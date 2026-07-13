"""
Nexus Step 3 — Regulatory Meaning.

Step 1 answered "what is the document" (Markdown + metadata). Step 2 answered
"how is it organized" (structure tree). Step 3 answers "what does each
clause mean": every operative unit in the Step-2 tree — a clause, or a
Section/Subsection whose text is never broken into clauses — gets tagged
with domain, concept, action, object, obligation_type, mandatory_flag, and
a self-reported confidence_score.

This step is inherently interpretive, so unlike Steps 1-2 it has no
deterministic path — every operative unit goes through the LLM. To keep
labels comparable across documents (so a later "find all AML/KYC record
retention clauses across every regulator" query actually works), domain,
obligation_type, and action are constrained to a fixed taxonomy with an
"Other"/"OTHER" escape hatch; concept and object stay free text since
they're naturally clause-specific.

The Step-2 structure file is left untouched — this step writes its own
artifact (Nexus/semantics/<stem>_semantics.json), a deep copy of the
structure tree with the seven fields above added to each operative unit's
node, consistent with the rest of the pipeline never mutating a prior
step's output in place.

Usage:
    python interpret_document.py --stem <file-stem>
    python interpret_document.py --stem <file-stem> --nexus Nexus
"""

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

import nexus_llm
import nexus_ontology
from nexus_llm import call_chat

# Nothing domain-specific is hardcoded here. The controlled vocabularies come
# from the LAYERED ontology (global + the document's jurisdiction), loaded at
# runtime via nexus_ontology; the thresholds and JSON Schemas come from
# <config-root>/step3_semantics/. The schemas' domain/action/obligation/modality
# enums are STAMPED from the merged ontology at load, so they are always correct
# for the jurisdiction in play and can never drift from the ontology.
def _stamp_enums(response_schema: dict, artifact_schema: dict, onto: dict) -> None:
    """Inject the merged-ontology enums into the two Step-3 schemas in place."""
    rp = response_schema["items"]["properties"]
    rp["domain"]["enum"] = list(onto["domain_labels"])
    rp["action"]["enum"] = list(onto["action"])
    rp["obligation_type"]["enum"] = list(onto["obligation_type"])
    rp["mandatory_flag"]["enum"] = list(onto["mandatory_flag"])
    # Artifact fields are nullable (unclassified nodes carry null), so keep null.
    ad = artifact_schema["$defs"]
    ad["domain"]["enum"] = list(onto["domain_labels"]) + [None]
    ad["action"]["enum"] = list(onto["action"]) + [None]
    ad["obligation_type"]["enum"] = list(onto["obligation_type"]) + [None]
    ad["mandatory_flag"]["enum"] = list(onto["mandatory_flag"]) + [None]


def _build_prompt(onto: dict) -> str:
    return (
        "You classify clauses from a regulatory document. For each numbered item "
        "below, read its context (the Part/Section/Subsection it sits under) and "
        "its text, then return an object with exactly these keys:\n"
        f"  domain: one of {list(onto['domain_labels'])}\n"
        "  concept: short free-text label for the regulatory concept, e.g. "
        "'Customer Due Diligence' (2-5 words)\n"
        f"  action: one of {list(onto['action'])} (the operative verb)\n"
        "  object: short free-text label for what the action applies to, e.g. "
        "'Customer Records' (2-5 words)\n"
        f"  obligation_type: one of {list(onto['obligation_type'])}\n"
        f"  mandatory_flag: one of {list(onto['mandatory_flag'])} — 'Not Applicable' if "
        "the item is definitional/explanatory rather than an obligation\n"
        "  confidence_score: your own confidence in this classification, a float "
        "0.0-1.0\n"
        "Classify only what the text explicitly states — never infer an obligation "
        "the clause does not state (a definition is 'Not Applicable', not "
        "'Mandatory'). When no listed value fits, use the escape hatch "
        f"(domain '{onto['escape_hatch']['domain']}', action '{onto['escape_hatch']['action']}', "
        f"obligation_type '{onto['escape_hatch']['obligation_type']}').\n"
        "Respond with a single JSON array of objects {\"index\": <int>, "
        "\"domain\": ..., \"concept\": ..., \"action\": ..., \"object\": ..., "
        "\"obligation_type\": ..., \"mandatory_flag\": ..., "
        "\"confidence_score\": ...} and nothing else, one entry per item given."
    )


def build_context(config_root: Path, onto: dict) -> dict:
    """Assemble everything a run needs: merged taxonomies, prompt, batch
    settings, and schema validators with jurisdiction-correct enums stamped in."""
    step3 = config_root / "step3_semantics"
    try:
        with open(step3 / "thresholds.yaml", encoding="utf-8") as f:
            thresholds = yaml.safe_load(f)
        with open(step3 / "classification_response.schema.json", encoding="utf-8") as f:
            response_schema = json.load(f)
        with open(step3 / "semantics_artifact.schema.json", encoding="utf-8") as f:
            artifact_schema = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Step-3 config missing: {exc.filename}\n"
            f"Expected thresholds.yaml and the two schemas under {step3}."
        )

    _stamp_enums(response_schema, artifact_schema, onto)
    return {
        "onto": onto,
        "domain": onto["domain_labels"],
        "obligation_type": onto["obligation_type"],
        "action": onto["action"],
        "mandatory_flag": onto["mandatory_flag"],
        "escape": onto["escape_hatch"],
        "batch_size": thresholds["batch"]["size"],
        "max_context_chars": thresholds["batch"]["max_context_chars"],
        "prompt": _build_prompt(onto),
        "response_validator": Draft202012Validator(response_schema),
        "artifact_validator": Draft202012Validator(artifact_schema),
    }


def collect_operative_units(nodes: list[dict], ancestors: list[str], units: list[dict]) -> None:
    """Collect every node that carries its own obligation text: clause nodes,
    leaf nodes, and numbered section/subsection nodes that have body text of
    their own (even if they also have children). Pure grouping headers — a
    numbered parent with no own text, e.g. "4.2 Market-makers and users" whose
    substance lives in 4.2.1/4.2.2 — are intentionally left unclassified."""
    for node in nodes:
        is_numbered_provision = node["type"] in ("section", "subsection") and node.get("number")
        is_operative = (
            node["type"] == "clause"
            or (not node["children"])
            or is_numbered_provision
        )
        if is_operative and node.get("text"):
            units.append({"node": node, "trail": " > ".join(ancestors) or "(document root)"})
        trail_entry = f"{node['type']} {node.get('number') or ''} {node.get('heading') or ''}".strip()
        collect_operative_units(node["children"], ancestors + [trail_entry], units)


def _validate(value, allowed: tuple[str, ...], fallback: str) -> str:
    return value if value in allowed else fallback


def classify_batch(batch: list[dict], ctx: dict) -> tuple[dict[int, dict], list[str]]:
    items = []
    for i, u in enumerate(batch):
        text = (u["node"].get("text") or "")[:ctx["max_context_chars"]]
        items.append(f'{i}. context="{u["trail"]}" text="{text}"')

    messages = [
        {"role": "system", "content": ctx["prompt"]},
        {"role": "user", "content": "\n".join(items)},
    ]
    content = call_chat(messages)

    bracket_index = content.find("[")
    parsed = json.loads(content[bracket_index: content.rfind("]") + 1])

    # Enforce the response contract. Violations are surfaced (so "the model
    # returned arbitrary JSON" is visible in the log, not silently masked) but
    # don't sink the batch: _validate() below still coerces every controlled
    # field to an allowed value, so the written artifact is always schema-clean.
    schema_errors = [
        f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}"
        for e in ctx["response_validator"].iter_errors(parsed)
    ]
    return {item["index"]: item for item in parsed if "index" in item}, schema_errors


def classify_units(units: list[dict], skip_llm: bool, log: dict, ctx: dict) -> None:
    if not units:
        return

    if skip_llm:
        for u in units:
            _apply_defaults(u["node"])
        log["warnings"].append(f"{len(units)} operative unit(s) left unclassified (--skip-llm was set).")
        return

    load_dotenv()
    if not nexus_llm.is_configured():
        for u in units:
            _apply_defaults(u["node"])
        log["warnings"].append(
            f"{len(units)} operative unit(s) left unclassified: no LLM provider configured "
            "(NVIDIA_* or AZURE_OPENAI_* in .env)."
        )
        return

    escape = ctx["escape"]
    classified_count = 0
    failed_batches = 0
    for start in range(0, len(units), ctx["batch_size"]):
        batch = units[start:start + ctx["batch_size"]]
        try:
            results, schema_errors = classify_batch(batch, ctx)
        except Exception as exc:
            failed_batches += 1
            for u in batch:
                _apply_defaults(u["node"])
            log["warnings"].append(
                f"Batch at offset {start} ({len(batch)} items) left unclassified: {exc}"
            )
            continue

        if schema_errors:
            log["warnings"].append(
                f"Batch at offset {start}: {len(schema_errors)} response-schema "
                f"violation(s), coerced to allowed values. First: {schema_errors[0]}"
            )

        for i, u in enumerate(batch):
            r = results.get(i)
            if r is None:
                _apply_defaults(u["node"])
                continue
            u["node"]["domain"] = _validate(r.get("domain"), ctx["domain"], escape["domain"])
            u["node"]["concept"] = r.get("concept") or None
            u["node"]["action"] = _validate(r.get("action"), ctx["action"], escape["action"])
            u["node"]["object"] = r.get("object") or None
            u["node"]["obligation_type"] = _validate(r.get("obligation_type"), ctx["obligation_type"], escape["obligation_type"])
            u["node"]["mandatory_flag"] = _validate(r.get("mandatory_flag"), ctx["mandatory_flag"], escape["mandatory_flag"])
            try:
                score = float(r.get("confidence_score"))
                u["node"]["confidence_score"] = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                u["node"]["confidence_score"] = None
            classified_count += 1

    log["steps"].append({
        "step": "llm_semantic_classification", "status": "ok" if failed_batches == 0 else "partial",
        "classified": classified_count, "total": len(units), "failed_batches": failed_batches,
        "providers": nexus_llm.configured_summary(),
        "jurisdiction": ctx["onto"]["jurisdiction"],
        "ontology_versions": ctx["onto"]["versions"],
    })


def _apply_defaults(node: dict) -> None:
    node["domain"] = None
    node["concept"] = None
    node["action"] = None
    node["object"] = None
    node["obligation_type"] = None
    node["mandatory_flag"] = None
    node["confidence_score"] = None


def interpret_document(stem: str, nexus_dir: Path, skip_llm: bool,
                       jurisdiction: str | None = None, config_dir: str | None = None) -> None:
    started_at = time.monotonic()
    structure_path = nexus_dir / "structure" / f"{stem}_structure.json"
    if not structure_path.exists():
        raise SystemExit(f"Structure file not found: {structure_path} (run structure_document.py first)")

    # Resolve config + jurisdiction, then load the merged (global + jurisdiction)
    # ontology and build the run context. Jurisdiction is auto-detected from the
    # document's Step-1 metadata unless forced with --jurisdiction.
    config_root = nexus_ontology.resolve_config_root(config_dir, nexus_dir)
    jur = nexus_ontology.resolve_jurisdiction(config_root, nexus_dir, stem, jurisdiction)
    onto = nexus_ontology.load_ontology(config_root, jur)
    ctx = build_context(config_root, onto)

    with open(structure_path, "r", encoding="utf-8") as f:
        structure_doc = json.load(f)

    semantics_doc = copy.deepcopy(structure_doc)
    log: dict = {
        "document": stem, "started_at": datetime.now(timezone.utc).isoformat(),
        "jurisdiction": jur, "ontology_versions": onto["versions"],
        "steps": [], "warnings": [],
    }

    units: list[dict] = []
    collect_operative_units(semantics_doc["structure"], [], units)
    log["steps"].append({"step": "collect_operative_units", "status": "ok", "count": len(units)})

    classify_units(units, skip_llm, log, ctx)

    # Self-check the artifact against its schema before writing. Degradation
    # principle: a schema-invalid artifact is still written, but the violation
    # is recorded so it can't pass downstream unnoticed.
    artifact_errors = [
        f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}"
        for e in ctx["artifact_validator"].iter_errors(semantics_doc)
    ]
    log["steps"].append({
        "step": "validate_artifact",
        "status": "ok" if not artifact_errors else "invalid",
        "error_count": len(artifact_errors),
        "errors": artifact_errors[:5],
    })

    (nexus_dir / "semantics").mkdir(parents=True, exist_ok=True)
    semantics_path = nexus_dir / "semantics" / f"{stem}_semantics.json"
    with open(semantics_path, "w", encoding="utf-8") as f:
        json.dump(semantics_doc, f, indent=2, ensure_ascii=False)
    log["steps"].append({"step": "semantics_write", "status": "ok", "path": str(semantics_path)})

    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    log["duration_seconds"] = round(time.monotonic() - started_at, 2)

    (nexus_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = nexus_dir / "logs" / f"{stem}_semantics_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"Wrote {semantics_path}")
    print(f"Wrote {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify each clause's regulatory meaning (Nexus Step 3).")
    parser.add_argument("--stem", required=True, help="File stem shared by Nexus/structure/<stem>_structure.json etc.")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--skip-llm", action="store_true", help="Leave all semantic fields null instead of calling the LLM")
    parser.add_argument("--jurisdiction", default=None,
                        help="Force a jurisdiction pack (e.g. IN_RBI); default: auto-detect from metadata regulator")
    parser.add_argument("--config", default=None,
                        help="Config root override (else NEXUS_CONFIG_DIR, then <nexus>/config, then packaged default)")
    args = parser.parse_args()

    interpret_document(stem=args.stem, nexus_dir=Path(args.nexus), skip_llm=args.skip_llm,
                       jurisdiction=args.jurisdiction, config_dir=args.config)


if __name__ == "__main__":
    main()
