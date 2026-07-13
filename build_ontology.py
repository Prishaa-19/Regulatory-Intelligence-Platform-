"""
Nexus Step 4 — Ontology Object Creation.

Steps 1-3 turn a regulator PDF into clauses tagged with a *free-text* concept.
Across the corpus that free text is almost never reused (298 distinct concepts
for 314 clauses), so it cannot answer "show every credit-risk obligation across
regulators". Step 4 maps each clause's free-text concept onto a **controlled
canonical concept** from the layered ontology (config/ontology/global/concepts.yaml
plus the document's jurisdiction extension) — a reusable banking vocabulary with
stable IDs, synonyms and a parent/child hierarchy, merged at runtime.

The one rule that makes this trustworthy: the matcher NEVER invents a canonical
ID. It only assigns IDs that already exist in the catalog, by string-matching
(exact label -> synonym -> normalized). Anything that does not match is
`unmapped` and flagged for review. That is the Step-4 quality gate — "every
approved concept uses a controlled ID; unknowns are flagged" — see the folder's
STANDARD.md for the full standard.

Input : Nexus/clauses/<stem>_clauses.json   (Step-3.5 flat clause export)
Output: Nexus/ontology/<stem>_ontology.json (one ontology object per clause)
        Nexus/logs/<stem>_ontology_log.json

Usage:
    python build_ontology.py --stem <file-stem>
    python build_ontology.py --stem <file-stem> --nexus Nexus
"""

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

import nexus_llm
import nexus_ontology

# --- Fuzzy-match token helpers (used by Matcher's fuzzy tier) ----------------
_CONCEPT_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Longest suffixes first so "-ations" is tried before "-tion"/"-s".
_STEM_SUFFIXES = ("ations", "ization", "isation", "ments", "ment",
                  "tions", "tion", "ings", "ing", "ies", "ers", "es", "s")


def _stem_token(t: str) -> str:
    """Very conservative suffix stripper. Goal is consistency (two surface forms
    of the same word collapse together), not linguistic correctness."""
    if len(t) <= 4:
        return t
    for suf in _STEM_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


def _concept_tokens(text: str | None, stem: bool) -> list[str]:
    toks = [t for t in _CONCEPT_TOKEN_RE.findall((text or "").casefold()) if len(t) > 1]
    return [_stem_token(t) for t in toks] if stem else toks


def _weighted_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    vec = {t: c * idf.get(t, 0.0) for t, c in tf.items()}
    vec = {t: w for t, w in vec.items() if w > 0.0}
    norm = math.sqrt(sum(w * w for w in vec.values()))
    return {t: w / norm for t, w in vec.items()} if norm else {}


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def _load_config(config_root: Path, jurisdiction: str | None) -> dict:
    """Load the merged (global + jurisdiction) concept catalog and domain token
    map via nexus_ontology, plus the Step-4 matching rules and output schema."""
    onto = nexus_ontology.load_ontology(config_root, jurisdiction)
    step4 = config_root / "step4_ontology"
    try:
        with open(step4 / "matching_rules.yaml", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
        with open(step4 / "ontology_object.schema.json", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Step-4 config missing: {exc.filename}\n"
            f"Expected matching_rules.yaml and ontology_object.schema.json "
            f"under {step4} (config root: {config_root})."
        )

    catalog_version = ";".join(f"{k}={v}" for k, v in onto["versions"].items())
    return {
        "concepts": onto["concepts"],
        "catalog_version": catalog_version,
        "rules": rules,
        "validator": Draft202012Validator(schema),
        "token_to_label": onto["token_to_label"],
        "jurisdiction": jurisdiction,
        "versions": onto["versions"],
    }


def _normalize(text: str | None, norm_cfg: dict, full: bool) -> str:
    """Normalize a concept string. `full` additionally strips trailing
    punctuation, which distinguishes a loose (normalized) match from a tight
    (exact/synonym) one."""
    if not text:
        return ""
    s = text
    if norm_cfg.get("casefold"):
        s = s.casefold()
    if norm_cfg.get("strip"):
        s = s.strip()
    if norm_cfg.get("collapse_whitespace"):
        s = re.sub(r"\s+", " ", s)
    if full and norm_cfg.get("strip_trailing_punct"):
        s = s.rstrip(norm_cfg.get("trailing_punctuation", ".,;:'\"")).strip()
    return s


class Matcher:
    """Deterministic catalog matcher. Assigns only IDs that exist in the
    catalog; never invents one."""

    def __init__(self, concepts: list[dict], rules: dict):
        self.concepts = {c["canonical_id"]: c for c in concepts}
        norm = rules["normalization"]
        self._norm = norm
        conf = {m["type"]: m["confidence"] for m in rules["match_precedence"]}
        self._conf = conf

        # tight = exact/synonym level; loose = normalized level.
        self._tight_label: dict[str, str] = {}
        self._tight_syn: dict[str, str] = {}
        self._loose_all: dict[str, str] = {}
        for c in concepts:
            cid = c["canonical_id"]
            self._tight_label[_normalize(c["label"], norm, False)] = cid
            self._loose_all[_normalize(c["label"], norm, True)] = cid
            for syn in c.get("synonyms") or []:
                self._tight_syn[_normalize(syn, norm, False)] = cid
                self._loose_all[_normalize(syn, norm, True)] = cid

        # Fuzzy tier: idf-weighted token vectors for every surface form (label +
        # synonyms), so a phrasing variant that misses the exact tiers can still
        # resolve by similarity. idf is computed across surface forms, so tokens
        # shared by many concepts (credit, risk, management) are down-weighted.
        self._fuzzy_cfg = rules.get("fuzzy") or {}
        self._fuzzy_forms: list[tuple[dict[str, float], str]] = []
        self._fuzzy_idf: dict[str, float] = {}
        self._fuzzy_stem = bool(self._fuzzy_cfg.get("light_stem", True))
        if self._fuzzy_cfg.get("enabled"):
            forms: list[tuple[list[str], str]] = []
            for c in concepts:
                cid = c["canonical_id"]
                for surface in [c["label"]] + list(c.get("synonyms") or []):
                    toks = _concept_tokens(surface, self._fuzzy_stem)
                    if toks:
                        forms.append((toks, cid))
            df: Counter = Counter()
            for toks, _ in forms:
                for t in set(toks):
                    df[t] += 1
            n = len(forms)
            self._fuzzy_idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
            self._fuzzy_forms = [(_weighted_vec(toks, self._fuzzy_idf), cid) for toks, cid in forms]

    def match(self, concept: str | None) -> tuple[str | None, str, float]:
        """Return (canonical_id, match_type, base_confidence)."""
        if not concept:
            return None, "unmapped", 0.0
        tight = _normalize(concept, self._norm, False)
        loose = _normalize(concept, self._norm, True)
        if tight in self._tight_label:
            return self._tight_label[tight], "exact_label", self._conf["exact_label"]
        if tight in self._tight_syn:
            return self._tight_syn[tight], "synonym", self._conf["synonym"]
        if loose in self._loose_all:
            return self._loose_all[loose], "normalized", self._conf["normalized"]

        if self._fuzzy_forms:
            qvec = _weighted_vec(_concept_tokens(concept, self._fuzzy_stem), self._fuzzy_idf)
            if qvec:
                best_sim, best_cid = 0.0, None
                for vec, cid in self._fuzzy_forms:
                    s = _cos(qvec, vec)
                    if s > best_sim:
                        best_sim, best_cid = s, cid
                if best_cid and best_sim >= self._fuzzy_cfg.get("min_similarity", 0.62):
                    conf = min(best_sim, self._fuzzy_cfg.get("max_confidence", 0.85))
                    return best_cid, "fuzzy", round(conf, 3)
        return None, "unmapped", 0.0


def build_object(clause: dict, matcher: Matcher, cfg: dict) -> dict:
    """Turn one flat clause record into an ontology object."""
    sem = clause.get("semantic") or {}
    concept = sem.get("concept")
    clause_domain = cfg["token_to_label"].get(sem.get("domain"))

    canonical_id, match_type, confidence = matcher.match(concept)
    matched = matcher.concepts.get(canonical_id) if canonical_id else None

    rules = cfg["rules"]
    review_cfg = rules["review_when"]
    domain_mismatch = bool(
        matched and clause_domain and matched.get("domain")
        and matched["domain"] != clause_domain
    )
    if domain_mismatch and review_cfg.get("domain_mismatch"):
        confidence = max(0.0, confidence - rules.get("domain_mismatch_penalty", 0.0))

    review_required = (
        (match_type == "unmapped" and review_cfg.get("unmapped", True))
        or (matched is not None and matched.get("status") == "draft"
            and review_cfg.get("matched_concept_is_draft", True))
        or (matched is not None and confidence < rules.get("auto_accept", 1.0))
        or domain_mismatch
    )

    return {
        "clause_id": clause.get("clause_id"),
        "clause_no": clause.get("clause_no"),
        "source_concept": concept,
        "canonical_id": canonical_id,
        "canonical_label": matched["label"] if matched else None,
        "domain": clause_domain,
        "action": sem.get("action"),
        "obligation_type": sem.get("obligation_type"),
        "mandatory_flag": sem.get("mandatory_flag"),
        "match_type": match_type,
        "match_confidence": round(confidence, 3),
        "review_required": review_required,
        # Filled only by the opt-in --suggest layer, for unmapped concepts.
        # A suggestion is never an approved mapping: canonical_id stays null.
        "suggested_canonical_id": None,
        "suggested_label": None,
        "suggestion_confidence": None,
    }


def _suggest_unmapped(concepts: list[str], catalog: list[dict],
                      rules: dict) -> tuple[dict[str, dict], dict]:
    """For each distinct unmapped concept, ask the LLM to pick the best-matching
    canonical_id FROM THE CATALOG, or null. The model is constrained to real
    catalog ids; anything else it returns is discarded. Suggestions are advisory
    — the caller never converts them into approved mappings."""
    load_dotenv()
    if not nexus_llm.is_configured():
        return {}, {"skipped": True, "reason": "no LLM provider configured (NVIDIA_*/AZURE_OPENAI_*)",
                    "distinct_unmapped": len(concepts), "suggested": 0}

    scfg = rules.get("suggestion", {}) or {}
    batch_size = scfg.get("batch_size", 40)
    min_conf = scfg.get("min_confidence", 0.5)
    valid_ids = {c["canonical_id"] for c in catalog}
    id_to_label = {c["canonical_id"]: c["label"] for c in catalog}
    catalog_lines = "\n".join(
        f'{c["canonical_id"]} | {c["label"]} | {c.get("definition", "")}' for c in catalog
    )
    system = (
        "You map free-text regulatory concept labels onto a fixed catalog of "
        "canonical banking concepts. For each input concept choose the single "
        "best-matching canonical_id from the catalog, or null if none is a good "
        "fit. NEVER invent an id — use only ids listed in the catalog. Return a "
        'JSON array of objects {"concept": <input>, "canonical_id": <CPT_...|null>, '
        '"confidence": <0.0-1.0>} and nothing else.\n\nCATALOG (id | label | definition):\n'
        + catalog_lines
    )

    suggestions: dict[str, dict] = {}
    warnings: list[str] = []
    failed = 0
    for start in range(0, len(concepts), batch_size):
        batch = concepts[start:start + batch_size]
        try:
            content = nexus_llm.call_chat([
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(f"- {c}" for c in batch)},
            ])
            parsed = json.loads(content[content.find("["): content.rfind("]") + 1])
        except Exception as exc:
            failed += 1
            warnings.append(f"Suggestion batch at offset {start} failed: {exc}")
            continue
        for item in parsed:
            concept, cid = item.get("concept"), item.get("canonical_id")
            try:
                conf = float(item.get("confidence"))
            except (TypeError, ValueError):
                conf = None
            # Keep only real catalog ids at/above the confidence floor.
            if concept and cid in valid_ids and conf is not None and conf >= min_conf:
                suggestions[concept] = {
                    "canonical_id": cid, "label": id_to_label[cid],
                    "confidence": round(max(0.0, min(1.0, conf)), 3),
                }
    meta = {
        "skipped": False, "distinct_unmapped": len(concepts),
        "suggested": len(suggestions), "failed_batches": failed,
        "min_confidence": min_conf, "providers": nexus_llm.configured_summary(),
    }
    if warnings:
        meta["warnings"] = warnings
    return suggestions, meta


def build_ontology(stem: str, nexus_dir: Path, jurisdiction: str | None = None,
                   config_dir: str | None = None, suggest: bool = False) -> None:
    config_root = nexus_ontology.resolve_config_root(config_dir, nexus_dir)
    jur = nexus_ontology.resolve_jurisdiction(config_root, nexus_dir, stem, jurisdiction)
    cfg = _load_config(config_root, jur)
    clauses_path = nexus_dir / "clauses" / f"{stem}_clauses.json"
    if not clauses_path.exists():
        raise SystemExit(
            f"Clauses file not found: {clauses_path} (run export_clauses.py first)"
        )

    with open(clauses_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    matcher = Matcher(cfg["concepts"], cfg["rules"])

    objects: list[dict] = []
    skipped_no_concept = 0
    for clause in doc.get("clauses", []):
        # Clauses with no concept (header-only / unclassified) have nothing to
        # map; skip them so coverage reflects only classified provisions.
        if not (clause.get("semantic") or {}).get("concept"):
            skipped_no_concept += 1
            continue
        objects.append(build_object(clause, matcher, cfg))

    # Opt-in LLM-suggest layer: propose a catalog id for each unmapped concept.
    # Advisory only — canonical_id and match_type are left untouched.
    suggest_meta = None
    if suggest:
        distinct_unmapped = sorted({
            o["source_concept"] for o in objects
            if o["match_type"] == "unmapped" and o["source_concept"]
        })
        suggestions, suggest_meta = _suggest_unmapped(distinct_unmapped, cfg["concepts"], cfg["rules"])
        for o in objects:
            if o["match_type"] == "unmapped":
                s = suggestions.get(o["source_concept"])
                if s:
                    o["suggested_canonical_id"] = s["canonical_id"]
                    o["suggested_label"] = s["label"]
                    o["suggestion_confidence"] = s["confidence"]

    total = len(objects)
    mapped = sum(1 for o in objects if o["canonical_id"] is not None)
    review = sum(1 for o in objects if o["review_required"])
    suggested = sum(1 for o in objects if o["suggested_canonical_id"] is not None)
    artifact = {
        "document_id": doc.get("document_id"),
        "concept_catalog_version": cfg["catalog_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "total": total,
            "mapped": mapped,
            "unmapped": total - mapped,
            "coverage_pct": round(100 * mapped / total, 1) if total else 0.0,
            "review_required": review,
            "suggested": suggested,
        },
        "objects": objects,
    }

    # Self-check against the output contract before writing.
    artifact_errors = [
        f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}"
        for e in cfg["validator"].iter_errors(artifact)
    ]

    (nexus_dir / "ontology").mkdir(parents=True, exist_ok=True)
    out_path = nexus_dir / "ontology" / f"{stem}_ontology.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    log = {
        "document": stem,
        "generated_at": artifact["generated_at"],
        "jurisdiction": cfg["jurisdiction"],
        "catalog_version": cfg["catalog_version"],
        "coverage": artifact["coverage"],
        "skipped_no_concept": skipped_no_concept,
        "suggestion": suggest_meta,
        "artifact_valid": not artifact_errors,
        "artifact_errors": artifact_errors[:5],
    }
    (nexus_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = nexus_dir / "logs" / f"{stem}_ontology_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    cov = artifact["coverage"]
    print(f"Wrote {out_path}")
    print(f"Wrote {log_path}")
    print(f"  {cov['mapped']}/{cov['total']} concepts mapped "
          f"({cov['coverage_pct']}%); {cov['unmapped']} unmapped, "
          f"{cov['review_required']} flagged for review"
          + (f"; {cov['suggested']} with LLM suggestion (unapproved)" if cov['suggested'] else "")
          + (f"; {skipped_no_concept} header-only clause(s) skipped" if skipped_no_concept else ""))
    if artifact_errors:
        print(f"  WARNING: artifact failed schema validation: {artifact_errors[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map free-text clause concepts onto canonical ontology objects (Nexus Step 4)."
    )
    parser.add_argument("--stem", required=True, help="File stem shared by Nexus/clauses/<stem>_clauses.json etc.")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--jurisdiction", default=None,
                        help="Force a jurisdiction pack (e.g. IN_RBI); default: auto-detect from metadata regulator")
    parser.add_argument("--config", default=None,
                        help="Config root override (else NEXUS_CONFIG_DIR, then <nexus>/config, then packaged default)")
    parser.add_argument("--suggest", action="store_true",
                        help="Use an LLM to propose a catalog id for each unmapped concept (advisory, review-gated, needs an LLM provider)")
    args = parser.parse_args()

    build_ontology(stem=args.stem, nexus_dir=Path(args.nexus),
                   jurisdiction=args.jurisdiction, config_dir=args.config, suggest=args.suggest)


if __name__ == "__main__":
    main()
