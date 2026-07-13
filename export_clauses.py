"""
Nexus Step 4 — Flat Clause Export.

Steps 1-3 produce a nested Document -> Part -> Section -> Subsection -> Clause
tree with semantic tags attached to each operative node. Downstream
consumers (search index, obligation register, diffing) usually want a flat
list of clause records, one per numbered provision, in this shape:

    {
      "clause_id": "clause_4_5_5",
      "clause_no": "4.5.5",
      "title": "Reporting",
      "text": "...",
      "page_reference": 13,
      "semantic": {
        "domain": "FINANCIAL_MARKETS",
        "concept": "...",
        "action": "REPORT",
        "object": "...",
        "obligation_type": "Reporting",
        "mandatory_flag": "Mandatory",
        "confidence_score": 0.87
      }
    }

This step flattens the Step-3 semantics tree into exactly that. Per the
chosen scope, *every numbered provision* (section, subsection, and clause
nodes carrying a real clause-style number) becomes a record; structural
wrappers (part/annex) and unnumbered noise are dropped. The domain value is
converted from the readable Step-3 taxonomy to a compact token (e.g.
"Anti-Money Laundering / KYC" -> "AML_KYC").

False-positive numbers that pypdf's plain-text extraction produces — bare
years like 1934/2023, or mid-sentence fragments — are filtered by
CLAUSE_NO_RE plus a sanity bound on bare integers (see _is_clause_number).

Usage:
    python export_clauses.py --stem <file-stem>
    python export_clauses.py --stem <file-stem> --nexus Nexus
"""

import argparse
import json
import re
from pathlib import Path

import nexus_ontology

# A real clause number is dotted (4.2, 4.5.5) or a small standalone integer.
# Bare integers >= MAX_BARE_INT are treated as years/citations, not clause nos.
CLAUSE_NO_RE = re.compile(r"^\d+(\.\d+)*$")
MAX_BARE_INT = 100

CLAUSE_NODE_TYPES = ("section", "subsection", "clause")


def _is_clause_number(number: str | None) -> bool:
    if not number or not CLAUSE_NO_RE.match(number):
        return False
    if "." not in number and int(number) >= MAX_BARE_INT:
        return False  # bare year/citation like 1934, 2023
    return True


def _domain_token(domain: str | None, label_to_token: dict, escape_token: str) -> str | None:
    # Readable Step-3 domain label -> compact token, from the merged ontology.
    if domain is None:
        return None
    return label_to_token.get(domain, escape_token)


def _semantic_block(node: dict, label_to_token: dict, escape_token: str) -> dict:
    return {
        "domain": _domain_token(node.get("domain"), label_to_token, escape_token),
        "concept": node.get("concept"),
        "action": node.get("action"),
        "object": node.get("object"),
        "obligation_type": node.get("obligation_type"),
        "mandatory_flag": node.get("mandatory_flag"),
        "confidence_score": node.get("confidence_score"),
    }


def flatten(nodes: list[dict], clauses: list[dict], seen_ids: dict[str, int],
            label_to_token: dict, escape_token: str) -> None:
    for node in nodes:
        if node["type"] in CLAUSE_NODE_TYPES and _is_clause_number(node.get("number")):
            number = node["number"]
            base_id = "clause_" + number.replace(".", "_")
            seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
            clause_id = base_id if seen_ids[base_id] == 1 else f"{base_id}__{seen_ids[base_id]}"

            clauses.append({
                "clause_id": clause_id,
                "clause_no": number,
                "title": node.get("heading"),
                "text": node.get("text"),
                "page_reference": node.get("page_start"),
                "semantic": _semantic_block(node, label_to_token, escape_token),
            })
        flatten(node.get("children", []), clauses, seen_ids, label_to_token, escape_token)


def export_clauses(stem: str, nexus_dir: Path, jurisdiction: str | None = None,
                   config_dir: str | None = None) -> None:
    semantics_path = nexus_dir / "semantics" / f"{stem}_semantics.json"
    if not semantics_path.exists():
        raise SystemExit(f"Semantics file not found: {semantics_path} (run interpret_document.py first)")

    # The label -> token map is jurisdiction-dependent (e.g. FEMA exists only for
    # IN_RBI), so resolve the jurisdiction and load the merged ontology.
    config_root = nexus_ontology.resolve_config_root(config_dir, nexus_dir)
    jur = nexus_ontology.resolve_jurisdiction(config_root, nexus_dir, stem, jurisdiction)
    onto = nexus_ontology.load_ontology(config_root, jur)
    label_to_token = onto["label_to_token"]
    escape_token = label_to_token.get(onto["escape_hatch"]["domain"], "OTHER")

    with open(semantics_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    clauses: list[dict] = []
    flatten(doc["structure"], clauses, {}, label_to_token, escape_token)

    # Only clauses that carry their own obligation text but got no tags are
    # genuine misses (an LLM outage during Step 3). Header-only provisions
    # legitimately have null semantics and are not counted here.
    header_only = sum(1 for c in clauses if not (c["text"] or "").strip())
    genuine_misses = sum(
        1 for c in clauses if c["semantic"]["action"] is None and (c["text"] or "").strip()
    )

    (nexus_dir / "clauses").mkdir(parents=True, exist_ok=True)
    out_path = nexus_dir / "clauses" / f"{stem}_clauses.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"document_id": doc.get("document_id"), "clause_count": len(clauses), "clauses": clauses},
            f, indent=2, ensure_ascii=False,
        )

    print(f"Wrote {out_path}")
    print(f"  {len(clauses)} clause records "
          f"({header_only} header-only w/o own text)"
          + (f"; {genuine_misses} tagged-text clause(s) MISSING semantics — re-run Step 3 while the LLM is healthy" if genuine_misses else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten the Step-3 semantics tree into flat clause records (Nexus Step 3.5).")
    parser.add_argument("--stem", required=True, help="File stem shared by Nexus/semantics/<stem>_semantics.json etc.")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--jurisdiction", default=None,
                        help="Force a jurisdiction pack (e.g. IN_RBI); default: auto-detect from metadata regulator")
    parser.add_argument("--config", default=None,
                        help="Config root override (else NEXUS_CONFIG_DIR, then <nexus>/config, then packaged default)")
    args = parser.parse_args()

    export_clauses(stem=args.stem, nexus_dir=Path(args.nexus),
                   jurisdiction=args.jurisdiction, config_dir=args.config)


if __name__ == "__main__":
    main()
