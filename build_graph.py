"""
Nexus Step 5A — Regulatory Knowledge Graph.

Steps 1-4 turn a regulator PDF into a flat list of clauses, each mapped to a
canonical ontology concept. That's enough to *list* obligations, but not to
answer relationship questions — "which clauses depend on this definition",
"every credit-risk obligation across regulators", "what breaks if this rule is
repealed". Step 5A assembles the Step-4 output into a graph: typed nodes joined
by typed, provenance-bearing edges.

Node types
    Document        one per regulation (id = metadata document_id, else stem)
    Regulator       the issuing authority (e.g. CBUAE, RBI)
    Domain          a regulatory domain (e.g. Prudential Norms / Capital Adequacy)
    Concept         a canonical ontology concept (CPT_*) with its hierarchy
    ObligationType  a controlled obligation category (e.g. Reporting)
    Clause          one per numbered provision, carrying its own text/page

Edge types (each edge records the evidence it came from)
    ISSUED_BY        Document  -> Regulator
    HAS_CLAUSE       Document  -> Clause
    MAPS_TO_CONCEPT  Clause    -> Concept    (+ match_type, confidence, review)
    IN_DOMAIN        Clause    -> Domain  and  Concept -> Domain
    HAS_OBLIGATION   Clause    -> ObligationType
    CHILD_OF         Concept   -> Concept    (ontology hierarchy, from catalog)

Node IDs are type-prefixed and GLOBAL (Concept:CPT_CREDIT_RISK, Domain:<label>,
Regulator:<name>), so per-document graphs union cleanly into a corpus graph
later (Step 6 reasons across documents). Clause nodes are document-scoped
(Clause:<doc>/<clause_id>) since a clause belongs to exactly one document.

The output is dependency-light node-link JSON (no networkx/Neo4j needed to
produce it; it is straightforward to load into either). The Step-5A quality
gate — "every relationship is traceable to source evidence" — is enforced by
attaching a `provenance` block to every non-structural edge.

Usage:
    python build_graph.py --stem <file-stem>
    python build_graph.py --stem <file-stem> --nexus Nexus
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nexus_ontology


def _node(nodes: dict, node_id: str, node_type: str, **props) -> str:
    """Insert-or-merge a node keyed by id; returns the id for edge wiring."""
    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = {"id": node_id, "type": node_type, "properties": {k: v for k, v in props.items() if v is not None}}
    else:
        # Merge any newly-known properties without clobbering existing ones.
        for k, v in props.items():
            if v is not None and k not in existing["properties"]:
                existing["properties"][k] = v
    return node_id


def _edge(edges: list, seen: set, source: str, target: str, rel: str, provenance: dict | None = None) -> None:
    """Append an edge. Structural edges (no provenance) are de-duplicated;
    evidence-bearing edges are kept per-occurrence so provenance isn't lost."""
    if provenance is None:
        key = (source, target, rel)
        if key in seen:
            return
        seen.add(key)
        edges.append({"source": source, "target": target, "type": rel})
    else:
        edges.append({"source": source, "target": target, "type": rel, "provenance": provenance})


def build_graph(stem: str, nexus_dir: Path, jurisdiction: str | None = None,
                config_dir: str | None = None) -> None:
    ontology_path = nexus_dir / "ontology" / f"{stem}_ontology.json"
    clauses_path = nexus_dir / "clauses" / f"{stem}_clauses.json"
    metadata_path = nexus_dir / "metadata" / f"{stem}_metadata.json"
    if not ontology_path.exists():
        raise SystemExit(f"Ontology file not found: {ontology_path} (run build_ontology.py first)")
    if not clauses_path.exists():
        raise SystemExit(f"Clauses file not found: {clauses_path} (run export_clauses.py first)")

    with open(ontology_path, encoding="utf-8") as f:
        onto_doc = json.load(f)
    with open(clauses_path, encoding="utf-8") as f:
        clauses_doc = json.load(f)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    # The concept catalog gives us the hierarchy (parent) and each concept's own
    # domain — data the per-clause ontology objects don't all carry.
    config_root = nexus_ontology.resolve_config_root(config_dir, nexus_dir)
    jur = nexus_ontology.resolve_jurisdiction(config_root, nexus_dir, stem, jurisdiction)
    onto = nexus_ontology.load_ontology(config_root, jur)
    catalog = {c["canonical_id"]: c for c in onto["concepts"]}

    document_id = metadata.get("document_id") or onto_doc.get("document_id") or stem
    regulator = metadata.get("regulator")

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen: set = set()

    doc_node = _node(
        nodes, f"Document:{document_id}", "Document",
        title=metadata.get("title"), source_type=metadata.get("source_type"),
        version=metadata.get("version"), jurisdiction=jur, stem=stem,
        regulator=regulator,
    )

    if regulator:
        reg_node = _node(nodes, f"Regulator:{regulator}", "Regulator", name=regulator,
                         country=metadata.get("regulator_country"))
        _edge(edges, seen, doc_node, reg_node, "ISSUED_BY")

    # Index the ontology objects by clause_id so each clause node can pull its
    # canonical mapping and the evidence that produced it.
    obj_by_clause = {o["clause_id"]: o for o in onto_doc.get("objects", [])}

    for clause in clauses_doc.get("clauses", []):
        cid = clause["clause_id"]
        clause_node = _node(
            nodes, f"Clause:{document_id}/{cid}", "Clause",
            clause_no=clause.get("clause_no"), title=clause.get("title"),
            text=clause.get("text"), page=clause.get("page_reference"),
        )
        sem = clause.get("semantic") or {}
        if sem.get("mandatory_flag"):
            nodes[clause_node]["properties"]["mandatory_flag"] = sem["mandatory_flag"]
        if sem.get("action"):
            nodes[clause_node]["properties"]["action"] = sem["action"]

        _edge(edges, seen, doc_node, clause_node, "HAS_CLAUSE")

        provenance = {
            "document_id": document_id, "clause_id": cid,
            "clause_no": clause.get("clause_no"), "page": clause.get("page_reference"),
        }

        # Clause -> Domain (from the clause's own Step-3 domain label).
        clause_domain = onto["token_to_label"].get(sem.get("domain")) if sem.get("domain") else None
        if clause_domain:
            dom_node = _node(nodes, f"Domain:{clause_domain}", "Domain", label=clause_domain)
            _edge(edges, seen, clause_node, dom_node, "IN_DOMAIN", provenance)

        # Clause -> ObligationType.
        obl = sem.get("obligation_type")
        if obl:
            obl_node = _node(nodes, f"ObligationType:{obl}", "ObligationType", label=obl)
            _edge(edges, seen, clause_node, obl_node, "HAS_OBLIGATION", provenance)

        # Clause -> Concept (the canonical mapping + the evidence for it).
        obj = obj_by_clause.get(cid)
        canonical_id = obj.get("canonical_id") if obj else None
        if canonical_id:
            cat = catalog.get(canonical_id, {})
            concept_node = _node(
                nodes, f"Concept:{canonical_id}", "Concept",
                canonical_id=canonical_id, label=obj.get("canonical_label") or cat.get("label"),
                domain=cat.get("domain"), status=cat.get("status"),
            )
            map_prov = dict(provenance)
            map_prov.update({
                "match_type": obj.get("match_type"),
                "match_confidence": obj.get("match_confidence"),
                "review_required": obj.get("review_required"),
            })
            _edge(edges, seen, clause_node, concept_node, "MAPS_TO_CONCEPT", map_prov)

    # Concept-level structural edges: hierarchy (CHILD_OF) and each concept's own
    # domain. Only for concepts that actually appear in this document's graph.
    catalog_version = onto_doc.get("concept_catalog_version") or ";".join(
        f"{k}={v}" for k, v in onto["versions"].items()
    )
    present_concepts = [nid for nid, n in nodes.items() if n["type"] == "Concept"]
    for nid in present_concepts:
        canonical_id = nodes[nid]["properties"]["canonical_id"]
        cat = catalog.get(canonical_id, {})
        parent = cat.get("parent")
        if parent:
            parent_cat = catalog.get(parent, {})
            parent_node = _node(nodes, f"Concept:{parent}", "Concept",
                                canonical_id=parent, label=parent_cat.get("label"),
                                domain=parent_cat.get("domain"), status=parent_cat.get("status"))
            _edge(edges, seen, nid, parent_node, "CHILD_OF", {"catalog_version": catalog_version})
        concept_domain = cat.get("domain")
        if concept_domain:
            dom_node = _node(nodes, f"Domain:{concept_domain}", "Domain", label=concept_domain)
            _edge(edges, seen, nid, dom_node, "IN_DOMAIN", {"catalog_version": catalog_version})

    # Stats.
    node_type_counts: dict[str, int] = {}
    for n in nodes.values():
        node_type_counts[n["type"]] = node_type_counts.get(n["type"], 0) + 1
    edge_type_counts: dict[str, int] = {}
    for e in edges:
        edge_type_counts[e["type"]] = edge_type_counts.get(e["type"], 0) + 1

    graph = {
        "document_id": document_id,
        "regulator": regulator,
        "jurisdiction": jur,
        "concept_catalog_version": catalog_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "directed": True,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "nodes_by_type": node_type_counts,
            "edges_by_type": edge_type_counts,
        },
        "nodes": list(nodes.values()),
        "edges": edges,
    }

    (nexus_dir / "graph").mkdir(parents=True, exist_ok=True)
    out_path = nexus_dir / "graph" / f"{stem}_graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    log = {
        "document": stem, "document_id": document_id, "jurisdiction": jur,
        "generated_at": graph["generated_at"], "stats": graph["stats"],
    }
    (nexus_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = nexus_dir / "logs" / f"{stem}_graph_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    s = graph["stats"]
    print(f"Wrote {out_path}")
    print(f"Wrote {log_path}")
    print(f"  {s['nodes']} nodes, {s['edges']} edges")
    print(f"  nodes: {node_type_counts}")
    print(f"  edges: {edge_type_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the Step-4 ontology into a regulatory knowledge graph (Nexus Step 5A).")
    parser.add_argument("--stem", required=True, help="File stem shared by Nexus/ontology/<stem>_ontology.json etc.")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--jurisdiction", default=None,
                        help="Force a jurisdiction pack (e.g. IN_RBI); default: auto-detect from metadata regulator")
    parser.add_argument("--config", default=None,
                        help="Config root override (else NEXUS_CONFIG_DIR, then <nexus>/config, then packaged default)")
    args = parser.parse_args()
    build_graph(stem=args.stem, nexus_dir=Path(args.nexus),
                jurisdiction=args.jurisdiction, config_dir=args.config)


if __name__ == "__main__":
    main()
