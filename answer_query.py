"""
Nexus Step 6 — Reasoning and Retrieval.

Steps 5A/5B built two complementary assets: a knowledge graph (relationships)
and a semantic index (similarity). Step 6 uses BOTH to answer a natural-language
question about the corpus, grounded in real clauses:

    1. Retrieve   — embed the query (Step-5B backend) and score it against every
                    clause vector across ALL indexed documents; keep the top-k.
    2. Reason     — for each hit, walk the Step-5A graph to attach its concept,
                    concept hierarchy, domain, obligation type, and sibling
                    clauses mapped to the same concept in OTHER documents. This
                    is the "reason across relationships" the flat index can't do.
    3. Ground     — assemble an evidence set where every item carries its source
                    (document, clause number, page) and mapping uncertainty
                    (review_required / match confidence).
    4. Synthesize — OPTIONAL: an LLM writes a prose answer that may cite ONLY the
                    supplied evidence; each citation is post-checked against the
                    evidence set, and anything ungrounded is flagged. With no LLM
                    configured, the structured evidence + reasoning path is the
                    answer (no free-text synthesis, nothing invented).

The Step-6 quality gate — "every answer includes source, reasoning path, and
uncertainty" — is structural here: the answer object always carries `evidence`
(sources), `reasoning_path`, and `confidence`, whether or not an LLM ran.

Every run is written to Nexus/answers/<timestamp>.json for auditability
(regulatory-memory principle: reasoning must be reproducible after the fact).

Usage:
    python answer_query.py --query "What must LFIs do when credit risk increases significantly?"
    python answer_query.py --query "..." --top-k 8 --no-llm
    python answer_query.py --query "..." --stem CBUAE_EN_5996_VER1   # restrict to one doc
"""

import argparse
import glob
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import nexus_llm
from build_index import tokenize, cosine, similarity, is_dense_index, _rebuild_backend

# Retrieval-score bands, per representation. tf-idf cosines are modest by nature;
# neural-embedding cosines (e.g. text-embedding-3-small) sit higher and cluster
# tighter, so a relevant hit that scores ~0.45 there would be mislabelled "Low"
# on the lexical scale. Bands are chosen per backend at scoring time.
STRONG, MODERATE = 0.30, 0.15               # tf-idf (lexical)
DENSE_STRONG, DENSE_MODERATE = 0.50, 0.32   # neural embeddings

ANSWER_SYSTEM_PROMPT = (
    "You are a regulatory-compliance analyst. Answer the user's question using "
    "ONLY the numbered evidence clauses provided — do not use outside knowledge "
    "or infer beyond what the clauses state. "
    "Synthesise across ALL clauses that bear on the question; do not stop at one "
    "or two. When the question asks what an entity must DO, lead with the concrete "
    "obligations and required actions (what it must actually do), then any "
    "supporting detection, measurement, reporting or governance duties — cover "
    "each distinct obligation the evidence contains. "
    "Every factual sentence must cite its source in square brackets as "
    "[<document_id> <clause_no>]. If the evidence is insufficient to answer, say "
    "so plainly. Keep it concise but complete. End with a single line beginning "
    "'Uncertainty:' that states what is unclear, missing, or based on clauses "
    "still pending compliance review."
)


def load_corpus(nexus_dir: Path, only_stem: str | None):
    """Load every per-document index and graph, returning the indexes and one
    merged corpus graph (node dict + edge list) built by unioning the per-doc
    graphs on their global node IDs."""
    index_paths = sorted(glob.glob(str(nexus_dir / "index" / "*_index.json")))
    graph_paths = sorted(glob.glob(str(nexus_dir / "graph" / "*_graph.json")))

    def _stem_of(p: str, suffix: str) -> str:
        return Path(p).name[: -len(suffix)]

    indexes = []
    stems_by_doc: dict[str, str] = {}
    for p in index_paths:
        stem = _stem_of(p, "_index.json")
        if only_stem and stem != only_stem:
            continue
        with open(p, encoding="utf-8") as f:
            index = json.load(f)
        indexes.append(index)
        stems_by_doc[index["document_id"]] = stem

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for p in graph_paths:
        if only_stem and _stem_of(p, "_graph.json") != only_stem:
            continue
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        for n in g["nodes"]:
            nodes.setdefault(n["id"], n)
        edges.extend(g["edges"])
    return indexes, nodes, edges, stems_by_doc


def attach_clause_text(evidence: list[dict], nexus_dir: Path, stems_by_doc: dict[str, str]) -> None:
    """Load each evidence clause's FULL text from its source clauses file and
    attach it as `text`. Retrieval/graph only carry the clause title (often a
    truncated first-line fragment), so without this the LLM synthesizes from
    headings, not the actual obligation wording — the single biggest lever on
    answer completeness."""
    cache: dict[str, dict] = {}
    for e in evidence:
        stem = stems_by_doc.get(e["document_id"])
        if not stem:
            continue
        if stem not in cache:
            path = nexus_dir / "clauses" / f"{stem}_clauses.json"
            try:
                data = json.load(open(path, encoding="utf-8"))
                cache[stem] = {c["clause_id"]: c for c in data.get("clauses", [])}
            except Exception:
                cache[stem] = {}
        rec = cache[stem].get(e["clause_id"])
        if rec:
            e["text"] = rec.get("text") or rec.get("title")


def _corpus_idf(indexes: list[dict]) -> tuple[dict[str, float], int]:
    """Build a SHARED idf over every clause in every index, from the raw term
    frequencies each index stores. This is the fix that makes cross-document
    ranking valid: a term's weight no longer depends on which document's local
    idf happens to apply. Falls back gracefully if an older index lacks `tf`."""
    df: Counter = Counter()
    n_docs = 0
    for index in indexes:
        for rec in index["clauses"]:
            tf = rec.get("tf")
            if tf is None:
                continue
            n_docs += 1
            for term in tf:
                df[term] += 1
    idf = {term: math.log((n_docs + 1) / (d + 1)) + 1.0 for term, d in df.items()}
    return idf, n_docs


def _weight(tf: dict, idf: dict) -> dict[str, float]:
    vec = {t: c * idf.get(t, 0.0) for t, c in tf.items()}
    vec = {t: w for t, w in vec.items() if w > 0.0}
    norm = math.sqrt(sum(w * w for w in vec.values()))
    return {t: w / norm for t, w in vec.items()} if norm > 0 else vec


def _hit(index: dict, rec: dict, score: float) -> dict:
    return {
        "score": round(score, 4), "document_id": index["document_id"],
        "clause_id": rec["clause_id"], "clause_no": rec["clause_no"],
        "title": rec["title"], "page": rec["page"], "concept": rec["concept"],
    }


def _retrieve_sparse(indexes: list[dict], query: str) -> list[dict]:
    """tf-idf retrieval in a single SHARED idf space so scores are comparable
    across documents (a term's weight no longer depends on which doc's local idf
    applies)."""
    idf, _ = _corpus_idf(indexes)
    qvec = _weight(dict(Counter(tokenize(query))), idf)
    if not qvec:
        return []
    scored = []
    for index in indexes:
        for rec in index["clauses"]:
            tf = rec.get("tf")
            cvec = _weight(tf, idf) if tf is not None else rec.get("vector", {})
            score = cosine(qvec, cvec)
            if score > 0:
                scored.append(_hit(index, rec, score))
    return scored


def _retrieve_dense(indexes: list[dict], query: str) -> list[dict]:
    """Neural-embedding retrieval. Dense vectors already share one model-defined
    space, so no idf reweighting is needed — the query is embedded once per
    (provider, model) and dotted against each clause vector."""
    scored = []
    qcache: dict[tuple, list] = {}
    for index in indexes:
        model = index.get("model", {})
        key = (model.get("provider") or index.get("backend"), model.get("model"))
        if key not in qcache:
            qcache[key] = _rebuild_backend(index).embed(query)
        qvec = qcache[key]
        for rec in index["clauses"]:
            score = similarity(qvec, rec["vector"])
            if score > 0:
                scored.append(_hit(index, rec, score))
    return scored


def retrieve(indexes: list[dict], query: str, top_k: int) -> list[dict]:
    """Score the query against every clause in every index and return the global
    top-k. tf-idf and neural-embedding indexes are each scored in their own
    space, then merged (a corpus is normally uniform; mixing only happens mid-
    migration, where within-document ranking still holds)."""
    dense = [ix for ix in indexes if is_dense_index(ix)]
    sparse = [ix for ix in indexes if not is_dense_index(ix)]
    scored: list[dict] = []
    if sparse:
        scored += _retrieve_sparse(sparse, query)
    if dense:
        scored += _retrieve_dense(dense, query)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _index_edges(edges: list[dict]):
    """Build lookup tables from the merged graph for fast relationship walks."""
    clause_to_concept: dict[str, dict] = {}   # clause node -> MAPS_TO_CONCEPT edge
    concept_to_clauses: dict[str, list] = {}   # concept node -> [clause edges]
    concept_parent: dict[str, str] = {}        # concept node -> parent concept node
    clause_obligation: dict[str, str] = {}     # clause node -> obligation type node
    for e in edges:
        if e["type"] == "MAPS_TO_CONCEPT":
            clause_to_concept[e["source"]] = e
            concept_to_clauses.setdefault(e["target"], []).append(e)
        elif e["type"] == "CHILD_OF":
            concept_parent[e["source"]] = e["target"]
        elif e["type"] == "HAS_OBLIGATION":
            clause_obligation[e["source"]] = e["target"]
    return clause_to_concept, concept_to_clauses, concept_parent, clause_obligation


def reason(hits: list[dict], nodes: dict, edges: list[dict]) -> tuple[list[dict], list[str], list[dict]]:
    """Enrich each hit with graph context and surface cross-document 'impact
    candidates' — clauses in OTHER documents mapped to the same concept."""
    c2concept, concept2clauses, parent, c2obl = _index_edges(edges)
    evidence = []
    concepts_seen: dict[str, dict] = {}

    for h in hits:
        clause_node = f"Clause:{h['document_id']}/{h['clause_id']}"
        item = dict(h)
        edge = c2concept.get(clause_node)
        if edge:
            concept_node = edge["target"]
            cprops = nodes.get(concept_node, {}).get("properties", {})
            prov = edge.get("provenance", {})
            item["canonical_id"] = cprops.get("canonical_id")
            item["canonical_label"] = cprops.get("label")
            item["mapping_confidence"] = prov.get("match_confidence")
            item["review_required"] = prov.get("review_required")
            parent_node = parent.get(concept_node)
            item["concept_parent"] = nodes.get(parent_node, {}).get("properties", {}).get("label") if parent_node else None
            concepts_seen[concept_node] = cprops
        obl_node = c2obl.get(clause_node)
        if obl_node:
            item["obligation_type"] = nodes.get(obl_node, {}).get("properties", {}).get("label")
        evidence.append(item)

    # Impact candidates: for each concept our evidence touched, list clauses in
    # OTHER documents mapped to the same concept (regulatory cross-reference).
    hit_docs = {h["document_id"] for h in hits}
    impact = []
    for concept_node, cprops in concepts_seen.items():
        for e in concept2clauses.get(concept_node, []):
            prov = e.get("provenance", {})
            if prov.get("document_id") not in hit_docs:
                impact.append({
                    "concept": cprops.get("label"), "canonical_id": cprops.get("canonical_id"),
                    "document_id": prov.get("document_id"), "clause_no": prov.get("clause_no"),
                    "page": prov.get("page"),
                })

    # Human-readable reasoning path.
    path = [
        f"Retrieved {len(hits)} clause(s) by semantic similarity across the indexed corpus.",
    ]
    if concepts_seen:
        labels = sorted({c.get("label") for c in concepts_seen.values() if c.get("label")})
        path.append(f"Evidence maps to {len(concepts_seen)} canonical concept(s): {', '.join(labels)}.")
    parents = sorted({e.get("concept_parent") for e in evidence if e.get("concept_parent")})
    if parents:
        path.append(f"Those concepts sit under parent concept(s): {', '.join(parents)}.")
    if impact:
        path.append(f"Found {len(impact)} related clause(s) in other documents sharing these concepts (cross-reference candidates).")
    n_review = sum(1 for e in evidence if e.get("review_required"))
    if n_review:
        path.append(f"{n_review} of {len(evidence)} evidence clause(s) rely on a concept mapping still pending compliance review.")
    return evidence, path, impact


def confidence(hits: list[dict], evidence: list[dict], dense: bool = False) -> dict:
    top = hits[0]["score"] if hits else 0.0
    mean = round(sum(h["score"] for h in hits) / len(hits), 4) if hits else 0.0
    strong, moderate = (DENSE_STRONG, DENSE_MODERATE) if dense else (STRONG, MODERATE)
    band = "High" if top >= strong else "Medium" if top >= moderate else "Low"
    n_review = sum(1 for e in evidence if e.get("review_required"))
    notes = []
    if band == "Low":
        kind = "semantic" if dense else "lexical"
        notes.append(f"Weak {kind} match — the corpus may not directly address this query.")
    if n_review:
        notes.append(f"{n_review} evidence clause(s) use draft concept mappings not yet compliance-approved.")
    return {"retrieval_band": band, "top_score": top, "mean_score": mean,
            "retrieval_mode": "embedding" if dense else "tfidf",
            "evidence_pending_review": n_review, "notes": notes}


CITATION_RE = re.compile(r"\[([^\]\[]+?)\s+([0-9]+(?:\.[0-9]+)*)\]")


def synthesize(query: str, evidence: list[dict]) -> tuple[str | None, list[str]]:
    """Optional LLM synthesis. Returns (answer, ungrounded_citations). The answer
    may cite only evidence clause numbers; citations outside the evidence set are
    reported so callers can flag ungrounded claims."""
    load_dotenv()
    if not nexus_llm.is_configured():
        return None, []

    lines = []
    for i, e in enumerate(evidence, 1):
        # Prefer the full clause text (attached by attach_clause_text); fall back
        # to the title only if text is unavailable (e.g. clauses file missing).
        body = (e.get("text") or e.get("title") or "").strip()
        tags = e.get("obligation_type") or e.get("canonical_label") or e.get("concept")
        lines.append(
            f"[{i}] document={e['document_id']} clause_no={e['clause_no']} "
            f"({tags}, p.{e.get('page')}):\n{body[:800]}"
        )
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nEvidence clauses:\n" + "\n".join(lines)},
    ]
    try:
        answer = nexus_llm.call_chat(messages)
    except Exception as exc:
        return f"(LLM synthesis failed: {exc})", []

    allowed = {(e["document_id"], str(e["clause_no"])) for e in evidence}
    ungrounded = []
    for doc, clause_no in CITATION_RE.findall(answer):
        # Accept a citation if the clause_no matches any evidence item (doc id
        # spelling from the model may differ slightly).
        if not any(clause_no == str(e["clause_no"]) for e in evidence):
            ungrounded.append(f"[{doc} {clause_no}]")
    return answer, ungrounded


def answer_query(query: str, nexus_dir: Path, top_k: int, use_llm: bool, only_stem: str | None) -> dict:
    indexes, nodes, edges, stems_by_doc = load_corpus(nexus_dir, only_stem)
    if not indexes:
        raise SystemExit(f"No indexes found under {nexus_dir/'index'} (run build_index.py first)")

    hits = retrieve(indexes, query, top_k)
    evidence, path, impact = reason(hits, nodes, edges)
    attach_clause_text(evidence, nexus_dir, stems_by_doc)
    dense = any(is_dense_index(ix) for ix in indexes)
    conf = confidence(hits, evidence, dense=dense)

    answer, ungrounded = (None, [])
    if use_llm and evidence:
        answer, ungrounded = synthesize(query, evidence)
        if ungrounded:
            conf["notes"].append(f"LLM cited {len(ungrounded)} clause(s) not in the evidence set: {', '.join(ungrounded)} — treat as ungrounded.")

    return {
        "query": query,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {"documents_indexed": len(indexes)},
        "answer": answer,
        "evidence": evidence,
        "impact_candidates": impact,
        "reasoning_path": path,
        "confidence": conf,
        "ungrounded_citations": ungrounded,
    }


def _print(result: dict) -> None:
    print(f"\nQ: {result['query']}\n")
    if result["answer"]:
        print(result["answer"].strip() + "\n")
    else:
        print("(No LLM configured - structured evidence only; nothing synthesized.)\n")
    print("Evidence:")
    for e in result["evidence"]:
        flag = " [pending review]" if e.get("review_required") else ""
        print(f"  [{e['score']:.3f}] {e['document_id']} clause {e['clause_no']} (p.{e.get('page')})"
              f" - {e.get('canonical_label') or e.get('concept')}{flag}")
        if e.get("title"):
            print(f"          {e['title'].strip()[:110]}")
    if result["impact_candidates"]:
        print("\nCross-document impact candidates (same concept, other regulations):")
        for c in result["impact_candidates"][:8]:
            print(f"  - {c['document_id']} clause {c['clause_no']} - {c['concept']}")
    print("\nReasoning path:")
    for step in result["reasoning_path"]:
        print(f"  - {step}")
    c = result["confidence"]
    print(f"\nConfidence: {c['retrieval_band']} (top={c['top_score']}, mean={c['mean_score']})")
    for note in c["notes"]:
        print(f"  ! {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question over the corpus using the graph + index (Nexus Step 6).")
    parser.add_argument("--query", required=True, help="Natural-language question")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--top-k", type=int, default=6, help="Number of evidence clauses to retrieve (default: 6)")
    parser.add_argument("--stem", default=None, help="Restrict reasoning to a single document stem")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis; return structured evidence only")
    args = parser.parse_args()

    result = answer_query(args.query, Path(args.nexus), args.top_k, not args.no_llm, args.stem)

    (Path(args.nexus) / "answers").mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.nexus) / "answers" / f"answer_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print(result)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
