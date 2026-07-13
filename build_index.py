"""
Nexus Step 5B — Embedding / Semantic-Retrieval Index.

Step 4 gives every clause an *exact* canonical concept — great for "show every
clause mapped to CPT_CREDIT_RISK", useless for "find clauses that read like this
one" when the wording never matched a concept. Step 5B is that complementary
layer: it turns each clause into a vector and supports similarity search, so
Step 6 can retrieve relevant clauses by meaning, not just by exact concept.

Pluggable backend
-----------------
Two backends live behind one tiny `Backend` interface, chosen with `--backend`:

  * tfidf    (default) — dependency-free, offline, zero-cost lexical vectors.
               Matches on SHARED WORDS only, so a paraphrased question or a
               "what is this about" query retrieves poorly. Good enough for a
               quick offline run; produces SPARSE term->weight vectors.

  * azure / nvidia — real neural embeddings from a hosted model (Azure OpenAI
               `text-embedding-3-small` by default, or an NVIDIA embedding
               model). Matches on MEANING, so paraphrases and topical questions
               retrieve well, and — unlike tfidf — vectors are directly
               comparable ACROSS documents without any shared-idf trick, because
               they already share one model-defined space. Produces DENSE
               float-vector embeddings. Requires credentials in .env (see the
               EmbeddingBackend docstring). Pick the provider explicitly, or use
               `--backend embedding` to auto-select whichever is configured.

Each clause is embedded over an ENRICHED text = title + body + canonical concept
+ object, so the semantic tags from Steps 3-4 sharpen retrieval rather than
sitting unused. Vectors are L2-normalized, so cosine similarity is a plain dot
product at query time.

Usage:
    python build_index.py --stem <file-stem>                      # tfidf (offline)
    python build_index.py --stem <file-stem> --backend azure      # neural embeddings
    python build_index.py --stem <file-stem> --backend embedding  # auto-pick provider
    python build_index.py --stem <file-stem> --query "credit risk monitoring"
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Small, domain-neutral stopword set. Kept inline (no nltk dependency); regulator
# text is formal so this modest list is enough to cut noise without hurting recall.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with",
    "as", "at", "is", "are", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "which", "such", "any", "all", "shall", "must", "may",
    "will", "must", "not", "no", "if", "than", "then", "from", "into", "under",
    "over", "per", "each", "other", "including", "include", "includes", "e", "g",
    "i", "ii", "iii", "iv", "v", "who", "whom", "their", "they", "them", "we",
    "our", "you", "your", "he", "she", "his", "her",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


# --- Embedding provider (Azure OpenAI / NVIDIA, OpenAI-compatible) -----------
# Read at call time (not import time) so a caller that runs load_dotenv() first
# gets its credentials. Azure `text-embedding-3-small` is the default when
# configured: it is symmetric (no query/passage distinction) and robust. NVIDIA
# retrieval models are asymmetric, so we pass input_type=query|passage for them.

EMBED_BATCH = 64
EMBED_MAX_CHARS = 8000


def _embed_provider(pref: str | None = None) -> dict:
    load_dotenv()
    pref = (pref or os.environ.get("NEXUS_EMBED_PROVIDER") or "").strip().lower()

    azure_ok = all(os.environ.get(k) for k in (
        "AZURE_OPENAI_EMBEDDINGS_ENDPOINT", "AZURE_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_API_KEY"))
    nvidia_ok = all(os.environ.get(k) for k in ("NVIDIA_BASE_URL", "NVIDIA_API_KEY"))

    if pref in ("", "embedding", "auto"):
        pref = "azure" if azure_ok else "nvidia" if nvidia_ok else ""

    if pref == "azure":
        if not azure_ok:
            raise RuntimeError(
                "Azure embeddings not configured — set AZURE_OPENAI_EMBEDDINGS_ENDPOINT, "
                "AZURE_EMBEDDING_DEPLOYMENT and AZURE_OPENAI_API_KEY in .env")
        endpoint = os.environ["AZURE_OPENAI_EMBEDDINGS_ENDPOINT"].strip().strip('"').rstrip("/")
        deployment = os.environ["AZURE_EMBEDDING_DEPLOYMENT"].strip().strip('"')
        version = (os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-05-01-preview").strip().strip('"')
        return {
            "provider": "azure", "model": deployment,
            "url": f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version={version}",
            "headers": {"api-key": os.environ["AZURE_OPENAI_API_KEY"].strip(), "Content-Type": "application/json"},
            "asymmetric": False,
        }
    if pref == "nvidia":
        if not nvidia_ok:
            raise RuntimeError(
                "NVIDIA embeddings not configured — set NVIDIA_BASE_URL and NVIDIA_API_KEY in .env")
        base = os.environ["NVIDIA_BASE_URL"].strip().strip('"').rstrip("/")
        model = (os.environ.get("NVIDIA_EMBED_MODEL") or "nvidia/nv-embedqa-e5-v5").strip().strip('"')
        return {
            "provider": "nvidia", "model": model,
            "url": base + "/embeddings",
            "headers": {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY'].strip()}", "Content-Type": "application/json"},
            "asymmetric": True,
        }
    raise RuntimeError(
        "No embedding provider configured. Set AZURE_OPENAI_EMBEDDINGS_ENDPOINT/"
        "AZURE_EMBEDDING_DEPLOYMENT/AZURE_OPENAI_API_KEY (or NVIDIA_BASE_URL/NVIDIA_API_KEY) "
        "in .env, or use --backend tfidf.")


def _embed_texts(texts: list[str], is_query: bool, cfg: dict) -> list[list[float]]:
    """Return one dense vector per input text, in the same order, batched."""
    out: list[list[float]] = []
    with httpx.Client(timeout=120) as client:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = [(t or " ")[:EMBED_MAX_CHARS] for t in texts[start:start + EMBED_BATCH]]
            body: dict = {"input": batch, "model": cfg["model"]}
            if cfg["asymmetric"]:
                body["input_type"] = "query" if is_query else "passage"
                body["truncate"] = "END"
            resp = client.post(cfg["url"], headers=cfg["headers"], json=body)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
    return out


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


# --- Backends ----------------------------------------------------------------

class Backend:
    """Minimal embedding-backend interface. `fit` learns any corpus-level state
    from the clause texts; `embed_passages` vectorizes the clauses to index;
    `embed` vectorizes one query. Vectors are either sparse term->weight dicts
    (is_dense=False) or dense float lists (is_dense=True), L2-normalized."""

    name = "base"
    is_dense = False

    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError

    def embed_passages(self, texts: list[str]) -> list:
        return [self.embed(t) for t in texts]

    def embed(self, text: str):
        raise NotImplementedError

    def serialize(self) -> dict:
        raise NotImplementedError


class TfidfBackend(Backend):
    """Dependency-free TF-IDF. idf = ln((N+1)/(df+1)) + 1 (smoothed); tf is raw
    term count. Vectors are sparse dicts, L2-normalized so cosine = dot product."""

    name = "tfidf"
    is_dense = False

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.n_docs = 0

    def fit(self, texts: list[str]) -> None:
        docs = [tokenize(t) for t in texts]
        self.n_docs = len(docs)
        df: Counter = Counter()
        for tokens in docs:
            for term in set(tokens):
                df[term] += 1
        self.idf = {term: math.log((self.n_docs + 1) / (d + 1)) + 1.0 for term, d in df.items()}

    def embed(self, text: str) -> dict[str, float]:
        tf = Counter(tokenize(text))
        vec = {term: count * self.idf.get(term, 0.0) for term, count in tf.items()}
        vec = {t: w for t, w in vec.items() if w > 0.0}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm > 0 else vec

    def serialize(self) -> dict:
        return {"idf": self.idf, "n_docs": self.n_docs, "vector_type": "sparse"}


class EmbeddingBackend(Backend):
    """Neural embeddings via a hosted, OpenAI-compatible endpoint.

    Provider is chosen from .env (see _embed_provider): Azure OpenAI
    `text-embedding-3-small` by default, or an NVIDIA embedding model. Pass a
    concrete provider ("azure"/"nvidia") to force one, or None to auto-select.
    On restore from a saved index, the stored provider/model are re-applied so
    queries embed in exactly the same space as the indexed clauses."""

    is_dense = True

    def __init__(self, provider: str | None = None):
        self._provider_pref = provider
        self._model_override: str | None = None
        self._cfg: dict | None = None
        self.dim: int | None = None

    @property
    def name(self) -> str:
        return self.cfg["provider"]

    @property
    def cfg(self) -> dict:
        if self._cfg is None:
            self._cfg = _embed_provider(self._provider_pref)
            if self._model_override:
                self._cfg = {**self._cfg, "model": self._model_override}
        return self._cfg

    def fit(self, texts: list[str]) -> None:
        # Nothing to learn locally — the representation lives in the hosted model.
        pass

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vecs = [_l2(v) for v in _embed_texts(texts, is_query=False, cfg=self.cfg)]
        if vecs:
            self.dim = len(vecs[0])
        return vecs

    def embed(self, text: str) -> list[float]:
        vec = _l2(_embed_texts([text], is_query=True, cfg=self.cfg)[0])
        self.dim = len(vec)
        return vec

    def serialize(self) -> dict:
        return {"provider": self.cfg["provider"], "model": self.cfg["model"],
                "dim": self.dim, "vector_type": "dense"}


# `embedding` auto-selects a configured provider; `azure`/`nvidia` force one.
BACKENDS = {
    "tfidf": lambda: TfidfBackend(),
    "embedding": lambda: EmbeddingBackend(None),
    "azure": lambda: EmbeddingBackend("azure"),
    "nvidia": lambda: EmbeddingBackend("nvidia"),
}


def _enriched_text(clause: dict) -> str:
    sem = clause.get("semantic") or {}
    parts = [
        clause.get("title") or "", clause.get("text") or "",
        sem.get("concept") or "", sem.get("object") or "",
    ]
    return " ".join(p for p in parts if p)


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    # Sparse cosine: both unit-normalized, so it's the dot product over the
    # smaller vector's terms.
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def similarity(qvec, cvec) -> float:
    """Cosine similarity for either representation: dense lists dot-product
    elementwise; sparse dicts use the term-overlap cosine above."""
    if isinstance(qvec, list):
        return sum(x * y for x, y in zip(qvec, cvec))
    return cosine(qvec, cvec)


def is_dense_index(index: dict) -> bool:
    return (index.get("model") or {}).get("vector_type") == "dense" or index.get("vector_type") == "dense"


def build_index(stem: str, nexus_dir: Path, backend_name: str) -> dict:
    clauses_path = nexus_dir / "clauses" / f"{stem}_clauses.json"
    if not clauses_path.exists():
        raise SystemExit(f"Clauses file not found: {clauses_path} (run export_clauses.py first)")

    with open(clauses_path, encoding="utf-8") as f:
        doc = json.load(f)
    clauses = doc.get("clauses", [])

    backend = BACKENDS[backend_name]()
    texts = [_enriched_text(c) for c in clauses]
    backend.fit(texts)
    vectors = backend.embed_passages(texts)

    records = []
    empty = 0
    for clause, text, vec in zip(clauses, texts, vectors):
        if not vec:
            empty += 1
        sem = clause.get("semantic") or {}
        rec = {
            "clause_id": clause.get("clause_id"),
            "clause_no": clause.get("clause_no"),
            "title": clause.get("title"),
            "page": clause.get("page_reference"),
            "concept": sem.get("concept"),
            "domain": sem.get("domain"),
            "vector": vec,
        }
        # Raw term frequencies are only meaningful for the sparse tfidf backend,
        # where a corpus-level consumer (Step 6) re-weights this clause with a
        # SHARED idf across documents. Dense embeddings are already in one shared
        # space, so they need no tf and we omit it to keep the file small.
        if not backend.is_dense:
            rec["tf"] = dict(Counter(tokenize(text)))
        records.append(rec)

    model = backend.serialize()
    index = {
        "document_id": doc.get("document_id") or stem,
        "backend": backend.name,
        "vector_type": "dense" if backend.is_dense else "sparse",
        "embedding_model": model.get("model"),
        "dim": model.get("dim"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clause_count": len(records),
        "vocabulary_size": len(model.get("idf", {})),
        "model": model,
        "clauses": records,
    }
    return index


def write_index(stem: str, nexus_dir: Path, index: dict) -> Path:
    (nexus_dir / "index").mkdir(parents=True, exist_ok=True)
    out_path = nexus_dir / "index" / f"{stem}_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    log = {
        "document": stem, "document_id": index["document_id"],
        "backend": index["backend"], "vector_type": index["vector_type"],
        "embedding_model": index.get("embedding_model"), "dim": index.get("dim"),
        "generated_at": index["generated_at"],
        "clause_count": index["clause_count"], "vocabulary_size": index["vocabulary_size"],
    }
    (nexus_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = nexus_dir / "logs" / f"{stem}_index_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return out_path


def _rebuild_backend(index: dict) -> Backend:
    backend = BACKENDS.get(index["backend"], BACKENDS["tfidf"])()
    model = index.get("model", {})
    if isinstance(backend, TfidfBackend):
        backend.idf = model.get("idf", {})
        backend.n_docs = model.get("n_docs", 0)
    elif isinstance(backend, EmbeddingBackend):
        # Pin the query to the exact provider/model the clauses were indexed with.
        backend._provider_pref = model.get("provider") or index.get("backend")
        backend._model_override = model.get("model")
        backend.dim = model.get("dim")
    return backend


def query_index(stem: str, nexus_dir: Path, query: str, top_k: int) -> list[dict]:
    """Load a prebuilt index (building a tfidf one if absent) and return the
    top-k clauses most similar to the query text."""
    index_path = nexus_dir / "index" / f"{stem}_index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = build_index(stem, nexus_dir, "tfidf")

    backend = _rebuild_backend(index)
    qvec = backend.embed(query)
    scored = []
    for rec in index["clauses"]:
        score = similarity(qvec, rec["vector"])
        if score > 0:
            scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"score": round(s, 4), "clause_no": r["clause_no"], "title": r["title"],
         "concept": r["concept"], "page": r["page"], "clause_id": r["clause_id"]}
        for s, r in scored[:top_k]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build (or query) a semantic-retrieval index over a document's clauses (Nexus Step 5B).")
    parser.add_argument("--stem", required=True, help="File stem shared by Nexus/clauses/<stem>_clauses.json etc.")
    parser.add_argument("--nexus", default="Nexus", help="Nexus root directory (default: Nexus)")
    parser.add_argument("--backend", default="tfidf", choices=sorted(BACKENDS),
                        help="Vectorizer: tfidf (offline lexical) or azure/nvidia/embedding (neural). Default: tfidf")
    parser.add_argument("--query", default=None, help="If given, search the index for this text instead of (re)building it")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return in --query mode (default: 5)")
    args = parser.parse_args()

    if args.query:
        results = query_index(args.stem, Path(args.nexus), args.query, args.top_k)
        if not results:
            print(f"No clauses matched: {args.query!r}")
            return
        print(f"Top {len(results)} clauses for {args.query!r}:")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['clause_no']}  {r['title'] or '(untitled)'}"
                  f"  - concept={r['concept']} (p.{r['page']})")
        return

    index = build_index(args.stem, Path(args.nexus), args.backend)
    out_path = write_index(args.stem, Path(args.nexus), index)
    print(f"Wrote {out_path}")
    detail = (f"vocabulary {index['vocabulary_size']} terms" if index["vector_type"] == "sparse"
              else f"{index['dim']}-dim {index['embedding_model']} embeddings")
    print(f"  {index['clause_count']} clauses indexed, {detail} (backend: {index['backend']})")


if __name__ == "__main__":
    main()
