# Nexus Regulatory-Document Pipeline — Internal Review

**Status:** Draft for internal review
**Date:** 2026-07-13
**Scope:** Steps 1 → 6 of the Nexus pipeline, with a deep focus on **Step 4 (Ontology Mapping)** — its limitations, its LLM-vs-deterministic design decision, folder architecture, where the ontology is stored, and open blockers.
**Corpus under test:** 5 regulator PDFs (1 CBUAE + 3 RBI + sample), driven end-to-end.

---

## 0. Executive summary

Nexus turns a regulator PDF into a queryable, auditable set of clause-level
regulatory obligations mapped to a controlled banking vocabulary. It is a
**multi-stage pipeline** where each stage writes its own immutable artifact and
never mutates the previous stage's output.

The central design tension — and the main subject of this review — is **Step 4
(Ontology Mapping)**: mapping each clause's free-text concept onto a stable
canonical concept ID. We deliberately do **not** let the LLM be the final
mapper. This document explains why (auditability, reproducibility, no invented
IDs, cost), documents a concrete failure we observed (a PDF that mapped 100%
on one run mapped 17.3% on a fresh run), and states the honest conclusion: the
durable path is the **semantic `--suggest` + human-approval loop**, not
hand-harvested exact strings.

---

## 1. Pipeline at a glance

| Step | Script | What it answers | Deterministic? |
|---|---|---|---|
| 1. Extract | `extract_document.py` | *What is this document?* (text + metadata) | Text/stats yes; metadata fields via 1 LLM call |
| 2. Structure | `structure_document.py` | *How is it organized?* (nested tree) | Mostly; LLM only for ambiguous headings |
| 3. Interpret | `interpret_document.py` | *What does each clause mean?* (semantic tags) | **No — every clause goes through the LLM** |
| 3.5. Export | `export_clauses.py` | *Flatten the tree into clause records* | Yes |
| 4. Ontology | `build_ontology.py` | *Map free-text concept → canonical ID* | **Yes (deterministic matcher); LLM only in opt-in `--suggest`** |
| 5A. Graph | `build_graph.py` | *Knowledge graph of clauses/concepts/regulators* | Yes |
| 5B. Index | `build_index.py` | *TF-IDF retrieval index* | Yes |
| 6. Answer | `answer_query.py` | *Answer a question with grounded evidence* | Retrieval yes; synthesis via LLM (optional) |

Shared infrastructure:
- `nexus_llm.py` — provider-fallback LLM client (**NVIDIA primary**, Azure/grok fallback).
- `nexus_ontology.py` — loads and merges the layered ontology (global + jurisdiction) at runtime.

---

## 2. Step-by-step: input, output, and summary

### Step 1 — `extract_document.py` (Document Acquisition & Metadata)

- **Input:** one regulator PDF (`--pdf path/to/file.pdf`), optional seeds
  (`--source-url`, `--regulator`, `--country`, `--source-type`).
- **Output:**
  - `Nexus/input/<stem>.pdf` — copy of the source (audit trail)
  - `Nexus/markdown/<stem>.md` — extracted text, lightly structured (headings tagged `## `)
  - `Nexus/metadata/<stem>_metadata.json` — 5-section metadata record
  - `Nexus/logs/<stem>_log.json` — per-step processing log
- **Summary:** Uses `pypdf` for text extraction (no layout/OCR engine, so
  `total_tables` is reported `null`, not guessed). Deterministic outputs (text,
  hashes, word/char/page counts) never depend on the LLM. Interpreted metadata
  fields (document_id, dates, version, status, supersedes, keywords, …) are
  filled by **one LLM call**, instructed to return `null` for anything not
  explicitly stated — nothing is inferred. If the LLM call fails, those fields
  stay `null` and the run still completes (graceful degradation).

### Step 2 — `structure_document.py` (Structural Segmentation)

- **Input:** `Nexus/markdown/<stem>.md` (+ `<stem>_metadata.json` for the doc id).
- **Output:** `Nexus/structure/<stem>_structure.json` — a nested
  `Document → Part → Section → Subsection → Clause` tree (Table/Annex/Footnote
  nodes attached where they occur). Plus `Nexus/logs/<stem>_structure_log.json`.
- **Summary:** Hybrid approach. A **deterministic pass** classifies each heading
  by numbering convention (PART/CHAPTER, `1.`, `1.1`, `(a)`/`(i)`, Table/Annex,
  Note). Headings that match no known convention are batched into **one LLM
  call** that assigns a structural level (skippable via `--skip-llm`, which
  defaults them to `clause`). A stack-based pass then assembles the tree by
  numbering depth. **Known limitation:** table/footnote *boundaries* are
  heuristic (a node's text runs until the next heading), because Step 1 has no
  table detection.

### Step 3 — `interpret_document.py` (Regulatory Meaning)

- **Input:** `Nexus/structure/<stem>_structure.json`.
- **Output:** `Nexus/semantics/<stem>_semantics.json` — a deep copy of the
  structure tree with 7 fields added to each operative node: `domain`,
  `concept`, `action`, `object`, `obligation_type`, `mandatory_flag`,
  `confidence_score`. Plus `Nexus/logs/<stem>_semantics_log.json`.
- **Summary:** The only **fully interpretive** step — every operative unit goes
  through the LLM (no deterministic path). To keep labels comparable across
  documents, `domain`, `action`, and `obligation_type` are constrained to a
  **fixed taxonomy** (with an `Other`/`OTHER` escape hatch) whose enums are
  *stamped into the JSON schema from the merged ontology at load time*, so they
  can never drift from the ontology. `concept` and `object` stay **free text**
  — and this free text is exactly what Step 4 has to grapple with. Batches are
  validated against a response schema; violations are logged but coerced to
  allowed values so the artifact is always schema-clean.
- **Critical property for Step 4:** Step 3 output is **non-deterministic**.
  Across the corpus it produced **298 distinct concept strings for 314 clauses**
  — almost one unique label per clause.

### Step 3.5 — `export_clauses.py` (Flat Clause Export)

- **Input:** `Nexus/semantics/<stem>_semantics.json`.
- **Output:** `Nexus/clauses/<stem>_clauses.json` — a flat list of clause
  records (`clause_id`, `clause_no`, `title`, `text`, `page_reference`,
  `semantic{...}`), one per numbered provision.
- **Summary:** Flattens the tree. Every numbered provision becomes a record;
  structural wrappers (part/annex) and unnumbered noise are dropped. Domain
  labels are converted to compact tokens (e.g. `AML_KYC`). False-positive
  numbers from pypdf (bare years like `1934`, mid-sentence fragments) are
  filtered. Flags "genuine misses" — clauses with their own text but no
  semantics — as a signal of an LLM outage during Step 3.

### Step 4 — `build_ontology.py` (Ontology Object Creation) — *focus of this review*

- **Input:** `Nexus/clauses/<stem>_clauses.json`.
- **Output:** `Nexus/ontology/<stem>_ontology.json` — one ontology object per
  clause, plus a coverage summary; `Nexus/logs/<stem>_ontology_log.json`.
- **Summary:** Maps each clause's **free-text** `concept` onto a **controlled
  canonical concept** (stable `CPT_<SLUG>` ID) from the layered ontology. **The
  one rule that makes this trustworthy: the matcher never invents a canonical
  ID** — it only assigns IDs that already exist in the human-owned catalog.
  Anything it can't match is `unmapped` and `review_required: true`. Details in
  §3–§5.

### Step 5A — `build_graph.py` (Knowledge Graph)

- **Input:** clauses + ontology objects.
- **Output:** `Nexus/graph/<stem>_graph.json` — node-link graph
  (Document/Regulator/Domain/Concept/ObligationType/Clause nodes;
  ISSUED_BY / HAS_CLAUSE / MAPS_TO_CONCEPT / IN_DOMAIN / HAS_OBLIGATION /
  CHILD_OF edges). Concept/Domain/Regulator node IDs are **global**, so per-doc
  graphs union into a corpus graph. Every evidence edge carries provenance.

### Step 5B — `build_index.py` (Retrieval Index)

- **Input:** clauses.
- **Output:** `Nexus/index/<stem>_index.json` — TF-IDF index (no new deps,
  pluggable `Backend` for real embeddings later). Records store raw `tf` so a
  shared-corpus idf can be computed later. `--query "..."` runs live top-k.

### Step 6 — `answer_query.py` (Grounded Q&A)

- **Input:** a natural-language query + the 5A graph and 5B index.
- **Output:** `Nexus/answers/answer_<ts>.json` — answer with evidence,
  reasoning path, and confidence.
- **Summary:** Retrieves top-k across the corpus using a **shared corpus idf**
  (critical fix: per-doc idf made cross-document ranking invalid), walks the
  graph for concept/hierarchy/obligation + cross-doc "impact candidates" (same
  concept, other regulators), then optional LLM synthesis with a
  **citation-grounding post-check** (flags citations not in the evidence set).
  `--no-llm` degrades to structured evidence. Always emits
  evidence + reasoning_path + confidence.

---

## 3. Step 4 deep dive — how the matcher works

For each clause concept, in strict precedence order (first match wins):

1. **`exact_label`** (confidence 1.0) — normalized concept == a canonical label.
2. **`synonym`** (0.95) — normalized concept == a listed synonym.
3. **`normalized`** (0.85) — matches only after casefold / whitespace-collapse /
   trailing-punctuation strip.
4. **`fuzzy`** (≤ 0.85) — idf-weighted token-cosine over every catalog surface
   form (label + synonyms), with light stemming. Generic tokens (`credit`,
   `risk`, `management`) are down-weighted; distinctive tokens (`collateral`,
   `staging`, `sicr`) carry the signal. **Tried only after the exact tiers
   miss.** A fuzzy match always scores below `auto_accept` (0.90), so it is
   **always routed to human review — never silently trusted.**
5. **`unmapped`** (0.0) — no match. `canonical_id: null`, `review_required: true`.

**Everything is currently `review_required: true`** regardless of confidence,
because `matching_rules.yaml → matched_concept_is_draft: true` and every
concept in the catalog is `status: draft`. That is intentional — see §7.

Config that governs all of this: `Nexus/config/step4_ontology/matching_rules.yaml`
(normalization, precedence, fuzzy thresholds, review policy, suggestion floor).

---

## 4. Step 4 limitations — what we observed (the core finding)

### 4.1 The 100% → 17.3% collapse

**The same PDF that mapped 100% earlier mapped only 17.3% on a fresh run.**

- **Cause:** the synonyms harvested into the catalog were the *exact free-text
  strings* produced by the **first** Step-3 run. A **second** LLM run phrases
  the same concepts differently. Where the catalog had `"Credit Risk Framework"`
  and `"SICR Indicators"`, the fresh run produced
  `"Credit Risk Framework Approval"` and `"SICR Credit Performance Indicators"`.
  Exact/synonym string-matching misses the variants.

- **What this means:** the earlier "100% coverage" was **real but overfit to one
  run.** Exact-synonym harvesting does **not generalize across re-runs**, because
  Step 3's output is non-deterministic. Hand-harvesting exact strings can never
  be the whole answer.

### 4.2 The mitigation we added — the fuzzy tier

A deterministic **fuzzy fallback** (idf-weighted token cosine + light stemming)
was added to absorb phrasing variants without an LLM at run time. Effect on the
fresh run: **17% → 73%** coverage, while the curated CBUAE run stayed at 100%
(no regression). RBI docs improved 26% → 56% (155MD) and 0% → 57% (407MD).
Fuzzy-match precision is roughly 75–80%; because fuzzy matches are always
review-flagged, the wrong ones are **review rejections, not silent data errors.**

### 4.3 The honest conclusion

The fuzzy tier **helps**, but the **durable** path for Step 4 is the
**`--suggest` (semantic) + human-approval loop**, not hand-harvested exact
strings. Exact synonyms are a fast cache for phrasings a human has already
approved; they cannot be the whole answer because the upstream (Step 3) is not
deterministic.

### 4.4 Current coverage snapshot (on-disk logs, 2026-07-13)

| Document | Jurisdiction | Clauses | Mapped | Coverage | Review-required |
|---|---|---|---|---|---|
| CBUAE_EN_5996_VER1 (curated) | AE_CBUAE | 177 | 177 | **100%** | 177 |
| 155MD | IN_RBI | 93 | 52 | 55.9% | 93 |
| 407MD… | IN_RBI | 28 | 16 | 57.1% | 28 |
| NOTI175… | IN_RBI | 1 | 0 | 0.0% | 1 |

CBUAE's catalog is mature (39 concepts, credit-risk + IFRS-9/ECL family); RBI
still needs its own concepts fleshed out in the `IN_RBI` pack.

---

## 5. Why isn't the LLM the sole / default mapper?

The LLM **is** very much in the loop — in `--suggest` (proposing catalog IDs)
and upstream in Step 3. The question is why it isn't the *final* mapper. Four
concrete reasons, all rooted in this being a **regulatory / compliance** system:

1. **Auditability.** A regulator or auditor can ask "why is this clause mapped
   to Credit Risk?" The deterministic matcher answers: *"exact/synonym match to
   a human-approved catalog entry."* An LLM answers: *"the model decided."* The
   first is defensible in a compliance review; the second is not.

2. **Reproducibility.** We saw this live: the same PDF through the LLM twice gave
   different concepts, crashing coverage 100% → 17%. LLMs are non-deterministic.
   If the LLM were the final mapper, the obligation register would silently
   change on every re-run — unacceptable for a compliance record.

3. **No invented IDs.** The deterministic matcher can only assign IDs that exist
   in the human-approved catalog — it **physically cannot hallucinate** a
   concept. The `--suggest` LLM is likewise constrained to real catalog IDs;
   anything else it returns is discarded.

4. **Cost & latency.** The deterministic matcher is instant and free. An LLM
   call per clause per run is neither (Step 3 alone already takes ~2 min).

**The role split, therefore:** LLM *proposes* (Step 3 tagging, Step 4
`--suggest`), human *approves* (promotes concepts/synonyms), deterministic
matcher *decides* at run time using only approved material. The catalog-growth
loop is: `build_ontology.py --stem <s> --suggest` → LLM proposes catalog IDs
for unmapped concepts → a human approves → approved concept/synonym is written
into the catalog → re-run **without** `--suggest` → coverage rises with **no LLM
at run time.**

---

## 6. Folder architecture

```
gdelt-news-monitor/
├── extract_document.py        # Step 1
├── structure_document.py      # Step 2
├── interpret_document.py      # Step 3
├── export_clauses.py          # Step 3.5
├── build_ontology.py          # Step 4        ← focus
├── build_graph.py             # Step 5A
├── build_index.py             # Step 5B
├── answer_query.py            # Step 6
├── nexus_llm.py               # shared LLM client (NVIDIA primary, Azure fallback)
├── nexus_ontology.py          # shared layered-ontology loader/merger
└── Nexus/
    ├── input/                 # copies of source PDFs (audit trail)
    ├── markdown/              # Step 1 extracted text
    ├── metadata/              # Step 1 metadata JSON
    ├── structure/             # Step 2 structural tree
    ├── semantics/             # Step 3 semantic tree
    ├── clauses/               # Step 3.5 flat clause records
    ├── ontology/              # Step 4 ontology objects  ← output
    ├── graph/                 # Step 5A knowledge graphs
    ├── index/                 # Step 5B retrieval indexes
    ├── answers/               # Step 6 Q&A audit records
    ├── logs/                  # per-step processing logs
    ├── docs/                  # this document
    └── config/
        ├── ontology/
        │   ├── global/                    # jurisdiction-neutral core
        │   │   ├── taxonomies.yaml         # obligation_type, action, modality, escape hatch
        │   │   ├── domains.yaml            # neutral domains + compact tokens
        │   │   └── concepts.yaml           # global canonical concept catalog
        │   └── jurisdictions/
        │       ├── registry.yaml           # regulator string → pack code (RBI→IN_RBI, CBUAE→AE_CBUAE)
        │       ├── AE_CBUAE/{concepts,domains}.yaml
        │       └── IN_RBI/{concepts,domains}.yaml
        ├── step3_semantics/   # Step 3 thresholds + schemas + STANDARD.md
        └── step4_ontology/    # Step 4 matching_rules.yaml + schema + STANDARD.md + concept_mappings.jsonl
```

**Layered ontology principle:** one codebase, many regulators. A
jurisdiction-neutral **global** layer plus a per-regulator extension, merged at
runtime by `nexus_ontology` (an extension entry with the same key overrides the
global one). Jurisdiction is auto-detected from the document's Step-1
`regulator` metadata via `registry.yaml`, or forced with `--jurisdiction`.

---

## 7. Where the ontology is saved

The **durable, cross-document asset** is the canonical concept catalog:

- **Global (jurisdiction-neutral):**
  `Nexus/config/ontology/global/concepts.yaml`
- **UAE / CBUAE extension:**
  `Nexus/config/ontology/jurisdictions/AE_CBUAE/concepts.yaml`
- (RBI extension exists at `.../jurisdictions/IN_RBI/concepts.yaml`, still thin.)

Each concept entry: `canonical_id` (`CPT_<SLUG>`, stable forever), `label`,
`definition`, `domain`, `parent` (hierarchy), `synonyms`, `status`
(`approved` | `draft`). **Everything is currently `status: draft`** — the file
header states it is DRAFT until a compliance expert signs off. The per-document
`Nexus/ontology/<stem>_ontology.json` files are just the *link* from a
document's clauses into that catalog; the catalog is the thing that must be
curated and version-controlled.

---

## 8. Blockers and potential issues

### Blockers

1. **Human sign-off is the real completion gate (not code).** Every concept is
   `status: draft`, and `matched_concept_is_draft: true` forces 100% of
   mappings into review. **No mapping is "authoritative" until a compliance
   expert promotes concepts draft → approved.** This is a process blocker, not a
   bug — but nothing downstream should be treated as a compliance record until
   it clears.

2. **RBI catalog is immature.** RBI docs sit at ~56–57% coverage (and one at
   0%). They need their own concepts + a fleshed-out `IN_RBI` pack, harvested via
   the `--suggest` + approval loop.

### Potential issues / risks

3. **Upstream non-determinism poisons any exact-string cache.** Because Step 3
   is non-deterministic, any Step-4 layer built on exact free-text strings will
   silently decay on re-runs (the 100%→17% event). The fuzzy tier softens this;
   the semantic+approval loop is the real fix. Treat coverage numbers from a
   single run as *point-in-time*, not stable.

4. **Fuzzy precision ~75–80%.** ~1 in 4–5 fuzzy matches is wrong. This is
   *contained* (fuzzy matches are always review-flagged, so wrong ones are
   rejected, not stored) — but it means fuzzy coverage is **not** the same as
   trustworthy coverage. Don't report fuzzy-inflated coverage as if it were
   auto-accepted.

5. **LLM provider fragility.** The pipeline routes through `nexus_llm.call_chat`
   with **NVIDIA primary** because the Azure/grok deployment was returning empty
   HTTP 200s in sustained streaks. If Step 3 tagging or Step 4 `--suggest`
   produces null/empty output, run `python test_llm.py` first to confirm which
   providers are healthy before assuming a code fault.

6. **No OCR / no table structure.** `pypdf` extraction means scanned/image PDFs
   yield little text (a warning is logged), and table boundaries are heuristic.
   Clause text can overshoot a table's true extent, which can pollute a clause's
   Step-3 concept and thus its Step-4 mapping.

7. **Coverage denominator excludes header-only clauses.** Step 4 skips clauses
   with no concept, so `coverage_pct` reflects only classified provisions. Good
   for signal, but comparisons across documents must account for it.

---

## 9. Recommendation

- Adopt the **`--suggest` + human-approval loop** as the *official* mechanism
  for growing coverage; treat hand-harvested exact synonyms as a convenience
  cache only, never as the strategy.
- Keep the fuzzy tier (it recovers most re-run drift) but **never** raise its
  `max_confidence` above `auto_accept` — fuzzy must stay review-gated.
- Prioritize a **compliance-expert review pass** to promote the global +
  CBUAE catalogs from `draft` → `approved`; until then, no output is a
  compliance record.
- Build out the `IN_RBI` pack next, using the same loop.
```
