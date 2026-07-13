# Nexus Step 3 — Semantic Classification Standard

**Owner:** Banking compliance expert (meaning) + developer (enforcement).
**Status:** DRAFT — not yet reviewed by a compliance expert.
**Applies to:** `interpret_document.py` (Nexus Step 3).

This document is the authority on *what Step 3 means*. The layered ontology
(`config/ontology/global/` + the document's jurisdiction extension, merged by
`nexus_ontology`), the machine contract (`classification_response.schema.json`,
whose domain/action/obligation/modality enums are stamped from the merged
ontology at load), and the golden set (`approved_clauses.jsonl`) all derive from
what is written here. If this document and the code disagree, this document is
wrong or the code is wrong — they are never both right. Fix one.

---

## 1. What Step 3 does

Step 3 reads the Step-2 structure tree and, for every **operative unit** (a
clause, a leaf node, or a numbered section/subsection that carries its own
body text), attaches seven fields describing what the provision *means*:

| Field | Type | Source of allowed values |
|---|---|---|
| `domain` | controlled enum | `ontology/…/domains.yaml` (global + jurisdiction) |
| `concept` | free text (2–5 words) | model, unconstrained |
| `action` | controlled enum | `ontology/global/taxonomies.yaml → action` |
| `object` | free text (2–5 words) | model, unconstrained |
| `obligation_type` | controlled enum | `ontology/global/taxonomies.yaml → obligation_type` |
| `mandatory_flag` | controlled enum | `ontology/global/taxonomies.yaml → mandatory_flag` |
| `confidence_score` | float 0.0–1.0 | model self-report |

The output is a new artifact, `Nexus/semantics/<stem>_semantics.json` — a deep
copy of the structure tree with these fields added. **Step 3 never mutates the
Step-2 output.**

## 2. What MUST be classified

- Every **clause** node.
- Every **leaf** node (no children) that has its own `text`.
- Every **numbered** section/subsection that has its own `text`, even if it
  also has children.

## 3. What MUST NOT be classified

- Pure grouping headers — a numbered parent whose substance lives entirely in
  its children (e.g. "4.2 Market-makers and users" with no body text of its
  own). These stay unclassified by design.
- Structural wrappers with no operative text (part/annex shells).

## 4. What MUST NOT be inferred

This is the core compliance rule. The model classifies **what the text says**,
never what a reader assumes it implies.

- Do **not** invent an obligation that the clause does not state. A definition
  ("'Commercial Banks' mean …") is `mandatory_flag: Not Applicable`, never
  `Mandatory`, no matter how important the defined term is.
- Do **not** upgrade a discretionary provision ("may", "at its option") to
  `Mandatory`.
- Do **not** guess a `domain` from the document title when the clause text is
  domain-neutral — use `Other`.
- `concept` and `object` must be grounded in the clause's own wording, not in
  outside knowledge of the regulation.

When the text does not support a controlled value, the model MUST fall back to
the escape hatch (`Other` / `OTHER`), not to its best guess.

## 5. Field definitions

- **domain** — the area of financial regulation the clause governs. One of the
  `domain` enum. Use `Other` when no listed domain clearly fits.
- **concept** — a short human label for the regulatory idea, e.g. "Customer Due
  Diligence", "Key Fact Statement". Free text, 2–5 words. Grounded in the text.
- **action** — the single operative verb the clause requires, e.g. `RETAIN`,
  `REPORT`, `DISCLOSE`. One of the `action` enum. `OTHER` if none fit or the
  clause is non-operative (definitional/explanatory).
- **object** — what the action applies to, e.g. "Customer Records". Free text,
  2–5 words.
- **obligation_type** — the regulatory category of the requirement. One of the
  `obligation_type` enum. `Other` when none fit.
- **mandatory_flag** — the binding strength:
  - `Mandatory` — the clause imposes a requirement ("shall", "must").
  - `Discretionary` — permissive ("may", "at its discretion").
  - `Conditional` — applies only if a stated condition holds ("if …, then …").
  - `Not Applicable` — definitional, explanatory, or scope text; no obligation.
- **confidence_score** — the model's own confidence in the *whole*
  classification, 0.0–1.0. Not a measure of clause importance.

## 6. Confidence and escalation rules

`confidence_score` drives human review, not correctness. Thresholds live in
`thresholds.yaml` so they can be tuned without editing this document.

| Band | Meaning | Routing (target behavior — not yet wired) |
|---|---|---|
| `>= auto_accept` | High confidence | Accept without review. |
| `review_floor`–`auto_accept` | Uncertain | Queue for Step 9 human review. |
| `< review_floor` | Low confidence | Mandatory human review before use downstream. |
| `null` | Not classified (LLM skipped/failed) | Treated as lowest band; must be reviewed. |

Any clause whose `domain`, `action`, or `obligation_type` fell back to the
escape hatch (`Other`/`OTHER`) should also be surfaced for review, regardless
of `confidence_score`.

## 7. Failure behavior (already implemented)

Step 3 degrades rather than aborting. If the LLM is unconfigured, a batch
errors, or a response row is missing, the affected units get **all seven
fields set to `null`** and the reason is recorded in
`Nexus/logs/<stem>_semantics_log.json`. A `null`-classified unit is a
review item, never a silent pass.

## 8. Approved examples

The golden set lives in `approved_clauses.jsonl`, one example per line. Each
line is a real clause plus its *expert-approved* classification. It is the
regression set: run it against every prompt or model change and require the
controlled fields (`domain`, `action`, `obligation_type`, `mandatory_flag`) to
match. Free-text fields (`concept`, `object`) are for human comparison, not
exact-match assertion.

Until a compliance expert signs off, example rows carry `"status": "draft"`.
Drafts must not be used to claim the model "passes".
