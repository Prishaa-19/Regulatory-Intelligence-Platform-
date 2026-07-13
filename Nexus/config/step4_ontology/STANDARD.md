# Nexus Step 4 — Ontology Object Creation Standard

**Owner:** Banking compliance expert (the concept catalog) + developer (matching & schema).
**Status:** DRAFT — catalog not yet reviewed by a compliance expert.
**Applies to:** `build_ontology.py` (Nexus Step 4).

## 1. Why this stage exists

Step 3 tags every clause with a **free-text** `concept` (and `object`). Across
the current corpus that produced **298 distinct concept strings for 314
clauses** — almost one unique label per clause. Free text cannot answer
"show every credit-risk-assessment obligation across all regulators", because
"Credit Risk Assessment", "Credit Risk Management", and "Assessment of Credit
Risk" are three different strings.

Step 4 fixes this. It maps each clause's free-text concept onto a **controlled
canonical concept** from the **layered ontology** (`config/ontology/global/concepts.yaml`
plus the document's jurisdiction extension under
`config/ontology/jurisdictions/<CODE>/concepts.yaml`, merged at runtime by
`nexus_ontology`) — a reusable, cross-document banking vocabulary with stable
IDs, synonyms, and a parent/child hierarchy. Concepts that do not map are
**flagged for review, never silently guessed**.

## 2. The one rule that makes this trustworthy

**The model never invents a canonical ID.** Canonical IDs exist only in
`concepts.yaml`. Step 4 assigns an ID by *matching* a clause's free-text
concept against the catalog's labels and synonyms. If nothing matches, the
clause is `unmapped` and `review_required: true`. This is what the quality
gate means: *every approved concept uses a controlled ID; unknowns are flagged.*

Consequences:
- Coverage starts **below 100%** and that is correct — the gate is "unknowns
  are flagged", not "everything is mapped". Low coverage is a signal to grow
  the catalog (with expert review), not to loosen matching.
- Growing the catalog is a compliance act: adding a canonical concept or a
  synonym is editing `concepts.yaml`, owned by the business team.

## 3. What a canonical concept is (`concepts.yaml`)

Each entry has:

| Field | Meaning |
|---|---|
| `canonical_id` | Stable controlled ID, `CPT_<SLUG>`. Never reused or renamed. |
| `label` | Human-readable canonical name. |
| `definition` | One sentence a compliance reviewer can approve against. |
| `domain` | The Step-3 domain label this concept belongs to. |
| `parent` | `canonical_id` of the broader concept, or `null` for a top family. |
| `synonyms` | Free-text variants (from real Step-3 output) that map here. |
| `status` | `approved` or `draft`. Only `approved` concepts satisfy the gate. |

A concept is a **reusable idea** ("Credit Risk Assessment"), not a
document-specific phrasing ("Credit Risk Assessment for Add-on Cardholders").

## 4. Matching rules (`matching_rules.yaml`)

For each clause concept, in precedence order:

1. **exact_label** — normalized concept equals a canonical `label`.
2. **synonym** — normalized concept equals a canonical `synonym`.
3. **normalized** — normalized concept equals a normalized label/synonym after
   stripping punctuation and collapsing whitespace.
4. **unmapped** — no match. `canonical_id: null`, `review_required: true`.

Normalization (casefold, trim, collapse internal whitespace, strip trailing
punctuation) is defined in `matching_rules.yaml` so it can be tuned without
touching code. Each match type carries a fixed `match_confidence`; a domain
mismatch between the clause and the matched concept lowers confidence and
forces review.

## 5. Output (`Nexus/ontology/<stem>_ontology.json`)

One **ontology object per clause**, plus a coverage summary. Each object
carries: the source clause id/no, the matched `canonical_id`/`label` (or null),
the controlled Step-3 fields (domain/action/obligation_type/mandatory_flag),
the `match_type`, `match_confidence`, and `review_required`. The full shape is
fixed by `ontology_object.schema.json`; Step 4 validates its own output against
it before writing.

The **catalog itself** (`concepts.yaml`) is the durable, cross-document asset;
the per-document file is the link from this document's clauses into it.

## 6. Review queue

`review_required: true` whenever any of:
- `match_type == unmapped`, or
- the matched concept's `status == draft`, or
- `match_confidence` below `matching_rules.yaml → auto_accept`, or
- the matched concept's `domain` differs from the clause's `domain`.

Everything flagged here is the input to Step 9 (Human Review) later, and every
approved correction becomes a new synonym or a new canonical concept.

## 7. Optional LLM-suggest layer (`--suggest`)

Deterministic matching alone leaves most of the hyper-specific free-text
concepts `unmapped`. Run `build_ontology.py --suggest` to add an advisory
semantic layer: for each distinct unmapped concept, an LLM proposes the
best-matching canonical concept **from the existing catalog** (or null).

Non-negotiable guarantees (all enforced in code and verified):
- **Never invents an id.** The model is constrained to catalog ids; any id it
  returns that is not in the catalog is discarded.
- **Never auto-accepts.** A suggestion fills `suggested_canonical_id` /
  `suggested_label` / `suggestion_confidence` only. `canonical_id` stays null,
  `match_type` stays `unmapped`, `review_required` stays true.
- **Confidence-floored.** Suggestions below `matching_rules.yaml →
  suggestion.min_confidence` are dropped.

Suggestions are the raw material for the review workflow (Step 9): a human
approves a suggestion, and it becomes a new synonym in the catalog + an approved
row in `concept_mappings.jsonl`. Until then, a suggestion is a hint, not a
mapping. The layer is off by default, so a plain run stays deterministic and
makes no LLM calls.

## 8. Approved examples (`concept_mappings.jsonl`)

One `{source_concept, canonical_id}` mapping per line, expert-approved. This is
the regression set: after any change to the catalog or matcher, every approved
mapping must still resolve to its stated `canonical_id`. Draft rows carry
`"status": "draft"` and do not count as passing.
