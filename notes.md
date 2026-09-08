# Notes: Full Qwen Relation Normalization

## Current Inventory
- Input: `data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/relation_inventory.json`
- Extracted records: 2,744
- Relation signatures: 2,425
- Singleton signatures: 2,315
- Signatures with count >= 2: 110
- Signatures with count >= 5: 19

## Contract
- A signature is keyed by normalized `relation_raw`, `subject_type`, and `answer_type`.
- Taxonomy relations define ID, definition, normalized subject type, normalized answer type, direction, and examples.
- Mapping decisions must be explicit and versioned.
- Unknown or unsafe signatures remain `ambiguous` or `out_of_taxonomy` with null normalized fields.
- Applying identical artifacts must be deterministic and preserve source/extraction provenance.

## Pending Findings
- Relation family profile
  - `term for definition` covers 428 records but means description-to-term.
  - `definition` and `meaning` mean term-to-definition.
  - `classification`, `is`, `name`, and other broad labels are overloaded and require type-aware or unresolved handling.
  - Initial lexical family coverage: definition 512 records; classification 248; location 110; cause/effect 110; measurement 96; function/purpose 56; requirement 51; part/whole 44.
- Candidate taxonomy size and coverage
  - Candidate relations: 33
  - Mapped signatures: 792 / 2,425 (32.66%)
  - Mapped records: 1,083 / 2,744 (39.47%)
  - Ambiguous: 221 signatures / 247 records
  - Out of taxonomy: 1,412 signatures / 1,414 records
- Ambiguous and out-of-taxonomy review queues
  - Persisted in `relation_review_queue_candidate_v1.json`, ordered by status and record support.
- Determinism and provenance checks
  - 13,625 input and output records with identical candidate order.
  - Every original input field is unchanged in normalized output.
  - All 10,881 non-extracted records are `not_applicable`.
  - Two in-memory applications are byte-equivalent after canonical JSON serialization.
  - Normalized canonical SHA-256: `1837e166041ea4254096fafdf064a14b52d20adf01ceec9da3cb033cf6da4030`.
