# Qwen Full Relation Normalization Report

## Status

Candidate v1 completed. Do not freeze as final taxonomy yet.

## Inputs

- Extraction ledger: `data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/triple_extractions.jsonl`
- Relation inventory: `data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/relation_inventory.json`
- Valid extracted records: 2,744
- Observed signatures: 2,425

## Candidate Artifacts

- Taxonomy: `relation_taxonomy_candidate_v1.json`
- Explicit mapping: `relation_mapping_candidate_v1.json`
- Mapping summary: `relation_mapping_candidate_v1_summary.json`
- Unresolved review queue: `relation_review_queue_candidate_v1.json`
- Normalized ledger: `triples_normalized_candidate_v1.jsonl`

## Coverage

| Status | Signatures | Records |
|---|---:|---:|
| mapped | 792 | 1,083 |
| ambiguous | 221 | 247 |
| out_of_taxonomy | 1,412 | 1,414 |
| total | 2,425 | 2,744 |

- Mapped signature coverage: 32.66%
- Mapped record coverage: 39.47%
- Candidate taxonomy relations: 33

The largest mapped families are `description_named_term` (455 records),
`entity_classification` (124), `description_named_entity` (81),
`entity_location` (70), and `cause_effect` (42).

## Validation

- Mapping schema validation passed for all 2,425 signatures.
- Output contains all 13,625 records in original candidate order.
- All original source and extraction fields are unchanged.
- All 2,744 extracted records have an explicit normalization status.
- All 10,881 non-extracted records are `not_applicable`.
- Every mapped relation ID exists in the candidate taxonomy.
- Applying the same inventory, taxonomy, and mapping twice is deterministic.
- Canonical normalized-output SHA-256:
  `1837e166041ea4254096fafdf064a14b52d20adf01ceec9da3cb033cf6da4030`
- Full test suite: 41 tests passed.

## Review Risks

- `classification`, `is`, `are`, `has`, `form`, and `source` are overloaded and
  remain ambiguous unless the observed types make the interpretation explicit.
- The inventory has 2,315 singleton signatures. Most out-of-taxonomy decisions
  are specific one-off relations rather than simple lexical synonyms.
- Broad automatic mapping would inflate coverage while merging different
  directions or semantic roles.
- Candidate normalized types intentionally use broad taxonomy types. They must
  be reviewed before being treated as final research labels.

## Freeze Recommendation

Do not freeze candidate v1 as final. Review the weighted unresolved queue first,
starting with ambiguous signatures and then recurring out-of-taxonomy patterns.
After review, publish new immutable versions such as `relation-taxonomy-v1` and
`relation-mapping-v1`, rerun normalization, and repeat the deterministic and
provenance checks.
