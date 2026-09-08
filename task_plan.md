# Task Plan: Full Qwen Relation Normalization

## Goal
Create an auditable candidate taxonomy and explicit mapping for all Qwen relation signatures, apply it deterministically, and report unresolved review work without presenting unreviewed labels as frozen gold.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Profile inventory and existing pilot artifacts
- [x] Phase 3: Build candidate taxonomy and complete signature mapping
- [x] Phase 4: Apply mapping and validate normalized output
- [x] Phase 5: Review risks and deliver report

## Key Questions
1. Which relation expressions can be merged without losing type or direction semantics?
2. How many records can be mapped deterministically, and which signatures remain ambiguous?
3. Does normalization preserve every source and extraction field and reproduce deterministically?

## Decisions Made
- Preserve `relation_raw`; only add versioned normalized fields.
- Produce candidate v1 artifacts first. They are not final frozen gold until review gates pass.
- Give every one of the 2,425 signatures an explicit `mapped`, `ambiguous`, or `out_of_taxonomy` decision.
- Separate description-to-term relations from term-to-definition relations; their directions are opposite.
- Use conservative deterministic rules for candidate v1. Broad or overloaded labels remain unresolved.
- Candidate v1 maps 1,083 of 2,744 extracted records; unresolved records remain explicit review items.

## Errors Encountered
- CodeGraph context lookup was unavailable because its service connection closed; continued with repository specifications and implementation source.

## Status
**Complete** - candidate artifacts generated, normalized output validated, review queue persisted, and freeze recommendation documented.
