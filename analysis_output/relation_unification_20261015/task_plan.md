# Unified relation and legacy-data audit

## Goal
Assess efficient relation normalization for formal PNT and the reuse of historical dual-model factual extractions, using current local artifacts. Deliver a concrete staged design and verified inventory; do not launch extraction or model experiments.

## Phases
- [x] Establish scope and inspect current protocols.
- [x] Inventory historical and public artifacts, provenance, coverage and overlap.
- [x] Design shared relation ontology, review routing, reuse and re-extraction rules.
- [x] Verify conclusions and deliver the analysis.

## Decisions
- Preserve existing experiment and split artifacts; propose a versioned successor for unification.
- Count facts, source records and model annotations separately.
- Current raw artifact availability must be checked before claiming full-pool union statistics.

## Errors / limitations
- Partial public pipeline entity artifacts are absent locally; full 7,663-row union/semantic classification requires restoration and SHA verification.
- First inventory run encountered null multilingual review on technical failures; null is now treated as not accepted. Rerun succeeded.
- An empty documentation patch was rejected without changing any file; corrected with a substantive patch.

## Status
Completed: reproducible offline inventory, ten hashed input artifacts, reconciled count checks, and unified_relation_reuse_plan.md. No source data or frozen experiments changed; no model calls performed.
