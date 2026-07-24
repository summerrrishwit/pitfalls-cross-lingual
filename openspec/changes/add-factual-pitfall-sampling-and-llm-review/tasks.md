## 1. Raw-source configuration and contracts

- [x] 1.1 Replace the bilingual-source configuration with a raw-source MVP configuration for the five `data/source/` files, 600 total candidates, source/subject strata, random seed, three target languages, weakness thresholds, and a new output root.
- [x] 1.2 Define and validate typed schemas for raw-source manifests, raw English snapshots, factual-review records, generation records, translation records, screen checkpoints, retained-pitfall records, and summaries.
- [x] 1.3 Define explicit configuration roles for `FACTUAL_AUDIT_MODEL=qwen3.7-plus`, perturbation generator, translator, screening model list, and answer extractor; reject missing or unsupported role configuration.

## 2. Deterministic raw English sampling

- [x] 2.1 Implement read-only loading and English QA schema validation for all five `data/source/` JSON files.
- [x] 2.2 Deduplicate normalized English questions across the full raw source pool and record exclusions by source path and original index.
- [x] 2.3 Implement seeded source/subject-stratified allocation, undersized-stratum redistribution, and deterministic selection of 600 raw candidates.
- [x] 2.4 Write a raw-source manifest containing file fingerprints, source snapshots, allocations, selected indices, and exclusions before any model request.
- [x] 2.5 Add and verify a no-network raw-source dry-run command that produces the manifest and reports per-source allocations.

## 3. qwen3.7-plus English factual audit

- [x] 3.1 Refactor the audit prompt to consume raw English QA snapshots only and exclude perturbations, translations, rates, and screen outputs.
- [x] 3.2 Update the audit runner to consume only raw-manifest candidates, call `FACTUAL_AUDIT_MODEL=qwen3.7-plus`, and persist terminal structured reviews.
- [x] 3.3 Validate tri-state factual-review outputs and allow only validated accepts to enter generation.
- [x] 3.4 Add transient-only retry, terminal failure records, and a bounded audit-pilot command for the raw manifest.

## 4. Perturbation generation and translation

- [x] 4.1 Add a versioned answer-preserving perturbation prompt and validation contract using each non-canonical English option as the distraction target.
- [x] 4.2 Implement configurable perturbation generation from factual-review accepts with checkpointed provenance and no modification of canonical choices or answers.
- [x] 4.3 Add a versioned translation prompt and structured validation for Chinese, Japanese, and French generated candidates.
- [x] 4.4 Implement configurable per-language translation with terminal failure records and translated question/choice/answer alignment checks.

## 5. Cross-lingual screening and retained dataset outputs

- [x] 5.1 Implement configurable English and target-language model screening with separate raw-response, answer-extraction, correctness, and score records.
- [x] 5.2 Compute configured `rate_ori`, `rate_trans`, and pitfall scores from the declared screening ensemble and retain only threshold-satisfying candidates.
- [x] 5.3 Add resumable per-stage and per-model checkpoints plus JSONL records and summaries for generated, translated, screened, failed, and retained candidates.

## 6. Verification and documentation

- [x] 6.1 Add unit tests for raw-source schema validation, cross-source deduplication, deterministic source/subject sampling, and manifest-before-model-call enforcement.
- [x] 6.2 Add unit tests for English-only qwen3.7-plus audit prompts, generation integrity validation, translation alignment validation, screening threshold calculations, and transient versus permanent failures.
- [x] 6.3 Run a no-network raw-source dry-run and inspect source fingerprints, allocations, and absence of downstream artifacts.
- [ ] 6.4 Run bounded factual-audit, generation, translation, and screening pilots from the same raw manifest; manually inspect representative terminal records before the 600-source run.
- [x] 6.5 Replace the current curation documentation with raw-source commands, required model-role configuration, stage output locations, resume behavior, and the planned manual-review procedure.
