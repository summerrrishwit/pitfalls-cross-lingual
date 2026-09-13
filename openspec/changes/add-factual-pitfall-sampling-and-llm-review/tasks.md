## 1. Retained raw-source sampling

- [x] 1.1 Load and validate the five immutable `data/source/` JSON datasets without modifying them.
- [x] 1.2 Deduplicate normalized English questions and preserve original source path/index provenance.
- [x] 1.3 Build deterministic stratified pilot and full-population raw manifests before model requests.

## 2. Replace the superseded factual-audit flow

- [x] 2.1 Remove the tri-state factual-audit prompt, validation, runner, monitor, analysis code, and audit-specific CLI/config names.
- [x] 2.2 Delete factual-review JSONL, audit summaries, supervisor state/log/lock files, and audit-analysis outputs while preserving `data/source/` and raw manifests.
- [x] 2.3 Replace the OpenSpec and README contracts with atomic-triple terminology and commands.

## 3. Initial atomic-triple extraction

- [x] 3.1 Add the versioned option-independent triple-extraction prompt and allowed exclusion reasons.
- [x] 3.2 Add local validation for extraction status, nullable-field consistency, canonical-answer grounding, confidence, and terminal failures.
- [x] 3.3 Write terminal records with explicit `source_id`, `source_dataset`, `source_question`, `source_choices`, `source_answer`, triple fields, model provenance, and validation errors.
- [x] 3.4 Add an `extract` CLI command with bounded runs, parallel workers, resume, retry-failed, prompt-version isolation, and summaries.

## 4. Relation inventory and normalization contracts

- [x] 4.1 Build a deterministic relation inventory keyed by relation text and subject/answer type signatures with counts and bounded examples.
- [x] 4.2 Define versioned taxonomy and mapping JSON schemas with direction/type validation and unresolved statuses.
- [x] 4.3 Add deterministic mapping application that writes normalized triples without changing source or extracted triple fields.

## 5. Verification and first experiment

- [x] 5.1 Replace old audit tests with extraction prompt, validator, provenance, resume-isolation, inventory, and mapping tests.
- [x] 5.2 Run the offline test suite and no-network manifest checks.
- [x] 5.3 Preflight `sensenova-6.7-flash-lite` and `qwen3.7-plus` through their configured endpoints without exposing credentials.
- [x] 5.4 Run a bounded online SenseNova extraction smoke test, inspect records, and build the first relation inventory.

## 6. Codex-calibrated dual-model extraction

- [x] 6.1 Freeze binary Codex reference labels and field-level review notes for the 100-row calibration manifest.
- [x] 6.2 Revise the extraction prompt and validator to admit single-definition facts, permit grounded answer normalization, and reject option-dependent non-unique facts.
- [x] 6.3 Add independent SenseNova and Qwen clients, model-slug output directories, concurrent workers, prompt/model resume isolation, and retry-failed support.
- [x] 6.4 Add calibration evaluation with a hard per-model `label_error_rate <= 0.03` expansion gate and disagreement records.
- [x] 6.5 Run both models on the 100-row calibration set, inspect field-level errors, and satisfy the expansion gate.
- [x] 6.6 Create a disjoint deterministic 300-row manifest and run both models to completion.
- [x] 6.7 Produce Codex summary artifacts for agreement, disagreement, relation inventory, and final-review candidates.
- [x] 6.8 Update tests and README, then run the full test suite and strict OpenSpec validation.

## 7. Canonical triples, prompts, and distractor candidates

- [x] 7.1 Align the full SenseNova and Qwen checkpoints and write conservative canonical decisions plus canonical triples.
- [x] 7.2 Build the canonical relation inventory, freeze the Codex taxonomy/mapping, and preserve reviewed unresolved signatures.
- [x] 7.3 Apply normalized relations and generate versioned strict-completion/open-answer factual prompts.
- [x] 7.4 Construct ordered source wrong options and expanded unverified distractor candidates.
- [x] 7.5 Add offline validation, tests, README instructions, and run the full pipeline determinism checks.

## 8. Codex adjudication of full-run extraction disagreements

- [x] 8.1 Replace automatic exclusion of single-model extractions and answer conflicts with a complete `pending_review` queue and downstream coverage gate.
- [x] 8.2 Freeze the item-level Codex adjudication prompt and review all 1,052 single-model rows plus 26 material answer conflicts.
- [x] 8.3 Preserve the 1,078 explicit decisions, reuse compatible prior Codex decisions, and validate exact queue coverage.
- [x] 8.4 Rebuild canonical triples, relation inventory/mapping, factual prompts, and distractor candidates from accepted adjudications.
- [x] 8.5 Update tests, README, and OpenSpec requirements, then run full validation.

## 9. Decouple factual prompts from relation normalization

- [x] 9.1 Change factual-prompt eligibility to depend on accepted canonical facts and prompt validation, not on `normalization_status=mapped` or a non-null `relation_normalized`.
- [x] 9.2 Preserve the existing relation taxonomy and mapping unchanged as an optional, versioned statistical layer, including explicit `ambiguous` and `out_of_taxonomy` outcomes.
- [x] 9.3 Version and regenerate factual-prompt artifacts; report strict-completion, open-answer-fallback, and rejected-prompt counts, and add coverage tests for mapped and unmapped canonical triples.
- [x] 9.4 Update downstream validation and distractor construction to use explicit prompt readiness rather than relation-mapping status.

## 10. Chinese factual-perturbation MVP before Paths Not Taken

- [x] 10.1 Validate and freeze the model-role configuration in `configs/factual_perturbation_zh_mvp_v1.json`; all eight selected models passed a separate minimal JSON request and representative role-specific structured request on 2026-09-09. Batch stability remains a later gate.
- [x] 10.2 Implement and preflight `qwen3.8-max` primary translation with `gpt-5.5` review, preserving translations, aliases, review decisions, prompt/model versions, and raw responses.
- [x] 10.3 Implement and preflight `gpt-5.6-sol` perturbation generation with independent `qwen3.8-max` distractor/perturbation validation; uncertain or invalid judge outputs are rejected.
- [x] 10.4 Use `qwen3.7-plus`, `bailian/deepseek-v4-flash-0731`, `hy4-preview`, and `gemini-3.7-flash` as the required four-family simulation ensemble; retain missing/failed model results and do not compute a primary score from partial coverage.
- [ ] 10.5 Freeze accepted candidates before evaluating `claude-opus-4-8` as the model-and-family holdout; do not call any SenseNova model in this run.
- [ ] 10.6 Run the bounded Chinese pilot with source-wrong-option and same-answer-type distractors; defer same-relation sampling and all Paths Not Taken mechanism work.
- [x] 10.7 Implement schema validation, bounded retry, redacted event logs, config fingerprints, and single-writer resumable JSONL checkpoints.
- [x] 10.8 Freeze and execute the deterministic 10-base-triple preflight. The original run exposed two `hy4-preview` failures and a duplicate-baseline inconsistency; the corrected three-arm rerun completed 112/112 calls with zero baseline inconsistency, while currency cost remains unauditable.
- [ ] 10.9 After a passing preflight, run the 100-base-triple pilot, freeze accepted candidates, and only then run `claude-opus-4-8` holdout evaluation.
- [x] 10.10 Emit a preflight report with attrition, per-model errors/retries/latency, token usage, cost-audit status, and isolation status.
- [x] 10.11 Freeze a complete Codex proxy review of 10 translations, 14 distractors, 22 perturbations, 3 provisional outcomes, and 2 failed calls with `not_human_gold=true`.
- [x] 10.12 Add shared Original and Neutral Context controls, candidate-specific Targeted arms, stable IDs, duplicate-baseline detection, and a resumable proxy-reviewed rerun.
- [x] 10.13 Run 20 new non-overlapping base triples, complete Codex proxy review, and evaluate one preselected perturbation for each of 12 eligible independent bases; 287/288 Simulation calls completed, with zero confirmed target effects and one control-sensitive item.
- [x] 10.14 Replace the binary endpoint with deterministic three-option scoring, matched neutral contexts, target-distractor hit tracking, and distinct non-target error reporting; complete the 20-base calibration with 480/480 Simulation calls and zero target-specific Chinese flips.
- [ ] 10.15 Run a preregistered 100-input-base diagnostic Chinese pilot under the frozen multi-option contract; if it remains insensitive, preregister a separate alternate-language transfer screen.

## 11. Deferred Paths Not Taken preparation

- [ ] 11.1 Define a separate nullable `probe_relation_id` contract for narrow, direction-preserving Paths Not Taken relation families, with version, status, confidence, reason, and review provenance.
- [ ] 11.2 Build and explicitly review the probe-relation inventory before freezing a small balanced Chinese-first main-experiment subset; retain long-tail and unresolved facts outside relation-conditioned mechanism experiments.
- [ ] 11.3 Add Paths Not Taken eligibility checks for answer uniqueness, bilingual equivalence, prompt-form consistency, aliases, and model/language-specific answer tokenization.
- [ ] 11.4 Verify that same-relation negatives, relation-preservation checks, relation-specific task vectors, and across-relation splits use `probe_relation_id` rather than the broad statistical taxonomy.
- [ ] 11.5 Freeze the selected open-weight model/tokenizer revisions, hidden-state and intervention hook contract, decoding/scoring policy, layer selection, seeds, and run manifest before mechanism experiments.
- [ ] 11.6 Run a bounded Chinese baseline and report independent base-triple counts for candidate, relation-eligible, both-correct control, recall-failure candidate, and conversion-failure candidate cohorts before approving full vector construction.
