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
- [x] 10.15 Run the 100-input-base diagnostic Chinese pilot under the multi-option contract as `diagnostic-100-five-model-v7b`: 73 candidates reached complete Simulation coverage and 14 model-specific strict signals were observed. Preserve `gate_passed=false`; this does not authorize candidate freeze, Claude holdout, or mechanism claims.

## 11. Offline Paths Not Taken preparation and GPU execution boundary

- [ ] 11.1 Define a separate nullable `probe_relation_id` contract for narrow, direction-preserving Paths Not Taken relation families, with version, status, confidence, reason, and review provenance.
- [ ] 11.2 Build and explicitly review the probe-relation inventory before freezing a small balanced Chinese-first main-experiment subset; retain long-tail and unresolved facts outside relation-conditioned mechanism experiments.
- [ ] 11.3 Add Paths Not Taken eligibility checks for answer uniqueness, bilingual equivalence, prompt-form consistency, aliases, and model/language-specific answer tokenization.
- [ ] 11.4 Verify that same-relation negatives, relation-preservation checks, relation-specific task vectors, and across-relation splits use `probe_relation_id` rather than the broad statistical taxonomy.
- [ ] 11.5 Freeze the selected open-weight model/tokenizer revisions, hidden-state and intervention hook contract, decoding/scoring policy, layer selection, seeds, and run manifest before mechanism experiments. This may be drafted offline, but model-specific tokenization is not final until the exact revision is selected.
- [ ] 11.6 When GPU execution is available, run a bounded exact-HF Chinese baseline and report independent base-triple counts for candidate, relation-eligible, both-correct control, recall-failure candidate, and conversion-failure candidate cohorts before approving hidden-state or vector construction.

Items 11.1-11.4 are offline data-contract work and MUST NOT be blocked solely by GPU
unavailability. Items 11.5-11.6 define the runtime boundary: API/Ollama behavior screening
MUST NOT be represented as exact-HF white-box evidence.

## 12. Public-benchmark single-model provisional bridge

- [x] 12.1 Implement producer v2, bind the `public-benchmarks-full-v1` Qwen input, admit only completed/extracted/validation-clean rows, and label every admitted row `provisional_single_model` rather than canonical gold; the current output is 9,059 triples.
- [x] 12.2 Deterministically cluster exact duplicate facts into 8,969 `base_fact_id` values and preserve source, raw response, retry, and local-adjudication provenance.
- [x] 12.3 Emit separate statistical relation, nullable probe-relation, and candidate-direction fields; relation mapping MUST NOT gate factual-prompt readiness, and `term for definition` MUST remain direction-unresolved.
- [x] 12.4 Produce the full semantic-review queue plus a 400-item coverage-oriented sample, a separate 400-item overall-random sample, and a 200-item targeted sample from the 376-fact prompt-risk frame. Record that only the overall-random sample supports conditional population estimation.
- [x] 12.5 Emit a versioned behavior input bundle with prompt tier, aliases, explicit distractor IDs/provenance, evidence status, record hashes, and split metadata.
- [x] 12.6 Make factual-perturbation preparation accept an explicit bundle path while retaining the legacy canonical-v2 default; require exact bundle-bound external review-freeze and split-freeze manifests for formal rows.
- [x] 12.7 Add deterministic tests and generate the first offline provisional artifacts without invoking model endpoints or overwriting existing v7b/canonical/PATH outputs.
- [x] 12.8 Add SHA-bound review `scope`, duplicate-member-expanded `export`, and staging-only `apply` with `accept`/`revise`/`reject`/`defer`, explicit Codex-proxy `human_gold=false`, and partial missing decisions that are not converted to reject. Do not implement automatic freeze.
- [x] 12.9 Run a 20-item pilot inside the overall-random 400 scope and preserve 16 `accept`, 2 `defer`, 1 `reject`, 1 `revise`, and 380 `missing` as staging only.
- [x] 12.10 Add and run lexical near-duplicate candidate audit v2; record 2,341 emitted representative pairs and 281 emitted cross-split pairs, plus the non-exhaustive-census and `semantic_review_complete=false` limitations.
- [x] 12.11 Regenerate the PATH review-only export with 8,969 review-only facts and 0 formal facts while external freeze manifests are absent.
- [x] 12.12 Implement immutable full-pool/review-scope cohort-universe declaration and a
  fail-closed evidence preflight. The preflight MUST remain unable to emit a reviewed bundle or
  either freeze manifest while the true finalizer is absent.
- [ ] 12.13 Predeclare and separately materialize the chosen real formal cohort universe. Keep the
  overall-random 400 as an inferential sample unless it is explicitly selected as the cohort
  source; in that case require 400/400 final review. Freezing the full pool requires item-level
  final review of all 8,969 facts. Resolve every included-scope `missing`, `defer`, and `revise`
  rather than treating Codex proxy output as human gold.
- [ ] 12.14 Apply revisions with lineage, re-review revised hashes, adjudicate emitted
  lexical/alias candidates, handle missed-paraphrase risk, and compute duplicate/exposure closure
  against the full provisional pool and historically exposed artifacts. Re-cluster accepted facts
  and recompute development/validation/sealed assignments.
- [ ] 12.15 Implement the reviewed-bundle/freeze producer. Bind review-freeze and split-freeze
  manifests to the exact cohort universe, final decisions and counts, revision/re-review lineage,
  duplicate/exposure adjudications, final row hashes, and recomputed split; then regenerate formal
  behavior and PATH inputs.
- [ ] 12.16 Freeze one exact HF checkpoint/tokenizer/runtime contract and use it unchanged for the
  open-completion baseline, hidden states and Logit Lens, task/difference vectors, `resid_pre`
  intervention, repair, controls, and regression evaluation.
- [ ] 12.17 Revoke or rotate the credential exposed in the external reference checkout at
  `demo_load_datasets_model.py:337`; ensure the new runtime loads Hugging Face credentials only
  through an environment variable or secret manager and never persists them in artifacts.

No GPU, Ollama, HF checkpoint, or external experiment/provider endpoint was run for items
12.1-12.11. The 20 semantic decisions in 12.9 are explicitly Codex proxy judgments, not human
gold.
