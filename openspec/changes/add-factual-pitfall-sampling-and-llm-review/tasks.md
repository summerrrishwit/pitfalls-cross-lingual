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
