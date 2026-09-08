## Context

`data/source/` contains immutable English multiple-choice records with `question`, `choices`, `answer`, `source`, and optional dataset-subject metadata. The correct answer is already known, so the model does not need to solve the MCQ. It must determine whether the question-answer pair expresses one atomic fact and, if so, orient a triple whose object is exactly the canonical source answer.

The revised experiment is intentionally limited to factual-triple construction and relation normalization:

```text
raw_source_manifest.json
  -> models/sensenova-6.7-flash-lite/triple_extractions.jsonl
  -> models/qwen3.7-plus/triple_extractions.jsonl
  -> calibration/codex_reference_labels.jsonl
  -> calibration/model_evaluation.json
  -> codex_summary.json
  -> relation_inventory.json
  -> relation_taxonomy_v1.json
  -> relation_mapping_v1.json
  -> triples_normalized.jsonl
  -> factual_prompts.jsonl
  -> distractor_candidates.jsonl
```

This phase now generates English factual prompts and unverified distractor candidates from
original wrong options. It does not generate perturbation text, translations, screening
scores, or retained badcases.

## Goals / Non-Goals

**Goals:**

- Run `sensenova-6.7-flash-lite` and `qwen3.7-plus` independently with the same extraction contract.
- Balance the calibration pilot across the five source datasets, then stratify within each source by optional subject metadata.
- Preserve source provenance and original choices without asking the model to reproduce them.
- Extract only atomic facts and record a machine-readable exclusion reason when no triple is extractable.
- Exclude full-sentence, explanatory, procedural, recommendation, and multi-clause canonical answers that are unsuitable as one triple object.
- Preserve the canonical `source_answer` exactly while permitting `answer` to be a shorter normalized entity or phrase explicitly entailed by it.
- Require triple direction to remain `subject -> canonical answer` even when the reverse relation has a more familiar name.
- Aggregate relation signatures globally before defining normalized labels.
- Freeze a versioned taxonomy and explicit mapping so normalization is reproducible.
- Reserve `qwen3.7-plus` or Codex for global taxonomy induction, mapping review, and ambiguous cases.
- Keep direct dual-model answer agreements and require explicit Codex adjudication before accepting or rejecting a single-model extraction or material answer conflict.
- Generate deterministic English recall prompts only for mapped relations and preserve prompt-quality tiers.
- Copy source wrong options into distractor candidates without prematurely claiming type or factual verification.

**Non-Goals:**

- Retaining or recalibrating the previous `accept` / `reject` / `needs_review` audit.
- Generating `prompt_en` during per-model triple extraction.
- Verifying or augmenting distractors with same-relation negatives, generating perturbation text, translating prompts, screening models, or searching cross-lingual badcases.
- Modifying raw files in `data/source/`.
- Forcing every record or every relation into the taxonomy.

## Decisions

### Preserve choices in code, not in the LLM response

The output writer copies `source_choices` from the immutable source snapshot. The extraction prompt intentionally omits choices so extractability is tested without option comparison. This also prevents the model from altering choice text or order.

### Keep extraction and normalization separate

The extraction model returns `relation_raw`, not `relation_normalized`. A global inventory groups observed signatures using:

```text
relation_raw + subject_type + answer_type + direction + examples + count
```

A stronger model can then propose a taxonomy and mapping with global visibility. After review, the taxonomy and mapping are frozen and applied programmatically. This is less expensive than two strong-model calls per row and more consistent than allowing each row to invent a normalized label.

The weak model's `subject_type` and `answer_type` are evidence rather than ground truth. Each mapped signature therefore also records `subject_type_normalized` and `answer_type_normalized`, which must match the frozen taxonomy and may correct weak-model type errors without rewriting the raw extraction.

### Keep an extraction gate without recreating the old audit

Every row has `extraction_status=extracted|not_extractable`. A non-extractable record includes one exclusion reason and null triple fields. This gate expresses only whether an atomic triple can be formed; it does not retain the old tri-state review decision or its audit findings.

### Canonical grounding and relation direction are hard constraints

The source answer is authoritative and remains unchanged in provenance. For an extracted row, `answer` must either copy it or normalize an explicitly stated entity or short phrase without adding unsupported information. For example, the source sentence `Obama was born in Hawaii, which is a US state` may normalize to `United States` for a `country_of_birth` answer. The model must not reverse the triple.

The subject must be an independently identifiable anchor from the question and must not repeat the answer. A single definitional description may be represented with a stable `definition_term` or classification relation. Multi-clue biographical riddles, scenario interpretation, and descriptions combining several independent facts remain clue solving.

### Calibrate before expansion

The 100-row pilot has a frozen Codex reference decision for every row. Model error is the fraction whose `extraction_status` differs from that binary reference label; field-level issues are reported separately and cannot be hidden by label accuracy. Both models must have at most three label errors out of 100 before a disjoint 300-row manifest is run. This is a calibration criterion against a Codex reference, not an estimate of human-gold accuracy.

### Isolate model outputs and endpoints

Every model writes to `models/<model-slug>/`. Resume reads only that model's checkpoint and rejects mixed prompt/model IDs. Model calls may run concurrently inside a model and the two model processes may run concurrently because they never share an output file. `qwen3.7-plus` uses `OPENAI_BASE_URL`, whose experiment value is `https://ctapi.csxdtx.com:16000/v1`; API credentials and base URLs are never persisted in result rows.

### Normalize only through explicit artifacts

`relation_taxonomy_v1.json` defines every relation ID, definition, subject type, answer type, direction, and examples. `relation_mapping_v1.json` maps observed relation signatures to those IDs. `triples_normalized.jsonl` is produced only from these artifacts. Unknown and ambiguous signatures remain unresolved and are never silently coerced.

### Canonicalize by agreement plus explicit Codex adjudication

Rows with two locally valid extractions and equivalent normalized answers enter directly;
SenseNova supplies their canonical fields and both annotations remain attached. A row with
exactly one valid extraction or two materially different answers is `pending_review`, not
excluded. The downstream inventory is blocked until a versioned Codex decision covers every
queued row. An accepted decision selects one of the eligible model annotations without editing
its fields; a rejected decision records an item-specific reason. Source provenance and choices
remain unchanged in either case.

### Separate prompt precision from relation statistics

`relation_normalized` supports grouping and statistics, but broad taxonomy templates can lose
the precise raw relation. Prompt generation therefore first removes a sentence-final answer
from `canonical_fact`, yielding a strict completion stem. When the answer does not occur at the
end, the system emits an option-free `Factual question: ... / Answer:` fallback and labels it as
`open_answer_fallback`. Unresolved relations receive no prompt.

### Treat source wrong options as candidates, not verified negatives

For every prompt-ready triple, the pipeline copies all original choices except the canonical
source answer and normalized answer. It ranks them deterministically by answer-text similarity
and selects the top candidate for convenience, but records `distractor_verified=false` and
does not claim that type compatibility or `(subject, relation, distractor)` falsity has been
independently verified.

### Version and isolate extraction outputs

The extraction prompt version, model ID, raw response, validation errors, retry history, and timestamps are retained. Resume rejects a checkpoint created by another prompt version or extraction model.

## Risks / Trade-offs

- [Weak-model extraction errors propagate] -> Run a stratified pilot, validate exact answers locally, and manually inspect relation direction and atomicity before scaling.
- [Sentence-like source answers waste calls or become malformed triples] -> Apply a conservative program-owned answer-suitability gate before the model and record the skipped row with full provenance.
- [Free-form relations fragment] -> Build a counted global inventory and freeze an explicit mapping before normalization.
- [A broad relation collapses different semantics] -> Include subject/answer types and representative canonical facts in inventory entries.
- [A reverse relation is forced into a familiar label] -> Validate direction against each taxonomy signature and permit `out_of_taxonomy`.
- [Source choices are lost] -> Copy them directly from the source snapshot into every terminal extraction record.
- [Strong-model endpoint changes] -> Record endpoint-independent model ID and run a preflight before taxonomy work; never record API keys.

## Migration Plan

1. Preserve `data/source/` and model-independent `raw_source_manifest.json` files.
2. Remove audit-specific checkpoints, logs, supervisor code, reports, prompt, validation, runner, and tests.
3. Introduce extraction-specific configuration, prompt, validation, CLI, records, and resume isolation.
4. Run offline unit tests and a bounded online SenseNova extraction smoke test.
5. Build `relation_inventory.json` from valid extracted rows.
6. Use Codex/`qwen3.7-plus` to propose and review `relation_taxonomy_v1.json` plus `relation_mapping_v1.json`.
7. Apply the frozen mapping to produce normalized triples and report unresolved signatures.
8. Generate factual prompts for mapped triples and construct traceable source-option distractor candidates.
