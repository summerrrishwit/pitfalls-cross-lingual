## Context

`data/source/` contains immutable English multiple-choice source datasets: AI2 ARC Easy, CommonsenseQA, MMLU, SciQ, and TruthfulQA. These records contain English `question`, `choices`, `answer`, and `source` fields (with optional `subject`), but do not contain translations or measured cross-lingual performance. By contrast, the repository's language-specific JSON files are derived artifacts from an earlier generation pipeline and already embed those later-stage properties.

The new dataset must make the construction path explicit:

```text
raw English source sample
  → qwen3.7-plus English factual audit
  → perturb English factual QA
  → translate candidate to each target language
  → screen English and target-language answers
  → retain cross-lingual factual pitfalls
```

The MVP keeps the design document's target languages—Chinese, Japanese, and French—and aims for up to 100 retained factual-pitfall records per language. It first creates and audits a reproducible 600-item calibration pilot. Pilot outcomes are used to revise and version the factual-audit prompt; only after a bounded v4 re-audit is accepted does the pipeline create a separate full-population manifest. The source manifest, prompt version, factual audit, generation, translation, and screening all have distinct provenance.

## Goals / Non-Goals

**Goals:**

- Deterministically sample 600 eligible raw English records for prompt calibration, then create a separate full-population manifest after the calibrated prompt is accepted.
- Use `qwen3.7-plus` to audit only the raw English QA for factual-recall suitability before generation or translation.
- Generate perturbations only for accepted English factual records and preserve the wrong option used, perturbation text, insertion position, and generator model configuration.
- Translate each generated English candidate into Chinese, Japanese, and French with an explicit translation model and preserve the translated question, choices, and answer.
- Screen English and target-language candidates with a configurable model ensemble; retain a pair only when it meets the configured `rate_ori`, `rate_trans`, and pitfall-score thresholds.
- Produce append-safe derived artifacts and checkpoints without modifying `data/source/` or the older language-specific files.

**Non-Goals:**

- Using existing `data/Chinese.json`, `data/Japanese.json`, or `data/French.json` as source inputs for the new construction run.
- Choosing one fixed perturbation generator, translator, or screening ensemble in the specification; each run must declare provider-valid model IDs explicitly.
- Final bilingual semantic-equivalence adjudication, train/dev/test splitting, logit-lens analysis, activation patching, vector intervention, or downstream benchmark reporting.
- Deleting the existing 600-item manifest or retroactively treating it as a raw-source manifest.

## Decisions

### Sample raw source records before all model stages

The pipeline creates a 600-item pilot manifest from the five `data/source/` files before calling `qwen3.7-plus`. It validates the English QA schema, deduplicates normalized English questions across the complete source pool, stratifies the pilot by source and optional subject, and records source-file fingerprints and excluded duplicates. A separate full configuration selects every eligible record only after prompt calibration.

Alternative considered: sample independently per target language or sample after factual review. Rejected because the raw English source population must be stable across languages and reviewers; target-language-specific selection happens only after translated screening outcomes exist.

### Keep the qwen3.7-plus audit English-only and upstream

`FACTUAL_AUDIT_MODEL=qwen3.7-plus` receives only a raw English question, choices, canonical answer, and optional English source/subject metadata. It returns an accept/reject/needs-review decision and, when accepted, an English subject-relation-answer triple and natural completion prompt. It does not receive target-language text, candidate perturbations, rates, or screen-model answers.

Alternative considered: audit generated bilingual candidates directly. Rejected because this would confound source factuality with generation and translation quality, making later error attribution impossible.

### Calibrate and version the audit prompt before the full run

The 600-item v1 audit is a calibration artifact rather than final full-run labeling. Analysis showed that 40 of 127 model accepts were downgraded because `answer_en` paraphrased rather than exactly copied the canonical answer, the model never selected `needs_review`, and several clue-solving or typical-commonsense items were accepted as direct facts. Iterations v2-v4 therefore require exact canonical copying, define the three decision boundaries, add contextual-inference, time-sensitivity, canonical-answer-quality, mutable-role and mutable-location checks, and record a structured reason code.

Prompt-calibration reruns write to a separate JSONL. Resume rejects checkpoints produced by another prompt version. This prevents v1 and v2 labels from being silently mixed.

### Treat perturbation, translation, and screening as explicit model roles

Each run configuration must declare `PERTURBATION_GENERATOR_MODEL`, `TRANSLATION_MODEL`, `SCREEN_MODELS`, and `ANSWER_EXTRACT_MODEL`, in addition to the fixed factual-audit role. The generation stage uses the accepted English factual QA and a selected incorrect option to produce a subtle, answer-preserving distraction. Translation creates a target-language version of the full candidate and its choices/answer. Screening evaluates the enhanced English and translated questions independently for every configured screen model, using the answer extractor only to normalize a model answer to an option.

Alternative considered: retain the `run.py` hard-coded GPT generation and translation values. Rejected because it obscures experimental variables and does not support controlled generator/translator comparisons.

### Screen only after generation and translation

Raw source records have no `rate_ori` or `rate_trans`. The pipeline computes these values only from screening results for each generated bilingual candidate. A candidate is retained when configured thresholds hold, initially following the dataset design: `rate_ori >= 0.8`, `rate_trans <= 0.5`, and `rate_ori - rate_trans >= 0.3`. The manifest records the model list so rate interpretation remains reproducible.

Alternative considered: infer weaknesses from source labels or historical language files. Rejected because the new perturbation and translation can change model behavior.

### Persist each stage separately

The output root contains one raw-source manifest, English factual-review JSONL, generation records, per-language translation records, screen checkpoints, retained-pitfall JSONL, and summaries. A terminal failure is recorded at its own stage and is not silently replaced by another raw source record. Re-running a stage consumes its preceding artifact rather than resampling.

## Risks / Trade-offs

- [Prompt calibration changes audit decisions] → Preserve v1 pilot results, write every calibration version to a separate checkpoint, and create a separate full v4 run instead of treating v1 labels as resumable v4 labels.
- [Generation changes the answer or introduces contradictory facts] → Require a generator validation prompt/contract, retain the wrong option and raw response, and reject candidates whose canonical answer or question integrity changes.
- [Translation changes answer semantics] → Preserve translated choices and answer, perform schema/option alignment checks before screening, and reserve semantic-equivalence adjudication for a later validation change.
- [Different model roles bias results] → Record all role/model IDs, temperatures, prompt versions, and per-model outputs; run controlled comparisons with one role changed at a time.
- [Screening is expensive or unstable] → Use per-stage checkpoints, conservative concurrency, transient-only retries, and bounded pilots before a full run.
- [A provider model ID is unavailable] → Validate configured model IDs with a small preflight call before creating a costly full run; do not fall back silently.

## Migration Plan

1. Retain the existing bilingual-source manifest as a historical prototype and do not reuse it as input.
2. Refactor the curation configuration and sampler to use `data/source/` and write a new raw-source manifest.
3. Implement and test the English factual audit against the raw manifest with `qwen3.7-plus`.
4. Analyze the completed 600-item v1 audit, revise the prompt, and run bounded v2-v4 calibration into separate checkpoints.
5. After manual acceptance of a representative v4 calibration, create a separate full-population run and implement generation, translation, and screening as distinct resumable stages.

## Open Questions

- Which provider-valid model IDs will be used for perturbation generation, translation, and the screening ensemble in the first controlled run?
- Should the 600 raw-source allocation be strictly proportional to source-pool size, or guarantee a minimum per source/subject stratum?
- Should screening retain only candidates that fail every screen model in the target language, or use the design document's aggregate `rate_trans <= 0.5` threshold?
