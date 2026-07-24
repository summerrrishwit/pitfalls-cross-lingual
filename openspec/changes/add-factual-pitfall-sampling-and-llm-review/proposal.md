## Why

The current curation workflow samples from already-generated bilingual Cross-Lingual Pitfalls files. That can analyze an existing benchmark, but it cannot create a new factual-pitfall dataset with controlled source selection, generation, translation, and screening.

This change moves the pipeline upstream to the raw English QA sources in `data/source/`. It will sample original questions first, audit their factual-recall suitability with `qwen3.7-plus`, then generate and translate perturbations before screening the resulting bilingual pairs for cross-lingual weaknesses.

## What Changes

- Replace sampling from `data/Chinese.json`, `data/Japanese.json`, and `data/French.json` with deterministic, source-stratified sampling from `data/source/ai2_arc_easy.json`, `commonsense_qa.json`, `mmlu.json`, `sciq.json`, and `truthful_qa.json`.
- Preserve a raw-source manifest before any LLM call, including file fingerprints, original indices, source metadata, random seed, allocations, and source snapshots.
- Retain the English-only `qwen3.7-plus` factual audit, now applied to sampled raw English QA records before any perturbation or translation is generated.
- Add configurable perturbation-generation, translation, and screening stages. The pipeline will create candidate bilingual pairs only from English factual-audit accepts, then retain candidates that satisfy configured English-versus-target-language weakness thresholds.
- Record the generator, translator, screening models, per-model answers, answer-extraction results, and all intermediate artifacts so retained cross-lingual weaknesses can be reproduced.
- Supersede the existing bilingual-source sampling manifest and its implementation for this change; retain it only as a historical derived artifact and do not delete or mutate it.

## Capabilities

### New Capabilities

- `raw-factual-source-sampling`: Deterministically sample immutable English QA records from `data/source/` before any model request.
- `english-factual-review`: Use `qwen3.7-plus` and a structured English-only prompt to determine whether a raw English QA item is a suitable factual-recall probe.
- `cross-lingual-pitfall-generation`: Generate perturbations, translate accepted English factual QA items, and screen bilingual candidates for configured cross-lingual weaknesses.

### Modified Capabilities

- None.

## Impact

- Replaces the curation input assumption; existing `data/<language>.json` files are no longer inputs to the new dataset-construction pipeline.
- Adds separate model roles for factual review, perturbation generation, translation, screening answers, and answer extraction.
- Requires derived manifests, candidate records, screening checkpoints, and run-level provenance under a new output root without changing raw source data.
- The existing prototype curation script and its 600-item bilingual manifest must be refactored in a future apply phase; current `run.py` and `eva.py` behavior remains unchanged until then.
