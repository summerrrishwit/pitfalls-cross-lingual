## Why

The current curation workflow samples from already-generated bilingual Cross-Lingual Pitfalls files. That can analyze an existing benchmark, but it cannot create a new factual-pitfall dataset with controlled source selection, generation, translation, and screening.

This change moves the pipeline upstream to the raw English QA sources in `data/source/`. It first audits a reproducible 600-item pilot, uses the observed failure modes to calibrate and version the `qwen3.7-plus` prompt, then creates a separate full-population run before generating, translating, and screening bilingual candidates.

## What Changes

- Replace sampling from `data/Chinese.json`, `data/Japanese.json`, and `data/French.json` with a deterministic 600-item calibration pilot followed by a separate full-population manifest from the five `data/source/` datasets.
- Preserve a raw-source manifest before any LLM call, including file fingerprints, original indices, source metadata, random seed, allocations, and source snapshots.
- Retain the English-only `qwen3.7-plus` factual audit, now applied to sampled raw English QA records before any perturbation or translation is generated.
- Add configurable perturbation-generation, translation, and screening stages. The pipeline will create candidate bilingual pairs only from English factual-audit accepts, then retain candidates that satisfy configured English-versus-target-language weakness thresholds.
- Record the generator, translator, screening models, per-model answers, answer-extraction results, and all intermediate artifacts so retained cross-lingual weaknesses can be reproduced.
- Supersede the existing bilingual-source sampling manifest and its implementation for this change; retain it only as a historical derived artifact and do not delete or mutate it.

## Capabilities

### New Capabilities

- `raw-factual-source-sampling`: Deterministically sample a 600-item calibration pilot, then include every immutable eligible English QA record in a separate full run after prompt calibration.
- `english-factual-review`: Use `qwen3.7-plus` and a structured English-only prompt to determine whether a raw English QA item is a suitable factual-recall probe.
- `cross-lingual-pitfall-generation`: Generate perturbations, translate accepted English factual QA items, and screen bilingual candidates for configured cross-lingual weaknesses.

### Modified Capabilities

- None.

## Impact

- Replaces the curation input assumption; existing `data/<language>.json` files are no longer inputs to the new dataset-construction pipeline.
- Adds separate model roles for factual review, perturbation generation, translation, screening answers, and answer extraction.
- Requires derived manifests, candidate records, screening checkpoints, and run-level provenance under a new output root without changing raw source data.
- The existing prototype curation script and its 600-item bilingual manifest must be refactored in a future apply phase; current `run.py` and `eva.py` behavior remains unchanged until then.
