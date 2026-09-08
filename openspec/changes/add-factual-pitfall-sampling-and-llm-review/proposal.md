## Why

The existing raw-English stage mixes two different concerns: deciding whether a multiple-choice item is suitable factual recall and freely generating `subject_en`, `relation_en`, `answer_en`, and `prompt_en`. Its tri-state review artifacts cannot serve as a reproducible atomic-triple dataset, and the free-form relation strings cannot support stable relation statistics.

This revision keeps deterministic loading and sampling from the five immutable `data/source/` datasets, removes the prior factual-review flow and its derived outputs, and introduces a focused two-stage experiment:

```text
raw English QA
  -> SenseNova and Qwen atomic factual triple extraction
  -> per-model validated triple records with source provenance
  -> Codex-reference calibration gate (label error <= 3%)
  -> global relation inventory
  -> versioned relation taxonomy and mapping
  -> normalized triples
  -> factual completion prompts
  -> source-wrong-option distractor candidates
```

## What Changes

- Preserve `data/source/` and model-independent raw-source manifests.
- Replace the `accept` / `reject` / `needs_review` factual-audit prompt with one versioned atomic-triple extraction contract run independently by `sensenova-6.7-flash-lite` and `qwen3.7-plus`.
- Persist explicit `source_id`, `source_dataset`, `source_question`, `source_choices`, and `source_answer` fields alongside extracted `subject`, `relation_raw`, and a concise normalized `answer` grounded in the canonical source answer.
- Store each model below a model-slug directory with independent JSONL checkpoints, summaries, retries, and prompt/model resume isolation.
- Freeze a 100-row Codex reference set, require each model's binary extraction-label error to be at most 3%, and only then run a separate deterministic 300-row experiment.
- Produce a deterministic Codex-review bundle containing model agreement, disagreements, candidate triples, relation inventories, and records requiring final review.
- Do not ask the extraction model to copy choices, create `prompt_en`, or invent `relation_normalized`.
- Build a global relation inventory containing relation text, subject/answer types, counts, and representative examples.
- Use a strong-model-assisted, human-freezable taxonomy and explicit mapping artifact before filling `relation_normalized`.
- Route unmapped or directionally ambiguous relations to `out_of_taxonomy` or `ambiguous` instead of forcing a label.
- Determine canonical triples from direct dual-model normalized-answer agreement plus explicit Codex adjudication of every single-model extraction and material answer conflict, while retaining both annotations and all source choices.
- Generate strict factual-completion prompts when the canonical fact ends in the answer and an explicit open-answer fallback otherwise.
- Preserve original wrong options as traceable, unverified distractor candidates; do not claim type/fact verification in this stage.
- Delete the superseded factual-review checkpoints, supervisor artifacts, audit-analysis outputs, and audit-specific code.

## Capabilities

### New Capabilities

- `atomic-factual-triple-extraction`: Extract validated atomic `(subject, relation_raw, answer)` records from raw English QA with complete source provenance.
- `relation-taxonomy-normalization`: Build a global relation inventory and reproducibly apply a versioned taxonomy/mapping to extracted triples.
- `canonical-triple-postprocessing`: Freeze conservative canonical triples, factual prompts, and source-option distractor candidates with stage-level provenance.

### Retained Capabilities

- `raw-factual-source-sampling`: Deterministically sample or enumerate immutable raw English source records before model calls.

## Impact

- Removes the old `FACTUAL_AUDIT_MODEL`, tri-state decision schema, `prompt_en`, audit checkpoints, and audit supervisor.
- Adds two configured triple-label models and records endpoint-independent model IDs plus extraction prompt version per row.
- Keeps generated perturbation text, translation, and screening outside this experiment until normalized triples, prompts, and distractor candidates are reviewed and frozen.
- Uses the configured OpenAI-compatible endpoint for later strong-model work; preflight confirmed `qwen3.7-plus` is callable through `https://ctapi.csxdtx.com:16000/v1`.
