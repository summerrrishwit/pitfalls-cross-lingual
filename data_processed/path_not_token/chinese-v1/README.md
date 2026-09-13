# Chinese PATH_not_token preparation v1

This directory contains deterministic preprocessing artifacts only. No model, hidden-state, task-vector, or intervention experiment was run.

## Counts

- Source rows: 342
- Unique source questions: 52
- Frozen canonical base facts: 13
- Retained perturbation instances: 91
- Excluded source questions: 39
- Excluded source rows: 251

## Files

- `base_facts.jsonl`: one row per independent canonical fact, with bilingual completion and open-question prompts.
- `perturbation_instances.jsonl`: all retained legacy perturbations linked to their base fact; choices are provenance only and are not embedded in prompts.
- `relation_inventory.json`: extracted relation signatures and their current unfrozen probe-relation status.
- `excluded_base_questions.jsonl`: source questions that did not pass the frozen canonical atomic-fact gate.
- `summary.json`: counts, hashes, prompt contract, and explicit not-run stages.

## Boundary

`probe_relation_id` remains null and answer-token fields remain pending. The records are ready for downstream tokenization and baseline screening, but they are not yet admitted to a relation-conditioned PATH_not_token mechanism experiment.
