# Factual perturbation report: calibration-15-frozen-ollama-v6

- Gate passed: `false`
- Base triples: 20
- Translation reviews accepted: 18
- Distractors accepted: 41
- Perturbations accepted: 15
- Fully simulated candidates: 15
- Failed terminal calls: 0
- Expected outcomes per candidate: 12
- Baseline inconsistencies: 0
- Total attempts: 388
- Mean/max latency ms: 7018.75 / 87941
- Token usage: `{"input_tokens": 88326, "output_tokens": 64210, "total_tokens": 152536}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["currency_cost_not_auditable"]`

## Simulation-only execution

- Physical calls: 180
- Completed: 180
- Attempts/retries: 184 / 4
- Token usage: `{"input_tokens": 18547, "output_tokens": 1089, "total_tokens": 19636}`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| gemma3:12b | 90 | 90 | 0.0000 | 0 | 1243.41 | 10675 |
| gpt-5.5 | 20 | 20 | 0.0000 | 0 | 13462.35 | 23199 |
| gpt-5.6-sol | 35 | 35 | 0.0000 | 3 | 39722.14 | 87941 |
| llama3.1:8b | 90 | 90 | 0.0000 | 4 | 618.62 | 4692 |
| qwen3.8-max | 146 | 146 | 0.0000 | 0 | 5801.62 | 14839 |

## Baseline integrity

- Inconsistent shared baselines: 0
## Isolation

- Paths Not Taken remained disabled.
- No SenseNova model was called.
- Credentials and endpoint URLs are absent from artifacts and event logs.

## Three-arm analysis

- Complete candidates: 15/15
- Paths Not Taken candidates: 2
- Strict signal count by model: `{"gemma3:12b": 0, "llama3.1:8b": 2}`
- Shared baseline calls saved: 0
- Neutral-control instabilities: 9
- Conclusion: At least one target-distractor-specific Chinese flip survived the shared-baseline, neutral-control rerun.

### Model-level arm scores

| Model | Arm | Completed | Accuracy | Distractor-hit rate |
|---|---|---:|---:|---:|
| gemma3:12b | en_original | 15/15 | 0.866667 | 0.0 |
| gemma3:12b | en_neutral | 15/15 | 0.866667 | 0.066667 |
| gemma3:12b | en_targeted | 15/15 | 0.866667 | 0.133333 |
| gemma3:12b | zh_original | 15/15 | 0.733333 | 0.2 |
| gemma3:12b | zh_neutral | 15/15 | 0.866667 | 0.066667 |
| gemma3:12b | zh_targeted | 15/15 | 0.6 | 0.266667 |
| llama3.1:8b | en_original | 15/15 | 0.8 | 0.066667 |
| llama3.1:8b | en_neutral | 15/15 | 0.866667 | 0.066667 |
| llama3.1:8b | en_targeted | 15/15 | 0.866667 | 0.066667 |
| llama3.1:8b | zh_original | 15/15 | 0.666667 | 0.266667 |
| llama3.1:8b | zh_neutral | 15/15 | 0.8 | 0.133333 |
| llama3.1:8b | zh_targeted | 15/15 | 0.666667 | 0.333333 |
