# Factual perturbation report: diagnostic-100-five-model-v7b

- Gate passed: `false`
- Base triples: 100
- Translation reviews accepted: 93
- Distractors accepted: 219
- Perturbations accepted: 73
- Fully simulated candidates: 73
- Failed terminal calls: 1
- Expected outcomes per candidate: 30
- Baseline inconsistencies: 0
- Total attempts: 3032
- Mean/max latency ms: 6622.4 / 902657
- Token usage: `{"input_tokens": 468018, "output_tokens": 253153, "total_tokens": 721171}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["translation_incomplete", "currency_cost_not_auditable"]`

## Simulation-only execution

- Physical calls: 2190
- Completed: 2190
- Attempts/retries: 2252 / 62
- Token usage: `{"input_tokens": 215316, "output_tokens": 68036, "total_tokens": 283352}`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| bailian/deepseek-v3.2 | 438 | 438 | 0.0000 | 43 | 5099.54 | 902657 |
| gemini-3.7-flash | 438 | 438 | 0.0000 | 19 | 11578.8 | 197282 |
| gemma3:12b | 438 | 438 | 0.0000 | 0 | 5270.56 | 14030 |
| gpt-5.5 | 99 | 99 | 0.0000 | 0 | 11774.18 | 21821 |
| gpt-5.6-sol | 91 | 91 | 0.0000 | 6 | 41301.36 | 98099 |
| llama3.1:8b | 438 | 438 | 0.0000 | 0 | 3538.59 | 8431 |
| qwen3.6-27b | 438 | 438 | 0.0000 | 0 | 1121.42 | 2928 |
| qwen3.8-max | 581 | 580 | 0.0017 | 3 | 5215.4 | 16863 |

## Baseline integrity

- Inconsistent shared baselines: 0
## Isolation

- Paths Not Taken remained disabled.
- No SenseNova model was called.
- Credentials and endpoint URLs are absent from artifacts and event logs.

## Three-arm analysis

- Complete candidates: 73/73
- Paths Not Taken candidates: 14
- Strict signal count by model: `{"bailian/deepseek-v3.2": 2, "gemini-3.7-flash": 2, "gemma3:12b": 8, "llama3.1:8b": 2, "qwen3.6-27b": 4}`
- Shared baseline calls saved: 0
- Neutral-control instabilities: 41
- Conclusion: At least one target-distractor-specific Chinese flip survived the shared-baseline, neutral-control rerun.

### Model-level arm scores

| Model | Arm | Completed | Accuracy | Distractor-hit rate |
|---|---|---:|---:|---:|
| qwen3.6-27b | en_original | 73/73 | 0.958904 | 0.027397 |
| qwen3.6-27b | en_neutral | 73/73 | 0.931507 | 0.041096 |
| qwen3.6-27b | en_targeted | 73/73 | 0.917808 | 0.054795 |
| qwen3.6-27b | zh_original | 73/73 | 0.931507 | 0.041096 |
| qwen3.6-27b | zh_neutral | 73/73 | 0.890411 | 0.068493 |
| qwen3.6-27b | zh_targeted | 73/73 | 0.821918 | 0.136986 |
| bailian/deepseek-v3.2 | en_original | 73/73 | 0.958904 | 0.027397 |
| bailian/deepseek-v3.2 | en_neutral | 73/73 | 0.958904 | 0.013699 |
| bailian/deepseek-v3.2 | en_targeted | 73/73 | 0.90411 | 0.082192 |
| bailian/deepseek-v3.2 | zh_original | 73/73 | 0.931507 | 0.054795 |
| bailian/deepseek-v3.2 | zh_neutral | 73/73 | 0.931507 | 0.041096 |
| bailian/deepseek-v3.2 | zh_targeted | 73/73 | 0.890411 | 0.09589 |
| gemini-3.7-flash | en_original | 73/73 | 0.986301 | 0.0 |
| gemini-3.7-flash | en_neutral | 73/73 | 0.986301 | 0.0 |
| gemini-3.7-flash | en_targeted | 73/73 | 0.986301 | 0.013699 |
| gemini-3.7-flash | zh_original | 73/73 | 0.972603 | 0.013699 |
| gemini-3.7-flash | zh_neutral | 73/73 | 0.986301 | 0.0 |
| gemini-3.7-flash | zh_targeted | 73/73 | 0.945205 | 0.041096 |
| gemma3:12b | en_original | 73/73 | 0.835616 | 0.054795 |
| gemma3:12b | en_neutral | 73/73 | 0.849315 | 0.054795 |
| gemma3:12b | en_targeted | 73/73 | 0.780822 | 0.164384 |
| gemma3:12b | zh_original | 73/73 | 0.794521 | 0.109589 |
| gemma3:12b | zh_neutral | 73/73 | 0.835616 | 0.082192 |
| gemma3:12b | zh_targeted | 73/73 | 0.643836 | 0.315068 |
| llama3.1:8b | en_original | 73/73 | 0.863014 | 0.068493 |
| llama3.1:8b | en_neutral | 73/73 | 0.808219 | 0.068493 |
| llama3.1:8b | en_targeted | 73/73 | 0.753425 | 0.205479 |
| llama3.1:8b | zh_original | 73/73 | 0.767123 | 0.109589 |
| llama3.1:8b | zh_neutral | 73/73 | 0.753425 | 0.109589 |
| llama3.1:8b | zh_targeted | 73/73 | 0.726027 | 0.178082 |
