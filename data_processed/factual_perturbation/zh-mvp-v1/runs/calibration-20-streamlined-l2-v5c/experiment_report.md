# Factual perturbation report: calibration-20-streamlined-l2-v5c

- Gate passed: `false`
- Base triples: 20
- Translation reviews accepted: 18
- Distractors accepted: 41
- Perturbations accepted: 15
- Fully simulated candidates: 15
- Failed terminal calls: 0
- Expected outcomes per candidate: 24
- Baseline inconsistencies: 0
- Total attempts: 595
- Mean/max latency ms: 6851.64 / 87941
- Token usage: `{"input_tokens": 103359, "output_tokens": 81393, "total_tokens": 184752}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["currency_cost_not_auditable"]`

## Simulation-only execution

- Physical calls: 360
- Completed: 360
- Attempts/retries: 391 / 31
- Token usage: `{"input_tokens": 33580, "output_tokens": 18272, "total_tokens": 51852}`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| bailian/deepseek-v3.2 | 90 | 90 | 0.0000 | 0 | 891.52 | 1697 |
| gemini-3.7-flash | 90 | 90 | 0.0000 | 6 | 5756.29 | 20229 |
| gpt-5-mini | 90 | 90 | 0.0000 | 25 | 7557.21 | 37883 |
| gpt-5.5 | 20 | 20 | 0.0000 | 0 | 13462.35 | 23199 |
| gpt-5.6-sol | 35 | 35 | 0.0000 | 3 | 39722.14 | 87941 |
| qwen3.6-27b | 90 | 90 | 0.0000 | 0 | 652.86 | 1854 |
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
- Shared baseline calls saved: 0
- Neutral-control instabilities: 0
- Conclusion: At least one target-distractor-specific Chinese flip survived the shared-baseline, neutral-control rerun.
