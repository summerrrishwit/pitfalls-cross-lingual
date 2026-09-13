# Factual perturbation report: screen-20-three-arm-v1

- Gate passed: `false`
- Base triples: 20
- Translation reviews accepted: 15
- Distractors accepted: 27
- Perturbations accepted: 12
- Fully simulated candidates: 11
- Failed terminal calls: 1
- Expected outcomes per candidate: 24
- Baseline inconsistencies: 0
- Total attempts: 494
- Mean/max latency ms: 17786.96 / 295216
- Token usage: `{"input_tokens": 60958, "output_tokens": 58498, "total_tokens": 119456}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["incomplete_four_model_simulation", "currency_cost_not_auditable"]`

## Simulation-only execution

- Physical calls: 288
- Completed: 287
- Attempts/retries: 339 / 51
- Token usage: `{"input_tokens": 24035, "output_tokens": 33012, "total_tokens": 57047}`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| bailian/deepseek-v4-flash-0731 | 72 | 72 | 0.0000 | 0 | 1538.58 | 2889 |
| gemini-3.7-flash | 72 | 72 | 0.0000 | 1 | 25479.22 | 214533 |
| gpt-5.5 | 20 | 20 | 0.0000 | 0 | 8381.5 | 11709 |
| gpt-5.6-sol | 27 | 27 | 0.0000 | 4 | 19643.04 | 85441 |
| hy4-preview | 72 | 71 | 0.0139 | 50 | 63847.17 | 295216 |
| qwen3.7-plus | 72 | 72 | 0.0000 | 0 | 1099.12 | 2293 |
| qwen3.8-max | 104 | 104 | 0.0000 | 0 | 4702.57 | 8998 |

## Baseline integrity

- Inconsistent shared baselines: 0
## Isolation

- Paths Not Taken remained disabled.
- No SenseNova model was called.
- Credentials and endpoint URLs are absent from artifacts and event logs.

## Three-arm analysis

- Complete candidates: 11/12
- Paths Not Taken candidates: 0
- Shared baseline calls saved: 0
- Neutral-control instabilities: 2
- Conclusion: No targeted Chinese accuracy degradation survived the shared-baseline, neutral-control rerun.
