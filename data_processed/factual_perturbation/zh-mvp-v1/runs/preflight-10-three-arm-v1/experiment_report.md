# Factual perturbation report: preflight-10-three-arm-v1

- Gate passed: `false`
- Base triples: 10
- Translation reviews accepted: 7
- Distractors accepted: 11
- Perturbations accepted: 6
- Fully simulated candidates: 6
- Failed terminal calls: 0
- Expected outcomes per candidate: 24
- Baseline inconsistencies: 0
- Total attempts: 187
- Mean/max latency ms: 15492.23 / 197070
- Token usage: `{"input_tokens": 24685, "output_tokens": 21249, "total_tokens": 45934}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["currency_cost_not_auditable"]`

## Simulation-only execution

- Physical calls: 112
- Completed: 112
- Attempts/retries: 120 / 8
- Token usage: `{"input_tokens": 9510, "output_tokens": 11782, "total_tokens": 21292}`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| bailian/deepseek-v4-flash-0731 | 28 | 28 | 0.0000 | 0 | 1980.96 | 6692 |
| gemini-3.7-flash | 28 | 28 | 0.0000 | 1 | 29352.25 | 70610 |
| gpt-5.5 | 10 | 10 | 0.0000 | 0 | 9927.0 | 15044 |
| gpt-5.6-sol | 11 | 11 | 0.0000 | 0 | 20382.73 | 30547 |
| hy4-preview | 28 | 28 | 0.0000 | 7 | 48508.54 | 197070 |
| qwen3.7-plus | 28 | 28 | 0.0000 | 0 | 1478.61 | 2898 |
| qwen3.8-max | 46 | 46 | 0.0000 | 0 | 3753.46 | 8078 |

## Baseline integrity

- Inconsistent shared baselines: 0
## Isolation

- Paths Not Taken remained disabled.
- No SenseNova model was called.
- Credentials and endpoint URLs are absent from artifacts and event logs.

## Three-arm analysis

- Complete candidates: 6/6
- Paths Not Taken candidates: 0
- Shared baseline calls saved: 32
- Neutral-control instabilities: 1
- Conclusion: No targeted Chinese accuracy degradation survived the shared-baseline, neutral-control rerun.
