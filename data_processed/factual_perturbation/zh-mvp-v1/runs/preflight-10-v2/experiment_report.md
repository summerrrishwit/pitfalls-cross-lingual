# Factual perturbation report: preflight-10-v2

- Gate passed: `false`
- Base triples: 10
- Translation reviews accepted: 7
- Distractors accepted: 11
- Perturbations accepted: 16
- Fully simulated candidates: 15
- Failed terminal calls: 2
- Total attempts: 435
- Mean/max latency ms: 20556.39 / 441399
- Token usage: `{"input_tokens": 37069, "output_tokens": 36628, "total_tokens": 73697}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["incomplete_four_model_simulation", "currency_cost_not_auditable"]`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| bailian/deepseek-v4-flash-0731 | 64 | 64 | 0.0000 | 0 | 1323.41 | 1970 |
| gemini-3.7-flash | 64 | 64 | 0.0000 | 0 | 23655.75 | 43070 |
| gpt-5.5 | 10 | 10 | 0.0000 | 0 | 9927.0 | 15044 |
| gpt-5.6-sol | 11 | 11 | 0.0000 | 0 | 20382.73 | 30547 |
| hy4-preview | 64 | 62 | 0.0312 | 112 | 69879.8 | 441399 |
| qwen3.7-plus | 64 | 64 | 0.0000 | 0 | 1134.41 | 3156 |
| qwen3.8-max | 46 | 46 | 0.0000 | 0 | 3753.46 | 8078 |

## Isolation

- Paths Not Taken remained disabled.
- No SenseNova model was called.
- Credentials and endpoint URLs are absent from artifacts and event logs.
