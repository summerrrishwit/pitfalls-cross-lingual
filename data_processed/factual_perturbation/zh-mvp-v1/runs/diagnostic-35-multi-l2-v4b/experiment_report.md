# Factual perturbation report: diagnostic-35-multi-l2-v4b

- Gate passed: `false`
- Base triples: 100
- Translation reviews accepted: 82
- Distractors accepted: 126
- Perturbations accepted: 35
- Fully simulated candidates: 35
- Failed terminal calls: 13
- Expected outcomes per candidate: 24
- Baseline inconsistencies: 0
- Total attempts: 1816
- Mean/max latency ms: 8258.97 / 222701
- Token usage: `{"input_tokens": 291070, "output_tokens": 196335, "total_tokens": 487405}`
- Cost: provider_price_table_not_configured; token usage retained for billing reconciliation
- Gate reasons: `["currency_cost_not_auditable"]`

## Simulation-only execution

- Physical calls: 840
- Completed: 840
- Attempts/retries: 851 / 11
- Token usage: `{"input_tokens": 80466, "output_tokens": 36212, "total_tokens": 116678}`

## Per-model reliability

| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |
|---|---:|---:|---:|---:|---:|---:|
| bailian/deepseek-v3.2 | 210 | 210 | 0.0000 | 0 | 882.46 | 1588 |
| gemini-3.7-flash | 210 | 210 | 0.0000 | 2 | 5098.26 | 15903 |
| gpt-5-mini | 210 | 210 | 0.0000 | 9 | 7734.38 | 29737 |
| gpt-5.5 | 100 | 100 | 0.0000 | 0 | 8960.09 | 15079 |
| gpt-5.6-sol | 126 | 113 | 0.1032 | 247 | 45674.83 | 222701 |
| qwen3.6-27b | 210 | 210 | 0.0000 | 0 | 609.27 | 3617 |
| qwen3.8-max | 488 | 488 | 0.0000 | 4 | 6506.72 | 194541 |

## Baseline integrity

- Inconsistent shared baselines: 0
## Isolation

- Paths Not Taken remained disabled.
- No SenseNova model was called.
- Credentials and endpoint URLs are absent from artifacts and event logs.

## Three-arm analysis

- Complete candidates: 35/35
- Paths Not Taken candidates: 1
- Shared baseline calls saved: 0
- Neutral-control instabilities: 2
- Conclusion: At least one target-distractor-specific Chinese flip survived the shared-baseline, neutral-control rerun.
