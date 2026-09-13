# Chinese sensitivity and stronger-perturbation assessment

## Decision

Do not start the 100-base-triple pilot. The revised model panel and the stronger truthful L2 perturbations both completed successfully, but neither produced a replicated Chinese-specific degradation. Keep Paths Not Taken and the `claude-opus-4-8` holdout disabled.

All new semantic decisions are Codex proxy review with `not_human_gold=true`. The proxy review was completed before the L2 Simulation outputs were observed.

## Model-panel calibration on the frozen 12 candidates

- Models: `qwen3.6-27b`, `bailian/deepseek-v3.2`, `gpt-5-mini`, and `gemini-3.7-flash`.
- Minimal and representative structured probes passed for all three newly introduced models. The first representative `gpt-5-mini` probe returned HTTP 500; its isolated retry passed.
- Simulation completed 288/288 physical calls and all 12 candidates had complete 24-cell coverage.
- One nominal model-level Chinese flip occurred for `qwen3.6-27b` on `raw_commonsense_qa_001993_source_wrong_1_p2`.
- That item is not admissible as cross-lingual evidence: its English distractor is misspelled as `string quintent`, while the Chinese version is the normal equivalent of string quintet. It was excluded from the stronger-perturbation screen.
- `gpt-5-mini` completed all 72 calls but required 11 retries and reached 444,059 ms maximum cumulative latency in this first batch.

## Stronger L2 perturbation funnel

| Stage | Count |
|---|---:|
| Frozen source/distractor pairs | 12 |
| Generated L1/L2 candidate pairs | 12 |
| Candidates reviewed by Codex proxy | 24 |
| Automatically accepted | 13/24 |
| Codex proxy accepted | 22/24 |
| Invalid bilingual distractor pairs excluded | 1 |
| L2 candidates admitted to Simulation | 11 |
| L2 Simulation calls completed | 264/264 |
| Complete L2 candidates | 11/11 |
| Confirmed Chinese targeted degradations | 0 |

The generation stage initially truncated six matched-pair outputs under a 768-token budget. The budget increase to 2,048 tokens was recorded as a manifest amendment, and resume completed all 23 generated distractor groups without replaying completed groups.

## L2 behavioral result

Across the 44 model-by-candidate comparisons per language:

| Language | Original correct | Matched Neutral correct | L2 Targeted correct | Qualified targeted flips |
|---|---:|---:|---:|---:|
| English | 43/44 | 43/44 | 44/44 | 0 |
| Chinese | 43/44 | 43/44 | 43/44 | 0 |

Per-model results:

| Model | Calls completed | Overall correct | Retries | Mean latency (ms) | Max latency (ms) |
|---|---:|---:|---:|---:|---:|
| `qwen3.6-27b` | 66/66 | 61/66 | 0 | 571.80 | 1,776 |
| `bailian/deepseek-v3.2` | 66/66 | 66/66 | 0 | 813.39 | 1,712 |
| `gpt-5-mini` | 66/66 | 66/66 | 0 | 3,287.76 | 5,531 |
| `gemini-3.7-flash` | 66/66 | 66/66 | 2 | 5,730.95 | 27,162 |

Simulation used 33,653 total tokens. Currency cost remains unauditable because no versioned provider price table is configured.

## Interpretation

The earlier ceiling explanation was only partial. Replacing two models with lower-capacity variants created slightly more baseline error, but valid L2 targeted contexts still did not produce any Original-and-Neutral-correct to Targeted-wrong transition. Stronger context also did not reduce aggregate Chinese accuracy.

The remaining bottleneck is the response endpoint: a two-choice factual item exposes only a coarse correctness bit. The next Chinese experiment should use original multi-option questions where available or a probability/rank-shift endpoint. Expanding the unchanged binary endpoint to 100 would mainly estimate a near-ceiling rate more precisely.

If the revised endpoint remains insensitive in Chinese, a predeclared alternate-language screen may be run to test whether the pipeline transfers. That language change must be a new experiment rather than a post-hoc reinterpretation of the Chinese cohort.

## Operational status

- Model routing, schema checks, retry logging, single-writer checkpoints, and resume succeeded.
- Matched candidate-level Neutral contexts and L1/L2 generation are implemented.
- No valid candidate requires adaptive replication because the valid L2 screen contained no arm disagreement.
- The 100-item pilot, candidate freeze, Claude holdout, and Paths Not Taken remain blocked.
