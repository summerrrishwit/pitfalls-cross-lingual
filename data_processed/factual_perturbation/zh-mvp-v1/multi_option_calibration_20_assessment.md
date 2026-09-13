# Chinese multi-option endpoint calibration (20 independent base triples)

## Decision

The 20-base-triple calibration is complete, but it did not pass the predeclared signal gate. Run a 100-base-triple **diagnostic Chinese pilot** next, with the denominator frozen at input base triples and all attrition reported. Do not freeze candidates, call `claude-opus-4-8`, or enable Paths Not Taken yet.

## Design and completion

- Independent unit: one canonical base triple.
- Final calibration sample: 20 distinct `source_id` values selected from 50 screened upstream base triples.
- Endpoint: three-option multiple choice with identical semantic option ordering in English and Chinese.
- Arms: Original, candidate-matched Neutral, and L2 Targeted.
- Simulation panel: `qwen3.6-27b`, `bailian/deepseek-v3.2`, `gpt-5-mini`, and `gemini-3.7-flash`.
- Codex review provenance: `reviewer_type=codex_proxy`, `not_human_gold=true`.
- Completion: 480/480 Simulation calls, 20/20 complete candidates, 120 calls per model, and every call had `option_count=3`.
- Safety boundary: `paths_not_taken_enabled=false`; holdout was not called.

The calibration is composed of three disjoint completed runs:

| Run | Independent bases | Simulation calls |
|---|---:|---:|
| `screen-10-multi-l2-v4` | 10 | 240/240 |
| `calibration-8b-multi-l2-v4` | 8 | 192/192 |
| `calibration-2c-multi-l2-v4` | 2 | 48/48 |

## Primary result

The primary endpoint required a model to be correct in Chinese Original and matched Neutral, become wrong under Chinese L2 Targeted, and choose the specifically manipulated distractor while English Targeted remained correct.

- Target-distractor-specific Chinese flips: **0/20 base triples** and **0/80 model-by-base comparisons**.
- The predeclared continuation threshold of 2–3 independent, repeatable flips was not met.
- A rough one-sided 95% upper bound from 0/20 is about 14%, but the screened and filtered sample is not an IID population sample, so this is only a sensitivity heuristic.

| Language / arm | Correct | Target distractor hits |
|---|---:|---:|
| English Original | 80/80 | 0/80 |
| English Neutral | 80/80 | 0/80 |
| English Targeted | 80/80 | 0/80 |
| Chinese Original | 78/80 | 1/80 |
| Chinese Neutral | 79/80 | 1/80 |
| Chinese Targeted | 79/80 | 0/80 |

One `qwen3.6-27b` response became wrong only in Chinese Targeted for the `Psycho` item, but it selected the non-target foil (`ketchup`) rather than the manipulated target (`tomato juice`). This is retained as a secondary category-level spillover observation, not counted as the primary targeted effect.

One `bailian/deepseek-v3.2` Chinese control changed between Original and Neutral on the largest-desert item. The targeted arm was correct, so this is control instability rather than evidence for the perturbation.

## Operational audit

- Simulation retries: 4 across 484 attempts; final terminal failures: 0.
- Simulation latency: mean 2,636.77 ms, p95 8,446.55 ms, maximum 14,800 ms.
- Simulation token usage: 64,499 total.
- Numeric currency cost remains unauditable because the repository has no versioned provider price table. Each run therefore retains `currency_cost_not_auditable` as its only gate reason.
- Upstream generation had resumable structured-output failures; reruns recovered all selected calibration candidates without replaying completed items.

## Attrition and limitations

- 50 upstream base triples were screened to obtain 20 reviewed, structurally eligible calibration bases (40% observed yield). The main losses were translation rejection, fewer than two accepted distractors, generation failures, invalid canonical facts, implausible foils, and semantically duplicate options.
- Final source mix: MMLU 5, SciQ 5, AI2 ARC Easy 5, TruthfulQA 4, CommonsenseQA 1. This is broad but not balanced and contains only one CommonsenseQA item.
- Relation mapping did not gate prompts: 12 selected bases were mapped and 8 were `out_of_taxonomy`.
- The near-ceiling English and Chinese accuracy shows that the endpoint is operationally stable, but the L2 contexts do not reliably move probability mass onto the intended distractor for this model panel.

## Next experiment

Choose the 100-base-triple diagnostic Chinese pilot before switching language. This keeps the now-validated pipeline fixed and estimates whether target-specific effects are merely rare. Pre-register the denominator as 100 input base triples, preserve the strict target-hit primary endpoint, report all attrition, and treat non-target wrong answers as a separate secondary endpoint.

If that diagnostic pilot again yields no target-specific signal, pre-register a separate alternate-language transfer screen. Do not reinterpret an alternate-language result as part of the Chinese cohort.
