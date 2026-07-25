# Statistical appendix

## Comparison definitions

- Metric direction: contract-valid rate and agreement are higher-is-more; decision shares are descriptive only.
- Unit: paired candidate.
- Repeated measure: the same candidate audited by v4 and v7.
- Sampling caveat: the 600 records are a stratified project pilot, not an iid population sample.
- Multiplicity: no family of accuracy hypotheses was tested. One McNemar test describes the pre-specified review-routing change.

## Exact summaries

| Run | N | Accept | Reject | Needs review | Raw contract-valid |
|---|---:|---:|---:|---:|---:|
| SenseNova v4 | 600 | 116 | 472 | 12 | 600 |
| SenseNova v7 | 600 | 113 | 457 | 30 | 597 |

## v4 to v7 transition matrix

Rows are v4 decisions; columns are v7 decisions.

| v4 \ v7 | Accept | Reject | Needs review |
|---|---:|---:|---:|
| accept | 108 | 4 | 4 |
| reject | 5 | 453 | 14 |
| needs_review | 0 | 0 | 12 |

## Agreement statistics

| Comparison | N | Exact agreement | Cohen's kappa | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| SenseNova v4 vs v7 | 600 | 573/600 | 0.876 | [0.829, 0.920] |
| Qwen v4 vs SenseNova v4 | 100 | 83/100 | 0.676 | [0.531, 0.807] |
| Qwen v4 vs SenseNova v7 | 100 | 80/100 | 0.639 | [0.498, 0.767] |

Bootstrap intervals use 10,000 candidate-level paired resamples with seed 20260725.

## Inferential boundary

The exact McNemar test for review routing compares 18 non-review→review changes against 0 review→non-review changes (p=0.00000763). It demonstrates an asymmetric routing change on this fixed pilot. It does not establish better labels. Parametric normality and variance tests are inapplicable to paired categorical decisions.
