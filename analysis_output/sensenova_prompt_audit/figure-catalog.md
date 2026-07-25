# Figure catalog

## figure-01-decision-distribution

- Files: `figures/figure-01-decision-distribution.png`, `figures/figure-01-decision-distribution.pdf`
- Purpose: Show whether the optimized prompt materially shifts the three audit outcomes.
- Data source: Paired 600-candidate SenseNova v4 and v7 JSONL outputs.
- Caption requirements: State that bars are exact counts on the same candidates and have no error bars because this is a fixed paired pilot.
- Observation: v7 mainly increases `needs_review` while leaving accept volume similar.
- Interpretation: The optimization is conservative routing, not evidence of accuracy improvement.
- Caveat: No human gold labels.

## figure-02-v4-v7-transition

- Files: `figures/figure-02-v4-v7-transition.png`, `figures/figure-02-v4-v7-transition.pdf`
- Purpose: Locate exactly which decision categories changed.
- Data source: Candidate-level paired v4/v7 decisions.
- Caption requirements: Rows are v4, columns are v7, cell text is candidate count.
- Observation: Most records stay on the diagonal; no v4 `needs_review` item leaves review.
- Interpretation: The new prompt preserves most decisions and adds conservative review routes.
- Caveat: Transition direction does not imply correctness.
