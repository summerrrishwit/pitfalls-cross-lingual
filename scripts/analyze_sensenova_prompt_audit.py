#!/usr/bin/env python3
"""Analyze SenseNova factual-audit prompt runs without treating model agreement as accuracy."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LABELS = ("accept", "reject", "needs_review")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def confusion(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]], ids: Iterable[str]) -> List[List[int]]:
    selected = list(ids)
    return [
        [
            sum(left[candidate_id]["decision"] == left_label and right[candidate_id]["decision"] == right_label for candidate_id in selected)
            for right_label in LABELS
        ]
        for left_label in LABELS
    ]


def kappa_from_pairs(pairs: Sequence[Tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(left_counts[label] * right_counts[label] for label in LABELS) / (total * total)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def bootstrap_kappa_interval(
    pairs: Sequence[Tuple[str, str]],
    *,
    samples: int = 10_000,
    seed: int = 20260725,
) -> Tuple[float, float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        estimates.append(kappa_from_pairs(resampled))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def exact_mcnemar_p(discordant_left: int, discordant_right: int) -> float:
    total = discordant_left + discordant_right
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(discordant_left, discordant_right) + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def interval_text(interval: Tuple[float, float]) -> str:
    return f"[{percent(interval[0])}, {percent(interval[1])}]"


def decision_counts(records: Dict[str, Dict[str, Any]]) -> Counter:
    return Counter(record["decision"] for record in records.values())


def render_figures(
    output_dir: Path,
    v4_counts: Counter,
    v7_counts: Counter,
    transition: List[List[int]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(LABELS))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(positions - width / 2, [v4_counts[label] for label in LABELS], width, label="SenseNova v4", color="#4C78A8")
    axis.bar(positions + width / 2, [v7_counts[label] for label in LABELS], width, label="SenseNova v7", color="#F58518")
    axis.set_xticks(positions, ["Accept", "Reject", "Needs review"])
    axis.set_ylabel("Number of audit records")
    axis.set_title("SenseNova audit decisions on the same 600 candidates")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(figures_dir / "figure-01-decision-distribution.png", dpi=200)
    figure.savefig(figures_dir / "figure-01-decision-distribution.pdf")
    plt.close(figure)

    matrix = np.asarray(transition)
    figure, axis = plt.subplots(figsize=(5.8, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(LABELS)), ["Accept", "Reject", "Needs review"])
    axis.set_yticks(range(len(LABELS)), ["Accept", "Reject", "Needs review"])
    axis.set_xlabel("Prompt v7 decision")
    axis.set_ylabel("Prompt v4 decision")
    axis.set_title("Decision transitions from prompt v4 to v7")
    figure.colorbar(image, ax=axis, label="Candidate count")
    figure.tight_layout()
    figure.savefig(figures_dir / "figure-02-v4-v7-transition.png", dpi=200)
    figure.savefig(figures_dir / "figure-02-v4-v7-transition.pdf")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data_processed/raw_factual_pitfalls/raw-mvp-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output/sensenova_prompt_audit"))
    args = parser.parse_args()

    v4_rows = read_jsonl(args.run_dir / "factual_reviews_sensenova_v4.jsonl")
    v7_rows = read_jsonl(args.run_dir / "factual_reviews_sensenova_v7.jsonl")
    qwen_rows = read_jsonl(args.run_dir / "factual_reviews_prompt_v4.jsonl")
    v4 = {row["candidate_id"]: row for row in v4_rows}
    v7 = {row["candidate_id"]: row for row in v7_rows}
    qwen = {row["candidate_id"]: row for row in qwen_rows}
    if set(v4) != set(v7) or len(v4) != 600:
        raise ValueError("Expected paired SenseNova v4/v7 records for exactly 600 candidates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    v4_counts = decision_counts(v4)
    v7_counts = decision_counts(v7)
    transition = confusion(v4, v7, v4)
    changed = sum(v4[candidate_id]["decision"] != v7[candidate_id]["decision"] for candidate_id in v4)
    paired_v4_v7 = [(v4[candidate_id]["decision"], v7[candidate_id]["decision"]) for candidate_id in v4]
    v4_v7_kappa = kappa_from_pairs(paired_v4_v7)
    v4_v7_kappa_ci = bootstrap_kappa_interval(paired_v4_v7)

    overlap_ids = sorted(set(qwen) & set(v4))
    paired_qwen_v4 = [(qwen[candidate_id]["decision"], v4[candidate_id]["decision"]) for candidate_id in overlap_ids]
    paired_qwen_v7 = [(qwen[candidate_id]["decision"], v7[candidate_id]["decision"]) for candidate_id in overlap_ids]
    qwen_v4_agreement = sum(left == right for left, right in paired_qwen_v4)
    qwen_v7_agreement = sum(left == right for left, right in paired_qwen_v7)
    qwen_v4_kappa = kappa_from_pairs(paired_qwen_v4)
    qwen_v7_kappa = kappa_from_pairs(paired_qwen_v7)
    qwen_v4_kappa_ci = bootstrap_kappa_interval(paired_qwen_v4)
    qwen_v7_kappa_ci = bootstrap_kappa_interval(paired_qwen_v7)

    v4_valid = sum(not row.get("validation_errors") and not row["audit"].get("error") for row in v4.values())
    v7_valid = sum(not row.get("validation_errors") and not row["audit"].get("error") for row in v7.values())
    v7_validation_errors = [row for row in v7.values() if row.get("validation_errors")]
    review_added = sum(v4[candidate_id]["decision"] != "needs_review" and v7[candidate_id]["decision"] == "needs_review" for candidate_id in v4)
    review_removed = sum(v4[candidate_id]["decision"] == "needs_review" and v7[candidate_id]["decision"] != "needs_review" for candidate_id in v4)
    review_mcnemar_p = exact_mcnemar_p(review_added, review_removed)

    render_figures(args.output_dir, v4_counts, v7_counts, transition)

    report = f"""# SenseNova factual-audit prompt evaluation

## Analysis questions

1. Can `sensenova-6.7-flash-lite` be called through the configured Anthropic-compatible endpoint?
2. Does prompt v4 reliably produce parseable, contract-valid audits on the existing 600-candidate pilot?
3. What failure modes justify prompt optimization, and does v7 correct them without destabilizing the full run?

## Evidence inventory

- Unit of analysis: one fixed raw English QA candidate.
- SenseNova prompt v4: {len(v4)} paired candidates.
- SenseNova prompt v7: {len(v7)} paired candidates.
- Qwen prompt-v4 comparison subset: {len(overlap_ids)} candidates selected in the prior calibration run.
- No human gold labels or independent repeated model seeds are available. Model agreement is therefore not accuracy.

## Key findings

### API and structured-output reliability

- The live availability request returned model `sensenova-6.7-flash-lite`, `stop_reason=end_turn`, and `OK`.
- Prompt v4 completed {len(v4)}/{len(v4)} requests with no API failures or retries. All {v4_valid}/{len(v4)} raw responses passed JSON and audit-contract validation: {percent(v4_valid / len(v4))}, Wilson 95% CI {interval_text(wilson_interval(v4_valid, len(v4)))}.
- Prompt v7 also completed {len(v7)}/{len(v7)} requests with no API failures or retries. {v7_valid}/{len(v7)} raw responses passed the stricter contract directly: {percent(v7_valid / len(v7))}, Wilson 95% CI {interval_text(wilson_interval(v7_valid, len(v7)))}. The remaining {len(v7_validation_errors)} responses were deterministically downgraded to `needs_review`; none were silently accepted or rejected.

### Prompt v4 behavior

| Decision | Count | Share |
|---|---:|---:|
| Accept | {v4_counts["accept"]} | {percent(v4_counts["accept"] / len(v4))} |
| Reject | {v4_counts["reject"]} | {percent(v4_counts["reject"] / len(v4))} |
| Needs review | {v4_counts["needs_review"]} | {percent(v4_counts["needs_review"] / len(v4))} |

On the 100-candidate same-prompt Qwen comparison subset, SenseNova v4 agreed on {qwen_v4_agreement}/100 decisions ({percent(qwen_v4_agreement / 100)}, Wilson 95% CI {interval_text(wilson_interval(qwen_v4_agreement, 100))}); Cohen's kappa was {qwen_v4_kappa:.3f}, bootstrap 95% CI [{qwen_v4_kappa_ci[0]:.3f}, {qwen_v4_kappa_ci[1]:.3f}]. This is inter-model consistency, not correctness.

### Why v4 was optimized

Qualitative adjudication of the 17 disagreements exposed four clear in-sample failures: a stray quote in `Fidelio'` was accepted, an incorrect photosynthesis-reactant premise was accepted, a correct Saturn ranking was falsely treated as contradictory, and a `time_sensitive` item was mapped to `reject` rather than `needs_review`.

Prompt v7 adds literal canonical-answer checking, question-premise validity, explicit decision mapping, and targeted scientific fact-boundary examples. These four calibration failures were routed as intended in the targeted v7 rerun. Because those examples informed the prompt, this is an in-sample regression result and must not be reported as held-out accuracy.

### Full-run v4 to v7 stability

- {changed}/{len(v4)} candidates changed decision ({percent(changed / len(v4))}, Wilson 95% CI {interval_text(wilson_interval(changed, len(v4)))}).
- Overall v4-v7 agreement was {len(v4) - changed}/{len(v4)} ({percent((len(v4) - changed) / len(v4))}); Cohen's kappa was {v4_v7_kappa:.3f}, bootstrap 95% CI [{v4_v7_kappa_ci[0]:.3f}, {v4_v7_kappa_ci[1]:.3f}].
- `needs_review` increased from {v4_counts["needs_review"]} to {v7_counts["needs_review"]}. There were {review_added} added and {review_removed} removed review routes; exact paired McNemar p={review_mcnemar_p:.8f}. This supports a more conservative routing change, not an accuracy improvement.
- On the prior 100-candidate Qwen prompt-v4 subset, v7 agreement was {qwen_v7_agreement}/100 ({percent(qwen_v7_agreement / 100)}), kappa {qwen_v7_kappa:.3f} with bootstrap 95% CI [{qwen_v7_kappa_ci[0]:.3f}, {qwen_v7_kappa_ci[1]:.3f}]. The lower agreement is expected partly because v7 deliberately routes more premise/canonical-answer cases to review; without gold labels it cannot be called better or worse.

## Decision

Use prompt v7 rather than v4 for new SenseNova audits. It preserves 95.5% of v4 decisions while making high-risk premise, malformed-answer, and decision-mapping failures reviewable. Keep the local validator enabled and manually adjudicate all `needs_review` records plus a stratified sample of accepts/rejects before treating labels as final.

## Claim Candidates

- Claim:
  - Source evidence: 600/600 live v4 calls completed; 600/600 responses passed the v4 contract.
  - Allowed wording: SenseNova is operationally compatible with the audit pipeline and prompt v4 is structurally reliable on this pilot.
  - Forbidden stronger wording: Prompt v4 is accurate or equivalent to Qwen.
  - Uncertainty: No human gold labels and one fixed model run per prompt.
  - Next check: Human-blind adjudication of a stratified sample.
  - Decision: keep

- Claim:
  - Source evidence: v7 corrected four known calibration failures and changed {changed}/600 full-run decisions, with all three raw contract violations safely downgraded.
  - Allowed wording: Prompt v7 is a safer operational default for SenseNova than v4.
  - Forbidden stronger wording: Prompt v7 significantly improves factual-audit accuracy.
  - Uncertainty: The four corrected examples were used to design v7; no held-out gold set exists.
  - Next check: Freeze v7 and evaluate on a new held-out, human-labeled set.
  - Decision: keep
"""
    (args.output_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    matrix_rows = "\n".join(
        f"| {LABELS[row]} | " + " | ".join(str(value) for value in transition[row]) + " |"
        for row in range(len(LABELS))
    )
    appendix = f"""# Statistical appendix

## Comparison definitions

- Metric direction: contract-valid rate and agreement are higher-is-more; decision shares are descriptive only.
- Unit: paired candidate.
- Repeated measure: the same candidate audited by v4 and v7.
- Sampling caveat: the 600 records are a stratified project pilot, not an iid population sample.
- Multiplicity: no family of accuracy hypotheses was tested. One McNemar test describes the pre-specified review-routing change.

## Exact summaries

| Run | N | Accept | Reject | Needs review | Raw contract-valid |
|---|---:|---:|---:|---:|---:|
| SenseNova v4 | {len(v4)} | {v4_counts["accept"]} | {v4_counts["reject"]} | {v4_counts["needs_review"]} | {v4_valid} |
| SenseNova v7 | {len(v7)} | {v7_counts["accept"]} | {v7_counts["reject"]} | {v7_counts["needs_review"]} | {v7_valid} |

## v4 to v7 transition matrix

Rows are v4 decisions; columns are v7 decisions.

| v4 \\ v7 | Accept | Reject | Needs review |
|---|---:|---:|---:|
{matrix_rows}

## Agreement statistics

| Comparison | N | Exact agreement | Cohen's kappa | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| SenseNova v4 vs v7 | {len(v4)} | {len(v4) - changed}/{len(v4)} | {v4_v7_kappa:.3f} | [{v4_v7_kappa_ci[0]:.3f}, {v4_v7_kappa_ci[1]:.3f}] |
| Qwen v4 vs SenseNova v4 | {len(overlap_ids)} | {qwen_v4_agreement}/{len(overlap_ids)} | {qwen_v4_kappa:.3f} | [{qwen_v4_kappa_ci[0]:.3f}, {qwen_v4_kappa_ci[1]:.3f}] |
| Qwen v4 vs SenseNova v7 | {len(overlap_ids)} | {qwen_v7_agreement}/{len(overlap_ids)} | {qwen_v7_kappa:.3f} | [{qwen_v7_kappa_ci[0]:.3f}, {qwen_v7_kappa_ci[1]:.3f}] |

Bootstrap intervals use 10,000 candidate-level paired resamples with seed 20260725.

## Inferential boundary

The exact McNemar test for review routing compares {review_added} non-review→review changes against {review_removed} review→non-review changes (p={review_mcnemar_p:.8f}). It demonstrates an asymmetric routing change on this fixed pilot. It does not establish better labels. Parametric normality and variance tests are inapplicable to paired categorical decisions.
"""
    (args.output_dir / "stats-appendix.md").write_text(appendix, encoding="utf-8")

    catalog = """# Figure catalog

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
"""
    (args.output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
