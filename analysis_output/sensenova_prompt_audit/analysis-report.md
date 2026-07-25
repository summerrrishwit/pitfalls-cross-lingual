# SenseNova factual-audit prompt evaluation

## Analysis questions

1. Can `sensenova-6.7-flash-lite` be called through the configured Anthropic-compatible endpoint?
2. Does prompt v4 reliably produce parseable, contract-valid audits on the existing 600-candidate pilot?
3. What failure modes justify prompt optimization, and does v7 correct them without destabilizing the full run?

## Evidence inventory

- Unit of analysis: one fixed raw English QA candidate.
- SenseNova prompt v4: 600 paired candidates.
- SenseNova prompt v7: 600 paired candidates.
- Qwen prompt-v4 comparison subset: 100 candidates selected in the prior calibration run.
- No human gold labels or independent repeated model seeds are available. Model agreement is therefore not accuracy.

## Key findings

### API and structured-output reliability

- The live availability request returned model `sensenova-6.7-flash-lite`, `stop_reason=end_turn`, and `OK`.
- Prompt v4 completed 600/600 requests with no API failures or retries. All 600/600 raw responses passed JSON and audit-contract validation: 100.0%, Wilson 95% CI [99.4%, 100.0%].
- Prompt v7 also completed 600/600 requests with no API failures or retries. 597/600 raw responses passed the stricter contract directly: 99.5%, Wilson 95% CI [98.5%, 99.8%]. The remaining 3 responses were deterministically downgraded to `needs_review`; none were silently accepted or rejected.

### Prompt v4 behavior

| Decision | Count | Share |
|---|---:|---:|
| Accept | 116 | 19.3% |
| Reject | 472 | 78.7% |
| Needs review | 12 | 2.0% |

On the 100-candidate same-prompt Qwen comparison subset, SenseNova v4 agreed on 83/100 decisions (83.0%, Wilson 95% CI [74.5%, 89.1%]); Cohen's kappa was 0.676, bootstrap 95% CI [0.531, 0.807]. This is inter-model consistency, not correctness.

### Why v4 was optimized

Qualitative adjudication of the 17 disagreements exposed four clear in-sample failures: a stray quote in `Fidelio'` was accepted, an incorrect photosynthesis-reactant premise was accepted, a correct Saturn ranking was falsely treated as contradictory, and a `time_sensitive` item was mapped to `reject` rather than `needs_review`.

Prompt v7 adds literal canonical-answer checking, question-premise validity, explicit decision mapping, and targeted scientific fact-boundary examples. These four calibration failures were routed as intended in the targeted v7 rerun. Because those examples informed the prompt, this is an in-sample regression result and must not be reported as held-out accuracy.

### Full-run v4 to v7 stability

- 27/600 candidates changed decision (4.5%, Wilson 95% CI [3.1%, 6.5%]).
- Overall v4-v7 agreement was 573/600 (95.5%); Cohen's kappa was 0.876, bootstrap 95% CI [0.829, 0.920].
- `needs_review` increased from 12 to 30. There were 18 added and 0 removed review routes; exact paired McNemar p=0.00000763. This supports a more conservative routing change, not an accuracy improvement.
- On the prior 100-candidate Qwen prompt-v4 subset, v7 agreement was 80/100 (80.0%), kappa 0.639 with bootstrap 95% CI [0.498, 0.767]. The lower agreement is expected partly because v7 deliberately routes more premise/canonical-answer cases to review; without gold labels it cannot be called better or worse.

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
  - Source evidence: v7 corrected four known calibration failures and changed 27/600 full-run decisions, with all three raw contract violations safely downgraded.
  - Allowed wording: Prompt v7 is a safer operational default for SenseNova than v4.
  - Forbidden stronger wording: Prompt v7 significantly improves factual-audit accuracy.
  - Uncertainty: The four corrected examples were used to design v7; no held-out gold set exists.
  - Next check: Freeze v7 and evaluate on a new held-out, human-labeled set.
  - Decision: keep
