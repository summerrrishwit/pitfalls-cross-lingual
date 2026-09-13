# Corrected preflight assessment

## Decision

Do not run the 100-base-triple pilot yet. Run an approximately 20-base-triple paired-control expansion after the revised perturbation validator is in place. Keep Paths Not Taken and the `claude-opus-4-8` holdout disabled.

This decision is based on a Codex proxy review, not human gold. The review is versioned as `codex-proxy-review-20260909-v1` and carries `not_human_gold=true`.

## What changed

The legacy two-arm run generated Original separately for every perturbation candidate. This created repeated calls for an identical condition and produced one directly conflicting `hy4-preview` Chinese baseline for the Georgia-peach distractor. The corrected design uses:

- shared Original by `(source_id, distractor_id, model, language)`;
- shared Neutral Context by the same key;
- candidate-specific Targeted context;
- exact 24-outcome coverage per candidate: 4 models x 2 languages x 3 arms.

For the six Codex-accepted perturbations, sharing reduced the physical Simulation count from 144 to 112, saving 32 calls (22.2%).

## Codex proxy-review funnel

| Stage | Reviewed | Accept | Reject | Needs adjudication |
|---|---:|---:|---:|---:|
| Translation | 10 | 7 | 3 | 0 |
| Distractor | 14 | 7 | 3 | 4 |
| Perturbation | 22 | 6 | 8 | 8 |
| Provisional old outcomes | 3 | 0 | 1 | 2 |
| Permanent failed calls | 2 | 0 | 2 | 0 |

The main semantic losses were not random: one disputed legal source fact, one time-sensitive agricultural ranking, implausible or type-mismatched distractors, prompt duplication inside generated context, and contexts that actively warned against the target distractor.

## Corrected three-arm result

- 6/6 candidates received complete four-model coverage.
- 112/112 physical Simulation calls completed.
- 120 attempts were required, for 8 retries.
- Simulation used 21,292 tokens.
- Maximum Simulation latency was 197,070 ms.
- `hy4-preview` required 7 retries; `gemini-3.7-flash` required 1; the other two Simulation models required none.
- No duplicate baseline outcome was present.
- No candidate showed a Chinese accuracy drop under Targeted context relative to both Original and Neutral Context.
- One Neutral Context instability occurred: `bailian/deepseek-v4-flash-0731` changed from correct to incorrect in English for the tomato-juice distractor, while its Targeted answer was correct.
- Therefore, the three previously provisional induced effects are not reproduced under the corrected design.

## Interpretation

The run validates the engineering path, checkpoint reuse, model coverage, and the value of a neutral control. It does not validate the Paths Not Taken hypothesis. The effective independent evidence is only three base triples contributing six accepted perturbations, so zero observed effects is not a precise estimate of a population rate.

As a rough screening calculation, 20 independent base triples have an 87.8% chance of observing at least one event if the true base-level event rate is 10%, but only 64.2% if it is 5%. A 100-item run would raise the latter probability to 99.4%, yet scaling now would be inefficient because the current generator/validator yielded no reproducible target effect and still admitted control-sensitive phrasing.

## Next experimental gate

Before the approximately 20-base-triple run:

1. Require `target_distractor_salience=true` and `no_prompt_duplication=true` in automated perturbation validation.
2. Exclude disputed or time-sensitive source facts unless an authoritative dated reference is frozen.
3. Keep Original, Neutral, and Targeted arms; use shared baselines.
4. For any apparent effect, add adaptive confirmation calls for the implicated model/language rather than treating one deterministic-looking API response as stable.
5. Record token use now; either add a versioned provider price table or explicitly approve currency cost as a reported limitation rather than a blocking gate.

Authorize the 100-base-triple pilot only if the 20-base-triple screen has adequate semantic yield across multiple independent base triples and produces replicated Targeted-vs-Neutral effects. Candidate freeze and `claude-opus-4-8` remain downstream of that gate.
