# Twenty-base-triple paired-control assessment

## Decision

Do not start the 100-base-triple pilot or the `claude-opus-4-8` holdout. Keep Paths Not Taken disabled and revise the sensitivity of the behavioral measurement first.

All semantic decisions in this screen are a versioned Codex proxy review with `not_human_gold=true`; unresolved items were excluded rather than treated as accepted gold.

## Independent-unit funnel

| Stage | Count |
|---|---:|
| Newly sampled base triples | 20 |
| Model translation reviews accepted | 15 |
| Codex proxy translation reviews accepted | 15 |
| Model distractor reviews accepted | 27/30 |
| Codex proxy distractor reviews accepted | 20/30 |
| Generated perturbations | 54 |
| Model perturbation reviews accepted | 13/54 |
| Codex proxy perturbation reviews accepted | 36/54 |
| Deterministically selected candidates | 12 from 12 distinct base triples |
| Complete three-arm candidates | 11/12 |
| Confirmed targeted Chinese degradations | 0 |

The large automated-versus-Codex perturbation disagreement is diagnostic. The automated judge accepted 13, the Codex proxy accepted 36, and only 11 overlapped. Many automated rejections incorrectly treated increased distractor salience as a defect even though it is the intended manipulation. The validator prompt has therefore been clarified for future runs.

## Simulation execution

- Physical calls: 288; completed: 287.
- Simulation attempts: 339; retries: 51.
- Simulation tokens: 57,047.
- Three models completed 72/72 each. `hy4-preview` completed 71/72, used 50 retries, and had a maximum recorded latency of 295,216 ms.
- One `hy4-preview` English Original call for `raw_sciq_000228_source_wrong_0_p1` remained failed after 8 cumulative attempts, leaving 11/12 candidates complete.
- No duplicate shared-baseline inconsistency was found.

## Behavioral result

No complete candidate had lower Chinese accuracy under Targeted context than under both Original and Neutral Context. Thus the corrected 20-base-triple screen produced zero Paths Not Taken candidates.

One control-sensitive item was observed: for `raw_commonsense_qa_001993_source_wrong_1_p2`, `qwen3.7-plus` changed between Original and Neutral in both English and Chinese. This is not a targeted perturbation effect and demonstrates that a single nominally deterministic call is insufficient for borderline candidates.

Across the earlier corrected preflight and this disjoint screen, 17 candidate runs from 14 distinct contributing base triples completed without a confirmed targeted Chinese degradation. This is evidence against the current setup's practical yield, not proof that the phenomenon is absent.

## Why 100 is not the next experiment

The dominant limitation is measurement sensitivity, not merely sample count:

1. Most Original, Neutral, and Targeted cells are at 4/4 accuracy, creating a binary-choice ceiling effect.
2. The old apparent effects disappeared after baseline sharing and neutral controls.
3. Control wording alone changed answers for one model/item.
4. `hy4-preview` creates substantial latency and retry cost.
5. Currency cost cannot be audited because no versioned provider price table is configured.

A larger run under the same measurement would estimate a near-ceiling accuracy more precisely, but would not necessarily expose a causal distractor effect.

## Required redesign before scale-up

1. Use original multi-option questions where possible, or record token log-probability/rank shifts when the provider supports them; binary correctness should remain a secondary endpoint.
2. Generate stronger but truthful target cues, with no corrective language and no embedded copy of the question.
3. Replace the single generic Neutral sentence with a length- and topic-matched neutral context that contains neither answer nor distractor cues.
4. Add adaptive replication only when Original, Neutral, and Targeted disagree; require the direction to reproduce before candidate admission.
5. Re-probe the clarified automated validator prompt against the Codex-reviewed disagreements.
6. Add a versioned price table, or explicitly classify monetary cost as a non-blocking operational limitation.

After these changes, run another bounded screen. A 100-base-triple pilot is justified only if the revised screen yields replicated effects across multiple independent base triples. Candidate freezing and Claude holdout remain downstream of that decision.
