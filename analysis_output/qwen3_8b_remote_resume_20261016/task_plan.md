# Qwen3-8B 160-fact remote continuation

## Goal
Continue the existing exact-checkpoint experiment on connect.cqa1.seetacloud.com:24946. All inference, activation collection, intervention, and execution checks run remotely; local work is limited to code, configuration, and artifact synchronization.

## Phases
- [x] Restore and verify frozen remote inputs; stage latest Development-only G2A and controls.
- [x] Implement/resume auditable option-free generation and exact-tokenizer audit.
- [x] Run remote smoke and Development three-arm generation/scoring.
- [x] Produce fact-level open/MCQ bridge, failure labels, and necessary analysis.
- [x] If stable open defects exist, develop bounded OOF mechanism/repair experiment with controls.
- [ ] Freeze repair rule before independent Validation; open Sealed only after confirmation. Not reached: all candidates fail G2.

## Decisions
- Exact checkpoint Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218, BF16, existing remote environment.
- Free generation correctness and designated-distractor hits are primary; candidate scoring is auxiliary and separate.
- Existing Original outputs may be reused only after exact prompt/tokenizer/decoding match.
- Keep all 96 Development facts and both variants; fact is the count unit.
- Never use Validation/Sealed to select prompts, vectors, layers, or scales.

## Errors encountered
Errors and recoveries are recorded below and in notes.md. Sandbox SSH DNS required network escalation; remote checks succeeded.

## Status
Remote Development and bounded repair runs complete. Baseline 960/960, positive repetition 82/82, 8 repair candidates / 4,256 generation calls. All candidates fail G2 preservation. No final repair rule or holdout execution; see final status and repair_outcome.md.

Remote audit error: tokenizer returned Mapping rather than dict; corrected dispatch and added a regression test before inference.

## Authorized continuation to independent confirmation (2026-10-16)
User explicitly requests continuation through frozen independent confirmation, remote execution only.
- [x] Publish bounded v3 search/selection/holdout contract before new activation or intervention outputs.
- [x] Collect open-answer layer trajectories and all-fold source vectors; verify no-op injection equivalence.
- [x] Run predeclared 12-setting OOF sequence screen; generate task/difference/combined candidates on full Development.
- [x] Review decisive ambiguous generations, evaluate retention, select or fail G2. All candidates fail; random controls are conditional on eligibility and were not started.
- [ ] Freeze rule and full-Development final vectors, then run complete Validation 32 without adjustment. Not reached: G2 failed.
- [ ] If G3 passes, run complete Sealed 32 once; otherwise preserve Sealed and report failed/insufficient confirmation. Not reached: G2 failed; both holdouts preserved.

Initial implementation decision: bounded v3 replaces the unexecuted full 35-layer/7-scale search prospectively; existing behavior and folds remain frozen. The following entries record execution history.

Remote collection completed: 960 open trajectories (36 resid_pre layers), 1440 OOF vector-source prompts, 4 source partitions of 72 facts. Final full-Development source remains unconsumed until rule selection. No-op audit passed four hash-selected ZH inputs; score delta below 5e-9. Fixed 12-setting score screen and three-candidate generation pipeline started.

V3 outcome: all three candidates fail a known preservation condition before independent confirmation. Task/combined harm unrelated controls; difference l24/-1 changes a resistant Margot Robbie case to other actors (known fact harm 1/14 exceeds 5%). Unnecessary semantic work on hard-failed candidates was pruned without imputing missing judgments. V4 is a separately published, finite Development-only refinement: task l12 +0.25/+0.5, difference l24 -0.25/-0.5, and the +0.25/-0.25 combination; selection uses actual generation without a sequence-score shortlist. The OOF source bank and all thresholds are unchanged; holdouts remain unused. V4 generation is running remotely.

Intermittent SSH transport failure during v4 semantic export: connection closed by remote endpoint before output. The inference job runs under tmux; retry read-only status first and do not restart completed calls. Last confirmed progress: task +0.25 completed 532, task +0.5 in progress. V4 semantic decision file may have been seeded from v3; check before any export rewrite.

## Final status

V3 and v4 completed, both exit 0: 8 candidates, 4,256 generation calls. All violate retention; v4 known resistant fact harms are 1/2/2/1/2 of 14 respectively. Explicit recheck resolved the weakest task candidate's primary-data definition as incorrect; original deferral and revision provenance retained. Incomplete date remains missing. No random inference, final source collection, rule freeze, Validation or Sealed execution occurred. Independent saved-artifact audit passed; 40 synchronized bound artifacts verified. Report: repair_outcome.md. The requested independent confirmation endpoint was not reached because the prospectively fixed G2 futility condition applies.

Synchronization recovery: extraction was attempted before the SSH transfer session finished, yielding a truncated-archive error. Waited for transfer exit 0, re-extracted the complete archive and verified bound SHA values; remote data unaffected.

## Authorized v5 continuation

User requests the proposed repair follow-up. Preserve the completed v3/v4 negative closure and all holdouts. Remote computation only.
- [x] Publish finite v5 contract and implement a prompt-prefill-only hook without changing archived common code.
- [x] Test hook equivalence remotely; run same-fact Neutral→Targeted and reverse local recovery diagnostics with self-patch controls.
- [x] If local recovery exists, estimate paired, fold-isolated directions and test norm-calibrated prefill-only repair on full Development plus controls.
- [x] Review decisive outputs, run timing/negative/random controls as specified; assess source-size learning curves if there is a repair signal.
- [x] Advance to final vector/rule freeze and independent confirmation only if all unchanged G2 gates pass; otherwise preserve holdouts and report the new evidence. G2 passed; fixed Validation completed and failed G3, so Sealed preserved.

Status: remote v5 local recovery diagnostic started in tmux qwen3-prefill-v5. Hook tests passed 2/2 remotely (prefill/teacher-forcing coordinate, cached-token exclusion, replacement and nonmutation). The previous final status applies to v3/v4 only; it is not being overwritten.

V5 initialization error: older seeded semantic decisions have no corresponding review export row. Corrected reuse to skip such entries rather than infer their text. Initialization stopped before self-patch or new inference and before creating v5 identity; safe to redeploy and restart. All old artifacts untouched.

V5 absolute self-patch failed score equivalence (generations matched). Explicit actual-model audit confirms cached/returned layer states exactly match true resid_pre; teacher-forced vs prefill states differ in the BF16 forward path (relative L2 .019–.026). Preserve failed v5 preflight. New v5b uses float32 source−target additive deltas for both generation and scoring; self delta is exactly zero. Three remote hook/timing tests passed; diagnostic restarted under v5b, no oracle trial had run in v5 and no holdout exposure occurred.

V5b local recovery completed 4×2×44=352 calls plus four self controls. All 56 unique ambiguous outputs explicitly proxy-reviewed, no unresolved diagnostic outputs. Layer 28 selected prospectively by fact-balanced generated recovery: 11/25 positive variants restored, fact-balanced recovery .470588 across 17 facts; 9/25 reverse positive variants become wrong; zero known harm among the 19 resistant variants in either direction. Layer 32 restored 10, layer 20 restored 2, layer 12 restored 0. This is same-fact diagnostic evidence only. Paired/centered prefill four-rule full Development generation started; source estimator unit checks passed 2/2 remotely. Do not recompute or change diagnostic summary after followup identity binds it.

Prefill independent execution adapter prepared and remote guard checks passed 2/2. Automatic SSH approval review timed out once during deployment (no command execution); permitted single retry completed successfully. First two paired candidates finished; rho .03 is generation-eligible provisionally, pending all four candidates and random comparison. Centered rho .01 already has known resistant harm. Review only decisive additional ambiguities; secondary canonical-equivalence/partial-date uncertainties stay missing without changing old semantics.

All four candidates finished 4×532, exit0. Paired rho .03 is the only generation-eligible rule: directed gain .0294118 (one of two variants of Captain America actor fact corrected from Yannick Bisson to Chris Evans), all-targeted gain .00537634, Original gain0, Neutral gain .00543478, directed margin +.691639, zero known resistant/control harms or unresolved primary pairs. Paired rho .01 and centered rho .03 lack generated efficacy; centered rho .01 harms a resistant fact. Two timing/negative trials completed 113 inputs each. Their conditional launcher checked the gate just before three remaining reviews were saved, so exited without random execution; a new explicit random→learning launcher was started after the complete gate was eligible. No duplicate generation. Random5 full532 and source-size18/36×seed0/1 on113-panel running; do not run the full --random selection analyzer while the launcher still needs generation_selection.status for the learning phase. Use resolve-only exports to curate partial random outputs.

Random5 (2,660 calls) and source-size4 (452 calls) completed, exit0. All five random directed generation gains0; observed .0294118 passes. All111 unique exported outputs explicitly proxy-reviewed; remaining semantic deferrals never imputed, no G2 primary missing. Source-size descriptive auxiliary summary complete, no pending reviews. V5b selected_rule status oof_selected_pending_final_vector. Final source/donor audit passes:480 ICL prompts,96 source facts, all donors in Development with relation/component restrictions. New7 remote tests pass. Final96 source collection→freeze→Validation pipeline started under tmux qwen3-prefill-validation; Sealed remains unopened. No further changes to inference/scoring scripts after freeze. A transient SSH connection closed; atomic retry succeeded, tmux did not repeat calls.

Final480 ICL source prompts collected and final96 paired vectors constructed; prefill rule is frozen. Validation engine has started with exact bound code; no code/scorer/cache changes from this point. Development top-level results and labels synced locally; 14 frozen code files and5 frozen artifact bindings match local SHA. Learning curves n18/n36 (two seeds) and n72 all yield the same directed generation gain .0294118; no observed monotonic benefit from extra sources in this bounded diagnostic. Await full Validation execution and blinded semantic review before G3 decision; do not invoke Dev analyzers that rewrite frozen artifacts.

User explicitly says continue unfinished task; continued same pipeline. Validation320 MCQs +320 no-op +173 repaired outputs complete; random-r0 complete173, r1 last114/173. Exported stable validation indices0..167 explicitly reviewed/saved, except five deliberately missing semantic decisions at53 (conflicting CN/EN Ray Bolger name),133 (only Marvin Gaye of duo),142 (generic tu quoque wording),161(seed control),162(treaty control). No unseen output imputed false. Foil identities explicitly assigned at4/7/11/60/63/75/95/102/114/122/134/143 to preserve induced-case definition. Remaining random groups still executing, so do not write a final G3 confirmation or open Sealed yet. Next: call analyze_qwen3_independent.resolve ONLY to export any new validation rows, then after execution_complete.json run the frozen analyze_qwen3_independent.py --config repair_config_v5b.json --split validation. If G3 fails/insufficient, preserve Sealed and finalize report; if passes, run same frozen prefill independent runner --phase sealed once and review it.

## V5b final closure (2026-10-17)

All phases complete through fixed independent Validation; status G3_validation_failed_or_insufficient. All32 facts executed;10 directed facts/18variants,0 corrected variants,0 new Targeted errors. Actual generation gain0 and random mean0; all-targeted margin+.782979, directed margin+.930901. Original/Neutral gains0; resistant panel2facts no harm, unrelated controls no new harm. Layer28 descriptive contrast gate failed. No too-few-positive gate failure.174 unique proxy judgments,5 missing and0 unreviewed;1 ambiguous Original fact excluded from main31-fact pairs, all32 raw records retained. No parameter/code/scorer changed after freeze. Sealed0/32 under futility, no further rule tuning within this completed protocol.

Independent integrity audit passed:14 frozen code files,320no-op,173repair and5x173random,320MCQs, text/decision SHA and Sealed unopened. Local20 bound Validation files+3final bindings verified. Dev5,818 main generation records; Validation1,358. Inference exited0; GPU no active process. Final report repair_v5_outcome.md; final_decision.json under remote_artifacts/repair-v5b/. Temporary validation archive included state tensors in transfer; extraction excluded .pt and archive removed, no local tensors retained. No remote data altered. Final multi-file documentation patch failed due an exact-text anchor comma mismatch; reapplied against the current file, no experiment data affected.
