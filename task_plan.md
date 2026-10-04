# Task Plan: Chinese-first Factual Perturbation Before Paths Not Taken

## Goal
Build an auditable independent review/freeze → Development MCQ weakness mining → natural EN/ZH
open-completion baseline and claim stratification → exact-HF attribution/intervention → independent
confirmation loop. Use Qwen public-benchmark extractions only as provenance-preserving single-model
provisional candidates; keep unreviewed data out of formal behavior/PATH cohorts, and stop at
offline preparation until a GPU-accessible exact HF checkpoint and tokenizer are frozen for the
same-model behavior, mechanism, and repair experiment.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Profile inventory and existing pilot artifacts
- [x] Phase 3: Build candidate taxonomy and complete signature mapping
- [x] Phase 4: Apply mapping and validate normalized output
- [x] Phase 5: Review risks and deliver report
- [x] Phase 6: Decouple factual-prompt generation from relation normalization
- [x] Phase 7: Configure and validate the Chinese translation/perturbation/simulation stack
- [x] Phase 8a: Run the bounded 100-input Chinese diagnostic and preserve its failed formal gate
- [x] Phase 8b: Close the five-model pilot as diagnostic proxy evidence without candidate freeze or holdout
- [x] Phase 9a: Build the producer-v2 public-benchmark provisional pool, review tooling, and offline Paths Not Taken review-only bridge
- [x] Phase 9b: Complete the cohort-specific Codex-proxy review, bounded full-pool lexical closure, provisional split recomputation, and post-closure offline artifacts
- [ ] Phase 9c: Complete translation/distractor review, historical-exposure and broader semantic closure, then explicitly freeze the offline cohort and split
- [ ] Phase 9d: Run Development exact-HF MCQ mining, natural baseline, attribution, intervention, and repair only after data and runtime are frozen
- [ ] Phase 9e: Open Validation for fixed confirmation and Sealed once for final evaluation only after the preceding rules are frozen and confirmed

## Key Questions
1. Which relation expressions can be merged without losing type or direction semantics?
2. How many records can be mapped deterministically, and which signatures remain ambiguous?
3. Does normalization preserve every source and extraction field and reproduce deterministically?
4. Can every accepted canonical triple generate a factual prompt without requiring a mapped `relation_normalized`?
5. Which narrow, direction-preserving `probe_relation_id` values have enough high-quality records for Paths Not Taken mechanism experiments?

## Decisions Made
- Preserve `relation_raw`; only add versioned normalized fields.
- Produce candidate v1 artifacts first. They are not final frozen gold until review gates pass.
- Give every one of the 2,425 signatures an explicit `mapped`, `ambiguous`, or `out_of_taxonomy` decision.
- Separate description-to-term relations from term-to-definition relations; their directions are
  opposite. Treat the raw label `term for definition` as direction-unresolved instead of
  automatically assigning `description -> term`.
- Use conservative deterministic rules for candidate v1. Broad or overloaded labels remain unresolved.
- Candidate v1 maps 1,083 of 2,744 extracted records; unresolved records remain explicit review items.
- Retain the current `relation_normalized` taxonomy as an optional statistical layer; it must not gate factual-prompt generation.
- Generate a versioned factual prompt for every accepted canonical triple when `canonical_fact` and `answer` pass prompt validation, using strict answer-suffix completion first and an option-free open-answer fallback otherwise.
- Add a separate nullable and versioned `probe_relation_id` for narrow Paths Not Taken relation families. Do not overload the broad statistical taxonomy for same-relation negatives, relation-specific task vectors, or across-relation evaluation.
- Keep unresolved or long-tail probe relations in the general factual-recall corpus while excluding them from relation-conditioned mechanism experiments until explicitly reviewed.
- Treat prompt readiness and Paths Not Taken eligibility as separate decisions. The mechanism subset additionally requires relation direction, answer uniqueness, bilingual equivalence, prompt-form consistency, and model-specific answer-token metadata.
- Do not require an exact join with the existing 16-language files for the first Paths Not Taken run. Build a new Chinese paired-prompt artifact directly from accepted canonical triples and retain source provenance.
- Treat one canonical base triple as the independent unit. Chinese translations, aliases, prompt variants, distractors, and perturbation rounds must remain grouped with that base triple in one split and must not inflate the reported sample size.
- Use a model with locally accessible hidden states, tokenizer logits, and intervention hooks. Prior API-model screening results are not substitutes for the selected mechanism model's English/Chinese baseline.
- Defer hidden-state, Logit Lens, task-vector, and activation-intervention work until the Chinese perturbation pipeline has produced a frozen, independently evaluated badcase set.
- The current 160-fact role contract is authoritative: MCQ Original/Neutral/Targeted behavior mines
  model-specific weaknesses; natural EN/ZH open completion is the unperturbed baseline and claim
  stratifier; PATH_not_token performs mechanism analysis and repair on the same frozen exact-HF
  identity. Validation and Sealed neither mine defects nor tune the repair.
- Keep four authorities separate before and during execution: deterministic source/rule validation
  establishes lineage only; evidence-backed fact verification establishes canonical support and
  distractor falsity; an independently identified strong semantic reviewer checks unique answer,
  relation, translation, Neutral, and Targeted contracts on final rendered inputs; the target
  exact-HF measures behavior only. Target outputs cannot retroactively approve or rewrite inputs.
- The following API-model assignments describe the completed historical diagnostic pipeline only:
  `qwen3.8-max` handled primary Chinese translation and distractor/perturbation validation,
  `gpt-5.5` reviewed translations, and `gpt-5.6-sol` generated controlled perturbations. These
  proxy roles do not establish formal fact truth, freeze status, or PATH eligibility.
- The historical perturbation generator was kept out of self-validation and Simulation scoring.
  Its four-family Simulation ensemble used `qwen3.7-plus`,
  `bailian/deepseek-v4-flash-0731`, `hy4-preview`, and `gemini-3.7-flash`; the proposed
  `claude-opus-4-8` holdout was not run. That panel is not the target exact-HF mechanism chain and
  all its signals remain `pnt_eligible=false`.
- Do not enable same-relation distractor sampling in the first pilot. Use verified source wrong options first and verified same-answer-type negatives second; add `probe_relation_id` before relation-conditioned negatives.
- Treat Codex review as a versioned proxy audit (`reviewer_type=codex_proxy`, `not_human_gold=true`), not as human gold. Exclude unresolved facts and candidates rather than silently accepting them.
- Retain three MCQ behavior arms—shared `original`, matched `neutral`, and candidate-specific
  `targeted`—for formal exact-HF Development mining. Reuse each baseline by
  `(source_id, distractor_id, model, language, variant)` and reject evidence with inconsistent
  duplicate baselines.
- Do not expand directly to 100 after the 10-item engineering preflight. First run approximately 20 new paired-control base triples because the corrected three-arm rerun reproduced zero of the prior provisional effects.
- The completed historical sensitivity screen used `qwen3.6-27b`,
  `bailian/deepseek-v3.2`, `gpt-5-mini`, and `gemini-3.7-flash`, with the generator kept out of
  Simulation. This remains a diagnostic-proxy design rather than a prescription for the 160-fact
  exact-HF run.
- Generate ordered L1/L2 truthful perturbations with candidate-specific, length/topic/style-matched Neutral contexts. Keep only Original shared in this design.
- The multi-option calibration completed 480/480 calls across 20 Codex-reviewed independent base
  triples and produced zero target-distractor-specific Chinese flips. At that historical point the
  next authorized step was the 100-input diagnostic later completed as v7b, not a production
  candidate freeze.
- The earlier v4 100-input diagnostic pilot completed 840/840 Simulation outcomes on 35 retained sources and produced one strict offline candidate. A later streamlined v5 calibration on 20 new sources reduced hard filtering through translation repair, distractor replenishment, advisory perturbation judging, and Simulation-blind Codex selection.
- Streamlined v5 retained 15/20 sources, completed 360/360 Simulation outcomes, had zero Neutral instability, and produced two strict Chinese-specific candidates. This is a calibration pass for proceeding to a new v5 100-input diagnostic pilot, not evidence sufficient for candidate freeze or holdout.
- The later `diagnostic-100-five-model-v7b` run completed the 100-base diagnostic: 93 translations and 73 perturbation candidates were retained, all 2,190 expected Simulation calls completed, and 14 model-specific strict signals were reported. Its formal gate remains false because translation coverage was incomplete and currency cost was not auditable; 41 Neutral-control instabilities also prevent treating it as a frozen production or mechanism cohort. It did not run the current 160-fact Validation or Sealed partitions, and every signal remains `pnt_eligible=false`.
- The audited `public-benchmarks-full-v1` Qwen extraction is a separate single-model provisional pool: 20,834/20,834 terminal records completed and 9,059 were extracted. It must be deduplicated, independently adjudicated, and versioned before any row becomes canonical or formally PATH_not_token eligible.
- Producer v2 groups the 9,059 validation-clean triples into 8,969 provisional base facts. Its
  400-item coverage-oriented sample is for heterogeneous failure discovery and does not support
  population inference. A separate 400-item overall-random sample supports whole-pool estimates
  only under its precommitted-seed, uniform-protocol, and nonresponse conditions. The 200-item
  prompt-risk sample targets a 376-fact risk frame and does not support whole-pool inference.
- Review `scope -> export -> apply` is a SHA-bound staging workflow. Exports include every
  duplicate-cluster member; decisions are `accept`, request-only `revise`, `reject`, or `defer`;
  partial-review missing items remain missing. Codex review is `reviewer_type=codex_proxy` and
  `human_gold=false`, and the tool has no automatic freeze operation.
- The current overall-random pilot is only staging: 20 decisions yielded 16 `accept`, 2 `defer`,
  1 `reject`, and 1 `revise`; 380 scoped items remain missing.
- A review sample and a frozen cohort are distinct artifacts. A formal freeze universe must be
  separately materialized with an immutable row inventory and exact hashes. It may be a
  predeclared subset, but every scoped item must reach final `accept` or `reject` after revisions;
  `missing`, `defer`, and unapplied `revise` block freeze.
- The overall-random 400 cannot automatically stand for a frozen 8,969-fact cohort. If selected as
  the formal cohort universe, it requires 400/400 review completion plus duplicate,
  near-duplicate, and historical-exposure closure against the full pool. Freezing the entire pool
  instead requires item-level terminal review of all 8,969 facts.
- Lexical near-duplicate audit v2 emitted 2,341 representative candidate pairs, including 281
  emitted cross-split pairs. It is not an exhaustive pair census, performs no semantic
  adjudication, and therefore blocks split freeze until reviewed.
- Formal behavior and PATH admission now require external review-freeze and split-freeze
  manifests bound to the exact bundle bytes; row-level status changes are insufficient. Those
  manifests have not been produced for this pool.
- Formal evidence validation now uses canonical evidence-tier values, requires row-level
  `human_gold`/`evidence_tier` to match the bound review decision, binds near-duplicate decisions
  to candidate and endpoint hashes, and rejects legacy unbound v1 evidence. PATH no longer coerces
  non-boolean `human_gold` values. Holdout execution binds both its runtime and frozen-candidate
  artifact before initializing a model router.
- `scripts/finalize_public_benchmark_bundle.py` can immutably declare a chosen full-pool or
  review-scope diagnostic universe and emit a fail-closed blocker report. Universe v2 accepts only
  the path-independent pinned `canonical-v2` comparison content contract and predeclares the full
  lexical near-audit policy; a later audit with different valid thresholds is rejected. It also
  records `historical_exposure_contract_status=not_predeclared`, so a runtime registry cannot
  authorize closure. The user has not yet supplied an authoritative historical-exposure inventory
  contract, and no actual finalizable formal universe has been selected or declared. The tool
  cannot apply revisions, materialize a reviewed bundle, or emit freeze manifests.
- A temporary diagnostic preflight over the existing 400-item scope (explicitly not retained as a
  formal declaration) reproduced 16 accept, 1 reject, 2 defer, 1 revise, and 380 missing. Of the
  2,341 lexical candidates, 205 touch this scope, 28 of those are cross-split, and 181 distinct
  out-of-scope base facts are linked. The review apply chain and lexical audit were independently
  replayed from their bound inputs. The v2 report exited 2 with 12 blocker classes, including the
  deliberately permanent historical-authority blocker, and emitted no formal or freeze artifact.
- GPU unavailability blocks exact-HF behavior, hidden-state, Logit Lens, activation-patching, task/difference-vector, and repair claims. It does not block provisional import, fact clustering, relation inventory, bilingual review, split freezing, or an HF-ready input bundle.

## Follow-up Tasks
- [x] 6.1 Update the prompt contract so `normalization_status` and `relation_normalized` no longer determine `factual_prompt_status`.
- [x] 6.2 Version the new prompt artifact and add tests proving that `mapped`, `ambiguous`, and `out_of_taxonomy` canonical triples can all receive valid prompts.
- [x] 6.3 Update downstream validation and distractor construction to consume prompt readiness explicitly instead of inferring it from relation mapping.
- [x] 6.4 Rebuild the full canonical prompt artifact and report strict-completion, open-answer-fallback, and failure counts without assuming all 2,450 records pass before validation.
- [x] 7.1 Validate each selected model ID with one minimal request and one representative structured workflow request before any batch run; all eight selected models passed on 2026-09-09, while batch stability remains unverified.
- [x] 7.2 Implement provider profiles and model-role routing from `configs/factual_perturbation_zh_mvp_v1.json` without storing credentials in artifacts.
- [x] 7.3 Build English/Chinese paired factual prompts from canonical triples, preserving aliases, translation provenance, and review outcomes.
- [x] 7.4 Verify source wrong options and same-answer-type compatibility before allowing them to serve as distractors; keep same-relation sampling disabled in the MVP.
- [x] 7.5 Generate controlled bilingual perturbations and require independent checks for ground-truth preservation, no explicit falsehood, relation preservation, bilingual equivalence, and naturalness.
- [x] 7.6 Record per-model English/Chinese original, neutral, and targeted outputs, correctness, distractor hits, failures, retries, model versions, and raw responses; share baseline calls across candidates.
- [x] 7.7 Complete a versioned Codex proxy review of all 10 translations, 14 distractors, 22 perturbations, 3 provisional outcomes, and 2 permanent failures without representing it as human gold.
- [x] 7.8 Rerun the 6 proxy-accepted perturbations under the corrected three-arm design: 112/112 Simulation calls completed, all 6 candidates had full 24-outcome coverage, and no targeted Chinese degradation survived.
- [ ] 7.9 Add a versioned provider price table or explicitly downgrade currency cost from a blocking gate to a reported limitation before expansion.
- [x] 7.10 Freeze 20 new, non-overlapping base triples; Codex proxy review retained 36 perturbations across 12 independent bases, one per base was deterministically selected, and the four-model three-arm run completed 287/288 calls with zero confirmed target effects.
- [ ] 7.11 Redesign the behavioral endpoint to reduce binary-choice ceiling effects. Matched Neutral contexts and the clarified validator have been exercised; adaptive replication was not triggered because the valid L2 screen had no arm disagreement.
- [x] 7.12 Probe and run the medium-capacity Simulation panel on the 12 previously frozen candidates: 288/288 calls completed; one nominal Qwen flip was excluded after detecting a bilingual distractor typo.
- [x] 7.13 Generate and Codex-review matched-neutral L1/L2 perturbations; retain 22/24 candidates across 11 valid independent source/distractor pairs.
- [x] 7.14 Run the 11 stronger L2 candidates with the revised four-model panel: 264/264 calls completed and zero qualified Chinese targeted flips were observed.
- [x] 7.15 Replace binary correctness with a deterministic three-option endpoint, identical English/Chinese option ordering, target-distractor hit tracking, and explicit separation of non-target wrong answers. The 20-base calibration completed 480/480 calls with zero target-specific flips.
- [x] 7.16 Run the 100-input-base multi-option diagnostic as `diagnostic-100-five-model-v7b`: 100 bases, 73 fully simulated candidates, 2,190/2,190 Simulation calls, and 14 model-specific strict signals. Preserve `gate_passed=false`; do not freeze candidates or run holdout from this diagnostic alone.
- [x] 7.17 Run the streamlined v5 calibration on 20 new sources: translation 18/20, complete option sets 17/20, Codex-retained sources 15/20, Simulation 360/360, Neutral instability 0/120, and strict Chinese-specific candidates 2/15.
- [x] 7.18 Add loopback-only, credential-free Ollama Simulation routing and rerun the latest 15 frozen v5c candidates with `gemma3:12b` and `llama3.1:8b`: 180/180 calls completed, with 2 model-specific strict signals from `llama3.1:8b` and 0 from `gemma3:12b`.
- [x] 8.1 Supersede the planned production 100-base API-panel pilot with the completed v7b diagnostic; do not reinterpret the diagnostic as a frozen cohort.
- [x] 8.2 Retain complete per-model coverage as a historical diagnostic reporting rule, but retire the multi-model panel as the primary score for the 160-fact exact-HF chain.
- [x] 8.3 Cancel the proposed Claude Opus 4.8 holdout for this diagnostic path; no candidate from the failed gate is frozen or made PATH-eligible.
- [x] 8.4 Preserve independent base-triple counts, attrition, per-model failures, English retention,
  Chinese change, distractor hits, and transfer as diagnostic reports without treating variants as
  independent samples or claiming mechanism evidence.
- [x] 9.1 Define the nullable `probe_relation_id` and offline admission schema now, while keeping model-specific tokenization and mechanism eligibility pending until an exact HF checkpoint/tokenizer revision is selected.
- [x] 9.2 Build and Codex-proxy review a relation-balanced 160-fact provisional subset without GPU; keep exact-HF baseline, hidden-state, task/difference-vector, intervention, and causal-repair execution deferred.
- [x] 9.3 Import `public-benchmarks-full-v1` through a dedicated single-model provisional adapter; never duplicate the Qwen checkpoint to simulate dual-model consensus.
- [x] 9.4 Assign deterministic `base_fact_id`/duplicate clusters, preserve extraction provenance, and separate strict completions from open-answer fallbacks.
- [x] 9.5 Emit a versioned behavior input bundle with explicit evidence tier, canonical status, aliases, distractor provenance, relation fields, record hashes, and a provisional answer-group split with a reserved sealed partition. Keep `split_status=provisional_not_frozen` until review is complete.
- [x] 9.6 Make perturbation preparation consume an explicit bundle path while preserving the legacy canonical-v2 default and existing frozen runs.
- [x] 9.7 Add a generic PATH bridge adapter for reviewed bundles; retain the legacy `Chinese.json` preparation unchanged.
- [x] 9.8 Upgrade the public producer to v2, emit the coverage-oriented 400, overall-random 400,
  and prompt-risk-targeted 200 samples, and record the 376-fact prompt-risk frame and inference
  limits explicitly.
- [x] 9.9 Add SHA-bound review `scope`, `export`, and staging-only `apply`; expand duplicate
  members, support `accept`/`revise`/`reject`/`defer`, preserve missing partial decisions, and
  require explicit non-human Codex-proxy provenance.
- [x] 9.10 Run the 20-item overall-random Codex-proxy pilot and retain its 16/2/1/1 outcomes plus
  380 missing entries as staging, without automatic canonical or split freeze.
- [x] 9.11 Add and run lexical near-duplicate audit v2 as bounded candidate generation; record
  2,341 emitted representative pairs and 281 emitted cross-split pairs without claiming an
  exhaustive census or semantic completion.
- [x] 9.12 Require exact bundle-bound external review-freeze and split-freeze manifests in formal
  factual-perturbation and PATH admission.
- [x] 9.13 Regenerate the PATH review-only bridge with 8,969 provisional facts and verify that the
  formal output remains empty.
- [ ] 9.14 Complete the remaining overall-random review under one protocol; resolve or explicitly
  exclude unresolved pilot `revise`/`defer` cases, without treating nonresponse as reject or human
  gold. This completes an inferential sample only; it does not itself define a formal cohort.
- [x] 9.14a Consolidate the separate relation-balanced cohort review chain: 251 terminal proxy
  outcomes produced 157 direct accepts, 13 direct rejects, 78 explicit cohort exclusions, and
  3 retained revisions; independently re-review all retained revisions without claiming human gold.
- [ ] 9.15 Semantically adjudicate lexical/alias candidates, regroup accepted duplicate or leakage
  components, and recompute the split. For a subset cohort, compute closure against all 8,969 pool
  facts and all historically exposed evaluation artifacts. Do not freeze while any relevant
  pair/component remains unresolved.
- [x] 9.15a Complete the bounded lexical closure against the full 8,969-fact pool for the selected
  160-fact cohort: adjudicate all 73 reachable candidate edges, preserve original split-group
  atomicity, retain all 160 facts, and recompute an exact 24/8/8 split for each of four relations.
  Keep broader semantic-paraphrase closure and historical-exposure closure explicitly pending.
- [ ] 9.16 Implement a provenance-preserving finalizer that separately materializes the exact
  formal cohort universe, applies and re-reviews revisions, re-clusters facts, records every
  inclusion/exclusion, and emits review-freeze and split-freeze manifests bound to the resulting
  bundle. Then regenerate formal behavior/PATH inputs.
- [x] 9.16a Implement immutable full-pool/review-scope diagnostic universe declaration and a
  fail-closed evidence preflight that always blocks while the reviewed-bundle/freeze producer is
  absent and while historical-exposure authority was not predeclared.
- [x] 9.16b Materialize the non-frozen post-closure 160-fact bundle, regenerate 320 same-relation
  same-final-split distractor candidates, and regenerate the PATH review-only bridge with 160 rows
  and zero formal rows. Preserve every pending blocker and keep perturbation authorization false.
- [ ] 9.17 Freeze the exact HF checkpoint/tokenizer, behavior/scoring contract, primary MCQ variants,
  and Development folds; keep all 160 rows `pnt_eligible=false` until the formal review/split gates
  and these manifests pass.
- [ ] 9.18 Run only Development 96 through exact-HF EN/ZH Original/Neutral/Targeted MCQ; derive
  strict induced defects and resistant controls by the frozen deterministic label rule.
- [ ] 9.19 Run natural EN/ZH open completion as the baseline and claim stratifier, then develop
  hidden-state/Logit-Lens attribution, task/difference vectors, and repair on Development with
  out-of-fold evaluation. Freeze the resulting mechanism and repair rule.
- [ ] 9.20 Open Validation 32 only for fixed confirmation. Do not replace distractors, tune labels,
  recompute vectors, or change layer/scale from its outputs.
- [ ] 9.21 Open Sealed 32 once only after Validation confirms the frozen protocol; report repair,
  English retention, Original/Neutral preservation, and unrelated-fact regression.

## Immediate Execution Order
1. Use only the SHA-bound authoritative chain under `preperturbation_v1`:
   `duplicate_closure_adjudications_v1` → `duplicate_closure_resolution_v1` →
   `postclosure_preperturbation_v2`, followed by the PATH output
   `public-benchmark-qwen-postclosure-provisional-v2`. Earlier v1/replay directories are
   non-authoritative and must not be mixed into this chain.
2. Generate and independently review EN→ZH translations for the 160 retained facts, including
   prompt/answer aliases, while preserving the English facts as the source of record.
3. Independently review all 320 regenerated distractors for answer disjointness, relation fit,
   single-answer behavior, and target-language rendering; failed items require replacement and a
   fresh bound review.
4. Obtain the user's authoritative historical-exposure inventory contract, bind it to the exact
   160-fact cohort, and add a declared broader semantic-paraphrase audit. The completed 73-edge
   lexical closure is bounded candidate coverage, not an exhaustive semantic census.
5. Decide whether Codex-proxy evidence is acceptable for the intended experiment or replace it
   with named independent human review. Only after that decision and all preceding checks may a
   producer emit exact-bundle review-freeze and split-freeze manifests.
6. Re-run formal admission: only then may factual-perturbation and PATH write formal records.
   The current PATH output deliberately contains 160 review-only and zero formal facts; all rows
   remain `pnt_eligible=false`.
7. Select and freeze one exact HF checkpoint/tokenizer plus prompt, scoring, retry, parsing, and
   primary-variant manifests before any model-specific execution.
8. Run Development 96 MCQ Original/Neutral/Targeted first to mine strict induced defects and retain
   resistant controls. Then run natural EN/ZH open completion for the unperturbed baseline and claim
   stratification, followed by Development-only PATH_not_token attribution and repair selection.
9. Freeze the behavior, attribution, vector, layer, scale, trigger, and repair contracts before
   opening Validation 32 for confirmation. Open Sealed 32 once only after Validation confirms them.
10. Do not call GPU, Ollama, HF, or external experiment/provider endpoints during the remaining
    offline review/freeze work; any further Codex proxy judgment must stay explicitly non-human.

## Errors Encountered
- CodeGraph context lookup was unavailable because its service connection closed; continued with repository specifications and implementation source.
- The first 10-item preflight exposed a `distractor_en`/`text_en` handoff mismatch; fixed and resumed without replaying completed upstream calls.
- `hy4-preview` exhausted its output budget on reasoning and intermittently returned `429006`; increasing its output budget recovered 17 of 19 failures, but 2/64 calls remain terminal failures after eight attempts each.
- Provider prices are not versioned in the repository, so token usage is auditable but currency cost is not. This remains a preflight gate failure.
- The original Simulation ID included `candidate_id` for every arm, causing duplicate baseline calls. One `hy4-preview` Chinese original baseline returned inconsistent A/B answers across the duplicated calls.
- The corrected three-arm rerun used 112 physical calls instead of 144 naive per-candidate calls, completed all calls, and found zero Paths Not Taken candidates among six Codex-accepted perturbations. One English neutral-control answer changed under `bailian/deepseek-v4-flash-0731`.
- The first representative `gpt-5-mini` probe returned HTTP 500 and passed on isolated retry. Its first 72-call Simulation batch completed but required 11 retries and reached 444,059 ms cumulative latency; the later 66-call L2 batch completed with no GPT retries.
- Matched L1/L2 generation initially failed 6/23 groups under a 768-token output budget. A versioned amendment raised the budget to 2,048 and resume completed all failed groups without replaying completed groups.
- The latest focused consolidation/pre-perturbation/closure/adjudication/finalizer/PATH suite
  passes 78/78. Full-suite discovery currently runs 271 tests: 270 pass and one unrelated import
  error remains because local Python lacks the pinned `anthropic` dependency.
- The first post-closure PATH conversion exposed an unhandled but producer-valid
  `relation_specific_open_completion` prompt tier. The adapter now maps that exact tier to
  `factual_completion`; unknown tiers still fail closed.
- The first streamlined proxy rerun exposed that frozen generated distractors were dropped when the runner reconstructed only original distractor inputs; the failed run was preserved, the frozen-checkpoint path was fixed, and the replacement run completed.
- Two long `gpt-5-mini` Simulation prompts repeatedly exhausted a 256-token completion budget and returned no parseable JSON. The model-specific budget was raised to 1,024 with a manifest amendment; checkpoint resume retried only the failures and reached 360/360.
- The external reference checkout `/Users/xiarongzhi/school task/paths_not_taken` contains a
  hard-coded Hugging Face credential in `demo_load_datasets_model.py:337`. Do not copy its value;
  revoke/rotate it and require environment-variable or secret-manager loading in any new runtime.

## Status
**The v7b 100-input diagnostic completed, but its formal gate did not pass** - 73/100 bases reached complete five-model Simulation coverage and 14 model-specific strict signals were reported. Translation incompleteness, unauditable currency cost, and 41 Neutral-control instabilities mean the run remains diagnostic proxy evidence. The proposed Claude holdout was not run; these records did not consume the current 160-fact Validation or Sealed partitions, cannot support causal Paths Not Taken claims, and remain `pnt_eligible=false`.

**The selected public-pool cohort is materialized through the last safe non-frozen step before
translation/distractor review and perturbation**

- Producer v2 contains 9,059 validation-clean Qwen triples and 8,969 provisional base facts. A
  separate relation-balanced review chain examined 251 records and materialized 160 retained
  facts across four relations. Seven facts carry explicit revisions with independent proxy
  re-review; all evidence remains `human_gold=false`.
- A full-pool-seeded bounded lexical closure emitted 73 reachable edges over 40 cohort and 63
  out-of-cohort nodes. All 73 were adjudicated (`30 same_fact`, `37 same_leakage_component`,
  `6 distinct`, `0 exclude_cohort`). Resolution retained 160/160 facts in 159 leakage components,
  enforced original split-group atomicity, and achieved exactly 24 development / 8 validation /
  8 sealed facts per relation. It does not claim exhaustive semantic-paraphrase or historical-
  exposure closure.
- The post-closure bundle contains 160 facts and 320 regenerated unverified distractors; the PATH
  bridge contains 160 review-only and 0 formal facts. Translation, distractor review, canonical
  freeze, split freeze, exact HF selection/tokenization, model inference, perturbation, hidden-
  state attribution, intervention, and repair remain unperformed. No GPU, Ollama, HF checkpoint,
  or external experiment/provider endpoint was used in this phase. Development MCQ mining has not
  started; Validation and Sealed have not run; all 160 rows remain `pnt_eligible=false`.

**Historical local Ollama compatibility run completed on the legacy frozen v5c records** - `gemma3:12b` and `llama3.1:8b` completed 180/180 Simulation calls over the same 15 candidates. Two model-specific strict signals were observed for `llama3.1:8b`; none were observed for `gemma3:12b`. Nine Original/Neutral choice changes mean this remains a compatibility/calibration result with `pnt_eligible=false`, not the current 160-fact split or a frozen mechanism cohort.
