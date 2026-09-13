# Task Plan: Chinese-first Factual Perturbation Before Paths Not Taken

## Goal
Create an auditable candidate taxonomy and explicit mapping for all Qwen relation signatures, apply it deterministically, and report unresolved review work without presenting unreviewed labels as frozen gold.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Profile inventory and existing pilot artifacts
- [x] Phase 3: Build candidate taxonomy and complete signature mapping
- [x] Phase 4: Apply mapping and validate normalized output
- [x] Phase 5: Review risks and deliver report
- [x] Phase 6: Decouple factual-prompt generation from relation normalization
- [x] Phase 7: Configure and validate the Chinese translation/perturbation/simulation stack
- [ ] Phase 8: Run the bounded Chinese factual-perturbation pilot
- [ ] Phase 9: Prepare Paths Not Taken only after the perturbation dataset is frozen

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
- Separate description-to-term relations from term-to-definition relations; their directions are opposite.
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
- Use API-catalog discovery plus live probes rather than only `.env` defaults: use `qwen3.8-max` for primary Chinese translation and strict distractor/perturbation validation, `gpt-5.5` for translation review, and `gpt-5.6-sol` for controlled perturbation generation. Reject uncertain or invalid judge outputs.
- Keep the perturbation generator out of both self-validation and simulation scoring. Use the four-family simulation ensemble `qwen3.7-plus`, `bailian/deepseek-v4-flash-0731`, `hy4-preview`, and `gemini-3.7-flash`; reserve `claude-opus-4-8` as a frozen-candidate model-and-family holdout. Do not call any SenseNova model in this run.
- Do not enable same-relation distractor sampling in the first pilot. Use verified source wrong options first and verified same-answer-type negatives second; add `probe_relation_id` before relation-conditioned negatives.
- Treat Codex review as a versioned proxy audit (`reviewer_type=codex_proxy`, `not_human_gold=true`), not as human gold. Exclude unresolved facts and candidates rather than silently accepting them.
- Use three Simulation arms: shared `original`, shared `neutral`, and candidate-specific `targeted`. Reuse each baseline by `(source_id, distractor_id, model, language, variant)` and reject evidence with inconsistent duplicate baselines.
- Do not expand directly to 100 after the 10-item engineering preflight. First run approximately 20 new paired-control base triples because the corrected three-arm rerun reproduced zero of the prior provisional effects.
- For the sensitivity screen, use `qwen3.6-27b`, `bailian/deepseek-v3.2`, `gpt-5-mini`, and `gemini-3.7-flash`; retain the latter as a strong anchor and keep `gpt-5.6-sol` out of Simulation because it generated the perturbations.
- Generate ordered L1/L2 truthful perturbations with candidate-specific, length/topic/style-matched Neutral contexts. Keep only Original shared in this design.
- The multi-option calibration is now complete: 480/480 calls across 20 Codex-reviewed independent base triples produced zero target-distractor-specific Chinese flips. The next authorized design is a 100-input-base diagnostic Chinese pilot, not a production candidate freeze.
- The earlier v4 100-input diagnostic pilot completed 840/840 Simulation outcomes on 35 retained sources and produced one strict offline candidate. A later streamlined v5 calibration on 20 new sources reduced hard filtering through translation repair, distractor replenishment, advisory perturbation judging, and Simulation-blind Codex selection.
- Streamlined v5 retained 15/20 sources, completed 360/360 Simulation outcomes, had zero Neutral instability, and produced two strict Chinese-specific candidates. This is a calibration pass for proceeding to a new v5 100-input diagnostic pilot, not evidence sufficient for candidate freeze or holdout.

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
- [ ] 7.16 Run a preregistered 100-input-base diagnostic Chinese pilot with the multi-option endpoint; if it again yields no target-specific signal, predeclare a separate alternate-language transfer screen.
- [x] 7.17 Run the streamlined v5 calibration on 20 new sources: translation 18/20, complete option sets 17/20, Codex-retained sources 15/20, Simulation 360/360, Neutral instability 0/120, and strict Chinese-specific candidates 2/15.
- [x] 7.18 Add loopback-only, credential-free Ollama Simulation routing and rerun the latest 15 frozen v5c candidates with `gemma3:12b` and `llama3.1:8b`: 180/180 calls completed, with 2 model-specific strict signals from `llama3.1:8b` and 0 from `gemma3:12b`.
- [ ] 8.1 Run the frozen 100-base-triple pilot with at most two verified distractors per triple, two generated perturbations per distractor, beam width 4, and maximum depth 2.
- [ ] 8.2 Require all four simulation models for primary search scores; report partial model coverage separately rather than silently averaging fewer models.
- [ ] 8.3 Freeze induced and amplified badcase candidates only after semantic checks and simulation scoring; then evaluate them once on the reserved Claude Opus 4.8 holdout.
- [ ] 8.4 Report independent base-triple counts, candidate attrition, model-specific failure rates, English-retention rate, Chinese accuracy drop, distractor-hit rate, and non-simulation transfer without treating perturbation variants as independent samples.
- [ ] 9.1 Define `probe_relation_id`, answer-token contracts, hidden-state hooks, and Paths Not Taken admission states only after the perturbation artifacts and model-behavior cohorts are stable.
- [ ] 9.2 Keep the previously specified Paths Not Taken relation-specific and causal-mechanism work deferred; do not block the Chinese perturbation MVP on it.

## Immediate Execution Order
1. Complete prompt decoupling tasks 6.1-6.4 and regenerate the versioned canonical factual-prompt artifact without requiring a mapped relation.
2. Implement provider/model routing, structured-response validation, redacted logging, retry policy, and single-writer resumable checkpoints from `configs/factual_perturbation_zh_mvp_v1.json`.
3. Freeze a deterministic 10-base-triple end-to-end preflight sample stratified by source dataset, prompt quality tier, and answer type.
4. Generate Chinese prompts with `qwen3.8-max`, review them with `gpt-5.5`, and reject unresolved translations rather than silently repairing them.
5. Verify source wrong options and same-answer-type distractors, generate perturbations with `gpt-5.6-sol`, and validate them independently with `qwen3.8-max`.
6. Run all four simulation models on every retained preflight candidate; preserve individual failures and do not compute the primary ensemble score unless all four results are present.
7. Audit the 10-triple preflight for semantic validity, resume correctness, latency, token usage, per-provider errors, and candidate attrition before authorizing the 100-base-triple pilot.
8. Run the frozen 100-base-triple pilot, freeze accepted induced/amplified candidates, then call `claude-opus-4-8` exactly once per frozen candidate as the holdout.
9. Produce the attrition, English-retention, Chinese accuracy-drop, distractor-hit, per-model failure, and holdout-transfer report. Keep Paths Not Taken disabled until this dataset and cohort definition are frozen.

## Errors Encountered
- CodeGraph context lookup was unavailable because its service connection closed; continued with repository specifications and implementation source.
- The first 10-item preflight exposed a `distractor_en`/`text_en` handoff mismatch; fixed and resumed without replaying completed upstream calls.
- `hy4-preview` exhausted its output budget on reasoning and intermittently returned `429006`; increasing its output budget recovered 17 of 19 failures, but 2/64 calls remain terminal failures after eight attempts each.
- Provider prices are not versioned in the repository, so token usage is auditable but currency cost is not. This remains a preflight gate failure.
- The original Simulation ID included `candidate_id` for every arm, causing duplicate baseline calls. One `hy4-preview` Chinese original baseline returned inconsistent A/B answers across the duplicated calls.
- The corrected three-arm rerun used 112 physical calls instead of 144 naive per-candidate calls, completed all calls, and found zero Paths Not Taken candidates among six Codex-accepted perturbations. One English neutral-control answer changed under `bailian/deepseek-v4-flash-0731`.
- The first representative `gpt-5-mini` probe returned HTTP 500 and passed on isolated retry. Its first 72-call Simulation batch completed but required 11 retries and reached 444,059 ms cumulative latency; the later 66-call L2 batch completed with no GPT retries.
- Matched L1/L2 generation initially failed 6/23 groups under a 768-token output budget. A versioned amendment raised the budget to 2,048 and resume completed all failed groups without replaying completed groups.
- Full-suite discovery remains environment-blocked for two unrelated modules because the local Python environment lacks `anthropic`; the 25 relevant factual-perturbation/canonical tests pass.
- The first streamlined proxy rerun exposed that frozen generated distractors were dropped when the runner reconstructed only original distractor inputs; the failed run was preserved, the frozen-checkpoint path was fixed, and the replacement run completed.
- Two long `gpt-5-mini` Simulation prompts repeatedly exhausted a 256-token completion budget and returned no parseable JSON. The model-specific budget was raised to 1,024 with a manifest amendment; checkpoint resume retried only the failures and reached 360/360.

## Status
**Streamlined v5 calibration completed; a new v5 100-input-base Chinese pilot is justified next** - 15/20 independent base triples survived the streamlined funnel and completed 360/360 three-option, three-arm, bilingual Simulation outcomes. Two candidates met the strict Chinese-specific offline condition, but exact paired evidence remains non-significant (`p=0.125`). Claude holdout and Paths Not Taken remain disabled.

**Local Ollama compatibility run completed on the latest frozen v5c records** - `gemma3:12b` and `llama3.1:8b` completed 180/180 Simulation calls over the same 15 candidates. Two model-specific strict signals were observed for `llama3.1:8b`; none were observed for `gemma3:12b`. Nine Original/Neutral choice changes mean this remains a compatibility/calibration result rather than a frozen mechanism cohort.
