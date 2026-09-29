## Context

`data/source/` contains immutable English multiple-choice records with `question`, `choices`, `answer`, `source`, and optional dataset-subject metadata. The correct answer is already known, so the model does not need to solve the MCQ. It must determine whether the question-answer pair expresses one atomic fact and, if so, orient a triple whose object is exactly the canonical source answer.

The original change covered factual-triple construction, relation normalization, and prompt
preparation:

```text
raw_source_manifest.json
  -> models/sensenova-6.7-flash-lite/triple_extractions.jsonl
  -> models/qwen3.7-plus/triple_extractions.jsonl
  -> calibration/codex_reference_labels.jsonl
  -> calibration/model_evaluation.json
  -> codex_summary.json
  -> relation_inventory.json
  -> relation_taxonomy_v1.json
  -> relation_mapping_v1.json
  -> triples_normalized.jsonl
  -> factual_prompts.jsonl
  -> distractor_candidates.jsonl
```

The current offline extension also consumes the completed public Qwen extraction as a strictly
single-model provisional source:

```text
qwen3.7-plus/triple_extractions.jsonl
  -> provisional_triples.jsonl + base_fact_clusters.jsonl
  -> review_queue.jsonl
     + coverage-oriented 400
     + overall-random 400
     + prompt-risk-targeted 200
  -> behavior_input_bundle.jsonl + split_manifest.json
  -> SHA-bound review scope -> expanded review export -> review staging
  -> bounded lexical near-duplicate candidates
  -> review_only PATH_not_token base facts
```

The next, not-yet-implemented production chain is deliberately separate:

```text
predeclared cohort-universe inventory
  -> complete final review dispositions
  -> provenance-preserving revisions and re-review
  -> re-clustered reviewed bundle
  -> full-pool duplicate/near-duplicate and historical-exposure closure
  -> recomputed development/validation/sealed split
  -> exact-bundle review-freeze and split-freeze manifests
  -> formal MCQ behavior and PATH inputs
  -> Development exact-HF Original/Neutral/Targeted MCQ weakness mining
  -> strict induced defects plus matched resistant controls
  -> natural EN/ZH open-completion baseline and claim stratification
  -> exact-HF attribution -> resid_pre intervention -> repair regression
  -> Validation fixed confirmation -> Sealed one-time evaluation
```

It generates English factual prompts and unverified distractor candidates, but does not promote
provisional rows, call a tokenizer/model, generate translations or perturbations, screen model
behavior, collect hidden states, or claim an intervention result.

## Goals / Non-Goals

**Goals:**

- Run `sensenova-6.7-flash-lite` and `qwen3.7-plus` independently with the same extraction contract.
- Balance the calibration pilot across the five source datasets, then stratify within each source by optional subject metadata.
- Preserve source provenance and original choices without asking the model to reproduce them.
- Extract only atomic facts and record a machine-readable exclusion reason when no triple is extractable.
- Exclude full-sentence, explanatory, procedural, recommendation, and multi-clause canonical answers that are unsuitable as one triple object.
- Preserve the canonical `source_answer` exactly while permitting `answer` to be a shorter normalized entity or phrase explicitly entailed by it.
- Require triple direction to remain `subject -> canonical answer` even when the reverse relation has a more familiar name.
- Aggregate relation signatures globally before defining normalized labels.
- Freeze a versioned taxonomy and explicit mapping so normalization is reproducible.
- Reserve `qwen3.7-plus` or Codex for global taxonomy induction, mapping review, and ambiguous cases.
- Keep direct dual-model answer agreements and require explicit Codex adjudication before accepting or rejecting a single-model extraction or material answer conflict.
- Generate deterministic English recall prompts for every reviewed prompt-valid fact, independent
  of relation mapping, and preserve prompt-quality tiers.
- Copy source wrong options into distractor candidates without prematurely claiming type or factual verification.
- Import the public Qwen extraction with immutable input/record hashes, deterministic fact
  clustering, an independent review queue, and provisional answer-group split metadata.
- Separate coverage-oriented, population-estimation, and targeted-risk review samples and state
  their inference limits in machine-readable metadata.
- Bind review scope, exported duplicate members, and decisions to exact source hashes; permit
  partial staging without interpreting missing decisions as rejects.
- Expose one versioned behavior/PATH bridge while keeping single-model evidence outside formal
  cohorts until review, evidence, prompt, and frozen-split gates all pass with external manifests
  bound to the exact bundle.
- Use MCQ perturbation as the weakness-mining entry point, natural English/Chinese open completion
  as the baseline and claim-stratification evidence, and one exact-HF checkpoint as the sole
  white-box PATH_not_token attribution-and-repair target.

**Non-Goals:**

- Retaining or recalibrating the previous `accept` / `reject` / `needs_review` audit.
- Generating `prompt_en` during per-model triple extraction.
- Verifying or augmenting distractors with same-relation negatives, generating perturbation text, translating prompts, screening models, or searching cross-lingual badcases.
- Modifying raw files in `data/source/`.
- Forcing every record or every relation into the taxonomy.
- Treating API/Ollama screening as exact-HF white-box attribution or running GPU-dependent
  tokenization, activation, vector, intervention, or repair experiments in this offline phase.
- Treating the historical five-model diagnostic panel, any API/Ollama strict signal, or an
  open-completion result by itself as a frozen induced-defect cohort or as PATH eligibility.

## Decisions

### Preserve choices in code, not in the LLM response

The output writer copies `source_choices` from the immutable source snapshot. The extraction prompt intentionally omits choices so extractability is tested without option comparison. This also prevents the model from altering choice text or order.

### Keep extraction and normalization separate

The extraction model returns `relation_raw`, not `relation_normalized`. A global inventory groups observed signatures using:

```text
relation_raw + subject_type + answer_type + direction + examples + count
```

A stronger model can then propose a taxonomy and mapping with global visibility. After review, the taxonomy and mapping are frozen and applied programmatically. This is less expensive than two strong-model calls per row and more consistent than allowing each row to invent a normalized label.

The weak model's `subject_type` and `answer_type` are evidence rather than ground truth. Each mapped signature therefore also records `subject_type_normalized` and `answer_type_normalized`, which must match the frozen taxonomy and may correct weak-model type errors without rewriting the raw extraction.

### Keep an extraction gate without recreating the old audit

Every row has `extraction_status=extracted|not_extractable`. A non-extractable record includes one exclusion reason and null triple fields. This gate expresses only whether an atomic triple can be formed; it does not retain the old tri-state review decision or its audit findings.

### Canonical grounding and relation direction are hard constraints

The source answer is authoritative and remains unchanged in provenance. For an extracted row, `answer` must either copy it or normalize an explicitly stated entity or short phrase without adding unsupported information. For example, the source sentence `Obama was born in Hawaii, which is a US state` may normalize to `United States` for a `country_of_birth` answer. The model must not reverse the triple.

The subject must be an independently identifiable anchor from the question and must not repeat the answer. A single definitional description may be represented with a stable `definition_term` or classification relation. Multi-clue biographical riddles, scenario interpretation, and descriptions combining several independent facts remain clue solving.

### Calibrate before expansion

The 100-row pilot has a frozen Codex reference decision for every row. Model error is the fraction whose `extraction_status` differs from that binary reference label; field-level issues are reported separately and cannot be hidden by label accuracy. Both models must have at most three label errors out of 100 before a disjoint 300-row manifest is run. This is a calibration criterion against a Codex reference, not an estimate of human-gold accuracy.

### Isolate model outputs and endpoints

Every model writes to `models/<model-slug>/`. Resume reads only that model's checkpoint and rejects mixed prompt/model IDs. Model calls may run concurrently inside a model and the two model processes may run concurrently because they never share an output file. `qwen3.7-plus` uses `OPENAI_BASE_URL`, whose experiment value is `https://ctapi.csxdtx.com:16000/v1`; API credentials and base URLs are never persisted in result rows.

### Normalize only through explicit artifacts

`relation_taxonomy_v1.json` defines every relation ID, definition, subject type, answer type, direction, and examples. `relation_mapping_v1.json` maps observed relation signatures to those IDs. `triples_normalized.jsonl` is produced only from these artifacts. Unknown and ambiguous signatures remain unresolved and are never silently coerced.

### Canonicalize by agreement plus explicit Codex adjudication

Rows with two locally valid extractions and equivalent normalized answers enter directly;
SenseNova supplies their canonical fields and both annotations remain attached. A row with
exactly one valid extraction or two materially different answers is `pending_review`, not
excluded. The downstream inventory is blocked until a versioned Codex decision covers every
queued row. An accepted decision selects one of the eligible model annotations without editing
its fields; a rejected decision records an item-specific reason. Source provenance and choices
remain unchanged in either case.

### Separate prompt precision from relation statistics

`relation_normalized` supports grouping and statistics, but broad taxonomy templates can lose
the precise raw relation. Prompt generation therefore first removes a sentence-final answer
from `canonical_fact`, yielding a strict completion stem. When the answer does not occur at the
end, the system emits an option-free `Factual question: ... / Answer:` fallback and labels it as
`open_answer_fallback`. Prompt readiness depends on reviewed canonical evidence and prompt
validation, not relation-normalization status: mapped, ambiguous, and out-of-taxonomy rows can
all receive prompts. A separate nullable `probe_relation_id` controls admission to
relation-conditioned mechanism experiments. Candidate mapping is also not evidence of direction:
the broad raw label `term for definition` is explicitly emitted with unresolved direction and
must not be treated as an automatic `description -> term` mapping.

### Treat source wrong options as candidates, not verified negatives

For every prompt-ready triple, the pipeline copies all original choices except the canonical
source answer and normalized answer. It ranks them deterministically by answer-text similarity
and selects the top candidate for convenience, but records `distractor_verified=false` and
does not claim that type compatibility or `(subject, relation, distractor)` falsity has been
independently verified.

### Keep the public bridge provisional until independent review

The producer-v2 public adapter admits only completed, extracted, validation-clean Qwen rows and
labels every one `canonical_status=pending_review`, `evidence_tier=provisional_single_model`, and
`human_gold=false`. Exact normalized duplicates share a stable `base_fact_id`; the full source,
raw extraction, retry, and local-adjudication provenance remains available. Statistical relation
labels, nullable probe-relation candidates, candidate direction, and factual-prompt readiness are
separate fields.

### Separate review samples by inferential purpose

The coverage-oriented sample uses balanced round-robin selection to expose more source/format,
prompt-tier, relation-candidate, and split strata. It carries no population weight and cannot
estimate whole-pool outcome rates. The overall-random sample uses deterministic uniform hash-rank
selection and may estimate the full review queue only when its seed precedes outcomes, all sampled
items receive the same protocol, and nonresponse is reported and handled. The prompt-risk sample
is drawn only from flagged fallback prompts; it is targeted diagnostic evidence and cannot be
extrapolated to the full pool.

### Make review auditable but staging-only

The review tool has three explicit operations. `scope` binds a selected set to the exact bytes of
the behavior bundle, review queue, base-fact clusters, and provisional triples. `export` expands
all members of duplicate clusters and binds each review item and member record hash. `apply`
accepts `accept`, request-only `revise`, `reject`, or `defer`, and emits only review staging.
Partial application is explicit: missing decisions remain missing and are not counted as reject.
Codex decisions are proxy evidence with `reviewer_type=codex_proxy` and `human_gold=false`.
Neither `apply` nor `revise` mutates source facts, promotes canonical status, or freezes a split.

### Define the formal cohort universe independently of review sampling

A review sample answers a sampling question; it does not become a formal cohort by implication.
The finalizer must materialize a distinct immutable cohort-universe artifact with an explicit ID,
ordered row inventory, source scope, inclusion policy, and hashes. The universe may be the complete
8,969-fact pool or a predeclared subset. Every scoped item must reach a final `accept` or `reject`
after requested revisions are applied and re-reviewed; `missing`, `defer`, and unapplied `revise`
block freeze.

The current universe-v2 utility is diagnostic-only. Its comparison corpus is selected by a
path-independent pinned canonical-v2 contract that fixes dataset ID/role, format, SHA-256,
byte/record counts, policy version, ordered identities, and ordered row hashes; a projection of the
provisional behavior rows cannot self-declare itself as that corpus. The declaration also freezes
the complete lexical-audit contract (schemas, normalization/blocking versions, thresholds, bucket
and example bounds, MinHash layout, and review rules). A later audit using different parameters is
invalid even if it replays deterministically. Because no authoritative historical-exposure
inventory has been supplied, v2 records `historical_exposure_contract_status=not_predeclared` and
always retains an authority-missing blocker. A runtime registry cannot retroactively establish that
trust boundary.

The overall-random 400 may estimate full-pool review outcomes under its stated assumptions, but it
cannot promote the other 8,569 facts. If it is separately predeclared as the cohort source, all
400 items must be reviewed under one protocol and its accepted output must undergo duplicate,
near-duplicate, and historical-exposure closure against the complete pool. Freezing the complete
pool instead requires terminal item-level review of all 8,969 facts.

### Treat lexical near-duplicate output as candidate generation

Exact indexes and bounded MinHash/lexical blocking generate review candidates for answer aliases,
questions, and canonical facts. Emitted edges are representative candidates, not an exhaustive
record-pair census. The audit may contain false positives and miss semantic paraphrases; it makes
no same-entity or semantic-equivalence decision and therefore cannot set
`semantic_review_complete=true`.

Revision lineage v2 reconstructs only a preflight semantic patch over the source behavior row. It
must declare `revised_record_role=preflight_reconstruction_not_final_behavior_row` and
`derived_fields_recomputed=false`, bind the exact source `revise` staging-row hash, and record
non-empty editor identity/method/time. The re-review separately records reviewer
identity/method/time and binds the revised-row hash. Any later producer must re-derive prompt text, distractors,
normalized relation/probe mapping, duplicate/cluster membership, and split assignment before a
revised row can become formal; the reconstruction cannot be copied directly into a final bundle.

### Require external, exact-bundle freeze evidence

The first split groups normalized answers to reduce direct answer leakage, reserves development,
validation, and sealed partitions, and remains `split_status=provisional_not_frozen`. It is not a
formal split until semantic/near-duplicate review is complete. A reviewed explicit bundle can
enter factual-perturbation preparation only when canonical, bundle, evidence, prompt, and split
contracts all pass and external review-freeze and split-freeze manifests bind the exact bundle
SHA. Row-local status claims cannot substitute for those manifests. The ordinary prepare defaults
to development only, and schema-less explicit bundles cannot fall back to the legacy loader. The
PATH adapter applies the same review and split contract, but still marks tokenization and all
mechanism stages pending/not-run.

The current repository implements consumers that validate these manifest types, but it does not
yet implement the reviewed-bundle/freeze producer. A trustworthy producer must bind the cohort
universe, scope/export/apply or equivalent final-decision artifacts, decision completeness and
counts, revision/re-review lineage, re-clustering result, duplicate/exposure adjudication, final
bundle, and split. Self-asserted completion booleans are not sufficient provenance.

### Separate source rules, factual verification, semantic review, and target behavior

Formal admission uses four non-interchangeable authorities. First, deterministic code binds source
snapshots, IDs, hashes, lineage, aliases, duplicate/leakage components, split locality, and
distractor-selection rules; this establishes reproducibility, not factual truth. Second, factual
verification records source evidence for the canonical proposition and for why each retained
distractor is not an acceptable answer under the same relation, entity, and temporal scope.
Unsupported or genuinely ambiguous cases defer or fail rather than passing from model memory.

Third, an independently identified strong semantic reviewer evaluates the fully rendered bilingual
material before target behavior is observed. Its frozen rubric covers answer uniqueness, relation
and direction preservation, translation equivalence, Neutral neutrality, and whether Targeted
content points only toward the designated distractor. Reviewer model/version, prompt SHA, structured
decision, uncertainty, and any adjudication remain auditable; proxy review never becomes human gold.
Any regenerated or edited variant invalidates the prior semantic decision.

Fourth, the frozen exact-HF target measures behavior. It does not judge data truth or semantic
admission. Its raw six-arm outputs are kept separate from the deterministic labeler that derives
strict, resistant, neutral-unstable, off-target, and incomplete statuses. A surprising target
response may trigger a new blinded review and a new data version, but it cannot silently rewrite
the reviewed inputs or their freeze evidence.

### Separate MCQ mining, natural baselines, and exact-HF mechanism work

The formal 160-fact loop begins with Original/Neutral/Targeted MCQ behavior on Development. MCQ is
the weakness-mining entry point: a strict induced defect requires a correct Original, a stable
Neutral, and a Targeted response that selects the predesignated distractor for one exact target
model. Resistant records exposed to the same manipulation but remaining correct are retained as
matched mechanism controls. Target-model behavior cannot retroactively approve facts,
translations, distractors, or perturbation wording, and it cannot be used to choose the primary
variant after observing outputs.

Natural English/Chinese open completion is required after mining as the unperturbed factual-recall
baseline and as a claim-stratification measurement. It distinguishes a natural cross-lingual recall
gap from an MCQ-conditioned induced vulnerability; it is not a replacement mining route, and an
MCQ-only effect must not be reported as a natural open-completion failure.

One exact Hugging Face model repository, commit revision, tokenizer revision, weight
precision/quantization, prompt formatting, and decoding/scoring policy must be frozen before formal
behavior execution. The same model identity must produce Development MCQ behavior, the natural
open-completion baseline, hidden states and Logit Lens, recall task vector,
translation-minus-recall difference vector, `resid_pre` intervention, repair evaluation, and
regression controls. Vector construction uses training records only; layer and scale selection use
Development data only. Validation receives only the frozen behavior, attribution, and repair
protocol; Sealed is opened once after Validation confirms it. Neither partition is used for mining
or tuning.

Ollama/API outputs are diagnostic behavior evidence only, even when their display name resembles
the HF model. They do not establish weight, tokenizer, chat-template, precision, or hook identity
and therefore cannot substitute for the white-box chain. In particular, the completed historical
five-model Simulation is a diagnostic proxy: its records remain `pnt_eligible=false`, and it did
not execute the current 160-fact Validation or Sealed partitions.

### Remove credentials from the reference implementation contract

The external reference checkout contains a hard-coded Hugging Face credential at
`/Users/xiarongzhi/school task/paths_not_taken/demo_load_datasets_model.py:337`. Its value must not
be copied into this repository or artifacts. It must be revoked/rotated before reuse, and any new
runtime must load credentials from an environment variable or secret manager. This change does
not modify the external checkout.

## Current Offline Snapshot

- Producer v2 retained 9,059 validation-clean triples and formed 8,969 provisional base facts.
- Review artifacts include a 400-item/98-stratum coverage sample, a distinct 400-item overall
  random sample, and a 200-item targeted sample drawn from a 376-fact prompt-risk frame. Only the
  overall-random sample supports conditional whole-pool estimation.
- The current 20-item Codex-proxy pilot has 16 `accept`, 2 `defer`, 1 `reject`, and 1 `revise`;
  the remaining 380 scope items are missing. Its output is staging, not a review freeze.
- Lexical audit v2 emitted 2,341 representative candidate pairs, 281 of them cross-split. It is
  not an exhaustive census, semantic review remains incomplete, and the provisional split is not
  freezeable.
- PATH export contains 8,969 review-only facts and 0 formal facts. Exact-bundle review-freeze and
  split-freeze manifests have not been produced.
- A fail-closed utility can materialize a separately chosen full-pool or review-scope diagnostic
  universe and report evidence blockers. It pins canonical-v2 comparison content and lexical
  policy but intentionally records historical authority as not predeclared. No real finalizable
  cohort has been selected or declared, and no reviewed-bundle/freeze producer exists; therefore
  the mining-to-repair workflow is not yet closed.
- This extension invoked no GPU, Ollama, HF checkpoint, or external experiment/provider endpoint;
  its 20 semantic decisions are explicitly labeled Codex proxy judgments, not human gold.
- The later authoritative post-closure v2 artifact retains 160 review-only facts in a provisional
  96/32/32 assignment and 320 unverified distractor candidates. Canonical and split freeze,
  translation/distractor review, broader semantic and historical-exposure closure, exact-HF
  selection, MCQ behavior, natural open-completion baseline, attribution, and repair are all still
  pending. Validation and Sealed have not been run, so every current 160-fact row remains
  `pnt_eligible=false` (serialized as `path_not_token_experiment_ready=false`).
- The historical `diagnostic-100-five-model-v7b` run completed 2,190/2,190 API Simulation calls
  over 73 complete candidates and reported 14 model-specific strict signals. Its formal gate stayed
  false because translation coverage and currency-cost audit were incomplete and 41 Neutral
  instabilities were present. Those outputs are diagnostic proxy evidence only, are not part of the
  160-fact split, and do not authorize Validation, Sealed, or PATH_not_token mechanism claims.

### Version and isolate extraction outputs

The extraction prompt version, model ID, raw response, validation errors, retry history, and timestamps are retained. Resume rejects a checkpoint created by another prompt version or extraction model.

## Risks / Trade-offs

- [Weak-model extraction errors propagate] -> Run a stratified pilot, validate exact answers locally, and manually inspect relation direction and atomicity before scaling.
- [Sentence-like source answers waste calls or become malformed triples] -> Apply a conservative program-owned answer-suitability gate before the model and record the skipped row with full provenance.
- [Free-form relations fragment] -> Build a counted global inventory and freeze an explicit mapping before normalization.
- [A broad relation collapses different semantics] -> Include subject/answer types and representative canonical facts in inventory entries.
- [A reverse relation is forced into a familiar label] -> Validate direction against each taxonomy signature and permit `out_of_taxonomy`.
- [Source choices are lost] -> Copy them directly from the source snapshot into every terminal extraction record.
- [Strong-model endpoint changes] -> Record endpoint-independent model ID and run a preflight before taxonomy work; never record API keys.
- [Coverage sample is misread as representative] -> Keep it separate from the overall-random
  sample and machine-record `population_estimation_supported=false`.
- [Partial review silently becomes rejection] -> Require explicit `--allow-partial`, retain
  `missing`, and forbid automatic canonical promotion.
- [Lexical candidates are treated as semantic duplicates] -> Record bounded blocking and
  non-exhaustive census limitations; require human or declared proxy adjudication of every
  blocking link/component before split freeze.
- [Row status is forged or becomes stale] -> Require external review/split manifests bound to the
  exact bundle SHA before formal behavior or PATH admission.
- [A random review sample is silently treated as the cohort] -> Materialize the cohort universe
  separately; require complete item-level disposition and full-pool duplicate/exposure closure.
- [Completion booleans are self-asserted] -> Bind the freeze manifests to final decisions,
  revisions, re-clustering, adjudication artifacts, exact universe/counts, and output row hashes.
- [Ollama and HF runs are conflated] -> Freeze one exact HF model/tokenizer identity for the entire
  MCQ-mining-to-baseline-to-repair chain and report Ollama/API results only as behavioral
  diagnostics with `pnt_eligible=false`.
- [Open completion and MCQ exchange roles] -> Use MCQ perturbation to mine directed defects and
  retain resistant controls; use natural EN/ZH open completion only for the unperturbed baseline
  and claim stratification before exact-HF attribution and repair.
- [Reference credential is reused] -> Revoke/rotate it and require environment or secret-manager
  loading without persisting the value.

## Migration Plan

1. Preserve `data/source/` and model-independent `raw_source_manifest.json` files.
2. Remove audit-specific checkpoints, logs, supervisor code, reports, prompt, validation, runner, and tests.
3. Introduce extraction-specific configuration, prompt, validation, CLI, records, and resume isolation.
4. Run offline unit tests and a bounded online SenseNova extraction smoke test.
5. Build `relation_inventory.json` from valid extracted rows.
6. Use Codex/`qwen3.7-plus` to propose and review `relation_taxonomy_v1.json` plus `relation_mapping_v1.json`.
7. Apply the frozen mapping to produce normalized triples and report unresolved signatures.
8. Generate factual prompts for every reviewed prompt-valid triple and construct traceable source-option distractor candidates.
9. Import the public Qwen output as provisional-only records, emit review/split/bundle artifacts,
   and verify exact overlap without changing the earlier canonical artifacts.
10. Build purpose-specific review samples and a SHA-bound scope/export/apply staging chain; keep
    Codex provenance non-human and partial missing decisions unresolved.
11. Predeclare and separately materialize the exact formal cohort universe. If it is the
    overall-random 400, complete 400/400 review; if it is the full pool, review all 8,969 facts.
12. Apply and re-review revisions, generate bounded lexical near-duplicate candidates, and
    semantically adjudicate blocking edges/components with full-pool and historical-exposure
    closure before re-clustering and recomputing the split.
13. Implement the reviewed-bundle/freeze producer and bind its manifests to the exact universe,
    final decisions, revisions, adjudications, resulting bundle, and split. Only then admit formal
    behavior or PATH inputs.
14. Freeze one exact HF checkpoint/tokenizer; mine strict induced defects and resistant controls
    with Development MCQ, run the natural EN/ZH open-completion baseline, and develop
    hidden-state/Logit-Lens attribution plus task/difference-vector repair on Development only.
15. Freeze the complete behavior, mechanism, and repair rule; then use Validation for fixed
    confirmation and open Sealed once for the final repair/retention/regression evaluation. Until
    these gates open, export provisional facts only to the PATH review artifact and keep
    `pnt_eligible=false`.
