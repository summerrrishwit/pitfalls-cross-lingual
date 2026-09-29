<h1 align="center">Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models</h1>

🌐 **Project page:** https://xzx34.github.io/cross-lingual-pitfalls/

📄 **ACL Anthology:** https://aclanthology.org/2025.acl-long.404/

📄 **arXiv:** https://arxiv.org/abs/2505.18673

🤗 **Dataset (Hugging Face):** https://huggingface.co/datasets/xzx34/cross-lingual-pitfalls

## Updates & News
- [05/15/2025] 🥂 **Cross-Lingual Pitfalls has been accepted by ACL 2025! See you in Vienna!**

## Introduction

We introduce a systematic framework for studying **Cross-Lingual Weakness**—a critical challenge where LLMs fail to generalize their English proficiency to other languages. Our methodology enables:  

1. **Automated Weakness Identification**  
   Beam search-based perturbation strategy leveraging high-quality English datasets to systematically uncover cross-lingual weaknesses.  

2. **Quantitative Cross-Lingual Assessment**  
   A benchmarking framework measuring performance disparities across 16 languages, analyzing linguistic similarity effects.  

3. **Mitigation Strategy Evaluation**  
   Comparative analysis of fine-tuning effectiveness across languages, revealing how linguistic proximity influences adaptation.  

<p align="center">
<img width="85%" alt="CLP Pipeline" src="images/pipeline.jpg">    
</p>

## Dataset

The full benchmark — **6,713 bilingual (English ↔ target-language) pairs across 16 languages**, built from five English QA benchmarks (MMLU, ARC, CommonsenseQA, TruthfulQA, SciQ) — is available on the Hugging Face Hub and mirrored under [`data/`](data/):

🤗 **https://huggingface.co/datasets/xzx34/cross-lingual-pitfalls**

```python
from datasets import load_dataset
ds = load_dataset("xzx34/cross-lingual-pitfalls", "Chinese")
```

## Installation

```bash
conda create -n clp python=3.9
conda activate clp
pip install -r requirements.txt
```

## Configuration

Place the .env file in the utils folder. You can selectively add API keys for the models you intend to use.

```properties
HTTP_PROXY=your_http_proxy
HTTPS_PROXY=your_https_proxy

OPENAI_BASE_URL=https://ctapi.csxdtx.com:16000/v1
OPENAI_API_KEY=your_openai_api_key

DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
DEEPINFRA_API_KEY=your_deepinfra_api_key

YI_BASE_URL=https://api.lingyiwanwu.com/v1/
YI_API_KEY=your_yi_api_key

# Alibaba Cloud Model Studio / Bailian (OpenAI-compatible)
# You only need to fill one key variable. BAILIAN_API_KEY is preferred;
# DASHSCOPE_API_KEY is also supported for compatibility with DashScope docs.
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
BAILIAN_TIMEOUT_SECONDS=120
BAILIAN_API_KEY=your_bailian_api_key
# DASHSCOPE_API_KEY=your_dashscope_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# SenseNova Anthropic SDK base URL; the SDK appends /v1/messages
SENSENOVA_BASE_URL=https://token.sensenova.cn
SENSENOVA_API_KEY=
```

## Usage

### Step 1: Generate cross-lingual weaknesses

```bash
python run.py 
  --input-file data/source/mmlu.json 
  --output-file data/mmlu_Chinese.json 
  --language Chinese 
  --max-good 3 
  --max-queue 12 
  --batch-size 4 
  --models gpt-4o llama-3.1-8B qwen-2.5-72B gpt-4o-mini gemma-2-27B   
```

### Step 2: Evaluate performance disparities
```bash
python eva.py \
  --lang Chinese \
  --input_file data/Chinese.json \
  --output_file data/Chinese_results.json \
  --models qwen3.7-plus qwen3.7-max
```

`--models` is required. Only the models passed on the command line are evaluated
and checkpointed. Checkpoints are stored separately under `data/Chinese/`, for
example `Chinese_results_qwen3.7-plus_progress.json`.

To evaluate only Bailian Qwen 3.7 models:

```bash
python eva.py \
  --lang Chinese \
  --input_file data/Chinese.json \
  --output_file data/Chinese_qwen37_results.json \
  --models qwen3.7-plus qwen3.7-max \
  --max-workers 2
```

To evaluate the additional OpenAI-compatible Qwen and DeepSeek models:

```bash
python eva.py \
  --lang Chinese \
  --input_file data/Chinese.json \
  --output_file data/Chinese_qwen_deepseek_results.json \
  --models qwen3.5-plus qwen3-max deepseek-v3 deepseek-v4-pro deepseek-v4-flash \
  --max-workers 2
```

### Step 3: Analyze Results
```bash
python visualization.py 
  --languages Chinese Japanese Korean French Spanish Italian Ukrainian German Bengali Hindi Arabic Hebrew Amharic Yoruba Swahili Zulu 
  --input_folder data/ 
  --output_folder visualizations/
```

### Atomic factual-triple extraction

This workflow is intentionally separate from `run.py` and `eva.py`. It starts from
the five immutable English QA datasets under `data/source/`, creates a raw-source
manifest before any model request, extracts atomic `(subject, relation_raw, answer)`
triples independently with SenseNova and Qwen, and builds per-model relation inventories.
After both full checkpoints finish, a separate offline postprocessor keeps direct answer
agreements and routes single-model extractions plus material answer conflicts through a frozen
Codex adjudication, then freezes a Codex-reviewed relation taxonomy/mapping, generates English
factual prompts, and constructs unverified source-wrong-option distractor candidates. Translation,
screening, and cross-lingual scores remain outside this phase.

Create and inspect the deterministic 100-item pilot without an API call:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  sample --run-id triple-pilot-v1 --dry-run
```

After inspecting
`data_processed/factual_triples/triple-pilot-v1/raw_source_manifest.json`, run a
dual-model extraction. Each LLM receives the question and canonical answer,
but not the answer choices. `source_choices` are copied directly from the manifest
into each output record.

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  extract \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --models sensenova-6.7-flash-lite qwen3.7-plus \
  --max-workers 1 \
  --model-parallelism 2
```

Outputs are isolated by model under `models/<model-name>/triple_extractions.jsonl`.
`qwen3.7-plus` uses `OPENAI_BASE_URL=https://ctapi.csxdtx.com:16000/v1`.

Resume a checkpoint without repeating completed records:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  extract \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --models sensenova-6.7-flash-lite qwen3.7-plus \
  --resume \
  --retry-failed \
  --max-workers 2
```

For SenseNova, high worker concurrency can be combined with a process-wide request-start
limit. For example, `SENSENOVA_MIN_REQUEST_INTERVAL_SECONDS=1.1` spaces all worker and
retry requests at roughly 54 starts per minute while keeping multiple responses in flight.

Build the global relation inventory from locally valid extracted triples:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  inventory \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --model qwen3.7-plus
```

Evaluate both model labels against the frozen 100-row Codex reference. The 300-row
experiment is eligible only when every required model has at most 3% label error:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py evaluate \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --reference data_processed/factual_triples/triple-pilot-v1/calibration/codex_reference_labels_v1.jsonl \
  --models sensenova-6.7-flash-lite qwen3.7-plus \
  --input-name triple_extractions_v6.jsonl \
  --threshold 0.03
```

Create a disjoint balanced 300-row manifest, then run the same two model labelers:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_300.json sample \
  --run-id triple-300-v1 \
  --output-dir data_processed/factual_triples/triple-300-v1 \
  --exclude-manifest data_processed/factual_triples/triple-pilot-v1/raw_source_manifest.json
```

Build the final agreement/disagreement bundle for Codex review:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_300.json summarize \
  --run-dir data_processed/factual_triples/triple-300-v1 \
  --models sensenova-6.7-flash-lite qwen3.7-plus
```

After Codex or `qwen3.7-plus` proposes and a reviewer freezes the taxonomy and
explicit mapping, apply them deterministically:

```bash
conda run -n clp python scripts/build_raw_factual_pitfalls.py \
  --config configs/raw_factual_pitfalls_mvp.json \
  normalize \
  --run-dir data_processed/factual_triples/triple-pilot-v1 \
  --model qwen3.7-plus \
  --taxonomy configs/relation_taxonomy_v1.json \
  --mapping configs/relation_mapping_v1.json
```

For the completed full dual-model run, first materialize the review queue without modifying
either model checkpoint. Exit status 2 is expected while adjudication is incomplete:

```bash
python scripts/build_canonical_factual_dataset.py \
  --run-dir data_processed/factual_triples/triple-full-v1 \
  --output-dir canonical-v2
```

Review `canonical_review_queue.jsonl` with
`configs/canonical_triple_adjudication_prompt_v1.md`, save one decision per queued row, and
then run the gated downstream pipeline:

```bash
python scripts/build_canonical_factual_dataset.py \
  --run-dir data_processed/factual_triples/triple-full-v1 \
  --output-dir canonical-v2 \
  --adjudication data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_adjudication_codex_v1.jsonl
```

The command writes `canonical-v2/` below the run directory. Important artifacts are
`canonical_review_queue.jsonl`, `canonical_adjudication_codex_v1.jsonl`,
`canonical_decisions.jsonl`, `canonical_triples.jsonl`, `relation_inventory.json`,
`relation_taxonomy_v1.json`, `relation_mapping_v1.json`, `relation_review_queue_v1.json`,
`triples_normalized.jsonl`, `factual_prompts.jsonl`, and `distractor_candidates.jsonl`.
Dual-model answer agreements enter directly. A single-model extraction or material answer
conflict can enter only after an explicit `codex_decision=accept` selects one eligible model;
missing review remains `pending_review` and cannot reach relation normalization. Unresolved
relation signatures retain a null `relation_normalized`; they are not forced into the taxonomy
and no longer block version-2 factual-prompt generation. Prompt readiness is based on accepted
canonical evidence and option-free prompt validation. Distractors are copied from original wrong
choices for every prompt-ready canonical triple and remain
`distractor_verified=false` until a later factual/type verification gate.

### Public-benchmark provisional bridge without GPU execution

The public Qwen extraction is a single-model candidate pool, not dual-model canonical gold.
Build deterministic review-only artifacts without calling a model endpoint:

```bash
/usr/bin/python3 scripts/prepare_public_benchmark_provisional.py \
  --input data_processed/factual_triples/public-benchmarks-full-v1/models/qwen3.7-plus/triple_extractions.jsonl \
  --output-dir data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1 \
  --comparison-canonical data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl
```

The current producer is `single-model-provisional-adapter-v2`. It admits only completed,
extracted, validation-clean rows: 9,059 provisional triples are grouped into 8,969
`base_fact_id` values. It writes a relation inventory, the full review queue,
`behavior_input_bundle.jsonl`, and three review samples while keeping every row
`canonical_status=pending_review`, `human_gold=false`, and `probe_relation_id=null`:

- `review_sample.jsonl` is a 400-item coverage-oriented sample across 98 strata. It is useful for
  finding heterogeneous failure modes, but it has no sampling weights and MUST NOT be used to
  estimate whole-pool rates.
- `review_sample_overall_random.jsonl` is a separate 400-item uniform random sample from all
  8,969 base facts. It supports whole-pool outcome estimates only if the seed was fixed before
  outcomes, every selected item receives the same review protocol, and nonresponse is reported
  and handled.
- `review_sample_prompt_risk.jsonl` targets 200 items from a 376-fact prompt-risk frame. It is a
  targeted diagnostic and MUST NOT be extrapolated to the whole pool.

A review sample is not a formal cohort by implication. A formal freeze universe must be
materialized as its own immutable bundle with an explicit universe ID, exact row inventory and
SHA-256 binding. The universe may be a predeclared subset of the 8,969-fact pool, but every item
considered for that universe must reach a final `accept` or `reject` disposition after any
requested revision is applied and re-reviewed. `missing`, `defer`, and unapplied `revise` outcomes
block freeze. If the overall-random 400 is deliberately reused as the cohort universe, review must
cover 400/400 under one protocol and the resulting accepted subset must still undergo duplicate,
near-duplicate, and historical-exposure closure against the complete 8,969-fact pool. Completing
the 400 does not promote or freeze the other 8,569 facts. Conversely, freezing the complete
8,969-fact pool requires terminal review coverage for all 8,969 facts; estimated rates from the
random sample cannot substitute for item-level admission.

The existing overall-random 400 was selected before its review outcomes, so it remains valid for
its stated estimation role. It was not, however, declared as a formal cohort before the 20-item
pilot was reviewed. Reusing it now as the formal cohort would therefore be a post-hoc exploratory
choice: record that limitation and re-review the pilot under the locked final protocol. For a
clean confirmatory formal cohort, prefer a newly seeded, predeclared subset (kept distinct from the
existing 400) or explicitly choose the complete 8,969-fact pool.

Probe-relation rules are versioned in
`configs/path_not_token_public_benchmark_relation_candidates_v2.json` and never gate prompt
generation. In particular, `term for definition` is no longer assumed to be
`description -> term`: it is emitted with `probe_relation_candidate_direction=unresolved` and
requires direction/semantic review.

Create a byte-bound review scope, export self-contained review items, and apply decisions into a
non-formal staging artifact with `scripts/review_public_benchmark_bundle.py`:

```bash
/usr/bin/python3 scripts/review_public_benchmark_bundle.py scope \
  --source-dir data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1 \
  --review-sample PATH_TO_SELECTED_REVIEW_SAMPLE.jsonl \
  --output-dir REVIEW_SCOPE_DIR

/usr/bin/python3 scripts/review_public_benchmark_bundle.py export \
  --scope-manifest REVIEW_SCOPE_DIR/review_scope_manifest.json \
  --output-dir REVIEW_SCOPE_DIR

/usr/bin/python3 scripts/review_public_benchmark_bundle.py apply \
  --scope-manifest REVIEW_SCOPE_DIR/review_scope_manifest.json \
  --export-manifest REVIEW_SCOPE_DIR/review_export_manifest.json \
  --decisions REVIEW_DECISIONS.jsonl \
  --output-dir REVIEW_STAGING_DIR \
  --allow-partial
```

The scope binds the exact bundle, queue, cluster, and triple bytes. Exported items expand every
duplicate cluster member and bind member hashes. Decisions are `accept`, request-only `revise`,
`reject`, or `defer`; missing partial-review decisions remain missing and are never converted to
reject. Codex decisions MUST use `reviewer_type=codex_proxy` and `human_gold=false`. `apply`
does not rewrite source artifacts, apply revised fact fields, promote canonical status, or freeze
a split. The current 20-item pilot over the 400-item overall-random scope produced 16 `accept`,
2 `defer`, 1 `reject`, and 1 `revise`; the other 380 entries remain `missing`. This is review
staging only, not a completed independent review or an end-to-end mining-to-repair loop. The
repository now has a fail-closed cohort-declaration and evidence-preflight utility, but still has
no producer that applies approved revisions, rebuilds duplicate components, materializes a
reviewed/frozen bundle, recomputes its split, or emits the two freeze manifests. The preflight and
existing formal-admission code are validators, not that producer.

The current v2 command can create a **diagnostic declaration** to inspect blockers without
emitting formal or freeze artifacts. It is not yet a finalizable real-universe declaration:
`historical_exposure_contract_status=not_predeclared` is mandatory until the user supplies an
authoritative, predeclared historical-exposure inventory contract and the schema is extended to
bind it.

```bash
/usr/bin/python3 scripts/finalize_public_benchmark_bundle.py declare-universe \
  --source-dir data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1 \
  --comparison-canonical data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl \
  --comparison-dataset-id canonical-v2 \
  --mode review-scope \
  --review-scope-manifest PATH_TO_PREDECLARED_SCOPE/review_scope_manifest.json \
  --universe-label DIAGNOSTIC_COHORT_LABEL \
  --selection-policy-id PREDECLARED_POLICY_ID \
  --near-question-threshold 0.80 \
  --near-fact-threshold 0.80 \
  --near-max-bucket-neighbors 24 \
  --near-max-examples 20 \
  --output-dir DIAGNOSTIC_COHORT_UNIVERSE_DIR

/usr/bin/python3 scripts/finalize_public_benchmark_bundle.py preflight \
  --universe-manifest DIAGNOSTIC_COHORT_UNIVERSE_DIR/formal_cohort_universe_manifest.json \
  --output FINALIZER_PREFLIGHT.json
```

`declare-universe` refuses to overwrite an existing declaration. `preflight` reports missing or
stale review, revision/re-review, near-duplicate, historical-exposure, re-clustering, and split
evidence and intentionally exits with status 2 while the reviewed-bundle finalizer is not
implemented. It deterministically replays the existing review `apply` chain and lexical audit from
their bound inputs rather than trusting staging/summary claims; the lexical policy therefore binds
`max_examples` as well as its thresholds and blocking parameters. It always records
`freeze_outputs_emitted=false` and refuses to overwrite any existing output.

The required `--comparison-canonical` input is a separate historical canonical-v2 comparison
corpus, not the provisional behavior bundle under another name. `--comparison-dataset-id
canonical-v2` selects a repository-pinned, path-independent trust contract: SHA-256, byte/record
counts, format, canonical policy, ordered record IDs, and ordered row hashes must all match. Every
row must contain non-empty
`source_id`, `candidate_id`, `source_dataset`, `source_question`, `canonical_fact`, `answer`,
`source_answer`, and `canonical_policy_version`, with `canonical_decision=keep`; the complete file
must use one policy version. The universe declaration records its dataset role, policy version,
ordered record-ID digest, ordered row digest, byte count, and SHA-256. Declaration and reload both
reject a comparison whose resolved path or bytes/SHA equal any of the four provisional source
artifacts.

The four `--near-*` values above are not late audit tuning knobs: their values, together with the
audit/pair schemas, normalization and blocking versions, MinHash layout, deterministic/no-network
flags, and pair-review contract, are included in the universe ID. A later lexical-audit summary
must match that complete predeclared policy exactly before deterministic replay is considered.
The preflight also always emits `historical_exposure_authority_missing` for the current v2 schema,
even if a runtime JSON file labels itself a complete registry; such a file is locally checkable
but cannot retroactively supply the missing authority boundary.

Revision lineage v2 is also staging evidence, not a finalized behavior row. Its
`revised_record_role=preflight_reconstruction_not_final_behavior_row` and
`derived_fields_recomputed=false` declarations are mandatory, and
`source_review_staging_row_sha256` must bind the exact upstream `revise` row. Editor identity,
method, and timestamp and the re-reviewer's identity, method, and timestamp are
required non-empty provenance; proxy editors/reviewers cannot claim human status. Before any later producer can emit
a reviewed bundle, it must re-derive and revalidate prompt text, distractors, normalized relation
and probe mapping, duplicate/cluster membership, and split assignment from the accepted semantic
payload; copying this preflight reconstruction directly into a formal bundle is invalid. Final
rereview decision v2 uses an exact field set and cannot self-declare an `evidence_tier` or freeze
authorization.

The diagnostic v2 validator records editor and re-reviewer identities but does not yet enforce
that they are distinct or validate timestamp ordering. If independent re-review is part of the
formal protocol, the future finalizer schema must enforce both conditions before freeze.

`overlap_audit.json` records exact normalized overlap with canonical-v2. The separate lexical
near-duplicate audit v2 emitted 2,341 representative candidate pairs, including 281 emitted
cross-split pairs. These counts are not an exhaustive record-pair census: bounded lexical
blocking can miss paraphrases, and every candidate still requires semantic adjudication.
Consequently `semantic_review_complete=false` and the current provisional split MUST NOT be
frozen.

```bash
/usr/bin/python3 scripts/audit_public_benchmark_near_duplicates.py \
  --input-bundle data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/behavior_input_bundle.jsonl \
  --comparison-canonical data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl \
  --output-dir data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/lexical-near-duplicate-audit-v2
```

Prepare the same bundle for later PATH_not_token work while retaining all provisional rows in a
separate review-only artifact:

```bash
/usr/bin/python3 scripts/prepare_path_not_token_bundle.py \
  --input-bundle data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/behavior_input_bundle.jsonl \
  --output-dir data_processed/path_not_token/public-benchmark-qwen-provisional-v1 \
  --allow-provisional
```

This command does not promote any fact or run a tokenizer/model. The current export contains
8,969 `review_only_base_facts.jsonl` rows and 0 formal `base_facts.jsonl` rows. Formal admission
requires `canonical_status=frozen`, a reviewed/frozen bundle, an accepted evidence tier,
completed semantic and near-duplicate review, source prompt readiness, a frozen valid split, and
external review-freeze and split-freeze manifests bound to the exact input-bundle SHA. Those
external manifests do not yet exist for this provisional pool. Answer tokenization remains
pending and mechanism analysis remains `not_run`.

### Historical bounded post-closure cohort

The former authoritative non-frozen chain is under
`data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/preperturbation_v1`.
It consolidates 251 Codex-proxy review outcomes, selects 160 facts over four probe relations,
applies seven explicitly re-reviewed revisions, and computes a cohort-seeded lexical closure
against all 8,969 public-pool facts. All 73 reachable candidate edges were adjudicated before the
provisional split was recomputed. Resolution retained 160 facts in 159 leakage components and
achieved 24 development / 8 validation / 8 sealed facts for each relation.

The authoritative downstream artifacts are:

- `duplicate_closure_adjudications_v1/duplicate_closure_adjudications.jsonl`
- `duplicate_closure_resolution_v1/resolution_manifest.json`
- `postclosure_preperturbation_v2/postclosure_preperturbation_behavior_input_bundle.jsonl`
- `data_processed/path_not_token/public-benchmark-qwen-postclosure-provisional-v2/review_only_base_facts.jsonl`

Earlier `postclosure_preperturbation_v1` / `public-benchmark-qwen-postclosure-provisional-v1`
outputs predate the row-level split-policy correction and are non-authoritative.

The post-closure bundle has 160 rows and 320 newly generated same-relation, same-final-split
distractor candidates. The PATH adapter accepts both `strict_factual_completion` and the
producer's `relation_specific_open_completion` as factual-completion prompts; unknown prompt tiers
still fail closed. The PATH result has 160 review-only rows and zero formal rows.

These artifacts were the last safe offline products at that historical stage. They were superseded
by the reviewed/frozen `static-g0a-160-dual-v8-20260916` chain; do not use the provisional PATH
adapter rows as current experiment state. See
`data_processed/factual_triples/public-benchmarks-full-v1/provisional/qwen3.7-plus-v1/preperturbation_v1/PRE_PERTURBATION_STATUS.md`
for the historical boundaries.

### Current 160-fact MCQ-to-PNT research contract

As of 2026-09-19, G0A/P0/G0B/G1, the Development natural EN/ZH baseline, and the reproducible
960-row MCQ decision-position attribution are complete for
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`. The canonical attribution is
`qwen3-8b-pnt-development-v1/mcq-decision-attribution-v2`; its independent repeat has identical
metrics, activation-index, and BF16 `resid_pre` SHA-256 values. This is MCQ-choice mechanism
evidence only. Answer-text recall/translation vectors, intervention, repair, Validation, and Sealed
remain unopened pending a frozen option-free/PNT adaptation contract.

The role of each measurement is fixed as follows:

```text
source provenance + deterministic rules
  -> evidence-backed canonical/distractor fact verification
  -> independent strong-model semantic review of rendered bilingual variants
  -> formal review/freeze
  -> Development 96 exact-HF Original/Neutral/Targeted MCQ weakness mining
  -> strict induced defects plus matched resistant controls
  -> natural EN/ZH open-completion baseline and claim stratification
  -> exact-HF hidden-state/Logit-Lens attribution and vector intervention
  -> frozen repair rule
  -> Validation 32 fixed confirmation
  -> Sealed 32 one-time final evaluation
```

MCQ perturbation is the weakness-mining entry point. Natural open completion is the unperturbed
factual-recall baseline: it determines whether a mined case supports a natural cross-lingual recall
gap or only an `MCQ-conditioned` vulnerability. PATH_not_token is the same-checkpoint exact-HF
mechanism-analysis and repair stage; API/Ollama panel outputs cannot substitute for it. Target-model
behavior may assign behavioral labels such as strict, resistant, neutral-unstable, or off-target,
but it cannot retroactively approve a fact, translation, distractor, semantic review, or primary
variant.

The first three gates also have different authority. Deterministic source checks prove lineage and
rule compliance, not truth. Factual verification must bind evidence showing that the canonical
answer is supported and the distractor is not a valid answer under the same scope. A separately
identified strong semantic reviewer then checks answer uniqueness, relation preservation,
translation equivalence, Neutral neutrality, and Targeted direction on the final rendered inputs;
it does not see target-model outputs or choose the variant most likely to flip. Proxy judgments
remain `human_gold=false`.

The completed historical `diagnostic-100-five-model-v7b` run remains useful only as a diagnostic
proxy: 73/100 bases had complete five-model coverage, 2,190/2,190 Simulation calls completed, and
14 model-specific strict signals were reported. Its formal gate was false because translation
coverage and currency-cost audit were incomplete and 41 Neutral instabilities were present. It is
not the target exact-HF mechanism cohort, did not run the current 160-fact Validation or Sealed
partitions, and every such diagnostic signal remains `pnt_eligible=false`.

After an independent adjudication produces a reviewed bundle, factual-perturbation preparation
can consume it explicitly:

```bash
/usr/bin/python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_zh_mvp_v1.json \
  prepare --run-id REVIEWED_RUN_ID --limit 100 \
  --input-bundle PATH_TO_REVIEWED_BEHAVIOR_INPUT_BUNDLE.jsonl
```

Formal preparation applies the same review/evidence/split contract and defaults
`inputs.allowed_splits` to `development`, so validation and sealed records are not sampled by an
ordinary prepare. Pending or incompletely reviewed rows are written to
`input_review_queue.jsonl`; rejected, prompt-incomplete, or disallowed-split rows are written to
`input_exclusions.jsonl`. Explicit bundles must use the v1 schema, consistent unique
`base_fact_id` values, and complete/consistent split-group metadata; a schema-less explicit JSONL
cannot fall back to legacy admission. A row-level claim of `frozen` is insufficient: the gate
validates both external manifests, their completion/integrity fields, and their binding to the
exact bundle before formal behavior or PATH admission. Once freeze manifests are supplied, every
row in the declared source universe must have terminal `frozen`/`rejected` status, and the bound
scope and decisions must cover every ID exactly once; `pending_review` is a fatal incomplete-freeze
error rather than a skippable row. Omitting `--input-bundle` preserves the legacy canonical-v2
routing derived from `inputs.canonical_triples`.

Formal review decisions and near-duplicate evidence use fail-closed v2 bindings. The gate requires
an exact canonical `evidence_tier`, requires row-level `human_gold` and `evidence_tier` to match
the bound decision, and binds every near-duplicate decision to the candidate row plus both endpoint
IDs and row hashes. The PATH adapter rejects non-boolean `human_gold` instead of coercing strings.
Candidate holdout freezes likewise bind the candidate JSONL bytes, schema, count, ordered ID
digest, and runtime fingerprint before any model router is initialized.

For an eventual formal explicit bundle, configure the two evidence files under `inputs`:

```json
{
  "review_freeze_manifest": "PATH_TO_REVIEW_FREEZE_MANIFEST.json",
  "split_freeze_manifest": "PATH_TO_SPLIT_FREEZE_MANIFEST.json"
}
```

The PATH adapter takes the same evidence through `--review-freeze-manifest` and
`--split-freeze-manifest`. The current provisional pool intentionally has neither file.

For the later PATH_not_token experiment, freeze one exact Hugging Face model checkpoint and one
tokenizer revision before generating model-specific answer-token metadata. The same unquantized or
explicitly declared quantized weights, tokenizer, prompt formatting, and decoding/scoring contract
must be used for Development MCQ mining, the natural open-completion baseline, hidden-state and
Logit Lens collection, recall task vector, translation-minus-recall difference vector,
`resid_pre` intervention, repair evaluation, and regression controls. API or Ollama
results—including an Ollama Q4 build of a nominally similar model—remain behavioral diagnostics
with `pnt_eligible=false` and cannot be substituted for this same-checkpoint white-box chain.

Security note: the external reference checkout
`/Users/xiarongzhi/school task/paths_not_taken/demo_load_datasets_model.py:337` contains a
hard-coded Hugging Face credential. Its value must never be copied into this repository or an
experiment artifact; revoke/rotate it before reuse and load future credentials only from an
environment variable or secret manager. The external checkout is reference material and is not
modified by this workflow.

All repository commands for producer-v2 generation, sampling, scope/export/apply staging,
lexical auditing, cohort-specific review, bounded closure, post-closure materialization, and PATH
  review-only export were offline. No GPU, Ollama, HF checkpoint, or external experiment/provider
endpoint was invoked; all semantic decisions in this phase are explicitly recorded as Codex proxy
judgments rather than human gold.

Resume is rejected when the checkpoint uses another extraction prompt version or
model. `--resume --retry-failed` retries only transport or validation failures while
preserving retry history. `source_answer` remains exact provenance; the extracted
`answer` may be a concise substring grounded in a sentence-like source answer. Before
freezing triples, inspect atomicity, subject selection, relation direction, answer
grounding, fact validity, and the Codex disagreement adjudication.

### Historical local Ollama diagnostic rerun

`configs/factual_perturbation_zh_ollama_v6.json` uses the loopback Ollama
OpenAI-compatible endpoint without an API credential and requests JSON-mode responses. Unauthenticated profiles are
accepted only for `localhost`/loopback URLs. The configuration runs one request at a
time and groups requests by model to avoid repeatedly swapping local model weights.

This command documents an already completed compatibility/calibration workflow; it is not the
current 160-fact protocol and does not authorize a new holdout, Validation, Sealed, or PATH run.
It reused an already frozen legacy candidate set without replaying translation, generation, or
review calls, then ran only the diagnostic Simulation panel:

```bash
/usr/bin/python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_zh_ollama_v6.json \
  --env-file /dev/null \
  prepare-simulation-rerun \
  --run-id calibration-15-frozen-ollama-v6 \
  --source-run-id calibration-20-streamlined-l2-v5c

/usr/bin/python3 scripts/run_factual_perturbation.py \
  --config configs/factual_perturbation_zh_ollama_v6.json \
  --env-file /dev/null \
  run --run-id calibration-15-frozen-ollama-v6
```

The new run records the source manifest hash and hashes every reused checkpoint. It
starts with no copied `simulation_results.jsonl`, so its scores contain only the two
configured local models. Its behavior signals remain diagnostic and `pnt_eligible=false`.

## Citation

If you use this code or dataset, please cite the ACL 2025 version:

```bibtex
@inproceedings{xu-etal-2025-cross,
    title = "Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models",
    author = "Xu, Zixiang  and Wang, Yanbo  and Huang, Yue  and Chen, Xiuying  and Zhao, Jieyu  and Jiang, Meng  and Zhang, Xiangliang",
    editor = "Che, Wanxiang  and Nabende, Joyce  and Shutova, Ekaterina  and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul, year = "2025", address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.404/",
    doi = "10.18653/v1/2025.acl-long.404",
    pages = "8254--8284", ISBN = "979-8-89176-251-0"
}
```

The arXiv preprint can also be cited:

```bibtex
@misc{xu2025crosslingualpitfallsautomaticprobing,
      title={Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models}, 
      author={Zixiang Xu and Yanbo Wang and Yue Huang and Xiuying Chen and Jieyu Zhao and Meng Jiang and Xiangliang Zhang},
      year={2025},
      eprint={2505.18673},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18673}, 
}
```
