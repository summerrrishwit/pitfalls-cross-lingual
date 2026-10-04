## ADDED Requirements

### Requirement: MCQ perturbation is the weakness-mining entry point
The formal PATH_not_token pipeline SHALL consume only facts admitted by exact-bundle review-freeze
and split-freeze manifests. Each fact SHALL have a stable `known_id` or `base_fact_id`, aligned
English and Chinese prompts and answer aliases, a predesignated distractor, frozen
Original/Neutral/Targeted MCQ variants, prompt-form metadata, and an explicit nullable
`probe_relation_id`.

Development MCQ behavior SHALL be used to mine model-specific directed weaknesses. A strict
induced defect SHALL require a correct Original response, a correct Neutral response, and a
Targeted response selecting the predesignated distractor under the frozen rule. The mechanism
cohort SHALL also retain resistant controls that receive the same type of Targeted manipulation but
remain correct. Target-model outputs MUST NOT retroactively change fact, translation, distractor,
semantic-review, or primary-variant admission decisions.

#### Scenario: A provisional fact has an Ollama flip
- **WHEN** a provisional fact exhibits a target-distractor-specific behavior change in Ollama
- **THEN** it remains diagnostic with `pnt_eligible=false` and is not admitted to the PATH_not_token mechanism cohort

#### Scenario: A Targeted response is wrong but misses the designated distractor
- **WHEN** Original and Neutral are correct but Targeted selects a non-designated wrong answer
- **THEN** the record is reported as an off-target failure and is not a strict induced defect

### Requirement: Natural open completion is a baseline and claim-stratification measurement
Every admitted mechanism fact SHALL retain aligned natural English and Chinese open-completion
prompts. The same exact-HF target SHALL run these prompts as the unperturbed factual-recall baseline.
Natural open completion SHALL distinguish natural cross-lingual recall gaps from MCQ-conditioned
induced vulnerabilities; it MUST NOT replace the Original/Neutral/Targeted MCQ mining step.

#### Scenario: MCQ Targeted flips but natural EN and ZH remain correct
- **WHEN** a strict MCQ induced defect has correct natural English and Chinese open-completion baselines
- **THEN** it is reported as an MCQ-conditioned induced vulnerability and not as a natural factual-recall failure

#### Scenario: Natural English is correct and natural Chinese fails
- **WHEN** the aligned open-completion baseline shows a reproducible EN-correct/ZH-failing gap
- **THEN** the record may support a natural cross-lingual recall-gap claim under the frozen scoring rule

### Requirement: One exact HF model identity spans the mechanism chain
Before execution, the system SHALL freeze the Hugging Face repository ID, immutable checkpoint
revision, tokenizer revision, weight precision or quantization, chat/prompt template, hook names,
layer mapping, decoding parameters, and scoring policy. The exact same model and tokenizer identity
SHALL be used for Development MCQ behavior, natural open-completion baselines, hidden-state
collection, Logit Lens, recall task-vector and translation-minus-recall difference-vector
construction, `resid_pre` intervention, repair evaluation, and regression controls.

#### Scenario: Ollama and HF share a display model name
- **WHEN** an Ollama Q4 model and an HF model have nominally matching names
- **THEN** their outputs may be compared as behavior systems but Ollama results do not count as exact-HF white-box evidence

#### Scenario: Checkpoint or tokenizer changes mid-experiment
- **WHEN** any frozen model identity field differs from the baseline manifest
- **THEN** downstream attribution/intervention results are isolated into a new experiment and cannot be merged with the original chain

### Requirement: Diagnostic panels cannot authorize the exact-HF mechanism chain
API/Ollama panels MAY provide model-specific diagnostic behavior evidence, but they MUST NOT define
the exact-HF mechanism cohort, construct vectors, select layers or scales, or authorize Validation
or Sealed. A diagnostic signal SHALL remain `pnt_eligible=false` unless the record independently
passes the formal data gates and the same frozen exact-HF identity supplies the MCQ behavior,
natural baseline, attribution, intervention, and repair evidence.

#### Scenario: The five-model diagnostic reports a strict signal
- **WHEN** `diagnostic-100-five-model-v7b` or another API/Ollama panel reports a model-specific strict signal
- **THEN** the signal remains diagnostic proxy evidence, contributes to neither current Validation nor Sealed, and has `pnt_eligible=false`

### Requirement: Vector construction and hyperparameter selection avoid leakage
Task and difference vectors SHALL be computed only from training records. Layer, scale, prompt,
and scoring choices SHALL be selected using Development data only. Validation SHALL remain unopened
until the MCQ behavior rule, attribution contrast, vector, layer, scale, trigger, and repair rule are
frozen; it SHALL then be used only for predeclared confirmation. Sealed evaluation SHALL remain
unopened until Validation confirms the frozen protocol and SHALL be opened once for final
evaluation. Within-relation and across-relation experiments SHALL use explicit, versioned split
policies and a reviewed `probe_relation_id` where relation conditioning is claimed.

#### Scenario: A sealed fact contributes to a task vector
- **WHEN** any sealed record is included in vector construction or hyperparameter selection
- **THEN** the run is invalid and cannot support a repair claim

#### Scenario: Current provisional 160-fact data are inspected
- **WHEN** review-freeze, split-freeze, exact-HF behavior, or the Development repair rule is incomplete
- **THEN** Validation and Sealed remain unrun and every row remains `pnt_eligible=false`

### Requirement: Attribution and repair use explicit metrics and controls
The system SHALL report alias-aware generation correctness, complete-answer sequence log
probability, answer-token rank, per-layer hidden-state or Logit-Lens measurements, and intervention
outcomes. It SHALL include no-intervention and norm-matched random-vector controls and separately
evaluate task-only, difference-only, and combined interventions. A repair claim SHALL report target
improvement together with English retention, unrelated-fact regression, and validation/sealed
performance at the independent base-fact level.

#### Scenario: Intervention improves selected Chinese examples only
- **WHEN** improvement is observed without passing the predeclared controls and regression checks
- **THEN** the result is reported as exploratory intervention behavior rather than validated repair

### Requirement: HF credentials are external secrets
The runtime MUST load Hugging Face credentials only from an environment variable or secret manager
and MUST NOT store them in source, configs, manifests, logs, or artifacts. The credential found in
the external reference checkout at `demo_load_datasets_model.py:337` SHALL be treated as exposed
and revoked or rotated before reuse.

#### Scenario: A credential literal is present in model-loading code
- **WHEN** preflight detects a hard-coded Hugging Face credential
- **THEN** model execution is blocked until the credential is removed from the runtime contract and replaced through secret injection
