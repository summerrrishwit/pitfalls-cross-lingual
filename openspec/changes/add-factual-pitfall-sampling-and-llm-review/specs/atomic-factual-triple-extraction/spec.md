## ADDED Requirements

### Requirement: Independent dual-model atomic-triple extraction
The system SHALL support independent submissions of every raw-manifest candidate to `sensenova-6.7-flash-lite` and `qwen3.7-plus` using the same versioned extraction contract. The prompt SHALL contain the raw English question, canonical English answer, source ID, dataset, and optional source subset. It MUST NOT contain source choices, target-language content, perturbations, rates, screening outputs, normalized relation labels, or a request for `prompt_en`.

#### Scenario: Extraction prompt is option-independent
- **WHEN** a raw-manifest candidate is submitted for extraction
- **THEN** its choices remain in program-owned provenance but are absent from the LLM prompt

#### Scenario: Extraction model identity is recorded and isolated
- **WHEN** an extraction request is issued
- **THEN** the terminal record identifies the selected model and prompt version, and its checkpoint is stored only below that model's slug directory

### Requirement: Structured atomic-triple result
The extraction response SHALL contain `extraction_status=extracted|not_extractable`, an exclusion reason when not extractable, and nullable `subject`, `subject_type`, `relation_raw`, `answer`, `answer_type`, `canonical_fact`, and confidence fields. An extracted result SHALL orient the triple from subject to an answer copied from or explicitly entailed by the canonical source answer. `relation_raw` SHALL be self-contained rather than vague question scaffolding, and explanatory, procedural, recommendation, or multi-clause facts SHALL be `answer_not_suitable`.

#### Scenario: Atomic fact is extracted
- **WHEN** the question-answer pair expresses one directly retrievable stable fact without essential choices, contextual inference, calculation, or multi-step reasoning
- **THEN** all triple fields are non-empty, `answer` is a concise grounded normalization of `source_answer`, and `subject` does not repeat `answer`

#### Scenario: Single definition is extractable
- **WHEN** one concise description directly defines or classifies one concept without multi-step reasoning or independent biographical clues
- **THEN** the row MAY be extracted with a stable definition or classification relation

#### Scenario: Description only identifies the answer
- **WHEN** the only possible subject is a descriptive clue whose purpose is to identify the canonical answer
- **THEN** the item is `not_extractable` with `clue_solving`

#### Scenario: Item is not extractable
- **WHEN** the item is option-dependent, negative, ambiguous, time-sensitive, malformed, subjective, contextual, computational, or otherwise not one atomic fact
- **THEN** `extraction_status` is `not_extractable`, one allowed exclusion reason is present, and every triple field is null

### Requirement: Source provenance is explicit
Every terminal extraction record SHALL copy `source_id`, `source_dataset`, `source_question`, `source_choices`, and `source_answer` from its raw manifest snapshot. The model response MUST NOT overwrite these fields.

#### Scenario: Source choices are preserved
- **WHEN** an extraction record is written
- **THEN** `source_choices` exactly equals the original ordered choices in the raw source snapshot

### Requirement: Local validation and resumable failures
The system SHALL locally validate response JSON, status/field consistency, answer grounding, confidence range, and allowed exclusion reasons. Transport failures SHALL produce `extraction_failed`; invalid model output SHALL produce `validation_failed`; neither SHALL be treated as an extracted triple. Resume SHALL reject mixed extraction prompt versions or model IDs.

#### Scenario: Model invents an answer
- **WHEN** an extracted response is neither copied from nor textually grounded in the canonical source answer
- **THEN** the record is `validation_failed` with `answer_not_grounded_in_source`

### Requirement: Codex calibration gate and resumable model runs
The system SHALL compare each model's binary extraction label with a frozen 100-row Codex reference. It SHALL block the 300-row expansion unless every required model has `label_error_rate <= 0.03`. Concurrent workers, resume, and retry-failed SHALL operate independently per model.

#### Scenario: Calibration passes
- **WHEN** both required models have at most three binary label disagreements over 100 completed reference rows
- **THEN** the 300-row experiment is eligible to run

#### Scenario: Calibration fails
- **WHEN** either model exceeds the error threshold or has incomplete or invalid rows
- **THEN** expansion is blocked and all disagreement records are persisted for prompt revision
