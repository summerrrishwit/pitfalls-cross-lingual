## ADDED Requirements

### Requirement: Fixed qwen3.7-plus raw-English factual review
The system SHALL audit every raw-manifest candidate's original English QA with `FACTUAL_AUDIT_MODEL=qwen3.7-plus`. The prompt SHALL contain only the raw English question, English choices, canonical English answer, and optional English source/subject metadata. The prompt MUST NOT contain target-language content, generated perturbations, translations, rates, or screen-model outputs.

#### Scenario: Prompt contains only raw English review context
- **WHEN** a raw-manifest candidate is submitted for factual review
- **THEN** the prompt includes its raw English question, choices, canonical answer, and available source/subject metadata and excludes all downstream-stage fields

#### Scenario: Audit model identity is recorded
- **WHEN** a raw English factual-audit request is issued
- **THEN** the request and terminal review record identify `qwen3.7-plus` under the `FACTUAL_AUDIT_MODEL` role

### Requirement: Structured factual-review decision and extraction
The audit response SHALL be valid JSON with a `decision` value of `accept`, `reject`, or `needs_review`; a structured reason code; factual-recall suitability; factual type; answer-uniqueness, ambiguity, option-dependence, contextual-inference, time-sensitivity, canonical-answer-quality, and reasoning-dependence findings; and a machine-readable reason. An `accept` decision SHALL additionally provide non-empty English subject, relation, answer, and natural English factual-recall completion prompt fields. Its `answer_en` SHALL copy the canonical English answer character-for-character.

#### Scenario: Raw English factual QA is accepted
- **WHEN** the raw English QA has a unique, directly retrievable factual answer and can be represented as a natural subject-relation-answer completion without essential choices or multi-step reasoning
- **THEN** the validated response has `decision` equal to `accept` and contains all required English factual-recall fields

#### Scenario: Raw English QA is unsuitable
- **WHEN** the audit identifies material ambiguity, essential option dependence, subjective judgment, or essential multi-step reasoning
- **THEN** the response has `decision` equal to `reject` or `needs_review` and includes the corresponding reason and findings

### Requirement: Pilot-driven prompt calibration and version isolation
The system SHALL preserve the completed 600-item v1 factual-review checkpoint as calibration evidence. A revised prompt SHALL use a new version and write calibration results to a separate checkpoint. The current calibrated candidate is `raw-english-factual-audit-v4`. Resume MUST reject an audit checkpoint whose recorded prompt version differs from the active prompt version.

#### Scenario: Re-audit the pilot with a revised prompt
- **WHEN** the same pilot candidate IDs are submitted using a revised prompt version
- **THEN** the system writes results to a separately named JSONL and retains all v1 records unchanged

#### Scenario: Resume encounters an older prompt version
- **WHEN** a newer prompt audit attempts to resume a checkpoint containing v1 records
- **THEN** the system refuses to continue and instructs the operator to use a separate output checkpoint

### Requirement: Conservative factual-audit failure handling
The system SHALL validate every parsed factual-review response. Invalid JSON, missing accepted-item fields, contradictory acceptance findings, or terminal audit-request failures MUST produce `needs_review` or `audit_failed` records and MUST NOT enter the generation stage. Retries SHALL be limited to transient transport, rate-limit, or server failures.

#### Scenario: Factual audit response is malformed
- **WHEN** `qwen3.7-plus` returns invalid JSON or an incomplete acceptance response after allowed retries
- **THEN** the system writes a terminal `needs_review` or `audit_failed` record and excludes the candidate from generation
