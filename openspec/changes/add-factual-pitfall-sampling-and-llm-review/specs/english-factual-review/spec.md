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
The audit response SHALL be valid JSON with a `decision` value of `accept`, `reject`, or `needs_review`; factual-recall suitability; factual type; answer-uniqueness, ambiguity, option-dependence, and reasoning-dependence findings; and a machine-readable reason. An `accept` decision SHALL additionally provide non-empty English subject, relation, answer, and natural English factual-recall completion prompt fields.

#### Scenario: Raw English factual QA is accepted
- **WHEN** the raw English QA has a unique, directly retrievable factual answer and can be represented as a natural subject-relation-answer completion without essential choices or multi-step reasoning
- **THEN** the validated response has `decision` equal to `accept` and contains all required English factual-recall fields

#### Scenario: Raw English QA is unsuitable
- **WHEN** the audit identifies material ambiguity, essential option dependence, subjective judgment, or essential multi-step reasoning
- **THEN** the response has `decision` equal to `reject` or `needs_review` and includes the corresponding reason and findings

### Requirement: Conservative factual-audit failure handling
The system SHALL validate every parsed factual-review response. Invalid JSON, missing accepted-item fields, contradictory acceptance findings, or terminal audit-request failures MUST produce `needs_review` or `audit_failed` records and MUST NOT enter the generation stage. Retries SHALL be limited to transient transport, rate-limit, or server failures.

#### Scenario: Factual audit response is malformed
- **WHEN** `qwen3.7-plus` returns invalid JSON or an incomplete acceptance response after allowed retries
- **THEN** the system writes a terminal `needs_review` or `audit_failed` record and excludes the candidate from generation
