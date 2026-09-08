## ADDED Requirements

### Requirement: Agreement plus Codex-adjudicated canonical triples
The system SHALL preserve both full model checkpoints and determine canonical triples in a
separate output directory. Equivalent dual-model answers SHALL enter directly. Every
single-model extraction and material dual-model answer conflict SHALL enter a frozen Codex
review queue and remain `pending_review` until an explicit adjudication accepts or rejects it.
Model failures where neither annotation is locally valid SHALL receive explicit exclusions.
Relation normalization MUST NOT start while any queued row lacks adjudication.

#### Scenario: Both models support the same answer
- **WHEN** SenseNova and Qwen both extract a row and their normalized answers agree
- **THEN** the canonical record preserves source provenance, source choices, both annotations, field conflicts, and the configured preferred field model

#### Scenario: Only one model extracts the row
- **WHEN** exactly one model has a valid extraction
- **THEN** the row receives `canonical_decision=pending_review` and preserves the proposed triple, the other model's exclusion, and all source provenance

#### Scenario: Codex accepts a queued row
- **WHEN** a version-matched adjudication accepts a single-model extraction or answer conflict and selects an eligible model
- **THEN** the selected unchanged annotation enters the canonical pool with the Codex reason and review provenance

#### Scenario: Adjudication coverage is incomplete
- **WHEN** one or more queued source IDs have no valid Codex decision
- **THEN** the system writes the complete review queue but does not build the relation inventory or any later artifact

### Requirement: Codex-reviewed taxonomy decisions
The system SHALL build one inventory from canonical triples and produce an explicit mapping
decision for every signature. Mapped signatures SHALL reference an existing versioned taxonomy
relation. Reviewed ambiguous and out-of-taxonomy signatures SHALL retain null normalized fields
and MUST NOT be coerced into a broad label.

#### Scenario: Relation remains unresolved after review
- **WHEN** no conservative unique taxonomy mapping matches its semantics and direction
- **THEN** the mapping records `codex_reviewed_unresolved`, a null `relation_normalized`, and an unresolved reason

### Requirement: Deterministic factual prompt generation
The system SHALL generate prompts only for mapped canonical triples. It SHALL first attempt to
remove a sentence-final answer from `canonical_fact` to create a strict completion stem. If this
is impossible, it SHALL emit an option-free source-question open-answer fallback. Every prompt
SHALL record its source, quality tier, expected answer, template ID, and version.

#### Scenario: Canonical fact ends in the answer
- **WHEN** the answer is the sentence-final completion of `canonical_fact`
- **THEN** the generated prompt has `prompt_quality_tier=strict_factual_completion`

#### Scenario: Canonical fact cannot be converted safely
- **WHEN** the answer is not sentence-final
- **THEN** the generated prompt uses the original question without choices and has `prompt_quality_tier=open_answer_fallback`

### Requirement: Traceable source-option distractor candidates
For every prompt-ready canonical triple, the system SHALL copy the original wrong choices in
their original order, create one expanded distractor record per candidate, and deterministically
rank candidates. It SHALL exclude the canonical source answer and normalized answer. It MUST
record candidates as unverified until a later type and factual-falsity gate is run.

#### Scenario: Source wrong options are constructed
- **WHEN** a mapped canonical triple has an English factual prompt
- **THEN** every distractor candidate occurs in `source_choices`, differs from both answer forms, and records `distractor_verified=false`

### Requirement: Offline deterministic validation
The postprocessor SHALL require no model SDK or network access, SHALL never mutate model
checkpoints, and SHALL validate source-choice preservation, record-ID conservation, mapping
coverage, prompt/mapping consistency, and distractor provenance before reporting success.

#### Scenario: Same checkpoints are processed twice
- **WHEN** the postprocessor runs twice over identical model JSONL inputs
- **THEN** canonical, mapping, normalized, prompt, and distractor JSONL contents are identical
