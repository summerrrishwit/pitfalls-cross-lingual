## ADDED Requirements

### Requirement: Global relation inventory
The system SHALL aggregate only locally valid extracted triples into a relation inventory keyed by normalized `relation_raw`, `subject_type`, and `answer_type`. Each inventory entry SHALL contain its count and bounded representative examples with subject, answer, canonical fact, source dataset, and source ID. Raw subject/answer types are evidence and MUST NOT be treated as uncorrectable truth.

#### Scenario: Synonymous-looking relations have different type signatures
- **WHEN** two rows share relation text but have different subject or answer types
- **THEN** they remain separate inventory signatures

### Requirement: Versioned taxonomy and explicit mapping
Normalization SHALL consume a versioned taxonomy defining relation ID, definition, subject type, answer type, direction, and examples, plus an explicit mapping from observed inventory signatures to taxonomy IDs. It MUST NOT infer new labels silently during application.

#### Scenario: Signature has a valid mapping
- **WHEN** the mapping references an existing taxonomy relation with compatible direction and types
- **THEN** the normalized record receives that `relation_normalized` and taxonomy version

#### Scenario: Signature is unknown or ambiguous
- **WHEN** no valid unique mapping exists
- **THEN** normalization status is `out_of_taxonomy` or `ambiguous` and `relation_normalized` is null

### Requirement: Programmatic normalized output
The system SHALL apply a frozen mapping deterministically to extracted triples and preserve all source and extraction provenance in `triples_normalized.jsonl`. A mapped signature SHALL include normalized subject and answer types matching the chosen taxonomy relation; these fields MAY correct the weak model's raw type labels.

#### Scenario: Same artifacts are reapplied
- **WHEN** identical extraction, taxonomy, and mapping files are used twice
- **THEN** normalized fields and unresolved counts are identical
