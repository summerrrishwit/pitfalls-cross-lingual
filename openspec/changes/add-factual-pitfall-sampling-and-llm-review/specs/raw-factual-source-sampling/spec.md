## ADDED Requirements

### Requirement: Immutable raw English source loading
The system SHALL load raw English QA records from configured immutable files under `data/source/`, including the original five sources and pinned public-benchmark imports. It SHALL validate that every candidate has a non-empty English question, a canonical English answer, and source metadata, without modifying any source file. Multiple-choice records SHALL also have a non-empty choice list containing the canonical answer. Choice-free records SHALL be accepted only when explicitly marked `source_format="open_qa"`, and their choices SHALL be absent or empty.

#### Scenario: Valid raw source record is eligible for sampling
- **WHEN** a configured multiple-choice record has complete English question, choices, answer, and source fields, or an explicitly marked open-QA record has complete English question, answer, and source fields
- **THEN** the system includes it in the raw-source eligible pool with its original index and source path

#### Scenario: Raw source record is incomplete
- **WHEN** a record is missing a required English field, an MCQ record has empty choices, or an open-QA record supplies non-empty choices
- **THEN** the system excludes it and records the exclusion reason in the raw-source manifest

### Requirement: Raw sampling precedes every model request
The system SHALL persist the raw-source manifest before issuing an atomic-triple extraction or later relation-normalization model request. Model outcomes, retries, and model configuration MUST NOT alter the selected raw original indices for that manifest.

#### Scenario: Triple extraction begins from a raw manifest
- **WHEN** an atomic-triple extraction run is started
- **THEN** the system extracts only candidate IDs and English snapshots listed in a completed raw-source manifest

#### Scenario: A later stage fails
- **WHEN** a triple extraction or normalization request reaches a terminal failure state
- **THEN** the selected raw candidate remains in the manifest and the failure is recorded without replacement sampling

### Requirement: Reproducible pilot and full-population manifests
The system SHALL deterministically sample a configurable pilot with balanced source-dataset quotas and optional-subject stratification inside each source. Source shortfalls SHALL be redistributed deterministically. After the extraction contract is accepted, a separate full configuration SHALL include every eligible record. Both modes SHALL deduplicate normalized English questions across the complete source pool and any configured immutable baseline sources, and record allocations, shortfalls, duplicate exclusions, and selected original indices per source file.

#### Scenario: Same source snapshots and seed reproduce pilot selection
- **WHEN** the pilot manifest builder runs twice with identical source-file fingerprints, configuration, and seed
- **THEN** both manifests contain the same selected source paths and original indices in the same per-stratum order

#### Scenario: Create the full run after extraction calibration
- **WHEN** the triple-extraction prompt and local validator are accepted
- **THEN** the system creates a separately identified full-population manifest without modifying the 600-item pilot manifest or its v1 review checkpoint

#### Scenario: Supplemental import excludes existing raw questions
- **WHEN** an existing raw-source collection is configured as a deduplication baseline
- **THEN** normalized English questions already present in that baseline are excluded from the supplemental manifest with baseline dataset and original-index provenance

### Requirement: Raw-source manifest provenance
The system SHALL write a machine-readable manifest containing run ID, source paths and fingerprints, source record counts, raw QA schema criteria, random seed, requested and actual sample counts, stratum allocations, selected original indices, duplicate exclusions, and immutable English source snapshots. The manifest MUST NOT contain translations, generated perturbations, rates, or LLM responses.

#### Scenario: Raw-source manifest is created successfully
- **WHEN** raw-source sampling completes
- **THEN** the manifest exists before any model request and contains no LLM extraction or normalization output
