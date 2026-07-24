## ADDED Requirements

### Requirement: Immutable raw English source loading
The system SHALL load raw English QA records only from `data/source/ai2_arc_easy.json`, `data/source/commonsense_qa.json`, `data/source/mmlu.json`, `data/source/sciq.json`, and `data/source/truthful_qa.json`. It SHALL validate that every candidate has a non-empty English question, a non-empty list of English choices, a canonical English answer, and source metadata, without modifying any source file.

#### Scenario: Valid raw source record is eligible for sampling
- **WHEN** a record from a configured source file has complete English question, choices, answer, and source fields
- **THEN** the system includes it in the raw-source eligible pool with its original index and source path

#### Scenario: Raw source record is incomplete
- **WHEN** a record is missing a required English field or has empty choices
- **THEN** the system excludes it and records the exclusion reason in the raw-source manifest

### Requirement: Raw sampling precedes every model request
The system SHALL persist the raw-source manifest before issuing a factual-audit, perturbation-generation, translation, answer-generation, or answer-extraction request. Model outcomes, retries, and model configuration MUST NOT alter the selected raw original indices for that manifest.

#### Scenario: Factual audit begins from a raw manifest
- **WHEN** an English factual-audit run is started
- **THEN** the system audits only candidate IDs and English snapshots listed in a completed raw-source manifest

#### Scenario: A later stage fails
- **WHEN** a factual audit, generation, translation, or screen request reaches a terminal failure state
- **THEN** the selected raw candidate remains in the manifest and the failure is recorded without replacement sampling

### Requirement: Reproducible source-stratified sampling
The system SHALL deterministically sample 600 eligible raw English records using a supplied seed and strata composed of `source` plus optional `subject`. It SHALL deduplicate normalized English questions across the complete source pool and record allocations, shortfalls, duplicate exclusions, and selected original indices per source file.

#### Scenario: Same source snapshots and seed reproduce selection
- **WHEN** the sampler runs twice with identical source-file fingerprints, configuration, and seed
- **THEN** both manifests contain the same selected source paths and original indices in the same per-stratum order

#### Scenario: A source stratum is undersized
- **WHEN** a configured source or subject stratum has fewer eligible unique records than its allocated count
- **THEN** the system selects all eligible records in that stratum and records the shortfall and deterministic redistribution in the manifest

### Requirement: Raw-source manifest provenance
The system SHALL write a machine-readable manifest containing run ID, source paths and fingerprints, source record counts, raw QA schema criteria, random seed, requested and actual sample counts, stratum allocations, selected original indices, duplicate exclusions, and immutable English source snapshots. The manifest MUST NOT contain translations, generated perturbations, rates, or LLM responses.

#### Scenario: Raw-source manifest is created successfully
- **WHEN** raw-source sampling completes
- **THEN** the manifest exists before any model request and contains no generated, translated, screened, or LLM-review output
