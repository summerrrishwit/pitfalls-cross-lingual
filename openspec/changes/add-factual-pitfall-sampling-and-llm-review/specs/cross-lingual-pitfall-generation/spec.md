## ADDED Requirements

### Requirement: Configurable answer-preserving perturbation generation
The system SHALL generate perturbation candidates only from factual-audit accepts. Each generation request SHALL use an explicitly configured perturbation-generator model, the accepted raw English QA, and one non-canonical English choice. It SHALL record the generator model, prompt version, chosen incorrect option, raw response, insertion position, and generated English candidate.

#### Scenario: Perturbation candidate is generated
- **WHEN** an accepted English factual QA is paired with a non-canonical choice
- **THEN** the system produces a provenance record for the generated distraction and enhanced English question without modifying the canonical answer or choices

#### Scenario: Generation changes answer integrity
- **WHEN** generation validation finds that the enhanced question contradicts or changes the canonical answer
- **THEN** the system rejects the candidate with a generation-validation failure and does not translate or screen it

### Requirement: Configurable translation of generated candidates
The system SHALL translate each generation-valid English candidate into every configured target language using an explicitly configured translation model. It SHALL preserve the translated question, choices, canonical answer, model identity, prompt version, raw response, and translation validation outcome per target language.

#### Scenario: Translation succeeds for a target language
- **WHEN** a generation-valid English candidate is translated into Chinese, Japanese, or French
- **THEN** the system writes a language-specific candidate record containing the translated question, choices, answer, and translation provenance

#### Scenario: Translation output is incomplete
- **WHEN** a translation omits a required question, choice, or answer field or cannot align the canonical answer to a translated choice
- **THEN** the system records a terminal translation-validation failure and does not screen that language-specific candidate

### Requirement: Configurable cross-lingual weakness screening
The system SHALL screen every translation-valid candidate in English and its target language with each explicitly configured screen model. It SHALL separately record raw answer responses, answer-extraction outcomes, correctness, and scores, then compute `rate_ori`, `rate_trans`, and `pitfall_score = rate_ori - rate_trans` for the model list declared by the run.

#### Scenario: Candidate satisfies cross-lingual-pitfall thresholds
- **WHEN** a candidate meets the configured minimum English rate, maximum target-language rate, and minimum pitfall score
- **THEN** the system writes a retained factual-pitfall record with all generation, translation, and per-model screening provenance

#### Scenario: Candidate fails screening thresholds
- **WHEN** English performance is below the configured minimum or target-language performance is above the configured maximum
- **THEN** the system records the screening result as not retained without discarding its intermediate provenance

### Requirement: Resumable stage outputs
The system SHALL store raw manifests, factual reviews, generation records, translation records, per-model screening checkpoints, retained-pitfall JSONL, and summaries in distinct stage outputs. A rerun SHALL consume the preceding completed artifact and MUST NOT resample or overwrite terminal records without explicit user selection.

#### Scenario: Screening resumes after interruption
- **WHEN** a screening run is restarted with an existing stage checkpoint
- **THEN** the system reuses completed per-model language results and evaluates only missing work
