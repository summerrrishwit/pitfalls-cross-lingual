## ADDED Requirements

### Requirement: Single-model public imports remain provisional
The system SHALL label every validation-clean fact imported from a single extraction model as
provisional and MUST NOT represent it as canonical, independently reviewed, human gold, or formal
PATH_not_token evidence. Codex-only review SHALL record `reviewer_type=codex_proxy` and
`human_gold=false`.

#### Scenario: A clean Qwen extraction is imported
- **WHEN** a completed Qwen row passes local extraction validation
- **THEN** it enters the review queue with provisional evidence and cannot enter a formal cohort

#### Scenario: A proxy-reviewed provisional row has a diagnostic behavior signal
- **WHEN** a Codex-proxy-reviewed or single-model provisional row has an API/Ollama/five-model diagnostic signal
- **THEN** the signal does not promote its canonical or split status, the row remains `pnt_eligible=false`, and Validation and Sealed remain unopened

### Requirement: Formal admission separates four evidence authorities
The system SHALL preserve distinct, versioned results for deterministic source/rule checks,
evidence-backed factual verification, strong-model semantic validation, and frozen target-model
behavior. A deterministic source/rule pass SHALL establish lineage and invariant compliance only;
it MUST NOT be treated as proof that the canonical proposition is true or that a distractor is
factually false.

Factual verification SHALL bind evidence for the canonical proposition and each retained
distractor under the same entity, relation, and temporal scope. Semantic validation SHALL inspect
the fully rendered bilingual Original/Neutral/Targeted material and SHALL explicitly decide answer
uniqueness, relation preservation, translation equivalence, Neutral neutrality, and Targeted
direction. The semantic reviewer identity, model/version where applicable, prompt/contract hash,
structured verdict, and adjudication provenance SHALL be recorded. A proxy reviewer MUST retain
`human_gold=false`.

The semantic reviewer SHALL be independent of the target exact-HF behavior output and MUST NOT use
observed flips to approve or select a variant. Target behavior SHALL run only after the preceding
artifacts are frozen and MAY derive behavioral labels only; it MUST NOT promote or rewrite fact,
translation, distractor, or semantic-review status.

#### Scenario: Source provenance passes but factual evidence is unresolved
- **WHEN** IDs, hashes, source grounding, and deterministic selection rules pass but the canonical fact or distractor falsity is ambiguous or unsupported
- **THEN** the row remains provisional and cannot pass review freeze

#### Scenario: A rendered perturbation changes after semantic validation
- **WHEN** any translation, answer alias, option order, Neutral context, or Targeted context differs from the semantically reviewed bytes
- **THEN** the prior semantic decision is stale and the variant cannot enter target-model behavior until re-reviewed and rebound

#### Scenario: Target behavior reveals a suspected data defect
- **WHEN** the frozen target output suggests that an admitted fact or perturbation may be defective
- **THEN** the original run is preserved, any re-review is blinded and versioned, and no target output silently changes the existing admission decision

### Requirement: Review sampling and formal cohort definition are separate
The system SHALL record the inferential purpose of each review sample. A review sample MUST NOT
become a formal freeze universe implicitly. The system SHALL materialize a formal cohort universe
as a separate immutable artifact containing its universe ID, selection policy, ordered
`base_fact_id` inventory, source bindings, row hashes, and total count.

The diagnostic universe SHALL accept its historical comparison corpus only through a pinned,
path-independent canonical-v2 contract that binds dataset ID/role, format, SHA-256, byte/record
counts, policy version, ordered record identities, and ordered row hashes. It SHALL also bind the
complete lexical near-duplicate policy before audit execution. Until an authoritative historical
exposure inventory contract is supplied and predeclared, the universe SHALL record
`historical_exposure_contract_status=not_predeclared` and MUST remain non-finalizable regardless of
any runtime registry's self-asserted completeness.

#### Scenario: Provisional rows are projected into a comparison-shaped file
- **WHEN** a file derived from the current provisional behavior rows supplies all required comparison fields but does not match the pinned canonical-v2 content contract
- **THEN** universe declaration fails even if the file uses a different path and SHA from every source artifact

#### Scenario: Audit policy changes after universe declaration
- **WHEN** a lexical audit is rerun with thresholds or blocking parameters different from the universe contract
- **THEN** preflight rejects the audit even if its output is internally deterministic

#### Scenario: Overall-random 400 is used only for estimation
- **WHEN** reviewers complete the overall-random sample to estimate full-pool outcome rates
- **THEN** no unreviewed member of the 8,969-fact pool is promoted or included in a formal cohort

#### Scenario: Overall-random 400 is also predeclared as a cohort source
- **WHEN** the 400-item scope is explicitly selected as the formal cohort source
- **THEN** all 400 items receive the same review protocol and the accepted output remains blocked until full-pool duplicate and historical-exposure closure completes

#### Scenario: Complete pool is selected as the cohort universe
- **WHEN** the formal universe is the complete 8,969-fact pool
- **THEN** estimated sample rates do not satisfy admission and every one of the 8,969 facts requires an item-level final disposition

#### Scenario: Historical exposure authority was not predeclared
- **WHEN** the user has not supplied an authoritative historical-exposure inventory at universe declaration time
- **THEN** the universe SHALL record `historical_exposure_contract_status=not_predeclared`, preflight SHALL retain `historical_exposure_authority_missing`, and a runtime self-attested registry SHALL NOT authorize finalization

### Requirement: Every scoped item has a terminal review disposition
Before freeze, the system SHALL account for every item in the declared review scope. Final
dispositions SHALL be `accept` or `reject`; `missing`, `defer`, and an unapplied `revise` SHALL
block review completion. A revised fact SHALL retain its source and prior-value lineage, be
re-clustered, and receive a new review bound to the revised row hash.

Revision lineage v2 SHALL carry an exact semantic payload and a source-hash-bound reconstruction
for re-review only. The reconstruction SHALL declare that it is not a final behavior row and that
derived fields were not recomputed. It SHALL bind the exact source `revise` staging-row hash and
record non-empty editor identity, method, and timestamp; the terminal re-review SHALL likewise
record reviewer identity, method, and timestamp and bind the reconstruction hash. Before formal
materialization, a future finalizer SHALL re-derive and validate prompts, distractors, normalized
relation/probe fields, duplicate clusters, and split assignment. Revision or re-review evidence
SHALL NOT self-promote canonical status, human-gold status, evidence tier, or freeze authorization.

#### Scenario: Partial staging is applied
- **WHEN** one or more scoped decisions are missing, deferred, or awaiting revision
- **THEN** the system may emit staging diagnostics but MUST NOT emit a review-freeze manifest

#### Scenario: A revision is accepted after re-review
- **WHEN** a requested revision is applied with provenance and the revised row is explicitly accepted
- **THEN** the re-review binds the staged reconstruction hash and the superseded value remains in lineage, but formal admission remains blocked until downstream fields are re-derived into a separately validated final row

### Requirement: Review freeze is bound to reproducible evidence
The review-freeze manifest SHALL bind the exact cohort universe, review scope/export, final
decision artifact, decision counts, revision and re-review lineage, re-clustered reviewed bundle,
and row inventory hashes. It SHALL prove complete coverage and zero unresolved dispositions.
Self-asserted completion booleans without those bindings MUST NOT authorize formal admission.
Each final bundle row's `human_gold` and canonical `evidence_tier` SHALL exactly match its bound
review decision; unknown tiers, type coercion, whitespace-normalized aliases, and conflicting
row-local claims MUST fail closed.

#### Scenario: A hand-written manifest asserts completion
- **WHEN** a manifest sets completion flags but lacks the required decision and lineage bindings
- **THEN** formal behavior and PATH admission fail closed

#### Scenario: Review finalization succeeds
- **WHEN** every universe item is accounted for, revisions are re-reviewed, hashes match, and unresolved counts are zero
- **THEN** the system may emit a review-freeze manifest for that exact reviewed bundle only

#### Scenario: A promoted row overstates its review evidence
- **WHEN** a bundle row's `human_gold` or `evidence_tier` differs from the bound final decision
- **THEN** formal behavior and PATH admission fail closed
