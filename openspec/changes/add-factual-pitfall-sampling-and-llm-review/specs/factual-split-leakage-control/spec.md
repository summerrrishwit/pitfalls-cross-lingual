## ADDED Requirements

### Requirement: Duplicate and exposure closure precedes split freeze
The system SHALL evaluate the declared cohort universe against itself, the complete provisional
source pool, comparison canonical data, and every recorded historically exposed evaluation set.
It SHALL preserve adjudications for exact duplicates, alias-connected candidates, lexical
near-duplicates, and protocol-defined semantic paraphrase checks. Bounded lexical candidate
generation alone MUST NOT be described as exhaustive semantic review.
Every near-duplicate adjudication SHALL bind the exact candidate-record hash plus both endpoint
identifiers and endpoint row hashes. A decision bound only to a reusable candidate ID MUST fail
closed.

#### Scenario: A subset cohort has a link to an out-of-subset pool fact
- **WHEN** duplicate or exposure evidence connects a cohort item to any of the other pool facts
- **THEN** the link is adjudicated and the resulting component is grouped, excluded, or explicitly resolved before split freeze

#### Scenario: A lexical audit still has unresolved candidates
- **WHEN** one or more blocking links or connected components lack semantic disposition
- **THEN** `semantic_review_complete` remains false and the split cannot be frozen

#### Scenario: Candidate content changes after adjudication
- **WHEN** a candidate row, either endpoint identity, or either endpoint row hash differs from the bound decision
- **THEN** the adjudication is stale and the split cannot be frozen

### Requirement: Splits are recomputed after review and re-clustering
The system SHALL construct leakage components only after accepted revisions and duplicate
re-clustering are complete. Every base fact, language variant, alias, prompt variant, distractor,
and perturbation derived from one component SHALL remain in one split. Development, validation,
and sealed assignments SHALL be recomputed from the final component inventory rather than copied
from the provisional split.

#### Scenario: Review merges two provisional base facts
- **WHEN** adjudication places two facts in the same duplicate or leakage component
- **THEN** all of their derived records receive one shared final split assignment

#### Scenario: A split assignment predates the final reviewed bundle
- **WHEN** a bundle revision or component merge changes the admitted row inventory
- **THEN** the previous split is invalidated and cannot be reused as frozen evidence

### Requirement: Split freeze is exact-bundle and review bound
The split-freeze manifest SHALL bind the final reviewed-bundle SHA, review-freeze manifest SHA,
split-policy version, component inventory, assignment counts, duplicate/near-duplicate
adjudication artifact, and historical-exposure audit. It SHALL require zero unresolved cross-split
components and zero exposure in both validation and sealed evaluation sets.

#### Scenario: Validation or sealed data was historically exposed
- **WHEN** the exposure audit finds any affected validation or sealed component
- **THEN** that component is removed or reassigned and the split is recomputed before freeze

#### Scenario: Split evidence is complete
- **WHEN** all bound artifacts match, all components have one split, and validation/sealed exposure counts are zero
- **THEN** the system may emit a split-freeze manifest for that exact reviewed bundle
