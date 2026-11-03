# Qwen3-8B Development open-answer bridge

Status: complete; semantic review: complete_with_deferrals.
Completed 960/960 inputs; 96/96 complete facts; 0 unique outputs pending review.

| Endpoint (facts) | True | False | Unresolved |
|---|---:|---:|---:|
| open_induced_directed | 17 | 74 | 5 |
| open_induced_any_error | 17 | 74 | 5 |
| open_language_specific | 5 | 89 | 2 |
| scored_preference_transfer | 57 | 39 | 0 |

Validation/Sealed remain unused. No intervention has run. Candidate-score transfer is auxiliary, not generated-answer success.
