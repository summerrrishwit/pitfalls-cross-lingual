# Superseded diagnostic attribution

This directory is not an authoritative PNT input.

The run used a single shared-prefix forward pass instead of the exact behavior-scoring
shape of one MCQ with three complete choice continuations. Under BF16 this produced
62 final vocabulary-rank differences and 29 choice-order differences across 960 renders.

Use `../mcq-decision-attribution-v2/` instead. Its behavior-shape replay has zero rank
or choice-order differences and its metrics, activation index, and `resid_pre` SHA-256
values were reproduced exactly in an independent full rerun. See
`../mcq-decision-attribution-v2/attribution_runtime_stability_audit.json`.
