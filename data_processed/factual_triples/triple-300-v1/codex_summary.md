# Codex summary: dual-model 300-row factual-triple experiment

## Calibration gate

- Reference set: 100 Codex-reviewed rows, 27 extracted and 73 not extractable.
- SenseNova v6 label error: 0/100 (0%).
- Qwen v6 label error: 1/100 (1%).
- Both models passed the required error threshold of at most 3%.

## Independent 300-row experiment

- The 300 candidates are disjoint from the 100-row calibration set.
- Each source dataset contributes 60 candidates.
- SenseNova: 117 extracted, 183 not extractable, 300 completed.
- Qwen: 127 extracted, 173 not extractable, 300 completed.
- Dual-model consensus: 108 extracted, 164 not extractable.
- Label disagreements requiring Codex review: 28.

## Codex disagreement adjudication

- Final extracted among disagreements: 6.
- Final not extractable among disagreements: 22.
- SenseNova matched Codex on 17 disagreements; Qwen matched Codex on 11.
- Consensus plus adjudication yields 114 candidate triples and 186 excluded rows.

The 114 rows are a candidate pool, not frozen human gold. Codex independently reviewed all model-label disagreements, but the 272 consensus rows were not exhaustively human-verified. Relation normalization should therefore operate on the 114 candidate pool with an additional field-level and fact-verification gate.

## Main error patterns

1. Scenario classification and procedural tool-choice questions were the largest source of over-extraction.
2. Full-sentence answers sometimes contained a recoverable entity, but explanatory propositions remained unsuitable triple objects.
3. Calculation and causal-effect questions could look like direct relations but violate the strict experiment scope.
4. Survey percentages and legal-status answers required provenance or time-stability checks beyond a date anchor.
5. The row-level `relation_raw` inventory remained fragmented: 103 signatures for 117 SenseNova triples and 120 signatures for 127 Qwen triples. A global closed taxonomy is still required before prompt generation.
