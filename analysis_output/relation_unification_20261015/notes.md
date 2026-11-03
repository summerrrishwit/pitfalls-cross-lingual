# Evidence notes

- Current public grouping is a deterministic candidate-family assignment, not formal PNT relation certification.
- Historical canonical-v2 artifacts and relation inventory exist locally; raw model ledger availability and row counts are being checked.
- Prior frozen 160 protocol requires same-relation, different-component donors and source/evaluation fold isolation.
- Root task_plan.md is historical; this audit uses its own plan to avoid overwriting it.
- Both historical raw model ledgers exist (13,625 rows each); public Qwen ledger has 20,834 rows. Ignored files are not discoverable with ordinary rg --files, so explicit paths were checked.
- Legacy canonical: 2,450; mapped 1,511, out-of-taxonomy 849, ambiguous 90. Of 1,986 answer-consensus rows, 1,033 have differing relation_raw strings; this is not a semantic error rate.
- Exact source-question overlap is zero, but normalized S-R-A overlap between retained legacy and valid public raw extractions is one. Semantic dedup remains necessary.
- 2,133 legacy relation signatures include 2,040 singletons. Semantic candidate grouping is needed before group review can materially reduce workload.
- Local public accepted/zh accepted entity files are incomplete. Restore authoritative artifacts before full unification; do not replace revised facts with raw extraction output.
- Report distinguishes per-relation donor feasibility, per-relation reporting and relation-specific intervention vectors; the current protocol does not equate them.
