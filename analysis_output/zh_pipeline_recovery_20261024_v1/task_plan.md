# Qwen gateway diagnosis and explicit recovery

User authorization: diagnose and fix Qwen3.8-max processing blockage; initialize CodeGraph (2026-10-24).

- [x] Read current run status, fatal evidence and frozen runtime recovery behavior.
- [x] Synthetic single request and8concurrent wave passed:9/9requests,36/36cases,expected Qwen label,no retries.
- [x] Implemented standalone explicit recovery;24tests passed,including8recovery tests. Frozen runtime/source/policy remain unchanged.
- [x] Original frozen worker30812 resumed once;real normalize/translate successes and count growth verified.
- [x] Translation-only policy retained;heartbeatACTIVE,target/current worktree checked,current handoff updated.

Current facts: run zh_normalize_translate_20261023_v1 stopped on repeated gateway zero-token failures. 558 normalized,547 translated/reviewed,532 accepted,15 rejected,38 technical record failures,3 normalization defer,6662 not yet normalized. No pending translation review at diagnosis start.

Constraints: no automatic semantic repairs or relation re-review; frozen input/catalog/runtime/manifest unchanged; old failures and exposure events remain auditable; no formal/PNT promotion.
