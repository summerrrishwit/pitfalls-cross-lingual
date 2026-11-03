# Automatic Qwen recovery and larger Luna assignments

Authorization: user requested self-repair on incomplete/stalled/failed Qwen work and a larger Luna review assignment on2026-10-24.

- [x] Inspect current automation, frozen run and explicit recovery implementation.
- [x] Added tested health classification,ownership checks,rolling budgets and bounded recovery;37tests passed.
- [x] Operations policy and heartbeat allow up to40rows per real Luna assignment,keeping separate20-row packet archives.
- [x] Existing heartbeat updated and saved TOML verified;ACTIVE,same target,same15minute schedule.
- [x] Verified policy/automation persistence and healthy worker; two real Luna tasks reviewed 80 rows, archived separately (77 accepted, 3 rejected).

Constraints: no frozen input/catalog/manifest/runtime changes, no semantic generation-to-acceptance loop, no new relation/fact review. Preserve failures, original attempts, exposure and all prior verdicts. Authentication, model identity or hash drift require user action. No duplicate live worker. Automatic recovery attempts limited to3per24hours with30/60minute backoff, and one extra retry epoch per technical-failure batch.
