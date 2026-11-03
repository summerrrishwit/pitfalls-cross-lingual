# Diagnosis notes

Scope: same configured Qwen model and route. Preserve frozen run and existing reviews. No duplicate workers.

Matched synthetic diagnostics: buffered8192 HTTP503 twice; SSE8192 passed twice (8.066s and76.144s), exact same prompt and route. DNS/TLS/auth minimal request passed. Gateway error body: network connection unstable, server_error. Smaller1024 synthetic passed, but production output budget remains8192 to avoid truncation. Streaming assembler and separately versioned overlay prepared; not yet activated.

Versioned production-compatible streaming probe: single1/1 and frozen8concurrency8/8, all36/36syntheticcase matches, correct response labels and nonzero usage.54tests passed. Controlled drain requested using a repair-owned STOP marker; no signal or duplicated worker. No production output accepted from probes.

Hardening before activation: known SSE503/429 remain bounded technical failures; SSE401 remains a fatal access error; unknown SSE errors fail closed. Zero-usage gateway wrappers are classified before treating the gateway lowercase label as model identity, matching the original handling. All59regression tests pass. Final single+8probe rerun binds the exact stream-helper SHA; earlier probe evidence retained separately. Old worker drained normally at2026-10-24T13:33:18UTC, all started requests accounted,124failedbatches retained,62packets including12-row tail.
