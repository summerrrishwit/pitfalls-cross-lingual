# Evidence

Explicit user request2026-10-24 to diagnose/fix recurrent403. Same configured credentials/model/route only. Preserve frozen artifacts and history.

Same configured route/credentials/model passed structured diagnosticHTTP200,4/4cases,nonzerousage. Frozen verification1/1single+8/8concurrent all36/36cases passed.71tests passed. Added sanitized HTTPerror body/type/code/message and selectedheaders via separately versioned diagnostics-only runtime; prompts/schema/validators/retry/stop policy unchanged. Historical403body absent so its internalcause remains unknown; currenttestsuccess excludes a continuing blanket access denial at validation time. Existingexplicit-user403restore archives and one-epoch access-interrupted recovery used,not automatic403policy change.

Final accessbatch outcomes: {"batch_00916.json": {"status": "failed", "translated_rows": 0, "last_http_status": null, "last_validation_code": "unsupported_evidence_quote"}, "batch_00919.json": {"status": "completed", "translated_rows": 2, "last_http_status": null, "last_validation_code": null}}
