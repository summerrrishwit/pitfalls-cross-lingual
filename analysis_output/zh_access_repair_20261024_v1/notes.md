# Evidence

User explicitly requested repair after the recorded HTTP403 stop. No credential/model/route changes; no repeated automated access probes.

Bindings and prior worker exit verified. Config.env mtime2026-10-22T06:52:11Z, route still matches frozen manifest. Exactly two production translation requests returned403 around14:05UTC; no response body was persisted by the old safe_error. Explicit synthetic access check at14:17:41Z hitReadTimeout after46.134s before response headers, no credential/model/route changes. One final structured same-route90s SSE diagnostic is pending; no production or automatic probes restarted.

Structured same-route SSE probe passed11.956s,all4cases,model identity andnonzero usage. Final boundprobe gates single1/1+8concurrency8/8,36/36cases passed.65tests passed. Recovery adds only an explicit user-requested403-restoration switch requiring all diagnostic requests succeed; automatic403 handling remainsblocked. Recovery plan preservesallhistory and selects onlybatch00825,00827(access403) and00831(normalizedinflightafterfatal) foroneimmediateextraepoch; oldtechnicalfailures remain deferred toexistingtailpolicy.
