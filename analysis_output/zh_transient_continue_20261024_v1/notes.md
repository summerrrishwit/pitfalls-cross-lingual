# Findings

The original worker already retries a request up to3 times and records failed batches; however16 consecutive transport errors trigger a global fatal stop. Fatal record construction also allows detail.error_code to overwrite the circuit-breaker code, explaining why a global stop appeared as a single zero-usage failure.

User now explicitly requests temporary failures be retained, other sources continue, and technical failures retried afterward. Implement an independently hash-bound runtime overlay retaining the original manifest/input/runtime and old packet bindings. Transport failure streaks will cause interruptible60/120/300second shared backoff, not global shutdown. One bounded tail retry epoch per technical failed batch; no reopening semantic/schema failures or previously exhausted recovery batches. Strict access/model/route/hash/unknown errors still stop. Existing independent translation review unchanged.

Activation health: keep the single synthetic request successful. For the8concurrent diagnostic, allow known transport-only failures when at least75percent requests succeed with exact identity and all4synthetic cases matching; failures remain recorded. This avoids treating one connection failure as proof the whole service is unusable. Historical all-success reports remain unchanged.

Validation:45tests passed. Initial test fixture accidentally used pytest reserved setup name; renamed setup_run and reran successfully. Added regression that a large retained failure queue cannot block recovery of untouched sources in tail mode; excess failures remain untouched for tail processing.
