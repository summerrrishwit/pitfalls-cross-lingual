# Scope and dependency findings

Current active pointer: analysis_output/active_data_pipeline.json; actual run zh_normalize_translate_20261023_v1 in worktree762b. Its four frozen bindings point only to copied input.jsonl, relation_catalog.json and runtime files within that active run. Current7250 input bytes match the old full input. Old relation-expansion result directories exist only under the recoverable archive, primary/ and worktree/.

Original extraction ledgers and current .env/shared scripts are outside the cleanup scope. Historical model exposure remains relevant despite deleting text/results; keep only source identifiers, stage/model metadata and artifact hashes, not old proposals or decisions. Unknown exposure stays unknown.

Pre-deletion verification:10 whitelisted obsolete archive directories,13061 files,2003568982 logical bytes. All archive manifest bindings verified. Protected guards cover current frozen bindings/manifest/pointer, primary/worktree raw extraction ledgers and .env without printing credentials. No current bound input/catalog/runtime archive references. Shared scripts are not deleted because the active run retains its own common probe helper. Exposure closure remains explicitly incomplete; metadata alone does not imply clean holdout.

Deletion completed successfully. Sandbox escalation approved under explicit user authorization. All pre/post guards passed. Current status remains running_qwen. Only provenance inventory, deletion receipt and compact exposure metadata remain in the old archive location. The old payloads cannot be recovered from that location.
