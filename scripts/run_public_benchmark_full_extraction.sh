#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${PUBLIC_BENCHMARK_CONFIG:-$PROJECT_ROOT/configs/raw_factual_pitfalls_public_full.json}"
RUN_DIR="${PUBLIC_BENCHMARK_RUN_DIR:-$PROJECT_ROOT/data_processed/factual_triples/public-benchmarks-full-v1}"
OUTPUT_PATH="$RUN_DIR/models/qwen3.7-plus/triple_extractions.jsonl"
TOKENHUB_ENV_FILE="${TOKENHUB_ENV_FILE:?TOKENHUB_ENV_FILE must point to a TokenHub .env file}"
MAX_REEXECUTIONS="${MAX_REEXECUTIONS:-3}"

set -a
source "$TOKENHUB_ENV_FILE"
set +a

export OPENAI_API_KEY="${TOKENHUB_API_KEY:?TOKENHUB_API_KEY is required}"
export OPENAI_BASE_URL="${TOKENHUB_BASE_URL:?TOKENHUB_BASE_URL is required}"
export QWEN_EXTRACTION_PROVIDER="openai"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pitfalls-public-benchmarks-pycache}"
if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$EXTRA_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi

run_extract() {
  python3 "$PROJECT_ROOT/scripts/build_raw_factual_pitfalls.py" \
    --config "$CONFIG_PATH" \
    extract \
    --run-dir "$RUN_DIR" \
    --models qwen3.7-plus \
    --max-workers 8 \
    --model-parallelism 1 \
    --checkpoint-every 100 \
    "$@"
}

if [[ -e "$OUTPUT_PATH" || -e "$OUTPUT_PATH.journal" ]]; then
  run_extract --resume
else
  run_extract
fi

for ((retry_pass = 1; retry_pass <= MAX_REEXECUTIONS; retry_pass++)); do
  failed_count="$(jq -s '[.[] | select(.terminal_status == "extraction_failed" or .terminal_status == "validation_failed")] | length' "$OUTPUT_PATH")"
  if [[ "$failed_count" == "0" ]]; then
    break
  fi
  printf 'Retry pass %d/%d: %s failed records remain.\n' "$retry_pass" "$MAX_REEXECUTIONS" "$failed_count"
  run_extract --resume --retry-failed
done

jq -s '{record_count:length,completed_count:([.[]|select(.terminal_status=="completed")]|length),extraction_failed_count:([.[]|select(.terminal_status=="extraction_failed")]|length),validation_failed_count:([.[]|select(.terminal_status=="validation_failed")]|length)}' "$OUTPUT_PATH"
