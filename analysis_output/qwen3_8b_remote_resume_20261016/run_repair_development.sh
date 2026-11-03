#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/pnt-open-development-20261016
trap 'task_status=$?; printf "%s\n" "$task_status" > repair-development.exit' EXIT
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
while ! test -f repair-collect.exit; do sleep 10; done
if ! test "$(cat repair-collect.exit)" = 0; then exit 1; fi
task_python=/root/autodl-tmp/venvs/vllm/bin/python
"$task_python" scripts/run_qwen3_open_repair_search.py --config repair_config_v3.json --phase screen
"$task_python" scripts/run_qwen3_open_repair_search.py --config repair_config_v3.json --phase generate
"$task_python" scripts/analyze_qwen3_open_repair.py --config repair_config_v3.json
