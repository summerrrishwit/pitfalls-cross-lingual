#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/pnt-open-development-20261016
trap 'task_status=$?; printf "%s\n" "$task_status" > pipeline.exit' EXIT
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
task_python=/root/autodl-tmp/venvs/vllm/bin/python
"$task_python" -m unittest discover -s tests -p 'test_*option_free*.py' -v
"$task_python" scripts/run_qwen3_option_free_remote.py --config run_config.json --smoke
"$task_python" scripts/run_qwen3_option_free_remote.py --config run_config.json
"$task_python" scripts/analyze_qwen3_option_free.py --run-dir .
