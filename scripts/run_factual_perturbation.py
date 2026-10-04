#!/usr/bin/env python3
"""Prepare and run the gated Chinese factual-perturbation MVP."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "factual_pitfalls" / "perturbation.py"
MODULE_SPEC = importlib.util.spec_from_file_location("factual_perturbation_runtime", MODULE_PATH)
runtime = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC and MODULE_SPEC.loader
sys.modules[MODULE_SPEC.name] = runtime
MODULE_SPEC.loader.exec_module(runtime)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge a small experiment overlay into its base config."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path) -> dict:
    config = runtime.read_json(path)
    parent = config.get("extends")
    if not parent:
        return config
    parent_path = (PROJECT_ROOT / parent).resolve()
    return _deep_merge(load_config(parent_path), config)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/factual_perturbation_zh_mvp_v1.json")
    result.add_argument("--env-file", default=".env")
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--limit", type=int, required=True)
    prepare.add_argument("--exclude-run-id", action="append", default=[])
    prepare.add_argument(
        "--input-bundle",
        type=Path,
        help=(
            "Explicit .json/.jsonl factual input bundle. When omitted, use "
            "inputs.input_bundle or derive the legacy sibling artifact from inputs.canonical_triples."
        ),
    )
    proxy_rerun = commands.add_parser("prepare-proxy-rerun")
    proxy_rerun.add_argument("--run-id", required=True)
    proxy_rerun.add_argument("--source-run-id", required=True)
    proxy_rerun.add_argument("--review-file", default="codex_proxy_review_v1.json")
    proxy_rerun.add_argument("--max-candidates-per-source", type=int)
    proxy_rerun.add_argument("--strength")
    static_proxy = commands.add_parser(
        "prepare-static-proxy-run",
        help=(
            "Bind the exact G0A frozen_static_bundle and expand only its Development "
            "stimuli over the frozen four-model proxy roster."
        ),
    )
    static_proxy.add_argument("--run-id", required=True)
    static_proxy.add_argument("--static-freeze-manifest", type=Path, required=True)
    static_single = commands.add_parser(
        "prepare-static-single-qwen3-run",
        help=(
            "Bind the exact G0A frozen_static_bundle and expand only its 960 "
            "Development stimuli for the frozen Qwen3-8B behavior-only route."
        ),
    )
    static_single.add_argument("--run-id", required=True)
    static_single.add_argument(
        "--static-freeze-manifest", type=Path, required=True
    )
    representative_probe = commands.add_parser(
        "probe-static-proxy-run",
        help=(
            "Call every frozen proxy model once on the same deterministic G0A-bound "
            "three-option stimulus and gate the full proxy run."
        ),
    )
    representative_probe.add_argument("--run-id", required=True)
    single_representative_probe = commands.add_parser(
        "probe-static-single-qwen3-run",
        help=(
            "Call the frozen Qwen3-8B route once on the deterministic "
            "representative G0A stimulus before its full 960-call run."
        ),
    )
    single_representative_probe.add_argument("--run-id", required=True)
    simulation_rerun = commands.add_parser("prepare-simulation-rerun")
    simulation_rerun.add_argument("--run-id", required=True)
    simulation_rerun.add_argument("--source-run-id", required=True)
    strength_rerun = commands.add_parser("prepare-strength-rerun")
    strength_rerun.add_argument("--run-id", required=True)
    strength_rerun.add_argument("--source-run-id", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run-id", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--run-id", required=True)
    analyze = commands.add_parser(
        "analyze-three-arm",
        help=(
            "Analyze Original/Neutral/Targeted behavior, retaining legacy strict labels and "
            "adding base_fact_id-level proxy_directed_candidate and proxy_zh_specific_strict summaries."
        ),
    )
    analyze.add_argument("--run-id", required=True)
    review_packet = commands.add_parser(
        "export-proxy-review",
        help="Export the Simulation-blind Codex adjudication contract for the configured manipulation family.",
    )
    review_packet.add_argument("--run-id", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run-id", required=True)
    holdout = commands.add_parser("holdout")
    holdout.add_argument("--run-id", required=True)
    report = commands.add_parser("report")
    report.add_argument("--run-id", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())
    root = PROJECT_ROOT / config["outputs"]["root"] / "runs"
    run_dir = root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env_path = (PROJECT_ROOT / args.env_file).resolve()
    if args.command == "prepare":
        exclude_source_ids = []
        for excluded_run_id in args.exclude_run_id:
            excluded_manifest = runtime.read_json(root / excluded_run_id / "run_manifest.json")
            exclude_source_ids.extend(row["source_id"] for row in excluded_manifest["records"])
        result = runtime.prepare_manifest(
            config,
            PROJECT_ROOT,
            run_dir,
            args.limit,
            args.run_id,
            exclude_source_ids,
            args.input_bundle,
        )
    elif args.command == "prepare-proxy-rerun":
        source_run_dir = root / args.source_run_id
        result = runtime.prepare_proxy_rerun(
            config,
            PROJECT_ROOT,
            source_run_dir,
            run_dir,
            args.run_id,
            source_run_dir / args.review_file,
            args.max_candidates_per_source,
            args.strength,
        )
    elif args.command == "prepare-static-proxy-run":
        result = runtime.prepare_static_proxy_run(
            config,
            PROJECT_ROOT,
            args.static_freeze_manifest,
            run_dir,
            args.run_id,
        )
    elif args.command == "prepare-static-single-qwen3-run":
        result = runtime.prepare_static_single_qwen3_run(
            config,
            PROJECT_ROOT,
            args.static_freeze_manifest,
            run_dir,
            args.run_id,
        )
    elif args.command in {
        "probe-static-proxy-run",
        "probe-static-single-qwen3-run",
    }:
        result = runtime.run_static_proxy_representative_probe(
            config, PROJECT_ROOT, run_dir, env_path
        )
    elif args.command == "prepare-strength-rerun":
        source_run_dir = root / args.source_run_id
        result = runtime.prepare_strength_rerun(
            config,
            PROJECT_ROOT,
            source_run_dir,
            run_dir,
            args.run_id,
        )
    elif args.command == "prepare-simulation-rerun":
        result = runtime.prepare_simulation_rerun(
            config,
            PROJECT_ROOT,
            root / args.source_run_id,
            run_dir,
            args.run_id,
        )
    elif args.command == "run":
        result = runtime.run_preholdout(config, PROJECT_ROOT, run_dir, env_path)
    elif args.command == "audit":
        result = runtime.audit_run(run_dir)
    elif args.command == "analyze-three-arm":
        result = runtime.analyze_three_arm(run_dir)
    elif args.command == "export-proxy-review":
        result = runtime.export_codex_review_packet(run_dir)
    elif args.command == "freeze":
        manifest = runtime.read_json(run_dir / "run_manifest.json")
        if manifest.get("execution_mode") in runtime.STATIC_BEHAVIOR_EXECUTION_MODES:
            raise SystemExit(
                "proxy behavior runs are not exact-HF evidence and cannot freeze PNT candidates"
            )
        summary = runtime.read_json(run_dir / "preholdout_summary.json")
        if not summary.get("gate_passed"):
            raise SystemExit(f"preholdout gate failed: {summary.get('gate_reasons')}")
        result = {"frozen_candidate_count": len(runtime.freeze_candidates(run_dir, config))}
    elif args.command == "holdout":
        result = runtime.run_holdout(config, run_dir, env_path)
    else:
        print(runtime.build_report(run_dir), end="")
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
