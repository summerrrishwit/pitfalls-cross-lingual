#!/usr/bin/env python3
"""Bounded translation-only concurrency probe; no production checkpoint mutation."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "multilingual_concurrency_runner", ROOT / "scripts/run_full_public_benchmark_multilingual_review.py"
)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--request-count", type=int, help="Bounded requests per level, at least the concurrency")
    parser.add_argument("--batch-offset", type=int, default=0)
    args = parser.parse_args()
    if not args.levels or any(level < 1 or level > 16 for level in args.levels):
        parser.error("probe concurrency must be between 1 and 16")
    if args.batch_offset < 0 or any(
        (args.request_count if args.request_count is not None else max(8, level)) < level
        or args.batch_offset + (args.request_count if args.request_count is not None else max(8, level)) > 16
        for level in args.levels
    ):
        parser.error("request count/offset must cover concurrency within the fixed 16 batches")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = runner.read_json(args.parent_manifest)
    bindings = parent["input_bindings"]
    mapping = {
        "candidate_projection_manifest_path": "candidate_projection_manifest",
        "candidate_base_facts_path": "candidate_base_facts",
        "projection_manifest_path": "projection_manifest",
        "projection_items_path": "projection_items",
        "source_review_manifest_path": "source_review_manifest",
        "source_records_path": "source_records",
    }
    loaded = runner.load_bound_inputs(**{k: Path(bindings[v]["path"]) for k, v in mapping.items()})
    # The same ordered static facts at every level; never select using model outcomes.
    items = loaded["items"][:64]
    config = runner.load_config(args.config.resolve())
    spec = {k: v for k, v in parent["models"]["generator"].items() if k != "role"}
    spec["max_retries"] = 0
    languages = runner._language_specs(list(runner.DEFAULT_LANGUAGES))
    report = {
        "schema_version": "multilingual-concurrency-probe-v1",
        "status": "running", "started_at": runner.utc_now(),
        "parent_manifest": runner.artifact_binding(args.parent_manifest),
        "runner_sha256": runner.sha256_file(Path(runner.__file__)),
        "probe_sha256": runner.sha256_file(Path(__file__)),
        "batch_size": 4, "levels_requested": args.levels, "levels": [],
        "input_policy": "fixed-first-64-static-facts-explicit-batch-offset",
        "batch_offset": args.batch_offset,
        "maximum_request_count": sum(args.request_count if args.request_count is not None else max(8, level)
                                     for level in args.levels),
        "production_outputs_reused": False, "production_outputs_written": False,
        "translation_only": True, "semantic_acceptance_measured": False,
        "no_automatic_retries_or_batch_split": True,
        "pnt_execution": False, "human_gold": False,
    }
    runner.write_json(out / "report.json", report)
    for level in args.levels:
        settings = copy.deepcopy(config)
        settings["execution"]["max_workers"] = level
        settings["execution"]["provider_max_concurrency"][spec["provider_profile"]] = level
        router = runner.runtime.ModelRouter(settings, args.env_file.resolve(), out / f"events-c{level}.jsonl")
        identity = runner.build_route_identity_set(
            config=settings, env_path=args.env_file, routes=[(spec, spec["expected_response_model"])],
            resolved_env=router.values,
        )
        expected = [r for r in parent["route_identity"]["records"]
                    if r["provider_profile"] == spec["provider_profile"]]
        if identity["records"] != expected:
            raise ValueError("probe route differs from bound production route")
        runner.attach_route_identity_guard(router, config=settings, env_path=args.env_file, identity_set=identity)
        event_lock = threading.Lock()
        event_writer = router._event
        def locked_event(*a, **kw):
            with event_lock:
                return event_writer(*a, **kw)
        router._event = locked_event
        request_count = args.request_count if args.request_count is not None else max(8, level)
        batches = [items[i * 4 : (i + 1) * 4]
                   for i in range(args.batch_offset, args.batch_offset + request_count)]
        stop = threading.Event()
        active = [0, 0]
        lock = threading.Lock()
        started = time.monotonic()
        def work(index, batch):
            if stop.is_set():
                return None
            with lock:
                active[0] += 1
                active[1] = max(active[1], active[0])
            validation_codes = []
            def validate(value):
                try:
                    runner.validate_translation_batch(value, batch, list(runner.DEFAULT_LANGUAGES))
                except ValueError as error:
                    # Validator messages are static schema codes, never model text.
                    code = str(error)
                    validation_codes.append(code if code.replace("_", "").isalnum() else "schema_validation_error")
                    raise
            try:
                result = router.request_json(
                    "multilingual_translation_concurrency_probe", f"c{level}-batch-{index}", spec,
                    runner.translation_batch_prompt(batch, languages), validate,
                )
                reason = runner.service_stop_reason(runner.BatchRequestFailure({"attempts": result.attempts}))
                transport_errors = [a for a in result.attempts if a.get("status") == "failed"
                                    and a.get("error_type") not in {"ValueError", "JSONDecodeError"}]
                if reason or transport_errors:
                    stop.set()
                return {
                    "concurrency": level, "batch_index": index,
                    "base_fact_ids": [r["base_fact_id"] for r in batch],
                    "prompt_sha256": runner.sha256_value(runner.translation_batch_prompt(batch, languages)),
                    "terminal_status": result.terminal_status, "latency_ms": result.latency_ms,
                    "attempts": runner.zh_tool._safe_attempts(result.attempts),
                    "usage": runner.zh_tool._safe_usage(result.usage),
                    "response_model_matches": result.response_model == spec["expected_response_model"],
                    "validation_codes": validation_codes,
                    "service_stop_reason": reason, "transport_error": bool(transport_errors),
                }
            finally:
                with lock:
                    active[0] -= 1
        records = []
        with ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(work, index, batch) for index, batch in enumerate(batches)]
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    records.append(row)
                    runner.append_jsonl_journal(out / "requests.jsonl", row)
        wall = time.monotonic() - started
        completed = [r for r in records if r["terminal_status"] == "completed"]
        summary = {
            "concurrency": level, "peak_in_flight": active[1], "request_count": len(records),
            "successful_requests": len(completed), "successful_facts": len(completed) * 4,
            "schema_failed_requests": sum(bool(r["validation_codes"]) for r in records),
            "json_parse_failed_requests": sum(any(a.get("error_type") == "JSONDecodeError"
                                                  for a in r["attempts"]) for r in records),
            "transport_failed_requests": sum(r["transport_error"] for r in records),
            "wall_seconds": round(wall, 3),
            "median_latency_seconds": percentile([r["latency_ms"] / 1000 for r in records], .5),
            "p95_latency_seconds": percentile([r["latency_ms"] / 1000 for r in records], .95),
            "successful_facts_per_minute": round(len(completed) * 4 * 60 / wall, 3),
            "stopped_for_service_error": stop.is_set(),
        }
        report["levels"].append(summary)
        runner.write_json(out / "report.json", report)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if stop.is_set():
            break
    report.update(status="completed_bounded_probe", completed_at=runner.utc_now())
    runner.write_json(out / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
