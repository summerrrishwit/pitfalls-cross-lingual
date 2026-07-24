#!/usr/bin/env python3
"""Construct Cross-Lingual Factual Pitfalls from raw English source datasets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls.pipeline import (  # noqa: E402
    build_raw_manifest,
    generate_candidate,
    load_config,
    make_client,
    read_jsonl,
    screen_translation,
    stage_summary,
    translate_candidate,
    audit_candidate,
    write_json,
    write_jsonl,
)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("raw-mvp-%Y%m%dT%H%M%SZ")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/raw_factual_pitfalls_mvp.json")
    commands = result.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Create a no-network raw-source manifest.")
    sample.add_argument("--run-id", default=None)
    sample.add_argument("--output-dir", default=None)
    sample.add_argument("--dry-run", action="store_true")
    for name, description in (
        ("audit", "Run qwen3.7-plus factual review for raw-manifest candidates."),
        ("generate", "Generate answer-preserving perturbations from factual-review accepts."),
        ("translate", "Translate generation-valid candidates into each target language."),
        ("screen", "Screen translated candidates and retain cross-lingual weaknesses."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--run-dir", required=True, help="Directory containing raw_source_manifest.json.")
        command.add_argument("--limit", type=int, default=None, help="Maximum pending inputs for a bounded pilot.")
        command.add_argument("--resume", action="store_true", help="Reuse terminal records already written for this stage.")
        command.add_argument("--retry-failed", action="store_true", help="With --resume, retry only existing records whose terminal status ends in _failed.")
    return result


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    import json

    path = run_dir / "raw_source_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ValueError(f"Invalid raw source manifest: {path}")
    return data


def resume_records(path: Path, key: str, resume: bool) -> Dict[str, Dict[str, Any]]:
    if not resume:
        return {}
    return {record[key]: record for record in read_jsonl(path) if key in record}


def run_stage(
    inputs: Iterable[Dict[str, Any]],
    output_path: Path,
    key: str,
    expected_keys: Callable[[Dict[str, Any]], List[str]],
    transform: Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]],
    resume: bool,
    retry_failed: bool,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    completed = resume_records(output_path, key, resume)
    processed = 0
    for item in inputs:
        expected = expected_keys(item)
        existing = [completed.get(record_key) for record_key in expected]
        if expected and all(existing):
            failed_records = [record for record in existing if str(record.get("terminal_status", "")).endswith("_failed")]
            if not retry_failed or not failed_records:
                continue
        produced = list(transform(item))
        for record in produced:
            previous = completed.get(record[key])
            if previous is not None:
                record["retry_history"] = previous.get("retry_history", []) + [
                    {
                        "terminal_status": previous.get("terminal_status"),
                        "created_at": previous.get("created_at"),
                        "error": (previous.get("audit") or previous.get("generator") or previous.get("translator") or {}).get("error"),
                    }
                ]
            completed[record[key]] = record
        write_jsonl(output_path, completed.values())
        processed += 1
        if limit is not None and processed >= limit:
            break
    write_jsonl(output_path, completed.values())
    return list(completed.values())


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "retry_failed", False) and not args.resume:
        parser().error("--retry-failed requires --resume")
    config = load_config(resolve_path(args.config))
    if args.command == "sample":
        run_id = args.run_id or run_id_now()
        run_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_root"]) / run_id
        manifest = build_raw_manifest(config, PROJECT_ROOT, run_id)
        manifest["dry_run"] = bool(args.dry_run)
        write_json(run_dir / "raw_source_manifest.json", manifest)
        allocation = manifest["source_datasets"]["allocation"]
        print(f"Wrote raw manifest: {run_dir / 'raw_source_manifest.json'}")
        print(f"Selected raw candidates: {allocation['actual_count']} / {allocation['requested_count']}")
        print("No model request was sent.")
        return 0

    run_dir = resolve_path(args.run_dir)
    manifest = load_manifest(run_dir)
    client = make_client(PROJECT_ROOT, config["runtime"]["timeout_seconds"])
    if args.command == "audit":
        output = run_dir / "factual_reviews.jsonl"
        records = run_stage(
            manifest["candidates"], output, "review_id", lambda candidate: [candidate["candidate_id"]], lambda candidate: [audit_candidate(candidate, config, client)], args.resume, args.retry_failed, args.limit
        )
    elif args.command == "generate":
        reviews = [record for record in read_jsonl(run_dir / "factual_reviews.jsonl") if record.get("decision") == "accept" and record.get("terminal_status") == "completed"]
        output = run_dir / "generation_records.jsonl"
        def generate_all(review: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
            choices = review["source_snapshot"]["record"]["choices"]
            answer = review["source_snapshot"]["record"]["answer"]
            return [generate_candidate(review, choice, index, config, client) for index, choice in enumerate(choices) if choice != answer]
        records = run_stage(
            reviews,
            output,
            "generation_id",
            lambda review: [f"{review['candidate_id']}_wrong_{index}" for index, choice in enumerate(review["source_snapshot"]["record"]["choices"]) if choice != review["source_snapshot"]["record"]["answer"]],
            generate_all,
            args.resume,
            args.retry_failed,
            args.limit,
        )
    elif args.command == "translate":
        generations = [record for record in read_jsonl(run_dir / "generation_records.jsonl") if record.get("terminal_status") == "completed"]
        output = run_dir / "translation_records.jsonl"
        records = run_stage(
            generations, output, "translation_id", lambda generation: [f"{generation['generation_id']}_{language.lower()}" for language in config["target_languages"]], lambda generation: [translate_candidate(generation, language, config, client) for language in config["target_languages"]], args.resume, args.retry_failed, args.limit
        )
    else:
        translations = [record for record in read_jsonl(run_dir / "translation_records.jsonl") if record.get("terminal_status") == "completed"]
        output = run_dir / "screening_records.jsonl"
        records = run_stage(translations, output, "screen_id", lambda translation: [translation["translation_id"]], lambda translation: [screen_translation(translation, config, client)], args.resume, args.retry_failed, args.limit)
        write_jsonl(run_dir / "retained_pitfalls.jsonl", [record for record in records if record.get("retained")])
    write_json(run_dir / f"{args.command}_summary.json", stage_summary(records, args.command))
    print(f"Wrote {len(records)} {args.command} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
