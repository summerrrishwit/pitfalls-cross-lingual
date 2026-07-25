#!/usr/bin/env python3
"""Construct Cross-Lingual Factual Pitfalls from raw English source datasets."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls.pipeline import (  # noqa: E402
    AUDIT_MODEL,
    PROMPT_VERSION,
    build_raw_manifest,
    generate_candidate,
    load_config,
    make_audit_client,
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
    sample.add_argument(
        "--expand-existing",
        action="store_true",
        help='Safely replace an existing manifest only when expanding it to raw_sample_size "all" while retaining its candidates.',
    )
    sample.add_argument(
        "--restore-reviewed-pilot",
        action="store_true",
        help="Restore an expanded run to the configured pilot manifest only when its candidate IDs exactly match the existing factual reviews.",
    )
    for name, description in (
        ("audit", "Run SenseNova factual review for raw-manifest candidates."),
        ("generate", "Generate answer-preserving perturbations from factual-review accepts."),
        ("translate", "Translate generation-valid candidates into each target language."),
        ("screen", "Screen translated candidates and retain cross-lingual weaknesses."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--run-dir", required=True, help="Directory containing raw_source_manifest.json.")
        command.add_argument("--limit", type=int, default=None, help="Maximum pending inputs for a bounded pilot.")
        command.add_argument("--resume", action="store_true", help="Reuse terminal records already written for this stage.")
        command.add_argument("--retry-failed", action="store_true", help="With --resume, retry only existing records whose terminal status ends in _failed.")
        if name == "audit":
            command.add_argument(
                "--max-workers",
                type=int,
                default=1,
                help="Concurrent audit requests. Keep this low if the API is rate-limited.",
            )
            command.add_argument(
                "--output-name",
                default="factual_reviews.jsonl",
                help="Audit JSONL basename. Use a separate name for prompt-calibration reruns.",
            )
            command.add_argument(
                "--candidate-ids-from",
                default=None,
                help="Optional JSONL whose candidate_id/review_id order defines the audit subset.",
            )
            command.add_argument(
                "--stratify-candidate-ids",
                action="store_true",
                help="Round-robin candidate IDs by prior decision and source; intended for representative prompt calibration.",
            )
    return result


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "raw_source_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ValueError(f"Invalid raw source manifest: {path}")
    return data


def expansion_metadata(previous: Dict[str, Any], expanded: Dict[str, Any], review_path: Path) -> Dict[str, Any]:
    """Verify that a full manifest preserves every old candidate before replacement."""
    old_candidates = {candidate.get("candidate_id"): candidate for candidate in previous["candidates"]}
    new_candidates = {candidate.get("candidate_id"): candidate for candidate in expanded["candidates"]}
    missing = sorted(set(old_candidates) - set(new_candidates))
    changed = [
        candidate_id
        for candidate_id, candidate in old_candidates.items()
        if candidate_id in new_candidates and candidate.get("source_snapshot") != new_candidates[candidate_id].get("source_snapshot")
    ]
    if missing or changed:
        details = []
        if missing:
            details.append(f"missing candidate IDs: {missing[:5]}")
        if changed:
            details.append(f"changed source snapshots: {changed[:5]}")
        raise ValueError("Refusing manifest expansion; existing checkpoint cannot be safely reused (" + "; ".join(details) + ")")
    review_ids = {record.get("review_id") for record in read_jsonl(review_path)}
    unknown_reviews = sorted(review_ids - set(new_candidates) - {None})
    if unknown_reviews:
        raise ValueError(f"Refusing manifest expansion; factual reviews contain unknown IDs: {unknown_reviews[:5]}")
    return {
        "kind": "full_population_expansion",
        "previous_candidate_count": len(old_candidates),
        "preserved_candidate_count": len(old_candidates),
        "existing_factual_review_count": len(review_ids - {None}),
        "expanded_at": datetime.now(timezone.utc).isoformat(),
    }


def pilot_restore_metadata(previous: Dict[str, Any], restored: Dict[str, Any], review_path: Path) -> Dict[str, Any]:
    review_ids = {record.get("review_id") for record in read_jsonl(review_path)} - {None}
    restored_ids = {candidate.get("candidate_id") for candidate in restored["candidates"]} - {None}
    if not review_ids or review_ids != restored_ids:
        missing_reviews = sorted(restored_ids - review_ids)
        extra_reviews = sorted(review_ids - restored_ids)
        raise ValueError(
            "Refusing pilot restore; configured candidates and factual-review IDs differ "
            f"(missing reviews: {missing_reviews[:5]}; extra reviews: {extra_reviews[:5]})"
        )
    return {
        "kind": "restore_reviewed_pilot",
        "previous_candidate_count": len(previous["candidates"]),
        "restored_candidate_count": len(restored_ids),
        "preserved_factual_review_count": len(review_ids),
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }


def resume_records(path: Path, key: str, resume: bool) -> Dict[str, Dict[str, Any]]:
    if not resume:
        return {}
    return {record[key]: record for record in read_jsonl(path) if key in record}


def audit_inputs(
    manifest: Dict[str, Any],
    run_dir: Path,
    candidate_ids_from: Optional[str],
    stratify_candidate_ids: bool = False,
) -> List[Dict[str, Any]]:
    candidates = manifest["candidates"]
    if not candidate_ids_from:
        return candidates
    source_path = Path(candidate_ids_from)
    if not source_path.is_absolute():
        source_path = run_dir / source_path
    source_records = read_jsonl(source_path)
    if stratify_candidate_ids:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in source_records:
            dataset = record.get("source_snapshot", {}).get("dataset", "unknown")
            group = f"{record.get('decision', 'unknown')}::{dataset}"
            groups.setdefault(group, []).append(record)
        source_records = []
        group_offsets = {group: 0 for group in groups}
        while any(group_offsets[group] < len(groups[group]) for group in groups):
            for group in sorted(groups):
                offset = group_offsets[group]
                if offset < len(groups[group]):
                    source_records.append(groups[group][offset])
                    group_offsets[group] += 1
    requested_ids = []
    for record in source_records:
        candidate_id = record.get("candidate_id") or record.get("review_id")
        if candidate_id and candidate_id not in requested_ids:
            requested_ids.append(candidate_id)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in requested_ids if candidate_id not in candidates_by_id]
    if missing:
        raise ValueError(f"Candidate ID source contains IDs absent from manifest: {missing[:5]}")
    return [candidates_by_id[candidate_id] for candidate_id in requested_ids]


def audit_output_path(run_dir: Path, output_name: str) -> Path:
    output = Path(output_name)
    if output.name != output_name or output.suffix != ".jsonl":
        raise ValueError("--output-name must be a .jsonl basename without directories")
    return run_dir / output


def validate_audit_resume_version(output_path: Path, resume: bool) -> None:
    if not resume:
        return
    versions = {
        record.get("audit", {}).get("prompt_version")
        for record in read_jsonl(output_path)
        if record.get("audit", {}).get("prompt_version")
    }
    incompatible = sorted(version for version in versions if version != PROMPT_VERSION)
    if incompatible:
        raise ValueError(
            f"Cannot resume {output_path.name} with {PROMPT_VERSION}; existing prompt versions are {incompatible}. "
            "Use --output-name for a separate prompt-calibration or full-run checkpoint."
        )
    models = {
        record.get("audit", {}).get("model")
        for record in read_jsonl(output_path)
        if record.get("audit", {}).get("model")
    }
    incompatible_models = sorted(model for model in models if model != AUDIT_MODEL)
    if incompatible_models:
        raise ValueError(
            f"Cannot resume {output_path.name} with {AUDIT_MODEL}; existing audit models are {incompatible_models}. "
            "Use --output-name for a separate SenseNova checkpoint."
        )


def run_stage(
    inputs: Iterable[Dict[str, Any]],
    output_path: Path,
    key: str,
    expected_keys: Callable[[Dict[str, Any]], List[str]],
    transform: Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]],
    resume: bool,
    retry_failed: bool,
    limit: Optional[int],
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    completed = resume_records(output_path, key, resume)
    pending = []
    for item in inputs:
        expected = expected_keys(item)
        existing = [completed.get(record_key) for record_key in expected]
        if expected and all(existing):
            failed_records = [record for record in existing if str(record.get("terminal_status", "")).endswith("_failed")]
            if not retry_failed or not failed_records:
                continue
        pending.append(item)
        if limit is not None and len(pending) >= limit:
            break
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    def store(produced: Iterable[Dict[str, Any]]) -> None:
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

    if max_workers == 1:
        for item in pending:
            store(transform(item))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(lambda value=item: list(transform(value))) for item in pending]
            for future in as_completed(futures):
                store(future.result())
    write_jsonl(output_path, completed.values())
    return list(completed.values())


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "retry_failed", False) and not args.resume:
        parser().error("--retry-failed requires --resume")
    config = load_config(resolve_path(args.config))
    if args.command == "sample":
        if args.expand_existing and args.restore_reviewed_pilot:
            parser().error("--expand-existing and --restore-reviewed-pilot are mutually exclusive")
        run_id = args.run_id or run_id_now()
        run_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_root"]) / run_id
        manifest = build_raw_manifest(config, PROJECT_ROOT, run_id)
        manifest_path = run_dir / "raw_source_manifest.json"
        if manifest_path.exists():
            previous = load_manifest(run_dir)
            if args.restore_reviewed_pilot:
                if not isinstance(config["raw_sample_size"], int):
                    parser().error("--restore-reviewed-pilot requires an integer raw_sample_size")
                manifest["migration"] = pilot_restore_metadata(previous, manifest, run_dir / "factual_reviews.jsonl")
            elif args.expand_existing:
                if config["raw_sample_size"] != "all":
                    parser().error('--expand-existing requires raw_sample_size set to "all"')
                manifest["migration"] = expansion_metadata(previous, manifest, run_dir / "factual_reviews.jsonl")
            else:
                parser().error(
                    f"{manifest_path} already exists; use --expand-existing or --restore-reviewed-pilot only for a validated migration"
                )
        manifest["dry_run"] = bool(args.dry_run)
        write_json(manifest_path, manifest)
        allocation = manifest["source_datasets"]["allocation"]
        print(f"Wrote raw manifest: {manifest_path}")
        print(f"Selected raw candidates: {allocation['actual_count']} / {allocation['requested_count']}")
        print("No model request was sent.")
        return 0

    run_dir = resolve_path(args.run_dir)
    manifest = load_manifest(run_dir)
    if args.command == "audit":
        client = make_audit_client(PROJECT_ROOT, config["runtime"]["timeout_seconds"])
        output = audit_output_path(run_dir, args.output_name)
        validate_audit_resume_version(output, args.resume)
        candidates = audit_inputs(manifest, run_dir, args.candidate_ids_from, args.stratify_candidate_ids)
        records = run_stage(
            candidates,
            output,
            "review_id",
            lambda candidate: [candidate["candidate_id"]],
            lambda candidate: [audit_candidate(candidate, config, client)],
            args.resume,
            args.retry_failed,
            args.limit,
            args.max_workers,
        )
        summary_path = run_dir / ("audit_summary.json" if args.output_name == "factual_reviews.jsonl" else f"{output.stem}_summary.json")
    elif args.command == "generate":
        client = make_client(PROJECT_ROOT, config["runtime"]["timeout_seconds"])
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
        client = make_client(PROJECT_ROOT, config["runtime"]["timeout_seconds"])
        generations = [record for record in read_jsonl(run_dir / "generation_records.jsonl") if record.get("terminal_status") == "completed"]
        output = run_dir / "translation_records.jsonl"
        records = run_stage(
            generations, output, "translation_id", lambda generation: [f"{generation['generation_id']}_{language.lower()}" for language in config["target_languages"]], lambda generation: [translate_candidate(generation, language, config, client) for language in config["target_languages"]], args.resume, args.retry_failed, args.limit
        )
    else:
        client = make_client(PROJECT_ROOT, config["runtime"]["timeout_seconds"])
        translations = [record for record in read_jsonl(run_dir / "translation_records.jsonl") if record.get("terminal_status") == "completed"]
        output = run_dir / "screening_records.jsonl"
        records = run_stage(translations, output, "screen_id", lambda translation: [translation["translation_id"]], lambda translation: [screen_translation(translation, config, client)], args.resume, args.retry_failed, args.limit)
        write_jsonl(run_dir / "retained_pitfalls.jsonl", [record for record in records if record.get("retained")])
    if args.command != "audit":
        summary_path = run_dir / f"{args.command}_summary.json"
    write_json(summary_path, stage_summary(records, args.command))
    print(f"Wrote {len(records)} {args.command} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
