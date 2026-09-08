#!/usr/bin/env python3
"""Build atomic factual triples from immutable raw English QA sources."""

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
    CALIBRATION_ERROR_THRESHOLD,
    TRIPLE_LABEL_MODELS,
    TRIPLE_EXTRACTION_MODEL,
    TRIPLE_EXTRACTION_PROMPT_VERSION,
    apply_extraction_adjudications,
    apply_relation_mapping,
    build_codex_reference,
    build_raw_manifest,
    build_relation_inventory,
    evaluate_model_against_codex_reference,
    extract_triple_candidate,
    extraction_summary,
    load_config,
    make_extraction_client,
    model_slug,
    read_jsonl,
    summarize_dual_model_records,
    write_json,
    write_jsonl,
)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("triple-pilot-%Y%m%dT%H%M%SZ")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/raw_factual_pitfalls_mvp.json")
    commands = result.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="Create a no-network raw-source manifest.")
    sample.add_argument("--run-id", default=None)
    sample.add_argument("--output-dir", default=None)
    sample.add_argument("--dry-run", action="store_true")
    sample.add_argument("--exclude-manifest", default=None)

    extract = commands.add_parser("extract", help="Extract triples independently with one or both label models.")
    extract.add_argument("--run-dir", required=True)
    extract.add_argument("--models", nargs="+", choices=TRIPLE_LABEL_MODELS, default=None)
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--resume", action="store_true")
    extract.add_argument("--retry-failed", action="store_true")
    extract.add_argument("--max-workers", type=int, default=1)
    extract.add_argument("--model-parallelism", type=int, default=2)
    extract.add_argument("--output-name", default="triple_extractions.jsonl")

    inventory = commands.add_parser("inventory", help="Build a global relation inventory from valid triples.")
    inventory.add_argument("--run-dir", required=True)
    inventory.add_argument("--model", choices=TRIPLE_LABEL_MODELS, required=True)
    inventory.add_argument("--input-name", default="triple_extractions.jsonl")
    inventory.add_argument("--output-name", default="relation_inventory.json")
    inventory.add_argument("--max-examples", type=int, default=5)

    normalize = commands.add_parser("normalize", help="Apply a frozen taxonomy and explicit relation mapping.")
    normalize.add_argument("--run-dir", required=True)
    normalize.add_argument("--model", choices=TRIPLE_LABEL_MODELS, required=True)
    normalize.add_argument("--input-name", default="triple_extractions.jsonl")
    normalize.add_argument("--inventory", default="relation_inventory.json")
    normalize.add_argument("--taxonomy", required=True)
    normalize.add_argument("--mapping", required=True)
    normalize.add_argument("--output-name", default="triples_normalized.jsonl")

    reference = commands.add_parser("reference", help="Expand a reviewed Codex reference policy to JSONL.")
    reference.add_argument("--base-extractions", required=True)
    reference.add_argument("--policy", required=True)
    reference.add_argument("--output", required=True)

    evaluate = commands.add_parser("evaluate", help="Evaluate per-model labels against Codex reference.")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--models", nargs="+", choices=TRIPLE_LABEL_MODELS, default=None)
    evaluate.add_argument("--input-name", default="triple_extractions.jsonl")
    evaluate.add_argument("--threshold", type=float, default=CALIBRATION_ERROR_THRESHOLD)
    evaluate.add_argument("--output", default="calibration/model_evaluation.json")

    summarize = commands.add_parser("summarize", help="Build a deterministic dual-model Codex-review bundle.")
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--models", nargs=2, choices=TRIPLE_LABEL_MODELS, default=list(TRIPLE_LABEL_MODELS))
    summarize.add_argument("--input-name", default="triple_extractions.jsonl")
    summarize.add_argument("--output", default="codex_summary.json")

    adjudicate = commands.add_parser("adjudicate", help="Apply reviewed local decisions to validation failures.")
    adjudicate.add_argument("--run-dir", required=True)
    adjudicate.add_argument("--model", choices=TRIPLE_LABEL_MODELS, required=True)
    adjudicate.add_argument("--policy", required=True)
    adjudicate.add_argument("--input-name", default="triple_extractions.jsonl")
    return result


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "raw_source_manifest.json"
    manifest = load_json(path)
    if not isinstance(manifest.get("candidates"), list):
        raise ValueError(f"Invalid raw source manifest: {path}")
    return manifest


def safe_basename(run_dir: Path, name: str, suffix: str) -> Path:
    path = Path(name)
    if path.name != name or path.suffix != suffix:
        raise ValueError(f"output name must be a {suffix} basename without directories")
    return run_dir / path


def model_output_dir(run_dir: Path, model: str) -> Path:
    return run_dir / "models" / model_slug(model)


def resume_records(path: Path, key: str, resume: bool) -> Dict[str, Dict[str, Any]]:
    if not resume:
        return {}
    return {record[key]: record for record in read_jsonl(path) if key in record}


def validate_extraction_resume(output_path: Path, resume: bool, model: str = TRIPLE_EXTRACTION_MODEL) -> None:
    if not resume:
        return
    records = read_jsonl(output_path)
    versions = {
        record.get("extraction", {}).get("prompt_version")
        for record in records
        if record.get("extraction", {}).get("prompt_version")
    }
    incompatible_versions = sorted(
        version for version in versions if version != TRIPLE_EXTRACTION_PROMPT_VERSION
    )
    if incompatible_versions:
        raise ValueError(
            f"Cannot resume {output_path.name} with {TRIPLE_EXTRACTION_PROMPT_VERSION}; "
            f"existing prompt versions are {incompatible_versions}."
        )
    models = {
        record.get("extraction", {}).get("model")
        for record in records
        if record.get("extraction", {}).get("model")
    }
    incompatible_models = sorted(existing_model for existing_model in models if existing_model != model)
    if incompatible_models:
        raise ValueError(
            f"Cannot resume {output_path.name} with {model}; "
            f"existing extraction models are {incompatible_models}."
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
            failed = [
                record
                for record in existing
                if record.get("terminal_status") in {"extraction_failed", "validation_failed"}
            ]
            if not retry_failed or not failed:
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
                        "error": previous.get("extraction", {}).get("error"),
                        "validation_errors": previous.get("validation_errors", []),
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
        run_id = args.run_id or run_id_now()
        run_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_root"]) / run_id
        manifest_path = run_dir / "raw_source_manifest.json"
        if manifest_path.exists():
            parser().error(f"{manifest_path} already exists; choose a new run ID or output directory")
        excluded_ids = None
        if args.exclude_manifest:
            prior_manifest = load_json(resolve_path(args.exclude_manifest))
            excluded_ids = [item["candidate_id"] for item in prior_manifest.get("candidates", [])]
        manifest = build_raw_manifest(config, PROJECT_ROOT, run_id, excluded_candidate_ids=excluded_ids)
        manifest["dry_run"] = bool(args.dry_run)
        write_json(manifest_path, manifest)
        allocation = manifest["source_datasets"]["allocation"]
        print(f"Wrote raw manifest: {manifest_path}")
        print(f"Selected raw candidates: {allocation['actual_count']} / {allocation['requested_count']}")
        print("No model request was sent.")
        return 0

    if args.command == "reference":
        base_records = read_jsonl(resolve_path(args.base_extractions))
        policy = load_json(resolve_path(args.policy))
        reference_records = build_codex_reference(base_records, policy)
        output = resolve_path(args.output)
        write_jsonl(output, reference_records)
        print(f"Wrote {len(reference_records)} Codex reference records to {output}")
        return 0

    run_dir = resolve_path(args.run_dir)
    if args.command == "adjudicate":
        output_dir = model_output_dir(run_dir, args.model)
        input_path = safe_basename(output_dir, args.input_name, ".jsonl")
        records = apply_extraction_adjudications(
            read_jsonl(input_path), load_json(resolve_path(args.policy)), args.model
        )
        write_jsonl(input_path, records)
        write_json(
            input_path.with_name(f"{input_path.stem}_summary.json"),
            extraction_summary(records, args.model),
        )
        print(f"Applied local adjudications to {input_path}")
        return 0

    if args.command == "extract":
        manifest = load_manifest(run_dir)
        models = args.models or config["roles"]["triple_label_models"]
        if args.model_parallelism <= 0:
            raise ValueError("model-parallelism must be a positive integer")

        def run_model(model: str) -> Dict[str, Any]:
            output_dir = model_output_dir(run_dir, model)
            output = safe_basename(output_dir, args.output_name, ".jsonl")
            validate_extraction_resume(output, args.resume, model)
            client = make_extraction_client(PROJECT_ROOT, config["runtime"]["timeout_seconds"], model)
            records = run_stage(
                manifest["candidates"],
                output,
                "candidate_id",
                lambda candidate: [candidate["candidate_id"]],
                lambda candidate: [extract_triple_candidate(candidate, config, client, model=model)],
                args.resume,
                args.retry_failed,
                args.limit,
                args.max_workers,
            )
            summary_path = output_dir / f"{output.stem}_summary.json"
            summary = extraction_summary(records, model=model)
            write_json(summary_path, summary)
            return {"model": model, "record_count": len(records), "output": str(output)}

        if len(models) == 1 or args.model_parallelism == 1:
            results = [run_model(model) for model in models]
        else:
            with ThreadPoolExecutor(max_workers=min(args.model_parallelism, len(models))) as executor:
                futures = {executor.submit(run_model, model): model for model in models}
                results = [future.result() for future in as_completed(futures)]
        for result in sorted(results, key=lambda item: item["model"]):
            print(f"Wrote {result['record_count']} records for {result['model']} to {result['output']}")
        return 0


    if args.command == "evaluate":
        models = args.models or config["roles"]["triple_label_models"]
        reference_records = read_jsonl(resolve_path(args.reference))
        evaluations = []
        for model in models:
            records = read_jsonl(safe_basename(model_output_dir(run_dir, model), args.input_name, ".jsonl"))
            evaluations.append(
                evaluate_model_against_codex_reference(
                    reference_records, records, model, threshold=args.threshold
                )
            )
        payload = {
            "evaluation_version": "codex-dual-model-calibration-v1",
            "threshold": args.threshold,
            "all_models_passed": bool(evaluations) and all(item["gate_passed"] for item in evaluations),
            "models": evaluations,
        }
        output = resolve_path(args.output) if Path(args.output).is_absolute() else run_dir / args.output
        write_json(output, payload)
        print(f"Wrote calibration evaluation to {output}; all_models_passed={payload['all_models_passed']}")
        return 0

    if args.command == "summarize":
        records_by_model = {
            model: read_jsonl(safe_basename(model_output_dir(run_dir, model), args.input_name, ".jsonl"))
            for model in args.models
        }
        payload = summarize_dual_model_records(records_by_model)
        output = resolve_path(args.output) if Path(args.output).is_absolute() else run_dir / args.output
        write_json(output, payload)
        print(f"Wrote dual-model summary to {output}")
        return 0

    if args.command == "inventory":
        output_dir = model_output_dir(run_dir, args.model)
        input_path = safe_basename(output_dir, args.input_name, ".jsonl")
        output = safe_basename(output_dir, args.output_name, ".json")
        inventory = build_relation_inventory(read_jsonl(input_path), max_examples=args.max_examples)
        write_json(output, inventory)
        print(f"Wrote {inventory['signature_count']} relation signatures to {output}")
        return 0

    output_dir = model_output_dir(run_dir, args.model)
    input_path = safe_basename(output_dir, args.input_name, ".jsonl")
    inventory_path = resolve_path(args.inventory) if Path(args.inventory).is_absolute() else output_dir / args.inventory
    taxonomy_path = resolve_path(args.taxonomy)
    mapping_path = resolve_path(args.mapping)
    output = safe_basename(output_dir, args.output_name, ".jsonl")
    normalized = apply_relation_mapping(
        read_jsonl(input_path),
        load_json(inventory_path),
        load_json(taxonomy_path),
        load_json(mapping_path),
    )
    write_jsonl(output, normalized)
    print(f"Wrote {len(normalized)} normalized records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
