#!/usr/bin/env python3
"""Audit repeated exact-HF behavior and document sensitivity to old runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", action="append", type=Path, default=[])
    parser.add_argument("--g0b-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def keyed(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {str(row["stimulus_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate stimulus_id")
    return result


def main() -> int:
    args = parse_args()
    result_name = "exact_hf_development_behavior_results.jsonl"
    label_name = "exact_hf_defect_control_labels.jsonl"
    canonical_result = args.canonical_dir / result_name
    repeat_result = args.repeat_dir / result_name
    canonical_labels = args.canonical_dir / label_name
    repeat_labels = args.repeat_dir / label_name
    canonical_rows = keyed(read_jsonl(canonical_result))
    if len(canonical_rows) != 960:
        raise ValueError("canonical run is not complete")
    repeat_rows = keyed(read_jsonl(repeat_result))
    if set(canonical_rows) != set(repeat_rows):
        raise ValueError("canonical and repeat stimulus sets differ")
    if sha256_file(canonical_result) != sha256_file(repeat_result):
        raise ValueError("frozen-runtime result files are not byte-identical")
    if sha256_file(canonical_labels) != sha256_file(repeat_labels):
        raise ValueError("frozen-runtime label files are not byte-identical")

    comparisons = []
    for directory in args.comparison_dir:
        other_path = directory / result_name
        other = keyed(read_jsonl(other_path))
        if set(other) != set(canonical_rows):
            raise ValueError(f"comparison stimulus set differs: {directory}")
        choice_differences = sorted(
            stimulus_id
            for stimulus_id, row in canonical_rows.items()
            if row["choice"] != other[stimulus_id]["choice"]
        )
        comparisons.append({
            "directory": directory.name,
            "result_sha256": sha256_file(other_path),
            "choice_difference_count": len(choice_differences),
            "choice_difference_stimulus_ids": choice_differences,
        })

    exact_ties = []
    for stimulus_id, row in canonical_rows.items():
        values = sorted(
            (float(score["sequence_logprob"]), choice)
            for choice, score in row["choice_scores"].items()
        )
        if values[-1][0] == values[-2][0]:
            exact_ties.append(stimulus_id)
    model_manifest_path = args.g0b_dir / "exact_hf_model_manifest.json"
    model_manifest = read_json(model_manifest_path)
    audit = {
        "schema_version": "qwen3-exact-hf-runtime-stability-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_frozen_runtime_reproduction_with_cross_runtime_sensitivity",
        "canonical_runtime": model_manifest["decoding_and_scoring"],
        "canonical": {
            "directory": args.canonical_dir.name,
            "result_sha256": sha256_file(canonical_result),
            "label_sha256": sha256_file(canonical_labels),
            "result_count": len(canonical_rows),
        },
        "same_runtime_repeat": {
            "directory": args.repeat_dir.name,
            "result_sha256": sha256_file(repeat_result),
            "label_sha256": sha256_file(repeat_labels),
            "results_byte_identical": True,
            "labels_byte_identical": True,
        },
        "superseded_runtime_comparisons": comparisons,
        "canonical_exact_top_score_tie_count": len(exact_ties),
        "canonical_exact_top_score_tie_stimulus_ids": sorted(exact_ties),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "interpretation": (
            "Only the canonical v3 runtime is admissible downstream. Differences from earlier "
            "batch-dependent runs are sensitivity evidence and must not be mixed with v3 labels."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
