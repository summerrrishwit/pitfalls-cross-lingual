#!/usr/bin/env python3
"""Describe lengths of all selected static-G0A context pairs.

This audit is intentionally non-gating.  It reuses the static G0A manager's
strict review-manifest, provenance, and completed-decision validation before
measuring the selected Neutral/Targeted contexts.  It never reads behavior or
Simulation outputs and never applies a post-hoc acceptance threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_static_g0a as g0a  # noqa: E402


TOOL_VERSION = "static-g0a-selected-context-length-audit-v1"
RECORD_SCHEMA = "static-g0a-selected-context-length-record-v1"
SUMMARY_SCHEMA = "static-g0a-selected-context-length-audit-v1"
EXPECTED_BASE_FACT_COUNT = 160
EXPECTED_VARIANT_COUNT = 320
RATIO_PRECISION = 6
RECORDS_FILENAME = "selected_context_lengths.jsonl"
SUMMARY_FILENAME = "selected_context_length_summary.json"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(g0a.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _snapshot_binding(
    path: Path,
    *,
    schema_version: str,
) -> Dict[str, Any]:
    """Capture the exact input-file identity used by this audit."""

    return g0a._binding(Path(path).resolve(), schema_version=schema_version)


def _assert_snapshot_unchanged(snapshot: Mapping[str, Any]) -> None:
    path = Path(snapshot["path"])
    if (
        g0a.sha256_file(path) != snapshot["sha256"]
        or path.stat().st_size != snapshot["byte_count"]
    ):
        raise RuntimeError(f"input changed while audit was running: {path}")


def _payload_binding(
    path: Path,
    payload: bytes,
    *,
    schema_version: str,
    record_count: int,
    id_values: Sequence[str],
    id_digest_field: str,
) -> Dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "schema_version": schema_version,
        "record_count": record_count,
        id_digest_field: g0a.sha256_value(sorted(id_values)),
    }


def _publish_output_dir(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    """Publish a complete directory in one rename, never a partial result set."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        for filename, payload in sorted(payloads.items()):
            if Path(filename).name != filename:
                raise ValueError(f"output filename must be a basename: {filename}")
            output_path = temporary_dir / filename
            with output_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if os.path.lexists(target):
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {target}"
            )
        os.rename(temporary_dir, target)
        temporary_dir = None
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return the deterministic R-7/NumPy-style linear percentile."""

    if not values:
        raise ValueError("cannot summarize an empty metric")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": round(min(numeric), RATIO_PRECISION),
        "p05": round(_percentile(numeric, 0.05), RATIO_PRECISION),
        "p25": round(_percentile(numeric, 0.25), RATIO_PRECISION),
        "median": round(_percentile(numeric, 0.50), RATIO_PRECISION),
        "p75": round(_percentile(numeric, 0.75), RATIO_PRECISION),
        "p95": round(_percentile(numeric, 0.95), RATIO_PRECISION),
        "max": round(max(numeric), RATIO_PRECISION),
        "mean": round(sum(numeric) / len(numeric), RATIO_PRECISION),
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = (
        "en_neutral_word_count",
        "en_targeted_word_count",
        "en_neutral_to_targeted_ratio",
        "zh_neutral_nonspace_char_count",
        "zh_targeted_nonspace_char_count",
        "zh_neutral_to_targeted_ratio",
    )
    result = {
        field: _distribution([row[field] for row in rows]) for field in metrics
    }
    for ratio_field in (
        "en_neutral_to_targeted_ratio",
        "zh_neutral_to_targeted_ratio",
    ):
        values = [float(row[ratio_field]) for row in rows]
        result[ratio_field].update(
            {
                "below_one_count": sum(value < 1.0 for value in values),
                "equal_one_count": sum(value == 1.0 for value in values),
                "above_one_count": sum(value > 1.0 for value in values),
            }
        )
    return result


def _nonspace_char_count(value: str) -> int:
    return sum(not character.isspace() for character in value)


def _selected_length_rows(
    review_by_id: Mapping[str, Mapping[str, Any]],
    decisions_by_id: Mapping[str, Mapping[str, Any]],
    raw_decision_sha256_by_id: Mapping[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base_fact_id in sorted(review_by_id):
        review = review_by_id[base_fact_id]
        decision = decisions_by_id[base_fact_id]
        review_variants = {
            variant["distractor_id"]: variant for variant in review["variants"]
        }
        for variant_decision in decision["variants"]:
            distractor_id = variant_decision["distractor_id"]
            review_variant = review_variants[distractor_id]
            selected_candidate_id = variant_decision["selected_candidate_id"]
            candidates = [
                candidate
                for candidate in review_variant["context_candidates"]
                if candidate["candidate_id"] == selected_candidate_id
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "selected context candidate is absent or ambiguous: "
                    f"{base_fact_id}:{distractor_id}:{selected_candidate_id}"
                )
            candidate = candidates[0]
            neutral_en = g0a._required_string(
                candidate.get("neutral_context_en"),
                f"neutral_context_en {base_fact_id}:{distractor_id}",
            )
            targeted_en = g0a._required_string(
                candidate.get("targeted_context_en"),
                f"targeted_context_en {base_fact_id}:{distractor_id}",
            )
            neutral_zh = g0a._required_string(
                candidate.get("neutral_context_zh"),
                f"neutral_context_zh {base_fact_id}:{distractor_id}",
            )
            targeted_zh = g0a._required_string(
                candidate.get("targeted_context_zh"),
                f"targeted_context_zh {base_fact_id}:{distractor_id}",
            )
            en_neutral = len(neutral_en.split())
            en_targeted = len(targeted_en.split())
            zh_neutral = _nonspace_char_count(neutral_zh)
            zh_targeted = _nonspace_char_count(targeted_zh)
            if min(en_neutral, en_targeted, zh_neutral, zh_targeted) <= 0:
                raise ValueError(
                    f"selected context has an empty measured arm: {base_fact_id}:{distractor_id}"
                )
            rows.append(
                {
                    "schema_version": RECORD_SCHEMA,
                    "base_fact_id": base_fact_id,
                    "split_assignment": review["source"]["split_assignment"],
                    "distractor_id": distractor_id,
                    "selected_candidate_id": selected_candidate_id,
                    "review_input_record_sha256": g0a.sha256_value(review),
                    "decision_record_sha256": raw_decision_sha256_by_id[
                        base_fact_id
                    ],
                    "context_candidate_record_sha256": candidate[
                        "candidate_record_sha256"
                    ],
                    "candidate_source": candidate.get(
                        "candidate_source", "upstream_context_generation"
                    ),
                    "en_neutral_word_count": en_neutral,
                    "en_targeted_word_count": en_targeted,
                    "en_neutral_to_targeted_ratio": round(
                        en_neutral / en_targeted, RATIO_PRECISION
                    ),
                    "zh_neutral_nonspace_char_count": zh_neutral,
                    "zh_targeted_nonspace_char_count": zh_targeted,
                    "zh_neutral_to_targeted_ratio": round(
                        zh_neutral / zh_targeted, RATIO_PRECISION
                    ),
                    "descriptive_only": True,
                    "hard_threshold_applied": False,
                }
            )
    rows.sort(
        key=lambda row: (
            row["base_fact_id"],
            row["distractor_id"],
            row["selected_candidate_id"],
        )
    )
    return rows


def audit_selected_context_lengths(
    *,
    review_input_manifest_path: Path,
    codex_decisions_path: Path,
    output_dir_path: Path,
) -> Dict[str, Any]:
    review_input_manifest_path = Path(review_input_manifest_path).resolve()
    codex_decisions_path = Path(codex_decisions_path).resolve()
    output_dir_path = Path(output_dir_path).resolve()
    if os.path.lexists(output_dir_path):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir_path}"
        )
    output_records_path = output_dir_path / RECORDS_FILENAME
    output_summary_path = output_dir_path / SUMMARY_FILENAME

    review_manifest_snapshot = _snapshot_binding(
        review_input_manifest_path,
        schema_version=g0a.REVIEW_INPUT_MANIFEST_SCHEMA,
    )
    decisions_snapshot = _snapshot_binding(
        codex_decisions_path,
        schema_version=g0a.CODEX_DECISION_SCHEMA,
    )
    preliminary_manifest = g0a.read_json(review_input_manifest_path)
    staging_binding = preliminary_manifest.get("staging_manifest")
    if not isinstance(staging_binding, dict):
        raise ValueError("review input manifest staging_manifest binding is missing")
    staging_manifest_path = g0a._resolve_binding_path(
        staging_binding, review_input_manifest_path
    )
    review_manifest, review_rows, review_by_id = g0a._load_review_input(
        review_input_manifest_path, staging_manifest_path
    )
    decisions_by_id, blockers, decision_counts = g0a._validate_codex_decisions(
        codex_decisions_path, review_by_id
    )
    if blockers:
        raise g0a.GateBlocked(blockers)
    raw_decisions_by_id: Dict[str, Dict[str, Any]] = {}
    for raw_decision in g0a.read_jsonl(codex_decisions_path):
        base_fact_id = g0a._required_string(
            raw_decision.get("base_fact_id"), "raw decision base_fact_id"
        )
        if base_fact_id in raw_decisions_by_id:
            raise ValueError(f"duplicate raw Codex decision: {base_fact_id}")
        raw_decisions_by_id[base_fact_id] = raw_decision
    if set(raw_decisions_by_id) != set(decisions_by_id):
        raise ValueError("raw and validated Codex decision coverage differs")
    raw_decision_sha256_by_id = {
        base_fact_id: g0a.sha256_value(raw_decision)
        for base_fact_id, raw_decision in raw_decisions_by_id.items()
    }
    _assert_snapshot_unchanged(review_manifest_snapshot)
    _assert_snapshot_unchanged(decisions_snapshot)
    if len(review_rows) != EXPECTED_BASE_FACT_COUNT:
        raise ValueError(
            "selected context audit requires exactly "
            f"{EXPECTED_BASE_FACT_COUNT} base facts, found {len(review_rows)}"
        )

    rows = _selected_length_rows(
        review_by_id,
        decisions_by_id,
        raw_decision_sha256_by_id,
    )
    if len(rows) != EXPECTED_VARIANT_COUNT:
        raise ValueError(
            "selected context audit requires exactly "
            f"{EXPECTED_VARIANT_COUNT} variants, found {len(rows)}"
        )
    split_counts = Counter(row["split_assignment"] for row in rows)
    expected_variant_count = review_manifest.get("variant_count")
    if expected_variant_count != len(rows):
        raise ValueError(
            "review input manifest variant_count does not equal selected variant count"
        )

    records_payload = _jsonl_payload(rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "tool_version": TOOL_VERSION,
        "status": "completed_descriptive_only",
        "descriptive_only": True,
        "hard_threshold_applied": False,
        "gate_decision_produced": False,
        "measurement_definition": {
            "en_word_count": "count of non-empty Unicode-whitespace-delimited spans",
            "zh_nonspace_char_count": "count of Unicode code points for which str.isspace() is false; punctuation and embedded Latin text are included",
            "neutral_to_targeted_ratio": "Neutral count divided by Targeted count, rounded to 6 decimal places",
            "percentiles": "R-7 linear interpolation over per-variant values",
        },
        "inputs": {
            "review_input_manifest": dict(review_manifest_snapshot),
            "review_input": dict(review_manifest["review_input"]),
            "codex_decisions": {
                **decisions_snapshot,
                "record_count": len(decisions_by_id),
                "base_fact_ids_sha256": g0a.sha256_value(
                    sorted(decisions_by_id)
                ),
            },
        },
        "counts": {
            "base_fact_count": len(review_rows),
            "selected_variant_count": len(rows),
            "decision_counts": dict(sorted(decision_counts.items())),
            "selected_variant_counts_by_split": dict(sorted(split_counts.items())),
        },
        "distribution": _metric_summary(rows),
        "distribution_by_split": {
            split: _metric_summary(
                [row for row in rows if row["split_assignment"] == split]
            )
            for split in sorted(split_counts)
        },
        "outputs": {
            "selected_context_lengths": _payload_binding(
                output_records_path,
                records_payload,
                schema_version=RECORD_SCHEMA,
                record_count=len(rows),
                id_values=[
                    ":".join(
                        (
                            row["base_fact_id"],
                            row["distractor_id"],
                            row["selected_candidate_id"],
                        )
                    )
                    for row in rows
                ],
                id_digest_field="selected_variant_ids_sha256",
            )
        },
    }
    summary_payload = _json_payload(summary)
    _publish_output_dir(
        output_dir_path,
        {
            RECORDS_FILENAME: records_payload,
            SUMMARY_FILENAME: summary_payload,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-input-manifest", type=Path, required=True)
    parser.add_argument("--codex-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_selected_context_lengths(
        review_input_manifest_path=args.review_input_manifest,
        codex_decisions_path=args.codex_decisions,
        output_dir_path=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "descriptive_only": summary["descriptive_only"],
                "selected_variant_count": summary["counts"][
                    "selected_variant_count"
                ],
                "output_dir": str(Path(args.output_dir).resolve()),
                "output_records": str(
                    (Path(args.output_dir).resolve() / RECORDS_FILENAME)
                ),
                "output_summary": str(
                    (Path(args.output_dir).resolve() / SUMMARY_FILENAME)
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
