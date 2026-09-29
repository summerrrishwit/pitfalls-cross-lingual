#!/usr/bin/env python3
"""Materialize resolver-ready duplicate-closure adjudications from review shards.

The input shards intentionally contain only the semantic judgments made by a
Codex proxy reviewer.  This command replays the bound duplicate-candidate
manifest, requires exact and duplicate-free pair coverage, binds every output
row to the replayed candidate and endpoint hashes, and publishes the resulting
directory atomically.  It is offline and does not publish a closure resolution,
freeze a split, authorize perturbation, or claim human review.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "public-benchmark-duplicate-adjudication-materializer-v1"
MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-adjudication-materialization-v1"
)
SUMMARY_SCHEMA_VERSION = "public-benchmark-duplicate-adjudication-summary-v1"
OUTPUT_ADJUDICATIONS_NAME = "duplicate_closure_adjudications.jsonl"
OUTPUT_SUMMARY_NAME = "duplicate_adjudication_summary.json"
OUTPUT_MANIFEST_NAME = "duplicate_adjudication_materialization_manifest.json"
DEFAULT_REVIEW_METHOD = "offline-semantic-duplicate-closure-review-v1"
SHARD_FIELDS = frozenset(
    {"pair_id", "decision", "rationale", "reviewer_id", "confidence"}
)
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})


def _load_closure_tool() -> Any:
    script_path = Path(__file__).resolve().with_name(
        "materialize_public_benchmark_duplicate_closure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "public_benchmark_duplicate_closure_for_adjudications", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load duplicate-closure tool: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


closure_tool = _load_closure_tool()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def serialize_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _absolute_path(path: Path) -> Path:
    """Resolve parent directories without following the final path component."""

    value = Path(path)
    if not value.is_absolute():
        value = Path.cwd() / value
    return value.parent.resolve() / value.name


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, serialize_jsonl(rows))


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_reviewed_at(value: Any) -> str:
    reviewed_at = _required_string(value, "reviewed_at")
    try:
        parsed = dt.datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    return reviewed_at


def _read_shard_snapshot(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Review shard is not UTF-8: {path}") from exc
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        keys = set(value)
        if keys != SHARD_FIELDS:
            missing = sorted(SHARD_FIELDS - keys)
            extra = sorted(keys - SHARD_FIELDS)
            raise ValueError(
                f"Invalid review shard fields at {path}:{line_number}: "
                f"missing={missing}, extra={extra}"
            )
        pair_id = _required_string(
            value.get("pair_id"), f"{path}:{line_number}.pair_id"
        )
        decision = value.get("decision")
        if not isinstance(decision, str) or decision not in closure_tool.DECISIONS:
            raise ValueError(
                f"Invalid duplicate-closure decision at {path}:{line_number}: "
                f"{decision}"
            )
        confidence = value.get("confidence")
        if not isinstance(confidence, str) or confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"Invalid confidence at {path}:{line_number}: {confidence}"
            )
        rows.append(
            {
                "pair_id": pair_id,
                "decision": decision,
                "rationale": _required_string(
                    value.get("rationale"), f"{path}:{line_number}.rationale"
                ),
                "reviewer_id": _required_string(
                    value.get("reviewer_id"), f"{path}:{line_number}.reviewer_id"
                ),
                "confidence": confidence,
            }
        )
    binding = {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "byte_count": len(payload),
        "record_count": len(rows),
        "ordered_pair_ids_sha256": sha256_value(
            [str(row["pair_id"]) for row in rows]
        ),
    }
    return rows, binding


def _load_review_shards(
    paths: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not paths:
        raise ValueError("At least one --review-shard is required")
    resolved_paths = [Path(path).resolve() for path in paths]
    duplicates = sorted(
        path for path, count in Counter(resolved_paths).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Review shard paths are repeated: {duplicates}")

    rows: List[Dict[str, Any]] = []
    bindings: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for path in sorted(resolved_paths, key=str):
        shard_rows, binding = _read_shard_snapshot(path)
        for row in shard_rows:
            pair_id = str(row["pair_id"])
            previous = seen.get(pair_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate review decision for {pair_id}: {previous}, {path}"
                )
            seen[pair_id] = str(path)
            rows.append(row)
        bindings.append(binding)
    return rows, bindings


def _candidate_manifest_binding(
    path: Path, *, payload_before: bytes, payload_after: bytes, manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    if payload_before != payload_after:
        raise ValueError("Candidate manifest changed during deterministic replay")
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_bytes(payload_after),
        "byte_count": len(payload_after),
        "schema_version": manifest.get("schema_version"),
    }


def _published_file_binding(
    staged_path: Path,
    published_path: Path,
    *,
    record_count: Optional[int] = None,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(_absolute_path(published_path)),
        "sha256": sha256_file(staged_path),
        "byte_count": Path(staged_path).stat().st_size,
    }
    if record_count is not None:
        result["record_count"] = record_count
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _assert_input_binding_current(binding: Mapping[str, Any], label: str) -> None:
    path = Path(_required_string(binding.get("path"), f"{label}.path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != binding.get("byte_count"):
        raise ValueError(f"{label} changed during adjudication materialization")
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"{label} changed during adjudication materialization")


def _iter_file_bindings(
    value: Any, *, label: str
) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(value, dict):
        if {"path", "sha256", "byte_count"} <= set(value):
            yield label, value
            return
        for key, child in value.items():
            yield from _iter_file_bindings(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_file_bindings(child, label=f"{label}[{index}]")


def _publish_directory_no_replace(staged_dir: Path, output_dir: Path) -> None:
    """Atomically publish a directory without replacing a concurrent target."""

    staged_dir = _absolute_path(staged_dir)
    output_dir = _absolute_path(output_dir)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("renamex_np is unavailable; refusing unsafe publish")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staged_dir), os.fsencode(output_dir), 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("renameat2 is unavailable; refusing unsafe publish")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100,
            os.fsencode(staged_dir),
            -100,
            os.fsencode(output_dir),
            0x00000001,
        )
    elif os.name == "nt":
        # Windows os.rename already fails when the destination exists.
        os.rename(staged_dir, output_dir)
        return
    else:
        raise RuntimeError(
            "Atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(output_dir)
        )
    raise OSError(error_number, os.strerror(error_number), str(output_dir))


def _pair_endpoint_value(pair: Mapping[str, Any], side: str, field: str) -> str:
    endpoint = pair.get(side)
    if not isinstance(endpoint, dict):
        raise ValueError(f"Candidate {pair.get('pair_id')} lacks {side} endpoint")
    provenance = endpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Candidate {pair.get('pair_id')} lacks {side} endpoint provenance"
        )
    return _required_string(
        provenance.get(field), f"candidate {pair.get('pair_id')}.{side}.{field}"
    )


def _excluded_cohort_ids(
    *, pair: Mapping[str, Any], decision: str, cohort_ids: set[str]
) -> List[str]:
    if decision != "exclude_cohort":
        return []
    candidates = sorted(
        {
            str(pair[side]["provenance"].get("base_fact_id"))
            for side in ("left", "right")
            if pair[side]["provenance"].get("input_role") == "current_bundle"
            and pair[side]["provenance"].get("base_fact_id") in cohort_ids
        }
    )
    if not candidates:
        raise ValueError(
            f"exclude_cohort pair has no cohort endpoint: {pair.get('pair_id')}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Simplified exclude_cohort decision is ambiguous for multiple cohort "
            f"endpoints: {pair.get('pair_id')}"
        )
    return candidates


def _materialize_rows(
    *,
    candidate_pairs: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    reviewed_at: str,
    review_method: str,
) -> List[Dict[str, Any]]:
    candidate_by_id: Dict[str, Mapping[str, Any]] = {}
    for pair in candidate_pairs:
        pair_id = _required_string(pair.get("pair_id"), "candidate pair_id")
        if pair_id in candidate_by_id:
            raise ValueError(f"Candidate manifest repeats pair_id: {pair_id}")
        candidate_by_id[pair_id] = pair

    review_by_id = {str(row["pair_id"]): row for row in review_rows}
    missing = sorted(set(candidate_by_id) - set(review_by_id))
    extra = sorted(set(review_by_id) - set(candidate_by_id))
    if missing or extra:
        raise ValueError(
            "Review shards must exactly cover candidate pairs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    cohort_ids = {
        _required_string(row.get("base_fact_id"), "cohort base_fact_id")
        for row in cohort_rows
    }
    outputs: List[Dict[str, Any]] = []
    for pair in candidate_pairs:
        pair_id = str(pair["pair_id"])
        review = review_by_id[pair_id]
        decision = str(review["decision"])
        match_types = pair.get("match_types")
        if not isinstance(match_types, list) or any(
            not isinstance(value, str) for value in match_types
        ):
            raise ValueError(f"Candidate match_types are invalid: {pair_id}")
        if decision == "distinct" and closure_tool.EXACT_DUPLICATE_MATCH_TYPES.intersection(
            match_types
        ):
            raise ValueError(f"Exact duplicate candidate cannot be distinct: {pair_id}")
        outputs.append(
            {
                "schema_version": closure_tool.ADJUDICATION_SCHEMA_VERSION,
                "pair_id": pair_id,
                "candidate_row_sha256": closure_tool.sha256_value(pair),
                "left_audit_record_id": _pair_endpoint_value(
                    pair, "left", "audit_record_id"
                ),
                "left_input_record_sha256": _pair_endpoint_value(
                    pair, "left", "input_record_sha256"
                ),
                "right_audit_record_id": _pair_endpoint_value(
                    pair, "right", "audit_record_id"
                ),
                "right_input_record_sha256": _pair_endpoint_value(
                    pair, "right", "input_record_sha256"
                ),
                "decision": decision,
                "excluded_cohort_base_fact_ids": _excluded_cohort_ids(
                    pair=pair, decision=decision, cohort_ids=cohort_ids
                ),
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "reviewer_id": review["reviewer_id"],
                "review_method": review_method,
                "reviewed_at": reviewed_at,
                "rationale": review["rationale"],
                "confidence": review["confidence"],
            }
        )
    return outputs


def materialize(
    *,
    candidate_manifest_path: Path,
    review_shard_paths: Sequence[Path],
    output_dir: Path,
    reviewed_at: str,
    review_method: str = DEFAULT_REVIEW_METHOD,
) -> Dict[str, Any]:
    """Validate shards and atomically publish resolver-ready adjudications."""

    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    output_dir = _absolute_path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    reviewed_at = _validate_reviewed_at(reviewed_at)
    review_method = _required_string(review_method, "review_method")

    manifest_before = candidate_manifest_path.read_bytes()
    candidate_manifest, cohort_rows, candidate_pairs = (
        closure_tool._validate_candidate_manifest(candidate_manifest_path)
    )
    manifest_after = candidate_manifest_path.read_bytes()
    candidate_binding = _candidate_manifest_binding(
        candidate_manifest_path,
        payload_before=manifest_before,
        payload_after=manifest_after,
        manifest=candidate_manifest,
    )
    review_rows, shard_bindings = _load_review_shards(review_shard_paths)
    full_rows = _materialize_rows(
        candidate_pairs=candidate_pairs,
        cohort_rows=cohort_rows,
        review_rows=review_rows,
        reviewed_at=reviewed_at,
        review_method=review_method,
    )

    decision_counts = {
        decision: Counter(row["decision"] for row in full_rows).get(decision, 0)
        for decision in sorted(closure_tool.DECISIONS)
    }
    confidence_counts = {
        confidence: Counter(row["confidence"] for row in full_rows).get(confidence, 0)
        for confidence in sorted(CONFIDENCE_LEVELS)
    }
    reviewer_id_counts = dict(
        sorted(Counter(str(row["reviewer_id"]) for row in full_rows).items())
    )
    pair_ids = [str(pair["pair_id"]) for pair in candidate_pairs]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        staged_adjudications = staged_dir / OUTPUT_ADJUDICATIONS_NAME
        staged_summary = staged_dir / OUTPUT_SUMMARY_NAME
        staged_manifest = staged_dir / OUTPUT_MANIFEST_NAME
        write_jsonl(staged_adjudications, full_rows)

        # Validate the exact bytes handed to the downstream resolver before publish.
        closure_tool._validate_adjudications(
            staged_adjudications,
            candidate_pairs,
            {
                _required_string(row.get("base_fact_id"), "cohort base_fact_id")
                for row in cohort_rows
            },
        )
        resolver_preflight_dir = staged_dir / ".resolver-preflight"
        try:
            closure_tool.resolve_candidates(
                candidate_manifest_path=candidate_manifest_path,
                adjudications_path=staged_adjudications,
                output_dir=resolver_preflight_dir,
            )
        finally:
            shutil.rmtree(resolver_preflight_dir, ignore_errors=True)

        published_adjudications = output_dir / OUTPUT_ADJUDICATIONS_NAME
        published_summary = output_dir / OUTPUT_SUMMARY_NAME
        published_manifest = output_dir / OUTPUT_MANIFEST_NAME
        adjudication_binding = _published_file_binding(
            staged_adjudications,
            published_adjudications,
            record_count=len(full_rows),
            schema_version=closure_tool.ADJUDICATION_SCHEMA_VERSION,
        )
        counts = {
            "candidate_pair_count": len(candidate_pairs),
            "review_shard_count": len(shard_bindings),
            "review_row_count": len(review_rows),
            "output_adjudication_count": len(full_rows),
            "cross_split_candidate_count": sum(
                pair.get("cross_split") is True for pair in candidate_pairs
            ),
            "exact_match_candidate_count": sum(
                bool(
                    closure_tool.EXACT_DUPLICATE_MATCH_TYPES.intersection(
                        pair.get("match_types", [])
                    )
                )
                for pair in candidate_pairs
            ),
            "excluded_cohort_base_fact_count": len(
                {
                    base_fact_id
                    for row in full_rows
                    for base_fact_id in row["excluded_cohort_base_fact_ids"]
                }
            ),
        }
        summary: Dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "resolver_ready_adjudications_materialized_non_human",
            "candidate_manifest": candidate_binding,
            "review_shards": shard_bindings,
            "adjudications": adjudication_binding,
            "counts": counts,
            "decision_counts": decision_counts,
            "confidence_counts": confidence_counts,
            "reviewer_id_counts": reviewer_id_counts,
            "ordered_pair_ids_sha256": sha256_value(pair_ids),
            "review": {
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "review_method": review_method,
                "reviewed_at": reviewed_at,
            },
            "validation": {
                "candidate_manifest_deterministic_replay_passed": True,
                "exact_pair_coverage_passed": True,
                "duplicate_pair_ids_absent": True,
                "exact_match_constraint_passed": True,
                "downstream_resolver_contract_passed": True,
                "transitive_input_bindings_current_at_publish": True,
                "atomic_no_replace_directory_publish": True,
            },
            "safety_contract": {
                "network_or_model_used": False,
                "human_gold": False,
                "closure_resolved": False,
                "closure_resolver_preflight_performed": True,
                "closure_resolution_published": False,
                "review_freeze_emitted": False,
                "split_freeze_emitted": False,
                "perturbation_authorized": False,
                "perturbation_performed": False,
            },
        }
        write_json(staged_summary, summary)
        summary_binding = _published_file_binding(
            staged_summary,
            published_summary,
            schema_version=SUMMARY_SCHEMA_VERSION,
        )
        output_manifest: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "resolver_ready_adjudications_materialized_non_human",
            "inputs": {
                "candidate_manifest": candidate_binding,
                "review_shards": shard_bindings,
            },
            "outputs": {
                "adjudications": adjudication_binding,
                "summary": summary_binding,
            },
            "counts": counts,
            "decision_counts": decision_counts,
            "confidence_counts": confidence_counts,
            "reviewer_id_counts": reviewer_id_counts,
            "review": summary["review"],
            "validation": summary["validation"],
            "safety_contract": summary["safety_contract"],
        }
        write_json(staged_manifest, output_manifest)

        _assert_input_binding_current(candidate_binding, "candidate manifest")
        for label, dependency_binding in _iter_file_bindings(
            candidate_manifest, label="candidate manifest"
        ):
            _assert_input_binding_current(dependency_binding, label)
        for index, shard_binding in enumerate(shard_bindings, start=1):
            _assert_input_binding_current(shard_binding, f"review shard {index}")

        _publish_directory_no_replace(staged_dir, output_dir)
        published = True
        return {
            "manifest_path": str(published_manifest),
            "manifest_sha256": sha256_file(published_manifest),
            "summary_path": str(published_summary),
            "adjudications_path": str(published_adjudications),
            "adjudications_sha256": sha256_file(published_adjudications),
            "adjudication_count": len(full_rows),
            "decision_counts": decision_counts,
            "human_gold": False,
            "perturbation_authorized": False,
        }
    finally:
        if not published:
            shutil.rmtree(staged_dir, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument(
        "--review-shard",
        type=Path,
        action="append",
        required=True,
        help="Simplified review JSONL; repeat for every shard.",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--reviewed-at",
        required=True,
        help="Shared timezone-aware ISO-8601 review timestamp.",
    )
    result.add_argument("--review-method", default=DEFAULT_REVIEW_METHOD)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = materialize(
        candidate_manifest_path=args.candidate_manifest,
        review_shard_paths=args.review_shard,
        output_dir=args.output_dir,
        reviewed_at=args.reviewed_at,
        review_method=args.review_method,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
