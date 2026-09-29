#!/usr/bin/env python3
"""Create and apply review packages for a provisional public-benchmark bundle.

This tool deliberately stops at review staging.  It never rewrites the source
artifacts, never emits a frozen canonical fact, and does not implement a freeze
operation.  The three commands form an auditable chain:

``scope``
    Select base facts and bind the selection to each source artifact's
    normalized local path, byte-level SHA-256, schema, size, and record count.
    Review-sample scopes additionally bind every sample row to its source
    review-queue row.
``export``
    Materialize self-contained review items, including every duplicate-cluster
    member and its canonical-JSON record hash, plus a decisions template.
``apply``
    Validate Codex-proxy decisions and emit review staging records.  Missing
    decisions are permitted only with ``--allow-partial`` and remain missing.

Version 2 supports ``accept``, ``reject``, ``defer``, and a request-only
``revise`` outcome.  Every active outcome must explicitly dispose every
duplicate-cluster member, and every supplied target is checked against its
exported metadata.  A revise decision cannot carry or apply rewritten fact
fields: the output stays pending review until a separate provenance-preserving
revision and re-clustering workflow is designed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "public-benchmark-review-tool-v2"
SCOPE_SCHEMA_VERSION = "public-benchmark-review-scope-manifest-v2"
SCOPE_SUMMARY_SCHEMA_VERSION = "public-benchmark-review-scope-summary-v2"
EXPORT_ITEM_SCHEMA_VERSION = "public-benchmark-review-export-item-v2"
DECISION_SCHEMA_VERSION = "public-benchmark-review-decision-v2"
EXPORT_MANIFEST_SCHEMA_VERSION = "public-benchmark-review-export-manifest-v2"
EXPORT_SUMMARY_SCHEMA_VERSION = "public-benchmark-review-export-summary-v2"
STAGING_SCHEMA_VERSION = "public-benchmark-review-staging-record-v2"
APPLY_MANIFEST_SCHEMA_VERSION = "public-benchmark-review-apply-manifest-v2"
APPLY_SUMMARY_SCHEMA_VERSION = "public-benchmark-review-apply-summary-v2"
REVIEW_SAMPLE_SCHEMA_VERSION = "provisional-semantic-review-sample-v1"
FULL_PUBLIC_BENCHMARK_COUNT = 8969

_COMPOSITE_AUTHORITY_TOOL: Optional[Any] = None


def _load_composite_authority_tool() -> Any:
    global _COMPOSITE_AUTHORITY_TOOL
    if _COMPOSITE_AUTHORITY_TOOL is None:
        path = Path(__file__).resolve().with_name(
            "build_full_public_benchmark_fact_review_composite.py"
        )
        spec = importlib.util.spec_from_file_location(
            "full_fact_review_composite_authority", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load composite authority tool: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _COMPOSITE_AUTHORITY_TOOL = module
    return _COMPOSITE_AUTHORITY_TOOL

SOURCE_ARTIFACTS = {
    "behavior_bundle": (
        "behavior_input_bundle.jsonl",
        "factual-perturbation-input-bundle-v1",
        "base_fact_id",
    ),
    "review_queue": (
        "review_queue.jsonl",
        "provisional-semantic-review-v1",
        "base_fact_id",
    ),
    "base_fact_clusters": (
        "base_fact_clusters.jsonl",
        "provisional-base-fact-cluster-v1",
        "base_fact_id",
    ),
    "provisional_triples": (
        "provisional_triples.jsonl",
        "provisional-factual-triple-v1",
        "candidate_id",
    ),
}

ALLOWED_DECISIONS = {"accept", "reject", "defer", "revise"}
ALLOWED_TARGET_DECISIONS = {"accept", "reject"}
SOURCE_BINDING_FIELDS = {
    "path",
    "sha256",
    "byte_count",
    "record_count",
    "schema_version",
}
REVIEW_SAMPLE_ROW_BINDING_FIELDS = {
    "sample_index",
    "base_fact_id",
    "sample_row_sha256",
    "review_queue_row_sha256",
}
EXPORT_ARTIFACT_BINDING_FIELDS = {
    "path",
    "sha256",
    "record_count",
    "schema_version",
}
DECISION_ALLOWED_FIELDS = {
    "schema_version",
    "scope_id",
    "base_fact_id",
    "review_item_sha256",
    "decision",
    "human_gold",
    "review_provenance",
    "member_reviews",
    "alias_reviews",
    "distractor_reviews",
    "notes",
}
FORBIDDEN_PROMOTION_FIELDS = {
    "canonical_status",
    "bundle_status",
    "evidence_tier",
    "semantic_review_complete",
    "split_status",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for record in records:
                handle.write(canonical_json_bytes(record))
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    if not records and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {path}")
    return records


def _require_nonempty_string(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing non-empty {field}")
    return text


def _unique_index(
    records: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Mapping[str, Any]]:
    output: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Expected object in {label} at index {index}")
        key = _require_nonempty_string(record.get(field), f"{label}[{index}].{field}")
        if key in output:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        output[key] = record
    return output


def source_paths(
    source_dir: Optional[Path] = None,
    overrides: Optional[Mapping[str, Path]] = None,
) -> Dict[str, Path]:
    supplied = dict(overrides or {})
    unknown = set(supplied) - set(SOURCE_ARTIFACTS)
    if unknown:
        raise ValueError(f"Unknown source artifact roles: {sorted(unknown)}")
    output: Dict[str, Path] = {}
    for role, (filename, _, _) in SOURCE_ARTIFACTS.items():
        candidate = supplied.get(role)
        if candidate is None:
            if source_dir is None:
                raise ValueError(
                    f"No path supplied for {role}; provide source_dir or every artifact path"
                )
            candidate = source_dir / filename
        output[role] = Path(candidate).resolve()
    return output


def _validate_provisional_status(role: str, record: Mapping[str, Any], key: str) -> None:
    if record.get("canonical_status") != "pending_review":
        raise ValueError(f"{role} {key} is not pending_review")
    if record.get("human_gold") is not False:
        raise ValueError(f"{role} {key} must have human_gold=false")
    if record.get("evidence_tier") != "provisional_single_model":
        raise ValueError(f"{role} {key} is not provisional_single_model")
    if role == "behavior_bundle" and record.get("bundle_status") != "draft_pending_review":
        raise ValueError(f"behavior_bundle {key} is not draft_pending_review")


def load_source_artifacts(
    paths: Mapping[str, Path],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Mapping[str, Any]]]]:
    """Load and strictly cross-check the four provisional source artifacts."""

    rows: Dict[str, List[Dict[str, Any]]] = {}
    indexes: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for role, (_, expected_schema, key_field) in SOURCE_ARTIFACTS.items():
        artifact_rows = read_jsonl(paths[role])
        for index, record in enumerate(artifact_rows):
            if record.get("schema_version") != expected_schema:
                raise ValueError(
                    f"Unsupported {role} schema at row {index}: "
                    f"{record.get('schema_version')!r}"
                )
            key = _require_nonempty_string(record.get(key_field), f"{role}.{key_field}")
            _validate_provisional_status(role, record, key)
        rows[role] = artifact_rows
        indexes[role] = _unique_index(artifact_rows, key_field, role)

    behavior_ids = set(indexes["behavior_bundle"])
    review_ids = set(indexes["review_queue"])
    cluster_ids = set(indexes["base_fact_clusters"])
    triple_ids_by_fact: Dict[str, List[str]] = defaultdict(list)
    for candidate_id, triple in indexes["provisional_triples"].items():
        fact_id = _require_nonempty_string(
            triple.get("base_fact_id"), f"provisional_triples[{candidate_id}].base_fact_id"
        )
        triple_ids_by_fact[fact_id].append(candidate_id)
    triple_fact_ids = set(triple_ids_by_fact)
    if not (behavior_ids == review_ids == cluster_ids == triple_fact_ids):
        raise ValueError(
            "Source artifact base_fact_id universes differ: "
            f"behavior={len(behavior_ids)}, review={len(review_ids)}, "
            f"clusters={len(cluster_ids)}, triples={len(triple_fact_ids)}"
        )

    for fact_id in sorted(cluster_ids):
        cluster = indexes["base_fact_clusters"][fact_id]
        review = indexes["review_queue"][fact_id]
        behavior = indexes["behavior_bundle"][fact_id]
        members = cluster.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Cluster {fact_id} has no members")
        member_index = _unique_index(members, "candidate_id", f"cluster {fact_id} members")
        expected_members = set(triple_ids_by_fact[fact_id])
        if set(member_index) != expected_members:
            raise ValueError(f"Cluster {fact_id} members do not match provisional triples")
        if cluster.get("member_count") != len(expected_members):
            raise ValueError(f"Cluster {fact_id} member_count is inconsistent")
        review_members = review.get("member_candidate_ids")
        if not isinstance(review_members, list) or len(review_members) != len(set(review_members)):
            raise ValueError(f"Review row {fact_id} has invalid member_candidate_ids")
        if set(str(value) for value in review_members) != expected_members:
            raise ValueError(f"Review row {fact_id} members do not match cluster")
        representative = _require_nonempty_string(
            cluster.get("representative_candidate_id"),
            f"cluster {fact_id}.representative_candidate_id",
        )
        if representative not in expected_members:
            raise ValueError(f"Cluster {fact_id} representative is not a member")
        if behavior.get("candidate_id") != representative:
            raise ValueError(f"Behavior row {fact_id} does not bind the cluster representative")
        for candidate_id, member in member_index.items():
            triple = indexes["provisional_triples"][candidate_id]
            provenance = triple.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"Triple {candidate_id} has no provenance")
            if member.get("input_record_sha256") != provenance.get("input_record_sha256"):
                raise ValueError(f"Cluster member {candidate_id} input provenance mismatch")

    return rows, indexes


def _artifact_bindings(
    paths: Mapping[str, Path], rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for role, (_, expected_schema, _) in SOURCE_ARTIFACTS.items():
        path = paths[role]
        output[role] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
            "record_count": len(rows[role]),
            "schema_version": expected_schema,
        }
    return output


def _read_explicit_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError("base-fact IDs JSON must be a list")
        raw_values = value
    elif path.suffix.casefold() == ".jsonl":
        raw_values = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"base-fact IDs JSONL row {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"base-fact IDs JSONL row {line_number} must be an object"
                    )
                base_fact_id = value.get("base_fact_id") or value.get(
                    "source_base_fact_id"
                )
                if not isinstance(base_fact_id, str) or not base_fact_id.strip():
                    raise ValueError(
                        f"base-fact IDs JSONL row {line_number} lacks base_fact_id "
                        "or source_base_fact_id"
                    )
                raw_values.append(base_fact_id)
    else:
        raw_values = [line for line in path.read_text(encoding="utf-8").splitlines()]
    return [str(value).strip() for value in raw_values if str(value).strip()]


def _validate_selected_ids(
    selected: Sequence[Any], available_ids: Sequence[str]
) -> List[str]:
    ids = [_require_nonempty_string(value, "base_fact_id") for value in selected]
    if not ids:
        raise ValueError("Review scope cannot be empty")
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate base_fact_id in scope selection: {duplicates[:10]}")
    unknown = sorted(set(ids) - set(available_ids))
    if unknown:
        raise ValueError(f"Unknown base_fact_id in scope selection: {unknown[:10]}")
    return ids


def _scope_id(
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
    selection_source: Mapping[str, Any],
) -> str:
    fingerprint = {
        "tool_version": TOOL_VERSION,
        "source_artifacts": {
            role: {
                field: artifact_bindings[role][field]
                for field in sorted(SOURCE_BINDING_FIELDS)
            }
            for role in sorted(artifact_bindings)
        },
        "base_fact_ids": list(selected_ids),
        "selection_source": selection_source,
    }
    return f"review_scope_{sha256_value(fingerprint)[:20]}"


def _review_sample_selection(
    sample: Sequence[Mapping[str, Any]],
    review_queue_index: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Validate that every sample row is an exact queue-row derivative."""

    for index, row in enumerate(sample):
        if row.get("schema_version") != REVIEW_SAMPLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported review sample schema at row {index}")
        if row.get("canonical_status") != "pending_review" or row.get("human_gold") is not False:
            raise ValueError(f"Review sample row {index} is not provisional")

    selected = _validate_selected_ids(
        [row.get("base_fact_id") for row in sample], review_queue_index.keys()
    )
    row_bindings: List[Dict[str, Any]] = []
    for index, (base_fact_id, sample_row) in enumerate(zip(selected, sample)):
        queue_row = review_queue_index[base_fact_id]
        mismatched_fields = sorted(
            field
            for field, expected in queue_row.items()
            if field != "schema_version"
            and (field not in sample_row or sample_row[field] != expected)
        )
        if mismatched_fields:
            raise ValueError(
                f"Review sample row {index} ({base_fact_id}) does not match "
                f"review_queue fields: {mismatched_fields[:10]}"
            )
        row_bindings.append(
            {
                "sample_index": index,
                "base_fact_id": base_fact_id,
                "sample_row_sha256": sha256_value(sample_row),
                "review_queue_row_sha256": sha256_value(queue_row),
            }
        )
    return selected, row_bindings


def create_scope(
    *,
    output_dir: Path,
    artifact_paths: Mapping[str, Path],
    base_fact_ids: Optional[Sequence[str]] = None,
    review_sample_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Freeze a review scope against immutable source-artifact hashes."""

    if (base_fact_ids is None) == (review_sample_path is None):
        raise ValueError("Choose exactly one scope source: base_fact_ids or review_sample_path")
    paths = {role: Path(path).resolve() for role, path in artifact_paths.items()}
    rows, indexes = load_source_artifacts(paths)

    if review_sample_path is not None:
        sample_path = Path(review_sample_path).resolve()
        sample = read_jsonl(sample_path)
        selected, row_bindings = _review_sample_selection(
            sample, indexes["review_queue"]
        )
        selection_source: Dict[str, Any] = {
            "mode": "review_sample",
            "path": str(sample_path),
            "sha256": sha256_file(sample_path),
            "byte_count": sample_path.stat().st_size,
            "record_count": len(sample),
            "schema_version": REVIEW_SAMPLE_SCHEMA_VERSION,
            "row_bindings": row_bindings,
            "row_bindings_sha256": sha256_value(row_bindings),
        }
    else:
        selected = _validate_selected_ids(
            list(base_fact_ids or []), indexes["behavior_bundle"].keys()
        )
        selection_source = {
            "mode": "explicit_base_fact_ids",
            "record_count": len(selected),
            "base_fact_ids_sha256": sha256_value(selected),
        }

    bindings = _artifact_bindings(paths, rows)
    scope_id = _scope_id(bindings, selected, selection_source)
    manifest = {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope_id,
        "scope_status": "frozen_review_scope",
        "canonical_freeze_authorized": False,
        "selection_source": selection_source,
        "base_fact_count": len(selected),
        "base_fact_ids": selected,
        "source_artifacts": bindings,
        "safety_contract": {
            "source_artifacts_are_read_only": True,
            "canonical_status_required": "pending_review",
            "human_gold_required": False,
            "automatic_promotion_allowed": False,
            "freeze_command_implemented": False,
        },
    }
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "review_scope_manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "schema_version": SCOPE_SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope_id,
        "status": "review_scope_created",
        "selection_mode": selection_source["mode"],
        "base_fact_count": len(selected),
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
        },
        "canonical_freeze_performed": False,
    }
    summary_path = output_dir / "review_scope_summary.json"
    write_json(summary_path, summary)
    return {
        "scope_id": scope_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "base_fact_count": len(selected),
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_selection_source_manifest(
    selection_source: Mapping[str, Any], base_fact_ids: Sequence[str]
) -> None:
    mode = selection_source.get("mode")
    if mode == "explicit_base_fact_ids":
        if set(selection_source) != {
            "mode",
            "record_count",
            "base_fact_ids_sha256",
        }:
            raise ValueError("Explicit selection source fields are invalid")
        if selection_source.get("record_count") != len(base_fact_ids):
            raise ValueError("Explicit selection record_count is inconsistent")
        if selection_source.get("base_fact_ids_sha256") != sha256_value(base_fact_ids):
            raise ValueError("Explicit selection base_fact_ids_sha256 is inconsistent")
        return
    if mode != "review_sample":
        raise ValueError(f"Unsupported review scope selection mode: {mode!r}")
    if set(selection_source) != {
        "mode",
        "path",
        "sha256",
        "byte_count",
        "record_count",
        "schema_version",
        "row_bindings",
        "row_bindings_sha256",
    }:
        raise ValueError("Review sample selection source fields are invalid")

    sample_path = Path(
        _require_nonempty_string(selection_source.get("path"), "selection_source.path")
    )
    if not sample_path.is_absolute() or str(sample_path.resolve()) != str(sample_path):
        raise ValueError("Review sample path must be absolute and normalized")
    if not _is_sha256(selection_source.get("sha256")):
        raise ValueError("Review sample SHA-256 binding is invalid")
    if selection_source.get("schema_version") != REVIEW_SAMPLE_SCHEMA_VERSION:
        raise ValueError("Review sample selection schema is invalid")
    if selection_source.get("record_count") != len(base_fact_ids):
        raise ValueError("Review sample selection record_count is inconsistent")
    if (
        not isinstance(selection_source.get("byte_count"), int)
        or selection_source["byte_count"] <= 0
    ):
        raise ValueError("Review sample selection byte_count is invalid")

    row_bindings = selection_source.get("row_bindings")
    if not isinstance(row_bindings, list) or len(row_bindings) != len(base_fact_ids):
        raise ValueError("Review sample row bindings do not cover the scope")
    for index, (base_fact_id, binding) in enumerate(zip(base_fact_ids, row_bindings)):
        if not isinstance(binding, dict) or set(binding) != REVIEW_SAMPLE_ROW_BINDING_FIELDS:
            raise ValueError(f"Review sample row binding {index} has invalid fields")
        if binding.get("sample_index") != index:
            raise ValueError(f"Review sample row binding {index} has the wrong sample_index")
        if binding.get("base_fact_id") != base_fact_id:
            raise ValueError(f"Review sample row binding {index} has the wrong base_fact_id")
        for field in ("sample_row_sha256", "review_queue_row_sha256"):
            if not _is_sha256(binding.get(field)):
                raise ValueError(f"Review sample row binding {index} has invalid {field}")
    if selection_source.get("row_bindings_sha256") != sha256_value(row_bindings):
        raise ValueError("Review sample row_bindings_sha256 is inconsistent")


def _validate_scope_manifest(manifest: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "tool_version",
        "scope_id",
        "scope_status",
        "canonical_freeze_authorized",
        "selection_source",
        "base_fact_count",
        "base_fact_ids",
        "source_artifacts",
        "safety_contract",
    }
    if set(manifest) != expected_fields:
        raise ValueError("Review scope manifest fields are invalid")
    if manifest.get("schema_version") != SCOPE_SCHEMA_VERSION:
        raise ValueError("Unsupported review scope manifest schema")
    if manifest.get("tool_version") != TOOL_VERSION:
        raise ValueError("Unsupported review scope tool version")
    if manifest.get("scope_status") != "frozen_review_scope":
        raise ValueError("Review scope manifest is not frozen")
    if manifest.get("canonical_freeze_authorized") is not False:
        raise ValueError("Review scope must not authorize canonical freeze")
    if manifest.get("safety_contract") != {
        "source_artifacts_are_read_only": True,
        "canonical_status_required": "pending_review",
        "human_gold_required": False,
        "automatic_promotion_allowed": False,
        "freeze_command_implemented": False,
    }:
        raise ValueError("Review scope safety contract is invalid")
    ids = manifest.get("base_fact_ids")
    if not isinstance(ids, list):
        raise ValueError("Review scope manifest has no base_fact_ids list")
    _validate_selected_ids(ids, ids)
    if manifest.get("base_fact_count") != len(ids):
        raise ValueError("Review scope base_fact_count is inconsistent")
    bindings = manifest.get("source_artifacts")
    if not isinstance(bindings, dict) or set(bindings) != set(SOURCE_ARTIFACTS):
        raise ValueError("Review scope source_artifacts are incomplete")
    for role, (_, expected_schema, _) in SOURCE_ARTIFACTS.items():
        binding = bindings[role]
        if not isinstance(binding, dict) or set(binding) != SOURCE_BINDING_FIELDS:
            raise ValueError(f"Review scope source binding is invalid for {role}")
        path = Path(_require_nonempty_string(binding.get("path"), f"{role}.path"))
        if not path.is_absolute() or str(path.resolve()) != str(path):
            raise ValueError(f"Review scope source path is not absolute and normalized: {path}")
        if not _is_sha256(binding.get("sha256")):
            raise ValueError(f"Review scope source SHA-256 is invalid for {role}")
        if not isinstance(binding.get("byte_count"), int) or binding["byte_count"] <= 0:
            raise ValueError(f"Review scope source byte_count is invalid for {role}")
        if not isinstance(binding.get("record_count"), int) or binding["record_count"] <= 0:
            raise ValueError(f"Review scope source record_count is invalid for {role}")
        if binding.get("schema_version") != expected_schema:
            raise ValueError(f"Review scope source schema is invalid for {role}")
    selection_source = manifest.get("selection_source")
    if not isinstance(selection_source, dict):
        raise ValueError("Review scope has no selection_source")
    _validate_selection_source_manifest(selection_source, ids)
    expected_id = _scope_id(bindings, ids, selection_source)
    if manifest.get("scope_id") != expected_id:
        raise ValueError("Review scope_id does not match manifest content")


def _scope_artifact_paths(manifest: Mapping[str, Any]) -> Dict[str, Path]:
    bindings = manifest["source_artifacts"]
    paths: Dict[str, Path] = {}
    for role in SOURCE_ARTIFACTS:
        binding = bindings[role]
        if not isinstance(binding, dict):
            raise ValueError(f"Invalid source binding for {role}")
        path = Path(_require_nonempty_string(binding.get("path"), f"{role}.path"))
        if not path.is_absolute():
            raise ValueError(f"Scope artifact path must be absolute: {path}")
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"Stale SHA-256 binding for {role}: {path}")
        if path.stat().st_size != binding.get("byte_count"):
            raise ValueError(f"Stale byte-count binding for {role}: {path}")
        paths[role] = path
    return paths


def _verify_scope_selection_source(
    manifest: Mapping[str, Any],
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    selection_source = manifest["selection_source"]
    if selection_source["mode"] != "review_sample":
        return
    sample_path = Path(selection_source["path"])
    if sha256_file(sample_path) != selection_source["sha256"]:
        raise ValueError(f"Stale SHA-256 binding for review_sample: {sample_path}")
    if sample_path.stat().st_size != selection_source["byte_count"]:
        raise ValueError(f"Stale byte-count binding for review_sample: {sample_path}")
    sample = read_jsonl(sample_path)
    selected, row_bindings = _review_sample_selection(
        sample, indexes["review_queue"]
    )
    if selected != list(manifest["base_fact_ids"]):
        raise ValueError("Review sample no longer selects the frozen scope in order")
    if len(sample) != selection_source["record_count"]:
        raise ValueError("Stale record-count binding for review_sample")
    if row_bindings != selection_source["row_bindings"]:
        raise ValueError("Review sample row bindings differ from the frozen review scope")


def _alias_targets(base_fact_id: str, behavior: Mapping[str, Any]) -> List[Dict[str, str]]:
    aliases = behavior.get("answer_aliases_en")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"Behavior row {base_fact_id} has no answer_aliases_en")
    output = []
    seen = set()
    for value in aliases:
        text = _require_nonempty_string(value, f"behavior {base_fact_id} answer alias")
        normalized = normalize_text(text)
        if normalized in seen:
            raise ValueError(f"Behavior row {base_fact_id} has duplicate normalized aliases")
        seen.add(normalized)
        alias_id = "alias_" + hashlib.sha256(
            f"{base_fact_id}\u241f{normalized}".encode("utf-8")
        ).hexdigest()[:16]
        output.append({"alias_id": alias_id, "text_en": text})
    answer = normalize_text(behavior.get("answer_en"))
    if not answer or answer not in seen:
        raise ValueError(f"Behavior row {base_fact_id} answer_en is not covered by aliases")
    return output


def _distractor_targets(
    base_fact_id: str, behavior: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    candidates = behavior.get("distractor_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"Behavior row {base_fact_id} has invalid distractor_candidates")
    output = []
    seen = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Behavior row {base_fact_id} distractor {index} is invalid")
        distractor_id = _require_nonempty_string(
            candidate.get("distractor_id"), f"behavior {base_fact_id} distractor_id"
        )
        if distractor_id in seen:
            raise ValueError(f"Behavior row {base_fact_id} has duplicate distractor_id")
        seen.add(distractor_id)
        text = _require_nonempty_string(
            candidate.get("text_en") or candidate.get("answer_en"),
            f"behavior {base_fact_id} distractor text",
        )
        output.append(
            {
                "distractor_id": distractor_id,
                "text_en": text,
                "source_choice_index": candidate.get("source_choice_index"),
            }
        )
    return output


def _build_export_item(
    scope_id: str,
    base_fact_id: str,
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    behavior = indexes["behavior_bundle"][base_fact_id]
    review = indexes["review_queue"][base_fact_id]
    cluster = indexes["base_fact_clusters"][base_fact_id]
    members = []
    for member in cluster["members"]:
        candidate_id = str(member["candidate_id"])
        triple = indexes["provisional_triples"][candidate_id]
        members.append(
            {
                "candidate_id": candidate_id,
                "triple_record_sha256": sha256_value(triple),
                "source_input_record_sha256": member["input_record_sha256"],
                "triple": triple,
            }
        )
    return {
        "schema_version": EXPORT_ITEM_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope_id,
        "base_fact_id": base_fact_id,
        "canonical_status": "pending_review",
        "human_gold": False,
        "source_row_bindings": {
            "behavior_row_sha256": sha256_value(behavior),
            "review_queue_row_sha256": sha256_value(review),
            "cluster_row_sha256": sha256_value(cluster),
        },
        "review_context": {
            "behavior_row": behavior,
            "review_queue_row": review,
            "cluster_row": cluster,
        },
        "member_count": len(members),
        "members": members,
        "review_targets": {
            "member_candidate_ids": [member["candidate_id"] for member in members],
            "aliases": _alias_targets(base_fact_id, behavior),
            "distractors": _distractor_targets(base_fact_id, behavior),
        },
        "review_contract": {
            "supported_decisions": sorted(ALLOWED_DECISIONS),
            "revise_outcome_supported": True,
            "revision_application_supported": False,
            "every_active_outcome_requires_all_member_reviews": True,
            "accept_requires_all_member_reviews": True,
            "accept_requires_complete_alias_review": True,
            "accept_requires_complete_distractor_review": True,
            "required_reviewer_type": "codex_proxy",
            "human_gold": False,
            "automatic_promotion_allowed": False,
        },
    }


def _decision_template(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "scope_id": item["scope_id"],
        "base_fact_id": item["base_fact_id"],
        "review_item_sha256": sha256_value(item),
        "decision": None,
        "human_gold": False,
        "review_provenance": {
            "reviewer_type": "codex_proxy",
            "reviewer_id": None,
            "review_method": None,
            "reviewed_at": None,
        },
        "member_reviews": [
            {
                "candidate_id": member["candidate_id"],
                "triple_record_sha256": member["triple_record_sha256"],
                "decision": None,
            }
            for member in item["members"]
        ],
        "alias_reviews": [
            {**target, "decision": None}
            for target in item["review_targets"]["aliases"]
        ],
        "distractor_reviews": [
            {**target, "decision": None}
            for target in item["review_targets"]["distractors"]
        ],
        "notes": None,
    }


def export_scope(*, scope_manifest_path: Path, output_dir: Path) -> Dict[str, Any]:
    scope_manifest_path = Path(scope_manifest_path).resolve()
    scope = read_json(scope_manifest_path)
    _validate_scope_manifest(scope)
    paths = _scope_artifact_paths(scope)
    rows, indexes = load_source_artifacts(paths)
    _verify_scope_selection_source(scope, indexes)
    for role, binding in scope["source_artifacts"].items():
        if len(rows[role]) != binding.get("record_count"):
            raise ValueError(f"Stale record-count binding for {role}")

    items = [
        _build_export_item(str(scope["scope_id"]), str(base_fact_id), indexes)
        for base_fact_id in scope["base_fact_ids"]
    ]
    templates = [_decision_template(item) for item in items]
    output_dir = Path(output_dir).resolve()
    items_path = output_dir / "review_items.jsonl"
    templates_path = output_dir / "review_decisions_template.jsonl"
    write_jsonl(items_path, items)
    write_jsonl(templates_path, templates)

    scope_sha = sha256_file(scope_manifest_path)
    manifest = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope["scope_id"],
        "scope_manifest": {
            "path": str(scope_manifest_path),
            "sha256": scope_sha,
        },
        "source_artifacts": scope["source_artifacts"],
        "base_fact_count": len(items),
        "artifacts": {
            "review_items": {
                "path": str(items_path),
                "sha256": sha256_file(items_path),
                "record_count": len(items),
                "schema_version": EXPORT_ITEM_SCHEMA_VERSION,
            },
            "review_decisions_template": {
                "path": str(templates_path),
                "sha256": sha256_file(templates_path),
                "record_count": len(templates),
                "schema_version": DECISION_SCHEMA_VERSION,
            },
        },
        "safety_contract": {
            "export_is_review_only": True,
            "canonical_freeze_performed": False,
            "automatic_promotion_allowed": False,
            "revise_outcome_supported": True,
            "revision_application_supported": False,
            "every_active_outcome_requires_all_member_reviews": True,
        },
    }
    manifest_path = output_dir / "review_export_manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "schema_version": EXPORT_SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope["scope_id"],
        "status": "review_package_exported",
        "base_fact_count": len(items),
        "member_count": sum(int(item["member_count"]) for item in items),
        "duplicate_cluster_count": sum(int(item["member_count"]) > 1 for item in items),
        "alias_target_count": sum(len(item["review_targets"]["aliases"]) for item in items),
        "distractor_target_count": sum(
            len(item["review_targets"]["distractors"]) for item in items
        ),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "artifacts": manifest["artifacts"],
        "canonical_freeze_performed": False,
    }
    summary_path = output_dir / "review_export_summary.json"
    write_json(summary_path, summary)
    return {
        "scope_id": scope["scope_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "base_fact_count": len(items),
    }


def _verify_bound_file(binding: Mapping[str, Any], label: str) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"Invalid {label} binding")
    path = Path(_require_nonempty_string(binding.get("path"), f"{label}.path"))
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} path must be absolute and normalized")
    if not _is_sha256(binding.get("sha256")):
        raise ValueError(f"{label} SHA-256 binding is invalid")
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"Stale SHA-256 binding for {label}: {path}")
    return path


def _validate_export_manifest(
    export_manifest: Mapping[str, Any],
    scope: Mapping[str, Any],
    scope_manifest_path: Path,
) -> Tuple[Path, Path]:
    expected_fields = {
        "schema_version",
        "tool_version",
        "scope_id",
        "scope_manifest",
        "source_artifacts",
        "base_fact_count",
        "artifacts",
        "safety_contract",
    }
    if set(export_manifest) != expected_fields:
        raise ValueError("Review export manifest fields are invalid")
    if export_manifest.get("schema_version") != EXPORT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported review export manifest schema")
    if export_manifest.get("tool_version") != TOOL_VERSION:
        raise ValueError("Unsupported review export tool version")
    if export_manifest.get("scope_id") != scope.get("scope_id"):
        raise ValueError("Export manifest scope_id does not match review scope")
    scope_binding = export_manifest.get("scope_manifest")
    if not isinstance(scope_binding, dict) or set(scope_binding) != {"path", "sha256"}:
        raise ValueError("Export manifest has no scope binding")
    bound_scope_path = Path(
        _require_nonempty_string(scope_binding.get("path"), "scope_manifest.path")
    )
    if not bound_scope_path.is_absolute() or bound_scope_path.resolve() != scope_manifest_path:
        raise ValueError("Export manifest scope path does not match review scope")
    if scope_binding.get("sha256") != sha256_file(scope_manifest_path):
        raise ValueError("Export manifest has a stale review scope SHA-256")
    if export_manifest.get("source_artifacts") != scope.get("source_artifacts"):
        raise ValueError("Export manifest source bindings differ from review scope")
    if export_manifest.get("safety_contract") != {
        "export_is_review_only": True,
        "canonical_freeze_performed": False,
        "automatic_promotion_allowed": False,
        "revise_outcome_supported": True,
        "revision_application_supported": False,
        "every_active_outcome_requires_all_member_reviews": True,
    }:
        raise ValueError("Export manifest safety contract is invalid")
    artifacts = export_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "review_items",
        "review_decisions_template",
    }:
        raise ValueError("Export manifest has no artifact bindings")
    items_binding = artifacts["review_items"]
    templates_binding = artifacts["review_decisions_template"]
    if (
        not isinstance(items_binding, dict)
        or set(items_binding) != EXPORT_ARTIFACT_BINDING_FIELDS
        or not isinstance(templates_binding, dict)
        or set(templates_binding) != EXPORT_ARTIFACT_BINDING_FIELDS
    ):
        raise ValueError("Export manifest has invalid artifact bindings")
    if items_binding.get("schema_version") != EXPORT_ITEM_SCHEMA_VERSION:
        raise ValueError("Export manifest review_items schema binding is invalid")
    if templates_binding.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError("Export manifest decision template schema binding is invalid")
    items_path = _verify_bound_file(items_binding, "review_items")
    templates_path = _verify_bound_file(
        templates_binding, "review_decisions_template"
    )
    if export_manifest.get("base_fact_count") != scope.get("base_fact_count"):
        raise ValueError("Export manifest base_fact_count does not match review scope")
    return items_path, templates_path


def _validate_review_provenance(decision: Mapping[str, Any], base_fact_id: str) -> Dict[str, str]:
    if decision.get("human_gold") is not False:
        raise ValueError(f"Decision {base_fact_id} must have human_gold=false")
    provenance = decision.get("review_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Decision {base_fact_id} has no review_provenance")
    if set(provenance) != {
        "reviewer_type",
        "reviewer_id",
        "review_method",
        "reviewed_at",
    }:
        raise ValueError(f"Decision {base_fact_id} review_provenance fields are invalid")
    if provenance.get("reviewer_type") != "codex_proxy":
        raise ValueError(f"Decision {base_fact_id} reviewer_type must be codex_proxy")
    return {
        "reviewer_type": "codex_proxy",
        "reviewer_id": _require_nonempty_string(
            provenance.get("reviewer_id"), f"decision {base_fact_id} reviewer_id"
        ),
        "review_method": _require_nonempty_string(
            provenance.get("review_method"), f"decision {base_fact_id} review_method"
        ),
        "reviewed_at": _require_nonempty_string(
            provenance.get("reviewed_at"), f"decision {base_fact_id} reviewed_at"
        ),
    }


def _review_entries(
    decision: Mapping[str, Any],
    field: str,
    id_field: str,
    base_fact_id: str,
) -> Dict[str, Mapping[str, Any]]:
    values = decision.get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"Decision {base_fact_id} {field} must be a list")
    return _unique_index(values, id_field, f"decision {base_fact_id} {field}")


def _validate_target_reviews(
    decision: Mapping[str, Any], item: Mapping[str, Any], *, require_complete: bool
) -> Dict[str, List[Dict[str, Any]]]:
    base_fact_id = str(item["base_fact_id"])
    target_specs = (
        ("member_reviews", "candidate_id", item["members"]),
        ("alias_reviews", "alias_id", item["review_targets"]["aliases"]),
        ("distractor_reviews", "distractor_id", item["review_targets"]["distractors"]),
    )
    validated: Dict[str, List[Dict[str, Any]]] = {}
    for field, id_field, targets in target_specs:
        allowed_fields = {
            "member_reviews": {"candidate_id", "triple_record_sha256", "decision"},
            "alias_reviews": {"alias_id", "text_en", "decision"},
            "distractor_reviews": {
                "distractor_id",
                "text_en",
                "source_choice_index",
                "decision",
            },
        }[field]
        expected = {str(target[id_field]): target for target in targets}
        all_supplied = _review_entries(decision, field, id_field, base_fact_id)
        unknown = sorted(set(all_supplied) - set(expected))
        if unknown:
            raise ValueError(f"Decision {base_fact_id} has unknown {field}: {unknown}")
        for target_id, entry in all_supplied.items():
            missing_fields = sorted(allowed_fields - set(entry))
            if missing_fields:
                raise ValueError(
                    f"Decision {base_fact_id} {field} {target_id} is missing target "
                    f"metadata: {missing_fields}"
                )
            extra_fields = sorted(set(entry) - allowed_fields)
            if extra_fields:
                raise ValueError(
                    f"Decision {base_fact_id} {field} {target_id} has unsupported fields: "
                    f"{extra_fields}"
                )
            target = expected[target_id]
            for metadata_field in sorted(allowed_fields - {"decision"}):
                if entry[metadata_field] != target[metadata_field]:
                    if field == "member_reviews" and metadata_field == "triple_record_sha256":
                        raise ValueError(
                            f"Decision {base_fact_id} member {target_id} has stale triple hash"
                        )
                    raise ValueError(
                        f"Decision {base_fact_id} {field} {target_id} {metadata_field} "
                        "does not match export"
                    )
            if entry.get("decision") not in ALLOWED_TARGET_DECISIONS | {None}:
                raise ValueError(
                    f"Decision {base_fact_id} {field} {target_id} must be accept, "
                    "reject, or unresolved"
                )

        field_requires_complete = require_complete or field == "member_reviews"
        supplied = (
            all_supplied
            if field_requires_complete
            else {
                target_id: entry
                for target_id, entry in all_supplied.items()
                if entry.get("decision") is not None
            }
        )
        missing = sorted(set(expected) - set(supplied))
        if field_requires_complete and missing:
            label = "Accept decision" if require_complete else "Decision"
            raise ValueError(f"{label} {base_fact_id} missing {field}: {missing}")
        field_output = []
        for target_id in sorted(supplied):
            entry = supplied[target_id]
            outcome = entry.get("decision")
            if outcome not in ALLOWED_TARGET_DECISIONS:
                raise ValueError(
                    f"Decision {base_fact_id} {field} {target_id} must be accept or reject"
                )
            target = expected[target_id]
            if field == "member_reviews":
                field_output.append(
                    {
                        "candidate_id": target_id,
                        "triple_record_sha256": target["triple_record_sha256"],
                        "decision": outcome,
                    }
                )
            else:
                field_output.append({**target, "decision": outcome})
        validated[field] = field_output

    if require_complete:
        if any(entry["decision"] != "accept" for entry in validated["member_reviews"]):
            raise ValueError(f"Accept decision {base_fact_id} requires every member to be accepted")
        behavior = item["review_context"]["behavior_row"]
        answer_normalized = normalize_text(behavior["answer_en"])
        accepted_aliases = {
            normalize_text(entry["text_en"])
            for entry in validated["alias_reviews"]
            if entry["decision"] == "accept"
        }
        if answer_normalized not in accepted_aliases:
            raise ValueError(
                f"Accept decision {base_fact_id} must retain an alias equal to answer_en"
            )
    return validated


def _validate_decision(
    decision: Mapping[str, Any], item: Mapping[str, Any]
) -> Tuple[str, Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    base_fact_id = str(item["base_fact_id"])
    if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported decision schema for {base_fact_id}")
    forbidden = sorted(FORBIDDEN_PROMOTION_FIELDS.intersection(decision))
    if forbidden:
        raise ValueError(
            f"Decision {base_fact_id} contains forbidden promotion fields: {forbidden}"
        )
    unsupported = sorted(set(decision) - DECISION_ALLOWED_FIELDS)
    if unsupported:
        raise ValueError(
            f"Decision {base_fact_id} contains unsupported decision fields; "
            f"revisions cannot be applied in v2: {unsupported}"
        )
    if decision.get("base_fact_id") != base_fact_id:
        raise ValueError(f"Decision {base_fact_id} has the wrong base_fact_id")
    if decision.get("scope_id") != item.get("scope_id"):
        raise ValueError(f"Decision {base_fact_id} has the wrong scope_id")
    if decision.get("review_item_sha256") != sha256_value(item):
        raise ValueError(f"Decision {base_fact_id} has a stale review_item_sha256")
    outcome = decision.get("decision")
    if outcome not in ALLOWED_DECISIONS:
        raise ValueError(f"Unknown decision for {base_fact_id}: {outcome!r}")
    provenance = _validate_review_provenance(decision, base_fact_id)
    reviews = _validate_target_reviews(
        decision, item, require_complete=outcome == "accept"
    )
    return str(outcome), provenance, reviews


def _staging_record(
    item: Mapping[str, Any],
    decision: Optional[Mapping[str, Any]],
    review_bindings: Mapping[str, Any],
) -> Dict[str, Any]:
    behavior = item["review_context"]["behavior_row"]
    base = {
        "schema_version": STAGING_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": item["scope_id"],
        "base_fact_id": item["base_fact_id"],
        "review_item_sha256": sha256_value(item),
        "canonical_status": "pending_review",
        "bundle_status": "draft_pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "subject_en": behavior["subject_en"],
        "relation_raw": behavior["relation_raw"],
        "answer_en": behavior["answer_en"],
        "canonical_fact_en": behavior["canonical_fact_en"],
        "source_row_bindings": item["source_row_bindings"],
        "review_evidence_bindings": {
            **review_bindings,
            "decision_record_sha256": (
                sha256_value(decision)
                if decision is not None and decision.get("decision") is not None
                else None
            ),
        },
        "automatic_promotion_performed": False,
        "canonical_freeze_performed": False,
    }
    if decision is None or decision.get("decision") is None:
        return {
            **base,
            "review_outcome": None,
            "staging_status": "decision_missing",
            "review_provenance": None,
            "review_completion": {
                "codex_proxy_scope_review_complete": False,
                "member_review_complete": False,
                "alias_review_complete": False,
                "distractor_review_complete": False,
            },
            "promotion_blockers": [
                "review_decision_missing",
                "canonical_freeze_not_implemented",
                "near_duplicate_resolution_not_bound",
                "split_not_frozen",
            ],
        }

    outcome, provenance, reviews = _validate_decision(decision, item)
    accepted_aliases = [
        entry["text_en"]
        for entry in reviews["alias_reviews"]
        if entry["decision"] == "accept"
    ]
    accepted_distractors = [
        {
            "distractor_id": entry["distractor_id"],
            "text_en": entry["text_en"],
        }
        for entry in reviews["distractor_reviews"]
        if entry["decision"] == "accept"
    ]
    completion = {
        "member_review_complete": len(reviews["member_reviews"]) == len(item["members"]),
        "alias_review_complete": len(reviews["alias_reviews"])
        == len(item["review_targets"]["aliases"]),
        "distractor_review_complete": len(reviews["distractor_reviews"])
        == len(item["review_targets"]["distractors"]),
    }
    status = {
        "accept": "accepted_pending_freeze_prerequisites",
        "reject": "rejected_review_staging",
        "defer": "deferred_review_staging",
        "revise": "revision_requested_review_staging",
    }[outcome]
    blockers = [
        "canonical_freeze_not_implemented",
        "near_duplicate_resolution_not_bound",
        "split_not_frozen",
    ]
    if outcome == "accept" and not accepted_distractors:
        blockers.append("verified_distractor_missing")
    if outcome == "revise":
        blockers.extend(["revision_not_applied", "recluster_required_before_revision"])
    return {
        **base,
        "review_outcome": outcome,
        "staging_status": status,
        "review_provenance": provenance,
        "review_completion": {
            "codex_proxy_scope_review_complete": all(completion.values()),
            **completion,
        },
        "member_reviews": reviews["member_reviews"],
        "alias_reviews": reviews["alias_reviews"],
        "distractor_reviews": reviews["distractor_reviews"],
        "accepted_answer_aliases_en": accepted_aliases,
        "accepted_distractors": accepted_distractors,
        "notes": decision.get("notes"),
        "promotion_blockers": blockers,
    }


def apply_decisions(
    *,
    scope_manifest_path: Path,
    export_manifest_path: Path,
    decisions_path: Optional[Path],
    output_dir: Path,
    allow_partial: bool = False,
    composite_authority_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate decisions and write staging records without canonical promotion."""

    scope_manifest_path = Path(scope_manifest_path).resolve()
    export_manifest_path = Path(export_manifest_path).resolve()
    if (decisions_path is None) == (composite_authority_manifest_path is None):
        raise ValueError(
            "Supply exactly one of decisions_path or composite_authority_manifest_path"
        )
    composite_authority: Optional[Dict[str, Any]] = None
    verified_composite_decisions: Optional[List[Dict[str, Any]]] = None
    verified_composite_decisions_sha256: Optional[str] = None
    if composite_authority_manifest_path is not None:
        if allow_partial:
            raise ValueError("Composite full apply forbids allow_partial")
        composite_authority = _load_composite_authority_tool().load_composite_authority(
            Path(composite_authority_manifest_path)
        )
        decisions_path = Path(composite_authority["decisions_path"])
        decisions_binding = composite_authority.get("decisions_binding")
        if not isinstance(decisions_binding, dict) or not _is_sha256(
            decisions_binding.get("sha256")
        ):
            raise ValueError("Composite authority decisions binding is invalid")
        verified_composite_decisions_sha256 = str(decisions_binding["sha256"])
        if sha256_file(decisions_path) != verified_composite_decisions_sha256:
            raise ValueError(
                "Composite decisions changed after authority validation"
            )
        authority_decisions = composite_authority.get("decisions")
        if not isinstance(authority_decisions, list):
            raise ValueError("Composite authority verified decisions are missing")
        verified_composite_decisions = [dict(row) for row in authority_decisions]
    assert decisions_path is not None
    decisions_path = Path(decisions_path).resolve()
    scope = read_json(scope_manifest_path)
    _validate_scope_manifest(scope)
    paths = _scope_artifact_paths(scope)
    rows, indexes = load_source_artifacts(paths)
    _verify_scope_selection_source(scope, indexes)
    for role, binding in scope["source_artifacts"].items():
        if len(rows[role]) != binding.get("record_count"):
            raise ValueError(f"Stale record-count binding for {role}")

    export_manifest = read_json(export_manifest_path)
    items_path, templates_path = _validate_export_manifest(
        export_manifest, scope, scope_manifest_path
    )
    items = read_jsonl(items_path)
    item_index = _unique_index(items, "base_fact_id", "review_items")
    expected_ids = [str(value) for value in scope["base_fact_ids"]]
    if allow_partial and len(expected_ids) == FULL_PUBLIC_BENCHMARK_COUNT:
        raise ValueError("Full 8,969-item apply forbids allow_partial")
    if composite_authority is not None:
        authority_manifest = composite_authority["manifest"]
        authority_export = authority_manifest.get("export_manifest")
        if not isinstance(authority_export, dict):
            raise ValueError("Composite authority export binding is missing")
        if (
            _verify_bound_file(authority_export, "composite export_manifest")
            != export_manifest_path
            or authority_export.get("sha256") != sha256_file(export_manifest_path)
        ):
            raise ValueError("Composite authority binds a different review export")
        if authority_manifest.get("scope_id") != scope.get("scope_id"):
            raise ValueError("Composite authority scope_id does not match review scope")
    if set(item_index) != set(expected_ids):
        raise ValueError("Review items do not exactly cover the frozen review scope")
    items_binding = export_manifest["artifacts"]["review_items"]
    if items_binding.get("record_count") != len(items):
        raise ValueError("Export manifest review_items record_count is stale")
    for item in items:
        if item.get("schema_version") != EXPORT_ITEM_SCHEMA_VERSION:
            raise ValueError("Unsupported review item schema")
        if item.get("scope_id") != scope.get("scope_id"):
            raise ValueError("Review item scope_id mismatch")
        if item.get("canonical_status") != "pending_review" or item.get("human_gold") is not False:
            raise ValueError("Review item is not provisional")
        expected_item = _build_export_item(
            str(scope["scope_id"]), str(item["base_fact_id"]), indexes
        )
        if sha256_value(item) != sha256_value(expected_item):
            raise ValueError(
                f"Review item {item['base_fact_id']} does not match SHA-bound source artifacts"
            )

    templates = read_jsonl(templates_path)
    template_index = _unique_index(
        templates, "base_fact_id", "review_decisions_template"
    )
    if set(template_index) != set(expected_ids):
        raise ValueError("Decision template does not exactly cover the frozen review scope")
    templates_binding = export_manifest["artifacts"]["review_decisions_template"]
    if templates_binding.get("record_count") != len(templates):
        raise ValueError("Export manifest decision template record_count is stale")
    for base_fact_id in expected_ids:
        template = template_index[base_fact_id]
        if template.get("schema_version") != DECISION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported decision template schema for {base_fact_id}"
            )
        expected_template = _decision_template(item_index[base_fact_id])
        if sha256_value(template) != sha256_value(expected_template):
            raise ValueError(
                f"Decision template {base_fact_id} does not match its review item"
            )

    decisions = (
        verified_composite_decisions
        if verified_composite_decisions is not None
        else read_jsonl(decisions_path, allow_empty=True)
    )
    if composite_authority is not None and [
        str(row.get("base_fact_id") or "") for row in decisions
    ] != expected_ids:
        raise ValueError(
            "Composite authority decisions do not match the full review scope order"
        )
    decision_index = _unique_index(decisions, "base_fact_id", "decisions")
    unknown_ids = sorted(set(decision_index) - set(expected_ids))
    if unknown_ids:
        raise ValueError(f"Unknown base_fact_id in decisions: {unknown_ids[:10]}")
    for base_fact_id, decision in decision_index.items():
        if decision.get("decision") is None and sha256_value(decision) != sha256_value(
            template_index[base_fact_id]
        ):
            raise ValueError(
                f"Missing decision placeholder {base_fact_id} does not match the "
                "bound decision template"
            )
    active_decisions = {
        base_fact_id: decision
        for base_fact_id, decision in decision_index.items()
        if decision.get("decision") is not None
    }
    missing_ids = [
        base_fact_id
        for base_fact_id in expected_ids
        if base_fact_id not in active_decisions
    ]
    if missing_ids and not allow_partial:
        raise ValueError(
            f"Missing decisions for {len(missing_ids)} scoped base facts; "
            "use --allow-partial to retain them as missing"
        )

    if verified_composite_decisions_sha256 is not None:
        if sha256_file(decisions_path) != verified_composite_decisions_sha256:
            raise ValueError(
                "Composite decisions changed after authority validation"
            )
        decisions_artifact_sha256 = verified_composite_decisions_sha256
    else:
        decisions_artifact_sha256 = sha256_file(decisions_path)

    staging = [
        _staging_record(
            item_index[base_fact_id],
            active_decisions.get(base_fact_id),
            {
                "scope_manifest_sha256": sha256_file(scope_manifest_path),
                "export_manifest_sha256": sha256_file(export_manifest_path),
                "decisions_artifact_sha256": decisions_artifact_sha256,
                "apply_manifest_schema_version": APPLY_MANIFEST_SCHEMA_VERSION,
            },
        )
        for base_fact_id in expected_ids
    ]
    outcome_counts = Counter(
        str(record["review_outcome"] or "missing") for record in staging
    )
    output_dir = Path(output_dir).resolve()
    staging_path = output_dir / "review_staging.jsonl"
    write_jsonl(staging_path, staging)
    manifest = {
        "schema_version": APPLY_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope["scope_id"],
        "scope_manifest": {
            "path": str(scope_manifest_path),
            "sha256": sha256_file(scope_manifest_path),
        },
        "export_manifest": {
            "path": str(export_manifest_path),
            "sha256": sha256_file(export_manifest_path),
        },
        "decisions": {
            "path": str(decisions_path),
            "sha256": decisions_artifact_sha256,
            "record_count": len(decisions),
        },
        "allow_partial": allow_partial,
        "scoped_base_fact_count": len(expected_ids),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "artifacts": {
            "review_staging": {
                "path": str(staging_path),
                "sha256": sha256_file(staging_path),
                "record_count": len(staging),
                "schema_version": STAGING_SCHEMA_VERSION,
            }
        },
        "safety_contract": {
            "output_is_staging_only": True,
            "source_artifacts_modified": False,
            "canonical_freeze_performed": False,
            "automatic_promotion_performed": False,
            "missing_is_reject": False,
            "revise_outcome_supported": True,
            "revision_application_supported": False,
        },
    }
    if composite_authority is not None:
        manifest["decision_authority"] = {
            "mode": "composite_fact_review_authority",
            "manifest": {
                "path": str(composite_authority["manifest_path"]),
                "sha256": sha256_file(composite_authority["manifest_path"]),
                "schema_version": composite_authority["manifest"][
                    "schema_version"
                ],
            },
            "authority_id": composite_authority["manifest"]["authority_id"],
        }
    manifest_path = output_dir / "review_apply_manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "schema_version": APPLY_SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "scope_id": scope["scope_id"],
        "status": "partial_review_staged" if missing_ids else "review_staged",
        "allow_partial": allow_partial,
        "scoped_base_fact_count": len(expected_ids),
        "decision_record_count": len(decisions),
        "active_decision_count": len(active_decisions),
        "missing_decision_count": len(missing_ids),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "artifacts": manifest["artifacts"],
        "canonical_freeze_performed": False,
        "automatic_promotion_performed": False,
    }
    summary_path = output_dir / "review_apply_summary.json"
    write_json(summary_path, summary)
    return {
        "scope_id": scope["scope_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "staging_path": str(staging_path),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "decision_authority_mode": (
            "composite_fact_review_authority"
            if composite_authority is not None
            else "direct_decisions"
        ),
    }


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-dir",
        help="Directory containing the four standard provisional artifacts",
    )
    parser.add_argument("--bundle", help="Override behavior_input_bundle.jsonl")
    parser.add_argument("--review-queue", help="Override review_queue.jsonl")
    parser.add_argument("--clusters", help="Override base_fact_clusters.jsonl")
    parser.add_argument("--triples", help="Override provisional_triples.jsonl")


def _paths_from_args(args: argparse.Namespace) -> Dict[str, Path]:
    overrides = {
        role: Path(value)
        for role, value in {
            "behavior_bundle": args.bundle,
            "review_queue": args.review_queue,
            "base_fact_clusters": args.clusters,
            "provisional_triples": args.triples,
        }.items()
        if value is not None
    }
    return source_paths(
        Path(args.source_dir) if args.source_dir is not None else None,
        overrides,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    scope = commands.add_parser(
        "scope", help="Create a normalized-path and SHA-bound v2 review scope"
    )
    _add_source_arguments(scope)
    scope.add_argument("--output-dir", required=True)
    selection = scope.add_mutually_exclusive_group(required=True)
    selection.add_argument("--base-fact-id", action="append", dest="base_fact_ids")
    selection.add_argument("--base-fact-ids-file")
    selection.add_argument("--review-sample")

    export = commands.add_parser(
        "export", help="Export self-contained v2 review items"
    )
    export.add_argument("--scope-manifest", required=True)
    export.add_argument("--output-dir", required=True)

    apply = commands.add_parser(
        "apply", help="Apply v2 decisions into review-only staging"
    )
    apply.add_argument("--scope-manifest", required=True)
    apply.add_argument("--export-manifest", required=True)
    decision_source = apply.add_mutually_exclusive_group(required=True)
    decision_source.add_argument("--decisions")
    decision_source.add_argument("--composite-authority-manifest")
    apply.add_argument("--output-dir", required=True)
    apply.add_argument("--allow-partial", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "scope":
        explicit_ids = args.base_fact_ids
        if args.base_fact_ids_file is not None:
            explicit_ids = _read_explicit_ids(Path(args.base_fact_ids_file))
        result = create_scope(
            output_dir=Path(args.output_dir),
            artifact_paths=_paths_from_args(args),
            base_fact_ids=explicit_ids,
            review_sample_path=(
                Path(args.review_sample) if args.review_sample is not None else None
            ),
        )
    elif args.command == "export":
        result = export_scope(
            scope_manifest_path=Path(args.scope_manifest),
            output_dir=Path(args.output_dir),
        )
    else:
        result = apply_decisions(
            scope_manifest_path=Path(args.scope_manifest),
            export_manifest_path=Path(args.export_manifest),
            decisions_path=(Path(args.decisions) if args.decisions is not None else None),
            output_dir=Path(args.output_dir),
            allow_partial=args.allow_partial,
            composite_authority_manifest_path=(
                Path(args.composite_authority_manifest)
                if args.composite_authority_manifest is not None
                else None
            ),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
