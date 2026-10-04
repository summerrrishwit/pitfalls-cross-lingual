#!/usr/bin/env python3
"""Create and merge SHA-bound shards for the static G0A Codex review.

This utility deliberately performs structural validation only.  A successful
merge is not a semantic acceptance decision; the merged file must still be
passed to ``scripts/manage_static_g0a.py`` for its complete decision checks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TOOL_VERSION = "codex-review-shards-v1"
SHARD_MANIFEST_SCHEMA = "static-g0a-codex-review-shard-manifest-v1"
MERGE_MANIFEST_SCHEMA = "static-g0a-codex-review-merge-manifest-v1"
REBASE_MANIFEST_SCHEMA = "static-g0a-codex-review-rebase-manifest-v1"
REBASE_COMPLETION_MANIFEST_SCHEMA = (
    "static-g0a-codex-review-rebase-completion-manifest-v1"
)
REVIEW_INPUT_MANIFEST_SCHEMA = "static-g0a-codex-review-input-manifest-v1"
REVIEW_INPUT_RECORD_SCHEMA = "static-g0a-codex-review-input-v1"
CODEX_DECISION_SCHEMA = "static-g0a-codex-decision-v1"
EXPECTED_RECORD_COUNT = 160
SHARD_SIZE = 40
SHARD_COUNT = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _read_bytes(path: Path, label: str) -> bytes:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved.read_bytes()


def _parse_json_payload(payload: bytes, path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _parse_jsonl_payload(
    payload: bytes, path: Path, label: str
) -> List[Dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 {label}: {path}: {exc}") from exc
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def read_json(path: Path, label: str = "JSON") -> Dict[str, Any]:
    resolved = Path(path).resolve()
    return _parse_json_payload(_read_bytes(resolved, label), resolved, label)


def read_jsonl(path: Path, label: str = "JSONL") -> List[Dict[str, Any]]:
    resolved = Path(path).resolve()
    return _parse_jsonl_payload(_read_bytes(resolved, label), resolved, label)


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
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
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _binding(
    path: Path,
    payload: bytes,
    *,
    schema_version: str,
    record_count: int | None = None,
    base_fact_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "schema_version": schema_version,
    }
    if record_count is not None:
        result["record_count"] = record_count
    if base_fact_ids is not None:
        result["base_fact_ids_sha256"] = sha256_value(sorted(base_fact_ids))
    return result


def _binding_path(owner_path: Path, binding: Mapping[str, Any], label: str) -> Path:
    raw_path = _required_string(binding.get("path"), f"{label}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(owner_path).resolve().parent / path
    return path.resolve()


def _validate_file_binding(
    owner_path: Path,
    binding: Any,
    label: str,
    *,
    expected_schema: str,
    jsonl: bool,
) -> Tuple[Path, bytes, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = _binding_path(owner_path, binding, label)
    payload = _read_bytes(path, label)
    if binding.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{label} SHA-256 mismatch")
    if binding.get("byte_count") != len(payload):
        raise ValueError(f"{label} byte_count mismatch")
    if binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version mismatch")
    value = (
        _parse_jsonl_payload(payload, path, label)
        if jsonl
        else _parse_json_payload(payload, path, label)
    )
    if jsonl and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record_count mismatch")
    return path, payload, value


def _unique_ids(rows: Sequence[Mapping[str, Any]], label: str) -> List[str]:
    ids: List[str] = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        base_fact_id = _required_string(
            row.get("base_fact_id"), f"{label} row {index} base_fact_id"
        )
        if base_fact_id in seen:
            raise ValueError(f"duplicate {label} base_fact_id: {base_fact_id}")
        seen.add(base_fact_id)
        ids.append(base_fact_id)
    return ids


def _load_parent_review(
    review_input_manifest_path: Path,
) -> Tuple[
    Dict[str, Any],
    bytes,
    Path,
    bytes,
    List[Dict[str, Any]],
    Path,
    bytes,
    List[Dict[str, Any]],
]:
    manifest_path = Path(review_input_manifest_path).resolve()
    manifest_payload = _read_bytes(manifest_path, "review input manifest")
    manifest = _parse_json_payload(
        manifest_payload, manifest_path, "review input manifest"
    )
    if manifest.get("schema_version") != REVIEW_INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported review input manifest schema_version")
    if manifest.get("base_fact_count") != EXPECTED_RECORD_COUNT:
        raise ValueError("review input manifest must contain exactly 160 base facts")
    if (
        manifest.get("status") != "codex_review_ready_behavior_blind"
        or manifest.get("review_blinded_to_behavior_results") is not True
        or manifest.get("simulation_results_checked") is not True
        or manifest.get("simulation_result_count") != 0
        or manifest.get("behavior_authorized") is not False
    ):
        raise ValueError("review input manifest is not behavior-blind and simulation-free")
    if manifest.get("reviewer_type") != "codex_proxy" or manifest.get("human_gold") is not False:
        raise ValueError("review input manifest reviewer identity is invalid")

    input_path, input_payload, input_rows = _validate_file_binding(
        manifest_path,
        manifest.get("review_input"),
        "review input",
        expected_schema=REVIEW_INPUT_RECORD_SCHEMA,
        jsonl=True,
    )
    template_path, template_payload, template_rows = _validate_file_binding(
        manifest_path,
        manifest.get("decision_template"),
        "decision template",
        expected_schema=CODEX_DECISION_SCHEMA,
        jsonl=True,
    )
    if len(input_rows) != EXPECTED_RECORD_COUNT or len(template_rows) != EXPECTED_RECORD_COUNT:
        raise ValueError("review input and template must each contain exactly 160 rows")

    input_ids = _unique_ids(input_rows, "review input")
    template_ids = _unique_ids(template_rows, "decision template")
    if set(input_ids) != set(template_ids):
        raise ValueError("review input and decision template base_fact_id sets differ")
    expected_ids_digest = sha256_value(sorted(input_ids))
    if (manifest.get("review_input") or {}).get("base_fact_ids_sha256") != expected_ids_digest:
        raise ValueError("review input base_fact_ids_sha256 mismatch")

    input_by_id = {row["base_fact_id"]: row for row in input_rows}
    for index, template in enumerate(template_rows, start=1):
        base_fact_id = template_ids[index - 1]
        if template.get("schema_version") != CODEX_DECISION_SCHEMA:
            raise ValueError(f"decision template row {index} schema_version mismatch")
        if template.get("reviewer_type") != "codex_proxy" or template.get("human_gold") is not False:
            raise ValueError(f"decision template row {index} reviewer identity is invalid")
        if template.get("review_input_record_sha256") != sha256_value(input_by_id[base_fact_id]):
            raise ValueError(f"decision template row {index} review input hash mismatch")
    for index, row in enumerate(input_rows, start=1):
        if row.get("schema_version") != REVIEW_INPUT_RECORD_SCHEMA:
            raise ValueError(f"review input row {index} schema_version mismatch")
        if row.get("reviewer_type") != "codex_proxy" or row.get("human_gold") is not False:
            raise ValueError(f"review input row {index} reviewer identity is invalid")
        if row.get("review_blinded_to_behavior_results") is not True:
            raise ValueError(f"review input row {index} is not behavior blind")

    return (
        manifest,
        manifest_payload,
        input_path,
        input_payload,
        input_rows,
        template_path,
        template_payload,
        template_rows,
    )


def create_shards(review_input_manifest_path: Path, output_dir: Path) -> Dict[str, Any]:
    (
        parent_manifest,
        parent_manifest_payload,
        input_path,
        input_payload,
        input_rows,
        template_path,
        template_payload,
        template_rows,
    ) = _load_parent_review(review_input_manifest_path)
    manifest_path = Path(review_input_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    input_by_id = {row["base_fact_id"]: row for row in input_rows}
    template_by_id = {row["base_fact_id"]: row for row in template_rows}
    ordered_ids = sorted(input_by_id)
    shard_specs = []
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        for offset in range(0, EXPECTED_RECORD_COUNT, SHARD_SIZE):
            shard_index = offset // SHARD_SIZE + 1
            shard_ids = ordered_ids[offset : offset + SHARD_SIZE]
            input_name = (
                f"codex_adjudication_input.shard-{shard_index:02d}-of-{SHARD_COUNT:02d}.jsonl"
            )
            template_name = (
                f"codex_adjudication_decisions.shard-{shard_index:02d}-of-{SHARD_COUNT:02d}.template.jsonl"
            )
            input_shard_rows = [input_by_id[value] for value in shard_ids]
            template_shard_rows = [template_by_id[value] for value in shard_ids]
            input_shard_payload = _jsonl_payload(input_shard_rows)
            template_shard_payload = _jsonl_payload(template_shard_rows)
            (temporary_dir / input_name).write_bytes(input_shard_payload)
            (temporary_dir / template_name).write_bytes(template_shard_payload)
            shard_specs.append(
                {
                    "shard_index": shard_index,
                    "record_count": len(shard_ids),
                    "base_fact_ids": shard_ids,
                    "base_fact_ids_sha256": sha256_value(shard_ids),
                    "review_input": _binding(
                        output_dir / input_name,
                        input_shard_payload,
                        schema_version=REVIEW_INPUT_RECORD_SCHEMA,
                        record_count=len(shard_ids),
                        base_fact_ids=shard_ids,
                    ),
                    "decision_template": _binding(
                        output_dir / template_name,
                        template_shard_payload,
                        schema_version=CODEX_DECISION_SCHEMA,
                        record_count=len(shard_ids),
                        base_fact_ids=shard_ids,
                    ),
                }
            )

        manifest = {
            "schema_version": SHARD_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "created_at": utc_now(),
            "status": "review_shards_created_pending_decisions",
            "ordering": "base_fact_id_ascending",
            "record_count": EXPECTED_RECORD_COUNT,
            "shard_count": SHARD_COUNT,
            "shard_size": SHARD_SIZE,
            "base_fact_ids_sha256": sha256_value(ordered_ids),
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "semantic_validation_performed": False,
            "parent_review_input_manifest": _binding(
                manifest_path,
                parent_manifest_payload,
                schema_version=REVIEW_INPUT_MANIFEST_SCHEMA,
            ),
            "parent_review_input": _binding(
                input_path,
                input_payload,
                schema_version=REVIEW_INPUT_RECORD_SCHEMA,
                record_count=len(input_rows),
                base_fact_ids=ordered_ids,
            ),
            "parent_decision_template": _binding(
                template_path,
                template_payload,
                schema_version=CODEX_DECISION_SCHEMA,
                record_count=len(template_rows),
                base_fact_ids=ordered_ids,
            ),
            "parent_manifest_status": parent_manifest["status"],
            "shards": shard_specs,
        }
        (temporary_dir / "shard_manifest.json").write_bytes(_json_payload(manifest))

        # Recheck the source artifacts just before publishing the directory.
        if _read_bytes(manifest_path, "review input manifest") != parent_manifest_payload:
            raise ValueError("parent review input manifest changed during shard creation")
        if _read_bytes(input_path, "review input") != input_payload:
            raise ValueError("parent review input changed during shard creation")
        if _read_bytes(template_path, "decision template") != template_payload:
            raise ValueError("parent decision template changed during shard creation")
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest


def _validate_current_binding(binding: Any, label: str) -> Tuple[Path, bytes]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = Path(_required_string(binding.get("path"), f"{label}.path")).resolve()
    payload = _read_bytes(path, label)
    if binding.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{label} changed after shard creation")
    if binding.get("byte_count") != len(payload):
        raise ValueError(f"{label} byte_count changed after shard creation")
    return path, payload


def _load_and_validate_shard_manifest(
    shard_manifest_path: Path,
) -> Tuple[Dict[str, Any], bytes, Dict[str, Dict[str, Any]], List[List[str]]]:
    shard_manifest_path = Path(shard_manifest_path).resolve()
    shard_manifest_payload = _read_bytes(shard_manifest_path, "shard manifest")
    manifest = _parse_json_payload(
        shard_manifest_payload, shard_manifest_path, "shard manifest"
    )
    if manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA:
        raise ValueError("unsupported shard manifest schema_version")
    if (
        manifest.get("record_count") != EXPECTED_RECORD_COUNT
        or manifest.get("shard_count") != SHARD_COUNT
        or manifest.get("shard_size") != SHARD_SIZE
        or manifest.get("ordering") != "base_fact_id_ascending"
        or manifest.get("status") != "review_shards_created_pending_decisions"
        or manifest.get("semantic_validation_performed") is not False
    ):
        raise ValueError("shard manifest count or ordering contract is invalid")
    if manifest.get("reviewer_type") != "codex_proxy" or manifest.get("human_gold") is not False:
        raise ValueError("shard manifest reviewer identity is invalid")

    parent_manifest_path, parent_manifest_payload = _validate_current_binding(
        manifest.get("parent_review_input_manifest"), "parent review input manifest"
    )
    (
        _,
        loaded_parent_manifest_payload,
        input_path,
        input_payload,
        input_rows,
        template_path,
        template_payload,
        template_rows,
    ) = _load_parent_review(parent_manifest_path)
    if loaded_parent_manifest_payload != parent_manifest_payload:
        raise ValueError("parent review input manifest binding changed")
    for binding, path, payload, label in (
        (manifest.get("parent_review_input"), input_path, input_payload, "parent review input"),
        (
            manifest.get("parent_decision_template"),
            template_path,
            template_payload,
            "parent decision template",
        ),
    ):
        bound_path, bound_payload = _validate_current_binding(binding, label)
        if bound_path != path or bound_payload != payload:
            raise ValueError(f"{label} binding does not match parent manifest")

    input_by_id = {row["base_fact_id"]: row for row in input_rows}
    template_by_id = {row["base_fact_id"]: row for row in template_rows}
    expected_ids = sorted(input_by_id)
    if manifest.get("base_fact_ids_sha256") != sha256_value(expected_ids):
        raise ValueError("shard manifest base_fact_ids_sha256 mismatch")
    expected_partitions = [
        expected_ids[offset : offset + SHARD_SIZE]
        for offset in range(0, EXPECTED_RECORD_COUNT, SHARD_SIZE)
    ]

    shard_specs = manifest.get("shards")
    if not isinstance(shard_specs, list) or len(shard_specs) != SHARD_COUNT:
        raise ValueError("shard manifest must contain exactly four shard entries")
    for expected_index, (spec, ids) in enumerate(
        zip(shard_specs, expected_partitions), start=1
    ):
        if not isinstance(spec, dict) or spec.get("shard_index") != expected_index:
            raise ValueError(f"invalid shard manifest entry {expected_index}")
        if (
            spec.get("record_count") != SHARD_SIZE
            or spec.get("base_fact_ids") != ids
            or spec.get("base_fact_ids_sha256") != sha256_value(ids)
        ):
            raise ValueError(f"shard {expected_index} ID contract mismatch")
        _, _, shard_inputs = _validate_file_binding(
            shard_manifest_path,
            spec.get("review_input"),
            f"review input shard {expected_index}",
            expected_schema=REVIEW_INPUT_RECORD_SCHEMA,
            jsonl=True,
        )
        _, _, shard_templates = _validate_file_binding(
            shard_manifest_path,
            spec.get("decision_template"),
            f"decision template shard {expected_index}",
            expected_schema=CODEX_DECISION_SCHEMA,
            jsonl=True,
        )
        if shard_inputs != [input_by_id[value] for value in ids]:
            raise ValueError(f"review input shard {expected_index} differs from parent")
        if shard_templates != [template_by_id[value] for value in ids]:
            raise ValueError(f"decision template shard {expected_index} differs from parent")
    return manifest, shard_manifest_payload, input_by_id, expected_partitions


def merge_decisions(
    shard_manifest_path: Path,
    decision_paths: Sequence[Path],
    output_decisions_path: Path,
    output_manifest_path: Path,
) -> Dict[str, Any]:
    if len(decision_paths) != SHARD_COUNT:
        raise ValueError("merge requires exactly four decision files")
    output_decisions_path = Path(output_decisions_path).resolve()
    output_manifest_path = Path(output_manifest_path).resolve()
    if output_decisions_path == output_manifest_path:
        raise ValueError("merged decisions and merge manifest paths must differ")
    for path in (output_decisions_path, output_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    (
        shard_manifest,
        shard_manifest_payload,
        input_by_id,
        expected_partitions,
    ) = _load_and_validate_shard_manifest(shard_manifest_path)
    merged_by_id: Dict[str, Dict[str, Any]] = {}
    decision_bindings = []
    for shard_index, (raw_path, expected_ids) in enumerate(
        zip(decision_paths, expected_partitions), start=1
    ):
        path = Path(raw_path).resolve()
        payload = _read_bytes(path, f"decision shard {shard_index}")
        rows = read_jsonl(path, f"decision shard {shard_index}")
        ids = _unique_ids(rows, f"decision shard {shard_index}")
        if len(rows) != SHARD_SIZE or set(ids) != set(expected_ids):
            raise ValueError(f"decision shard {shard_index} does not cover its assigned 40 IDs")
        for row_index, row in enumerate(rows, start=1):
            base_fact_id = ids[row_index - 1]
            if row.get("schema_version") != CODEX_DECISION_SCHEMA:
                raise ValueError(
                    f"decision shard {shard_index} row {row_index} schema_version mismatch"
                )
            if row.get("reviewer_type") != "codex_proxy" or row.get("human_gold") is not False:
                raise ValueError(
                    f"decision shard {shard_index} row {row_index} reviewer identity is invalid"
                )
            if row.get("terminal_status") != "completed":
                raise ValueError(
                    f"decision shard {shard_index} row {row_index} is not terminal"
                )
            if row.get("review_input_record_sha256") != sha256_value(input_by_id[base_fact_id]):
                raise ValueError(
                    f"decision shard {shard_index} row {row_index} review input hash mismatch"
                )
            if base_fact_id in merged_by_id:
                raise ValueError(f"duplicate merged base_fact_id: {base_fact_id}")
            merged_by_id[base_fact_id] = row
        decision_bindings.append(
            _binding(
                path,
                payload,
                schema_version=CODEX_DECISION_SCHEMA,
                record_count=len(rows),
                base_fact_ids=ids,
            )
        )

    expected_all_ids = sorted(input_by_id)
    if set(merged_by_id) != set(expected_all_ids):
        missing = sorted(set(expected_all_ids) - set(merged_by_id))
        extra = sorted(set(merged_by_id) - set(expected_all_ids))
        raise ValueError(f"merged decisions have incomplete coverage; missing={missing[:5]} extra={extra[:5]}")
    merged_rows = [merged_by_id[value] for value in expected_all_ids]
    decisions_payload = _jsonl_payload(merged_rows)

    # Revalidate every parent and shard binding after reading all decisions and
    # before publishing any output, closing the most likely stale-input race.
    current_manifest, current_manifest_payload, _, current_partitions = (
        _load_and_validate_shard_manifest(shard_manifest_path)
    )
    if current_manifest_payload != shard_manifest_payload or current_manifest != shard_manifest:
        raise ValueError("shard manifest changed during merge")
    if current_partitions != expected_partitions:
        raise ValueError("shard partition changed during merge")
    for index, binding in enumerate(decision_bindings, start=1):
        _validate_current_binding(binding, f"decision shard {index}")

    merge_manifest = {
        "schema_version": MERGE_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "structurally_merged_pending_manage_static_g0a_semantic_validation",
        "record_count": len(merged_rows),
        "base_fact_ids_sha256": sha256_value(expected_all_ids),
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "structural_validation_complete": True,
        "semantic_validation_performed": False,
        "next_validator": "scripts/manage_static_g0a.py",
        "parent_review_input_manifest": shard_manifest[
            "parent_review_input_manifest"
        ],
        "parent_review_input": shard_manifest["parent_review_input"],
        "parent_decision_template": shard_manifest["parent_decision_template"],
        "shard_manifest": _binding(
            Path(shard_manifest_path).resolve(),
            shard_manifest_payload,
            schema_version=SHARD_MANIFEST_SCHEMA,
        ),
        "decision_shards": decision_bindings,
        "merged_decisions": _binding(
            output_decisions_path,
            decisions_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(merged_rows),
            base_fact_ids=expected_all_ids,
        ),
    }
    manifest_payload = _json_payload(merge_manifest)

    # The decisions file is published first; the manifest is the commit marker.
    _atomic_write(output_decisions_path, decisions_payload)
    try:
        _atomic_write(output_manifest_path, manifest_payload)
    except Exception:
        try:
            output_decisions_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return merge_manifest


def _load_completed_decisions_for_review(
    decision_path: Path,
    input_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Path, bytes, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    path = Path(decision_path).resolve()
    payload = _read_bytes(path, "old Codex decisions")
    rows = _parse_jsonl_payload(payload, path, "old Codex decisions")
    ids = _unique_ids(rows, "old Codex decisions")
    if len(rows) != EXPECTED_RECORD_COUNT or set(ids) != set(input_by_id):
        raise ValueError("old Codex decisions do not exactly cover old review input")
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        base_fact_id = ids[index - 1]
        if row.get("schema_version") != CODEX_DECISION_SCHEMA:
            raise ValueError(f"old Codex decision row {index} schema_version mismatch")
        if row.get("reviewer_type") != "codex_proxy" or row.get("human_gold") is not False:
            raise ValueError(
                f"old Codex decision row {index} reviewer identity is invalid"
            )
        if row.get("terminal_status") != "completed":
            raise ValueError(f"old Codex decision row {index} is not completed")
        if row.get("review_input_record_sha256") != sha256_value(
            input_by_id[base_fact_id]
        ):
            raise ValueError(
                f"old Codex decision row {index} review input hash mismatch"
            )
        by_id[base_fact_id] = row
    return path, payload, rows, by_id


def rebase_decisions(
    old_review_input_manifest_path: Path,
    old_decisions_path: Path,
    new_review_input_manifest_path: Path,
    output_decisions_path: Path,
    output_manifest_path: Path,
) -> Dict[str, Any]:
    """Copy decisions only for byte-canonically unchanged review rows.

    A changed row is reset to the new review's untouched pending/defer template.
    In particular, this function never updates an old decision's input hash.
    """

    output_decisions_path = Path(output_decisions_path).resolve()
    output_manifest_path = Path(output_manifest_path).resolve()
    if output_decisions_path == output_manifest_path:
        raise ValueError("rebased decisions and rebase manifest paths must differ")
    for path in (output_decisions_path, output_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    (
        _,
        old_manifest_payload,
        old_input_path,
        old_input_payload,
        old_input_rows,
        _,
        _,
        _,
    ) = _load_parent_review(old_review_input_manifest_path)
    (
        _,
        new_manifest_payload,
        new_input_path,
        new_input_payload,
        new_input_rows,
        new_template_path,
        new_template_payload,
        new_template_rows,
    ) = _load_parent_review(new_review_input_manifest_path)
    old_input_by_id = {row["base_fact_id"]: row for row in old_input_rows}
    new_input_by_id = {row["base_fact_id"]: row for row in new_input_rows}
    if set(old_input_by_id) != set(new_input_by_id):
        raise ValueError("old and new review inputs have different base fact universes")
    new_template_by_id = {row["base_fact_id"]: row for row in new_template_rows}
    for base_fact_id, template in new_template_by_id.items():
        if template.get("terminal_status") != "pending" or template.get("decision") != "defer":
            raise ValueError(
                f"new decision template is not pending/defer: {base_fact_id}"
            )

    (
        old_decisions_path,
        old_decisions_payload,
        old_decision_rows,
        old_decision_by_id,
    ) = _load_completed_decisions_for_review(old_decisions_path, old_input_by_id)

    copied_ids: List[str] = []
    pending_ids: List[str] = []
    rebased_rows: List[Dict[str, Any]] = []
    for base_fact_id in sorted(new_input_by_id):
        old_hash = sha256_value(old_input_by_id[base_fact_id])
        new_hash = sha256_value(new_input_by_id[base_fact_id])
        if old_hash == new_hash:
            copied_ids.append(base_fact_id)
            rebased_rows.append(copy.deepcopy(old_decision_by_id[base_fact_id]))
        else:
            pending_ids.append(base_fact_id)
            template = copy.deepcopy(new_template_by_id[base_fact_id])
            if template.get("review_input_record_sha256") != new_hash:
                raise ValueError(
                    f"new decision template hash mismatch: {base_fact_id}"
                )
            rebased_rows.append(template)
    rebased_payload = _jsonl_payload(rebased_rows)

    old_manifest_path = Path(old_review_input_manifest_path).resolve()
    new_manifest_path = Path(new_review_input_manifest_path).resolve()
    inputs_to_recheck = (
        (old_manifest_path, old_manifest_payload, "old review input manifest"),
        (old_input_path, old_input_payload, "old review input"),
        (old_decisions_path, old_decisions_payload, "old Codex decisions"),
        (new_manifest_path, new_manifest_payload, "new review input manifest"),
        (new_input_path, new_input_payload, "new review input"),
        (new_template_path, new_template_payload, "new decision template"),
    )
    for path, payload, label in inputs_to_recheck:
        if _read_bytes(path, label) != payload:
            raise ValueError(f"{label} changed during decision rebase")

    ids = sorted(new_input_by_id)
    manifest = {
        "schema_version": REBASE_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "unchanged_decisions_copied_changed_rows_pending_review",
        "record_count": len(rebased_rows),
        "base_fact_ids_sha256": sha256_value(ids),
        "copied_count": len(copied_ids),
        "copied_base_fact_ids": copied_ids,
        "copied_base_fact_ids_sha256": sha256_value(copied_ids),
        "pending_count": len(pending_ids),
        "pending_base_fact_ids": pending_ids,
        "pending_base_fact_ids_sha256": sha256_value(pending_ids),
        "copy_rule": "canonical_review_input_record_sha256_exact_match_only",
        "changed_row_policy": "use_new_pending_defer_template_without_decision_reuse",
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "semantic_validation_performed": False,
        "old_review_input_manifest": _binding(
            old_manifest_path,
            old_manifest_payload,
            schema_version=REVIEW_INPUT_MANIFEST_SCHEMA,
        ),
        "old_review_input": _binding(
            old_input_path,
            old_input_payload,
            schema_version=REVIEW_INPUT_RECORD_SCHEMA,
            record_count=len(old_input_rows),
            base_fact_ids=ids,
        ),
        "old_decisions": _binding(
            old_decisions_path,
            old_decisions_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(old_decision_rows),
            base_fact_ids=ids,
        ),
        "new_review_input_manifest": _binding(
            new_manifest_path,
            new_manifest_payload,
            schema_version=REVIEW_INPUT_MANIFEST_SCHEMA,
        ),
        "new_review_input": _binding(
            new_input_path,
            new_input_payload,
            schema_version=REVIEW_INPUT_RECORD_SCHEMA,
            record_count=len(new_input_rows),
            base_fact_ids=ids,
        ),
        "new_decision_template": _binding(
            new_template_path,
            new_template_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(new_template_rows),
            base_fact_ids=ids,
        ),
        "rebased_decisions": _binding(
            output_decisions_path,
            rebased_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(rebased_rows),
            base_fact_ids=ids,
        ),
    }
    manifest_payload = _json_payload(manifest)
    _atomic_write(output_decisions_path, rebased_payload)
    try:
        _atomic_write(output_manifest_path, manifest_payload)
    except Exception:
        try:
            output_decisions_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return manifest


def _validate_completed_decision_for_review(
    decision: Mapping[str, Any],
    review: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    label: str,
) -> None:
    base_fact_id = _required_string(review.get("base_fact_id"), f"{label} review ID")
    if decision.get("schema_version") != CODEX_DECISION_SCHEMA:
        raise ValueError(f"{label} schema_version mismatch: {base_fact_id}")
    if decision.get("base_fact_id") != base_fact_id:
        raise ValueError(f"{label} base_fact_id mismatch: {base_fact_id}")
    if (
        decision.get("reviewer_type") != "codex_proxy"
        or decision.get("human_gold") is not False
    ):
        raise ValueError(f"{label} reviewer identity is invalid: {base_fact_id}")
    if decision.get("terminal_status") != "completed":
        raise ValueError(f"{label} is not completed: {base_fact_id}")
    if decision.get("decision") not in {"accept", "revise", "reject", "defer"}:
        raise ValueError(f"{label} decision is invalid: {base_fact_id}")
    if decision.get("review_input_record_sha256") != sha256_value(review):
        raise ValueError(f"{label} review input hash mismatch: {base_fact_id}")

    template_checks = template.get("checks")
    checks = decision.get("checks")
    if (
        not isinstance(template_checks, dict)
        or not template_checks
        or not isinstance(checks, dict)
        or set(checks) != set(template_checks)
        or any(type(value) is not bool for value in checks.values())
    ):
        raise ValueError(f"{label} checks are incomplete: {base_fact_id}")
    if decision.get("decision") in {"accept", "revise"} and not all(
        checks.values()
    ):
        raise ValueError(f"{label} accepted checks are not all true: {base_fact_id}")

    review_variants = review.get("variants")
    template_variants = template.get("variants")
    decision_variants = decision.get("variants")
    if not all(
        isinstance(value, list)
        for value in (review_variants, template_variants, decision_variants)
    ) or not (
        len(review_variants) == len(template_variants) == len(decision_variants)
    ):
        raise ValueError(f"{label} variant coverage is invalid: {base_fact_id}")
    review_by_distractor = {
        value.get("distractor_id"): value
        for value in review_variants
        if isinstance(value, dict)
    }
    template_by_distractor = {
        value.get("distractor_id"): value
        for value in template_variants
        if isinstance(value, dict)
    }
    if (
        len(review_by_distractor) != len(review_variants)
        or set(template_by_distractor) != set(review_by_distractor)
    ):
        raise ValueError(f"{label} review/template variants are invalid: {base_fact_id}")
    seen = set()
    for variant in decision_variants:
        if not isinstance(variant, dict):
            raise ValueError(f"{label} variant is invalid: {base_fact_id}")
        distractor_id = _required_string(
            variant.get("distractor_id"), f"{label} variant distractor_id"
        )
        if distractor_id in seen or distractor_id not in review_by_distractor:
            raise ValueError(f"{label} variant identity mismatch: {base_fact_id}")
        seen.add(distractor_id)
        if variant.get("decision") not in {"accept", "reject", "defer"}:
            raise ValueError(
                f"{label} variant decision is invalid: {base_fact_id}:{distractor_id}"
            )
        expected_checks = template_by_distractor[distractor_id].get("checks")
        variant_checks = variant.get("checks")
        if (
            not isinstance(expected_checks, dict)
            or not expected_checks
            or not isinstance(variant_checks, dict)
            or set(variant_checks) != set(expected_checks)
            or any(type(value) is not bool for value in variant_checks.values())
        ):
            raise ValueError(
                f"{label} variant checks are incomplete: {base_fact_id}:{distractor_id}"
            )
        if variant.get("decision") == "accept" and not all(
            variant_checks.values()
        ):
            raise ValueError(
                f"{label} accepted variant checks are not all true: "
                f"{base_fact_id}:{distractor_id}"
            )
        selected_candidate_id = _required_string(
            variant.get("selected_candidate_id"),
            f"{label} selected_candidate_id {base_fact_id}:{distractor_id}",
        )
        candidates = review_by_distractor[distractor_id].get(
            "context_candidates"
        )
        if not isinstance(candidates, list) or selected_candidate_id not in {
            value.get("candidate_id")
            for value in candidates
            if isinstance(value, dict)
        }:
            raise ValueError(
                f"{label} selected candidate is absent: "
                f"{base_fact_id}:{distractor_id}"
            )
    if seen != set(review_by_distractor):
        raise ValueError(f"{label} does not cover all variants: {base_fact_id}")


def complete_rebase(
    rebase_manifest_path: Path,
    pending_decisions_path: Path,
    output_decisions_path: Path,
    output_manifest_path: Path,
) -> Dict[str, Any]:
    """Replace only rebase-pending templates with newly completed decisions."""

    output_decisions_path = Path(output_decisions_path).resolve()
    output_manifest_path = Path(output_manifest_path).resolve()
    if output_decisions_path == output_manifest_path:
        raise ValueError("completed decisions and completion manifest paths must differ")
    for path in (output_decisions_path, output_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    rebase_manifest_path = Path(rebase_manifest_path).resolve()
    rebase_manifest_payload = _read_bytes(rebase_manifest_path, "rebase manifest")
    rebase_manifest = _parse_json_payload(
        rebase_manifest_payload, rebase_manifest_path, "rebase manifest"
    )
    if rebase_manifest.get("schema_version") != REBASE_MANIFEST_SCHEMA:
        raise ValueError("unsupported rebase manifest schema_version")
    if (
        rebase_manifest.get("status")
        != "unchanged_decisions_copied_changed_rows_pending_review"
        or rebase_manifest.get("record_count") != EXPECTED_RECORD_COUNT
        or rebase_manifest.get("copy_rule")
        != "canonical_review_input_record_sha256_exact_match_only"
        or rebase_manifest.get("changed_row_policy")
        != "use_new_pending_defer_template_without_decision_reuse"
        or rebase_manifest.get("reviewer_type") != "codex_proxy"
        or rebase_manifest.get("human_gold") is not False
        or rebase_manifest.get("semantic_validation_performed") is not False
    ):
        raise ValueError("rebase manifest contract is invalid")

    new_manifest_path, new_manifest_payload, _ = _validate_file_binding(
        rebase_manifest_path,
        rebase_manifest.get("new_review_input_manifest"),
        "new review input manifest",
        expected_schema=REVIEW_INPUT_MANIFEST_SCHEMA,
        jsonl=False,
    )
    (
        _,
        loaded_new_manifest_payload,
        new_input_path,
        new_input_payload,
        new_input_rows,
        new_template_path,
        new_template_payload,
        new_template_rows,
    ) = _load_parent_review(new_manifest_path)
    if loaded_new_manifest_payload != new_manifest_payload:
        raise ValueError("new review input manifest binding changed")
    for binding_name, expected_path, expected_payload, label in (
        (
            "new_review_input",
            new_input_path,
            new_input_payload,
            "new review input",
        ),
        (
            "new_decision_template",
            new_template_path,
            new_template_payload,
            "new decision template",
        ),
    ):
        bound_path, bound_payload = _validate_current_binding(
            rebase_manifest.get(binding_name), label
        )
        if bound_path != expected_path or bound_payload != expected_payload:
            raise ValueError(f"rebase {label} binding mismatch")

    rebased_path, rebased_payload, rebased_rows = _validate_file_binding(
        rebase_manifest_path,
        rebase_manifest.get("rebased_decisions"),
        "rebased decisions",
        expected_schema=CODEX_DECISION_SCHEMA,
        jsonl=True,
    )
    rebased_ids = _unique_ids(rebased_rows, "rebased decisions")
    new_input_by_id = {row["base_fact_id"]: row for row in new_input_rows}
    new_template_by_id = {row["base_fact_id"]: row for row in new_template_rows}
    expected_ids = sorted(new_input_by_id)
    if len(rebased_rows) != EXPECTED_RECORD_COUNT or set(rebased_ids) != set(
        expected_ids
    ):
        raise ValueError("rebased decisions do not cover the new review input")
    if rebase_manifest.get("base_fact_ids_sha256") != sha256_value(expected_ids):
        raise ValueError("rebase base_fact_ids_sha256 mismatch")

    copied_ids = rebase_manifest.get("copied_base_fact_ids")
    pending_ids = rebase_manifest.get("pending_base_fact_ids")
    if not isinstance(copied_ids, list) or not isinstance(pending_ids, list):
        raise ValueError("rebase copied/pending IDs must be arrays")
    copied_ids = [
        _required_string(value, "rebase copied base_fact_id") for value in copied_ids
    ]
    pending_ids = [
        _required_string(value, "rebase pending base_fact_id") for value in pending_ids
    ]
    if (
        copied_ids != sorted(set(copied_ids))
        or pending_ids != sorted(set(pending_ids))
        or set(copied_ids) & set(pending_ids)
        or sorted([*copied_ids, *pending_ids]) != expected_ids
        or rebase_manifest.get("copied_count") != len(copied_ids)
        or rebase_manifest.get("pending_count") != len(pending_ids)
        or rebase_manifest.get("copied_base_fact_ids_sha256")
        != sha256_value(copied_ids)
        or rebase_manifest.get("pending_base_fact_ids_sha256")
        != sha256_value(pending_ids)
    ):
        raise ValueError("rebase copied/pending ID contract is invalid")
    rebased_by_id = {row["base_fact_id"]: row for row in rebased_rows}
    for base_fact_id in copied_ids:
        _validate_completed_decision_for_review(
            rebased_by_id[base_fact_id],
            new_input_by_id[base_fact_id],
            new_template_by_id[base_fact_id],
            label="copied rebased decision",
        )
    for base_fact_id in pending_ids:
        template = new_template_by_id[base_fact_id]
        if (
            template.get("terminal_status") != "pending"
            or template.get("decision") != "defer"
            or rebased_by_id[base_fact_id] != template
        ):
            raise ValueError(
                f"rebased pending row differs from new template: {base_fact_id}"
            )

    pending_decisions_path = Path(pending_decisions_path).resolve()
    pending_payload = _read_bytes(pending_decisions_path, "pending decisions")
    pending_rows = _parse_jsonl_payload(
        pending_payload, pending_decisions_path, "pending decisions"
    )
    supplied_ids = _unique_ids(pending_rows, "pending decisions")
    if set(supplied_ids) != set(pending_ids) or len(supplied_ids) != len(
        pending_ids
    ):
        missing = sorted(set(pending_ids) - set(supplied_ids))
        extra = sorted(set(supplied_ids) - set(pending_ids))
        raise ValueError(
            f"pending decisions coverage mismatch; missing={missing}; extra={extra}"
        )
    pending_by_id = {row["base_fact_id"]: row for row in pending_rows}
    for base_fact_id in pending_ids:
        _validate_completed_decision_for_review(
            pending_by_id[base_fact_id],
            new_input_by_id[base_fact_id],
            new_template_by_id[base_fact_id],
            label="new pending decision",
        )

    completed_rows = [
        copy.deepcopy(
            pending_by_id.get(base_fact_id, rebased_by_id[base_fact_id])
        )
        for base_fact_id in expected_ids
    ]
    completed_payload = _jsonl_payload(completed_rows)
    for path, payload, label in (
        (rebase_manifest_path, rebase_manifest_payload, "rebase manifest"),
        (rebased_path, rebased_payload, "rebased decisions"),
        (pending_decisions_path, pending_payload, "pending decisions"),
        (new_manifest_path, new_manifest_payload, "new review input manifest"),
        (new_input_path, new_input_payload, "new review input"),
        (new_template_path, new_template_payload, "new decision template"),
    ):
        if _read_bytes(path, label) != payload:
            raise ValueError(f"{label} changed during rebase completion")

    completion_manifest = {
        "schema_version": REBASE_COMPLETION_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "rebase_completed_pending_manage_static_g0a_semantic_validation",
        "record_count": len(completed_rows),
        "base_fact_ids_sha256": sha256_value(expected_ids),
        "copied_decision_count": len(copied_ids),
        "newly_completed_decision_count": len(pending_ids),
        "newly_completed_base_fact_ids": pending_ids,
        "newly_completed_base_fact_ids_sha256": sha256_value(pending_ids),
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "structural_validation_complete": True,
        "semantic_validation_performed": False,
        "next_validator": "scripts/manage_static_g0a.py",
        "rebase_manifest": _binding(
            rebase_manifest_path,
            rebase_manifest_payload,
            schema_version=REBASE_MANIFEST_SCHEMA,
        ),
        "rebased_decisions": _binding(
            rebased_path,
            rebased_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(rebased_rows),
            base_fact_ids=expected_ids,
        ),
        "pending_decisions": _binding(
            pending_decisions_path,
            pending_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(pending_rows),
            base_fact_ids=pending_ids,
        ),
        "new_review_input_manifest": rebase_manifest["new_review_input_manifest"],
        "new_review_input": rebase_manifest["new_review_input"],
        "completed_decisions": _binding(
            output_decisions_path,
            completed_payload,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(completed_rows),
            base_fact_ids=expected_ids,
        ),
    }
    completion_manifest_payload = _json_payload(completion_manifest)
    _atomic_write(output_decisions_path, completed_payload)
    try:
        _atomic_write(output_manifest_path, completion_manifest_payload)
    except Exception:
        try:
            output_decisions_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return completion_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or structurally merge SHA-bound Codex review shards. "
            "A merge does not replace manage_static_g0a semantic validation."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create four immutable 40-row shards")
    create.add_argument("--review-input-manifest", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)

    merge = commands.add_parser("merge", help="structurally merge four completed decision shards")
    merge.add_argument("--shard-manifest", type=Path, required=True)
    merge.add_argument(
        "--decision",
        dest="decisions",
        action="append",
        type=Path,
        required=True,
        help="completed decision shard in shard order; pass exactly four times",
    )
    merge.add_argument("--output-decisions", type=Path, required=True)
    merge.add_argument("--output-manifest", type=Path, required=True)

    rebase = commands.add_parser(
        "rebase",
        help=(
            "copy old completed decisions only for unchanged review rows and "
            "leave changed rows at the new pending/defer template"
        ),
    )
    rebase.add_argument("--old-review-input-manifest", type=Path, required=True)
    rebase.add_argument("--old-decisions", type=Path, required=True)
    rebase.add_argument("--new-review-input-manifest", type=Path, required=True)
    rebase.add_argument("--output-decisions", type=Path, required=True)
    rebase.add_argument("--output-manifest", type=Path, required=True)

    complete = commands.add_parser(
        "complete-rebase",
        help=(
            "replace exactly the rebase-pending rows with newly completed "
            "decisions and emit a new immutable 160-row decision file"
        ),
    )
    complete.add_argument("--rebase-manifest", type=Path, required=True)
    complete.add_argument("--pending-decisions", type=Path, required=True)
    complete.add_argument("--output-decisions", type=Path, required=True)
    complete.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create":
        result = create_shards(args.review_input_manifest, args.output_dir)
    elif args.command == "merge":
        result = merge_decisions(
            args.shard_manifest,
            args.decisions,
            args.output_decisions,
            args.output_manifest,
        )
    elif args.command == "rebase":
        result = rebase_decisions(
            args.old_review_input_manifest,
            args.old_decisions,
            args.new_review_input_manifest,
            args.output_decisions,
            args.output_manifest,
        )
    else:
        result = complete_rebase(
            args.rebase_manifest,
            args.pending_decisions,
            args.output_decisions,
            args.output_manifest,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
