#!/usr/bin/env python3
"""Build one exact full-pool fact-review authority from parent plus supplement.

The parent run is immutable evidence for 8,966 completed decisions and three
failed items.  The supplement is allowed to cover exactly those three failed
IDs.  This tool verifies both run contracts and all bound artifacts, then
writes a deterministic 8,969-row decision artifact without rewriting either
source run.

No model, API, behavior evaluation, canonical freeze, or human-gold promotion
is performed here.
"""

from __future__ import annotations

import argparse
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


TOOL_VERSION = "full-public-benchmark-fact-review-composite-builder-v1"
MANIFEST_SCHEMA_VERSION = (
    "full-public-benchmark-fact-review-composite-authority-manifest-v1"
)
LINEAGE_SCHEMA_VERSION = (
    "full-public-benchmark-fact-review-composite-decision-lineage-v1"
)
STATUS = "completed_exact_full_export"

PARENT_RUN_MANIFEST_SCHEMA = "full-public-benchmark-fact-review-run-manifest-v1"
PARENT_TOOL_VERSION = "full-public-benchmark-fact-review-runner-v2"
PARENT_CHECKPOINT_SCHEMA = "full-public-benchmark-fact-review-checkpoint-v1"
SUPPLEMENT_RUN_MANIFEST_SCHEMA = (
    "full-public-benchmark-fact-review-supplement-run-manifest-v1"
)
SUPPLEMENT_TOOL_VERSION = "full-public-benchmark-fact-review-supplement-runner-v2"
SUPPLEMENT_REVIEW_METHOD = (
    "behavior_blind_dual_model_singleton_fact_review_recovery_v2"
)
SUPPLEMENT_RESPONSE_IDENTITY_ENFORCEMENT = (
    "runner_exact_post_response_without_static_proxy_guard_v2"
)
SUPPLEMENT_CHECKPOINT_SCHEMA = (
    "full-public-benchmark-fact-review-supplement-checkpoint-v1"
)
DECISION_SCHEMA = "public-benchmark-review-decision-v2"

EXPECTED_TOTAL_COUNT = 8969
EXPECTED_PARENT_COMPLETED_COUNT = 8966
EXPECTED_SUPPLEMENT_COUNT = 3

_SUPPLEMENT_RUNNER: Optional[Any] = None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_supplement_runner() -> Any:
    global _SUPPLEMENT_RUNNER
    if _SUPPLEMENT_RUNNER is None:
        path = Path(__file__).resolve().with_name(
            "run_full_public_benchmark_fact_review_supplement.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_full_fact_review_supplement_for_composite", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load supplement validator: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SUPPLEMENT_RUNNER = module
    return _SUPPLEMENT_RUNNER


def read_json(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object at {resolved}:{line_number}"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {resolved}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=str(resolved.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, resolved)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    relative_to: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    rendered = (
        resolved.relative_to(Path(relative_to).resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    result: Dict[str, Any] = {
        "filename" if relative_to is not None else "path": rendered,
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _resolve_binding_path(
    binding: Mapping[str, Any], owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    candidate = Path(_required_string(raw, f"{label}.path"))
    if not candidate.is_absolute():
        candidate = Path(owner_path).resolve().parent / candidate
    return candidate.resolve()


def _verify_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: Optional[str] = None,
    jsonl: bool = False,
    allow_empty: bool = False,
) -> Tuple[Path, Any, Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    path = _resolve_binding_path(binding, owner_path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if "byte_count" in binding and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count is stale")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
    value: Any = (
        read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    )
    if jsonl and "record_count" in binding and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record_count is stale")
    return path, value, dict(binding)


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label}[{row_number}].{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = dict(row)
    return result


def _same_binding(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    actual_owner: Path,
    expected_owner: Path,
    label: str,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} binding is missing")
    if _resolve_binding_path(actual, actual_owner, label) != _resolve_binding_path(
        expected, expected_owner, label
    ):
        raise ValueError(f"{label} path differs from parent authority")
    if actual.get("sha256") != expected.get("sha256"):
        raise ValueError(f"{label} SHA-256 differs from parent authority")
    for field in ("record_count", "schema_version"):
        if (
            field in actual
            and field in expected
            and actual.get(field) != expected.get(field)
        ):
            raise ValueError(f"{label} {field} differs from parent authority")


def _validate_run_contract(
    manifest: Mapping[str, Any], *, label: str
) -> Tuple[Dict[str, Any], str]:
    contract = manifest.get("run_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{label} run_contract is missing")
    digest = manifest.get("run_contract_sha256")
    if digest != sha256_value(contract):
        raise ValueError(f"{label} run_contract SHA-256 is stale")
    if contract.get("human_gold") is not False:
        raise ValueError(f"{label} run_contract must keep human_gold=false")
    return dict(contract), str(digest)


def _validate_decision(
    decision: Mapping[str, Any], *, base_fact_id: str, label: str
) -> None:
    if decision.get("schema_version") != DECISION_SCHEMA:
        raise ValueError(f"{label} decision schema is unsupported: {base_fact_id}")
    if decision.get("base_fact_id") != base_fact_id:
        raise ValueError(f"{label} decision base_fact_id is stale: {base_fact_id}")
    if decision.get("human_gold") is not False:
        raise ValueError(f"{label} decision became human gold: {base_fact_id}")
    if decision.get("decision") not in {"accept", "reject", "defer", "revise"}:
        raise ValueError(f"{label} decision is not terminal: {base_fact_id}")


def _validate_checkpoint_sequence(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    checkpoint_schema: str,
    tool_version: str,
    run_contract_sha256: str,
    expected_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    _unique_index(rows, "base_fact_id", f"{label} checkpoint")
    completed_ids: List[str] = []
    failed_ids: List[str] = []
    completed_decisions: List[Dict[str, Any]] = []
    for offset, row in enumerate(rows):
        base_fact_id = _required_string(
            row.get("base_fact_id"), f"{label} checkpoint[{offset}].base_fact_id"
        )
        if row.get("schema_version") != checkpoint_schema:
            raise ValueError(f"{label} checkpoint schema is unsupported: {base_fact_id}")
        if row.get("tool_version") != tool_version:
            raise ValueError(f"{label} checkpoint tool_version is stale: {base_fact_id}")
        if row.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(f"{label} checkpoint run contract is stale: {base_fact_id}")
        if row.get("human_gold") is not False or row.get("behavior_blind") is not True:
            raise ValueError(f"{label} checkpoint safety contract is invalid: {base_fact_id}")
        if expected_ids is not None and base_fact_id != expected_ids[offset]:
            raise ValueError(f"{label} checkpoint ID order is stale: {base_fact_id}")
        terminal_status = row.get("terminal_status")
        decision = row.get("strict_decision")
        if terminal_status == "completed":
            if not isinstance(decision, dict):
                raise ValueError(
                    f"{label} completed checkpoint lacks a decision: {base_fact_id}"
                )
            _validate_decision(decision, base_fact_id=base_fact_id, label=label)
            completed_ids.append(base_fact_id)
            completed_decisions.append(dict(decision))
        elif terminal_status == "failed":
            if decision is not None:
                raise ValueError(
                    f"{label} failed checkpoint contains a decision: {base_fact_id}"
                )
            failed_ids.append(base_fact_id)
        else:
            raise ValueError(
                f"{label} checkpoint terminal_status is invalid: {base_fact_id}"
            )
    return completed_ids, failed_ids, completed_decisions


def _validate_decisions_equal_checkpoint(
    decisions: Sequence[Mapping[str, Any]],
    checkpoint_decisions: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if list(decisions) != list(checkpoint_decisions):
        raise ValueError(f"{label} decisions differ from completed checkpoint decisions")
    _unique_index(decisions, "base_fact_id", f"{label} decisions")
    for row in decisions:
        base_fact_id = _required_string(
            row.get("base_fact_id"), f"{label} decision.base_fact_id"
        )
        _validate_decision(row, base_fact_id=base_fact_id, label=label)


def _validate_export_bindings(
    manifest: Mapping[str, Any], manifest_path: Path, *, label: str
) -> Dict[str, Dict[str, Any]]:
    export_path, export, export_binding = _verify_binding(
        manifest.get("export_manifest"),
        owner_path=manifest_path,
        label=f"{label} export_manifest",
    )
    items_path, _, items_binding = _verify_binding(
        manifest.get("review_items"),
        owner_path=manifest_path,
        label=f"{label} review_items",
        jsonl=True,
    )
    template_path, _, template_binding = _verify_binding(
        manifest.get("decision_template"),
        owner_path=manifest_path,
        label=f"{label} decision_template",
        jsonl=True,
    )
    if export.get("base_fact_count") != EXPECTED_TOTAL_COUNT:
        raise ValueError(f"{label} export is not the exact 8,969-item scope")
    artifacts = export.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{label} export artifacts are missing")
    _same_binding(
        items_binding,
        artifacts.get("review_items"),
        actual_owner=manifest_path,
        expected_owner=export_path,
        label=f"{label} review_items",
    )
    _same_binding(
        template_binding,
        artifacts.get("review_decisions_template"),
        actual_owner=manifest_path,
        expected_owner=export_path,
        label=f"{label} decision_template",
    )
    return {
        "export_manifest": export_binding,
        "review_items": items_binding,
        "decision_template": template_binding,
        "export_manifest_path": {"path": str(export_path)},
        "review_items_path": {"path": str(items_path)},
        "decision_template_path": {"path": str(template_path)},
    }


def _load_parent_authority(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != PARENT_RUN_MANIFEST_SCHEMA:
        raise ValueError("Parent run manifest schema is unsupported")
    if manifest.get("tool_version") != PARENT_TOOL_VERSION:
        raise ValueError("Parent fact-review runner version is unsupported")
    if manifest.get("status") != "completed_with_failures":
        raise ValueError("Parent run must be completed_with_failures")
    if manifest.get("human_gold") is not False:
        raise ValueError("Parent run must keep human_gold=false")
    contract, contract_sha = _validate_run_contract(manifest, label="parent")
    if not all(
        (
            manifest.get("selected_count") == EXPECTED_TOTAL_COUNT,
            manifest.get("full_export_count") == EXPECTED_TOTAL_COUNT,
            contract.get("selected_count") == EXPECTED_TOTAL_COUNT,
            contract.get("full_export_count") == EXPECTED_TOTAL_COUNT,
            contract.get("selection_is_full_export") is True,
            contract.get("behavior_blind") is True,
        )
    ):
        raise ValueError("Parent run is not the exact full 8,969-item export")
    export_bindings = _validate_export_bindings(manifest, path, label="parent")
    supplement_runner = _load_supplement_runner()
    parent_runner = supplement_runner.parent_runner
    _, loaded_items_path, loaded_templates_path, units = parent_runner.load_review_units(
        export_manifest_path=Path(
            export_bindings["export_manifest_path"]["path"]
        ),
        review_items_path=Path(export_bindings["review_items_path"]["path"]),
        decision_template_path=Path(
            export_bindings["decision_template_path"]["path"]
        ),
    )
    if (
        loaded_items_path
        != Path(export_bindings["review_items_path"]["path"])
        or loaded_templates_path
        != Path(export_bindings["decision_template_path"]["path"])
    ):
        raise ValueError("Parent review-unit bindings did not replay exactly")
    units_by_id = {unit.base_fact_id: unit for unit in units}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Parent run artifacts are missing")
    checkpoint_path, checkpoints, checkpoint_binding = _verify_binding(
        artifacts.get("checkpoint"),
        owner_path=path,
        label="parent checkpoint",
        expected_schema=PARENT_CHECKPOINT_SCHEMA,
        jsonl=True,
    )
    decisions_path, decisions, decisions_binding = _verify_binding(
        artifacts.get("decisions"),
        owner_path=path,
        label="parent decisions",
        expected_schema=DECISION_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    completed_ids, failed_ids, checkpoint_decisions = _validate_checkpoint_sequence(
        checkpoints,
        label="parent",
        checkpoint_schema=PARENT_CHECKPOINT_SCHEMA,
        tool_version=PARENT_TOOL_VERSION,
        run_contract_sha256=contract_sha,
    )
    for row in checkpoints:
        parent_runner._validate_checkpoint_row(
            row,
            units_by_id=units_by_id,
            run_contract_sha256=contract_sha,
        )
    if len(checkpoints) != EXPECTED_TOTAL_COUNT:
        raise ValueError("Parent checkpoint must contain exactly 8,969 rows")
    if len(completed_ids) != EXPECTED_PARENT_COMPLETED_COUNT:
        raise ValueError("Parent completed set must contain exactly 8,966 IDs")
    if len(failed_ids) != EXPECTED_SUPPLEMENT_COUNT:
        raise ValueError("Parent failed set must contain exactly three IDs")
    expected_terminal_counts = {"failed": EXPECTED_SUPPLEMENT_COUNT}
    if EXPECTED_PARENT_COMPLETED_COUNT:
        expected_terminal_counts["completed"] = EXPECTED_PARENT_COMPLETED_COUNT
    if manifest.get("terminal_status_counts") != dict(
        sorted(expected_terminal_counts.items())
    ):
        raise ValueError("Parent terminal_status_counts are stale")
    if contract.get("selected_base_fact_ids_sha256") != sha256_value(
        [str(row["base_fact_id"]) for row in checkpoints]
    ):
        raise ValueError("Parent selected ID order is stale")
    _validate_decisions_equal_checkpoint(
        decisions, checkpoint_decisions, label="parent"
    )
    if len(decisions) != EXPECTED_PARENT_COMPLETED_COUNT:
        raise ValueError("Parent decisions must contain exactly 8,966 rows")
    decision_counts = dict(
        sorted(Counter(str(row["decision"]) for row in decisions).items())
    )
    if manifest.get("decision_counts") != decision_counts:
        raise ValueError("Parent decision_counts are stale")
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(
            path, schema_version=PARENT_RUN_MANIFEST_SCHEMA
        ),
        "run_contract": contract,
        "run_contract_sha256": contract_sha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_binding": checkpoint_binding,
        "checkpoints": checkpoints,
        "decisions_path": decisions_path,
        "decisions_binding": decisions_binding,
        "decisions": decisions,
        "completed_ids": completed_ids,
        "failed_ids": failed_ids,
        "checkpoint_decisions": checkpoint_decisions,
        "units_by_id": units_by_id,
        **export_bindings,
    }


def _load_supplement_authority(
    path: Path, *, parent: Mapping[str, Any]
) -> Dict[str, Any]:
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != SUPPLEMENT_RUN_MANIFEST_SCHEMA:
        raise ValueError("Supplement run manifest schema is unsupported")
    if manifest.get("tool_version") != SUPPLEMENT_TOOL_VERSION:
        raise ValueError("Supplement fact-review runner version is unsupported")
    if manifest.get("status") != "completed":
        raise ValueError("Supplement run is not complete")
    if manifest.get("human_gold") is not False:
        raise ValueError("Supplement run must keep human_gold=false")
    contract, contract_sha = _validate_run_contract(manifest, label="supplement")
    failed_ids = list(parent["failed_ids"])
    if not all(
        (
            manifest.get("selected_count") == EXPECTED_SUPPLEMENT_COUNT,
            manifest.get("batch_size") == 1,
            manifest.get("failed_base_fact_ids") == failed_ids,
            manifest.get("failed_base_fact_ids_sha256") == sha256_value(failed_ids),
            manifest.get("parent_run_contract_sha256")
            == parent["run_contract_sha256"],
            contract.get("selected_count") == EXPECTED_SUPPLEMENT_COUNT,
            contract.get("failed_base_fact_ids") == failed_ids,
            contract.get("failed_base_fact_ids_sha256") == sha256_value(failed_ids),
            contract.get("batch_size") == 1,
            contract.get("behavior_blind") is True,
            contract.get("human_gold") is False,
            contract.get("tool_version") == SUPPLEMENT_TOOL_VERSION,
            contract.get("review_method") == SUPPLEMENT_REVIEW_METHOD,
            contract.get("response_identity_enforcement")
            == SUPPLEMENT_RESPONSE_IDENTITY_ENFORCEMENT,
            contract.get("parent_selected_count") == EXPECTED_TOTAL_COUNT,
            contract.get("parent_completed_count")
            == EXPECTED_PARENT_COMPLETED_COUNT,
            contract.get("parent_failed_count") == EXPECTED_SUPPLEMENT_COUNT,
            contract.get("parent_decisions_count")
            == EXPECTED_PARENT_COMPLETED_COUNT,
        )
    ):
        raise ValueError("Supplement selection does not exactly match parent failures")
    runtime_sources = contract.get("runtime_sources")
    if not isinstance(runtime_sources, dict):
        raise ValueError("Supplement runtime source bindings are missing")
    _load_supplement_runner()._verify_bindings_unchanged(runtime_sources)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Supplement inputs are missing")
    expected_input_bindings = {
        "parent_run_manifest": parent["manifest_binding"],
        "parent_checkpoint": parent["checkpoint_binding"],
        "parent_decisions": parent["decisions_binding"],
        "export_manifest": parent["export_manifest"],
        "review_items": parent["review_items"],
        "decision_template": parent["decision_template"],
    }
    for label, expected in expected_input_bindings.items():
        _same_binding(
            inputs.get(label),
            expected,
            actual_owner=path,
            expected_owner=parent["manifest_path"],
            label=f"supplement {label}",
        )
    expected_contract_hashes = {
        "parent_run_manifest_sha256": parent["manifest_binding"]["sha256"],
        "parent_checkpoint_sha256": parent["checkpoint_binding"]["sha256"],
        "parent_decisions_sha256": parent["decisions_binding"]["sha256"],
        "export_manifest_sha256": parent["export_manifest"]["sha256"],
        "review_items_sha256": parent["review_items"]["sha256"],
        "decision_template_sha256": parent["decision_template"]["sha256"],
        "parent_run_contract_sha256": parent["run_contract_sha256"],
    }
    for field, expected in expected_contract_hashes.items():
        if contract.get(field) != expected:
            raise ValueError(f"Supplement run_contract {field} is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Supplement artifacts are missing")
    checkpoint_path, checkpoints, checkpoint_binding = _verify_binding(
        artifacts.get("checkpoint"),
        owner_path=path,
        label="supplement checkpoint",
        expected_schema=SUPPLEMENT_CHECKPOINT_SCHEMA,
        jsonl=True,
    )
    decisions_path, decisions, decisions_binding = _verify_binding(
        artifacts.get("decisions"),
        owner_path=path,
        label="supplement decisions",
        expected_schema=DECISION_SCHEMA,
        jsonl=True,
    )
    completed_ids, supplement_failed_ids, checkpoint_decisions = (
        _validate_checkpoint_sequence(
            checkpoints,
            label="supplement",
            checkpoint_schema=SUPPLEMENT_CHECKPOINT_SCHEMA,
            tool_version=SUPPLEMENT_TOOL_VERSION,
            run_contract_sha256=contract_sha,
            expected_ids=failed_ids,
        )
    )
    generator_spec = contract.get("generator_model_spec")
    reviewer_spec = contract.get("reviewer_model_spec")
    if not isinstance(generator_spec, dict) or not isinstance(reviewer_spec, dict):
        raise ValueError("Supplement model specifications are missing")
    expected_model_specs = {
        "fact_review_supplement_generator": generator_spec,
        "fact_review_supplement_reviewer": reviewer_spec,
    }
    expected_response_models = [
        _required_string(
            spec.get("expected_response_model"),
            f"supplement {stage} expected_response_model",
        )
        for stage, spec in expected_model_specs.items()
    ]
    if len(set(expected_response_models)) != 2:
        raise ValueError("Supplement generator and reviewer identities must differ")
    supplement_runner = _load_supplement_runner()
    parent_checkpoint_index = _unique_index(
        parent["checkpoints"], "base_fact_id", "parent checkpoint"
    )
    for row in checkpoints:
        supplement_runner._validate_checkpoint_row(
            row,
            units_by_id=parent["units_by_id"],
            parent_rows=parent_checkpoint_index,
            run_contract_sha256=contract_sha,
            expected_model_specs=expected_model_specs,
        )
    if completed_ids != failed_ids or supplement_failed_ids:
        raise ValueError("Supplement must complete exactly the three parent failures")
    if manifest.get("terminal_status_counts") != {
        "completed": EXPECTED_SUPPLEMENT_COUNT
    }:
        raise ValueError("Supplement terminal_status_counts are stale")
    _validate_decisions_equal_checkpoint(
        decisions, checkpoint_decisions, label="supplement"
    )
    if [str(row["base_fact_id"]) for row in decisions] != failed_ids:
        raise ValueError("Supplement decision order differs from parent failed order")
    for row in checkpoints:
        base_fact_id = str(row["base_fact_id"])
        parent_failed_row = parent_checkpoint_index[base_fact_id]
        if row.get("input_index") != parent_failed_row.get("input_index"):
            raise ValueError(
                f"Supplement input_index differs from parent failure: {base_fact_id}"
            )
        if row.get("parent_failed_checkpoint_record_sha256") != sha256_value(
            parent_failed_row
        ):
            raise ValueError(
                f"Supplement parent failure binding is stale: {base_fact_id}"
            )
    decision_counts = dict(
        sorted(Counter(str(row["decision"]) for row in decisions).items())
    )
    if manifest.get("decision_counts") != decision_counts:
        raise ValueError("Supplement decision_counts are stale")
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(
            path, schema_version=SUPPLEMENT_RUN_MANIFEST_SCHEMA
        ),
        "run_contract": contract,
        "run_contract_sha256": contract_sha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_binding": checkpoint_binding,
        "checkpoints": checkpoints,
        "decisions_path": decisions_path,
        "decisions_binding": decisions_binding,
        "decisions": decisions,
        "completed_ids": completed_ids,
    }


def _authority_summary(authority: Mapping[str, Any]) -> Dict[str, Any]:
    decisions = authority["decisions"]
    return {
        "run_manifest": authority["manifest_binding"],
        "run_contract_sha256": authority["run_contract_sha256"],
        "checkpoint": authority["checkpoint_binding"],
        "decisions": authority["decisions_binding"],
        "ordered_decision_record_sha256s_sha256": sha256_value(
            [sha256_value(row) for row in decisions]
        ),
        "decision_count": len(decisions),
        "human_gold": False,
    }


def build_composite_authority(
    *,
    parent_run_manifest_path: Path,
    supplement_run_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    parent = _load_parent_authority(parent_run_manifest_path)
    supplement = _load_supplement_authority(
        supplement_run_manifest_path, parent=parent
    )
    parent_ids = list(parent["completed_ids"])
    supplement_ids = list(supplement["completed_ids"])
    if set(parent_ids).intersection(supplement_ids):
        raise ValueError("Parent and supplement decisions overlap")
    if supplement_ids != parent["failed_ids"]:
        raise ValueError("Supplement IDs do not exactly equal parent failed IDs")

    parent_decisions = {
        str(row["base_fact_id"]): dict(row) for row in parent["decisions"]
    }
    supplement_decisions = {
        str(row["base_fact_id"]): dict(row) for row in supplement["decisions"]
    }
    checkpoint_ids = [
        str(row["base_fact_id"]) for row in parent["checkpoints"]
    ]
    if set(checkpoint_ids) != set(parent_decisions) | set(supplement_decisions):
        raise ValueError("Composite decision union has omissions or unknown IDs")
    composite_decisions: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    parent_checkpoint_index = _unique_index(
        parent["checkpoints"], "base_fact_id", "parent checkpoint"
    )
    supplement_checkpoint_index = _unique_index(
        supplement["checkpoints"], "base_fact_id", "supplement checkpoint"
    )
    for selection_index, base_fact_id in enumerate(checkpoint_ids):
        if base_fact_id in parent_decisions:
            source = "parent_completed"
            authority = parent
            checkpoint = parent_checkpoint_index[base_fact_id]
            decision = parent_decisions[base_fact_id]
        else:
            source = "supplement_recovery"
            authority = supplement
            checkpoint = supplement_checkpoint_index[base_fact_id]
            decision = supplement_decisions[base_fact_id]
        composite_decisions.append(dict(decision))
        lineage.append(
            {
                "schema_version": LINEAGE_SCHEMA_VERSION,
                "selection_index": selection_index,
                "base_fact_id": base_fact_id,
                "source_authority": source,
                "source_run_manifest_sha256": authority["manifest_binding"][
                    "sha256"
                ],
                "source_run_contract_sha256": authority["run_contract_sha256"],
                "source_checkpoint_record_sha256": sha256_value(checkpoint),
                "source_decision_record_sha256": sha256_value(decision),
                "human_gold": False,
            }
        )
    if len(composite_decisions) != EXPECTED_TOTAL_COUNT:
        raise ValueError("Composite authority must contain exactly 8,969 decisions")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    decisions_path = output_dir / "composite_review_decisions.jsonl"
    lineage_path = output_dir / "composite_decision_lineage.jsonl"
    manifest_path = output_dir / "fact_review_composite_authority_manifest.json"
    parent_hashes_before = {
        "manifest": sha256_file(parent["manifest_path"]),
        "checkpoint": sha256_file(parent["checkpoint_path"]),
        "decisions": sha256_file(parent["decisions_path"]),
    }
    try:
        write_jsonl(decisions_path, composite_decisions)
        write_jsonl(lineage_path, lineage)
        decision_binding = _binding(
            decisions_path,
            schema_version=DECISION_SCHEMA,
            record_count=len(composite_decisions),
            relative_to=output_dir,
        )
        lineage_binding = _binding(
            lineage_path,
            schema_version=LINEAGE_SCHEMA_VERSION,
            record_count=len(lineage),
            relative_to=output_dir,
        )
        parent_summary = _authority_summary(parent)
        parent_summary.update(
            {
                "completed_base_fact_ids_sha256": sha256_value(parent_ids),
                "failed_base_fact_ids": list(parent["failed_ids"]),
                "failed_base_fact_ids_sha256": sha256_value(parent["failed_ids"]),
            }
        )
        supplement_summary = _authority_summary(supplement)
        supplement_summary.update(
            {
                "selected_base_fact_ids": supplement_ids,
                "selected_base_fact_ids_sha256": sha256_value(supplement_ids),
            }
        )
        authority_payload = {
            "parent_run_manifest_sha256": parent["manifest_binding"]["sha256"],
            "parent_run_contract_sha256": parent["run_contract_sha256"],
            "supplement_run_manifest_sha256": supplement["manifest_binding"][
                "sha256"
            ],
            "supplement_run_contract_sha256": supplement["run_contract_sha256"],
            "composite_decisions_sha256": decision_binding["sha256"],
            "lineage_sha256": lineage_binding["sha256"],
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": STATUS,
            "authority_id": "fact_review_composite_"
            + sha256_value(authority_payload)[:20],
            "scope_id": parent["manifest"].get("scope_id"),
            "export_manifest": parent["export_manifest"],
            "review_items": parent["review_items"],
            "decision_template": parent["decision_template"],
            "parent_authority": parent_summary,
            "supplement_authority": supplement_summary,
            "counts": {
                "parent_completed": EXPECTED_PARENT_COMPLETED_COUNT,
                "parent_failed": EXPECTED_SUPPLEMENT_COUNT,
                "supplement_completed": EXPECTED_SUPPLEMENT_COUNT,
                "composite_decisions": EXPECTED_TOTAL_COUNT,
            },
            "composition": {
                "ordered_base_fact_ids_sha256": sha256_value(checkpoint_ids),
                "parent_and_supplement_overlap_count": 0,
                "missing_base_fact_count": 0,
                "unknown_base_fact_count": 0,
                "supplement_exactly_equals_parent_failed_set": True,
                "parent_decision_rows_copied_without_change": True,
                "parent_decisions_artifact_modified": False,
            },
            "artifacts": {
                "decisions": decision_binding,
                "decision_lineage": lineage_binding,
            },
            "human_gold": False,
            "safety_contract": {
                "source_run_artifacts_modified": False,
                "external_api_or_model_used": False,
                "behavior_execution_performed": False,
                "hf_model_execution_performed": False,
                "canonical_freeze_performed": False,
                "allow_partial_apply_authorized": False,
                "human_gold": False,
            },
        }
        write_json(manifest_path, manifest)
        parent_hashes_after = {
            "manifest": sha256_file(parent["manifest_path"]),
            "checkpoint": sha256_file(parent["checkpoint_path"]),
            "decisions": sha256_file(parent["decisions_path"]),
        }
        if parent_hashes_after != parent_hashes_before:
            raise RuntimeError("Parent authority changed while building composite")
        loaded = load_composite_authority(manifest_path)
        return {
            "status": loaded["manifest"]["status"],
            "authority_id": loaded["manifest"]["authority_id"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "decisions_path": str(decisions_path),
            "decisions_sha256": sha256_file(decisions_path),
            "decision_count": len(composite_decisions),
        }
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def load_composite_authority(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Composite authority manifest schema is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION or manifest.get("status") != STATUS:
        raise ValueError("Composite authority manifest is incomplete")
    if manifest.get("human_gold") is not False:
        raise ValueError("Composite authority must keep human_gold=false")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or safety.get("allow_partial_apply_authorized") is not False:
        raise ValueError("Composite authority safety contract is invalid")
    parent_summary = manifest.get("parent_authority")
    supplement_summary = manifest.get("supplement_authority")
    if not isinstance(parent_summary, dict) or not isinstance(
        supplement_summary, dict
    ):
        raise ValueError("Composite source authorities are missing")
    parent_manifest_path, _, _ = _verify_binding(
        parent_summary.get("run_manifest"),
        owner_path=path,
        label="composite parent run manifest",
        expected_schema=PARENT_RUN_MANIFEST_SCHEMA,
    )
    parent = _load_parent_authority(parent_manifest_path)
    expected_parent_summary = _authority_summary(parent)
    expected_parent_summary.update(
        {
            "completed_base_fact_ids_sha256": sha256_value(
                parent["completed_ids"]
            ),
            "failed_base_fact_ids": list(parent["failed_ids"]),
            "failed_base_fact_ids_sha256": sha256_value(parent["failed_ids"]),
        }
    )
    if parent_summary != expected_parent_summary:
        raise ValueError("Composite parent authority binding is stale")
    supplement_manifest_path, _, _ = _verify_binding(
        supplement_summary.get("run_manifest"),
        owner_path=path,
        label="composite supplement run manifest",
        expected_schema=SUPPLEMENT_RUN_MANIFEST_SCHEMA,
    )
    supplement = _load_supplement_authority(
        supplement_manifest_path, parent=parent
    )
    expected_supplement_summary = _authority_summary(supplement)
    expected_supplement_summary.update(
        {
            "selected_base_fact_ids": list(supplement["completed_ids"]),
            "selected_base_fact_ids_sha256": sha256_value(
                supplement["completed_ids"]
            ),
        }
    )
    if supplement_summary != expected_supplement_summary:
        raise ValueError("Composite supplement authority binding is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Composite authority artifacts are missing")
    decisions_path, decisions, decision_binding = _verify_binding(
        artifacts.get("decisions"),
        owner_path=path,
        label="composite decisions",
        expected_schema=DECISION_SCHEMA,
        jsonl=True,
    )
    lineage_path, lineage, lineage_binding = _verify_binding(
        artifacts.get("decision_lineage"),
        owner_path=path,
        label="composite decision lineage",
        expected_schema=LINEAGE_SCHEMA_VERSION,
        jsonl=True,
    )
    if len(decisions) != EXPECTED_TOTAL_COUNT or len(lineage) != EXPECTED_TOTAL_COUNT:
        raise ValueError("Composite authority does not contain exactly 8,969 rows")
    decision_index = _unique_index(decisions, "base_fact_id", "composite decisions")
    lineage_index = _unique_index(lineage, "base_fact_id", "composite lineage")
    ordered_ids = [str(row["base_fact_id"]) for row in decisions]
    if set(decision_index) != set(lineage_index):
        raise ValueError("Composite decision lineage does not exactly cover decisions")
    for selection_index, (decision, row) in enumerate(zip(decisions, lineage)):
        base_fact_id = str(decision["base_fact_id"])
        _validate_decision(decision, base_fact_id=base_fact_id, label="composite")
        if not all(
            (
                row.get("schema_version") == LINEAGE_SCHEMA_VERSION,
                row.get("selection_index") == selection_index,
                row.get("base_fact_id") == base_fact_id,
                row.get("source_authority")
                in {"parent_completed", "supplement_recovery"},
                row.get("source_decision_record_sha256") == sha256_value(decision),
                row.get("human_gold") is False,
            )
        ):
            raise ValueError(f"Composite decision lineage is stale: {base_fact_id}")
    parent_decisions = {
        str(row["base_fact_id"]): dict(row) for row in parent["decisions"]
    }
    supplement_decisions = {
        str(row["base_fact_id"]): dict(row) for row in supplement["decisions"]
    }
    parent_checkpoints = _unique_index(
        parent["checkpoints"], "base_fact_id", "parent checkpoint"
    )
    supplement_checkpoints = _unique_index(
        supplement["checkpoints"], "base_fact_id", "supplement checkpoint"
    )
    expected_decisions: List[Dict[str, Any]] = []
    expected_lineage: List[Dict[str, Any]] = []
    for selection_index, base_fact_id in enumerate(
        str(row["base_fact_id"]) for row in parent["checkpoints"]
    ):
        if base_fact_id in parent_decisions:
            source = "parent_completed"
            authority = parent
            checkpoint = parent_checkpoints[base_fact_id]
            decision = parent_decisions[base_fact_id]
        else:
            source = "supplement_recovery"
            authority = supplement
            checkpoint = supplement_checkpoints[base_fact_id]
            decision = supplement_decisions[base_fact_id]
        expected_decisions.append(decision)
        expected_lineage.append(
            {
                "schema_version": LINEAGE_SCHEMA_VERSION,
                "selection_index": selection_index,
                "base_fact_id": base_fact_id,
                "source_authority": source,
                "source_run_manifest_sha256": authority["manifest_binding"][
                    "sha256"
                ],
                "source_run_contract_sha256": authority["run_contract_sha256"],
                "source_checkpoint_record_sha256": sha256_value(checkpoint),
                "source_decision_record_sha256": sha256_value(decision),
                "human_gold": False,
            }
        )
    if decisions != expected_decisions or lineage != expected_lineage:
        raise ValueError("Composite decisions do not replay from bound authorities")
    counts = manifest.get("counts")
    if counts != {
        "parent_completed": EXPECTED_PARENT_COMPLETED_COUNT,
        "parent_failed": EXPECTED_SUPPLEMENT_COUNT,
        "supplement_completed": EXPECTED_SUPPLEMENT_COUNT,
        "composite_decisions": EXPECTED_TOTAL_COUNT,
    }:
        raise ValueError("Composite authority counts are stale")
    composition = manifest.get("composition")
    if not isinstance(composition, dict) or not all(
        (
            composition.get("ordered_base_fact_ids_sha256")
            == sha256_value(ordered_ids),
            composition.get("parent_and_supplement_overlap_count") == 0,
            composition.get("missing_base_fact_count") == 0,
            composition.get("unknown_base_fact_count") == 0,
            composition.get("supplement_exactly_equals_parent_failed_set") is True,
            composition.get("parent_decision_rows_copied_without_change") is True,
            composition.get("parent_decisions_artifact_modified") is False,
        )
    ):
        raise ValueError("Composite authority composition contract is invalid")
    expected_authority_id = "fact_review_composite_" + sha256_value(
        {
            "parent_run_manifest_sha256": parent["manifest_binding"]["sha256"],
            "parent_run_contract_sha256": parent["run_contract_sha256"],
            "supplement_run_manifest_sha256": supplement["manifest_binding"][
                "sha256"
            ],
            "supplement_run_contract_sha256": supplement["run_contract_sha256"],
            "composite_decisions_sha256": decision_binding["sha256"],
            "lineage_sha256": lineage_binding["sha256"],
        }
    )[:20]
    if manifest.get("authority_id") != expected_authority_id:
        raise ValueError("Composite authority_id is stale")
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(path, schema_version=MANIFEST_SCHEMA_VERSION),
        "decisions": decisions,
        "decisions_path": decisions_path,
        "decisions_binding": decision_binding,
        "lineage": lineage,
        "lineage_path": lineage_path,
        "lineage_binding": lineage_binding,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parent-run-manifest", type=Path, required=True)
    result.add_argument("--supplement-run-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = build_composite_authority(
        parent_run_manifest_path=args.parent_run_manifest,
        supplement_run_manifest_path=args.supplement_run_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
