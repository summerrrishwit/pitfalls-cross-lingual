#!/usr/bin/env python3
"""Materialize and validate the full-pool formal-universe-v2 successor.

This tool creates a new immutable declaration around the exact ordered scope of
the existing formal-universe-v1.  The successor pre-binds the pending v2
historical-exposure authority contract, its challenge, and its untouched
scope-owner attestation template.  It does not complete historical-exposure
review, bind a completed attestation, rebuild post-review data, freeze a split,
load an HF artifact, or run model behavior.

The optional attestation argument to ``validate`` is read-only.  It produces a
SHA-bound candidate for a later immutable successor/addendum; it never edits
this declaration in place and never turns any authorization flag on.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import (  # noqa: E402
    materialize_full_public_benchmark_historical_exposure_contract_v2 as exposure,
)


TOOL_VERSION = "full-public-benchmark-formal-universe-v2-successor-v1"
ARTIFACT_VERSION = "full-8969-formal-universe-v2"
MANIFEST_SCHEMA_VERSION = "public-benchmark-formal-cohort-universe-successor-v1"
ITEM_SCHEMA_VERSION = "public-benchmark-formal-cohort-successor-item-v1"
EXPECTED_RECORD_COUNT = 8969
EXPECTED_SOURCE_TOOL_VERSION = "public-benchmark-finalizer-preflight-v2"

MANIFEST_NAME = "formal_cohort_universe_manifest.json"
ITEMS_NAME = "formal_cohort_universe_items.jsonl"
UNIVERSE_STATUS = (
    "declared_immutable_not_reviewed_pending_scope_owner_attestation"
)
CONTRACT_STATUS = "predeclared_pending_scope_owner_attestation"
FUTURE_BINDING_POLICY_VERSION = (
    "new-immutable-successor-or-sha-bound-addendum-no-in-place-edit-v1"
)


class UniverseValidationError(ValueError):
    """The successor declaration fails closed."""


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


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UniverseValidationError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise UniverseValidationError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            rows.append(value)
    if not rows:
        raise UniverseValidationError(f"JSONL file is empty: {path}")
    return rows


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UniverseValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _resolve_binding_path(binding: Mapping[str, Any], owner_path: Path) -> Path:
    value = Path(_required_string(binding.get("path"), "binding.path"))
    if value.is_absolute():
        return value.resolve()
    return (Path(owner_path).resolve().parent / value).resolve()


def _binding(
    path: Path,
    *,
    schema_version: str,
    record_count: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(Path(output_path).resolve() if output_path else path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _validate_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    schema_version: str,
    jsonl: bool = False,
) -> Tuple[Path, Any]:
    if not isinstance(binding, dict):
        raise UniverseValidationError(f"{label} binding must be an object")
    expected_fields = {"path", "sha256", "byte_count", "schema_version"}
    if jsonl:
        expected_fields.add("record_count")
    if set(binding) != expected_fields:
        raise UniverseValidationError(f"{label} binding fields are invalid")
    if binding.get("schema_version") != schema_version:
        raise UniverseValidationError(f"{label} schema binding is invalid")
    path = _resolve_binding_path(binding, owner_path)
    if not path.is_file():
        raise UniverseValidationError(f"{label} is missing: {path}")
    if path.stat().st_size != binding.get("byte_count"):
        raise UniverseValidationError(f"{label} byte_count binding is stale")
    if sha256_file(path) != binding.get("sha256"):
        raise UniverseValidationError(f"{label} SHA binding is stale")
    value: Any = read_jsonl(path) if jsonl else read_json(path)
    if jsonl and binding.get("record_count") != len(value):
        raise UniverseValidationError(f"{label} record_count binding is stale")
    return path, value


def _load_source_universe(
    manifest_path: Path, *, expected_record_count: int
) -> Dict[str, Any]:
    try:
        result = exposure._load_source_universe(  # pylint: disable=protected-access
            manifest_path, expected_record_count=expected_record_count
        )
    except exposure.ContractValidationError as exc:
        raise UniverseValidationError(str(exc)) from exc
    if result["manifest"].get("tool_version") != EXPECTED_SOURCE_TOOL_VERSION:
        raise UniverseValidationError("predecessor universe tool_version is unsupported")
    selection_policy = result["manifest"].get("selection_policy")
    if not isinstance(selection_policy, dict) or selection_policy.get("mode") != "full-pool":
        raise UniverseValidationError("predecessor universe is not a full-pool declaration")
    return result


def _load_pending_contract(
    contract_path: Path, source: Mapping[str, Any]
) -> Dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    try:
        validation = exposure.validate_contract(contract_path=contract_path)
    except exposure.ContractValidationError as exc:
        raise UniverseValidationError(str(exc)) from exc
    if validation.get("status") != (
        "valid_pending_contract_waiting_for_scope_owner"
    ):
        raise UniverseValidationError("historical-exposure contract is not pending")
    if validation.get("ready_for_successor_universe_binding") is not False:
        raise UniverseValidationError(
            "pending historical-exposure contract cannot be attestation-complete"
        )
    contract = read_json(contract_path)
    source_binding = contract.get("source_universe_v1")
    if not isinstance(source_binding, dict):
        raise UniverseValidationError("contract source_universe_v1 binding is missing")
    if source_binding.get("universe_id") != source["universe_id"]:
        raise UniverseValidationError("contract targets a different predecessor universe")
    if source_binding.get("sha256") != source["manifest_binding"]["sha256"]:
        raise UniverseValidationError(
            "contract predecessor manifest SHA does not match the requested universe"
        )
    if contract.get("source_universe_items") != source["items_binding"]:
        raise UniverseValidationError(
            "contract predecessor items binding does not match the requested universe"
        )
    return {
        "value": contract,
        "path": contract_path,
        "binding": _binding(
            contract_path, schema_version=exposure.CONTRACT_SCHEMA_VERSION
        ),
    }


def _future_attestation_policy(contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "policy_version": FUTURE_BINDING_POLICY_VERSION,
        "status": "not_bound_pending_scope_owner_attestation",
        "completed_scope_owner_attestation": None,
        "current_manifest_must_remain_immutable": True,
        "in_place_manifest_edit_allowed": False,
        "required_attestation_schema_version": exposure.ATTESTATION_SCHEMA_VERSION,
        "required_contract_id": contract["contract_id"],
        "required_attestation_challenge_sha256": contract[
            "attestation_challenge_sha256"
        ],
        "required_binding_fields": [
            "path",
            "sha256",
            "byte_count",
            "schema_version",
            "contract_id",
            "attestation_challenge_sha256",
            "known_historical_exposure_count",
            "ordered_known_historical_exposure_source_base_fact_ids_sha256",
        ],
        "binding_target": (
            "new_immutable_successor_manifest_or_sha_bound_binding_addendum"
        ),
    }


def _declaration_payload(
    *,
    universe_label: str,
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    source_manifest = source["manifest"]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "universe_label": universe_label,
        "predecessor_universe_id": source["universe_id"],
        "predecessor_manifest_sha256": source["manifest_binding"]["sha256"],
        "predecessor_items_sha256": source["items_binding"]["sha256"],
        "predecessor_ordered_item_row_hashes_sha256": source_manifest[
            "ordered_item_row_hashes_sha256"
        ],
        "record_count": len(source["items"]),
        "ordered_cohort_item_ids_sha256": source_manifest[
            "ordered_cohort_item_ids_sha256"
        ],
        "ordered_source_base_fact_ids_sha256": source_manifest[
            "ordered_source_base_fact_ids_sha256"
        ],
        "historical_exposure_contract_status": CONTRACT_STATUS,
        "contract_id": contract["contract_id"],
        "attestation_challenge_sha256": contract[
            "attestation_challenge_sha256"
        ],
        "contract_sha256": contract_binding["sha256"],
        "scope_owner_attestation_template_sha256": contract[
            "scope_owner_attestation_template"
        ]["sha256"],
        "future_attestation_binding_policy_version": (
            FUTURE_BINDING_POLICY_VERSION
        ),
    }


def _successor_item(
    source_item: Mapping[str, Any], *, successor_universe_id: str
) -> Dict[str, Any]:
    return {
        "schema_version": ITEM_SCHEMA_VERSION,
        "universe_id": successor_universe_id,
        "selection_index": source_item["selection_index"],
        "cohort_item_id": source_item["cohort_item_id"],
        "source_base_fact_id": source_item["source_base_fact_id"],
        "source_id": source_item.get("source_id"),
        "source_row_bindings": copy.deepcopy(source_item.get("source_row_bindings")),
        "member_count": source_item.get("member_count"),
        "member_bindings": copy.deepcopy(source_item.get("member_bindings")),
        "predecessor_item_binding": {
            "schema_version": exposure.SOURCE_ITEM_SCHEMA_VERSION,
            "universe_id": source_item["universe_id"],
            "cohort_item_id": source_item["cohort_item_id"],
            "row_sha256": sha256_value(source_item),
        },
    }


def _lineage(source: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = source["manifest"]
    return {
        "relation": "immutable_exact_ordered_scope_successor",
        "predecessor_artifact_version": "full-8969-formal-universe-v1",
        "predecessor_universe_id": source["universe_id"],
        "predecessor_manifest": source["manifest_binding"],
        "predecessor_items": source["items_binding"],
        "predecessor_historical_exposure_contract_status": manifest[
            "historical_exposure_contract_status"
        ],
        "source_scope_copied_without_addition_removal_or_reordering": True,
    }


def _scope_copy(source: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = source["manifest"]
    return {
        "policy": "exact-ordered-source-base-fact-and-item-payload-copy-v1",
        "record_count": len(source["items"]),
        "ordered_cohort_item_ids_sha256": manifest[
            "ordered_cohort_item_ids_sha256"
        ],
        "ordered_source_base_fact_ids_sha256": manifest[
            "ordered_source_base_fact_ids_sha256"
        ],
        "ordered_predecessor_item_row_hashes_sha256": manifest[
            "ordered_item_row_hashes_sha256"
        ],
        "exact_ordered_scope_equal_to_predecessor": True,
    }


def _historical_authority(
    *, contract: Mapping[str, Any], contract_binding: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "status": CONTRACT_STATUS,
        "contract_id": contract["contract_id"],
        "attestation_challenge_sha256": contract[
            "attestation_challenge_sha256"
        ],
        "source_universe_manifest_sha256": contract["source_universe_v1"][
            "sha256"
        ],
        "source_universe_items_sha256": contract["source_universe_items"][
            "sha256"
        ],
        "contract": dict(contract_binding),
        "scope_owner_attestation_template": copy.deepcopy(
            contract["scope_owner_attestation_template"]
        ),
        "future_attestation_binding": _future_attestation_policy(contract),
    }


def _inherited_fields(source_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "selection_policy",
        "selection_source",
        "source_artifacts",
        "comparison_canonical",
        "near_duplicate_audit_contract",
    )
    missing = [field for field in required if field not in source_manifest]
    if missing:
        raise UniverseValidationError(
            "predecessor universe is missing inherited fields: " + ", ".join(missing)
        )
    return {field: copy.deepcopy(source_manifest[field]) for field in required}


def materialize_successor_universe(
    *,
    source_universe_manifest_path: Path,
    historical_exposure_contract_path: Path,
    output_dir: Path,
    universe_label: str,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    universe_label = _required_string(universe_label, "universe_label")
    source = _load_source_universe(
        source_universe_manifest_path,
        expected_record_count=expected_record_count,
    )
    pending = _load_pending_contract(historical_exposure_contract_path, source)
    contract = pending["value"]
    declaration = _declaration_payload(
        universe_label=universe_label,
        source=source,
        contract=contract,
        contract_binding=pending["binding"],
    )
    universe_id = "formal_cohort_" + sha256_value(declaration)[:24]
    items = [
        _successor_item(row, successor_universe_id=universe_id)
        for row in source["items"]
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir: Optional[Path] = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent)
        )
    )
    try:
        staged_items = staged_dir / ITEMS_NAME
        staged_manifest = staged_dir / MANIFEST_NAME
        staged_items.write_bytes(_jsonl_payload(items))
        item_binding = _binding(
            staged_items,
            schema_version=ITEM_SCHEMA_VERSION,
            record_count=len(items),
            output_path=output_dir / ITEMS_NAME,
        )
        source_manifest = source["manifest"]
        manifest: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "universe_id": universe_id,
            "universe_label": universe_label,
            "universe_status": UNIVERSE_STATUS,
            "lineage": _lineage(source),
            "scope_copy": _scope_copy(source),
            **_inherited_fields(source_manifest),
            "historical_exposure_contract_status": CONTRACT_STATUS,
            "historical_exposure_authority": _historical_authority(
                contract=contract, contract_binding=pending["binding"]
            ),
            "record_count": len(items),
            "ordered_cohort_item_ids_sha256": sha256_value(
                [row["cohort_item_id"] for row in items]
            ),
            "ordered_source_base_fact_ids_sha256": sha256_value(
                [row["source_base_fact_id"] for row in items]
            ),
            "ordered_item_row_hashes_sha256": sha256_value(
                [sha256_value(row) for row in items]
            ),
            "items": item_binding,
            "historical_exposure_complete": False,
            "review_complete": False,
            "review_freeze_authorized": False,
            "canonical_freeze_authorized": False,
            "split_freeze_authorized": False,
            "exact_hf_authorized": False,
            "behavior_authorized": False,
        }
        staged_manifest.write_bytes(_json_payload(manifest))
        if os.path.lexists(output_dir):
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        os.rename(staged_dir, output_dir)
        staged_dir = None
    finally:
        if staged_dir is not None and staged_dir.exists():
            shutil.rmtree(staged_dir)

    manifest_path = output_dir / MANIFEST_NAME
    return {
        "status": "materialized_pending_formal_universe_v2_successor",
        "artifact_version": ARTIFACT_VERSION,
        "universe_id": universe_id,
        "record_count": len(items),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "items_path": str(output_dir / ITEMS_NAME),
        "items_sha256": item_binding["sha256"],
        "contract_id": contract["contract_id"],
        "attestation_challenge_sha256": contract[
            "attestation_challenge_sha256"
        ],
        "historical_exposure_complete": False,
        "review_freeze_authorized": False,
        "canonical_freeze_authorized": False,
        "split_freeze_authorized": False,
        "exact_hf_authorized": False,
        "behavior_authorized": False,
    }


def _expected_top_fields() -> set[str]:
    return {
        "schema_version",
        "tool_version",
        "artifact_version",
        "universe_id",
        "universe_label",
        "universe_status",
        "lineage",
        "scope_copy",
        "selection_policy",
        "selection_source",
        "source_artifacts",
        "comparison_canonical",
        "near_duplicate_audit_contract",
        "historical_exposure_contract_status",
        "historical_exposure_authority",
        "record_count",
        "ordered_cohort_item_ids_sha256",
        "ordered_source_base_fact_ids_sha256",
        "ordered_item_row_hashes_sha256",
        "items",
        "historical_exposure_complete",
        "review_complete",
        "review_freeze_authorized",
        "canonical_freeze_authorized",
        "split_freeze_authorized",
        "exact_hf_authorized",
        "behavior_authorized",
    }


def _future_attestation_binding_candidate(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    attestation_path: Path,
) -> Dict[str, Any]:
    authority = manifest["historical_exposure_authority"]
    contract_path = _resolve_binding_path(authority["contract"], manifest_path)
    try:
        result = exposure.validate_contract(
            contract_path=contract_path,
            scope_owner_attestation_path=attestation_path,
        )
    except exposure.ContractValidationError as exc:
        raise UniverseValidationError(str(exc)) from exc
    if result.get("ready_for_successor_universe_binding") is not True:
        raise UniverseValidationError(
            "scope-owner attestation is not ready for future successor binding"
        )
    if result.get("contract_id") != authority["contract_id"]:
        raise UniverseValidationError("attestation contract_id does not match universe")
    ids = result["known_historical_exposure_source_base_fact_ids"]
    return {
        "path": str(Path(attestation_path).resolve()),
        "sha256": sha256_file(attestation_path),
        "byte_count": Path(attestation_path).resolve().stat().st_size,
        "schema_version": exposure.ATTESTATION_SCHEMA_VERSION,
        "contract_id": result["contract_id"],
        "attestation_challenge_sha256": authority[
            "attestation_challenge_sha256"
        ],
        "known_historical_exposure_count": len(ids),
        "ordered_known_historical_exposure_source_base_fact_ids_sha256": (
            sha256_value(ids)
        ),
    }


def validate_successor_universe(
    *,
    universe_manifest_path: Path,
    scope_owner_attestation_path: Optional[Path] = None,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    manifest_path = Path(universe_manifest_path).resolve()
    manifest = read_json(manifest_path)
    if set(manifest) != _expected_top_fields():
        raise UniverseValidationError("successor manifest fields are invalid")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise UniverseValidationError("successor manifest schema_version is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION:
        raise UniverseValidationError("successor manifest tool_version is unsupported")
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise UniverseValidationError("successor artifact_version is unsupported")
    if manifest.get("universe_status") != UNIVERSE_STATUS:
        raise UniverseValidationError("successor universe_status is invalid")
    if manifest.get("historical_exposure_contract_status") != CONTRACT_STATUS:
        raise UniverseValidationError(
            "successor historical-exposure contract status is invalid"
        )
    for field in (
        "historical_exposure_complete",
        "review_complete",
        "review_freeze_authorized",
        "canonical_freeze_authorized",
        "split_freeze_authorized",
        "exact_hf_authorized",
        "behavior_authorized",
    ):
        if manifest.get(field) is not False:
            raise UniverseValidationError(f"successor manifest must keep {field}=false")
    if manifest.get("record_count") != expected_record_count:
        raise UniverseValidationError(
            f"successor has {manifest.get('record_count')} records, expected {expected_record_count}"
        )

    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict):
        raise UniverseValidationError("successor lineage is missing")
    predecessor_path, _ = _validate_binding(
        lineage.get("predecessor_manifest"),
        owner_path=manifest_path,
        label="predecessor universe manifest",
        schema_version=exposure.SOURCE_UNIVERSE_SCHEMA_VERSION,
    )
    source = _load_source_universe(
        predecessor_path, expected_record_count=expected_record_count
    )
    if lineage != _lineage(source):
        raise UniverseValidationError("successor predecessor lineage is stale")

    authority = manifest.get("historical_exposure_authority")
    if not isinstance(authority, dict):
        raise UniverseValidationError("historical-exposure authority binding is missing")
    contract_path, _ = _validate_binding(
        authority.get("contract"),
        owner_path=manifest_path,
        label="historical-exposure contract",
        schema_version=exposure.CONTRACT_SCHEMA_VERSION,
    )
    pending = _load_pending_contract(contract_path, source)
    contract = pending["value"]
    if authority != _historical_authority(
        contract=contract, contract_binding=pending["binding"]
    ):
        raise UniverseValidationError(
            "historical-exposure authority binding is stale or completed in place"
        )

    label = _required_string(manifest.get("universe_label"), "universe_label")
    declaration = _declaration_payload(
        universe_label=label,
        source=source,
        contract=contract,
        contract_binding=pending["binding"],
    )
    expected_universe_id = "formal_cohort_" + sha256_value(declaration)[:24]
    if manifest.get("universe_id") != expected_universe_id:
        raise UniverseValidationError("successor universe_id is stale")

    if manifest.get("scope_copy") != _scope_copy(source):
        raise UniverseValidationError("successor scope-copy declaration is stale")
    inherited = _inherited_fields(source["manifest"])
    for field, expected in inherited.items():
        if manifest.get(field) != expected:
            raise UniverseValidationError(f"successor inherited {field} is stale")

    _, items = _validate_binding(
        manifest.get("items"),
        owner_path=manifest_path,
        label="successor universe items",
        schema_version=ITEM_SCHEMA_VERSION,
        jsonl=True,
    )
    expected_items = [
        _successor_item(row, successor_universe_id=expected_universe_id)
        for row in source["items"]
    ]
    if items != expected_items:
        raise UniverseValidationError(
            "successor items do not exactly copy the predecessor ordered scope"
        )
    digest_checks = {
        "ordered_cohort_item_ids_sha256": sha256_value(
            [row["cohort_item_id"] for row in items]
        ),
        "ordered_source_base_fact_ids_sha256": sha256_value(
            [row["source_base_fact_id"] for row in items]
        ),
        "ordered_item_row_hashes_sha256": sha256_value(
            [sha256_value(row) for row in items]
        ),
    }
    for field, expected in digest_checks.items():
        if manifest.get(field) != expected:
            raise UniverseValidationError(f"successor {field} is stale")

    binding_candidate = None
    if scope_owner_attestation_path is not None:
        binding_candidate = _future_attestation_binding_candidate(
            manifest=manifest,
            manifest_path=manifest_path,
            attestation_path=Path(scope_owner_attestation_path).resolve(),
        )
    return {
        "status": (
            "valid_pending_successor_with_compatible_attestation_candidate"
            if binding_candidate is not None
            else "valid_pending_formal_universe_v2_successor"
        ),
        "artifact_version": ARTIFACT_VERSION,
        "universe_id": expected_universe_id,
        "record_count": len(items),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "contract_id": contract["contract_id"],
        "attestation_challenge_sha256": contract[
            "attestation_challenge_sha256"
        ],
        "scope_owner_attestation_bound": False,
        "future_attestation_binding_candidate": binding_candidate,
        "historical_exposure_complete": False,
        "review_freeze_authorized": False,
        "canonical_freeze_authorized": False,
        "split_freeze_authorized": False,
        "exact_hf_authorized": False,
        "behavior_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser(
        "materialize", help="emit a pending immutable formal-universe-v2 successor"
    )
    materialize.add_argument(
        "--source-universe-manifest", type=Path, required=True
    )
    materialize.add_argument(
        "--historical-exposure-contract", type=Path, required=True
    )
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--universe-label", required=True)
    validate = commands.add_parser(
        "validate", help="validate the successor and optional future attestation"
    )
    validate.add_argument("--universe-manifest", type=Path, required=True)
    validate.add_argument("--scope-owner-attestation", type=Path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "materialize":
            result = materialize_successor_universe(
                source_universe_manifest_path=args.source_universe_manifest,
                historical_exposure_contract_path=args.historical_exposure_contract,
                output_dir=args.output_dir,
                universe_label=args.universe_label,
            )
        else:
            result = validate_successor_universe(
                universe_manifest_path=args.universe_manifest,
                scope_owner_attestation_path=args.scope_owner_attestation,
            )
    except (
        UniverseValidationError,
        exposure.ContractValidationError,
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked_fail_closed",
                    "historical_exposure_complete": False,
                    "split_freeze_authorized": False,
                    "exact_hf_authorized": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
