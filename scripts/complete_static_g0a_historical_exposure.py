#!/usr/bin/env python3
"""Complete historical-exposure evidence from a user scope attestation.

The pending materializer can prove only what its repository inventory scanned.
This tool therefore requires a separately authored, explicit scope-owner JSON
attestation for every remaining scope.  It never infers assent, never edits the
pending packet, and publishes no complete artifacts when any repository hit,
stale binding, missing scope, or disclosed exposure is present.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_static_g0a as g0a  # noqa: E402
from scripts import materialize_static_g0a_historical_exposure_pending as pending  # noqa: E402


TOOL_VERSION = "static-g0a-historical-exposure-attestation-completer-v1"
POLICY_VERSION = "repository-inventory-plus-explicit-scope-owner-attestation-v1"
ATTESTATION_SCHEMA = "static-g0a-historical-exposure-scope-attestation-v1"
COMPLETE_CHECKS_NAME = "historical_exposure_checks.jsonl"
COMPLETE_REGISTRY_NAME = "historical_exposure_registry.jsonl"
COMPLETE_MANIFEST_NAME = "historical_exposure_manifest.json"
EXPECTED_PENDING_STATUS = "pending_external_scope_attestation"
EXPECTED_PROXY_MODEL_IDS = (
    "qwen3.6-27b",
    "bailian/deepseek-v3.2",
    "gemini-3.7-flash",
    "gemma3:12b",
    "llama3.1:8b",
)
REQUIRED_SCOPE_FLAGS = (
    "validation_and_sealed_facts",
    "five_proxy_models",
    "target_models",
    "external_directories",
    "other_scripts",
    "external_platforms",
)
REQUIRED_PENDING_BLOCKERS = frozenset(
    {
        "external_directories_not_attested",
        "external_script_run_history_not_attested",
        "external_platform_run_history_not_attested",
    }
)


class HistoricalExposureBlockedError(ValueError):
    """The evidence cannot safely be completed."""


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(g0a.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], stderr=subprocess.STDOUT
    )


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise HistoricalExposureBlockedError(f"{label} must be a non-empty ISO timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HistoricalExposureBlockedError(f"{label} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoricalExposureBlockedError(f"{label} must include a timezone")
    return parsed


def _require_true(mapping: Mapping[str, Any], field: str, label: str) -> None:
    if mapping.get(field) is not True:
        raise HistoricalExposureBlockedError(f"{label}.{field} must be explicitly true")


def _require_zero_repository_hits(
    *, matched_count: Any, matches: Any, label: str
) -> None:
    if (
        not isinstance(matched_count, int)
        or isinstance(matched_count, bool)
        or matched_count != 0
        or matches != []
    ):
        raise HistoricalExposureBlockedError(f"{label} contains a validation/sealed hit")


def _validate_binding_value(
    bindings: Mapping[str, Any], field: str, expected: Any
) -> None:
    if bindings.get(field) != expected:
        raise HistoricalExposureBlockedError(f"scope attestation binding mismatch: {field}")


def _repository_scan_inventory_sha256(
    pending_manifest: Mapping[str, Any],
) -> str:
    scan = pending_manifest.get("repository_artifact_scan")
    if not isinstance(scan, dict) or not isinstance(scan.get("inventory"), dict):
        raise HistoricalExposureBlockedError(
            "pending repository artifact inventory binding is missing"
        )
    value = scan["inventory"].get("sha256")
    if not isinstance(value, str) or not value:
        raise HistoricalExposureBlockedError(
            "pending repository artifact inventory SHA is missing"
        )
    return value


def _validate_scope_attestation(
    *,
    attestation_path: Path,
    pending_manifest_path: Path,
    pending_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    attestation_path = Path(attestation_path).resolve()
    value = g0a.read_json(attestation_path)
    if value.get("schema_version") != ATTESTATION_SCHEMA:
        raise HistoricalExposureBlockedError("unsupported scope attestation schema_version")
    if value.get("status") != "affirmed":
        raise HistoricalExposureBlockedError(
            "scope attestation status must be explicitly affirmed"
        )
    if value.get("attestation_source") != "user_explicit_confirmation":
        raise HistoricalExposureBlockedError(
            "scope attestation must identify user_explicit_confirmation as its source"
        )
    if value.get("attester_role") != "scope_owner":
        raise HistoricalExposureBlockedError(
            "scope attestation requires attester_role=scope_owner"
        )
    g0a._required_string(value.get("attested_by"), "scope attestation attested_by")
    attested_at = _parse_timestamp(value.get("attested_at"), "scope attestation attested_at")
    pending_created_at = _parse_timestamp(
        pending_manifest.get("created_at"), "pending manifest created_at"
    )
    if attested_at < pending_created_at:
        raise HistoricalExposureBlockedError(
            "scope attestation predates the repository scan packet"
        )

    statement = value.get("statement")
    if not isinstance(statement, dict):
        raise HistoricalExposureBlockedError("scope attestation statement must be an object")
    _require_true(
        statement,
        "no_validation_or_sealed_historical_behavior_exposure",
        "scope attestation statement",
    )
    disclosed = statement.get("known_historical_exposures")
    if disclosed != []:
        raise HistoricalExposureBlockedError(
            "scope attestation must contain an explicit empty known_historical_exposures list"
        )
    scope = statement.get("scope_includes")
    if not isinstance(scope, dict):
        raise HistoricalExposureBlockedError("scope_includes must be an object")
    for field in REQUIRED_SCOPE_FLAGS:
        _require_true(scope, field, "scope attestation statement.scope_includes")
    proxy_model_ids = scope.get("proxy_model_ids")
    if not isinstance(proxy_model_ids, list) or sorted(proxy_model_ids) != sorted(
        EXPECTED_PROXY_MODEL_IDS
    ):
        raise HistoricalExposureBlockedError(
            "scope attestation must list exactly the five frozen proxy model ids"
        )

    bindings = value.get("bindings")
    if not isinstance(bindings, dict):
        raise HistoricalExposureBlockedError("scope attestation bindings must be an object")
    repository_scope = pending_manifest.get("repository_scope")
    scan_summary = pending_manifest.get("simulation_results_scan")
    if not isinstance(repository_scope, dict) or not isinstance(scan_summary, dict):
        raise HistoricalExposureBlockedError("pending repository scan metadata is missing")
    scan_binding = scan_summary.get("scans")
    if not isinstance(scan_binding, dict):
        raise HistoricalExposureBlockedError("pending repository scan result binding is missing")
    repository_artifact_scan = pending_manifest.get("repository_artifact_scan")
    if not isinstance(repository_artifact_scan, dict) or not isinstance(
        repository_artifact_scan.get("results"), dict
    ):
        raise HistoricalExposureBlockedError(
            "pending full-repository scan result binding is missing"
        )
    resolved_manifest_binding = pending_manifest.get("resolved_static_draft_manifest")
    if not isinstance(resolved_manifest_binding, dict):
        raise HistoricalExposureBlockedError(
            "pending resolved_static_draft_manifest binding is missing"
        )
    expected_bindings = {
        "pending_manifest_sha256": g0a.sha256_file(pending_manifest_path),
        "input_bundle_sha256": pending_manifest["input_bundle"].get("sha256"),
        "effective_staging_bundle_sha256": pending_manifest[
            "effective_staging_bundle"
        ].get("sha256"),
        "resolved_static_draft_sha256": pending_manifest[
            "resolved_static_draft"
        ].get("sha256"),
        "resolved_static_draft_manifest_sha256": resolved_manifest_binding.get("sha256"),
        "repository_git_head": repository_scope.get("git_head"),
        "repository_tracked_paths_sha256": repository_scope.get("tracked_paths_sha256"),
        "repository_scan_inventory_sha256": _repository_scan_inventory_sha256(
            pending_manifest
        ),
        "repository_scan_results_sha256": repository_artifact_scan["results"].get(
            "sha256"
        ),
        "simulation_results_scan_sha256": scan_binding.get("sha256"),
    }
    for field, expected in expected_bindings.items():
        if not expected:
            raise HistoricalExposureBlockedError(
                f"pending evidence has no immutable value for attestation binding: {field}"
            )
        _validate_binding_value(bindings, field, expected)
    return value


def _validate_repository_artifact_scan(
    *, manifest: Mapping[str, Any], owner_path: Path
) -> Dict[str, Any]:
    scan = manifest.get("repository_artifact_scan")
    if not isinstance(scan, dict):
        raise HistoricalExposureBlockedError("pending full-repository scan is missing")
    if scan.get("policy") != "exact-holdout-identifiers-plus-behavior-signature-v1":
        raise HistoricalExposureBlockedError("unsupported full-repository scan policy")
    for field in ("current_worktree_complete", "reachable_git_blobs_complete"):
        _require_true(scan, field, "pending repository_artifact_scan")
    _require_zero_repository_hits(
        matched_count=scan.get("behavior_hit_base_fact_count"),
        matches=scan.get("behavior_hit_base_fact_ids"),
        label="full-repository artifact scan",
    )
    inventory_path, inventory = g0a._validate_file_binding(
        scan.get("inventory"),
        owner_path=owner_path,
        label="repository artifact inventory",
        expected_schema=pending.REPOSITORY_INVENTORY_SCHEMA,
        jsonl=True,
    )
    results_path, results = g0a._validate_file_binding(
        scan.get("results"),
        owner_path=owner_path,
        label="repository exposure scan results",
        expected_schema=pending.REPOSITORY_SCAN_RESULT_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    inventory_by_id: Dict[str, Dict[str, Any]] = {}
    source_counts = {"current_worktree_file": 0, "reachable_git_blob": 0}
    matching_inventory_ids = set()
    for index, row in enumerate(inventory, start=1):
        if row.get("schema_version") != pending.REPOSITORY_INVENTORY_SCHEMA:
            raise HistoricalExposureBlockedError(
                f"repository inventory row {index} has unsupported schema"
            )
        inventory_id = g0a._required_string(
            row.get("inventory_id"), f"repository inventory row {index} inventory_id"
        )
        if inventory_id in inventory_by_id:
            raise HistoricalExposureBlockedError(
                f"duplicate repository inventory id: {inventory_id}"
            )
        source_kind = row.get("source_kind")
        if source_kind not in source_counts:
            raise HistoricalExposureBlockedError(
                f"repository inventory has unsupported source_kind: {inventory_id}"
            )
        source_counts[source_kind] += 1
        if row.get("scan_status") != "completed_exact_identifier_and_behavior_signature_scan":
            raise HistoricalExposureBlockedError(
                f"repository inventory scan is incomplete: {inventory_id}"
            )
        relative_paths = row.get("relative_paths")
        if not isinstance(relative_paths, list) or not relative_paths or any(
            not isinstance(path, str) or not path for path in relative_paths
        ):
            raise HistoricalExposureBlockedError(
                f"repository inventory paths are invalid: {inventory_id}"
            )
        for field in (
            "identifier_match_count",
            "matched_holdout_base_fact_count",
            "behavior_hit_base_fact_count",
        ):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise HistoricalExposureBlockedError(
                    f"repository inventory has invalid {field}: {inventory_id}"
                )
        if row["behavior_hit_base_fact_count"] != 0:
            raise HistoricalExposureBlockedError(
                f"repository inventory contains a validation/sealed hit: {inventory_id}"
            )
        if row["identifier_match_count"] > 0:
            matching_inventory_ids.add(inventory_id)
        inventory_by_id[inventory_id] = dict(row)
    inventory_binding = scan["inventory"]
    if inventory_binding.get("inventory_ids_sha256") != g0a.sha256_value(
        sorted(inventory_by_id)
    ):
        raise HistoricalExposureBlockedError(
            "repository artifact inventory id digest is stale"
        )

    result_ids = set()
    for index, row in enumerate(results, start=1):
        if row.get("schema_version") != pending.REPOSITORY_SCAN_RESULT_SCHEMA:
            raise HistoricalExposureBlockedError(
                f"repository scan result {index} has unsupported schema"
            )
        inventory_id = g0a._required_string(
            row.get("inventory_id"), f"repository scan result {index} inventory_id"
        )
        if inventory_id in result_ids or inventory_id not in inventory_by_id:
            raise HistoricalExposureBlockedError(
                f"repository scan result coverage is invalid: {inventory_id}"
            )
        if row.get("source_kind") != inventory_by_id[inventory_id]["source_kind"]:
            raise HistoricalExposureBlockedError(
                f"repository scan result source_kind mismatch: {inventory_id}"
            )
        if row.get("scan_status") != "completed_exact_identifier_and_behavior_signature_scan":
            raise HistoricalExposureBlockedError(
                f"repository scan result is incomplete: {inventory_id}"
            )
        matched_identifiers = row.get("matched_identifiers")
        matched_ids = row.get("matched_holdout_base_fact_ids")
        if (
            not isinstance(matched_identifiers, list)
            or not matched_identifiers
            or not isinstance(matched_ids, list)
            or not matched_ids
        ):
            raise HistoricalExposureBlockedError(
                f"repository scan result has no exact identifier evidence: {inventory_id}"
            )
        _require_zero_repository_hits(
            matched_count=len(row.get("behavior_hit_base_fact_ids", [])),
            matches=row.get("behavior_hit_base_fact_ids"),
            label=f"repository scan result {inventory_id}",
        )
        if row.get("historical_behavior_exposure_detected") is not False:
            raise HistoricalExposureBlockedError(
                f"repository scan result exposes a holdout: {inventory_id}"
            )
        if row.get("behavior_signature_names") != []:
            raise HistoricalExposureBlockedError(
                f"repository scan result has unresolved behavior signatures: {inventory_id}"
            )
        inventory_row = inventory_by_id[inventory_id]
        if inventory_row["identifier_match_count"] != len(matched_identifiers) or inventory_row[
            "matched_holdout_base_fact_count"
        ] != len(matched_ids):
            raise HistoricalExposureBlockedError(
                f"repository scan result counts disagree with inventory: {inventory_id}"
            )
        result_ids.add(inventory_id)
    results_binding = scan["results"]
    if results_binding.get("inventory_ids_sha256") != g0a.sha256_value(
        sorted(result_ids)
    ):
        raise HistoricalExposureBlockedError(
            "repository exposure result id digest is stale"
        )
    if result_ids != matching_inventory_ids:
        raise HistoricalExposureBlockedError(
            "repository scan results do not cover every identifier-matching inventory item"
        )
    if scan.get("worktree_file_count") != source_counts["current_worktree_file"]:
        raise HistoricalExposureBlockedError("repository worktree inventory count is stale")
    if scan.get("reachable_git_blob_count") != source_counts["reachable_git_blob"]:
        raise HistoricalExposureBlockedError("repository Git blob inventory count is stale")
    if scan.get("identifier_matching_artifact_count") != len(results):
        raise HistoricalExposureBlockedError(
            "repository identifier-matching artifact count is stale"
        )
    if len(inventory) != sum(source_counts.values()):
        raise HistoricalExposureBlockedError("repository inventory source counts are stale")
    return {
        "summary": scan,
        "inventory": inventory,
        "inventory_path": inventory_path,
        "results": results,
        "results_path": results_path,
    }


def _validate_pending_packet(
    pending_manifest_path: Path,
) -> Dict[str, Any]:
    pending_manifest_path = Path(pending_manifest_path).resolve()
    manifest = g0a.read_json(pending_manifest_path)
    if manifest.get("schema_version") != g0a.EXPOSURE_MANIFEST_SCHEMA:
        raise HistoricalExposureBlockedError(
            "unsupported pending exposure manifest schema_version"
        )
    if manifest.get("status") != EXPECTED_PENDING_STATUS:
        raise HistoricalExposureBlockedError("historical exposure packet is not pending")
    for field in (
        "historical_exposure_complete",
        "zero_registry_supported_by_per_fact_checks",
        "static_frozen",
        "behavior_authorized",
        "simulation_authorized",
    ):
        if manifest.get(field) is not False:
            raise HistoricalExposureBlockedError(f"pending manifest must keep {field}=false")
    if manifest.get("review_blinded_to_behavior_results") is not True:
        raise HistoricalExposureBlockedError("pending exposure review is not behavior blind")
    if manifest.get("behavior_result_count") != 0:
        raise HistoricalExposureBlockedError("pending packet contains behavior results")
    blockers = manifest.get("known_blockers")
    if (
        not isinstance(blockers, list)
        or any(not isinstance(item, str) for item in blockers)
        or not REQUIRED_PENDING_BLOCKERS.issubset(set(blockers))
    ):
        raise HistoricalExposureBlockedError(
            "pending manifest does not preserve all scope blockers"
        )

    resolved_manifest_path, _ = g0a._validate_file_binding(
        manifest.get("resolved_static_draft_manifest"),
        owner_path=pending_manifest_path,
        label="pending resolved static draft manifest",
        expected_schema=g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
    )
    loaded = pending._load_resolved(resolved_manifest_path)
    g0a._validate_source_binding(
        manifest.get("input_bundle"),
        loaded["staging_manifest"]["source_bundle"],
        "pending historical exposure",
    )
    g0a._validate_binding_identity(
        manifest.get("effective_staging_bundle"),
        loaded["staging_manifest"]["effective_staging_bundle"],
        "pending historical exposure effective_staging_bundle",
    )
    g0a._validate_binding_identity(
        manifest.get("resolved_static_draft"),
        loaded["manifest"]["resolved_rows"],
        "pending historical exposure resolved_static_draft",
    )

    _, checks = g0a._validate_file_binding(
        manifest.get("checks"),
        owner_path=pending_manifest_path,
        label="pending historical exposure checks",
        expected_schema=g0a.EXPOSURE_CHECK_SCHEMA,
        jsonl=True,
    )
    _, registry = g0a._validate_file_binding(
        manifest.get("registry"),
        owner_path=pending_manifest_path,
        label="pending historical exposure registry",
        expected_schema=g0a.EXPOSURE_REGISTRY_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    scans_path, scans = g0a._validate_file_binding(
        manifest.get("simulation_results_scan", {}).get("scans"),
        owner_path=pending_manifest_path,
        label="pending simulation-results scans",
        expected_schema=pending.SIMULATION_SCAN_SCHEMA,
        jsonl=True,
    )
    repository_artifact_scan = _validate_repository_artifact_scan(
        manifest=manifest,
        owner_path=pending_manifest_path,
    )
    if registry:
        raise HistoricalExposureBlockedError("pending historical exposure registry is not empty")

    holdout_ids = set(loaded["holdout_ids"])
    checked_sources: Optional[List[str]] = None
    check_by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(checks, start=1):
        if row.get("schema_version") != g0a.EXPOSURE_CHECK_SCHEMA:
            raise HistoricalExposureBlockedError(
                f"pending historical exposure check {index} has unsupported schema"
            )
        base_fact_id = row.get("base_fact_id")
        if base_fact_id not in holdout_ids or base_fact_id in check_by_id:
            raise HistoricalExposureBlockedError(
                f"pending historical exposure check coverage is invalid: {base_fact_id}"
            )
        staging = loaded["staging_by_id"][base_fact_id]
        resolved = loaded["resolved_by_id"][base_fact_id]
        expected_fields = {
            "source_record_sha256": staging["source_record_sha256"],
            "effective_source_record_sha256": staging[
                "effective_source_record_sha256"
            ],
            "input_record_sha256": g0a.sha256_value(resolved),
            "split_assignment": resolved["split_assignment"],
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise HistoricalExposureBlockedError(
                    f"pending exposure check has stale {field}: {base_fact_id}"
                )
        if row.get("terminal_status") != EXPECTED_PENDING_STATUS:
            raise HistoricalExposureBlockedError(
                f"pending exposure check has unexpected status: {base_fact_id}"
            )
        if row.get("historically_exposed") is not None:
            raise HistoricalExposureBlockedError(
                f"pending exposure check contains a hit or non-pending result: {base_fact_id}"
            )
        local_hits = row.get("local_simulation_identifier_hits")
        _require_zero_repository_hits(
            matched_count=len(local_hits) if isinstance(local_hits, list) else None,
            matches=local_hits,
            label=f"repository identifier scan for {base_fact_id}",
        )
        repository_behavior_hits = row.get("repository_behavior_hit_inventory_ids")
        _require_zero_repository_hits(
            matched_count=(
                len(repository_behavior_hits)
                if isinstance(repository_behavior_hits, list)
                else None
            ),
            matches=repository_behavior_hits,
            label=f"full-repository behavior scan for {base_fact_id}",
        )
        sources = row.get("checked_sources")
        if not isinstance(sources, list) or not sources or any(
            not isinstance(item, str) or not item.strip() for item in sources
        ):
            raise HistoricalExposureBlockedError(
                f"pending exposure check has no concrete repository sources: {base_fact_id}"
            )
        if checked_sources is None:
            checked_sources = sources
        elif sources != checked_sources:
            raise HistoricalExposureBlockedError("pending checks disagree on repository sources")
        check_by_id[base_fact_id] = dict(row)
    if set(check_by_id) != holdout_ids or len(check_by_id) != pending.EXPECTED_HOLDOUT_COUNT:
        raise HistoricalExposureBlockedError(
            "pending exposure checks do not exactly cover 64 validation/sealed facts"
        )

    scan_paths: List[str] = []
    scan_record_count = 0
    for index, scan in enumerate(scans, start=1):
        if scan.get("schema_version") != pending.SIMULATION_SCAN_SCHEMA:
            raise HistoricalExposureBlockedError(f"repository scan {index} has unsupported schema")
        _require_zero_repository_hits(
            matched_count=scan.get("matched_holdout_base_fact_count"),
            matches=scan.get("matches"),
            label=f"repository scan {scan.get('relative_path')}",
        )
        if scan.get("behavior_outcome_fields_inspected") is not False:
            raise HistoricalExposureBlockedError("repository scan behavior-blind contract changed")
        if scan.get("local_file_scan_status") != "completed_identifier_metadata_only":
            raise HistoricalExposureBlockedError("repository identifier scan is incomplete")
        relative_path = scan.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise HistoricalExposureBlockedError("repository scan has no relative_path")
        record_count = scan.get("record_count")
        if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 0:
            raise HistoricalExposureBlockedError("repository scan has invalid record_count")
        scan_paths.append(relative_path)
        scan_record_count += record_count

    scan_summary = manifest.get("simulation_results_scan")
    if not isinstance(scan_summary, dict):
        raise HistoricalExposureBlockedError("pending simulation_results_scan is missing")
    _require_zero_repository_hits(
        matched_count=scan_summary.get("matched_holdout_base_fact_count"),
        matches=[],
        label="repository scan summary",
    )
    if scan_summary.get("file_count") != len(scans):
        raise HistoricalExposureBlockedError("repository scan file_count is stale")
    if scan_summary.get("record_count") != scan_record_count:
        raise HistoricalExposureBlockedError("repository scan record_count is stale")
    if checked_sources != scan_paths:
        raise HistoricalExposureBlockedError(
            "per-fact checked_sources do not match repository scan inventory"
        )
    if manifest.get("holdout_check_count") != len(checks) or manifest.get(
        "unresolved_check_count"
    ) != len(checks):
        raise HistoricalExposureBlockedError("pending holdout counts are stale")

    repository_scope = manifest.get("repository_scope")
    if not isinstance(repository_scope, dict):
        raise HistoricalExposureBlockedError("pending repository_scope is missing")
    if repository_scope.get("current_worktree_simulation_identifier_scan_complete") is not True:
        raise HistoricalExposureBlockedError("current worktree identifier scan is incomplete")
    if repository_scope.get("current_worktree_all_artifact_content_scan_complete") is not True:
        raise HistoricalExposureBlockedError("current worktree artifact scan is incomplete")
    if repository_scope.get("reachable_git_history_content_scan_complete") is not True:
        raise HistoricalExposureBlockedError("reachable Git blob scan is incomplete")
    repo_root = Path(
        g0a._required_string(repository_scope.get("repo_root"), "repo_root")
    ).resolve()
    git_toplevel = Path(
        _git(repo_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if git_toplevel != repo_root:
        raise HistoricalExposureBlockedError("pending repo_root is not the Git toplevel")
    if _git(repo_root, "rev-parse", "HEAD").decode().strip() != repository_scope.get(
        "git_head"
    ):
        raise HistoricalExposureBlockedError("repository HEAD changed after pending scan")
    reachable_commit_count = int(
        _git(repo_root, "rev-list", "--all", "--count").decode().strip()
    )
    if reachable_commit_count != repository_scope.get("reachable_commit_count"):
        raise HistoricalExposureBlockedError(
            "repository reachable commit inventory changed after pending scan"
        )
    tracked_paths = sorted(
        item.decode("utf-8")
        for item in _git(repo_root, "ls-files", "-z").split(b"\0")
        if item
    )
    if len(tracked_paths) != repository_scope.get("tracked_file_count") or g0a.sha256_value(
        tracked_paths
    ) != repository_scope.get("tracked_paths_sha256"):
        raise HistoricalExposureBlockedError("repository tracked-path inventory changed")
    for scan in scans:
        relative_path = scan["relative_path"]
        g0a._validate_file_binding(
            scan.get("file"),
            owner_path=scans_path,
            label=f"repository scan source {relative_path}",
            expected_path=repo_root / relative_path,
            jsonl=True,
            allow_empty=True,
        )

    return {
        "manifest": manifest,
        "manifest_path": pending_manifest_path,
        "loaded": loaded,
        "checks": [check_by_id[base_fact_id] for base_fact_id in sorted(holdout_ids)],
        "registry": registry,
        "scans": scans,
        "scans_path": scans_path,
        "repository_artifact_scan": repository_artifact_scan,
        "repo_root": repo_root,
    }


def complete_historical_exposure(
    *,
    pending_manifest_path: Path,
    scope_attestation_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    packet = _validate_pending_packet(pending_manifest_path)
    attestation = _validate_scope_attestation(
        attestation_path=scope_attestation_path,
        pending_manifest_path=packet["manifest_path"],
        pending_manifest=packet["manifest"],
    )
    attestation_path = Path(scope_attestation_path).resolve()
    attestation_binding = g0a._binding(
        attestation_path, schema_version=ATTESTATION_SCHEMA
    )
    pending_binding = g0a._binding(
        packet["manifest_path"], schema_version=g0a.EXPOSURE_MANIFEST_SCHEMA
    )
    simulation_scan_binding = copy.deepcopy(
        packet["manifest"]["simulation_results_scan"]["scans"]
    )
    repository_inventory_binding = copy.deepcopy(
        packet["manifest"]["repository_artifact_scan"]["inventory"]
    )
    repository_results_binding = copy.deepcopy(
        packet["manifest"]["repository_artifact_scan"]["results"]
    )

    completed_checks: List[Dict[str, Any]] = []
    for row in packet["checks"]:
        completed = copy.deepcopy(row)
        completed.update(
            {
                "terminal_status": "completed",
                "historically_exposed": False,
                "checked_at": attestation["attested_at"],
                "reviewer_type": "user_scope_owner_attestation",
                "human_gold": False,
                "pending_check_record_sha256": g0a.sha256_value(row),
                "scope_attestation_sha256": attestation_binding["sha256"],
                "repository_scan_inventory_sha256": repository_inventory_binding[
                    "sha256"
                ],
                "repository_scan_results_sha256": repository_results_binding["sha256"],
            }
        )
        completed_checks.append(completed)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent))
    )
    try:
        checks_path = temporary_dir / COMPLETE_CHECKS_NAME
        registry_path = temporary_dir / COMPLETE_REGISTRY_NAME
        manifest_path = temporary_dir / COMPLETE_MANIFEST_NAME
        checks_path.write_bytes(_jsonl_payload(completed_checks))
        registry_path.write_bytes(_jsonl_payload(packet["registry"]))
        manifest = {
            "schema_version": g0a.EXPOSURE_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "created_at": g0a.utc_now(),
            "status": "complete",
            "policy_version": POLICY_VERSION,
            "input_bundle": copy.deepcopy(packet["manifest"]["input_bundle"]),
            "effective_staging_bundle": copy.deepcopy(
                packet["manifest"]["effective_staging_bundle"]
            ),
            "resolved_static_draft": copy.deepcopy(
                packet["manifest"]["resolved_static_draft"]
            ),
            "resolved_static_draft_manifest": copy.deepcopy(
                packet["manifest"]["resolved_static_draft_manifest"]
            ),
            "pending_exposure_packet": pending_binding,
            "scope_attestation": attestation_binding,
            "repository_scan_evidence": {
                "repo_root": str(packet["repo_root"]),
                "git_head": packet["manifest"]["repository_scope"]["git_head"],
                "tracked_paths_sha256": packet["manifest"]["repository_scope"][
                    "tracked_paths_sha256"
                ],
                "policy": packet["manifest"]["repository_artifact_scan"]["policy"],
                "inventory": repository_inventory_binding,
                "results": repository_results_binding,
                "simulation_results_scans": simulation_scan_binding,
                "matched_holdout_base_fact_count": 0,
            },
            "checks": g0a._binding(
                checks_path,
                schema_version=g0a.EXPOSURE_CHECK_SCHEMA,
                record_count=len(completed_checks),
                id_values=[row["base_fact_id"] for row in completed_checks],
                id_digest_field="base_fact_ids_sha256",
            ),
            "registry": g0a._binding(
                registry_path,
                schema_version=g0a.EXPOSURE_REGISTRY_SCHEMA,
                record_count=len(packet["registry"]),
            ),
            "holdout_check_count": len(completed_checks),
            "unresolved_check_count": 0,
            "review_blinded_to_behavior_results": True,
            "behavior_result_count": 0,
            "zero_registry_supported_by_per_fact_checks": True,
            "historical_exposure_complete": True,
            "known_blockers": [],
            "static_frozen": False,
            "behavior_authorized": False,
            "simulation_authorized": False,
        }
        for key, filename in (
            ("checks", COMPLETE_CHECKS_NAME),
            ("registry", COMPLETE_REGISTRY_NAME),
        ):
            manifest[key]["path"] = str(output_dir / filename)
        manifest_path.write_bytes(_json_payload(manifest))
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        os.rename(temporary_dir, output_dir)
        temporary_dir = None
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return {
        "status": "complete",
        "historical_exposure_manifest": str(output_dir / COMPLETE_MANIFEST_NAME),
        "holdout_check_count": len(completed_checks),
        "historical_exposure_complete": True,
        "behavior_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pending-manifest", type=Path, required=True)
    result.add_argument("--scope-attestation", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = complete_historical_exposure(
            pending_manifest_path=args.pending_manifest,
            scope_attestation_path=args.scope_attestation,
            output_dir=args.output_dir,
        )
    except (HistoricalExposureBlockedError, FileNotFoundError, subprocess.SubprocessError) as exc:
        result = {
            "status": "blocked_fail_closed",
            "historical_exposure_complete": False,
            "behavior_authorized": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
