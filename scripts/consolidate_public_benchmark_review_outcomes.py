#!/usr/bin/env python3
"""Consolidate disjoint review-apply chains into one audited review chain.

The input review manifests are replayed before any output is written.  Existing
accept/reject decisions are preserved.  Every revise/defer row must have one
explicit resolution-plan entry: a revise may remain ``retain_revise`` for a
later provenance-preserving edit, while ``cohort_exclude`` becomes a reject
whose note states that exclusion is not a factual-falsity judgment.

This tool only rebuilds review scope/export/decision/apply artifacts.  It does
not edit facts, run models, translate, perturb prompts, or freeze a cohort.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_VERSION = "public-benchmark-review-outcome-consolidator-v1"
PLAN_SCHEMA_VERSION = "public-benchmark-review-outcome-resolution-plan-v1"
MANIFEST_SCHEMA_VERSION = "public-benchmark-review-outcome-consolidation-v1"
SUMMARY_SCHEMA_VERSION = "public-benchmark-review-outcome-consolidation-summary-v1"
COHORT_RESOLUTION_SCHEMA_VERSION = (
    "public-benchmark-cohort-eligibility-resolution-v1"
)

PLAN_FIELDS = frozenset(
    {"schema_version", "review_apply_manifests", "resolutions"}
)
PLAN_MANIFEST_BINDING_FIELDS = frozenset({"path", "sha256"})
RESOLUTION_FIELDS = frozenset(
    {
        "base_fact_id",
        "source_review_outcome",
        "source_review_staging_row_sha256",
        "disposition",
        "reason",
        "reviewer_provenance",
    }
)
REVIEWER_PROVENANCE_FIELDS = frozenset(
    {"reviewer_type", "reviewer_id", "review_method", "reviewed_at"}
)
UNRESOLVED_OUTCOMES = frozenset({"revise", "defer"})
TERMINAL_INPUT_OUTCOMES = frozenset({"accept", "reject"})
DISPOSITIONS = frozenset({"retain_revise", "cohort_exclude"})
SHA256_RE = re.compile(r"[0-9a-f]{64}")

COHORT_EXCLUSION_NOTE = (
    "Excluded from this pre-perturbation cohort only; this disposition does "
    "not assert that the underlying factual claim is false."
)
RETAIN_REVISE_NOTE = (
    "Retained as revise for a later provenance-preserving edit and independent "
    "rereview; this consolidation did not apply a factual revision."
)

SAFETY_CONTRACT = {
    "source_artifacts_modified": False,
    "model_or_api_used": False,
    "fact_revision_applied": False,
    "translation_performed": False,
    "perturbation_performed": False,
    "canonical_freeze_performed": False,
    "split_freeze_performed": False,
    "factual_rejections_created_from_cohort_exclusions": False,
    "output_is_review_staging_only": True,
}

COHORT_ELIGIBILITY_CONTRACT = {
    "authoritative_semantics": "cohort_eligibility_disposition",
    "cohort_exclude_applies_to_current_preperturbation_cohort_only": True,
    "cohort_exclude_asserts_factual_falsehood": False,
    "review_tool_v2_reject_is_adapter_encoding_for_cohort_exclude": True,
    "selector_must_consult_cohort_eligibility_resolutions": True,
    "retain_revise_requires_later_revision_and_independent_rereview": True,
}


def _load_sibling_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_timestamp(value: Any, label: str) -> None:
    text = _required_string(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")


def read_json(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: List[Dict[str, Any]] = []
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL: {resolved}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL at {resolved}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {resolved}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"JSONL artifact is empty: {resolved}")
    return rows


def _unique_index(
    rows: Iterable[Mapping[str, Any]], key: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        value = _required_string(row.get(key), f"{label} row {position}.{key}")
        if value in result:
            raise ValueError(f"Duplicate {key} in {label}: {value}")
        result[value] = dict(row)
    return result


def _resolved_path(value: Any, owner_path: Path, label: str) -> Path:
    text = _required_string(value, label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = owner_path.resolve().parent / path
    return path.resolve()


def _bound_json(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
) -> Tuple[Dict[str, Any], Path]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} binding fields are invalid")
    path = _resolved_path(binding.get("path"), owner_path, f"{label}.path")
    expected_sha = _require_sha256(binding.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label}.sha256 is stale")
    return read_json(path), path


def _bound_jsonl(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
) -> Tuple[List[Dict[str, Any]], Path]:
    if not isinstance(binding, dict) or set(binding) != {
        "path",
        "sha256",
        "record_count",
    }:
        raise ValueError(f"{label} binding fields are invalid")
    path = _resolved_path(binding.get("path"), owner_path, f"{label}.path")
    expected_sha = _require_sha256(binding.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label}.sha256 is stale")
    rows = read_jsonl(path, allow_empty=True)
    if binding.get("record_count") != len(rows):
        raise ValueError(f"{label}.record_count is stale")
    return rows, path


def _scope_for_apply_manifest(
    manifest_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    manifest = read_json(manifest_path)
    scope, scope_path = _bound_json(
        manifest.get("scope_manifest"),
        owner_path=manifest_path,
        label="review apply scope_manifest",
    )
    return manifest, scope, scope_path


def _source_context(
    scope: Mapping[str, Any],
    *,
    scope_path: Path,
    review_tool: Any,
) -> Tuple[Dict[str, Path], Dict[str, Any], List[str]]:
    bindings = scope.get("source_artifacts")
    if not isinstance(bindings, dict) or set(bindings) != set(
        review_tool.SOURCE_ARTIFACTS
    ):
        raise ValueError("Review scope source_artifacts are invalid")
    paths: Dict[str, Path] = {}
    normalized: Dict[str, Any] = {}
    for role, (_, expected_schema, _) in review_tool.SOURCE_ARTIFACTS.items():
        binding = bindings.get(role)
        if not isinstance(binding, dict) or set(binding) != review_tool.SOURCE_BINDING_FIELDS:
            raise ValueError(f"Review scope source binding is invalid: {role}")
        path = _resolved_path(
            binding.get("path"), scope_path, f"source_artifacts.{role}.path"
        )
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"Review scope source SHA is stale: {role}")
        if path.stat().st_size != binding.get("byte_count"):
            raise ValueError(f"Review scope source byte_count is stale: {role}")
        if binding.get("schema_version") != expected_schema:
            raise ValueError(f"Review scope source schema is invalid: {role}")
        paths[role] = path
        normalized[role] = {**binding, "path": str(path)}
    rows, indexes = review_tool.load_source_artifacts(paths)
    for role, binding in bindings.items():
        if len(rows[role]) != binding.get("record_count"):
            raise ValueError(f"Review scope source record_count is stale: {role}")
    return paths, normalized, list(indexes["behavior_bundle"])


def _validate_plan_manifest_bindings(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    manifest_paths: Sequence[Path],
) -> None:
    if set(plan) != PLAN_FIELDS:
        raise ValueError("Resolution plan fields are invalid")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported resolution plan schema_version")
    bindings = plan.get("review_apply_manifests")
    if not isinstance(bindings, list) or len(bindings) != len(manifest_paths):
        raise ValueError("Resolution plan review apply manifest count mismatch")
    for position, (binding, actual_path) in enumerate(
        zip(bindings, manifest_paths), start=1
    ):
        if not isinstance(binding, dict) or set(binding) != PLAN_MANIFEST_BINDING_FIELDS:
            raise ValueError(
                f"review_apply_manifests[{position}] binding fields are invalid"
            )
        declared_path = _resolved_path(
            binding.get("path"),
            plan_path,
            f"review_apply_manifests[{position}].path",
        )
        if declared_path != actual_path:
            raise ValueError(
                f"review_apply_manifests[{position}].path does not match CLI input"
            )
        if _require_sha256(
            binding.get("sha256"), f"review_apply_manifests[{position}].sha256"
        ) != sha256_file(actual_path):
            raise ValueError(f"review_apply_manifests[{position}].sha256 is stale")


def _outcome(row: Mapping[str, Any]) -> str:
    outcome = row.get("review_outcome")
    if outcome is None and row.get("staging_status") == "decision_missing":
        return "missing"
    if outcome not in TERMINAL_INPUT_OUTCOMES | UNRESOLVED_OUTCOMES:
        raise ValueError(
            f"Unsupported review outcome for {row.get('base_fact_id')}: {outcome}"
        )
    return str(outcome)


def _validate_reviewer_provenance(
    value: Any,
    *,
    base_fact_id: str,
    original_staging: Mapping[str, Any],
) -> Dict[str, str]:
    if not isinstance(value, dict) or set(value) != REVIEWER_PROVENANCE_FIELDS:
        raise ValueError(f"Resolution reviewer_provenance is invalid: {base_fact_id}")
    if value.get("reviewer_type") != "codex_proxy":
        raise ValueError(
            f"Resolution reviewer_type must be codex_proxy: {base_fact_id}"
        )
    reviewer_id = _required_string(
        value.get("reviewer_id"), f"resolution {base_fact_id}.reviewer_id"
    )
    _required_string(
        value.get("review_method"), f"resolution {base_fact_id}.review_method"
    )
    _validate_timestamp(
        value.get("reviewed_at"), f"resolution {base_fact_id}.reviewed_at"
    )
    prior = original_staging.get("review_provenance")
    if not isinstance(prior, dict):
        raise ValueError(f"Original review provenance is missing: {base_fact_id}")
    if reviewer_id == prior.get("reviewer_id"):
        raise ValueError(
            f"Resolution reviewer must differ from original reviewer: {base_fact_id}"
        )
    return {str(key): str(value[key]) for key in REVIEWER_PROVENANCE_FIELDS}


def _validate_resolutions(
    raw_resolutions: Any,
    *,
    staging_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_resolutions, list):
        raise ValueError("Resolution plan resolutions must be a list")
    unresolved_ids = {
        base_fact_id
        for base_fact_id, row in staging_by_id.items()
        if _outcome(row) in UNRESOLVED_OUTCOMES
    }
    result: Dict[str, Dict[str, Any]] = {}
    for position, entry in enumerate(raw_resolutions, start=1):
        if not isinstance(entry, dict) or set(entry) != RESOLUTION_FIELDS:
            raise ValueError(f"Resolution entry {position} fields are invalid")
        base_fact_id = _required_string(
            entry.get("base_fact_id"), f"resolution {position}.base_fact_id"
        )
        if base_fact_id in result:
            raise ValueError(f"Duplicate resolution base_fact_id: {base_fact_id}")
        staging = staging_by_id.get(base_fact_id)
        if staging is None:
            raise ValueError(f"Resolution is outside input review scopes: {base_fact_id}")
        source_outcome = _outcome(staging)
        if source_outcome not in UNRESOLVED_OUTCOMES:
            raise ValueError(
                f"Resolution targets terminal input outcome: {base_fact_id} "
                f"({source_outcome})"
            )
        if entry.get("source_review_outcome") != source_outcome:
            raise ValueError(f"Resolution source outcome is stale: {base_fact_id}")
        expected_hash = _require_sha256(
            entry.get("source_review_staging_row_sha256"),
            f"resolution {base_fact_id}.source_review_staging_row_sha256",
        )
        if expected_hash != sha256_value(staging):
            raise ValueError(f"Resolution staging row hash is stale: {base_fact_id}")
        disposition = entry.get("disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"Resolution disposition is invalid: {base_fact_id}")
        if source_outcome == "defer" and disposition != "cohort_exclude":
            raise ValueError(f"Deferred review cannot remain revise: {base_fact_id}")
        reason = _required_string(
            entry.get("reason"), f"resolution {base_fact_id}.reason"
        )
        provenance = _validate_reviewer_provenance(
            entry.get("reviewer_provenance"),
            base_fact_id=base_fact_id,
            original_staging=staging,
        )
        result[base_fact_id] = {
            "base_fact_id": base_fact_id,
            "source_review_outcome": source_outcome,
            "source_review_staging_row_sha256": expected_hash,
            "disposition": str(disposition),
            "reason": reason,
            "reviewer_provenance": provenance,
        }
    missing = sorted(unresolved_ids - set(result))
    extra = sorted(set(result) - unresolved_ids)
    if missing or extra:
        raise ValueError(
            "Resolution plan must exactly cover revise/defer rows; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return result


def _resolution_note(
    original_note: Any,
    *,
    disposition: str,
    reason: str,
) -> str:
    parts: List[str] = []
    if isinstance(original_note, str) and original_note.strip():
        parts.append(f"Original review note: {original_note.strip()}")
    parts.append(
        COHORT_EXCLUSION_NOTE
        if disposition == "cohort_exclude"
        else RETAIN_REVISE_NOTE
    )
    parts.append(f"Resolution rationale: {reason.strip()}")
    return " ".join(parts)


def _rebind_decision(
    original: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    staging: Mapping[str, Any],
    resolution: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    decision = copy.deepcopy(dict(original))
    decision["scope_id"] = template["scope_id"]
    decision["review_item_sha256"] = template["review_item_sha256"]
    source_outcome = _outcome(staging)
    if source_outcome in TERMINAL_INPUT_OUTCOMES:
        if resolution is not None:
            raise ValueError(
                f"Terminal decision unexpectedly has a resolution: "
                f"{staging['base_fact_id']}"
            )
        if decision.get("decision") != source_outcome:
            raise ValueError(
                f"Decision artifact disagrees with staging: {staging['base_fact_id']}"
            )
        return decision

    if resolution is None:
        raise ValueError(f"Missing resolution: {staging['base_fact_id']}")
    decision["decision"] = (
        "revise"
        if resolution["disposition"] == "retain_revise"
        else "reject"
    )
    decision["review_provenance"] = copy.deepcopy(
        resolution["reviewer_provenance"]
    )
    decision["notes"] = _resolution_note(
        original.get("notes"),
        disposition=str(resolution["disposition"]),
        reason=str(resolution["reason"]),
    )
    decision["human_gold"] = False
    return decision


def _binding(
    path: Path,
    *,
    record_count: int | None = None,
    schema: str | None = None,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
    }
    if record_count is not None:
        value["record_count"] = record_count
    if schema is not None:
        value["schema_version"] = schema
    return value


def consolidate_review_outcomes(
    *,
    review_apply_manifest_paths: Sequence[Path],
    resolution_plan_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Validate, consolidate, and replay a single replacement review chain."""

    review_tool = _load_sibling_module(
        "review_public_benchmark_bundle.py",
        "_pitfalls_review_public_benchmark_bundle_consolidation",
    )
    finalizer = _load_sibling_module(
        "finalize_public_benchmark_bundle.py",
        "_pitfalls_finalize_public_benchmark_bundle_consolidation",
    )
    manifest_paths = [Path(path).resolve() for path in review_apply_manifest_paths]
    if not manifest_paths:
        raise ValueError("At least one review apply manifest is required")
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("Duplicate review apply manifest path")
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    plan_path = Path(resolution_plan_path).resolve()
    plan = read_json(plan_path)
    _validate_plan_manifest_bindings(
        plan,
        plan_path=plan_path,
        manifest_paths=manifest_paths,
    )

    preliminary: List[Tuple[Path, Dict[str, Any], Dict[str, Any], Path]] = []
    source_paths: Dict[str, Path] | None = None
    normalized_sources: Dict[str, Any] | None = None
    universe_source_ids: List[str] | None = None
    for manifest_path in manifest_paths:
        manifest, scope, scope_path = _scope_for_apply_manifest(manifest_path)
        paths, normalized, source_ids = _source_context(
            scope, scope_path=scope_path, review_tool=review_tool
        )
        if normalized_sources is None:
            source_paths = paths
            normalized_sources = normalized
            universe_source_ids = source_ids
        elif canonical_json_bytes(normalized_sources) != canonical_json_bytes(normalized):
            raise ValueError("Review apply manifests bind different source artifacts")
        preliminary.append((manifest_path, manifest, scope, scope_path))
    assert source_paths is not None
    assert normalized_sources is not None
    assert universe_source_ids is not None

    staging_by_id: Dict[str, Dict[str, Any]] = {}
    decisions_by_id: Dict[str, Dict[str, Any]] = {}
    source_row_origins: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    input_bindings: List[Dict[str, Any]] = []
    input_outcomes: Counter[str] = Counter()
    for manifest_path, manifest, scope, _ in preliminary:
        staging, replay_provenance = finalizer.validate_review_apply_manifest(
            manifest_path,
            universe={"source_artifacts": scope["source_artifacts"]},
            universe_source_ids=universe_source_ids,
        )
        decisions, decisions_path = _bound_jsonl(
            manifest.get("decisions"),
            owner_path=manifest_path,
            label="review apply decisions",
        )
        decision_index = _unique_index(decisions, "base_fact_id", "review decisions")
        scope_ids = [str(value) for value in scope.get("base_fact_ids", [])]
        if set(decision_index) != set(scope_ids):
            raise ValueError("Review decisions do not exactly cover their scope")
        if set(staging) != set(scope_ids):
            raise ValueError("Review staging does not exactly cover its scope")
        for base_fact_id in scope_ids:
            if base_fact_id in staging_by_id:
                raise ValueError(
                    "Duplicate base_fact_id across review apply manifests: "
                    f"{base_fact_id}"
                )
            row = staging[base_fact_id]
            outcome = _outcome(row)
            if outcome == "missing":
                raise ValueError(
                    f"Missing review decision cannot be consolidated: {base_fact_id}"
                )
            decision = decision_index[base_fact_id]
            if decision.get("decision") != outcome:
                raise ValueError(
                    f"Decision/staging outcome mismatch: {base_fact_id}"
                )
            staging_by_id[base_fact_id] = copy.deepcopy(row)
            decisions_by_id[base_fact_id] = copy.deepcopy(decision)
            source_row_origins[base_fact_id] = {
                "review_apply_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                "review_staging": {
                    "path": str(
                        replay_provenance["review_staging"]["path"]
                    ),
                    "artifact_sha256": replay_provenance["review_staging"][
                        "sha256"
                    ],
                    "record_sha256": sha256_value(row),
                },
                "decision": {
                    "path": str(decisions_path),
                    "artifact_sha256": sha256_file(decisions_path),
                    "record_sha256": sha256_value(decision),
                },
            }
            ordered_ids.append(base_fact_id)
            input_outcomes[outcome] += 1
        input_bindings.append(
            {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "scope_id": manifest.get("scope_id"),
                "record_count": len(scope_ids),
                "outcome_counts": copy.deepcopy(manifest.get("outcome_counts")),
                "deterministic_replay_performed": bool(
                    replay_provenance.get("deterministic_replay", {}).get("performed")
                ),
            }
        )

    resolutions = _validate_resolutions(
        plan.get("resolutions"), staging_by_id=staging_by_id
    )
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"Output directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    try:
        scope_result = review_tool.create_scope(
            output_dir=output_path / "scope",
            artifact_paths=source_paths,
            base_fact_ids=ordered_ids,
        )
        export_result = review_tool.export_scope(
            scope_manifest_path=Path(scope_result["manifest_path"]),
            output_dir=output_path / "export",
        )
        export_manifest = read_json(Path(export_result["manifest_path"]))
        templates_path = _resolved_path(
            export_manifest["artifacts"]["review_decisions_template"]["path"],
            Path(export_result["manifest_path"]),
            "new review_decisions_template.path",
        )
        templates = read_jsonl(templates_path)
        template_index = _unique_index(
            templates, "base_fact_id", "new decision templates"
        )
        if set(template_index) != set(ordered_ids):
            raise ValueError("New decision templates do not cover the consolidated scope")

        consolidated_decisions = [
            _rebind_decision(
                decisions_by_id[base_fact_id],
                template_index[base_fact_id],
                staging=staging_by_id[base_fact_id],
                resolution=resolutions.get(base_fact_id),
            )
            for base_fact_id in ordered_ids
        ]
        cohort_resolution_rows = []
        for base_fact_id in ordered_ids:
            resolution = resolutions.get(base_fact_id)
            if resolution is None:
                continue
            disposition = str(resolution["disposition"])
            cohort_resolution_rows.append(
                {
                    "schema_version": COHORT_RESOLUTION_SCHEMA_VERSION,
                    "base_fact_id": base_fact_id,
                    "original_review_outcome": _outcome(
                        staging_by_id[base_fact_id]
                    ),
                    "original_review_bindings": copy.deepcopy(
                        source_row_origins[base_fact_id]
                    ),
                    "resolution_plan_entry_sha256": sha256_value(resolution),
                    "cohort_eligibility_disposition": disposition,
                    "cohort_exclusion_reason": (
                        str(resolution["reason"])
                        if disposition == "cohort_exclude"
                        else None
                    ),
                    "encoded_review_outcome": (
                        "reject" if disposition == "cohort_exclude" else "revise"
                    ),
                    "factual_rejection_asserted": False,
                    "resolution_reviewer_provenance": copy.deepcopy(
                        resolution["reviewer_provenance"]
                    ),
                }
            )
        cohort_resolution_path = (
            output_path
            / "resolutions"
            / "cohort_eligibility_resolutions.jsonl"
        )
        review_tool.write_jsonl(cohort_resolution_path, cohort_resolution_rows)
        decisions_path = output_path / "decisions" / "review_decisions.jsonl"
        review_tool.write_jsonl(decisions_path, consolidated_decisions)
        apply_result = review_tool.apply_decisions(
            scope_manifest_path=Path(scope_result["manifest_path"]),
            export_manifest_path=Path(export_result["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=output_path / "apply",
            allow_partial=False,
        )
        output_staging, output_replay = finalizer.validate_review_apply_manifest(
            Path(apply_result["manifest_path"]),
            universe={
                "source_artifacts": read_json(Path(scope_result["manifest_path"]))[
                    "source_artifacts"
                ]
            },
            universe_source_ids=universe_source_ids,
        )
        if set(output_staging) != set(ordered_ids):
            raise ValueError("Consolidated apply output does not cover the input union")
        output_outcomes = Counter(_outcome(output_staging[key]) for key in ordered_ids)
        for base_fact_id in ordered_ids:
            before = _outcome(staging_by_id[base_fact_id])
            after = _outcome(output_staging[base_fact_id])
            if before in TERMINAL_INPUT_OUTCOMES and after != before:
                raise ValueError(
                    f"Terminal review outcome changed during consolidation: {base_fact_id}"
                )
            if before in UNRESOLVED_OUTCOMES:
                expected = (
                    "revise"
                    if resolutions[base_fact_id]["disposition"] == "retain_revise"
                    else "reject"
                )
                if after != expected:
                    raise ValueError(
                        f"Resolved outcome mismatch for {base_fact_id}: {after}"
                    )

        disposition_counts = Counter(
            entry["disposition"] for entry in resolutions.values()
        )
        manifest_path = output_path / "consolidation_manifest.json"
        consolidation_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "review_outcomes_consolidated_pending_fact_revision",
            "resolution_plan": _binding(plan_path),
            "inputs": {"review_apply_manifests": input_bindings},
            "source_artifacts": normalized_sources,
            "counts": {
                "input_record_count": len(ordered_ids),
                "input_outcome_counts": dict(sorted(input_outcomes.items())),
                "resolution_count": len(resolutions),
                "resolution_disposition_counts": dict(
                    sorted(disposition_counts.items())
                ),
                "output_outcome_counts": dict(sorted(output_outcomes.items())),
            },
            "artifacts": {
                "scope_manifest": _binding(Path(scope_result["manifest_path"])),
                "export_manifest": _binding(Path(export_result["manifest_path"])),
                "decisions": _binding(
                    decisions_path,
                    record_count=len(consolidated_decisions),
                    schema=review_tool.DECISION_SCHEMA_VERSION,
                ),
                "cohort_eligibility_resolutions": _binding(
                    cohort_resolution_path,
                    record_count=len(cohort_resolution_rows),
                    schema=COHORT_RESOLUTION_SCHEMA_VERSION,
                ),
                "review_apply_manifest": _binding(
                    Path(apply_result["manifest_path"])
                ),
            },
            "deterministic_replay": {
                "all_input_apply_manifests_replayed": all(
                    item["deterministic_replay_performed"] for item in input_bindings
                ),
                "output_apply_manifest_replayed": bool(
                    output_replay.get("deterministic_replay", {}).get("performed")
                ),
            },
            "cohort_eligibility_contract": COHORT_ELIGIBILITY_CONTRACT,
            "safety_contract": SAFETY_CONTRACT,
        }
        review_tool.write_json(manifest_path, consolidation_manifest)
        summary_path = output_path / "summary.json"
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "review_outcomes_consolidated_pending_fact_revision",
            "record_count": len(ordered_ids),
            "input_outcome_counts": dict(sorted(input_outcomes.items())),
            "resolution_disposition_counts": dict(sorted(disposition_counts.items())),
            "output_outcome_counts": dict(sorted(output_outcomes.items())),
            "manifest": _binding(manifest_path),
            "cohort_eligibility_resolutions": _binding(
                cohort_resolution_path,
                record_count=len(cohort_resolution_rows),
                schema=COHORT_RESOLUTION_SCHEMA_VERSION,
            ),
            "cohort_eligibility_contract": COHORT_ELIGIBILITY_CONTRACT,
            "safety_contract": SAFETY_CONTRACT,
        }
        review_tool.write_json(summary_path, summary)
    except Exception:
        shutil.rmtree(output_path)
        raise

    return {
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "review_apply_manifest_path": str(apply_result["manifest_path"]),
        "record_count": len(ordered_ids),
        "outcome_counts": dict(sorted(output_outcomes.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-apply-manifest",
        action="append",
        required=True,
        type=Path,
        help="Input review_apply_manifest.json; repeat for each disjoint scope.",
    )
    parser.add_argument("--resolution-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = consolidate_review_outcomes(
            review_apply_manifest_paths=args.review_apply_manifest,
            resolution_plan_path=args.resolution_plan,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
