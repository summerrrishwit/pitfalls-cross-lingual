#!/usr/bin/env python3
"""Rebuild the reviewed full public-benchmark cohort before exact-HF binding.

The preferred path consumes the exact retained/revised artifact from a complete
fact-resolution manifest plus complete adjudications for the newly materialized
bounded semantic candidate set.  It removes exclusions before graph building,
reconstructs retained exact-answer/frozen groups, merges adjudicated positive
leakage edges, reruns the existing component split policy, and regenerates the
existing dual distractor/Neutral candidate policy on the resulting split.  A
legacy accept/reject-only fact-review apply path remains available for older
fixtures, but an excluded row is never accepted as a semantic endpoint or
transitive bridge.

This remains a non-human, provisional, offline artifact.  Bounded candidate
review does not guarantee exhaustive semantic recall.  The tool neither emits
review/split freezes nor binds or executes an HF checkpoint/tokenizer.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "data_processed"
    / "factual_triples"
    / "public-benchmarks-full-v1"
    / "provisional"
    / "qwen3.7-plus-v1"
)
DEFAULT_FULL_BASE_FACTS = DEFAULT_SOURCE_ROOT / "full-8969-pre-hf-v1" / "full_base_facts.jsonl"
DEFAULT_SEMANTIC_CANDIDATE_MANIFEST = (
    DEFAULT_SOURCE_ROOT
    / "full-8969-formal-universe-v1"
    / "semantic-closure-candidates-v1"
    / "semantic_closure_candidate_manifest.json"
)

TOOL_VERSION = "public-benchmark-full-postreview-rebuild-v2"
MANIFEST_SCHEMA_VERSION = "public-benchmark-full-postreview-rebuild-manifest-v2"
SUMMARY_SCHEMA_VERSION = "public-benchmark-full-postreview-rebuild-summary-v1"
INTEGRITY_SCHEMA_VERSION = "public-benchmark-full-postreview-integrity-audit-v1"
EXCLUSION_SCHEMA_VERSION = "public-benchmark-full-postreview-cohort-exclusion-v1"

POSITIVE_RELATIONSHIPS = frozenset({"same_fact", "same_leakage_component"})
UNRESOLVED_FACT_OUTCOMES = frozenset({None, "defer", "revise"})


def _load_sibling(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pre_hf = _load_sibling(
    "_full_postreview_pre_hf_policy", "materialize_full_public_benchmark_pre_hf.py"
)
review_tool = _load_sibling(
    "_full_postreview_fact_review_contract", "review_public_benchmark_bundle.py"
)
semantic_runner = _load_sibling(
    "_full_postreview_semantic_review_contract",
    "run_full_public_benchmark_semantic_review.py",
)


def _load_semantic_composite_builder() -> Any:
    return _load_sibling(
        "_full_postreview_semantic_composite_contract",
        "build_full_public_benchmark_semantic_review_composite.py",
    )


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _index_unique(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label} row {row_number}.{field}")
        if key in output:
            raise ValueError(f"Duplicate {label} {field}: {key}")
        output[key] = dict(row)
    return output


def _resolved_binding_path(
    binding: Mapping[str, Any], owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    candidate = Path(_required_string(raw, f"{label}.path"))
    if not candidate.is_absolute():
        candidate = owner_path.parent / candidate
    return candidate.resolve()


def _verify_bound_file(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    explicit_path: Optional[Path] = None,
) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is invalid")
    path = _resolved_binding_path(binding, owner_path, label)
    if explicit_path is not None and path != Path(explicit_path).resolve():
        raise ValueError(f"{label} binding does not match the supplied path")
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != pre_hf.sha256_file(path):
        raise ValueError(f"{label} SHA-256 binding is stale")
    if binding.get("byte_count") is not None and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count binding is stale")
    return path


def _input_binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    value: Dict[str, Any] = {
        "path": str(path),
        "sha256": pre_hf.sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        value["schema_version"] = schema_version
    if record_count is not None:
        value["record_count"] = record_count
    return value


def _output_binding(
    path: Path, schema_version: str, record_count: Optional[int] = None
) -> Dict[str, Any]:
    return pre_hf.output_binding(path, schema_version, record_count)


def _assert_snapshots_current(bindings: Sequence[Mapping[str, Any]]) -> None:
    for binding in bindings:
        path = Path(str(binding["path"])).resolve()
        if pre_hf.sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"Input changed during rebuild: {path}")


def _bound_file_snapshots(
    value: Any, *, owner_path: Path
) -> List[Dict[str, Any]]:
    """Collect current snapshots for every path+SHA binding in a manifest subtree."""

    output: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        if "sha256" in value and ("path" in value or "filename" in value):
            path = _resolved_binding_path(value, owner_path, "manifest dependency")
            if not path.is_file():
                raise FileNotFoundError(path)
            if value.get("sha256") != pre_hf.sha256_file(path):
                raise ValueError(f"Manifest dependency SHA-256 binding is stale: {path}")
            output.append(_input_binding(path))
        for nested in value.values():
            output.extend(_bound_file_snapshots(nested, owner_path=owner_path))
    elif isinstance(value, list):
        for nested in value:
            output.extend(_bound_file_snapshots(nested, owner_path=owner_path))
    unique: Dict[str, Dict[str, Any]] = {}
    for binding in output:
        unique[str(binding["path"])] = binding
    return [unique[path] for path in sorted(unique)]


def _validate_full_base_rows(
    path: Path, expected_input_count: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rows = pre_hf.read_jsonl(path)
    if len(rows) != expected_input_count:
        raise ValueError(
            "full_base_facts record count differs from expected_input_count: "
            f"{len(rows)} != {expected_input_count}"
        )
    index = _index_unique(rows, "base_fact_id", "full_base_facts")
    for base_fact_id, row in index.items():
        if row.get("schema_version") != pre_hf.FULL_FACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported full-base schema: {base_fact_id}")
        if row.get("canonical_status") != "pending_review":
            raise ValueError(f"Full-base fact is not pending_review: {base_fact_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"Full-base fact must have human_gold=false: {base_fact_id}")
        if row.get("split_status") != "provisional_not_frozen":
            raise ValueError(f"Full-base split is not provisional: {base_fact_id}")
        if row.get("split_assignment") not in pre_hf.SPLITS:
            raise ValueError(f"Full-base split is invalid: {base_fact_id}")
        if row.get("hf_model_execution_status") != "not_run":
            raise ValueError(f"Full-base fact already has HF model execution: {base_fact_id}")
        if row.get("hf_tokenizer_execution_status") != "not_run":
            raise ValueError(
                f"Full-base fact already has HF tokenizer execution: {base_fact_id}"
            )
        for field in (
            "leakage_component_id",
            "relation_partition_id",
            "answer_type_bucket",
        ):
            _required_string(row.get(field), f"{base_fact_id}.{field}")
    binding = {
        **_input_binding(
            path,
            schema_version=pre_hf.FULL_FACT_SCHEMA_VERSION,
            record_count=len(rows),
        ),
        "ordered_base_fact_ids_sha256": pre_hf.sha256_value(
            [row["base_fact_id"] for row in rows]
        ),
        "ordered_row_hashes_sha256": pre_hf.sha256_value(
            [pre_hf.sha256_value(row) for row in rows]
        ),
    }
    return rows, index, binding


def classify_fact_review_staging(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    staging_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Partition resolved accepts/rejects and fail closed on unresolved outcomes."""

    outcomes: Counter[str] = Counter()
    retained_ids: List[str] = []
    rejected_ids: List[str] = []
    unresolved: List[Tuple[str, Optional[str]]] = []
    for full_row in full_rows:
        base_fact_id = str(full_row["base_fact_id"])
        if base_fact_id not in staging_index:
            unresolved.append((base_fact_id, None))
            outcomes["missing"] += 1
            continue
        staging = staging_index[base_fact_id]
        for field in ("subject_en", "relation_raw", "answer_en", "canonical_fact_en"):
            if staging.get(field) != full_row.get(field):
                raise ValueError(
                    f"Fact-review staging {field} differs from full_base_facts: "
                    f"{base_fact_id}"
                )
        outcome = staging.get("review_outcome")
        outcomes[str(outcome or "missing")] += 1
        if outcome == "accept":
            completion = staging.get("review_completion")
            if not isinstance(completion, dict) or not all(
                completion.get(field) is True
                for field in (
                    "codex_proxy_scope_review_complete",
                    "member_review_complete",
                    "alias_review_complete",
                    "distractor_review_complete",
                )
            ):
                raise ValueError(f"Accepted fact review is incomplete: {base_fact_id}")
            retained_ids.append(base_fact_id)
        elif outcome == "reject":
            rejected_ids.append(base_fact_id)
        elif outcome in UNRESOLVED_FACT_OUTCOMES:
            unresolved.append((base_fact_id, outcome))
        else:
            raise ValueError(f"Unsupported fact-review outcome: {base_fact_id}={outcome!r}")

    extra = sorted(set(staging_index) - {str(row["base_fact_id"]) for row in full_rows})
    if extra:
        raise ValueError(f"Fact-review staging contains unknown facts: {extra[:5]}")
    if unresolved:
        preview = ", ".join(
            f"{base_fact_id}:{outcome or 'missing'}"
            for base_fact_id, outcome in unresolved[:5]
        )
        raise ValueError(
            "Unresolved fact-review outcomes fail closed; resolve revise/defer/missing "
            f"before rebuild: {preview}"
        )
    if not retained_ids:
        raise ValueError("Fact review retained no facts")
    return {
        "retained_ids": retained_ids,
        "rejected_ids": rejected_ids,
        "outcome_counts": dict(sorted(outcomes.items())),
    }


def _replay_fact_review_apply(
    *,
    apply_manifest_path: Path,
    full_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate the apply chain and replay every staging row exactly."""

    apply_manifest_path = Path(apply_manifest_path).resolve()
    manifest = review_tool.read_json(apply_manifest_path)
    if manifest.get("schema_version") != review_tool.APPLY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Fact-review apply manifest schema_version is unsupported")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or not all(
        (
            safety.get("output_is_staging_only") is True,
            safety.get("source_artifacts_modified") is False,
            safety.get("canonical_freeze_performed") is False,
            safety.get("automatic_promotion_performed") is False,
            safety.get("missing_is_reject") is False,
            safety.get("revise_outcome_supported") is True,
            safety.get("revision_application_supported") is False,
        )
    ):
        raise ValueError("Fact-review apply safety contract is invalid")

    scope_path = _verify_bound_file(
        manifest.get("scope_manifest"),
        owner_path=apply_manifest_path,
        label="fact-review scope manifest",
    )
    export_path = _verify_bound_file(
        manifest.get("export_manifest"),
        owner_path=apply_manifest_path,
        label="fact-review export manifest",
    )
    decisions_path = _verify_bound_file(
        manifest.get("decisions"),
        owner_path=apply_manifest_path,
        label="fact-review decisions",
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Fact-review apply artifacts are missing")
    staging_binding = artifacts.get("review_staging")
    staging_path = _verify_bound_file(
        staging_binding,
        owner_path=apply_manifest_path,
        label="fact-review staging",
    )

    scope = review_tool.read_json(scope_path)
    review_tool._validate_scope_manifest(scope)
    if manifest.get("scope_id") != scope.get("scope_id"):
        raise ValueError("Fact-review apply scope_id mismatch")
    full_ids = [str(row["base_fact_id"]) for row in full_rows]
    if scope.get("base_fact_ids") != full_ids:
        raise ValueError("Fact-review scope does not exactly match full_base_facts order")
    if manifest.get("scoped_base_fact_count") != len(full_ids):
        raise ValueError("Fact-review scoped_base_fact_count is stale")

    source_paths = review_tool._scope_artifact_paths(scope)
    source_rows, source_indexes = review_tool.load_source_artifacts(source_paths)
    review_tool._verify_scope_selection_source(scope, source_indexes)
    for role, binding in scope["source_artifacts"].items():
        if binding.get("record_count") != len(source_rows[role]):
            raise ValueError(f"Fact-review scope {role} record_count is stale")

    export = review_tool.read_json(export_path)
    item_path, template_path = review_tool._validate_export_manifest(
        export, scope, scope_path
    )
    items = review_tool.read_jsonl(item_path)
    templates = review_tool.read_jsonl(template_path)
    item_index = _index_unique(items, "base_fact_id", "fact-review items")
    template_index = _index_unique(
        templates, "base_fact_id", "fact-review decision templates"
    )
    if set(item_index) != set(full_ids) or set(template_index) != set(full_ids):
        raise ValueError("Fact-review export does not exactly cover full_base_facts")
    for base_fact_id in full_ids:
        expected_item = review_tool._build_export_item(
            str(scope["scope_id"]), base_fact_id, source_indexes
        )
        if item_index[base_fact_id] != expected_item:
            raise ValueError(f"Fact-review item is stale: {base_fact_id}")
        if template_index[base_fact_id] != review_tool._decision_template(expected_item):
            raise ValueError(f"Fact-review decision template is stale: {base_fact_id}")

    decisions = review_tool.read_jsonl(decisions_path, allow_empty=True)
    if manifest["decisions"].get("record_count") != len(decisions):
        raise ValueError("Fact-review decision record_count is stale")
    decision_index = _index_unique(decisions, "base_fact_id", "fact-review decisions")
    unknown_decisions = sorted(set(decision_index) - set(full_ids))
    if unknown_decisions:
        raise ValueError(
            f"Fact-review decisions contain unknown facts: {unknown_decisions[:5]}"
        )
    active_decisions = {
        base_fact_id: decision
        for base_fact_id, decision in decision_index.items()
        if decision.get("decision") is not None
    }
    staging_rows = review_tool.read_jsonl(staging_path)
    if not isinstance(staging_binding, dict) or staging_binding.get("record_count") != len(
        staging_rows
    ):
        raise ValueError("Fact-review staging record_count is stale")
    if staging_binding.get("schema_version") != review_tool.STAGING_SCHEMA_VERSION:
        raise ValueError("Fact-review staging schema_version is unsupported")
    staging_index = _index_unique(staging_rows, "base_fact_id", "fact-review staging")
    if set(staging_index) != set(full_ids):
        raise ValueError("Fact-review staging does not exactly cover full_base_facts")

    replay_bindings = {
        "scope_manifest_sha256": pre_hf.sha256_file(scope_path),
        "export_manifest_sha256": pre_hf.sha256_file(export_path),
        "decisions_artifact_sha256": pre_hf.sha256_file(decisions_path),
        "apply_manifest_schema_version": review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
    }
    for full_row in full_rows:
        base_fact_id = str(full_row["base_fact_id"])
        decision = active_decisions.get(base_fact_id)
        expected_staging = review_tool._staging_record(
            item_index[base_fact_id], decision, replay_bindings
        )
        staging = staging_index[base_fact_id]
        if staging != expected_staging:
            raise ValueError(f"Fact-review staging replay mismatch: {base_fact_id}")
    dispositions = classify_fact_review_staging(
        full_rows=full_rows, staging_index=staging_index
    )
    expected_counts = dispositions["outcome_counts"]
    if manifest.get("outcome_counts") != expected_counts:
        raise ValueError("Fact-review apply outcome_counts are stale")

    snapshot_paths = {
        apply_manifest_path,
        scope_path,
        export_path,
        decisions_path,
        staging_path,
        item_path,
        template_path,
        *source_paths.values(),
    }
    return {
        "manifest": manifest,
        "staging_rows": staging_rows,
        "staging_index": staging_index,
        "retained_ids": dispositions["retained_ids"],
        "rejected_ids": dispositions["rejected_ids"],
        "outcome_counts": expected_counts,
        "bindings": {
            "apply_manifest": _input_binding(
                apply_manifest_path,
                schema_version=review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
            ),
            "review_staging": _input_binding(
                staging_path,
                schema_version=review_tool.STAGING_SCHEMA_VERSION,
                record_count=len(staging_rows),
            ),
            "scope_manifest": _input_binding(scope_path),
            "export_manifest": _input_binding(export_path),
            "decisions": _input_binding(decisions_path, record_count=len(decisions)),
        },
        "snapshots": [_input_binding(path) for path in sorted(snapshot_paths)],
    }


def _load_fact_resolution_evidence(
    *,
    manifest_path: Path,
    resolved_full_base_facts_path: Path,
) -> Dict[str, Any]:
    """Validate the complete resolution chain and expose its successor cohort."""

    context = semantic_runner.materializer.load_fact_resolution_context(
        fact_resolution_manifest_path=manifest_path,
        resolved_full_base_facts_path=resolved_full_base_facts_path,
    )
    manifest = context["manifest"]
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Fact-resolution inputs are missing")
    staging_path = _verify_bound_file(
        inputs.get("review_staging"),
        owner_path=Path(manifest_path).resolve(),
        label="fact-resolution source review_staging",
    )
    staging_rows = review_tool.read_jsonl(staging_path)
    staging_index = _index_unique(
        staging_rows, "base_fact_id", "fact-resolution source review staging"
    )
    if set(staging_index) != set(context["source_ids"]):
        raise ValueError(
            "Fact-resolution source review staging does not cover source universe"
        )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Fact-resolution counts are missing")
    return {
        "mode": "fact_resolution_successor_cohort",
        "manifest": manifest,
        "manifest_path": context["manifest_path"],
        "source_rows": context["source_rows"],
        "source_ids": context["source_ids"],
        "retained_ids": context["retained_ids"],
        "rejected_ids": context["excluded_ids"],
        "revised_ids": context["revised_ids"],
        "ledger_rows": context["ledger_rows"],
        "ledger_index": context["ledger_index"],
        "lineage_rows": context["lineage_rows"],
        "lineage_index": context["lineage_index"],
        "source_exclusion_rows": context["exclusion_rows"],
        "source_exclusion_index": context["exclusion_index"],
        "staging_rows": staging_rows,
        "staging_index": staging_index,
        "outcome_counts": copy.deepcopy(counts.get("source_review_outcome_counts") or {}),
        "successor_id": context["successor_id"],
        "successor_payload": context["successor_payload"],
        "bindings": copy.deepcopy(context["bindings"]),
        "snapshots": context["snapshots"],
    }


def _validate_semantic_adjudications(
    *,
    candidates: Sequence[Mapping[str, Any]],
    units: Sequence[Any],
    adjudications: Sequence[Mapping[str, Any]],
    authority_metadata_by_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    candidate_index = _index_unique(candidates, "pair_id", "semantic candidates")
    unit_index = {str(unit.pair_id): unit for unit in units}
    if set(unit_index) != set(candidate_index):
        raise ValueError("Semantic review units do not exactly cover candidate pairs")
    decision_index = _index_unique(
        adjudications, "pair_id", "semantic adjudications"
    )
    missing = sorted(set(candidate_index) - set(decision_index))
    extra = sorted(set(decision_index) - set(candidate_index))
    if missing or extra:
        raise ValueError(
            "Semantic adjudications must exactly cover bounded candidates: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    expected_fields = {
        "schema_version",
        "pair_id",
        "candidate_row_sha256",
        "adjudication_template_row_sha256",
        "left_endpoint_id",
        "left_input_record_sha256",
        "right_endpoint_id",
        "right_input_record_sha256",
        *semantic_runner.FIELD_NAMES,
        "relationship",
        "rationale",
        "confidence",
        "reviewer_type",
        "reviewer_id",
        "requested_model",
        "expected_response_model",
        "response_model",
        "response_model_identity_status",
        "review_method",
        "reviewed_at",
        "behavior_blind",
        "human_gold",
    }
    for pair_id, decision in decision_index.items():
        if set(decision) != expected_fields:
            raise ValueError(f"Semantic adjudication fields are invalid: {pair_id}")
        unit = unit_index[pair_id]
        expected_bindings = {
            "candidate_row_sha256": unit.candidate_row_sha256,
            "adjudication_template_row_sha256": unit.template_row_sha256,
            "left_endpoint_id": unit.left_endpoint_id,
            "left_input_record_sha256": unit.left_input_record_sha256,
            "right_endpoint_id": unit.right_endpoint_id,
            "right_input_record_sha256": unit.right_input_record_sha256,
        }
        if decision.get("schema_version") != semantic_runner.ADJUDICATION_SCHEMA:
            raise ValueError(f"Semantic adjudication schema is invalid: {pair_id}")
        for field, expected in expected_bindings.items():
            if decision.get(field) != expected:
                raise ValueError(
                    f"Semantic adjudication {field} binding is stale: {pair_id}"
                )
        if decision.get("reviewer_type") != "independent_model_proxy":
            raise ValueError(f"Semantic adjudication reviewer_type is invalid: {pair_id}")
        expected_identity = (
            dict(authority_metadata_by_id[pair_id])
            if authority_metadata_by_id is not None
            else {
                "reviewer_id": (
                    f"{semantic_runner.PROTOCOL_REVIEWER_PROVIDER_PROFILE}:"
                    f"{semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL}"
                ),
                "requested_model": semantic_runner.DEFAULT_REVIEWER_ROUTE,
                "expected_response_model": semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
                "response_model": semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
                "response_model_identity_status": "matched",
                "review_method": semantic_runner.REVIEW_METHOD,
            }
        )
        if set(expected_identity) != {
            "reviewer_id",
            "requested_model",
            "expected_response_model",
            "response_model",
            "response_model_identity_status",
            "review_method",
        }:
            raise ValueError(
                f"Semantic authority metadata is invalid: {pair_id}"
            )
        for field, expected in expected_identity.items():
            if decision.get(field) != expected:
                raise ValueError(
                    f"Semantic adjudication model identity is invalid: {pair_id}.{field}"
                )
        for field in ("reviewer_id", "reviewed_at"):
            _required_string(decision.get(field), f"{pair_id}.{field}")
        if decision.get("behavior_blind") is not True:
            raise ValueError(f"Semantic adjudication is not behavior-blind: {pair_id}")
        if decision.get("human_gold") is not False:
            raise ValueError(f"Semantic adjudication must have human_gold=false: {pair_id}")
        verdict = {
            "pair_id": pair_id,
            "relationship": decision.get("relationship"),
            **{field: decision.get(field) for field in semantic_runner.FIELD_NAMES},
            "rationale": decision.get("rationale"),
            "confidence": decision.get("confidence"),
        }
        semantic_runner.validate_batch_response(
            {
                "schema_version": semantic_runner.BATCH_RESPONSE_SCHEMA,
                "records": [verdict],
            },
            [unit],
        )
    return decision_index


def _validate_semantic_run_manifest(
    *,
    run_manifest_path: Path,
    candidate_manifest_path: Path,
    candidate_path: Path,
    template_path: Path,
    adjudications_path: Path,
    units: Sequence[Any],
) -> Dict[str, Any]:
    """Validate the exact model-route and source-code evidence for adjudications."""

    run_manifest_path = Path(run_manifest_path).resolve()
    run_manifest = pre_hf.read_json(run_manifest_path)
    if run_manifest.get("schema_version") != semantic_runner.RUN_MANIFEST_SCHEMA:
        raise ValueError("Semantic run manifest schema is unsupported")
    if run_manifest.get("tool_version") != semantic_runner.TOOL_VERSION:
        raise ValueError("Semantic run manifest tool version is unsupported")
    if run_manifest.get("status") != "completed":
        raise ValueError("Semantic run manifest is incomplete")

    run_contract = run_manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("Semantic run contract is missing")
    run_contract_sha256 = semantic_runner.sha256_value(run_contract)
    if run_manifest.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("Semantic run contract SHA is stale")
    if run_contract.get("tool_version") != semantic_runner.TOOL_VERSION:
        raise ValueError("Semantic run contract tool version is unsupported")
    if run_contract.get("source_fingerprints") != semantic_runner.source_fingerprints():
        raise ValueError("Semantic run source fingerprints are stale")

    protocol_identity = {
        "provider_profile": semantic_runner.PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": semantic_runner.DEFAULT_REVIEWER_ROUTE,
        "expected_response_model": semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if run_contract.get("protocol_reviewer_identity") != protocol_identity:
        raise ValueError("Semantic run protocol reviewer identity is invalid")
    reviewer_spec = run_contract.get("reviewer_spec")
    if not isinstance(reviewer_spec, dict) or any(
        reviewer_spec.get(field) != expected
        for field, expected in {
            "provider_profile": protocol_identity["provider_profile"],
            "model": protocol_identity["requested_model"],
        }.items()
    ):
        raise ValueError("Semantic run reviewer identity is invalid")
    if (
        run_contract.get("expected_response_model")
        != protocol_identity["expected_response_model"]
    ):
        raise ValueError("Semantic run expected response model is invalid")
    semantic_runner.validate_route_identity_set(
        run_contract.get("route_identity"),
        expected_routes=[
            (
                protocol_identity["provider_profile"],
                protocol_identity["requested_model"],
                protocol_identity["expected_response_model"],
            )
        ],
    )

    expected_prompt_contract = semantic_runner.sha256_value(
        {
            "prompt_version": semantic_runner.PROMPT_VERSION,
            "instructions": semantic_runner.REVIEW_INSTRUCTIONS,
            "projection_schema": semantic_runner.PROJECTION_SCHEMA,
            "batch_response_schema": semantic_runner.BATCH_RESPONSE_SCHEMA,
            "field_names": semantic_runner.FIELD_NAMES,
        }
    )
    if run_contract.get("prompt_contract_sha256") != expected_prompt_contract:
        raise ValueError("Semantic run prompt contract is stale")

    expected_ids = [str(unit.pair_id) for unit in units]
    if not all(
        (
            run_contract.get("selected_pair_ids_sha256")
            == semantic_runner.sha256_value(expected_ids),
            run_contract.get("selected_pair_count") == len(units),
            run_contract.get("full_candidate_pair_count") == len(units),
            run_contract.get("selection_is_full_candidate_set") is True,
            run_manifest.get("selected_pair_count") == len(units),
            run_manifest.get("full_candidate_pair_count") == len(units),
            run_manifest.get("selection_is_full_candidate_set") is True,
            run_manifest.get("candidate_adjudication_complete_for_full_set") is True,
            run_manifest.get("terminal_status_counts")
            == ({"completed": len(units)} if units else {}),
        )
    ):
        raise ValueError("Semantic run selected pair contract is stale")

    for field, path in (
        ("candidate_manifest", candidate_manifest_path),
        ("candidate_pairs", candidate_path),
        ("adjudication_template", template_path),
    ):
        _verify_bound_file(
            run_contract.get(field),
            owner_path=run_manifest_path,
            label=f"semantic run {field}",
            explicit_path=path,
        )

    artifacts = run_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Semantic run artifacts are missing")
    _verify_bound_file(
        artifacts.get("adjudications"),
        owner_path=run_manifest_path,
        label="semantic run adjudications",
        explicit_path=adjudications_path,
    )
    checkpoint_path = _verify_bound_file(
        artifacts.get("checkpoint"),
        owner_path=run_manifest_path,
        label="semantic run checkpoint",
    )
    checkpoint = semantic_runner._checkpoint_index(
        path=checkpoint_path,
        units_by_id={str(unit.pair_id): unit for unit in units},
        run_contract_sha256=run_contract_sha256,
        protocol_identity=protocol_identity,
    )
    if set(checkpoint) != set(expected_ids) or any(
        row.get("terminal_status") != "completed" for row in checkpoint.values()
    ):
        raise ValueError("Semantic run checkpoint coverage is incomplete")

    boundary = run_manifest.get("evidence_boundary")
    if not isinstance(boundary, dict) or not all(
        (
            boundary.get("reviewer_type") == "independent_model_proxy",
            boundary.get("human_gold") is False,
            boundary.get("behavior_blind") is True,
            boundary.get("target_behavior_consumed") is False,
            boundary.get("hf_model_execution_count") == 0,
            boundary.get("hf_tokenizer_execution_count") == 0,
            boundary.get("validation_behavior_exposure_count") == 0,
            boundary.get("sealed_behavior_exposure_count") == 0,
        )
    ):
        raise ValueError("Semantic run evidence boundary is invalid")

    return {
        "manifest": run_manifest,
        "binding": _input_binding(
            run_manifest_path,
            schema_version=semantic_runner.RUN_MANIFEST_SCHEMA,
        ),
        "run_contract_sha256": run_contract_sha256,
        "route_identity_set_sha256": run_contract["route_identity"][
            "route_identity_set_sha256"
        ],
    }


def _load_semantic_review_evidence(
    *,
    candidate_manifest_path: Path,
    adjudications_path: Optional[Path],
    run_manifest_path: Optional[Path],
    composite_manifest_path: Optional[Path],
    full_base_facts_path: Path,
    fact_resolution: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    direct_supplied = adjudications_path is not None or run_manifest_path is not None
    if composite_manifest_path is None:
        if adjudications_path is None or run_manifest_path is None:
            raise ValueError(
                "Supply both semantic adjudications/run manifest or one semantic "
                "composite manifest"
            )
    elif direct_supplied:
        raise ValueError(
            "Supply direct semantic evidence XOR semantic composite authority"
        )
    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    manifest, candidate_path, template_path, units = semantic_runner.load_review_units(
        candidate_manifest_path=candidate_manifest_path
    )
    current_binding = manifest["inputs"]["current_full_base_facts"]
    current_path = _resolved_binding_path(
        current_binding, candidate_manifest_path, "semantic current_full_base_facts"
    )
    if current_path != Path(full_base_facts_path).resolve():
        raise ValueError(
            "Semantic candidate manifest is not bound to the supplied full_base_facts"
        )
    if current_binding.get("sha256") != pre_hf.sha256_file(full_base_facts_path):
        raise ValueError("Semantic candidate full_base_facts binding is stale")
    candidates = pre_hf.read_jsonl(candidate_path, allow_empty=True)
    authority_metadata_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    composite_authority: Optional[Dict[str, Any]] = None
    if composite_manifest_path is not None:
        composite_tool = _load_semantic_composite_builder()
        composite_authority = composite_tool.load_composite_authority(
            Path(composite_manifest_path).resolve()
        )
        bound_candidate = composite_authority["manifest"].get(
            "candidate_manifest"
        )
        bound_candidate_path = _resolved_binding_path(
            bound_candidate,
            composite_authority["manifest_path"],
            "semantic composite candidate_manifest",
        )
        if bound_candidate_path != candidate_manifest_path or bound_candidate.get(
            "sha256"
        ) != pre_hf.sha256_file(candidate_manifest_path):
            raise ValueError(
                "Semantic composite is not bound to the supplied candidate manifest"
            )
        adjudications_path = composite_authority["adjudications_path"]
        adjudications = copy.deepcopy(composite_authority["adjudications"])
        parent_identity = composite_authority["parent"].manifest[
            "run_contract"
        ]["protocol_reviewer_identity"]
        supplement_spec = composite_authority["supplement"]["reviewer_spec"]
        metadata_by_source = {
            "parent_completed": {
                "reviewer_id": (
                    f"{parent_identity['provider_profile']}:"
                    f"{parent_identity['expected_response_model']}"
                ),
                "requested_model": parent_identity["requested_model"],
                "expected_response_model": parent_identity[
                    "expected_response_model"
                ],
                "response_model": parent_identity["expected_response_model"],
                "response_model_identity_status": "matched",
                "review_method": semantic_runner.REVIEW_METHOD,
            },
            "supplement_recovery": {
                "reviewer_id": (
                    f"{supplement_spec['provider_profile']}:"
                    f"{supplement_spec['expected_response_model']}"
                ),
                "requested_model": supplement_spec["model"],
                "expected_response_model": supplement_spec[
                    "expected_response_model"
                ],
                "response_model": supplement_spec["expected_response_model"],
                "response_model_identity_status": "matched",
                "review_method": composite_tool.supplement_runner.REVIEW_METHOD,
            },
        }
        authority_metadata_by_id = {
            str(lineage["pair_id"]): copy.deepcopy(
                metadata_by_source[str(lineage["source_authority"])]
            )
            for lineage in composite_authority["lineage"]
        }
    else:
        adjudications_path = Path(adjudications_path).resolve()
        run_manifest_path = Path(run_manifest_path).resolve()
        adjudications = pre_hf.read_jsonl(adjudications_path, allow_empty=True)
    decisions = _validate_semantic_adjudications(
        candidates=candidates,
        units=units,
        adjudications=adjudications,
        authority_metadata_by_id=authority_metadata_by_id,
    )
    run_evidence = (
        _validate_semantic_run_manifest(
            run_manifest_path=run_manifest_path,
            candidate_manifest_path=candidate_manifest_path,
            candidate_path=candidate_path,
            template_path=template_path,
            adjudications_path=adjudications_path,
            units=units,
        )
        if composite_authority is None
        else None
    )

    component_binding = manifest["inputs"]["current_leakage_components"]
    split_binding = manifest["inputs"]["current_split_manifest"]
    component_path = _verify_bound_file(
        component_binding,
        owner_path=candidate_manifest_path,
        label="semantic current_leakage_components",
    )
    split_path = _verify_bound_file(
        split_binding,
        owner_path=candidate_manifest_path,
        label="semantic current_split_manifest",
    )
    components = pre_hf.read_jsonl(component_path)
    split_manifest = pre_hf.read_json(split_path)
    expected_count = manifest.get("candidate_summary", {}).get("candidate_pair_count")
    if expected_count != len(candidates) or len(decisions) != len(candidates):
        raise ValueError("Semantic candidate/adjudication counts are stale")
    limitations = manifest.get("limitations")
    if not isinstance(limitations, dict) or not all(
        (
            limitations.get("candidate_generation_is_bounded") is True,
            limitations.get("all_record_pairs_enumerated") is False,
            limitations.get("semantic_near_duplicate_recall_guaranteed") is False,
        )
    ):
        raise ValueError("Semantic bounded-recall limitations are invalid")

    if fact_resolution is not None:
        resolved_contract = manifest.get("resolved_cohort")
        expected_contract = {
            "source_formal_universe_id": manifest.get("source_formal_universe_id"),
            "successor_universe_id": fact_resolution["successor_id"],
            "selection_is_complete_retained_cohort": True,
            "source_record_count": len(fact_resolution["source_ids"]),
            "retained_record_count": len(fact_resolution["retained_ids"]),
            "revised_record_count": len(fact_resolution["revised_ids"]),
            "excluded_record_count": len(fact_resolution["rejected_ids"]),
            "ordered_source_base_fact_ids_sha256": pre_hf.sha256_value(
                fact_resolution["source_ids"]
            ),
            "ordered_retained_base_fact_ids_sha256": pre_hf.sha256_value(
                fact_resolution["retained_ids"]
            ),
            "ordered_revised_base_fact_ids_sha256": pre_hf.sha256_value(
                fact_resolution["revised_ids"]
            ),
            "ordered_excluded_base_fact_ids_sha256": pre_hf.sha256_value(
                fact_resolution["rejected_ids"]
            ),
            "excluded_ids_are_candidate_endpoints": False,
            "excluded_ids_are_component_members": False,
            "source_universe_lineage_preserved": True,
        }
        if resolved_contract != expected_contract:
            raise ValueError("Semantic candidate successor-cohort contract is stale")
        if manifest.get("universe_id") != fact_resolution["successor_id"]:
            raise ValueError("Semantic candidate successor universe ID is stale")
        inputs = manifest.get("inputs")
        if not isinstance(inputs, dict) or inputs.get("fact_resolution") != fact_resolution[
            "bindings"
        ]:
            raise ValueError("Semantic candidate fact-resolution binding is stale")
        invalidation = manifest.get("upstream_review_invalidation")
        if not isinstance(invalidation, dict) or not all(
            (
                invalidation.get("fact_resolution_applied") is True,
                invalidation.get("pre_resolution_semantic_candidates_valid") is False,
                invalidation.get("pre_resolution_semantic_adjudications_valid") is False,
                invalidation.get("pre_resolution_translation_reviews_valid") is False,
                invalidation.get("pre_resolution_distractor_reviews_valid") is False,
                invalidation.get("pre_resolution_neutral_reviews_valid") is False,
                invalidation.get("pair_id_alone_is_sufficient_for_review_reuse") is False,
            )
        ):
            raise ValueError("Semantic candidate upstream-review invalidation is missing")
        retained = set(fact_resolution["retained_ids"])
        excluded = set(fact_resolution["rejected_ids"])
        endpoint_ids = set()
        for candidate in candidates:
            for side in ("left", "right"):
                provenance = candidate[side]["provenance"]
                if provenance.get("input_role") == "current_full_base":
                    base_fact_id = str(provenance.get("base_fact_id") or "")
                    endpoint_ids.add(base_fact_id)
                    if base_fact_id not in retained or base_fact_id in excluded:
                        raise ValueError(
                            "Semantic candidate references an excluded or unknown fact: "
                            f"{base_fact_id}"
                        )

    return {
        "manifest": manifest,
        "candidates": candidates,
        "decisions": decisions,
        "components": components,
        "split_manifest": split_manifest,
        "component_path": component_path,
        "split_path": split_path,
        "bindings": {
            "authority_mode": (
                "direct_semantic_review_run"
                if composite_authority is None
                else "composite_semantic_review_authority"
            ),
            "run_manifest": (
                run_evidence["binding"] if run_evidence is not None else None
            ),
            "composite_manifest": (
                None
                if composite_authority is None
                else composite_authority["manifest_binding"]
            ),
            "candidate_manifest": _input_binding(
                candidate_manifest_path,
                schema_version=semantic_runner.materializer.MANIFEST_SCHEMA,
            ),
            "candidate_pairs": _input_binding(
                candidate_path,
                schema_version=semantic_runner.materializer.CANDIDATE_SCHEMA,
                record_count=len(candidates),
            ),
            "adjudication_template": _input_binding(
                template_path,
                schema_version=semantic_runner.materializer.ADJUDICATION_TEMPLATE_SCHEMA,
                record_count=len(units),
            ),
            "adjudications": _input_binding(
                adjudications_path,
                schema_version=semantic_runner.ADJUDICATION_SCHEMA,
                record_count=len(adjudications),
            ),
            "current_leakage_components": _input_binding(
                component_path,
                schema_version=pre_hf.COMPONENT_SCHEMA_VERSION,
                record_count=len(components),
            ),
            "current_split_manifest": _input_binding(
                split_path,
                schema_version=pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
            ),
            "source_run_contract_sha256": (
                {"direct": run_evidence["run_contract_sha256"]}
                if run_evidence is not None
                else composite_authority["source_run_contract_sha256"]
            ),
            "source_route_identity_set_sha256": (
                {"direct": run_evidence["route_identity_set_sha256"]}
                if run_evidence is not None
                else composite_authority["source_route_identity_set_sha256"]
            ),
        },
        "snapshots": (
            [
                run_evidence["binding"],
                _input_binding(candidate_manifest_path),
                _input_binding(adjudications_path),
                *_bound_file_snapshots(manifest, owner_path=candidate_manifest_path),
            ]
            if run_evidence is not None
            else [
                *composite_authority["snapshot_bindings"],
                _input_binding(candidate_manifest_path),
                *_bound_file_snapshots(manifest, owner_path=candidate_manifest_path),
            ]
        ),
        "authority_mode": (
            "direct_semantic_review_run"
            if composite_authority is None
            else "composite_semantic_review_authority"
        ),
        "run_contract_sha256": (
            run_evidence["run_contract_sha256"]
            if run_evidence is not None
            else None
        ),
        "route_identity_set_sha256": (
            run_evidence["route_identity_set_sha256"]
            if run_evidence is not None
            else None
        ),
    }


def _validate_current_components(
    *,
    full_index: Mapping[str, Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    component_index = _index_unique(
        components, "leakage_component_id", "current leakage components"
    )
    component_by_fact: Dict[str, str] = {}
    for component_id, component in component_index.items():
        members = component.get("base_fact_ids")
        if not isinstance(members, list) or component.get("member_count") != len(members):
            raise ValueError(f"Current component members are invalid: {component_id}")
        if component.get("split_assignment") not in pre_hf.SPLITS:
            raise ValueError(f"Current component split is invalid: {component_id}")
        for raw_id in members:
            base_fact_id = _required_string(
                raw_id, f"{component_id}.base_fact_ids"
            )
            if base_fact_id in component_by_fact:
                raise ValueError(f"Fact occurs in multiple current components: {base_fact_id}")
            if base_fact_id not in full_index:
                raise ValueError(f"Current component contains unknown fact: {base_fact_id}")
            if full_index[base_fact_id].get("leakage_component_id") != component_id:
                raise ValueError(f"Current component assignment is stale: {base_fact_id}")
            if full_index[base_fact_id].get("split_assignment") != component.get(
                "split_assignment"
            ):
                raise ValueError(f"Current component split is stale: {base_fact_id}")
            component_by_fact[base_fact_id] = component_id
    if set(component_by_fact) != set(full_index):
        raise ValueError("Current components do not exactly cover full_base_facts")
    return component_index, component_by_fact


def _load_frozen_constraints(
    *,
    split_manifest: Mapping[str, Any],
    split_manifest_path: Path,
    full_ids: Sequence[str],
    source_universe_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    binding = split_manifest.get("frozen_split_constraint")
    if binding is None:
        return {
            "binding": None,
            "path": None,
            "rows": [],
            "split_by_fact": {},
            "component_by_fact": {},
            "snapshot": None,
            "excluded_source_fact_count": 0,
        }
    path = _verify_bound_file(
        binding,
        owner_path=split_manifest_path,
        label="frozen split constraint",
    )
    rows = pre_hf.read_jsonl(path)
    if not isinstance(binding, dict) or binding.get("record_count") != len(rows):
        raise ValueError("Frozen split constraint record_count is stale")
    available = set(full_ids)
    source_available = set(source_universe_ids or full_ids)
    split_by_fact: Dict[str, str] = {}
    component_by_fact: Dict[str, str] = {}
    excluded_source_fact_count = 0
    for row in rows:
        base_fact_id = _required_string(
            row.get("base_fact_id"), "frozen split base_fact_id"
        )
        if base_fact_id not in source_available:
            raise ValueError(f"Frozen split fact is outside full_base_facts: {base_fact_id}")
        if base_fact_id not in available:
            excluded_source_fact_count += 1
            continue
        if base_fact_id in split_by_fact:
            raise ValueError(f"Duplicate frozen split fact: {base_fact_id}")
        split = row.get("split_assignment")
        if split not in pre_hf.SPLITS:
            raise ValueError(f"Frozen split assignment is invalid: {base_fact_id}")
        component_id = _required_string(
            row.get("leakage_component_id") or row.get("split_group_id"),
            f"frozen {base_fact_id}.leakage_component_id",
        )
        split_by_fact[base_fact_id] = str(split)
        component_by_fact[base_fact_id] = component_id
    return {
        "binding": copy.deepcopy(dict(binding)),
        "path": path,
        "rows": rows,
        "split_by_fact": split_by_fact,
        "component_by_fact": component_by_fact,
        "snapshot": _input_binding(path),
        "excluded_source_fact_count": excluded_source_fact_count,
    }


def _endpoint_graph_node(endpoint: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    provenance = endpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Semantic candidate endpoint provenance is invalid")
    role = provenance.get("input_role")
    if role == "current_full_base":
        base_fact_id = _required_string(
            provenance.get("base_fact_id"), "semantic current endpoint base_fact_id"
        )
        return f"current::{base_fact_id}", base_fact_id
    if role == "comparison_canonical":
        endpoint_id = _required_string(
            provenance.get("endpoint_id"), "semantic comparison endpoint_id"
        )
        return f"comparison::{endpoint_id}", None
    raise ValueError(f"Unsupported semantic endpoint role: {role!r}")


def build_reviewed_components(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    retained_ids: Sequence[str],
    rejected_ids: Sequence[str],
    current_components: Sequence[Mapping[str, Any]],
    semantic_candidates: Sequence[Mapping[str, Any]],
    semantic_decisions: Mapping[str, Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    """Rebuild on retained nodes, then add adjudicated positive edges.

    Pre-resolution components are lineage only.  They are deliberately not
    unioned wholesale: doing so could let an excluded fact remain an implicit
    donor or transitive bridge.  Exact normalized-answer groups and retained
    frozen groups are reconstructed from the retained cohort itself.
    """

    full_index = _index_unique(full_rows, "base_fact_id", "full_base_facts")
    component_index, current_component_by_fact = _validate_current_components(
        full_index=full_index, components=current_components
    )
    retained_set = set(retained_ids)
    rejected_set = set(rejected_ids)
    if retained_set.intersection(rejected_set):
        raise ValueError("Fact-review retained/rejected sets overlap")
    full_set = set(full_index)
    if retained_set != full_set and retained_set | rejected_set != full_set:
        raise ValueError("Fact-review dispositions do not cover full_base_facts")
    if not retained_set.issubset(full_set):
        raise ValueError("Retained facts are outside full_base_facts")

    candidate_index = _index_unique(
        semantic_candidates, "pair_id", "semantic candidates"
    )
    if set(candidate_index) != set(semantic_decisions):
        raise ValueError("Semantic decisions do not exactly cover candidates")

    graph_nodes = {f"current::{base_fact_id}" for base_fact_id in retained_set}
    endpoint_nodes: Dict[Tuple[str, str], Tuple[str, Optional[str]]] = {}
    for candidate in semantic_candidates:
        pair_id = str(candidate["pair_id"])
        for side in ("left", "right"):
            endpoint = candidate.get(side)
            if not isinstance(endpoint, dict):
                raise ValueError(f"Semantic candidate {pair_id} has invalid {side} endpoint")
            node = _endpoint_graph_node(endpoint)
            endpoint_nodes[(pair_id, side)] = node
            graph_nodes.add(node[0])
            if node[1] is not None:
                if node[1] in rejected_set:
                    raise ValueError(
                        "Semantic candidates must be regenerated after fact resolution; "
                        f"excluded fact remains an endpoint: {node[1]}"
                    )
                if node[1] not in retained_set:
                    raise ValueError(
                        f"Semantic candidate {pair_id} refers to unknown current fact: {node[1]}"
                    )

    union_find = pre_hf.UnionFind(graph_nodes)
    normalized_answer_groups: MutableMapping[str, List[str]] = defaultdict(list)
    for base_fact_id in sorted(retained_set):
        normalized = pre_hf.normalize_text(full_index[base_fact_id].get("answer_en"))
        if not normalized:
            raise ValueError(f"Retained fact has an empty normalized answer: {base_fact_id}")
        normalized_answer_groups[normalized].append(f"current::{base_fact_id}")
    for members in normalized_answer_groups.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    retained_frozen_groups: MutableMapping[str, List[str]] = defaultdict(list)
    for base_fact_id, frozen_component_id in frozen.get(
        "component_by_fact", {}
    ).items():
        if base_fact_id in retained_set:
            retained_frozen_groups[str(frozen_component_id)].append(
                f"current::{base_fact_id}"
            )
    for members in retained_frozen_groups.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    positive_pair_ids: List[str] = []
    positive_relationship_counts: Counter[str] = Counter()
    for pair_id in sorted(candidate_index):
        relationship = semantic_decisions[pair_id].get("relationship")
        if relationship in POSITIVE_RELATIONSHIPS:
            left_node = endpoint_nodes[(pair_id, "left")][0]
            right_node = endpoint_nodes[(pair_id, "right")][0]
            union_find.union(left_node, right_node)
            positive_pair_ids.append(pair_id)
            positive_relationship_counts[str(relationship)] += 1
        elif relationship != "distinct":
            raise ValueError(f"Unsupported semantic relationship: {pair_id}={relationship!r}")

    current_ids_by_root: MutableMapping[str, List[str]] = defaultdict(list)
    retained_ids_by_root: MutableMapping[str, List[str]] = defaultdict(list)
    comparison_ids_by_root: MutableMapping[str, List[str]] = defaultdict(list)
    for node in graph_nodes:
        root = union_find.find(node)
        if node.startswith("current::"):
            base_fact_id = node.split("::", 1)[1]
            current_ids_by_root[root].append(base_fact_id)
            if base_fact_id in retained_set:
                retained_ids_by_root[root].append(base_fact_id)
        else:
            comparison_ids_by_root[root].append(node.split("::", 1)[1])

    positive_pairs_by_root: MutableMapping[str, List[str]] = defaultdict(list)
    for pair_id in positive_pair_ids:
        root = union_find.find(endpoint_nodes[(pair_id, "left")][0])
        positive_pairs_by_root[root].append(pair_id)

    frozen_split_by_fact = dict(frozen.get("split_by_fact", {}))
    frozen_component_by_fact = dict(frozen.get("component_by_fact", {}))
    components: List[Dict[str, Any]] = []
    component_by_fact: Dict[str, str] = {}
    external_bridge_count = 0
    for root, retained_members_raw in sorted(
        retained_ids_by_root.items(), key=lambda item: min(item[1])
    ):
        retained_members = sorted(retained_members_raw)
        all_current_members = sorted(current_ids_by_root[root])
        comparison_endpoints = sorted(set(comparison_ids_by_root.get(root, [])))
        if comparison_endpoints:
            external_bridge_count += 1
        pinned = {
            frozen_split_by_fact[base_fact_id]
            for base_fact_id in retained_members
            if base_fact_id in frozen_split_by_fact
        }
        if len(pinned) > 1:
            raise ValueError(
                "Pinned split conflict after semantic closure: "
                f"members={retained_members[:5]}, splits={sorted(pinned)}"
            )
        source_component_ids = sorted(
            {current_component_by_fact[base_fact_id] for base_fact_id in retained_members}
        )
        source_split_groups = sorted(
            {
                str(group_id)
                for component_id in source_component_ids
                for group_id in component_index[component_id].get(
                    "source_split_group_ids", []
                )
                if str(group_id)
            }
        )
        frozen_semantic_components = sorted(
            {
                str(component_id)
                for base_fact_id in all_current_members
                for component_id in (
                    [frozen_component_by_fact[base_fact_id]]
                    if base_fact_id in frozen_component_by_fact
                    else (
                        component_index[current_component_by_fact[base_fact_id]].get(
                            "frozen_semantic_component_ids", []
                        )
                        or []
                    )
                )
                if str(component_id)
            }
        )
        relation_counts = Counter(
            str(full_index[base_fact_id]["relation_partition_id"])
            for base_fact_id in retained_members
        )
        source_counts = Counter(
            str(full_index[base_fact_id].get("source_dataset") or "unknown")
            for base_fact_id in retained_members
        )
        answer_counts = Counter(
            str(full_index[base_fact_id]["answer_type_bucket"])
            for base_fact_id in retained_members
        )
        pair_ids = sorted(positive_pairs_by_root.get(root, []))
        relationship_counts = Counter(
            str(semantic_decisions[pair_id]["relationship"]) for pair_id in pair_ids
        )
        component_id = pre_hf.stable_id(
            "full_leakage_component", retained_members
        )
        semantics = ["preserved_pre_review_leakage_component"]
        if frozen_semantic_components:
            semantics.append("preserved_frozen_semantic_component")
        if pair_ids:
            semantics.append("adjudicated_bounded_semantic_positive_edge")
        component = {
            "schema_version": pre_hf.COMPONENT_SCHEMA_VERSION,
            "leakage_component_id": component_id,
            "member_count": len(retained_members),
            "base_fact_ids": retained_members,
            "source_leakage_component_ids": source_component_ids,
            "source_split_group_ids": source_split_groups,
            "frozen_semantic_component_ids": frozen_semantic_components,
            "relation_partition_counts": dict(sorted(relation_counts.items())),
            "source_dataset_counts": dict(sorted(source_counts.items())),
            "answer_type_bucket_counts": dict(sorted(answer_counts.items())),
            "pinned_split_assignment": next(iter(pinned)) if pinned else None,
            "pinned_fact_count": sum(
                base_fact_id in frozen_split_by_fact
                for base_fact_id in retained_members
            ),
            "historical_bridge_pinned_fact_count": sum(
                base_fact_id in frozen_split_by_fact
                for base_fact_id in retained_members
            ),
            "excluded_bridge_base_fact_ids": [],
            "comparison_bridge_endpoint_ids": comparison_endpoints,
            "bounded_semantic_positive_pair_ids": pair_ids,
            "bounded_semantic_positive_relationship_counts": dict(
                sorted(relationship_counts.items())
            ),
            "component_semantics": semantics,
        }
        components.append(component)
        for base_fact_id in retained_members:
            component_by_fact[base_fact_id] = component_id

    if set(component_by_fact) != retained_set:
        raise ValueError("Rebuilt components do not exactly cover retained facts")
    audit = {
        "input_component_count": len(current_components),
        "rebuilt_component_count": len(components),
        "positive_semantic_edge_count": len(positive_pair_ids),
        "positive_relationship_counts": dict(
            sorted(positive_relationship_counts.items())
        ),
        "positive_edge_pair_ids_sha256": pre_hf.sha256_value(positive_pair_ids),
        "excluded_fact_bridge_count": 0,
        "excluded_fact_endpoint_count": 0,
        "pre_resolution_components_used_as_union_edges": False,
        "retained_normalized_answer_groups_recomputed": True,
        "comparison_bridge_component_count": external_bridge_count,
        "bounded_candidate_recall_guaranteed": False,
    }
    return components, component_by_fact, audit


def _component_size_summary(components: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    sizes = sorted(int(component["member_count"]) for component in components)
    if not sizes:
        raise ValueError("No retained leakage components")
    return {
        "maximum": max(sizes),
        "median": sizes[len(sizes) // 2],
        "p95_nearest_rank": sizes[max(0, math.ceil(0.95 * len(sizes)) - 1)],
        "singleton_component_count": sum(size == 1 for size in sizes),
        "multi_fact_component_count": sum(size > 1 for size in sizes),
        "at_least_10_fact_component_count": sum(size >= 10 for size in sizes),
    }


def _reviewed_base_rows(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    retained_ids: Sequence[str],
    staging_index: Mapping[str, Mapping[str, Any]],
    component_by_fact: Mapping[str, str],
    split_by_fact: Mapping[str, str],
    fact_resolution: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    retained = set(retained_ids)
    output: List[Dict[str, Any]] = []
    for source in sorted(full_rows, key=lambda row: str(row["base_fact_id"])):
        base_fact_id = str(source["base_fact_id"])
        if base_fact_id not in retained:
            continue
        staging = staging_index[base_fact_id]
        review_notes = staging.get("notes")
        structured_notes = None
        if isinstance(review_notes, str):
            try:
                parsed_notes = json.loads(review_notes)
            except json.JSONDecodeError:
                parsed_notes = None
            if isinstance(parsed_notes, dict):
                structured_notes = parsed_notes
        row = copy.deepcopy(dict(source))
        if fact_resolution is None:
            row["answer_aliases_en"] = list(
                staging.get("accepted_answer_aliases_en") or []
            )
        row["leakage_component_id"] = component_by_fact[base_fact_id]
        row["split_assignment"] = split_by_fact[base_fact_id]
        row["split_policy_version"] = pre_hf.SPLIT_POLICY_VERSION
        row["split_status"] = "provisional_not_frozen"
        row["canonical_status"] = "pending_review"
        row["human_gold"] = False
        review_evidence = {
            "review_outcome": (
                "accept"
                if fact_resolution is None
                else fact_resolution["ledger_index"][base_fact_id][
                    "source_review_outcome"
                ]
            ),
            "staging_status": staging["staging_status"],
            "review_provenance": copy.deepcopy(staging["review_provenance"]),
            "review_item_sha256": staging["review_item_sha256"],
            "review_completion": copy.deepcopy(staging["review_completion"]),
            "member_reviews": copy.deepcopy(staging.get("member_reviews") or []),
            "alias_reviews": copy.deepcopy(staging.get("alias_reviews") or []),
            "source_choice_distractor_reviews": copy.deepcopy(
                staging.get("distractor_reviews") or []
            ),
            "review_notes": review_notes,
            "structured_review_notes": copy.deepcopy(structured_notes),
            "proxy_evidence_only": True,
            "human_gold": False,
        }
        if fact_resolution is not None:
            ledger = fact_resolution["ledger_index"][base_fact_id]
            if ledger.get("final_disposition") not in {
                "retain_accepted",
                "retain_revised",
            }:
                raise ValueError(
                    f"Resolved retained fact has non-retained ledger state: {base_fact_id}"
                )
            review_evidence.update(
                {
                    "resolution_successor_universe_id": fact_resolution[
                        "successor_id"
                    ],
                    "final_disposition": ledger["final_disposition"],
                    "fact_resolution_ledger_row_sha256": pre_hf.sha256_value(
                        ledger
                    ),
                    "fact_revision_lineage": copy.deepcopy(
                        fact_resolution["lineage_index"].get(base_fact_id)
                    ),
                    "revision_applied": ledger["final_disposition"]
                    == "retain_revised",
                }
            )
        row["fact_review"] = review_evidence
        row["semantic_component_status"] = (
            "bounded_candidate_adjudications_applied_recall_not_guaranteed"
        )
        row["distractor_status"] = (
            "regenerated_after_review_and_bounded_semantic_closure_pending_review"
        )
        row["translation_status"] = "pending_translation_or_rebind_to_final_candidates"
        row["hf_model_execution_status"] = "not_run"
        row["hf_tokenizer_execution_status"] = "not_run"
        output.append(row)
    return output


def _exclusion_rows(
    rejected_ids: Sequence[str], staging_index: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    for base_fact_id in sorted(rejected_ids):
        staging = staging_index[base_fact_id]
        rows.append(
            {
                "schema_version": EXCLUSION_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "disposition": "cohort_exclude",
                "source_review_outcome": "reject",
                "source_staging_status": staging["staging_status"],
                "review_item_sha256": staging["review_item_sha256"],
                "review_provenance": copy.deepcopy(staging["review_provenance"]),
                "notes": staging.get("notes"),
                "materializer_asserts_factual_falsehood": False,
                "human_gold": False,
            }
        )
    return rows


def _resolution_exclusion_rows(
    fact_resolution: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base_fact_id in fact_resolution["rejected_ids"]:
        source = fact_resolution["source_exclusion_index"][base_fact_id]
        ledger = fact_resolution["ledger_index"][base_fact_id]
        if source.get("factual_falsehood_asserted") is not False:
            raise ValueError(
                f"Resolution exclusion asserts factual falsehood: {base_fact_id}"
            )
        rows.append(
            {
                "schema_version": EXCLUSION_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "disposition": "cohort_exclude",
                "source_review_outcome": source.get("source_review_outcome"),
                "source_staging_status": None,
                "review_item_sha256": None,
                "review_provenance": copy.deepcopy(
                    source.get("resolution_provenance")
                ),
                "notes": source.get("cohort_exclusion_reason"),
                "source_fact_resolution_exclusion_row_sha256": pre_hf.sha256_value(
                    source
                ),
                "source_fact_resolution_ledger_row_sha256": pre_hf.sha256_value(
                    ledger
                ),
                "resolution_successor_universe_id": fact_resolution[
                    "successor_id"
                ],
                "materializer_asserts_factual_falsehood": False,
                "human_gold": False,
            }
        )
    return rows


def _candidate_shortfalls(
    *,
    base_fact_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    distractor_ids: Mapping[str, Sequence[str]],
    neutral_ids: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base_fact_id in sorted(base_fact_ids):
        distractor_count = len(distractor_ids.get(base_fact_id, []))
        neutral_count = len(neutral_ids.get(base_fact_id, []))
        if distractor_count == 2 and neutral_count == 2:
            continue
        reasons = []
        if distractor_count < 2:
            reasons.append("fewer_than_two_same_split_relation_partition_distractors")
        if neutral_count < 2:
            reasons.append(
                "fewer_than_two_same_split_different_relation_neutral_references"
            )
        row = rows_by_id[base_fact_id]
        rows.append(
            {
                "schema_version": pre_hf.SHORTFALL_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "split_assignment": row["split_assignment"],
                "relation_partition_id": row["relation_partition_id"],
                "distractor_candidate_count": distractor_count,
                "neutral_reference_candidate_count": neutral_count,
                "shortfall_reasons": reasons,
                "fact_excluded_from_full_pool": False,
                "hf_behavior_authorized": False,
            }
        )
    return rows


def _audit_outputs(
    *,
    input_full_ids: Sequence[str],
    reviewed_rows: Sequence[Mapping[str, Any]],
    rejected_ids: Sequence[str],
    components: Sequence[Mapping[str, Any]],
    semantic_candidates: Sequence[Mapping[str, Any]],
    semantic_decisions: Mapping[str, Mapping[str, Any]],
    distractors: Sequence[Mapping[str, Any]],
    neutrals: Sequence[Mapping[str, Any]],
    shortfalls: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> Dict[str, Any]:
    reviewed_index = _index_unique(reviewed_rows, "base_fact_id", "reviewed facts")
    retained_ids = set(reviewed_index)
    if retained_ids | set(rejected_ids) != set(input_full_ids):
        raise ValueError("Reviewed and excluded facts do not cover the input universe")
    if retained_ids.intersection(rejected_ids):
        raise ValueError("Reviewed and excluded fact universes overlap")

    component_by_fact: Dict[str, str] = {}
    component_split: Dict[str, str] = {}
    for component in components:
        component_id = str(component["leakage_component_id"])
        if component_id in component_split:
            raise ValueError(f"Duplicate rebuilt component: {component_id}")
        component_split[component_id] = str(component["split_assignment"])
        for base_fact_id in component["base_fact_ids"]:
            if base_fact_id in component_by_fact:
                raise ValueError(f"Fact occurs in multiple rebuilt components: {base_fact_id}")
            component_by_fact[str(base_fact_id)] = component_id
    if set(component_by_fact) != retained_ids:
        raise ValueError("Rebuilt components do not exactly cover reviewed facts")
    for base_fact_id, row in reviewed_index.items():
        component_id = component_by_fact[base_fact_id]
        if row["leakage_component_id"] != component_id:
            raise ValueError(f"Reviewed fact component is stale: {base_fact_id}")
        if row["split_assignment"] != component_split[component_id]:
            raise ValueError(f"Reviewed fact split is stale: {base_fact_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"Reviewed fact was promoted to human gold: {base_fact_id}")
        if row.get("hf_model_execution_status") != "not_run":
            raise ValueError(f"Reviewed fact contains HF execution: {base_fact_id}")
        if row.get("hf_tokenizer_execution_status") != "not_run":
            raise ValueError(f"Reviewed fact contains tokenizer execution: {base_fact_id}")

    candidate_index = _index_unique(
        semantic_candidates, "pair_id", "semantic candidates"
    )
    positive_current_pairs = 0
    excluded_semantic_endpoint_count = 0
    rejected_set = set(rejected_ids)
    for pair_id, decision in semantic_decisions.items():
        candidate = candidate_index[pair_id]
        current_ids = []
        for side in ("left", "right"):
            provenance = candidate[side]["provenance"]
            if provenance["input_role"] == "current_full_base":
                endpoint_id = str(provenance["base_fact_id"])
                current_ids.append(endpoint_id)
                if endpoint_id in rejected_set or endpoint_id not in retained_ids:
                    excluded_semantic_endpoint_count += 1
        if excluded_semantic_endpoint_count:
            raise ValueError(
                "Semantic candidates contain excluded or unknown current endpoints"
            )
        if decision["relationship"] not in POSITIVE_RELATIONSHIPS:
            continue
        retained_current = [value for value in current_ids if value in retained_ids]
        if len(retained_current) == 2:
            positive_current_pairs += 1
            if component_by_fact[retained_current[0]] != component_by_fact[
                retained_current[1]
            ]:
                raise ValueError(f"Positive semantic edge crosses components: {pair_id}")

    distractor_by_fact: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in distractors:
        base_fact_id = str(row["base_fact_id"])
        source_id = str(row["source_base_fact_id"])
        if base_fact_id not in retained_ids or source_id not in retained_ids:
            raise ValueError("Distractor references an excluded or unknown fact")
        if reviewed_index[base_fact_id]["split_assignment"] != reviewed_index[source_id][
            "split_assignment"
        ]:
            raise ValueError("Distractor crosses rebuilt splits")
        if component_by_fact[base_fact_id] == component_by_fact[source_id]:
            raise ValueError("Distractor crosses within one rebuilt leakage component")
        if reviewed_index[base_fact_id]["relation_partition_id"] != reviewed_index[
            source_id
        ]["relation_partition_id"]:
            raise ValueError("Distractor crosses relation partitions")
        if row.get("verified") is not False:
            raise ValueError("Regenerated distractor was promoted without review")
        distractor_by_fact[base_fact_id].append(row)

    neutral_by_fact: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in neutrals:
        base_fact_id = str(row["base_fact_id"])
        source_id = str(row["source_base_fact_id"])
        if base_fact_id not in retained_ids or source_id not in retained_ids:
            raise ValueError("Neutral candidate references an excluded or unknown fact")
        if reviewed_index[base_fact_id]["split_assignment"] != reviewed_index[source_id][
            "split_assignment"
        ]:
            raise ValueError("Neutral candidate crosses rebuilt splits")
        if component_by_fact[base_fact_id] == component_by_fact[source_id]:
            raise ValueError("Neutral candidate is in the target leakage component")
        if reviewed_index[base_fact_id]["relation_partition_id"] == reviewed_index[
            source_id
        ]["relation_partition_id"]:
            raise ValueError("Neutral candidate shares the target relation partition")
        if row.get("verified_unrelated") is not False:
            raise ValueError("Regenerated Neutral candidate was promoted without review")
        neutral_by_fact[base_fact_id].append(row)
    if any(len(values) > 2 for values in distractor_by_fact.values()):
        raise ValueError("A fact has more than two distractor candidates")
    if any(len(values) > 2 for values in neutral_by_fact.values()):
        raise ValueError("A fact has more than two Neutral candidates")

    expected_shortfalls = {
        base_fact_id
        for base_fact_id in retained_ids
        if len(distractor_by_fact[base_fact_id]) < 2
        or len(neutral_by_fact[base_fact_id]) < 2
    }
    supplied_shortfalls = [str(row["base_fact_id"]) for row in shortfalls]
    if len(supplied_shortfalls) != len(set(supplied_shortfalls)) or set(
        supplied_shortfalls
    ) != expected_shortfalls:
        raise ValueError("Candidate shortfall ledger is not exhaustive")

    frozen_mismatches = []
    for base_fact_id, expected_split in frozen.get("split_by_fact", {}).items():
        if base_fact_id in retained_ids and reviewed_index[base_fact_id][
            "split_assignment"
        ] != expected_split:
            frozen_mismatches.append(base_fact_id)
    if frozen_mismatches:
        raise ValueError(f"Frozen split assignments changed: {frozen_mismatches[:5]}")

    return {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "all_checks_passed": True,
        "checks": {
            "fact_review_dispositions_cover_input_universe": True,
            "unresolved_fact_review_outcomes_absent": True,
            "semantic_candidate_adjudications_exactly_complete": True,
            "semantic_candidates_exclude_cohort_exclusions": True,
            "positive_current_edges_share_rebuilt_component": True,
            "rebuilt_components_cover_retained_facts_once": True,
            "component_splits_are_atomic": True,
            "frozen_split_constraints_preserved_for_retained_facts": True,
            "distractors_are_same_split_same_relation_different_component": True,
            "neutral_candidates_are_same_split_different_relation_different_component": True,
            "candidate_shortfalls_are_exhaustive": True,
            "human_gold_not_claimed": True,
            "hf_model_or_tokenizer_not_executed": True,
        },
        "counts": {
            "input_base_facts": len(input_full_ids),
            "retained_base_facts": len(retained_ids),
            "rejected_base_facts": len(rejected_ids),
            "rebuilt_components": len(components),
            "semantic_candidate_pairs": len(semantic_candidates),
            "positive_current_current_pairs": positive_current_pairs,
            "distractor_candidates": len(distractors),
            "neutral_candidates": len(neutrals),
            "candidate_shortfalls": len(shortfalls),
        },
        "violations": {
            "unresolved_fact_reviews": 0,
            "missing_semantic_adjudications": 0,
            "excluded_semantic_endpoints": 0,
            "positive_edge_component": 0,
            "component_split": 0,
            "frozen_split": 0,
            "cross_split_distractor": 0,
            "same_component_distractor": 0,
            "cross_split_neutral": 0,
            "same_component_neutral": 0,
        },
        "limitations": {
            "semantic_candidate_generation_is_bounded": True,
            "all_record_pairs_enumerated": False,
            "semantic_near_duplicate_recall_guaranteed": False,
            "semantic_closure_complete_for_full_pool": False,
        },
    }


def materialize(
    *,
    full_base_facts_path: Path,
    fact_review_apply_manifest_path: Optional[Path],
    semantic_candidate_manifest_path: Path,
    semantic_adjudications_path: Optional[Path],
    semantic_run_manifest_path: Optional[Path],
    semantic_composite_manifest_path: Optional[Path] = None,
    output_dir: Path,
    fact_resolution_manifest_path: Optional[Path] = None,
    expected_input_count: int = 8969,
    split_seed: str = pre_hf.DEFAULT_SPLIT_SEED,
) -> Dict[str, Any]:
    """Validate reviewed evidence and atomically publish a provisional rebuild."""

    expected_input_count = _required_positive_int(
        expected_input_count, "expected_input_count"
    )
    split_seed = _required_string(split_seed, "split_seed")
    full_base_facts_path = Path(full_base_facts_path).resolve()
    if (fact_review_apply_manifest_path is None) == (
        fact_resolution_manifest_path is None
    ):
        raise ValueError(
            "Supply exactly one of fact_review_apply_manifest_path or "
            "fact_resolution_manifest_path"
        )
    fact_review_apply_manifest_path = (
        Path(fact_review_apply_manifest_path).resolve()
        if fact_review_apply_manifest_path is not None
        else None
    )
    fact_resolution_manifest_path = (
        Path(fact_resolution_manifest_path).resolve()
        if fact_resolution_manifest_path is not None
        else None
    )
    direct_semantic_supplied = (
        semantic_adjudications_path is not None
        or semantic_run_manifest_path is not None
    )
    if semantic_composite_manifest_path is None:
        if (
            semantic_adjudications_path is None
            or semantic_run_manifest_path is None
        ):
            raise ValueError(
                "Supply both semantic_adjudications_path and "
                "semantic_run_manifest_path, or semantic_composite_manifest_path"
            )
    elif direct_semantic_supplied:
        raise ValueError(
            "Supply direct semantic run XOR semantic composite authority"
        )
    semantic_candidate_manifest_path = Path(semantic_candidate_manifest_path).resolve()
    semantic_adjudications_path = (
        Path(semantic_adjudications_path).resolve()
        if semantic_adjudications_path is not None
        else None
    )
    semantic_run_manifest_path = (
        Path(semantic_run_manifest_path).resolve()
        if semantic_run_manifest_path is not None
        else None
    )
    semantic_composite_manifest_path = (
        Path(semantic_composite_manifest_path).resolve()
        if semantic_composite_manifest_path is not None
        else None
    )
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    if fact_resolution_manifest_path is not None:
        fact_review = _load_fact_resolution_evidence(
            manifest_path=fact_resolution_manifest_path,
            resolved_full_base_facts_path=full_base_facts_path,
        )
        if len(fact_review["source_ids"]) != expected_input_count:
            raise ValueError(
                "Fact-resolution source count differs from expected_input_count"
            )
        full_rows, full_index, full_binding = _validate_full_base_rows(
            full_base_facts_path, len(fact_review["retained_ids"])
        )
        if [str(row["base_fact_id"]) for row in full_rows] != fact_review[
            "retained_ids"
        ]:
            raise ValueError(
                "Resolved full_base_facts is not the complete retained cohort"
            )
    else:
        full_rows, full_index, full_binding = _validate_full_base_rows(
            full_base_facts_path, expected_input_count
        )
        fact_review = _replay_fact_review_apply(
            apply_manifest_path=fact_review_apply_manifest_path,
            full_rows=full_rows,
        )
        fact_review["mode"] = "legacy_fact_review_apply"
        fact_review["source_ids"] = [
            str(row["base_fact_id"]) for row in full_rows
        ]
        fact_review["revised_ids"] = []
    semantic = _load_semantic_review_evidence(
        candidate_manifest_path=semantic_candidate_manifest_path,
        adjudications_path=semantic_adjudications_path,
        run_manifest_path=semantic_run_manifest_path,
        composite_manifest_path=semantic_composite_manifest_path,
        full_base_facts_path=full_base_facts_path,
        fact_resolution=(
            fact_review
            if fact_review["mode"] == "fact_resolution_successor_cohort"
            else None
        ),
    )
    if semantic["manifest"].get("base_fact_count") != len(full_rows):
        raise ValueError("Semantic candidate manifest base_fact_count is stale")

    component_index, _ = _validate_current_components(
        full_index=full_index, components=semantic["components"]
    )
    split_manifest = semantic["split_manifest"]
    if split_manifest.get("split_policy_version") != next(
        iter(full_rows)
    ).get("split_policy_version"):
        raise ValueError("Current split manifest policy differs from full_base_facts")
    frozen = _load_frozen_constraints(
        split_manifest=split_manifest,
        split_manifest_path=semantic["split_path"],
        full_ids=list(full_index),
        source_universe_ids=fact_review["source_ids"],
    )

    components, component_by_fact, component_audit = build_reviewed_components(
        full_rows=full_rows,
        retained_ids=fact_review["retained_ids"],
        rejected_ids=fact_review["rejected_ids"],
        current_components=list(component_index.values()),
        semantic_candidates=semantic["candidates"],
        semantic_decisions=semantic["decisions"],
        frozen=frozen,
    )
    component_assignments, split_audit = pre_hf.assign_component_splits(
        components, split_seed
    )
    split_by_fact: Dict[str, str] = {}
    for component in components:
        component_id = str(component["leakage_component_id"])
        split = component_assignments[component_id]
        for base_fact_id in component["base_fact_ids"]:
            split_by_fact[str(base_fact_id)] = split
    if set(split_by_fact) != set(fact_review["retained_ids"]):
        raise ValueError("Recomputed split does not exactly cover retained facts")

    reviewed_rows = _reviewed_base_rows(
        full_rows=full_rows,
        retained_ids=fact_review["retained_ids"],
        staging_index=fact_review["staging_index"],
        component_by_fact=component_by_fact,
        split_by_fact=split_by_fact,
        fact_resolution=(
            fact_review
            if fact_review["mode"] == "fact_resolution_successor_cohort"
            else None
        ),
    )
    reviewed_index = _index_unique(reviewed_rows, "base_fact_id", "reviewed facts")
    relation_by_fact = {
        base_fact_id: {
            "relation_partition_id": row["relation_partition_id"],
            "answer_type_bucket": row["answer_type_bucket"],
        }
        for base_fact_id, row in reviewed_index.items()
    }
    distractors, neutrals, selection = pre_hf.select_static_candidates(
        reviewed_rows,
        relation_by_fact,
        component_by_fact,
        split_by_fact,
    )
    distractor_ids = selection.pop("distractor_ids")
    neutral_ids = selection.pop("neutral_ids")
    shortfalls = _candidate_shortfalls(
        base_fact_ids=list(reviewed_index),
        rows_by_id=reviewed_index,
        distractor_ids=distractor_ids,
        neutral_ids=neutral_ids,
    )
    exclusions = (
        _resolution_exclusion_rows(fact_review)
        if fact_review["mode"] == "fact_resolution_successor_cohort"
        else _exclusion_rows(
            fact_review["rejected_ids"], fact_review["staging_index"]
        )
    )

    frozen_retained_count = sum(
        base_fact_id in reviewed_index for base_fact_id in frozen["split_by_fact"]
    )
    frozen_excluded_count = len(frozen["split_by_fact"]) - frozen_retained_count
    frozen_excluded_count += int(frozen.get("excluded_source_fact_count", 0))
    split_manifest_out = {
        "schema_version": pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "split_policy_version": pre_hf.SPLIT_POLICY_VERSION,
        "split_seed": split_seed,
        "split_ratios": pre_hf.SPLIT_RATIOS,
        "split_status": "provisional_not_frozen",
        "formal_split_freeze_performed": False,
        "input_base_fact_count": len(fact_review["source_ids"]),
        "base_fact_count": len(reviewed_rows),
        "fact_review_reject_excluded_count": sum(
            row.get("source_review_outcome") == "reject" for row in exclusions
        ),
        "fact_resolution_cohort_excluded_count": len(exclusions),
        "fact_resolution_revision_count": len(fact_review["revised_ids"]),
        "leakage_component_count": len(components),
        "leakage_grouping_sources": [
            "preserved_pre_review_leakage_components",
            "adjudicated_same_fact_positive_edges",
            "adjudicated_same_leakage_component_positive_edges",
            "comparison_canonical_bridge_nodes",
        ],
        "bounded_semantic_candidate_adjudication_complete": True,
        "semantic_paraphrase_closure_complete_for_full_pool": False,
        "semantic_near_duplicate_recall_guaranteed": False,
        "historical_exposure_complete_for_full_pool": False,
        "frozen_split_constraint": frozen["binding"],
        "retained_frozen_split_fact_count": frozen_retained_count,
        "excluded_frozen_split_fact_count": frozen_excluded_count,
        "frozen_split_mismatch_count": 0,
        "cross_component_split_violation_count": 0,
        "component_size_summary": _component_size_summary(components),
        "semantic_component_rebuild": component_audit,
        **split_audit,
    }

    integrity = _audit_outputs(
        input_full_ids=fact_review["source_ids"],
        reviewed_rows=reviewed_rows,
        rejected_ids=fact_review["rejected_ids"],
        components=components,
        semantic_candidates=semantic["candidates"],
        semantic_decisions=semantic["decisions"],
        distractors=distractors,
        neutrals=neutrals,
        shortfalls=shortfalls,
        frozen=frozen,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        paths = {
            "full_base_facts": staged_dir / "full_base_facts.jsonl",
            "cohort_exclusions": staged_dir / "cohort_exclusions.jsonl",
            "leakage_components": staged_dir / "leakage_components.jsonl",
            "split_manifest": staged_dir / "split_manifest.json",
            "distractor_candidates": staged_dir / "distractor_candidates.jsonl",
            "neutral_reference_candidates": staged_dir
            / "neutral_reference_candidates.jsonl",
            "candidate_shortfalls": staged_dir / "candidate_shortfalls.jsonl",
            "integrity_audit": staged_dir / "integrity_audit.json",
            "summary": staged_dir / "summary.json",
            "manifest": staged_dir / "postreview_rebuild_manifest.json",
        }
        pre_hf.write_jsonl(paths["full_base_facts"], reviewed_rows)
        pre_hf.write_jsonl(paths["cohort_exclusions"], exclusions)
        pre_hf.write_jsonl(paths["leakage_components"], components)
        pre_hf.write_json(paths["split_manifest"], split_manifest_out)
        pre_hf.write_jsonl(paths["distractor_candidates"], distractors)
        pre_hf.write_jsonl(paths["neutral_reference_candidates"], neutrals)
        pre_hf.write_jsonl(paths["candidate_shortfalls"], shortfalls)
        pre_hf.write_json(paths["integrity_audit"], integrity)

        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
            "counts": {
                "input_base_facts": len(fact_review["source_ids"]),
                "retained_base_facts": len(reviewed_rows),
                "rejected_base_facts": len(exclusions),
                "revised_base_facts": len(fact_review["revised_ids"]),
                "leakage_components": len(components),
                "semantic_candidate_pairs": len(semantic["candidates"]),
                "semantic_positive_edges": component_audit[
                    "positive_semantic_edge_count"
                ],
                "distractor_candidates": len(distractors),
                "neutral_reference_candidates": len(neutrals),
                "candidate_shortfalls": len(shortfalls),
            },
            "fact_review_outcome_counts": fact_review["outcome_counts"],
            "split": {
                "policy_version": pre_hf.SPLIT_POLICY_VERSION,
                "base_fact_counts": split_audit["actual_base_fact_counts"],
                "component_counts": split_audit["component_counts"],
                "formal_split_freeze_performed": False,
            },
            "static_candidates": selection,
            "limitations": integrity["limitations"],
            "execution": {
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
                "hidden_state_collection_count": 0,
                "intervention_count": 0,
            },
        }
        pre_hf.write_json(paths["summary"], summary)

        output_artifacts = {
            "full_base_facts": _output_binding(
                paths["full_base_facts"],
                pre_hf.FULL_FACT_SCHEMA_VERSION,
                len(reviewed_rows),
            ),
            "cohort_exclusions": _output_binding(
                paths["cohort_exclusions"], EXCLUSION_SCHEMA_VERSION, len(exclusions)
            ),
            "leakage_components": _output_binding(
                paths["leakage_components"],
                pre_hf.COMPONENT_SCHEMA_VERSION,
                len(components),
            ),
            "split_manifest": _output_binding(
                paths["split_manifest"], pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION
            ),
            "distractor_candidates": _output_binding(
                paths["distractor_candidates"],
                pre_hf.DISTRACTOR_SCHEMA_VERSION,
                len(distractors),
            ),
            "neutral_reference_candidates": _output_binding(
                paths["neutral_reference_candidates"],
                pre_hf.NEUTRAL_SCHEMA_VERSION,
                len(neutrals),
            ),
            "candidate_shortfalls": _output_binding(
                paths["candidate_shortfalls"],
                pre_hf.SHORTFALL_SCHEMA_VERSION,
                len(shortfalls),
            ),
            "integrity_audit": _output_binding(
                paths["integrity_audit"], INTEGRITY_SCHEMA_VERSION
            ),
            "summary": _output_binding(paths["summary"], SUMMARY_SCHEMA_VERSION),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
            "inputs": {
                "full_base_facts": full_binding,
                "fact_review": fact_review["bindings"],
                "fact_resolution": (
                    fact_review["bindings"]
                    if fact_review["mode"]
                    == "fact_resolution_successor_cohort"
                    else None
                ),
                "semantic_review": semantic["bindings"],
                "frozen_split_constraint": frozen["binding"],
            },
            "outputs": output_artifacts,
            "counts": summary["counts"],
            "fact_review_contract": {
                "mode": fact_review["mode"],
                "scope_exactly_matches_input_universe": True,
                "staging_deterministically_replayed": True,
                "selection_is_complete_retained_cohort": True,
                "source_universe_base_fact_count": len(fact_review["source_ids"]),
                "accepted_facts_retained": len(reviewed_rows),
                "revised_facts_retained_after_independent_rereview": len(
                    fact_review["revised_ids"]
                ),
                "facts_cohort_excluded": len(exclusions),
                "revise_defer_missing_count": 0,
                "outcome_policy": {
                    "accept": "retain_with_review_evidence",
                    "reject": (
                        "retain_only_if_eligible_revision_override_passed_independent_"
                        "rereview_else_explicit_cohort_exclude"
                    ),
                    "revise": "retain_only_after_applied_revision_and_independent_rereview",
                    "defer": "explicit_cohort_exclude_after_resolution",
                    "missing": "fail_closed_until_resolved",
                },
                "proxy_review_only": True,
                "human_gold": False,
            },
            "semantic_review_contract": {
                "authority_mode": semantic["authority_mode"],
                "candidate_adjudication_complete_for_bounded_set": True,
                "runtime_identity_evidence_bound": True,
                "semantic_run_contract_sha256": semantic[
                    "run_contract_sha256"
                ],
                "route_identity_set_sha256": semantic[
                    "route_identity_set_sha256"
                ],
                "source_run_contract_sha256": semantic["bindings"][
                    "source_run_contract_sha256"
                ],
                "source_route_identity_set_sha256": semantic["bindings"][
                    "source_route_identity_set_sha256"
                ],
                "positive_relationships_merged": sorted(POSITIVE_RELATIONSHIPS),
                "same_fact_edges_remove_records": False,
                "excluded_facts_are_semantic_endpoints": False,
                "excluded_facts_are_component_bridges": False,
                "pre_resolution_components_used_as_union_edges": False,
                "candidate_generation_is_bounded": True,
                "all_record_pairs_enumerated": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "semantic_closure_complete_for_full_pool": False,
                "reviewer_evidence_is_human_gold": False,
            },
            "policy_reuse": {
                "source_tool": str(
                    Path(pre_hf.__file__).resolve().relative_to(PROJECT_ROOT)
                ),
                "source_tool_version": pre_hf.TOOL_VERSION,
                "split_policy_version": pre_hf.SPLIT_POLICY_VERSION,
                "split_seed": split_seed,
                "split_ratios": pre_hf.SPLIT_RATIOS,
                "distractor_policy_version": pre_hf.DISTRACTOR_POLICY_VERSION,
                "neutral_policy_version": pre_hf.NEUTRAL_POLICY_VERSION,
                "split_function_reused_directly": True,
                "candidate_function_reused_directly": True,
            },
            "lineage_digests": {
                "ordered_input_base_fact_ids_sha256": pre_hf.sha256_value(
                    fact_review["source_ids"]
                ),
                "ordered_retained_base_fact_ids_sha256": pre_hf.sha256_value(
                    [row["base_fact_id"] for row in reviewed_rows]
                ),
                "ordered_rejected_base_fact_ids_sha256": pre_hf.sha256_value(
                    [row["base_fact_id"] for row in exclusions]
                ),
                "ordered_revised_base_fact_ids_sha256": pre_hf.sha256_value(
                    fact_review["revised_ids"]
                ),
                "ordered_component_ids_sha256": pre_hf.sha256_value(
                    [row["leakage_component_id"] for row in components]
                ),
                "ordered_component_assignments_sha256": pre_hf.sha256_value(
                    [
                        [row["base_fact_id"], row["leakage_component_id"]]
                        for row in reviewed_rows
                    ]
                ),
                "ordered_split_assignments_sha256": pre_hf.sha256_value(
                    [
                        [row["base_fact_id"], row["split_assignment"]]
                        for row in reviewed_rows
                    ]
                ),
                "ordered_distractor_ids_sha256": pre_hf.sha256_value(
                    [row["distractor_id"] for row in distractors]
                ),
                "ordered_neutral_candidate_ids_sha256": pre_hf.sha256_value(
                    [row["neutral_candidate_id"] for row in neutrals]
                ),
            },
            "review_invalidation": {
                "pre_resolution_semantic_candidates_valid": False,
                "pre_resolution_semantic_adjudications_valid": False,
                "pre_resolution_translation_reviews_valid": False,
                "pre_resolution_distractor_reviews_valid": False,
                "pre_resolution_neutral_reviews_valid": False,
                "pre_postreview_split_translation_reviews_valid": False,
                "pre_postreview_distractor_reviews_valid": False,
                "pre_postreview_neutral_reviews_valid": False,
                "current_semantic_adjudications_applied_only_via_exact_manifest_binding": True,
                "new_distractor_review_required": True,
                "new_neutral_review_required": True,
                "new_translation_and_translation_review_required": True,
            },
            "integrity": {
                "all_checks_passed": integrity["all_checks_passed"],
                "integrity_audit_sha256": output_artifacts["integrity_audit"][
                    "sha256"
                ],
            },
            "safety_contract": {
                "output_is_provisional": True,
                "canonical_freeze_emitted": False,
                "review_freeze_emitted": False,
                "split_recomputed": True,
                "split_freeze_emitted": False,
                "distractor_candidates_verified": False,
                "neutral_candidates_verified_unrelated": False,
                "translation_review_complete": False,
                "historical_exposure_complete_for_full_pool": False,
                "hf_checkpoint_bound": False,
                "hf_tokenizer_bound": False,
                "hf_model_executed": False,
                "hf_tokenizer_executed": False,
                "behavior_executed": False,
                "validation_exposed": False,
                "sealed_exposed": False,
                "perturbation_authorized": False,
                "path_not_token_authorized": False,
                "human_gold": False,
            },
            "next_required_steps": [
                "resolve any remaining distractor or Neutral candidate shortfalls",
                "review distractor falsehood and uniqueness",
                "review Neutral unrelatedness and length",
                "translate and review zh against the final split and candidate bindings",
                "complete historical-exposure authority attestation",
                "address bounded semantic-recall limitation before claiming full closure",
                "emit exact-bundle review and split freeze manifests",
                "only then bind an exact HF checkpoint and tokenizer",
            ],
        }
        pre_hf.write_json(paths["manifest"], manifest)

        snapshots = [full_binding, *fact_review["snapshots"], *semantic["snapshots"]]
        if frozen["snapshot"] is not None:
            snapshots.append(frozen["snapshot"])
        _assert_snapshots_current(snapshots)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        staged_dir.replace(output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged_dir, ignore_errors=True)

    manifest_path = output_dir / "postreview_rebuild_manifest.json"
    return {
        "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
        "manifest_path": str(manifest_path),
        "manifest_sha256": pre_hf.sha256_file(manifest_path),
        "retained_base_fact_count": len(reviewed_rows),
        "rejected_base_fact_count": len(exclusions),
        "leakage_component_count": len(components),
        "candidate_shortfall_count": len(shortfalls),
        "semantic_closure_complete_for_full_pool": False,
        "split_freeze_emitted": False,
        "hf_checkpoint_bound": False,
        "hf_tokenizer_bound": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--full-base-facts", default=str(DEFAULT_FULL_BASE_FACTS))
    fact_source = result.add_mutually_exclusive_group(required=True)
    fact_source.add_argument("--fact-review-apply-manifest")
    fact_source.add_argument(
        "--fact-resolution-manifest",
        help=(
            "Completed fact_resolution_manifest.json; --full-base-facts must "
            "be its revised_full_base_facts artifact."
        ),
    )
    result.add_argument(
        "--semantic-candidate-manifest",
        default=str(DEFAULT_SEMANTIC_CANDIDATE_MANIFEST),
    )
    semantic_source = result.add_mutually_exclusive_group(required=True)
    semantic_source.add_argument(
        "--semantic-run-manifest",
        help="Completed direct semantic run manifest; also requires --semantic-adjudications.",
    )
    semantic_source.add_argument("--semantic-composite-manifest")
    result.add_argument("--semantic-adjudications")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--expected-input-count", type=int, default=8969)
    result.add_argument("--split-seed", default=pre_hf.DEFAULT_SPLIT_SEED)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = materialize(
        full_base_facts_path=Path(args.full_base_facts),
        fact_review_apply_manifest_path=(
            Path(args.fact_review_apply_manifest)
            if args.fact_review_apply_manifest
            else None
        ),
        fact_resolution_manifest_path=(
            Path(args.fact_resolution_manifest)
            if args.fact_resolution_manifest
            else None
        ),
        semantic_candidate_manifest_path=Path(args.semantic_candidate_manifest),
        semantic_adjudications_path=(
            Path(args.semantic_adjudications)
            if args.semantic_adjudications
            else None
        ),
        semantic_run_manifest_path=(
            Path(args.semantic_run_manifest) if args.semantic_run_manifest else None
        ),
        semantic_composite_manifest_path=(
            Path(args.semantic_composite_manifest)
            if args.semantic_composite_manifest
            else None
        ),
        output_dir=Path(args.output_dir),
        expected_input_count=args.expected_input_count,
        split_seed=args.split_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
