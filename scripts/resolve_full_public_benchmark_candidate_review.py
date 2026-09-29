#!/usr/bin/env python3
"""Resolve post-review candidate failures without weakening candidate policy.

The workflow is deliberately two phase:

1. ``materialize-resolution`` binds one immutable post-review predecessor and,
   optionally, its complete candidate-review run.  Reject/defer decisions are
   converted into pair-scoped exclusions.  Structural shortfalls and exhausted
   replacement pools cause target facts to be excluded from this cohort only.
   Exclusions are computed to a fixed point because an excluded target can also
   have been a donor for another target.
2. ``materialize-successor`` deterministically replays that resolution and
   publishes a new post-review package.  Components, provisional splits, and
   both candidate kinds are rebuilt from the retained cohort.  Every prior
   candidate/translation review is explicitly invalidated.

The tool never calls a model, never treats proxy evidence as human gold, never
promotes a provisional relation partition, and never weakens the same-split,
different-component candidate constraints.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_TOOL_VERSION = "full-public-benchmark-candidate-resolution-v1"
LEGACY_TOOL_VERSION = "full-public-benchmark-candidate-resolution-v2"
TOOL_VERSION = "full-public-benchmark-candidate-resolution-v3"
SUPPORTED_TOOL_VERSIONS = frozenset(
    {V1_TOOL_VERSION, LEGACY_TOOL_VERSION, TOOL_VERSION}
)
RESOLUTION_MANIFEST_SCHEMA = "public-benchmark-full-candidate-resolution-manifest-v2"
LEGACY_RESOLUTION_MANIFEST_SCHEMA = (
    "public-benchmark-full-candidate-resolution-manifest-v1"
)
SUPPORTED_RESOLUTION_MANIFEST_SCHEMAS = frozenset(
    {LEGACY_RESOLUTION_MANIFEST_SCHEMA, RESOLUTION_MANIFEST_SCHEMA}
)
SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA = (
    "public-benchmark-full-postreview-rebuild-manifest-v3"
)
BASE_POSTREVIEW_MANIFEST_SCHEMA = "public-benchmark-full-postreview-rebuild-manifest-v2"
SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS = frozenset(
    {BASE_POSTREVIEW_MANIFEST_SCHEMA, SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA}
)
PAIR_EXCLUSION_SCHEMA = "public-benchmark-full-candidate-pair-exclusion-v1"
COHORT_EXCLUSION_SCHEMA = "public-benchmark-full-candidate-cohort-exclusion-v1"
RESOLUTION_STATUS = "candidate_resolution_materialized_offline"
TARGET_INELIGIBILITY_POLICY_VERSION = (
    "complete-v3-distractor-answer-uniqueness-fail-closed-v1"
)
TARGET_INELIGIBILITY_STAGE = "candidate_target_eligibility_resolution"
TARGET_INELIGIBILITY_REASON = (
    "complete_published_distractor_review_unanimously_found_"
    "target_answer_not_unique"
)
CANDIDATE_EVIDENCE_REUSE_POLICY = {
    "predecessor_candidate_reviews_blanket_valid": False,
    "exact_v3_accepted_projection_evidence_carry_forward_eligible": True,
    "candidate_id_or_row_sha_alone_sufficient_for_review_reuse": False,
    "full_candidate_evidence_coverage_required": True,
    "full_candidate_model_rereview_required": False,
    "changed_or_new_candidate_fresh_model_review_required": True,
}
LEGACY_CANDIDATE_REVIEW_TOOL_VERSION = (
    "full-public-benchmark-candidate-review-runner-v2"
)
LEGACY_CANDIDATE_CHECKPOINT_SCHEMA = (
    "public-benchmark-full-candidate-review-checkpoint-v1"
)
LEGACY_CANDIDATE_ADJUDICATION_SCHEMA = (
    "public-benchmark-full-candidate-adjudication-v2"
)
LEGACY_CANDIDATE_PROMPT_CONTRACT_SHA256 = (
    "c40fba7bb1dd0972c62d7cf9e2a86fb2f650ae78e777e9eb910b5a1f06f3fe54"
)
LEGACY_CANDIDATE_PROMPT_VERSION = "public-benchmark-full-candidate-semantic-review-v1"
LEGACY_CANDIDATE_REVIEW_METHOD = (
    "independent_behavior_blind_candidate_semantic_review_v1"
)


def _load_sibling(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


postreview = _load_sibling(
    "_candidate_resolution_postreview", "materialize_full_public_benchmark_postreview.py"
)
candidate_review = _load_sibling(
    "_candidate_resolution_review", "run_full_public_benchmark_candidate_review.py"
)
pre_hf = postreview.pre_hf


def validate_candidate_evidence_reuse_policy(
    value: Any, *, label: str
) -> Dict[str, Any]:
    if not isinstance(value, dict) or any(
        value.get(field) is not expected
        for field, expected in CANDIDATE_EVIDENCE_REUSE_POLICY.items()
    ):
        raise ValueError(f"{label} candidate evidence reuse policy is invalid")
    return dict(value)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _resolve_path(binding: Mapping[str, Any], owner_path: Path, label: str) -> Path:
    raw = binding.get("path") or binding.get("filename")
    path = Path(_required_string(raw, f"{label}.path"))
    if not path.is_absolute():
        path = owner_path.parent / path
    return path.resolve()


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    relative_to: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    rendered_path = (
        path.relative_to(Path(relative_to).resolve()).as_posix()
        if relative_to is not None
        else str(path)
    )
    result: Dict[str, Any] = {
        "path": rendered_path,
        "sha256": pre_hf.sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _verify_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: Optional[str] = None,
    jsonl: bool = False,
    allow_empty: bool = False,
    explicit_path: Optional[Path] = None,
) -> Tuple[Path, Any, Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    path = _resolve_path(binding, owner_path, label)
    if explicit_path is not None and path != Path(explicit_path).resolve():
        raise ValueError(f"{label} binding differs from the supplied path")
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != pre_hf.sha256_file(path):
        raise ValueError(f"{label} SHA-256 mismatch")
    if binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count mismatch")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version mismatch")
    value = (
        pre_hf.read_jsonl(path, allow_empty=allow_empty)
        if jsonl
        else pre_hf.read_json(path)
    )
    if jsonl and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record_count mismatch")
    return path, value, dict(binding)


def _index_unique(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label}[{row_number}].{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = dict(row)
    return result


POSTREVIEW_OUTPUT_SPECS = {
    "full_base_facts": (True, pre_hf.FULL_FACT_SCHEMA_VERSION, False),
    "cohort_exclusions": (True, postreview.EXCLUSION_SCHEMA_VERSION, True),
    "leakage_components": (True, pre_hf.COMPONENT_SCHEMA_VERSION, False),
    "split_manifest": (False, pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION, False),
    "distractor_candidates": (True, pre_hf.DISTRACTOR_SCHEMA_VERSION, True),
    "neutral_reference_candidates": (True, pre_hf.NEUTRAL_SCHEMA_VERSION, True),
    "candidate_shortfalls": (True, pre_hf.SHORTFALL_SCHEMA_VERSION, True),
    "integrity_audit": (False, postreview.INTEGRITY_SCHEMA_VERSION, False),
    "summary": (False, postreview.SUMMARY_SCHEMA_VERSION, False),
}


def _load_postreview(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    manifest = pre_hf.read_json(path)
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS:
        raise ValueError("Unsupported postreview manifest schema_version")
    if manifest_schema == SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA:
        validate_candidate_evidence_reuse_policy(
            manifest.get("candidate_repair_contract"),
            label="Postreview successor",
        )
    if manifest.get("status") != "postreview_rebuilt_bounded_semantic_recall_not_frozen":
        raise ValueError("Postreview predecessor status is invalid")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or safety.get("human_gold") is not False:
        raise ValueError("Postreview predecessor evidence boundary is invalid")
    for field in (
        "canonical_freeze_emitted",
        "review_freeze_emitted",
        "split_freeze_emitted",
        "distractor_candidates_verified",
        "neutral_candidates_verified_unrelated",
        "hf_checkpoint_bound",
        "hf_tokenizer_bound",
        "hf_model_executed",
        "hf_tokenizer_executed",
        "behavior_executed",
        "validation_exposed",
        "sealed_exposed",
        "perturbation_authorized",
        "path_not_token_authorized",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"Postreview predecessor safety flag must be false: {field}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Postreview predecessor outputs are missing")
    artifacts: Dict[str, Dict[str, Any]] = {}
    for label, (is_jsonl, schema, allow_empty) in POSTREVIEW_OUTPUT_SPECS.items():
        artifact_path, value, artifact_binding = _verify_binding(
            outputs.get(label),
            owner_path=path,
            label=f"postreview {label}",
            expected_schema=schema,
            jsonl=is_jsonl,
            allow_empty=allow_empty,
        )
        artifacts[label] = {
            "path": artifact_path,
            "value": value,
            "binding": artifact_binding,
        }
    rows = artifacts["full_base_facts"]["value"]
    facts = candidate_review._validate_full_facts(rows)
    candidate_review._validate_candidate_rows(
        distractors=artifacts["distractor_candidates"]["value"],
        neutrals=artifacts["neutral_reference_candidates"]["value"],
        facts=facts,
    )
    exclusion_index = _index_unique(
        artifacts["cohort_exclusions"]["value"],
        "base_fact_id",
        "postreview cohort exclusions",
    )
    if set(facts).intersection(exclusion_index):
        raise ValueError("Postreview retained and excluded facts overlap")
    shortfall_index = _index_unique(
        artifacts["candidate_shortfalls"]["value"],
        "base_fact_id",
        "postreview candidate shortfalls",
    )
    distractor_counts = Counter(
        str(row["base_fact_id"])
        for row in artifacts["distractor_candidates"]["value"]
    )
    neutral_counts = Counter(
        str(row["base_fact_id"])
        for row in artifacts["neutral_reference_candidates"]["value"]
    )
    expected_shortfalls = {
        base_fact_id
        for base_fact_id in facts
        if distractor_counts[base_fact_id] != 2 or neutral_counts[base_fact_id] != 2
    }
    if set(shortfall_index) != expected_shortfalls:
        raise ValueError("Postreview candidate shortfall ledger is not exhaustive")
    components = artifacts["leakage_components"]["value"]
    component_by_fact: Dict[str, str] = {}
    for component in components:
        component_id = _required_string(
            component.get("leakage_component_id"), "leakage_component_id"
        )
        members = component.get("base_fact_ids")
        if not isinstance(members, list) or component.get("member_count") != len(members):
            raise ValueError(f"Invalid postreview component: {component_id}")
        for base_fact_id in members:
            base_fact_id = str(base_fact_id)
            if base_fact_id in component_by_fact or base_fact_id not in facts:
                raise ValueError(f"Invalid postreview component coverage: {base_fact_id}")
            component_by_fact[base_fact_id] = component_id
    if set(component_by_fact) != set(facts):
        raise ValueError("Postreview components do not cover retained facts")
    return {
        "path": path,
        "manifest": manifest,
        "binding": _binding(path, schema_version=str(manifest_schema)),
        "artifacts": artifacts,
        "rows": rows,
        "row_index": facts,
        "exclusions": artifacts["cohort_exclusions"]["value"],
        "exclusion_index": exclusion_index,
        "components": components,
        "shortfalls": artifacts["candidate_shortfalls"]["value"],
        "distractors": artifacts["distractor_candidates"]["value"],
        "neutrals": artifacts["neutral_reference_candidates"]["value"],
    }


def _load_candidate_review(
    *,
    predecessor_path: Path,
    review_manifest_path: Path,
    allow_legacy_v2: bool = False,
) -> Dict[str, Any]:
    predecessor_path = Path(predecessor_path).resolve()
    review_manifest_path = Path(review_manifest_path).resolve()
    raw_manifest = pre_hf.read_json(review_manifest_path)
    if raw_manifest.get("tool_version") == LEGACY_CANDIDATE_REVIEW_TOOL_VERSION:
        if not allow_legacy_v2:
            raise ValueError(
                "Legacy v2 candidate review is migration-only and cannot authorize a new resolution"
            )
        return _load_legacy_candidate_review(
            predecessor_path=predecessor_path,
            review_manifest_path=review_manifest_path,
            manifest=raw_manifest,
        )
    try:
        evidence = candidate_review.load_review_evidence_manifest(
            review_manifest_path
        )
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError(f"Candidate review evidence lineage is invalid: {exc}") from exc
    manifest = evidence["manifest"]
    if manifest.get("schema_version") != candidate_review.RUN_MANIFEST_SCHEMA:
        raise ValueError("Unsupported candidate review manifest schema_version")
    if manifest.get("status") != "completed":
        raise ValueError("Candidate review contains failed terminal rows")
    if not all(
        manifest.get(field) is True
        for field in (
            "selection_is_full_candidate_set",
            "candidate_review_complete_for_selected_scope",
            "candidate_review_complete_for_full_set",
        )
    ):
        raise ValueError("Candidate review is not complete for the full candidate set")
    _, manifest_kind, input_bindings, units = candidate_review.load_review_units(
        candidate_manifest_path=predecessor_path
    )
    if manifest_kind != "postreview_rebuild":
        raise ValueError("Candidate review predecessor kind is invalid")
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict) or manifest.get(
        "run_contract_sha256"
    ) != candidate_review.sha256_value(run_contract):
        raise ValueError("Candidate review run contract is stale")
    protocol_identity = {
        "provider_profile": candidate_review.PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": candidate_review.DEFAULT_REVIEWER_MODEL,
        "expected_response_model": candidate_review.DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if run_contract.get("protocol_reviewer_identity") != protocol_identity:
        raise ValueError("Candidate review protocol reviewer identity is stale")
    reviewer_spec = run_contract.get("reviewer_spec")
    if not isinstance(reviewer_spec, dict) or any(
        (
            reviewer_spec.get("provider_profile")
            != protocol_identity["provider_profile"],
            reviewer_spec.get("model") != protocol_identity["requested_model"],
            run_contract.get("expected_response_model")
            != protocol_identity["expected_response_model"],
        )
    ):
        raise ValueError("Candidate review model route or response identity is unsupported")
    source_binding = run_contract.get("candidate_manifest")
    if not isinstance(source_binding, dict):
        raise ValueError("Candidate review source manifest binding is missing")
    if _resolve_path(source_binding, review_manifest_path, "candidate source") != predecessor_path:
        raise ValueError("Candidate review is bound to a different postreview predecessor")
    if source_binding.get("sha256") != pre_hf.sha256_file(predecessor_path):
        raise ValueError("Candidate review predecessor SHA-256 is stale")
    if evidence["candidate_manifest_path"] != predecessor_path:
        raise ValueError("Candidate review evidence binds a different postreview predecessor")
    if run_contract.get("input_artifacts") != input_bindings:
        raise ValueError("Candidate review input artifact bindings are stale")
    unit_ids = [str(unit.candidate_id) for unit in units]
    if not units or run_contract.get("selected_candidate_ids_sha256") != pre_hf.sha256_value(
        unit_ids
    ):
        raise ValueError("Candidate review selected IDs are stale")
    if run_contract.get("selected_candidate_count") != len(units) or run_contract.get(
        "full_candidate_count"
    ) != len(units):
        raise ValueError("Candidate review selected counts are stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Candidate review artifacts are missing")
    checkpoint_path, checkpoints, checkpoint_binding = _verify_binding(
        artifacts.get("checkpoint"),
        owner_path=review_manifest_path,
        label="candidate review checkpoint",
        expected_schema=candidate_review.CHECKPOINT_SCHEMA,
        jsonl=True,
    )
    adjudications_path, adjudications, adjudication_binding = _verify_binding(
        artifacts.get("adjudications"),
        owner_path=review_manifest_path,
        label="candidate review adjudications",
        expected_schema=candidate_review.ADJUDICATION_SCHEMA,
        jsonl=True,
    )
    if len(checkpoints) != len(units) or len(adjudications) != len(units):
        raise ValueError("Candidate review artifacts do not cover the full set")
    checkpoint_index = _index_unique(checkpoints, "candidate_id", "candidate checkpoints")
    adjudication_index = _index_unique(
        adjudications, "candidate_id", "candidate adjudications"
    )
    if set(checkpoint_index) != set(unit_ids) or set(adjudication_index) != set(unit_ids):
        raise ValueError("Candidate review candidate ID coverage is stale")
    units_by_id = {str(unit.candidate_id): unit for unit in units}
    for unit in units:
        candidate_id = str(unit.candidate_id)
        _, checkpoint = candidate_review._validate_checkpoint_row(
            checkpoint_index[candidate_id],
            units_by_id=units_by_id,
            run_contract_sha256=str(manifest["run_contract_sha256"]),
            review_semantics_contract_sha256=evidence[
                "review_semantics_contract_sha256"
            ],
        )
        if checkpoint.get("terminal_status") != "completed":
            raise ValueError(f"Candidate review is not terminal: {candidate_id}")
        decision = adjudication_index[candidate_id]
        if checkpoint.get("strict_adjudication") != decision:
            raise ValueError(f"Candidate checkpoint/adjudication mismatch: {candidate_id}")
        expected = {
            "candidate_id": candidate_id,
            "candidate_kind": unit.candidate_kind,
            "candidate_row_sha256": unit.candidate_row_sha256,
            "target_base_fact_id": unit.target_base_fact_id,
            "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
            "source_base_fact_id": unit.source_base_fact_id,
            "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
        }
        for field, value in expected.items():
            if decision.get(field) != value:
                raise ValueError(f"Candidate adjudication {field} is stale: {candidate_id}")
        if decision.get("human_gold") is not False or decision.get("behavior_blind") is not True:
            raise ValueError(f"Candidate adjudication evidence boundary is invalid: {candidate_id}")
        expected_identity = {
            "reviewer_id": (
                f"{protocol_identity['provider_profile']}:"
                f"{protocol_identity['expected_response_model']}"
            ),
            "requested_model": protocol_identity["requested_model"],
            "expected_response_model": protocol_identity["expected_response_model"],
            "response_model": protocol_identity["expected_response_model"],
            "response_model_identity_status": "matched",
        }
        for field, value in expected_identity.items():
            if decision.get(field) != value:
                raise ValueError(
                    f"Candidate adjudication model identity is invalid: {candidate_id}.{field}"
                )
        verdict = {
            "candidate_id": candidate_id,
            "candidate_kind": unit.candidate_kind,
            "overall_decision": decision.get("overall_decision"),
            **{field: decision.get(field) for field in candidate_review.FIELD_NAMES},
            "rationale": decision.get("rationale"),
            "confidence": decision.get("confidence"),
        }
        candidate_review.validate_batch_response(
            {
                "schema_version": candidate_review.BATCH_RESPONSE_SCHEMA,
                "records": [verdict],
            },
            [unit],
        )
    return {
        "manifest": manifest,
        "manifest_path": review_manifest_path,
        "manifest_binding": _binding(
            review_manifest_path,
            schema_version=candidate_review.RUN_MANIFEST_SCHEMA,
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_binding": checkpoint_binding,
        "adjudications_path": adjudications_path,
        "adjudications_binding": adjudication_binding,
        "adjudications": adjudications,
        "adjudication_index": adjudication_index,
        "units": units,
        "review_semantics_contract_sha256": evidence[
            "review_semantics_contract_sha256"
        ],
        "evidence_origin_counts": dict(manifest["evidence_origin_counts"]),
        "fresh_model_review_candidate_count": manifest[
            "fresh_model_review_candidate_count"
        ],
        "carried_forward_candidate_count": manifest[
            "carried_forward_candidate_count"
        ],
        "full_evidence_partition_complete": manifest[
            "full_evidence_partition_complete"
        ],
        "legacy_migration_only": False,
    }


def _load_legacy_candidate_review(
    *,
    predecessor_path: Path,
    review_manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate immutable v2 evidence only for replaying an existing v1 resolution."""

    if manifest.get("schema_version") != candidate_review.RUN_MANIFEST_SCHEMA:
        raise ValueError("Unsupported legacy candidate review manifest schema_version")
    if manifest.get("status") != "completed" or not all(
        manifest.get(field) is True
        for field in (
            "selection_is_full_candidate_set",
            "candidate_review_complete_for_selected_scope",
            "candidate_review_complete_for_full_set",
        )
    ):
        raise ValueError("Legacy candidate review is not complete for the full set")
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict) or manifest.get(
        "run_contract_sha256"
    ) != candidate_review.sha256_value(run_contract):
        raise ValueError("Legacy candidate review run contract is stale")
    if run_contract.get("tool_version") != LEGACY_CANDIDATE_REVIEW_TOOL_VERSION:
        raise ValueError("Legacy candidate review run contract tool version is unsupported")
    if run_contract.get(
        "prompt_contract_sha256"
    ) != LEGACY_CANDIDATE_PROMPT_CONTRACT_SHA256:
        raise ValueError("Legacy candidate review prompt contract is unsupported")
    _, manifest_kind, input_bindings, units = candidate_review.load_review_units(
        candidate_manifest_path=predecessor_path
    )
    if manifest_kind != "postreview_rebuild":
        raise ValueError("Legacy candidate review predecessor kind is invalid")
    source_binding = run_contract.get("candidate_manifest")
    if not isinstance(source_binding, dict) or _resolve_path(
        source_binding,
        review_manifest_path,
        "legacy candidate source",
    ) != predecessor_path:
        raise ValueError("Legacy candidate review binds a different predecessor")
    if source_binding.get("sha256") != pre_hf.sha256_file(predecessor_path):
        raise ValueError("Legacy candidate review predecessor SHA-256 is stale")
    unit_ids = [unit.candidate_id for unit in units]
    if run_contract.get("input_artifacts") != input_bindings or not all(
        (
            run_contract.get("selected_candidate_count") == len(units),
            run_contract.get("full_candidate_count") == len(units),
            run_contract.get("selection_is_full_candidate_set") is True,
            run_contract.get("selected_candidate_ids_sha256")
            == pre_hf.sha256_value(unit_ids),
        )
    ):
        raise ValueError("Legacy candidate review coverage contract is stale")
    protocol_identity = {
        "provider_profile": candidate_review.PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": candidate_review.DEFAULT_REVIEWER_MODEL,
        "expected_response_model": candidate_review.DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if run_contract.get("protocol_reviewer_identity") != protocol_identity:
        raise ValueError("Legacy candidate review protocol identity is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Legacy candidate review artifacts are missing")
    checkpoint_path, checkpoints, checkpoint_binding = _verify_binding(
        artifacts.get("checkpoint"),
        owner_path=review_manifest_path,
        label="legacy candidate review checkpoint",
        expected_schema=LEGACY_CANDIDATE_CHECKPOINT_SCHEMA,
        jsonl=True,
    )
    adjudications_path, adjudications, adjudications_binding = _verify_binding(
        artifacts.get("adjudications"),
        owner_path=review_manifest_path,
        label="legacy candidate review adjudications",
        expected_schema=LEGACY_CANDIDATE_ADJUDICATION_SCHEMA,
        jsonl=True,
    )
    if len(checkpoints) != len(units) or len(adjudications) != len(units):
        raise ValueError("Legacy candidate review artifacts do not cover the full set")
    checkpoints_by_id = _index_unique(
        checkpoints, "candidate_id", "legacy candidate checkpoints"
    )
    adjudications_by_id = _index_unique(
        adjudications, "candidate_id", "legacy candidate adjudications"
    )
    if set(checkpoints_by_id) != set(unit_ids) or set(adjudications_by_id) != set(
        unit_ids
    ):
        raise ValueError("Legacy candidate review ID coverage is stale")
    for unit in units:
        candidate_id = unit.candidate_id
        checkpoint = checkpoints_by_id[candidate_id]
        decision = adjudications_by_id[candidate_id]
        expected_binding = {
            "candidate_kind": unit.candidate_kind,
            "candidate_row_sha256": unit.candidate_row_sha256,
            "target_base_fact_id": unit.target_base_fact_id,
            "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
            "source_base_fact_id": unit.source_base_fact_id,
            "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
        }
        if checkpoint.get("schema_version") != LEGACY_CANDIDATE_CHECKPOINT_SCHEMA or checkpoint.get(
            "tool_version"
        ) != LEGACY_CANDIDATE_REVIEW_TOOL_VERSION:
            raise ValueError(f"Legacy candidate checkpoint identity is invalid: {candidate_id}")
        if checkpoint.get("run_contract_sha256") != manifest.get(
            "run_contract_sha256"
        ) or checkpoint.get("terminal_status") != "completed":
            raise ValueError(f"Legacy candidate checkpoint is not terminal: {candidate_id}")
        if checkpoint.get("strict_adjudication") != decision:
            raise ValueError(f"Legacy candidate evidence mismatch: {candidate_id}")
        for field, expected in expected_binding.items():
            if checkpoint.get(field) != expected or decision.get(field) != expected:
                raise ValueError(
                    f"Legacy candidate evidence {field} is stale: {candidate_id}"
                )
        if decision.get("schema_version") != LEGACY_CANDIDATE_ADJUDICATION_SCHEMA or decision.get(
            "review_method"
        ) != LEGACY_CANDIDATE_REVIEW_METHOD:
            raise ValueError(f"Legacy candidate adjudication identity is invalid: {candidate_id}")
        if decision.get("overall_decision") not in {"accept", "reject", "defer"}:
            raise ValueError(f"Legacy candidate decision is invalid: {candidate_id}")
        if decision.get("behavior_blind") is not True or decision.get(
            "human_gold"
        ) is not False:
            raise ValueError(f"Legacy candidate evidence boundary is invalid: {candidate_id}")
        calls = checkpoint.get("model_calls")
        final_call = calls[-1] if isinstance(calls, list) and calls else None
        if not isinstance(final_call, dict) or any(
            final_call.get(field) != expected
            for field, expected in {
                "prompt_version": LEGACY_CANDIDATE_PROMPT_VERSION,
                "provider_profile": protocol_identity["provider_profile"],
                "requested_model": protocol_identity["requested_model"],
                "expected_response_model": protocol_identity[
                    "expected_response_model"
                ],
                "response_model": protocol_identity["expected_response_model"],
                "response_model_identity_status": "matched",
            }.items()
        ):
            raise ValueError(f"Legacy candidate model identity is invalid: {candidate_id}")
        for field in (
            "hf_model_execution_count",
            "hf_tokenizer_execution_count",
            "behavior_execution_count",
            "validation_behavior_exposure_count",
            "sealed_behavior_exposure_count",
        ):
            if checkpoint.get(field) != 0:
                raise ValueError(
                    f"Legacy candidate execution boundary is invalid: {candidate_id}.{field}"
                )
    return {
        "manifest": dict(manifest),
        "manifest_path": review_manifest_path,
        "manifest_binding": _binding(
            review_manifest_path,
            schema_version=candidate_review.RUN_MANIFEST_SCHEMA,
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_binding": checkpoint_binding,
        "adjudications_path": adjudications_path,
        "adjudications_binding": adjudications_binding,
        "adjudications": adjudications,
        "adjudication_index": adjudications_by_id,
        "units": units,
        "legacy_migration_only": True,
    }


def _semantic_current_ids(candidate: Mapping[str, Any]) -> Set[str]:
    result: Set[str] = set()
    for side in ("left", "right"):
        endpoint = candidate.get(side)
        provenance = endpoint.get("provenance") if isinstance(endpoint, dict) else None
        if not isinstance(provenance, dict):
            raise ValueError("Semantic candidate endpoint provenance is invalid")
        if provenance.get("input_role") == "current_full_base":
            result.add(
                _required_string(
                    provenance.get("base_fact_id"),
                    "semantic candidate current base_fact_id",
                )
            )
    return result


def _candidate_cohort_exclusion_row(
    *,
    base_fact_id: str,
    shortfall: Mapping[str, Any],
    root: Mapping[str, Any],
) -> Dict[str, Any]:
    source_row = root["fact_base_index"][base_fact_id]
    return {
        "schema_version": COHORT_EXCLUSION_SCHEMA,
        "base_fact_id": base_fact_id,
        "disposition": "cohort_exclude",
        "exclusion_stage": "candidate_availability_resolution",
        "exclusion_reason": (
            "fewer_than_two_candidates_under_unchanged_same_split_"
            "different_component_relation_policy"
        ),
        "fixed_point_round": shortfall["fixed_point_round"],
        "distractor_candidate_count_before_exclusion": shortfall[
            "distractor_candidate_count"
        ],
        "neutral_candidate_count_before_exclusion": shortfall[
            "neutral_reference_candidate_count"
        ],
        "shortfall_reasons": copy.deepcopy(shortfall["shortfall_reasons"]),
        "source_root_postreview_row_sha256": pre_hf.sha256_value(source_row),
        "candidate_policy_relaxed": False,
        "factual_falsehood_asserted": False,
        "human_gold": False,
    }


def _target_ineligibility_evidence(
    adjudications: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Find target-level failures without generalizing pair-specific judgments.

    Candidate review is already required to cover the predecessor's complete
    published candidate set.  A target is ineligible only when every published
    distractor judgment for that target is non-accept with
    ``answer_unique == False``, and the evidence covers at least two distinct
    candidates and sources.  Neutral judgments never participate.
    """

    by_target: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in adjudications:
        if row.get("candidate_kind") != "distractor":
            continue
        target_id = _required_string(
            row.get("target_base_fact_id"),
            "candidate adjudication target_base_fact_id",
        )
        by_target[target_id].append(dict(row))

    result: Dict[str, Dict[str, Any]] = {}
    for target_id in sorted(by_target):
        rows = sorted(
            by_target[target_id], key=lambda row: str(row.get("candidate_id") or "")
        )
        candidate_ids = [
            _required_string(row.get("candidate_id"), "candidate adjudication candidate_id")
            for row in rows
        ]
        source_ids = [
            _required_string(
                row.get("source_base_fact_id"),
                "candidate adjudication source_base_fact_id",
            )
            for row in rows
        ]
        if len(rows) < 2 or len(set(candidate_ids)) < 2 or len(set(source_ids)) < 2:
            continue
        if not all(
            row.get("overall_decision") in {"reject", "defer"}
            and row.get("answer_unique") is False
            for row in rows
        ):
            continue
        candidate_row_hashes = [
            _required_string(
                row.get("candidate_row_sha256"),
                "candidate adjudication candidate_row_sha256",
            )
            for row in rows
        ]
        adjudication_row_hashes = [pre_hf.sha256_value(row) for row in rows]
        result[target_id] = {
            "policy_version": TARGET_INELIGIBILITY_POLICY_VERSION,
            "target_base_fact_id": target_id,
            "published_distractor_judgment_count": len(rows),
            "distinct_distractor_candidate_count": len(set(candidate_ids)),
            "distinct_distractor_source_count": len(set(source_ids)),
            "candidate_ids": candidate_ids,
            "candidate_ids_sha256": pre_hf.sha256_value(candidate_ids),
            "source_base_fact_ids": source_ids,
            "source_base_fact_ids_sha256": pre_hf.sha256_value(source_ids),
            "candidate_row_sha256s": candidate_row_hashes,
            "candidate_row_sha256s_sha256": pre_hf.sha256_value(
                candidate_row_hashes
            ),
            "adjudication_row_sha256s": adjudication_row_hashes,
            "adjudication_row_sha256s_sha256": pre_hf.sha256_value(
                adjudication_row_hashes
            ),
            "all_published_distractor_judgments_non_accept": True,
            "all_published_distractor_answer_unique_false": True,
            "proxy_review_only": True,
            "human_gold": False,
        }
    return result


def _target_ineligibility_contract(
    evidence_by_target: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    target_ids = sorted(evidence_by_target)
    ordered_evidence = [
        copy.deepcopy(dict(evidence_by_target[target_id])) for target_id in target_ids
    ]
    return {
        "policy_version": TARGET_INELIGIBILITY_POLICY_VERSION,
        "candidate_kind": "distractor",
        "trigger_field": "answer_unique",
        "trigger_value": False,
        "complete_full_candidate_review_required": True,
        "all_published_target_distractor_judgments_required": True,
        "all_triggering_judgments_non_accept_required": True,
        "minimum_distinct_candidate_count": 2,
        "minimum_distinct_source_count": 2,
        "neutral_judgments_participate": False,
        "cohort_scope_only": True,
        "factual_falsehood_asserted": False,
        "human_gold": False,
        "new_target_ineligibility_exclusion_count": len(target_ids),
        "new_target_ineligibility_base_fact_ids": target_ids,
        "ordered_new_target_ineligibility_base_fact_ids_sha256": pre_hf.sha256_value(
            target_ids
        ),
        "target_evidence": ordered_evidence,
        "ordered_target_evidence_sha256": pre_hf.sha256_value(ordered_evidence),
    }


def _candidate_target_ineligibility_row(
    *,
    base_fact_id: str,
    evidence: Mapping[str, Any],
    review: Mapping[str, Any],
    root: Mapping[str, Any],
) -> Dict[str, Any]:
    source_row = root["fact_base_index"][base_fact_id]
    return {
        "schema_version": COHORT_EXCLUSION_SCHEMA,
        "base_fact_id": base_fact_id,
        "disposition": "cohort_exclude",
        "exclusion_stage": TARGET_INELIGIBILITY_STAGE,
        "exclusion_reason": TARGET_INELIGIBILITY_REASON,
        "target_ineligibility_evidence": copy.deepcopy(dict(evidence)),
        "source_candidate_review_manifest_sha256": review["manifest_binding"][
            "sha256"
        ],
        "source_candidate_review_adjudications_sha256": review[
            "adjudications_binding"
        ]["sha256"],
        "source_candidate_review_checkpoint_sha256": review["checkpoint_binding"][
            "sha256"
        ],
        "source_root_postreview_row_sha256": pre_hf.sha256_value(source_row),
        "candidate_policy_relaxed": False,
        "factual_falsehood_asserted": False,
        "human_gold": False,
    }


def _semantic_fact_resolution_context(
    resolution: Mapping[str, Any],
) -> Dict[str, Any]:
    """Adapt the semantic materializer's public resolution context.

    ``load_fact_resolution_context`` calls cohort removals ``excluded_ids``;
    postreview's semantic-authority replay uses the equivalent
    ``rejected_ids`` name.  Keep the source object immutable and fail closed if
    a caller ever supplies conflicting aliases.
    """

    adapted = dict(resolution)
    excluded = adapted.get("excluded_ids")
    if not isinstance(excluded, list) or any(
        not isinstance(base_fact_id, str) or not base_fact_id
        for base_fact_id in excluded
    ):
        raise ValueError("Fact-resolution excluded_ids are invalid")
    rejected = adapted.get("rejected_ids")
    if rejected is not None and rejected != excluded:
        raise ValueError("Fact-resolution excluded/rejected ID aliases conflict")
    adapted["rejected_ids"] = list(excluded)
    return adapted


def _load_root_context(root_path: Path) -> Dict[str, Any]:
    root = _load_postreview(root_path)
    inputs = root["manifest"].get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Root postreview inputs are missing")
    if inputs.get("candidate_repair") is not None:
        raise ValueError("Root postreview must precede candidate repair")
    input_path, source_rows, source_binding = _verify_binding(
        inputs.get("full_base_facts"),
        owner_path=root["path"],
        label="root postreview full_base_facts input",
        expected_schema=pre_hf.FULL_FACT_SCHEMA_VERSION,
        jsonl=True,
    )
    source_index = _index_unique(source_rows, "base_fact_id", "root component source facts")
    fact_contract = root["manifest"].get("fact_review_contract")
    if not isinstance(fact_contract, dict):
        raise ValueError("Root fact review contract is missing")
    mode = fact_contract.get("mode")
    revised_ids: List[str] = []
    fact_resolution_context: Optional[Dict[str, Any]] = None
    if mode == "fact_resolution_successor_cohort":
        fact_resolution = inputs.get("fact_resolution")
        if not isinstance(fact_resolution, dict):
            raise ValueError("Root fact resolution bindings are missing")
        resolution_manifest_path, _, _ = _verify_binding(
            fact_resolution.get("fact_resolution_manifest"),
            owner_path=root["path"],
            label="root fact resolution manifest",
            expected_schema=postreview.semantic_runner.materializer.RESOLUTION_MANIFEST_SCHEMA,
        )
        resolution = postreview.semantic_runner.materializer.load_fact_resolution_context(
            fact_resolution_manifest_path=resolution_manifest_path,
            resolved_full_base_facts_path=input_path,
        )
        fact_resolution_context = _semantic_fact_resolution_context(resolution)
        original_ids = list(resolution["source_ids"])
        fact_retained_ids = list(resolution["retained_ids"])
        fact_excluded_ids = list(resolution["excluded_ids"])
        revised_ids = list(resolution["revised_ids"])
    elif mode == "legacy_fact_review_apply":
        original_ids = [str(row["base_fact_id"]) for row in source_rows]
        fact_retained_ids = [str(row["base_fact_id"]) for row in root["rows"]]
        fact_excluded_ids = [str(row["base_fact_id"]) for row in root["exclusions"]]
    else:
        raise ValueError(f"Unsupported root fact review mode: {mode!r}")
    if [str(row["base_fact_id"]) for row in root["rows"]] != fact_retained_ids:
        raise ValueError("Root retained facts differ from fact-review lineage")
    if [str(row["base_fact_id"]) for row in root["exclusions"]] != fact_excluded_ids:
        raise ValueError("Root exclusions differ from fact-review lineage")

    semantic = inputs.get("semantic_review")
    if not isinstance(semantic, dict):
        raise ValueError("Root semantic review inputs are missing")
    semantic_manifest_path, _, _ = _verify_binding(
        semantic.get("candidate_manifest"),
        owner_path=root["path"],
        label="root semantic candidate manifest",
        expected_schema=postreview.semantic_runner.materializer.MANIFEST_SCHEMA,
    )
    adjudications_path, _, _ = _verify_binding(
        semantic.get("adjudications"),
        owner_path=root["path"],
        label="root semantic adjudications",
        expected_schema=postreview.semantic_runner.ADJUDICATION_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    authority_mode = semantic.get("authority_mode", "direct_semantic_review_run")
    direct_run_binding = semantic.get("run_manifest")
    composite_binding = semantic.get("composite_manifest")
    if authority_mode == "direct_semantic_review_run":
        if not isinstance(direct_run_binding, dict) or composite_binding is not None:
            raise ValueError("Root semantic direct/composite authority is inconsistent")
        semantic_run_manifest_path = _resolve_path(
            direct_run_binding, root["path"], "root semantic run manifest"
        )
        semantic_composite_manifest_path = None
        direct_adjudications_path: Optional[Path] = adjudications_path
    elif authority_mode == "composite_semantic_review_authority":
        if direct_run_binding is not None or not isinstance(composite_binding, dict):
            raise ValueError("Root semantic direct/composite authority is inconsistent")
        semantic_run_manifest_path = None
        semantic_composite_manifest_path = _resolve_path(
            composite_binding, root["path"], "root semantic composite manifest"
        )
        direct_adjudications_path = None
    else:
        raise ValueError(f"Unsupported root semantic authority mode: {authority_mode!r}")
    semantic_evidence = postreview._load_semantic_review_evidence(
        candidate_manifest_path=semantic_manifest_path,
        adjudications_path=direct_adjudications_path,
        run_manifest_path=semantic_run_manifest_path,
        composite_manifest_path=semantic_composite_manifest_path,
        full_base_facts_path=input_path,
        fact_resolution=fact_resolution_context,
    )
    semantic_manifest = semantic_evidence["manifest"]
    candidates = semantic_evidence["candidates"]
    decisions = semantic_evidence["decisions"]
    semantic_inputs = semantic_manifest.get("inputs")
    if not isinstance(semantic_inputs, dict):
        raise ValueError("Semantic candidate inputs are missing")
    current_binding = semantic_inputs.get("current_full_base_facts")
    current_path = _resolve_path(
        current_binding, semantic_manifest_path, "semantic current full_base_facts"
    )
    if current_path != input_path or current_binding.get("sha256") != pre_hf.sha256_file(
        input_path
    ):
        raise ValueError("Root semantic candidate full-base binding is stale")
    component_path = semantic_evidence["component_path"]
    components = semantic_evidence["components"]
    split_path = semantic_evidence["split_path"]
    split_manifest = semantic_evidence["split_manifest"]
    postreview._validate_current_components(
        full_index=source_index, components=components
    )
    frozen = postreview._load_frozen_constraints(
        split_manifest=split_manifest,
        split_manifest_path=split_path,
        full_ids=list(source_index),
        source_universe_ids=original_ids,
    )
    return {
        "root": root,
        "source_path": input_path,
        "source_rows": source_rows,
        "source_index": source_index,
        "source_binding": source_binding,
        "fact_base_index": root["row_index"],
        "fact_exclusions": root["exclusions"],
        "fact_excluded_ids": fact_excluded_ids,
        "fact_retained_ids": fact_retained_ids,
        "original_ids": original_ids,
        "revised_ids": revised_ids,
        "fact_mode": mode,
        "semantic_candidates": candidates,
        "semantic_decisions": decisions,
        "semantic_components": components,
        "semantic_component_path": component_path,
        "semantic_split_manifest": split_manifest,
        "semantic_split_path": split_path,
        "frozen": frozen,
        "semantic_adjudications_path": adjudications_path,
    }


def load_resolution_context(
    *,
    resolution_manifest_path: Path,
    predecessor_postreview_manifest_path: Optional[Path] = None,
    _lineage_stack: Optional[Set[Path]] = None,
) -> Dict[str, Any]:
    path = Path(resolution_manifest_path).resolve()
    lineage_stack = set(_lineage_stack or set())
    if path in lineage_stack:
        raise ValueError("Candidate resolution lineage contains a cycle")
    lineage_stack.add(path)
    manifest = pre_hf.read_json(path)
    resolution_schema = manifest.get("schema_version")
    if resolution_schema not in SUPPORTED_RESOLUTION_MANIFEST_SCHEMAS:
        raise ValueError("Unsupported candidate resolution manifest schema_version")
    tool_version = manifest.get("tool_version")
    if tool_version not in SUPPORTED_TOOL_VERSIONS:
        raise ValueError("Unsupported candidate resolution tool_version")
    if (
        resolution_schema == LEGACY_RESOLUTION_MANIFEST_SCHEMA
        and tool_version != V1_TOOL_VERSION
    ) or (
        resolution_schema == RESOLUTION_MANIFEST_SCHEMA
        and tool_version not in {LEGACY_TOOL_VERSION, TOOL_VERSION}
    ):
        raise ValueError("Candidate resolution schema/tool version combination is invalid")
    target_ineligibility_enabled = tool_version == TOOL_VERSION
    if target_ineligibility_enabled and resolution_schema != RESOLUTION_MANIFEST_SCHEMA:
        raise ValueError("Target-ineligibility resolution uses an unsupported schema")
    if not target_ineligibility_enabled and manifest.get(
        "target_ineligibility_contract"
    ) is not None:
        raise ValueError(
            "Legacy candidate resolution unexpectedly declares target ineligibility"
        )
    if manifest.get("status") != RESOLUTION_STATUS:
        raise ValueError("Candidate resolution manifest status is invalid")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Candidate resolution inputs/outputs are missing")
    predecessor_path, _, predecessor_binding = _verify_binding(
        inputs.get("predecessor_postreview_manifest"),
        owner_path=path,
        label="candidate resolution predecessor postreview",
        explicit_path=predecessor_postreview_manifest_path,
    )
    root_path, _, root_binding = _verify_binding(
        inputs.get("root_postreview_manifest"),
        owner_path=path,
        label="candidate resolution root postreview",
    )
    pair_path, pair_rows, pair_binding = _verify_binding(
        outputs.get("candidate_pair_exclusions"),
        owner_path=path,
        label="candidate pair exclusions",
        expected_schema=PAIR_EXCLUSION_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    exclusion_path, exclusion_rows, exclusion_binding = _verify_binding(
        outputs.get("candidate_cohort_exclusions"),
        owner_path=path,
        label="candidate cohort exclusions",
        expected_schema=COHORT_EXCLUSION_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    pair_keys: List[Tuple[str, str, str]] = []
    for row in pair_rows:
        if row.get("schema_version") != PAIR_EXCLUSION_SCHEMA:
            raise ValueError("Candidate pair exclusion row schema is invalid")
        key = (
            _required_string(row.get("candidate_kind"), "pair candidate_kind"),
            _required_string(row.get("target_base_fact_id"), "pair target_base_fact_id"),
            _required_string(row.get("source_base_fact_id"), "pair source_base_fact_id"),
        )
        if key[0] not in candidate_review.CANDIDATE_KINDS:
            raise ValueError(f"Candidate pair exclusion kind is invalid: {key[0]}")
        if row.get("pair_disposition") != "exclude_pair_from_candidate_selection":
            raise ValueError("Candidate pair exclusion disposition is invalid")
        if row.get("source_fact_falsehood_asserted") is not False or row.get(
            "human_gold"
        ) is not False:
            raise ValueError("Candidate pair exclusion evidence boundary is invalid")
        pair_keys.append(key)
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("Candidate pair exclusion ledger repeats a pair")
    excluded_ids: List[str] = []
    for row in exclusion_rows:
        if row.get("schema_version") != COHORT_EXCLUSION_SCHEMA:
            raise ValueError("Candidate cohort exclusion row schema is invalid")
        base_fact_id = _required_string(
            row.get("base_fact_id"), "candidate cohort exclusion base_fact_id"
        )
        if row.get("disposition") != "cohort_exclude" or row.get(
            "factual_falsehood_asserted"
        ) is not False:
            raise ValueError("Candidate cohort exclusion disposition is invalid")
        if row.get("human_gold") is not False:
            raise ValueError("Candidate cohort exclusion claims human gold")
        exclusion_stage = _required_string(
            row.get("exclusion_stage"), "candidate cohort exclusion stage"
        )
        allowed_stages = {"candidate_availability_resolution"}
        if target_ineligibility_enabled:
            allowed_stages.add(TARGET_INELIGIBILITY_STAGE)
        if exclusion_stage not in allowed_stages:
            raise ValueError("Candidate cohort exclusion stage is invalid")
        excluded_ids.append(base_fact_id)
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("Candidate cohort exclusion ledger repeats a fact")
    counts = manifest.get("counts")
    plan = manifest.get("resolution_plan")
    if not isinstance(counts, dict) or not isinstance(plan, dict):
        raise ValueError("Candidate resolution counts/plan are missing")
    if counts.get("candidate_pair_exclusion_count") != len(pair_rows) or counts.get(
        "candidate_cohort_exclusion_count"
    ) != len(exclusion_rows):
        raise ValueError("Candidate resolution counts are stale")
    if plan.get("ordered_candidate_pair_keys_sha256") != pre_hf.sha256_value(
        [list(key) for key in pair_keys]
    ) or plan.get("ordered_candidate_excluded_base_fact_ids_sha256") != pre_hf.sha256_value(
        excluded_ids
    ):
        raise ValueError("Candidate resolution lineage digests are stale")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or not all(
        (
            safety.get("human_gold") is False,
            safety.get("candidate_policy_relaxed") is False,
            safety.get("cross_split_candidates_allowed") is False,
            safety.get("same_component_candidates_allowed") is False,
            safety.get("hf_model_executed") is False,
            safety.get("hf_tokenizer_executed") is False,
            safety.get("behavior_executed") is False,
            safety.get("validation_exposed") is False,
            safety.get("sealed_exposed") is False,
        )
    ):
        raise ValueError("Candidate resolution safety contract is invalid")
    if resolution_schema == RESOLUTION_MANIFEST_SCHEMA:
        validate_candidate_evidence_reuse_policy(
            manifest.get("review_invalidation"),
            label="Candidate resolution",
        )
    else:
        legacy_invalidation = manifest.get("review_invalidation")
        if not isinstance(legacy_invalidation, dict) or not all(
            (
                legacy_invalidation.get(
                    "predecessor_candidate_reviews_valid_for_successor"
                )
                is False,
                legacy_invalidation.get("full_successor_candidate_rereview_required")
                is True,
                legacy_invalidation.get(
                    "candidate_id_or_row_sha_alone_sufficient_for_reuse"
                )
                is False,
            )
        ):
            raise ValueError("Legacy candidate resolution invalidation contract is stale")
    predecessor = _load_postreview(predecessor_path)
    root = _load_root_context(root_path)
    if predecessor_binding.get("schema_version") != predecessor["manifest"].get(
        "schema_version"
    ):
        raise ValueError("Candidate resolution predecessor schema binding is stale")
    if root_binding.get("schema_version") != root["root"]["manifest"].get(
        "schema_version"
    ):
        raise ValueError("Candidate resolution root schema binding is stale")
    previous_binding = inputs.get("previous_candidate_resolution_manifest")
    predecessor_repair = predecessor["manifest"].get("inputs", {}).get(
        "candidate_repair"
    )
    if previous_binding is None:
        if predecessor_repair is not None:
            raise ValueError(
                "Candidate resolution omitted predecessor repair lineage"
            )
        previous_pair_keys: List[Tuple[str, str, str]] = []
        previous_pair_rows: List[Dict[str, Any]] = []
        previous_excluded_ids: List[str] = []
        previous_exclusion_rows: List[Dict[str, Any]] = []
    else:
        previous_path, _, previous_verified_binding = _verify_binding(
            previous_binding,
            owner_path=path,
            label="previous candidate resolution manifest",
        )
        previous = load_resolution_context(
            resolution_manifest_path=previous_path,
            _lineage_stack=lineage_stack,
        )
        if previous_verified_binding.get("sha256") != previous[
            "manifest_binding"
        ].get("sha256"):
            raise ValueError("Previous candidate resolution binding is stale")
        if previous_verified_binding.get("schema_version") != previous[
            "resolution_schema_version"
        ]:
            raise ValueError("Previous candidate resolution schema binding is stale")
        if previous["root_path"] != root_path:
            raise ValueError("Previous candidate resolution has a different root")
        if not isinstance(predecessor_repair, dict):
            raise ValueError(
                "Predecessor postreview does not bind the declared previous resolution"
            )
        predecessor_previous_path, _, predecessor_previous_binding = _verify_binding(
            predecessor_repair.get("candidate_resolution_manifest"),
            owner_path=predecessor_path,
            label="predecessor previous candidate resolution manifest",
            explicit_path=previous_path,
        )
        if (
            predecessor_previous_path != previous["manifest_path"]
            or predecessor_previous_binding.get("sha256")
            != previous["manifest_binding"].get("sha256")
        ):
            raise ValueError(
                "Predecessor postreview does not bind the declared previous resolution"
            )
        previous_pair_keys = list(previous["pair_keys"])
        previous_pair_rows = [dict(row) for row in previous["pair_rows"]]
        previous_excluded_ids = list(previous["candidate_excluded_ids"])
        previous_exclusion_rows = [
            dict(row) for row in previous["cohort_exclusion_rows"]
        ]

    review_binding = inputs.get("candidate_review_manifest")
    adjudication_input_binding = inputs.get("candidate_review_adjudications")
    checkpoint_input_binding = inputs.get("candidate_review_checkpoint")
    reviewed_nonaccept_keys: List[Tuple[str, str, str]] = []
    reviewed_nonaccept_rows: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    review_decision_counts: Counter[str] = Counter()
    review: Optional[Dict[str, Any]] = None
    if review_binding is None:
        if adjudication_input_binding is not None or checkpoint_input_binding is not None:
            raise ValueError("Candidate resolution evidence lacks a review manifest")
        if not predecessor["shortfalls"]:
            raise ValueError(
                "Candidate resolution without review requires predecessor shortfalls"
            )
    else:
        review_path = _resolve_path(
            review_binding, path, "candidate resolution candidate review"
        )
        review = _load_candidate_review(
            predecessor_path=predecessor_path,
            review_manifest_path=review_path,
            allow_legacy_v2=(
                resolution_schema == LEGACY_RESOLUTION_MANIFEST_SCHEMA
            ),
        )
        if review_binding.get("sha256") != review["manifest_binding"].get("sha256"):
            raise ValueError("Candidate resolution review manifest binding is stale")
        if not isinstance(adjudication_input_binding, dict) or adjudication_input_binding.get(
            "sha256"
        ) != review["adjudications_binding"].get("sha256") or _resolve_path(
            adjudication_input_binding,
            path,
            "candidate resolution adjudications",
        ) != review["adjudications_path"]:
            raise ValueError("Candidate resolution adjudication binding is stale")
        if resolution_schema == RESOLUTION_MANIFEST_SCHEMA:
            if not isinstance(checkpoint_input_binding, dict) or checkpoint_input_binding.get(
                "sha256"
            ) != review["checkpoint_binding"].get("sha256") or _resolve_path(
                checkpoint_input_binding,
                path,
                "candidate resolution checkpoint",
            ) != review["checkpoint_path"]:
                raise ValueError("Candidate resolution checkpoint binding is stale")
        elif checkpoint_input_binding is not None:
            raise ValueError("Legacy candidate resolution unexpectedly binds a checkpoint")
        reviewed_nonaccept_keys = sorted(
            (
                str(row["candidate_kind"]),
                str(row["target_base_fact_id"]),
                str(row["source_base_fact_id"]),
            )
            for row in review["adjudications"]
            if row.get("overall_decision") in {"reject", "defer"}
        )
        review_decision_counts.update(
            str(row.get("overall_decision")) for row in review["adjudications"]
        )
        for row in review["adjudications"]:
            if row.get("overall_decision") not in {"reject", "defer"}:
                continue
            expected_row = _pair_exclusion_from_adjudication(row)
            key = (
                str(expected_row["candidate_kind"]),
                str(expected_row["target_base_fact_id"]),
                str(expected_row["source_base_fact_id"]),
            )
            reviewed_nonaccept_rows[key] = expected_row
    target_ineligibility_evidence = (
        _target_ineligibility_evidence(review["adjudications"])
        if target_ineligibility_enabled and review is not None
        else {}
    )
    expected_target_ineligibility_contract = _target_ineligibility_contract(
        target_ineligibility_evidence
    )
    if target_ineligibility_enabled:
        if (
            manifest.get("target_ineligibility_contract")
            != expected_target_ineligibility_contract
        ):
            raise ValueError("Candidate target-ineligibility contract is stale")
    repeated_reviewed_keys = set(previous_pair_keys).intersection(
        reviewed_nonaccept_keys
    )
    if repeated_reviewed_keys:
        raise ValueError(
            "Candidate review repeats an already excluded pair: "
            f"{sorted(repeated_reviewed_keys)[0]}"
        )
    if not set(previous_pair_keys).issubset(pair_keys):
        raise ValueError("Candidate resolution dropped a prior pair exclusion")
    if set(pair_keys) != set(previous_pair_keys) | set(reviewed_nonaccept_keys):
        raise ValueError("Candidate pair exclusions do not match review lineage")
    expected_pair_index = {
        (
            str(row["candidate_kind"]),
            str(row["target_base_fact_id"]),
            str(row["source_base_fact_id"]),
        ): dict(row)
        for row in previous_pair_rows
    }
    expected_pair_index.update(reviewed_nonaccept_rows)
    expected_pair_rows = [expected_pair_index[key] for key in sorted(expected_pair_index)]
    if pair_rows != expected_pair_rows:
        raise ValueError("Candidate pair exclusion ledger differs from review lineage")
    if not set(previous_excluded_ids).issubset(excluded_ids):
        raise ValueError("Candidate resolution dropped a prior cohort exclusion")
    target_ineligible_ids = sorted(target_ineligibility_evidence)
    repeated_target_ineligibility_ids = set(previous_excluded_ids).intersection(
        target_ineligible_ids
    )
    if repeated_target_ineligibility_ids:
        raise ValueError(
            "Target ineligibility repeats a prior cohort exclusion: "
            f"{sorted(repeated_target_ineligibility_ids)[0]}"
        )
    state = _resolve_fixed_point(
        root=root,
        excluded_pairs=set(pair_keys),
        prior_candidate_excluded_ids=previous_excluded_ids,
        target_ineligible_ids=target_ineligible_ids,
    )
    if state["candidate_excluded_ids"] != excluded_ids:
        raise ValueError("Candidate resolution cohort-exclusion fixed point is stale")
    newly_excluded_ids = sorted(
        set(state["candidate_excluded_ids"]) - set(previous_excluded_ids)
    )
    expected_exclusion_rows = [dict(row) for row in previous_exclusion_rows]
    if target_ineligible_ids and review is None:
        raise ValueError("Target ineligibility evidence lacks a candidate review")
    expected_exclusion_rows.extend(
        _candidate_target_ineligibility_row(
            base_fact_id=base_fact_id,
            evidence=target_ineligibility_evidence[base_fact_id],
            review=review,
            root=root,
        )
        for base_fact_id in target_ineligible_ids
    )
    availability_excluded_ids = sorted(
        set(newly_excluded_ids) - set(target_ineligible_ids)
    )
    expected_exclusion_rows.extend(
        _candidate_cohort_exclusion_row(
            base_fact_id=base_fact_id,
            shortfall=state["latest_shortfall_by_id"][base_fact_id],
            root=root,
        )
        for base_fact_id in availability_excluded_ids
    )
    expected_exclusion_rows.sort(key=lambda row: str(row["base_fact_id"]))
    if exclusion_rows != expected_exclusion_rows:
        raise ValueError(
            "Candidate cohort exclusion ledger differs from fixed-point replay"
        )
    if _state_plan(
        state,
        pair_keys,
        target_ineligibility_evidence if target_ineligibility_enabled else None,
    ) != plan:
        raise ValueError("Candidate resolution deterministic plan is stale")
    if manifest.get("fixed_point_rounds") != state["rounds"]:
        raise ValueError("Candidate resolution fixed-point rounds are stale")
    expected_counts = {
        "root_fact_review_retained_count": len(root["fact_retained_ids"]),
        "predecessor_retained_count": len(predecessor["rows"]),
        "candidate_pair_exclusion_count": len(pair_rows),
        "new_candidate_pair_exclusion_count": len(pair_rows)
        - len(previous_pair_rows),
        "candidate_cohort_exclusion_count": len(exclusion_rows),
        "new_candidate_cohort_exclusion_count": len(exclusion_rows)
        - len(previous_exclusion_rows),
        "final_retained_count": len(state["retained_ids"]),
        "new_review_decision_counts": dict(sorted(review_decision_counts.items())),
    }
    if target_ineligibility_enabled:
        expected_counts.update(
            {
                "candidate_target_ineligibility_exclusion_count": sum(
                    row.get("exclusion_stage") == TARGET_INELIGIBILITY_STAGE
                    for row in exclusion_rows
                ),
                "new_candidate_target_ineligibility_exclusion_count": len(
                    target_ineligible_ids
                ),
            }
        )
    if any(counts.get(field) != value for field, value in expected_counts.items()):
        raise ValueError("Candidate resolution counts are stale")
    expected_resolution_id = "candidate_resolution_" + pre_hf.sha256_value(
        {
            "root_postreview_sha256": root_binding["sha256"],
            "predecessor_postreview_sha256": predecessor_binding["sha256"],
            "plan": plan,
        }
    )[:24]
    if manifest.get("resolution_id") != expected_resolution_id:
        raise ValueError("Candidate resolution ID is stale")
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(
            path, schema_version=str(resolution_schema)
        ),
        "predecessor_path": predecessor_path,
        "predecessor_binding": predecessor_binding,
        "root_path": root_path,
        "root_binding": root_binding,
        "pair_rows": pair_rows,
        "pair_keys": pair_keys,
        "pair_path": pair_path,
        "pair_binding": pair_binding,
        "cohort_exclusion_rows": exclusion_rows,
        "candidate_excluded_ids": excluded_ids,
        "previous_candidate_excluded_ids": previous_excluded_ids,
        "target_ineligibility_enabled": target_ineligibility_enabled,
        "target_ineligibility_evidence": target_ineligibility_evidence,
        "new_target_ineligible_ids": target_ineligible_ids,
        "all_target_ineligible_ids": sorted(
            str(row["base_fact_id"])
            for row in exclusion_rows
            if row.get("exclusion_stage") == TARGET_INELIGIBILITY_STAGE
        ),
        "cohort_exclusion_path": exclusion_path,
        "cohort_exclusion_binding": exclusion_binding,
        "resolution_id": _required_string(
            manifest.get("resolution_id"), "candidate resolution_id"
        ),
        "resolution_schema_version": str(resolution_schema),
        "legacy_migration_only": resolution_schema == LEGACY_RESOLUTION_MANIFEST_SCHEMA,
        "plan": plan,
    }


def _prior_resolution(predecessor: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    inputs = predecessor["manifest"].get("inputs")
    candidate_repair = inputs.get("candidate_repair") if isinstance(inputs, dict) else None
    if candidate_repair is None:
        return None
    if not isinstance(candidate_repair, dict):
        raise ValueError("Postreview candidate_repair input is invalid")
    resolution_binding = candidate_repair.get("candidate_resolution_manifest")
    resolution_path = _resolve_path(
        resolution_binding,
        predecessor["path"],
        "postreview candidate resolution",
    )
    context = load_resolution_context(resolution_manifest_path=resolution_path)
    if context["manifest_binding"]["sha256"] != resolution_binding.get("sha256"):
        raise ValueError("Postreview candidate resolution binding is stale")
    if context["predecessor_path"] == predecessor["path"]:
        raise ValueError("Candidate repair lineage is self-referential")
    return context


def _pair_exclusion_from_adjudication(row: Mapping[str, Any]) -> Dict[str, Any]:
    decision = str(row["overall_decision"])
    if decision not in {"reject", "defer"}:
        raise ValueError("Only reject/defer can create a pair exclusion")
    return {
        "schema_version": PAIR_EXCLUSION_SCHEMA,
        "candidate_kind": str(row["candidate_kind"]),
        "target_base_fact_id": str(row["target_base_fact_id"]),
        "source_base_fact_id": str(row["source_base_fact_id"]),
        "source_candidate_id": str(row["candidate_id"]),
        "source_candidate_row_sha256": str(row["candidate_row_sha256"]),
        "source_adjudication_row_sha256": pre_hf.sha256_value(row),
        "source_overall_decision": decision,
        "pair_disposition": "exclude_pair_from_candidate_selection",
        "defer_treated_as_non_accept_not_false": decision == "defer",
        "source_fact_falsehood_asserted": False,
        "human_gold": False,
    }


def _filtered_semantic(
    *,
    candidates: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    retained_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    kept_decisions: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        pair_id = str(candidate["pair_id"])
        if not _semantic_current_ids(candidate).issubset(retained_ids):
            continue
        kept.append(dict(candidate))
        kept_decisions[pair_id] = dict(decisions[pair_id])
    return kept, kept_decisions


def _candidate_repair_rows(
    *,
    root_rows_by_id: Mapping[str, Mapping[str, Any]],
    retained_ids: Set[str],
    component_by_fact: Mapping[str, str],
    split_by_fact: Mapping[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base_fact_id in sorted(retained_ids):
        row = copy.deepcopy(dict(root_rows_by_id[base_fact_id]))
        row["leakage_component_id"] = component_by_fact[base_fact_id]
        row["split_assignment"] = split_by_fact[base_fact_id]
        row["split_policy_version"] = pre_hf.SPLIT_POLICY_VERSION
        row["split_status"] = "provisional_not_frozen"
        row["canonical_status"] = "pending_review"
        row["human_gold"] = False
        row["distractor_status"] = (
            "regenerated_after_candidate_resolution_pending_full_rereview"
        )
        row["translation_status"] = (
            "pending_translation_or_rebind_to_final_candidates"
        )
        row["hf_model_execution_status"] = "not_run"
        row["hf_tokenizer_execution_status"] = "not_run"
        rows.append(row)
    return rows


def _resolve_fixed_point(
    *,
    root: Mapping[str, Any],
    excluded_pairs: Set[Tuple[str, str, str]],
    prior_candidate_excluded_ids: Sequence[str],
    target_ineligible_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    root_ids = set(root["fact_retained_ids"])
    prior_excluded = set(prior_candidate_excluded_ids)
    target_ineligible = set(target_ineligible_ids)
    if prior_excluded.intersection(target_ineligible):
        raise ValueError("New target ineligibility repeats a prior cohort exclusion")
    candidate_excluded = prior_excluded | target_ineligible
    if not candidate_excluded.issubset(root_ids):
        raise ValueError("Candidate exclusions are outside the root retained cohort")
    rounds: List[Dict[str, Any]] = []
    latest_shortfall_by_id: Dict[str, Dict[str, Any]] = {}
    while True:
        retained = root_ids - candidate_excluded
        if not retained:
            raise ValueError("Candidate repair would exclude the entire retained cohort")
        semantic_candidates, semantic_decisions = _filtered_semantic(
            candidates=root["semantic_candidates"],
            decisions=root["semantic_decisions"],
            retained_ids=retained,
        )
        component_rejected = set(root["source_index"]) - retained
        components, component_by_fact, component_audit = postreview.build_reviewed_components(
            full_rows=root["source_rows"],
            retained_ids=sorted(retained),
            rejected_ids=sorted(component_rejected),
            current_components=root["semantic_components"],
            semantic_candidates=semantic_candidates,
            semantic_decisions=semantic_decisions,
            frozen=root["frozen"],
        )
        component_assignments, split_audit = pre_hf.assign_component_splits(
            components, root["semantic_split_manifest"]["split_seed"]
        )
        split_by_fact: Dict[str, str] = {}
        for component in components:
            component_id = str(component["leakage_component_id"])
            for base_fact_id in component["base_fact_ids"]:
                split_by_fact[str(base_fact_id)] = component_assignments[component_id]
        rows = _candidate_repair_rows(
            root_rows_by_id=root["fact_base_index"],
            retained_ids=retained,
            component_by_fact=component_by_fact,
            split_by_fact=split_by_fact,
        )
        row_index = _index_unique(rows, "base_fact_id", "candidate repair rows")
        relation_by_fact = {
            base_fact_id: {
                "relation_partition_id": row["relation_partition_id"],
                "answer_type_bucket": row["answer_type_bucket"],
            }
            for base_fact_id, row in row_index.items()
        }
        distractors, neutrals, selection = pre_hf.select_static_candidates(
            rows,
            relation_by_fact,
            component_by_fact,
            split_by_fact,
            excluded_candidate_pairs=excluded_pairs,
        )
        distractor_ids = selection.pop("distractor_ids")
        neutral_ids = selection.pop("neutral_ids")
        shortfalls = postreview._candidate_shortfalls(
            base_fact_ids=list(row_index),
            rows_by_id=row_index,
            distractor_ids=distractor_ids,
            neutral_ids=neutral_ids,
        )
        if not shortfalls:
            return {
                "retained_ids": sorted(retained),
                "candidate_excluded_ids": sorted(candidate_excluded),
                "components": components,
                "component_by_fact": component_by_fact,
                "component_audit": component_audit,
                "split_by_fact": split_by_fact,
                "split_audit": split_audit,
                "rows": rows,
                "row_index": row_index,
                "distractors": distractors,
                "neutrals": neutrals,
                "selection": selection,
                "semantic_candidates": semantic_candidates,
                "semantic_decisions": semantic_decisions,
                "rounds": rounds,
                "latest_shortfall_by_id": latest_shortfall_by_id,
            }
        new_ids = sorted(
            {str(row["base_fact_id"]) for row in shortfalls} - candidate_excluded
        )
        if not new_ids:
            raise ValueError("Candidate repair fixed point made no progress")
        round_number = len(rounds) + 1
        for row in shortfalls:
            base_fact_id = str(row["base_fact_id"])
            if base_fact_id in new_ids:
                latest_shortfall_by_id[base_fact_id] = {
                    **copy.deepcopy(dict(row)),
                    "fixed_point_round": round_number,
                }
        rounds.append(
            {
                "round": round_number,
                "input_retained_count": len(retained),
                "new_candidate_cohort_exclusion_count": len(new_ids),
                "new_candidate_cohort_excluded_base_fact_ids_sha256": pre_hf.sha256_value(
                    new_ids
                ),
            }
        )
        candidate_excluded.update(new_ids)


def _state_plan(
    state: Mapping[str, Any],
    pair_keys: Sequence[Tuple[str, str, str]],
    target_ineligibility_evidence: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    result = {
        "final_retained_base_fact_count": len(state["retained_ids"]),
        "candidate_cohort_exclusion_count": len(state["candidate_excluded_ids"]),
        "distractor_candidate_count": len(state["distractors"]),
        "neutral_candidate_count": len(state["neutrals"]),
        "candidate_shortfall_count": 0,
        "fixed_point_round_count": len(state["rounds"]),
        "ordered_candidate_pair_keys_sha256": pre_hf.sha256_value(
            [list(key) for key in pair_keys]
        ),
        "ordered_candidate_excluded_base_fact_ids_sha256": pre_hf.sha256_value(
            state["candidate_excluded_ids"]
        ),
        "ordered_final_retained_base_fact_ids_sha256": pre_hf.sha256_value(
            state["retained_ids"]
        ),
        "ordered_component_assignments_sha256": pre_hf.sha256_value(
            [
                [row["base_fact_id"], row["leakage_component_id"]]
                for row in state["rows"]
            ]
        ),
        "ordered_split_assignments_sha256": pre_hf.sha256_value(
            [[row["base_fact_id"], row["split_assignment"]] for row in state["rows"]]
        ),
        "ordered_distractor_source_assignments_sha256": pre_hf.sha256_value(
            [
                [
                    row["base_fact_id"],
                    row["source_base_fact_id"],
                    row["slot"],
                ]
                for row in state["distractors"]
            ]
        ),
        "ordered_neutral_source_assignments_sha256": pre_hf.sha256_value(
            [
                [
                    row["base_fact_id"],
                    row["source_base_fact_id"],
                    row["slot"],
                ]
                for row in state["neutrals"]
            ]
        ),
    }
    if target_ineligibility_evidence is not None:
        target_ids = sorted(target_ineligibility_evidence)
        ordered_evidence = [
            dict(target_ineligibility_evidence[target_id])
            for target_id in target_ids
        ]
        result.update(
            {
                "new_target_ineligibility_exclusion_count": len(target_ids),
                "ordered_new_target_ineligibility_base_fact_ids_sha256": pre_hf.sha256_value(
                    target_ids
                ),
                "ordered_target_ineligibility_evidence_sha256": pre_hf.sha256_value(
                    ordered_evidence
                ),
            }
        )
    return result


def materialize_resolution(
    *,
    predecessor_postreview_manifest_path: Path,
    output_dir: Path,
    candidate_review_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    predecessor = _load_postreview(predecessor_postreview_manifest_path)
    prior = _prior_resolution(predecessor)
    root_path = prior["root_path"] if prior is not None else predecessor["path"]
    root = _load_root_context(root_path)
    if prior is not None and prior["root_path"] != root["root"]["path"]:
        raise ValueError("Candidate repair root lineage is inconsistent")
    prior_pair_rows = list(prior["pair_rows"]) if prior is not None else []
    prior_exclusion_rows = (
        list(prior["cohort_exclusion_rows"]) if prior is not None else []
    )
    pair_index = {
        (
            str(row["candidate_kind"]),
            str(row["target_base_fact_id"]),
            str(row["source_base_fact_id"]),
        ): dict(row)
        for row in prior_pair_rows
    }
    review: Optional[Dict[str, Any]] = None
    target_ineligibility_evidence: Dict[str, Dict[str, Any]] = {}
    new_decision_counts: Counter[str] = Counter()
    if candidate_review_manifest_path is not None:
        review = _load_candidate_review(
            predecessor_path=predecessor["path"],
            review_manifest_path=Path(candidate_review_manifest_path),
        )
        for row in review["adjudications"]:
            decision = str(row["overall_decision"])
            new_decision_counts[decision] += 1
            if decision == "accept":
                continue
            pair_row = _pair_exclusion_from_adjudication(row)
            key = (
                str(pair_row["candidate_kind"]),
                str(pair_row["target_base_fact_id"]),
                str(pair_row["source_base_fact_id"]),
            )
            if key in pair_index:
                raise ValueError(f"Candidate review repeats an already excluded pair: {key}")
            pair_index[key] = pair_row
        target_ineligibility_evidence = _target_ineligibility_evidence(
            review["adjudications"]
        )
    elif not predecessor["shortfalls"]:
        raise ValueError(
            "A candidate review is required when the predecessor has no structural shortfall"
        )

    prior_excluded_ids = {
        str(row["base_fact_id"]) for row in prior_exclusion_rows
    }
    repeated_target_ineligibility_ids = prior_excluded_ids.intersection(
        target_ineligibility_evidence
    )
    if repeated_target_ineligibility_ids:
        raise ValueError(
            "Target ineligibility repeats a prior cohort exclusion: "
            f"{sorted(repeated_target_ineligibility_ids)[0]}"
        )
    target_ineligible_ids = sorted(target_ineligibility_evidence)
    pair_keys = sorted(pair_index)
    state = _resolve_fixed_point(
        root=root,
        excluded_pairs=set(pair_keys),
        prior_candidate_excluded_ids=sorted(prior_excluded_ids),
        target_ineligible_ids=target_ineligible_ids,
    )
    newly_excluded_ids = sorted(
        set(state["candidate_excluded_ids"]) - prior_excluded_ids
    )
    availability_excluded_ids = sorted(
        set(newly_excluded_ids) - set(target_ineligible_ids)
    )
    if not newly_excluded_ids and not (
        set(pair_keys)
        - {
            (
                str(row["candidate_kind"]),
                str(row["target_base_fact_id"]),
                str(row["source_base_fact_id"]),
            )
            for row in prior_pair_rows
        }
    ):
        raise ValueError("Candidate resolution is a no-op")

    plan = _state_plan(state, pair_keys, target_ineligibility_evidence)
    resolution_id = "candidate_resolution_" + pre_hf.sha256_value(
        {
            "root_postreview_sha256": root["root"]["binding"]["sha256"],
            "predecessor_postreview_sha256": predecessor["binding"]["sha256"],
            "plan": plan,
        }
    )[:24]
    cohort_rows = [copy.deepcopy(dict(row)) for row in prior_exclusion_rows]
    if target_ineligible_ids and review is None:
        raise ValueError("Target ineligibility evidence lacks a candidate review")
    for base_fact_id in target_ineligible_ids:
        cohort_rows.append(
            _candidate_target_ineligibility_row(
                base_fact_id=base_fact_id,
                evidence=target_ineligibility_evidence[base_fact_id],
                review=review,
                root=root,
            )
        )
    for base_fact_id in availability_excluded_ids:
        cohort_rows.append(
            _candidate_cohort_exclusion_row(
                base_fact_id=base_fact_id,
                shortfall=state["latest_shortfall_by_id"][base_fact_id],
                root=root,
            )
        )
    cohort_rows.sort(key=lambda row: str(row["base_fact_id"]))
    pair_rows = [pair_index[key] for key in pair_keys]

    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        pair_path = staged_dir / "candidate_pair_exclusions.jsonl"
        exclusion_path = staged_dir / "candidate_cohort_exclusions.jsonl"
        manifest_path = staged_dir / "candidate_resolution_manifest.json"
        pre_hf.write_jsonl(pair_path, pair_rows)
        pre_hf.write_jsonl(exclusion_path, cohort_rows)
        inputs: Dict[str, Any] = {
            "root_postreview_manifest": root["root"]["binding"],
            "predecessor_postreview_manifest": predecessor["binding"],
            "previous_candidate_resolution_manifest": (
                prior["manifest_binding"] if prior is not None else None
            ),
            "candidate_review_manifest": (
                review["manifest_binding"] if review is not None else None
            ),
            "candidate_review_adjudications": (
                review["adjudications_binding"] if review is not None else None
            ),
            "candidate_review_checkpoint": (
                review["checkpoint_binding"] if review is not None else None
            ),
        }
        manifest = {
            "schema_version": RESOLUTION_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": RESOLUTION_STATUS,
            "resolution_id": resolution_id,
            "inputs": inputs,
            "outputs": {
                "candidate_pair_exclusions": _binding(
                    pair_path,
                    schema_version=PAIR_EXCLUSION_SCHEMA,
                    record_count=len(pair_rows),
                    relative_to=staged_dir,
                ),
                "candidate_cohort_exclusions": _binding(
                    exclusion_path,
                    schema_version=COHORT_EXCLUSION_SCHEMA,
                    record_count=len(cohort_rows),
                    relative_to=staged_dir,
                ),
            },
            "counts": {
                "root_fact_review_retained_count": len(root["fact_retained_ids"]),
                "predecessor_retained_count": len(predecessor["rows"]),
                "candidate_pair_exclusion_count": len(pair_rows),
                "new_candidate_pair_exclusion_count": len(pair_rows)
                - len(prior_pair_rows),
                "candidate_cohort_exclusion_count": len(cohort_rows),
                "new_candidate_cohort_exclusion_count": len(newly_excluded_ids),
                "candidate_target_ineligibility_exclusion_count": sum(
                    row.get("exclusion_stage") == TARGET_INELIGIBILITY_STAGE
                    for row in cohort_rows
                ),
                "new_candidate_target_ineligibility_exclusion_count": len(
                    target_ineligible_ids
                ),
                "final_retained_count": len(state["retained_ids"]),
                "new_review_decision_counts": dict(sorted(new_decision_counts.items())),
            },
            "fixed_point_rounds": state["rounds"],
            "resolution_plan": plan,
            "target_ineligibility_contract": _target_ineligibility_contract(
                target_ineligibility_evidence
            ),
            "review_invalidation": {
                **CANDIDATE_EVIDENCE_REUSE_POLICY,
                "predecessor_zh_reviews_valid_for_successor": False,
                "full_successor_zh_translation_and_review_required": True,
            },
            "safety_contract": {
                "offline_only": True,
                "candidate_policy_relaxed": False,
                "cross_split_candidates_allowed": False,
                "same_component_candidates_allowed": False,
                "relation_partition_promoted_to_probe_relation_id": False,
                "source_fact_falsehood_inferred_from_pair_rejection": False,
                "candidate_cohort_exclusion_asserts_factual_falsehood": False,
                "human_gold": False,
                "hf_model_executed": False,
                "hf_tokenizer_executed": False,
                "behavior_executed": False,
                "validation_exposed": False,
                "sealed_exposed": False,
            },
        }
        pre_hf.write_json(manifest_path, manifest)
        # Revalidate the exact staged bytes before the atomic publish.
        load_resolution_context(resolution_manifest_path=manifest_path)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        staged_dir.replace(output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged_dir, ignore_errors=True)
    final_manifest = output_dir / "candidate_resolution_manifest.json"
    return {
        "status": RESOLUTION_STATUS,
        "manifest_path": str(final_manifest),
        "manifest_sha256": pre_hf.sha256_file(final_manifest),
        "resolution_id": resolution_id,
        "candidate_pair_exclusion_count": len(pair_rows),
        "candidate_cohort_exclusion_count": len(cohort_rows),
        "candidate_target_ineligibility_exclusion_count": sum(
            row.get("exclusion_stage") == TARGET_INELIGIBILITY_STAGE
            for row in cohort_rows
        ),
        "final_retained_count": len(state["retained_ids"]),
        "candidate_shortfall_count": 0,
        "human_gold": False,
    }


def _verify_plan(state: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    expected = _state_plan(
        state,
        context["pair_keys"],
        context["target_ineligibility_evidence"]
        if context["target_ineligibility_enabled"]
        else None,
    )
    if expected != context["plan"]:
        raise ValueError("Candidate resolution deterministic plan is stale")


def _candidate_postreview_exclusion(
    row: Mapping[str, Any], *, resolution_id: str
) -> Dict[str, Any]:
    return {
        "schema_version": postreview.EXCLUSION_SCHEMA_VERSION,
        "base_fact_id": row["base_fact_id"],
        "disposition": "cohort_exclude",
        "exclusion_stage": row["exclusion_stage"],
        "source_review_outcome": "accept",
        "source_staging_status": None,
        "review_item_sha256": None,
        "review_provenance": {
            "reviewer_type": "deterministic_candidate_resolution",
            "resolution_id": resolution_id,
        },
        "notes": row["exclusion_reason"],
        "source_candidate_resolution_row_sha256": pre_hf.sha256_value(row),
        "materializer_asserts_factual_falsehood": False,
        "human_gold": False,
    }


def materialize_successor(
    *,
    predecessor_postreview_manifest_path: Path,
    candidate_resolution_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    predecessor = _load_postreview(predecessor_postreview_manifest_path)
    resolution = load_resolution_context(
        resolution_manifest_path=candidate_resolution_manifest_path,
        predecessor_postreview_manifest_path=predecessor["path"],
    )
    target_ineligibility_enabled = bool(
        resolution["target_ineligibility_enabled"]
    )
    successor_tool_version = (
        TOOL_VERSION if target_ineligibility_enabled else LEGACY_TOOL_VERSION
    )
    root = _load_root_context(resolution["root_path"])
    if Path(resolution["root_path"]).resolve() != Path(
        root["root"]["path"]
    ).resolve():
        raise ValueError("Candidate resolution root postreview binding is stale")
    state = _resolve_fixed_point(
        root=root,
        excluded_pairs=set(resolution["pair_keys"]),
        prior_candidate_excluded_ids=resolution[
            "previous_candidate_excluded_ids"
        ],
        target_ineligible_ids=resolution["new_target_ineligible_ids"],
    )
    _verify_plan(state, resolution)
    if state["candidate_excluded_ids"] != resolution["candidate_excluded_ids"]:
        raise ValueError("Candidate resolution exclusion fixed point is stale")
    if state["retained_ids"] == [str(row["base_fact_id"]) for row in predecessor["rows"]] and (
        state["distractors"] == predecessor["distractors"]
        and state["neutrals"] == predecessor["neutrals"]
    ):
        raise ValueError("Candidate successor would not change the predecessor")

    reviewed_rows = state["rows"]
    for row in reviewed_rows:
        row["candidate_repair"] = {
            "resolution_id": resolution["resolution_id"],
            "source_root_postreview_row_sha256": pre_hf.sha256_value(
                root["fact_base_index"][str(row["base_fact_id"])]
            ),
            "candidate_reviews_reused": False,
            "exact_accepted_candidate_evidence_carry_forward_eligible": True,
            "zh_reviews_reused": False,
            "human_gold": False,
        }
    candidate_exclusion_rows = [
        _candidate_postreview_exclusion(
            row, resolution_id=resolution["resolution_id"]
        )
        for row in resolution["cohort_exclusion_rows"]
    ]
    candidate_target_ineligibility_exclusion_count = sum(
        row.get("exclusion_stage") == TARGET_INELIGIBILITY_STAGE
        for row in resolution["cohort_exclusion_rows"]
    )
    exclusions = [
        *[copy.deepcopy(dict(row)) for row in root["fact_exclusions"]],
        *candidate_exclusion_rows,
    ]
    all_excluded_ids = [str(row["base_fact_id"]) for row in exclusions]
    if len(all_excluded_ids) != len(set(all_excluded_ids)):
        raise ValueError("Fact-stage and candidate-stage exclusions overlap")

    semantic_candidates = state["semantic_candidates"]
    semantic_decisions = state["semantic_decisions"]
    frozen = root["frozen"]
    integrity = postreview._audit_outputs(
        input_full_ids=root["original_ids"],
        reviewed_rows=reviewed_rows,
        rejected_ids=all_excluded_ids,
        components=state["components"],
        semantic_candidates=semantic_candidates,
        semantic_decisions=semantic_decisions,
        distractors=state["distractors"],
        neutrals=state["neutrals"],
        shortfalls=[],
        frozen=frozen,
    )
    integrity["candidate_repair"] = {
        "resolution_id": resolution["resolution_id"],
        "pair_exclusion_count": len(resolution["pair_rows"]),
        "candidate_stage_cohort_exclusion_count": len(candidate_exclusion_rows),
        "fixed_point_reached": True,
        "policy_relaxed": False,
        "predecessor_candidate_reviews_reused": False,
        "predecessor_zh_reviews_reused": False,
    }
    if target_ineligibility_enabled:
        integrity["candidate_repair"][
            "candidate_target_ineligibility_exclusion_count"
        ] = candidate_target_ineligibility_exclusion_count

    split_seed = _required_string(
        root["semantic_split_manifest"].get("split_seed"), "split_seed"
    )
    frozen_retained_count = sum(
        base_fact_id in state["row_index"] for base_fact_id in frozen["split_by_fact"]
    )
    frozen_excluded_count = len(frozen["split_by_fact"]) - frozen_retained_count
    frozen_excluded_count += int(frozen.get("excluded_source_fact_count", 0))
    fact_reject_count = sum(
        row.get("source_review_outcome") == "reject"
        for row in root["fact_exclusions"]
    )
    split_manifest = {
        "schema_version": pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
        "tool_version": successor_tool_version,
        "split_policy_version": pre_hf.SPLIT_POLICY_VERSION,
        "split_seed": split_seed,
        "split_ratios": pre_hf.SPLIT_RATIOS,
        "split_status": "provisional_not_frozen",
        "formal_split_freeze_performed": False,
        "input_base_fact_count": len(root["original_ids"]),
        "base_fact_count": len(reviewed_rows),
        "fact_review_reject_excluded_count": fact_reject_count,
        "fact_resolution_cohort_excluded_count": len(root["fact_exclusions"]),
        "candidate_repair_cohort_excluded_count": len(candidate_exclusion_rows),
        "total_cohort_excluded_count": len(exclusions),
        "fact_resolution_revision_count": len(root["revised_ids"]),
        "leakage_component_count": len(state["components"]),
        "leakage_grouping_sources": [
            "preserved_pre_review_leakage_components",
            "adjudicated_same_fact_positive_edges",
            "adjudicated_same_leakage_component_positive_edges",
            "comparison_canonical_bridge_nodes",
            "candidate_resolution_induced_subgraph",
        ],
        "bounded_semantic_candidate_adjudication_complete": True,
        "semantic_paraphrase_closure_complete_for_full_pool": False,
        "semantic_near_duplicate_recall_guaranteed": False,
        "historical_exposure_complete_for_full_pool": False,
        "frozen_split_constraint": copy.deepcopy(frozen["binding"]),
        "retained_frozen_split_fact_count": frozen_retained_count,
        "excluded_frozen_split_fact_count": frozen_excluded_count,
        "frozen_split_mismatch_count": 0,
        "cross_component_split_violation_count": 0,
        "component_size_summary": postreview._component_size_summary(
            state["components"]
        ),
        "semantic_component_rebuild": state["component_audit"],
        **state["split_audit"],
    }
    if target_ineligibility_enabled:
        split_manifest["candidate_target_ineligibility_excluded_count"] = (
            candidate_target_ineligibility_exclusion_count
        )

    counts = {
        "input_base_facts": len(root["original_ids"]),
        "retained_base_facts": len(reviewed_rows),
        "rejected_base_facts": len(exclusions),
        "fact_stage_cohort_exclusions": len(root["fact_exclusions"]),
        "candidate_stage_cohort_exclusions": len(candidate_exclusion_rows),
        "revised_base_facts": len(root["revised_ids"]),
        "leakage_components": len(state["components"]),
        "semantic_candidate_pairs": len(semantic_candidates),
        "semantic_positive_edges": state["component_audit"][
            "positive_semantic_edge_count"
        ],
        "distractor_candidates": len(state["distractors"]),
        "neutral_reference_candidates": len(state["neutrals"]),
        "candidate_shortfalls": 0,
    }
    if target_ineligibility_enabled:
        counts["candidate_target_ineligibility_exclusions"] = (
            candidate_target_ineligibility_exclusion_count
        )
    summary = {
        "schema_version": postreview.SUMMARY_SCHEMA_VERSION,
        "tool_version": successor_tool_version,
        "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
        "counts": counts,
        "fact_review_outcome_counts": copy.deepcopy(
            root["root"]["manifest"].get("fact_review_outcome_counts") or {}
        ),
        "split": {
            "policy_version": pre_hf.SPLIT_POLICY_VERSION,
            "base_fact_counts": state["split_audit"]["actual_base_fact_counts"],
            "component_counts": state["split_audit"]["component_counts"],
            "formal_split_freeze_performed": False,
        },
        "static_candidates": state["selection"],
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

    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
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
        pre_hf.write_jsonl(paths["leakage_components"], state["components"])
        pre_hf.write_json(paths["split_manifest"], split_manifest)
        pre_hf.write_jsonl(paths["distractor_candidates"], state["distractors"])
        pre_hf.write_jsonl(paths["neutral_reference_candidates"], state["neutrals"])
        pre_hf.write_jsonl(paths["candidate_shortfalls"], [])
        pre_hf.write_json(paths["integrity_audit"], integrity)
        pre_hf.write_json(paths["summary"], summary)
        output_artifacts = {
            "full_base_facts": postreview._output_binding(
                paths["full_base_facts"], pre_hf.FULL_FACT_SCHEMA_VERSION, len(reviewed_rows)
            ),
            "cohort_exclusions": postreview._output_binding(
                paths["cohort_exclusions"], postreview.EXCLUSION_SCHEMA_VERSION, len(exclusions)
            ),
            "leakage_components": postreview._output_binding(
                paths["leakage_components"], pre_hf.COMPONENT_SCHEMA_VERSION, len(state["components"])
            ),
            "split_manifest": postreview._output_binding(
                paths["split_manifest"], pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION
            ),
            "distractor_candidates": postreview._output_binding(
                paths["distractor_candidates"], pre_hf.DISTRACTOR_SCHEMA_VERSION, len(state["distractors"])
            ),
            "neutral_reference_candidates": postreview._output_binding(
                paths["neutral_reference_candidates"], pre_hf.NEUTRAL_SCHEMA_VERSION, len(state["neutrals"])
            ),
            "candidate_shortfalls": postreview._output_binding(
                paths["candidate_shortfalls"], pre_hf.SHORTFALL_SCHEMA_VERSION, 0
            ),
            "integrity_audit": postreview._output_binding(
                paths["integrity_audit"], postreview.INTEGRITY_SCHEMA_VERSION
            ),
            "summary": postreview._output_binding(
                paths["summary"], postreview.SUMMARY_SCHEMA_VERSION
            ),
        }
        root_manifest = root["root"]["manifest"]
        root_fact_contract = copy.deepcopy(root_manifest["fact_review_contract"])
        root_fact_contract.update(
            {
                "selection_is_complete_retained_cohort": True,
                "fact_stage_retained_base_fact_count": len(root["fact_retained_ids"]),
                "accepted_facts_retained": len(reviewed_rows),
                "facts_cohort_excluded": len(exclusions),
                "fact_stage_cohort_excluded": len(root["fact_exclusions"]),
                "candidate_stage_cohort_excluded": len(candidate_exclusion_rows),
            }
        )
        if target_ineligibility_enabled:
            root_fact_contract["candidate_target_ineligibility_excluded"] = (
                candidate_target_ineligibility_exclusion_count
            )
        root_semantic_contract = copy.deepcopy(root_manifest["semantic_review_contract"])
        root_semantic_contract.update(
            {
                "candidate_stage_exclusion_induced_subgraph_used": True,
                "new_pairs_after_candidate_stage_exclusion_enumerated": False,
                "bounded_recall_claim_not_strengthened": True,
            }
        )
        root_inputs = root_manifest["inputs"]
        manifest = {
            "schema_version": SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA,
            "tool_version": successor_tool_version,
            "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
            "inputs": {
                "full_base_facts": copy.deepcopy(root_inputs["full_base_facts"]),
                "fact_review": copy.deepcopy(root_inputs["fact_review"]),
                "fact_resolution": copy.deepcopy(root_inputs.get("fact_resolution")),
                "semantic_review": copy.deepcopy(root_inputs["semantic_review"]),
                "frozen_split_constraint": copy.deepcopy(
                    root_inputs.get("frozen_split_constraint")
                ),
                "candidate_repair": {
                    "root_postreview_manifest": root["root"]["binding"],
                    "predecessor_postreview_manifest": predecessor["binding"],
                    "candidate_resolution_manifest": resolution["manifest_binding"],
                    "candidate_pair_exclusions": _binding(
                        resolution["pair_path"],
                        schema_version=PAIR_EXCLUSION_SCHEMA,
                        record_count=len(resolution["pair_rows"]),
                    ),
                    "candidate_cohort_exclusions": _binding(
                        resolution["cohort_exclusion_path"],
                        schema_version=COHORT_EXCLUSION_SCHEMA,
                        record_count=len(resolution["cohort_exclusion_rows"]),
                    ),
                },
            },
            "outputs": output_artifacts,
            "counts": counts,
            "fact_review_contract": root_fact_contract,
            "semantic_review_contract": root_semantic_contract,
            "candidate_repair_contract": {
                "resolution_id": resolution["resolution_id"],
                "resolution_applied": True,
                "pair_exclusion_count": len(resolution["pair_rows"]),
                "candidate_stage_cohort_exclusion_count": len(
                    candidate_exclusion_rows
                ),
                "fixed_point_reached": True,
                "candidate_shortfall_count": 0,
                "candidate_policy_relaxed": False,
                **CANDIDATE_EVIDENCE_REUSE_POLICY,
                "predecessor_zh_reviews_valid": False,
                "full_zh_translation_and_review_required": True,
                "human_gold": False,
            },
            "policy_reuse": {
                **copy.deepcopy(root_manifest.get("policy_reuse") or {}),
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
                "candidate_pair_exclusion_overlay_applied": True,
            },
            "lineage_digests": {
                "ordered_input_base_fact_ids_sha256": pre_hf.sha256_value(
                    root["original_ids"]
                ),
                "ordered_fact_stage_retained_base_fact_ids_sha256": pre_hf.sha256_value(
                    root["fact_retained_ids"]
                ),
                "ordered_retained_base_fact_ids_sha256": pre_hf.sha256_value(
                    state["retained_ids"]
                ),
                "ordered_fact_stage_excluded_base_fact_ids_sha256": pre_hf.sha256_value(
                    root["fact_excluded_ids"]
                ),
                "ordered_candidate_stage_excluded_base_fact_ids_sha256": pre_hf.sha256_value(
                    state["candidate_excluded_ids"]
                ),
                "ordered_rejected_base_fact_ids_sha256": pre_hf.sha256_value(
                    all_excluded_ids
                ),
                "ordered_revised_base_fact_ids_sha256": pre_hf.sha256_value(
                    root["revised_ids"]
                ),
                "ordered_component_ids_sha256": pre_hf.sha256_value(
                    [row["leakage_component_id"] for row in state["components"]]
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
                    [row["distractor_id"] for row in state["distractors"]]
                ),
                "ordered_neutral_candidate_ids_sha256": pre_hf.sha256_value(
                    [row["neutral_candidate_id"] for row in state["neutrals"]]
                ),
                "ordered_candidate_pair_exclusions_sha256": pre_hf.sha256_value(
                    [list(key) for key in resolution["pair_keys"]]
                ),
            },
            "review_invalidation": {
                **copy.deepcopy(root_manifest.get("review_invalidation") or {}),
                **CANDIDATE_EVIDENCE_REUSE_POLICY,
                "predecessor_translation_reviews_valid": False,
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
                (
                    "cover the complete successor candidate set by carrying only exact "
                    "v3 accepted projection evidence and freshly reviewing changed or new candidates"
                ),
                "translate and review zh against the successor candidate bindings",
                "complete historical-exposure authority attestation",
                "emit exact-bundle review and split freeze manifests",
                "only then bind an exact HF checkpoint and tokenizer",
            ],
        }
        if target_ineligibility_enabled:
            manifest["candidate_repair_contract"].update(
                {
                    "candidate_target_ineligibility_exclusion_count": (
                        candidate_target_ineligibility_exclusion_count
                    ),
                    "target_ineligibility_policy_version": (
                        TARGET_INELIGIBILITY_POLICY_VERSION
                    ),
                }
            )
            manifest["lineage_digests"][
                "ordered_candidate_target_ineligible_base_fact_ids_sha256"
            ] = pre_hf.sha256_value(resolution["all_target_ineligible_ids"])
        pre_hf.write_json(paths["manifest"], manifest)
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
        "fact_stage_cohort_excluded_count": len(root["fact_exclusions"]),
        "candidate_stage_cohort_excluded_count": len(candidate_exclusion_rows),
        "candidate_target_ineligibility_exclusion_count": (
            candidate_target_ineligibility_exclusion_count
        ),
        "candidate_shortfall_count": 0,
        "candidate_evidence_coverage_required": True,
        "candidate_model_rereview_required": False,
        "zh_rereview_required": True,
        "human_gold": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("materialize-resolution")
    resolve.add_argument("--predecessor-postreview-manifest", type=Path, required=True)
    resolve.add_argument("--candidate-review-manifest", type=Path)
    resolve.add_argument("--output-dir", type=Path, required=True)
    successor = subparsers.add_parser("materialize-successor")
    successor.add_argument("--predecessor-postreview-manifest", type=Path, required=True)
    successor.add_argument("--candidate-resolution-manifest", type=Path, required=True)
    successor.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-resolution":
        result = materialize_resolution(
            predecessor_postreview_manifest_path=args.predecessor_postreview_manifest,
            candidate_review_manifest_path=args.candidate_review_manifest,
            output_dir=args.output_dir,
        )
    else:
        result = materialize_successor(
            predecessor_postreview_manifest_path=args.predecessor_postreview_manifest,
            candidate_resolution_manifest_path=args.candidate_resolution_manifest,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
