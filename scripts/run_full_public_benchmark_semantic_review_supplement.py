#!/usr/bin/env python3
"""Recover exactly the persistent failures from one bound semantic review.

This recovery protocol is deliberately separate from the parent review.  It
derives its selection only from a fully validated parent checkpoint, reviews
each failed pair as a singleton with a generator and an independent reviewer,
and requires strict JSON Schema on both calls.  Parent artifacts are immutable.

The output remains behavior-blind model-proxy evidence with ``human_gold=false``.
It does not initialize an HF model, execute target behavior, expose Validation
or Sealed data, freeze a split, or claim infrastructure independence.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls.route_identity import (  # noqa: E402
    attach_route_identity_guard,
    build_route_identity_set,
    validate_route_identity_set,
)


TOOL_VERSION = "full-public-benchmark-semantic-review-supplement-runner-v1"
RUN_MANIFEST_SCHEMA = (
    "public-benchmark-full-semantic-review-supplement-run-manifest-v1"
)
CHECKPOINT_SCHEMA = (
    "public-benchmark-full-semantic-review-supplement-checkpoint-v1"
)
RESPONSE_SCHEMA = "public-benchmark-full-semantic-review-supplement-response-v1"
GENERATOR_PROMPT_VERSION = (
    "public-benchmark-full-semantic-review-supplement-generator-v2"
)
REVIEWER_PROMPT_VERSION = (
    "public-benchmark-full-semantic-review-supplement-reviewer-v2"
)
REVIEW_METHOD = (
    "behavior_blind_dual_model_singleton_semantic_review_recovery_v1"
)
RESPONSE_IDENTITY_ENFORCEMENT = "runner_exact_post_response_v1"

EXPECTED_PARENT_SELECTED_COUNT = 1826
EXPECTED_PARENT_COMPLETED_COUNT = 1798
EXPECTED_FAILED_COUNT = 28
EXPECTED_PARENT_ADJUDICATION_COUNT = 1798
FIXED_BATCH_SIZE = 1
MAX_ALLOWED_RETRIES = 2

MODEL_SPEC_FIELDS = frozenset(
    {
        "provider_profile",
        "model",
        "expected_response_model",
        "temperature",
        "max_output_tokens",
        "reasoning_effort",
        "disable_thinking",
        "timeout_seconds",
        "max_retries",
    }
)
VALIDATION_ERROR_CODES = frozenset(
    {
        "response_fields_invalid",
        "response_schema_version_invalid",
        "pair_id_invalid",
        "relationship_invalid",
        "alias_valid_invalid",
        "boolean_field_invalid",
        "alias_entity_inconsistent",
        "relationship_semantic_duplicate_inconsistent",
        "same_fact_fields_inconsistent",
        "rationale_invalid",
        "confidence_invalid",
        "json_parse_failed",
        "response_model_identity_missing",
        "response_model_identity_mismatch",
        "static_proxy_identity_failure",
        "strict_schema_route_unsupported",
        "strict_schema_provider_rejected",
        "parsed_response_missing",
        "provider_or_transport_failure",
    }
)
MODEL_CALL_FIELDS = frozenset(
    {
        "stage",
        "prompt_version",
        "pair_id",
        "provider_profile",
        "requested_model",
        "expected_response_model",
        "response_model",
        "response_model_identity_status",
        "terminal_status",
        "validation_error_code",
        "structured_output_mode",
        "strict_json_schema_sha256",
        "prompt_sha256",
        "request_sha256",
        "response_sha256",
        "raw_response_sha256",
        "usage",
        "latency_ms",
        "attempt_count",
        "attempts",
        "credentials_or_endpoints_included",
    }
)
CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "tool_version",
        "pair_id",
        "input_index",
        "candidate_row_sha256",
        "adjudication_template_row_sha256",
        "left_endpoint_id",
        "left_input_record_sha256",
        "right_endpoint_id",
        "right_input_record_sha256",
        "parent_failed_checkpoint_record_sha256",
        "run_contract_sha256",
        "terminal_status",
        "failure_stage",
        "validation_error_code",
        "generator_verdict",
        "reviewer_verdict",
        "strict_adjudication",
        "model_calls",
        "retry_history",
        "completed_at",
        "behavior_blind",
        "human_gold",
        "hf_model_execution_count",
        "hf_tokenizer_execution_count",
        "validation_behavior_exposure_count",
        "sealed_behavior_exposure_count",
        "behavior_execution_count",
    }
)

PARENT_RUN_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "tool_version",
        "status",
        "run_contract",
        "run_contract_sha256",
        "invocation_resumed",
        "started_or_resumed_at",
        "completed_at",
        "resume_enabled",
        "retry_failed_enabled",
        "execution_policy",
        "batch_size",
        "max_workers",
        "selected_pair_count",
        "full_candidate_pair_count",
        "selection_is_full_candidate_set",
        "terminal_status_counts",
        "relationship_counts",
        "candidate_adjudication_complete_for_selected_scope",
        "candidate_adjudication_complete_for_full_set",
        "semantic_closure_complete",
        "artifacts",
        "evidence_boundary",
    }
)
PARENT_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "tool_version",
        "pair_id",
        "input_index",
        "candidate_row_sha256",
        "adjudication_template_row_sha256",
        "left_endpoint_id",
        "left_input_record_sha256",
        "right_endpoint_id",
        "right_input_record_sha256",
        "run_contract_sha256",
        "terminal_status",
        "failure_stage",
        "model_verdict",
        "strict_adjudication",
        "model_calls",
        "retry_history",
        "completed_at",
        "behavior_blind",
        "human_gold",
        "hf_model_execution_count",
        "behavior_execution_count",
    }
)
PARENT_MODEL_CALL_FIELDS = frozenset(
    {
        "stage",
        "batch_id",
        "batch_item_count",
        "ordered_pair_ids_sha256",
        "provider_profile",
        "requested_model",
        "expected_response_model",
        "response_model",
        "response_model_identity_status",
        "terminal_status",
        "post_validation_error_type",
        "prompt_version",
        "prompt_sha256",
        "request_sha256",
        "response_sha256",
        "raw_response_sha256",
        "usage",
        "usage_scope",
        "latency_ms",
        "attempt_count",
        "attempts",
        "credentials_or_endpoints_included",
    }
)

GENERATOR_INSTRUCTIONS = """You are the proposal stage for a singleton semantic-leakage recovery review.
Use only the supplied English pair projection. Target-model behavior, splits, components, HF
results, perturbation results, and evaluation results are unavailable and forbidden. Return only
an object matching the supplied strict JSON Schema and copy pair_id exactly. A shared answer alone
is insufficient. Decide same_fact only for the same proposition under the same entities, relation,
polarity, time, and answer scope; use same_leakage_component only for a distinct but closely linked
proposition whose separation could leak the answer or proposition; otherwise use distinct.
Before returning, enforce these field invariants exactly: alias_valid must be null when
answer_alias_exact is absent; when answer_alias_exact is present, alias_valid=true means the shared
alias validly names the same answer entity in both endpoints and therefore requires
same_answer_entity=true. relationship=same_fact if and only if semantic_duplicate=true, and a
same_fact result requires same_answer_entity, same_subject_entity, relation_semantics_same,
temporal_scope_compatible, and answer_compatible all to be true."""

REVIEWER_INSTRUCTIONS = """You are the independent final reviewer for a singleton semantic-leakage
recovery review. Re-evaluate the supplied English evidence yourself; the generator draft is
non-authoritative and may be overturned. Target-model behavior, splits, components, HF results,
perturbation results, and evaluation results are unavailable and forbidden. Return only an object
matching the supplied strict JSON Schema and copy pair_id exactly. A shared answer alone is
insufficient. Decide same_fact only for the same proposition under the same entities, relation,
polarity, time, and answer scope; use same_leakage_component only for a distinct but closely linked
proposition whose separation could leak the answer or proposition; otherwise use distinct. Ground
the rationale in the supplied text. Before returning, enforce these field invariants exactly:
alias_valid must be null when answer_alias_exact is absent; when answer_alias_exact is present,
alias_valid=true means the shared alias validly names the same answer entity in both endpoints and
therefore requires same_answer_entity=true. relationship=same_fact if and only if
semantic_duplicate=true, and a same_fact result requires same_answer_entity, same_subject_entity,
relation_semantics_same, temporal_scope_compatible, and answer_compatible all to be true."""


def _load_parent_runner() -> Any:
    path = Path(__file__).resolve().with_name(
        "run_full_public_benchmark_semantic_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_semantic_review_parent_for_supplement", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load parent semantic runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent_runner = _load_parent_runner()


class SupplementValidationError(ValueError):
    """A finite and non-sensitive response-contract failure."""

    def __init__(self, code: str):
        if code not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Unknown supplement validation error code: {code}")
        self.code = code
        super().__init__(code)


class StrictSchemaRouteUnsupportedError(RuntimeError):
    """The selected provider protocol cannot carry strict JSON Schema."""


class ParentAuthority(NamedTuple):
    manifest: Dict[str, Any]
    manifest_path: Path
    units: List[Any]
    checkpoints: List[Dict[str, Any]]
    adjudications: List[Dict[str, Any]]
    completed_ids: List[str]
    failed_ids: List[str]
    failed_units: List[Any]
    failed_checkpoint_rows: Dict[str, Dict[str, Any]]
    inputs: Dict[str, Dict[str, Any]]


def _required_string(value: Any, label: str) -> str:
    return parent_runner._required_string(value, label)


def _validate_timestamp(value: Any, label: str) -> str:
    rendered = _required_string(value, label)
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return rendered


def protocol_reviewer_identity(config: Mapping[str, Any]) -> Dict[str, str]:
    """Return the reviewer identity frozen by the semantic-review protocol."""

    roles = config.get("model_roles")
    roles = roles if isinstance(roles, dict) else {}
    role = roles.get("full_pool_semantic_closure_review")
    role = role if isinstance(role, dict) else {}
    reviewer = role.get("reviewer")
    reviewer = reviewer if isinstance(reviewer, dict) else {}
    identity = {
        "provider_profile": _required_string(
            reviewer.get("provider_profile"),
            "semantic supplement protocol reviewer provider_profile",
        ),
        "requested_model": _required_string(
            reviewer.get("model"),
            "semantic supplement protocol reviewer model",
        ),
        "expected_response_model": _required_string(
            reviewer.get("expected_response_model"),
            "semantic supplement protocol reviewer expected_response_model",
        ),
    }
    expected = {
        "provider_profile": parent_runner.PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": parent_runner.DEFAULT_REVIEWER_ROUTE,
        "expected_response_model": parent_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if identity != expected:
        raise ValueError("semantic supplement protocol reviewer identity is unsupported")
    return identity


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(resolved),
        "sha256": parent_runner.sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _bound_path(binding: Any, owner_path: Path, label: str) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    raw = binding.get("path") or binding.get("filename")
    path = Path(_required_string(raw, f"{label}.path"))
    if not path.is_absolute():
        path = Path(owner_path).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != parent_runner.sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if "byte_count" in binding and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count is stale")
    return path


def _same_bound_file(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    actual_owner: Path,
    expected_owner: Path,
    label: str,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} binding is missing")
    if _bound_path(actual, actual_owner, label) != _bound_path(
        expected, expected_owner, label
    ):
        raise ValueError(f"{label} path differs from parent authority")
    if actual.get("sha256") != expected.get("sha256"):
        raise ValueError(f"{label} SHA-256 differs from parent authority")
    for field in ("schema_version", "record_count"):
        if field in actual and field in expected and actual.get(field) != expected.get(field):
            raise ValueError(f"{label} {field} differs from parent authority")


def _runtime_source_bindings() -> Dict[str, Dict[str, Any]]:
    paths = {
        "supplement_runner": Path(__file__).resolve(),
        "parent_runner": Path(parent_runner.__file__).resolve(),
        "semantic_candidate_contract": Path(parent_runner.materializer.__file__).resolve(),
        "model_router_runtime": PROJECT_ROOT / "factual_pitfalls" / "perturbation.py",
        "route_identity_helper": PROJECT_ROOT / "factual_pitfalls" / "route_identity.py",
    }
    return {name: _binding(path) for name, path in paths.items()}


def _verify_bindings_unchanged(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for label, binding in bindings.items():
        _bound_path(binding, Path(binding.get("path") or "."), label)


def _validate_parent_manifest_shape(
    manifest: Mapping[str, Any],
    *,
    expected_selected_count: int,
    expected_completed_count: int,
    expected_failed_count: int,
) -> Tuple[Dict[str, Any], str]:
    if set(manifest) != PARENT_RUN_MANIFEST_FIELDS:
        raise ValueError("Parent semantic run manifest fields are invalid")
    if manifest.get("schema_version") != parent_runner.RUN_MANIFEST_SCHEMA:
        raise ValueError("Parent semantic run manifest schema is unsupported")
    if manifest.get("tool_version") != parent_runner.TOOL_VERSION:
        raise ValueError("Parent semantic runner version is unsupported")
    if manifest.get("status") != "completed_with_failures":
        raise ValueError("Parent semantic run must be completed_with_failures")
    _validate_timestamp(
        manifest.get("started_or_resumed_at"),
        "parent semantic started_or_resumed_at",
    )
    _validate_timestamp(manifest.get("completed_at"), "parent semantic completed_at")
    contract = manifest.get("run_contract")
    if not isinstance(contract, dict):
        raise ValueError("Parent semantic run_contract is missing")
    contract_sha = parent_runner.sha256_value(contract)
    if manifest.get("run_contract_sha256") != contract_sha:
        raise ValueError("Parent semantic run_contract SHA-256 is stale")
    expected_counts = {
        "completed": expected_completed_count,
        "failed": expected_failed_count,
    }
    if manifest.get("terminal_status_counts") != expected_counts:
        raise ValueError(f"Parent semantic terminal counts must equal {expected_counts}")
    if expected_completed_count + expected_failed_count != expected_selected_count:
        raise ValueError("Expected parent counts do not cover selected_count")
    if not all(
        (
            manifest.get("selected_pair_count") == expected_selected_count,
            manifest.get("full_candidate_pair_count") == expected_selected_count,
            manifest.get("selection_is_full_candidate_set") is True,
            manifest.get("candidate_adjudication_complete_for_full_set") is False,
            contract.get("selected_pair_count") == expected_selected_count,
            contract.get("full_candidate_pair_count") == expected_selected_count,
            contract.get("selection_is_full_candidate_set") is True,
            contract.get("behavior_blind") is True,
            contract.get("human_gold") is False,
        )
    ):
        raise ValueError("Parent semantic full-candidate contract is invalid")
    if contract.get("source_fingerprints") != parent_runner.source_fingerprints():
        raise ValueError("Parent semantic runtime source fingerprints are stale")
    boundary = manifest.get("evidence_boundary")
    if not isinstance(boundary, dict) or not all(
        (
            boundary.get("human_gold") is False,
            boundary.get("behavior_blind") is True,
            boundary.get("target_behavior_consumed") is False,
            boundary.get("hf_model_execution_count") == 0,
            boundary.get("hf_tokenizer_execution_count") == 0,
            boundary.get("validation_behavior_exposure_count") == 0,
            boundary.get("sealed_behavior_exposure_count") == 0,
        )
    ):
        raise ValueError("Parent semantic evidence boundary is invalid")
    return dict(contract), contract_sha


def _validate_parent_adjudication(
    decision: Mapping[str, Any], *, unit: Any, protocol_identity: Mapping[str, str]
) -> None:
    expected_fields = {
        "schema_version",
        "pair_id",
        "candidate_row_sha256",
        "adjudication_template_row_sha256",
        "left_endpoint_id",
        "left_input_record_sha256",
        "right_endpoint_id",
        "right_input_record_sha256",
        *parent_runner.FIELD_NAMES,
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
    pair_id = str(unit.pair_id)
    if set(decision) != expected_fields:
        raise ValueError(f"Parent semantic adjudication fields are invalid: {pair_id}")
    expected = {
        "schema_version": parent_runner.ADJUDICATION_SCHEMA,
        "pair_id": pair_id,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        "reviewer_type": "independent_model_proxy",
        "reviewer_id": (
            f"{protocol_identity['provider_profile']}:"
            f"{protocol_identity['expected_response_model']}"
        ),
        "requested_model": protocol_identity["requested_model"],
        "expected_response_model": protocol_identity["expected_response_model"],
        "response_model": protocol_identity["expected_response_model"],
        "response_model_identity_status": "matched",
        "review_method": parent_runner.REVIEW_METHOD,
        "behavior_blind": True,
        "human_gold": False,
    }
    for field, value in expected.items():
        if decision.get(field) != value:
            raise ValueError(
                f"Parent semantic adjudication {field} is stale: {pair_id}"
            )
    _validate_timestamp(decision.get("reviewed_at"), f"{pair_id}.reviewed_at")
    parent_runner.validate_batch_response(
        {
            "schema_version": parent_runner.BATCH_RESPONSE_SCHEMA,
            "records": [
                {
                    "pair_id": pair_id,
                    "relationship": decision.get("relationship"),
                    **{
                        field: decision.get(field)
                        for field in parent_runner.FIELD_NAMES
                    },
                    "rationale": decision.get("rationale"),
                    "confidence": decision.get("confidence"),
                }
            ],
        },
        [unit],
    )


def _validate_parent_model_call(
    call: Any,
    *,
    pair_id: str,
    protocol_identity: Mapping[str, str],
) -> None:
    if not isinstance(call, dict) or set(call) != PARENT_MODEL_CALL_FIELDS:
        raise ValueError(f"Parent semantic model call fields are invalid: {pair_id}")
    expected = {
        "stage": "full_semantic_closure_review",
        "prompt_version": parent_runner.PROMPT_VERSION,
        "provider_profile": protocol_identity["provider_profile"],
        "requested_model": protocol_identity["requested_model"],
        "expected_response_model": protocol_identity["expected_response_model"],
        "usage_scope": "shared_batch_not_per_item",
        "credentials_or_endpoints_included": False,
    }
    if any(call.get(field) != value for field, value in expected.items()):
        raise ValueError(f"Parent semantic model call contract is stale: {pair_id}")
    if not _required_string(call.get("batch_id"), f"{pair_id}.batch_id").startswith(
        "semantic_batch_"
    ):
        raise ValueError(f"Parent semantic batch_id is invalid: {pair_id}")
    batch_count = call.get("batch_item_count")
    if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count < 1:
        raise ValueError(f"Parent semantic batch size is invalid: {pair_id}")
    for field in (
        "ordered_pair_ids_sha256",
        "prompt_sha256",
        "request_sha256",
        "response_sha256",
        "raw_response_sha256",
    ):
        if not _is_sha256(call.get(field)):
            raise ValueError(f"Parent semantic model call hash is invalid: {pair_id}")
    if call.get("response_sha256") != call.get("raw_response_sha256"):
        raise ValueError(f"Parent semantic response hashes disagree: {pair_id}")
    attempts = call.get("attempts")
    if attempts != parent_runner._safe_attempts(attempts) or not attempts:
        raise ValueError(f"Parent semantic attempts are invalid: {pair_id}")
    attempt_count = call.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count != len(attempts)
    ):
        raise ValueError(f"Parent semantic attempt_count is invalid: {pair_id}")
    for index, attempt in enumerate(attempts, start=1):
        if attempt.get("attempt") != index or attempt.get("status") not in {
            "ok",
            "failed",
        }:
            raise ValueError(f"Parent semantic attempt trace is invalid: {pair_id}")
    usage = call.get("usage")
    if (
        not isinstance(usage, dict)
        or not set(usage).issubset({"input_tokens", "output_tokens", "total_tokens"})
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in usage.values()
        )
    ):
        raise ValueError(f"Parent semantic usage trace is invalid: {pair_id}")
    response_model = call.get("response_model")
    expected_identity_status = (
        "not_observed"
        if response_model is None
        else "matched"
        if response_model == protocol_identity["expected_response_model"]
        else "mismatch"
    )
    if call.get("response_model_identity_status") != expected_identity_status:
        raise ValueError(f"Parent semantic response identity is inconsistent: {pair_id}")
    status = call.get("terminal_status")
    if status == "completed":
        if (
            expected_identity_status != "matched"
            or attempts[-1].get("status") != "ok"
            or call.get("post_validation_error_type") is not None
        ):
            raise ValueError(f"Parent semantic completed call is invalid: {pair_id}")
    elif status == "failed":
        if attempts[-1].get("status") not in {"ok", "failed"}:
            raise ValueError(f"Parent semantic failed call is invalid: {pair_id}")
    else:
        raise ValueError(f"Parent semantic model call status is invalid: {pair_id}")


def _validate_parent_checkpoint_row(
    row: Mapping[str, Any],
    *,
    unit: Any,
    run_contract_sha256: str,
    protocol_identity: Mapping[str, str],
) -> None:
    pair_id = str(unit.pair_id)
    if set(row) != PARENT_CHECKPOINT_FIELDS:
        raise ValueError(f"Parent semantic checkpoint fields are invalid: {pair_id}")
    expected = {
        "schema_version": parent_runner.CHECKPOINT_SCHEMA,
        "tool_version": parent_runner.TOOL_VERSION,
        "pair_id": pair_id,
        "input_index": unit.input_index,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        "run_contract_sha256": run_contract_sha256,
        "behavior_blind": True,
        "human_gold": False,
        "hf_model_execution_count": 0,
        "behavior_execution_count": 0,
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise ValueError(f"Parent semantic checkpoint binding is stale: {pair_id}")
    _validate_timestamp(row.get("completed_at"), f"{pair_id}.completed_at")
    calls = row.get("model_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"Parent semantic model calls are missing: {pair_id}")
    for call in calls:
        _validate_parent_model_call(
            call, pair_id=pair_id, protocol_identity=protocol_identity
        )
    retry_history = row.get("retry_history")
    if not isinstance(retry_history, list):
        raise ValueError(f"Parent semantic retry_history is invalid: {pair_id}")
    for prior in retry_history:
        if not isinstance(prior, dict) or set(prior) != {
            "terminal_status",
            "failure_stage",
            "model_calls",
            "completed_at",
        }:
            raise ValueError(f"Parent semantic retry history is invalid: {pair_id}")
        prior_calls = prior.get("model_calls")
        if (
            prior.get("terminal_status") != "failed"
            or prior.get("failure_stage") != "reviewer"
            or not isinstance(prior_calls, list)
            or not prior_calls
            or prior_calls[-1].get("terminal_status") != "failed"
        ):
            raise ValueError(f"Parent semantic retry failure is invalid: {pair_id}")
        _validate_timestamp(prior.get("completed_at"), f"{pair_id}.retry.completed_at")
        for call in prior_calls:
            _validate_parent_model_call(
                call, pair_id=pair_id, protocol_identity=protocol_identity
            )
    status = row.get("terminal_status")
    if status == "completed":
        if (
            row.get("failure_stage") is not None
            or not isinstance(row.get("model_verdict"), dict)
            or not isinstance(row.get("strict_adjudication"), dict)
            or calls[-1].get("terminal_status") != "completed"
        ):
            raise ValueError(f"Parent completed checkpoint is invalid: {pair_id}")
        parent_runner.validate_batch_response(
            {
                "schema_version": parent_runner.BATCH_RESPONSE_SCHEMA,
                "records": [dict(row["model_verdict"])],
            },
            [unit],
        )
    elif status == "failed":
        if (
            row.get("failure_stage") != "reviewer"
            or row.get("model_verdict") is not None
            or row.get("strict_adjudication") is not None
            or calls[-1].get("terminal_status") != "failed"
        ):
            raise ValueError(f"Parent failed checkpoint is invalid: {pair_id}")
    else:
        raise ValueError(f"Parent semantic checkpoint status is invalid: {pair_id}")


def _inspect_parent_authority(
    parent_run_manifest_path: Path,
    *,
    expected_selected_count: int,
    expected_completed_count: int,
    expected_failed_count: int,
    expected_adjudication_count: int,
) -> ParentAuthority:
    manifest_path = Path(parent_run_manifest_path).resolve()
    manifest = parent_runner.read_json(manifest_path)
    contract, contract_sha = _validate_parent_manifest_shape(
        manifest,
        expected_selected_count=expected_selected_count,
        expected_completed_count=expected_completed_count,
        expected_failed_count=expected_failed_count,
    )
    protocol_identity = {
        "provider_profile": parent_runner.PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": parent_runner.DEFAULT_REVIEWER_ROUTE,
        "expected_response_model": parent_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if contract.get("protocol_reviewer_identity") != protocol_identity:
        raise ValueError("Parent semantic reviewer identity is unsupported")
    reviewer_spec = contract.get("reviewer_spec")
    if not isinstance(reviewer_spec, dict) or any(
        reviewer_spec.get(field) != expected
        for field, expected in {
            "provider_profile": protocol_identity["provider_profile"],
            "model": protocol_identity["requested_model"],
        }.items()
    ):
        raise ValueError("Parent semantic reviewer spec is unsupported")
    if contract.get("expected_response_model") != protocol_identity[
        "expected_response_model"
    ]:
        raise ValueError("Parent semantic expected response identity is unsupported")
    validate_route_identity_set(
        contract.get("route_identity"),
        expected_routes=[
            (
                protocol_identity["provider_profile"],
                protocol_identity["requested_model"],
                protocol_identity["expected_response_model"],
            )
        ],
    )

    candidate_manifest_path = _bound_path(
        contract.get("candidate_manifest"), manifest_path, "parent candidate_manifest"
    )
    candidate_pairs_path = _bound_path(
        contract.get("candidate_pairs"), manifest_path, "parent candidate_pairs"
    )
    template_path = _bound_path(
        contract.get("adjudication_template"),
        manifest_path,
        "parent adjudication_template",
    )
    candidate_manifest, loaded_pairs_path, loaded_template_path, units = (
        parent_runner.load_review_units(
            candidate_manifest_path=candidate_manifest_path,
            candidate_pairs_path=candidate_pairs_path,
            adjudication_template_path=template_path,
        )
    )
    if loaded_pairs_path != candidate_pairs_path or loaded_template_path != template_path:
        raise ValueError("Parent candidate artifacts differ from manifest bindings")
    if len(units) != expected_selected_count:
        raise ValueError("Parent semantic units do not cover the expected full set")
    ordered_ids = [str(unit.pair_id) for unit in units]
    if contract.get("selected_pair_ids_sha256") != parent_runner.sha256_value(
        ordered_ids
    ):
        raise ValueError("Parent semantic selected pair digest is stale")
    if contract.get("formal_universe_id") != candidate_manifest.get("universe_id"):
        raise ValueError("Parent semantic universe binding is stale")
    expected_prompt_contract_sha = parent_runner.sha256_value(
        {
            "prompt_version": parent_runner.PROMPT_VERSION,
            "instructions": parent_runner.REVIEW_INSTRUCTIONS,
            "projection_schema": parent_runner.PROJECTION_SCHEMA,
            "batch_response_schema": parent_runner.BATCH_RESPONSE_SCHEMA,
            "field_names": parent_runner.FIELD_NAMES,
        }
    )
    if contract.get("prompt_contract_sha256") != expected_prompt_contract_sha:
        raise ValueError("Parent semantic prompt contract is stale")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Parent semantic artifacts are missing")
    checkpoint_binding = artifacts.get("checkpoint")
    adjudications_binding = artifacts.get("adjudications")
    checkpoint_path = _bound_path(
        checkpoint_binding, manifest_path, "parent semantic checkpoint"
    )
    adjudications_path = _bound_path(
        adjudications_binding, manifest_path, "parent semantic adjudications"
    )
    if not isinstance(checkpoint_binding, dict) or not isinstance(
        adjudications_binding, dict
    ):
        raise ValueError("Parent semantic artifact bindings are invalid")
    if checkpoint_binding.get("schema_version") != parent_runner.CHECKPOINT_SCHEMA:
        raise ValueError("Parent semantic checkpoint schema is unsupported")
    if adjudications_binding.get("schema_version") != parent_runner.ADJUDICATION_SCHEMA:
        raise ValueError("Parent semantic adjudication schema is unsupported")
    checkpoint_rows = parent_runner.read_jsonl(checkpoint_path)
    adjudications = parent_runner.read_jsonl(adjudications_path, allow_empty=True)
    if checkpoint_binding.get("record_count") != len(checkpoint_rows):
        raise ValueError("Parent semantic checkpoint record_count is stale")
    if adjudications_binding.get("record_count") != len(adjudications):
        raise ValueError("Parent semantic adjudication record_count is stale")
    if len(checkpoint_rows) != expected_selected_count:
        raise ValueError("Parent semantic checkpoint is not full-set")
    if len(adjudications) != expected_adjudication_count:
        raise ValueError("Parent semantic adjudication count is stale")

    checkpoint_index = parent_runner._checkpoint_index(
        path=checkpoint_path,
        units_by_id={str(unit.pair_id): unit for unit in units},
        run_contract_sha256=contract_sha,
        protocol_identity=protocol_identity,
    )
    if len(checkpoint_index) != len(units):
        raise ValueError("Parent semantic checkpoint coverage is incomplete")
    completed_ids: List[str] = []
    failed_ids: List[str] = []
    failed_rows: Dict[str, Dict[str, Any]] = {}
    expected_adjudications: List[Dict[str, Any]] = []
    for unit, row in zip(units, checkpoint_rows):
        pair_id = str(unit.pair_id)
        if row.get("pair_id") != pair_id or row.get("input_index") != unit.input_index:
            raise ValueError("Parent semantic checkpoint order differs from candidates")
        if row.get("tool_version") != parent_runner.TOOL_VERSION:
            raise ValueError(f"Parent semantic checkpoint tool version is stale: {pair_id}")
        validated = checkpoint_index[pair_id]
        _validate_parent_checkpoint_row(
            validated,
            unit=unit,
            run_contract_sha256=contract_sha,
            protocol_identity=protocol_identity,
        )
        if validated.get("terminal_status") == "completed":
            _validate_parent_adjudication(
                validated["strict_adjudication"],
                unit=unit,
                protocol_identity=protocol_identity,
            )
            completed_ids.append(pair_id)
            expected_adjudications.append(dict(validated["strict_adjudication"]))
        else:
            if validated.get("strict_adjudication") is not None or validated.get(
                "model_verdict"
            ) is not None:
                raise ValueError(
                    f"Parent failed checkpoint contains an adjudication: {pair_id}"
                )
            failed_ids.append(pair_id)
            failed_rows[pair_id] = dict(validated)
    if len(completed_ids) != expected_completed_count or len(failed_ids) != expected_failed_count:
        raise ValueError("Parent semantic checkpoint terminal counts are stale")
    if parent_runner.canonical_json_bytes(adjudications) != parent_runner.canonical_json_bytes(
        expected_adjudications
    ):
        raise ValueError("Parent semantic adjudications differ from completed checkpoints")
    relationship_counts = Counter(
        str(row["relationship"]) for row in expected_adjudications
    )
    if manifest.get("relationship_counts") != dict(sorted(relationship_counts.items())):
        raise ValueError("Parent semantic relationship_counts are stale")

    failed_set = set(failed_ids)
    failed_units = [unit for unit in units if str(unit.pair_id) in failed_set]
    if [str(unit.pair_id) for unit in failed_units] != failed_ids:
        raise ValueError("Parent semantic failed pair ordering is not deterministic")
    inputs = {
        "parent_run_manifest": _binding(
            manifest_path, schema_version=parent_runner.RUN_MANIFEST_SCHEMA
        ),
        "parent_checkpoint": _binding(
            checkpoint_path,
            schema_version=parent_runner.CHECKPOINT_SCHEMA,
            record_count=len(checkpoint_rows),
        ),
        "parent_adjudications": _binding(
            adjudications_path,
            schema_version=parent_runner.ADJUDICATION_SCHEMA,
            record_count=len(adjudications),
        ),
        "candidate_manifest": _binding(
            candidate_manifest_path,
            schema_version=parent_runner.materializer.MANIFEST_SCHEMA,
        ),
        "candidate_pairs": _binding(
            candidate_pairs_path,
            schema_version=parent_runner.materializer.CANDIDATE_SCHEMA,
            record_count=len(units),
        ),
        "adjudication_template": _binding(
            template_path,
            schema_version=parent_runner.materializer.ADJUDICATION_TEMPLATE_SCHEMA,
            record_count=len(units),
        ),
    }
    return ParentAuthority(
        manifest=dict(manifest),
        manifest_path=manifest_path,
        units=units,
        checkpoints=[dict(row) for row in checkpoint_rows],
        adjudications=[dict(row) for row in adjudications],
        completed_ids=completed_ids,
        failed_ids=failed_ids,
        failed_units=failed_units,
        failed_checkpoint_rows=failed_rows,
        inputs=inputs,
    )


def inspect_parent_authority(parent_run_manifest_path: Path) -> ParentAuthority:
    """Validate the authoritative 1,826-pair parent and derive its 28 failures."""

    return _inspect_parent_authority(
        parent_run_manifest_path,
        expected_selected_count=EXPECTED_PARENT_SELECTED_COUNT,
        expected_completed_count=EXPECTED_PARENT_COMPLETED_COUNT,
        expected_failed_count=EXPECTED_FAILED_COUNT,
        expected_adjudication_count=EXPECTED_PARENT_ADJUDICATION_COUNT,
    )


def _safe_model_spec(spec: Mapping[str, Any], role: str) -> Dict[str, Any]:
    output = {
        field: copy.deepcopy(spec[field])
        for field in MODEL_SPEC_FIELDS
        if field in spec and spec[field] is not None
    }
    _required_string(output.get("provider_profile"), f"{role} provider_profile")
    _required_string(output.get("model"), f"{role} model")
    _required_string(
        output.get("expected_response_model"), f"{role} expected_response_model"
    )
    retries = output.get("max_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= MAX_ALLOWED_RETRIES:
        raise ValueError(f"{role} max_retries must be between 0 and {MAX_ALLOWED_RETRIES}")
    output["max_retries"] = retries
    max_tokens = output.get("max_output_tokens", 8192)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"{role} max_output_tokens must be positive")
    output["max_output_tokens"] = max_tokens
    return output


def _validate_strict_schema_routes(
    config: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]
) -> None:
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("config.provider_profiles must be an object")
    for role, spec in zip(("generator", "reviewer"), specs):
        profile_name = str(spec["provider_profile"])
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(f"Unknown {role} provider_profile: {profile_name}")
        if profile.get("protocol") not in {"openai", "openai_compatible"}:
            raise ValueError(
                f"{role} route does not support required strict JSON Schema: {profile_name}"
            )


def resolve_model_specs(
    config: Mapping[str, Any],
    *,
    generator_provider_profile: Optional[str] = None,
    generator_model: Optional[str] = None,
    generator_expected_response_model: Optional[str] = None,
    generator_reasoning_effort: Optional[str] = None,
    generator_max_output_tokens: Optional[int] = None,
    reviewer_provider_profile: Optional[str] = None,
    reviewer_model: Optional[str] = None,
    reviewer_expected_response_model: Optional[str] = None,
    reviewer_reasoning_effort: Optional[str] = None,
    reviewer_max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    roles = config.get("model_roles")
    if not isinstance(roles, dict):
        raise ValueError("config.model_roles must be an object")
    translation = roles.get("translation")
    semantic = roles.get("full_pool_semantic_closure_review")
    if not isinstance(translation, dict) or not isinstance(semantic, dict):
        raise ValueError("Required semantic supplement model roles are missing")
    generator = dict(translation.get("primary") or {})
    reviewer = dict(semantic.get("reviewer") or {})
    overrides = (
        (
            generator,
            generator_provider_profile,
            generator_model,
            generator_expected_response_model,
            generator_reasoning_effort,
            generator_max_output_tokens,
        ),
        (
            reviewer,
            reviewer_provider_profile,
            reviewer_model,
            reviewer_expected_response_model,
            reviewer_reasoning_effort,
            reviewer_max_output_tokens,
        ),
    )
    safe_specs: List[Dict[str, Any]] = []
    for role, values in zip(("generator", "reviewer"), overrides):
        spec, profile, model, expected, effort, max_tokens = values
        if profile is not None:
            spec["provider_profile"] = profile
        if model is not None:
            spec["model"] = model
        if expected is not None:
            spec["expected_response_model"] = expected
        if effort is not None:
            spec["reasoning_effort"] = effort
        if max_tokens is not None:
            spec["max_output_tokens"] = max_tokens
        if max_retries is not None:
            spec["max_retries"] = max_retries
        spec.pop("json_mode", None)
        spec.pop("require_response_model_identity", None)
        safe_specs.append(_safe_model_spec(spec, role))
    generator_safe, reviewer_safe = safe_specs
    _validate_strict_schema_routes(config, safe_specs)
    reviewer_identity = protocol_reviewer_identity(config)
    if (
        reviewer_safe["provider_profile"] != reviewer_identity["provider_profile"]
        or reviewer_safe["model"] != reviewer_identity["requested_model"]
        or reviewer_safe["expected_response_model"]
        != reviewer_identity["expected_response_model"]
    ):
        raise ValueError("reviewer identity differs from the protocol config")
    if (
        generator_safe["provider_profile"],
        generator_safe["model"],
    ) == (
        reviewer_safe["provider_profile"],
        reviewer_safe["model"],
    ):
        raise ValueError("generator and reviewer must use distinct model identities")
    if generator_safe["expected_response_model"] == reviewer_safe[
        "expected_response_model"
    ]:
        raise ValueError("generator and reviewer must use distinct resolved identities")
    return generator_safe, reviewer_safe


def strict_response_schema(unit: Any) -> Dict[str, Any]:
    alias_required = "answer_alias_exact" in unit.projection["match_types"]
    fields: Dict[str, Any] = {
        "schema_version": {"type": "string", "enum": [RESPONSE_SCHEMA]},
        "pair_id": {"type": "string", "enum": [unit.pair_id]},
        "relationship": {
            "type": "string",
            "enum": sorted(parent_runner.RELATIONSHIPS),
        },
        "alias_valid": {"type": "boolean"} if alias_required else {"type": "null"},
        **{
            field: {"type": "boolean"}
            for field in parent_runner.FIELD_NAMES[1:]
        },
        "rationale": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": sorted(parent_runner.CONFIDENCE_LEVELS),
        },
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def _raise(code: str) -> None:
    raise SupplementValidationError(code)


def validate_response(value: Dict[str, Any], unit: Any) -> None:
    expected_fields = {
        "schema_version",
        "pair_id",
        "relationship",
        *parent_runner.FIELD_NAMES,
        "rationale",
        "confidence",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        _raise("response_fields_invalid")
    if value.get("schema_version") != RESPONSE_SCHEMA:
        _raise("response_schema_version_invalid")
    if value.get("pair_id") != unit.pair_id:
        _raise("pair_id_invalid")
    relationship = value.get("relationship")
    if relationship not in parent_runner.RELATIONSHIPS:
        _raise("relationship_invalid")
    alias_required = "answer_alias_exact" in unit.projection["match_types"]
    alias_valid = value.get("alias_valid")
    if (alias_required and not isinstance(alias_valid, bool)) or (
        not alias_required and alias_valid is not None
    ):
        _raise("alias_valid_invalid")
    if any(not isinstance(value.get(field), bool) for field in parent_runner.FIELD_NAMES[1:]):
        _raise("boolean_field_invalid")
    if alias_valid is True and value.get("same_answer_entity") is not True:
        _raise("alias_entity_inconsistent")
    if (relationship == "same_fact") != value.get("semantic_duplicate"):
        _raise("relationship_semantic_duplicate_inconsistent")
    if relationship == "same_fact" and not all(
        value.get(field) is True
        for field in (
            "same_answer_entity",
            "same_subject_entity",
            "relation_semantics_same",
            "temporal_scope_compatible",
            "answer_compatible",
        )
    ):
        _raise("same_fact_fields_inconsistent")
    if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
        _raise("rationale_invalid")
    if value.get("confidence") not in parent_runner.CONFIDENCE_LEVELS:
        _raise("confidence_invalid")


def _prompt_payload(prompt: str) -> Dict[str, Any]:
    marker = "INPUT_JSON:\n"
    if marker not in prompt:
        raise ValueError("Prompt lacks INPUT_JSON marker")
    value = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(value, dict):
        raise ValueError("Prompt INPUT_JSON must be an object")
    return value


def build_generator_prompt(unit: Any, schema: Mapping[str, Any]) -> str:
    payload = {"strict_response_json_schema": schema, "evidence": unit.projection}
    return (
        f"{GENERATOR_INSTRUCTIONS}\nINPUT_JSON:\n"
        f"{parent_runner.canonical_json_bytes(payload).decode('utf-8')}"
    )


def build_reviewer_prompt(
    unit: Any, draft: Mapping[str, Any], schema: Mapping[str, Any]
) -> str:
    payload = {
        "strict_response_json_schema": schema,
        "evidence": unit.projection,
        "generator_draft": draft,
    }
    return (
        f"{REVIEWER_INSTRUCTIONS}\nINPUT_JSON:\n"
        f"{parent_runner.canonical_json_bytes(payload).decode('utf-8')}"
    )


def _strict_schema_router_class() -> Any:
    from factual_pitfalls import perturbation as runtime

    class StrictJSONSchemaModelRouter(runtime.ModelRouter):
        strict_json_schema_no_fallback = True

        def _one_call_unlocked(
            self,
            profile: Dict[str, Any],
            api_key: str,
            base_url: str,
            model_spec: Dict[str, Any],
            prompt: str,
        ) -> Tuple[str, Optional[str], Dict[str, int]]:
            strict_schema = model_spec.get("strict_json_schema")
            if not isinstance(strict_schema, dict):
                raise StrictSchemaRouteUnsupportedError("strict_json_schema_missing")
            if profile.get("protocol") not in {"openai", "openai_compatible"}:
                raise StrictSchemaRouteUnsupportedError(
                    "strict_json_schema_requires_openai_compatible_protocol"
                )
            model = str(model_spec["model"])
            timeout = float(
                model_spec.get(
                    "timeout_seconds", profile.get("timeout_seconds", self.timeout)
                )
            )
            client = runtime.OpenAI(
                api_key=api_key,
                base_url=runtime.api_root(base_url),
                timeout=timeout,
                max_retries=0,
            )
            request: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": strict_schema},
            }
            if model.startswith("gpt-"):
                request["max_completion_tokens"] = model_spec.get(
                    "max_output_tokens", 8192
                )
                if model_spec.get("reasoning_effort"):
                    request["reasoning_effort"] = model_spec["reasoning_effort"]
            else:
                request["max_tokens"] = model_spec.get("max_output_tokens", 8192)
            if model_spec.get("disable_thinking") is True:
                request["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            elif model.startswith("qwen") or "deepseek" in model:
                request["extra_body"] = {"enable_thinking": False}
            if "temperature" in model_spec and not model.startswith("gpt-"):
                request["temperature"] = model_spec["temperature"]
            response = client.chat.completions.create(**request)
            return (
                self._openai_text(response),
                getattr(response, "model", None),
                self._usage(getattr(response, "usage", None)),
            )

    return StrictJSONSchemaModelRouter


def _create_runtime_router(
    config: Mapping[str, Any], env_path: Path, event_log_path: Path
) -> Any:
    return _strict_schema_router_class()(
        dict(config), Path(env_path).resolve(), Path(event_log_path).resolve()
    )


def _failure_code(result: Any, validator_code: Optional[str]) -> str:
    attempts = result.attempts if isinstance(result.attempts, list) else []
    last = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    if validator_code is not None and last.get("error_type") == "SupplementValidationError":
        return validator_code
    error_types = {
        str(row.get("error_type"))
        for row in attempts
        if isinstance(row, dict) and row.get("error_type")
    }
    statuses = {row.get("http_status") for row in attempts if isinstance(row, dict)}
    if error_types.intersection({"JSONDecodeError", "JSONDecodeFailure"}):
        return "json_parse_failed"
    if "StrictSchemaRouteUnsupportedError" in error_types:
        return "strict_schema_route_unsupported"
    if "StaticProxyIdentityError" in error_types:
        return "static_proxy_identity_failure"
    if 400 in statuses or error_types.intersection(
        {"BadRequestError", "UnprocessableEntityError"}
    ):
        return "strict_schema_provider_rejected"
    return "provider_or_transport_failure"


def _call_model(
    *,
    router: Any,
    stage: str,
    prompt_version: str,
    unit: Any,
    model_spec: Mapping[str, Any],
    prompt: str,
    schema: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    safe_spec = _safe_model_spec(model_spec, stage)
    expected_response_model = safe_spec["expected_response_model"]
    schema_binding = {
        "name": "semantic_review_supplement_response",
        "strict": True,
        "schema": dict(schema),
    }
    request_spec = dict(safe_spec)
    request_spec["strict_json_schema"] = schema_binding
    validator_state: Dict[str, Optional[str]] = {"code": None}

    def validator(value: Dict[str, Any]) -> None:
        try:
            validate_response(value, unit)
        except SupplementValidationError as error:
            validator_state["code"] = error.code
            raise

    request_binding = {
        "stage": stage,
        "pair_id": unit.pair_id,
        "model_spec": safe_spec,
        "prompt": prompt,
        "strict_json_schema_sha256": parent_runner.sha256_value(schema_binding),
    }
    try:
        result = router.request_json(
            stage, unit.pair_id, request_spec, prompt, validator
        )
    except Exception as error:
        code = (
            "strict_schema_route_unsupported"
            if isinstance(error, StrictSchemaRouteUnsupportedError)
            else "static_proxy_identity_failure"
            if type(error).__name__ == "StaticProxyIdentityError"
            else "provider_or_transport_failure"
        )
        return None, {
            "stage": stage,
            "prompt_version": prompt_version,
            "pair_id": unit.pair_id,
            "provider_profile": safe_spec["provider_profile"],
            "requested_model": safe_spec["model"],
            "expected_response_model": expected_response_model,
            "response_model": None,
            "response_model_identity_status": "not_observed",
            "terminal_status": "failed",
            "validation_error_code": code,
            "structured_output_mode": "strict_json_schema_no_fallback",
            "strict_json_schema_sha256": parent_runner.sha256_value(schema_binding),
            "prompt_sha256": parent_runner.sha256_text(prompt),
            "request_sha256": parent_runner.sha256_value(request_binding),
            "response_sha256": None,
            "raw_response_sha256": None,
            "usage": {},
            "latency_ms": None,
            "attempt_count": 0,
            "attempts": [],
            "credentials_or_endpoints_included": False,
        }
    raw_response = result.raw_response if isinstance(result.raw_response, str) else None
    identity_status = (
        "not_observed"
        if result.response_model is None
        else "matched"
        if result.response_model == expected_response_model
        else "mismatch"
    )
    terminal_status = result.terminal_status
    validation_error_code: Optional[str] = None
    if terminal_status == "completed" and identity_status != "matched":
        terminal_status = "failed"
        validation_error_code = (
            "response_model_identity_missing"
            if identity_status == "not_observed"
            else "response_model_identity_mismatch"
        )
    elif terminal_status != "completed":
        validation_error_code = _failure_code(result, validator_state["code"])
    elif not isinstance(result.parsed_response, dict):
        terminal_status = "failed"
        validation_error_code = "parsed_response_missing"
    else:
        try:
            validate_response(result.parsed_response, unit)
        except SupplementValidationError as error:
            terminal_status = "failed"
            validation_error_code = error.code
    record = {
        "stage": stage,
        "prompt_version": prompt_version,
        "pair_id": unit.pair_id,
        "provider_profile": safe_spec["provider_profile"],
        "requested_model": safe_spec["model"],
        "expected_response_model": expected_response_model,
        "response_model": result.response_model,
        "response_model_identity_status": identity_status,
        "terminal_status": terminal_status,
        "validation_error_code": validation_error_code,
        "structured_output_mode": "strict_json_schema_no_fallback",
        "strict_json_schema_sha256": parent_runner.sha256_value(schema_binding),
        "prompt_sha256": parent_runner.sha256_text(prompt),
        "request_sha256": parent_runner.sha256_value(request_binding),
        "response_sha256": (
            parent_runner.sha256_text(raw_response) if raw_response is not None else None
        ),
        "raw_response_sha256": (
            parent_runner.sha256_text(raw_response) if raw_response is not None else None
        ),
        "usage": {
            key: number
            for key, number in dict(result.usage or {}).items()
            if key in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(number, int)
        },
        "latency_ms": result.latency_ms,
        "attempt_count": result.attempt_count,
        "attempts": parent_runner._safe_attempts(result.attempts),
        "credentials_or_endpoints_included": False,
    }
    return (
        (dict(result.parsed_response), record)
        if terminal_status == "completed"
        else (None, record)
    )


def _strict_adjudication(
    unit: Any,
    verdict: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    reviewed_at: str,
) -> Dict[str, Any]:
    response_model = reviewer_spec["expected_response_model"]
    decision = {
        "schema_version": parent_runner.ADJUDICATION_SCHEMA,
        "pair_id": unit.pair_id,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        **{field: verdict[field] for field in parent_runner.FIELD_NAMES},
        "relationship": verdict["relationship"],
        "rationale": verdict["rationale"],
        "confidence": verdict["confidence"],
        "reviewer_type": "independent_model_proxy",
        "reviewer_id": f"{reviewer_spec['provider_profile']}:{response_model}",
        "requested_model": reviewer_spec["model"],
        "expected_response_model": response_model,
        "response_model": response_model,
        "response_model_identity_status": "matched",
        "review_method": REVIEW_METHOD,
        "reviewed_at": reviewed_at,
        "behavior_blind": True,
        "human_gold": False,
    }
    _validate_adjudication(decision, unit=unit, reviewer_spec=reviewer_spec)
    return decision


def _validate_adjudication(
    decision: Mapping[str, Any], *, unit: Any, reviewer_spec: Mapping[str, Any]
) -> None:
    expected_fields = {
        "schema_version",
        "pair_id",
        "candidate_row_sha256",
        "adjudication_template_row_sha256",
        "left_endpoint_id",
        "left_input_record_sha256",
        "right_endpoint_id",
        "right_input_record_sha256",
        *parent_runner.FIELD_NAMES,
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
    if set(decision) != expected_fields:
        raise ValueError(f"Supplement adjudication fields are invalid: {unit.pair_id}")
    expected = {
        "schema_version": parent_runner.ADJUDICATION_SCHEMA,
        "pair_id": unit.pair_id,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        "reviewer_type": "independent_model_proxy",
        "reviewer_id": (
            f"{reviewer_spec['provider_profile']}:"
            f"{reviewer_spec['expected_response_model']}"
        ),
        "requested_model": reviewer_spec["model"],
        "expected_response_model": reviewer_spec["expected_response_model"],
        "response_model": reviewer_spec["expected_response_model"],
        "response_model_identity_status": "matched",
        "review_method": REVIEW_METHOD,
        "behavior_blind": True,
        "human_gold": False,
    }
    for field, value in expected.items():
        if decision.get(field) != value:
            raise ValueError(
                f"Supplement adjudication {field} is invalid: {unit.pair_id}"
            )
    _required_string(decision.get("reviewed_at"), f"{unit.pair_id}.reviewed_at")
    response = {
        "schema_version": RESPONSE_SCHEMA,
        "pair_id": unit.pair_id,
        "relationship": decision.get("relationship"),
        **{field: decision.get(field) for field in parent_runner.FIELD_NAMES},
        "rationale": decision.get("rationale"),
        "confidence": decision.get("confidence"),
    }
    validate_response(response, unit)


def _failed_checkpoint_row(
    *,
    unit: Any,
    parent_row: Mapping[str, Any],
    run_contract_sha256: str,
    failure_stage: str,
    model_calls: Sequence[Mapping[str, Any]],
    generator_verdict: Optional[Mapping[str, Any]],
    now_fn: Callable[[], str],
) -> Dict[str, Any]:
    code = str(model_calls[-1].get("validation_error_code"))
    if code not in VALIDATION_ERROR_CODES:
        raise ValueError("Failed semantic supplement call has no finite code")
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "pair_id": unit.pair_id,
        "input_index": unit.input_index,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        "parent_failed_checkpoint_record_sha256": parent_runner.sha256_value(parent_row),
        "run_contract_sha256": run_contract_sha256,
        "terminal_status": "failed",
        "failure_stage": failure_stage,
        "validation_error_code": code,
        "generator_verdict": dict(generator_verdict) if generator_verdict else None,
        "reviewer_verdict": None,
        "strict_adjudication": None,
        "model_calls": [dict(call) for call in model_calls],
        "retry_history": [],
        "completed_at": now_fn(),
        "behavior_blind": True,
        "human_gold": False,
        "hf_model_execution_count": 0,
        "hf_tokenizer_execution_count": 0,
        "validation_behavior_exposure_count": 0,
        "sealed_behavior_exposure_count": 0,
        "behavior_execution_count": 0,
    }


def _process_unit(
    *,
    unit: Any,
    parent_row: Mapping[str, Any],
    router: Any,
    generator_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    run_contract_sha256: str,
    now_fn: Callable[[], str],
) -> Dict[str, Any]:
    schema = strict_response_schema(unit)
    draft, generator_call = _call_model(
        router=router,
        stage="semantic_review_supplement_generator",
        prompt_version=GENERATOR_PROMPT_VERSION,
        unit=unit,
        model_spec=generator_spec,
        prompt=build_generator_prompt(unit, schema),
        schema=schema,
    )
    if draft is None:
        return _failed_checkpoint_row(
            unit=unit,
            parent_row=parent_row,
            run_contract_sha256=run_contract_sha256,
            failure_stage="generator",
            model_calls=[generator_call],
            generator_verdict=None,
            now_fn=now_fn,
        )
    verdict, reviewer_call = _call_model(
        router=router,
        stage="semantic_review_supplement_reviewer",
        prompt_version=REVIEWER_PROMPT_VERSION,
        unit=unit,
        model_spec=reviewer_spec,
        prompt=build_reviewer_prompt(unit, draft, schema),
        schema=schema,
    )
    calls = [generator_call, reviewer_call]
    if verdict is None:
        return _failed_checkpoint_row(
            unit=unit,
            parent_row=parent_row,
            run_contract_sha256=run_contract_sha256,
            failure_stage="reviewer",
            model_calls=calls,
            generator_verdict=draft,
            now_fn=now_fn,
        )
    completed_at = now_fn()
    decision = _strict_adjudication(unit, verdict, reviewer_spec, completed_at)
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "pair_id": unit.pair_id,
        "input_index": unit.input_index,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        "parent_failed_checkpoint_record_sha256": parent_runner.sha256_value(parent_row),
        "run_contract_sha256": run_contract_sha256,
        "terminal_status": "completed",
        "failure_stage": None,
        "validation_error_code": None,
        "generator_verdict": dict(draft),
        "reviewer_verdict": dict(verdict),
        "strict_adjudication": decision,
        "model_calls": calls,
        "retry_history": [],
        "completed_at": completed_at,
        "behavior_blind": True,
        "human_gold": False,
        "hf_model_execution_count": 0,
        "hf_tokenizer_execution_count": 0,
        "validation_behavior_exposure_count": 0,
        "sealed_behavior_exposure_count": 0,
        "behavior_execution_count": 0,
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_model_call_trace(
    call: Any,
    *,
    pair_id: str,
    expected_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(call, dict) or not set(call).issubset(MODEL_CALL_FIELDS):
        raise ValueError(f"Semantic supplement model call contains unsafe fields: {pair_id}")
    if not MODEL_CALL_FIELDS.issubset(call):
        raise ValueError(f"Semantic supplement model call fields are incomplete: {pair_id}")
    if call.get("pair_id") != pair_id:
        raise ValueError(f"Semantic supplement model call pair_id is stale: {pair_id}")
    if call.get("credentials_or_endpoints_included") is not False:
        raise ValueError(f"Semantic supplement model trace is not redacted: {pair_id}")
    attempts = call.get("attempts")
    if attempts != parent_runner._safe_attempts(attempts):
        raise ValueError(f"Semantic supplement attempts contain unsafe fields: {pair_id}")
    attempt_count = call.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count != len(attempts)
    ):
        raise ValueError(f"Semantic supplement attempt_count is invalid: {pair_id}")
    for index, attempt in enumerate(attempts, start=1):
        if attempt.get("attempt") != index or attempt.get("status") not in {
            "ok",
            "failed",
        }:
            raise ValueError(f"Semantic supplement attempt trace is invalid: {pair_id}")
    stage = str(call.get("stage") or "")
    spec = expected_specs.get(stage)
    if spec is None:
        raise ValueError(f"Semantic supplement model stage is invalid: {pair_id}")
    prompt_version = {
        "semantic_review_supplement_generator": GENERATOR_PROMPT_VERSION,
        "semantic_review_supplement_reviewer": REVIEWER_PROMPT_VERSION,
    }[stage]
    if call.get("prompt_version") != prompt_version:
        raise ValueError(f"Semantic supplement prompt version is stale: {pair_id}")
    expected_identity = {
        "provider_profile": spec["provider_profile"],
        "requested_model": spec["model"],
        "expected_response_model": spec["expected_response_model"],
    }
    if any(call.get(field) != value for field, value in expected_identity.items()):
        raise ValueError(f"Semantic supplement model identity is stale: {pair_id}")
    if call.get("structured_output_mode") != "strict_json_schema_no_fallback":
        raise ValueError(f"Semantic supplement strict schema trace is missing: {pair_id}")
    for field in ("strict_json_schema_sha256", "prompt_sha256", "request_sha256"):
        if not _is_sha256(call.get(field)):
            raise ValueError(f"Semantic supplement call hash is invalid: {pair_id}")
    response_sha = call.get("response_sha256")
    if response_sha != call.get("raw_response_sha256") or (
        response_sha is not None and not _is_sha256(response_sha)
    ):
        raise ValueError(f"Semantic supplement response hash is invalid: {pair_id}")
    usage = call.get("usage")
    if not isinstance(usage, dict) or not set(usage).issubset(
        {"input_tokens", "output_tokens", "total_tokens"}
    ) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in usage.values()
    ):
        raise ValueError(f"Semantic supplement usage trace is invalid: {pair_id}")
    response_model = call.get("response_model")
    expected_identity_status = (
        "not_observed"
        if response_model is None
        else "matched"
        if response_model == spec["expected_response_model"]
        else "mismatch"
    )
    if call.get("response_model_identity_status") != expected_identity_status:
        raise ValueError(f"Semantic supplement response identity is inconsistent: {pair_id}")
    status = call.get("terminal_status")
    code = call.get("validation_error_code")
    if status == "completed":
        if (
            code is not None
            or expected_identity_status != "matched"
            or not _is_sha256(response_sha)
            or attempt_count < 1
            or attempts[-1].get("status") != "ok"
        ):
            raise ValueError(f"Completed semantic supplement call is invalid: {pair_id}")
        if call.get("response_model") != spec["expected_response_model"]:
            raise ValueError(f"Completed semantic supplement identity is invalid: {pair_id}")
    elif status == "failed":
        if code not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Failed semantic supplement call lacks finite code: {pair_id}")
        if attempt_count == 0 and (
            response_model is not None
            or response_sha is not None
            or code
            not in {
                "strict_schema_route_unsupported",
                "static_proxy_identity_failure",
                "provider_or_transport_failure",
            }
        ):
            raise ValueError(f"Failed semantic supplement call trace is invalid: {pair_id}")
    else:
        raise ValueError(f"Semantic supplement call status is invalid: {pair_id}")


def _validate_call_bindings(
    call: Mapping[str, Any],
    *,
    unit: Any,
    model_spec: Mapping[str, Any],
    prompt: str,
) -> None:
    schema_binding = {
        "name": "semantic_review_supplement_response",
        "strict": True,
        "schema": strict_response_schema(unit),
    }
    request_binding = {
        "stage": call["stage"],
        "pair_id": unit.pair_id,
        "model_spec": _safe_model_spec(model_spec, str(call["stage"])),
        "prompt": prompt,
        "strict_json_schema_sha256": parent_runner.sha256_value(schema_binding),
    }
    expected = {
        "strict_json_schema_sha256": parent_runner.sha256_value(schema_binding),
        "prompt_sha256": parent_runner.sha256_text(prompt),
        "request_sha256": parent_runner.sha256_value(request_binding),
    }
    for field, value in expected.items():
        if call.get(field) != value:
            raise ValueError(
                f"Semantic supplement {field} cannot be replayed: {unit.pair_id}"
            )


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    units_by_id: Mapping[str, Any],
    parent_rows: Mapping[str, Mapping[str, Any]],
    run_contract_sha256: str,
    expected_specs: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    pair_id = _required_string(row.get("pair_id"), "checkpoint.pair_id")
    if set(row) != CHECKPOINT_FIELDS:
        raise ValueError(f"Semantic supplement checkpoint fields are invalid: {pair_id}")
    unit = units_by_id.get(pair_id)
    if unit is None:
        raise ValueError(f"Semantic supplement checkpoint has unbound ID: {pair_id}")
    expected_bindings = {
        "schema_version": CHECKPOINT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "input_index": unit.input_index,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        "parent_failed_checkpoint_record_sha256": parent_runner.sha256_value(
            parent_rows[pair_id]
        ),
        "run_contract_sha256": run_contract_sha256,
        "behavior_blind": True,
        "human_gold": False,
        "hf_model_execution_count": 0,
        "hf_tokenizer_execution_count": 0,
        "validation_behavior_exposure_count": 0,
        "sealed_behavior_exposure_count": 0,
        "behavior_execution_count": 0,
    }
    for field, expected in expected_bindings.items():
        if row.get(field) != expected:
            raise ValueError(f"Semantic supplement checkpoint {field} is stale: {pair_id}")
    calls = row.get("model_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"Semantic supplement model calls are missing: {pair_id}")
    for call in calls:
        _validate_model_call_trace(call, pair_id=pair_id, expected_specs=expected_specs)
    expected_stage_prefix = [
        "semantic_review_supplement_generator",
        "semantic_review_supplement_reviewer",
    ]
    observed_stages = [call.get("stage") for call in calls]
    if observed_stages != expected_stage_prefix[: len(observed_stages)]:
        raise ValueError(f"Semantic supplement call order is invalid: {pair_id}")
    generator_prompt = build_generator_prompt(unit, strict_response_schema(unit))
    _validate_call_bindings(
        calls[0],
        unit=unit,
        model_spec=expected_specs["semantic_review_supplement_generator"],
        prompt=generator_prompt,
    )
    if len(calls) == 2:
        generator_verdict = row.get("generator_verdict")
        if not isinstance(generator_verdict, dict):
            raise ValueError(
                f"Semantic supplement reviewer call lacks generator draft: {pair_id}"
            )
        validate_response(generator_verdict, unit)
        reviewer_prompt = build_reviewer_prompt(
            unit, generator_verdict, strict_response_schema(unit)
        )
        _validate_call_bindings(
            calls[1],
            unit=unit,
            model_spec=expected_specs["semantic_review_supplement_reviewer"],
            prompt=reviewer_prompt,
        )
    retry_history = row.get("retry_history")
    if not isinstance(retry_history, list):
        raise ValueError(f"Semantic supplement retry_history is invalid: {pair_id}")
    for prior in retry_history:
        if not isinstance(prior, dict) or set(prior) != {
            "terminal_status",
            "failure_stage",
            "validation_error_code",
            "model_calls",
            "completed_at",
        }:
            raise ValueError(f"Semantic supplement retry history is invalid: {pair_id}")
        if prior.get("terminal_status") != "failed" or prior.get(
            "validation_error_code"
        ) not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Semantic supplement retry status is invalid: {pair_id}")
        prior_calls = prior.get("model_calls")
        if not isinstance(prior_calls, list) or not prior_calls:
            raise ValueError(f"Semantic supplement retry calls are invalid: {pair_id}")
        prior_stages = [call.get("stage") for call in prior_calls]
        if prior_stages != expected_stage_prefix[: len(prior_stages)] or len(
            prior_stages
        ) not in {1, 2}:
            raise ValueError(f"Semantic supplement retry call order is invalid: {pair_id}")
        expected_prior_failure_stage = (
            "generator" if len(prior_calls) == 1 else "reviewer"
        )
        expected_prior_statuses = (
            ["failed"] if len(prior_calls) == 1 else ["completed", "failed"]
        )
        if (
            prior.get("failure_stage") != expected_prior_failure_stage
            or [call.get("terminal_status") for call in prior_calls]
            != expected_prior_statuses
            or prior_calls[-1].get("validation_error_code")
            != prior.get("validation_error_code")
        ):
            raise ValueError(f"Semantic supplement retry failure is invalid: {pair_id}")
        _validate_timestamp(
            prior.get("completed_at"), f"{pair_id}.retry.completed_at"
        )
        for call in prior_calls:
            _validate_model_call_trace(
                call, pair_id=pair_id, expected_specs=expected_specs
            )
    status = row.get("terminal_status")
    completed_at = _validate_timestamp(
        row.get("completed_at"), f"{pair_id}.completed_at"
    )
    if status == "completed":
        if observed_stages != expected_stage_prefix or any(
            call.get("terminal_status") != "completed" for call in calls
        ):
            raise ValueError(f"Completed semantic supplement lacks both roles: {pair_id}")
        if row.get("validation_error_code") is not None or row.get("failure_stage") is not None:
            raise ValueError(f"Completed semantic supplement has failure metadata: {pair_id}")
        if not isinstance(row.get("generator_verdict"), dict) or not isinstance(
            row.get("reviewer_verdict"), dict
        ):
            raise ValueError(f"Completed semantic supplement lacks verdicts: {pair_id}")
        validate_response(row["generator_verdict"], unit)
        validate_response(row["reviewer_verdict"], unit)
        decision = row.get("strict_adjudication")
        if not isinstance(decision, dict):
            raise ValueError(f"Completed semantic supplement lacks adjudication: {pair_id}")
        _validate_adjudication(
            decision,
            unit=unit,
            reviewer_spec=expected_specs["semantic_review_supplement_reviewer"],
        )
        expected_decision = _strict_adjudication(
            unit,
            row["reviewer_verdict"],
            expected_specs["semantic_review_supplement_reviewer"],
            completed_at,
        )
        if decision != expected_decision:
            raise ValueError(
                f"Semantic supplement adjudication does not replay: {pair_id}"
            )
        response_models = [str(call.get("response_model")) for call in calls]
        if len(set(response_models)) != 2:
            raise ValueError(f"Semantic supplement role identities collide: {pair_id}")
    elif status == "failed":
        if row.get("strict_adjudication") is not None:
            raise ValueError(f"Failed semantic supplement has adjudication: {pair_id}")
        if row.get("failure_stage") not in {"generator", "reviewer"} or row.get(
            "validation_error_code"
        ) not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Failed semantic supplement metadata is invalid: {pair_id}")
        expected_failure_stage = "generator" if len(calls) == 1 else "reviewer"
        expected_call_statuses = (
            ["failed"] if len(calls) == 1 else ["completed", "failed"]
        )
        if (
            len(calls) not in {1, 2}
            or row.get("failure_stage") != expected_failure_stage
            or [call.get("terminal_status") for call in calls]
            != expected_call_statuses
            or calls[-1].get("validation_error_code")
            != row.get("validation_error_code")
            or row.get("reviewer_verdict") is not None
            or (
                expected_failure_stage == "generator"
                and row.get("generator_verdict") is not None
            )
            or (
                expected_failure_stage == "reviewer"
                and not isinstance(row.get("generator_verdict"), dict)
            )
        ):
            raise ValueError(f"Failed semantic supplement lineage is invalid: {pair_id}")
    else:
        raise ValueError(f"Semantic supplement terminal status is invalid: {pair_id}")
    return pair_id, dict(row)


def _write_runtime_artifacts(
    checkpoint_path: Path,
    adjudications_path: Path,
    checkpoint: Mapping[str, Mapping[str, Any]],
    units: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = [
        dict(checkpoint[str(unit.pair_id)])
        for unit in units
        if str(unit.pair_id) in checkpoint
    ]
    adjudications = [
        dict(row["strict_adjudication"])
        for row in rows
        if row.get("terminal_status") == "completed"
    ]
    parent_runner.write_jsonl(checkpoint_path, rows)
    parent_runner.write_jsonl(adjudications_path, adjudications)
    return rows, adjudications


def run_recovery(
    *,
    parent_run_manifest_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    env_path: Path,
    generator_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    batch_size: int = FIXED_BATCH_SIZE,
    resume: bool = True,
    retry_failed: bool = False,
    now_fn: Callable[[], str] = parent_runner.utc_now,
) -> Dict[str, Any]:
    if batch_size != FIXED_BATCH_SIZE:
        raise ValueError("Semantic supplement batch_size is fixed at 1")
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    authority = inspect_parent_authority(parent_run_manifest_path)
    generator_spec = _safe_model_spec(generator_spec, "generator")
    reviewer_spec = _safe_model_spec(reviewer_spec, "reviewer")
    protocol_identity = protocol_reviewer_identity(config)
    parent_contract = authority.manifest.get("run_contract")
    if not isinstance(parent_contract, dict):
        raise ValueError("Parent semantic run contract is missing")
    if parent_contract.get("config_sha256") != parent_runner.sha256_value(config):
        raise ValueError("Semantic supplement config differs from parent authority")
    if parent_contract.get("protocol_reviewer_identity") != protocol_identity:
        raise ValueError("Semantic supplement reviewer protocol differs from parent authority")
    if (
        reviewer_spec["provider_profile"] != protocol_identity["provider_profile"]
        or reviewer_spec["model"] != protocol_identity["requested_model"]
        or reviewer_spec["expected_response_model"]
        != protocol_identity["expected_response_model"]
    ):
        raise ValueError("reviewer identity differs from the protocol config")
    _validate_strict_schema_routes(config, (generator_spec, reviewer_spec))
    if (
        generator_spec["provider_profile"], generator_spec["model"]
    ) == (reviewer_spec["provider_profile"], reviewer_spec["model"]):
        raise ValueError("generator and reviewer must use distinct model identities")
    if generator_spec["expected_response_model"] == reviewer_spec[
        "expected_response_model"
    ]:
        raise ValueError("generator and reviewer must use distinct resolved identities")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "supplement_semantic_review_checkpoint.jsonl"
    adjudications_path = output_dir / "supplement_semantic_adjudications.jsonl"
    manifest_path = output_dir / "supplement_semantic_review_run_manifest.json"
    event_log_path = output_dir / "supplement_redacted_model_events.jsonl"
    managed = (checkpoint_path, adjudications_path, manifest_path, event_log_path)
    if not resume and any(path.exists() for path in managed):
        raise FileExistsError("Semantic supplement outputs exist; choose a new directory")

    router = _create_runtime_router(config, Path(env_path), event_log_path)
    route_identity = build_route_identity_set(
        config=config,
        env_path=Path(env_path).resolve(),
        routes=[
            (generator_spec, generator_spec["expected_response_model"]),
            (reviewer_spec, reviewer_spec["expected_response_model"]),
        ],
        resolved_env=(
            router.values
            if isinstance(getattr(router, "values", None), Mapping)
            else None
        ),
    )
    validate_route_identity_set(
        route_identity,
        expected_routes=[
            (
                generator_spec["provider_profile"],
                generator_spec["model"],
                generator_spec["expected_response_model"],
            ),
            (
                reviewer_spec["provider_profile"],
                reviewer_spec["model"],
                reviewer_spec["expected_response_model"],
            ),
        ],
    )
    attach_route_identity_guard(
        router,
        config=config,
        env_path=Path(env_path).resolve(),
        identity_set=route_identity,
    )

    failed_ids = list(authority.failed_ids)
    prompt_contract = {
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "generator_instructions": GENERATOR_INSTRUCTIONS,
        "reviewer_prompt_version": REVIEWER_PROMPT_VERSION,
        "reviewer_instructions": REVIEWER_INSTRUCTIONS,
        "response_schema_version": RESPONSE_SCHEMA,
        "strict_json_schema": True,
        "json_mode_fallback_allowed": False,
        "validation_error_codes": sorted(VALIDATION_ERROR_CODES),
    }
    runtime_sources = _runtime_source_bindings()
    run_contract = {
        "tool_version": TOOL_VERSION,
        "runtime_sources": runtime_sources,
        "parent_run_manifest_sha256": authority.inputs["parent_run_manifest"]["sha256"],
        "parent_run_contract_sha256": authority.manifest["run_contract_sha256"],
        "parent_checkpoint_sha256": authority.inputs["parent_checkpoint"]["sha256"],
        "parent_adjudications_sha256": authority.inputs["parent_adjudications"]["sha256"],
        "candidate_manifest_sha256": authority.inputs["candidate_manifest"]["sha256"],
        "candidate_pairs_sha256": authority.inputs["candidate_pairs"]["sha256"],
        "adjudication_template_sha256": authority.inputs["adjudication_template"]["sha256"],
        "parent_selected_count": len(authority.units),
        "parent_completed_count": len(authority.completed_ids),
        "parent_failed_count": len(failed_ids),
        "failed_pair_ids": failed_ids,
        "failed_pair_ids_sha256": parent_runner.sha256_value(failed_ids),
        "selected_pair_count": len(failed_ids),
        "batch_size": FIXED_BATCH_SIZE,
        "config_sha256": parent_runner.sha256_value(config),
        "generator_model_spec": generator_spec,
        "reviewer_model_spec": reviewer_spec,
        "protocol_reviewer_identity": protocol_identity,
        "model_identity_distinct": True,
        "resolved_response_identity_distinct": True,
        "infrastructure_distinct": False,
        "infrastructure_independence_claimed": False,
        "route_identity": route_identity,
        "prompt_contract_sha256": parent_runner.sha256_value(prompt_contract),
        "review_method": REVIEW_METHOD,
        "response_identity_enforcement": RESPONSE_IDENTITY_ENFORCEMENT,
        "behavior_blind": True,
        "human_gold": False,
    }
    run_contract_sha = parent_runner.sha256_value(run_contract)
    if resume and manifest_path.is_file():
        old = parent_runner.read_json(manifest_path)
        if old.get("run_contract_sha256") != run_contract_sha:
            raise ValueError("Resume semantic supplement run contract mismatch")

    units_by_id = {str(unit.pair_id): unit for unit in authority.failed_units}
    expected_specs = {
        "semantic_review_supplement_generator": generator_spec,
        "semantic_review_supplement_reviewer": reviewer_spec,
    }
    checkpoint: Dict[str, Dict[str, Any]] = {}
    if resume and checkpoint_path.is_file():
        for row in parent_runner.read_jsonl(checkpoint_path, allow_empty=True):
            pair_id, validated = _validate_checkpoint_row(
                row,
                units_by_id=units_by_id,
                parent_rows=authority.failed_checkpoint_rows,
                run_contract_sha256=run_contract_sha,
                expected_specs=expected_specs,
            )
            if pair_id in checkpoint:
                raise ValueError(f"Duplicate semantic supplement checkpoint: {pair_id}")
            checkpoint[pair_id] = validated
    elif not resume:
        parent_runner.write_jsonl(checkpoint_path, [])
        parent_runner.write_jsonl(adjudications_path, [])

    pending = [
        unit
        for unit in authority.failed_units
        if str(unit.pair_id) not in checkpoint
        or (
            retry_failed
            and checkpoint[str(unit.pair_id)].get("terminal_status") == "failed"
        )
    ]
    for unit in pending:
        pair_id = str(unit.pair_id)
        row = _process_unit(
            unit=unit,
            parent_row=authority.failed_checkpoint_rows[pair_id],
            router=router,
            generator_spec=generator_spec,
            reviewer_spec=reviewer_spec,
            run_contract_sha256=run_contract_sha,
            now_fn=now_fn,
        )
        previous = checkpoint.get(pair_id)
        if previous is not None and previous.get("terminal_status") == "failed":
            history = list(previous.get("retry_history") or [])
            history.append(
                {
                    "terminal_status": "failed",
                    "failure_stage": previous.get("failure_stage"),
                    "validation_error_code": previous.get("validation_error_code"),
                    "model_calls": copy.deepcopy(previous.get("model_calls") or []),
                    "completed_at": previous.get("completed_at"),
                }
            )
            row["retry_history"] = history
        checkpoint[pair_id] = row
        _write_runtime_artifacts(
            checkpoint_path, adjudications_path, checkpoint, authority.failed_units
        )

    rows, adjudications = _write_runtime_artifacts(
        checkpoint_path, adjudications_path, checkpoint, authority.failed_units
    )
    counts = Counter(str(row["terminal_status"]) for row in rows)
    status = "completed" if counts.get("completed", 0) == len(failed_ids) else "completed_with_failures"
    relationship_counts = Counter(
        str(row["relationship"]) for row in adjudications
    )

    _verify_bindings_unchanged(authority.inputs)
    _verify_bindings_unchanged(runtime_sources)
    current_route_identity = build_route_identity_set(
        config=config,
        env_path=Path(env_path).resolve(),
        routes=[
            (generator_spec, generator_spec["expected_response_model"]),
            (reviewer_spec, reviewer_spec["expected_response_model"]),
        ],
    )
    if current_route_identity != route_identity:
        raise ValueError("Bound semantic supplement route changed during run")

    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "status": status,
        "formal_universe_id": authority.manifest["run_contract"].get(
            "formal_universe_id"
        ),
        "inputs": authority.inputs,
        "parent_run_contract_sha256": authority.manifest["run_contract_sha256"],
        "parent_terminal_status_counts": {
            "completed": len(authority.completed_ids),
            "failed": len(failed_ids),
        },
        "failed_pair_ids": failed_ids,
        "failed_pair_ids_sha256": parent_runner.sha256_value(failed_ids),
        "selected_pair_count": len(failed_ids),
        "batch_size": FIXED_BATCH_SIZE,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha,
        "prompt_contract": prompt_contract,
        "terminal_status_counts": dict(sorted(counts.items())),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "artifacts": {
            "checkpoint": _binding(
                checkpoint_path,
                schema_version=CHECKPOINT_SCHEMA,
                record_count=len(rows),
            ),
            "adjudications": _binding(
                adjudications_path,
                schema_version=parent_runner.ADJUDICATION_SCHEMA,
                record_count=len(adjudications),
            ),
        },
        "model_roles": {
            "generator": generator_spec,
            "reviewer": reviewer_spec,
            "model_identity_distinct": True,
            "resolved_response_identity_distinct": True,
            "infrastructure_distinct": False,
            "infrastructure_independence_claimed": False,
        },
        "reviewer_type": "independent_model_proxy",
        "human_gold": False,
        "evidence_boundary": {
            "behavior_blind": True,
            "target_behavior_consumed": False,
            "hf_model_execution_count": 0,
            "hf_tokenizer_execution_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
            "human_gold": False,
        },
        "safety_contract": {
            "parent_artifacts_modified": False,
            "failed_ids_derived_only_from_parent_checkpoint": True,
            "caller_supplied_selection_allowed": False,
            "batch_size_fixed_at_one": True,
            "strict_json_schema_required": True,
            "json_mode_fallback_allowed": False,
            "exact_response_identity_required": True,
            "model_identity_independent": True,
            "infrastructure_distinct": False,
            "infrastructure_independence_claimed": False,
            "behavior_blind_projection_only": True,
            "target_behavior_consumed": False,
            "hf_model_initialized": False,
            "hf_tokenizer_initialized": False,
            "validation_exposed": False,
            "sealed_exposed": False,
            "human_gold": False,
            "credentials_or_endpoints_included": False,
        },
    }
    parent_runner.write_json(manifest_path, manifest)
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": parent_runner.sha256_file(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "adjudications_path": str(adjudications_path),
        "selected_pair_count": len(failed_ids),
        "terminal_status_counts": manifest["terminal_status_counts"],
        "relationship_counts": manifest["relationship_counts"],
    }


def load_supplement_authority(
    path: Path, *, parent: Optional[ParentAuthority] = None
) -> Dict[str, Any]:
    """Strictly replay a completed supplement against its immutable parent."""

    path = Path(path).resolve()
    manifest = parent_runner.read_json(path)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("Semantic supplement manifest schema is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION or manifest.get("status") != "completed":
        raise ValueError("Semantic supplement authority is incomplete")
    if manifest.get("human_gold") is not False:
        raise ValueError("Semantic supplement must keep human_gold=false")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Semantic supplement inputs are missing")
    parent_manifest_path = _bound_path(
        inputs.get("parent_run_manifest"), path, "supplement parent run manifest"
    )
    parent = parent or inspect_parent_authority(parent_manifest_path)
    for label, expected in parent.inputs.items():
        _same_bound_file(
            inputs.get(label),
            expected,
            actual_owner=path,
            expected_owner=parent.manifest_path,
            label=f"supplement {label}",
        )
    if manifest.get("failed_pair_ids") != parent.failed_ids or manifest.get(
        "failed_pair_ids_sha256"
    ) != parent_runner.sha256_value(parent.failed_ids):
        raise ValueError("Semantic supplement selection differs from parent failures")
    if not all(
        (
            manifest.get("selected_pair_count") == len(parent.failed_ids),
            manifest.get("batch_size") == FIXED_BATCH_SIZE,
            manifest.get("parent_terminal_status_counts")
            == {
                "completed": len(parent.completed_ids),
                "failed": len(parent.failed_ids),
            },
        )
    ):
        raise ValueError("Semantic supplement top-level counts are stale")

    contract = manifest.get("run_contract")
    if not isinstance(contract, dict) or manifest.get(
        "run_contract_sha256"
    ) != parent_runner.sha256_value(contract):
        raise ValueError("Semantic supplement run contract is stale")
    expected_contract_values = {
        "tool_version": TOOL_VERSION,
        "parent_run_manifest_sha256": parent.inputs["parent_run_manifest"]["sha256"],
        "parent_run_contract_sha256": parent.manifest["run_contract_sha256"],
        "parent_checkpoint_sha256": parent.inputs["parent_checkpoint"]["sha256"],
        "parent_adjudications_sha256": parent.inputs["parent_adjudications"]["sha256"],
        "candidate_manifest_sha256": parent.inputs["candidate_manifest"]["sha256"],
        "candidate_pairs_sha256": parent.inputs["candidate_pairs"]["sha256"],
        "adjudication_template_sha256": parent.inputs["adjudication_template"]["sha256"],
        "parent_selected_count": len(parent.units),
        "parent_completed_count": len(parent.completed_ids),
        "parent_failed_count": len(parent.failed_ids),
        "failed_pair_ids": parent.failed_ids,
        "failed_pair_ids_sha256": parent_runner.sha256_value(parent.failed_ids),
        "selected_pair_count": len(parent.failed_ids),
        "batch_size": FIXED_BATCH_SIZE,
        "config_sha256": parent.manifest["run_contract"]["config_sha256"],
        "protocol_reviewer_identity": parent.manifest["run_contract"][
            "protocol_reviewer_identity"
        ],
        "review_method": REVIEW_METHOD,
        "response_identity_enforcement": RESPONSE_IDENTITY_ENFORCEMENT,
        "model_identity_distinct": True,
        "resolved_response_identity_distinct": True,
        "infrastructure_distinct": False,
        "infrastructure_independence_claimed": False,
        "behavior_blind": True,
        "human_gold": False,
    }
    if any(contract.get(field) != value for field, value in expected_contract_values.items()):
        raise ValueError("Semantic supplement run contract differs from parent authority")
    runtime_sources = contract.get("runtime_sources")
    if runtime_sources != _runtime_source_bindings():
        raise ValueError("Semantic supplement runtime source bindings are stale")
    generator_spec = _safe_model_spec(contract.get("generator_model_spec") or {}, "generator")
    reviewer_spec = _safe_model_spec(contract.get("reviewer_model_spec") or {}, "reviewer")
    protocol_identity = contract.get("protocol_reviewer_identity")
    if not isinstance(protocol_identity, dict) or protocol_identity != parent.manifest[
        "run_contract"
    ].get("protocol_reviewer_identity"):
        raise ValueError("Semantic supplement protocol reviewer identity is stale")
    if (
        reviewer_spec["provider_profile"] != protocol_identity["provider_profile"]
        or reviewer_spec["model"] != protocol_identity["requested_model"]
        or reviewer_spec["expected_response_model"]
        != protocol_identity["expected_response_model"]
    ):
        raise ValueError("Semantic supplement reviewer identity is unsupported")
    if (
        generator_spec["provider_profile"], generator_spec["model"]
    ) == (reviewer_spec["provider_profile"], reviewer_spec["model"]) or generator_spec[
        "expected_response_model"
    ] == reviewer_spec["expected_response_model"]:
        raise ValueError("Semantic supplement model roles are not independent")
    validate_route_identity_set(
        contract.get("route_identity"),
        expected_routes=[
            (
                generator_spec["provider_profile"],
                generator_spec["model"],
                generator_spec["expected_response_model"],
            ),
            (
                reviewer_spec["provider_profile"],
                reviewer_spec["model"],
                reviewer_spec["expected_response_model"],
            ),
        ],
    )
    prompt_contract = manifest.get("prompt_contract")
    expected_prompt_contract = {
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "generator_instructions": GENERATOR_INSTRUCTIONS,
        "reviewer_prompt_version": REVIEWER_PROMPT_VERSION,
        "reviewer_instructions": REVIEWER_INSTRUCTIONS,
        "response_schema_version": RESPONSE_SCHEMA,
        "strict_json_schema": True,
        "json_mode_fallback_allowed": False,
        "validation_error_codes": sorted(VALIDATION_ERROR_CODES),
    }
    if prompt_contract != expected_prompt_contract or contract.get(
        "prompt_contract_sha256"
    ) != parent_runner.sha256_value(expected_prompt_contract):
        raise ValueError("Semantic supplement prompt contract is stale")
    expected_model_roles = {
        "generator": generator_spec,
        "reviewer": reviewer_spec,
        "model_identity_distinct": True,
        "resolved_response_identity_distinct": True,
        "infrastructure_distinct": False,
        "infrastructure_independence_claimed": False,
    }
    if manifest.get("model_roles") != expected_model_roles:
        raise ValueError("Semantic supplement model role contract is stale")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Semantic supplement artifacts are missing")
    checkpoint_path = _bound_path(
        artifacts.get("checkpoint"), path, "semantic supplement checkpoint"
    )
    adjudications_path = _bound_path(
        artifacts.get("adjudications"), path, "semantic supplement adjudications"
    )
    checkpoint_rows = parent_runner.read_jsonl(checkpoint_path)
    adjudications = parent_runner.read_jsonl(adjudications_path)
    if artifacts["checkpoint"].get("schema_version") != CHECKPOINT_SCHEMA or artifacts[
        "checkpoint"
    ].get("record_count") != len(checkpoint_rows):
        raise ValueError("Semantic supplement checkpoint binding is stale")
    if artifacts["adjudications"].get("schema_version") != parent_runner.ADJUDICATION_SCHEMA or artifacts[
        "adjudications"
    ].get("record_count") != len(adjudications):
        raise ValueError("Semantic supplement adjudication binding is stale")
    if len(checkpoint_rows) != len(parent.failed_ids) or len(adjudications) != len(
        parent.failed_ids
    ):
        raise ValueError("Semantic supplement does not exactly cover parent failures")
    units_by_id = {str(unit.pair_id): unit for unit in parent.failed_units}
    expected_specs = {
        "semantic_review_supplement_generator": generator_spec,
        "semantic_review_supplement_reviewer": reviewer_spec,
    }
    validated_rows: List[Dict[str, Any]] = []
    for expected_id, row in zip(parent.failed_ids, checkpoint_rows):
        pair_id, validated = _validate_checkpoint_row(
            row,
            units_by_id=units_by_id,
            parent_rows=parent.failed_checkpoint_rows,
            run_contract_sha256=str(manifest["run_contract_sha256"]),
            expected_specs=expected_specs,
        )
        if pair_id != expected_id or validated.get("terminal_status") != "completed":
            raise ValueError("Semantic supplement checkpoint order or status is invalid")
        validated_rows.append(validated)
    expected_adjudications = [dict(row["strict_adjudication"]) for row in validated_rows]
    if adjudications != expected_adjudications:
        raise ValueError("Semantic supplement adjudications differ from checkpoints")
    if manifest.get("terminal_status_counts") != {"completed": len(parent.failed_ids)}:
        raise ValueError("Semantic supplement terminal counts are stale")
    relationships = Counter(str(row["relationship"]) for row in adjudications)
    if manifest.get("relationship_counts") != dict(sorted(relationships.items())):
        raise ValueError("Semantic supplement relationship_counts are stale")
    boundary = manifest.get("evidence_boundary")
    if not isinstance(boundary, dict) or not all(
        (
            boundary.get("behavior_blind") is True,
            boundary.get("target_behavior_consumed") is False,
            boundary.get("hf_model_execution_count") == 0,
            boundary.get("hf_tokenizer_execution_count") == 0,
            boundary.get("validation_behavior_exposure_count") == 0,
            boundary.get("sealed_behavior_exposure_count") == 0,
            boundary.get("human_gold") is False,
        )
    ):
        raise ValueError("Semantic supplement evidence boundary is invalid")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or not all(
        (
            safety.get("failed_ids_derived_only_from_parent_checkpoint") is True,
            safety.get("strict_json_schema_required") is True,
            safety.get("json_mode_fallback_allowed") is False,
            safety.get("infrastructure_distinct") is False,
            safety.get("infrastructure_independence_claimed") is False,
            safety.get("target_behavior_consumed") is False,
            safety.get("hf_model_initialized") is False,
            safety.get("hf_tokenizer_initialized") is False,
            safety.get("validation_exposed") is False,
            safety.get("sealed_exposed") is False,
            safety.get("human_gold") is False,
        )
    ):
        raise ValueError("Semantic supplement safety contract is invalid")
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(path, schema_version=RUN_MANIFEST_SCHEMA),
        "run_contract_sha256": manifest["run_contract_sha256"],
        "checkpoint_path": checkpoint_path,
        "checkpoints": checkpoint_rows,
        "checkpoint_binding": _binding(
            checkpoint_path,
            schema_version=CHECKPOINT_SCHEMA,
            record_count=len(checkpoint_rows),
        ),
        "adjudications_path": adjudications_path,
        "adjudications": adjudications,
        "adjudications_binding": _binding(
            adjudications_path,
            schema_version=parent_runner.ADJUDICATION_SCHEMA,
            record_count=len(adjudications),
        ),
        "completed_ids": list(parent.failed_ids),
        "generator_spec": generator_spec,
        "reviewer_spec": reviewer_spec,
        "route_identity": copy.deepcopy(contract["route_identity"]),
        "parent": parent,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parent-run-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=parent_runner.DEFAULT_CONFIG)
    result.add_argument("--env-file", type=Path, default=parent_runner.DEFAULT_ENV_FILE)
    result.add_argument("--generator-provider-profile")
    result.add_argument("--generator-model")
    result.add_argument("--generator-expected-response-model")
    result.add_argument("--generator-reasoning-effort")
    result.add_argument("--generator-max-output-tokens", type=int)
    result.add_argument("--reviewer-provider-profile")
    result.add_argument("--reviewer-model")
    result.add_argument("--reviewer-expected-response-model")
    result.add_argument("--reviewer-reasoning-effort")
    result.add_argument("--reviewer-max-output-tokens", type=int)
    result.add_argument("--max-retries", type=int)
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--no-resume", action="store_true")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    config = parent_runner.load_config(args.config)
    generator_spec, reviewer_spec = resolve_model_specs(
        config,
        generator_provider_profile=args.generator_provider_profile,
        generator_model=args.generator_model,
        generator_expected_response_model=args.generator_expected_response_model,
        generator_reasoning_effort=args.generator_reasoning_effort,
        generator_max_output_tokens=args.generator_max_output_tokens,
        reviewer_provider_profile=args.reviewer_provider_profile,
        reviewer_model=args.reviewer_model,
        reviewer_expected_response_model=args.reviewer_expected_response_model,
        reviewer_reasoning_effort=args.reviewer_reasoning_effort,
        reviewer_max_output_tokens=args.reviewer_max_output_tokens,
        max_retries=args.max_retries,
    )
    result = run_recovery(
        parent_run_manifest_path=args.parent_run_manifest,
        output_dir=args.output_dir,
        config=config,
        env_path=args.env_file,
        generator_spec=generator_spec,
        reviewer_spec=reviewer_spec,
        resume=not args.no_resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
