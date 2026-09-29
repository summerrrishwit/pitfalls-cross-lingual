#!/usr/bin/env python3
"""Recover exactly the failed rows from a bound full fact-review run.

This is a separate, versioned recovery protocol.  It never edits the parent
run, never accepts a caller-supplied ID list, fixes the batch size at one, and
requires strict JSON Schema on both the proposal and independent review calls.
All decisions remain behavior-blind model-proxy evidence with
``human_gold=false``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_VERSION = "full-public-benchmark-fact-review-supplement-runner-v2"
CHECKPOINT_SCHEMA_VERSION = (
    "full-public-benchmark-fact-review-supplement-checkpoint-v1"
)
RUN_MANIFEST_SCHEMA_VERSION = (
    "full-public-benchmark-fact-review-supplement-run-manifest-v1"
)
RESPONSE_SCHEMA_VERSION = (
    "full-public-benchmark-fact-review-supplement-response-v1"
)
GENERATOR_PROMPT_VERSION = (
    "full-public-benchmark-fact-review-supplement-generator-v1"
)
REVIEWER_PROMPT_VERSION = (
    "full-public-benchmark-fact-review-supplement-reviewer-v1"
)
REVIEW_METHOD = "behavior_blind_dual_model_singleton_fact_review_recovery_v2"
EXPECTED_PARENT_SELECTED_COUNT = 8969
EXPECTED_PARENT_COMPLETED_COUNT = 8966
EXPECTED_PARENT_DECISIONS_COUNT = 8966
EXPECTED_FAILED_COUNT = 3
FIXED_BATCH_SIZE = 1
MAX_ALLOWED_RETRIES = 2

OVERALL_DECISIONS = frozenset({"accept", "reject", "defer", "revise"})
TARGET_DECISIONS = frozenset({"accept", "reject"})
VALIDATION_ERROR_CODES = frozenset(
    {
        "response_fields_invalid",
        "response_schema_version_invalid",
        "base_fact_id_invalid",
        "overall_decision_invalid",
        "fact_review_invalid",
        "relation_review_invalid",
        "notes_invalid",
        "member_reviews_invalid",
        "member_review_coverage_invalid",
        "alias_reviews_invalid",
        "alias_review_coverage_invalid",
        "distractor_reviews_invalid",
        "distractor_review_coverage_invalid",
        "accept_fact_inconsistent",
        "accept_relation_inconsistent",
        "accept_member_inconsistent",
        "accept_alias_inconsistent",
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
        "base_fact_id",
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

GENERATOR_INSTRUCTIONS = """You are the evidence-first proposal stage for a three-item recovery protocol.
You receive exactly one item. Review only its supplied source snapshot, extracted triples, aliases,
and distractors. Target-model behavior, split assignment, HF results, and Simulation results are
forbidden. Return only an object matching the supplied strict JSON Schema. Copy every supplied ID
exactly. Target decisions are only accept or reject. If overall_decision is accept, fact_review and
relation_review must both be accept, every member must be accept, and at least one accepted alias
must exactly normalize to the canonical answer. Use defer when source evidence is insufficient,
reject for a clear falsehood, and revise only for a correctable extraction."""

REVIEWER_INSTRUCTIONS = """You are the independent final reviewer for a three-item recovery protocol.
You receive exactly one evidence item plus a non-authoritative generator draft. Re-evaluate the
source evidence yourself and overturn the draft when needed. Target-model behavior, split
assignment, HF results, and Simulation results are forbidden. Return only an object matching the
supplied strict JSON Schema. Copy every supplied ID exactly. Target decisions are only accept or
reject. If overall_decision is accept, fact_review and relation_review must both be accept, every
member must be accept, and at least one accepted alias must exactly normalize to the canonical
answer. Prefer defer over unsupported confidence."""


def _load_parent_runner() -> Any:
    path = Path(__file__).resolve().with_name(
        "run_full_public_benchmark_fact_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "full_fact_review_parent_runner_for_supplement", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load parent fact-review runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parent_runner = _load_parent_runner()
review_tool = parent_runner.review_tool


class SupplementValidationError(ValueError):
    """A finite, non-sensitive response-contract failure."""

    def __init__(self, code: str):
        if code not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Unknown supplement validation error code: {code}")
        self.code = code
        super().__init__(code)


class StrictSchemaRouteUnsupportedError(RuntimeError):
    """The selected provider protocol cannot carry strict JSON Schema."""


class ParentAuthority(NamedTuple):
    manifest: Dict[str, Any]
    units: List[Any]
    failed_units: List[Any]
    failed_checkpoint_rows: Dict[str, Dict[str, Any]]
    inputs: Dict[str, Dict[str, Any]]


def _required_string(value: Any, field: str) -> str:
    return parent_runner._required_string(value, field)


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(path),
        "sha256": parent_runner.sha256_file(path),
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _runtime_source_bindings() -> Dict[str, Dict[str, str]]:
    """Bind every local implementation that defines the recovery contract."""

    sources = {
        "supplement_runner": Path(__file__).resolve(),
        "parent_runner": Path(parent_runner.__file__).resolve(),
        "review_contract": Path(review_tool.__file__).resolve(),
    }
    return {
        name: {
            "path": str(path),
            "sha256": parent_runner.sha256_file(path),
        }
        for name, path in sources.items()
    }


def _verify_bindings_unchanged(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    for label, binding in bindings.items():
        path = Path(_required_string(binding.get("path"), f"{label}.path")).resolve()
        expected_sha = _required_string(binding.get("sha256"), f"{label}.sha256")
        if not path.is_file() or parent_runner.sha256_file(path) != expected_sha:
            raise ValueError(f"Bound input changed during supplement run: {label}")


def _bound_path(binding: Any, label: str) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = Path(_required_string(binding.get("path"), f"{label}.path")).resolve()
    expected_sha = _required_string(binding.get("sha256"), f"{label}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if parent_runner.sha256_file(path) != expected_sha:
        raise ValueError(f"{label} sha256 mismatch")
    return path


def _validate_parent_manifest_shape(
    manifest: Mapping[str, Any],
    *,
    expected_selected_count: int,
    expected_completed_count: int,
    expected_failed_count: int,
) -> None:
    if manifest.get("schema_version") != parent_runner.RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported parent run manifest schema_version")
    if manifest.get("tool_version") != parent_runner.TOOL_VERSION:
        raise ValueError("Unsupported parent fact-review tool_version")
    if manifest.get("status") != "completed_with_failures":
        raise ValueError("Parent run must have status completed_with_failures")
    if manifest.get("human_gold") is not False:
        raise ValueError("Parent run must remain human_gold=false")
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("Parent run_contract must be an object")
    if manifest.get("run_contract_sha256") != parent_runner.sha256_value(
        run_contract
    ):
        raise ValueError("Parent run_contract_sha256 mismatch")
    selected_count = manifest.get("selected_count")
    if selected_count != expected_selected_count:
        raise ValueError(
            f"Parent selected_count must be exactly {expected_selected_count}"
        )
    if manifest.get("full_export_count") != selected_count:
        raise ValueError("Parent run must cover the full review export")
    if run_contract.get("selected_count") != selected_count:
        raise ValueError("Parent run_contract selected_count mismatch")
    if run_contract.get("selection_is_full_export") is not True:
        raise ValueError("Parent run_contract must bind the full export")
    if run_contract.get("human_gold") is not False:
        raise ValueError("Parent run_contract must remain human_gold=false")
    counts = manifest.get("terminal_status_counts")
    expected_counts = {
        "completed": expected_completed_count,
        "failed": expected_failed_count,
    }
    normalized_counts = (
        {key: counts.get(key, 0) for key in expected_counts}
        if isinstance(counts, dict)
        else None
    )
    if (
        normalized_counts != expected_counts
        or not isinstance(counts, dict)
        or not set(counts).issubset(expected_counts)
    ):
        raise ValueError(
            f"Parent terminal_status_counts must equal {expected_counts}"
        )
    if expected_completed_count + expected_failed_count != selected_count:
        raise ValueError("Parent terminal_status_counts do not cover selected_count")


def _inspect_parent_authority(
    parent_run_manifest_path: Path,
    *,
    expected_selected_count: int,
    expected_completed_count: int,
    expected_failed_count: int,
    expected_decisions_count: int,
) -> ParentAuthority:
    parent_run_manifest_path = Path(parent_run_manifest_path).resolve()
    manifest = parent_runner.read_json(parent_run_manifest_path)
    _validate_parent_manifest_shape(
        manifest,
        expected_selected_count=expected_selected_count,
        expected_completed_count=expected_completed_count,
        expected_failed_count=expected_failed_count,
    )

    export_path = _bound_path(manifest.get("export_manifest"), "export_manifest")
    review_items_path = _bound_path(manifest.get("review_items"), "review_items")
    decision_template_path = _bound_path(
        manifest.get("decision_template"), "decision_template"
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Parent artifacts must be an object")
    checkpoint_binding = artifacts.get("checkpoint")
    decisions_binding = artifacts.get("decisions")
    checkpoint_path = _bound_path(checkpoint_binding, "parent checkpoint")
    decisions_path = _bound_path(decisions_binding, "parent decisions")

    export_manifest, loaded_items_path, loaded_templates_path, units = (
        parent_runner.load_review_units(
            export_manifest_path=export_path,
            review_items_path=review_items_path,
            decision_template_path=decision_template_path,
        )
    )
    if loaded_items_path != review_items_path or loaded_templates_path != decision_template_path:
        raise ValueError("Parent review artifacts are not the export-manifest bindings")
    if len(units) != manifest["selected_count"]:
        raise ValueError("Parent review unit count mismatch")

    checkpoint_rows = parent_runner.read_jsonl(checkpoint_path)
    if not isinstance(checkpoint_binding, dict):
        raise ValueError("Parent checkpoint binding must be an object")
    if checkpoint_binding.get("record_count") != len(checkpoint_rows):
        raise ValueError("Parent checkpoint record_count mismatch")
    if checkpoint_binding.get("schema_version") != parent_runner.CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Parent checkpoint schema_version mismatch")
    if len(checkpoint_rows) != len(units):
        raise ValueError("Parent checkpoint does not cover the full export")
    units_by_id = {unit.base_fact_id: unit for unit in units}
    parent_contract_sha = str(manifest["run_contract_sha256"])
    validated_rows: List[Dict[str, Any]] = []
    for expected_unit, row in zip(units, checkpoint_rows):
        base_fact_id, validated = parent_runner._validate_checkpoint_row(
            row,
            units_by_id=units_by_id,
            run_contract_sha256=parent_contract_sha,
        )
        if base_fact_id != expected_unit.base_fact_id:
            raise ValueError("Parent checkpoint ordering differs from the review export")
        if validated.get("tool_version") != parent_runner.TOOL_VERSION:
            raise ValueError(f"Parent checkpoint tool_version mismatch: {base_fact_id}")
        if validated.get("input_index") != expected_unit.input_index:
            raise ValueError(f"Parent checkpoint input_index mismatch: {base_fact_id}")
        validated_rows.append(validated)

    failed_rows = [
        row for row in validated_rows if row.get("terminal_status") == "failed"
    ]
    if len(failed_rows) != expected_failed_count:
        raise ValueError(
            f"Parent checkpoint must contain exactly {expected_failed_count} failed rows"
        )
    failed_ids = [str(row["base_fact_id"]) for row in failed_rows]
    failed_id_set = set(failed_ids)
    failed_units = [unit for unit in units if unit.base_fact_id in failed_id_set]
    if [unit.base_fact_id for unit in failed_units] != failed_ids:
        raise ValueError("Parent failed ID ordering is not deterministic")

    parent_decisions = parent_runner.read_jsonl(decisions_path, allow_empty=True)
    if not isinstance(decisions_binding, dict):
        raise ValueError("Parent decisions binding must be an object")
    if decisions_binding.get("record_count") != len(parent_decisions):
        raise ValueError("Parent decisions record_count mismatch")
    if len(parent_decisions) != expected_decisions_count:
        raise ValueError(
            f"Parent decisions must contain exactly {expected_decisions_count} rows"
        )
    if decisions_binding.get("schema_version") != review_tool.DECISION_SCHEMA_VERSION:
        raise ValueError("Parent decisions schema_version mismatch")
    expected_decisions = [
        row["strict_decision"]
        for row in validated_rows
        if row.get("terminal_status") == "completed"
    ]
    if parent_runner.canonical_json_bytes(parent_decisions) != parent_runner.canonical_json_bytes(
        expected_decisions
    ):
        raise ValueError("Parent decisions differ from completed checkpoint rows")

    export_artifacts = export_manifest.get("artifacts")
    if not isinstance(export_artifacts, dict):
        raise ValueError("Review export artifacts must be an object")
    inputs = {
        "parent_run_manifest": _binding(
            parent_run_manifest_path,
            schema_version=parent_runner.RUN_MANIFEST_SCHEMA_VERSION,
        ),
        "parent_checkpoint": _binding(
            checkpoint_path,
            schema_version=parent_runner.CHECKPOINT_SCHEMA_VERSION,
            record_count=len(checkpoint_rows),
        ),
        "parent_decisions": _binding(
            decisions_path,
            schema_version=review_tool.DECISION_SCHEMA_VERSION,
            record_count=len(parent_decisions),
        ),
        "export_manifest": _binding(
            export_path,
            schema_version=export_manifest.get("schema_version"),
        ),
        "review_items": _binding(
            review_items_path,
            schema_version=review_tool.EXPORT_ITEM_SCHEMA_VERSION,
            record_count=len(units),
        ),
        "decision_template": _binding(
            decision_template_path,
            schema_version=review_tool.DECISION_SCHEMA_VERSION,
            record_count=len(units),
        ),
    }
    return ParentAuthority(
        manifest=dict(manifest),
        units=units,
        failed_units=failed_units,
        failed_checkpoint_rows={str(row["base_fact_id"]): row for row in failed_rows},
        inputs=inputs,
    )


def inspect_parent_authority(parent_run_manifest_path: Path) -> ParentAuthority:
    """Validate the one authoritative 8,969-row parent run and derive 3 failures."""

    return _inspect_parent_authority(
        parent_run_manifest_path,
        expected_selected_count=EXPECTED_PARENT_SELECTED_COUNT,
        expected_completed_count=EXPECTED_PARENT_COMPLETED_COUNT,
        expected_failed_count=EXPECTED_FAILED_COUNT,
        expected_decisions_count=EXPECTED_PARENT_DECISIONS_COUNT,
    )


def _reasoned_decision_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": sorted(OVERALL_DECISIONS)},
            "reason": {"type": "string"},
        },
        "required": ["decision", "reason"],
        "additionalProperties": False,
    }


def _target_array_schema(id_field: str, ids: Sequence[str]) -> Dict[str, Any]:
    id_schema: Dict[str, Any] = {"type": "string"}
    # JSON Schema forbids an empty enum.  A zero-length target array is instead
    # closed by minItems=maxItems=0; non-empty arrays remain enum-bound.
    if ids:
        id_schema["enum"] = list(ids)
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                id_field: id_schema,
                "decision": {"type": "string", "enum": sorted(TARGET_DECISIONS)},
                "reason": {"type": "string"},
            },
            "required": [id_field, "decision", "reason"],
            "additionalProperties": False,
        },
        "minItems": len(ids),
        "maxItems": len(ids),
    }


def strict_response_schema(unit: Any) -> Dict[str, Any]:
    member_ids = [str(row["candidate_id"]) for row in unit.template["member_reviews"]]
    alias_ids = [str(row["alias_id"]) for row in unit.template["alias_reviews"]]
    distractor_ids = [
        str(row["distractor_id"]) for row in unit.template["distractor_reviews"]
    ]
    fields = {
        "schema_version": {"type": "string", "enum": [RESPONSE_SCHEMA_VERSION]},
        "base_fact_id": {"type": "string", "enum": [unit.base_fact_id]},
        "overall_decision": {
            "type": "string",
            "enum": sorted(OVERALL_DECISIONS),
        },
        "fact_review": _reasoned_decision_schema(),
        "relation_review": _reasoned_decision_schema(),
        "member_reviews": _target_array_schema("candidate_id", member_ids),
        "alias_reviews": _target_array_schema("alias_id", alias_ids),
        "distractor_reviews": _target_array_schema(
            "distractor_id", distractor_ids
        ),
        "notes": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def _raise(code: str) -> None:
    raise SupplementValidationError(code)


def _validate_reasoned(value: Any, code: str) -> None:
    if not isinstance(value, dict) or set(value) != {"decision", "reason"}:
        _raise(code)
    if value.get("decision") not in OVERALL_DECISIONS:
        _raise(code)
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        _raise(code)


def _validate_targets(
    values: Any,
    *,
    id_field: str,
    expected_ids: Sequence[str],
    invalid_code: str,
    coverage_code: str,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(values, list):
        _raise(invalid_code)
    output: Dict[str, Dict[str, Any]] = {}
    for entry in values:
        if not isinstance(entry, dict) or set(entry) != {
            id_field,
            "decision",
            "reason",
        }:
            _raise(invalid_code)
        target_id = entry.get(id_field)
        if not isinstance(target_id, str) or not target_id or target_id in output:
            _raise(coverage_code)
        if entry.get("decision") not in TARGET_DECISIONS:
            _raise(invalid_code)
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            _raise(invalid_code)
        output[target_id] = dict(entry)
    if set(output) != set(expected_ids):
        _raise(coverage_code)
    return output


def validate_response(value: Dict[str, Any], unit: Any) -> None:
    fields = {
        "schema_version",
        "base_fact_id",
        "overall_decision",
        "fact_review",
        "relation_review",
        "member_reviews",
        "alias_reviews",
        "distractor_reviews",
        "notes",
    }
    if not isinstance(value, dict) or set(value) != fields:
        _raise("response_fields_invalid")
    if value.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        _raise("response_schema_version_invalid")
    if value.get("base_fact_id") != unit.base_fact_id:
        _raise("base_fact_id_invalid")
    if value.get("overall_decision") not in OVERALL_DECISIONS:
        _raise("overall_decision_invalid")
    _validate_reasoned(value.get("fact_review"), "fact_review_invalid")
    _validate_reasoned(value.get("relation_review"), "relation_review_invalid")
    if not isinstance(value.get("notes"), str) or not value["notes"].strip():
        _raise("notes_invalid")

    member_ids = [str(row["candidate_id"]) for row in unit.template["member_reviews"]]
    alias_ids = [str(row["alias_id"]) for row in unit.template["alias_reviews"]]
    distractor_ids = [
        str(row["distractor_id"]) for row in unit.template["distractor_reviews"]
    ]
    members = _validate_targets(
        value.get("member_reviews"),
        id_field="candidate_id",
        expected_ids=member_ids,
        invalid_code="member_reviews_invalid",
        coverage_code="member_review_coverage_invalid",
    )
    aliases = _validate_targets(
        value.get("alias_reviews"),
        id_field="alias_id",
        expected_ids=alias_ids,
        invalid_code="alias_reviews_invalid",
        coverage_code="alias_review_coverage_invalid",
    )
    _validate_targets(
        value.get("distractor_reviews"),
        id_field="distractor_id",
        expected_ids=distractor_ids,
        invalid_code="distractor_reviews_invalid",
        coverage_code="distractor_review_coverage_invalid",
    )
    if value["overall_decision"] == "accept":
        if value["fact_review"]["decision"] != "accept":
            _raise("accept_fact_inconsistent")
        if value["relation_review"]["decision"] != "accept":
            _raise("accept_relation_inconsistent")
        if any(row["decision"] != "accept" for row in members.values()):
            _raise("accept_member_inconsistent")
        answer = unit.projection["members"][0]["extracted_triple"].get("answer")
        alias_text_by_id = {
            str(row["alias_id"]): row.get("text_en")
            for row in unit.projection["alias_targets"]
        }
        accepted_aliases = {
            review_tool.normalize_text(alias_text_by_id.get(alias_id))
            for alias_id, row in aliases.items()
            if row["decision"] == "accept"
        }
        if review_tool.normalize_text(answer) not in accepted_aliases:
            _raise("accept_alias_inconsistent")


def _prompt_payload(prompt: str) -> Dict[str, Any]:
    marker = "INPUT_JSON:\n"
    if marker not in prompt:
        raise ValueError("Prompt lacks INPUT_JSON marker")
    value = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(value, dict):
        raise ValueError("Prompt INPUT_JSON must be an object")
    return value


def build_generator_prompt(unit: Any, schema: Mapping[str, Any]) -> str:
    payload = {
        "strict_response_json_schema": schema,
        "evidence": unit.projection,
    }
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
    """Create a ModelRouter subclass with no non-schema fallback."""

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
                raise StrictSchemaRouteUnsupportedError(
                    "strict_json_schema_missing"
                )
            if profile.get("protocol") not in {"openai", "openai_compatible"}:
                raise StrictSchemaRouteUnsupportedError(
                    "strict_json_schema_requires_openai_compatible_protocol"
                )
            model = str(model_spec["model"])
            timeout = float(
                model_spec.get(
                    "timeout_seconds",
                    profile.get("timeout_seconds", self.timeout),
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
                "response_format": {
                    "type": "json_schema",
                    "json_schema": strict_schema,
                },
            }
            if model.startswith("gpt-"):
                request["max_completion_tokens"] = model_spec.get(
                    "max_output_tokens", 768
                )
                if model_spec.get("reasoning_effort"):
                    request["reasoning_effort"] = model_spec["reasoning_effort"]
            else:
                request["max_tokens"] = model_spec.get("max_output_tokens", 512)
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
    """Construct the only router allowed by the authoritative run path.

    There is intentionally no router-injection argument on ``run_recovery``.
    Unit tests replace this private factory in-process; the CLI cannot select a
    fake or a weaker transport implementation.
    """

    router_class = _strict_schema_router_class()
    return router_class(dict(config), Path(env_path).resolve(), event_log_path)


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
                f"{role} route does not support required strict JSON Schema: "
                f"{profile_name}"
            )


def resolve_model_specs(
    config: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    *,
    generator_provider_profile: Optional[str] = None,
    generator_model: Optional[str] = None,
    reviewer_provider_profile: Optional[str] = None,
    reviewer_model: Optional[str] = None,
    generator_expected_response_model: Optional[str] = None,
    reviewer_expected_response_model: Optional[str] = None,
    generator_reasoning_effort: Optional[str] = None,
    reviewer_reasoning_effort: Optional[str] = None,
    generator_max_output_tokens: Optional[int] = None,
    reviewer_max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parent_roles = parent_manifest.get("model_roles")
    if not isinstance(parent_roles, dict):
        raise ValueError("Parent manifest model_roles must be an object")
    generator = dict(parent_roles.get("generator") or {})
    reviewer = dict(parent_roles.get("reviewer") or {})
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
    for spec, profile, model, expected, effort, max_tokens in overrides:
        if profile:
            spec["provider_profile"] = profile
        if model:
            spec["model"] = model
        if expected:
            spec["expected_response_model"] = expected
        if effort:
            spec["reasoning_effort"] = effort
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_output_tokens must be positive")
            spec["max_output_tokens"] = max_tokens
        # This protocol uses JSON Schema structured output, not JSON-object
        # mode.  Remove the inherited parent flag so the run contract cannot
        # imply that a weaker fallback is allowed.
        spec.pop("json_mode", None)
        # The shared router's require_response_model_identity flag invokes a
        # static-proxy compatibility policy that is unrelated to this review
        # protocol and can reject an otherwise exact model match.  Identity is
        # checked exactly by this runner after every response instead.
        spec.pop("require_response_model_identity", None)
        retries = spec.get("max_retries", 0) if max_retries is None else max_retries
        if not isinstance(retries, int) or not 0 <= retries <= MAX_ALLOWED_RETRIES:
            raise ValueError(
                f"max_retries must be between 0 and {MAX_ALLOWED_RETRIES}"
            )
        spec["max_retries"] = retries
    generator_safe = parent_runner._safe_model_spec(generator)
    reviewer_safe = parent_runner._safe_model_spec(reviewer)
    if (
        generator_safe["provider_profile"],
        generator_safe["model"],
    ) == (
        reviewer_safe["provider_profile"],
        reviewer_safe["model"],
    ):
        raise ValueError("generator and reviewer must use distinct model identities")
    if (
        generator_safe["expected_response_model"]
        == reviewer_safe["expected_response_model"]
    ):
        raise ValueError(
            "generator and reviewer must use distinct expected response models"
        )
    _validate_strict_schema_routes(config, (generator_safe, reviewer_safe))
    return generator_safe, reviewer_safe


def _failure_code(result: Any, validator_code: Optional[str]) -> str:
    attempts = result.attempts if isinstance(result.attempts, list) else []
    last_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    if (
        validator_code is not None
        and last_attempt.get("error_type") == "SupplementValidationError"
    ):
        return validator_code
    error_types = {
        str(row.get("error_type"))
        for row in attempts
        if isinstance(row, dict) and row.get("error_type")
    }
    http_statuses = {
        row.get("http_status") for row in attempts if isinstance(row, dict)
    }
    if error_types.intersection({"JSONDecodeError", "JSONDecodeFailure"}):
        return "json_parse_failed"
    if "StrictSchemaRouteUnsupportedError" in error_types:
        return "strict_schema_route_unsupported"
    if "StaticProxyIdentityError" in error_types:
        return "static_proxy_identity_failure"
    if 400 in http_statuses or error_types.intersection(
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
    safe_spec = parent_runner._safe_model_spec(model_spec)
    expected_response_model = safe_spec["expected_response_model"]
    schema_binding = {
        "name": "fact_review_supplement_response",
        "strict": True,
        "schema": dict(schema),
    }
    request_spec = dict(model_spec)
    request_spec["strict_json_schema"] = schema_binding
    request_spec.pop("json_mode", None)
    request_spec.pop("require_response_model_identity", None)
    validator_state: Dict[str, Optional[str]] = {"code": None}

    def validator(value: Dict[str, Any]) -> None:
        try:
            validate_response(value, unit)
        except SupplementValidationError as error:
            validator_state["code"] = error.code
            raise

    request_binding = {
        "stage": stage,
        "base_fact_id": unit.base_fact_id,
        "model_spec": safe_spec,
        "prompt": prompt,
        "strict_json_schema_sha256": parent_runner.sha256_value(schema_binding),
    }
    try:
        result = router.request_json(
            stage,
            unit.base_fact_id,
            request_spec,
            prompt,
            validator,
        )
    except Exception as error:
        if isinstance(error, StrictSchemaRouteUnsupportedError):
            code = "strict_schema_route_unsupported"
        elif type(error).__name__ == "StaticProxyIdentityError":
            code = "static_proxy_identity_failure"
        else:
            code = "provider_or_transport_failure"
        return None, {
            "stage": stage,
            "prompt_version": prompt_version,
            "base_fact_id": unit.base_fact_id,
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
            "error_type": type(error).__name__,
            "credentials_or_endpoints_included": False,
        }
    raw_response = result.raw_response if isinstance(result.raw_response, str) else None
    identity_status = (
        "not_observed"
        if result.response_model is None
        else (
            "matched"
            if result.response_model == expected_response_model
            else "mismatch"
        )
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
        "base_fact_id": unit.base_fact_id,
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
    if terminal_status != "completed":
        return None, record
    return dict(result.parsed_response), record


def _strict_decision(
    unit: Any,
    verdict: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    reviewer_response_model: Optional[str],
    reviewed_at: str,
) -> Dict[str, Any]:
    output = copy.deepcopy(unit.template)
    output["decision"] = verdict["overall_decision"]
    output["human_gold"] = False
    output["review_provenance"] = {
        "reviewer_type": "codex_proxy",
        "reviewer_id": (
            f"{reviewer_spec['provider_profile']}:"
            f"{reviewer_response_model or reviewer_spec['model']}"
        ),
        "review_method": REVIEW_METHOD,
        "reviewed_at": reviewed_at,
    }
    target_specs = (
        ("member_reviews", "candidate_id"),
        ("alias_reviews", "alias_id"),
        ("distractor_reviews", "distractor_id"),
    )
    for field, id_field in target_specs:
        decisions = {
            str(row[id_field]): str(row["decision"]) for row in verdict[field]
        }
        output[field] = [
            {**target, "decision": decisions[str(target[id_field])]}
            for target in unit.template[field]
        ]
    output["notes"] = parent_runner.canonical_json_bytes(
        {
            "fact_review": verdict["fact_review"],
            "relation_review": verdict["relation_review"],
            "notes": verdict["notes"],
        }
    ).decode("utf-8")
    if set(output) != review_tool.DECISION_ALLOWED_FIELDS:
        raise ValueError(f"Strict decision fields drifted: {unit.base_fact_id}")
    review_tool._validate_decision(output, unit.review_item)
    return output


def _failed_checkpoint_row(
    *,
    unit: Any,
    parent_row: Mapping[str, Any],
    run_contract_sha256: str,
    failure_stage: str,
    model_calls: Sequence[Mapping[str, Any]],
    now_fn: Callable[[], str],
) -> Dict[str, Any]:
    code = str(model_calls[-1]["validation_error_code"])
    if code not in VALIDATION_ERROR_CODES:
        raise ValueError("Failed call lacks a finite validation_error_code")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "base_fact_id": unit.base_fact_id,
        "input_index": unit.input_index,
        "review_item_sha256": unit.review_item_sha256,
        "decision_template_sha256": unit.decision_template_sha256,
        "parent_failed_checkpoint_record_sha256": parent_runner.sha256_value(
            parent_row
        ),
        "run_contract_sha256": run_contract_sha256,
        "terminal_status": "failed",
        "failure_stage": failure_stage,
        "validation_error_code": code,
        "generator_verdict": None,
        "reviewer_verdict": None,
        "strict_decision": None,
        "model_calls": [dict(row) for row in model_calls],
        "retry_history": [],
        "completed_at": now_fn(),
        "behavior_blind": True,
        "human_gold": False,
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
    generator_prompt = build_generator_prompt(unit, schema)
    draft, generator_call = _call_model(
        router=router,
        stage="fact_review_supplement_generator",
        prompt_version=GENERATOR_PROMPT_VERSION,
        unit=unit,
        model_spec=generator_spec,
        prompt=generator_prompt,
        schema=schema,
    )
    if draft is None:
        return _failed_checkpoint_row(
            unit=unit,
            parent_row=parent_row,
            run_contract_sha256=run_contract_sha256,
            failure_stage="generator",
            model_calls=[generator_call],
            now_fn=now_fn,
        )

    reviewer_prompt = build_reviewer_prompt(unit, draft, schema)
    verdict, reviewer_call = _call_model(
        router=router,
        stage="fact_review_supplement_reviewer",
        prompt_version=REVIEWER_PROMPT_VERSION,
        unit=unit,
        model_spec=reviewer_spec,
        prompt=reviewer_prompt,
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
            now_fn=now_fn,
        )

    reviewed_at = now_fn()
    decision = _strict_decision(
        unit,
        verdict,
        reviewer_spec,
        reviewer_call.get("response_model"),
        reviewed_at,
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "base_fact_id": unit.base_fact_id,
        "input_index": unit.input_index,
        "review_item_sha256": unit.review_item_sha256,
        "decision_template_sha256": unit.decision_template_sha256,
        "parent_failed_checkpoint_record_sha256": parent_runner.sha256_value(
            parent_row
        ),
        "run_contract_sha256": run_contract_sha256,
        "terminal_status": "completed",
        "failure_stage": None,
        "validation_error_code": None,
        "generator_verdict": draft,
        "reviewer_verdict": verdict,
        "strict_decision": decision,
        "model_calls": calls,
        "retry_history": [],
        "completed_at": reviewed_at,
        "behavior_blind": True,
        "human_gold": False,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_model_call_trace(
    call: Any,
    *,
    base_fact_id: str,
    expected_model_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(call, dict):
        raise ValueError(f"Supplement model call is invalid: {base_fact_id}")
    if not set(call).issubset(MODEL_CALL_FIELDS | {"error_type"}):
        raise ValueError(f"Supplement model call contains unsafe fields: {base_fact_id}")
    if not MODEL_CALL_FIELDS.issubset(call):
        raise ValueError(
            f"Supplement model call is missing required fields: {base_fact_id}"
        )
    if call.get("base_fact_id") != base_fact_id:
        raise ValueError(f"Supplement model call ID mismatch: {base_fact_id}")
    if call.get("credentials_or_endpoints_included") is not False:
        raise ValueError(f"Supplement model call safety flag is invalid: {base_fact_id}")
    attempts = call.get("attempts")
    if attempts != parent_runner._safe_attempts(attempts):
        raise ValueError(f"Supplement model attempts contain unsafe fields: {base_fact_id}")
    if not isinstance(call.get("attempt_count"), int) or call["attempt_count"] != len(
        attempts
    ):
        raise ValueError(f"Supplement model attempt_count is invalid: {base_fact_id}")
    stage = call.get("stage")
    expected_spec = expected_model_specs.get(str(stage))
    if expected_spec is None:
        raise ValueError(f"Supplement model call stage is invalid: {base_fact_id}")
    expected_prompt_version = {
        "fact_review_supplement_generator": GENERATOR_PROMPT_VERSION,
        "fact_review_supplement_reviewer": REVIEWER_PROMPT_VERSION,
    }[str(stage)]
    if call.get("prompt_version") != expected_prompt_version:
        raise ValueError(f"Supplement prompt version drifted: {base_fact_id}")
    if (
        call.get("provider_profile") != expected_spec["provider_profile"]
        or call.get("requested_model") != expected_spec["model"]
        or call.get("expected_response_model")
        != expected_spec["expected_response_model"]
    ):
        raise ValueError(f"Supplement model identity contract drifted: {base_fact_id}")
    if call.get("structured_output_mode") != "strict_json_schema_no_fallback":
        raise ValueError(f"Supplement strict schema trace is missing: {base_fact_id}")
    for hash_field in (
        "strict_json_schema_sha256",
        "prompt_sha256",
        "request_sha256",
    ):
        if not _is_sha256(call.get(hash_field)):
            raise ValueError(f"Supplement model call hash is invalid: {base_fact_id}")
    response_sha = call.get("response_sha256")
    raw_response_sha = call.get("raw_response_sha256")
    if response_sha != raw_response_sha or (
        response_sha is not None and not _is_sha256(response_sha)
    ):
        raise ValueError(f"Supplement response hash is invalid: {base_fact_id}")
    usage = call.get("usage")
    if not isinstance(usage, dict) or not set(usage).issubset(
        {"input_tokens", "output_tokens", "total_tokens"}
    ) or any(not isinstance(value, int) or value < 0 for value in usage.values()):
        raise ValueError(f"Supplement usage trace is invalid: {base_fact_id}")
    status = call.get("terminal_status")
    code = call.get("validation_error_code")
    if status == "completed":
        if code is not None:
            raise ValueError(f"Completed supplement call has an error code: {base_fact_id}")
        if (
            call.get("response_model_identity_status") != "matched"
            or not isinstance(call.get("response_model"), str)
            or call.get("response_model") != expected_spec["expected_response_model"]
        ):
            raise ValueError(
                f"Completed supplement call lacks exact response identity: {base_fact_id}"
            )
    elif status == "failed":
        if code not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Failed supplement call lacks a finite code: {base_fact_id}")
        if call.get("response_model_identity_status") not in {
            "matched",
            "mismatch",
            "not_observed",
        }:
            raise ValueError(f"Failed supplement identity trace is invalid: {base_fact_id}")
    else:
        raise ValueError(f"Supplement model call status is invalid: {base_fact_id}")
    if "error_type" in call and not isinstance(call["error_type"], str):
        raise ValueError(f"Supplement model error_type is invalid: {base_fact_id}")


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    units_by_id: Mapping[str, Any],
    parent_rows: Mapping[str, Mapping[str, Any]],
    run_contract_sha256: str,
    expected_model_specs: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    base_fact_id = _required_string(row.get("base_fact_id"), "checkpoint.base_fact_id")
    unit = units_by_id.get(base_fact_id)
    if unit is None:
        raise ValueError(f"Supplement checkpoint contains an unbound ID: {base_fact_id}")
    if row.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported supplement checkpoint schema: {base_fact_id}")
    if row.get("tool_version") != TOOL_VERSION:
        raise ValueError(f"Supplement checkpoint tool_version mismatch: {base_fact_id}")
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError(f"Supplement checkpoint run contract is stale: {base_fact_id}")
    if row.get("input_index") != unit.input_index:
        raise ValueError(f"Supplement checkpoint input_index mismatch: {base_fact_id}")
    if row.get("review_item_sha256") != unit.review_item_sha256:
        raise ValueError(f"Supplement review item is stale: {base_fact_id}")
    if row.get("decision_template_sha256") != unit.decision_template_sha256:
        raise ValueError(f"Supplement decision template is stale: {base_fact_id}")
    if row.get("parent_failed_checkpoint_record_sha256") != parent_runner.sha256_value(
        parent_rows[base_fact_id]
    ):
        raise ValueError(f"Supplement parent failure binding is stale: {base_fact_id}")
    if row.get("human_gold") is not False or row.get("behavior_blind") is not True:
        raise ValueError(f"Supplement checkpoint safety contract is invalid: {base_fact_id}")
    status = row.get("terminal_status")
    if status not in {"completed", "failed"}:
        raise ValueError(f"Supplement terminal_status is invalid: {base_fact_id}")
    calls = row.get("model_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"Supplement model_calls are invalid: {base_fact_id}")
    for call in calls:
        _validate_model_call_trace(
            call,
            base_fact_id=base_fact_id,
            expected_model_specs=expected_model_specs,
        )
    retry_history = row.get("retry_history")
    if not isinstance(retry_history, list):
        raise ValueError(f"Supplement retry_history is invalid: {base_fact_id}")
    for prior in retry_history:
        if not isinstance(prior, dict) or set(prior) != {
            "terminal_status",
            "failure_stage",
            "validation_error_code",
            "model_calls",
            "completed_at",
        }:
            raise ValueError(f"Supplement retry_history fields are invalid: {base_fact_id}")
        if prior.get("terminal_status") != "failed":
            raise ValueError(f"Supplement retry_history status is invalid: {base_fact_id}")
        if prior.get("validation_error_code") not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Supplement retry_history code is invalid: {base_fact_id}")
        prior_calls = prior.get("model_calls")
        if not isinstance(prior_calls, list) or not prior_calls:
            raise ValueError(f"Supplement retry_history calls are invalid: {base_fact_id}")
        for prior_call in prior_calls:
            _validate_model_call_trace(
                prior_call,
                base_fact_id=base_fact_id,
                expected_model_specs=expected_model_specs,
            )
    if status == "completed":
        if row.get("validation_error_code") is not None:
            raise ValueError(f"Completed supplement row has an error code: {base_fact_id}")
        decision = row.get("strict_decision")
        if not isinstance(decision, dict):
            raise ValueError(f"Completed supplement row lacks a decision: {base_fact_id}")
        review_tool._validate_decision(decision, unit.review_item)
        completed_response_models: List[str] = []
        for stage in (
            "fact_review_supplement_generator",
            "fact_review_supplement_reviewer",
        ):
            matching_calls = [
                call
                for call in calls
                if (
                call.get("stage") == stage
                and call.get("terminal_status") == "completed"
                and call.get("response_model_identity_status") == "matched"
                and call.get("validation_error_code") is None
                and call.get("response_model")
                == expected_model_specs[stage]["expected_response_model"]
                )
            ]
            if len(matching_calls) != 1:
                raise ValueError(
                    f"Completed supplement row lacks a matched {stage}: {base_fact_id}"
                )
            completed_response_models.append(str(matching_calls[0]["response_model"]))
        if len(set(completed_response_models)) != 2:
            raise ValueError(
                f"Completed supplement row reuses one response identity: {base_fact_id}"
            )
    else:
        if row.get("strict_decision") is not None:
            raise ValueError(f"Failed supplement row has a decision: {base_fact_id}")
        if row.get("validation_error_code") not in VALIDATION_ERROR_CODES:
            raise ValueError(f"Failed supplement row lacks a finite code: {base_fact_id}")
    return base_fact_id, dict(row)


def _write_runtime_artifacts(
    checkpoint_path: Path,
    decisions_path: Path,
    checkpoint: Mapping[str, Mapping[str, Any]],
    units: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = [
        dict(checkpoint[unit.base_fact_id])
        for unit in units
        if unit.base_fact_id in checkpoint
    ]
    decisions = [
        row["strict_decision"]
        for row in rows
        if row.get("terminal_status") == "completed"
    ]
    parent_runner.write_jsonl(checkpoint_path, rows)
    parent_runner.write_jsonl(decisions_path, decisions)
    return rows, decisions


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
        raise ValueError("Supplement recovery batch_size is fixed at 1")
    authority = inspect_parent_authority(parent_run_manifest_path)
    generator_input = dict(generator_spec)
    reviewer_input = dict(reviewer_spec)
    generator_input.pop("json_mode", None)
    reviewer_input.pop("json_mode", None)
    generator_input.pop("require_response_model_identity", None)
    reviewer_input.pop("require_response_model_identity", None)
    generator_spec = parent_runner._safe_model_spec(generator_input)
    reviewer_spec = parent_runner._safe_model_spec(reviewer_input)
    for role, spec in (("generator", generator_spec), ("reviewer", reviewer_spec)):
        retries = spec.get("max_retries", 0)
        if not isinstance(retries, int) or not 0 <= retries <= MAX_ALLOWED_RETRIES:
            raise ValueError(
                f"{role} max_retries must be between 0 and {MAX_ALLOWED_RETRIES}"
            )
    _validate_strict_schema_routes(config, (generator_spec, reviewer_spec))
    if (
        generator_spec["provider_profile"],
        generator_spec["model"],
    ) == (
        reviewer_spec["provider_profile"],
        reviewer_spec["model"],
    ):
        raise ValueError("generator and reviewer must use distinct model identities")
    if (
        generator_spec["expected_response_model"]
        == reviewer_spec["expected_response_model"]
    ):
        raise ValueError(
            "generator and reviewer must use distinct expected response models"
        )

    failed_ids = [unit.base_fact_id for unit in authority.failed_units]
    prompt_contract = {
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "generator_instructions": GENERATOR_INSTRUCTIONS,
        "reviewer_prompt_version": REVIEWER_PROMPT_VERSION,
        "reviewer_instructions": REVIEWER_INSTRUCTIONS,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "strict_json_schema": True,
        "json_mode_fallback_allowed": False,
        "validation_error_codes": sorted(VALIDATION_ERROR_CODES),
    }
    run_contract = {
        "tool_version": TOOL_VERSION,
        "runtime_sources": _runtime_source_bindings(),
        "parent_run_manifest_sha256": authority.inputs["parent_run_manifest"]["sha256"],
        "parent_run_contract_sha256": authority.manifest["run_contract_sha256"],
        "parent_selected_count": EXPECTED_PARENT_SELECTED_COUNT,
        "parent_completed_count": EXPECTED_PARENT_COMPLETED_COUNT,
        "parent_failed_count": EXPECTED_FAILED_COUNT,
        "parent_decisions_count": EXPECTED_PARENT_DECISIONS_COUNT,
        "parent_checkpoint_sha256": authority.inputs["parent_checkpoint"]["sha256"],
        "parent_decisions_sha256": authority.inputs["parent_decisions"]["sha256"],
        "export_manifest_sha256": authority.inputs["export_manifest"]["sha256"],
        "review_items_sha256": authority.inputs["review_items"]["sha256"],
        "decision_template_sha256": authority.inputs["decision_template"]["sha256"],
        "config_sha256": parent_runner.sha256_value(config),
        "generator_model_spec": generator_spec,
        "reviewer_model_spec": reviewer_spec,
        "prompt_contract_sha256": parent_runner.sha256_value(prompt_contract),
        "review_method": REVIEW_METHOD,
        "response_identity_enforcement": (
            "runner_exact_post_response_without_static_proxy_guard_v2"
        ),
        "failed_base_fact_ids": failed_ids,
        "failed_base_fact_ids_sha256": parent_runner.sha256_value(failed_ids),
        "selected_count": EXPECTED_FAILED_COUNT,
        "batch_size": FIXED_BATCH_SIZE,
        "behavior_blind": True,
        "human_gold": False,
    }
    run_contract_sha = parent_runner.sha256_value(run_contract)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "supplement_fact_review_checkpoint.jsonl"
    decisions_path = output_dir / "supplement_review_decisions_codex_proxy.jsonl"
    manifest_path = output_dir / "supplement_fact_review_run_manifest.json"
    event_log_path = output_dir / "supplement_redacted_model_events.jsonl"
    managed_paths = (checkpoint_path, decisions_path, manifest_path, event_log_path)
    if not resume and any(path.exists() for path in managed_paths):
        raise FileExistsError(
            "supplement run artifacts already exist; use a new output directory or enable resume"
        )
    if resume and manifest_path.is_file():
        old_manifest = parent_runner.read_json(manifest_path)
        if old_manifest.get("run_contract_sha256") != run_contract_sha:
            raise ValueError("resume supplement run contract mismatch")

    units_by_id = {unit.base_fact_id: unit for unit in authority.failed_units}
    expected_model_specs = {
        "fact_review_supplement_generator": generator_spec,
        "fact_review_supplement_reviewer": reviewer_spec,
    }
    checkpoint: Dict[str, Dict[str, Any]] = {}
    if resume and checkpoint_path.is_file():
        for row in parent_runner.read_jsonl(checkpoint_path, allow_empty=True):
            base_fact_id, validated = _validate_checkpoint_row(
                row,
                units_by_id=units_by_id,
                parent_rows=authority.failed_checkpoint_rows,
                run_contract_sha256=run_contract_sha,
                expected_model_specs=expected_model_specs,
            )
            if base_fact_id in checkpoint:
                raise ValueError(f"Duplicate supplement checkpoint ID: {base_fact_id}")
            checkpoint[base_fact_id] = validated
    elif not resume:
        parent_runner.write_jsonl(checkpoint_path, [])
        parent_runner.write_jsonl(decisions_path, [])

    router = _create_runtime_router(config, env_path, event_log_path)

    pending = [
        unit
        for unit in authority.failed_units
        if unit.base_fact_id not in checkpoint
        or (
            retry_failed
            and checkpoint[unit.base_fact_id].get("terminal_status") == "failed"
        )
    ]
    for unit in pending:
        row = _process_unit(
            unit=unit,
            parent_row=authority.failed_checkpoint_rows[unit.base_fact_id],
            router=router,
            generator_spec=generator_spec,
            reviewer_spec=reviewer_spec,
            run_contract_sha256=run_contract_sha,
            now_fn=now_fn,
        )
        previous = checkpoint.get(unit.base_fact_id)
        if previous is not None and previous.get("terminal_status") == "failed":
            history = list(previous.get("retry_history", []))
            history.append(
                {
                    "terminal_status": "failed",
                    "failure_stage": previous.get("failure_stage"),
                    "validation_error_code": previous.get("validation_error_code"),
                    "model_calls": list(previous.get("model_calls", [])),
                    "completed_at": previous.get("completed_at"),
                }
            )
            row["retry_history"] = history
        checkpoint[unit.base_fact_id] = row
        _write_runtime_artifacts(
            checkpoint_path,
            decisions_path,
            checkpoint,
            authority.failed_units,
        )

    ordered_rows, decisions = _write_runtime_artifacts(
        checkpoint_path,
        decisions_path,
        checkpoint,
        authority.failed_units,
    )
    status_counts = Counter(row["terminal_status"] for row in ordered_rows)
    decision_counts = Counter(str(row["decision"]) for row in decisions)
    status = (
        "completed"
        if status_counts.get("completed", 0) == EXPECTED_FAILED_COUNT
        else "completed_with_failures"
    )
    _verify_bindings_unchanged(authority.inputs)
    _verify_bindings_unchanged(run_contract["runtime_sources"])
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": status,
        "scope_id": authority.manifest["scope_id"],
        "inputs": authority.inputs,
        "parent_run_contract_sha256": authority.manifest["run_contract_sha256"],
        "parent_terminal_status_counts": {
            "completed": EXPECTED_PARENT_COMPLETED_COUNT,
            "failed": EXPECTED_FAILED_COUNT,
        },
        "failed_base_fact_ids": failed_ids,
        "failed_base_fact_ids_sha256": parent_runner.sha256_value(failed_ids),
        "selected_count": EXPECTED_FAILED_COUNT,
        "batch_size": FIXED_BATCH_SIZE,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha,
        "prompt_contract": prompt_contract,
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "artifacts": {
            "checkpoint": _binding(
                checkpoint_path,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                record_count=len(ordered_rows),
            ),
            "decisions": _binding(
                decisions_path,
                schema_version=review_tool.DECISION_SCHEMA_VERSION,
                record_count=len(decisions),
            ),
        },
        "model_roles": {
            "generator": generator_spec,
            "reviewer": reviewer_spec,
            "role_identity_distinct": True,
            "expected_response_model_distinct": True,
        },
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "safety_contract": {
            "parent_run_modified": False,
            "failed_ids_derived_only_from_parent_checkpoint": True,
            "exact_failed_count_required": EXPECTED_FAILED_COUNT,
            "batch_size_fixed_at_one": True,
            "strict_json_schema_required": True,
            "json_mode_fallback_allowed": False,
            "static_proxy_identity_guard_used": False,
            "runner_exact_response_identity_required": True,
            "completed_role_response_identities_distinct": True,
            "finite_validation_error_codes": sorted(VALIDATION_ERROR_CODES),
            "behavior_blind_projection_only": True,
            "target_behavior_consumed": False,
            "hf_model_initialized": False,
            "simulation_called": False,
            "canonical_freeze_performed": False,
            "automatic_promotion_performed": False,
            "credentials_or_endpoints_included": False,
        },
    }
    parent_runner.write_json(manifest_path, manifest)
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": parent_runner.sha256_file(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "decisions_path": str(decisions_path),
        "selected_count": EXPECTED_FAILED_COUNT,
        "terminal_status_counts": manifest["terminal_status_counts"],
        "decision_counts": manifest["decision_counts"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parent-run-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "full_public_benchmark_8969_pre_exact_hf_zh_v1.json"
        ),
    )
    result.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
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
    authority = inspect_parent_authority(args.parent_run_manifest)
    generator_spec, reviewer_spec = resolve_model_specs(
        config,
        authority.manifest,
        generator_provider_profile=args.generator_provider_profile,
        generator_model=args.generator_model,
        reviewer_provider_profile=args.reviewer_provider_profile,
        reviewer_model=args.reviewer_model,
        generator_expected_response_model=args.generator_expected_response_model,
        reviewer_expected_response_model=args.reviewer_expected_response_model,
        generator_reasoning_effort=args.generator_reasoning_effort,
        reviewer_reasoning_effort=args.reviewer_reasoning_effort,
        generator_max_output_tokens=args.generator_max_output_tokens,
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
