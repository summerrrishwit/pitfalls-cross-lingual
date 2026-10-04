#!/usr/bin/env python3
"""Run resumable, behavior-blind semantic-closure review with GPT-5.5.

The runner consumes a SHA-bound semantic-closure candidate manifest, sends
only a whitelist projection of each lexical/alias pair to an independent
review model, and writes one terminal checkpoint row per pair.  It supports
real multi-item requests, concurrent batches, recursive binary fallback for a
failed batch, deterministic prefix pilots, and resume/retry of failed rows.

This is proxy review only.  It never reads target-model behavior, initializes
an HF model, recomputes or freezes a split, or promotes any decision to human
gold.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls.route_identity import (  # noqa: E402
    attach_route_identity_guard,
    build_route_identity_set,
    validate_route_identity_set,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "full_public_benchmark_8969_pre_exact_hf_zh_v1.json"
)
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_REVIEWER_ROUTE = "gpt-5.5"
DEFAULT_EXPECTED_RESPONSE_MODEL = "gpt-5.5-2026-04-24"
PROTOCOL_REVIEWER_PROVIDER_PROFILE = "openai"

TOOL_VERSION = "full-public-benchmark-semantic-review-runner-v2"
RUN_MANIFEST_SCHEMA = "public-benchmark-full-semantic-review-run-manifest-v1"
CHECKPOINT_SCHEMA = "public-benchmark-full-semantic-review-checkpoint-v1"
ADJUDICATION_SCHEMA = "public-benchmark-full-semantic-closure-adjudication-v2"
BATCH_RESPONSE_SCHEMA = "public-benchmark-full-semantic-review-batch-response-v1"
PROJECTION_SCHEMA = "public-benchmark-full-semantic-review-projection-v1"
PROMPT_VERSION = "public-benchmark-full-semantic-closure-review-v1"
REVIEW_METHOD = "independent_behavior_blind_gpt_5_5_semantic_closure_review_v1"

RELATIONSHIPS = frozenset({"same_fact", "same_leakage_component", "distinct"})
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
FIELD_NAMES = (
    "alias_valid",
    "same_answer_entity",
    "same_subject_entity",
    "relation_semantics_same",
    "temporal_scope_compatible",
    "answer_compatible",
    "semantic_duplicate",
)
FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "split_assignment",
        "split_status",
        "split_policy_version",
        "leakage_component_id",
        "cross_split",
        "same_leakage_component",
        "hf_model_execution_status",
        "hf_tokenizer_execution_status",
        "experiment_status",
        "behavior_output",
    }
)
MODEL_SPEC_FIELDS = (
    "provider_profile",
    "model",
    "temperature",
    "max_output_tokens",
    "reasoning_effort",
    "disable_thinking",
    "json_mode",
    "timeout_seconds",
    "max_retries",
)

REVIEW_INSTRUCTIONS = """You are an independent reviewer of possible factual-semantic leakage.
Use only the supplied English pair text and lexical evidence. No target-model behavior, split,
component, HF, perturbation, or evaluation result is available or allowed. Decide same_fact only
when both endpoints express the same proposition under the same entities, relation, polarity,
time, and answer scope. Decide same_leakage_component when they are not the same fact but are
semantically linked closely enough that putting them in different evaluation splits could leak the
answer or proposition. Otherwise decide distinct. A shared answer string alone is not sufficient.
Treat a canonical-v2 endpoint as comparison evidence, not automatic truth for the provisional row.
Return JSON only and cover every supplied pair exactly once."""


# Loaded lazily so input validation and unit tests do not require provider SDKs.
ModelRouter: Optional[Any] = None


def _load_materializer() -> Any:
    path = Path(__file__).resolve().with_name(
        "materialize_full_public_benchmark_semantic_closure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_full_semantic_closure_materializer_contract", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load semantic-closure materializer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


materializer = _load_materializer()


def _load_model_router_class() -> Any:
    global ModelRouter
    if ModelRouter is None:
        from factual_pitfalls.perturbation import ModelRouter as runtime_router

        ModelRouter = runtime_router
    return ModelRouter


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints() -> Dict[str, str]:
    """Bind resume to the runner, router, and redacted route-binding code."""

    return {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_router_runtime_sha256": sha256_file(
            PROJECT_ROOT / "factual_pitfalls" / "perturbation.py"
        ),
        "route_identity_helper_sha256": sha256_file(
            PROJECT_ROOT / "factual_pitfalls" / "route_identity.py"
        ),
    }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
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
            yield value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode(
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


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path, *, _seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Config extends cycle: {path}")
    seen.add(path)
    config = read_json(path)
    parent = config.get("extends")
    if not parent:
        return config
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        project_relative = (PROJECT_ROOT / parent_path).resolve()
        parent_path = (
            project_relative
            if project_relative.is_file()
            else (path.parent / parent_path).resolve()
        )
    return _deep_merge(load_config(parent_path, _seen=seen), config)


def _safe_model_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    output = {
        field: copy.deepcopy(spec[field])
        for field in MODEL_SPEC_FIELDS
        if field in spec and spec[field] is not None
    }
    _required_string(output.get("provider_profile"), "reviewer provider_profile")
    model = _required_string(output.get("model"), "reviewer model")
    if model != DEFAULT_REVIEWER_ROUTE:
        raise ValueError("semantic closure reviewer must use gpt-5.5")
    max_tokens = output.get("max_output_tokens", 8192)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("reviewer max_output_tokens must be positive")
    output["max_output_tokens"] = max_tokens
    retries = output.get("max_retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("reviewer max_retries must be non-negative")
    output["max_retries"] = retries
    output["json_mode"] = True
    return output


def protocol_reviewer_identity(config: Mapping[str, Any]) -> Dict[str, str]:
    """Resolve and validate the reviewer identity declared by this protocol."""

    roles = config.get("model_roles")
    roles = roles if isinstance(roles, dict) else {}
    role = roles.get("full_pool_semantic_closure_review")
    role = role if isinstance(role, dict) else {}
    reviewer = role.get("reviewer")
    reviewer = reviewer if isinstance(reviewer, dict) else {}
    identity = {
        "provider_profile": _required_string(
            reviewer.get("provider_profile", PROTOCOL_REVIEWER_PROVIDER_PROFILE),
            "protocol reviewer provider_profile",
        ),
        "requested_model": _required_string(
            reviewer.get("model", DEFAULT_REVIEWER_ROUTE),
            "protocol reviewer model",
        ),
        "expected_response_model": _required_string(
            reviewer.get("expected_response_model", DEFAULT_EXPECTED_RESPONSE_MODEL),
            "protocol expected_response_model",
        ),
    }
    expected = {
        "provider_profile": PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": DEFAULT_REVIEWER_ROUTE,
        "expected_response_model": DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if identity != expected:
        raise ValueError("semantic closure protocol reviewer identity is unsupported")
    return identity


def resolve_expected_response_model(
    config: Mapping[str, Any], override: Optional[str]
) -> str:
    declared = protocol_reviewer_identity(config)["expected_response_model"]
    if override is None:
        return declared
    supplied = _required_string(override, "expected_response_model")
    if supplied != declared:
        raise ValueError("expected_response_model differs from the protocol identity")
    return declared


def resolve_reviewer_spec(
    config: Mapping[str, Any],
    *,
    provider_profile: Optional[str] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    protocol_identity = protocol_reviewer_identity(config)
    roles = config.get("model_roles")
    roles = roles if isinstance(roles, dict) else {}
    role = roles.get("full_pool_semantic_closure_review")
    role = role if isinstance(role, dict) else {}
    spec = copy.deepcopy(dict(role.get("reviewer") or {}))
    spec.setdefault("provider_profile", "openai")
    spec.setdefault("model", DEFAULT_REVIEWER_ROUTE)
    if provider_profile:
        spec["provider_profile"] = provider_profile
    if model:
        spec["model"] = model
    if reasoning_effort:
        spec["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        spec["max_output_tokens"] = max_output_tokens
    if max_retries is not None:
        spec["max_retries"] = max_retries
    spec.setdefault("reasoning_effort", "low")
    spec.setdefault("max_output_tokens", 8192)
    execution = config.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    spec.setdefault("max_retries", int(execution.get("max_retries", 0)))
    spec.setdefault("temperature", 0.0)
    safe = _safe_model_spec(spec)
    if safe["provider_profile"] != protocol_identity["provider_profile"]:
        raise ValueError("reviewer provider_profile differs from the protocol identity")
    if safe["model"] != protocol_identity["requested_model"]:
        raise ValueError("reviewer model differs from the protocol identity")
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("config.provider_profiles must be an object")
    if safe["provider_profile"] not in profiles:
        raise ValueError(
            f"Unknown reviewer provider_profile: {safe['provider_profile']}"
        )
    return safe


def _resolve_artifact_path(
    *,
    manifest_path: Path,
    binding: Mapping[str, Any],
    explicit_path: Optional[Path],
    label: str,
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    bound = Path(_required_string(raw, f"{label} path"))
    if not bound.is_absolute():
        bound = manifest_path.parent / bound
    bound = bound.resolve()
    if explicit_path is not None and Path(explicit_path).resolve() != bound:
        raise ValueError(f"Explicit {label} path differs from candidate-manifest binding")
    return bound


def _verify_file_binding(
    *,
    manifest_path: Path,
    binding: Any,
    explicit_path: Optional[Path],
    label: str,
    expected_schema: Optional[str] = None,
    jsonl: bool = False,
    allow_empty: bool = False,
) -> Tuple[Path, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is invalid")
    path = _resolve_artifact_path(
        manifest_path=manifest_path,
        binding=binding,
        explicit_path=explicit_path,
        label=label,
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 mismatch")
    if binding.get("byte_count") is not None and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count mismatch")
    value = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if jsonl and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record_count mismatch")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version mismatch")
    return path, value


class ReviewUnit(NamedTuple):
    input_index: int
    pair_id: str
    candidate_row_sha256: str
    template_row_sha256: str
    left_endpoint_id: str
    left_input_record_sha256: str
    right_endpoint_id: str
    right_input_record_sha256: str
    template: Dict[str, Any]
    projection: Dict[str, Any]


def _endpoint_projection(endpoint: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = endpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Candidate endpoint provenance is invalid")
    return {
        "input_role": provenance.get("input_role"),
        "endpoint_id": provenance.get("endpoint_id"),
        "base_fact_id": provenance.get("base_fact_id"),
        "candidate_id": provenance.get("candidate_id"),
        "source_id": provenance.get("source_id"),
        "source_dataset": provenance.get("source_dataset"),
        "question_en": endpoint.get("question_raw"),
        "canonical_fact_en": endpoint.get("canonical_fact_raw"),
        "answer_en": endpoint.get("answer_raw"),
        "answer_aliases_en": copy.deepcopy(endpoint.get("answer_aliases_raw") or []),
    }


def project_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Whitelist the only candidate fields allowed into the reviewer prompt."""

    review = candidate.get("review_contract")
    if not isinstance(review, dict):
        raise ValueError("Candidate review_contract is invalid")
    projection = {
        "schema_version": PROJECTION_SCHEMA,
        "pair_id": candidate.get("pair_id"),
        "match_types": copy.deepcopy(candidate.get("match_types")),
        "match_evidence": copy.deepcopy(candidate.get("match_evidence")),
        "required_adjudications": copy.deepcopy(
            review.get("required_adjudications")
        ),
        "left": _endpoint_projection(candidate["left"]),
        "right": _endpoint_projection(candidate["right"]),
    }
    serialized = canonical_json_bytes(projection).decode("utf-8")
    for forbidden in FORBIDDEN_PROMPT_KEYS:
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"Behavior-blind projection leaked forbidden field: {forbidden}")
    return projection


def _validate_endpoint(
    *,
    endpoint: Any,
    side: str,
    pair_id: str,
    current_full_path: Path,
    comparison_path: Path,
    input_cache: Dict[str, Tuple[str, List[Dict[str, Any]]]],
) -> None:
    if not isinstance(endpoint, dict) or not isinstance(endpoint.get("provenance"), dict):
        raise ValueError(f"Candidate {pair_id} has invalid {side} endpoint")
    provenance = endpoint["provenance"]
    role = provenance.get("input_role")
    expected_path = (
        current_full_path
        if role == "current_full_base"
        else comparison_path
        if role == "comparison_canonical"
        else None
    )
    if expected_path is None:
        raise ValueError(f"Candidate {pair_id} has unsupported endpoint role")
    input_path = Path(
        _required_string(provenance.get("input_path"), f"{pair_id}.{side}.input_path")
    ).resolve()
    if input_path != expected_path:
        raise ValueError(f"Candidate {pair_id} {side} endpoint path mismatch")
    cache_key = str(input_path)
    if cache_key not in input_cache:
        input_cache[cache_key] = (sha256_file(input_path), read_jsonl(input_path))
    file_sha256, rows = input_cache[cache_key]
    if provenance.get("input_sha256") != file_sha256:
        raise ValueError(f"Candidate {pair_id} {side} endpoint file hash is stale")
    line_number = provenance.get("input_line_number")
    if isinstance(line_number, bool) or not isinstance(line_number, int):
        raise ValueError(f"Candidate {pair_id} {side} input_line_number is invalid")
    if line_number < 1 or line_number > len(rows):
        raise ValueError(f"Candidate {pair_id} {side} input_line_number is out of range")
    input_row = rows[line_number - 1]
    row_sha256 = sha256_value(input_row)
    if provenance.get("input_record_sha256") != row_sha256:
        raise ValueError(f"Candidate {pair_id} {side} endpoint row hash is stale")
    for field in ("base_fact_id", "candidate_id", "source_id"):
        if provenance.get(field) != input_row.get(field):
            raise ValueError(f"Candidate {pair_id} {side} endpoint {field} mismatch")
    expected_endpoint_id = "closure_endpoint_" + sha256_value(
        (
            {
                "input_role": "current_full_base",
                "base_fact_id": input_row.get("base_fact_id"),
                "input_record_sha256": row_sha256,
            }
            if role == "current_full_base"
            else {
                "input_role": "comparison_canonical",
                "candidate_id": input_row.get("candidate_id"),
                "source_id": input_row.get("source_id"),
                "input_record_sha256": row_sha256,
            }
        )
    )[:24]
    if provenance.get("endpoint_id") != expected_endpoint_id:
        raise ValueError(f"Candidate {pair_id} {side} endpoint_id is stale")


def load_review_units(
    *,
    candidate_manifest_path: Path,
    candidate_pairs_path: Optional[Path] = None,
    adjudication_template_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], Path, Path, List[ReviewUnit]]:
    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    manifest = read_json(candidate_manifest_path)
    if manifest.get("schema_version") != materializer.MANIFEST_SCHEMA:
        raise ValueError("Semantic-closure candidate manifest schema is unsupported")
    if manifest.get("status") != "bounded_candidates_materialized_semantic_decisions_pending":
        raise ValueError("Semantic-closure candidate manifest is not review-pending")
    limitations = manifest.get("limitations")
    if not isinstance(limitations, dict):
        raise ValueError("Semantic-closure candidate limitations are missing")
    if limitations.get("candidate_generation_is_bounded") is not True:
        raise ValueError("Semantic-closure candidate generation is not declared bounded")
    if limitations.get("semantic_decisions_pending") is not True:
        raise ValueError("Semantic decisions are not pending")
    if limitations.get("semantic_near_duplicate_recall_guaranteed") is not False:
        raise ValueError("Semantic recall must not be claimed")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict):
        raise ValueError("Semantic-closure candidate safety contract is missing")
    for field in (
        "network_or_model_used",
        "semantic_closure_complete",
        "review_freeze_emitted",
        "split_freeze_emitted",
        "hf_checkpoint_bound",
        "hf_tokenizer_bound",
        "hf_behavior_executed",
        "validation_exposed",
        "sealed_exposed",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"Semantic-closure candidate safety flag is invalid: {field}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Semantic-closure candidate artifacts are missing")
    candidates_path, candidates = _verify_file_binding(
        manifest_path=candidate_manifest_path,
        binding=artifacts.get("candidate_pairs"),
        explicit_path=candidate_pairs_path,
        label="candidate_pairs",
        expected_schema=materializer.CANDIDATE_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    templates_path, templates = _verify_file_binding(
        manifest_path=candidate_manifest_path,
        binding=artifacts.get("adjudication_template"),
        explicit_path=adjudication_template_path,
        label="adjudication_template",
        expected_schema=materializer.ADJUDICATION_TEMPLATE_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    if len(candidates) != len(templates):
        raise ValueError("Candidate/template record counts differ")
    summary = manifest.get("candidate_summary")
    if not isinstance(summary, dict) or summary.get("candidate_pair_count") != len(candidates):
        raise ValueError("Candidate manifest candidate_pair_count is stale")
    pair_ids = [str(row.get("pair_id") or "") for row in candidates]
    if any(not value for value in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Candidate pair IDs are missing or duplicated")
    if summary.get("ordered_pair_ids_sha256") != sha256_value(pair_ids):
        raise ValueError("Candidate manifest ordered pair-ID digest is stale")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Semantic-closure candidate inputs are missing")
    current_binding = inputs.get("current_full_base_facts")
    comparison_binding = inputs.get("comparison_canonical")
    current_path, _ = _verify_file_binding(
        manifest_path=candidate_manifest_path,
        binding=current_binding,
        explicit_path=None,
        label="current_full_base_facts",
        expected_schema=materializer.FULL_BASE_SCHEMA,
        jsonl=True,
    )
    comparison_path, _ = _verify_file_binding(
        manifest_path=candidate_manifest_path,
        binding=comparison_binding,
        explicit_path=None,
        label="comparison_canonical",
        jsonl=True,
    )
    for label in (
        "formal_universe_manifest",
        "formal_universe_items",
        "current_split_manifest",
        "current_leakage_components",
    ):
        _verify_file_binding(
            manifest_path=candidate_manifest_path,
            binding=inputs.get(label),
            explicit_path=None,
            label=label,
            jsonl=label in {"formal_universe_items", "current_leakage_components"},
        )
    original_sources = inputs.get("original_source_artifacts")
    if not isinstance(original_sources, dict):
        raise ValueError("Candidate manifest original_source_artifacts are missing")
    for label, binding in original_sources.items():
        _verify_file_binding(
            manifest_path=candidate_manifest_path,
            binding=binding,
            explicit_path=None,
            label=f"original_source_artifacts.{label}",
            jsonl=True,
        )

    input_cache: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    units: List[ReviewUnit] = []
    for input_index, (candidate, template) in enumerate(zip(candidates, templates)):
        pair_id = pair_ids[input_index]
        if candidate.get("schema_version") != materializer.CANDIDATE_SCHEMA:
            raise ValueError(f"Candidate schema is unsupported: {pair_id}")
        expected_pair_id = "semantic_pair_" + sha256_value(
            {
                "left_endpoint_id": candidate["left"]["provenance"]["endpoint_id"],
                "right_endpoint_id": candidate["right"]["provenance"]["endpoint_id"],
                "match_types": candidate.get("match_types"),
                "match_evidence": candidate.get("match_evidence"),
            }
        )[:24]
        if pair_id != expected_pair_id:
            raise ValueError(f"Candidate pair_id is stale: {pair_id}")
        _validate_endpoint(
            endpoint=candidate.get("left"),
            side="left",
            pair_id=pair_id,
            current_full_path=current_path,
            comparison_path=comparison_path,
            input_cache=input_cache,
        )
        _validate_endpoint(
            endpoint=candidate.get("right"),
            side="right",
            pair_id=pair_id,
            current_full_path=current_path,
            comparison_path=comparison_path,
            input_cache=input_cache,
        )
        expected_template = materializer._adjudication_template(candidate)
        if template != expected_template:
            raise ValueError(f"Adjudication template is stale: {pair_id}")
        if template.get("schema_version") != materializer.ADJUDICATION_TEMPLATE_SCHEMA:
            raise ValueError(f"Adjudication template schema is unsupported: {pair_id}")
        candidate_sha = sha256_value(candidate)
        if template.get("candidate_row_sha256") != candidate_sha:
            raise ValueError(f"Adjudication template candidate hash is stale: {pair_id}")
        left_provenance = candidate["left"]["provenance"]
        right_provenance = candidate["right"]["provenance"]
        units.append(
            ReviewUnit(
                input_index=input_index,
                pair_id=pair_id,
                candidate_row_sha256=candidate_sha,
                template_row_sha256=sha256_value(template),
                left_endpoint_id=str(left_provenance["endpoint_id"]),
                left_input_record_sha256=str(left_provenance["input_record_sha256"]),
                right_endpoint_id=str(right_provenance["endpoint_id"]),
                right_input_record_sha256=str(right_provenance["input_record_sha256"]),
                template=dict(template),
                projection=project_candidate(candidate),
            )
        )
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        units = units[:limit]
    return manifest, candidates_path, templates_path, units


def _response_contract() -> Dict[str, Any]:
    return {
        "top_level": {
            "schema_version": BATCH_RESPONSE_SCHEMA,
            "records": "exactly one result for every input pair_id",
        },
        "record": {
            "pair_id": "copy the exact input pair_id",
            # The concrete tokens are defined in the prose instructions.  Do not
            # repeat them inside INPUT_JSON: one token is also the name of an
            # internal leakage-component field and the behavior-blind payload
            # must never contain that field name as a quoted JSON value/key.
            "relationship": "one relationship token defined in the instructions",
            "alias_valid": "boolean; null only when answer_alias_exact is absent",
            "same_answer_entity": "boolean",
            "same_subject_entity": "boolean",
            "relation_semantics_same": "boolean",
            "temporal_scope_compatible": "boolean",
            "answer_compatible": "boolean",
            "semantic_duplicate": "boolean",
            "rationale": "non-empty evidence-grounded explanation",
            "confidence": sorted(CONFIDENCE_LEVELS),
        },
    }


def build_review_prompt(units: Sequence[ReviewUnit]) -> str:
    payload = {
        "response_contract": _response_contract(),
        "pairs": [unit.projection for unit in units],
    }
    return f"{REVIEW_INSTRUCTIONS}\nINPUT_JSON:\n{canonical_json_bytes(payload).decode('utf-8')}"


def _prompt_payload(prompt: str) -> Dict[str, Any]:
    marker = "INPUT_JSON:\n"
    if marker not in prompt:
        raise ValueError("Prompt has no INPUT_JSON marker")
    value = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(value, dict):
        raise ValueError("Prompt payload is not an object")
    return value


def validate_batch_response(value: Dict[str, Any], units: Sequence[ReviewUnit]) -> None:
    if set(value) != {"schema_version", "records"}:
        raise ValueError("Batch response must contain exactly schema_version and records")
    if value.get("schema_version") != BATCH_RESPONSE_SCHEMA:
        raise ValueError("Batch response schema_version is unsupported")
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError("Batch response records must be a list")
    expected = {unit.pair_id: unit for unit in units}
    by_id: Dict[str, Mapping[str, Any]] = {}
    required_fields = {"pair_id", "relationship", *FIELD_NAMES, "rationale", "confidence"}
    for record in records:
        if not isinstance(record, dict) or set(record) != required_fields:
            raise ValueError("Batch response record fields are invalid")
        pair_id = _required_string(record.get("pair_id"), "response pair_id")
        if pair_id in by_id:
            raise ValueError(f"Duplicate response pair_id: {pair_id}")
        unit = expected.get(pair_id)
        if unit is None:
            raise ValueError(f"Unknown response pair_id: {pair_id}")
        relationship = record.get("relationship")
        if relationship not in RELATIONSHIPS:
            raise ValueError(f"Invalid relationship: {pair_id}")
        alias_required = "answer_alias_exact" in unit.projection["match_types"]
        alias_valid = record.get("alias_valid")
        if alias_required:
            if not isinstance(alias_valid, bool):
                raise ValueError(f"alias_valid must be boolean: {pair_id}")
        elif alias_valid is not None:
            raise ValueError(f"alias_valid must be null without alias evidence: {pair_id}")
        for field in FIELD_NAMES[1:]:
            if not isinstance(record.get(field), bool):
                raise ValueError(f"{field} must be boolean: {pair_id}")
        if alias_valid is True and record.get("same_answer_entity") is not True:
            raise ValueError(f"alias_valid requires same_answer_entity: {pair_id}")
        semantic_duplicate = record["semantic_duplicate"]
        if (relationship == "same_fact") != semantic_duplicate:
            raise ValueError(f"relationship conflicts with semantic_duplicate: {pair_id}")
        if relationship == "same_fact" and not all(
            record[field]
            for field in (
                "same_answer_entity",
                "same_subject_entity",
                "relation_semantics_same",
                "temporal_scope_compatible",
                "answer_compatible",
            )
        ):
            raise ValueError(f"same_fact field adjudications are inconsistent: {pair_id}")
        _required_string(record.get("rationale"), f"{pair_id}.rationale")
        if record.get("confidence") not in CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence: {pair_id}")
        by_id[pair_id] = record
    if set(by_id) != set(expected):
        raise ValueError("Batch response does not exactly cover requested pair IDs")


def _safe_attempts(value: Any) -> List[Dict[str, Any]]:
    allowed = {
        "attempt",
        "status",
        "latency_ms",
        "error_type",
        "http_status",
        "provider_type",
        "provider_code",
        "provider_param",
    }
    if not isinstance(value, list):
        return []
    return [
        {key: entry[key] for key in sorted(set(entry).intersection(allowed))}
        for entry in value
        if isinstance(entry, dict)
    ]


def _exception_record(error: Exception) -> Dict[str, Any]:
    result: Dict[str, Any] = {"error_type": type(error).__name__}
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    return result


def _response_map(value: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["pair_id"]): dict(row) for row in value["records"]}


def _call_model(
    *,
    router: Any,
    units: Sequence[ReviewUnit],
    reviewer_spec: Mapping[str, Any],
    expected_response_model: str,
    prompt: str,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    pair_ids = [unit.pair_id for unit in units]
    batch_id = "semantic_batch_" + sha256_value(pair_ids)[:20]
    safe_spec = _safe_model_spec(reviewer_spec)
    request_binding = {
        "stage": "full_semantic_closure_review",
        "batch_id": batch_id,
        "ordered_pair_ids": pair_ids,
        "model_spec": safe_spec,
        "expected_response_model": expected_response_model,
        "prompt": prompt,
    }
    try:
        result = router.request_json(
            "full_semantic_closure_review",
            batch_id,
            {
                **dict(reviewer_spec),
                "expected_response_model": expected_response_model,
            },
            prompt,
            lambda value: validate_batch_response(value, units),
        )
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
        post_validation_error_type = None
        if terminal_status == "completed" and identity_status != "matched":
            terminal_status = "failed"
            post_validation_error_type = (
                "ResponseModelMissing"
                if identity_status == "not_observed"
                else "ResponseModelMismatch"
            )
        record = {
            "stage": "full_semantic_closure_review",
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch_id,
            "batch_item_count": len(units),
            "ordered_pair_ids_sha256": sha256_value(pair_ids),
            "provider_profile": safe_spec["provider_profile"],
            "requested_model": safe_spec["model"],
            "expected_response_model": expected_response_model,
            "response_model": result.response_model,
            "response_model_identity_status": identity_status,
            "terminal_status": terminal_status,
            "prompt_sha256": sha256_text(prompt),
            "request_sha256": sha256_value(request_binding),
            "response_sha256": sha256_text(raw_response) if raw_response is not None else None,
            "raw_response_sha256": sha256_text(raw_response) if raw_response is not None else None,
            "usage": {
                key: number
                for key, number in dict(result.usage or {}).items()
                if key in {"input_tokens", "output_tokens", "total_tokens"}
                and isinstance(number, int)
            },
            "usage_scope": "shared_batch_not_per_item",
            "latency_ms": result.latency_ms,
            "attempt_count": result.attempt_count,
            "attempts": _safe_attempts(result.attempts),
            "post_validation_error_type": post_validation_error_type,
            "credentials_or_endpoints_included": False,
        }
        if terminal_status != "completed" or not isinstance(result.parsed_response, dict):
            return None, record
        try:
            validate_batch_response(result.parsed_response, units)
        except Exception as error:
            record["terminal_status"] = "failed"
            record["post_validation_error_type"] = type(error).__name__
            return None, record
        return _response_map(result.parsed_response), record
    except Exception as error:
        return None, {
            "stage": "full_semantic_closure_review",
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch_id,
            "batch_item_count": len(units),
            "ordered_pair_ids_sha256": sha256_value(pair_ids),
            "provider_profile": safe_spec["provider_profile"],
            "requested_model": safe_spec["model"],
            "expected_response_model": expected_response_model,
            "response_model": None,
            "response_model_identity_status": "not_observed",
            "terminal_status": "failed",
            "prompt_sha256": sha256_text(prompt),
            "request_sha256": sha256_value(request_binding),
            "response_sha256": None,
            "raw_response_sha256": None,
            "usage": {},
            "usage_scope": "shared_batch_not_per_item",
            "latency_ms": None,
            "attempt_count": 0,
            "attempts": [],
            "post_validation_error_type": None,
            **_exception_record(error),
            "credentials_or_endpoints_included": False,
        }


def _should_split(call: Mapping[str, Any], unit_count: int) -> bool:
    if unit_count <= 1:
        return False
    error_types = {
        str(entry.get("error_type"))
        for entry in call.get("attempts", [])
        if isinstance(entry, dict) and entry.get("error_type")
    }
    for field in ("error_type", "post_validation_error_type"):
        if call.get(field):
            error_types.add(str(call[field]))
    non_splittable = {
        "AuthenticationError",
        "PermissionDeniedError",
        "StaticProxyIdentityError",
        "ResponseModelMissing",
        "ResponseModelMismatch",
    }
    return not bool(error_types.intersection(non_splittable))


def _strict_adjudication(
    *,
    unit: ReviewUnit,
    verdict: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    expected_response_model: str,
    response_model: str,
    reviewed_at: str,
) -> Dict[str, Any]:
    return {
        "schema_version": ADJUDICATION_SCHEMA,
        "pair_id": unit.pair_id,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "adjudication_template_row_sha256": unit.template_row_sha256,
        "left_endpoint_id": unit.left_endpoint_id,
        "left_input_record_sha256": unit.left_input_record_sha256,
        "right_endpoint_id": unit.right_endpoint_id,
        "right_input_record_sha256": unit.right_input_record_sha256,
        **{field: verdict[field] for field in FIELD_NAMES},
        "relationship": verdict["relationship"],
        "rationale": verdict["rationale"],
        "confidence": verdict["confidence"],
        "reviewer_type": "independent_model_proxy",
        "reviewer_id": (
            f"{reviewer_spec['provider_profile']}:"
            f"{response_model}"
        ),
        "requested_model": reviewer_spec["model"],
        "expected_response_model": expected_response_model,
        "response_model": response_model,
        "response_model_identity_status": "matched",
        "review_method": REVIEW_METHOD,
        "reviewed_at": reviewed_at,
        "behavior_blind": True,
        "human_gold": False,
    }


def _failed_rows(
    *,
    units: Sequence[ReviewUnit],
    traces: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
    now_fn: Callable[[], str],
) -> List[Dict[str, Any]]:
    return [
        {
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
            "run_contract_sha256": run_contract_sha256,
            "terminal_status": "failed",
            "failure_stage": "reviewer",
            "model_verdict": None,
            "strict_adjudication": None,
            "model_calls": [dict(trace) for trace in traces],
            "retry_history": [],
            "completed_at": now_fn(),
            "behavior_blind": True,
            "human_gold": False,
            "hf_model_execution_count": 0,
            "behavior_execution_count": 0,
        }
        for unit in units
    ]


def _process_batch(
    *,
    units: Sequence[ReviewUnit],
    router: Any,
    reviewer_spec: Mapping[str, Any],
    expected_response_model: str,
    run_contract_sha256: str,
    now_fn: Callable[[], str],
    inherited_traces: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    prompt = build_review_prompt(units)
    verdicts, call = _call_model(
        router=router,
        units=units,
        reviewer_spec=reviewer_spec,
        expected_response_model=expected_response_model,
        prompt=prompt,
    )
    traces = [*inherited_traces, call]
    if verdicts is None:
        if _should_split(call, len(units)):
            midpoint = len(units) // 2
            return [
                *_process_batch(
                    units=units[:midpoint],
                    router=router,
                    reviewer_spec=reviewer_spec,
                    expected_response_model=expected_response_model,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
                *_process_batch(
                    units=units[midpoint:],
                    router=router,
                    reviewer_spec=reviewer_spec,
                    expected_response_model=expected_response_model,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
            ]
        return _failed_rows(
            units=units,
            traces=traces,
            run_contract_sha256=run_contract_sha256,
            now_fn=now_fn,
        )

    response_model = _required_string(call.get("response_model"), "response_model")
    output: List[Dict[str, Any]] = []
    for unit in units:
        reviewed_at = now_fn()
        verdict = verdicts[unit.pair_id]
        adjudication = _strict_adjudication(
            unit=unit,
            verdict=verdict,
            reviewer_spec=reviewer_spec,
            expected_response_model=expected_response_model,
            response_model=response_model,
            reviewed_at=reviewed_at,
        )
        output.append(
            {
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
                "run_contract_sha256": run_contract_sha256,
                "terminal_status": "completed",
                "failure_stage": None,
                "model_verdict": dict(verdict),
                "strict_adjudication": adjudication,
                "model_calls": [dict(trace) for trace in traces],
                "retry_history": [],
                "completed_at": reviewed_at,
                "behavior_blind": True,
                "human_gold": False,
                "hf_model_execution_count": 0,
                "behavior_execution_count": 0,
            }
        )
    return output


def _chunks(values: Sequence[ReviewUnit], size: int) -> Iterator[List[ReviewUnit]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _checkpoint_index(
    *,
    path: Path,
    units_by_id: Mapping[str, ReviewUnit],
    run_contract_sha256: str,
    protocol_identity: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = read_jsonl(path, allow_empty=True)
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pair_id = _required_string(row.get("pair_id"), "checkpoint.pair_id")
        if pair_id in output:
            raise ValueError(f"Duplicate checkpoint pair_id: {pair_id}")
        unit = units_by_id.get(pair_id)
        if unit is None:
            raise ValueError(f"Checkpoint pair is outside selected input prefix: {pair_id}")
        if row.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError(f"Checkpoint schema is unsupported: {pair_id}")
        if row.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(f"Checkpoint run contract is stale: {pair_id}")
        expected_bindings = {
            "candidate_row_sha256": unit.candidate_row_sha256,
            "adjudication_template_row_sha256": unit.template_row_sha256,
            "left_endpoint_id": unit.left_endpoint_id,
            "left_input_record_sha256": unit.left_input_record_sha256,
            "right_endpoint_id": unit.right_endpoint_id,
            "right_input_record_sha256": unit.right_input_record_sha256,
        }
        for field, expected in expected_bindings.items():
            if row.get(field) != expected:
                raise ValueError(f"Checkpoint {field} is stale: {pair_id}")
        if row.get("terminal_status") not in {"completed", "failed"}:
            raise ValueError(f"Checkpoint terminal_status is invalid: {pair_id}")
        if row.get("behavior_blind") is not True or row.get("human_gold") is not False:
            raise ValueError(f"Checkpoint safety contract is invalid: {pair_id}")
        if row.get("hf_model_execution_count") != 0:
            raise ValueError(f"Checkpoint unexpectedly contains HF execution: {pair_id}")
        if row.get("behavior_execution_count") != 0:
            raise ValueError(f"Checkpoint unexpectedly contains behavior execution: {pair_id}")
        if row["terminal_status"] == "completed":
            decision = row.get("strict_adjudication")
            if not isinstance(decision, dict):
                raise ValueError(f"Checkpoint lacks strict adjudication: {pair_id}")
            if decision.get("candidate_row_sha256") != unit.candidate_row_sha256:
                raise ValueError(f"Checkpoint adjudication candidate hash is stale: {pair_id}")
            expected_identity = {
                "requested_model": protocol_identity["requested_model"],
                "expected_response_model": protocol_identity["expected_response_model"],
                "response_model": protocol_identity["expected_response_model"],
                "response_model_identity_status": "matched",
            }
            for field, expected in expected_identity.items():
                if decision.get(field) != expected:
                    raise ValueError(
                        f"Checkpoint adjudication model identity is invalid: {pair_id}.{field}"
                    )
            calls = row.get("model_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"Checkpoint model identity trace is missing: {pair_id}")
            final_call = calls[-1]
            if not isinstance(final_call, dict) or any(
                final_call.get(field) != expected
                for field, expected in {
                    "provider_profile": protocol_identity["provider_profile"],
                    "requested_model": protocol_identity["requested_model"],
                    "expected_response_model": protocol_identity[
                        "expected_response_model"
                    ],
                    "response_model": protocol_identity["expected_response_model"],
                    "response_model_identity_status": "matched",
                }.items()
            ):
                raise ValueError(f"Checkpoint response model is not verified: {pair_id}")
        output[pair_id] = dict(row)
    return output


def _ordered_checkpoint_rows(
    checkpoint: Mapping[str, Mapping[str, Any]], units: Sequence[ReviewUnit]
) -> List[Dict[str, Any]]:
    return [dict(checkpoint[unit.pair_id]) for unit in units if unit.pair_id in checkpoint]


def _write_runtime_artifacts(
    *,
    checkpoint_path: Path,
    decisions_path: Path,
    checkpoint: Mapping[str, Mapping[str, Any]],
    units: Sequence[ReviewUnit],
) -> None:
    rows = _ordered_checkpoint_rows(checkpoint, units)
    write_jsonl(checkpoint_path, rows)
    decisions = [
        row["strict_adjudication"]
        for row in rows
        if row.get("terminal_status") == "completed"
    ]
    write_jsonl(decisions_path, decisions)


def run_review(
    *,
    candidate_manifest_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    env_path: Path,
    reviewer_spec: Mapping[str, Any],
    expected_response_model: Optional[str],
    candidate_pairs_path: Optional[Path] = None,
    adjudication_template_path: Optional[Path] = None,
    limit: Optional[int] = None,
    batch_size: int = 8,
    max_workers: Optional[int] = None,
    resume: bool = False,
    retry_failed: bool = False,
    router: Optional[Any] = None,
    now_fn: Callable[[], str] = utc_now,
) -> Dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    execution = config.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    worker_count = int(max_workers if max_workers is not None else execution.get("max_workers", 1))
    if worker_count < 1:
        raise ValueError("max_workers must be positive")
    reviewer_spec = _safe_model_spec(reviewer_spec)
    protocol_identity = protocol_reviewer_identity(config)
    if reviewer_spec["provider_profile"] != protocol_identity["provider_profile"]:
        raise ValueError("reviewer provider_profile differs from the protocol identity")
    if reviewer_spec["model"] != protocol_identity["requested_model"]:
        raise ValueError("reviewer model differs from the protocol identity")
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict) or reviewer_spec["provider_profile"] not in profiles:
        raise ValueError("Reviewer provider_profile is absent from config")
    if expected_response_model is None:
        raise ValueError("expected_response_model is required")
    expected_response_model = _required_string(
        expected_response_model, "expected_response_model"
    )
    if expected_response_model != protocol_identity["expected_response_model"]:
        raise ValueError("expected_response_model differs from the protocol identity")

    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    candidate_manifest, candidates_path, templates_path, units = load_review_units(
        candidate_manifest_path=candidate_manifest_path,
        candidate_pairs_path=candidate_pairs_path,
        adjudication_template_path=adjudication_template_path,
        limit=limit,
    )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "semantic_review_checkpoint.jsonl"
    decisions_path = output_dir / "semantic_closure_adjudications.jsonl"
    run_manifest_path = output_dir / "semantic_review_run_manifest.json"
    event_log_path = output_dir / "redacted_model_events.jsonl"
    if not resume and any(
        path.exists() for path in (checkpoint_path, decisions_path, run_manifest_path)
    ):
        raise FileExistsError("Review outputs exist; pass --resume or choose a new output")

    if router is None:
        router = _load_model_router_class()(
            dict(config), Path(env_path).resolve(), event_log_path
        )
    route_identity = build_route_identity_set(
        config=config,
        env_path=Path(env_path).resolve(),
        routes=[(reviewer_spec, expected_response_model)],
        resolved_env=(
            router.values if isinstance(getattr(router, "values", None), Mapping) else None
        ),
    )
    attach_route_identity_guard(
        router,
        config=config,
        env_path=Path(env_path).resolve(),
        identity_set=route_identity,
    )

    selected_ids = [unit.pair_id for unit in units]
    run_contract = {
        "tool_version": TOOL_VERSION,
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "candidate_pairs": {
            "path": str(candidates_path),
            "sha256": sha256_file(candidates_path),
        },
        "adjudication_template": {
            "path": str(templates_path),
            "sha256": sha256_file(templates_path),
        },
        "formal_universe_id": candidate_manifest.get("universe_id"),
        "selected_pair_ids_sha256": sha256_value(selected_ids),
        "selected_pair_count": len(units),
        "full_candidate_pair_count": candidate_manifest["candidate_summary"][
            "candidate_pair_count"
        ],
        "selection_is_full_candidate_set": len(units)
        == candidate_manifest["candidate_summary"]["candidate_pair_count"],
        "config_sha256": sha256_value(config),
        "source_fingerprints": source_fingerprints(),
        "route_identity": route_identity,
        "reviewer_spec": reviewer_spec,
        "expected_response_model": expected_response_model,
        "protocol_reviewer_identity": protocol_identity,
        "prompt_contract_sha256": sha256_value(
            {
                "prompt_version": PROMPT_VERSION,
                "instructions": REVIEW_INSTRUCTIONS,
                "projection_schema": PROJECTION_SCHEMA,
                "batch_response_schema": BATCH_RESPONSE_SCHEMA,
                "field_names": FIELD_NAMES,
            }
        ),
        "batch_size": batch_size,
        "max_workers": worker_count,
        "behavior_blind": True,
        "human_gold": False,
    }
    run_contract_sha256 = sha256_value(run_contract)
    if run_manifest_path.is_file():
        previous_manifest = read_json(run_manifest_path)
        if not resume:
            raise FileExistsError(f"Run manifest exists: {run_manifest_path}")
        if previous_manifest.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(
                "Resume run contract mismatch; input, selection, batch, route, or config changed"
            )

    running_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "status": "running",
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "invocation_resumed": bool(resume),
        "started_or_resumed_at": now_fn(),
        "evidence_boundary": {
            "reviewer_type": "independent_model_proxy",
            "human_gold": False,
            "behavior_blind": True,
            "target_behavior_consumed": False,
            "hf_model_execution_count": 0,
            "hf_tokenizer_execution_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
            "semantic_closure_complete": False,
            "review_freeze_performed": False,
            "split_freeze_performed": False,
        },
    }
    write_json(run_manifest_path, running_manifest)

    units_by_id = {unit.pair_id: unit for unit in units}
    checkpoint = (
        _checkpoint_index(
            path=checkpoint_path,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha256,
            protocol_identity=protocol_identity,
        )
        if resume
        else {}
    )
    pending = [
        unit
        for unit in units
        if unit.pair_id not in checkpoint
        or (
            retry_failed
            and checkpoint[unit.pair_id].get("terminal_status") == "failed"
        )
    ]
    if pending and hasattr(router, "_event"):
        event_lock = threading.Lock()
        original_event = router._event

        def locked_event(*args: Any, **kwargs: Any) -> Any:
            with event_lock:
                return original_event(*args, **kwargs)

        router._event = locked_event

    wave_size = batch_size * worker_count
    for wave_start in range(0, len(pending), wave_size):
        wave = pending[wave_start : wave_start + wave_size]
        batches = list(_chunks(wave, batch_size))

        def commit(rows: Sequence[Mapping[str, Any]]) -> None:
            for raw_row in rows:
                row = dict(raw_row)
                pair_id = str(row["pair_id"])
                previous = checkpoint.get(pair_id)
                if previous is not None and previous.get("terminal_status") == "failed":
                    history = list(previous.get("retry_history", []))
                    history.append(
                        {
                            "terminal_status": "failed",
                            "failure_stage": previous.get("failure_stage"),
                            "model_calls": copy.deepcopy(previous.get("model_calls", [])),
                            "completed_at": previous.get("completed_at"),
                        }
                    )
                    row["retry_history"] = history
                checkpoint[pair_id] = row
                # This coordinator is the only checkpoint/decision writer.
                _write_runtime_artifacts(
                    checkpoint_path=checkpoint_path,
                    decisions_path=decisions_path,
                    checkpoint=checkpoint,
                    units=units,
                )

        if worker_count == 1:
            for batch in batches:
                commit(
                    _process_batch(
                        units=batch,
                        router=router,
                        reviewer_spec=reviewer_spec,
                        expected_response_model=expected_response_model,
                        run_contract_sha256=run_contract_sha256,
                        now_fn=now_fn,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(batches))) as executor:
                futures = [
                    executor.submit(
                        _process_batch,
                        units=batch,
                        router=router,
                        reviewer_spec=reviewer_spec,
                        expected_response_model=expected_response_model,
                        run_contract_sha256=run_contract_sha256,
                        now_fn=now_fn,
                    )
                    for batch in batches
                ]
                for future in as_completed(futures):
                    commit(future.result())

    if not checkpoint_path.exists():
        _write_runtime_artifacts(
            checkpoint_path=checkpoint_path,
            decisions_path=decisions_path,
            checkpoint=checkpoint,
            units=units,
        )
    ordered_rows = _ordered_checkpoint_rows(checkpoint, units)
    status_counts = Counter(str(row["terminal_status"]) for row in ordered_rows)
    decisions = [
        row["strict_adjudication"]
        for row in ordered_rows
        if row.get("terminal_status") == "completed"
    ]
    relationship_counts = Counter(str(row["relationship"]) for row in decisions)
    all_selected_completed = status_counts.get("completed", 0) == len(units)
    selection_is_full = run_contract["selection_is_full_candidate_set"]
    final_manifest = {
        **running_manifest,
        "status": "completed" if all_selected_completed else "completed_with_failures",
        "completed_at": now_fn(),
        "selected_pair_count": len(units),
        "full_candidate_pair_count": run_contract["full_candidate_pair_count"],
        "selection_is_full_candidate_set": selection_is_full,
        "batch_size": batch_size,
        "max_workers": worker_count,
        "resume_enabled": resume,
        "retry_failed_enabled": retry_failed,
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "relationship_counts": dict(sorted(relationship_counts.items())),
        "candidate_adjudication_complete_for_selected_scope": all_selected_completed,
        "candidate_adjudication_complete_for_full_set": (
            all_selected_completed and selection_is_full
        ),
        "semantic_closure_complete": False,
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "byte_count": checkpoint_path.stat().st_size,
                "record_count": len(ordered_rows),
                "schema_version": CHECKPOINT_SCHEMA,
            },
            "adjudications": {
                "path": str(decisions_path),
                "sha256": sha256_file(decisions_path),
                "byte_count": decisions_path.stat().st_size,
                "record_count": len(decisions),
                "schema_version": ADJUDICATION_SCHEMA,
            },
        },
        "execution_policy": {
            "one_model_request_per_batch": True,
            "real_batch_size": batch_size,
            "max_workers": worker_count,
            "recursive_binary_split_on_batch_failure": True,
            "atomic_single_writer_checkpoint": True,
            "resume_supported": True,
        },
    }
    write_json(run_manifest_path, final_manifest)
    return {
        "status": final_manifest["status"],
        "manifest_path": str(run_manifest_path),
        "manifest_sha256": sha256_file(run_manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "adjudications_path": str(decisions_path),
        "selected_pair_count": len(units),
        "terminal_status_counts": final_manifest["terminal_status_counts"],
        "relationship_counts": final_manifest["relationship_counts"],
        "semantic_closure_complete": False,
    }


def execute(
    args: argparse.Namespace,
    *,
    router_factory: Optional[Callable[..., Any]] = None,
    now_fn: Callable[[], str] = utc_now,
) -> Dict[str, Any]:
    config = load_config(Path(args.config))
    spec = resolve_reviewer_spec(
        config,
        provider_profile=args.reviewer_provider_profile,
        model=args.reviewer_model,
        reasoning_effort=args.reviewer_reasoning_effort,
        max_output_tokens=args.reviewer_max_output_tokens,
        max_retries=args.max_retries,
    )
    expected_response_model = resolve_expected_response_model(
        config, args.expected_response_model
    )
    router = None
    if router_factory is not None:
        router = router_factory(
            config,
            Path(args.env_file).resolve(),
            Path(args.output_dir).resolve() / "redacted_model_events.jsonl",
        )
    return run_review(
        candidate_manifest_path=Path(args.candidate_manifest),
        candidate_pairs_path=(Path(args.candidates) if args.candidates else None),
        adjudication_template_path=(
            Path(args.adjudication_template) if args.adjudication_template else None
        ),
        output_dir=Path(args.output_dir),
        config=config,
        env_path=Path(args.env_file),
        reviewer_spec=spec,
        expected_response_model=expected_response_model,
        limit=args.limit,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        resume=args.resume,
        retry_failed=args.retry_failed,
        router=router,
        now_fn=now_fn,
    )


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument("--candidates", type=Path)
    result.add_argument("--adjudication-template", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    result.add_argument(
        "--reviewer-route",
        "--reviewer-model",
        dest="reviewer_model",
        default=DEFAULT_REVIEWER_ROUTE,
    )
    result.add_argument("--reviewer-provider-profile", default="openai")
    result.add_argument("--reviewer-reasoning-effort", default="low")
    result.add_argument("--reviewer-max-output-tokens", type=int, default=8192)
    result.add_argument(
        "--expected-response-model", default=DEFAULT_EXPECTED_RESPONSE_MODEL
    )
    result.add_argument("--max-retries", type=int)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--max-workers", type=int)
    result.add_argument("--limit", type=int)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--retry-failed", action="store_true")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
