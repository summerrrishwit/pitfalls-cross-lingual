#!/usr/bin/env python3
"""Review regenerated distractor and Neutral candidates before exact-HF binding.

The runner consumes only SHA-bound candidate artifacts from a post-review rebuild
manifest (or the explicitly supported provisional pre-HF summary schema).  It
sends a behavior-blind whitelist projection to one independent proxy reviewer,
writes one field-level adjudication per candidate, and supports concurrent real
batches, recursive binary fallback, a single-writer durable append journal,
resume, and failed-row retry.

It does not initialize an HF model, run behavior, expose Validation/Sealed, make
human-gold claims, freeze a split, or promote candidates into a formal bundle.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls.route_identity import (  # noqa: E402
    attach_route_identity_guard,
    build_route_identity_set,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "full_public_benchmark_8969_pre_exact_hf_zh_v1.json"
)
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_REVIEWER_MODEL = "gpt-5.5"
DEFAULT_EXPECTED_RESPONSE_MODEL = "gpt-5.5-2026-04-24"
PROTOCOL_REVIEWER_PROVIDER_PROFILE = "openai"

TOOL_VERSION = "full-public-benchmark-candidate-review-runner-v3"
RUN_MANIFEST_SCHEMA = "public-benchmark-full-candidate-review-run-manifest-v1"
CHECKPOINT_SCHEMA = "public-benchmark-full-candidate-review-checkpoint-v3"
ADJUDICATION_SCHEMA = "public-benchmark-full-candidate-adjudication-v3"
BATCH_RESPONSE_SCHEMA = "public-benchmark-full-candidate-review-batch-response-v2"
PROJECTION_SCHEMA = "public-benchmark-full-candidate-review-projection-v2"
PROMPT_VERSION = "public-benchmark-full-candidate-semantic-review-v2"
REVIEW_METHOD = "independent_behavior_blind_candidate_semantic_review_v2"
EVIDENCE_IDENTITY_SCHEMA = "public-benchmark-candidate-review-evidence-identity-v1"
EVIDENCE_REUSE_CONTRACT_SCHEMA = (
    "public-benchmark-candidate-review-evidence-reuse-contract-v1"
)
RESPONSE_VALIDATOR_CONTRACT = "candidate-review-response-validator-v2"
FRESH_EVIDENCE_ORIGIN = "fresh_model_review"
CARRIED_EVIDENCE_ORIGIN = "carried_forward_exact_accept"
EVIDENCE_ORIGINS = frozenset({FRESH_EVIDENCE_ORIGIN, CARRIED_EVIDENCE_ORIGIN})
LEGACY_RESOLUTION_MANIFEST_SCHEMA = (
    "public-benchmark-full-candidate-resolution-manifest-v1"
)
CURRENT_RESOLUTION_MANIFEST_SCHEMA = (
    "public-benchmark-full-candidate-resolution-manifest-v2"
)

POSTREVIEW_MANIFEST_SCHEMA = "public-benchmark-full-postreview-rebuild-manifest-v2"
POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA = (
    "public-benchmark-full-postreview-rebuild-manifest-v3"
)
SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS = frozenset(
    {POSTREVIEW_MANIFEST_SCHEMA, POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA}
)
POSTREVIEW_STATUS = "postreview_rebuilt_bounded_semantic_recall_not_frozen"
PRE_HF_SUMMARY_SCHEMA = "public-benchmark-full-pre-hf-summary-v1"
PRE_HF_STATUS = "completed_provisional_offline_pre_hf"
FULL_FACT_SCHEMA = "public-benchmark-full-provisional-base-fact-v1"
DISTRACTOR_SCHEMA = "public-benchmark-dual-distractor-candidate-v1"
NEUTRAL_SCHEMA = "public-benchmark-neutral-reference-candidate-v1"

CANDIDATE_KINDS = frozenset({"distractor", "neutral"})
OVERALL_DECISIONS = frozenset({"accept", "reject", "defer"})
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
JUDGMENT_FIELD_NAMES = (
    "distractor_factually_false",
    "answer_unique",
    "answer_alias_disjoint",
    "neutral_semantically_neutral",
    "neutral_unrelated",
)
ALIAS_EVIDENCE_FIELD_NAMES = (
    "overlapping_target_aliases",
    "answer_alias_disjoint_unresolved_evidence",
)
# Public aggregate used by the downstream resolver when reconstructing the
# exact response record from a persisted adjudication.
FIELD_NAMES = (*JUDGMENT_FIELD_NAMES, *ALIAS_EVIDENCE_FIELD_NAMES)
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
FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "split_assignment",
        "split_status",
        "split_policy_version",
        "leakage_component_id",
        "same_split",
        "same_leakage_component",
        "hf_model_execution_status",
        "hf_tokenizer_execution_status",
        "experiment_status",
        "behavior_output",
        "validation_exposed",
        "sealed_exposed",
    }
)

REVIEW_INSTRUCTIONS = """You are an independent proxy reviewer of English factual candidates.
Use only the supplied text. No target-model behavior, split assignment, leakage component, HF
checkpoint/tokenizer result, Validation result, or Sealed result is available or allowed.

For a distractor, judge whether it is factually false as an answer to the target question under the
target relation and scope; whether the target question has a unique intended canonical answer; and
whether the distractor is disjoint from every accepted answer alias. answer_alias_disjoint is scoped
exclusively to target_fact.answer_aliases_en. candidate_source_fact is provenance-only: its answer,
text, and any aliases must never be treated as target aliases or as answer_alias_disjoint evidence.
Consider the sibling option when judging answer uniqueness. For a Neutral candidate, judge whether
adding the context leaves the
target proposition semantically neutral (it neither answers, contradicts, redirects, presupposes, nor
leaks the target answer) and whether it is genuinely unrelated to the target entities, proposition,
and answer aliases. Shared words alone do not imply relatedness, but an entity, alias, causal,
taxonomic, temporal, or answer-bearing link does.

Relevant fields are true or false when decidable. Use null only for a relevant field that cannot be
decided from the supplied evidence. For a distractor, overlapping_target_aliases must be an empty
list when answer_alias_disjoint is true, a non-empty list copied exactly from
target_fact.answer_aliases_en when it is false, and an empty list when it is null. When it is null,
answer_alias_disjoint_unresolved_evidence must be a non-empty explanation of what evidence is
missing; otherwise that field must be null. For a Neutral candidate, both alias-evidence fields are
irrelevant and must be null. Irrelevant judgment fields must be null. overall_decision is accept only
when every relevant judgment is true, reject when any relevant judgment is false, and defer only when
no relevant judgment is false and at least one is null. Return JSON only and cover every supplied
candidate ID exactly once."""


# Lazy import keeps offline validation and tests independent of provider SDKs.
ModelRouter: Optional[Any] = None


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
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
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
    _required_string(output.get("model"), "reviewer model")
    max_tokens = output.get("max_output_tokens", 16384)
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
    role = roles.get("full_pool_candidate_semantic_review")
    if not isinstance(role, dict):
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
            reviewer.get("model", DEFAULT_REVIEWER_MODEL),
            "protocol reviewer model",
        ),
        "expected_response_model": _required_string(
            reviewer.get("expected_response_model", DEFAULT_EXPECTED_RESPONSE_MODEL),
            "protocol expected_response_model",
        ),
    }
    expected = {
        "provider_profile": PROTOCOL_REVIEWER_PROVIDER_PROFILE,
        "requested_model": DEFAULT_REVIEWER_MODEL,
        "expected_response_model": DEFAULT_EXPECTED_RESPONSE_MODEL,
    }
    if identity != expected:
        raise ValueError("candidate review protocol reviewer identity is unsupported")
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
    role = roles.get("full_pool_candidate_semantic_review")
    if not isinstance(role, dict):
        # The first version can reuse the already declared independent semantic
        # reviewer without mutating the shared config.
        role = roles.get("full_pool_semantic_closure_review")
    role = role if isinstance(role, dict) else {}
    spec = copy.deepcopy(dict(role.get("reviewer") or {}))
    spec.setdefault("provider_profile", "openai")
    spec.setdefault("model", DEFAULT_REVIEWER_MODEL)
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
    execution = config.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    spec.setdefault("reasoning_effort", "low")
    spec.setdefault("temperature", 0.0)
    spec.setdefault("max_output_tokens", 16384)
    spec.setdefault("max_retries", int(execution.get("max_retries", 0)))
    safe = _safe_model_spec(spec)
    if safe["provider_profile"] != protocol_identity["provider_profile"]:
        raise ValueError("reviewer provider_profile differs from the protocol identity")
    if safe["model"] != protocol_identity["requested_model"]:
        raise ValueError("reviewer model differs from the protocol identity")
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("config.provider_profiles must be an object")
    if safe["provider_profile"] not in profiles:
        raise ValueError("Reviewer provider_profile is absent from config")
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
        raise ValueError(f"Explicit {label} path differs from manifest binding")
    return bound


def _verify_jsonl_binding(
    *,
    manifest_path: Path,
    binding: Any,
    explicit_path: Optional[Path],
    label: str,
    expected_schema: str,
    allow_empty: bool,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
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
    byte_count = binding.get("byte_count")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int):
        raise ValueError(f"{label} byte_count is invalid")
    if byte_count != path.stat().st_size:
        raise ValueError(f"{label} byte_count mismatch")
    if binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version mismatch")
    rows = read_jsonl(path, allow_empty=allow_empty)
    record_count = binding.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise ValueError(f"{label} record_count is invalid")
    if record_count != len(rows):
        raise ValueError(f"{label} record_count mismatch")
    return path, rows, dict(binding)


def _manifest_artifacts(manifest: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    schema = manifest.get("schema_version")
    if schema in SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS:
        if manifest.get("status") != POSTREVIEW_STATUS:
            raise ValueError("Post-review rebuild manifest is not in the expected status")
        safety = manifest.get("safety_contract")
        if not isinstance(safety, dict):
            raise ValueError("Post-review rebuild safety_contract is missing")
        required_false = (
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
            "human_gold",
        )
        for field in required_false:
            if safety.get(field) is not False:
                raise ValueError(f"Post-review safety flag must be false: {field}")
        if safety.get("output_is_provisional") is not True:
            raise ValueError("Post-review output must remain provisional")
        if safety.get("split_recomputed") is not True:
            raise ValueError("Post-review split must have been recomputed")
        fact_contract = manifest.get("fact_review_contract")
        if not isinstance(fact_contract, dict):
            raise ValueError("Post-review fact_review_contract is missing")
        if fact_contract.get("revise_defer_missing_count") != 0:
            raise ValueError("Post-review fact review is not terminal")
        if fact_contract.get("proxy_review_only") is not True:
            raise ValueError("Post-review fact evidence boundary is invalid")
        if fact_contract.get("human_gold") is not False:
            raise ValueError("Post-review fact evidence was promoted to human gold")
        semantic_contract = manifest.get("semantic_review_contract")
        if not isinstance(semantic_contract, dict):
            raise ValueError("Post-review semantic_review_contract is missing")
        if semantic_contract.get("candidate_adjudication_complete_for_bounded_set") is not True:
            raise ValueError("Post-review bounded semantic adjudication is incomplete")
        if semantic_contract.get("reviewer_evidence_is_human_gold") is not False:
            raise ValueError("Post-review semantic evidence was promoted to human gold")
        integrity = manifest.get("integrity")
        if not isinstance(integrity, dict) or integrity.get("all_checks_passed") is not True:
            raise ValueError("Post-review integrity checks are not complete")
        if schema == POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA:
            repair = manifest.get("candidate_repair_contract")
            if not isinstance(repair, dict) or not all(
                (
                    repair.get("predecessor_candidate_reviews_blanket_valid")
                    is False,
                    repair.get(
                        "exact_v3_accepted_projection_evidence_carry_forward_eligible"
                    )
                    is True,
                    repair.get(
                        "candidate_id_or_row_sha_alone_sufficient_for_review_reuse"
                    )
                    is False,
                    repair.get("full_candidate_evidence_coverage_required") is True,
                    repair.get("full_candidate_model_rereview_required") is False,
                    repair.get("changed_or_new_candidate_fresh_model_review_required")
                    is True,
                )
            ):
                raise ValueError(
                    "Post-review successor candidate evidence reuse contract is invalid"
                )
        artifacts = manifest.get("outputs")
        kind = "postreview_rebuild"
    elif schema == PRE_HF_SUMMARY_SCHEMA:
        if manifest.get("status") != PRE_HF_STATUS:
            raise ValueError("Pre-HF compatibility manifest is not completed provisional")
        execution = manifest.get("execution")
        if not isinstance(execution, dict):
            raise ValueError("Pre-HF execution boundary is missing")
        expected_zero = (
            "behavior_output_count",
            "development_behavior_exposure_count",
            "hidden_state_collection_count",
            "intervention_count",
            "validation_behavior_exposure_count",
            "sealed_behavior_exposure_count",
        )
        for field in expected_zero:
            if execution.get(field) != 0:
                raise ValueError(f"Pre-HF execution count must be zero: {field}")
        for field in ("hf_model_execution", "hf_tokenizer_execution"):
            if execution.get(field) is not False:
                raise ValueError(f"Pre-HF execution flag must be false: {field}")
        artifacts = manifest.get("artifacts")
        kind = "pre_hf_schema_compatibility"
    else:
        raise ValueError("Candidate manifest schema is unsupported")
    if not isinstance(artifacts, dict):
        raise ValueError("Candidate manifest artifact bindings are missing")
    return kind, artifacts


def _index_unique(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label}[{row_number}].{field}")
        if key in output:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        output[key] = dict(row)
    return output


def _same_text(left: Any, right: Any) -> bool:
    return " ".join(str(left or "").strip().casefold().split()) == " ".join(
        str(right or "").strip().casefold().split()
    )


class ReviewUnit(NamedTuple):
    input_index: int
    candidate_id: str
    candidate_kind: str
    candidate_row_sha256: str
    target_base_fact_id: str
    target_base_fact_row_sha256: str
    source_base_fact_id: str
    source_base_fact_row_sha256: str
    projection: Dict[str, Any]


def _fact_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "projection_role": "target_fact_and_exclusive_answer_alias_scope",
        "base_fact_id": row.get("base_fact_id"),
        "source_question_en": row.get("source_question_en"),
        "subject_en": row.get("subject_en"),
        "relation_raw": row.get("relation_raw"),
        "canonical_fact_en": row.get("canonical_fact_en"),
        "answer_en": row.get("answer_en"),
        "answer_aliases_en": copy.deepcopy(row.get("answer_aliases_en") or []),
        "answer_type": row.get("answer_type"),
    }


def _candidate_source_fact_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose source provenance without creating a second alias scope."""

    projection = _fact_projection(row)
    projection["projection_role"] = "provenance_only_not_target_alias_evidence"
    projection.pop("answer_aliases_en")
    return projection


def _project_candidate(
    *,
    candidate_kind: str,
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    siblings: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if candidate_kind == "distractor":
        candidate_id = candidate["distractor_id"]
        candidate_evidence = {
            "distractor_id": candidate_id,
            "source_base_fact_id": candidate["source_base_fact_id"],
            "distractor_text_en": candidate["distractor_text_en"],
            "sibling_distractors": [
                {
                    "distractor_id": row["distractor_id"],
                    "distractor_text_en": row["distractor_text_en"],
                }
                for row in siblings
                if row["distractor_id"] != candidate_id
            ],
        }
        required_fields = list(JUDGMENT_FIELD_NAMES[:3])
    elif candidate_kind == "neutral":
        candidate_id = candidate["neutral_candidate_id"]
        candidate_evidence = {
            "neutral_candidate_id": candidate_id,
            "source_base_fact_id": candidate["source_base_fact_id"],
            "neutral_context_candidate_en": candidate[
                "neutral_context_candidate_en"
            ],
            "sibling_neutral_candidates": [
                {
                    "neutral_candidate_id": row["neutral_candidate_id"],
                    "neutral_context_candidate_en": row[
                        "neutral_context_candidate_en"
                    ],
                }
                for row in siblings
                if row["neutral_candidate_id"] != candidate_id
            ],
        }
        required_fields = list(JUDGMENT_FIELD_NAMES[3:])
    else:
        raise ValueError(f"Unsupported candidate kind: {candidate_kind}")
    projection = {
        "schema_version": PROJECTION_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "required_field_judgments": required_fields,
        "target_fact": _fact_projection(target),
        "candidate_evidence": candidate_evidence,
        "candidate_source_fact": _candidate_source_fact_projection(source),
    }
    serialized = canonical_json_bytes(projection).decode("utf-8")
    for forbidden in FORBIDDEN_PROMPT_KEYS:
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"Behavior-blind projection leaked forbidden field: {forbidden}")
    return projection


def _validate_full_facts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = _index_unique(rows, "base_fact_id", "full_base_facts")
    for base_fact_id, row in index.items():
        if row.get("schema_version") != FULL_FACT_SCHEMA:
            raise ValueError(f"Unsupported full-base schema: {base_fact_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"Full-base row must remain human_gold=false: {base_fact_id}")
        if row.get("hf_model_execution_status") != "not_run":
            raise ValueError(f"Full-base row contains HF model execution: {base_fact_id}")
        if row.get("hf_tokenizer_execution_status") != "not_run":
            raise ValueError(f"Full-base row contains HF tokenizer execution: {base_fact_id}")
        _required_string(row.get("canonical_fact_en"), f"{base_fact_id}.canonical_fact_en")
        _required_string(row.get("answer_en"), f"{base_fact_id}.answer_en")
        aliases = row.get("answer_aliases_en")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"Full-base row has no answer aliases: {base_fact_id}")
    return index


def _validate_candidate_rows(
    *,
    distractors: Sequence[Mapping[str, Any]],
    neutrals: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Mapping[str, Any]],
) -> None:
    seen_ids: set[str] = set()
    seen_slots: set[Tuple[str, str, int]] = set()
    for kind, rows, id_field, schema in (
        ("distractor", distractors, "distractor_id", DISTRACTOR_SCHEMA),
        ("neutral", neutrals, "neutral_candidate_id", NEUTRAL_SCHEMA),
    ):
        for row_number, row in enumerate(rows, start=1):
            candidate_id = _required_string(
                row.get(id_field), f"{kind}[{row_number}].{id_field}"
            )
            if candidate_id in seen_ids:
                raise ValueError(f"Duplicate candidate ID: {candidate_id}")
            seen_ids.add(candidate_id)
            if row.get("schema_version") != schema:
                raise ValueError(f"Unsupported {kind} schema: {candidate_id}")
            base_fact_id = _required_string(
                row.get("base_fact_id"), f"{candidate_id}.base_fact_id"
            )
            source_id = _required_string(
                row.get("source_base_fact_id"), f"{candidate_id}.source_base_fact_id"
            )
            if base_fact_id not in facts or source_id not in facts:
                raise ValueError(f"Candidate references an unknown fact: {candidate_id}")
            if base_fact_id == source_id:
                raise ValueError(f"Candidate references its target as source: {candidate_id}")
            slot = row.get("slot")
            if isinstance(slot, bool) or slot not in {1, 2}:
                raise ValueError(f"Candidate slot is invalid: {candidate_id}")
            slot_key = (kind, base_fact_id, int(slot))
            if slot_key in seen_slots:
                raise ValueError(f"Duplicate candidate slot: {kind}:{base_fact_id}:{slot}")
            seen_slots.add(slot_key)
            target = facts[base_fact_id]
            source = facts[source_id]
            if row.get("same_split") is not True:
                raise ValueError(f"Candidate is not same-split: {candidate_id}")
            if row.get("same_leakage_component") is not False:
                raise ValueError(f"Candidate shares a leakage component: {candidate_id}")
            if not (
                row.get("split_assignment") == target.get("split_assignment")
                == source.get("split_assignment")
            ):
                raise ValueError(f"Candidate split lineage is stale: {candidate_id}")
            if kind == "distractor":
                if row.get("verified") is not False:
                    raise ValueError(f"Distractor was already promoted: {candidate_id}")
                if row.get("review_status") != "pending_factual_uniqueness_and_semantic_review":
                    raise ValueError(f"Distractor review_status is invalid: {candidate_id}")
                if not (
                    row.get("relation_partition_id")
                    == target.get("relation_partition_id")
                    == source.get("relation_partition_id")
                ):
                    raise ValueError(f"Distractor relation lineage is stale: {candidate_id}")
                if not _same_text(row.get("distractor_text_en"), source.get("answer_en")):
                    raise ValueError(f"Distractor answer lineage is stale: {candidate_id}")
            else:
                if row.get("verified_unrelated") is not False:
                    raise ValueError(f"Neutral candidate was already promoted: {candidate_id}")
                if row.get("review_status") != "pending_unrelatedness_and_length_review":
                    raise ValueError(f"Neutral review_status is invalid: {candidate_id}")
                if row.get("target_relation_partition_id") != target.get(
                    "relation_partition_id"
                ):
                    raise ValueError(f"Neutral target relation lineage is stale: {candidate_id}")
                if row.get("source_relation_partition_id") != source.get(
                    "relation_partition_id"
                ):
                    raise ValueError(f"Neutral source relation lineage is stale: {candidate_id}")
                if row.get("target_relation_partition_id") == row.get(
                    "source_relation_partition_id"
                ):
                    raise ValueError(f"Neutral candidate shares target relation: {candidate_id}")
                if not _same_text(
                    row.get("neutral_context_candidate_en"), source.get("canonical_fact_en")
                ):
                    raise ValueError(f"Neutral context lineage is stale: {candidate_id}")


def load_review_units(
    *,
    candidate_manifest_path: Path,
    full_base_facts_path: Optional[Path] = None,
    distractor_candidates_path: Optional[Path] = None,
    neutral_candidates_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], str, Dict[str, Dict[str, Any]], List[ReviewUnit]]:
    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    manifest = read_json(candidate_manifest_path)
    manifest_kind, artifacts = _manifest_artifacts(manifest)
    full_path, full_rows, full_binding = _verify_jsonl_binding(
        manifest_path=candidate_manifest_path,
        binding=artifacts.get("full_base_facts"),
        explicit_path=full_base_facts_path,
        label="full_base_facts",
        expected_schema=FULL_FACT_SCHEMA,
        allow_empty=False,
    )
    distractor_path, distractors, distractor_binding = _verify_jsonl_binding(
        manifest_path=candidate_manifest_path,
        binding=artifacts.get("distractor_candidates"),
        explicit_path=distractor_candidates_path,
        label="distractor_candidates",
        expected_schema=DISTRACTOR_SCHEMA,
        allow_empty=True,
    )
    neutral_path, neutrals, neutral_binding = _verify_jsonl_binding(
        manifest_path=candidate_manifest_path,
        binding=artifacts.get("neutral_reference_candidates"),
        explicit_path=neutral_candidates_path,
        label="neutral_reference_candidates",
        expected_schema=NEUTRAL_SCHEMA,
        allow_empty=True,
    )
    if not distractors and not neutrals:
        raise ValueError("Candidate artifacts contain no reviewable records")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Candidate manifest counts are missing")
    expected_counts = {
        "full_base_facts": len(full_rows),
        "distractor_candidates": len(distractors),
        "neutral_reference_candidates": len(neutrals),
    }
    count_keys = {
        "full_base_facts": (
            "retained_base_facts" if manifest_kind == "postreview_rebuild" else "base_facts"
        ),
        "distractor_candidates": "distractor_candidates",
        "neutral_reference_candidates": "neutral_reference_candidates",
    }
    for label, actual in expected_counts.items():
        if counts.get(count_keys[label]) != actual:
            raise ValueError(f"Candidate manifest {count_keys[label]} count is stale")

    facts = _validate_full_facts(full_rows)
    _validate_candidate_rows(distractors=distractors, neutrals=neutrals, facts=facts)
    distractors_by_fact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    neutrals_by_fact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in distractors:
        distractors_by_fact[str(row["base_fact_id"])].append(row)
    for row in neutrals:
        neutrals_by_fact[str(row["base_fact_id"])].append(row)
    for rows in (*distractors_by_fact.values(), *neutrals_by_fact.values()):
        rows.sort(key=lambda row: (int(row["slot"]), str(row.get("distractor_id") or row.get("neutral_candidate_id"))))

    ordered: List[Tuple[str, Dict[str, Any]]] = []
    for base_fact_id in sorted(set(distractors_by_fact) | set(neutrals_by_fact)):
        ordered.extend(("distractor", row) for row in distractors_by_fact[base_fact_id])
        ordered.extend(("neutral", row) for row in neutrals_by_fact[base_fact_id])
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        ordered = ordered[:limit]

    units: List[ReviewUnit] = []
    for input_index, (kind, candidate) in enumerate(ordered):
        id_field = "distractor_id" if kind == "distractor" else "neutral_candidate_id"
        candidate_id = str(candidate[id_field])
        target_id = str(candidate["base_fact_id"])
        source_id = str(candidate["source_base_fact_id"])
        siblings = (
            distractors_by_fact[target_id]
            if kind == "distractor"
            else neutrals_by_fact[target_id]
        )
        units.append(
            ReviewUnit(
                input_index=input_index,
                candidate_id=candidate_id,
                candidate_kind=kind,
                candidate_row_sha256=sha256_value(candidate),
                target_base_fact_id=target_id,
                target_base_fact_row_sha256=sha256_value(facts[target_id]),
                source_base_fact_id=source_id,
                source_base_fact_row_sha256=sha256_value(facts[source_id]),
                projection=_project_candidate(
                    candidate_kind=kind,
                    candidate=candidate,
                    target=facts[target_id],
                    source=facts[source_id],
                    siblings=siblings,
                ),
            )
        )
    bindings = {
        "full_base_facts": {**full_binding, "path": str(full_path)},
        "distractor_candidates": {**distractor_binding, "path": str(distractor_path)},
        "neutral_reference_candidates": {**neutral_binding, "path": str(neutral_path)},
    }
    return manifest, manifest_kind, bindings, units


def _response_contract() -> Dict[str, Any]:
    return {
        "top_level": {
            "schema_version": BATCH_RESPONSE_SCHEMA,
            "records": "exactly one result for every supplied candidate_id",
        },
        "record": {
            "candidate_id": "copy the exact candidate ID",
            "candidate_kind": sorted(CANDIDATE_KINDS),
            "overall_decision": sorted(OVERALL_DECISIONS),
            "distractor_factually_false": "boolean/null under the instructions",
            "answer_unique": "boolean/null under the instructions",
            "answer_alias_disjoint": "boolean/null under the instructions",
            "neutral_semantically_neutral": "boolean/null under the instructions",
            "neutral_unrelated": "boolean/null under the instructions",
            "overlapping_target_aliases": (
                "distractor: [] when disjoint or unresolved, otherwise a non-empty "
                "subset copied exactly from target_fact.answer_aliases_en; neutral: null"
            ),
            "answer_alias_disjoint_unresolved_evidence": (
                "distractor: non-empty string only when answer_alias_disjoint is null; "
                "otherwise null; neutral: null"
            ),
            "rationale": "non-empty evidence-grounded explanation",
            "confidence": sorted(CONFIDENCE_LEVELS),
        },
    }


def prompt_contract_payload() -> Dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "instructions": REVIEW_INSTRUCTIONS,
        "projection_schema": PROJECTION_SCHEMA,
        "batch_response_schema": BATCH_RESPONSE_SCHEMA,
        "field_names": FIELD_NAMES,
        "judgment_field_names": JUDGMENT_FIELD_NAMES,
        "review_method": REVIEW_METHOD,
        "response_validator_contract": RESPONSE_VALIDATOR_CONTRACT,
        "response_contract": _response_contract(),
    }


def review_semantics_contract(
    *,
    reviewer_spec: Mapping[str, Any],
    expected_response_model: str,
    protocol_identity: Mapping[str, str],
    route_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    semantic_spec_fields = (
        "provider_profile",
        "model",
        "temperature",
        "reasoning_effort",
        "max_output_tokens",
        "disable_thinking",
        "json_mode",
    )
    semantic_spec = {
        field: copy.deepcopy(reviewer_spec[field])
        for field in semantic_spec_fields
        if field in reviewer_spec
    }
    answer_alias_contract = {
        "answer_alias_disjoint_scope": "target_fact.answer_aliases_en_only",
        "candidate_source_fact_role": "provenance_only",
        "candidate_source_aliases_in_scope": False,
        "contradictory_alias_evidence_fails_closed": True,
    }
    return {
        "schema_version": "public-benchmark-candidate-review-semantics-contract-v1",
        "prompt_contract_sha256": sha256_value(prompt_contract_payload()),
        "prompt_version": PROMPT_VERSION,
        "projection_schema": PROJECTION_SCHEMA,
        "batch_response_schema": BATCH_RESPONSE_SCHEMA,
        "adjudication_schema": ADJUDICATION_SCHEMA,
        "review_method": REVIEW_METHOD,
        "response_validator_contract": RESPONSE_VALIDATOR_CONTRACT,
        "answer_alias_contract": answer_alias_contract,
        "reviewer_spec": semantic_spec,
        "expected_response_model": expected_response_model,
        "protocol_reviewer_identity": dict(protocol_identity),
        "route_identity_sha256": sha256_value(route_identity),
        "behavior_blind": True,
        "target_behavior_consumed": False,
        "human_gold": False,
    }


def unit_projection_sha256(unit: ReviewUnit) -> str:
    return sha256_value(unit.projection)


def unit_evidence_identity(
    unit: ReviewUnit, review_semantics_contract_sha256: str
) -> Dict[str, Any]:
    return {
        "schema_version": EVIDENCE_IDENTITY_SCHEMA,
        "candidate_id": unit.candidate_id,
        "candidate_kind": unit.candidate_kind,
        "projection_schema": unit.projection.get("schema_version"),
        "projection_sha256": unit_projection_sha256(unit),
        "review_semantics_contract_sha256": review_semantics_contract_sha256,
    }


def unit_evidence_identity_sha256(
    unit: ReviewUnit, review_semantics_contract_sha256: str
) -> str:
    return sha256_value(
        unit_evidence_identity(unit, review_semantics_contract_sha256)
    )


def current_unit_binding_sha256(unit: ReviewUnit) -> str:
    return sha256_value(
        {
            "candidate_id": unit.candidate_id,
            "candidate_kind": unit.candidate_kind,
            "candidate_row_sha256": unit.candidate_row_sha256,
            "target_base_fact_id": unit.target_base_fact_id,
            "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
            "source_base_fact_id": unit.source_base_fact_id,
            "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
        }
    )


def build_review_prompt(units: Sequence[ReviewUnit]) -> str:
    payload = {
        "response_contract": _response_contract(),
        "candidates": [unit.projection for unit in units],
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


def _validate_alias_evidence(
    record: Mapping[str, Any], unit: ReviewUnit, candidate_id: str
) -> None:
    overlaps = record.get("overlapping_target_aliases")
    unresolved = record.get("answer_alias_disjoint_unresolved_evidence")
    if unit.candidate_kind == "neutral":
        if overlaps is not None or unresolved is not None:
            raise ValueError(
                f"Neutral alias-evidence fields must be null: {candidate_id}"
            )
        return

    if not isinstance(overlaps, list):
        raise ValueError(
            f"Distractor overlapping_target_aliases must be a list: {candidate_id}"
        )
    target_fact = unit.projection.get("target_fact")
    if not isinstance(target_fact, Mapping):
        raise ValueError(f"Target fact projection is invalid: {candidate_id}")
    target_aliases = target_fact.get("answer_aliases_en")
    if not isinstance(target_aliases, list) or not target_aliases:
        raise ValueError(f"Target alias scope is invalid: {candidate_id}")
    checked_target_aliases = [
        _required_string(alias, f"{candidate_id}.target_fact.answer_aliases_en")
        for alias in target_aliases
    ]
    checked_overlaps = [
        _required_string(alias, f"{candidate_id}.overlapping_target_aliases")
        for alias in overlaps
    ]
    if len(checked_overlaps) != len(set(checked_overlaps)):
        raise ValueError(f"Overlapping target aliases must be unique: {candidate_id}")
    if any(alias not in checked_target_aliases for alias in checked_overlaps):
        raise ValueError(
            f"Overlapping aliases must come only from target aliases: {candidate_id}"
        )

    alias_judgment = record.get("answer_alias_disjoint")
    if alias_judgment is True:
        if checked_overlaps:
            raise ValueError(
                f"Disjoint=true requires no overlapping target aliases: {candidate_id}"
            )
        if unresolved is not None:
            raise ValueError(
                f"Resolved alias judgment requires null unresolved evidence: {candidate_id}"
            )
    elif alias_judgment is False:
        if not checked_overlaps:
            raise ValueError(
                f"Disjoint=false requires overlapping target aliases: {candidate_id}"
            )
        if unresolved is not None:
            raise ValueError(
                f"Resolved alias judgment requires null unresolved evidence: {candidate_id}"
            )
    elif alias_judgment is None:
        if checked_overlaps:
            raise ValueError(
                f"Unresolved alias judgment requires no claimed overlaps: {candidate_id}"
            )
        _required_string(
            unresolved,
            f"{candidate_id}.answer_alias_disjoint_unresolved_evidence",
        )
    else:
        raise ValueError(
            f"answer_alias_disjoint must be boolean or null: {candidate_id}"
        )

    evidence = unit.projection.get("candidate_evidence")
    distractor_text = (
        evidence.get("distractor_text_en") if isinstance(evidence, Mapping) else None
    )
    distractor_text = _required_string(
        distractor_text, f"{candidate_id}.candidate_evidence.distractor_text_en"
    )
    exact_target_aliases = [
        alias for alias in checked_target_aliases if _same_text(alias, distractor_text)
    ]
    if exact_target_aliases and alias_judgment is not False:
        raise ValueError(
            f"Exact target-alias match contradicts answer_alias_disjoint: {candidate_id}"
        )
    if any(alias not in checked_overlaps for alias in exact_target_aliases):
        raise ValueError(
            f"Exact target-alias matches must be reported as overlaps: {candidate_id}"
        )


def validate_batch_response(value: Dict[str, Any], units: Sequence[ReviewUnit]) -> None:
    if set(value) != {"schema_version", "records"}:
        raise ValueError("Batch response must contain exactly schema_version and records")
    if value.get("schema_version") != BATCH_RESPONSE_SCHEMA:
        raise ValueError("Batch response schema_version is unsupported")
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError("Batch response records must be a list")
    expected = {unit.candidate_id: unit for unit in units}
    by_id: Dict[str, Dict[str, Any]] = {}
    required_fields = {
        "candidate_id",
        "candidate_kind",
        "overall_decision",
        *FIELD_NAMES,
        "rationale",
        "confidence",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != required_fields:
            raise ValueError("Batch response record fields are invalid")
        candidate_id = _required_string(record.get("candidate_id"), "candidate_id")
        if candidate_id in by_id:
            raise ValueError(f"Duplicate response candidate_id: {candidate_id}")
        unit = expected.get(candidate_id)
        if unit is None:
            raise ValueError(f"Unknown response candidate_id: {candidate_id}")
        if record.get("candidate_kind") != unit.candidate_kind:
            raise ValueError(f"Response candidate_kind mismatch: {candidate_id}")
        if record.get("overall_decision") not in OVERALL_DECISIONS:
            raise ValueError(f"Invalid overall_decision: {candidate_id}")
        relevant = (
            JUDGMENT_FIELD_NAMES[:3]
            if unit.candidate_kind == "distractor"
            else JUDGMENT_FIELD_NAMES[3:]
        )
        irrelevant = (
            JUDGMENT_FIELD_NAMES[3:]
            if unit.candidate_kind == "distractor"
            else JUDGMENT_FIELD_NAMES[:3]
        )
        for field in irrelevant:
            if record.get(field) is not None:
                raise ValueError(f"Irrelevant field must be null: {candidate_id}.{field}")
        relevant_values = [record.get(field) for field in relevant]
        if any(
            value is not None and not isinstance(value, bool)
            for value in relevant_values
        ):
            raise ValueError(f"Relevant field must be boolean or null: {candidate_id}")
        _validate_alias_evidence(record, unit, candidate_id)
        decision = record["overall_decision"]
        if decision == "accept" and not all(value is True for value in relevant_values):
            raise ValueError(f"Accept requires every relevant field true: {candidate_id}")
        if decision == "reject" and not any(value is False for value in relevant_values):
            raise ValueError(f"Reject requires a false relevant field: {candidate_id}")
        if decision == "defer" and not (
            all(value is not False for value in relevant_values)
            and any(value is None for value in relevant_values)
        ):
            raise ValueError(f"Defer requires unresolved relevant evidence: {candidate_id}")
        _required_string(record.get("rationale"), f"{candidate_id}.rationale")
        if record.get("confidence") not in CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence: {candidate_id}")
        by_id[candidate_id] = dict(record)
    if set(by_id) != set(expected):
        raise ValueError("Batch response does not exactly cover requested candidate IDs")


def _response_map(value: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["candidate_id"]): dict(row) for row in value["records"]}


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
    record: Dict[str, Any] = {"error_type": type(error).__name__}
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        record["http_status"] = status
    return record


def _call_model(
    *,
    router: Any,
    units: Sequence[ReviewUnit],
    reviewer_spec: Mapping[str, Any],
    expected_response_model: str,
    prompt: str,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    candidate_ids = [unit.candidate_id for unit in units]
    batch_id = "candidate_batch_" + sha256_value(candidate_ids)[:20]
    safe_spec = _safe_model_spec(reviewer_spec)
    request_binding = {
        "stage": "full_candidate_semantic_review",
        "batch_id": batch_id,
        "ordered_candidate_ids": candidate_ids,
        "model_spec": safe_spec,
        "expected_response_model": expected_response_model,
        "prompt": prompt,
    }
    try:
        result = router.request_json(
            "full_candidate_semantic_review",
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
            "stage": "full_candidate_semantic_review",
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch_id,
            "batch_item_count": len(units),
            "ordered_candidate_ids_sha256": sha256_value(candidate_ids),
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
            "stage": "full_candidate_semantic_review",
            "prompt_version": PROMPT_VERSION,
            "batch_id": batch_id,
            "batch_item_count": len(units),
            "ordered_candidate_ids_sha256": sha256_value(candidate_ids),
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
        "candidate_id": unit.candidate_id,
        "candidate_kind": unit.candidate_kind,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "target_base_fact_id": unit.target_base_fact_id,
        "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
        "source_base_fact_id": unit.source_base_fact_id,
        "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
        "overall_decision": verdict["overall_decision"],
        **{field: verdict[field] for field in FIELD_NAMES},
        "rationale": verdict["rationale"],
        "confidence": verdict["confidence"],
        "reviewer_type": "independent_model_proxy",
        "reviewer_id": f"{reviewer_spec['provider_profile']}:{response_model}",
        "requested_model": reviewer_spec["model"],
        "expected_response_model": expected_response_model,
        "response_model": response_model,
        "response_model_identity_status": "matched",
        "review_method": REVIEW_METHOD,
        "reviewed_at": reviewed_at,
        "behavior_blind": True,
        "target_behavior_consumed": False,
        "human_gold": False,
    }


def _failed_rows(
    *,
    units: Sequence[ReviewUnit],
    traces: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
    review_semantics_contract_sha256: str,
    now_fn: Callable[[], str],
) -> List[Dict[str, Any]]:
    return [
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "tool_version": TOOL_VERSION,
            "candidate_id": unit.candidate_id,
            "candidate_kind": unit.candidate_kind,
            "input_index": unit.input_index,
            "candidate_row_sha256": unit.candidate_row_sha256,
            "target_base_fact_id": unit.target_base_fact_id,
            "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
            "source_base_fact_id": unit.source_base_fact_id,
            "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
            "run_contract_sha256": run_contract_sha256,
            "review_semantics_contract_sha256": review_semantics_contract_sha256,
            "projection_sha256": unit_projection_sha256(unit),
            "evidence_identity_sha256": unit_evidence_identity_sha256(
                unit, review_semantics_contract_sha256
            ),
            "current_unit_binding_sha256": current_unit_binding_sha256(unit),
            "evidence_origin": FRESH_EVIDENCE_ORIGIN,
            "model_calls_scope": "current_run",
            "carry_forward_provenance": None,
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
            "hf_tokenizer_execution_count": 0,
            "behavior_execution_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
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
    review_semantics_contract_sha256: str,
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
                    review_semantics_contract_sha256=review_semantics_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
                *_process_batch(
                    units=units[midpoint:],
                    router=router,
                    reviewer_spec=reviewer_spec,
                    expected_response_model=expected_response_model,
                    run_contract_sha256=run_contract_sha256,
                    review_semantics_contract_sha256=review_semantics_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
            ]
        return _failed_rows(
            units=units,
            traces=traces,
            run_contract_sha256=run_contract_sha256,
            review_semantics_contract_sha256=review_semantics_contract_sha256,
            now_fn=now_fn,
        )
    response_model = str(call["response_model"])
    output: List[Dict[str, Any]] = []
    for unit in units:
        reviewed_at = now_fn()
        verdict = verdicts[unit.candidate_id]
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
                "candidate_id": unit.candidate_id,
                "candidate_kind": unit.candidate_kind,
                "input_index": unit.input_index,
                "candidate_row_sha256": unit.candidate_row_sha256,
                "target_base_fact_id": unit.target_base_fact_id,
                "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
                "source_base_fact_id": unit.source_base_fact_id,
                "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
                "run_contract_sha256": run_contract_sha256,
                "review_semantics_contract_sha256": review_semantics_contract_sha256,
                "projection_sha256": unit_projection_sha256(unit),
                "evidence_identity_sha256": unit_evidence_identity_sha256(
                    unit, review_semantics_contract_sha256
                ),
                "current_unit_binding_sha256": current_unit_binding_sha256(unit),
                "evidence_origin": FRESH_EVIDENCE_ORIGIN,
                "model_calls_scope": "current_run",
                "carry_forward_provenance": None,
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
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
            }
        )
    return output


def _chunks(values: Sequence[ReviewUnit], size: int) -> Iterator[List[ReviewUnit]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    units_by_id: Mapping[str, ReviewUnit],
    run_contract_sha256: str,
    review_semantics_contract_sha256: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    candidate_id = _required_string(row.get("candidate_id"), "checkpoint.candidate_id")
    unit = units_by_id.get(candidate_id)
    if unit is None:
        raise ValueError(f"Checkpoint candidate is outside selected scope: {candidate_id}")
    if row.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Checkpoint schema is unsupported: {candidate_id}")
    if row.get("tool_version") != TOOL_VERSION:
        raise ValueError(f"Checkpoint tool version is unsupported: {candidate_id}")
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError(f"Checkpoint run contract is stale: {candidate_id}")
    if review_semantics_contract_sha256 is None:
        review_semantics_contract_sha256 = _required_string(
            row.get("review_semantics_contract_sha256"),
            f"{candidate_id}.review_semantics_contract_sha256",
        )
    if row.get("review_semantics_contract_sha256") != review_semantics_contract_sha256:
        raise ValueError(f"Checkpoint review semantics are stale: {candidate_id}")
    expected = {
        "candidate_kind": unit.candidate_kind,
        "input_index": unit.input_index,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "target_base_fact_id": unit.target_base_fact_id,
        "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
        "source_base_fact_id": unit.source_base_fact_id,
        "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"Checkpoint {field} is stale: {candidate_id}")
    expected_evidence = {
        "projection_sha256": unit_projection_sha256(unit),
        "evidence_identity_sha256": unit_evidence_identity_sha256(
            unit, review_semantics_contract_sha256
        ),
        "current_unit_binding_sha256": current_unit_binding_sha256(unit),
    }
    for field, value in expected_evidence.items():
        if row.get(field) != value:
            raise ValueError(f"Checkpoint {field} is stale: {candidate_id}")
    origin = row.get("evidence_origin")
    if origin not in EVIDENCE_ORIGINS:
        raise ValueError(f"Checkpoint evidence_origin is invalid: {candidate_id}")
    if origin == FRESH_EVIDENCE_ORIGIN:
        if row.get("model_calls_scope") != "current_run":
            raise ValueError(f"Fresh checkpoint model call scope is invalid: {candidate_id}")
        if row.get("carry_forward_provenance") is not None:
            raise ValueError(f"Fresh checkpoint has carry provenance: {candidate_id}")
    else:
        if row.get("terminal_status") != "completed":
            raise ValueError(f"Carried checkpoint cannot be failed: {candidate_id}")
        if row.get("model_calls_scope") != "none_carried_forward":
            raise ValueError(f"Carried checkpoint model call scope is invalid: {candidate_id}")
        if row.get("model_calls") != [] or row.get("retry_history") != []:
            raise ValueError(f"Carried checkpoint contains current model activity: {candidate_id}")
        if not isinstance(row.get("carry_forward_provenance"), dict):
            raise ValueError(f"Carried checkpoint lacks provenance: {candidate_id}")
    if row.get("terminal_status") not in {"completed", "failed"}:
        raise ValueError(f"Checkpoint terminal_status is invalid: {candidate_id}")
    if row.get("behavior_blind") is not True or row.get("human_gold") is not False:
        raise ValueError(f"Checkpoint evidence boundary is invalid: {candidate_id}")
    for field in (
        "hf_model_execution_count",
        "hf_tokenizer_execution_count",
        "behavior_execution_count",
        "validation_behavior_exposure_count",
        "sealed_behavior_exposure_count",
    ):
        if row.get(field) != 0:
            raise ValueError(f"Checkpoint execution boundary is invalid: {candidate_id}.{field}")
    if not isinstance(row.get("model_calls"), list) or not isinstance(
        row.get("retry_history"), list
    ):
        raise ValueError(f"Checkpoint trace/history is invalid: {candidate_id}")
    if row["terminal_status"] == "completed":
        decision = row.get("strict_adjudication")
        if not isinstance(decision, dict):
            raise ValueError(f"Checkpoint lacks adjudication: {candidate_id}")
        verdict = row.get("model_verdict")
        if not isinstance(verdict, dict):
            raise ValueError(f"Checkpoint lacks model verdict: {candidate_id}")
        validate_batch_response(
            {"schema_version": BATCH_RESPONSE_SCHEMA, "records": [verdict]},
            [unit],
        )
        if decision.get("candidate_id") != candidate_id:
            raise ValueError(f"Checkpoint adjudication ID is stale: {candidate_id}")
        if decision.get("schema_version") != ADJUDICATION_SCHEMA:
            raise ValueError(f"Checkpoint adjudication schema is stale: {candidate_id}")
        if decision.get("review_method") != REVIEW_METHOD:
            raise ValueError(f"Checkpoint adjudication method is stale: {candidate_id}")
        if decision.get("candidate_row_sha256") != unit.candidate_row_sha256:
            raise ValueError(f"Checkpoint adjudication row hash is stale: {candidate_id}")
        if decision.get("behavior_blind") is not True or decision.get("human_gold") is not False:
            raise ValueError(f"Checkpoint adjudication boundary is invalid: {candidate_id}")
        expected_identity = {
            "requested_model": DEFAULT_REVIEWER_MODEL,
            "expected_response_model": DEFAULT_EXPECTED_RESPONSE_MODEL,
            "response_model": DEFAULT_EXPECTED_RESPONSE_MODEL,
            "response_model_identity_status": "matched",
        }
        for field, expected in expected_identity.items():
            if decision.get(field) != expected:
                raise ValueError(
                    f"Checkpoint adjudication model identity is invalid: {candidate_id}.{field}"
                )
        for field in ("overall_decision", *FIELD_NAMES, "rationale", "confidence"):
            if decision.get(field) != verdict.get(field):
                raise ValueError(
                    f"Checkpoint adjudication/model verdict mismatch: {candidate_id}.{field}"
                )
        if origin == FRESH_EVIDENCE_ORIGIN:
            calls = row["model_calls"]
            final_call = calls[-1] if calls and isinstance(calls[-1], dict) else {}
            if any(
                final_call.get(field) != expected
                for field, expected in {
                    "prompt_version": PROMPT_VERSION,
                    "provider_profile": PROTOCOL_REVIEWER_PROVIDER_PROFILE,
                    "requested_model": DEFAULT_REVIEWER_MODEL,
                    "expected_response_model": DEFAULT_EXPECTED_RESPONSE_MODEL,
                    "response_model": DEFAULT_EXPECTED_RESPONSE_MODEL,
                    "response_model_identity_status": "matched",
                }.items()
            ):
                raise ValueError(f"Checkpoint response model is not verified: {candidate_id}")
        elif decision.get("overall_decision") != "accept":
            raise ValueError(f"Carried checkpoint is not accepted: {candidate_id}")
    elif row.get("strict_adjudication") is not None:
        raise ValueError(f"Failed checkpoint contains adjudication: {candidate_id}")
    return candidate_id, dict(row)


def _read_journal(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    path = Path(path)
    if not path.is_file():
        return [], False
    payload = path.read_bytes()
    if not payload:
        return [], False
    rows: List[Dict[str, Any]] = []
    needs_repair = False
    lines = payload.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        is_last = index == len(lines) - 1
        terminated = raw_line.endswith((b"\n", b"\r"))
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if is_last and not terminated:
                needs_repair = True
                break
            raise ValueError(f"Malformed checkpoint journal line {index + 1}: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in checkpoint journal line {index + 1}")
        rows.append(value)
        if is_last and not terminated:
            needs_repair = True
    return rows, needs_repair


def _checkpoint_state(
    *,
    checkpoint_path: Path,
    journal_path: Path,
    units_by_id: Mapping[str, ReviewUnit],
    run_contract_sha256: str,
    review_semantics_contract_sha256: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], bool]:
    output: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.is_file():
        for row in read_jsonl(checkpoint_path, allow_empty=True):
            candidate_id, validated = _validate_checkpoint_row(
                row,
                units_by_id=units_by_id,
                run_contract_sha256=run_contract_sha256,
                review_semantics_contract_sha256=review_semantics_contract_sha256,
            )
            if candidate_id in output:
                raise ValueError(f"Duplicate compact checkpoint candidate_id: {candidate_id}")
            output[candidate_id] = validated
    journal_rows, needs_repair = _read_journal(journal_path)
    for row in journal_rows:
        candidate_id, validated = _validate_checkpoint_row(
            row,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha256,
            review_semantics_contract_sha256=review_semantics_contract_sha256,
        )
        output[candidate_id] = validated
    return output, journal_rows, needs_repair


def _ordered_checkpoint_rows(
    checkpoint: Mapping[str, Mapping[str, Any]], units: Sequence[ReviewUnit]
) -> List[Dict[str, Any]]:
    return [
        dict(checkpoint[unit.candidate_id])
        for unit in units
        if unit.candidate_id in checkpoint
    ]


def _append_journal(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_json_binding(
    *, owner_path: Path, binding: Any, label: str, expected_schema: Optional[str] = None
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is invalid")
    path = _resolve_artifact_path(
        manifest_path=owner_path,
        binding=binding,
        explicit_path=None,
        label=label,
    )
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 mismatch")
    if binding.get("byte_count") is not None and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count mismatch")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version mismatch")
    value = read_json(path)
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError(f"{label} payload schema_version mismatch")
    return path, value, dict(binding)


def _same_bound_file(
    left: Mapping[str, Any], left_owner: Path, right: Mapping[str, Any], right_owner: Path
) -> bool:
    return (
        _resolve_artifact_path(
            manifest_path=left_owner,
            binding=left,
            explicit_path=None,
            label="left binding",
        )
        == _resolve_artifact_path(
            manifest_path=right_owner,
            binding=right,
            explicit_path=None,
            label="right binding",
        )
        and left.get("sha256") == right.get("sha256")
    )


def discover_direct_predecessor_review(
    *, candidate_manifest_path: Path, candidate_manifest: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Resolve the only review authority eligible for monotone carry-forward."""

    inputs = candidate_manifest.get("inputs")
    repair = inputs.get("candidate_repair") if isinstance(inputs, dict) else None
    if repair is None:
        return None
    if not isinstance(repair, dict):
        raise ValueError("Candidate repair lineage is invalid")
    resolution_path, resolution, resolution_binding = _verify_json_binding(
        owner_path=candidate_manifest_path,
        binding=repair.get("candidate_resolution_manifest"),
        label="candidate resolution manifest",
        expected_schema=None,
    )
    resolution_schema = resolution.get("schema_version")
    if resolution_schema not in {
        LEGACY_RESOLUTION_MANIFEST_SCHEMA,
        CURRENT_RESOLUTION_MANIFEST_SCHEMA,
    } or resolution_binding.get("schema_version") != resolution_schema:
        raise ValueError("Candidate resolution manifest schema is unsupported")
    resolution_inputs = resolution.get("inputs")
    if not isinstance(resolution_inputs, dict):
        raise ValueError("Candidate resolution inputs are missing")
    predecessor_binding = repair.get("predecessor_postreview_manifest")
    resolution_predecessor = resolution_inputs.get("predecessor_postreview_manifest")
    if not isinstance(predecessor_binding, dict) or not isinstance(
        resolution_predecessor, dict
    ) or not _same_bound_file(
        predecessor_binding,
        candidate_manifest_path,
        resolution_predecessor,
        resolution_path,
    ):
        raise ValueError("Direct predecessor postreview lineage is stale")
    review_binding = resolution_inputs.get("candidate_review_manifest")
    review_path, review_manifest, verified_review_binding = _verify_json_binding(
        owner_path=resolution_path,
        binding=review_binding,
        label="direct predecessor candidate review",
        expected_schema=RUN_MANIFEST_SCHEMA,
    )
    adjudication_binding = resolution_inputs.get("candidate_review_adjudications")
    if not isinstance(adjudication_binding, dict):
        raise ValueError("Direct predecessor candidate adjudication binding is missing")
    checkpoint_binding = resolution_inputs.get("candidate_review_checkpoint")
    if checkpoint_binding is None and resolution_schema == LEGACY_RESOLUTION_MANIFEST_SCHEMA:
        artifacts = review_manifest.get("artifacts")
        checkpoint_binding = artifacts.get("checkpoint") if isinstance(artifacts, dict) else None
    if not isinstance(checkpoint_binding, dict):
        raise ValueError("Direct predecessor candidate checkpoint binding is missing")
    return {
        "review_path": review_path,
        "review_manifest": review_manifest,
        "review_binding": verified_review_binding,
        "resolution_manifest": resolution_binding,
        "resolution_manifest_path": str(resolution_path),
        "predecessor_postreview_manifest": dict(predecessor_binding),
        "resolution_adjudications": dict(adjudication_binding),
        "resolution_checkpoint": dict(checkpoint_binding),
    }


def _validate_evidence_partition(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], units: Sequence[ReviewUnit]
) -> None:
    if len(rows) != len(units):
        raise ValueError("Evidence partition does not cover the selected candidate set")
    origins = [str(row.get("evidence_origin")) for row in rows]
    counts = dict(sorted(Counter(origins).items()))
    if set(counts).difference(EVIDENCE_ORIGINS):
        raise ValueError("Evidence partition contains an unsupported origin")
    if manifest.get("evidence_origin_counts") != counts:
        raise ValueError("Evidence origin counts are stale")
    fresh_ids = [
        unit.candidate_id
        for unit, row in zip(units, rows)
        if row.get("evidence_origin") == FRESH_EVIDENCE_ORIGIN
    ]
    carried_ids = [
        unit.candidate_id
        for unit, row in zip(units, rows)
        if row.get("evidence_origin") == CARRIED_EVIDENCE_ORIGIN
    ]
    expected = {
        "fresh_model_review_candidate_count": len(fresh_ids),
        "carried_forward_candidate_count": len(carried_ids),
        "fresh_model_review_candidate_ids_sha256": sha256_value(fresh_ids),
        "carried_forward_candidate_ids_sha256": sha256_value(carried_ids),
        "full_evidence_partition_complete": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Evidence partition field is stale: {field}")


def load_review_evidence_manifest(
    review_manifest_path: Path,
    *,
    _lineage_stack: Tuple[Path, ...] = (),
) -> Dict[str, Any]:
    """Validate a complete review plus every bound carry-forward ancestor."""

    review_manifest_path = Path(review_manifest_path).resolve()
    if review_manifest_path in _lineage_stack:
        raise ValueError("Candidate review carry-forward lineage cycle detected")
    manifest = read_json(review_manifest_path)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ValueError("Candidate review manifest schema is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION:
        raise ValueError("Candidate review manifest tool version is unsupported")
    if manifest.get("status") != "completed":
        raise ValueError("Candidate review source is not completed")
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict) or manifest.get(
        "run_contract_sha256"
    ) != sha256_value(run_contract):
        raise ValueError("Candidate review source run contract is stale")
    if run_contract.get("tool_version") != TOOL_VERSION or run_contract.get(
        "source_fingerprints"
    ) != source_fingerprints():
        raise ValueError("Candidate review source implementation identity is stale")
    semantics = run_contract.get("review_semantics_contract")
    semantics_sha = run_contract.get("review_semantics_contract_sha256")
    if not isinstance(semantics, dict) or semantics_sha != sha256_value(semantics):
        raise ValueError("Candidate review semantics contract is stale")
    candidate_binding = run_contract.get("candidate_manifest")
    if not isinstance(candidate_binding, dict):
        raise ValueError("Candidate review source manifest binding is missing")
    candidate_path = _resolve_artifact_path(
        manifest_path=review_manifest_path,
        binding=candidate_binding,
        explicit_path=None,
        label="candidate review source manifest",
    )
    if candidate_binding.get("sha256") != sha256_file(candidate_path):
        raise ValueError("Candidate review source manifest SHA-256 is stale")
    candidate_manifest, manifest_kind, input_bindings, units = load_review_units(
        candidate_manifest_path=candidate_path
    )
    if manifest_kind != candidate_binding.get("manifest_kind"):
        raise ValueError("Candidate review source manifest kind is stale")
    unit_ids = [unit.candidate_id for unit in units]
    if run_contract.get("input_artifacts") != input_bindings or not all(
        (
            run_contract.get("selected_candidate_count") == len(units),
            run_contract.get("full_candidate_count") == len(units),
            run_contract.get("selection_is_full_candidate_set") is True,
            run_contract.get("selected_candidate_ids_sha256") == sha256_value(unit_ids),
        )
    ):
        raise ValueError("Candidate review source coverage contract is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Candidate review source artifacts are missing")
    checkpoint_path, checkpoints, checkpoint_binding = _verify_jsonl_binding(
        manifest_path=review_manifest_path,
        binding=artifacts.get("checkpoint"),
        explicit_path=None,
        label="candidate review checkpoint",
        expected_schema=CHECKPOINT_SCHEMA,
        allow_empty=False,
    )
    adjudications_path, adjudications, adjudication_binding = _verify_jsonl_binding(
        manifest_path=review_manifest_path,
        binding=artifacts.get("adjudications"),
        explicit_path=None,
        label="candidate review adjudications",
        expected_schema=ADJUDICATION_SCHEMA,
        allow_empty=False,
    )
    if len(checkpoints) != len(units) or len(adjudications) != len(units):
        raise ValueError("Candidate review source artifacts do not fully cover candidates")
    checkpoint_by_id = _index_unique(checkpoints, "candidate_id", "candidate checkpoints")
    adjudication_by_id = _index_unique(
        adjudications, "candidate_id", "candidate adjudications"
    )
    if set(checkpoint_by_id) != set(unit_ids) or set(adjudication_by_id) != set(unit_ids):
        raise ValueError("Candidate review source artifact ID coverage is stale")
    units_by_id = {unit.candidate_id: unit for unit in units}
    ordered_checkpoints: List[Dict[str, Any]] = []
    for unit in units:
        _, checkpoint = _validate_checkpoint_row(
            checkpoint_by_id[unit.candidate_id],
            units_by_id=units_by_id,
            run_contract_sha256=str(manifest["run_contract_sha256"]),
            review_semantics_contract_sha256=str(semantics_sha),
        )
        if checkpoint.get("terminal_status") != "completed":
            raise ValueError(f"Candidate review source is not terminal: {unit.candidate_id}")
        if checkpoint.get("strict_adjudication") != adjudication_by_id[unit.candidate_id]:
            raise ValueError(
                f"Candidate review source checkpoint/adjudication mismatch: {unit.candidate_id}"
            )
        ordered_checkpoints.append(checkpoint)
    _validate_evidence_partition(manifest, ordered_checkpoints, units)

    carried = [
        row
        for row in ordered_checkpoints
        if row.get("evidence_origin") == CARRIED_EVIDENCE_ORIGIN
    ]
    reuse = run_contract.get("evidence_reuse_contract")
    if not isinstance(reuse, dict) or reuse.get("schema_version") != EVIDENCE_REUSE_CONTRACT_SCHEMA:
        raise ValueError("Candidate review evidence reuse contract is invalid")
    source_bundle = None
    source_binding = reuse.get("source_review_manifest")
    if source_binding is not None:
        if reuse.get("mode") != "direct_predecessor_exact_accept":
            raise ValueError("Carry-forward source mode is invalid")
        source_path, _, verified_source_binding = _verify_json_binding(
            owner_path=review_manifest_path,
            binding=source_binding,
            label="carry-forward source review",
            expected_schema=RUN_MANIFEST_SCHEMA,
        )
        source_bundle = load_review_evidence_manifest(
            source_path, _lineage_stack=(*_lineage_stack, review_manifest_path)
        )
        direct = discover_direct_predecessor_review(
            candidate_manifest_path=candidate_path,
            candidate_manifest=candidate_manifest,
        )
        if direct is None or direct["review_path"] != source_path:
            raise ValueError("Carry-forward source is not the direct predecessor review")
        declared_lineage = reuse.get("direct_predecessor_lineage")
        expected_lineage = {
            "candidate_resolution_manifest": direct["resolution_manifest"],
            "predecessor_postreview_manifest": direct[
                "predecessor_postreview_manifest"
            ],
        }
        if declared_lineage != expected_lineage:
            raise ValueError("Carry-forward direct predecessor lineage is stale")
        if source_bundle["review_semantics_contract_sha256"] != semantics_sha:
            raise ValueError("Carry-forward review semantics contract changed")
        if reuse.get("source_run_contract_sha256") != source_bundle[
            "run_contract_sha256"
        ]:
            raise ValueError("Carry-forward source run contract is stale")
        for row in carried:
            candidate_id = str(row["candidate_id"])
            provenance = row["carry_forward_provenance"]
            source_checkpoint = source_bundle["checkpoint_by_id"].get(candidate_id)
            source_adjudication = source_bundle["adjudication_by_id"].get(candidate_id)
            if source_checkpoint is None or source_adjudication is None:
                raise ValueError(f"Carry-forward source evidence is missing: {candidate_id}")
            expected_provenance = {
                "source_review_manifest_sha256": verified_source_binding["sha256"],
                "source_run_contract_sha256": source_bundle["run_contract_sha256"],
                "source_checkpoint_artifact_sha256": source_bundle["checkpoint_binding"]["sha256"],
                "source_adjudications_artifact_sha256": source_bundle["adjudication_binding"]["sha256"],
                "source_checkpoint_row_sha256": sha256_value(source_checkpoint),
                "source_adjudication_row_sha256": sha256_value(source_adjudication),
                "source_evidence_identity_sha256": source_checkpoint[
                    "evidence_identity_sha256"
                ],
                "source_candidate_row_sha256": source_checkpoint["candidate_row_sha256"],
                "source_target_base_fact_row_sha256": source_checkpoint[
                    "target_base_fact_row_sha256"
                ],
                "source_source_base_fact_row_sha256": source_checkpoint[
                    "source_base_fact_row_sha256"
                ],
                "carry_depth": int(
                    (source_checkpoint.get("carry_forward_provenance") or {}).get(
                        "carry_depth", 0
                    )
                )
                + 1,
            }
            for field, value in expected_provenance.items():
                if provenance.get(field) != value:
                    raise ValueError(
                        f"Carry-forward provenance is stale: {candidate_id}.{field}"
                    )
            if source_checkpoint.get("evidence_identity_sha256") != row.get(
                "evidence_identity_sha256"
            ) or source_adjudication.get("overall_decision") != "accept":
                raise ValueError(f"Carry-forward source evidence is not an exact accept: {candidate_id}")
    elif carried:
        raise ValueError("Carried evidence has no source review manifest")
    elif reuse.get("mode") != "fresh_full_v3_baseline":
        raise ValueError("Fresh baseline evidence reuse mode is invalid")
    planned = reuse.get("planned_evidence_partition")
    if not isinstance(planned, dict) or any(
        planned.get(field) != manifest.get(field)
        for field in (
            "fresh_model_review_candidate_count",
            "carried_forward_candidate_count",
            "fresh_model_review_candidate_ids_sha256",
            "carried_forward_candidate_ids_sha256",
            "full_evidence_partition_complete",
        )
    ):
        raise ValueError("Planned evidence partition is stale")
    return {
        "manifest": manifest,
        "manifest_path": review_manifest_path,
        "run_contract": run_contract,
        "run_contract_sha256": str(manifest["run_contract_sha256"]),
        "review_semantics_contract": semantics,
        "review_semantics_contract_sha256": str(semantics_sha),
        "candidate_manifest_path": candidate_path,
        "units": units,
        "checkpoint_path": checkpoint_path,
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_by_id": checkpoint_by_id,
        "adjudications_path": adjudications_path,
        "adjudication_binding": adjudication_binding,
        "adjudication_by_id": adjudication_by_id,
        "source_bundle": source_bundle,
    }


def _carried_checkpoint_row(
    *,
    unit: ReviewUnit,
    source_bundle: Mapping[str, Any],
    run_contract_sha256: str,
    review_semantics_contract_sha256: str,
    now_fn: Callable[[], str],
) -> Dict[str, Any]:
    source_checkpoint = source_bundle["checkpoint_by_id"][unit.candidate_id]
    source_adjudication = source_bundle["adjudication_by_id"][unit.candidate_id]
    adjudication = copy.deepcopy(source_adjudication)
    adjudication.update(
        {
            "candidate_row_sha256": unit.candidate_row_sha256,
            "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
            "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
        }
    )
    source_provenance = source_checkpoint.get("carry_forward_provenance") or {}
    provenance = {
        "source_review_manifest_sha256": sha256_file(source_bundle["manifest_path"]),
        "source_run_contract_sha256": source_bundle["run_contract_sha256"],
        "source_checkpoint_artifact_sha256": source_bundle["checkpoint_binding"]["sha256"],
        "source_adjudications_artifact_sha256": source_bundle["adjudication_binding"]["sha256"],
        "source_checkpoint_row_sha256": sha256_value(source_checkpoint),
        "source_adjudication_row_sha256": sha256_value(source_adjudication),
        "source_evidence_identity_sha256": source_checkpoint["evidence_identity_sha256"],
        "source_candidate_row_sha256": source_checkpoint["candidate_row_sha256"],
        "source_target_base_fact_row_sha256": source_checkpoint[
            "target_base_fact_row_sha256"
        ],
        "source_source_base_fact_row_sha256": source_checkpoint[
            "source_base_fact_row_sha256"
        ],
        "carry_depth": int(source_provenance.get("carry_depth", 0)) + 1,
        "carried_forward_at": now_fn(),
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "candidate_id": unit.candidate_id,
        "candidate_kind": unit.candidate_kind,
        "input_index": unit.input_index,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "target_base_fact_id": unit.target_base_fact_id,
        "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
        "source_base_fact_id": unit.source_base_fact_id,
        "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
        "run_contract_sha256": run_contract_sha256,
        "review_semantics_contract_sha256": review_semantics_contract_sha256,
        "projection_sha256": unit_projection_sha256(unit),
        "evidence_identity_sha256": unit_evidence_identity_sha256(
            unit, review_semantics_contract_sha256
        ),
        "current_unit_binding_sha256": current_unit_binding_sha256(unit),
        "evidence_origin": CARRIED_EVIDENCE_ORIGIN,
        "model_calls_scope": "none_carried_forward",
        "carry_forward_provenance": provenance,
        "terminal_status": "completed",
        "failure_stage": None,
        "model_verdict": copy.deepcopy(source_checkpoint["model_verdict"]),
        "strict_adjudication": adjudication,
        "model_calls": [],
        "retry_history": [],
        "completed_at": provenance["carried_forward_at"],
        "behavior_blind": True,
        "human_gold": False,
        "hf_model_execution_count": 0,
        "hf_tokenizer_execution_count": 0,
        "behavior_execution_count": 0,
        "validation_behavior_exposure_count": 0,
        "sealed_behavior_exposure_count": 0,
    }


def _compact_runtime_artifacts(
    *,
    checkpoint_path: Path,
    journal_path: Path,
    adjudications_path: Path,
    checkpoint: Mapping[str, Mapping[str, Any]],
    units: Sequence[ReviewUnit],
    final: bool,
) -> None:
    rows = _ordered_checkpoint_rows(checkpoint, units)
    write_jsonl(checkpoint_path, rows)
    decisions = [
        row["strict_adjudication"]
        for row in rows
        if row.get("terminal_status") == "completed"
    ]
    write_jsonl(adjudications_path, decisions)
    if final:
        try:
            journal_path.unlink()
        except FileNotFoundError:
            pass
    else:
        write_jsonl(journal_path, [])


def run_review(
    *,
    candidate_manifest_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    env_path: Path,
    reviewer_spec: Mapping[str, Any],
    expected_response_model: str,
    full_base_facts_path: Optional[Path] = None,
    distractor_candidates_path: Optional[Path] = None,
    neutral_candidates_path: Optional[Path] = None,
    carry_forward_review_manifest_path: Optional[Path] = None,
    limit: Optional[int] = None,
    batch_size: int = 8,
    checkpoint_compact_every: int = 128,
    max_workers: Optional[int] = None,
    resume: bool = False,
    retry_failed: bool = False,
    router: Optional[Any] = None,
    now_fn: Callable[[], str] = utc_now,
) -> Dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if (
        isinstance(checkpoint_compact_every, bool)
        or not isinstance(checkpoint_compact_every, int)
        or checkpoint_compact_every < 1
    ):
        raise ValueError("checkpoint_compact_every must be positive")
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    execution = config.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    worker_count = int(
        max_workers if max_workers is not None else execution.get("max_workers", 1)
    )
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
    manifest, manifest_kind, bindings, units = load_review_units(
        candidate_manifest_path=candidate_manifest_path,
        full_base_facts_path=full_base_facts_path,
        distractor_candidates_path=distractor_candidates_path,
        neutral_candidates_path=neutral_candidates_path,
        limit=limit,
    )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "candidate_review_checkpoint.jsonl"
    journal_path = output_dir / "candidate_review_checkpoint.journal.jsonl"
    adjudications_path = output_dir / "candidate_semantic_adjudications.jsonl"
    run_manifest_path = output_dir / "candidate_review_run_manifest.json"
    event_log_path = output_dir / "redacted_model_events.jsonl"
    managed_paths = (
        checkpoint_path,
        journal_path,
        adjudications_path,
        run_manifest_path,
        event_log_path,
    )
    if not resume and any(path.exists() for path in managed_paths):
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

    semantics_contract = review_semantics_contract(
        reviewer_spec=reviewer_spec,
        expected_response_model=expected_response_model,
        protocol_identity=protocol_identity,
        route_identity=route_identity,
    )
    semantics_contract_sha256 = sha256_value(semantics_contract)
    direct_source = discover_direct_predecessor_review(
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest=manifest,
    )
    explicit_carry_path = (
        Path(carry_forward_review_manifest_path).resolve()
        if carry_forward_review_manifest_path is not None
        else None
    )
    if explicit_carry_path is not None and (
        direct_source is None or explicit_carry_path != direct_source["review_path"]
    ):
        raise ValueError(
            "Carry-forward review must be the direct predecessor bound by candidate resolution"
        )
    source_bundle: Optional[Dict[str, Any]] = None
    if direct_source is not None:
        source_manifest = direct_source["review_manifest"]
        source_is_current_protocol = source_manifest.get("tool_version") == TOOL_VERSION
        if explicit_carry_path is not None and not source_is_current_protocol:
            raise ValueError("Explicit carry-forward source is not a v3 review")
        if source_is_current_protocol:
            source_bundle = load_review_evidence_manifest(direct_source["review_path"])
            if (
                source_bundle["review_semantics_contract_sha256"]
                != semantics_contract_sha256
            ):
                raise ValueError("Direct predecessor review semantics contract changed")
            predecessor_binding = direct_source["predecessor_postreview_manifest"]
            predecessor_path = _resolve_artifact_path(
                manifest_path=candidate_manifest_path,
                binding=predecessor_binding,
                explicit_path=None,
                label="direct predecessor postreview manifest",
            )
            if source_bundle["candidate_manifest_path"] != predecessor_path:
                raise ValueError("Direct predecessor review binds a different candidate universe")
            if direct_source["resolution_adjudications"].get("sha256") != source_bundle[
                "adjudication_binding"
            ].get("sha256"):
                raise ValueError("Candidate resolution binds different source adjudications")
            if direct_source["resolution_checkpoint"].get("sha256") != source_bundle[
                "checkpoint_binding"
            ].get("sha256"):
                raise ValueError("Candidate resolution binds different source checkpoints")

    carry_units: List[ReviewUnit] = []
    fresh_units: List[ReviewUnit] = []
    if source_bundle is None:
        fresh_units = list(units)
    else:
        for unit in units:
            source_checkpoint = source_bundle["checkpoint_by_id"].get(unit.candidate_id)
            if source_checkpoint is None:
                fresh_units.append(unit)
                continue
            current_identity = unit_evidence_identity_sha256(
                unit, semantics_contract_sha256
            )
            if source_checkpoint.get("evidence_identity_sha256") != current_identity:
                fresh_units.append(unit)
                continue
            source_decision = source_bundle["adjudication_by_id"][unit.candidate_id].get(
                "overall_decision"
            )
            if source_decision != "accept":
                raise ValueError(
                    "Exact predecessor non-accept survived candidate resolution: "
                    f"{unit.candidate_id}"
                )
            carry_units.append(unit)

    carry_ids = [unit.candidate_id for unit in carry_units]
    fresh_ids = [unit.candidate_id for unit in fresh_units]
    planned_partition = {
        "fresh_model_review_candidate_count": len(fresh_ids),
        "carried_forward_candidate_count": len(carry_ids),
        "fresh_model_review_candidate_ids_sha256": sha256_value(fresh_ids),
        "carried_forward_candidate_ids_sha256": sha256_value(carry_ids),
        "full_evidence_partition_complete": len(fresh_ids) + len(carry_ids) == len(units),
    }
    if not planned_partition["full_evidence_partition_complete"] or set(fresh_ids).intersection(
        carry_ids
    ):
        raise ValueError("Planned evidence partition is not a disjoint full cover")

    selected_ids = [unit.candidate_id for unit in units]
    selected_by_kind = {
        kind: [unit.candidate_id for unit in units if unit.candidate_kind == kind]
        for kind in sorted(CANDIDATE_KINDS)
    }
    full_candidate_count = sum(
        int(bindings[label]["record_count"])
        for label in ("distractor_candidates", "neutral_reference_candidates")
    )
    run_contract = {
        "tool_version": TOOL_VERSION,
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "sha256": sha256_file(candidate_manifest_path),
            "schema_version": manifest.get("schema_version"),
            "manifest_kind": manifest_kind,
        },
        "input_artifacts": bindings,
        "selected_candidate_ids_sha256": sha256_value(selected_ids),
        "selected_candidate_count": len(units),
        "selected_candidate_ids_by_kind": {
            kind: {
                "record_count": len(ids),
                "ordered_ids_sha256": sha256_value(ids),
            }
            for kind, ids in selected_by_kind.items()
        },
        "full_candidate_count": full_candidate_count,
        "selection_is_full_candidate_set": len(units) == full_candidate_count,
        "config_sha256": sha256_value(config),
        "source_fingerprints": source_fingerprints(),
        "route_identity": route_identity,
        "reviewer_spec": reviewer_spec,
        "expected_response_model": expected_response_model,
        "protocol_reviewer_identity": protocol_identity,
        "prompt_contract_sha256": sha256_value(prompt_contract_payload()),
        "review_semantics_contract": semantics_contract,
        "review_semantics_contract_sha256": semantics_contract_sha256,
        "evidence_reuse_contract": {
            "schema_version": EVIDENCE_REUSE_CONTRACT_SCHEMA,
            "mode": (
                "direct_predecessor_exact_accept"
                if source_bundle is not None
                else "fresh_full_v3_baseline"
            ),
            "source_review_manifest": (
                {
                    "path": str(source_bundle["manifest_path"]),
                    "sha256": sha256_file(source_bundle["manifest_path"]),
                    "byte_count": source_bundle["manifest_path"].stat().st_size,
                    "schema_version": RUN_MANIFEST_SCHEMA,
                }
                if source_bundle is not None
                else None
            ),
            "source_run_contract_sha256": (
                source_bundle["run_contract_sha256"] if source_bundle is not None else None
            ),
            "direct_predecessor_lineage": (
                {
                    "candidate_resolution_manifest": direct_source[
                        "resolution_manifest"
                    ],
                    "predecessor_postreview_manifest": direct_source[
                        "predecessor_postreview_manifest"
                    ],
                }
                if source_bundle is not None and direct_source is not None
                else None
            ),
            "identity_schema": EVIDENCE_IDENTITY_SCHEMA,
            "only_prior_accept": True,
            "exact_nonaccept_policy": "fail_closed",
            "changed_or_new_policy": "fresh_model_review",
            "operator_rereview_override_allowed": False,
            "planned_evidence_partition": planned_partition,
        },
        "batch_size": batch_size,
        "checkpoint_compact_every": checkpoint_compact_every,
        "max_workers": worker_count,
        "answer_alias_contract": semantics_contract["answer_alias_contract"],
        "behavior_blind": True,
        "human_gold": False,
    }
    run_contract_sha256 = sha256_value(run_contract)
    if resume and run_manifest_path.is_file():
        previous = read_json(run_manifest_path)
        if previous.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(
                "Resume run contract mismatch; manifest, artifacts, selection, route, or config changed"
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
            "behavior_execution_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
            "review_freeze_performed": False,
            "split_freeze_performed": False,
            "candidate_promotion_performed": False,
        },
    }
    write_json(run_manifest_path, running_manifest)

    units_by_id = {unit.candidate_id: unit for unit in units}
    if resume:
        checkpoint, journal_rows, journal_needs_repair = _checkpoint_state(
            checkpoint_path=checkpoint_path,
            journal_path=journal_path,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha256,
            review_semantics_contract_sha256=semantics_contract_sha256,
        )
        if journal_needs_repair:
            write_jsonl(journal_path, journal_rows)
    else:
        checkpoint, journal_rows = {}, []
        write_jsonl(checkpoint_path, [])
        write_jsonl(adjudications_path, [])
        write_jsonl(journal_path, [])
    planned_origin = {
        **{unit.candidate_id: FRESH_EVIDENCE_ORIGIN for unit in fresh_units},
        **{unit.candidate_id: CARRIED_EVIDENCE_ORIGIN for unit in carry_units},
    }
    for candidate_id, row in checkpoint.items():
        if row.get("evidence_origin") != planned_origin.get(candidate_id):
            raise ValueError(f"Checkpoint evidence partition is stale: {candidate_id}")
    newly_carried = [
        _carried_checkpoint_row(
            unit=unit,
            source_bundle=source_bundle,
            run_contract_sha256=run_contract_sha256,
            review_semantics_contract_sha256=semantics_contract_sha256,
            now_fn=now_fn,
        )
        for unit in carry_units
        if unit.candidate_id not in checkpoint
    ] if source_bundle is not None else []
    if newly_carried:
        _append_journal(journal_path, newly_carried)
        journal_rows.extend(newly_carried)
        checkpoint.update(
            {str(row["candidate_id"]): row for row in newly_carried}
        )
    pending = [
        unit
        for unit in fresh_units
        if unit.candidate_id not in checkpoint
        or (
            retry_failed
            and checkpoint[unit.candidate_id].get("terminal_status") == "failed"
        )
    ]
    if pending and hasattr(router, "_event"):
        event_lock = threading.Lock()
        original_event = router._event

        def locked_event(*args: Any, **kwargs: Any) -> Any:
            with event_lock:
                return original_event(*args, **kwargs)

        router._event = locked_event

    journaled_since_compact = len(journal_rows)
    batches = list(_chunks(pending, batch_size))
    if batches:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_batch,
                    units=batch,
                    router=router,
                    reviewer_spec=reviewer_spec,
                    expected_response_model=expected_response_model,
                    run_contract_sha256=run_contract_sha256,
                    review_semantics_contract_sha256=semantics_contract_sha256,
                    now_fn=now_fn,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                committed_rows = sorted(
                    future.result(), key=lambda row: int(row["input_index"])
                )
                for row in committed_rows:
                    candidate_id = str(row["candidate_id"])
                    previous = checkpoint.get(candidate_id)
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
                # Worker threads never write checkpoints.  This coordinator is
                # the single durable writer.
                _append_journal(journal_path, committed_rows)
                for row in committed_rows:
                    checkpoint[str(row["candidate_id"])] = row
                journaled_since_compact += len(committed_rows)
                if journaled_since_compact >= checkpoint_compact_every:
                    _compact_runtime_artifacts(
                        checkpoint_path=checkpoint_path,
                        journal_path=journal_path,
                        adjudications_path=adjudications_path,
                        checkpoint=checkpoint,
                        units=units,
                        final=False,
                    )
                    journaled_since_compact = 0

    _compact_runtime_artifacts(
        checkpoint_path=checkpoint_path,
        journal_path=journal_path,
        adjudications_path=adjudications_path,
        checkpoint=checkpoint,
        units=units,
        final=True,
    )
    ordered_rows = _ordered_checkpoint_rows(checkpoint, units)
    status_counts = Counter(str(row["terminal_status"]) for row in ordered_rows)
    decisions = [
        row["strict_adjudication"]
        for row in ordered_rows
        if row.get("terminal_status") == "completed"
    ]
    overall_counts = Counter(str(row["overall_decision"]) for row in decisions)
    kind_counts = Counter(str(row["candidate_kind"]) for row in decisions)
    accepted_by_kind = Counter(
        str(row["candidate_kind"])
        for row in decisions
        if row["overall_decision"] == "accept"
    )
    evidence_origin_counts = Counter(
        str(row.get("evidence_origin")) for row in ordered_rows
    )
    actual_fresh_ids = [
        str(row["candidate_id"])
        for row in ordered_rows
        if row.get("evidence_origin") == FRESH_EVIDENCE_ORIGIN
    ]
    actual_carry_ids = [
        str(row["candidate_id"])
        for row in ordered_rows
        if row.get("evidence_origin") == CARRIED_EVIDENCE_ORIGIN
    ]
    all_selected_completed = status_counts.get("completed", 0) == len(units)
    selection_is_full = run_contract["selection_is_full_candidate_set"]
    final_manifest = {
        **running_manifest,
        "status": "completed" if all_selected_completed else "completed_with_failures",
        "completed_at": now_fn(),
        "selected_candidate_count": len(units),
        "full_candidate_count": full_candidate_count,
        "selection_is_full_candidate_set": selection_is_full,
        "selected_candidate_ids_sha256": run_contract[
            "selected_candidate_ids_sha256"
        ],
        "selected_candidate_ids_by_kind": run_contract[
            "selected_candidate_ids_by_kind"
        ],
        "batch_size": batch_size,
        "checkpoint_compact_every": checkpoint_compact_every,
        "max_workers": worker_count,
        "resume_enabled": resume,
        "retry_failed_enabled": retry_failed,
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "completed_kind_counts": dict(sorted(kind_counts.items())),
        "accepted_kind_counts": dict(sorted(accepted_by_kind.items())),
        "overall_decision_counts": dict(sorted(overall_counts.items())),
        "evidence_origin_counts": dict(sorted(evidence_origin_counts.items())),
        "fresh_model_review_candidate_count": len(actual_fresh_ids),
        "carried_forward_candidate_count": len(actual_carry_ids),
        "fresh_model_review_candidate_ids_sha256": sha256_value(actual_fresh_ids),
        "carried_forward_candidate_ids_sha256": sha256_value(actual_carry_ids),
        "full_evidence_partition_complete": (
            len(actual_fresh_ids) + len(actual_carry_ids) == len(units)
            and not set(actual_fresh_ids).intersection(actual_carry_ids)
            and {
                **planned_partition,
            }
            == {
                "fresh_model_review_candidate_count": len(actual_fresh_ids),
                "carried_forward_candidate_count": len(actual_carry_ids),
                "fresh_model_review_candidate_ids_sha256": sha256_value(
                    actual_fresh_ids
                ),
                "carried_forward_candidate_ids_sha256": sha256_value(
                    actual_carry_ids
                ),
                "full_evidence_partition_complete": True,
            }
        ),
        "candidate_review_complete_for_selected_scope": all_selected_completed,
        "candidate_review_complete_for_full_set": all_selected_completed and selection_is_full,
        "formal_review_complete": False,
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "byte_count": checkpoint_path.stat().st_size,
                "record_count": len(ordered_rows),
                "schema_version": CHECKPOINT_SCHEMA,
            },
            "adjudications": {
                "path": str(adjudications_path),
                "sha256": sha256_file(adjudications_path),
                "byte_count": adjudications_path.stat().st_size,
                "record_count": len(decisions),
                "schema_version": ADJUDICATION_SCHEMA,
            },
        },
        "execution_policy": {
            "one_model_request_per_batch": True,
            "real_batch_size": batch_size,
            "max_workers": worker_count,
            "recursive_binary_split_on_batch_failure": True,
            "response_model_identity_fail_closed": True,
            "contradictory_alias_evidence_fail_closed": True,
            "answer_alias_disjoint_uses_target_aliases_only": True,
            "single_writer_append_journal_with_atomic_compaction": True,
            "journal_fsync_after_each_commit": True,
            "incomplete_final_journal_line_ignored_on_resume": True,
            "terminal_journal_removed": not journal_path.exists(),
            "resume_supported": True,
            "retry_failed_supported": True,
            "monotone_exact_accept_carry_forward": True,
            "exact_prior_accept_rereview_forbidden": True,
            "exact_prior_nonaccept_survival_fails_closed": True,
        },
        "limitations": {
            "proxy_review_only": True,
            "human_gold": False,
            "candidate_rows_promoted": False,
            "review_freeze_emitted": False,
            "split_freeze_emitted": False,
            "pre_hf_compatibility_input_is_not_final_postreview_input": (
                manifest_kind == "pre_hf_schema_compatibility"
            ),
        },
    }
    if not final_manifest["full_evidence_partition_complete"]:
        raise ValueError("Final evidence partition differs from the run contract")
    _validate_evidence_partition(final_manifest, ordered_rows, units)
    write_json(run_manifest_path, final_manifest)
    return {
        "status": final_manifest["status"],
        "manifest_path": str(run_manifest_path),
        "manifest_sha256": sha256_file(run_manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "adjudications_path": str(adjudications_path),
        "selected_candidate_count": len(units),
        "terminal_status_counts": final_manifest["terminal_status_counts"],
        "overall_decision_counts": final_manifest["overall_decision_counts"],
        "fresh_model_review_candidate_count": final_manifest[
            "fresh_model_review_candidate_count"
        ],
        "carried_forward_candidate_count": final_manifest[
            "carried_forward_candidate_count"
        ],
        "formal_review_complete": False,
    }


def execute(
    args: argparse.Namespace,
    *,
    router_factory: Optional[Callable[..., Any]] = None,
    now_fn: Callable[[], str] = utc_now,
) -> Dict[str, Any]:
    config = load_config(Path(args.config))
    reviewer_spec = resolve_reviewer_spec(
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
        full_base_facts_path=(
            Path(args.full_base_facts) if args.full_base_facts else None
        ),
        distractor_candidates_path=(
            Path(args.distractor_candidates) if args.distractor_candidates else None
        ),
        neutral_candidates_path=(
            Path(args.neutral_candidates) if args.neutral_candidates else None
        ),
        carry_forward_review_manifest_path=(
            Path(args.carry_forward_review_manifest)
            if args.carry_forward_review_manifest
            else None
        ),
        output_dir=Path(args.output_dir),
        config=config,
        env_path=Path(args.env_file),
        reviewer_spec=reviewer_spec,
        expected_response_model=expected_response_model,
        limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_compact_every=args.checkpoint_compact_every,
        max_workers=args.max_workers,
        resume=args.resume,
        retry_failed=args.retry_failed,
        router=router,
        now_fn=now_fn,
    )


def build_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate-manifest", type=Path, required=True)
    result.add_argument("--full-base-facts", type=Path)
    result.add_argument("--distractor-candidates", type=Path)
    result.add_argument("--neutral-candidates", type=Path)
    result.add_argument(
        "--carry-forward-review-manifest",
        type=Path,
        help=(
            "Optional assertion of the direct predecessor review. Compatible v3 "
            "predecessors are auto-discovered and cannot be bypassed."
        ),
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    result.add_argument("--reviewer-provider-profile", default="openai")
    result.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    result.add_argument("--reviewer-reasoning-effort", default="low")
    result.add_argument("--reviewer-max-output-tokens", type=int, default=16384)
    result.add_argument(
        "--expected-response-model", default=DEFAULT_EXPECTED_RESPONSE_MODEL
    )
    result.add_argument("--max-retries", type=int)
    result.add_argument("--limit", type=int)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--checkpoint-compact-every", type=int, default=128)
    result.add_argument("--max-workers", type=int)
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
