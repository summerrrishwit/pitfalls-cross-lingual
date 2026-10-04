#!/usr/bin/env python3
"""Run resumable dual-model resolution for non-accept full-pool fact reviews.

The input is the manifest produced by
``resolve_full_public_benchmark_fact_review.py materialize-template``.  A Qwen
proposal stage recommends either a field-level revision or current-cohort
exclusion, and an independent GPT reviewer makes the final proxy decision.

Source-review ``revise`` rows and narrowly eligible target-only ``reject`` rows
may be retained through ``apply_revision``.  The current schema does not allow
``defer`` or ineligible ``reject`` rows to be revised, so those rows are
emitted as explicit current-cohort exclusions.
All evidence remains model-proxy evidence with ``human_gold=false``.

The runner stores no prompts, raw responses, credentials, or endpoints.  It
uses an append-only, single-writer journal between atomic compactions and can
resume or retry terminal failures.  A full successful run writes a decisions
JSONL that is strictly consumable by the resolver's ``apply`` command.  A
limited/selected pilot writes schema-valid rows but intentionally lacks full
apply coverage until every materialized resolution item is processed.
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

TOOL_VERSION = "full-public-benchmark-fact-resolution-runner-v2"
CHECKPOINT_SCHEMA_VERSION = "full-public-benchmark-fact-resolution-checkpoint-v2"
RUN_MANIFEST_SCHEMA_VERSION = "full-public-benchmark-fact-resolution-run-manifest-v2"
RUN_CONTRACT_VERSION = "full-public-benchmark-fact-resolution-run-contract-v2"
PROPOSAL_RESPONSE_SCHEMA_VERSION = "full-public-benchmark-fact-resolution-proposal-batch-v2"
REVIEW_RESPONSE_SCHEMA_VERSION = "full-public-benchmark-fact-resolution-review-batch-v2"
PROJECTION_SCHEMA_VERSION = "full-public-benchmark-fact-resolution-projection-v1"
PROPOSER_PROMPT_VERSION = "full-public-benchmark-fact-resolution-proposer-v2"
REVIEWER_PROMPT_VERSION = "full-public-benchmark-fact-resolution-reviewer-v2"
REVISION_EDIT_METHOD = "dual_model_field_level_fact_resolution_proposal_v2"
REVISION_REVIEW_METHOD = "independent_field_level_fact_resolution_rereview_v2"
EXCLUSION_REVIEW_METHOD = "independent_current_cohort_exclusion_resolution_v2"
RESPONSE_IDENTITY_ENFORCEMENT = (
    "runner_exact_post_response_without_static_proxy_guard_v2"
)
SOURCE_REVIEWER_IDENTITY_FALLBACK_REASON = (
    "A target-only repair was proposed, but the configured resolution reviewer "
    "is the same identity as the source reject reviewer; independent reject-override "
    "rereview is therefore unavailable."
)

DEFAULT_PROPOSER_ROUTE = "qwen3.8-max-0902/bailian/bailian"
DEFAULT_PROPOSER_RESPONSE_MODEL = "qwen3.8-max"
DEFAULT_REVIEWER_MODEL = "gpt-5.5"
DEFAULT_REVIEWER_RESPONSE_MODEL = "gpt-5.5-2026-04-24"

# Loaded lazily so imports, offline validation, and unit tests need no SDK.
ModelRouter: Optional[Any] = None

PROPOSER_INSTRUCTIONS = """You propose resolutions for non-accept factual reviews.
Use only the supplied source snapshots, extracted facts, review evidence, and current mutable
values. Never use target-model behavior, split assignment, HF output, or Simulation output.
For a source outcome of revise, prefer apply_revision when an alias-only cleanup or another
source-grounded extraction correction can make the row reliable. A reject may use apply_revision
only when its supplied allowed_dispositions explicitly permits it; that is a narrowly derived
target-only override, not permission to reverse a core fact or relation rejection. Change only the
allowed mutable fields and keep all fields mutually consistent. If no verifiable correction is
possible, use cohort_exclude. A defer must be cohort_exclude. Cohort exclusion means only exclusion
from this pre-exact-HF cohort and does not assert that the underlying fact is false. Return JSON
only."""

REVIEWER_INSTRUCTIONS = """You independently resolve non-accept factual reviews.
Re-evaluate the supplied evidence yourself; the Qwen proposal is non-authoritative. Never use
target-model behavior, split assignment, HF output, or Simulation output. A revise row should use
apply_revision when an alias-only cleanup or source-grounded extraction correction is reliable.
Every retained revision must be internally consistent and receive an explicit accept review for
all mutable fields, including unchanged fields. Accept a Qwen revision only by repeating its exact
updates; otherwise use cohort_exclude. A reject may be revised only when allowed_dispositions says
so and the supplied eligibility record proves a target-only override. A defer must be
cohort_exclude. Exclusion is only from the current cohort and is not a factual-falsehood judgment.
Return JSON only."""


def _load_resolver() -> Any:
    path = Path(__file__).resolve().with_name(
        "resolve_full_public_benchmark_fact_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_full_fact_resolution_apply_contract", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load resolution contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()


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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


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


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not have surrounding whitespace")
    return value


MODEL_SPEC_OUTPUT_FIELDS = (
    "provider_profile",
    "model",
    "expected_response_model",
    "temperature",
    "max_output_tokens",
    "reasoning_effort",
    "disable_thinking",
    "json_mode",
    "timeout_seconds",
    "max_retries",
)


def _safe_model_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    output = {
        field: spec[field]
        for field in MODEL_SPEC_OUTPUT_FIELDS
        if field in spec and spec[field] is not None
    }
    _required_string(output.get("provider_profile"), "model provider_profile")
    _required_string(output.get("model"), "model name")
    _required_string(
        output.get("expected_response_model"), "model expected_response_model"
    )
    return output


def _inferred_expected_response_model(requested_model: Any) -> Optional[str]:
    requested = str(requested_model or "").strip()
    return {
        DEFAULT_PROPOSER_ROUTE: DEFAULT_PROPOSER_RESPONSE_MODEL,
        DEFAULT_REVIEWER_MODEL: DEFAULT_REVIEWER_RESPONSE_MODEL,
    }.get(requested)


def _runtime_source_bindings() -> Dict[str, Dict[str, str]]:
    """Bind every local implementation that defines this run contract."""

    sources = {
        "resolution_runner": Path(__file__).resolve(),
        "resolution_contract": Path(resolver.__file__).resolve(),
        "model_router": PROJECT_ROOT / "factual_pitfalls" / "perturbation.py",
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }


def _verify_runtime_sources_unchanged(
    bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    for label, binding in bindings.items():
        path = Path(_required_string(binding.get("path"), f"{label}.path")).resolve()
        expected_sha = _required_string(binding.get("sha256"), f"{label}.sha256")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"Bound runtime source changed during resolution run: {label}")


def _route_fingerprints(
    router: Any,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Hash effective routes without persisting endpoints or credentials."""

    from factual_pitfalls import perturbation as runtime

    output: Dict[str, Dict[str, Any]] = {}
    for role, spec in (
        ("proposer", proposer_spec),
        ("reviewer", reviewer_spec),
    ):
        route = runtime._static_proxy_route_identity_record(router, dict(spec))
        fingerprint = {
            "provider_profile": route["provider_profile"],
            "protocol": route["protocol"],
            "requested_model": route["model"],
            "expected_response_model": spec["expected_response_model"],
            "base_url_sha256": route["base_url_sha256"],
            "request_route_sha256": route["request_route_sha256"],
            "credentials_or_endpoints_included": False,
        }
        fingerprint["route_fingerprint_sha256"] = sha256_value(fingerprint)
        output[role] = fingerprint
    return output


def _verify_routes_unchanged(
    router: Any,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if _route_fingerprints(router, proposer_spec, reviewer_spec) != expected:
        raise ValueError("Effective model route changed during resolution run")


def _validate_role_identity_contract(
    proposer_spec: Mapping[str, Any], reviewer_spec: Mapping[str, Any]
) -> None:
    if (
        proposer_spec["provider_profile"],
        proposer_spec["model"],
    ) == (
        reviewer_spec["provider_profile"],
        reviewer_spec["model"],
    ):
        raise ValueError("proposer and reviewer must use distinct model identities")
    if (
        proposer_spec["expected_response_model"]
        == reviewer_spec["expected_response_model"]
    ):
        raise ValueError(
            "proposer and reviewer must use distinct expected response models"
        )


def _role_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_roles = config.get("model_roles")
    model_roles = model_roles if isinstance(model_roles, dict) else {}
    candidates = (
        config.get("fact_resolution"),
        config.get("full_public_benchmark_fact_resolution"),
        model_roles.get("fact_resolution"),
        model_roles.get("full_pool_fact_resolution"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    # The current zh-first configuration already carries the intended Qwen
    # primary and GPT fact-review identities.  Reuse those identities without
    # persisting provider endpoints or credentials.
    translation = model_roles.get("translation")
    fact_review = model_roles.get("full_pool_fact_review")
    return {
        "proposer": (
            dict(translation.get("primary") or {})
            if isinstance(translation, dict)
            else {}
        ),
        "reviewer": (
            dict(fact_review.get("reviewer") or {})
            if isinstance(fact_review, dict)
            else (
                dict(translation.get("reviewer") or {})
                if isinstance(translation, dict)
                else {}
            )
        ),
    }


def resolve_model_specs(
    config: Mapping[str, Any],
    *,
    proposer_provider_profile: Optional[str] = None,
    proposer_model: Optional[str] = None,
    reviewer_provider_profile: Optional[str] = None,
    reviewer_model: Optional[str] = None,
    proposer_expected_response_model: Optional[str] = None,
    reviewer_expected_response_model: Optional[str] = None,
    proposer_reasoning_effort: Optional[str] = None,
    reviewer_reasoning_effort: Optional[str] = None,
    proposer_max_output_tokens: Optional[int] = None,
    reviewer_max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    role = _role_config(config)
    proposer = dict(role.get("proposer") or role.get("generator") or {})
    reviewer = dict(role.get("reviewer") or {})
    execution = config.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    overrides = (
        (
            proposer,
            proposer_provider_profile,
            proposer_model,
            proposer_expected_response_model,
            proposer_reasoning_effort,
            proposer_max_output_tokens,
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
        spec.setdefault("max_output_tokens", 4096)
        spec.setdefault("temperature", 0.0)
        spec["json_mode"] = True
        # The shared router's static-proxy guard compares the requested route
        # name to response.model.  That is not valid for alias routes such as
        # qwen3.8-max-0902/bailian/bailian -> qwen3.8-max.  This runner checks
        # the configured expected_response_model exactly after every response.
        spec.pop("require_response_model_identity", None)
        spec.setdefault(
            "expected_response_model",
            _inferred_expected_response_model(spec.get("model")),
        )
        retries = max_retries
        if retries is None:
            retries = spec.get("max_retries", execution.get("max_retries", 2))
        if not isinstance(retries, int) or retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        spec["max_retries"] = retries
    proposer_safe = _safe_model_spec(proposer)
    reviewer_safe = _safe_model_spec(reviewer)
    _validate_role_identity_contract(proposer_safe, reviewer_safe)
    return proposer_safe, reviewer_safe


class ResolutionUnit(NamedTuple):
    input_index: int
    base_fact_id: str
    resolution_item_sha256: str
    decision_template_sha256: str
    source_full_fact_row_sha256: str
    item: Dict[str, Any]
    template: Dict[str, Any]
    source_row: Dict[str, Any]
    projection: Dict[str, Any]


def _source_snapshot_projection(triple: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = triple.get("source_snapshot")
    record = snapshot.get("record") if isinstance(snapshot, dict) else None
    record = record if isinstance(record, dict) else {}
    return {
        "source": record.get("source", triple.get("source_dataset")),
        "source_format": record.get("source_format", triple.get("source_format")),
        "upstream_id": record.get("upstream_id"),
        "upstream_revision": record.get("upstream_revision"),
        "question": record.get("question", triple.get("source_question")),
        "answer": record.get("answer", triple.get("source_answer")),
        "choices": record.get("choices", triple.get("source_choices", [])),
    }


def project_resolution_item(
    item: Mapping[str, Any], review_item: Mapping[str, Any]
) -> Dict[str, Any]:
    """Whitelist source evidence; never copy behavior, split, or HF fields."""

    members: List[Dict[str, Any]] = []
    for member in review_item.get("members", []):
        if not isinstance(member, dict) or not isinstance(member.get("triple"), dict):
            raise ValueError(f"Invalid review member for {item.get('base_fact_id')}")
        triple = member["triple"]
        members.append(
            {
                "candidate_id": member.get("candidate_id"),
                "triple_record_sha256": member.get("triple_record_sha256"),
                "extracted_triple": {
                    "subject": triple.get("subject"),
                    "subject_type": triple.get("subject_type"),
                    "relation_raw": triple.get("relation_raw"),
                    "answer": triple.get("answer"),
                    "answer_type": triple.get("answer_type"),
                    "canonical_fact": triple.get("canonical_fact"),
                    "answer_aliases_en": triple.get("answer_aliases_en", []),
                },
                "source_snapshot": _source_snapshot_projection(triple),
            }
        )
    targets = review_item.get("review_targets")
    if not isinstance(targets, dict):
        raise ValueError(f"Invalid review targets for {item.get('base_fact_id')}")
    aliases = targets.get("aliases")
    distractors = targets.get("distractors")
    if not isinstance(aliases, list) or not isinstance(distractors, list):
        raise ValueError(f"Invalid review targets for {item.get('base_fact_id')}")
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "base_fact_id": item.get("base_fact_id"),
        "source_review_outcome": item.get("source_review_outcome"),
        "allowed_dispositions": copy.deepcopy(item.get("allowed_dispositions")),
        "mutable_fields": list(resolver.MUTABLE_FIELDS),
        "source_values": copy.deepcopy(item.get("source_values")),
        "source_review_evidence": copy.deepcopy(item.get("source_review_evidence")),
        "source_reject_override_eligibility": copy.deepcopy(
            item.get("source_reject_override_eligibility")
        ),
        "members": members,
        "alias_targets": [
            {"alias_id": target.get("alias_id"), "text_en": target.get("text_en")}
            for target in aliases
        ],
        "distractor_targets": [
            {
                "distractor_id": target.get("distractor_id"),
                "text_en": target.get("text_en"),
                "source_choice_index": target.get("source_choice_index"),
            }
            for target in distractors
        ],
        "cohort_exclusion_scope": resolver.EXCLUSION_SCOPE,
        "human_gold": False,
    }


def _selected_ids(path: Path) -> List[str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("base_fact_ids")
        if not isinstance(value, list):
            raise ValueError("Selected-ID JSON must be an array or contain base_fact_ids")
        raw = value
    else:
        raw = []
        for row in iter_jsonl(path):
            raw.append(row.get("base_fact_id", row.get("source_base_fact_id")))
    ids = [_required_string(value, "selected base_fact_id") for value in raw]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected base_fact_ids must be unique")
    if not ids:
        raise ValueError("Selected base_fact_ids must not be empty")
    return ids


def load_resolution_units(
    *,
    template_manifest_path: Path,
    base_fact_ids_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[ResolutionUnit], Optional[Dict[str, Any]]]:
    if base_fact_ids_path is not None and limit is not None:
        raise ValueError("base_fact_ids_path and limit are mutually exclusive")
    context = resolver._load_template_context(Path(template_manifest_path).resolve())
    items = list(context["items"])
    templates = list(context["templates"])
    if len(items) != len(templates):
        raise ValueError("Resolution item/template count mismatch")
    item_by_id = {str(item["base_fact_id"]): item for item in items}
    template_by_id = {str(row["base_fact_id"]): row for row in templates}
    if len(item_by_id) != len(items) or len(template_by_id) != len(templates):
        raise ValueError("Resolution items/templates contain duplicate base_fact_id")
    if set(item_by_id) != set(template_by_id):
        raise ValueError("Resolution item/template ID coverage mismatch")
    all_ids = [str(item["base_fact_id"]) for item in items]
    selection_binding: Optional[Dict[str, Any]] = None
    if base_fact_ids_path is not None:
        selected = _selected_ids(base_fact_ids_path)
        missing = sorted(set(selected) - set(item_by_id))
        if missing:
            raise ValueError(f"Selected IDs are not resolution items: {missing[:10]}")
        selection_binding = {
            "path": str(Path(base_fact_ids_path).resolve()),
            "sha256": sha256_file(base_fact_ids_path),
            "record_count": len(selected),
        }
    else:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        selected = all_ids if limit is None else all_ids[: min(limit, len(all_ids))]

    units: List[ResolutionUnit] = []
    for input_index, base_fact_id in enumerate(selected):
        item = item_by_id[base_fact_id]
        template = template_by_id[base_fact_id]
        if item.get("schema_version") != resolver.RESOLUTION_ITEM_SCHEMA:
            raise ValueError(f"Unsupported resolution item schema: {base_fact_id}")
        if item.get("source_review_outcome") not in resolver.NONACCEPT_OUTCOMES:
            raise ValueError(f"Resolution runner received an accept item: {base_fact_id}")
        reject_eligible = bool(
            isinstance(item.get("source_reject_override_eligibility"), dict)
            and item["source_reject_override_eligibility"].get("eligible") is True
        )
        expected_allowed = (
            ["apply_revision", "cohort_exclude"]
            if item["source_review_outcome"] == "revise"
            or (item["source_review_outcome"] == "reject" and reject_eligible)
            else ["cohort_exclude"]
        )
        if item.get("allowed_dispositions") != expected_allowed:
            raise ValueError(f"Resolution disposition contract drifted: {base_fact_id}")
        expected_template = resolver._decision_template(item)
        if template != expected_template:
            raise ValueError(f"Stale resolution decision template: {base_fact_id}")
        source_row = context["chain"]["full_index"][base_fact_id]
        source_sha = sha256_value(source_row)
        if item.get("source_bindings", {}).get("full_fact_row_sha256") != source_sha:
            raise ValueError(f"Stale source full-fact binding: {base_fact_id}")
        review_item = context["chain"]["item_index"][base_fact_id]
        units.append(
            ResolutionUnit(
                input_index=input_index,
                base_fact_id=base_fact_id,
                resolution_item_sha256=sha256_value(item),
                decision_template_sha256=sha256_value(template),
                source_full_fact_row_sha256=source_sha,
                item=copy.deepcopy(item),
                template=copy.deepcopy(template),
                source_row=copy.deepcopy(source_row),
                projection=project_resolution_item(item, review_item),
            )
        )
    return context, units, selection_binding


def _prompt_payload(prompt: str) -> Dict[str, Any]:
    marker = "INPUT_JSON:\n"
    return json.loads(prompt.split(marker, 1)[1])


def _response_contract(*, reviewer: bool) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "base_fact_id": "copy the exact input ID",
        "disposition": ["apply_revision", "cohort_exclude"],
        "updates": "non-empty allowed-field object for apply_revision; null otherwise",
        "revision_reason": "non-empty for apply_revision; null otherwise",
        "cohort_exclusion_reason": "non-empty for cohort_exclude; null otherwise",
        "rationale": "non-empty evidence-grounded rationale",
    }
    if reviewer:
        item["field_reviews"] = {
            field: {
                "decision": "accept",
                "reason": "non-empty verification reason for the final revised row",
            }
            for field in resolver.MUTABLE_FIELDS
        }
    return {
        "top_level": {
            "schema_version": (
                REVIEW_RESPONSE_SCHEMA_VERSION
                if reviewer
                else PROPOSAL_RESPONSE_SCHEMA_VERSION
            ),
            "items": "exactly one result for every input base_fact_id",
        },
        "item": item,
        "hard_rules": {
            "defer_disposition": "cohort_exclude",
            "reject_disposition": (
                "apply_revision only when explicitly listed in allowed_dispositions; "
                "otherwise cohort_exclude"
            ),
            "cohort_exclude_field_reviews": None,
            "cohort_exclusion_asserts_factual_falsehood": False,
            "human_gold": False,
        },
    }


def build_proposer_prompt(units: Sequence[ResolutionUnit]) -> str:
    payload = {
        "response_contract": _response_contract(reviewer=False),
        "items": [unit.projection for unit in units],
    }
    return f"{PROPOSER_INSTRUCTIONS}\nINPUT_JSON:\n{canonical_json_bytes(payload).decode('utf-8')}"


def build_reviewer_prompt(
    units: Sequence[ResolutionUnit], proposals: Mapping[str, Mapping[str, Any]]
) -> str:
    payload = {
        "response_contract": _response_contract(reviewer=True),
        "items": [
            {
                "evidence": unit.projection,
                "qwen_proposal": proposals[unit.base_fact_id],
            }
            for unit in units
        ],
    }
    return f"{REVIEWER_INSTRUCTIONS}\nINPUT_JSON:\n{canonical_json_bytes(payload).decode('utf-8')}"


def _validate_updates(value: Any, unit: ResolutionUnit) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Revision updates must be an object: {unit.base_fact_id}")
    if not value and unit.item["source_review_outcome"] != "reject":
        raise ValueError(f"Revision updates must be non-empty: {unit.base_fact_id}")
    unsupported = sorted(set(value) - set(resolver.MUTABLE_FIELDS))
    if unsupported:
        raise ValueError(
            f"Revision updates immutable fields for {unit.base_fact_id}: {unsupported}"
        )
    updates = copy.deepcopy(value)
    for field, update in updates.items():
        if canonical_json_bytes(update) == canonical_json_bytes(
            unit.source_row.get(field)
        ):
            raise ValueError(f"Revision does not change {field}: {unit.base_fact_id}")
        if field in resolver.STRING_MUTABLE_FIELDS:
            _required_string(update, f"revision {unit.base_fact_id}.{field}")
    if "answer_en" in updates and "answer_aliases_en" not in updates:
        raise ValueError(
            f"Revision changing answer_en must update answer_aliases_en: {unit.base_fact_id}"
        )
    revised = copy.deepcopy(unit.source_row)
    revised.update(updates)
    resolver._validate_aliases(
        revised.get("answer_aliases_en"),
        answer=str(revised["answer_en"]),
        label=f"revised {unit.base_fact_id}.answer_aliases_en",
    )
    if unit.item["source_review_outcome"] == "reject":
        eligibility = unit.item.get("source_reject_override_eligibility")
        if not isinstance(eligibility, dict) or eligibility.get("eligible") is not True:
            raise ValueError(
                f"Source reject is not eligible for revision override: {unit.base_fact_id}"
            )
        rejected_alias_ids = eligibility.get("target_resolutions", {}).get(
            "rejected_alias_ids", []
        )
        if rejected_alias_ids and "answer_aliases_en" not in updates:
            raise ValueError(
                f"Reject override must edit answer_aliases_en: {unit.base_fact_id}"
            )
        alias_targets = {
            str(entry.get("alias_id")): str(entry.get("text_en"))
            for entry in unit.item.get("source_review_evidence", {}).get(
                "alias_reviews", []
            )
            if isinstance(entry, dict)
            and isinstance(entry.get("alias_id"), str)
            and isinstance(entry.get("text_en"), str)
        }
        rejected_alias_text = {
            resolver.normalize_text(alias_targets[alias_id])
            for alias_id in rejected_alias_ids
            if alias_id in alias_targets
        }
        if rejected_alias_text.intersection(
            resolver.normalize_text(alias) for alias in revised["answer_aliases_en"]
        ):
            raise ValueError(
                f"Reject override retains a rejected answer alias: {unit.base_fact_id}"
            )
    return updates


def _validate_model_item(
    row: Any, unit: ResolutionUnit, *, reviewer: bool
) -> Dict[str, Any]:
    base_fields = {
        "base_fact_id",
        "disposition",
        "updates",
        "revision_reason",
        "cohort_exclusion_reason",
        "rationale",
    }
    expected_fields = base_fields | ({"field_reviews"} if reviewer else set())
    if not isinstance(row, dict) or set(row) != expected_fields:
        raise ValueError(f"Resolution response fields are invalid: {unit.base_fact_id}")
    if row.get("base_fact_id") != unit.base_fact_id:
        raise ValueError(f"Resolution response ID mismatch: {unit.base_fact_id}")
    disposition = row.get("disposition")
    if disposition not in unit.item["allowed_dispositions"]:
        raise ValueError(f"Resolution disposition is invalid: {unit.base_fact_id}")
    _required_string(row.get("rationale"), f"{unit.base_fact_id}.rationale")
    output = copy.deepcopy(row)
    if disposition == "apply_revision":
        if disposition not in unit.item["allowed_dispositions"]:
            raise ValueError(f"Revision is not allowed: {unit.base_fact_id}")
        output["updates"] = _validate_updates(row.get("updates"), unit)
        _required_string(
            row.get("revision_reason"), f"{unit.base_fact_id}.revision_reason"
        )
        if row.get("cohort_exclusion_reason") is not None:
            raise ValueError(
                f"Revision cannot include cohort exclusion: {unit.base_fact_id}"
            )
        if reviewer:
            field_reviews = row.get("field_reviews")
            if not isinstance(field_reviews, dict) or set(field_reviews) != set(
                resolver.MUTABLE_FIELDS
            ):
                raise ValueError(
                    f"Revision field_reviews are incomplete: {unit.base_fact_id}"
                )
            for field in resolver.MUTABLE_FIELDS:
                review = field_reviews[field]
                if not isinstance(review, dict) or set(review) != {
                    "decision",
                    "reason",
                }:
                    raise ValueError(
                        f"Revision field review fields are invalid: {unit.base_fact_id}.{field}"
                    )
                if review.get("decision") != "accept":
                    raise ValueError(
                        f"Revision field review must accept {field}: {unit.base_fact_id}"
                    )
                _required_string(
                    review.get("reason"),
                    f"{unit.base_fact_id}.field_reviews.{field}.reason",
                )
    else:
        if row.get("updates") is not None or row.get("revision_reason") is not None:
            raise ValueError(
                f"Cohort exclusion cannot include revision fields: {unit.base_fact_id}"
            )
        _required_string(
            row.get("cohort_exclusion_reason"),
            f"{unit.base_fact_id}.cohort_exclusion_reason",
        )
        if reviewer and row.get("field_reviews") is not None:
            raise ValueError(
                f"Cohort exclusion field_reviews must be null: {unit.base_fact_id}"
            )
    return output


def validate_batch_response(
    value: Dict[str, Any], units: Sequence[ResolutionUnit], *, reviewer: bool
) -> None:
    if set(value) != {"schema_version", "items"}:
        raise ValueError("Batch response must contain exactly schema_version and items")
    expected_schema = (
        REVIEW_RESPONSE_SCHEMA_VERSION if reviewer else PROPOSAL_RESPONSE_SCHEMA_VERSION
    )
    if value.get("schema_version") != expected_schema:
        raise ValueError("Unsupported batch response schema_version")
    rows = value.get("items")
    if not isinstance(rows, list):
        raise ValueError("Batch response items must be a list")
    expected = {unit.base_fact_id: unit for unit in units}
    found: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Batch response item must be an object")
        base_fact_id = _required_string(row.get("base_fact_id"), "base_fact_id")
        if base_fact_id not in expected:
            raise ValueError(f"Unknown response base_fact_id: {base_fact_id}")
        if base_fact_id in found:
            raise ValueError(f"Duplicate response base_fact_id: {base_fact_id}")
        found[base_fact_id] = _validate_model_item(
            row, expected[base_fact_id], reviewer=reviewer
        )
    if set(found) != set(expected):
        raise ValueError(
            f"Batch response coverage mismatch: expected={sorted(expected)}, "
            f"actual={sorted(found)}"
        )


def _response_map(value: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["base_fact_id"]): copy.deepcopy(row) for row in value["items"]}


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
    stage: str,
    prompt_version: str,
    units: Sequence[ResolutionUnit],
    model_spec: Mapping[str, Any],
    prompt: str,
    reviewer: bool,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    base_fact_ids = [unit.base_fact_id for unit in units]
    batch_id = f"batch_{sha256_value(base_fact_ids)[:16]}"
    safe_spec = _safe_model_spec(model_spec)
    expected_response_model = safe_spec["expected_response_model"]
    request_spec = dict(safe_spec)
    # Identity belongs to this protocol's exact expected-response contract.
    # Never invoke the shared requested-route compatibility guard here.
    request_spec.pop("require_response_model_identity", None)
    request_binding = {
        "stage": stage,
        "batch_id": batch_id,
        "ordered_base_fact_ids": base_fact_ids,
        "model_spec": safe_spec,
        "prompt": prompt,
    }
    try:
        result = router.request_json(
            stage,
            batch_id,
            request_spec,
            prompt,
            lambda value: validate_batch_response(value, units, reviewer=reviewer),
        )
        raw_response = result.raw_response if isinstance(result.raw_response, str) else None
        response_model_observed = (
            isinstance(result.response_model, str)
            and bool(result.response_model.strip())
        )
        identity_status = (
            "not_observed"
            if not response_model_observed
            else (
                "matched"
                if result.response_model == expected_response_model
                else "mismatch"
            )
        )
        terminal_status = result.terminal_status
        if terminal_status == "completed" and identity_status != "matched":
            terminal_status = "failed"
        record = {
            "stage": stage,
            "prompt_version": prompt_version,
            "batch_id": batch_id,
            "ordered_base_fact_ids": base_fact_ids,
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
            "latency_ms": result.latency_ms,
            "attempt_count": result.attempt_count,
            "attempts": _safe_attempts(result.attempts),
            "credentials_or_endpoints_included": False,
        }
        if result.terminal_status == "completed" and identity_status != "matched":
            record["post_validation_error_type"] = "ResponseModelIdentityError"
            return None, record
        if result.terminal_status != "completed" or not isinstance(
            result.parsed_response, dict
        ):
            return None, record
        try:
            validate_batch_response(result.parsed_response, units, reviewer=reviewer)
        except Exception as error:
            record["terminal_status"] = "failed"
            record["post_validation_error_type"] = type(error).__name__
            return None, record
        return _response_map(result.parsed_response), record
    except Exception as error:
        return None, {
            "stage": stage,
            "prompt_version": prompt_version,
            "batch_id": batch_id,
            "ordered_base_fact_ids": base_fact_ids,
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
            "latency_ms": None,
            "attempt_count": 0,
            "attempts": [],
            **_exception_record(error),
            "credentials_or_endpoints_included": False,
        }


def _should_split(record: Mapping[str, Any], unit_count: int) -> bool:
    if unit_count <= 1:
        return False
    error_types = {
        str(entry.get("error_type"))
        for entry in record.get("attempts", [])
        if isinstance(entry, dict) and entry.get("error_type")
    }
    error_type = record.get("error_type") or record.get("post_validation_error_type")
    if error_type:
        error_types.add(str(error_type))
    return not bool(
        error_types.intersection(
            {
                "AuthenticationError",
                "PermissionDeniedError",
                "StaticProxyIdentityError",
                "ResponseModelIdentityError",
            }
        )
    )


def _strict_resolution(
    unit: ResolutionUnit,
    proposal: Mapping[str, Any],
    verdict: Mapping[str, Any],
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    proposer_response_model: Optional[str],
    reviewer_response_model: Optional[str],
    reviewed_at: str,
) -> Dict[str, Any]:
    output = copy.deepcopy(unit.template)
    disposition = str(verdict["disposition"])
    output["disposition"] = disposition
    output["human_gold"] = False
    proposer_response_model = _required_string(
        proposer_response_model, "proposer response model"
    )
    reviewer_response_model = _required_string(
        reviewer_response_model, "reviewer response model"
    )
    if proposer_response_model == reviewer_response_model:
        raise ValueError(f"Resolution model identities are not independent: {unit.base_fact_id}")
    proposer_id = f"{proposer_spec['provider_profile']}:{proposer_response_model}"
    reviewer_id = f"{reviewer_spec['provider_profile']}:{reviewer_response_model}"
    source_reviewer_id = unit.item.get("source_review_evidence", {}).get(
        "review_provenance", {}
    ).get("reviewer_id")
    forced_exclusion_reason: Optional[str] = None
    if (
        disposition == "apply_revision"
        and unit.item["source_review_outcome"] == "reject"
        and source_reviewer_id == reviewer_id
    ):
        # The apply schema requires a reject-override rereviewer distinct from
        # the original reject reviewer.  Do not fake independence by changing
        # an identifier; retain the evidence and fail closed to cohort scope.
        disposition = "cohort_exclude"
        output["disposition"] = disposition
        forced_exclusion_reason = SOURCE_REVIEWER_IDENTITY_FALLBACK_REASON
    if disposition == "apply_revision":
        if proposal.get("disposition") != "apply_revision" or canonical_json_bytes(
            proposal.get("updates")
        ) != canonical_json_bytes(verdict.get("updates")):
            raise ValueError(
                f"Reviewer cannot invent or alter proposer updates: {unit.base_fact_id}"
            )
        updates = _validate_updates(verdict["updates"], unit)
        revised = copy.deepcopy(unit.source_row)
        revised.update(copy.deepcopy(updates))
        output["revision"] = {
            "updates": updates,
            "revision_reason": proposal["revision_reason"],
            "source_reject_override": (
                {
                    "original_review_outcome": "reject",
                    "eligibility_evidence_sha256": sha256_value(
                        unit.item["source_reject_override_eligibility"]
                    ),
                    "override_reason": verdict["rationale"],
                    "target_resolutions": copy.deepcopy(
                        unit.item["source_reject_override_eligibility"][
                            "target_resolutions"
                        ]
                    ),
                }
                if unit.item["source_review_outcome"] == "reject"
                else None
            ),
            "editor_provenance": {
                "editor_type": "codex_proxy",
                "editor_id": proposer_id,
                "edit_method": REVISION_EDIT_METHOD,
                "edited_at": reviewed_at,
                "human_gold": False,
            },
            "rereview": {
                "decision": "accept",
                "revised_full_fact_row_sha256": sha256_value(revised),
                "field_reviews": copy.deepcopy(verdict["field_reviews"]),
                "rationale": verdict["rationale"],
                "reviewer_type": "independent_model_proxy",
                "reviewer_id": reviewer_id,
                "review_method": REVISION_REVIEW_METHOD,
                "reviewed_at": reviewed_at,
                "human_gold": False,
            },
        }
        output["cohort_exclusion"] = None
        resolver._validate_revision(
            output["revision"],
            base_fact_id=unit.base_fact_id,
            source_row=unit.source_row,
            resolution_item=unit.item,
        )
    else:
        output["revision"] = None
        output["cohort_exclusion"] = {
            "reason": forced_exclusion_reason or verdict["cohort_exclusion_reason"],
            "scope": resolver.EXCLUSION_SCOPE,
            "factual_falsehood_asserted": False,
            "resolution_provenance": {
                "reviewer_type": "independent_model_proxy",
                "reviewer_id": reviewer_id,
                "review_method": EXCLUSION_REVIEW_METHOD,
                "reviewed_at": reviewed_at,
                "human_gold": False,
            },
        }
        resolver._validate_exclusion(
            output["cohort_exclusion"], base_fact_id=unit.base_fact_id
        )
    _validate_strict_resolution(output, unit)
    return output


def _validate_strict_resolution(
    decision: Mapping[str, Any], unit: ResolutionUnit
) -> None:
    if set(decision) != resolver.DECISION_FIELDS:
        raise ValueError(f"Strict resolution fields drifted: {unit.base_fact_id}")
    if decision.get("schema_version") != resolver.RESOLUTION_DECISION_SCHEMA:
        raise ValueError(f"Strict resolution schema drifted: {unit.base_fact_id}")
    if decision.get("base_fact_id") != unit.base_fact_id:
        raise ValueError(f"Strict resolution ID drifted: {unit.base_fact_id}")
    if decision.get("resolution_item_sha256") != unit.resolution_item_sha256:
        raise ValueError(f"Strict resolution item SHA is stale: {unit.base_fact_id}")
    if decision.get("human_gold") is not False:
        raise ValueError(f"Strict resolution must have human_gold=false: {unit.base_fact_id}")
    disposition = decision.get("disposition")
    if disposition not in unit.item["allowed_dispositions"]:
        raise ValueError(f"Strict resolution disposition is invalid: {unit.base_fact_id}")
    if disposition == "apply_revision":
        if decision.get("cohort_exclusion") is not None:
            raise ValueError(f"Revision also contains exclusion: {unit.base_fact_id}")
        resolver._validate_revision(
            decision.get("revision"),
            base_fact_id=unit.base_fact_id,
            source_row=unit.source_row,
            resolution_item=unit.item,
        )
    else:
        if decision.get("revision") is not None:
            raise ValueError(f"Exclusion also contains revision: {unit.base_fact_id}")
        resolver._validate_exclusion(
            decision.get("cohort_exclusion"), base_fact_id=unit.base_fact_id
        )


def _failed_rows(
    units: Sequence[ResolutionUnit],
    traces: Sequence[Mapping[str, Any]],
    failure_stage: str,
    run_contract_sha256: str,
    now_fn: Callable[[], str],
) -> List[Dict[str, Any]]:
    return [
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "base_fact_id": unit.base_fact_id,
            "input_index": unit.input_index,
            "resolution_item_sha256": unit.resolution_item_sha256,
            "decision_template_sha256": unit.decision_template_sha256,
            "source_full_fact_row_sha256": unit.source_full_fact_row_sha256,
            "run_contract_sha256": run_contract_sha256,
            "terminal_status": "failed",
            "failure_stage": failure_stage,
            "proposer_verdict": None,
            "reviewer_verdict": None,
            "strict_resolution": None,
            "model_calls": [dict(trace) for trace in traces],
            "retry_history": [],
            "completed_at": now_fn(),
            "source_evidence_only": True,
            "human_gold": False,
        }
        for unit in units
    ]


def _review_subbatch(
    *,
    units: Sequence[ResolutionUnit],
    proposals: Mapping[str, Mapping[str, Any]],
    proposer_response_models: Mapping[str, Optional[str]],
    inherited_traces: Sequence[Mapping[str, Any]],
    router: Any,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    run_contract_sha256: str,
    now_fn: Callable[[], str],
) -> List[Dict[str, Any]]:
    prompt = build_reviewer_prompt(units, proposals)
    verdicts, call = _call_model(
        router=router,
        stage="fact_resolution_reviewer",
        prompt_version=REVIEWER_PROMPT_VERSION,
        units=units,
        model_spec=reviewer_spec,
        prompt=prompt,
        reviewer=True,
    )
    traces = [*inherited_traces, call]
    if verdicts is None:
        if _should_split(call, len(units)):
            midpoint = len(units) // 2
            return [
                *_review_subbatch(
                    units=units[:midpoint],
                    proposals=proposals,
                    proposer_response_models=proposer_response_models,
                    inherited_traces=traces,
                    router=router,
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                ),
                *_review_subbatch(
                    units=units[midpoint:],
                    proposals=proposals,
                    proposer_response_models=proposer_response_models,
                    inherited_traces=traces,
                    router=router,
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                ),
            ]
        return _failed_rows(units, traces, "reviewer", run_contract_sha256, now_fn)

    reviewer_response_model = call.get("response_model")
    rows: List[Dict[str, Any]] = []
    for unit in units:
        reviewed_at = now_fn()
        verdict = verdicts[unit.base_fact_id]
        try:
            resolution = _strict_resolution(
                unit,
                proposals[unit.base_fact_id],
                verdict,
                proposer_spec,
                reviewer_spec,
                proposer_response_models[unit.base_fact_id],
                reviewer_response_model,
                reviewed_at,
            )
        except Exception as error:
            invalid_call = dict(call)
            invalid_call["terminal_status"] = "failed"
            invalid_call["post_validation_error_type"] = type(error).__name__
            failed = _failed_rows(
                [unit],
                [*inherited_traces, invalid_call],
                "strict_resolution_validation",
                run_contract_sha256,
                now_fn,
            )[0]
            failed["proposer_verdict"] = copy.deepcopy(
                proposals[unit.base_fact_id]
            )
            failed["reviewer_verdict"] = copy.deepcopy(verdict)
            rows.append(failed)
            continue
        rows.append(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "tool_version": TOOL_VERSION,
                "base_fact_id": unit.base_fact_id,
                "input_index": unit.input_index,
                "resolution_item_sha256": unit.resolution_item_sha256,
                "decision_template_sha256": unit.decision_template_sha256,
                "source_full_fact_row_sha256": unit.source_full_fact_row_sha256,
                "run_contract_sha256": run_contract_sha256,
                "terminal_status": "completed",
                "failure_stage": None,
                "proposer_verdict": copy.deepcopy(proposals[unit.base_fact_id]),
                "reviewer_verdict": copy.deepcopy(verdict),
                "strict_resolution": resolution,
                "model_calls": [dict(trace) for trace in traces],
                "retry_history": [],
                "completed_at": reviewed_at,
                "source_evidence_only": True,
                "human_gold": False,
            }
        )
    return rows


def _process_batch(
    *,
    units: Sequence[ResolutionUnit],
    router: Any,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    run_contract_sha256: str,
    now_fn: Callable[[], str],
    inherited_traces: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    prompt = build_proposer_prompt(units)
    proposals, call = _call_model(
        router=router,
        stage="fact_resolution_proposer",
        prompt_version=PROPOSER_PROMPT_VERSION,
        units=units,
        model_spec=proposer_spec,
        prompt=prompt,
        reviewer=False,
    )
    traces = [*inherited_traces, call]
    if proposals is None:
        if _should_split(call, len(units)):
            midpoint = len(units) // 2
            return [
                *_process_batch(
                    units=units[:midpoint],
                    router=router,
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
                *_process_batch(
                    units=units[midpoint:],
                    router=router,
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
            ]
        return _failed_rows(units, traces, "proposer", run_contract_sha256, now_fn)
    response_model = call.get("response_model")
    response_models = {unit.base_fact_id: response_model for unit in units}
    return _review_subbatch(
        units=units,
        proposals=proposals,
        proposer_response_models=response_models,
        inherited_traces=traces,
        router=router,
        proposer_spec=proposer_spec,
        reviewer_spec=reviewer_spec,
        run_contract_sha256=run_contract_sha256,
        now_fn=now_fn,
    )


def _chunks(values: Sequence[ResolutionUnit], size: int) -> Iterator[List[ResolutionUnit]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "tool_version",
        "base_fact_id",
        "input_index",
        "resolution_item_sha256",
        "decision_template_sha256",
        "source_full_fact_row_sha256",
        "run_contract_sha256",
        "terminal_status",
        "failure_stage",
        "proposer_verdict",
        "reviewer_verdict",
        "strict_resolution",
        "model_calls",
        "retry_history",
        "completed_at",
        "source_evidence_only",
        "human_gold",
    }
)


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    units_by_id: Mapping[str, ResolutionUnit],
    run_contract_sha256: str,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    base_fact_id = _required_string(row.get("base_fact_id"), "checkpoint.base_fact_id")
    unit = units_by_id.get(base_fact_id)
    if unit is None:
        raise ValueError(f"Checkpoint contains an unselected item: {base_fact_id}")
    if set(row) != CHECKPOINT_FIELDS:
        raise ValueError(f"Checkpoint fields are invalid: {base_fact_id}")
    if row.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {base_fact_id}")
    if row.get("tool_version") != TOOL_VERSION:
        raise ValueError(f"Checkpoint tool version is stale: {base_fact_id}")
    if row.get("input_index") != unit.input_index:
        raise ValueError(f"Checkpoint input index is stale: {base_fact_id}")
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError(f"Checkpoint run contract is stale: {base_fact_id}")
    if row.get("resolution_item_sha256") != unit.resolution_item_sha256:
        raise ValueError(f"Checkpoint resolution item is stale: {base_fact_id}")
    if row.get("decision_template_sha256") != unit.decision_template_sha256:
        raise ValueError(f"Checkpoint decision template is stale: {base_fact_id}")
    if row.get("source_full_fact_row_sha256") != unit.source_full_fact_row_sha256:
        raise ValueError(f"Checkpoint source full-fact row is stale: {base_fact_id}")
    if row.get("terminal_status") not in {"completed", "failed"}:
        raise ValueError(f"Checkpoint terminal status is invalid: {base_fact_id}")
    if row.get("human_gold") is not False or row.get("source_evidence_only") is not True:
        raise ValueError(f"Checkpoint safety contract is invalid: {base_fact_id}")
    calls = row.get("model_calls")
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise ValueError(f"Checkpoint model_calls are invalid: {base_fact_id}")
    if any(
        any(key in call for key in ("prompt", "raw_response", "api_key", "base_url"))
        for call in calls
    ):
        raise ValueError(f"Checkpoint contains sensitive model-call fields: {base_fact_id}")
    if row["terminal_status"] == "completed":
        if not isinstance(row.get("proposer_verdict"), dict) or not isinstance(
            row.get("reviewer_verdict"), dict
        ):
            raise ValueError(f"Completed checkpoint lacks verdicts: {base_fact_id}")
        validate_batch_response(
            {
                "schema_version": PROPOSAL_RESPONSE_SCHEMA_VERSION,
                "items": [row["proposer_verdict"]],
            },
            [unit],
            reviewer=False,
        )
        validate_batch_response(
            {
                "schema_version": REVIEW_RESPONSE_SCHEMA_VERSION,
                "items": [row["reviewer_verdict"]],
            },
            [unit],
            reviewer=True,
        )
        decision = row.get("strict_resolution")
        if not isinstance(decision, dict):
            raise ValueError(f"Completed checkpoint has no strict resolution: {base_fact_id}")
        _validate_strict_resolution(decision, unit)
        completed_response_models: List[str] = []
        expected_specs = {
            "fact_resolution_proposer": proposer_spec,
            "fact_resolution_reviewer": reviewer_spec,
        }
        for stage, expected_spec in expected_specs.items():
            matching_calls = [
                call
                for call in calls
                if call.get("stage") == stage
                and call.get("terminal_status") == "completed"
                and call.get("response_model_identity_status") == "matched"
                and call.get("provider_profile")
                == expected_spec["provider_profile"]
                and call.get("requested_model") == expected_spec["model"]
                and call.get("expected_response_model")
                == expected_spec["expected_response_model"]
                and call.get("response_model")
                == expected_spec["expected_response_model"]
            ]
            if len(matching_calls) != 1:
                raise ValueError(
                    f"Completed checkpoint lacks a matched {stage} call: {base_fact_id}"
                )
            completed_response_models.append(
                str(matching_calls[0]["response_model"])
            )
        if len(set(completed_response_models)) != 2:
            raise ValueError(
                f"Completed checkpoint reuses one response identity: {base_fact_id}"
            )
    elif row.get("strict_resolution") is not None:
        raise ValueError(f"Failed checkpoint has a strict resolution: {base_fact_id}")
    return base_fact_id, copy.deepcopy(dict(row))


def _read_journal(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """Read all durable rows and ignore only an interrupted final fragment."""

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
            raise ValueError(
                f"Malformed checkpoint journal line {index + 1}: {path}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected JSON object in checkpoint journal line {index + 1}"
            )
        rows.append(value)
        if is_last and not terminated:
            needs_repair = True
    return rows, needs_repair


def _checkpoint_state(
    checkpoint_path: Path,
    journal_path: Path,
    *,
    units_by_id: Mapping[str, ResolutionUnit],
    run_contract_sha256: str,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], bool]:
    output: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.is_file():
        for row in read_jsonl(checkpoint_path, allow_empty=True):
            base_fact_id, validated = _validate_checkpoint_row(
                row,
                units_by_id=units_by_id,
                run_contract_sha256=run_contract_sha256,
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
            )
            if base_fact_id in output:
                raise ValueError(f"Duplicate compact checkpoint ID: {base_fact_id}")
            output[base_fact_id] = validated
    journal_rows, needs_repair = _read_journal(journal_path)
    for row in journal_rows:
        base_fact_id, validated = _validate_checkpoint_row(
            row,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha256,
            proposer_spec=proposer_spec,
            reviewer_spec=reviewer_spec,
        )
        output[base_fact_id] = validated
    return output, journal_rows, needs_repair


def _ordered_checkpoint_rows(
    checkpoint: Mapping[str, Mapping[str, Any]], units: Sequence[ResolutionUnit]
) -> List[Dict[str, Any]]:
    return [
        copy.deepcopy(dict(checkpoint[unit.base_fact_id]))
        for unit in units
        if unit.base_fact_id in checkpoint
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


def _compact_runtime_artifacts(
    *,
    checkpoint_path: Path,
    journal_path: Path,
    decisions_path: Path,
    checkpoint: Mapping[str, Mapping[str, Any]],
    units: Sequence[ResolutionUnit],
    final: bool,
) -> None:
    rows = _ordered_checkpoint_rows(checkpoint, units)
    write_jsonl(checkpoint_path, rows)
    decisions = [
        row["strict_resolution"]
        for row in rows
        if row.get("terminal_status") == "completed"
    ]
    write_jsonl(decisions_path, decisions)
    if final:
        try:
            journal_path.unlink()
        except FileNotFoundError:
            pass
    else:
        write_jsonl(journal_path, [])


def _assert_run_inputs_current(
    *,
    context: Mapping[str, Any],
    template_manifest_path: Path,
    template_manifest_sha256: str,
    resolution_items_sha256: str,
    decision_template_sha256: str,
    selection_binding: Optional[Mapping[str, Any]],
) -> None:
    resolver._assert_snapshots_current(context["chain"]["snapshots"])
    checks = (
        (template_manifest_path, template_manifest_sha256, "template manifest"),
        (context["items_path"], resolution_items_sha256, "resolution items"),
        (context["templates_path"], decision_template_sha256, "decision template"),
    )
    for path, expected_sha, label in checks:
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{label} changed during resolution run")
    if selection_binding is not None:
        selection_path = Path(str(selection_binding["path"])).resolve()
        if sha256_file(selection_path) != selection_binding["sha256"]:
            raise ValueError("Selected-ID artifact changed during resolution run")


def run_resolution(
    *,
    template_manifest_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    env_path: Path,
    proposer_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    base_fact_ids_path: Optional[Path] = None,
    limit: Optional[int] = None,
    batch_size: int = 4,
    checkpoint_compact_every: int = 128,
    max_workers: Optional[int] = None,
    resume: bool = True,
    retry_failed: bool = False,
    router: Optional[Any] = None,
    now_fn: Callable[[], str] = utc_now,
) -> Dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if checkpoint_compact_every <= 0:
        raise ValueError("checkpoint_compact_every must be positive")
    execution = config.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    worker_count = int(max_workers or execution.get("max_workers", 4))
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    proposer_spec = _safe_model_spec(proposer_spec)
    reviewer_spec = _safe_model_spec(reviewer_spec)
    _validate_role_identity_contract(proposer_spec, reviewer_spec)
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("config.provider_profiles must be an object")
    for label, spec in (("proposer", proposer_spec), ("reviewer", reviewer_spec)):
        if spec["provider_profile"] not in profiles:
            raise ValueError(f"Unknown {label} provider_profile: {spec['provider_profile']}")
    template_manifest_path = Path(template_manifest_path).resolve()
    context, units, selection_binding = load_resolution_units(
        template_manifest_path=template_manifest_path,
        base_fact_ids_path=base_fact_ids_path,
        limit=limit,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fact_resolution_checkpoint.jsonl"
    journal_path = output_dir / "fact_resolution_checkpoint.journal.jsonl"
    decisions_path = output_dir / "fact_resolutions_codex_proxy.jsonl"
    run_manifest_path = output_dir / "fact_resolution_run_manifest.json"
    event_log_path = output_dir / "redacted_model_events.jsonl"
    if router is None:
        router = _load_model_router_class()(
            dict(config), Path(env_path).resolve(), event_log_path
        )

    selected_ids = [unit.base_fact_id for unit in units]
    all_ids = [str(item["base_fact_id"]) for item in context["items"]]
    selected_ids_sha = sha256_value(selected_ids)
    selection_is_full = len(selected_ids) == len(all_ids) and set(selected_ids) == set(
        all_ids
    )
    template_manifest_sha = sha256_file(template_manifest_path)
    resolution_items_sha = sha256_file(context["items_path"])
    decision_template_sha = sha256_file(context["templates_path"])
    runtime_sources = _runtime_source_bindings()
    route_fingerprints = _route_fingerprints(
        router, proposer_spec, reviewer_spec
    )
    endpoint_identities_distinct = (
        route_fingerprints["proposer"]["base_url_sha256"]
        != route_fingerprints["reviewer"]["base_url_sha256"]
    )
    prompt_contract_sha = sha256_value(
        {
            "proposer_prompt_version": PROPOSER_PROMPT_VERSION,
            "proposer_instructions": PROPOSER_INSTRUCTIONS,
            "reviewer_prompt_version": REVIEWER_PROMPT_VERSION,
            "reviewer_instructions": REVIEWER_INSTRUCTIONS,
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "proposal_response_schema_version": PROPOSAL_RESPONSE_SCHEMA_VERSION,
            "review_response_schema_version": REVIEW_RESPONSE_SCHEMA_VERSION,
        }
    )
    run_contract: Dict[str, Any] = {
        "run_contract_version": RUN_CONTRACT_VERSION,
        "tool_version": TOOL_VERSION,
        "runtime_sources": runtime_sources,
        "route_fingerprints": route_fingerprints,
        "template_manifest_sha256": template_manifest_sha,
        "resolution_items_sha256": resolution_items_sha,
        "decision_template_sha256": decision_template_sha,
        "config_sha256": sha256_value(config),
        "proposer_model_spec": proposer_spec,
        "reviewer_model_spec": reviewer_spec,
        "prompt_contract_sha256": prompt_contract_sha,
        "resolution_methods": {
            "revision_edit_method": REVISION_EDIT_METHOD,
            "revision_review_method": REVISION_REVIEW_METHOD,
            "exclusion_review_method": EXCLUSION_REVIEW_METHOD,
        },
        "response_identity_enforcement": RESPONSE_IDENTITY_ENFORCEMENT,
        "batch_size": batch_size,
        "selected_base_fact_ids_sha256": selected_ids_sha,
        "selected_count": len(units),
        "full_resolution_item_count": len(all_ids),
        "selection_is_full_template": selection_is_full,
        "selected_ids_artifact": copy.deepcopy(selection_binding),
        "source_evidence_only": True,
        "human_gold": False,
    }
    run_contract_sha = sha256_value(run_contract)
    units_by_id = {unit.base_fact_id: unit for unit in units}
    managed_paths = (
        checkpoint_path,
        journal_path,
        decisions_path,
        run_manifest_path,
        event_log_path,
    )
    if not resume and any(path.exists() for path in managed_paths):
        raise FileExistsError(
            "run artifacts already exist; use a new output directory or enable resume"
        )
    if resume and run_manifest_path.is_file():
        previous_manifest = read_json(run_manifest_path)
        if previous_manifest.get("run_contract_sha256") != run_contract_sha:
            raise ValueError("resume run contract mismatch")
    _assert_run_inputs_current(
        context=context,
        template_manifest_path=template_manifest_path,
        template_manifest_sha256=template_manifest_sha,
        resolution_items_sha256=resolution_items_sha,
        decision_template_sha256=decision_template_sha,
        selection_binding=selection_binding,
    )
    if resume:
        checkpoint, journal_rows, journal_needs_repair = _checkpoint_state(
            checkpoint_path,
            journal_path,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha,
            proposer_spec=proposer_spec,
            reviewer_spec=reviewer_spec,
        )
        if journal_needs_repair:
            write_jsonl(journal_path, journal_rows)
    else:
        checkpoint, journal_rows = {}, []
        write_jsonl(checkpoint_path, [])
        write_jsonl(decisions_path, [])
        write_jsonl(journal_path, [])

    pending = [
        unit
        for unit in units
        if unit.base_fact_id not in checkpoint
        or (
            retry_failed
            and checkpoint[unit.base_fact_id].get("terminal_status") == "failed"
        )
    ]
    if pending and hasattr(router, "_event"):
        event_lock = threading.Lock()
        original_event = router._event

        def locked_event(*args: Any, **kwargs: Any) -> Any:
            with event_lock:
                return original_event(*args, **kwargs)

        router._event = locked_event

    batches = list(_chunks(pending, batch_size))
    journaled_since_compact = len(journal_rows)
    if batches:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_batch,
                    units=batch,
                    router=router,
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha,
                    now_fn=now_fn,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                committed_rows = sorted(
                    future.result(), key=lambda row: int(row["input_index"])
                )
                for row in committed_rows:
                    previous = checkpoint.get(str(row["base_fact_id"]))
                    if previous is not None and previous.get("terminal_status") == "failed":
                        prior_history = list(previous.get("retry_history", []))
                        prior_history.append(
                            {
                                "terminal_status": "failed",
                                "failure_stage": previous.get("failure_stage"),
                                "model_calls": list(previous.get("model_calls", [])),
                                "completed_at": previous.get("completed_at"),
                            }
                        )
                        row["retry_history"] = prior_history
                # Only the coordinator writes durable state.
                _append_journal(journal_path, committed_rows)
                for row in committed_rows:
                    checkpoint[str(row["base_fact_id"])] = row
                journaled_since_compact += len(committed_rows)
                if journaled_since_compact >= checkpoint_compact_every:
                    _compact_runtime_artifacts(
                        checkpoint_path=checkpoint_path,
                        journal_path=journal_path,
                        decisions_path=decisions_path,
                        checkpoint=checkpoint,
                        units=units,
                        final=False,
                    )
                    journaled_since_compact = 0

    _assert_run_inputs_current(
        context=context,
        template_manifest_path=template_manifest_path,
        template_manifest_sha256=template_manifest_sha,
        resolution_items_sha256=resolution_items_sha,
        decision_template_sha256=decision_template_sha,
        selection_binding=selection_binding,
    )
    _verify_runtime_sources_unchanged(runtime_sources)
    _verify_routes_unchanged(
        router,
        proposer_spec,
        reviewer_spec,
        route_fingerprints,
    )
    _compact_runtime_artifacts(
        checkpoint_path=checkpoint_path,
        journal_path=journal_path,
        decisions_path=decisions_path,
        checkpoint=checkpoint,
        units=units,
        final=True,
    )
    ordered_rows = _ordered_checkpoint_rows(checkpoint, units)
    status_counts = Counter(str(row["terminal_status"]) for row in ordered_rows)
    decisions = [
        row["strict_resolution"]
        for row in ordered_rows
        if row.get("terminal_status") == "completed"
    ]
    disposition_counts = Counter(str(row["disposition"]) for row in decisions)
    source_reviewer_identity_fallback_count = sum(
        1
        for row in decisions
        if row.get("disposition") == "cohort_exclude"
        and row.get("cohort_exclusion", {}).get("reason")
        == SOURCE_REVIEWER_IDENTITY_FALLBACK_REASON
    )
    all_completed = status_counts.get("completed", 0) == len(units)
    strict_apply_ready = selection_is_full and all_completed
    if strict_apply_ready:
        # Exercise the exact downstream loader before advertising this file as
        # complete input for ``resolve ... apply``.
        resolver._load_resolution_decisions(decisions_path, context=context)
    manifest_output = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": "completed" if all_completed else "completed_with_failures",
        "template_manifest": {
            "path": str(template_manifest_path),
            "sha256": sha256_file(template_manifest_path),
            "schema_version": resolver.TEMPLATE_MANIFEST_SCHEMA,
        },
        "resolution_items": {
            "path": str(context["items_path"]),
            "sha256": sha256_file(context["items_path"]),
            "schema_version": resolver.RESOLUTION_ITEM_SCHEMA,
        },
        "decision_template": {
            "path": str(context["templates_path"]),
            "sha256": sha256_file(context["templates_path"]),
            "schema_version": resolver.RESOLUTION_DECISION_SCHEMA,
        },
        "selection": {
            "selected_base_fact_ids_sha256": selected_ids_sha,
            "selected_count": len(units),
            "full_resolution_item_count": len(all_ids),
            "selection_is_full_template": selection_is_full,
            "selected_ids_artifact": copy.deepcopy(selection_binding),
        },
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha,
        "batch_size": batch_size,
        "checkpoint_compact_every": checkpoint_compact_every,
        "max_workers": worker_count,
        "resume_enabled": resume,
        "retry_failed_enabled": retry_failed,
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "source_reviewer_identity_fallback_count": (
            source_reviewer_identity_fallback_count
        ),
        "strict_resolution_apply_ready": strict_apply_ready,
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "record_count": len(ordered_rows),
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
            },
            "resolutions": {
                "path": str(decisions_path),
                "sha256": sha256_file(decisions_path),
                "record_count": len(decisions),
                "schema_version": resolver.RESOLUTION_DECISION_SCHEMA,
                "strict_apply_coverage_complete": strict_apply_ready,
            },
        },
        "model_roles": {
            "proposer": proposer_spec,
            "reviewer": reviewer_spec,
            "provider_profiles_distinct": (
                proposer_spec["provider_profile"]
                != reviewer_spec["provider_profile"]
            ),
            "requested_model_identities_distinct": (
                proposer_spec["model"] != reviewer_spec["model"]
            ),
            "expected_response_model_identities_distinct": True,
            "effective_endpoint_identities_distinct": (
                endpoint_identities_distinct
            ),
            "infrastructure_distinct": False,
            "infrastructure_independence_claimed": False,
        },
        "reviewer_type": "independent_model_proxy",
        "human_gold": False,
        "safety_contract": {
            "source_evidence_projection_only": True,
            "target_behavior_consumed": False,
            "hf_model_initialized": False,
            "simulation_called": False,
            "canonical_freeze_performed": False,
            "split_freeze_performed": False,
            "credentials_or_endpoints_included": False,
            "response_model_identity_fail_closed": True,
            "static_proxy_response_model_identity_guard_used": False,
            "expected_response_model_identities_distinct": True,
            "completed_response_model_identities_distinct": True,
            "effective_routes_sha256_bound_without_endpoint_disclosure": True,
            "shared_gateway_allowed": True,
            "infrastructure_independence_claimed": False,
            "defer_and_ineligible_reject_forced_to_cohort_exclude": True,
            "eligible_reject_override_bound_to_derived_eligibility": True,
            "cohort_exclusion_asserts_factual_falsehood": False,
            "single_writer_append_journal": True,
            "terminal_journal_removed": not journal_path.exists(),
        },
    }
    write_json(run_manifest_path, manifest_output)
    return {
        "status": manifest_output["status"],
        "manifest_path": str(run_manifest_path),
        "manifest_sha256": sha256_file(run_manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "resolutions_path": str(decisions_path),
        "selected_count": len(units),
        "terminal_status_counts": manifest_output["terminal_status_counts"],
        "disposition_counts": manifest_output["disposition_counts"],
        "source_reviewer_identity_fallback_count": (
            source_reviewer_identity_fallback_count
        ),
        "strict_resolution_apply_ready": strict_apply_ready,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--template-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--base-fact-ids-file", type=Path)
    result.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "full_public_benchmark_8969_pre_exact_hf_zh_v1.json",
    )
    result.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    result.add_argument("--proposer-provider-profile", "--generator-provider-profile", dest="proposer_provider_profile")
    result.add_argument("--proposer-model", "--generator-model", dest="proposer_model")
    result.add_argument("--proposer-expected-response-model", "--generator-expected-response-model", dest="proposer_expected_response_model")
    result.add_argument("--proposer-reasoning-effort", "--generator-reasoning-effort", dest="proposer_reasoning_effort")
    result.add_argument("--proposer-max-output-tokens", "--generator-max-output-tokens", dest="proposer_max_output_tokens", type=int)
    result.add_argument("--reviewer-provider-profile", type=str)
    result.add_argument("--reviewer-model")
    result.add_argument("--reviewer-expected-response-model")
    result.add_argument("--reviewer-reasoning-effort")
    result.add_argument("--reviewer-max-output-tokens", type=int)
    result.add_argument("--max-retries", type=int)
    result.add_argument("--limit", type=int)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--checkpoint-compact-every", type=int, default=128)
    result.add_argument("--max-workers", type=int)
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--no-resume", action="store_true")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    proposer_spec, reviewer_spec = resolve_model_specs(
        config,
        proposer_provider_profile=args.proposer_provider_profile,
        proposer_model=args.proposer_model,
        reviewer_provider_profile=args.reviewer_provider_profile,
        reviewer_model=args.reviewer_model,
        proposer_expected_response_model=args.proposer_expected_response_model,
        reviewer_expected_response_model=args.reviewer_expected_response_model,
        proposer_reasoning_effort=args.proposer_reasoning_effort,
        reviewer_reasoning_effort=args.reviewer_reasoning_effort,
        proposer_max_output_tokens=args.proposer_max_output_tokens,
        reviewer_max_output_tokens=args.reviewer_max_output_tokens,
        max_retries=args.max_retries,
    )
    result = run_resolution(
        template_manifest_path=args.template_manifest,
        output_dir=args.output_dir,
        config=config,
        env_path=args.env_file,
        proposer_spec=proposer_spec,
        reviewer_spec=reviewer_spec,
        base_fact_ids_path=args.base_fact_ids_file,
        limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_compact_every=args.checkpoint_compact_every,
        max_workers=args.max_workers,
        resume=not args.no_resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
