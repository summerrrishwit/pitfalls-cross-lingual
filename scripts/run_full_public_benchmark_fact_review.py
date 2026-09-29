#!/usr/bin/env python3
"""Run resumable, behavior-blind proxy fact review over a bound review export.

This runner calls a proposal model and an independent review model in small
batches.  It writes one atomic checkpoint row per ``base_fact_id`` and derives
a strict decision JSONL accepted by ``review_public_benchmark_bundle.py apply``.
It never initializes an HF model, runs Simulation, promotes facts, or freezes a
split.  All outputs remain Codex-proxy evidence with ``human_gold=false``.
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

TOOL_VERSION = "full-public-benchmark-fact-review-runner-v2"
CHECKPOINT_SCHEMA_VERSION = "full-public-benchmark-fact-review-checkpoint-v1"
RUN_MANIFEST_SCHEMA_VERSION = "full-public-benchmark-fact-review-run-manifest-v1"
BATCH_RESPONSE_SCHEMA_VERSION = "full-public-benchmark-fact-review-batch-response-v1"
PROJECTION_SCHEMA_VERSION = "full-public-benchmark-fact-review-projection-v1"
GENERATOR_PROMPT_VERSION = "full-public-benchmark-fact-review-generator-v1"
REVIEWER_PROMPT_VERSION = "full-public-benchmark-fact-review-reviewer-v1"
REVIEW_METHOD = "behavior_blind_dual_model_batch_fact_review_v1"
DEFAULT_GENERATOR_ROUTE = "qwen3.8-max-0902/bailian/bailian"
DEFAULT_GENERATOR_RESPONSE_MODEL = "qwen3.8-max"
DEFAULT_REVIEWER_MODEL = "gpt-5.5"
DEFAULT_REVIEWER_RESPONSE_MODEL = "gpt-5.5-2026-04-24"

# Kept lazy so offline validation and unit tests do not require provider SDKs.
# Tests may also replace this binding with a compatible fake router.
ModelRouter: Optional[Any] = None

OVERALL_DECISIONS = frozenset({"accept", "reject", "defer", "revise"})
TARGET_DECISIONS = frozenset({"accept", "reject"})
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
    "require_response_model_identity",
)

GENERATOR_INSTRUCTIONS = """You are the evidence-first proposal stage for a factual review.
Review only the supplied source snapshot, extracted triples, aliases, and candidate distractors.
No target-model behavior, split assignment, HF result, or Simulation result is supplied or allowed.
Do not accept a claim merely from model memory. If the supplied source evidence cannot support the
canonical proposition under the same entity, relation, and temporal scope, use defer. Reject clear
falsehoods. Use revise only when the current extraction needs a correctable rewrite. A distractor is
accepted only when it is not a valid answer under exactly the same scope. Return JSON only."""

REVIEWER_INSTRUCTIONS = """You are the final independent proxy reviewer for a factual review.
Re-evaluate the supplied source evidence yourself. The generator draft is non-authoritative and may
be overturned. No target-model behavior, split assignment, HF result, or Simulation result is
available or allowed. Accept only when the canonical fact and relation are supported, all duplicate
members are valid, and the canonical answer remains an accepted alias. Mark every member, alias,
and distractor accept or reject. Prefer defer over unsupported confidence. Return JSON only."""


def _load_review_tool() -> Any:
    path = Path(__file__).resolve().with_name("review_public_benchmark_bundle.py")
    spec = importlib.util.spec_from_file_location("full_fact_review_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load review contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review_tool = _load_review_tool()


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
    return value.strip()


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
        DEFAULT_GENERATOR_ROUTE: DEFAULT_GENERATOR_RESPONSE_MODEL,
        DEFAULT_REVIEWER_MODEL: DEFAULT_REVIEWER_RESPONSE_MODEL,
    }.get(requested)


def _role_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = (
        config.get("fact_review"),
        config.get("full_public_benchmark_fact_review"),
        (config.get("model_roles") or {}).get("fact_review")
        if isinstance(config.get("model_roles"), dict)
        else None,
        (config.get("model_roles") or {}).get("full_public_benchmark_fact_review")
        if isinstance(config.get("model_roles"), dict)
        else None,
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def resolve_model_specs(
    config: Mapping[str, Any],
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
    role = _role_config(config)
    generator = dict(role.get("generator") or {})
    reviewer = dict(role.get("reviewer") or {})
    overrides = (
        (
            generator,
            generator_provider_profile,
            generator_model,
            generator_reasoning_effort,
            generator_max_output_tokens,
        ),
        (
            reviewer,
            reviewer_provider_profile,
            reviewer_model,
            reviewer_reasoning_effort,
            reviewer_max_output_tokens,
        ),
    )
    expected_models = (
        generator_expected_response_model,
        reviewer_expected_response_model,
    )
    for (spec, profile, model, effort, max_tokens), expected_model in zip(
        overrides, expected_models
    ):
        if profile:
            spec["provider_profile"] = profile
        if model:
            spec["model"] = model
        if expected_model:
            spec["expected_response_model"] = expected_model
        if effort:
            spec["reasoning_effort"] = effort
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_output_tokens must be positive")
            spec["max_output_tokens"] = max_tokens
        spec.setdefault("max_output_tokens", 2048)
        spec.setdefault("temperature", 0.0)
        spec["json_mode"] = True
        spec.setdefault(
            "expected_response_model",
            _inferred_expected_response_model(spec.get("model")),
        )
        if max_retries is not None:
            if max_retries < 0:
                raise ValueError("max_retries must be non-negative")
            spec["max_retries"] = max_retries
    return _safe_model_spec(generator), _safe_model_spec(reviewer)


class ReviewUnit(NamedTuple):
    input_index: int
    base_fact_id: str
    review_item_sha256: str
    decision_template_sha256: str
    template: Dict[str, Any]
    projection: Dict[str, Any]
    review_item: Dict[str, Any]


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


def project_review_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the only payload allowed into model prompts.

    The projection is deliberately whitelist-based.  In particular it never
    copies ``review_context.behavior_row``, split fields, admission flags, or
    any HF/Simulation output.
    """

    members: List[Dict[str, Any]] = []
    for member in item.get("members", []):
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
    targets = item.get("review_targets")
    if not isinstance(targets, dict):
        raise ValueError(f"Invalid review targets for {item.get('base_fact_id')}")
    aliases = targets.get("aliases")
    distractors = targets.get("distractors")
    if not isinstance(aliases, list) or not isinstance(distractors, list):
        raise ValueError(f"Invalid aliases or distractors for {item.get('base_fact_id')}")
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "base_fact_id": item.get("base_fact_id"),
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
    }


def _resolve_bound_artifact(
    explicit_path: Optional[Path], bound_path: Path, label: str
) -> Path:
    if explicit_path is None:
        return bound_path
    resolved = Path(explicit_path).resolve()
    if resolved != bound_path:
        raise ValueError(f"Explicit {label} path differs from export-manifest binding")
    return resolved


def load_review_units(
    *,
    export_manifest_path: Path,
    review_items_path: Optional[Path] = None,
    decision_template_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], Path, Path, List[ReviewUnit]]:
    export_manifest_path = Path(export_manifest_path).resolve()
    manifest = read_json(export_manifest_path)
    scope_binding = manifest.get("scope_manifest")
    if not isinstance(scope_binding, dict):
        raise ValueError("Review export manifest is missing scope_manifest")
    scope_path = Path(
        _required_string(scope_binding.get("path"), "scope_manifest.path")
    ).resolve()
    scope = read_json(scope_path)
    review_tool._validate_scope_manifest(scope)
    bound_items, bound_templates = review_tool._validate_export_manifest(
        manifest, scope, scope_path
    )
    items_path = _resolve_bound_artifact(review_items_path, bound_items, "review_items")
    templates_path = _resolve_bound_artifact(
        decision_template_path, bound_templates, "decision_template"
    )
    total = manifest.get("base_fact_count")
    if not isinstance(total, int) or total <= 0:
        raise ValueError("Review export manifest has invalid base_fact_count")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected_count = total if limit is None else min(limit, total)

    item_iter = iter_jsonl(items_path)
    template_iter = iter_jsonl(templates_path)
    units: List[ReviewUnit] = []
    seen: set[str] = set()
    for input_index in range(selected_count):
        try:
            item = next(item_iter)
            template = next(template_iter)
        except StopIteration as exc:
            raise ValueError("Review items/templates end before manifest record_count") from exc
        base_fact_id = _required_string(item.get("base_fact_id"), "base_fact_id")
        if base_fact_id in seen:
            raise ValueError(f"Duplicate base_fact_id in selected review items: {base_fact_id}")
        seen.add(base_fact_id)
        if item.get("schema_version") != review_tool.EXPORT_ITEM_SCHEMA_VERSION:
            raise ValueError(f"Unsupported review item schema: {base_fact_id}")
        if item.get("scope_id") != manifest.get("scope_id"):
            raise ValueError(f"Review item scope mismatch: {base_fact_id}")
        if item.get("human_gold") is not False:
            raise ValueError(f"Review item must remain human_gold=false: {base_fact_id}")
        if template.get("base_fact_id") != base_fact_id:
            raise ValueError(f"Review item/template ordering mismatch: {base_fact_id}")
        expected_template = review_tool._decision_template(item)
        if sha256_value(template) != sha256_value(expected_template):
            raise ValueError(f"Stale or invalid decision template: {base_fact_id}")
        item_sha = sha256_value(item)
        if template.get("review_item_sha256") != item_sha:
            raise ValueError(f"Stale review_item_sha256: {base_fact_id}")
        units.append(
            ReviewUnit(
                input_index=input_index,
                base_fact_id=base_fact_id,
                review_item_sha256=item_sha,
                decision_template_sha256=sha256_value(template),
                template=dict(template),
                projection=project_review_item(item),
                review_item=dict(item),
            )
        )
    if selected_count == total:
        try:
            next(item_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("Review items exceed manifest record_count")
        try:
            next(template_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("Decision templates exceed manifest record_count")
    return manifest, items_path, templates_path, units


def _prompt_payload(prompt: str) -> Dict[str, Any]:
    marker = "INPUT_JSON:\n"
    return json.loads(prompt.split(marker, 1)[1])


def _response_contract() -> Dict[str, Any]:
    return {
        "top_level": {
            "schema_version": BATCH_RESPONSE_SCHEMA_VERSION,
            "items": "exactly one result for every input base_fact_id",
        },
        "item": {
            "base_fact_id": "copy the exact input ID",
            "overall_decision": sorted(OVERALL_DECISIONS),
            "fact_review": {
                "decision": sorted(OVERALL_DECISIONS),
                "reason": "non-empty evidence-grounded reason",
            },
            "relation_review": {
                "decision": sorted(OVERALL_DECISIONS),
                "reason": "non-empty evidence-grounded reason",
            },
            "member_reviews": [
                {
                    "candidate_id": "copy every supplied member ID exactly once",
                    "decision": sorted(TARGET_DECISIONS),
                    "reason": "non-empty reason",
                }
            ],
            "alias_reviews": [
                {
                    "alias_id": "copy every supplied alias ID exactly once",
                    "decision": sorted(TARGET_DECISIONS),
                    "reason": "non-empty reason",
                }
            ],
            "distractor_reviews": [
                {
                    "distractor_id": "copy every supplied distractor ID exactly once",
                    "decision": sorted(TARGET_DECISIONS),
                    "reason": "non-empty reason",
                }
            ],
            "notes": "non-empty concise overall rationale",
        },
    }


def build_generator_prompt(units: Sequence[ReviewUnit]) -> str:
    payload = {
        "response_contract": _response_contract(),
        "items": [unit.projection for unit in units],
    }
    return f"{GENERATOR_INSTRUCTIONS}\nINPUT_JSON:\n{canonical_json_bytes(payload).decode('utf-8')}"


def build_reviewer_prompt(
    units: Sequence[ReviewUnit], drafts: Mapping[str, Mapping[str, Any]]
) -> str:
    payload = {
        "response_contract": _response_contract(),
        "items": [
            {
                "evidence": unit.projection,
                "generator_draft": drafts[unit.base_fact_id],
            }
            for unit in units
        ],
    }
    return f"{REVIEWER_INSTRUCTIONS}\nINPUT_JSON:\n{canonical_json_bytes(payload).decode('utf-8')}"


def _validate_reasoned_decision(value: Any, field: str, allowed: frozenset[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"decision", "reason"}:
        raise ValueError(f"{field} must contain exactly decision and reason")
    if value.get("decision") not in allowed:
        raise ValueError(f"{field}.decision is invalid")
    _required_string(value.get("reason"), f"{field}.reason")


def _validate_target_response(
    values: Any,
    *,
    field: str,
    id_field: str,
    expected_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    output: Dict[str, Dict[str, Any]] = {}
    for entry in values:
        if not isinstance(entry, dict) or set(entry) != {id_field, "decision", "reason"}:
            raise ValueError(f"{field} entries must contain {id_field}, decision, reason")
        target_id = _required_string(entry.get(id_field), f"{field}.{id_field}")
        if target_id in output:
            raise ValueError(f"Duplicate {field} target: {target_id}")
        if entry.get("decision") not in TARGET_DECISIONS:
            raise ValueError(f"{field}.{target_id}.decision is invalid")
        _required_string(entry.get("reason"), f"{field}.{target_id}.reason")
        output[target_id] = dict(entry)
    if set(output) != set(expected_ids):
        raise ValueError(
            f"{field} coverage mismatch: expected={sorted(expected_ids)}, "
            f"actual={sorted(output)}"
        )
    return output


def validate_batch_response(
    value: Dict[str, Any], units: Sequence[ReviewUnit]
) -> None:
    if set(value) != {"schema_version", "items"}:
        raise ValueError("Batch response must contain exactly schema_version and items")
    if value.get("schema_version") != BATCH_RESPONSE_SCHEMA_VERSION:
        raise ValueError("Unsupported batch response schema_version")
    rows = value.get("items")
    if not isinstance(rows, list):
        raise ValueError("Batch response items must be a list")
    by_id: Dict[str, Dict[str, Any]] = {}
    expected = {unit.base_fact_id: unit for unit in units}
    item_fields = {
        "base_fact_id",
        "overall_decision",
        "fact_review",
        "relation_review",
        "member_reviews",
        "alias_reviews",
        "distractor_reviews",
        "notes",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != item_fields:
            raise ValueError("Each batch response item has invalid fields")
        base_fact_id = _required_string(row.get("base_fact_id"), "base_fact_id")
        if base_fact_id in by_id:
            raise ValueError(f"Duplicate response base_fact_id: {base_fact_id}")
        if base_fact_id not in expected:
            raise ValueError(f"Unknown response base_fact_id: {base_fact_id}")
        if row.get("overall_decision") not in OVERALL_DECISIONS:
            raise ValueError(f"Invalid overall_decision: {base_fact_id}")
        _validate_reasoned_decision(
            row.get("fact_review"), f"{base_fact_id}.fact_review", OVERALL_DECISIONS
        )
        _validate_reasoned_decision(
            row.get("relation_review"),
            f"{base_fact_id}.relation_review",
            OVERALL_DECISIONS,
        )
        _required_string(row.get("notes"), f"{base_fact_id}.notes")
        unit = expected[base_fact_id]
        member_ids = [
            str(entry["candidate_id"])
            for entry in unit.template.get("member_reviews", [])
        ]
        alias_ids = [
            str(entry["alias_id"]) for entry in unit.template.get("alias_reviews", [])
        ]
        distractor_ids = [
            str(entry["distractor_id"])
            for entry in unit.template.get("distractor_reviews", [])
        ]
        member_reviews = _validate_target_response(
            row.get("member_reviews"),
            field=f"{base_fact_id}.member_reviews",
            id_field="candidate_id",
            expected_ids=member_ids,
        )
        alias_reviews = _validate_target_response(
            row.get("alias_reviews"),
            field=f"{base_fact_id}.alias_reviews",
            id_field="alias_id",
            expected_ids=alias_ids,
        )
        _validate_target_response(
            row.get("distractor_reviews"),
            field=f"{base_fact_id}.distractor_reviews",
            id_field="distractor_id",
            expected_ids=distractor_ids,
        )
        if row["overall_decision"] == "accept":
            if row["fact_review"]["decision"] != "accept":
                raise ValueError(f"Accept requires accepted fact_review: {base_fact_id}")
            if row["relation_review"]["decision"] != "accept":
                raise ValueError(f"Accept requires accepted relation_review: {base_fact_id}")
            if any(entry["decision"] != "accept" for entry in member_reviews.values()):
                raise ValueError(f"Accept requires every member: {base_fact_id}")
            answer = unit.projection["members"][0]["extracted_triple"].get("answer")
            accepted_alias_texts = {
                review_tool.normalize_text(target.get("text_en"))
                for target, response in zip(
                    unit.template.get("alias_reviews", []),
                    [alias_reviews[str(target["alias_id"])] for target in unit.template.get("alias_reviews", [])],
                )
                if response["decision"] == "accept"
            }
            if review_tool.normalize_text(answer) not in accepted_alias_texts:
                raise ValueError(
                    f"Accept must retain an alias equal to the canonical answer: {base_fact_id}"
                )
        by_id[base_fact_id] = dict(row)
    if set(by_id) != set(expected):
        raise ValueError(
            f"Batch response coverage mismatch: expected={sorted(expected)}, "
            f"actual={sorted(by_id)}"
        )


def _response_map(value: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["base_fact_id"]): dict(row) for row in value["items"]}


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
    units: Sequence[ReviewUnit],
    model_spec: Mapping[str, Any],
    prompt: str,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    base_fact_ids = [unit.base_fact_id for unit in units]
    batch_id = f"batch_{sha256_value(base_fact_ids)[:16]}"
    safe_spec = _safe_model_spec(model_spec)
    expected_response_model = safe_spec["expected_response_model"]
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
            dict(model_spec),
            prompt,
            lambda value: validate_batch_response(value, units),
        )
        raw_response = result.raw_response if isinstance(result.raw_response, str) else None
        identity_status = (
            "matched"
            if result.response_model == expected_response_model
            else "mismatch"
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
            validate_batch_response(result.parsed_response, units)
        except Exception as error:
            record = {
                **record,
                "terminal_status": "failed",
                "post_validation_error_type": type(error).__name__,
            }
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
    non_splittable = {
        "AuthenticationError",
        "PermissionDeniedError",
        "StaticProxyIdentityError",
        "ResponseModelIdentityError",
    }
    return not bool(error_types.intersection(non_splittable))


def _target_response_map(
    verdict: Mapping[str, Any], field: str, id_field: str
) -> Dict[str, str]:
    return {
        str(entry[id_field]): str(entry["decision"])
        for entry in verdict[field]
    }


def _strict_decision(
    unit: ReviewUnit,
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
    member_map = _target_response_map(verdict, "member_reviews", "candidate_id")
    alias_map = _target_response_map(verdict, "alias_reviews", "alias_id")
    distractor_map = _target_response_map(
        verdict, "distractor_reviews", "distractor_id"
    )
    output["member_reviews"] = [
        {**target, "decision": member_map[str(target["candidate_id"])]}
        for target in unit.template["member_reviews"]
    ]
    output["alias_reviews"] = [
        {**target, "decision": alias_map[str(target["alias_id"])]}
        for target in unit.template["alias_reviews"]
    ]
    output["distractor_reviews"] = [
        {**target, "decision": distractor_map[str(target["distractor_id"])]}
        for target in unit.template["distractor_reviews"]
    ]
    output["notes"] = canonical_json_bytes(
        {
            "fact_review": verdict["fact_review"],
            "relation_review": verdict["relation_review"],
            "notes": verdict["notes"],
        }
    ).decode("utf-8")
    if set(output) != review_tool.DECISION_ALLOWED_FIELDS:
        raise ValueError(f"Strict decision fields drifted: {unit.base_fact_id}")
    return output


def _failed_rows(
    units: Sequence[ReviewUnit],
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
            "review_item_sha256": unit.review_item_sha256,
            "decision_template_sha256": unit.decision_template_sha256,
            "run_contract_sha256": run_contract_sha256,
            "terminal_status": "failed",
            "failure_stage": failure_stage,
            "generator_verdict": None,
            "reviewer_verdict": None,
            "strict_decision": None,
            "model_calls": [dict(trace) for trace in traces],
            "retry_history": [],
            "completed_at": now_fn(),
            "behavior_blind": True,
            "human_gold": False,
        }
        for unit in units
    ]


def _review_subbatch(
    *,
    units: Sequence[ReviewUnit],
    drafts: Mapping[str, Mapping[str, Any]],
    inherited_traces: Sequence[Mapping[str, Any]],
    router: Any,
    reviewer_spec: Mapping[str, Any],
    run_contract_sha256: str,
    now_fn: Callable[[], str],
) -> List[Dict[str, Any]]:
    prompt = build_reviewer_prompt(units, drafts)
    verdicts, call = _call_model(
        router=router,
        stage="fact_review_reviewer",
        prompt_version=REVIEWER_PROMPT_VERSION,
        units=units,
        model_spec=reviewer_spec,
        prompt=prompt,
    )
    traces = [*inherited_traces, call]
    if verdicts is None:
        if _should_split(call, len(units)):
            midpoint = len(units) // 2
            return [
                *_review_subbatch(
                    units=units[:midpoint],
                    drafts=drafts,
                    inherited_traces=traces,
                    router=router,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                ),
                *_review_subbatch(
                    units=units[midpoint:],
                    drafts=drafts,
                    inherited_traces=traces,
                    router=router,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                ),
            ]
        return _failed_rows(
            units, traces, "reviewer", run_contract_sha256, now_fn
        )

    response_model = call.get("response_model")
    rows: List[Dict[str, Any]] = []
    for unit in units:
        reviewed_at = now_fn()
        verdict = verdicts[unit.base_fact_id]
        decision = _strict_decision(
            unit, verdict, reviewer_spec, response_model, reviewed_at
        )
        rows.append(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "tool_version": TOOL_VERSION,
                "base_fact_id": unit.base_fact_id,
                "input_index": unit.input_index,
                "review_item_sha256": unit.review_item_sha256,
                "decision_template_sha256": unit.decision_template_sha256,
                "run_contract_sha256": run_contract_sha256,
                "terminal_status": "completed",
                "failure_stage": None,
                "generator_verdict": dict(drafts[unit.base_fact_id]),
                "reviewer_verdict": dict(verdict),
                "strict_decision": decision,
                "model_calls": [dict(trace) for trace in traces],
                "retry_history": [],
                "completed_at": reviewed_at,
                "behavior_blind": True,
                "human_gold": False,
            }
        )
    return rows


def _process_batch(
    *,
    units: Sequence[ReviewUnit],
    router: Any,
    generator_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    run_contract_sha256: str,
    now_fn: Callable[[], str],
    inherited_traces: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    prompt = build_generator_prompt(units)
    drafts, call = _call_model(
        router=router,
        stage="fact_review_generator",
        prompt_version=GENERATOR_PROMPT_VERSION,
        units=units,
        model_spec=generator_spec,
        prompt=prompt,
    )
    traces = [*inherited_traces, call]
    if drafts is None:
        if _should_split(call, len(units)):
            midpoint = len(units) // 2
            return [
                *_process_batch(
                    units=units[:midpoint],
                    router=router,
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
                *_process_batch(
                    units=units[midpoint:],
                    router=router,
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    run_contract_sha256=run_contract_sha256,
                    now_fn=now_fn,
                    inherited_traces=traces,
                ),
            ]
        return _failed_rows(
            units, traces, "generator", run_contract_sha256, now_fn
        )
    return _review_subbatch(
        units=units,
        drafts=drafts,
        inherited_traces=traces,
        router=router,
        reviewer_spec=reviewer_spec,
        run_contract_sha256=run_contract_sha256,
        now_fn=now_fn,
    )


def _chunks(values: Sequence[ReviewUnit], size: int) -> Iterator[List[ReviewUnit]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _validate_checkpoint_row(
    row: Mapping[str, Any],
    *,
    units_by_id: Mapping[str, ReviewUnit],
    run_contract_sha256: str,
) -> Tuple[str, Dict[str, Any]]:
    base_fact_id = _required_string(row.get("base_fact_id"), "checkpoint.base_fact_id")
    unit = units_by_id.get(base_fact_id)
    if unit is None:
        raise ValueError(
            f"Checkpoint contains an item outside the selected input prefix: {base_fact_id}"
        )
    if row.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema: {base_fact_id}")
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError(f"Checkpoint run contract is stale: {base_fact_id}")
    if row.get("review_item_sha256") != unit.review_item_sha256:
        raise ValueError(f"Checkpoint review item is stale: {base_fact_id}")
    if row.get("decision_template_sha256") != unit.decision_template_sha256:
        raise ValueError(f"Checkpoint decision template is stale: {base_fact_id}")
    if row.get("terminal_status") not in {"completed", "failed"}:
        raise ValueError(f"Checkpoint terminal_status is invalid: {base_fact_id}")
    if row.get("human_gold") is not False or row.get("behavior_blind") is not True:
        raise ValueError(f"Checkpoint safety contract is invalid: {base_fact_id}")
    calls = row.get("model_calls")
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise ValueError(f"Checkpoint model_calls are invalid: {base_fact_id}")
    if row.get("terminal_status") == "completed":
        decision = row.get("strict_decision")
        if not isinstance(decision, dict):
            raise ValueError(f"Completed checkpoint has no strict decision: {base_fact_id}")
        review_tool._validate_decision(decision, unit.review_item)
        for stage in ("fact_review_generator", "fact_review_reviewer"):
            if not any(
                call.get("stage") == stage
                and call.get("terminal_status") == "completed"
                and call.get("response_model_identity_status") == "matched"
                for call in calls
            ):
                raise ValueError(
                    f"Completed checkpoint lacks a matched {stage} call: {base_fact_id}"
                )
    elif row.get("strict_decision") is not None:
        raise ValueError(f"Failed checkpoint has a strict decision: {base_fact_id}")
    return base_fact_id, dict(row)


def _read_journal(path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """Read a journal, ignoring only an interrupted final partial JSON line."""

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
    checkpoint_path: Path,
    journal_path: Path,
    *,
    units_by_id: Mapping[str, ReviewUnit],
    run_contract_sha256: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], bool]:
    output: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.is_file():
        for row in read_jsonl(checkpoint_path, allow_empty=True):
            base_fact_id, validated = _validate_checkpoint_row(
                row,
                units_by_id=units_by_id,
                run_contract_sha256=run_contract_sha256,
            )
            if base_fact_id in output:
                raise ValueError(f"Duplicate compact checkpoint base_fact_id: {base_fact_id}")
            output[base_fact_id] = validated
    journal_rows, needs_repair = _read_journal(journal_path)
    for row in journal_rows:
        base_fact_id, validated = _validate_checkpoint_row(
            row,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha256,
        )
        # Journals are ordered updates; the last durable row wins on resume.
        output[base_fact_id] = validated
    return output, journal_rows, needs_repair


def _ordered_checkpoint_rows(
    checkpoint: Mapping[str, Mapping[str, Any]], units: Sequence[ReviewUnit]
) -> List[Dict[str, Any]]:
    return [dict(checkpoint[unit.base_fact_id]) for unit in units if unit.base_fact_id in checkpoint]


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
    units: Sequence[ReviewUnit],
    final: bool,
) -> None:
    rows = _ordered_checkpoint_rows(checkpoint, units)
    write_jsonl(checkpoint_path, rows)
    decisions = [
        row["strict_decision"]
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
        # Atomic replacement makes a crash between compact and reset harmless:
        # duplicated journal rows are valid ordered updates on the next resume.
        write_jsonl(journal_path, [])


def run_review(
    *,
    export_manifest_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    env_path: Path,
    generator_spec: Mapping[str, Any],
    reviewer_spec: Mapping[str, Any],
    review_items_path: Optional[Path] = None,
    decision_template_path: Optional[Path] = None,
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
    execution = config.get("execution") if isinstance(config.get("execution"), dict) else {}
    worker_count = int(max_workers or execution.get("max_workers", 4))
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    generator_spec = _safe_model_spec(generator_spec)
    reviewer_spec = _safe_model_spec(reviewer_spec)
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict):
        raise ValueError("config.provider_profiles must be an object")
    for label, spec in (("generator", generator_spec), ("reviewer", reviewer_spec)):
        if spec["provider_profile"] not in profiles:
            raise ValueError(f"Unknown {label} provider_profile: {spec['provider_profile']}")
    if (
        generator_spec["provider_profile"],
        generator_spec["model"],
    ) == (
        reviewer_spec["provider_profile"],
        reviewer_spec["model"],
    ):
        raise ValueError("generator and reviewer must use distinct model identities")

    manifest, items_path, templates_path, units = load_review_units(
        export_manifest_path=export_manifest_path,
        review_items_path=review_items_path,
        decision_template_path=decision_template_path,
        limit=limit,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fact_review_checkpoint.jsonl"
    journal_path = output_dir / "fact_review_checkpoint.journal.jsonl"
    decisions_path = output_dir / "review_decisions_codex_proxy.jsonl"
    run_manifest_path = output_dir / "fact_review_run_manifest.json"
    event_log_path = output_dir / "redacted_model_events.jsonl"
    config_sha = sha256_value(config)
    prompt_contract_sha = sha256_value(
        {
            "generator_prompt_version": GENERATOR_PROMPT_VERSION,
            "generator_instructions": GENERATOR_INSTRUCTIONS,
            "reviewer_prompt_version": REVIEWER_PROMPT_VERSION,
            "reviewer_instructions": REVIEWER_INSTRUCTIONS,
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "batch_response_schema_version": BATCH_RESPONSE_SCHEMA_VERSION,
        }
    )
    run_contract = {
        "tool_version": TOOL_VERSION,
        "export_manifest_sha256": sha256_file(export_manifest_path),
        "review_items_sha256": sha256_file(items_path),
        "decision_template_sha256": sha256_file(templates_path),
        "config_sha256": config_sha,
        "generator_model_spec": generator_spec,
        "reviewer_model_spec": reviewer_spec,
        "prompt_contract_sha256": prompt_contract_sha,
        "batch_size": batch_size,
        "selected_base_fact_ids_sha256": sha256_value(
            [unit.base_fact_id for unit in units]
        ),
        "selected_count": len(units),
        "full_export_count": manifest["base_fact_count"],
        "selection_is_full_export": len(units) == manifest["base_fact_count"],
        "behavior_blind": True,
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
    if resume:
        checkpoint, journal_rows, journal_needs_repair = _checkpoint_state(
            checkpoint_path,
            journal_path,
            units_by_id=units_by_id,
            run_contract_sha256=run_contract_sha,
        )
        if journal_needs_repair:
            write_jsonl(journal_path, journal_rows)
    else:
        checkpoint, journal_rows = {}, []
        # Starting fresh must not leave an older compact file able to reappear
        # after an interruption before the first new compaction.
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
    if router is None:
        router = _load_model_router_class()(
            dict(config), Path(env_path).resolve(), event_log_path
        )
    if hasattr(router, "_event"):
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
                    generator_spec=generator_spec,
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
                    base_fact_id = str(row["base_fact_id"])
                    previous = checkpoint.get(base_fact_id)
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
                # Only this coordinator appends journal rows and compacts files.
                # Journal durability comes before the in-memory state update.
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

    # A normal return always leaves one deterministic compact checkpoint and no
    # journal.  If execution is interrupted, the durable journal is merged on
    # the next resume before any new model calls.
    _compact_runtime_artifacts(
        checkpoint_path=checkpoint_path,
        journal_path=journal_path,
        decisions_path=decisions_path,
        checkpoint=checkpoint,
        units=units,
        final=True,
    )

    ordered_rows = _ordered_checkpoint_rows(checkpoint, units)
    status_counts = Counter(row["terminal_status"] for row in ordered_rows)
    decisions = [
        row["strict_decision"]
        for row in ordered_rows
        if row.get("terminal_status") == "completed"
    ]
    decision_counts = Counter(str(row["decision"]) for row in decisions)
    manifest_output = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": (
            "completed"
            if status_counts.get("completed", 0) == len(units)
            else "completed_with_failures"
        ),
        "scope_id": manifest["scope_id"],
        "export_manifest": {
            "path": str(Path(export_manifest_path).resolve()),
            "sha256": sha256_file(export_manifest_path),
        },
        "review_items": {
            "path": str(items_path),
            "sha256": sha256_file(items_path),
        },
        "decision_template": {
            "path": str(templates_path),
            "sha256": sha256_file(templates_path),
        },
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha,
        "selected_count": len(units),
        "full_export_count": manifest["base_fact_count"],
        "batch_size": batch_size,
        "checkpoint_compact_every": checkpoint_compact_every,
        "max_workers": worker_count,
        "resume_enabled": resume,
        "retry_failed_enabled": retry_failed,
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "record_count": len(ordered_rows),
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
            },
            "decisions": {
                "path": str(decisions_path),
                "sha256": sha256_file(decisions_path),
                "record_count": len(decisions),
                "schema_version": review_tool.DECISION_SCHEMA_VERSION,
            },
        },
        "model_roles": {
            "generator": generator_spec,
            "reviewer": reviewer_spec,
            "role_identity_distinct": (
                generator_spec["provider_profile"], generator_spec["model"]
            )
            != (reviewer_spec["provider_profile"], reviewer_spec["model"]),
        },
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "safety_contract": {
            "behavior_blind_projection_only": True,
            "target_behavior_consumed": False,
            "hf_model_initialized": False,
            "simulation_called": False,
            "canonical_freeze_performed": False,
            "automatic_promotion_performed": False,
            "credentials_or_endpoints_included": False,
            "strict_apply_decisions_only": True,
            "response_model_identity_fail_closed": True,
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
        "decisions_path": str(decisions_path),
        "selected_count": len(units),
        "terminal_status_counts": manifest_output["terminal_status_counts"],
        "decision_counts": manifest_output["decision_counts"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--export-manifest",
        "--review-export-manifest",
        dest="export_manifest",
        type=Path,
        required=True,
    )
    result.add_argument("--review-items", type=Path)
    result.add_argument(
        "--decision-template",
        "--review-decisions-template",
        dest="decision_template",
        type=Path,
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "factual_perturbation_zh_mvp_v1.json",
    )
    result.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    result.add_argument(
        "--generator-provider-profile",
        "--generator-profile",
        dest="generator_provider_profile",
    )
    result.add_argument("--generator-model")
    result.add_argument("--generator-expected-response-model")
    result.add_argument("--generator-reasoning-effort")
    result.add_argument("--generator-max-output-tokens", type=int)
    result.add_argument(
        "--reviewer-provider-profile",
        "--reviewer-profile",
        dest="reviewer_provider_profile",
    )
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
    generator_spec, reviewer_spec = resolve_model_specs(
        config,
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
    result = run_review(
        export_manifest_path=args.export_manifest,
        review_items_path=args.review_items,
        decision_template_path=args.decision_template,
        output_dir=args.output_dir,
        config=config,
        env_path=args.env_file,
        generator_spec=generator_spec,
        reviewer_spec=reviewer_spec,
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
