#!/usr/bin/env python3
"""Resolve non-accept full-pool fact reviews before semantic closure.

The command has three deliberately separate phases:

``materialize-template``
    Replay a complete ``review_public_benchmark_bundle.py apply`` chain and
    emit one resolution item/template for every revise, defer, or reject row.

``apply``
    Require an exact, SHA-bound resolution for every emitted item.  Only a
    ``revise`` row may be retained through a field-level edit, and that edit
    must be accepted by an independent rereviewer.  A source ``reject`` may
    use the same path only when its preserved reviewer evidence accepts both
    the core fact and relation, retains the canonical answer alias, and limits
    the problem to auditable member/alias/distractor targets.  Any non-accept
    row may instead be explicitly excluded from the current cohort; cohort
    exclusion is never encoded as an assertion that the underlying fact is
    false.

``validate``
    Replay the source review chain and resolution decisions, then compare all
    output rows and manifest contracts without writing anything.

The tool is pool-size agnostic, performs no network/model/HF work, and never
freezes facts or splits.  A changed output SHA invalidates every semantic
candidate manifest bound to the source full-base artifact and requires a new
semantic/component/split/candidate materialization.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_VERSION = "full-public-benchmark-fact-resolution-v1"
STRICT_FACT_REVIEW_METHOD = "behavior_blind_dual_model_batch_fact_review_v1"

FULL_BASE_SCHEMA = "public-benchmark-full-provisional-base-fact-v1"
RESOLUTION_ITEM_SCHEMA = "full-public-benchmark-fact-resolution-item-v1"
RESOLUTION_DECISION_SCHEMA = "full-public-benchmark-fact-resolution-decision-v1"
TEMPLATE_MANIFEST_SCHEMA = "full-public-benchmark-fact-resolution-template-manifest-v1"
TEMPLATE_SUMMARY_SCHEMA = "full-public-benchmark-fact-resolution-template-summary-v1"
RESOLUTION_LEDGER_SCHEMA = "full-public-benchmark-fact-resolution-ledger-v1"
REVISION_LINEAGE_SCHEMA = "full-public-benchmark-fact-revision-lineage-v1"
COHORT_EXCLUSION_SCHEMA = "full-public-benchmark-fact-cohort-exclusion-v1"
OUTPUT_MANIFEST_SCHEMA = "full-public-benchmark-fact-resolution-manifest-v1"
OUTPUT_SUMMARY_SCHEMA = "full-public-benchmark-fact-resolution-summary-v1"

NONACCEPT_OUTCOMES = frozenset({"revise", "defer", "reject"})
ALL_OUTCOMES = NONACCEPT_OUTCOMES | {"accept"}
MUTABLE_FIELDS = (
    "subject_en",
    "relation_raw",
    "answer_en",
    "answer_aliases_en",
    "canonical_fact_en",
)
STRING_MUTABLE_FIELDS = frozenset(MUTABLE_FIELDS) - {"answer_aliases_en"}
EDITOR_TYPES = frozenset({"human", "codex_proxy"})
REREVIEWER_TYPES = frozenset({"human", "codex_proxy", "independent_model_proxy"})
SHA256_RE = re.compile(r"[0-9a-f]{64}")

DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "base_fact_id",
        "resolution_item_sha256",
        "disposition",
        "revision",
        "cohort_exclusion",
        "human_gold",
    }
)
REVISION_FIELDS = frozenset(
    {
        "updates",
        "revision_reason",
        "source_reject_override",
        "editor_provenance",
        "rereview",
    }
)
SOURCE_REJECT_OVERRIDE_FIELDS = frozenset(
    {
        "original_review_outcome",
        "eligibility_evidence_sha256",
        "override_reason",
        "target_resolutions",
    }
)
TARGET_RESOLUTION_FIELDS = frozenset(
    {
        "rejected_member_candidate_ids",
        "rejected_alias_ids",
        "rejected_distractor_ids",
        "member_action",
        "alias_action",
        "distractor_action",
    }
)
EDITOR_PROVENANCE_FIELDS = frozenset(
    {"editor_type", "editor_id", "edit_method", "edited_at", "human_gold"}
)
REREVIEW_FIELDS = frozenset(
    {
        "decision",
        "revised_full_fact_row_sha256",
        "field_reviews",
        "rationale",
        "reviewer_type",
        "reviewer_id",
        "review_method",
        "reviewed_at",
        "human_gold",
    }
)
FIELD_REVIEW_FIELDS = frozenset({"decision", "reason"})
COHORT_EXCLUSION_FIELDS = frozenset(
    {
        "reason",
        "scope",
        "factual_falsehood_asserted",
        "resolution_provenance",
    }
)
RESOLUTION_PROVENANCE_FIELDS = frozenset(
    {"reviewer_type", "reviewer_id", "review_method", "reviewed_at", "human_gold"}
)
EXCLUSION_SCOPE = "current_pre_exact_hf_cohort_only"

SAFETY_CONTRACT = {
    "network_or_model_used": False,
    "hf_model_or_tokenizer_bound": False,
    "behavior_execution_performed": False,
    "hidden_state_or_intervention_run": False,
    "translation_performed": False,
    "canonical_freeze_performed": False,
    "split_freeze_performed": False,
    "validation_or_sealed_exposed": False,
    "human_gold": False,
    "output_remains_provisional_pre_exact_hf": True,
}

RESOLUTION_CONTRACT = {
    "every_nonaccept_review_requires_exactly_one_resolution": True,
    "apply_revision_source_outcomes": ["revise", "eligible_reject_override"],
    "reject_override_requires_core_fact_and_relation_accept": True,
    "reject_override_requires_canonical_answer_alias_accept": True,
    "reject_override_requires_primary_member_accept": True,
    "reject_override_limited_to_member_alias_distractor_issues": True,
    "reject_override_preserves_original_reject_and_reason": True,
    "reject_override_rereviewer_differs_from_source_reviewer": True,
    "revision_requires_independent_field_level_rereview": True,
    "defer_and_reject_cannot_be_retained_without_new_review": True,
    "cohort_exclusion_is_current_cohort_only": True,
    "cohort_exclusion_asserts_factual_falsehood": False,
    "proxy_review_is_human_gold": False,
    "missing_or_inconsistent_resolution_fails_closed": True,
}


def _load_review_tool() -> Any:
    path = Path(__file__).resolve().with_name("review_public_benchmark_bundle.py")
    spec = importlib.util.spec_from_file_location(
        "_full_fact_resolution_review_contract", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load review tool: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


review_tool = _load_review_tool()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value


def _required_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_timestamp(value: Any, label: str) -> str:
    text = _required_string(value, label)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"JSONL artifact is empty: {path}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(Path(path), payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(Path(path), payload)


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label} row {row_number}.{field}")
        if key in output:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        output[key] = dict(row)
    return output


def _resolved_path(binding: Mapping[str, Any], owner_path: Path, label: str) -> Path:
    raw = binding.get("path") or binding.get("filename")
    path = Path(_required_string(raw, f"{label}.path"))
    if not path.is_absolute():
        path = owner_path.resolve().parent / path
    return path.resolve()


def _verify_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    explicit_path: Optional[Path] = None,
    expected_schema: Optional[str] = None,
) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = _resolved_path(binding, owner_path, label)
    if explicit_path is not None and path != Path(explicit_path).resolve():
        raise ValueError(f"{label} path does not match the supplied input")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = _required_sha(binding.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label} SHA-256 binding is stale")
    if binding.get("byte_count") is not None and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count is stale")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
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
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        value["schema_version"] = schema_version
    if record_count is not None:
        value["record_count"] = record_count
    return value


def _output_binding(
    path: Path, *, schema_version: str, record_count: Optional[int] = None
) -> Dict[str, Any]:
    path = Path(path)
    value: Dict[str, Any] = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def _validate_aliases(value: Any, *, answer: str, label: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    output: List[str] = []
    seen = set()
    for position, raw in enumerate(value, start=1):
        alias = _required_string(raw, f"{label}[{position}]")
        normalized = normalize_text(alias)
        if normalized in seen:
            raise ValueError(f"{label} contains duplicate normalized aliases")
        seen.add(normalized)
        output.append(alias)
    if normalize_text(answer) not in seen:
        raise ValueError(f"{label} must contain answer_en")
    return output


def _validate_full_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows = read_jsonl(path)
    index = _unique_index(rows, "base_fact_id", "full_base_facts")
    for base_fact_id, row in index.items():
        if row.get("schema_version") != FULL_BASE_SCHEMA:
            raise ValueError(f"Unsupported full-base schema: {base_fact_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"Full-base row must have human_gold=false: {base_fact_id}")
        for field in STRING_MUTABLE_FIELDS:
            _required_string(row.get(field), f"full_base_facts {base_fact_id}.{field}")
        _validate_aliases(
            row.get("answer_aliases_en"),
            answer=str(row["answer_en"]),
            label=f"full_base_facts {base_fact_id}.answer_aliases_en",
        )
    return rows, index


def _manifest_outcome(row: Mapping[str, Any]) -> str:
    outcome = row.get("review_outcome")
    if outcome not in ALL_OUTCOMES:
        raise ValueError(
            f"Complete fact review required; {row.get('base_fact_id')} has "
            f"outcome {outcome!r}"
        )
    return str(outcome)


def _reasoned_core_review(value: Any) -> Optional[Dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"decision", "reason"}:
        return None
    decision = value.get("decision")
    reason = value.get("reason")
    if decision not in ALL_OUTCOMES or not isinstance(reason, str) or not reason.strip():
        return None
    return {"decision": str(decision), "reason": reason.strip()}


def _source_reject_override_eligibility(
    *,
    source_row: Mapping[str, Any],
    item: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive (never merely trust) whether a source reject is repairable.

    The strict fact runner preserves its core fact/relation verdicts as JSON in
    ``notes`` because the public review-decision-v2 schema has no top-level
    fields for them.  A hand-authored or legacy note therefore cannot qualify
    for a reject override.  Target completeness and the canonical alias are
    independently rechecked against the SHA-bound review item.
    """

    blockers: List[str] = []
    source_provenance = decision.get("review_provenance")
    source_review_method = (
        source_provenance.get("review_method")
        if isinstance(source_provenance, dict)
        else None
    )
    if source_review_method != STRICT_FACT_REVIEW_METHOD:
        blockers.append("source_review_method_not_strict_fact_runner")
    notes_payload: Optional[Dict[str, Any]] = None
    notes = decision.get("notes")
    if isinstance(notes, str):
        try:
            parsed = json.loads(notes)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and set(parsed) == {
            "fact_review",
            "relation_review",
            "notes",
        }:
            notes_payload = parsed
    if notes_payload is None:
        blockers.append("strict_core_review_evidence_missing")
        fact_review = None
        relation_review = None
    else:
        fact_review = _reasoned_core_review(notes_payload.get("fact_review"))
        relation_review = _reasoned_core_review(notes_payload.get("relation_review"))
        if fact_review is None or relation_review is None:
            blockers.append("strict_core_review_evidence_invalid")
    if fact_review is None or fact_review.get("decision") != "accept":
        blockers.append("core_fact_review_not_accept")
    if relation_review is None or relation_review.get("decision") != "accept":
        blockers.append("core_relation_review_not_accept")

    target_specs = (
        ("member_reviews", "candidate_id", item.get("members", [])),
        ("alias_reviews", "alias_id", item.get("review_targets", {}).get("aliases", [])),
        (
            "distractor_reviews",
            "distractor_id",
            item.get("review_targets", {}).get("distractors", []),
        ),
    )
    target_indexes: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    rejected: Dict[str, List[str]] = {}
    for field, id_field, expected_rows in target_specs:
        supplied = decision.get(field)
        if not isinstance(supplied, list):
            supplied = []
        supplied_index = {
            str(entry.get(id_field)): entry
            for entry in supplied
            if isinstance(entry, dict) and isinstance(entry.get(id_field), str)
        }
        expected_ids = [str(entry[id_field]) for entry in expected_rows]
        complete = set(supplied_index) == set(expected_ids) and all(
            supplied_index[target_id].get("decision") in {"accept", "reject"}
            for target_id in expected_ids
        )
        if not complete:
            blockers.append(f"{field}_incomplete")
        target_indexes[field] = supplied_index
        rejected[field] = [
            target_id
            for target_id in expected_ids
            if supplied_index.get(target_id, {}).get("decision") == "reject"
        ]

    aliases = item.get("review_targets", {}).get("aliases", [])
    canonical_alias_ids = [
        str(target["alias_id"])
        for target in aliases
        if normalize_text(target.get("text_en")) == normalize_text(source_row.get("answer_en"))
    ]
    canonical_alias_accepted = bool(canonical_alias_ids) and all(
        target_indexes["alias_reviews"].get(alias_id, {}).get("decision") == "accept"
        for alias_id in canonical_alias_ids
    )
    if not canonical_alias_accepted:
        blockers.append("canonical_answer_alias_not_accept")

    behavior = item.get("review_context", {}).get("behavior_row", {})
    primary_member_id = source_row.get("candidate_id") or behavior.get("candidate_id")
    primary_member_accepted = (
        isinstance(primary_member_id, str)
        and target_indexes["member_reviews"].get(primary_member_id, {}).get("decision")
        == "accept"
    )
    if not primary_member_accepted:
        blockers.append("primary_member_not_accept")
    rejected_target_count = sum(len(values) for values in rejected.values())
    if rejected_target_count == 0:
        blockers.append("no_editable_target_rejection")

    target_resolutions = {
        "rejected_member_candidate_ids": rejected["member_reviews"],
        "rejected_alias_ids": rejected["alias_reviews"],
        "rejected_distractor_ids": rejected["distractor_reviews"],
        "member_action": "exclude_rejected_members_from_revised_support",
        "alias_action": "remove_rejected_aliases_from_revised_answer_aliases",
        "distractor_action": "discard_and_regenerate_after_semantic_closure",
    }
    return {
        "eligible": not blockers,
        "source_review_method": source_review_method,
        "fact_review": copy.deepcopy(fact_review),
        "relation_review": copy.deepcopy(relation_review),
        "canonical_answer_alias_accepted": canonical_alias_accepted,
        "primary_member_candidate_id": primary_member_id,
        "primary_member_accepted": primary_member_accepted,
        "all_target_reviews_complete": not any(
            blocker.endswith("_incomplete") for blocker in blockers
        ),
        "rejected_target_count": rejected_target_count,
        "target_resolutions": target_resolutions,
        "blocking_reasons": sorted(set(blockers)),
    }


def _load_review_chain(
    *,
    full_base_facts_path: Path,
    review_apply_manifest_path: Path,
    review_export_manifest_path: Path,
) -> Dict[str, Any]:
    """Validate and deterministically replay the complete review chain."""

    full_base_facts_path = Path(full_base_facts_path).resolve()
    review_apply_manifest_path = Path(review_apply_manifest_path).resolve()
    review_export_manifest_path = Path(review_export_manifest_path).resolve()
    full_rows, full_index = _validate_full_rows(full_base_facts_path)
    full_ids = [str(row["base_fact_id"]) for row in full_rows]

    apply_manifest = read_json(review_apply_manifest_path)
    if apply_manifest.get("schema_version") != review_tool.APPLY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Fact-review apply manifest schema_version is unsupported")
    if apply_manifest.get("allow_partial") is not False:
        raise ValueError("Fact-review apply manifest must be complete (allow_partial=false)")
    safety = apply_manifest.get("safety_contract")
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

    scope_path = _verify_binding(
        apply_manifest.get("scope_manifest"),
        owner_path=review_apply_manifest_path,
        label="fact-review scope_manifest",
    )
    export_path = _verify_binding(
        apply_manifest.get("export_manifest"),
        owner_path=review_apply_manifest_path,
        label="fact-review export_manifest",
        explicit_path=review_export_manifest_path,
        expected_schema=None,
    )
    decisions_path = _verify_binding(
        apply_manifest.get("decisions"),
        owner_path=review_apply_manifest_path,
        label="fact-review decisions",
    )
    artifacts = apply_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Fact-review apply artifacts are missing")
    staging_binding = artifacts.get("review_staging")
    staging_path = _verify_binding(
        staging_binding,
        owner_path=review_apply_manifest_path,
        label="fact-review review_staging",
        expected_schema=review_tool.STAGING_SCHEMA_VERSION,
    )

    scope = review_tool.read_json(scope_path)
    review_tool._validate_scope_manifest(scope)
    if scope.get("base_fact_ids") != full_ids:
        raise ValueError("Fact-review scope must exactly match full_base_facts order")
    if apply_manifest.get("scope_id") != scope.get("scope_id"):
        raise ValueError("Fact-review apply scope_id mismatch")
    if apply_manifest.get("scoped_base_fact_count") != len(full_ids):
        raise ValueError("Fact-review scoped_base_fact_count is stale")

    source_paths = review_tool._scope_artifact_paths(scope)
    source_rows, source_indexes = review_tool.load_source_artifacts(source_paths)
    review_tool._verify_scope_selection_source(scope, source_indexes)
    for role, binding in scope["source_artifacts"].items():
        if binding.get("record_count") != len(source_rows[role]):
            raise ValueError(f"Fact-review scope {role} record_count is stale")

    export_manifest = review_tool.read_json(export_path)
    items_path, templates_path = review_tool._validate_export_manifest(
        export_manifest, scope, scope_path
    )
    items = review_tool.read_jsonl(items_path)
    templates = review_tool.read_jsonl(templates_path)
    item_index = _unique_index(items, "base_fact_id", "fact-review items")
    template_index = _unique_index(
        templates, "base_fact_id", "fact-review decision templates"
    )
    if set(item_index) != set(full_ids) or set(template_index) != set(full_ids):
        raise ValueError("Fact-review export must exactly cover full_base_facts")
    for base_fact_id in full_ids:
        expected_item = review_tool._build_export_item(
            str(scope["scope_id"]), base_fact_id, source_indexes
        )
        if item_index[base_fact_id] != expected_item:
            raise ValueError(f"Fact-review item is stale: {base_fact_id}")
        if template_index[base_fact_id] != review_tool._decision_template(expected_item):
            raise ValueError(f"Fact-review decision template is stale: {base_fact_id}")

    decisions = review_tool.read_jsonl(decisions_path, allow_empty=True)
    if not isinstance(apply_manifest.get("decisions"), dict) or apply_manifest[
        "decisions"
    ].get("record_count") != len(decisions):
        raise ValueError("Fact-review decisions record_count is stale")
    decision_index = _unique_index(decisions, "base_fact_id", "fact-review decisions")
    if set(decision_index) != set(full_ids):
        raise ValueError("Complete fact-review decisions must exactly cover full_base_facts")
    for base_fact_id in full_ids:
        decision = decision_index[base_fact_id]
        if decision.get("decision") is None:
            raise ValueError(f"Missing fact-review decision: {base_fact_id}")
        review_tool._validate_decision(decision, item_index[base_fact_id])

    staging_rows = review_tool.read_jsonl(staging_path)
    if not isinstance(staging_binding, dict) or staging_binding.get("record_count") != len(
        staging_rows
    ):
        raise ValueError("Fact-review staging record_count is stale")
    staging_index = _unique_index(staging_rows, "base_fact_id", "fact-review staging")
    if set(staging_index) != set(full_ids):
        raise ValueError("Fact-review staging must exactly cover full_base_facts")
    replay_bindings = {
        "scope_manifest_sha256": sha256_file(scope_path),
        "export_manifest_sha256": sha256_file(export_path),
        "decisions_artifact_sha256": sha256_file(decisions_path),
        "apply_manifest_schema_version": review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
    }
    outcome_counts: Counter[str] = Counter()
    for base_fact_id in full_ids:
        expected = review_tool._staging_record(
            item_index[base_fact_id], decision_index[base_fact_id], replay_bindings
        )
        staging = staging_index[base_fact_id]
        if staging != expected:
            raise ValueError(f"Fact-review staging replay mismatch: {base_fact_id}")
        outcome = _manifest_outcome(staging)
        outcome_counts[outcome] += 1
        full_row = full_index[base_fact_id]
        behavior = item_index[base_fact_id]["review_context"]["behavior_row"]
        for field in ("subject_en", "relation_raw", "answer_en", "canonical_fact_en"):
            if full_row.get(field) != behavior.get(field) or staging.get(field) != full_row.get(field):
                raise ValueError(
                    f"Full-base/review-export/staging {field} mismatch: {base_fact_id}"
                )
        if full_row.get("answer_aliases_en") != behavior.get("answer_aliases_en"):
            raise ValueError(
                f"Full-base/review-export answer_aliases_en mismatch: {base_fact_id}"
            )
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
            _validate_aliases(
                staging.get("accepted_answer_aliases_en"),
                answer=str(full_row["answer_en"]),
                label=f"accepted aliases {base_fact_id}",
            )
    expected_counts = dict(sorted(outcome_counts.items()))
    if apply_manifest.get("outcome_counts") != expected_counts:
        raise ValueError("Fact-review apply outcome_counts are stale")

    snapshot_paths = {
        full_base_facts_path,
        review_apply_manifest_path,
        export_path,
        scope_path,
        decisions_path,
        staging_path,
        items_path,
        templates_path,
        *source_paths.values(),
    }
    bindings = {
        "full_base_facts": _input_binding(
            full_base_facts_path,
            schema_version=FULL_BASE_SCHEMA,
            record_count=len(full_rows),
        ),
        "review_apply_manifest": _input_binding(
            review_apply_manifest_path,
            schema_version=review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
        ),
        "review_export_manifest": _input_binding(
            export_path, schema_version=review_tool.EXPORT_MANIFEST_SCHEMA_VERSION
        ),
        "review_scope_manifest": _input_binding(scope_path),
        "review_decisions": _input_binding(
            decisions_path,
            schema_version=review_tool.DECISION_SCHEMA_VERSION,
            record_count=len(decisions),
        ),
        "review_staging": _input_binding(
            staging_path,
            schema_version=review_tool.STAGING_SCHEMA_VERSION,
            record_count=len(staging_rows),
        ),
        "review_items": _input_binding(
            items_path,
            schema_version=review_tool.EXPORT_ITEM_SCHEMA_VERSION,
            record_count=len(items),
        ),
        "review_decision_templates": _input_binding(
            templates_path,
            schema_version=review_tool.DECISION_SCHEMA_VERSION,
            record_count=len(templates),
        ),
    }
    return {
        "full_rows": full_rows,
        "full_index": full_index,
        "full_ids": full_ids,
        "item_index": item_index,
        "decision_index": decision_index,
        "staging_index": staging_index,
        "outcome_counts": expected_counts,
        "bindings": bindings,
        "snapshots": [_input_binding(path) for path in sorted(snapshot_paths)],
    }


def _resolution_item(base_fact_id: str, chain: Mapping[str, Any]) -> Dict[str, Any]:
    full_row = chain["full_index"][base_fact_id]
    staging = chain["staging_index"][base_fact_id]
    decision = chain["decision_index"][base_fact_id]
    outcome = _manifest_outcome(staging)
    if outcome not in NONACCEPT_OUTCOMES:
        raise ValueError(f"Resolution item requested for accept row: {base_fact_id}")
    allowed = ["cohort_exclude"]
    reject_override_eligibility = None
    if outcome == "reject":
        reject_override_eligibility = _source_reject_override_eligibility(
            source_row=full_row,
            item=chain["item_index"][base_fact_id],
            decision=decision,
        )
    if outcome == "revise" or (
        outcome == "reject" and reject_override_eligibility["eligible"] is True
    ):
        allowed.insert(0, "apply_revision")
    return {
        "schema_version": RESOLUTION_ITEM_SCHEMA,
        "tool_version": TOOL_VERSION,
        "base_fact_id": base_fact_id,
        "source_review_outcome": outcome,
        "allowed_dispositions": allowed,
        "source_bindings": {
            "full_fact_row_sha256": sha256_value(full_row),
            "review_item_sha256": sha256_value(chain["item_index"][base_fact_id]),
            "review_decision_record_sha256": sha256_value(decision),
            "review_staging_row_sha256": sha256_value(staging),
        },
        "source_values": {field: copy.deepcopy(full_row[field]) for field in MUTABLE_FIELDS},
        "source_review_evidence": {
            "review_provenance": copy.deepcopy(staging.get("review_provenance")),
            "review_completion": copy.deepcopy(staging.get("review_completion")),
            "member_reviews": copy.deepcopy(staging.get("member_reviews", [])),
            "alias_reviews": copy.deepcopy(staging.get("alias_reviews", [])),
            "distractor_reviews": copy.deepcopy(staging.get("distractor_reviews", [])),
            "notes": staging.get("notes"),
        },
        "source_reject_override_eligibility": reject_override_eligibility,
        "resolution_contract": {
            "revision_mutable_fields": list(MUTABLE_FIELDS),
            "revision_requires_all_field_reviews_accept": True,
            "revision_rereviewer_must_differ_from_editor": True,
            "eligible_source_reject_override_must_preserve_original_reject": True,
            "cohort_exclusion_asserts_factual_falsehood": False,
            "human_gold": False,
        },
    }


def _decision_template(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": RESOLUTION_DECISION_SCHEMA,
        "base_fact_id": item["base_fact_id"],
        "resolution_item_sha256": sha256_value(item),
        "disposition": None,
        "revision": None,
        "cohort_exclusion": None,
        "human_gold": False,
    }


def _materialized_rows(chain: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items = [
        _resolution_item(base_fact_id, chain)
        for base_fact_id in chain["full_ids"]
        if _manifest_outcome(chain["staging_index"][base_fact_id]) in NONACCEPT_OUTCOMES
    ]
    return items, [_decision_template(item) for item in items]


def _stage_directory(output_dir: Path) -> Tuple[Path, Path]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    return output_dir, staged


def _assert_snapshots_current(snapshots: Sequence[Mapping[str, Any]]) -> None:
    for binding in snapshots:
        path = Path(str(binding["path"])).resolve()
        if sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"Input changed during operation: {path}")


def materialize_template(
    *,
    full_base_facts_path: Path,
    review_apply_manifest_path: Path,
    review_export_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    chain = _load_review_chain(
        full_base_facts_path=full_base_facts_path,
        review_apply_manifest_path=review_apply_manifest_path,
        review_export_manifest_path=review_export_manifest_path,
    )
    items, templates = _materialized_rows(chain)
    output_path, staged = _stage_directory(output_dir)
    try:
        items_path = staged / "fact_resolution_items.jsonl"
        templates_path = staged / "fact_resolution_decisions_template.jsonl"
        write_jsonl(items_path, items)
        write_jsonl(templates_path, templates)
        counts = Counter(item["source_review_outcome"] for item in items)
        manifest = {
            "schema_version": TEMPLATE_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "resolution_template_materialized",
            "inputs": copy.deepcopy(chain["bindings"]),
            "source_record_count": len(chain["full_rows"]),
            "source_review_outcome_counts": copy.deepcopy(chain["outcome_counts"]),
            "resolution_item_count": len(items),
            "resolution_item_outcome_counts": dict(sorted(counts.items())),
            "artifacts": {
                "resolution_items": _output_binding(
                    items_path,
                    schema_version=RESOLUTION_ITEM_SCHEMA,
                    record_count=len(items),
                ),
                "resolution_decisions_template": _output_binding(
                    templates_path,
                    schema_version=RESOLUTION_DECISION_SCHEMA,
                    record_count=len(templates),
                ),
            },
            "resolution_contract": RESOLUTION_CONTRACT,
            "safety_contract": SAFETY_CONTRACT,
        }
        manifest_path = staged / "fact_resolution_template_manifest.json"
        write_json(manifest_path, manifest)
        summary = {
            "schema_version": TEMPLATE_SUMMARY_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "resolution_template_materialized",
            "source_record_count": len(chain["full_rows"]),
            "source_review_outcome_counts": copy.deepcopy(chain["outcome_counts"]),
            "resolution_item_count": len(items),
            "resolution_item_outcome_counts": dict(sorted(counts.items())),
            "manifest": _output_binding(
                manifest_path, schema_version=TEMPLATE_MANIFEST_SCHEMA
            ),
            "resolution_contract": RESOLUTION_CONTRACT,
            "safety_contract": SAFETY_CONTRACT,
        }
        summary_path = staged / "summary.json"
        write_json(summary_path, summary)
        _assert_snapshots_current(chain["snapshots"])
        os.replace(staged, output_path)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_path),
        "manifest_path": str(output_path / manifest_path.name),
        "manifest_sha256": sha256_file(output_path / manifest_path.name),
        "summary_path": str(output_path / summary_path.name),
        "resolution_item_count": len(items),
    }


def _load_template_context(template_manifest_path: Path) -> Dict[str, Any]:
    template_manifest_path = Path(template_manifest_path).resolve()
    manifest = read_json(template_manifest_path)
    if manifest.get("schema_version") != TEMPLATE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported fact-resolution template manifest schema_version")
    if manifest.get("resolution_contract") != RESOLUTION_CONTRACT:
        raise ValueError("Fact-resolution template contract is invalid")
    if manifest.get("safety_contract") != SAFETY_CONTRACT:
        raise ValueError("Fact-resolution template safety contract is invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Fact-resolution template inputs are missing")
    full_path = _verify_binding(
        inputs.get("full_base_facts"),
        owner_path=template_manifest_path,
        label="template full_base_facts",
        expected_schema=FULL_BASE_SCHEMA,
    )
    apply_path = _verify_binding(
        inputs.get("review_apply_manifest"),
        owner_path=template_manifest_path,
        label="template review_apply_manifest",
        expected_schema=review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
    )
    export_path = _verify_binding(
        inputs.get("review_export_manifest"),
        owner_path=template_manifest_path,
        label="template review_export_manifest",
        expected_schema=review_tool.EXPORT_MANIFEST_SCHEMA_VERSION,
    )
    chain = _load_review_chain(
        full_base_facts_path=full_path,
        review_apply_manifest_path=apply_path,
        review_export_manifest_path=export_path,
    )
    if manifest.get("inputs") != chain["bindings"]:
        raise ValueError("Fact-resolution template input bindings are stale")
    expected_items, expected_templates = _materialized_rows(chain)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Fact-resolution template artifacts are missing")
    items_path = _verify_binding(
        artifacts.get("resolution_items"),
        owner_path=template_manifest_path,
        label="resolution_items",
        expected_schema=RESOLUTION_ITEM_SCHEMA,
    )
    templates_path = _verify_binding(
        artifacts.get("resolution_decisions_template"),
        owner_path=template_manifest_path,
        label="resolution_decisions_template",
        expected_schema=RESOLUTION_DECISION_SCHEMA,
    )
    actual_items = read_jsonl(items_path, allow_empty=True)
    actual_templates = read_jsonl(templates_path, allow_empty=True)
    if actual_items != expected_items:
        raise ValueError("Resolution items do not replay from the bound review chain")
    if actual_templates != expected_templates:
        raise ValueError("Resolution decision templates do not replay from their items")
    for binding, rows, label in (
        (artifacts["resolution_items"], actual_items, "resolution_items"),
        (
            artifacts["resolution_decisions_template"],
            actual_templates,
            "resolution_decisions_template",
        ),
    ):
        if binding.get("record_count") != len(rows):
            raise ValueError(f"{label} record_count is stale")
    counts = Counter(item["source_review_outcome"] for item in expected_items)
    if manifest.get("source_record_count") != len(chain["full_rows"]):
        raise ValueError("Template source_record_count is stale")
    if manifest.get("source_review_outcome_counts") != chain["outcome_counts"]:
        raise ValueError("Template source_review_outcome_counts are stale")
    if manifest.get("resolution_item_count") != len(expected_items):
        raise ValueError("Template resolution_item_count is stale")
    if manifest.get("resolution_item_outcome_counts") != dict(sorted(counts.items())):
        raise ValueError("Template resolution_item_outcome_counts are stale")
    return {
        "manifest": manifest,
        "manifest_path": template_manifest_path,
        "chain": chain,
        "items": expected_items,
        "templates": expected_templates,
        "items_path": items_path,
        "templates_path": templates_path,
    }


def _validate_person_provenance(
    value: Any,
    *,
    label: str,
    type_field: str,
    id_field: str,
    method_field: str,
    time_field: str,
    expected_fields: frozenset[str],
    allowed_types: frozenset[str],
) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{label} fields are invalid")
    if value.get(type_field) not in allowed_types:
        raise ValueError(f"{label}.{type_field} is invalid")
    _required_string(value.get(id_field), f"{label}.{id_field}")
    _required_string(value.get(method_field), f"{label}.{method_field}")
    _validate_timestamp(value.get(time_field), f"{label}.{time_field}")
    if value.get("human_gold") is not False:
        raise ValueError(f"{label} must have human_gold=false")
    return copy.deepcopy(value)


def _validate_revision(
    value: Any,
    *,
    base_fact_id: str,
    source_row: Mapping[str, Any],
    resolution_item: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != REVISION_FIELDS:
        raise ValueError(f"Revision fields are invalid: {base_fact_id}")
    updates = value.get("updates")
    if not isinstance(updates, dict):
        raise ValueError(f"Revision updates must be an object: {base_fact_id}")
    source_outcome = resolution_item.get("source_review_outcome")
    if not updates and source_outcome != "reject":
        raise ValueError(f"Revision updates must be non-empty: {base_fact_id}")
    unsupported = sorted(set(updates) - set(MUTABLE_FIELDS))
    if unsupported:
        raise ValueError(
            f"Revision updates immutable fields for {base_fact_id}: {unsupported}"
        )
    for field, update in updates.items():
        if canonical_json_bytes(update) == canonical_json_bytes(source_row.get(field)):
            raise ValueError(f"Revision does not change {field}: {base_fact_id}")
        if field in STRING_MUTABLE_FIELDS:
            _required_string(update, f"revision {base_fact_id}.{field}")
    if "answer_en" in updates and "answer_aliases_en" not in updates:
        raise ValueError(
            f"Revision changing answer_en must update answer_aliases_en: {base_fact_id}"
        )
    revised = copy.deepcopy(dict(source_row))
    revised.update(copy.deepcopy(updates))
    for field in STRING_MUTABLE_FIELDS:
        _required_string(revised.get(field), f"revised {base_fact_id}.{field}")
    revised["answer_aliases_en"] = _validate_aliases(
        revised.get("answer_aliases_en"),
        answer=str(revised["answer_en"]),
        label=f"revised {base_fact_id}.answer_aliases_en",
    )
    if revised.get("human_gold") is not False:
        raise ValueError(f"Revised row must retain human_gold=false: {base_fact_id}")

    reject_override = value.get("source_reject_override")
    if source_outcome == "reject":
        eligibility = resolution_item.get("source_reject_override_eligibility")
        if not isinstance(eligibility, dict) or eligibility.get("eligible") is not True:
            raise ValueError(f"Source reject is not eligible for revision override: {base_fact_id}")
        if not isinstance(reject_override, dict) or set(reject_override) != SOURCE_REJECT_OVERRIDE_FIELDS:
            raise ValueError(f"Source reject override fields are invalid: {base_fact_id}")
        if reject_override.get("original_review_outcome") != "reject":
            raise ValueError(f"Source reject override loses original outcome: {base_fact_id}")
        if reject_override.get("eligibility_evidence_sha256") != sha256_value(eligibility):
            raise ValueError(f"Source reject override evidence SHA is stale: {base_fact_id}")
        _required_string(
            reject_override.get("override_reason"),
            f"revision {base_fact_id}.source_reject_override.override_reason",
        )
        target_resolutions = reject_override.get("target_resolutions")
        if not isinstance(target_resolutions, dict) or set(target_resolutions) != TARGET_RESOLUTION_FIELDS:
            raise ValueError(f"Source reject target resolutions are invalid: {base_fact_id}")
        if target_resolutions != eligibility.get("target_resolutions"):
            raise ValueError(f"Source reject target resolutions are stale: {base_fact_id}")
        rejected_alias_ids = set(target_resolutions["rejected_alias_ids"])
        if rejected_alias_ids and "answer_aliases_en" not in updates:
            raise ValueError(
                f"Reject override must edit answer_aliases_en to remove rejected aliases: "
                f"{base_fact_id}"
            )
        alias_targets = {
            str(target["alias_id"]): str(target["text_en"])
            for target in resolution_item.get("source_review_evidence", {}).get(
                "alias_reviews", []
            )
        }
        rejected_alias_text = {
            normalize_text(alias_targets[alias_id])
            for alias_id in rejected_alias_ids
            if alias_id in alias_targets
        }
        if rejected_alias_text.intersection(
            normalize_text(alias) for alias in revised["answer_aliases_en"]
        ):
            raise ValueError(
                f"Reject override retains a rejected answer alias: {base_fact_id}"
            )
    elif source_outcome == "revise":
        if reject_override is not None:
            raise ValueError(
                f"Revise outcome must not invent a source reject override: {base_fact_id}"
            )
    else:
        raise ValueError(f"Unsupported revision source outcome: {base_fact_id}")

    _required_string(value.get("revision_reason"), f"revision {base_fact_id}.revision_reason")
    editor = _validate_person_provenance(
        value.get("editor_provenance"),
        label=f"revision {base_fact_id}.editor_provenance",
        type_field="editor_type",
        id_field="editor_id",
        method_field="edit_method",
        time_field="edited_at",
        expected_fields=EDITOR_PROVENANCE_FIELDS,
        allowed_types=EDITOR_TYPES,
    )
    rereview = value.get("rereview")
    if not isinstance(rereview, dict) or set(rereview) != REREVIEW_FIELDS:
        raise ValueError(f"Revision rereview fields are invalid: {base_fact_id}")
    if rereview.get("decision") != "accept":
        raise ValueError(f"Revision must pass independent rereview: {base_fact_id}")
    if rereview.get("revised_full_fact_row_sha256") != sha256_value(revised):
        raise ValueError(f"Revision rereview row SHA is stale: {base_fact_id}")
    field_reviews = rereview.get("field_reviews")
    if not isinstance(field_reviews, dict) or set(field_reviews) != set(MUTABLE_FIELDS):
        raise ValueError(f"Revision field_reviews are incomplete: {base_fact_id}")
    for field in MUTABLE_FIELDS:
        field_review = field_reviews[field]
        if not isinstance(field_review, dict) or set(field_review) != FIELD_REVIEW_FIELDS:
            raise ValueError(
                f"Revision field review fields are invalid: {base_fact_id}.{field}"
            )
        if field_review.get("decision") != "accept":
            raise ValueError(
                f"Revision field review must accept {field}: {base_fact_id}"
            )
        _required_string(
            field_review.get("reason"),
            f"revision {base_fact_id}.field_reviews.{field}.reason",
        )
    reviewer = _validate_person_provenance(
        rereview,
        label=f"revision {base_fact_id}.rereview",
        type_field="reviewer_type",
        id_field="reviewer_id",
        method_field="review_method",
        time_field="reviewed_at",
        expected_fields=REREVIEW_FIELDS,
        allowed_types=REREVIEWER_TYPES,
    )
    _required_string(rereview.get("rationale"), f"revision {base_fact_id}.rereview.rationale")
    if reviewer["reviewer_id"] == editor["editor_id"]:
        raise ValueError(
            f"Revision editor and rereviewer must be independent: {base_fact_id}"
        )
    if source_outcome == "reject":
        source_review_provenance = resolution_item.get(
            "source_review_evidence", {}
        ).get("review_provenance")
        if isinstance(source_review_provenance, dict) and reviewer[
            "reviewer_id"
        ] == source_review_provenance.get("reviewer_id"):
            raise ValueError(
                "Reject override rereviewer must differ from the source reject "
                f"reviewer: {base_fact_id}"
            )
    return revised, {
        "updates": copy.deepcopy(updates),
        "revision_reason": value["revision_reason"],
        "source_reject_override": copy.deepcopy(reject_override),
        "editor_provenance": editor,
        "rereview": reviewer,
    }


def _validate_exclusion(value: Any, *, base_fact_id: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != COHORT_EXCLUSION_FIELDS:
        raise ValueError(f"Cohort exclusion fields are invalid: {base_fact_id}")
    reason = _required_string(
        value.get("reason"), f"cohort exclusion {base_fact_id}.reason"
    )
    if value.get("scope") != EXCLUSION_SCOPE:
        raise ValueError(f"Cohort exclusion scope is invalid: {base_fact_id}")
    if value.get("factual_falsehood_asserted") is not False:
        raise ValueError(
            f"Cohort exclusion must not assert factual falsehood: {base_fact_id}"
        )
    provenance = _validate_person_provenance(
        value.get("resolution_provenance"),
        label=f"cohort exclusion {base_fact_id}.resolution_provenance",
        type_field="reviewer_type",
        id_field="reviewer_id",
        method_field="review_method",
        time_field="reviewed_at",
        expected_fields=RESOLUTION_PROVENANCE_FIELDS,
        allowed_types=REREVIEWER_TYPES,
    )
    return {
        "reason": reason,
        "scope": EXCLUSION_SCOPE,
        "factual_falsehood_asserted": False,
        "resolution_provenance": provenance,
    }


def _load_resolution_decisions(
    path: Path,
    *,
    context: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path, allow_empty=True)
    index = _unique_index(rows, "base_fact_id", "fact resolutions")
    item_index = _unique_index(context["items"], "base_fact_id", "resolution items")
    if set(index) != set(item_index):
        missing = sorted(set(item_index) - set(index))
        extra = sorted(set(index) - set(item_index))
        raise ValueError(
            "Resolution decisions must exactly cover every non-accept review; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    validated: Dict[str, Dict[str, Any]] = {}
    chain = context["chain"]
    for base_fact_id in [str(item["base_fact_id"]) for item in context["items"]]:
        row = index[base_fact_id]
        item = item_index[base_fact_id]
        if set(row) != DECISION_FIELDS:
            raise ValueError(f"Resolution decision fields are invalid: {base_fact_id}")
        if row.get("schema_version") != RESOLUTION_DECISION_SCHEMA:
            raise ValueError(f"Resolution decision schema is invalid: {base_fact_id}")
        if row.get("resolution_item_sha256") != sha256_value(item):
            raise ValueError(f"Resolution item SHA is stale: {base_fact_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"Resolution must have human_gold=false: {base_fact_id}")
        disposition = row.get("disposition")
        if disposition not in item["allowed_dispositions"]:
            raise ValueError(f"Resolution disposition is invalid: {base_fact_id}")
        if disposition == "apply_revision":
            if item["source_review_outcome"] not in {"revise", "reject"}:
                raise ValueError(
                    f"Only revise or an eligible reject may apply a revision: {base_fact_id}"
                )
            if item["source_review_outcome"] == "reject" and not item.get(
                "source_reject_override_eligibility", {}
            ).get("eligible"):
                raise ValueError(
                    f"Source reject is not eligible for revision override: {base_fact_id}"
                )
            if row.get("cohort_exclusion") is not None:
                raise ValueError(
                    f"Revision cannot also be a cohort exclusion: {base_fact_id}"
                )
            revised, revision = _validate_revision(
                row.get("revision"),
                base_fact_id=base_fact_id,
                source_row=chain["full_index"][base_fact_id],
                resolution_item=item,
            )
            validated[base_fact_id] = {
                "decision": copy.deepcopy(row),
                "disposition": disposition,
                "revised_row": revised,
                "revision": revision,
                "cohort_exclusion": None,
            }
        else:
            if row.get("revision") is not None:
                raise ValueError(
                    f"Cohort exclusion cannot also carry a revision: {base_fact_id}"
                )
            exclusion = _validate_exclusion(
                row.get("cohort_exclusion"), base_fact_id=base_fact_id
            )
            validated[base_fact_id] = {
                "decision": copy.deepcopy(row),
                "disposition": disposition,
                "revised_row": None,
                "revision": None,
                "cohort_exclusion": exclusion,
            }
    return validated


def _accepted_row(
    source: Mapping[str, Any], staging: Mapping[str, Any], *, base_fact_id: str
) -> Dict[str, Any]:
    accepted = staging.get("accepted_answer_aliases_en")
    accepted_values = _validate_aliases(
        accepted,
        answer=str(source["answer_en"]),
        label=f"accepted aliases {base_fact_id}",
    )
    accepted_norm = {normalize_text(value) for value in accepted_values}
    filtered = [
        value
        for value in source["answer_aliases_en"]
        if normalize_text(value) in accepted_norm
    ]
    output = copy.deepcopy(dict(source))
    output["answer_aliases_en"] = _validate_aliases(
        filtered,
        answer=str(output["answer_en"]),
        label=f"retained aliases {base_fact_id}",
    )
    return output


def _resolve_outputs(
    context: Mapping[str, Any], decisions: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    chain = context["chain"]
    retained: List[Dict[str, Any]] = []
    ledger: List[Dict[str, Any]] = []
    lineages: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    alias_filter_changes = 0
    disposition_counts: Counter[str] = Counter()
    for base_fact_id in chain["full_ids"]:
        source = chain["full_index"][base_fact_id]
        staging = chain["staging_index"][base_fact_id]
        source_outcome = _manifest_outcome(staging)
        common = {
            "base_fact_id": base_fact_id,
            "source_review_outcome": source_outcome,
            "source_full_fact_row_sha256": sha256_value(source),
            "source_review_staging_row_sha256": sha256_value(staging),
            "source_review_decision_record_sha256": sha256_value(
                chain["decision_index"][base_fact_id]
            ),
            "human_gold": False,
        }
        if source_outcome == "accept":
            output_row = _accepted_row(source, staging, base_fact_id=base_fact_id)
            if output_row["answer_aliases_en"] != source["answer_aliases_en"]:
                alias_filter_changes += 1
            retained.append(output_row)
            disposition_counts["retain_accepted"] += 1
            ledger.append(
                {
                    "schema_version": RESOLUTION_LEDGER_SCHEMA,
                    **common,
                    "final_disposition": "retain_accepted",
                    "resolution_decision_row_sha256": None,
                    "output_full_fact_row_sha256": sha256_value(output_row),
                    "independent_revision_rereview_complete": False,
                    "source_reject_override_applied": False,
                    "source_reject_override_reason": None,
                    "cohort_exclusion_asserts_factual_falsehood": False,
                }
            )
            continue

        resolution = decisions[base_fact_id]
        resolution_sha = sha256_value(resolution["decision"])
        if resolution["disposition"] == "apply_revision":
            output_row = copy.deepcopy(resolution["revised_row"])
            retained.append(output_row)
            disposition_counts["retain_revised"] += 1
            revision = resolution["revision"]
            lineage = {
                "schema_version": REVISION_LINEAGE_SCHEMA,
                **common,
                "resolution_decision_row_sha256": resolution_sha,
                "before_full_fact_row_sha256": sha256_value(source),
                "after_full_fact_row_sha256": sha256_value(output_row),
                "changed_fields": sorted(revision["updates"]),
                "updates": copy.deepcopy(revision["updates"]),
                "revision_reason": revision["revision_reason"],
                "source_reject_override": copy.deepcopy(
                    revision["source_reject_override"]
                ),
                "editor_provenance": copy.deepcopy(revision["editor_provenance"]),
                "rereview": copy.deepcopy(revision["rereview"]),
                "canonical_freeze_performed": False,
                "split_freeze_performed": False,
            }
            lineages.append(lineage)
            ledger.append(
                {
                    "schema_version": RESOLUTION_LEDGER_SCHEMA,
                    **common,
                    "final_disposition": "retain_revised",
                    "resolution_decision_row_sha256": resolution_sha,
                    "output_full_fact_row_sha256": sha256_value(output_row),
                    "independent_revision_rereview_complete": True,
                    "source_reject_override_applied": source_outcome == "reject",
                    "source_reject_override_reason": (
                        revision["source_reject_override"]["override_reason"]
                        if source_outcome == "reject"
                        else None
                    ),
                    "cohort_exclusion_asserts_factual_falsehood": False,
                }
            )
        else:
            disposition_counts["cohort_exclude"] += 1
            exclusion = resolution["cohort_exclusion"]
            exclusions.append(
                {
                    "schema_version": COHORT_EXCLUSION_SCHEMA,
                    **common,
                    "resolution_decision_row_sha256": resolution_sha,
                    "cohort_eligibility_disposition": "cohort_exclude",
                    "cohort_scope": EXCLUSION_SCOPE,
                    "cohort_exclusion_reason": exclusion["reason"],
                    "factual_falsehood_asserted": False,
                    "source_review_outcome_preserved_as_proxy_evidence": True,
                    "resolution_provenance": copy.deepcopy(
                        exclusion["resolution_provenance"]
                    ),
                }
            )
            ledger.append(
                {
                    "schema_version": RESOLUTION_LEDGER_SCHEMA,
                    **common,
                    "final_disposition": "cohort_exclude",
                    "resolution_decision_row_sha256": resolution_sha,
                    "output_full_fact_row_sha256": None,
                    "independent_revision_rereview_complete": False,
                    "source_reject_override_applied": False,
                    "source_reject_override_reason": None,
                    "cohort_exclusion_asserts_factual_falsehood": False,
                }
            )
    return {
        "full_rows": retained,
        "ledger": ledger,
        "lineages": lineages,
        "exclusions": exclusions,
        "source_reject_override_count": sum(
            row["source_review_outcome"] == "reject" for row in lineages
        ),
        "alias_filter_change_count": alias_filter_changes,
        "disposition_counts": dict(sorted(disposition_counts.items())),
    }


def _validate_prior_semantic_manifest(
    path: Optional[Path], *, source_full_binding: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    path = Path(path).resolve()
    manifest = read_json(path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Prior semantic candidate manifest inputs are missing")
    binding = inputs.get("current_full_base_facts")
    if not isinstance(binding, dict):
        raise ValueError(
            "Prior semantic candidate manifest does not bind current_full_base_facts"
        )
    bound_path = _resolved_path(binding, path, "prior semantic current_full_base_facts")
    source_path = Path(str(source_full_binding["path"])).resolve()
    if bound_path != source_path or binding.get("sha256") != source_full_binding.get("sha256"):
        raise ValueError(
            "Prior semantic candidate manifest is not bound to source full_base_facts"
        )
    if sha256_file(bound_path) != binding.get("sha256"):
        raise ValueError("Prior semantic candidate full-base binding is stale")
    return _input_binding(path, schema_version=str(manifest.get("schema_version")))


def _semantic_invalidation(
    *,
    source_sha: str,
    output_sha: str,
    revision_count: int,
    exclusion_count: int,
    alias_filter_change_count: int,
    prior_manifest_bound: bool,
) -> Dict[str, Any]:
    changed = source_sha != output_sha
    resolution_state_changed = bool(
        revision_count or exclusion_count or alias_filter_change_count
    )
    requires_rematerialization = changed or resolution_state_changed
    return {
        "source_full_base_facts_sha256": source_sha,
        "resolved_full_base_facts_sha256": output_sha,
        "source_and_resolved_sha256_differ": changed,
        "resolution_state_changed": resolution_state_changed,
        "applied_revision_count": revision_count,
        "cohort_exclusion_count": exclusion_count,
        "accepted_alias_filter_change_count": alias_filter_change_count,
        "specific_prior_semantic_candidate_manifest_bound": prior_manifest_bound,
        "prior_semantic_candidate_manifest_valid_for_resolved_facts": not requires_rematerialization,
        "prior_semantic_candidate_manifest_invalidated": requires_rematerialization,
        "semantic_candidates_require_rematerialization": requires_rematerialization,
        "base_fact_clusters_require_rematerialization": requires_rematerialization,
        "review_export_requires_rematerialization": requires_rematerialization,
        "source_target_resolutions_must_be_applied": requires_rematerialization,
        "leakage_components_require_rematerialization": requires_rematerialization,
        "split_assignments_require_recomputation": requires_rematerialization,
        "distractor_candidates_require_regeneration": requires_rematerialization,
        "neutral_candidates_require_regeneration": requires_rematerialization,
        "translations_for_revised_or_excluded_universe_require_rebinding": requires_rematerialization,
        "old_semantic_decisions_must_not_be_rebound_by_pair_id_alone": requires_rematerialization,
        "invalidation_basis": (
            "resolved full-base SHA differs from the source SHA to which prior "
            "semantic candidates were bound"
            if changed
            else (
                "resolution lineage changed and the resolved artifact path/contract "
                "is not the source artifact bound by prior semantic candidates"
                if resolution_state_changed
                else "resolved full-base bytes and resolution state equal the source"
            )
        ),
    }


def apply_resolutions(
    *,
    template_manifest_path: Path,
    resolutions_path: Path,
    output_dir: Path,
    prior_semantic_candidate_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    context = _load_template_context(template_manifest_path)
    resolutions_path = Path(resolutions_path).resolve()
    decisions = _load_resolution_decisions(resolutions_path, context=context)
    outputs = _resolve_outputs(context, decisions)
    prior_binding = _validate_prior_semantic_manifest(
        prior_semantic_candidate_manifest_path,
        source_full_binding=context["chain"]["bindings"]["full_base_facts"],
    )
    output_path, staged = _stage_directory(output_dir)
    try:
        full_path = staged / "revised_full_base_facts.jsonl"
        ledger_path = staged / "fact_resolution_ledger.jsonl"
        lineage_path = staged / "fact_revision_lineage.jsonl"
        exclusion_path = staged / "cohort_exclusions.jsonl"
        write_jsonl(full_path, outputs["full_rows"])
        write_jsonl(ledger_path, outputs["ledger"])
        write_jsonl(lineage_path, outputs["lineages"])
        write_jsonl(exclusion_path, outputs["exclusions"])
        source_binding = context["chain"]["bindings"]["full_base_facts"]
        invalidation = _semantic_invalidation(
            source_sha=str(source_binding["sha256"]),
            output_sha=sha256_file(full_path),
            revision_count=len(outputs["lineages"]),
            exclusion_count=len(outputs["exclusions"]),
            alias_filter_change_count=int(outputs["alias_filter_change_count"]),
            prior_manifest_bound=prior_binding is not None,
        )
        manifest_inputs: Dict[str, Any] = {
            "template_manifest": _input_binding(
                Path(template_manifest_path).resolve(),
                schema_version=TEMPLATE_MANIFEST_SCHEMA,
            ),
            "resolution_decisions": _input_binding(
                resolutions_path,
                schema_version=RESOLUTION_DECISION_SCHEMA,
                record_count=len(decisions),
            ),
            **copy.deepcopy(context["chain"]["bindings"]),
        }
        if prior_binding is not None:
            manifest_inputs["prior_semantic_candidate_manifest"] = prior_binding
        manifest = {
            "schema_version": OUTPUT_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "completed_provisional_offline_fact_resolution",
            "inputs": manifest_inputs,
            "counts": {
                "source_record_count": len(context["chain"]["full_rows"]),
                "source_review_outcome_counts": copy.deepcopy(
                    context["chain"]["outcome_counts"]
                ),
                "resolution_decision_count": len(decisions),
                "retained_record_count": len(outputs["full_rows"]),
                "revision_count": len(outputs["lineages"]),
                "source_reject_override_count": int(
                    outputs["source_reject_override_count"]
                ),
                "cohort_exclusion_count": len(outputs["exclusions"]),
                "accepted_alias_filter_change_count": int(
                    outputs["alias_filter_change_count"]
                ),
                "final_disposition_counts": copy.deepcopy(
                    outputs["disposition_counts"]
                ),
                "unresolved_count": 0,
            },
            "artifacts": {
                "revised_full_base_facts": _output_binding(
                    full_path,
                    schema_version=FULL_BASE_SCHEMA,
                    record_count=len(outputs["full_rows"]),
                ),
                "fact_resolution_ledger": _output_binding(
                    ledger_path,
                    schema_version=RESOLUTION_LEDGER_SCHEMA,
                    record_count=len(outputs["ledger"]),
                ),
                "fact_revision_lineage": _output_binding(
                    lineage_path,
                    schema_version=REVISION_LINEAGE_SCHEMA,
                    record_count=len(outputs["lineages"]),
                ),
                "cohort_exclusions": _output_binding(
                    exclusion_path,
                    schema_version=COHORT_EXCLUSION_SCHEMA,
                    record_count=len(outputs["exclusions"]),
                ),
            },
            "completion_contract": {
                "source_review_chain_replayed": True,
                "source_review_scope_complete": True,
                "all_nonaccept_rows_resolved": True,
                "all_retained_revisions_independently_rereviewed": True,
                "unresolved_count": 0,
                "formal_fact_or_split_freeze_completed": False,
            },
            "semantic_closure_invalidation": invalidation,
            "resolution_contract": RESOLUTION_CONTRACT,
            "safety_contract": SAFETY_CONTRACT,
        }
        manifest_path = staged / "fact_resolution_manifest.json"
        write_json(manifest_path, manifest)
        summary = {
            "schema_version": OUTPUT_SUMMARY_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": manifest["status"],
            "counts": copy.deepcopy(manifest["counts"]),
            "semantic_closure_invalidation": invalidation,
            "manifest": _output_binding(
                manifest_path, schema_version=OUTPUT_MANIFEST_SCHEMA
            ),
            "safety_contract": SAFETY_CONTRACT,
        }
        summary_path = staged / "summary.json"
        write_json(summary_path, summary)
        _assert_snapshots_current(context["chain"]["snapshots"])
        if sha256_file(resolutions_path) != manifest_inputs["resolution_decisions"]["sha256"]:
            raise ValueError("Resolution decisions changed during apply")
        if prior_binding is not None and sha256_file(
            Path(str(prior_binding["path"]))
        ) != prior_binding["sha256"]:
            raise ValueError("Prior semantic candidate manifest changed during apply")
        os.replace(staged, output_path)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    final_manifest = output_path / manifest_path.name
    validate_resolution_output(final_manifest)
    return {
        "output_dir": str(output_path),
        "manifest_path": str(final_manifest),
        "manifest_sha256": sha256_file(final_manifest),
        "summary_path": str(output_path / summary_path.name),
        "retained_record_count": len(outputs["full_rows"]),
        "revision_count": len(outputs["lineages"]),
        "source_reject_override_count": int(outputs["source_reject_override_count"]),
        "cohort_exclusion_count": len(outputs["exclusions"]),
    }


def validate_resolution_output(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != OUTPUT_MANIFEST_SCHEMA:
        raise ValueError("Unsupported fact-resolution output manifest schema_version")
    if manifest.get("resolution_contract") != RESOLUTION_CONTRACT:
        raise ValueError("Fact-resolution output contract is invalid")
    if manifest.get("safety_contract") != SAFETY_CONTRACT:
        raise ValueError("Fact-resolution output safety contract is invalid")
    if manifest.get("status") != "completed_provisional_offline_fact_resolution":
        raise ValueError("Fact-resolution output status is invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Fact-resolution output inputs are missing")
    template_path = _verify_binding(
        inputs.get("template_manifest"),
        owner_path=manifest_path,
        label="output template_manifest",
        expected_schema=TEMPLATE_MANIFEST_SCHEMA,
    )
    resolutions_path = _verify_binding(
        inputs.get("resolution_decisions"),
        owner_path=manifest_path,
        label="output resolution_decisions",
        expected_schema=RESOLUTION_DECISION_SCHEMA,
    )
    context = _load_template_context(template_path)
    decisions = _load_resolution_decisions(resolutions_path, context=context)
    expected = _resolve_outputs(context, decisions)
    for role, binding in context["chain"]["bindings"].items():
        if inputs.get(role) != binding:
            raise ValueError(f"Output source binding is stale: {role}")
    prior_binding = None
    if "prior_semantic_candidate_manifest" in inputs:
        prior_path = _verify_binding(
            inputs["prior_semantic_candidate_manifest"],
            owner_path=manifest_path,
            label="output prior_semantic_candidate_manifest",
        )
        prior_binding = _validate_prior_semantic_manifest(
            prior_path,
            source_full_binding=context["chain"]["bindings"]["full_base_facts"],
        )
        if prior_binding != inputs["prior_semantic_candidate_manifest"]:
            raise ValueError("Prior semantic candidate manifest binding is stale")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Fact-resolution output artifacts are missing")
    specs = (
        (
            "revised_full_base_facts",
            FULL_BASE_SCHEMA,
            expected["full_rows"],
        ),
        ("fact_resolution_ledger", RESOLUTION_LEDGER_SCHEMA, expected["ledger"]),
        ("fact_revision_lineage", REVISION_LINEAGE_SCHEMA, expected["lineages"]),
        ("cohort_exclusions", COHORT_EXCLUSION_SCHEMA, expected["exclusions"]),
    )
    paths: Dict[str, Path] = {}
    for role, schema, rows in specs:
        path = _verify_binding(
            artifacts.get(role),
            owner_path=manifest_path,
            label=f"output {role}",
            expected_schema=schema,
        )
        actual = read_jsonl(path, allow_empty=True)
        if actual != rows:
            raise ValueError(f"Output {role} does not replay from bound inputs")
        if artifacts[role].get("record_count") != len(actual):
            raise ValueError(f"Output {role} record_count is stale")
        paths[role] = path
    counts = {
        "source_record_count": len(context["chain"]["full_rows"]),
        "source_review_outcome_counts": copy.deepcopy(context["chain"]["outcome_counts"]),
        "resolution_decision_count": len(decisions),
        "retained_record_count": len(expected["full_rows"]),
        "revision_count": len(expected["lineages"]),
        "source_reject_override_count": int(expected["source_reject_override_count"]),
        "cohort_exclusion_count": len(expected["exclusions"]),
        "accepted_alias_filter_change_count": int(expected["alias_filter_change_count"]),
        "final_disposition_counts": copy.deepcopy(expected["disposition_counts"]),
        "unresolved_count": 0,
    }
    if manifest.get("counts") != counts:
        raise ValueError("Fact-resolution output counts are stale")
    expected_invalidation = _semantic_invalidation(
        source_sha=str(context["chain"]["bindings"]["full_base_facts"]["sha256"]),
        output_sha=sha256_file(paths["revised_full_base_facts"]),
        revision_count=len(expected["lineages"]),
        exclusion_count=len(expected["exclusions"]),
        alias_filter_change_count=int(expected["alias_filter_change_count"]),
        prior_manifest_bound=prior_binding is not None,
    )
    if manifest.get("semantic_closure_invalidation") != expected_invalidation:
        raise ValueError("Semantic-closure invalidation contract is stale")
    completion = manifest.get("completion_contract")
    if completion != {
        "source_review_chain_replayed": True,
        "source_review_scope_complete": True,
        "all_nonaccept_rows_resolved": True,
        "all_retained_revisions_independently_rereviewed": True,
        "unresolved_count": 0,
        "formal_fact_or_split_freeze_completed": False,
    }:
        raise ValueError("Fact-resolution completion contract is invalid")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "retained_record_count": counts["retained_record_count"],
        "revision_count": counts["revision_count"],
        "source_reject_override_count": counts["source_reject_override_count"],
        "cohort_exclusion_count": counts["cohort_exclusion_count"],
        "validated": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-template")
    materialize.add_argument("--full-base-facts", required=True, type=Path)
    materialize.add_argument("--review-apply-manifest", required=True, type=Path)
    materialize.add_argument("--review-export-manifest", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--template-manifest", required=True, type=Path)
    apply_parser.add_argument("--resolutions", required=True, type=Path)
    apply_parser.add_argument("--output-dir", required=True, type=Path)
    apply_parser.add_argument("--prior-semantic-candidate-manifest", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "materialize-template":
            result = materialize_template(
                full_base_facts_path=args.full_base_facts,
                review_apply_manifest_path=args.review_apply_manifest,
                review_export_manifest_path=args.review_export_manifest,
                output_dir=args.output_dir,
            )
        elif args.command == "apply":
            result = apply_resolutions(
                template_manifest_path=args.template_manifest,
                resolutions_path=args.resolutions,
                output_dir=args.output_dir,
                prior_semantic_candidate_manifest_path=(
                    args.prior_semantic_candidate_manifest
                ),
            )
        else:
            result = validate_resolution_output(args.manifest)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
