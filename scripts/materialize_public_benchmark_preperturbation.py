#!/usr/bin/env python3
"""Materialize a reviewed, provisional bundle that stops before perturbation.

The tool joins a hash-bound source behavior bundle, the predeclared relation
review frame, one or more deterministic review-apply manifests, and an explicit
revision plan.  It selects the first terminally eligible rows for every
relation/split quota and creates deterministic, *unverified* distractor
candidates.  It never calls a model, translates text, perturbs a prompt, or
freezes canonical facts or splits.
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
import string
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELATION_POLICY = (
    PROJECT_ROOT / "configs" / "public_benchmark_probe_relation_review_v1.json"
)

TOOL_VERSION = "public-benchmark-preperturbation-materializer-v1"
REVISION_PLAN_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-revision-plan-v1"
)
SELECTION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-selection-manifest-v1"
)
SELECTED_IDS_SCHEMA_VERSION = "public-benchmark-preperturbation-selected-ids-v1"
REVISION_LINEAGE_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-revision-lineage-v1"
)
REREVIEW_SCHEMA_VERSION = "public-benchmark-preperturbation-rereview-v1"
DISTRACTOR_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-distractor-candidate-v1"
)
SUMMARY_SCHEMA_VERSION = "public-benchmark-preperturbation-summary-v1"
CONSOLIDATION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-review-outcome-consolidation-v1"
)
COHORT_RESOLUTION_SCHEMA_VERSION = (
    "public-benchmark-cohort-eligibility-resolution-v1"
)

SOURCE_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
FRAME_SCHEMA_VERSION = "probe-relation-fact-review-frame-v1"
POLICY_SCHEMA_VERSION = "probe-relation-review-policy-v1"
STAGING_SCHEMA_VERSION = "public-benchmark-review-staging-record-v2"

SPLITS = ("development", "validation", "sealed")
ALLOWED_REVIEW_OUTCOMES = frozenset(
    {"accept", "reject", "defer", "revise", "missing"}
)
REVISION_MUTABLE_FIELDS = frozenset(
    {
        "subject_en",
        "relation_raw",
        "answer_en",
        "answer_type",
        "answer_aliases_en",
        "canonical_fact",
        "canonical_fact_en",
    }
)
REVISION_TRIGGERS = frozenset(
    {"review_requested_revision", "semantic_closure_disambiguation"}
)
EDITOR_TYPES = frozenset({"human", "codex_proxy"})
REREVIEW_DECISIONS = frozenset({"accept", "reject"})
PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "source_behavior_bundle",
        "frame",
        "relation_policy",
        "review_apply_manifests",
        "revisions",
    }
)
OPTIONAL_PLAN_FIELDS = frozenset({"review_outcome_consolidation"})
PLAN_FILE_BINDING_FIELDS = frozenset({"path", "sha256", "record_count"})
PLAN_POLICY_BINDING_FIELDS = frozenset({"path", "sha256"})
PLAN_MANIFEST_BINDING_FIELDS = frozenset({"path", "sha256"})
REVISION_FIELDS = frozenset(
    {
        "base_fact_id",
        "trigger",
        "source_review_staging_row_sha256",
        "updates",
        "editor_type",
        "editor_id",
        "edit_method",
        "edited_at",
        "rereview",
    }
)
REREVIEW_FIELDS = frozenset(
    {
        "decision",
        "reviewer_type",
        "reviewer_id",
        "review_method",
        "reviewed_at",
        "human_gold",
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
DISTRACTORS_PER_FACT = 2

SAFETY_CONTRACT = {
    "network_or_model_used": False,
    "model_inference_performed": False,
    "translation_performed": False,
    "perturbation_performed": False,
    "canonical_freeze_performed": False,
    "split_freeze_performed": False,
    "path_not_token_experiment_performed": False,
    "distractors_are_verified": False,
    "output_is_provisional_preperturbation_only": True,
}


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


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def read_json(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
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
                f"Invalid JSON at {resolved}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {resolved}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {resolved}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, payload)


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_timestamp(value: Any, field: str) -> str:
    text = _required_string(value, field)
    parsed_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(parsed_text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return text


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label} row {position}.{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = row
    return result


def _resolved_binding_path(raw: Any, owner_path: Path, field: str) -> Path:
    declared = Path(_required_string(raw, field))
    if not declared.is_absolute():
        declared = owner_path.resolve().parent / declared
    return declared.resolve()


def _file_binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    published_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str((published_path or resolved).resolve()),
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
    }
    if record_count is not None:
        result["record_count"] = record_count
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _load_finalizer() -> Any:
    module_name = "_pitfalls_preperturbation_finalizer_validation"
    module_path = Path(__file__).resolve().with_name(
        "finalize_public_benchmark_bundle.py"
    )
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sibling validator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_policy(policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported relation policy schema_version")
    if policy.get("status") != "pre_perturbation_offline_only":
        raise ValueError("Relation policy must remain pre-perturbation only")
    if policy.get("human_gold") is not False:
        raise ValueError("Relation policy must not claim human gold")
    if policy.get("broad_relation_normalized_required") is not False:
        raise ValueError("Broad relation normalization must not gate this cohort")
    contract = policy.get("selection_contract")
    required_contract = {
        "prompt_ready_required": True,
        "prompt_risk_flags_must_be_empty": True,
        "one_base_fact_per_split_group": True,
        "model_outputs_must_not_be_used": True,
        "selection_precedes_fact_review": True,
        "shortfall_policy": "fail_closed",
    }
    if contract != required_contract:
        raise ValueError("Relation policy selection_contract is invalid")
    relations = policy.get("relations")
    if not isinstance(relations, list) or not relations:
        raise ValueError("Relation policy relations must be a non-empty list")
    result: Dict[str, Dict[str, Any]] = {}
    formatter = string.Formatter()
    for position, item in enumerate(relations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Relation policy entry {position} must be an object")
        relation_id = _required_string(
            item.get("probe_relation_id"),
            f"relation policy entry {position}.probe_relation_id",
        )
        if relation_id in result:
            raise ValueError(f"Duplicate probe_relation_id in policy: {relation_id}")
        direction = _required_string(
            item.get("direction"), f"relation policy {relation_id}.direction"
        )
        template = _required_string(
            item.get("prompt_template_en"),
            f"relation policy {relation_id}.prompt_template_en",
        )
        fields = [
            name
            for _, name, _, _ in formatter.parse(template)
            if name is not None
        ]
        if fields != ["subject"]:
            raise ValueError(
                f"Relation policy {relation_id} prompt template must contain "
                "exactly one {subject} field"
            )
        try:
            template.format(subject="subject")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Relation policy {relation_id} has an invalid prompt template"
            ) from exc
        frame_counts = item.get("review_frame_split_counts")
        target_counts = item.get("target_accept_split_counts")
        if not isinstance(frame_counts, dict) or set(frame_counts) != set(SPLITS):
            raise ValueError(
                f"Relation policy {relation_id} review_frame_split_counts are invalid"
            )
        if not isinstance(target_counts, dict) or set(target_counts) != set(SPLITS):
            raise ValueError(
                f"Relation policy {relation_id} target_accept_split_counts are invalid"
            )
        normalized_frame_counts: Dict[str, int] = {}
        normalized_target_counts: Dict[str, int] = {}
        for split in SPLITS:
            frame_count = _require_int(
                frame_counts[split],
                f"relation policy {relation_id}.review_frame_split_counts.{split}",
                minimum=1,
            )
            target_count = _require_int(
                target_counts[split],
                f"relation policy {relation_id}.target_accept_split_counts.{split}",
                minimum=1,
            )
            if target_count > frame_count:
                raise ValueError(
                    f"Relation policy {relation_id}/{split} target exceeds frame"
                )
            normalized_frame_counts[split] = frame_count
            normalized_target_counts[split] = target_count
        result[relation_id] = {
            "probe_relation_id": relation_id,
            "direction": direction,
            "prompt_template_en": template,
            "review_frame_split_counts": normalized_frame_counts,
            "target_accept_split_counts": normalized_target_counts,
        }
    return result


def _validate_source_bundle(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    index = _unique_index(rows, "base_fact_id", "source behavior bundle")
    for position, row in enumerate(rows, start=1):
        base_fact_id = str(row["base_fact_id"])
        if row.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Source behavior row {position} has unsupported schema_version"
            )
        if row.get("canonical_status") != "pending_review":
            raise ValueError(f"Source behavior row is not pending_review: {base_fact_id}")
        if row.get("bundle_status") != "draft_pending_review":
            raise ValueError(
                f"Source behavior row is not draft_pending_review: {base_fact_id}"
            )
        if row.get("evidence_tier") != "provisional_single_model":
            raise ValueError(
                f"Source behavior row has unsupported evidence_tier: {base_fact_id}"
            )
        if row.get("human_gold") is not False:
            raise ValueError(f"Source behavior row must have human_gold=false: {base_fact_id}")
        if row.get("split_assignment") not in SPLITS:
            raise ValueError(f"Source behavior row has invalid split: {base_fact_id}")
    return index


def _prompt_for(
    relation_spec: Mapping[str, Any], subject: Any, *, base_fact_id: str
) -> str:
    subject_text = _required_string(subject, f"{base_fact_id}.subject_en")
    return str(relation_spec["prompt_template_en"]).format(subject=subject_text)


def _canonical_answer_suffix_prompt(
    row: Mapping[str, Any], *, base_fact_id: str
) -> Optional[str]:
    canonical_fact = _required_string(
        row.get("canonical_fact_en"), f"{base_fact_id}.canonical_fact_en"
    )
    _answer_terms(row, base_fact_id=base_fact_id)
    raw_aliases = [row.get("answer_en"), *(row.get("answer_aliases_en") or [])]
    aliases: List[str] = []
    seen: set[str] = set()
    for value in raw_aliases:
        alias = unicodedata.normalize("NFKC", str(value)).strip()
        normalized = normalize_text(alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(alias)
    aliases.sort(key=lambda value: len(normalize_text(value)), reverse=True)

    fact = unicodedata.normalize("NFKC", canonical_fact).strip()
    fact_variants = [fact]
    if fact and fact[-1] in ".!?。！？":
        fact_variants.append(fact[:-1].rstrip())
    for candidate in fact_variants:
        for alias in aliases:
            match = re.search(re.escape(alias) + r"$", candidate, flags=re.IGNORECASE)
            if match is None:
                continue
            if (
                match.start() > 0
                and candidate[match.start() - 1].isalnum()
                and alias[0].isalnum()
            ):
                continue
            prompt = candidate[: match.start()].rstrip()
            if prompt:
                return prompt
    return None


def _derive_open_completion_prompt(
    relation_spec: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    relation_id: str,
    base_fact_id: str,
) -> Tuple[str, Dict[str, str]]:
    canonical_prompt = _canonical_answer_suffix_prompt(
        row, base_fact_id=base_fact_id
    )
    if canonical_prompt is not None:
        return canonical_prompt, {
            "prompt_template_id": "canonical_fact_answer_suffix_en_provisional_v1",
            "prompt_quality_tier": "strict_factual_completion",
            "prompt_derivation_method": (
                "canonical_fact_en_terminal_answer_or_accepted_alias_suffix_split"
            ),
        }
    return _prompt_for(
        relation_spec, row.get("subject_en"), base_fact_id=base_fact_id
    ), {
        "prompt_template_id": (
            f"probe_relation_open_completion_en_{relation_id}_v1"
        ),
        "prompt_quality_tier": "relation_specific_open_completion",
        "prompt_derivation_method": "relation_template_fallback",
    }


def _answer_terms(row: Mapping[str, Any], *, base_fact_id: str) -> List[str]:
    answer = _required_string(row.get("answer_en"), f"{base_fact_id}.answer_en")
    aliases = row.get("answer_aliases_en")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"{base_fact_id}.answer_aliases_en must be a non-empty list")
    normalized: List[str] = []
    for index, alias in enumerate(aliases):
        text = _required_string(alias, f"{base_fact_id}.answer_aliases_en[{index}]")
        normalized.append(normalize_text(text))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{base_fact_id}.answer_aliases_en contains duplicates")
    if normalize_text(answer) not in set(normalized):
        raise ValueError(f"{base_fact_id}.answer_aliases_en does not contain answer_en")
    return normalized


def _validate_prompt_no_answer(
    prompt: str, row: Mapping[str, Any], *, base_fact_id: str
) -> None:
    prompt_normalized = normalize_text(prompt)
    for answer in _answer_terms(row, base_fact_id=base_fact_id):
        if answer and answer in prompt_normalized:
            raise ValueError(f"Relation prompt leaks an answer alias: {base_fact_id}")
    if row.get("answer_type") == "date":
        date_texts = [row.get("answer_en"), *(row.get("answer_aliases_en") or [])]
        answer_years = {
            match.group(0)
            for value in date_texts
            for match in re.finditer(r"(?<!\d)\d{4}(?!\d)", normalize_text(value))
        }
        for year in answer_years:
            if re.search(rf"(?<!\d){re.escape(year)}(?!\d)", prompt_normalized):
                raise ValueError(
                    f"Relation prompt leaks a date answer year: {base_fact_id}"
                )


def _validate_frame(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    source_index: Mapping[str, Mapping[str, Any]],
    relation_specs: Mapping[str, Mapping[str, Any]],
    expected_frame_count: Optional[int],
) -> Tuple[
    Dict[str, Mapping[str, Any]],
    Dict[Tuple[str, str], List[Mapping[str, Any]]],
]:
    if expected_frame_count is not None and len(frame_rows) != expected_frame_count:
        raise ValueError(
            f"Frame record_count mismatch: expected {expected_frame_count}, "
            f"found {len(frame_rows)}"
        )
    frame_index = _unique_index(frame_rows, "base_fact_id", "relation review frame")
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    seen_ranks: Dict[Tuple[str, str], set[int]] = defaultdict(set)
    seen_split_groups: set[str] = set()
    last_rank: Dict[Tuple[str, str], int] = {}
    closed_groups: set[Tuple[str, str]] = set()
    active_group: Optional[Tuple[str, str]] = None

    copied_fields = (
        "candidate_id",
        "relation_signature_id",
        "subject_en",
        "answer_en",
        "answer_aliases_en",
        "canonical_fact_en",
        "source_question_en",
        "source_dataset",
        "source_format",
        "split_group_id",
        "split_assignment",
        "split_status",
        "probe_relation_candidate",
        "probe_relation_candidate_direction",
    )
    for position, frame in enumerate(frame_rows, start=1):
        base_fact_id = str(frame["base_fact_id"])
        if frame.get("schema_version") != FRAME_SCHEMA_VERSION:
            raise ValueError(f"Frame row has unsupported schema_version: {base_fact_id}")
        source = source_index.get(base_fact_id)
        if source is None:
            raise ValueError(f"Frame row is absent from source bundle: {base_fact_id}")
        if frame.get("source_record_sha256") != sha256_value(source):
            raise ValueError(f"Frame source_record_sha256 is stale: {base_fact_id}")
        for field in copied_fields:
            if canonical_json_bytes(frame.get(field)) != canonical_json_bytes(
                source.get(field)
            ):
                raise ValueError(f"Frame/source {field} mismatch: {base_fact_id}")
        relation_id = _required_string(
            frame.get("probe_relation_id"), f"frame row {base_fact_id}.probe_relation_id"
        )
        relation_spec = relation_specs.get(relation_id)
        if relation_spec is None:
            raise ValueError(f"Frame row has unsupported probe relation: {base_fact_id}")
        if frame.get("probe_relation_candidate") != relation_id:
            raise ValueError(f"Frame relation candidate mismatch: {base_fact_id}")
        if frame.get("probe_relation_candidate_direction") != relation_spec["direction"]:
            raise ValueError(f"Frame relation direction mismatch: {base_fact_id}")
        split = frame.get("split_assignment")
        if split not in SPLITS:
            raise ValueError(f"Frame row has invalid split: {base_fact_id}")
        if frame.get("signature_review_status") != "approved":
            raise ValueError(f"Frame signature is not approved: {base_fact_id}")
        if frame.get("selection_before_fact_review") is not True:
            raise ValueError(f"Frame was not selected before fact review: {base_fact_id}")
        if frame.get("canonical_status") != "pending_review":
            raise ValueError(f"Frame row is not pending_review: {base_fact_id}")
        if frame.get("perturbation_ready") is not False:
            raise ValueError(f"Frame row unexpectedly claims perturbation readiness: {base_fact_id}")
        rank = _require_int(
            frame.get("frame_rank_within_relation_split"),
            f"frame row {base_fact_id}.frame_rank_within_relation_split",
            minimum=1,
        )
        group = (relation_id, str(split))
        max_rank = int(relation_spec["review_frame_split_counts"][str(split)])
        if rank > max_rank:
            raise ValueError(f"Frame rank exceeds policy limit: {base_fact_id}")
        if rank in seen_ranks[group]:
            raise ValueError(
                f"Duplicate frame rank for {relation_id}/{split}: {rank}"
            )
        if rank <= last_rank.get(group, 0):
            raise ValueError(
                f"Frame ranks are not strictly increasing for {relation_id}/{split}"
            )
        if active_group != group:
            if active_group is not None:
                closed_groups.add(active_group)
            if group in closed_groups:
                raise ValueError(f"Frame relation/split group is interleaved: {group}")
            active_group = group
        seen_ranks[group].add(rank)
        last_rank[group] = rank
        split_group_id = _required_string(
            frame.get("split_group_id"), f"frame row {base_fact_id}.split_group_id"
        )
        if split_group_id in seen_split_groups:
            raise ValueError(f"Duplicate split_group_id in frame: {split_group_id}")
        seen_split_groups.add(split_group_id)
        expected_target = int(
            relation_spec["target_accept_split_counts"][str(split)]
        )
        if frame.get("target_accept_count_for_split") != expected_target:
            raise ValueError(f"Frame target_accept_count is stale: {base_fact_id}")
        if frame.get("initial_target_slot") is not (rank <= expected_target):
            raise ValueError(f"Frame initial_target_slot is stale: {base_fact_id}")
        expected_prompt = _prompt_for(
            relation_spec, source.get("subject_en"), base_fact_id=base_fact_id
        )
        if frame.get("proposed_probe_prompt_en") != expected_prompt:
            raise ValueError(f"Frame proposed prompt is stale: {base_fact_id}")
        grouped[group].append(frame)

    for relation_id in relation_specs:
        for split in SPLITS:
            grouped[(relation_id, split)].sort(
                key=lambda row: int(row["frame_rank_within_relation_split"])
            )
    return frame_index, grouped


def _scope_source_bindings(
    apply_manifest_path: Path, finalizer: Any
) -> Mapping[str, Any]:
    manifest = read_json(apply_manifest_path)
    scope_binding = manifest.get("scope_manifest")
    if not isinstance(scope_binding, dict):
        raise ValueError(f"Review apply manifest has no scope binding: {apply_manifest_path}")
    scope_path = _resolved_binding_path(
        scope_binding.get("path"), apply_manifest_path, "scope_manifest.path"
    )
    scope = read_json(scope_path)
    bindings = scope.get("source_artifacts")
    if not isinstance(bindings, dict) or set(bindings) != set(finalizer.SOURCE_ARTIFACTS):
        raise ValueError("Review scope source_artifacts are invalid")
    for role, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != finalizer.SOURCE_SCOPE_BINDING_FIELDS:
            raise ValueError(f"Review scope source binding is invalid: {role}")
    return bindings


def _review_outcome(row: Mapping[str, Any]) -> str:
    """Normalize the review tool's explicit missing-decision staging row."""

    outcome = row.get("review_outcome")
    if outcome is None and row.get("staging_status") == "decision_missing":
        return "missing"
    if not isinstance(outcome, str) or outcome not in ALLOWED_REVIEW_OUTCOMES:
        raise ValueError(
            f"Unsupported review outcome for {row.get('base_fact_id')}: {outcome}"
        )
    return outcome


def _validate_review_manifests(
    manifest_paths: Sequence[Path],
    *,
    frame_ids: Sequence[str],
    source_bundle_path: Path,
    source_bundle_count: int,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Mapping[str, Any],
    List[Dict[str, Any]],
]:
    if not manifest_paths:
        raise ValueError("At least one review apply manifest is required")
    finalizer = _load_finalizer()
    normalized_paths = [Path(path).resolve() for path in manifest_paths]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("Duplicate review apply manifest path")
    expected_sources: Optional[Mapping[str, Any]] = None
    staging_by_id: Dict[str, Dict[str, Any]] = {}
    staging_origin: Dict[str, Dict[str, Any]] = {}
    manifest_bindings: List[Dict[str, Any]] = []
    for manifest_path in normalized_paths:
        sources = _scope_source_bindings(manifest_path, finalizer)
        if expected_sources is None:
            expected_sources = copy.deepcopy(sources)
        elif canonical_json_bytes(expected_sources) != canonical_json_bytes(sources):
            raise ValueError("Review apply manifests bind different source artifacts")
        staging, provenance = finalizer.validate_review_apply_manifest(
            manifest_path,
            universe={"source_artifacts": expected_sources},
            universe_source_ids=list(frame_ids),
        )
        manifest_binding = _file_binding(manifest_path)
        manifest_bindings.append(manifest_binding)
        for base_fact_id, row in staging.items():
            if base_fact_id in staging_by_id:
                raise ValueError(
                    f"Duplicate base_fact_id across review apply manifests: {base_fact_id}"
                )
            if row.get("schema_version") != STAGING_SCHEMA_VERSION:
                raise ValueError(f"Unsupported review staging schema: {base_fact_id}")
            _review_outcome(row)
            staging_by_id[base_fact_id] = dict(row)
            staging_origin[base_fact_id] = {
                "review_apply_manifest_path": str(manifest_path),
                "review_apply_manifest_sha256": manifest_binding["sha256"],
                "scope_id": row.get("scope_id"),
                "review_staging_record_sha256": sha256_value(row),
                "deterministic_replay_performed": bool(
                    provenance.get("deterministic_replay", {}).get("performed")
                ),
            }
    assert expected_sources is not None
    behavior_binding = expected_sources["behavior_bundle"]
    if Path(str(behavior_binding.get("path"))).resolve() != source_bundle_path.resolve():
        raise ValueError("Review scopes bind a different source behavior bundle path")
    if behavior_binding.get("sha256") != sha256_file(source_bundle_path):
        raise ValueError("Review scopes bind a stale source behavior bundle SHA")
    if behavior_binding.get("record_count") != source_bundle_count:
        raise ValueError("Review scopes bind a stale source behavior record_count")
    if behavior_binding.get("schema_version") != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Review scopes bind an unsupported behavior bundle schema")
    return staging_by_id, staging_origin, expected_sources, manifest_bindings


def _validate_plan_file_binding(
    binding: Any,
    *,
    owner_path: Path,
    actual_path: Path,
    actual_count: int,
    label: str,
) -> None:
    if not isinstance(binding, dict) or set(binding) != PLAN_FILE_BINDING_FIELDS:
        raise ValueError(f"{label} binding fields are invalid")
    declared_path = _resolved_binding_path(binding.get("path"), owner_path, f"{label}.path")
    if declared_path != actual_path.resolve():
        raise ValueError(f"{label}.path does not match the CLI input")
    if binding.get("sha256") != sha256_file(actual_path):
        raise ValueError(f"{label}.sha256 is stale")
    if binding.get("record_count") != actual_count:
        raise ValueError(f"{label}.record_count is stale")


def _validate_plan_policy_binding(
    binding: Any, *, owner_path: Path, actual_path: Path
) -> None:
    if not isinstance(binding, dict) or set(binding) != PLAN_POLICY_BINDING_FIELDS:
        raise ValueError("relation_policy binding fields are invalid")
    declared_path = _resolved_binding_path(
        binding.get("path"), owner_path, "relation_policy.path"
    )
    if declared_path != actual_path.resolve():
        raise ValueError("relation_policy.path does not match the CLI input")
    if binding.get("sha256") != sha256_file(actual_path):
        raise ValueError("relation_policy.sha256 is stale")


def _validate_revision_plan_bindings(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    source_bundle_path: Path,
    source_count: int,
    frame_path: Path,
    frame_count: int,
    relation_policy_path: Path,
    review_manifest_paths: Sequence[Path],
) -> None:
    if not PLAN_FIELDS.issubset(plan) or set(plan) - PLAN_FIELDS != (
        set(plan).intersection(OPTIONAL_PLAN_FIELDS)
    ):
        raise ValueError("Revision plan fields are invalid")
    if plan.get("schema_version") != REVISION_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported revision plan schema_version")
    _validate_plan_file_binding(
        plan.get("source_behavior_bundle"),
        owner_path=plan_path,
        actual_path=source_bundle_path,
        actual_count=source_count,
        label="source_behavior_bundle",
    )
    _validate_plan_file_binding(
        plan.get("frame"),
        owner_path=plan_path,
        actual_path=frame_path,
        actual_count=frame_count,
        label="frame",
    )
    _validate_plan_policy_binding(
        plan.get("relation_policy"),
        owner_path=plan_path,
        actual_path=relation_policy_path,
    )
    manifest_bindings = plan.get("review_apply_manifests")
    if not isinstance(manifest_bindings, list) or not manifest_bindings:
        raise ValueError("review_apply_manifests must be a non-empty list")
    if len(manifest_bindings) != len(review_manifest_paths):
        raise ValueError("Revision plan review apply manifest count mismatch")
    for position, (binding, actual_path) in enumerate(
        zip(manifest_bindings, review_manifest_paths), start=1
    ):
        if not isinstance(binding, dict) or set(binding) != PLAN_MANIFEST_BINDING_FIELDS:
            raise ValueError(
                f"review_apply_manifests[{position}] binding fields are invalid"
            )
        declared_path = _resolved_binding_path(
            binding.get("path"),
            plan_path,
            f"review_apply_manifests[{position}].path",
        )
        if declared_path != Path(actual_path).resolve():
            raise ValueError(
                f"review_apply_manifests[{position}].path does not match CLI input"
            )
        if binding.get("sha256") != sha256_file(actual_path):
            raise ValueError(f"review_apply_manifests[{position}].sha256 is stale")


def _validate_consolidation_evidence(
    binding: Any,
    *,
    plan_path: Path,
    review_manifest_paths: Sequence[Path],
    staging_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    if binding is None:
        return {}, None
    if not isinstance(binding, dict) or set(binding) != PLAN_MANIFEST_BINDING_FIELDS:
        raise ValueError("review_outcome_consolidation binding fields are invalid")
    manifest_path = _resolved_binding_path(
        binding.get("path"), plan_path, "review_outcome_consolidation.path"
    )
    if binding.get("sha256") != sha256_file(manifest_path):
        raise ValueError("review_outcome_consolidation.sha256 is stale")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != CONSOLIDATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported review outcome consolidation schema_version")
    contract = manifest.get("cohort_eligibility_contract")
    if not isinstance(contract, dict) or not all(
        (
            contract.get(
                "cohort_exclude_applies_to_current_preperturbation_cohort_only"
            )
            is True,
            contract.get("cohort_exclude_asserts_factual_falsehood") is False,
            contract.get(
                "review_tool_v2_reject_is_adapter_encoding_for_cohort_exclude"
            )
            is True,
            contract.get("selector_must_consult_cohort_eligibility_resolutions")
            is True,
        )
    ):
        raise ValueError("Review outcome consolidation eligibility contract is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Review outcome consolidation artifacts are missing")
    apply_binding = artifacts.get("review_apply_manifest")
    if not isinstance(apply_binding, dict) or set(apply_binding) != {
        "path",
        "sha256",
    }:
        raise ValueError("Consolidated review apply binding is invalid")
    bound_apply_path = _resolved_binding_path(
        apply_binding.get("path"), manifest_path, "consolidated review apply.path"
    )
    if len(review_manifest_paths) != 1 or bound_apply_path != Path(
        review_manifest_paths[0]
    ).resolve():
        raise ValueError(
            "Consolidation evidence must bind the sole review apply manifest input"
        )
    if apply_binding.get("sha256") != sha256_file(bound_apply_path):
        raise ValueError("Consolidated review apply SHA is stale")
    ledger_binding = artifacts.get("cohort_eligibility_resolutions")
    if not isinstance(ledger_binding, dict) or set(ledger_binding) != {
        "path",
        "sha256",
        "record_count",
        "schema_version",
    }:
        raise ValueError("Cohort eligibility resolution binding is invalid")
    ledger_path = _resolved_binding_path(
        ledger_binding.get("path"), manifest_path, "cohort eligibility ledger.path"
    )
    if ledger_binding.get("sha256") != sha256_file(ledger_path):
        raise ValueError("Cohort eligibility resolution SHA is stale")
    if ledger_binding.get("schema_version") != COHORT_RESOLUTION_SCHEMA_VERSION:
        raise ValueError("Cohort eligibility resolution schema binding is invalid")
    rows = read_jsonl(ledger_path, allow_empty=True)
    if ledger_binding.get("record_count") != len(rows):
        raise ValueError("Cohort eligibility resolution record_count is stale")
    resolution_index = _unique_index(
        rows, "base_fact_id", "cohort eligibility resolutions"
    )
    disposition_counts: Counter[str] = Counter()
    reconstructed_input_outcomes: Counter[str] = Counter(
        _review_outcome(row) for row in staging_by_id.values()
    )
    for base_fact_id, row in resolution_index.items():
        if row.get("schema_version") != COHORT_RESOLUTION_SCHEMA_VERSION:
            raise ValueError(
                f"Cohort eligibility resolution schema is invalid: {base_fact_id}"
            )
        if base_fact_id not in staging_by_id:
            raise ValueError(
                f"Cohort eligibility resolution is outside review staging: "
                f"{base_fact_id}"
            )
        original_outcome = row.get("original_review_outcome")
        if original_outcome not in {"revise", "defer"}:
            raise ValueError(
                f"Cohort resolution has invalid original outcome: {base_fact_id}"
            )
        disposition = row.get("cohort_eligibility_disposition")
        if disposition not in {"retain_revise", "cohort_exclude"}:
            raise ValueError(
                f"Cohort resolution has invalid disposition: {base_fact_id}"
            )
        disposition_counts[str(disposition)] += 1
        if row.get("factual_rejection_asserted") is not False:
            raise ValueError(
                f"Cohort exclusion must not assert factual falsehood: {base_fact_id}"
            )
        expected_encoded = "revise" if disposition == "retain_revise" else "reject"
        if row.get("encoded_review_outcome") != expected_encoded:
            raise ValueError(
                f"Cohort resolution encoded outcome is invalid: {base_fact_id}"
            )
        if _review_outcome(staging_by_id[base_fact_id]) != expected_encoded:
            raise ValueError(
                f"Cohort resolution disagrees with consolidated staging: "
                f"{base_fact_id}"
            )
        reconstructed_input_outcomes[expected_encoded] -= 1
        reconstructed_input_outcomes[str(original_outcome)] += 1
        if disposition == "cohort_exclude":
            _required_string(
                row.get("cohort_exclusion_reason"),
                f"cohort resolution {base_fact_id}.cohort_exclusion_reason",
            )
        original_bindings = row.get("original_review_bindings")
        if not isinstance(original_bindings, dict):
            raise ValueError(
                f"Cohort resolution lacks original review bindings: {base_fact_id}"
            )
        for role in ("review_apply_manifest", "review_staging", "decision"):
            if not isinstance(original_bindings.get(role), dict):
                raise ValueError(
                    f"Cohort resolution lacks original {role} binding: "
                    f"{base_fact_id}"
                )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Review outcome consolidation counts are missing")
    output_outcomes = Counter(
        _review_outcome(row) for row in staging_by_id.values()
    )
    if counts.get("input_record_count") != len(staging_by_id):
        raise ValueError("Review outcome consolidation input_record_count is stale")
    if counts.get("resolution_count") != len(resolution_index):
        raise ValueError("Review outcome consolidation resolution_count is stale")
    if counts.get("input_outcome_counts") != dict(
        sorted((key, value) for key, value in reconstructed_input_outcomes.items() if value)
    ):
        raise ValueError("Review outcome consolidation input_outcome_counts are stale")
    if counts.get("output_outcome_counts") != dict(sorted(output_outcomes.items())):
        raise ValueError("Review outcome consolidation output_outcome_counts are stale")
    if counts.get("resolution_disposition_counts") != dict(
        sorted(disposition_counts.items())
    ):
        raise ValueError(
            "Review outcome consolidation resolution_disposition_counts are stale"
        )
    replay = manifest.get("deterministic_replay")
    if not isinstance(replay, dict) or not all(
        (
            replay.get("all_input_apply_manifests_replayed") is True,
            replay.get("output_apply_manifest_replayed") is True,
        )
    ):
        raise ValueError("Review outcome consolidation replay evidence is invalid")
    retained_revise_ids = {
        base_fact_id
        for base_fact_id, row in staging_by_id.items()
        if _review_outcome(row) == "revise"
    }
    ledger_retained_ids = {
        base_fact_id
        for base_fact_id, row in resolution_index.items()
        if row.get("cohort_eligibility_disposition") == "retain_revise"
    }
    if retained_revise_ids != ledger_retained_ids:
        raise ValueError(
            "Consolidated revise outcomes do not match retain_revise resolutions"
        )
    return (
        {str(key): dict(value) for key, value in resolution_index.items()},
        _file_binding(
            manifest_path,
            schema_version=CONSOLIDATION_MANIFEST_SCHEMA_VERSION,
        ),
    )


def _validate_revisions(
    raw_revisions: Any,
    *,
    source_index: Mapping[str, Mapping[str, Any]],
    staging_by_id: Mapping[str, Mapping[str, Any]],
    cohort_resolutions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_revisions, list):
        raise ValueError("Revision plan revisions must be a list")
    result: Dict[str, Dict[str, Any]] = {}
    for position, revision in enumerate(raw_revisions, start=1):
        if not isinstance(revision, dict) or set(revision) != REVISION_FIELDS:
            raise ValueError(f"Revision entry {position} fields are invalid")
        base_fact_id = _required_string(
            revision.get("base_fact_id"), f"revision {position}.base_fact_id"
        )
        if base_fact_id in result:
            raise ValueError(f"Duplicate revision base_fact_id: {base_fact_id}")
        source = source_index.get(base_fact_id)
        if source is None:
            raise ValueError(f"Revision is absent from source bundle: {base_fact_id}")
        staging = staging_by_id.get(base_fact_id)
        if staging is None:
            raise ValueError(f"Revision has no review staging row: {base_fact_id}")
        staging_hash = _require_sha256(
            revision.get("source_review_staging_row_sha256"),
            f"revision {base_fact_id}.source_review_staging_row_sha256",
        )
        if staging_hash != sha256_value(staging):
            raise ValueError(f"Revision staging row hash is stale: {base_fact_id}")
        trigger = revision.get("trigger")
        if trigger not in REVISION_TRIGGERS:
            raise ValueError(f"Revision trigger is invalid: {base_fact_id}")
        outcome = _review_outcome(staging)
        if outcome == "revise" and trigger != "review_requested_revision":
            raise ValueError(
                f"Revise outcome requires review_requested_revision: {base_fact_id}"
            )
        if outcome == "revise" and cohort_resolutions and (
            cohort_resolutions.get(base_fact_id, {}).get(
                "cohort_eligibility_disposition"
            )
            != "retain_revise"
        ):
            raise ValueError(
                f"Revision is not retained by cohort resolution: {base_fact_id}"
            )
        if outcome == "accept" and trigger != "semantic_closure_disambiguation":
            raise ValueError(
                f"Accept outcome revisions require semantic_closure_disambiguation: "
                f"{base_fact_id}"
            )
        if outcome not in {"accept", "revise"}:
            raise ValueError(
                f"Review outcome does not permit a revision: {base_fact_id} ({outcome})"
            )
        updates = revision.get("updates")
        if not isinstance(updates, dict) or not updates:
            raise ValueError(f"Revision updates must be non-empty: {base_fact_id}")
        unsupported = sorted(set(updates) - REVISION_MUTABLE_FIELDS)
        if unsupported:
            raise ValueError(
                f"Revision updates immutable fields for {base_fact_id}: {unsupported}"
            )
        for field, value in updates.items():
            if canonical_json_bytes(value) == canonical_json_bytes(source.get(field)):
                raise ValueError(
                    f"Revision update does not change {field}: {base_fact_id}"
                )
        if "answer_en" in updates and "answer_aliases_en" not in updates:
            raise ValueError(
                f"Revision changing answer_en must update answer_aliases_en: {base_fact_id}"
            )
        revised = copy.deepcopy(dict(source))
        for field, value in updates.items():
            revised[field] = copy.deepcopy(value)
        for field in (
            "subject_en",
            "relation_raw",
            "answer_en",
            "answer_type",
            "canonical_fact",
            "canonical_fact_en",
        ):
            _required_string(revised.get(field), f"revised {base_fact_id}.{field}")
        _answer_terms(revised, base_fact_id=base_fact_id)
        if revised.get("canonical_fact") != revised.get("canonical_fact_en"):
            raise ValueError(f"Revised canonical facts differ: {base_fact_id}")
        editor_type = revision.get("editor_type")
        if editor_type not in EDITOR_TYPES:
            raise ValueError(f"Revision editor_type is invalid: {base_fact_id}")
        editor_id = _required_string(
            revision.get("editor_id"), f"revision {base_fact_id}.editor_id"
        )
        _required_string(
            revision.get("edit_method"), f"revision {base_fact_id}.edit_method"
        )
        _validate_timestamp(
            revision.get("edited_at"), f"revision {base_fact_id}.edited_at"
        )
        rereview = revision.get("rereview")
        if not isinstance(rereview, dict) or set(rereview) != REREVIEW_FIELDS:
            raise ValueError(f"Revision rereview fields are invalid: {base_fact_id}")
        decision = rereview.get("decision")
        if decision not in REREVIEW_DECISIONS:
            raise ValueError(f"Revision rereview decision is invalid: {base_fact_id}")
        if outcome == "accept" and decision != "accept":
            raise ValueError(
                f"Semantic-closure revision must pass rereview: {base_fact_id}"
            )
        reviewer_type = rereview.get("reviewer_type")
        if reviewer_type not in EDITOR_TYPES:
            raise ValueError(f"Revision reviewer_type is invalid: {base_fact_id}")
        reviewer_id = _required_string(
            rereview.get("reviewer_id"), f"revision {base_fact_id}.rereview.reviewer_id"
        )
        if reviewer_id == editor_id:
            raise ValueError(
                f"Revision editor and rereviewer must be independent: {base_fact_id}"
            )
        _required_string(
            rereview.get("review_method"),
            f"revision {base_fact_id}.rereview.review_method",
        )
        _validate_timestamp(
            rereview.get("reviewed_at"),
            f"revision {base_fact_id}.rereview.reviewed_at",
        )
        if rereview.get("human_gold") is not False:
            raise ValueError(f"Revision rereview must not claim human gold: {base_fact_id}")
        result[base_fact_id] = {
            "entry": copy.deepcopy(revision),
            "revised_source": revised,
            "before_record_sha256": sha256_value(source),
            "after_semantic_revision_sha256": sha256_value(revised),
            "changed_fields": sorted(updates),
        }
    return result


def _lineage_rows(
    consumed_revision_ids: Sequence[str],
    *,
    revisions: Mapping[str, Mapping[str, Any]],
    frame_index: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lineage: List[Dict[str, Any]] = []
    rereviews: List[Dict[str, Any]] = []
    for base_fact_id in consumed_revision_ids:
        revision = revisions[base_fact_id]
        entry = revision["entry"]
        frame = frame_index[base_fact_id]
        rereview = entry["rereview"]
        lineage.append(
            {
                "schema_version": REVISION_LINEAGE_SCHEMA_VERSION,
                "source_base_fact_id": base_fact_id,
                "trigger": entry["trigger"],
                "source_review_staging_row_sha256": entry[
                    "source_review_staging_row_sha256"
                ],
                "before_record_sha256": revision["before_record_sha256"],
                "after_semantic_revision_sha256": revision[
                    "after_semantic_revision_sha256"
                ],
                "changed_fields": revision["changed_fields"],
                "updates": copy.deepcopy(entry["updates"]),
                "editor_type": entry["editor_type"],
                "editor_id": entry["editor_id"],
                "edit_method": entry["edit_method"],
                "edited_at": entry["edited_at"],
                "probe_relation_id": frame["probe_relation_id"],
                "split_assignment": frame["split_assignment"],
                "frame_rank_within_relation_split": frame[
                    "frame_rank_within_relation_split"
                ],
                "derived_prompt_recomputed": True,
                "canonical_or_split_freeze_performed": False,
            }
        )
        rereviews.append(
            {
                "schema_version": REREVIEW_SCHEMA_VERSION,
                "source_base_fact_id": base_fact_id,
                "trigger": entry["trigger"],
                "revised_record_sha256": revision[
                    "after_semantic_revision_sha256"
                ],
                "decision": rereview["decision"],
                "reviewer_type": rereview["reviewer_type"],
                "reviewer_id": rereview["reviewer_id"],
                "review_method": rereview["review_method"],
                "reviewed_at": rereview["reviewed_at"],
                "human_gold": False,
            }
        )
    return lineage, rereviews


def _select_rows(
    *,
    grouped_frame: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
    relation_specs: Mapping[str, Mapping[str, Any]],
    staging_by_id: Mapping[str, Mapping[str, Any]],
    staging_origin: Mapping[str, Mapping[str, Any]],
    source_index: Mapping[str, Mapping[str, Any]],
    revisions: Mapping[str, Mapping[str, Any]],
    cohort_resolutions: Mapping[str, Mapping[str, Any]],
    quotas: Mapping[str, int],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    consumed_revision_ids: List[str] = []
    consumed_revision_set: set[str] = set()
    scan_outcomes: Counter[str] = Counter()
    scan_counts: Dict[str, Dict[str, int]] = {}

    for relation_id, relation_spec in relation_specs.items():
        for split in SPLITS:
            quota = quotas[split]
            policy_capacity = int(
                relation_spec["review_frame_split_counts"][split]
            )
            if quota > policy_capacity:
                raise ValueError(
                    f"Quota exceeds policy frame capacity for {relation_id}/{split}"
                )
            candidates = grouped_frame.get((relation_id, split), [])
            selected_in_group = 0
            scanned = 0
            for frame in candidates:
                if selected_in_group >= quota:
                    break
                scanned += 1
                base_fact_id = str(frame["base_fact_id"])
                staging = staging_by_id.get(base_fact_id)
                if staging is None:
                    raise ValueError(
                        f"Fail-closed review gap before quota for {relation_id}/{split}: "
                        f"rank {frame['frame_rank_within_relation_split']} "
                        f"{base_fact_id} has no review staging row"
                    )
                bindings = staging.get("source_row_bindings")
                if not isinstance(bindings, dict) or bindings.get(
                    "behavior_row_sha256"
                ) != sha256_value(source_index[base_fact_id]):
                    raise ValueError(
                        f"Review staging source behavior hash is stale: {base_fact_id}"
                    )
                outcome = _review_outcome(staging)
                scan_outcomes[outcome] += 1
                revision = revisions.get(base_fact_id)
                if outcome in {"missing", "defer"}:
                    raise ValueError(
                        f"Fail-closed {outcome} review before quota for "
                        f"{relation_id}/{split}: rank "
                        f"{frame['frame_rank_within_relation_split']} {base_fact_id}"
                    )
                if outcome == "revise" and revision is None:
                    raise ValueError(
                        f"Fail-closed unresolved revise before quota for "
                        f"{relation_id}/{split}: rank "
                        f"{frame['frame_rank_within_relation_split']} {base_fact_id}"
                    )
                if revision is not None:
                    consumed_revision_set.add(base_fact_id)
                    consumed_revision_ids.append(base_fact_id)
                if outcome == "reject":
                    continue
                if outcome == "revise" and revision["entry"]["rereview"][
                    "decision"
                ] == "reject":
                    continue
                if outcome not in {"accept", "revise"}:
                    raise ValueError(f"Unsupported review outcome: {base_fact_id}")
                source = (
                    revision["revised_source"]
                    if revision is not None
                    else copy.deepcopy(dict(source_index[base_fact_id]))
                )
                source = copy.deepcopy(dict(source))
                if outcome == "accept" and (
                    revision is None
                    or "answer_aliases_en" not in revision["entry"]["updates"]
                ):
                    accepted_aliases = staging.get("accepted_answer_aliases_en")
                    if not isinstance(accepted_aliases, list) or not accepted_aliases:
                        raise ValueError(
                            f"Accepted review has no accepted answer aliases: "
                            f"{base_fact_id}"
                        )
                    source["answer_aliases_en"] = copy.deepcopy(accepted_aliases)
                    _answer_terms(source, base_fact_id=base_fact_id)
                final_prompt, _ = _derive_open_completion_prompt(
                    relation_spec,
                    source,
                    relation_id=relation_id,
                    base_fact_id=base_fact_id,
                )
                _validate_prompt_no_answer(
                    final_prompt, source, base_fact_id=base_fact_id
                )
                selected.append(
                    {
                        "base_fact_id": base_fact_id,
                        "source": copy.deepcopy(dict(source)),
                        "original_source_record_sha256": sha256_value(
                            source_index[base_fact_id]
                        ),
                        "frame": copy.deepcopy(dict(frame)),
                        "staging": copy.deepcopy(dict(staging)),
                        "staging_origin": copy.deepcopy(
                            dict(staging_origin[base_fact_id])
                        ),
                        "revision": copy.deepcopy(revision),
                        "cohort_resolution": copy.deepcopy(
                            cohort_resolutions.get(base_fact_id)
                        ),
                    }
                )
                selected_in_group += 1
            scan_counts[f"{relation_id}/{split}"] = {
                "quota": quota,
                "scanned": scanned,
                "selected": selected_in_group,
                "available_in_frame": len(candidates),
            }
            if selected_in_group != quota:
                raise ValueError(
                    f"Insufficient eligible rows for {relation_id}/{split}: "
                    f"required {quota}, selected {selected_in_group}"
                )
    unused = sorted(set(revisions) - consumed_revision_set)
    if unused:
        raise ValueError(f"Revision plan contains unconsumed entries: {unused}")
    return selected, consumed_revision_ids, {
        "scan_outcome_counts": dict(sorted(scan_outcomes.items())),
        "relation_split_scan_counts": scan_counts,
    }


def _distractor_candidates(
    selected: Sequence[Mapping[str, Any]], *, policy_id: str
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    pools: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for item in selected:
        frame = item["frame"]
        pools[(str(frame["probe_relation_id"]), str(frame["split_assignment"]))].append(
            item
        )
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    flattened: List[Dict[str, Any]] = []
    for target in selected:
        target_id = str(target["base_fact_id"])
        target_frame = target["frame"]
        target_source = target["source"]
        relation_id = str(target_frame["probe_relation_id"])
        split = str(target_frame["split_assignment"])
        target_terms = set(_answer_terms(target_source, base_fact_id=target_id))
        eligible: List[Tuple[str, Mapping[str, Any]]] = []
        for donor in pools[(relation_id, split)]:
            donor_id = str(donor["base_fact_id"])
            if donor_id == target_id:
                continue
            donor_source = donor["source"]
            donor_terms = set(_answer_terms(donor_source, base_fact_id=donor_id))
            if target_terms.intersection(donor_terms):
                continue
            score = sha256_value(
                [
                    policy_id,
                    target_id,
                    donor_id,
                    donor_source.get("answer_en"),
                ]
            )
            eligible.append((score, donor))
        eligible.sort(key=lambda pair: (pair[0], str(pair[1]["base_fact_id"])))
        if len(eligible) < DISTRACTORS_PER_FACT:
            raise ValueError(
                f"Fewer than {DISTRACTORS_PER_FACT} same-relation/split, "
                f"alias-disjoint distractors for {target_id}"
            )
        candidates: List[Dict[str, Any]] = []
        for rank, (score, donor) in enumerate(
            eligible[:DISTRACTORS_PER_FACT], start=1
        ):
            donor_id = str(donor["base_fact_id"])
            donor_source = donor["source"]
            candidate_id = "pre_dist_" + sha256_value(
                [target_id, donor_id, score]
            )[:20]
            candidate = {
                "schema_version": DISTRACTOR_SCHEMA_VERSION,
                "distractor_id": candidate_id,
                "target_base_fact_id": target_id,
                "donor_base_fact_id": donor_id,
                "probe_relation_id": relation_id,
                "split_assignment": split,
                "rank": rank,
                "text_en": donor_source["answer_en"],
                "answer_en": donor_source["answer_en"],
                "answer_aliases_en": copy.deepcopy(
                    donor_source["answer_aliases_en"]
                ),
                "selection_score_sha256": score,
                "selection_method": (
                    "deterministic_same_relation_split_alias_disjoint_v1"
                ),
                "verification_status": "codex_proxy_pending_review",
                "status": "codex_proxy_pending_review",
                "verified": False,
                "human_gold": False,
            }
            candidates.append(candidate)
            flattened.append(candidate)
        by_target[target_id] = candidates
    return by_target, flattened


def _materialized_behavior_row(
    item: Mapping[str, Any],
    *,
    relation_spec: Mapping[str, Any],
    distractors: Sequence[Mapping[str, Any]],
    source_bundle_sha256: str,
    frame_sha256: str,
    revision_plan_sha256: str,
    consolidation_manifest_sha256: Optional[str],
) -> Dict[str, Any]:
    base_fact_id = str(item["base_fact_id"])
    row = copy.deepcopy(dict(item["source"]))
    frame = item["frame"]
    for field in (
        "subject_en",
        "relation_raw",
        "answer_en",
        "answer_type",
        "canonical_fact",
        "canonical_fact_en",
    ):
        _required_string(row.get(field), f"selected {base_fact_id}.{field}")
    if row.get("canonical_fact") != row.get("canonical_fact_en"):
        raise ValueError(f"Selected canonical facts differ: {base_fact_id}")
    _answer_terms(row, base_fact_id=base_fact_id)
    prompt, prompt_metadata = _derive_open_completion_prompt(
        relation_spec,
        row,
        relation_id=str(frame["probe_relation_id"]),
        base_fact_id=base_fact_id,
    )
    _validate_prompt_no_answer(prompt, row, base_fact_id=base_fact_id)
    cohort_resolution = item.get("cohort_resolution")
    row.update(
        {
            "bundle_status": "preperturbation_provisional_not_frozen",
            "canonical_status": "pending_review",
            "canonical_freeze_status": "provisional_not_frozen",
            "evidence_tier": "provisional_single_model",
            "human_gold": False,
            "probe_relation_id": frame["probe_relation_id"],
            "probe_relation_status": (
                "item_reviewed_preperturbation_pending_freeze"
            ),
            "relation_normalized": None,
            "normalization_status": "not_required_for_first_probe_cohort",
            "prompt_en": prompt,
            "prompt_template_id": prompt_metadata["prompt_template_id"],
            "prompt_version": (
                "public-benchmark-preperturbation-open-completion-v1"
            ),
            "prompt_quality_tier": prompt_metadata["prompt_quality_tier"],
            "prompt_derivation_method": prompt_metadata[
                "prompt_derivation_method"
            ],
            "prompt_ready": True,
            "split_assignment": frame["split_assignment"],
            "split_group_id": frame["split_group_id"],
            "split_status": "provisional_not_frozen",
            "distractor_candidates": [copy.deepcopy(dict(value)) for value in distractors],
            "subject_target": None,
            "answer_target": None,
            "prompt_target": None,
            "target_language": None,
            "subject_zh": None,
            "answer_zh": None,
            "answer_aliases_zh": None,
            "canonical_fact_zh": None,
            "prompt_zh": None,
            "distractor_candidates_zh": None,
            "translation_status": "pending_generation_and_review",
            "perturbation_ready": False,
            "perturbation_status": "not_run",
            "path_not_token_experiment_ready": False,
            "sealed_evaluation_status": "not_run",
            "experiment_status": {
                "ollama_behavior_screening": "not_run",
                "hf_exact_checkpoint_reproduction": "not_run",
                "answer_tokenization": "not_run",
                "hidden_state_collection": "not_run",
                "activation_patching": "not_run",
                "vector_intervention": "not_run",
                "repair_validation": "not_run",
            },
            "admission": {
                "prompt_ready": True,
                "semantic_review_complete": False,
                "behavior_screening_ready": False,
                "hf_bridge_ready": False,
                "relation_conditioned_mechanism_eligible": False,
                "path_not_token_experiment_ready": False,
                "blocking_reasons": [
                    "canonical_freeze_pending",
                    "split_freeze_pending",
                    "target_translation_pending",
                    "distractor_review_pending",
                    "perturbation_not_run",
                ],
            },
            "preperturbation_lineage": {
                "tool_version": TOOL_VERSION,
                "source_behavior_bundle_sha256": source_bundle_sha256,
                "source_record_sha256": item[
                    "original_source_record_sha256"
                ],
                "frame_artifact_sha256": frame_sha256,
                "frame_record_sha256": sha256_value(frame),
                "review_staging_record_sha256": item["staging_origin"][
                    "review_staging_record_sha256"
                ],
                "review_apply_manifest_sha256": item["staging_origin"][
                    "review_apply_manifest_sha256"
                ],
                "revision_plan_sha256": revision_plan_sha256,
                "review_outcome_consolidation_manifest_sha256": (
                    consolidation_manifest_sha256
                ),
                "cohort_eligibility_resolution_record_sha256": (
                    sha256_value(cohort_resolution)
                    if isinstance(cohort_resolution, dict)
                    else None
                ),
                "cohort_eligibility_disposition": (
                    cohort_resolution.get("cohort_eligibility_disposition")
                    if isinstance(cohort_resolution, dict)
                    else None
                ),
                "cohort_resolution_asserts_factual_rejection": (
                    cohort_resolution.get("factual_rejection_asserted")
                    if isinstance(cohort_resolution, dict)
                    else None
                ),
                "revision_applied": item["revision"] is not None,
                "revision_trigger": (
                    item["revision"]["entry"]["trigger"]
                    if item["revision"] is not None
                    else None
                ),
                "source_review_outcome": item["staging"]["review_outcome"],
                "accepted_alias_filter_applied": item["staging"][
                    "review_outcome"
                ]
                == "accept",
            },
        }
    )
    return row


def materialize_preperturbation_bundle(
    *,
    source_bundle_path: Path,
    frame_path: Path,
    review_apply_manifest_paths: Sequence[Path],
    revision_plan_path: Path,
    relation_policy_path: Path,
    output_dir: Path,
    development_quota: int = 24,
    validation_quota: int = 8,
    sealed_quota: int = 8,
    expected_frame_count: Optional[int] = 311,
) -> Dict[str, Any]:
    source_bundle_path = Path(source_bundle_path).resolve()
    frame_path = Path(frame_path).resolve()
    revision_plan_path = Path(revision_plan_path).resolve()
    relation_policy_path = Path(relation_policy_path).resolve()
    review_apply_manifest_paths = [
        Path(path).resolve() for path in review_apply_manifest_paths
    ]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    quotas = {
        "development": _require_int(
            development_quota, "development_quota", minimum=1
        ),
        "validation": _require_int(validation_quota, "validation_quota", minimum=1),
        "sealed": _require_int(sealed_quota, "sealed_quota", minimum=1),
    }
    if expected_frame_count is not None:
        _require_int(expected_frame_count, "expected_frame_count", minimum=1)

    source_rows = read_jsonl(source_bundle_path)
    source_index = _validate_source_bundle(source_rows)
    policy = read_json(relation_policy_path)
    relation_specs = _validate_policy(policy)
    frame_rows = read_jsonl(frame_path)
    frame_index, grouped_frame = _validate_frame(
        frame_rows,
        source_index=source_index,
        relation_specs=relation_specs,
        expected_frame_count=expected_frame_count,
    )
    (
        staging_by_id,
        staging_origin,
        review_source_bindings,
        review_manifest_bindings,
    ) = _validate_review_manifests(
        review_apply_manifest_paths,
        frame_ids=list(frame_index),
        source_bundle_path=source_bundle_path,
        source_bundle_count=len(source_rows),
    )
    plan = read_json(revision_plan_path)
    _validate_revision_plan_bindings(
        plan,
        plan_path=revision_plan_path,
        source_bundle_path=source_bundle_path,
        source_count=len(source_rows),
        frame_path=frame_path,
        frame_count=len(frame_rows),
        relation_policy_path=relation_policy_path,
        review_manifest_paths=review_apply_manifest_paths,
    )
    cohort_resolutions, consolidation_evidence = _validate_consolidation_evidence(
        plan.get("review_outcome_consolidation"),
        plan_path=revision_plan_path,
        review_manifest_paths=review_apply_manifest_paths,
        staging_by_id=staging_by_id,
    )
    revisions = _validate_revisions(
        plan.get("revisions"),
        source_index=source_index,
        staging_by_id=staging_by_id,
        cohort_resolutions=cohort_resolutions,
    )
    selected, consumed_revision_ids, scan_summary = _select_rows(
        grouped_frame=grouped_frame,
        relation_specs=relation_specs,
        staging_by_id=staging_by_id,
        staging_origin=staging_origin,
        source_index=source_index,
        revisions=revisions,
        cohort_resolutions=cohort_resolutions,
        quotas=quotas,
    )
    policy_id = _required_string(policy.get("policy_id"), "relation policy.policy_id")
    distractors_by_target, distractor_rows = _distractor_candidates(
        selected, policy_id=policy_id
    )
    source_bundle_sha256 = sha256_file(source_bundle_path)
    frame_sha256 = sha256_file(frame_path)
    revision_plan_sha256 = sha256_file(revision_plan_path)
    behavior_rows: List[Dict[str, Any]] = []
    for item in selected:
        relation_id = str(item["frame"]["probe_relation_id"])
        behavior_rows.append(
            _materialized_behavior_row(
                item,
                relation_spec=relation_specs[relation_id],
                distractors=distractors_by_target[str(item["base_fact_id"])],
                source_bundle_sha256=source_bundle_sha256,
                frame_sha256=frame_sha256,
                revision_plan_sha256=revision_plan_sha256,
                consolidation_manifest_sha256=(
                    str(consolidation_evidence["sha256"])
                    if consolidation_evidence is not None
                    else None
                ),
            )
        )
    lineage_rows, rereview_rows = _lineage_rows(
        consumed_revision_ids,
        revisions=revisions,
        frame_index=frame_index,
    )

    selected_ids = [str(item["base_fact_id"]) for item in selected]
    selected_by_group: Dict[str, List[str]] = defaultdict(list)
    selected_counts: Dict[str, int] = Counter()
    for item in selected:
        group = (
            f"{item['frame']['probe_relation_id']}/"
            f"{item['frame']['split_assignment']}"
        )
        selected_by_group[group].append(str(item["base_fact_id"]))
        selected_counts[group] += 1
    selected_ids_document = {
        "schema_version": SELECTED_IDS_SCHEMA_VERSION,
        "relation_order": list(relation_specs),
        "split_order": list(SPLITS),
        "quotas": quotas,
        "selected_base_fact_count": len(selected_ids),
        "selected_base_fact_ids": selected_ids,
        "selected_base_fact_ids_by_relation_split": dict(selected_by_group),
    }
    loaded_outcomes = Counter(_review_outcome(row) for row in staging_by_id.values())
    revision_trigger_counts = Counter(
        str(revisions[base_fact_id]["entry"]["trigger"])
        for base_fact_id in consumed_revision_ids
    )
    rereview_decision_counts = Counter(
        str(revisions[base_fact_id]["entry"]["rereview"]["decision"])
        for base_fact_id in consumed_revision_ids
    )
    cohort_resolution_disposition_counts = Counter(
        str(row["cohort_eligibility_disposition"])
        for row in cohort_resolutions.values()
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": "provisional_preperturbation_bundle_materialized",
        "source_behavior_record_count": len(source_rows),
        "frame_record_count": len(frame_rows),
        "review_staging_record_count": len(staging_by_id),
        "review_outcome_counts_loaded": dict(sorted(loaded_outcomes.items())),
        "review_outcome_consolidation_used": consolidation_evidence is not None,
        "cohort_resolution_record_count": len(cohort_resolutions),
        "cohort_resolution_disposition_counts": dict(
            sorted(cohort_resolution_disposition_counts.items())
        ),
        "selected_base_fact_count": len(selected_ids),
        "selected_counts_by_relation_split": dict(sorted(selected_counts.items())),
        "revision_count": len(consumed_revision_ids),
        "revision_trigger_counts": dict(sorted(revision_trigger_counts.items())),
        "semantic_closure_disambiguation_count": revision_trigger_counts.get(
            "semantic_closure_disambiguation", 0
        ),
        "review_requested_revision_count": revision_trigger_counts.get(
            "review_requested_revision", 0
        ),
        "rereview_decision_counts": dict(sorted(rereview_decision_counts.items())),
        "distractor_candidate_count": len(distractor_rows),
        "distractors_per_fact": DISTRACTORS_PER_FACT,
        **scan_summary,
        "safety_contract": SAFETY_CONTRACT,
        "next_required_stage": (
            "independent_distractor_review_then_explicit_freeze_decision"
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        filenames = {
            "selected_base_fact_ids": "selected_base_fact_ids.json",
            "revision_lineage": "revision_lineage.jsonl",
            "revision_rereview": "revision_rereview.jsonl",
            "behavior_bundle": "preperturbation_behavior_input_bundle.jsonl",
            "distractor_candidates": "distractor_candidates.jsonl",
            "summary": "summary.json",
            "selection_manifest": "selection_manifest.json",
        }
        write_json(temporary_dir / filenames["selected_base_fact_ids"], selected_ids_document)
        write_jsonl(temporary_dir / filenames["revision_lineage"], lineage_rows)
        write_jsonl(temporary_dir / filenames["revision_rereview"], rereview_rows)
        write_jsonl(temporary_dir / filenames["behavior_bundle"], behavior_rows)
        write_jsonl(temporary_dir / filenames["distractor_candidates"], distractor_rows)
        write_json(temporary_dir / filenames["summary"], summary)

        def output_binding(
            key: str, schema_version: str, record_count: Optional[int] = None
        ) -> Dict[str, Any]:
            return _file_binding(
                temporary_dir / filenames[key],
                schema_version=schema_version,
                record_count=record_count,
                published_path=output_dir / filenames[key],
            )

        selection_inputs = {
            "source_behavior_bundle": _file_binding(
                source_bundle_path,
                schema_version=SOURCE_BUNDLE_SCHEMA_VERSION,
                record_count=len(source_rows),
            ),
            "frame": _file_binding(
                frame_path,
                schema_version=FRAME_SCHEMA_VERSION,
                record_count=len(frame_rows),
            ),
            "relation_policy": _file_binding(
                relation_policy_path,
                schema_version=POLICY_SCHEMA_VERSION,
            ),
            "review_apply_manifests": review_manifest_bindings,
            "review_scope_source_artifacts": review_source_bindings,
            "revision_plan": _file_binding(
                revision_plan_path,
                schema_version=REVISION_PLAN_SCHEMA_VERSION,
            ),
        }
        if consolidation_evidence is not None:
            selection_inputs["review_outcome_consolidation"] = consolidation_evidence

        selection_manifest = {
            "schema_version": SELECTION_MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "inputs": selection_inputs,
            "selection_contract": {
                "relation_order": list(relation_specs),
                "split_order": list(SPLITS),
                "quotas": quotas,
                "rank_field": "frame_rank_within_relation_split",
                "accepted_review_outcomes": ["accept"],
                "accepted_revised_outcome": "revise_with_terminal_rereview_accept",
                "semantic_closure_revision_trigger": (
                    "semantic_closure_disambiguation"
                ),
                "unresolved_before_quota_policy": "fail_closed",
                "unused_revision_entry_policy": "fail_closed",
                "cohort_exclude_semantics": (
                    "current_preperturbation_cohort_only_not_factual_rejection"
                    if consolidation_evidence is not None
                    else "not_applicable"
                ),
            },
            "selected_base_fact_count": len(selected_ids),
            "selected_counts_by_relation_split": dict(sorted(selected_counts.items())),
            "revision_count": len(consumed_revision_ids),
            "revision_trigger_counts": dict(sorted(revision_trigger_counts.items())),
            "cohort_resolution_record_count": len(cohort_resolutions),
            "cohort_resolution_disposition_counts": dict(
                sorted(cohort_resolution_disposition_counts.items())
            ),
            "distractor_candidate_count": len(distractor_rows),
            "artifacts": {
                "selected_base_fact_ids": output_binding(
                    "selected_base_fact_ids", SELECTED_IDS_SCHEMA_VERSION
                ),
                "revision_lineage": output_binding(
                    "revision_lineage",
                    REVISION_LINEAGE_SCHEMA_VERSION,
                    len(lineage_rows),
                ),
                "revision_rereview": output_binding(
                    "revision_rereview", REREVIEW_SCHEMA_VERSION, len(rereview_rows)
                ),
                "preperturbation_behavior_input_bundle": output_binding(
                    "behavior_bundle",
                    SOURCE_BUNDLE_SCHEMA_VERSION,
                    len(behavior_rows),
                ),
                "distractor_candidates": output_binding(
                    "distractor_candidates",
                    DISTRACTOR_SCHEMA_VERSION,
                    len(distractor_rows),
                ),
                "summary": output_binding("summary", SUMMARY_SCHEMA_VERSION),
            },
            "safety_contract": SAFETY_CONTRACT,
        }
        write_json(
            temporary_dir / filenames["selection_manifest"], selection_manifest
        )
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "selected_base_fact_count": len(selected_ids),
        "distractor_candidate_count": len(distractor_rows),
        "revision_count": len(consumed_revision_ids),
        "selection_manifest_path": str(output_dir / "selection_manifest.json"),
        "summary_path": str(output_dir / "summary.json"),
        "behavior_bundle_path": str(
            output_dir / "preperturbation_behavior_input_bundle.jsonl"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument(
        "--review-apply-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--revision-plan", type=Path, required=True)
    parser.add_argument(
        "--relation-policy", type=Path, default=DEFAULT_RELATION_POLICY
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-quota", type=int, default=24)
    parser.add_argument("--validation-quota", type=int, default=8)
    parser.add_argument("--sealed-quota", type=int, default=8)
    parser.add_argument("--expected-frame-count", type=int, default=311)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_preperturbation_bundle(
        source_bundle_path=args.source_bundle,
        frame_path=args.frame,
        review_apply_manifest_paths=args.review_apply_manifest,
        revision_plan_path=args.revision_plan,
        relation_policy_path=args.relation_policy,
        output_dir=args.output_dir,
        development_quota=args.development_quota,
        validation_quota=args.validation_quota,
        sealed_quota=args.sealed_quota,
        expected_frame_count=args.expected_frame_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
