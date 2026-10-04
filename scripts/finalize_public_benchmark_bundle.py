#!/usr/bin/env python3
"""Declare a formal public-benchmark cohort and preflight later finalization.

This tool is intentionally fail-closed.  It can materialize an immutable cohort
universe and audit whether the evidence needed by a future reviewed-bundle
finalizer is present.  It does not apply revisions, promote rows, recompute a
split, or emit review/split freeze manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "public-benchmark-finalizer-preflight-v2"
UNIVERSE_ITEM_SCHEMA_VERSION = "public-benchmark-formal-cohort-item-v1"
UNIVERSE_MANIFEST_SCHEMA_VERSION = "public-benchmark-formal-cohort-universe-v2"
PREFLIGHT_SCHEMA_VERSION = "public-benchmark-finalizer-preflight-v2"

REVIEW_SCOPE_SCHEMA_VERSION = "public-benchmark-review-scope-manifest-v2"
REVIEW_TOOL_VERSION = "public-benchmark-review-tool-v2"
REVIEW_APPLY_SCHEMA_VERSION = "public-benchmark-review-apply-manifest-v2"
REVIEW_STAGING_SCHEMA_VERSION = "public-benchmark-review-staging-record-v2"
NEAR_AUDIT_SCHEMA_VERSION = "public-benchmark-lexical-candidate-audit-v2"
NEAR_PAIR_SCHEMA_VERSION = "public-benchmark-lexical-candidate-pair-v2"
NEAR_ADJUDICATION_SCHEMA_VERSION = (
    "public-benchmark-near-duplicate-adjudication-v1"
)
REVISION_LINEAGE_SCHEMA_VERSION = "public-benchmark-revision-lineage-v2"
REREVIEW_DECISION_SCHEMA_VERSION = "public-benchmark-final-rereview-decision-v2"
SEMANTIC_CLOSURE_SCHEMA_VERSION = "public-benchmark-semantic-closure-attestation-v1"
HISTORICAL_REGISTRY_SCHEMA_VERSION = (
    "public-benchmark-historical-exposure-registry-v1"
)
HISTORICAL_CHECK_SCHEMA_VERSION = "public-benchmark-historical-exposure-check-v1"
RECLUSTER_MANIFEST_SCHEMA_VERSION = "public-benchmark-recluster-manifest-v1"
SPLIT_RECOMPUTE_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-split-recompute-manifest-v1"
)

SOURCE_ARTIFACTS = {
    "behavior_bundle": (
        "behavior_input_bundle.jsonl",
        "factual-perturbation-input-bundle-v1",
        "base_fact_id",
    ),
    "review_queue": (
        "review_queue.jsonl",
        "provisional-semantic-review-v1",
        "base_fact_id",
    ),
    "base_fact_clusters": (
        "base_fact_clusters.jsonl",
        "provisional-base-fact-cluster-v1",
        "base_fact_id",
    ),
    "provisional_triples": (
        "provisional_triples.jsonl",
        "provisional-factual-triple-v1",
        "candidate_id",
    ),
}

SOURCE_SCOPE_BINDING_FIELDS = {
    "path",
    "sha256",
    "byte_count",
    "record_count",
    "schema_version",
}
COMPARISON_BINDING_FIELDS = {
    "path",
    "sha256",
    "byte_count",
    "record_count",
    "dataset_id",
    "dataset_role",
    "format_id",
    "policy_versions",
    "ordered_record_ids_sha256",
    "ordered_row_hashes_sha256",
}
COMPARISON_CANONICAL_REQUIRED_FIELDS = frozenset(
    {
        "source_id",
        "candidate_id",
        "source_dataset",
        "source_question",
        "canonical_fact",
        "answer",
        "source_answer",
        "canonical_policy_version",
        "canonical_decision",
    }
)
COMPARISON_CANONICAL_DATASET_ROLE = "historical_canonical_v2_comparison"
DEFAULT_COMPARISON_DATASET_ID = "canonical-v2"
TRUSTED_COMPARISON_CONTRACT_FIELDS = frozenset(
    {
        "dataset_role",
        "format_id",
        "sha256",
        "byte_count",
        "record_count",
        "policy_versions",
        "ordered_record_ids_sha256",
        "ordered_row_hashes_sha256",
    }
)
TRUSTED_COMPARISON_CONTRACTS: Dict[str, Dict[str, Any]] = {
    DEFAULT_COMPARISON_DATASET_ID: {
        "dataset_role": COMPARISON_CANONICAL_DATASET_ROLE,
        "format_id": "canonical-triples-v2",
        "sha256": "4fb7addaf141c63588a9442edfe59a77c3ab402d0dfc2799dd7b3619f0067acc",
        "byte_count": 6358433,
        "record_count": 2450,
        "policy_versions": ["codex-adjudicated-canonical-v2"],
        "ordered_record_ids_sha256": (
            "ccd533be2cb14a7909c7639afb6b438ef55dcd2f553a664a997e26d25e3e40d1"
        ),
        "ordered_row_hashes_sha256": (
            "5c849d1c8fd8be01435bdb21d728fcffd507bc0834f00aa4c7a2693ebce4691f"
        ),
    }
}
NEAR_AUDIT_CONTRACT_FIELDS = frozenset(
    {"summary_schema_version", "pair_schema_version", "policy"}
)
NEAR_AUDIT_POLICY_FIELDS = frozenset(
    {
        "text_normalization_version",
        "answer_normalization_version",
        "blocking_version",
        "question_near_threshold",
        "canonical_fact_near_threshold",
        "max_bucket_neighbors",
        "max_examples",
        "minhash_components",
        "minhash_band_rows",
        "deterministic",
        "network_or_model_used",
        "pair_review_contract",
    }
)
HISTORICAL_EXPOSURE_CONTRACT_STATUS = "not_predeclared"
FREEZE_OUTPUT_NAMES = {
    "reviewed_behavior_input_bundle.jsonl",
    "review_freeze_manifest.json",
    "split_freeze_manifest.json",
}
FREEZE_OUTPUT_NAMES_CASEFOLD = frozenset(
    name.casefold() for name in FREEZE_OUTPUT_NAMES
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

REVIEW_APPLY_FIELDS = {
    "schema_version",
    "tool_version",
    "scope_id",
    "scope_manifest",
    "export_manifest",
    "decisions",
    "allow_partial",
    "scoped_base_fact_count",
    "outcome_counts",
    "artifacts",
    "safety_contract",
}
REVIEW_APPLY_COMPOSITE_FIELDS = REVIEW_APPLY_FIELDS | {"decision_authority"}
COMPOSITE_REVIEW_AUTHORITY_MODE = "composite_fact_review_authority"
COMPOSITE_REVIEW_AUTHORITY_SCHEMA_VERSION = (
    "full-public-benchmark-fact-review-composite-authority-manifest-v1"
)
REVIEW_APPLY_SAFETY_CONTRACT = {
    "output_is_staging_only": True,
    "source_artifacts_modified": False,
    "canonical_freeze_performed": False,
    "automatic_promotion_performed": False,
    "missing_is_reject": False,
    "revise_outcome_supported": True,
    "revision_application_supported": False,
}
EXACT_DUPLICATE_MATCH_TYPES = frozenset(
    {"question_exact", "canonical_fact_exact"}
)
REVISION_PAYLOAD_FIELDS = frozenset(
    {
        "base_fact_id",
        "subject_en",
        "relation_raw",
        "answer_en",
        "answer_type",
        "answer_aliases_en",
        "canonical_fact",
        "canonical_fact_en",
    }
)
REVISION_MUTABLE_FIELDS = REVISION_PAYLOAD_FIELDS - {"base_fact_id"}
REVISION_LINEAGE_FIELDS = frozenset(
    {
        "schema_version",
        "source_base_fact_id",
        "universe_id",
        "cohort_item_id",
        "revision_status",
        "revised_record_role",
        "derived_fields_recomputed",
        "source_review_staging_row_sha256",
        "editor_type",
        "editor_id",
        "edit_method",
        "edited_at",
        "before_record_sha256",
        "after_record_sha256",
        "revision_payload",
        "changed_fields",
        "revised_record",
    }
)
REREVIEW_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "source_base_fact_id",
        "universe_id",
        "cohort_item_id",
        "revised_record_sha256",
        "decision",
        "reviewer_type",
        "reviewer_id",
        "review_method",
        "reviewed_at",
        "human_gold",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _validate_integer_fields(value: Any, label: str) -> None:
    """Reject bool/float count and index lookalikes throughout bound JSON."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if child is not None and isinstance(key, str):
                if key == "count" or key.endswith("_count"):
                    _require_int(child, child_label)
                elif key == "index" or key.endswith("_index"):
                    _require_int(child, child_label)
                elif key == "line_number" or key.endswith("_line_number"):
                    _require_int(child, child_label, minimum=1)
            _validate_integer_fields(child, child_label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_integer_fields(child, f"{label}[{index}]")


def _mapping_list(value: Any, field: str, *, allow_empty: bool = True) -> List[Mapping[str, Any]]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a{' non-empty' if not allow_empty else ''} list")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
    return value


def _load_sibling_module(filename: str, module_name: str) -> Any:
    """Load a sibling producer explicitly so preflight can deterministically replay it."""

    path = Path(__file__).resolve().with_name(filename)
    existing = sys.modules.get(module_name)
    if existing is not None and Path(str(existing.__file__)).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
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
        os.replace(temporary_name, str(path))
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


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    _atomic_write(path, payload)


def _decode_json(raw: bytes, path: Path) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON snapshot: {path}: {exc}") from exc


def read_json_snapshot(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    value = _decode_json(raw, path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    _validate_integer_fields(value, str(path))
    return value, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "schema_version": value.get("schema_version"),
    }


def read_jsonl_snapshot(
    path: Path, *, allow_empty: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL snapshot: {path}: {exc}") from exc
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        _validate_integer_fields(value, f"{path}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "record_count": len(rows),
    }


def read_jsonl_line_snapshot(
    path: Path,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Read JSONL while preserving physical line numbers used by audit provenance."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL snapshot: {path}: {exc}") from exc
    rows_by_line: Dict[int, Dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        _validate_integer_fields(value, f"{path}:{line_number}")
        rows_by_line[line_number] = value
    if not rows_by_line:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows_by_line, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "record_count": len(rows_by_line),
    }


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalized_semantic_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        identifier = _required_string(row.get(field), f"{label} row {index}.{field}")
        if identifier in result:
            raise ValueError(f"Duplicate {field} in {label}: {identifier}")
        result[identifier] = row
    return result


def source_paths(source_dir: Path) -> Dict[str, Path]:
    root = Path(source_dir).resolve()
    return {
        role: (root / filename).resolve()
        for role, (filename, _, _) in SOURCE_ARTIFACTS.items()
    }


def _trusted_comparison_contract(dataset_id: str) -> Dict[str, Any]:
    identifier = _required_string(dataset_id, "comparison_dataset_id")
    contract = TRUSTED_COMPARISON_CONTRACTS.get(identifier)
    if not isinstance(contract, dict) or set(contract) != (
        TRUSTED_COMPARISON_CONTRACT_FIELDS
    ):
        raise ValueError(
            f"Unknown or invalid trusted comparison dataset contract: {identifier}"
        )
    return dict(contract)


def load_comparison_canonical_snapshot(
    path: Path, dataset_id: str
) -> Dict[str, Any]:
    identifier = _required_string(dataset_id, "comparison_dataset_id")
    trusted_contract = _trusted_comparison_contract(identifier)
    rows, snapshot = read_jsonl_snapshot(path)
    record_ids: List[Dict[str, str]] = []
    seen_source_ids = set()
    seen_candidate_ids = set()
    policy_versions = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(COMPARISON_CANONICAL_REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"comparison_canonical row {index} is missing required fields: {missing}"
            )
        required = {
            field: _required_string(
                row.get(field), f"comparison_canonical row {index}.{field}"
            )
            for field in sorted(COMPARISON_CANONICAL_REQUIRED_FIELDS)
        }
        if required["canonical_decision"] != "keep":
            raise ValueError(
                f"comparison_canonical row {index} canonical_decision must be keep"
            )
        source_id = required["source_id"]
        candidate_id = required["candidate_id"]
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate source_id in comparison_canonical: {source_id}")
        if candidate_id in seen_candidate_ids:
            raise ValueError(
                f"Duplicate candidate_id in comparison_canonical: {candidate_id}"
            )
        seen_source_ids.add(source_id)
        seen_candidate_ids.add(candidate_id)
        policy_versions.add(required["canonical_policy_version"])
        record_ids.append(
            {"source_id": source_id, "candidate_id": candidate_id}
        )
    if len(policy_versions) != 1:
        raise ValueError(
            "comparison_canonical must use exactly one non-empty "
            "canonical_policy_version"
        )
    actual_contract = {
        "dataset_role": COMPARISON_CANONICAL_DATASET_ROLE,
        "format_id": "canonical-triples-v2",
        "sha256": snapshot["sha256"],
        "byte_count": snapshot["byte_count"],
        "record_count": snapshot["record_count"],
        "policy_versions": sorted(policy_versions),
        "ordered_record_ids_sha256": sha256_value(record_ids),
        "ordered_row_hashes_sha256": sha256_value(
            [sha256_value(row) for row in rows]
        ),
    }
    if not _json_equal(actual_contract, trusted_contract):
        raise ValueError(
            "comparison_canonical does not match its trusted dataset contract: "
            f"{identifier}"
        )
    return {
        **{
            field: snapshot[field]
            for field in ("path", "sha256", "byte_count", "record_count")
        },
        "dataset_id": identifier,
        **{
            field: trusted_contract[field]
            for field in (
                "dataset_role",
                "format_id",
                "policy_versions",
                "ordered_record_ids_sha256",
                "ordered_row_hashes_sha256",
            )
        },
    }


def build_near_audit_contract(
    *,
    question_threshold: float,
    fact_threshold: float,
    max_bucket_neighbors: int,
    max_examples: int,
) -> Dict[str, Any]:
    if type(question_threshold) is not float or not 0.0 < question_threshold <= 1.0:
        raise ValueError("near question threshold must be a float in (0, 1]")
    if type(fact_threshold) is not float or not 0.0 < fact_threshold <= 1.0:
        raise ValueError("near fact threshold must be a float in (0, 1]")
    neighbors = _require_int(
        max_bucket_neighbors, "near max_bucket_neighbors", minimum=1
    )
    examples = _require_int(max_examples, "near max_examples", minimum=1)
    return {
        "summary_schema_version": NEAR_AUDIT_SCHEMA_VERSION,
        "pair_schema_version": NEAR_PAIR_SCHEMA_VERSION,
        "policy": {
            "text_normalization_version": "nfkc-casefold-unicode-word-v1",
            "answer_normalization_version": "nfkc-casefold-whitespace-v1",
            "blocking_version": "token-minhash-16x2-lexical-window-v1",
            "question_near_threshold": question_threshold,
            "canonical_fact_near_threshold": fact_threshold,
            "max_bucket_neighbors": neighbors,
            "max_examples": examples,
            "minhash_components": 16,
            "minhash_band_rows": 2,
            "deterministic": True,
            "network_or_model_used": False,
            "pair_review_contract": {
                "status": "pending_adjudication",
                "answer_alias_exact_requires": [
                    "alias_valid",
                    "same_answer_entity",
                ],
                "alias_valid_scope": (
                    "pair_level_at_least_one_shared_alias_is_valid_for_both_records"
                ),
                "question_or_fact_exact_or_near_requires": [
                    "semantic_duplicate"
                ],
                "mixed_evidence_requires_union": True,
            },
        },
    }


def validate_comparison_source_separation(
    comparison_path: Path,
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject a comparison corpus that aliases any provisional source artifact."""

    resolved = Path(comparison_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    comparison_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    for role, binding in source_bindings.items():
        source_path = Path(
            _required_string(binding.get("path"), f"source_artifacts.{role}.path")
        ).resolve()
        if resolved == source_path:
            raise ValueError(
                "comparison_canonical must not reuse source artifact path: "
                f"{role}"
            )
        if comparison_sha256 == binding.get("sha256"):
            raise ValueError(
                "comparison_canonical must not reuse source artifact SHA: "
                f"{role}"
            )


def load_source_snapshot(
    source_dir: Path,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, Mapping[str, Any]]],
    Dict[str, Dict[str, Any]],
]:
    """Load and cross-check the four provisional artifacts from byte snapshots."""

    paths = source_paths(source_dir)
    rows_by_role: Dict[str, List[Dict[str, Any]]] = {}
    indexes: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    bindings: Dict[str, Dict[str, Any]] = {}
    for role, (_, schema_version, id_field) in SOURCE_ARTIFACTS.items():
        rows, snapshot = read_jsonl_snapshot(paths[role])
        for index, row in enumerate(rows, start=1):
            if row.get("schema_version") != schema_version:
                raise ValueError(
                    f"Unsupported {role} schema at row {index}: "
                    f"{row.get('schema_version')!r}"
                )
            if row.get("canonical_status") != "pending_review":
                raise ValueError(f"{role} row {index} is not pending_review")
            if row.get("human_gold") is not False:
                raise ValueError(f"{role} row {index} must have human_gold=false")
            if row.get("evidence_tier") != "provisional_single_model":
                raise ValueError(f"{role} row {index} is not provisional_single_model")
            if role == "behavior_bundle" and row.get("bundle_status") != "draft_pending_review":
                raise ValueError(f"behavior_bundle row {index} is not draft_pending_review")
        index = _unique_index(rows, id_field, role)
        identifiers = list(index)
        rows_by_role[role] = rows
        indexes[role] = index
        bindings[role] = {
            **snapshot,
            "schema_version": schema_version,
            "record_ids_sha256": sha256_value(identifiers),
            "record_row_hashes_sha256": sha256_value(
                [sha256_value(row) for row in rows]
            ),
        }

    base_id_sets = [
        set(indexes[role])
        for role in ("behavior_bundle", "review_queue", "base_fact_clusters")
    ]
    triple_ids_by_fact: Dict[str, List[str]] = {}
    for candidate_id, row in indexes["provisional_triples"].items():
        base_fact_id = _required_string(
            row.get("base_fact_id"),
            f"provisional_triples[{candidate_id}].base_fact_id",
        )
        triple_ids_by_fact.setdefault(base_fact_id, []).append(candidate_id)
    if not (base_id_sets[0] == base_id_sets[1] == base_id_sets[2] == set(triple_ids_by_fact)):
        raise ValueError("Source artifact base_fact_id universes differ")

    for base_fact_id in sorted(base_id_sets[0]):
        cluster = indexes["base_fact_clusters"][base_fact_id]
        review = indexes["review_queue"][base_fact_id]
        behavior = indexes["behavior_bundle"][base_fact_id]
        members = _mapping_list(
            cluster.get("members"),
            f"cluster {base_fact_id}.members",
            allow_empty=False,
        )
        member_index = _unique_index(
            members, "candidate_id", f"cluster {base_fact_id} members"
        )
        expected_members = set(triple_ids_by_fact[base_fact_id])
        if set(member_index) != expected_members:
            raise ValueError(f"Cluster {base_fact_id} members do not match triples")
        if cluster.get("member_count") != len(expected_members):
            raise ValueError(f"Cluster {base_fact_id} member_count is inconsistent")
        review_members = review.get("member_candidate_ids")
        if (
            not isinstance(review_members, list)
            or any(not isinstance(value, str) or not value.strip() for value in review_members)
            or len(review_members) != len(set(review_members))
            or set(review_members) != expected_members
        ):
            raise ValueError(f"Review row {base_fact_id} members do not match cluster")
        representative = _required_string(
            cluster.get("representative_candidate_id"),
            f"cluster {base_fact_id}.representative_candidate_id",
        )
        if representative not in expected_members:
            raise ValueError(f"Cluster {base_fact_id} representative is not a member")
        if behavior.get("candidate_id") != representative:
            raise ValueError(
                f"Behavior row {base_fact_id} does not bind the representative"
            )
        for candidate_id, member in member_index.items():
            triple = indexes["provisional_triples"][candidate_id]
            provenance = triple.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"Triple {candidate_id} has no provenance")
            if member.get("input_record_sha256") != provenance.get(
                "input_record_sha256"
            ):
                raise ValueError(f"Cluster member {candidate_id} provenance mismatch")
    return rows_by_role, indexes, bindings


def _scope_compatible_binding(binding: Mapping[str, Any]) -> Dict[str, Any]:
    return {field: binding[field] for field in SOURCE_SCOPE_BINDING_FIELDS}


def _scope_id(
    source_bindings: Mapping[str, Mapping[str, Any]],
    base_fact_ids: Sequence[str],
    selection_source: Mapping[str, Any],
) -> str:
    return "review_scope_" + sha256_value(
        {
            "tool_version": REVIEW_TOOL_VERSION,
            "source_artifacts": {
                role: {
                    field: source_bindings[role][field]
                    for field in sorted(SOURCE_SCOPE_BINDING_FIELDS)
                }
                for role in sorted(source_bindings)
            },
            "base_fact_ids": list(base_fact_ids),
            "selection_source": selection_source,
        }
    )[:20]


def validate_review_scope(
    path: Path,
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest, provenance = read_json_snapshot(path)
    if manifest.get("schema_version") != REVIEW_SCOPE_SCHEMA_VERSION:
        raise ValueError("Unsupported review scope schema_version")
    if manifest.get("tool_version") != REVIEW_TOOL_VERSION:
        raise ValueError("Unsupported review scope tool_version")
    if manifest.get("scope_status") != "frozen_review_scope":
        raise ValueError("Review scope is not frozen")
    if manifest.get("canonical_freeze_authorized") is not False:
        raise ValueError("Review scope must not authorize canonical freeze")
    selected = manifest.get("base_fact_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Review scope base_fact_ids must be non-empty")
    selected_ids = [
        _required_string(value, "review scope base_fact_id") for value in selected
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Review scope contains duplicate base_fact_id values")
    if manifest.get("base_fact_count") != len(selected_ids):
        raise ValueError("Review scope base_fact_count mismatch")
    unknown = set(selected_ids) - set(indexes["behavior_bundle"])
    if unknown:
        raise ValueError(f"Review scope contains unknown base_fact_id: {sorted(unknown)[:5]}")

    declared_sources = manifest.get("source_artifacts")
    if not isinstance(declared_sources, dict) or set(declared_sources) != set(
        SOURCE_ARTIFACTS
    ):
        raise ValueError("Review scope source_artifacts are incomplete")
    for role in SOURCE_ARTIFACTS:
        if not _json_equal(
            declared_sources[role],
            _scope_compatible_binding(source_bindings[role]),
        ):
            raise ValueError(f"Review scope has stale source binding for {role}")

    selection_source = manifest.get("selection_source")
    if not isinstance(selection_source, dict):
        raise ValueError("Review scope selection_source is missing")
    mode = selection_source.get("mode")
    if mode == "explicit_base_fact_ids":
        if selection_source.get("record_count") != len(selected_ids):
            raise ValueError("Explicit review scope record_count mismatch")
        if selection_source.get("base_fact_ids_sha256") != sha256_value(selected_ids):
            raise ValueError("Explicit review scope ID digest mismatch")
    elif mode == "review_sample":
        sample_path = Path(
            _required_string(selection_source.get("path"), "selection_source.path")
        ).resolve()
        sample_rows, sample_snapshot = read_jsonl_snapshot(sample_path)
        for field in ("sha256", "byte_count", "record_count"):
            if selection_source.get(field) != sample_snapshot[field]:
                raise ValueError(f"Review scope has stale review sample {field}")
        if selection_source.get("schema_version") != "provisional-semantic-review-sample-v1":
            raise ValueError("Review scope sample schema_version mismatch")
        sample_ids = [str(row.get("base_fact_id") or "").strip() for row in sample_rows]
        if sample_ids != selected_ids:
            raise ValueError("Review scope sample order differs from base_fact_ids")
        expected_bindings = []
        for index, (base_fact_id, sample_row) in enumerate(
            zip(selected_ids, sample_rows)
        ):
            if sample_row.get("schema_version") != (
                "provisional-semantic-review-sample-v1"
            ):
                raise ValueError(
                    f"Review sample row {base_fact_id} has invalid schema_version"
                )
            queue_row = indexes["review_queue"][base_fact_id]
            mismatches = [
                field
                for field, value in queue_row.items()
                if field != "schema_version" and sample_row.get(field) != value
            ]
            if mismatches:
                raise ValueError(
                    f"Review sample row {base_fact_id} differs from review queue"
                )
            expected_bindings.append(
                {
                    "sample_index": index,
                    "base_fact_id": base_fact_id,
                    "sample_row_sha256": sha256_value(sample_row),
                    "review_queue_row_sha256": sha256_value(queue_row),
                }
            )
        if not _json_equal(selection_source.get("row_bindings"), expected_bindings):
            raise ValueError("Review scope sample row bindings mismatch")
        if selection_source.get("row_bindings_sha256") != sha256_value(
            expected_bindings
        ):
            raise ValueError("Review scope sample row binding digest mismatch")
    else:
        raise ValueError(f"Unsupported review scope selection mode: {mode!r}")

    expected_scope_id = _scope_id(
        declared_sources, selected_ids, selection_source
    )
    if manifest.get("scope_id") != expected_scope_id:
        raise ValueError("Review scope_id does not match its content")
    provenance.update({"scope_id": expected_scope_id, "record_count": len(selected_ids)})
    return manifest, provenance


def _universe_item(
    *,
    universe_id: str,
    selection_index: int,
    base_fact_id: str,
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_bundle_sha256: str,
) -> Dict[str, Any]:
    behavior = indexes["behavior_bundle"][base_fact_id]
    review = indexes["review_queue"][base_fact_id]
    cluster = indexes["base_fact_clusters"][base_fact_id]
    member_bindings = []
    for member in cluster["members"]:
        candidate_id = str(member["candidate_id"])
        triple = indexes["provisional_triples"][candidate_id]
        member_bindings.append(
            {
                "candidate_id": candidate_id,
                "triple_record_sha256": sha256_value(triple),
                "source_input_record_sha256": member["input_record_sha256"],
            }
        )
    return {
        "schema_version": UNIVERSE_ITEM_SCHEMA_VERSION,
        "universe_id": universe_id,
        "selection_index": selection_index,
        "cohort_item_id": "cohort_item_"
        + sha256_value([source_bundle_sha256, base_fact_id])[:24],
        "source_base_fact_id": base_fact_id,
        "source_id": behavior.get("source_id"),
        "source_row_bindings": {
            "behavior_row_sha256": sha256_value(behavior),
            "review_queue_row_sha256": sha256_value(review),
            "base_fact_cluster_row_sha256": sha256_value(cluster),
        },
        "member_count": len(member_bindings),
        "member_bindings": member_bindings,
    }


def materialize_cohort_universe(
    *,
    source_dir: Path,
    output_dir: Path,
    mode: str,
    universe_label: str,
    selection_policy_id: str,
    comparison_canonical_path: Path,
    comparison_dataset_id: str,
    near_question_threshold: float = 0.80,
    near_fact_threshold: float = 0.80,
    near_max_bucket_neighbors: int = 24,
    near_max_examples: int = 20,
    review_scope_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Materialize a selected universe without making any review/freeze claim."""

    if mode not in {"full-pool", "review-scope"}:
        raise ValueError("mode must be full-pool or review-scope")
    label = _required_string(universe_label, "universe_label")
    policy_id = _required_string(selection_policy_id, "selection_policy_id")
    comparison_id = _required_string(
        comparison_dataset_id, "comparison_dataset_id"
    )
    near_audit_contract = build_near_audit_contract(
        question_threshold=near_question_threshold,
        fact_threshold=near_fact_threshold,
        max_bucket_neighbors=near_max_bucket_neighbors,
        max_examples=near_max_examples,
    )
    rows_by_role, indexes, source_bindings = load_source_snapshot(source_dir)
    validate_comparison_source_separation(
        comparison_canonical_path, source_bindings
    )
    comparison_binding = load_comparison_canonical_snapshot(
        comparison_canonical_path, comparison_id
    )

    if mode == "full-pool":
        if review_scope_manifest_path is not None:
            raise ValueError("full-pool mode must not receive a review scope")
        selected_ids = [str(row["base_fact_id"]) for row in rows_by_role["behavior_bundle"]]
        selection_source: Dict[str, Any] = {
            "mode": "full_pool",
            "source_role": "behavior_bundle",
            "record_count": len(selected_ids),
            "ordered_base_fact_ids_sha256": sha256_value(selected_ids),
        }
    else:
        if review_scope_manifest_path is None:
            raise ValueError("review-scope mode requires review_scope_manifest_path")
        scope, scope_provenance = validate_review_scope(
            review_scope_manifest_path, indexes, source_bindings
        )
        selected_ids = [str(value) for value in scope["base_fact_ids"]]
        selection_source = {
            "mode": "review_scope",
            "path": scope_provenance["path"],
            "sha256": scope_provenance["sha256"],
            "schema_version": REVIEW_SCOPE_SCHEMA_VERSION,
            "scope_id": scope["scope_id"],
            "scope_selection_mode": scope["selection_source"]["mode"],
            "record_count": len(selected_ids),
            "ordered_base_fact_ids_sha256": sha256_value(selected_ids),
        }

    universe_id = "formal_cohort_" + sha256_value(
        {
            "tool_version": TOOL_VERSION,
            "universe_label": label,
            "selection_policy_id": policy_id,
            "selection_source": selection_source,
            "source_artifacts": source_bindings,
            "comparison_canonical": comparison_binding,
            "near_duplicate_audit_contract": near_audit_contract,
            "historical_exposure_contract_status": (
                HISTORICAL_EXPOSURE_CONTRACT_STATUS
            ),
            "ordered_base_fact_ids": selected_ids,
        }
    )[:24]
    items = [
        _universe_item(
            universe_id=universe_id,
            selection_index=index,
            base_fact_id=base_fact_id,
            indexes=indexes,
            source_bundle_sha256=source_bindings["behavior_bundle"]["sha256"],
        )
        for index, base_fact_id in enumerate(selected_ids)
    ]

    output_dir = Path(output_dir).resolve()
    items_path = output_dir / "formal_cohort_universe_items.jsonl"
    manifest_path = output_dir / "formal_cohort_universe_manifest.json"
    for path in (items_path, manifest_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing cohort declaration artifact: {path}"
            )
    write_jsonl(items_path, items)
    item_row_hashes = [sha256_value(item) for item in items]
    manifest = {
        "schema_version": UNIVERSE_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "universe_id": universe_id,
        "universe_label": label,
        "universe_status": "declared_immutable_not_reviewed",
        "selection_policy": {
            "policy_id": policy_id,
            "mode": mode,
            "formal_cohort_purpose_explicit": True,
            "review_sample_inference_role_not_inherited": True,
        },
        "selection_source": selection_source,
        "source_artifacts": source_bindings,
        "comparison_canonical": comparison_binding,
        "near_duplicate_audit_contract": near_audit_contract,
        "historical_exposure_contract_status": (
            HISTORICAL_EXPOSURE_CONTRACT_STATUS
        ),
        "record_count": len(items),
        "ordered_cohort_item_ids_sha256": sha256_value(
            [item["cohort_item_id"] for item in items]
        ),
        "ordered_source_base_fact_ids_sha256": sha256_value(selected_ids),
        "ordered_item_row_hashes_sha256": sha256_value(item_row_hashes),
        "items": {
            "path": str(items_path),
            "sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
            "byte_count": items_path.stat().st_size,
            "record_count": len(items),
            "schema_version": UNIVERSE_ITEM_SCHEMA_VERSION,
        },
        "review_complete": False,
        "canonical_freeze_authorized": False,
        "split_freeze_authorized": False,
    }
    write_json(manifest_path, manifest)
    return {
        "universe_id": universe_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "items_path": str(items_path),
        "items_sha256": manifest["items"]["sha256"],
        "record_count": len(items),
        "comparison_canonical_sha256": comparison_binding["sha256"],
        "canonical_freeze_authorized": False,
    }


def load_cohort_universe(
    manifest_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    manifest, provenance = read_json_snapshot(manifest_path)
    if manifest.get("schema_version") != UNIVERSE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported formal cohort universe schema_version")
    if manifest.get("tool_version") != TOOL_VERSION:
        raise ValueError("Unsupported formal cohort universe tool_version")
    universe_id = _required_string(manifest.get("universe_id"), "universe_id")
    if manifest.get("universe_status") != "declared_immutable_not_reviewed":
        raise ValueError("Formal cohort universe status is invalid")
    if manifest.get("canonical_freeze_authorized") is not False:
        raise ValueError("Cohort declaration cannot authorize canonical freeze")
    if manifest.get("split_freeze_authorized") is not False:
        raise ValueError("Cohort declaration cannot authorize split freeze")
    if manifest.get("review_complete") is not False:
        raise ValueError("Cohort declaration cannot claim review completion")

    item_binding = manifest.get("items")
    if not isinstance(item_binding, dict):
        raise ValueError("Formal cohort universe is missing items binding")
    item_path_value = _required_string(item_binding.get("path"), "items.path")
    item_path = Path(item_path_value)
    if not item_path.is_absolute():
        item_path = (Path(manifest_path).resolve().parent / item_path).resolve()
    items, snapshot = read_jsonl_snapshot(item_path)
    for field in ("sha256", "byte_count", "record_count"):
        if item_binding.get(field) != snapshot[field]:
            raise ValueError(f"Formal cohort items {field} binding is stale")
    if item_binding.get("schema_version") != UNIVERSE_ITEM_SCHEMA_VERSION:
        raise ValueError("Formal cohort items schema binding is invalid")
    if manifest.get("record_count") != len(items):
        raise ValueError("Formal cohort universe record_count mismatch")

    cohort_ids: List[str] = []
    source_ids: List[str] = []
    for index, row in enumerate(items):
        if row.get("schema_version") != UNIVERSE_ITEM_SCHEMA_VERSION:
            raise ValueError(f"Formal cohort item {index} has invalid schema_version")
        if row.get("universe_id") != universe_id:
            raise ValueError(f"Formal cohort item {index} has wrong universe_id")
        if row.get("selection_index") != index:
            raise ValueError(f"Formal cohort item {index} has wrong selection_index")
        cohort_ids.append(_required_string(row.get("cohort_item_id"), "cohort_item_id"))
        source_ids.append(
            _required_string(row.get("source_base_fact_id"), "source_base_fact_id")
        )
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("Formal cohort universe contains duplicate cohort_item_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Formal cohort universe contains duplicate source_base_fact_id")
    if manifest.get("ordered_cohort_item_ids_sha256") != sha256_value(cohort_ids):
        raise ValueError("Formal cohort cohort-item digest mismatch")
    if manifest.get("ordered_source_base_fact_ids_sha256") != sha256_value(source_ids):
        raise ValueError("Formal cohort source-ID digest mismatch")
    if manifest.get("ordered_item_row_hashes_sha256") != sha256_value(
        [sha256_value(row) for row in items]
    ):
        raise ValueError("Formal cohort row-hash digest mismatch")

    source_bindings = manifest.get("source_artifacts")
    if not isinstance(source_bindings, dict) or set(source_bindings) != set(
        SOURCE_ARTIFACTS
    ):
        raise ValueError("Formal cohort source_artifacts are incomplete")
    for role, binding in source_bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"Formal cohort source binding is invalid for {role}")

    behavior_path = _resolve_binding_path(
        source_bindings["behavior_bundle"],
        Path(manifest_path).resolve(),
        "behavior_bundle",
    )
    rows_by_role, indexes, actual_source_bindings = load_source_snapshot(
        behavior_path.parent
    )
    if source_bindings != actual_source_bindings:
        raise ValueError("Formal cohort source artifact bindings are stale or incomplete")

    comparison_binding = manifest.get("comparison_canonical")
    if not isinstance(comparison_binding, dict) or set(comparison_binding) != (
        COMPARISON_BINDING_FIELDS
    ):
        raise ValueError("Formal cohort comparison_canonical binding is invalid")
    comparison_path = _resolve_binding_path(
        comparison_binding,
        Path(manifest_path).resolve(),
        "comparison_canonical",
    )
    validate_comparison_source_separation(
        comparison_path, actual_source_bindings
    )
    actual_comparison_binding = load_comparison_canonical_snapshot(
        comparison_path,
        _required_string(
            comparison_binding.get("dataset_id"),
            "comparison_canonical.dataset_id",
        ),
    )
    if not _json_equal(comparison_binding, actual_comparison_binding):
        raise ValueError("Formal cohort comparison_canonical binding is stale")

    near_audit_contract = manifest.get("near_duplicate_audit_contract")
    if not isinstance(near_audit_contract, dict) or set(near_audit_contract) != (
        NEAR_AUDIT_CONTRACT_FIELDS
    ):
        raise ValueError("Formal cohort near-duplicate audit contract is invalid")
    near_policy = near_audit_contract.get("policy")
    if not isinstance(near_policy, dict) or set(near_policy) != (
        NEAR_AUDIT_POLICY_FIELDS
    ):
        raise ValueError("Formal cohort near-duplicate policy is invalid")
    expected_near_audit_contract = build_near_audit_contract(
        question_threshold=near_policy.get("question_near_threshold"),
        fact_threshold=near_policy.get("canonical_fact_near_threshold"),
        max_bucket_neighbors=near_policy.get("max_bucket_neighbors"),
        max_examples=near_policy.get("max_examples"),
    )
    if not _json_equal(near_audit_contract, expected_near_audit_contract):
        raise ValueError("Formal cohort near-duplicate audit contract is unsupported")
    if manifest.get("historical_exposure_contract_status") != (
        HISTORICAL_EXPOSURE_CONTRACT_STATUS
    ):
        raise ValueError("Formal cohort historical exposure contract status is invalid")

    universe_label = _required_string(manifest.get("universe_label"), "universe_label")
    selection_policy = manifest.get("selection_policy")
    if not isinstance(selection_policy, dict):
        raise ValueError("Formal cohort selection_policy is invalid")
    policy_id = _required_string(
        selection_policy.get("policy_id"), "selection_policy.policy_id"
    )
    mode = selection_policy.get("mode")
    expected_policy = {
        "policy_id": policy_id,
        "mode": mode,
        "formal_cohort_purpose_explicit": True,
        "review_sample_inference_role_not_inherited": True,
    }
    if not _json_equal(selection_policy, expected_policy) or mode not in {
        "full-pool",
        "review-scope",
    }:
        raise ValueError("Formal cohort selection_policy is invalid")
    selection_source = manifest.get("selection_source")
    if not isinstance(selection_source, dict):
        raise ValueError("Formal cohort selection_source is invalid")

    if mode == "full-pool":
        selected_ids = [
            str(row["base_fact_id"]) for row in rows_by_role["behavior_bundle"]
        ]
        expected_selection_source: Dict[str, Any] = {
            "mode": "full_pool",
            "source_role": "behavior_bundle",
            "record_count": len(selected_ids),
            "ordered_base_fact_ids_sha256": sha256_value(selected_ids),
        }
    else:
        scope_path_value = _required_string(
            selection_source.get("path"), "selection_source.path"
        )
        scope_path = Path(scope_path_value)
        if not scope_path.is_absolute():
            scope_path = (
                Path(manifest_path).resolve().parent / scope_path
            ).resolve()
        scope, scope_provenance = validate_review_scope(
            scope_path,
            indexes,
            actual_source_bindings,
        )
        selected_ids = [str(value) for value in scope["base_fact_ids"]]
        expected_selection_source = {
            "mode": "review_scope",
            "path": scope_provenance["path"],
            "sha256": scope_provenance["sha256"],
            "schema_version": REVIEW_SCOPE_SCHEMA_VERSION,
            "scope_id": scope["scope_id"],
            "scope_selection_mode": scope["selection_source"]["mode"],
            "record_count": len(selected_ids),
            "ordered_base_fact_ids_sha256": sha256_value(selected_ids),
        }
    if not _json_equal(selection_source, expected_selection_source):
        raise ValueError("Formal cohort selection_source is stale or inconsistent")
    if source_ids != selected_ids:
        raise ValueError("Formal cohort items do not match the declared selection")

    expected_universe_id = "formal_cohort_" + sha256_value(
        {
            "tool_version": TOOL_VERSION,
            "universe_label": universe_label,
            "selection_policy_id": policy_id,
            "selection_source": expected_selection_source,
            "source_artifacts": actual_source_bindings,
            "comparison_canonical": actual_comparison_binding,
            "near_duplicate_audit_contract": expected_near_audit_contract,
            "historical_exposure_contract_status": (
                HISTORICAL_EXPOSURE_CONTRACT_STATUS
            ),
            "ordered_base_fact_ids": selected_ids,
        }
    )[:24]
    if universe_id != expected_universe_id:
        raise ValueError("Formal cohort universe_id does not match its declaration")

    for index, (base_fact_id, row) in enumerate(zip(selected_ids, items)):
        expected_item = _universe_item(
            universe_id=expected_universe_id,
            selection_index=index,
            base_fact_id=base_fact_id,
            indexes=indexes,
            source_bundle_sha256=actual_source_bindings["behavior_bundle"]["sha256"],
        )
        if not _json_equal(row, expected_item):
            raise ValueError(
                f"Formal cohort item {index} does not match its source artifacts"
            )
    provenance.update(
        {
            "universe_id": universe_id,
            "record_count": len(items),
            "comparison_canonical": actual_comparison_binding,
            "near_duplicate_audit_contract": expected_near_audit_contract,
            "historical_exposure_contract_status": (
                HISTORICAL_EXPOSURE_CONTRACT_STATUS
            ),
        }
    )
    return manifest, items, provenance


def _resolve_binding_path(binding: Mapping[str, Any], owner_path: Path, label: str) -> Path:
    value = _required_string(binding.get("path"), f"{label}.path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (owner_path.parent / path).resolve()


def _verify_file_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    jsonl: bool,
    allow_empty: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = _resolve_binding_path(binding, owner_path, label)
    if jsonl:
        value, snapshot = read_jsonl_snapshot(path, allow_empty=allow_empty)
        if binding.get("record_count") != snapshot["record_count"]:
            raise ValueError(f"{label}.record_count is stale")
    else:
        value, snapshot = read_json_snapshot(path)
    if binding.get("sha256") != snapshot["sha256"]:
        raise ValueError(f"{label}.sha256 is stale")
    return value, {**snapshot, "path": str(path)}


def validate_review_apply_manifest(
    path: Path,
    *,
    universe: Mapping[str, Any],
    universe_source_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    path = Path(path).resolve()
    manifest, provenance = read_json_snapshot(path)
    manifest_fields = set(manifest)
    if manifest_fields not in {
        frozenset(REVIEW_APPLY_FIELDS),
        frozenset(REVIEW_APPLY_COMPOSITE_FIELDS),
    }:
        raise ValueError("Review apply manifest fields are invalid")
    if manifest.get("schema_version") != REVIEW_APPLY_SCHEMA_VERSION:
        raise ValueError("Unsupported review apply manifest schema_version")
    if manifest.get("tool_version") != REVIEW_TOOL_VERSION:
        raise ValueError("Unsupported review apply tool_version")
    _required_string(manifest.get("scope_id"), "review apply scope_id")
    if type(manifest.get("allow_partial")) is not bool:
        raise ValueError("Review apply allow_partial must be boolean")
    if not _json_equal(
        manifest.get("safety_contract"), REVIEW_APPLY_SAFETY_CONTRACT
    ):
        raise ValueError("Review apply safety_contract is invalid")
    scoped_count = _require_int(
        manifest.get("scoped_base_fact_count"),
        "review apply scoped_base_fact_count",
        minimum=1,
    )
    outcome_counts = manifest.get("outcome_counts")
    if not isinstance(outcome_counts, dict) or not outcome_counts:
        raise ValueError("Review apply outcome_counts must be a non-empty object")
    allowed_outcomes = {"accept", "reject", "defer", "revise", "missing"}
    if not set(outcome_counts).issubset(allowed_outcomes):
        raise ValueError("Review apply outcome_counts has unsupported outcomes")
    for outcome, count in outcome_counts.items():
        _require_int(count, f"review apply outcome_counts.{outcome}")
    if sum(outcome_counts.values()) != scoped_count:
        raise ValueError("Review apply outcome_counts do not sum to scope size")

    scope_binding = manifest.get("scope_manifest")
    if not isinstance(scope_binding, dict) or set(scope_binding) != {"path", "sha256"}:
        raise ValueError("Review apply scope_manifest binding is invalid")
    scope, scope_snapshot = _verify_file_binding(
        scope_binding,
        owner_path=path,
        label="review apply scope_manifest",
        jsonl=False,
    )
    if scope.get("schema_version") != REVIEW_SCOPE_SCHEMA_VERSION:
        raise ValueError("Review apply references unsupported scope schema")
    if scope.get("scope_id") != manifest.get("scope_id"):
        raise ValueError("Review apply scope_id mismatch")
    source_bindings = universe["source_artifacts"]
    declared_scope_sources = scope.get("source_artifacts")
    if not isinstance(declared_scope_sources, dict):
        raise ValueError("Review apply scope has no source bindings")
    for role in SOURCE_ARTIFACTS:
        if not _json_equal(
            declared_scope_sources.get(role),
            _scope_compatible_binding(source_bindings[role]),
        ):
            raise ValueError(f"Review apply scope source binding mismatch for {role}")
    scoped_ids = scope.get("base_fact_ids")
    if (
        not isinstance(scoped_ids, list)
        or not scoped_ids
        or any(not isinstance(value, str) or not value.strip() for value in scoped_ids)
        or len(scoped_ids) != len(set(scoped_ids))
    ):
        raise ValueError("Review apply scope IDs are invalid")
    if scoped_count != len(scoped_ids):
        raise ValueError("Review apply scoped_base_fact_count mismatch")
    outside = set(scoped_ids) - set(universe_source_ids)
    if outside:
        raise ValueError("Review apply scope contains IDs outside the formal cohort")

    export_binding = manifest.get("export_manifest")
    if not isinstance(export_binding, dict) or set(export_binding) != {"path", "sha256"}:
        raise ValueError("Review apply export_manifest binding is invalid")
    _, export_snapshot = _verify_file_binding(
        export_binding,
        owner_path=path,
        label="review apply export_manifest",
        jsonl=False,
    )
    decisions_binding = manifest.get("decisions")
    if not isinstance(decisions_binding, dict) or set(decisions_binding) != {
        "path",
        "sha256",
        "record_count",
    }:
        raise ValueError("Review apply decisions binding is invalid")
    decisions, decision_snapshot = _verify_file_binding(
        decisions_binding,
        owner_path=path,
        label="review apply decisions",
        jsonl=True,
        allow_empty=True,
    )
    decision_authority_snapshot: Optional[Dict[str, Any]] = None
    composite_authority_path: Optional[Path] = None
    if "decision_authority" in manifest:
        decision_authority = manifest.get("decision_authority")
        if not isinstance(decision_authority, dict):
            raise ValueError("Review apply decision_authority must be an object")
        if decision_authority.get("mode") != COMPOSITE_REVIEW_AUTHORITY_MODE:
            raise ValueError("Review apply decision_authority mode is invalid")
        authority_id = _required_string(
            decision_authority.get("authority_id"),
            "review apply decision_authority.authority_id",
        )
        authority_binding = decision_authority.get("manifest")
        if not isinstance(authority_binding, dict) or set(authority_binding) != {
            "path",
            "sha256",
            "schema_version",
        }:
            raise ValueError("Review apply decision_authority manifest binding is invalid")
        if (
            authority_binding.get("schema_version")
            != COMPOSITE_REVIEW_AUTHORITY_SCHEMA_VERSION
        ):
            raise ValueError("Review apply decision_authority schema is unsupported")
        authority_manifest, decision_authority_snapshot = _verify_file_binding(
            authority_binding,
            owner_path=path,
            label="review apply decision_authority manifest",
            jsonl=False,
        )
        if (
            authority_manifest.get("schema_version")
            != COMPOSITE_REVIEW_AUTHORITY_SCHEMA_VERSION
            or authority_manifest.get("authority_id") != authority_id
        ):
            raise ValueError("Review apply decision_authority binding is inconsistent")
        composite_authority_path = Path(decision_authority_snapshot["path"])
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"review_staging"}:
        raise ValueError("Review apply artifacts are missing")
    staging_binding = artifacts.get("review_staging")
    if not isinstance(staging_binding, dict) or set(staging_binding) != {
        "path",
        "sha256",
        "record_count",
        "schema_version",
    }:
        raise ValueError("Review apply review_staging binding is invalid")
    staging, staging_snapshot = _verify_file_binding(
        staging_binding,
        owner_path=path,
        label="review apply review_staging",
        jsonl=True,
    )
    if staging_binding.get("schema_version") != REVIEW_STAGING_SCHEMA_VERSION:
        raise ValueError("Review staging schema binding is invalid")

    review_tool = _load_sibling_module(
        "review_public_benchmark_bundle.py",
        "_pitfalls_review_public_benchmark_bundle_replay",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="review-apply-replay-") as directory:
            replay_kwargs: Dict[str, Any] = {
                "scope_manifest_path": Path(scope_snapshot["path"]),
                "export_manifest_path": Path(export_snapshot["path"]),
                "decisions_path": (
                    None
                    if composite_authority_path is not None
                    else Path(decision_snapshot["path"])
                ),
                "output_dir": Path(directory),
                "allow_partial": manifest["allow_partial"],
            }
            if composite_authority_path is not None:
                replay_kwargs["composite_authority_manifest_path"] = (
                    composite_authority_path
                )
            replay_result = review_tool.apply_decisions(**replay_kwargs)
            replay_manifest, _ = read_json_snapshot(
                Path(replay_result["manifest_path"])
            )
            replay_staging, replay_staging_snapshot = read_jsonl_snapshot(
                Path(replay_result["staging_path"])
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"Review apply deterministic replay failed: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Review apply deterministic replay raised {type(exc).__name__}: {exc}"
        ) from exc

    if len(staging) != len(replay_staging) or any(
        not _json_equal(actual, expected)
        for actual, expected in zip(staging, replay_staging)
    ):
        raise ValueError("Review staging differs from deterministic apply replay")
    normalized_manifest = json.loads(json.dumps(manifest))
    normalized_replay = json.loads(json.dumps(replay_manifest))
    normalized_manifest["artifacts"]["review_staging"]["path"] = "<staging>"
    normalized_replay["artifacts"]["review_staging"]["path"] = "<staging>"
    if not _json_equal(normalized_manifest, normalized_replay):
        raise ValueError("Review apply manifest differs from deterministic replay")

    staging_index = _unique_index(staging, "base_fact_id", "review staging")
    if set(staging_index) != set(scoped_ids):
        raise ValueError("Review staging does not exactly cover its scope")
    provenance.update(
        {
            "scope_manifest": scope_snapshot,
            "export_manifest": export_snapshot,
            "decisions": decision_snapshot,
            "review_staging": staging_snapshot,
            "deterministic_replay": {
                "performed": True,
                "staging_sha256": replay_staging_snapshot["sha256"],
                "record_count": len(replay_staging),
            },
            "record_count": len(staging),
        }
    )
    if decision_authority_snapshot is not None:
        provenance["decision_authority"] = decision_authority_snapshot
    return {str(key): dict(value) for key, value in staging_index.items()}, provenance


def validate_near_audit(
    path: Path,
    *,
    universe: Mapping[str, Any],
    universe_source_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
    path = Path(path).resolve()
    summary, provenance = read_json_snapshot(path)
    expected_contract = universe.get("near_duplicate_audit_contract")
    if not isinstance(expected_contract, dict) or set(expected_contract) != (
        NEAR_AUDIT_CONTRACT_FIELDS
    ):
        raise ValueError("Formal cohort has no valid near-duplicate audit contract")
    expected_policy = expected_contract.get("policy")
    if not isinstance(expected_policy, dict) or set(expected_policy) != (
        NEAR_AUDIT_POLICY_FIELDS
    ):
        raise ValueError("Formal cohort has no valid near-duplicate policy")
    if summary.get("schema_version") != expected_contract.get(
        "summary_schema_version"
    ):
        raise ValueError("Unsupported near-duplicate audit schema_version")
    inputs = summary.get("input")
    if not isinstance(inputs, dict) or set(inputs) != {
        "behavior_bundle",
        "comparison_canonical",
    }:
        raise ValueError("Near-duplicate audit input bindings are invalid")
    behavior = inputs.get("behavior_bundle")
    if not isinstance(behavior, dict):
        raise ValueError("Near-duplicate audit has no behavior bundle binding")
    expected_behavior = universe["source_artifacts"]["behavior_bundle"]
    for field in ("path", "sha256", "record_count"):
        if behavior.get(field) != expected_behavior.get(field):
            raise ValueError(f"Near-duplicate audit behavior bundle {field} mismatch")
    comparison = inputs.get("comparison_canonical")
    if not isinstance(comparison, dict) or set(comparison) != {
        "path",
        "sha256",
        "record_count",
    }:
        raise ValueError(
            "Near-duplicate audit must bind a comparison_canonical artifact"
        )
    expected_comparison = universe.get("comparison_canonical")
    if not isinstance(expected_comparison, dict):
        raise ValueError("Formal cohort has no comparison_canonical binding")
    for field in ("path", "sha256", "record_count"):
        if not _json_equal(comparison.get(field), expected_comparison.get(field)):
            raise ValueError(
                f"Near-duplicate audit comparison_canonical {field} differs from universe"
            )
    comparison_path = _resolve_binding_path(
        comparison, path, "near-duplicate comparison_canonical"
    )
    _, comparison_snapshot = read_jsonl_snapshot(comparison_path)
    if comparison.get("sha256") != comparison_snapshot["sha256"]:
        raise ValueError("Near-duplicate audit comparison_canonical SHA is stale")
    if comparison.get("record_count") != comparison_snapshot["record_count"]:
        raise ValueError("Near-duplicate audit comparison_canonical count is stale")
    if comparison_snapshot["byte_count"] != expected_comparison.get("byte_count"):
        raise ValueError(
            "Near-duplicate audit comparison_canonical bytes differ from universe"
        )

    policy = summary.get("policy")
    if not isinstance(policy, dict) or not _json_equal(policy, expected_policy):
        raise ValueError(
            "Near-duplicate audit policy differs from the predeclared universe contract"
        )
    question_threshold = policy.get("question_near_threshold")
    fact_threshold = policy.get("canonical_fact_near_threshold")
    if type(question_threshold) is not float or not 0.0 < question_threshold <= 1.0:
        raise ValueError("Near-duplicate question threshold must be a float in (0, 1]")
    if type(fact_threshold) is not float or not 0.0 < fact_threshold <= 1.0:
        raise ValueError("Near-duplicate fact threshold must be a float in (0, 1]")
    max_bucket_neighbors = _require_int(
        policy.get("max_bucket_neighbors"),
        "near-duplicate policy.max_bucket_neighbors",
        minimum=1,
    )
    max_examples = _require_int(
        policy.get("max_examples"),
        "near-duplicate policy.max_examples",
        minimum=1,
    )
    summary_artifacts = summary.get("artifacts")
    if not isinstance(summary_artifacts, dict) or set(summary_artifacts) != {
        "candidate_pairs"
    }:
        raise ValueError("Near-duplicate audit artifacts are invalid")
    candidate_binding = summary_artifacts.get("candidate_pairs")
    candidates, candidate_snapshot = _verify_file_binding(
        candidate_binding,
        owner_path=path,
        label="near-duplicate candidate_pairs",
        jsonl=True,
        allow_empty=True,
    )
    if candidate_binding.get("schema_version") != expected_contract.get(
        "pair_schema_version"
    ):
        raise ValueError("Near-duplicate candidate schema binding is invalid")

    audit_tool = _load_sibling_module(
        "audit_public_benchmark_near_duplicates.py",
        "_pitfalls_audit_public_benchmark_near_duplicates_replay",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="near-audit-replay-") as directory:
            replay_summary = audit_tool.audit(
                Path(expected_behavior["path"]),
                Path(directory),
                comparison_canonical_path=comparison_path,
                question_threshold=question_threshold,
                fact_threshold=fact_threshold,
                max_bucket_neighbors=max_bucket_neighbors,
                max_examples=max_examples,
            )
            replay_summary["policy"]["max_examples"] = max_examples
            replay_candidates_path = (
                Path(directory) / "lexical_candidate_pairs.jsonl"
            )
            replay_candidate_bytes = replay_candidates_path.read_bytes()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"Near-duplicate deterministic replay failed: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            f"Near-duplicate deterministic replay raised {type(exc).__name__}: {exc}"
        ) from exc
    if Path(candidate_snapshot["path"]).read_bytes() != replay_candidate_bytes:
        raise ValueError("Near-duplicate candidate bytes differ from deterministic replay")
    if not _json_equal(summary, replay_summary):
        raise ValueError("Near-duplicate summary differs from deterministic replay")

    pair_index = _unique_index(candidates, "pair_id", "near-duplicate candidates")
    universe_ids = set(universe_source_ids)
    touching = 0
    cross_split_touching = 0
    external_pool_ids = set()
    verified_input_files: Dict[
        str, Tuple[str, Dict[int, Dict[str, Any]]]
    ] = {}
    for pair_id, row in pair_index.items():
        if row.get("schema_version") != expected_contract.get(
            "pair_schema_version"
        ):
            raise ValueError(f"Near-duplicate pair has invalid schema: {pair_id}")
        endpoint_ids = []
        for side in ("left", "right"):
            endpoint = row.get(side)
            endpoint_provenance = (
                endpoint.get("provenance") if isinstance(endpoint, dict) else None
            )
            if not isinstance(endpoint_provenance, dict):
                raise ValueError(f"Near-duplicate pair has invalid {side}: {pair_id}")
            audit_record_id = _required_string(
                endpoint_provenance.get("audit_record_id"),
                f"near pair {pair_id}.{side}.audit_record_id",
            )
            input_role = _required_string(
                endpoint_provenance.get("input_role"),
                f"near pair {pair_id}.{side}.input_role",
            )
            input_line_number = endpoint_provenance.get("input_line_number")
            if type(input_line_number) is not int or input_line_number < 1:
                raise ValueError(
                    f"Near-duplicate pair has invalid input line: {pair_id}"
                )
            input_record_sha256 = endpoint_provenance.get("input_record_sha256")
            if not _is_sha256(input_record_sha256):
                raise ValueError(f"Near-duplicate pair has invalid endpoint hash: {pair_id}")
            input_path = Path(
                _required_string(
                    endpoint_provenance.get("input_path"),
                    f"near pair {pair_id}.{side}.input_path",
                )
            ).resolve()
            declared_input_sha = endpoint_provenance.get("input_sha256")
            cached_input = verified_input_files.get(str(input_path))
            if cached_input is None:
                input_rows, input_snapshot = read_jsonl_line_snapshot(input_path)
                cached_input = (input_snapshot["sha256"], input_rows)
                verified_input_files[str(input_path)] = cached_input
            cached_sha, input_rows = cached_input
            if cached_sha != declared_input_sha:
                raise ValueError(f"Near-duplicate pair has stale input file: {pair_id}")
            input_row = input_rows.get(input_line_number)
            if input_row is None:
                raise ValueError(
                    f"Near-duplicate pair input line does not exist: {pair_id}"
                )
            actual_record_sha256 = sha256_value(input_row)
            if input_record_sha256 != actual_record_sha256:
                raise ValueError(
                    f"Near-duplicate pair endpoint row hash mismatch: {pair_id}"
                )
            for field in ("base_fact_id", "candidate_id", "source_id"):
                if endpoint_provenance.get(field) != input_row.get(field):
                    raise ValueError(
                        f"Near-duplicate pair endpoint {field} mismatch: {pair_id}"
                    )
            expected_audit_record_id = "audit_rec_" + sha256_value(
                {
                    "role": input_role,
                    "base_fact_id": input_row.get("base_fact_id"),
                    "candidate_id": input_row.get("candidate_id"),
                    "source_id": input_row.get("source_id"),
                    "record_sha256": actual_record_sha256,
                    "line_number": input_line_number,
                }
            )[:24]
            if audit_record_id != expected_audit_record_id:
                raise ValueError(
                    f"Near-duplicate pair audit_record_id mismatch: {pair_id}"
                )
            endpoint_ids.append(str(endpoint_provenance.get("base_fact_id") or ""))
        if any(value in universe_ids for value in endpoint_ids):
            touching += 1
            cross_split_touching += int(row.get("cross_split") is True)
            for value in endpoint_ids:
                if value and value not in universe_ids:
                    external_pool_ids.add(value)
    candidate_generation = summary.get("candidate_generation")
    if not isinstance(candidate_generation, dict):
        raise ValueError("Near-duplicate audit candidate_generation is invalid")
    summary_count = candidate_generation.get("candidate_pair_count")
    if type(summary_count) is not int or summary_count != len(candidates):
        raise ValueError("Near-duplicate audit candidate_pair_count mismatch")
    provenance.update(
        {
            "candidate_pairs": candidate_snapshot,
            "comparison_canonical": comparison_snapshot,
            "deterministic_replay": {
                "performed": True,
                "candidate_bytes_sha256": hashlib.sha256(
                    replay_candidate_bytes
                ).hexdigest(),
            },
        }
    )
    return candidates, provenance, {
        "candidate_pair_count": len(candidates),
        "pairs_touching_universe": touching,
        "cross_split_pairs_touching_universe": cross_split_touching,
        "out_of_universe_base_fact_count": len(external_pool_ids),
    }


def validate_near_adjudications(
    path: Path, candidates: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    rows, snapshot = read_jsonl_snapshot(path, allow_empty=True)
    decisions = _unique_index(rows, "pair_id", "near-duplicate adjudications")
    candidate_index = _unique_index(candidates, "pair_id", "near-duplicate candidates")
    if set(decisions) != set(candidate_index):
        raise ValueError("Near-duplicate adjudications do not exactly cover candidates")
    for pair_id, row in decisions.items():
        if row.get("schema_version") != NEAR_ADJUDICATION_SCHEMA_VERSION:
            raise ValueError(f"Near adjudication has invalid schema: {pair_id}")
        candidate = candidate_index[pair_id]
        if row.get("candidate_row_sha256") != sha256_value(candidate):
            raise ValueError(f"Near adjudication has stale candidate hash: {pair_id}")
        for side in ("left", "right"):
            endpoint = candidate[side]["provenance"]
            if row.get(f"{side}_audit_record_id") != endpoint.get("audit_record_id"):
                raise ValueError(f"Near adjudication has wrong {side} endpoint: {pair_id}")
            if row.get(f"{side}_input_record_sha256") != endpoint.get(
                "input_record_sha256"
            ):
                raise ValueError(f"Near adjudication has stale {side} endpoint hash: {pair_id}")
        decision = row.get("decision")
        if not isinstance(decision, str) or decision not in {
            "same_fact",
            "same_leakage_component",
            "distinct",
            "exclude_cohort",
        }:
            raise ValueError(f"Near adjudication is unresolved: {pair_id}")
        match_types = candidate.get("match_types")
        if (
            not isinstance(match_types, list)
            or any(not isinstance(value, str) for value in match_types)
        ):
            raise ValueError(f"Near-duplicate candidate match_types are invalid: {pair_id}")
        if decision == "distinct" and EXACT_DUPLICATE_MATCH_TYPES.intersection(
            match_types
        ):
            raise ValueError(
                f"Exact duplicate candidate cannot be adjudicated distinct: {pair_id}"
            )
        reviewer_type = row.get("reviewer_type")
        if not isinstance(reviewer_type, str) or reviewer_type not in {
            "human",
            "codex_proxy",
        }:
            raise ValueError(f"Near adjudication has invalid reviewer_type: {pair_id}")
        if type(row.get("human_gold")) is not bool:
            raise ValueError(f"Near adjudication human_gold must be boolean: {pair_id}")
        if reviewer_type == "codex_proxy" and row.get("human_gold") is not False:
            raise ValueError(f"Near adjudication codex_proxy cannot be human gold: {pair_id}")
    return {**snapshot, "schema_version": NEAR_ADJUDICATION_SCHEMA_VERSION}


def _validate_completion_manifest(
    path: Path,
    *,
    expected_schema: str,
    universe_id: str,
    status_field: str = "status",
    expected_status: str = "complete",
) -> Dict[str, Any]:
    value, snapshot = read_json_snapshot(path)
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"Unsupported completion artifact schema: {path}")
    if value.get("universe_id") != universe_id:
        raise ValueError(f"Completion artifact universe_id mismatch: {path}")
    if value.get(status_field) != expected_status:
        raise ValueError(f"Completion artifact is not complete: {path}")
    if type(value.get("unresolved_count")) is not int or value.get("unresolved_count") != 0:
        raise ValueError(f"Completion artifact has unresolved records: {path}")
    return snapshot


def _validate_revision_evidence(
    lineage_path: Path,
    rereview_path: Path,
    *,
    universe_id: str,
    revised_source_ids: Sequence[str],
    item_by_source_id: Mapping[str, Mapping[str, Any]],
    source_behavior_by_id: Mapping[str, Mapping[str, Any]],
    source_review_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    lineage, lineage_snapshot = read_jsonl_snapshot(lineage_path)
    rereviews, rereview_snapshot = read_jsonl_snapshot(rereview_path)
    lineage_index = _unique_index(lineage, "source_base_fact_id", "revision lineage")
    rereview_index = _unique_index(rereviews, "source_base_fact_id", "revision rereview")
    revised = set(revised_source_ids)
    if set(lineage_index) != revised or set(rereview_index) != revised:
        raise ValueError("Revision lineage and rereview must exactly cover revise outcomes")
    rejected_ids: List[str] = []
    for base_fact_id in sorted(revised):
        lineage_row = lineage_index[base_fact_id]
        rereview = rereview_index[base_fact_id]
        item = item_by_source_id[base_fact_id]
        original_behavior = source_behavior_by_id.get(base_fact_id)
        if not isinstance(original_behavior, dict):
            raise ValueError(f"Revision has no bound source behavior row: {base_fact_id}")
        source_review = source_review_by_id.get(base_fact_id)
        if not isinstance(source_review, dict) or source_review.get(
            "review_outcome"
        ) != "revise":
            raise ValueError(
                f"Revision has no bound revise staging row: {base_fact_id}"
            )
        if set(lineage_row) != REVISION_LINEAGE_FIELDS:
            raise ValueError(f"Revision lineage fields are invalid: {base_fact_id}")
        if set(rereview) != REREVIEW_DECISION_FIELDS:
            raise ValueError(f"Revision rereview fields are invalid: {base_fact_id}")
        if lineage_row.get("schema_version") != REVISION_LINEAGE_SCHEMA_VERSION:
            raise ValueError(f"Revision lineage schema mismatch: {base_fact_id}")
        if rereview.get("schema_version") != REREVIEW_DECISION_SCHEMA_VERSION:
            raise ValueError(f"Revision rereview schema mismatch: {base_fact_id}")
        if lineage_row.get("universe_id") != universe_id:
            raise ValueError(f"Revision lineage universe mismatch: {base_fact_id}")
        if lineage_row.get("cohort_item_id") != item.get("cohort_item_id"):
            raise ValueError(f"Revision lineage cohort item mismatch: {base_fact_id}")
        if lineage_row.get("revision_status") != "semantic_patch_staged_for_rereview":
            raise ValueError(f"Revision semantic patch is not staged: {base_fact_id}")
        if lineage_row.get("revised_record_role") != (
            "preflight_reconstruction_not_final_behavior_row"
        ):
            raise ValueError(f"Revision revised_record role is invalid: {base_fact_id}")
        if lineage_row.get("derived_fields_recomputed") is not False:
            raise ValueError(
                f"Revision must declare derived fields not recomputed: {base_fact_id}"
            )
        source_review_hash = lineage_row.get("source_review_staging_row_sha256")
        if not _is_sha256(source_review_hash) or source_review_hash != sha256_value(
            source_review
        ):
            raise ValueError(
                f"Revision source review staging hash is invalid: {base_fact_id}"
            )
        editor_type = lineage_row.get("editor_type")
        if not isinstance(editor_type, str) or editor_type not in {
            "human",
            "codex_proxy",
        }:
            raise ValueError(f"Revision editor_type is invalid: {base_fact_id}")
        for field in ("editor_id", "edit_method", "edited_at"):
            _required_string(
                lineage_row.get(field), f"revision lineage {base_fact_id}.{field}"
            )
        before_hash = lineage_row.get("before_record_sha256")
        after_hash = lineage_row.get("after_record_sha256")
        if not _is_sha256(before_hash) or not _is_sha256(after_hash) or before_hash == after_hash:
            raise ValueError(f"Revision hash chain is invalid: {base_fact_id}")
        expected_before_hash = sha256_value(original_behavior)
        if (
            before_hash != expected_before_hash
            or before_hash
            != (item.get("source_row_bindings") or {}).get("behavior_row_sha256")
        ):
            raise ValueError(
                f"Revision before hash is not bound to the universe item: {base_fact_id}"
            )
        revised_record = lineage_row.get("revised_record")
        if not isinstance(revised_record, dict) or not revised_record:
            raise ValueError(f"Revision has no revised_record object: {base_fact_id}")
        revision_payload = lineage_row.get("revision_payload")
        if not isinstance(revision_payload, dict) or set(revision_payload) != (
            REVISION_PAYLOAD_FIELDS
        ):
            raise ValueError(f"Revision semantic payload fields are invalid: {base_fact_id}")
        if revision_payload.get("base_fact_id") != base_fact_id:
            raise ValueError(f"Revision semantic payload has wrong base_fact_id: {base_fact_id}")
        for field in (
            "subject_en",
            "relation_raw",
            "answer_en",
            "answer_type",
            "canonical_fact",
            "canonical_fact_en",
        ):
            _required_string(
                revision_payload.get(field),
                f"revision payload {base_fact_id}.{field}",
            )
        aliases = revision_payload.get("answer_aliases_en")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(
                f"Revision payload answer_aliases_en must be a non-empty list: {base_fact_id}"
            )
        normalized_aliases: List[str] = []
        for index, alias in enumerate(aliases):
            normalized_aliases.append(
                _normalized_semantic_text(
                    _required_string(
                        alias,
                        f"revision payload {base_fact_id}.answer_aliases_en[{index}]",
                    )
                )
            )
        if len(normalized_aliases) != len(set(normalized_aliases)):
            raise ValueError(f"Revision payload aliases are not unique: {base_fact_id}")
        if _normalized_semantic_text(revision_payload["answer_en"]) not in set(
            normalized_aliases
        ):
            raise ValueError(
                f"Revision payload aliases do not contain answer_en: {base_fact_id}"
            )
        if revision_payload["canonical_fact"] != revision_payload["canonical_fact_en"]:
            raise ValueError(
                f"Revision payload canonical facts differ: {base_fact_id}"
            )
        reconstructed_record = dict(original_behavior)
        for field in REVISION_MUTABLE_FIELDS:
            reconstructed_record[field] = revision_payload[field]
        if not _json_equal(revised_record, reconstructed_record):
            raise ValueError(
                f"Revision revised_record is not the source plus semantic payload: {base_fact_id}"
            )
        if set(revised_record) != set(original_behavior):
            raise ValueError(
                f"Revision revised_record fields differ from source schema: {base_fact_id}"
            )
        required_status = {
            "schema_version": original_behavior.get("schema_version"),
            "base_fact_id": base_fact_id,
            "canonical_status": "pending_review",
            "bundle_status": "draft_pending_review",
            "human_gold": False,
            "evidence_tier": "provisional_single_model",
            "split_status": "provisional_not_frozen",
        }
        for field, expected in required_status.items():
            if not _json_equal(revised_record.get(field), expected):
                raise ValueError(
                    f"Revision revised_record has invalid {field}: {base_fact_id}"
                )
        changed_fields = lineage_row.get("changed_fields")
        if (
            not isinstance(changed_fields, list)
            or any(not isinstance(field, str) or not field for field in changed_fields)
            or len(changed_fields) != len(set(changed_fields))
            or changed_fields != sorted(changed_fields)
        ):
            raise ValueError(f"Revision changed_fields are invalid: {base_fact_id}")
        actual_changed_fields = sorted(
            field
            for field in REVISION_MUTABLE_FIELDS
            if not _json_equal(original_behavior.get(field), revised_record.get(field))
        )
        if changed_fields != actual_changed_fields:
            raise ValueError(
                f"Revision changed_fields do not match revised_record: {base_fact_id}"
            )
        unsupported_changes = sorted(set(changed_fields) - REVISION_MUTABLE_FIELDS)
        if unsupported_changes:
            raise ValueError(
                f"Revision changes immutable fields {unsupported_changes}: {base_fact_id}"
            )
        if sha256_value(revised_record) != after_hash:
            raise ValueError(f"Revision after hash does not bind revised_record: {base_fact_id}")
        if rereview.get("universe_id") != universe_id:
            raise ValueError(f"Revision rereview universe mismatch: {base_fact_id}")
        if rereview.get("cohort_item_id") != item.get("cohort_item_id"):
            raise ValueError(f"Revision rereview cohort item mismatch: {base_fact_id}")
        if rereview.get("revised_record_sha256") != after_hash:
            raise ValueError(f"Revision rereview has stale revised row hash: {base_fact_id}")
        decision = rereview.get("decision")
        if not isinstance(decision, str) or decision not in {"accept", "reject"}:
            raise ValueError(f"Revision rereview is not terminal: {base_fact_id}")
        reviewer_type = rereview.get("reviewer_type")
        if not isinstance(reviewer_type, str) or reviewer_type not in {
            "human",
            "codex_proxy",
        }:
            raise ValueError(f"Revision rereview reviewer_type is invalid: {base_fact_id}")
        for field in ("reviewer_id", "review_method", "reviewed_at"):
            _required_string(
                rereview.get(field), f"revision rereview {base_fact_id}.{field}"
            )
        if type(rereview.get("human_gold")) is not bool:
            raise ValueError(
                f"Revision rereview human_gold must be boolean: {base_fact_id}"
            )
        if reviewer_type == "codex_proxy" and rereview.get("human_gold") is not False:
            raise ValueError(f"Revision rereview codex_proxy cannot be human gold: {base_fact_id}")
        if rereview.get("human_gold") is not revised_record.get("human_gold"):
            raise ValueError(
                f"Revision rereview human_gold differs from revised row: {base_fact_id}"
            )
        if decision == "reject":
            rejected_ids.append(base_fact_id)
    return lineage_snapshot, rereview_snapshot, rejected_ids


def _validate_historical_registry(
    path: Path, universe_id: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path).resolve()
    value, snapshot = read_json_snapshot(path)
    if set(value) != {
        "schema_version",
        "universe_id",
        "inventory_status",
        "source_count",
        "sources_sha256",
        "sources",
    }:
        raise ValueError("Historical exposure registry fields are invalid")
    if value.get("schema_version") != HISTORICAL_REGISTRY_SCHEMA_VERSION:
        raise ValueError("Unsupported historical exposure registry schema")
    if value.get("universe_id") != universe_id:
        raise ValueError("Historical exposure registry universe_id mismatch")
    if value.get("inventory_status") != "complete_attested":
        raise ValueError("Historical exposure registry is not complete and attested")
    sources = _mapping_list(
        value.get("sources"),
        "historical exposure registry sources",
        allow_empty=False,
    )
    if _require_int(
        value.get("source_count"),
        "historical exposure registry source_count",
        minimum=1,
    ) != len(sources):
        raise ValueError("Historical exposure registry source_count mismatch")
    if value.get("sources_sha256") != sha256_value(sources):
        raise ValueError("Historical exposure registry sources_sha256 mismatch")
    source_ids = set()
    for index, source in enumerate(sources):
        if set(source) != {"exposure_source_id", "path", "sha256", "byte_count"}:
            raise ValueError(f"Historical exposure source {index} fields are invalid")
        source_id = _required_string(
            source.get("exposure_source_id"),
            f"historical exposure source {index}.exposure_source_id",
        )
        if source_id in source_ids:
            raise ValueError(f"Duplicate historical exposure source: {source_id}")
        source_ids.add(source_id)
        source_path = _resolve_binding_path(
            source,
            path,
            f"historical exposure source {index}",
        )
        byte_count = _require_int(
            source.get("byte_count"),
            f"historical exposure source {index}.byte_count",
            minimum=1,
        )
        if source_path.stat().st_size != byte_count:
            raise ValueError(f"Historical exposure source has stale size: {source_id}")
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source.get("sha256"):
            raise ValueError(f"Historical exposure source has stale SHA: {source_id}")
    return value, {
        **snapshot,
        "source_count": len(sources),
        "sources_sha256": value["sources_sha256"],
    }


def _validate_historical_checks(
    path: Path,
    universe_id: str,
    item_by_cohort_id: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
    registry_sha256: str,
) -> Tuple[Dict[str, Any], List[str]]:
    rows, snapshot = read_jsonl_snapshot(path)
    checks = _unique_index(rows, "cohort_item_id", "historical exposure checks")
    expected_ids = set(item_by_cohort_id)
    if set(checks) != expected_ids:
        missing = sorted(expected_ids - set(checks))
        extra = sorted(set(checks) - expected_ids)
        raise ValueError(
            "Historical exposure checks do not exactly cover the formal cohort: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for index, row in enumerate(rows, start=1):
        cohort_item_id = str(row["cohort_item_id"])
        item = item_by_cohort_id[cohort_item_id]
        if set(row) != {
            "schema_version",
            "universe_id",
            "cohort_item_id",
            "source_base_fact_id",
            "source_row_bindings",
            "registry_sha256",
            "source_results",
            "status",
            "historically_exposed",
        }:
            raise ValueError(f"Historical exposure check {index} fields are invalid")
        if row.get("schema_version") != HISTORICAL_CHECK_SCHEMA_VERSION:
            raise ValueError(f"Historical exposure check {index} has invalid schema")
        if row.get("universe_id") != universe_id:
            raise ValueError(f"Historical exposure check {index} has wrong universe_id")
        if row.get("source_base_fact_id") != item.get("source_base_fact_id"):
            raise ValueError(
                f"Historical exposure check {index} has wrong source_base_fact_id"
            )
        if not _json_equal(
            row.get("source_row_bindings"), item.get("source_row_bindings")
        ):
            raise ValueError(
                f"Historical exposure check {index} has stale source row bindings"
            )
        if row.get("registry_sha256") != registry_sha256:
            raise ValueError(
                f"Historical exposure check {index} has stale registry SHA"
            )
        if row.get("status") != "resolved":
            raise ValueError(f"Historical exposure check {index} is unresolved")
        if type(row.get("historically_exposed")) is not bool:
            raise ValueError(f"Historical exposure check {index} lacks a boolean result")
        results = _mapping_list(
            row.get("source_results"),
            f"historical exposure check {index}.source_results",
            allow_empty=False,
        )
        result_index = _unique_index(
            results,
            "exposure_source_id",
            f"historical exposure check {index} source results",
        )
        registry_sources = registry["sources"]
        expected_source_ids = [
            str(source["exposure_source_id"]) for source in registry_sources
        ]
        if list(result_index) != expected_source_ids:
            raise ValueError(
                f"Historical exposure check {index} does not cover registry sources in order"
            )
        per_source_exposed = []
        for source in registry_sources:
            source_id = str(source["exposure_source_id"])
            result = result_index[source_id]
            if set(result) != {
                "exposure_source_id",
                "source_sha256",
                "status",
                "historically_exposed",
            }:
                raise ValueError(
                    f"Historical exposure source result fields are invalid: {source_id}"
                )
            if result.get("source_sha256") != source.get("sha256"):
                raise ValueError(
                    f"Historical exposure source result has stale SHA: {source_id}"
                )
            if result.get("status") != "resolved":
                raise ValueError(
                    f"Historical exposure source result is unresolved: {source_id}"
                )
            if type(result.get("historically_exposed")) is not bool:
                raise ValueError(
                    f"Historical exposure source result lacks boolean result: {source_id}"
                )
            per_source_exposed.append(result["historically_exposed"])
        if row["historically_exposed"] is not any(per_source_exposed):
            raise ValueError(
                f"Historical exposure aggregate result is inconsistent: {cohort_item_id}"
            )
    exposed_ids = [
        str(row["cohort_item_id"])
        for row in rows
        if row["historically_exposed"] is True
    ]
    return {
        **snapshot,
        "registry_sha256": registry_sha256,
        "exposed_count": len(exposed_ids),
    }, exposed_ids


def _blocker(
    code: str,
    *,
    count: Optional[int] = None,
    examples: Sequence[str] = (),
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"code": code}
    if count is not None:
        result["count"] = count
    if examples:
        result["examples"] = list(examples[:5])
    if detail:
        result["detail"] = detail
    return result


def build_preflight_report(
    *,
    universe_manifest_path: Path,
    output_path: Path,
    review_apply_manifest_paths: Sequence[Path] = (),
    revision_lineage_path: Optional[Path] = None,
    revision_rereview_path: Optional[Path] = None,
    near_audit_summary_path: Optional[Path] = None,
    near_adjudications_path: Optional[Path] = None,
    semantic_closure_attestation_path: Optional[Path] = None,
    historical_exposure_registry_path: Optional[Path] = None,
    historical_exposure_checks_path: Optional[Path] = None,
    recluster_manifest_path: Optional[Path] = None,
    split_recompute_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a diagnostic report and never emit formal/freeze artifacts."""

    output_path = Path(output_path).resolve()
    if output_path.name.casefold() in FREEZE_OUTPUT_NAMES_CASEFOLD:
        raise ValueError("Preflight output path cannot use a formal freeze artifact name")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight output: {output_path}")
    universe, items, universe_provenance = load_cohort_universe(
        universe_manifest_path
    )
    universe_id = str(universe["universe_id"])
    source_ids = [str(item["source_base_fact_id"]) for item in items]
    item_by_source_id = {
        str(item["source_base_fact_id"]): item for item in items
    }
    item_by_cohort_id = {
        str(item["cohort_item_id"]): item for item in items
    }
    behavior_path = _resolve_binding_path(
        universe["source_artifacts"]["behavior_bundle"],
        Path(universe_manifest_path).resolve(),
        "formal cohort behavior_bundle",
    )
    behavior_rows, _ = read_jsonl_snapshot(behavior_path)
    source_behavior_by_id = _unique_index(
        behavior_rows,
        "base_fact_id",
        "formal cohort behavior bundle",
    )
    blockers: List[Dict[str, Any]] = [
        _blocker(
            "reviewed_bundle_finalizer_not_implemented",
            detail=(
                "This command validates prerequisite presence and consistency only; "
                "it cannot authorize or emit a reviewed bundle or split freeze."
            ),
        ),
        _blocker(
            "historical_exposure_authority_missing",
            detail=(
                "The universe was declared with "
                "historical_exposure_contract_status=not_predeclared; a runtime "
                "registry cannot retroactively become the authoritative inventory."
            ),
        ),
    ]
    evidence: Dict[str, Any] = {
        "review_apply_manifests": [],
        "revision_lineage": None,
        "revision_rereview": None,
        "near_duplicate_audit": None,
        "near_duplicate_adjudications": None,
        "semantic_closure_attestation": None,
        "historical_exposure_registry": None,
        "historical_exposure_checks": None,
        "recluster_manifest": None,
        "split_recompute_manifest": None,
    }

    review_by_id: Dict[str, Dict[str, Any]] = {}
    for apply_path in review_apply_manifest_paths:
        try:
            staging, provenance = validate_review_apply_manifest(
                apply_path,
                universe=universe,
                universe_source_ids=source_ids,
            )
            duplicates = sorted(set(review_by_id) & set(staging))
            if duplicates:
                raise ValueError(
                    f"duplicate review evidence across apply manifests: {duplicates[:5]}"
                )
            review_by_id.update(staging)
            evidence["review_apply_manifests"].append(provenance)
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(
                _blocker(
                    "review_apply_manifest_invalid",
                    detail=f"{Path(apply_path).resolve()}: {exc}",
                )
            )
    if not review_apply_manifest_paths:
        blockers.append(_blocker("review_apply_manifest_missing"))

    missing_ids = [base_fact_id for base_fact_id in source_ids if base_fact_id not in review_by_id]
    outcome_ids: Dict[str, List[str]] = {
        "accept": [],
        "reject": [],
        "defer": [],
        "revise": [],
        "missing": list(missing_ids),
        "incomplete": [],
    }
    for base_fact_id, row in review_by_id.items():
        outcome = str(row.get("review_outcome") or "missing")
        outcome_ids.setdefault(outcome, []).append(base_fact_id)
        completion = row.get("review_completion")
        if outcome == "accept" and (
            not isinstance(completion, dict)
            or completion.get("codex_proxy_scope_review_complete") is not True
        ):
            outcome_ids["incomplete"].append(base_fact_id)
        if outcome == "reject" and (
            not isinstance(completion, dict)
            or completion.get("member_review_complete") is not True
        ):
            outcome_ids["incomplete"].append(base_fact_id)
    for outcome in outcome_ids:
        outcome_ids[outcome] = sorted(set(outcome_ids[outcome]))
    if outcome_ids["missing"]:
        blockers.append(
            _blocker(
                "review_decision_missing",
                count=len(outcome_ids["missing"]),
                examples=outcome_ids["missing"],
            )
        )
    if outcome_ids["defer"]:
        blockers.append(
            _blocker(
                "review_decision_deferred",
                count=len(outcome_ids["defer"]),
                examples=outcome_ids["defer"],
            )
        )
    if outcome_ids["incomplete"]:
        blockers.append(
            _blocker(
                "review_target_checks_incomplete",
                count=len(outcome_ids["incomplete"]),
                examples=outcome_ids["incomplete"],
            )
        )

    revised_ids = outcome_ids["revise"]
    if revised_ids:
        if revision_lineage_path is None:
            blockers.append(
                _blocker(
                    "revision_lineage_missing",
                    count=len(revised_ids),
                    examples=revised_ids,
                )
            )
        if revision_rereview_path is None:
            blockers.append(
                _blocker(
                    "revision_rereview_missing",
                    count=len(revised_ids),
                    examples=revised_ids,
                )
            )
        if revision_lineage_path is not None and revision_rereview_path is not None:
            try:
                lineage, rereview, rejected_revisions = _validate_revision_evidence(
                    revision_lineage_path,
                    revision_rereview_path,
                    universe_id=universe_id,
                    revised_source_ids=revised_ids,
                    item_by_source_id=item_by_source_id,
                    source_behavior_by_id=source_behavior_by_id,
                    source_review_by_id=review_by_id,
                )
                evidence["revision_lineage"] = lineage
                evidence["revision_rereview"] = {
                    **rereview,
                    "rejected_count": len(rejected_revisions),
                }
                if rejected_revisions:
                    blockers.append(
                        _blocker(
                            "revision_rereview_rejected",
                            count=len(rejected_revisions),
                            examples=rejected_revisions,
                        )
                    )
            except (FileNotFoundError, OSError, ValueError) as exc:
                blockers.append(_blocker("revision_evidence_invalid", detail=str(exc)))

    candidates: List[Dict[str, Any]] = []
    if near_audit_summary_path is None:
        blockers.append(_blocker("near_duplicate_audit_missing"))
    else:
        try:
            candidates, near_provenance, near_counts = validate_near_audit(
                near_audit_summary_path,
                universe=universe,
                universe_source_ids=source_ids,
            )
            evidence["near_duplicate_audit"] = {
                **near_provenance,
                **near_counts,
            }
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(_blocker("near_duplicate_audit_invalid", detail=str(exc)))
    if candidates:
        if near_adjudications_path is None:
            blockers.append(
                _blocker(
                    "near_duplicate_adjudications_missing",
                    count=len(candidates),
                    examples=[str(row.get("pair_id")) for row in candidates],
                )
            )
        else:
            try:
                evidence["near_duplicate_adjudications"] = validate_near_adjudications(
                    near_adjudications_path, candidates
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                blockers.append(
                    _blocker("near_duplicate_adjudications_invalid", detail=str(exc))
                )
    elif near_audit_summary_path is not None and near_adjudications_path is None:
        blockers.append(_blocker("near_duplicate_adjudications_missing", count=0))

    completion_inputs = (
        (
            "semantic_closure_attestation",
            semantic_closure_attestation_path,
            SEMANTIC_CLOSURE_SCHEMA_VERSION,
            "semantic_paraphrase_closure_missing",
        ),
        (
            "recluster_manifest",
            recluster_manifest_path,
            RECLUSTER_MANIFEST_SCHEMA_VERSION,
            "recluster_result_missing",
        ),
        (
            "split_recompute_manifest",
            split_recompute_manifest_path,
            SPLIT_RECOMPUTE_MANIFEST_SCHEMA_VERSION,
            "split_recomputation_missing",
        ),
    )
    for evidence_key, artifact_path, schema, missing_code in completion_inputs:
        if artifact_path is None:
            blockers.append(_blocker(missing_code))
            continue
        try:
            snapshot = _validate_completion_manifest(
                artifact_path,
                expected_schema=schema,
                universe_id=universe_id,
            )
            evidence[evidence_key] = {
                **snapshot,
                "verification_level": "presence_only_self_attested",
                "can_authorize_finalization": False,
            }
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(_blocker(f"{evidence_key}_invalid", detail=str(exc)))

    historical_registry: Optional[Dict[str, Any]] = None
    historical_registry_snapshot: Optional[Dict[str, Any]] = None
    if historical_exposure_registry_path is None:
        blockers.append(_blocker("historical_exposure_registry_missing"))
    else:
        try:
            historical_registry, historical_registry_snapshot = (
                _validate_historical_registry(
                historical_exposure_registry_path, universe_id
                )
            )
            evidence["historical_exposure_registry"] = {
                **historical_registry_snapshot,
                "verification_level": "runtime_self_attested_not_predeclared",
                "inventory_completeness_locally_provable": False,
                "can_authorize_finalization": False,
            }
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(
                _blocker("historical_exposure_registry_invalid", detail=str(exc))
            )
    if historical_exposure_checks_path is None:
        blockers.append(_blocker("historical_exposure_checks_missing"))
    elif historical_registry is None or historical_registry_snapshot is None:
        blockers.append(
            _blocker(
                "historical_exposure_checks_invalid",
                detail="A valid bound historical exposure registry is required",
            )
        )
    else:
        try:
            historical_checks, exposed_ids = _validate_historical_checks(
                historical_exposure_checks_path,
                universe_id,
                item_by_cohort_id,
                historical_registry,
                str(historical_registry_snapshot["sha256"]),
            )
            evidence["historical_exposure_checks"] = historical_checks
            if exposed_ids:
                blockers.append(
                    _blocker(
                        "historical_exposure_detected",
                        count=len(exposed_ids),
                        examples=exposed_ids,
                    )
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            blockers.append(
                _blocker("historical_exposure_checks_invalid", detail=str(exc))
            )

    blocker_codes = [blocker["code"] for blocker in blockers]
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "universe": {
            "path": universe_provenance["path"],
            "sha256": universe_provenance["sha256"],
            "universe_id": universe_id,
            "record_count": len(items),
            "selection_mode": universe["selection_policy"]["mode"],
        },
        "review_counts": {
            "universe": len(items),
            "accept": len(outcome_ids["accept"]),
            "reject": len(outcome_ids["reject"]),
            "defer": len(outcome_ids["defer"]),
            "revise": len(outcome_ids["revise"]),
            "missing": len(outcome_ids["missing"]),
            "incomplete_target_checks": len(outcome_ids["incomplete"]),
        },
        "evidence": evidence,
        "blockers": blockers,
        "blocker_codes": blocker_codes,
        "ready_for_future_finalizer": not blockers,
        "reviewed_bundle_finalizer_implemented": False,
        "freeze_outputs_emitted": False,
        "formal_bundle_emitted": False,
        "integrity_scope": "local_byte_and_sha_consistency_without_digital_signature",
        "trust_boundaries": {
            "historical_exposure_inventory_completeness": (
                "not_predeclared_requires_user_supplied_external_authority_contract"
            ),
            "completion_attestations": "presence_only_self_attested",
        },
    }
    write_json(output_path, report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    declare = commands.add_parser(
        "declare-universe", help="Materialize an immutable formal cohort universe"
    )
    declare.add_argument("--source-dir", type=Path, required=True)
    declare.add_argument("--output-dir", type=Path, required=True)
    declare.add_argument("--mode", choices=("full-pool", "review-scope"), required=True)
    declare.add_argument("--universe-label", required=True)
    declare.add_argument("--selection-policy-id", required=True)
    declare.add_argument("--comparison-canonical", type=Path, required=True)
    declare.add_argument("--comparison-dataset-id", required=True)
    declare.add_argument("--near-question-threshold", type=float, default=0.80)
    declare.add_argument("--near-fact-threshold", type=float, default=0.80)
    declare.add_argument("--near-max-bucket-neighbors", type=int, default=24)
    declare.add_argument("--near-max-examples", type=int, default=20)
    declare.add_argument("--review-scope-manifest", type=Path)

    preflight = commands.add_parser(
        "preflight", help="Report blockers without emitting a formal bundle or freeze"
    )
    preflight.add_argument("--universe-manifest", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--review-apply-manifest", type=Path, action="append", default=[])
    preflight.add_argument("--revision-lineage", type=Path)
    preflight.add_argument("--revision-rereview", type=Path)
    preflight.add_argument("--near-audit-summary", type=Path)
    preflight.add_argument("--near-adjudications", type=Path)
    preflight.add_argument("--semantic-closure-attestation", type=Path)
    preflight.add_argument("--historical-exposure-registry", type=Path)
    preflight.add_argument("--historical-exposure-checks", type=Path)
    preflight.add_argument("--recluster-manifest", type=Path)
    preflight.add_argument("--split-recompute-manifest", type=Path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "declare-universe":
        result = materialize_cohort_universe(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            universe_label=args.universe_label,
            selection_policy_id=args.selection_policy_id,
            comparison_canonical_path=args.comparison_canonical,
            comparison_dataset_id=args.comparison_dataset_id,
            near_question_threshold=args.near_question_threshold,
            near_fact_threshold=args.near_fact_threshold,
            near_max_bucket_neighbors=args.near_max_bucket_neighbors,
            near_max_examples=args.near_max_examples,
            review_scope_manifest_path=args.review_scope_manifest,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    report = build_preflight_report(
        universe_manifest_path=args.universe_manifest,
        output_path=args.output,
        review_apply_manifest_paths=args.review_apply_manifest,
        revision_lineage_path=args.revision_lineage,
        revision_rereview_path=args.revision_rereview,
        near_audit_summary_path=args.near_audit_summary,
        near_adjudications_path=args.near_adjudications,
        semantic_closure_attestation_path=args.semantic_closure_attestation,
        historical_exposure_registry_path=args.historical_exposure_registry,
        historical_exposure_checks_path=args.historical_exposure_checks,
        recluster_manifest_path=args.recluster_manifest,
        split_recompute_manifest_path=args.split_recompute_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready_for_future_finalizer"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
