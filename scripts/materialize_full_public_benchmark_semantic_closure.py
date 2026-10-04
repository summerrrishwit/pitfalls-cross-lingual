#!/usr/bin/env python3
"""Materialize review-ready semantic-closure candidates for the full pool.

This command is deliberately offline and non-adjudicative.  It validates the
declared formal universe and its four original source artifacts, adapts the
current full-base rows to the repository lexical/alias candidate generator,
then rebinds every emitted endpoint to the *current* base/split/component
artifacts.  The temporary adapter, its row hashes, and the audit tool's
``cross_split`` values are never reused as current-state evidence.

The output is a bounded candidate set, not an exhaustive semantic census.  All
semantic decisions remain pending; no review/split freeze or HF work is
authorized or performed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "factual_triples"
    / "public-benchmarks-full-v1"
    / "provisional"
    / "qwen3.7-plus-v1"
)
DEFAULT_FORMAL_UNIVERSE_MANIFEST = (
    DEFAULT_SOURCE_DIR
    / "full-8969-formal-universe-v1"
    / "formal_cohort_universe_manifest.json"
)
DEFAULT_FULL_BASE_FACTS = (
    DEFAULT_SOURCE_DIR / "full-8969-pre-hf-v1" / "full_base_facts.jsonl"
)
DEFAULT_COMPARISON_CANONICAL = (
    PROJECT_ROOT
    / "data_processed"
    / "factual_triples"
    / "triple-full-v1"
    / "canonical-v2"
    / "canonical_triples.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_SOURCE_DIR
    / "full-8969-formal-universe-v1"
    / "semantic-closure-candidates-v1"
)
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_public_benchmark_near_duplicates.py"
FACT_RESOLUTION_SCRIPT = (
    PROJECT_ROOT / "scripts" / "resolve_full_public_benchmark_fact_review.py"
)

TOOL_VERSION = "public-benchmark-full-semantic-closure-materializer-v1"
UNIVERSE_MANIFEST_SCHEMA = "public-benchmark-formal-cohort-universe-v2"
UNIVERSE_ITEM_SCHEMA = "public-benchmark-formal-cohort-item-v1"
FULL_BASE_SCHEMA = "public-benchmark-full-provisional-base-fact-v1"
SPLIT_MANIFEST_SCHEMA = "public-benchmark-full-provisional-split-manifest-v1"
COMPONENT_SCHEMA = "public-benchmark-full-leakage-component-v1"
AUDIT_INPUT_SCHEMA = "factual-perturbation-input-bundle-v1"
AUDIT_SUMMARY_SCHEMA = "public-benchmark-lexical-candidate-audit-v2"
AUDIT_PAIR_SCHEMA = "public-benchmark-lexical-candidate-pair-v2"

CANDIDATE_SCHEMA = "public-benchmark-full-semantic-closure-candidate-v1"
ADJUDICATION_TEMPLATE_SCHEMA = (
    "public-benchmark-full-semantic-closure-adjudication-template-v1"
)
MANIFEST_SCHEMA = "public-benchmark-full-semantic-closure-candidate-manifest-v1"
RESOLUTION_MANIFEST_SCHEMA = "full-public-benchmark-fact-resolution-manifest-v1"
RESOLUTION_LEDGER_SCHEMA = "full-public-benchmark-fact-resolution-ledger-v1"
REVISION_LINEAGE_SCHEMA = "full-public-benchmark-fact-revision-lineage-v1"
COHORT_EXCLUSION_SCHEMA = "full-public-benchmark-fact-cohort-exclusion-v1"

ALLOWED_SPLITS = frozenset({"development", "validation", "sealed"})
ALLOWED_RELATIONSHIP_DECISIONS = (
    "same_fact",
    "same_leakage_component",
    "distinct",
    "exclude_cohort",
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


def serialize_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
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
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Empty JSONL input: {path}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, serialize_jsonl(rows))


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _index_unique(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        value = _required_string(row.get(field), f"{label} row {row_number}.{field}")
        if value in output:
            raise ValueError(f"Duplicate {field} in {label}: {value}")
        output[value] = dict(row)
    return output


def _input_binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _output_binding(
    path: Path, *, schema_version: str, record_count: Optional[int] = None
) -> Dict[str, Any]:
    result = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _resolve_declared_path(
    value: Any, *, owner_path: Path, label: str
) -> Path:
    raw = _required_string(value, f"{label}.path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = owner_path.parent / candidate
    return candidate.resolve()


def _validate_file_binding(
    binding: Any,
    *,
    owner_path: Path,
    expected_path: Path,
    label: str,
    expected_schema: Optional[str] = None,
    record_count: Optional[int] = None,
) -> None:
    value = _required_mapping(binding, label)
    path = _resolve_declared_path(value.get("path"), owner_path=owner_path, label=label)
    expected_path = Path(expected_path).resolve()
    if path != expected_path:
        raise ValueError(f"{label} path does not match the supplied input")
    if not path.is_file():
        raise FileNotFoundError(path)
    if value.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if value.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte count is stale")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
    if record_count is not None and value.get("record_count") != record_count:
        raise ValueError(f"{label} record_count is stale")


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_full_pool_semantic_closure_lexical_audit", AUDIT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load lexical candidate generator: {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fact_resolution_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_full_pool_semantic_closure_fact_resolution", FACT_RESOLUTION_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load fact-resolution validator: {FACT_RESOLUTION_SCRIPT}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_source_artifacts(
    *,
    universe_manifest: Mapping[str, Any],
    universe_manifest_path: Path,
    source_dir: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    declared = _required_mapping(
        universe_manifest.get("source_artifacts"), "formal universe source_artifacts"
    )
    source_dir = Path(source_dir).resolve()
    rows_by_role: Dict[str, List[Dict[str, Any]]] = {}
    bindings: Dict[str, Dict[str, Any]] = {}
    indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for role, (filename, schema_version, identifier_field) in SOURCE_ARTIFACTS.items():
        path = source_dir / filename
        rows = read_jsonl(path)
        binding = _required_mapping(declared.get(role), f"source_artifacts.{role}")
        _validate_file_binding(
            binding,
            owner_path=universe_manifest_path,
            expected_path=path,
            label=f"source_artifacts.{role}",
            expected_schema=schema_version,
            record_count=len(rows),
        )
        for row_number, row in enumerate(rows, start=1):
            if row.get("schema_version") != schema_version:
                raise ValueError(
                    f"source {role} row {row_number} has unsupported schema_version"
                )
        index = _index_unique(rows, identifier_field, f"source {role}")
        identifiers = [str(row[identifier_field]) for row in rows]
        if binding.get("record_ids_sha256") != sha256_value(identifiers):
            raise ValueError(f"source_artifacts.{role} record ID digest is stale")
        if binding.get("record_row_hashes_sha256") != sha256_value(
            [sha256_value(row) for row in rows]
        ):
            raise ValueError(f"source_artifacts.{role} row digest is stale")
        rows_by_role[role] = rows
        indexes[role] = index
        bindings[role] = copy.deepcopy(dict(binding))

    base_universe = set(indexes["behavior_bundle"])
    if set(indexes["review_queue"]) != base_universe:
        raise ValueError("source review_queue base-fact universe differs")
    if set(indexes["base_fact_clusters"]) != base_universe:
        raise ValueError("source base_fact_clusters base-fact universe differs")
    provisional_base_ids = {
        _required_string(row.get("base_fact_id"), "provisional_triples.base_fact_id")
        for row in rows_by_role["provisional_triples"]
    }
    if provisional_base_ids != base_universe:
        raise ValueError("source provisional_triples base-fact universe differs")
    return rows_by_role, bindings


def _validate_universe(
    *,
    manifest_path: Path,
    expected_record_count: int,
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != UNIVERSE_MANIFEST_SCHEMA:
        raise ValueError("Formal universe manifest schema_version is unsupported")
    if manifest.get("universe_status") != "declared_immutable_not_reviewed":
        raise ValueError("Formal universe is not in declared immutable review-pending state")
    if manifest.get("review_complete") is not False:
        raise ValueError("Formal universe unexpectedly claims review completion")
    if manifest.get("canonical_freeze_authorized") is not False:
        raise ValueError("Formal universe unexpectedly authorizes canonical freeze")
    if manifest.get("split_freeze_authorized") is not False:
        raise ValueError("Formal universe unexpectedly authorizes split freeze")
    if manifest.get("record_count") != expected_record_count:
        raise ValueError("Formal universe record_count differs from expected_record_count")

    item_binding = _required_mapping(manifest.get("items"), "formal universe items")
    items_path = _resolve_declared_path(
        item_binding.get("path"), owner_path=manifest_path, label="formal universe items"
    )
    items = read_jsonl(items_path)
    _validate_file_binding(
        item_binding,
        owner_path=manifest_path,
        expected_path=items_path,
        label="formal universe items",
        expected_schema=UNIVERSE_ITEM_SCHEMA,
        record_count=len(items),
    )
    if len(items) != expected_record_count:
        raise ValueError("Formal universe items count differs from expected_record_count")

    universe_id = _required_string(manifest.get("universe_id"), "universe_id")
    behavior = _index_unique(source_rows["behavior_bundle"], "base_fact_id", "behavior")
    review = _index_unique(source_rows["review_queue"], "base_fact_id", "review queue")
    clusters = _index_unique(
        source_rows["base_fact_clusters"], "base_fact_id", "base-fact clusters"
    )
    triples = _index_unique(
        source_rows["provisional_triples"], "candidate_id", "provisional triples"
    )
    item_index: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    for index, item in enumerate(items):
        if item.get("schema_version") != UNIVERSE_ITEM_SCHEMA:
            raise ValueError(f"Formal universe item {index + 1} has unsupported schema")
        if item.get("universe_id") != universe_id:
            raise ValueError(f"Formal universe item {index + 1} has wrong universe_id")
        if item.get("selection_index") != index:
            raise ValueError(f"Formal universe item {index + 1} has stale selection_index")
        base_fact_id = _required_string(
            item.get("source_base_fact_id"), f"formal universe item {index + 1}.source_base_fact_id"
        )
        if base_fact_id in item_index:
            raise ValueError(f"Duplicate formal-universe base_fact_id: {base_fact_id}")
        if base_fact_id not in behavior:
            raise ValueError(f"Formal universe item is absent from source: {base_fact_id}")
        if item.get("source_id") != behavior[base_fact_id].get("source_id"):
            raise ValueError(f"Formal universe source_id is stale: {base_fact_id}")
        row_bindings = _required_mapping(
            item.get("source_row_bindings"), f"formal universe item {base_fact_id}.source_row_bindings"
        )
        expected_row_hashes = {
            "behavior_row_sha256": sha256_value(behavior[base_fact_id]),
            "review_queue_row_sha256": sha256_value(review[base_fact_id]),
            "base_fact_cluster_row_sha256": sha256_value(clusters[base_fact_id]),
        }
        if dict(row_bindings) != expected_row_hashes:
            raise ValueError(f"Formal universe source-row hashes are stale: {base_fact_id}")
        members = clusters[base_fact_id].get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Source cluster has no members: {base_fact_id}")
        expected_members = []
        for member in members:
            member = _required_mapping(member, f"cluster member for {base_fact_id}")
            candidate_id = _required_string(
                member.get("candidate_id"), f"cluster member candidate_id for {base_fact_id}"
            )
            if candidate_id not in triples:
                raise ValueError(f"Unknown source candidate_id: {candidate_id}")
            expected_members.append(
                {
                    "candidate_id": candidate_id,
                    "source_input_record_sha256": member.get("input_record_sha256"),
                    "triple_record_sha256": sha256_value(triples[candidate_id]),
                }
            )
        if item.get("member_count") != len(expected_members):
            raise ValueError(f"Formal universe member_count is stale: {base_fact_id}")
        if item.get("member_bindings") != expected_members:
            raise ValueError(f"Formal universe member bindings are stale: {base_fact_id}")
        item_index[base_fact_id] = dict(item)
        ordered_ids.append(base_fact_id)

    if set(ordered_ids) != set(behavior):
        raise ValueError("Formal universe does not exactly cover the source base facts")
    if manifest.get("ordered_source_base_fact_ids_sha256") != sha256_value(ordered_ids):
        raise ValueError("Formal universe ordered base-fact digest is stale")
    if manifest.get("ordered_cohort_item_ids_sha256") != sha256_value(
        [item["cohort_item_id"] for item in items]
    ):
        raise ValueError("Formal universe ordered item-ID digest is stale")
    if manifest.get("ordered_item_row_hashes_sha256") != sha256_value(
        [sha256_value(item) for item in items]
    ):
        raise ValueError("Formal universe ordered item-row digest is stale")
    selection_source = _required_mapping(
        manifest.get("selection_source"), "formal universe selection_source"
    )
    if selection_source.get("record_count") != len(ordered_ids):
        raise ValueError("Formal universe selection_source record_count is stale")
    if selection_source.get("ordered_base_fact_ids_sha256") != sha256_value(ordered_ids):
        raise ValueError("Formal universe selection_source ID digest is stale")
    return manifest, items, item_index


def _validate_comparison_canonical(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    comparison_path: Path,
    source_bindings: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    comparison_path = Path(comparison_path).resolve()
    rows = read_jsonl(comparison_path)
    binding = _required_mapping(
        manifest.get("comparison_canonical"), "comparison_canonical"
    )
    _validate_file_binding(
        binding,
        owner_path=manifest_path,
        expected_path=comparison_path,
        label="comparison_canonical",
        record_count=len(rows),
    )
    if binding.get("dataset_id") != "canonical-v2":
        raise ValueError("comparison_canonical dataset_id must be canonical-v2")
    record_ids = [
        {"source_id": row.get("source_id"), "candidate_id": row.get("candidate_id")}
        for row in rows
    ]
    if binding.get("ordered_record_ids_sha256") != sha256_value(record_ids):
        raise ValueError("comparison_canonical ordered record-ID digest is stale")
    if binding.get("ordered_row_hashes_sha256") != sha256_value(
        [sha256_value(row) for row in rows]
    ):
        raise ValueError("comparison_canonical ordered row digest is stale")
    for role, source in source_bindings.items():
        if Path(str(source["path"])).resolve() == comparison_path:
            raise ValueError(f"comparison_canonical aliases source artifact path: {role}")
        if source.get("sha256") == binding.get("sha256"):
            raise ValueError(f"comparison_canonical aliases source artifact SHA: {role}")
    return rows, copy.deepcopy(dict(binding))


def _validate_full_base_and_context(
    *,
    full_base_path: Path,
    split_manifest_path: Path,
    components_path: Path,
    expected_record_count: int,
    universe_ids: Sequence[str],
    source_behavior_rows: Sequence[Mapping[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    full_base_path = Path(full_base_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    components_path = Path(components_path).resolve()
    full_rows = read_jsonl(full_base_path)
    if len(full_rows) != expected_record_count:
        raise ValueError("full_base_facts count differs from expected_record_count")
    full_index = _index_unique(full_rows, "base_fact_id", "full_base_facts")
    ordered_ids = [str(row["base_fact_id"]) for row in full_rows]
    if ordered_ids != list(universe_ids):
        raise ValueError("full_base_facts order/universe differs from formal universe")
    source_behavior = _index_unique(
        source_behavior_rows, "base_fact_id", "source behavior bundle"
    )
    lineage_fields = (
        "candidate_id",
        "source_id",
        "source_dataset",
        "source_subset",
        "source_question_en",
        "canonical_fact_en",
        "answer_en",
        "answer_aliases_en",
    )
    for row_number, row in enumerate(full_rows, start=1):
        base_fact_id = str(row["base_fact_id"])
        if row.get("schema_version") != FULL_BASE_SCHEMA:
            raise ValueError(f"full_base_facts row {row_number} has unsupported schema")
        if row.get("split_assignment") not in ALLOWED_SPLITS:
            raise ValueError(f"full_base_facts row has invalid split: {base_fact_id}")
        _required_string(row.get("leakage_component_id"), f"{base_fact_id}.leakage_component_id")
        _required_string(row.get("split_policy_version"), f"{base_fact_id}.split_policy_version")
        if row.get("split_status") != "provisional_not_frozen":
            raise ValueError(f"full_base_facts row unexpectedly claims frozen split: {base_fact_id}")
        if row.get("hf_model_execution_status") != "not_run":
            raise ValueError(f"full_base_facts row already has HF model execution: {base_fact_id}")
        if row.get("hf_tokenizer_execution_status") != "not_run":
            raise ValueError(f"full_base_facts row already has HF tokenizer execution: {base_fact_id}")
        source = source_behavior[base_fact_id]
        for field in lineage_fields:
            if row.get(field) != source.get(field):
                raise ValueError(f"full_base_facts {field} drifted from source: {base_fact_id}")

    split_manifest = read_json(split_manifest_path)
    if split_manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("split_manifest schema_version is unsupported")
    if split_manifest.get("split_status") != "provisional_not_frozen":
        raise ValueError("split_manifest is not provisional_not_frozen")
    if split_manifest.get("formal_split_freeze_performed") is not False:
        raise ValueError("split_manifest unexpectedly claims formal split freeze")
    if split_manifest.get("semantic_paraphrase_closure_complete_for_full_pool") is not False:
        raise ValueError("split_manifest unexpectedly claims semantic closure completion")
    if split_manifest.get("base_fact_count") != len(full_rows):
        raise ValueError("split_manifest base_fact_count is stale")
    split_policy = _required_string(
        split_manifest.get("split_policy_version"), "split_manifest.split_policy_version"
    )
    if any(row.get("split_policy_version") != split_policy for row in full_rows):
        raise ValueError("full_base_facts split_policy_version differs from split_manifest")
    observed_split_counts = Counter(str(row["split_assignment"]) for row in full_rows)
    if split_manifest.get("actual_base_fact_counts") != {
        split: observed_split_counts[split] for split in ("development", "validation", "sealed")
    }:
        raise ValueError("split_manifest actual_base_fact_counts is stale")

    component_rows = read_jsonl(components_path)
    component_index = _index_unique(
        component_rows, "leakage_component_id", "leakage_components"
    )
    if split_manifest.get("leakage_component_count") != len(component_rows):
        raise ValueError("split_manifest leakage_component_count is stale")
    covered: Dict[str, str] = {}
    for component_id, component in component_index.items():
        if component.get("schema_version") != COMPONENT_SCHEMA:
            raise ValueError(f"leakage component has unsupported schema: {component_id}")
        members = component.get("base_fact_ids")
        if not isinstance(members, list) or component.get("member_count") != len(members):
            raise ValueError(f"leakage component member list is stale: {component_id}")
        component_split = component.get("split_assignment")
        if component_split not in ALLOWED_SPLITS:
            raise ValueError(f"leakage component has invalid split: {component_id}")
        for raw_id in members:
            base_fact_id = _required_string(raw_id, f"{component_id}.base_fact_ids")
            if base_fact_id in covered:
                raise ValueError(f"Base fact occurs in multiple components: {base_fact_id}")
            if base_fact_id not in full_index:
                raise ValueError(f"Leakage component contains unknown fact: {base_fact_id}")
            row = full_index[base_fact_id]
            if row.get("leakage_component_id") != component_id:
                raise ValueError(f"Current component assignment is stale: {base_fact_id}")
            if row.get("split_assignment") != component_split:
                raise ValueError(f"Current component split is stale: {base_fact_id}")
            covered[base_fact_id] = component_id
    if set(covered) != set(full_index):
        raise ValueError("leakage_components do not exactly cover full_base_facts")
    return full_rows, full_index, split_manifest, component_rows


def _resolve_manifest_binding_path(
    binding: Mapping[str, Any], *, owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    return _resolve_declared_path(raw, owner_path=owner_path, label=label)


def _verified_resolution_artifact(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: Optional[str] = None,
) -> Path:
    value = _required_mapping(binding, label)
    path = _resolve_manifest_binding_path(value, owner_path=owner_path, label=label)
    if not path.is_file():
        raise FileNotFoundError(path)
    if value.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if value.get("byte_count") is not None and value.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte count is stale")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
    return path


def _manifest_binding_snapshots(
    value: Any, *, owner_path: Path
) -> List[Dict[str, Any]]:
    """Collect and verify every file binding reachable from a manifest."""

    snapshots: Dict[str, Dict[str, Any]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "sha256" in node and ("path" in node or "filename" in node):
                path = _resolve_manifest_binding_path(
                    node, owner_path=owner_path, label="fact-resolution dependency"
                )
                if not path.is_file() or node.get("sha256") != sha256_file(path):
                    raise ValueError(
                        f"Fact-resolution dependency SHA-256 is stale: {path}"
                    )
                snapshots[str(path)] = _input_binding(path)
            for nested in node.values():
                visit(nested)
        elif isinstance(node, list):
            for nested in node:
                visit(nested)

    visit(value)
    return [snapshots[path] for path in sorted(snapshots)]


def load_fact_resolution_context(
    *,
    fact_resolution_manifest_path: Path,
    resolved_full_base_facts_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Replay a resolution manifest and derive its exact successor cohort.

    The resolver remains the authority for edit/exclusion semantics.  This
    loader adds the downstream selection invariant: the revised artifact must
    contain every and only ``retain_*`` ledger row, in source-universe order.
    """

    manifest_path = Path(fact_resolution_manifest_path).resolve()
    resolution_tool = _load_fact_resolution_module()
    resolution_tool.validate_resolution_output(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != RESOLUTION_MANIFEST_SCHEMA:
        raise ValueError("Fact-resolution manifest schema_version is unsupported")
    if manifest.get("status") != "completed_provisional_offline_fact_resolution":
        raise ValueError("Fact-resolution manifest is not complete")
    completion = _required_mapping(
        manifest.get("completion_contract"), "fact-resolution completion_contract"
    )
    if not all(
        (
            completion.get("source_review_chain_replayed") is True,
            completion.get("source_review_scope_complete") is True,
            completion.get("all_nonaccept_rows_resolved") is True,
            completion.get("all_retained_revisions_independently_rereviewed") is True,
            completion.get("unresolved_count") == 0,
            completion.get("formal_fact_or_split_freeze_completed") is False,
        )
    ):
        raise ValueError("Fact-resolution completion contract is invalid")

    inputs = _required_mapping(manifest.get("inputs"), "fact-resolution inputs")
    artifacts = _required_mapping(
        manifest.get("artifacts"), "fact-resolution artifacts"
    )
    source_path = _verified_resolution_artifact(
        inputs.get("full_base_facts"),
        owner_path=manifest_path,
        label="fact-resolution source full_base_facts",
        expected_schema=FULL_BASE_SCHEMA,
    )
    revised_path = _verified_resolution_artifact(
        artifacts.get("revised_full_base_facts"),
        owner_path=manifest_path,
        label="fact-resolution revised_full_base_facts",
        expected_schema=FULL_BASE_SCHEMA,
    )
    if resolved_full_base_facts_path is not None and revised_path != Path(
        resolved_full_base_facts_path
    ).resolve():
        raise ValueError(
            "Supplied full_base_facts is not the fact-resolution revised artifact"
        )
    ledger_path = _verified_resolution_artifact(
        artifacts.get("fact_resolution_ledger"),
        owner_path=manifest_path,
        label="fact-resolution ledger",
        expected_schema=RESOLUTION_LEDGER_SCHEMA,
    )
    lineage_path = _verified_resolution_artifact(
        artifacts.get("fact_revision_lineage"),
        owner_path=manifest_path,
        label="fact-resolution revision lineage",
        expected_schema=REVISION_LINEAGE_SCHEMA,
    )
    exclusion_path = _verified_resolution_artifact(
        artifacts.get("cohort_exclusions"),
        owner_path=manifest_path,
        label="fact-resolution cohort exclusions",
        expected_schema=COHORT_EXCLUSION_SCHEMA,
    )

    source_rows = read_jsonl(source_path)
    revised_rows = read_jsonl(revised_path)
    ledger_rows = read_jsonl(ledger_path)
    lineage_rows = read_jsonl(lineage_path, allow_empty=True)
    exclusion_rows = read_jsonl(exclusion_path, allow_empty=True)
    source_index = _index_unique(source_rows, "base_fact_id", "resolution source facts")
    revised_index = _index_unique(
        revised_rows, "base_fact_id", "resolution retained facts"
    )
    ledger_index = _index_unique(ledger_rows, "base_fact_id", "resolution ledger")
    lineage_index = _index_unique(
        lineage_rows, "base_fact_id", "resolution revision lineage"
    )
    exclusion_index = _index_unique(
        exclusion_rows, "base_fact_id", "resolution cohort exclusions"
    )
    source_ids = [str(row["base_fact_id"]) for row in source_rows]
    if [str(row["base_fact_id"]) for row in ledger_rows] != source_ids:
        raise ValueError("Fact-resolution ledger does not preserve source-universe order")

    retained_ids: List[str] = []
    excluded_ids: List[str] = []
    revised_ids: List[str] = []
    for base_fact_id in source_ids:
        ledger = ledger_index[base_fact_id]
        disposition = ledger.get("final_disposition")
        if disposition in {"retain_accepted", "retain_revised"}:
            retained_ids.append(base_fact_id)
            if disposition == "retain_revised":
                revised_ids.append(base_fact_id)
            output_row = revised_index.get(base_fact_id)
            if output_row is None or ledger.get("output_full_fact_row_sha256") != sha256_value(
                output_row
            ):
                raise ValueError(
                    f"Fact-resolution retained row binding is stale: {base_fact_id}"
                )
        elif disposition == "cohort_exclude":
            excluded_ids.append(base_fact_id)
            if ledger.get("output_full_fact_row_sha256") is not None:
                raise ValueError(
                    f"Excluded fact unexpectedly binds an output row: {base_fact_id}"
                )
        else:
            raise ValueError(
                f"Fact-resolution ledger disposition is invalid: {base_fact_id}"
            )
    if [str(row["base_fact_id"]) for row in revised_rows] != retained_ids:
        raise ValueError(
            "Fact-resolution revised artifact is not the complete ordered retained cohort"
        )
    if set(lineage_index) != set(revised_ids):
        raise ValueError("Fact-resolution revision lineage does not match revised facts")
    if [str(row["base_fact_id"]) for row in exclusion_rows] != excluded_ids:
        raise ValueError("Fact-resolution exclusions do not match excluded ledger rows")
    if set(retained_ids).intersection(excluded_ids) or set(retained_ids) | set(
        excluded_ids
    ) != set(source_ids):
        raise ValueError("Fact-resolution retained/excluded selection is not exhaustive")

    counts = _required_mapping(manifest.get("counts"), "fact-resolution counts")
    expected_counts = {
        "source_record_count": len(source_ids),
        "retained_record_count": len(retained_ids),
        "revision_count": len(revised_ids),
        "cohort_exclusion_count": len(excluded_ids),
        "unresolved_count": 0,
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise ValueError(f"Fact-resolution {field} is stale")

    successor_payload = {
        "source_full_base_facts_sha256": sha256_file(source_path),
        "fact_resolution_manifest_sha256": sha256_file(manifest_path),
        "resolved_full_base_facts_sha256": sha256_file(revised_path),
        "ordered_retained_base_fact_ids_sha256": sha256_value(retained_ids),
        "ordered_excluded_base_fact_ids_sha256": sha256_value(excluded_ids),
        "ordered_revised_base_fact_ids_sha256": sha256_value(revised_ids),
    }
    successor_id = "resolved_cohort_" + sha256_value(successor_payload)[:24]
    bindings = {
        "fact_resolution_manifest": _input_binding(
            manifest_path, schema_version=RESOLUTION_MANIFEST_SCHEMA
        ),
        "source_full_base_facts": _input_binding(
            source_path, schema_version=FULL_BASE_SCHEMA, record_count=len(source_rows)
        ),
        "revised_full_base_facts": _input_binding(
            revised_path,
            schema_version=FULL_BASE_SCHEMA,
            record_count=len(revised_rows),
        ),
        "fact_resolution_ledger": _input_binding(
            ledger_path,
            schema_version=RESOLUTION_LEDGER_SCHEMA,
            record_count=len(ledger_rows),
        ),
        "fact_revision_lineage": _input_binding(
            lineage_path,
            schema_version=REVISION_LINEAGE_SCHEMA,
            record_count=len(lineage_rows),
        ),
        "cohort_exclusions": _input_binding(
            exclusion_path,
            schema_version=COHORT_EXCLUSION_SCHEMA,
            record_count=len(exclusion_rows),
        ),
    }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "source_path": source_path,
        "source_rows": source_rows,
        "source_index": source_index,
        "revised_path": revised_path,
        "revised_rows": revised_rows,
        "revised_index": revised_index,
        "ledger_rows": ledger_rows,
        "ledger_index": ledger_index,
        "lineage_rows": lineage_rows,
        "lineage_index": lineage_index,
        "exclusion_rows": exclusion_rows,
        "exclusion_index": exclusion_index,
        "source_ids": source_ids,
        "retained_ids": retained_ids,
        "excluded_ids": excluded_ids,
        "revised_ids": revised_ids,
        "successor_id": successor_id,
        "successor_payload": successor_payload,
        "bindings": bindings,
        "snapshots": [
            _input_binding(manifest_path),
            *_manifest_binding_snapshots(manifest, owner_path=manifest_path),
        ],
    }


def _validate_resolved_full_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    retained_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    index = _index_unique(rows, "base_fact_id", "resolved full_base_facts")
    if [str(row["base_fact_id"]) for row in rows] != list(retained_ids):
        raise ValueError("Resolved full_base_facts order differs from retained cohort")
    for row_number, row in enumerate(rows, start=1):
        base_fact_id = str(row["base_fact_id"])
        if row.get("schema_version") != FULL_BASE_SCHEMA:
            raise ValueError(
                f"resolved full_base_facts row {row_number} has unsupported schema"
            )
        if row.get("split_assignment") not in ALLOWED_SPLITS:
            raise ValueError(f"Resolved fact has invalid inherited split: {base_fact_id}")
        _required_string(
            row.get("leakage_component_id"),
            f"resolved {base_fact_id}.leakage_component_id",
        )
        _required_string(
            row.get("split_policy_version"),
            f"resolved {base_fact_id}.split_policy_version",
        )
        if row.get("split_status") != "provisional_not_frozen":
            raise ValueError(f"Resolved fact unexpectedly has a frozen split: {base_fact_id}")
        if row.get("hf_model_execution_status") != "not_run":
            raise ValueError(f"Resolved fact already has HF execution: {base_fact_id}")
        if row.get("hf_tokenizer_execution_status") != "not_run":
            raise ValueError(f"Resolved fact already has tokenizer execution: {base_fact_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"Resolved fact must remain proxy evidence: {base_fact_id}")
    return index


def _project_retained_components(
    *,
    source_components: Sequence[Mapping[str, Any]],
    resolved_rows: Sequence[Mapping[str, Any]],
    excluded_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Remove excluded members without allowing them to remain graph nodes."""

    resolved_index = _index_unique(
        resolved_rows, "base_fact_id", "resolved full_base_facts"
    )
    retained = set(resolved_index)
    excluded = set(excluded_ids)
    output: List[Dict[str, Any]] = []
    covered: set[str] = set()
    for source_component in source_components:
        raw_members = source_component.get("base_fact_ids")
        if not isinstance(raw_members, list):
            raise ValueError("Source component base_fact_ids is invalid")
        members = [str(value) for value in raw_members if str(value) in retained]
        excluded_members = [str(value) for value in raw_members if str(value) in excluded]
        unknown = sorted(set(str(value) for value in raw_members) - retained - excluded)
        if unknown:
            raise ValueError(f"Source component contains unknown resolution IDs: {unknown[:5]}")
        if not members:
            continue
        component = copy.deepcopy(dict(source_component))
        component["member_count"] = len(members)
        component["base_fact_ids"] = members
        component["relation_partition_counts"] = dict(
            sorted(
                Counter(
                    str(resolved_index[value].get("relation_partition_id"))
                    for value in members
                ).items()
            )
        )
        component["source_dataset_counts"] = dict(
            sorted(
                Counter(
                    str(resolved_index[value].get("source_dataset") or "unknown")
                    for value in members
                ).items()
            )
        )
        component["answer_type_bucket_counts"] = dict(
            sorted(
                Counter(
                    str(resolved_index[value].get("answer_type_bucket") or "other")
                    for value in members
                ).items()
            )
        )
        component["resolution_projection"] = {
            "selection_is_complete_retained_component_projection": True,
            "retained_member_count": len(members),
            "excluded_member_count": len(excluded_members),
            "ordered_retained_base_fact_ids_sha256": sha256_value(members),
            "ordered_excluded_base_fact_ids_sha256": sha256_value(excluded_members),
            "excluded_members_are_graph_nodes": False,
        }
        output.append(component)
        for base_fact_id in members:
            if base_fact_id in covered:
                raise ValueError(
                    f"Resolved fact occurs in multiple projected components: {base_fact_id}"
                )
            row = resolved_index[base_fact_id]
            if row.get("leakage_component_id") != component.get(
                "leakage_component_id"
            ):
                raise ValueError(
                    f"Resolved fact inherited component binding is stale: {base_fact_id}"
                )
            if row.get("split_assignment") != component.get("split_assignment"):
                raise ValueError(
                    f"Resolved fact inherited component split is stale: {base_fact_id}"
                )
            covered.add(base_fact_id)
    if covered != retained:
        raise ValueError("Projected components do not exactly cover retained facts")
    return output


def _project_retained_split_manifest(
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_path: Path,
    resolved_rows: Sequence[Mapping[str, Any]],
    projected_components: Sequence[Mapping[str, Any]],
    resolution: Mapping[str, Any],
) -> Dict[str, Any]:
    projected = copy.deepcopy(dict(source_manifest))
    counts = Counter(str(row["split_assignment"]) for row in resolved_rows)
    projected["base_fact_count"] = len(resolved_rows)
    projected["leakage_component_count"] = len(projected_components)
    projected["actual_base_fact_counts"] = {
        split: counts[split] for split in ("development", "validation", "sealed")
    }
    frozen_binding = projected.get("frozen_split_constraint")
    if frozen_binding is not None:
        frozen_path = _verified_resolution_artifact(
            frozen_binding,
            owner_path=source_manifest_path,
            label="source frozen_split_constraint",
        )
        frozen_rows = read_jsonl(frozen_path)
        projected["frozen_split_constraint"] = _input_binding(
            frozen_path,
            schema_version=str(frozen_binding.get("schema_version") or "unknown"),
            record_count=len(frozen_rows),
        )
    projected["resolution_projection"] = {
        "source_pre_resolution_base_fact_count": len(resolution["source_ids"]),
        "retained_base_fact_count": len(resolution["retained_ids"]),
        "excluded_base_fact_count": len(resolution["excluded_ids"]),
        "selection_is_complete_retained_cohort": True,
        "ordered_retained_base_fact_ids_sha256": sha256_value(
            resolution["retained_ids"]
        ),
        "ordered_excluded_base_fact_ids_sha256": sha256_value(
            resolution["excluded_ids"]
        ),
        "split_assignments_are_inherited_provisional_context": True,
        "split_recomputation_required_after_semantic_adjudication": True,
    }
    return projected


def _audit_policy(
    universe_manifest: Mapping[str, Any], audit_tool: Any
) -> Dict[str, Any]:
    contract = _required_mapping(
        universe_manifest.get("near_duplicate_audit_contract"),
        "formal universe near_duplicate_audit_contract",
    )
    if contract.get("summary_schema_version") != audit_tool.SUMMARY_SCHEMA_VERSION:
        raise ValueError("Formal universe lexical audit summary schema is unsupported")
    if contract.get("pair_schema_version") != audit_tool.PAIR_SCHEMA_VERSION:
        raise ValueError("Formal universe lexical audit pair schema is unsupported")
    policy = copy.deepcopy(
        dict(_required_mapping(contract.get("policy"), "lexical audit policy"))
    )
    expected_fixed = {
        "text_normalization_version": audit_tool.TEXT_NORMALIZATION_VERSION,
        "answer_normalization_version": audit_tool.ANSWER_NORMALIZATION_VERSION,
        "blocking_version": audit_tool.BLOCKING_VERSION,
        "minhash_components": audit_tool.MINHASH_COMPONENTS,
        "minhash_band_rows": audit_tool.MINHASH_BAND_ROWS,
        "deterministic": True,
        "network_or_model_used": False,
    }
    for field, expected in expected_fixed.items():
        if policy.get(field) != expected:
            raise ValueError(f"Formal universe lexical audit policy mismatch: {field}")
    for field in ("question_near_threshold", "canonical_fact_near_threshold"):
        value = policy.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"Formal universe lexical audit policy invalid: {field}")
    for field in ("max_bucket_neighbors", "max_examples"):
        _required_positive_int(policy.get(field), f"lexical audit policy.{field}")
    return policy


def _adapter_rows(
    full_rows: Sequence[Mapping[str, Any]],
    source_behavior: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for full_row in full_rows:
        base_fact_id = str(full_row["base_fact_id"])
        row = copy.deepcopy(dict(full_row))
        row["schema_version"] = AUDIT_INPUT_SCHEMA
        row["split_group_id"] = full_row["leakage_component_id"]
        row["provenance"] = copy.deepcopy(source_behavior[base_fact_id].get("provenance"))
        rows.append(row)
    return rows


def _fresh_endpoint(
    *,
    raw_endpoint: Mapping[str, Any],
    full_path: Path,
    full_sha256: str,
    full_rows: Sequence[Mapping[str, Any]],
    full_index: Mapping[str, Mapping[str, Any]],
    full_lines: Mapping[str, int],
    item_index: Mapping[str, Mapping[str, Any]],
    source_behavior: Mapping[str, Mapping[str, Any]],
    source_behavior_lines: Mapping[str, int],
    canonical_path: Path,
    canonical_sha256: str,
    canonical_rows: Sequence[Mapping[str, Any]],
    audit_tool: Any,
) -> Dict[str, Any]:
    raw_provenance = _required_mapping(raw_endpoint.get("provenance"), "audit endpoint provenance")
    role = raw_provenance.get("input_role")
    if role == "current_bundle":
        base_fact_id = _required_string(raw_provenance.get("base_fact_id"), "current endpoint base_fact_id")
        if base_fact_id not in full_index:
            raise ValueError(f"Audit endpoint refers to unknown current fact: {base_fact_id}")
        row = full_index[base_fact_id]
        line_number = full_lines[base_fact_id]
        record_sha256 = sha256_value(row)
        item = item_index[base_fact_id]
        endpoint_id = "closure_endpoint_" + sha256_value(
            {
                "input_role": "current_full_base",
                "base_fact_id": base_fact_id,
                "input_record_sha256": record_sha256,
            }
        )[:24]
        provenance = {
            "endpoint_id": endpoint_id,
            "input_role": "current_full_base",
            "input_path": str(full_path),
            "input_sha256": full_sha256,
            "input_line_number": line_number,
            "input_record_sha256": record_sha256,
            "base_fact_id": base_fact_id,
            "candidate_id": row.get("candidate_id"),
            "source_id": row.get("source_id"),
            "source_dataset": row.get("source_dataset"),
            "source_subset": row.get("source_subset"),
            "formal_cohort_item_id": item.get("cohort_item_id"),
            "formal_cohort_item_sha256": sha256_value(item),
            "source_behavior_input_line_number": source_behavior_lines[base_fact_id],
            "source_behavior_row_sha256": sha256_value(source_behavior[base_fact_id]),
            "source_row_bindings": copy.deepcopy(item.get("source_row_bindings")),
            "source_member_bindings": copy.deepcopy(item.get("member_bindings")),
        }
        return {
            "provenance": provenance,
            "split_assignment": row.get("split_assignment"),
            "split_status": row.get("split_status"),
            "split_policy_version": row.get("split_policy_version"),
            "leakage_component_id": row.get("leakage_component_id"),
            "question_raw": row.get("source_question_en"),
            "canonical_fact_raw": row.get("canonical_fact_en"),
            "answer_raw": row.get("answer_en"),
            "answer_aliases_raw": copy.deepcopy(row.get("answer_aliases_en") or []),
        }
    if role != "comparison_canonical":
        raise ValueError(f"Audit endpoint has unsupported input_role: {role!r}")
    line_number = raw_provenance.get("input_line_number")
    if isinstance(line_number, bool) or not isinstance(line_number, int):
        raise ValueError("Comparison endpoint input_line_number is invalid")
    if line_number < 1 or line_number > len(canonical_rows):
        raise ValueError("Comparison endpoint input_line_number is out of range")
    row = canonical_rows[line_number - 1]
    for field in ("candidate_id", "source_id"):
        if raw_provenance.get(field) != row.get(field):
            raise ValueError(f"Comparison endpoint {field} differs from canonical input")
    record_sha256 = sha256_value(row)
    endpoint_id = "closure_endpoint_" + sha256_value(
        {
            "input_role": "comparison_canonical",
            "candidate_id": row.get("candidate_id"),
            "source_id": row.get("source_id"),
            "input_record_sha256": record_sha256,
        }
    )[:24]
    provenance = {
        "endpoint_id": endpoint_id,
        "input_role": "comparison_canonical",
        "input_path": str(canonical_path),
        "input_sha256": canonical_sha256,
        "input_line_number": line_number,
        "input_record_sha256": record_sha256,
        "base_fact_id": row.get("base_fact_id"),
        "candidate_id": row.get("candidate_id"),
        "source_id": row.get("source_id"),
        "source_dataset": row.get("source_dataset"),
        "source_subset": row.get("source_subset"),
        "canonical_policy_version": row.get("canonical_policy_version"),
        "canonical_decision": row.get("canonical_decision"),
    }
    answer = audit_tool.first_string(row, ("answer_en", "answer", "source_answer"))
    return {
        "provenance": provenance,
        "split_assignment": None,
        "split_status": None,
        "split_policy_version": None,
        "leakage_component_id": None,
        "question_raw": audit_tool.first_string(
            row, ("source_question_en", "source_question", "question")
        ),
        "canonical_fact_raw": audit_tool.first_string(
            row, ("canonical_fact_en", "canonical_fact")
        ),
        "answer_raw": answer,
        "answer_aliases_raw": list(audit_tool.extract_aliases(row, answer)),
    }


def _rebind_pairs(
    *,
    raw_pairs: Sequence[Mapping[str, Any]],
    full_path: Path,
    full_rows: Sequence[Mapping[str, Any]],
    full_index: Mapping[str, Mapping[str, Any]],
    item_index: Mapping[str, Mapping[str, Any]],
    source_behavior_rows: Sequence[Mapping[str, Any]],
    canonical_path: Path,
    canonical_rows: Sequence[Mapping[str, Any]],
    audit_tool: Any,
) -> List[Dict[str, Any]]:
    full_sha256 = sha256_file(full_path)
    canonical_sha256 = sha256_file(canonical_path)
    full_lines = {str(row["base_fact_id"]): index for index, row in enumerate(full_rows, start=1)}
    source_behavior = _index_unique(
        source_behavior_rows, "base_fact_id", "source behavior bundle"
    )
    source_behavior_lines = {
        str(row["base_fact_id"]): index
        for index, row in enumerate(source_behavior_rows, start=1)
    }
    output: List[Dict[str, Any]] = []
    seen_ids = set()
    for raw_pair in raw_pairs:
        if raw_pair.get("schema_version") != AUDIT_PAIR_SCHEMA:
            raise ValueError("Lexical candidate generator emitted unsupported pair schema")
        left = _fresh_endpoint(
            raw_endpoint=_required_mapping(raw_pair.get("left"), "raw pair left"),
            full_path=full_path,
            full_sha256=full_sha256,
            full_rows=full_rows,
            full_index=full_index,
            full_lines=full_lines,
            item_index=item_index,
            source_behavior=source_behavior,
            source_behavior_lines=source_behavior_lines,
            canonical_path=canonical_path,
            canonical_sha256=canonical_sha256,
            canonical_rows=canonical_rows,
            audit_tool=audit_tool,
        )
        right = _fresh_endpoint(
            raw_endpoint=_required_mapping(raw_pair.get("right"), "raw pair right"),
            full_path=full_path,
            full_sha256=full_sha256,
            full_rows=full_rows,
            full_index=full_index,
            full_lines=full_lines,
            item_index=item_index,
            source_behavior=source_behavior,
            source_behavior_lines=source_behavior_lines,
            canonical_path=canonical_path,
            canonical_sha256=canonical_sha256,
            canonical_rows=canonical_rows,
            audit_tool=audit_tool,
        )
        left_role = left["provenance"]["input_role"]
        right_role = right["provenance"]["input_role"]
        both_current = left_role == right_role == "current_full_base"
        scope = (
            "within_current_full_base"
            if both_current
            else "current_full_base_vs_comparison_canonical"
        )
        match_types = sorted(
            _required_string(value, "match_type")
            for value in raw_pair.get("match_types", [])
        )
        if not match_types:
            raise ValueError("Lexical candidate pair has no match_types")
        evidence = copy.deepcopy(
            dict(_required_mapping(raw_pair.get("match_evidence"), "match_evidence"))
        )
        required = []
        if "answer_alias_exact" in match_types:
            required.extend(("alias_valid", "same_answer_entity"))
        if any(value in audit_tool.TEXT_EVIDENCE_TYPES for value in match_types):
            required.append("semantic_duplicate")
        required = list(dict.fromkeys(required))
        pair_id = "semantic_pair_" + sha256_value(
            {
                "left_endpoint_id": left["provenance"]["endpoint_id"],
                "right_endpoint_id": right["provenance"]["endpoint_id"],
                "match_types": match_types,
                "match_evidence": evidence,
            }
        )[:24]
        if pair_id in seen_ids:
            raise ValueError(f"Rebound candidate pair ID collision: {pair_id}")
        seen_ids.add(pair_id)
        output.append(
            {
                "schema_version": CANDIDATE_SCHEMA,
                "pair_id": pair_id,
                "candidate_generation_status": "bounded_lexical_alias_candidate_only",
                "audit_scope": scope,
                "match_types": match_types,
                "match_evidence": evidence,
                "split_comparison_applicable": both_current,
                "cross_split": (
                    left["split_assignment"] != right["split_assignment"]
                    if both_current
                    else None
                ),
                "component_comparison_applicable": both_current,
                "same_leakage_component": (
                    left["leakage_component_id"] == right["leakage_component_id"]
                    if both_current
                    else None
                ),
                "left": left,
                "right": right,
                "review_contract": {
                    "status": "pending_adjudication",
                    "required_adjudications": required,
                    **{field: None for field in required},
                    "relationship_decision": None,
                    "allowed_relationship_decisions": list(
                        ALLOWED_RELATIONSHIP_DECISIONS
                    ),
                },
            }
        )
    output.sort(key=lambda row: str(row["pair_id"]))
    return output


def _adjudication_template(pair: Mapping[str, Any]) -> Dict[str, Any]:
    review = _required_mapping(pair.get("review_contract"), "candidate review_contract")
    required = review.get("required_adjudications")
    if not isinstance(required, list):
        raise ValueError("candidate required_adjudications is invalid")
    fields = {
        field: {
            "required": field in required,
            "value": None,
        }
        for field in ("alias_valid", "same_answer_entity", "semantic_duplicate")
    }
    return {
        "schema_version": ADJUDICATION_TEMPLATE_SCHEMA,
        "pair_id": pair["pair_id"],
        "candidate_row_sha256": sha256_value(pair),
        "left_endpoint_id": pair["left"]["provenance"]["endpoint_id"],
        "left_input_record_sha256": pair["left"]["provenance"][
            "input_record_sha256"
        ],
        "right_endpoint_id": pair["right"]["provenance"]["endpoint_id"],
        "right_input_record_sha256": pair["right"]["provenance"][
            "input_record_sha256"
        ],
        "required_adjudications": list(required),
        "field_adjudications": fields,
        "relationship_decision": None,
        "allowed_relationship_decisions": list(ALLOWED_RELATIONSHIP_DECISIONS),
        "excluded_cohort_base_fact_ids": None,
        "review_status": "pending_adjudication",
        "reviewer_type": None,
        "reviewer_id": None,
        "review_method": None,
        "reviewed_at": None,
        "rationale": None,
        "confidence": None,
        "human_gold": False,
    }


def _assert_unchanged(snapshots: Sequence[Mapping[str, Any]]) -> None:
    for snapshot in snapshots:
        path = Path(str(snapshot["path"]))
        if (
            not path.is_file()
            or sha256_file(path) != snapshot["sha256"]
            or path.stat().st_size != snapshot["byte_count"]
        ):
            raise RuntimeError(f"Input changed while materializing semantic closure: {path}")


def materialize(
    *,
    formal_universe_manifest_path: Path,
    full_base_facts_path: Path,
    source_dir: Path,
    comparison_canonical_path: Path,
    output_dir: Path,
    split_manifest_path: Optional[Path] = None,
    leakage_components_path: Optional[Path] = None,
    fact_resolution_manifest_path: Optional[Path] = None,
    expected_record_count: int = 8969,
) -> Dict[str, Any]:
    """Validate current state and emit bounded candidates plus a review template."""

    expected_record_count = _required_positive_int(
        expected_record_count, "expected_record_count"
    )
    formal_universe_manifest_path = Path(formal_universe_manifest_path).resolve()
    full_base_facts_path = Path(full_base_facts_path).resolve()
    source_dir = Path(source_dir).resolve()
    comparison_canonical_path = Path(comparison_canonical_path).resolve()
    resolution = (
        load_fact_resolution_context(
            fact_resolution_manifest_path=Path(fact_resolution_manifest_path),
            resolved_full_base_facts_path=full_base_facts_path,
        )
        if fact_resolution_manifest_path is not None
        else None
    )
    context_base_path = (
        Path(resolution["source_path"]) if resolution is not None else full_base_facts_path
    )
    split_manifest_path = (
        Path(split_manifest_path).resolve()
        if split_manifest_path is not None
        else context_base_path.parent / "split_manifest.json"
    )
    leakage_components_path = (
        Path(leakage_components_path).resolve()
        if leakage_components_path is not None
        else context_base_path.parent / "leakage_components.jsonl"
    )
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    preliminary_manifest = read_json(formal_universe_manifest_path)
    source_rows, source_bindings = _validate_source_artifacts(
        universe_manifest=preliminary_manifest,
        universe_manifest_path=formal_universe_manifest_path,
        source_dir=source_dir,
    )
    universe_manifest, universe_items, item_index = _validate_universe(
        manifest_path=formal_universe_manifest_path,
        expected_record_count=expected_record_count,
        source_rows=source_rows,
    )
    canonical_rows, canonical_binding = _validate_comparison_canonical(
        manifest=universe_manifest,
        manifest_path=formal_universe_manifest_path,
        comparison_path=comparison_canonical_path,
        source_bindings=source_bindings,
    )
    universe_ids = [str(item["source_base_fact_id"]) for item in universe_items]
    if resolution is not None and resolution["source_ids"] != universe_ids:
        raise ValueError(
            "Fact-resolution source universe differs from formal universe v2 lineage"
        )
    context_full_path = (
        Path(resolution["source_path"]) if resolution is not None else full_base_facts_path
    )
    source_full_rows, _, source_split_manifest, source_component_rows = (
        _validate_full_base_and_context(
            full_base_path=context_full_path,
            split_manifest_path=split_manifest_path,
            components_path=leakage_components_path,
            expected_record_count=expected_record_count,
            universe_ids=universe_ids,
            source_behavior_rows=source_rows["behavior_bundle"],
        )
    )
    if resolution is None:
        full_rows = source_full_rows
        full_index = _index_unique(full_rows, "base_fact_id", "full_base_facts")
        split_manifest = source_split_manifest
        component_rows = source_component_rows
    else:
        full_rows = list(resolution["revised_rows"])
        full_index = _validate_resolved_full_rows(
            rows=full_rows, retained_ids=resolution["retained_ids"]
        )
        component_rows = _project_retained_components(
            source_components=source_component_rows,
            resolved_rows=full_rows,
            excluded_ids=resolution["excluded_ids"],
        )
        split_manifest = _project_retained_split_manifest(
            source_manifest=source_split_manifest,
            source_manifest_path=split_manifest_path,
            resolved_rows=full_rows,
            projected_components=component_rows,
            resolution=resolution,
        )
    audit_tool = _load_audit_module()
    policy = _audit_policy(universe_manifest, audit_tool)
    source_behavior = _index_unique(
        source_rows["behavior_bundle"], "base_fact_id", "source behavior bundle"
    )
    adapter_rows = _adapter_rows(full_rows, source_behavior)

    with tempfile.TemporaryDirectory(prefix="full-semantic-closure-audit-") as directory:
        temporary_root = Path(directory)
        adapter_path = temporary_root / "current_full_base_adapter.jsonl"
        audit_output = temporary_root / "audit"
        write_jsonl(adapter_path, adapter_rows)
        adapter_sha256 = sha256_file(adapter_path)
        audit_summary = audit_tool.audit(
            adapter_path,
            audit_output,
            comparison_canonical_path=comparison_canonical_path,
            question_threshold=float(policy["question_near_threshold"]),
            fact_threshold=float(policy["canonical_fact_near_threshold"]),
            max_bucket_neighbors=int(policy["max_bucket_neighbors"]),
            max_examples=int(policy["max_examples"]),
        )
        if audit_summary.get("schema_version") != AUDIT_SUMMARY_SCHEMA:
            raise ValueError("Lexical candidate generator emitted unsupported summary")
        if audit_summary.get("policy") != policy:
            raise ValueError("Lexical candidate generator policy differs from formal universe")
        raw_pairs = read_jsonl(
            audit_output / "lexical_candidate_pairs.jsonl", allow_empty=True
        )
        candidates = _rebind_pairs(
            raw_pairs=raw_pairs,
            full_path=full_base_facts_path,
            full_rows=full_rows,
            full_index=full_index,
            item_index=item_index,
            source_behavior_rows=source_rows["behavior_bundle"],
            canonical_path=comparison_canonical_path,
            canonical_rows=canonical_rows,
            audit_tool=audit_tool,
        )

    templates = [_adjudication_template(pair) for pair in candidates]
    pair_ids = [str(pair["pair_id"]) for pair in candidates]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Candidate pair IDs are not unique")
    if [row["pair_id"] for row in templates] != pair_ids:
        raise ValueError("Adjudication template does not exactly cover candidate pairs")

    universe_manifest_binding = _input_binding(
        formal_universe_manifest_path, schema_version=UNIVERSE_MANIFEST_SCHEMA
    )
    universe_items_path = _resolve_declared_path(
        universe_manifest["items"]["path"],
        owner_path=formal_universe_manifest_path,
        label="formal universe items",
    )
    universe_items_binding = _input_binding(
        universe_items_path,
        schema_version=UNIVERSE_ITEM_SCHEMA,
        record_count=len(universe_items),
    )
    full_binding = {
        **_input_binding(
            full_base_facts_path,
            schema_version=FULL_BASE_SCHEMA,
            record_count=len(full_rows),
        ),
        "ordered_base_fact_ids_sha256": sha256_value(
            [str(row["base_fact_id"]) for row in full_rows]
        ),
        "ordered_row_hashes_sha256": sha256_value(
            [sha256_value(row) for row in full_rows]
        ),
        "split_assignments_sha256": sha256_value(
            [[row["base_fact_id"], row["split_assignment"]] for row in full_rows]
        ),
        "component_assignments_sha256": sha256_value(
            [[row["base_fact_id"], row["leakage_component_id"]] for row in full_rows]
        ),
    }
    source_split_binding = _input_binding(
        split_manifest_path, schema_version=SPLIT_MANIFEST_SCHEMA
    )
    source_component_binding = {
        **_input_binding(
            leakage_components_path,
            schema_version=COMPONENT_SCHEMA,
            record_count=len(source_component_rows),
        ),
        "ordered_component_ids_sha256": sha256_value(
            [row["leakage_component_id"] for row in source_component_rows]
        ),
        "ordered_row_hashes_sha256": sha256_value(
            [sha256_value(row) for row in source_component_rows]
        ),
    }
    snapshots = [
        universe_manifest_binding,
        universe_items_binding,
        full_binding,
        source_split_binding,
        source_component_binding,
        canonical_binding,
        *source_bindings.values(),
    ]
    if resolution is not None:
        snapshots.extend(resolution["snapshots"])

    scope_counts = Counter(str(row["audit_scope"]) for row in candidates)
    match_type_counts: Counter[str] = Counter()
    for row in candidates:
        match_type_counts.update(str(value) for value in row["match_types"])
    current_pairs = [
        row for row in candidates if row["split_comparison_applicable"] is True
    ]
    audit_result = {
        "summary_schema_version": audit_summary["schema_version"],
        "audit_status": audit_summary.get("audit_status"),
        "semantic_review_status": audit_summary.get("semantic_review_status"),
        "semantic_review_complete": audit_summary.get("semantic_review_complete"),
        "candidate_generation": copy.deepcopy(audit_summary.get("candidate_generation")),
        "split_integrity": copy.deepcopy(audit_summary.get("split_integrity")),
        "limitations": copy.deepcopy(audit_summary.get("limitations")),
        "recommended_next_step": audit_summary.get("recommended_next_step"),
    }
    audit_result["payload_sha256"] = sha256_value(audit_result)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        candidate_path = staged_dir / "semantic_closure_candidate_pairs.jsonl"
        template_path = staged_dir / "semantic_closure_adjudication_template.jsonl"
        manifest_path = staged_dir / "semantic_closure_candidate_manifest.json"
        write_jsonl(candidate_path, candidates)
        write_jsonl(template_path, templates)
        if resolution is None:
            split_binding = source_split_binding
            component_binding = source_component_binding
            projected_context_artifacts: Dict[str, Any] = {}
        else:
            projected_components_path = (
                staged_dir / "retained_presemantic_leakage_components.jsonl"
            )
            projected_split_path = staged_dir / "retained_presemantic_split_manifest.json"
            write_jsonl(projected_components_path, component_rows)
            write_json(projected_split_path, split_manifest)
            component_binding = {
                **_output_binding(
                    projected_components_path,
                    schema_version=COMPONENT_SCHEMA,
                    record_count=len(component_rows),
                ),
                "ordered_component_ids_sha256": sha256_value(
                    [row["leakage_component_id"] for row in component_rows]
                ),
                "ordered_row_hashes_sha256": sha256_value(
                    [sha256_value(row) for row in component_rows]
                ),
            }
            split_binding = _output_binding(
                projected_split_path, schema_version=SPLIT_MANIFEST_SCHEMA
            )
            projected_context_artifacts = {
                "retained_presemantic_leakage_components": component_binding,
                "retained_presemantic_split_manifest": split_binding,
            }
        successor_contract = None
        if resolution is not None:
            successor_contract = {
                "source_formal_universe_id": universe_manifest["universe_id"],
                "successor_universe_id": resolution["successor_id"],
                "selection_is_complete_retained_cohort": True,
                "source_record_count": len(resolution["source_ids"]),
                "retained_record_count": len(resolution["retained_ids"]),
                "revised_record_count": len(resolution["revised_ids"]),
                "excluded_record_count": len(resolution["excluded_ids"]),
                "ordered_source_base_fact_ids_sha256": sha256_value(
                    resolution["source_ids"]
                ),
                "ordered_retained_base_fact_ids_sha256": sha256_value(
                    resolution["retained_ids"]
                ),
                "ordered_revised_base_fact_ids_sha256": sha256_value(
                    resolution["revised_ids"]
                ),
                "ordered_excluded_base_fact_ids_sha256": sha256_value(
                    resolution["excluded_ids"]
                ),
                "excluded_ids_are_candidate_endpoints": False,
                "excluded_ids_are_component_members": False,
                "source_universe_lineage_preserved": True,
            }
        manifest_inputs: Dict[str, Any] = {
            "formal_universe_manifest": universe_manifest_binding,
            "formal_universe_items": universe_items_binding,
            "current_full_base_facts": full_binding,
            "current_split_manifest": split_binding,
            "current_leakage_components": component_binding,
            "original_source_artifacts": source_bindings,
            "comparison_canonical": canonical_binding,
        }
        if resolution is not None:
            manifest_inputs["fact_resolution"] = copy.deepcopy(
                resolution["bindings"]
            )
            manifest_inputs["source_pre_resolution_full_base_facts"] = (
                resolution["bindings"]["source_full_base_facts"]
            )
            manifest_inputs["source_pre_resolution_split_manifest"] = (
                source_split_binding
            )
            manifest_inputs["source_pre_resolution_leakage_components"] = (
                source_component_binding
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "bounded_candidates_materialized_semantic_decisions_pending",
            "universe_id": (
                resolution["successor_id"]
                if resolution is not None
                else universe_manifest["universe_id"]
            ),
            "source_formal_universe_id": universe_manifest["universe_id"],
            "base_fact_count": len(full_rows),
            "source_formal_universe_record_count": len(universe_ids),
            "resolved_cohort": successor_contract,
            "inputs": manifest_inputs,
            "current_context": {
                "split_status": split_manifest["split_status"],
                "split_policy_version": split_manifest["split_policy_version"],
                "leakage_component_count": len(component_rows),
                "current_base_split_component_binding_sha256": sha256_value(
                    {
                        "base": full_binding["sha256"],
                        "split": split_binding["sha256"],
                        "components": component_binding["sha256"],
                    }
                ),
            },
            "lexical_alias_candidate_generation": {
                "engine": _input_binding(AUDIT_SCRIPT),
                "original_audit_contract": copy.deepcopy(
                    universe_manifest["near_duplicate_audit_contract"]
                ),
                "original_audit_policy": policy,
                "original_audit_result": audit_result,
                "temporary_adapter": {
                    "schema_version": AUDIT_INPUT_SCHEMA,
                    "record_count": len(adapter_rows),
                    "sha256": adapter_sha256,
                    "persisted": False,
                    "derivation_policy": (
                        "copy_current_full_base_set_schema_to_factual_perturbation_"
                        "input_bundle_v1_and_set_split_group_to_current_component_v1"
                    ),
                },
                "all_emitted_candidates_retained": len(candidates) == len(raw_pairs),
                "audit_pair_ids_reused": False,
                "audit_cross_split_flags_reused": False,
                "audit_current_endpoint_hashes_reused": False,
            },
            "candidate_summary": {
                "candidate_pair_count": len(candidates),
                "ordered_pair_ids_sha256": sha256_value(pair_ids),
                "scope_counts": dict(sorted(scope_counts.items())),
                "match_type_counts": dict(sorted(match_type_counts.items())),
                "current_pair_count": len(current_pairs),
                "current_cross_split_candidate_pair_count": sum(
                    row["cross_split"] is True for row in current_pairs
                ),
                "current_same_component_candidate_pair_count": sum(
                    row["same_leakage_component"] is True for row in current_pairs
                ),
                "adjudication_template_count": len(templates),
                "semantic_decision_count": 0,
            },
            "artifacts": {
                "candidate_pairs": _output_binding(
                    candidate_path,
                    schema_version=CANDIDATE_SCHEMA,
                    record_count=len(candidates),
                ),
                "adjudication_template": _output_binding(
                    template_path,
                    schema_version=ADJUDICATION_TEMPLATE_SCHEMA,
                    record_count=len(templates),
                ),
                **projected_context_artifacts,
            },
            "upstream_review_invalidation": {
                "fact_resolution_applied": resolution is not None,
                "pre_resolution_semantic_candidates_valid": resolution is None,
                "pre_resolution_semantic_adjudications_valid": resolution is None,
                "pre_resolution_translation_reviews_valid": resolution is None,
                "pre_resolution_distractor_reviews_valid": resolution is None,
                "pre_resolution_neutral_reviews_valid": resolution is None,
                "pair_id_alone_is_sufficient_for_review_reuse": False,
                "required_rebinding": [
                    "resolved_full_base_facts_sha256",
                    "candidate_manifest_sha256",
                    "candidate_row_sha256",
                    "endpoint_input_record_sha256",
                    "split_assignments_sha256",
                ],
            },
            "limitations": {
                "candidate_generation_is_bounded": True,
                "all_record_pairs_enumerated": False,
                "semantic_decisions_pending": True,
                "semantic_adjudication_performed": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "candidate_edges_are_positive_leakage_edges": False,
            },
            "safety_contract": {
                "network_or_model_used": False,
                "semantic_closure_complete": False,
                "review_freeze_emitted": False,
                "split_recomputed": False,
                "split_freeze_emitted": False,
                "hf_checkpoint_bound": False,
                "hf_tokenizer_bound": False,
                "hf_behavior_executed": False,
                "validation_exposed": False,
                "sealed_exposed": False,
                "perturbation_authorized": False,
                "path_not_token_authorized": False,
            },
            "reviewer_contract": {
                "template_status": "pending_adjudication",
                "one_template_row_per_candidate_pair": True,
                "candidate_row_sha256_required": True,
                "endpoint_row_sha256_required": True,
                "field_level_adjudications_required_before_relationship_decision": True,
                "allowed_relationship_decisions": list(ALLOWED_RELATIONSHIP_DECISIONS),
                "positive_relationships_require_component_recompute": [
                    "same_fact",
                    "same_leakage_component",
                ],
            },
        }
        write_json(manifest_path, manifest)
        _assert_unchanged(snapshots)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        staged_dir.replace(output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged_dir, ignore_errors=True)

    return {
        "manifest_path": str(output_dir / "semantic_closure_candidate_manifest.json"),
        "manifest_sha256": sha256_file(
            output_dir / "semantic_closure_candidate_manifest.json"
        ),
        "candidate_pair_count": len(candidates),
        "adjudication_template_count": len(templates),
        "semantic_decision_count": 0,
        "semantic_closure_complete": False,
        "split_freeze_emitted": False,
        "hf_checkpoint_bound": False,
        "hf_tokenizer_bound": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--formal-universe-manifest", default=str(DEFAULT_FORMAL_UNIVERSE_MANIFEST)
    )
    result.add_argument("--full-base-facts", default=str(DEFAULT_FULL_BASE_FACTS))
    result.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    result.add_argument(
        "--comparison-canonical", default=str(DEFAULT_COMPARISON_CANONICAL)
    )
    result.add_argument("--split-manifest")
    result.add_argument("--leakage-components")
    result.add_argument(
        "--fact-resolution-manifest",
        help=(
            "Completed fact_resolution_manifest.json. When supplied, "
            "--full-base-facts must be its revised_full_base_facts artifact."
        ),
    )
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    result.add_argument("--expected-record-count", type=int, default=8969)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = materialize(
        formal_universe_manifest_path=Path(args.formal_universe_manifest),
        full_base_facts_path=Path(args.full_base_facts),
        source_dir=Path(args.source_dir),
        comparison_canonical_path=Path(args.comparison_canonical),
        output_dir=Path(args.output_dir),
        split_manifest_path=(Path(args.split_manifest) if args.split_manifest else None),
        leakage_components_path=(
            Path(args.leakage_components) if args.leakage_components else None
        ),
        fact_resolution_manifest_path=(
            Path(args.fact_resolution_manifest)
            if args.fact_resolution_manifest
            else None
        ),
        expected_record_count=args.expected_record_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
