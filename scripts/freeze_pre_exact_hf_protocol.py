#!/usr/bin/env python3
"""Freeze behavior-blind Development folds and the pre-exact-HF protocol.

This tool is deliberately unable to consume proxy behavior outputs.  It reads
only the frozen G0A manifest/bundle, its split manifest, and the authoritative
protocol document.  The resulting supplementary freeze is prospective for
exact-HF replay and PNT, but is not represented as having existed before P0.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_VERSION = "freeze-pre-exact-hf-protocol-v1"
FOLD_SCHEMA_VERSION = "factual-development-fold-manifest-v1"
PROTOCOL_SCHEMA_VERSION = "factual-perturbation-protocol-manifest-v1"
FOLD_POLICY_VERSION = (
    "relation-stratified-leakage-component-exact-four-fold-sha256-v1"
)
PROTOCOL_VERSION = "mcq-dual-distractor-pnt-pre-exact-hf-v1"
EXPECTED_SPLIT_COUNTS = {"development": 96, "validation": 32, "sealed": 32}
EXPECTED_RELATIONS = (
    "character_to_actor",
    "song_to_performer",
    "term_to_definition",
    "work_to_release_date",
)
FOLD_IDS = tuple(f"dev-fold-{index:02d}" for index in range(1, 5))
TARGET_RELATION_COUNT_PER_FOLD = 6
TARGET_FACT_COUNT_PER_FOLD = 24
EXPECTED_VARIANTS_PER_FACT = 2
EXPECTED_STATIC_INPUTS_PER_FACT = 10
COMPLETED_PROXY_RUN_ID = "static-g0a-dev96-four-model-v10-20260916"
EXPECTED_STATIC_FREEZE_MANIFEST_SHA256 = (
    "f033c9cf2d4529b2cde03878a4d01b84a462d167ce8aaaaec13cb3e9972e8636"
)
EXPECTED_FROZEN_STATIC_BUNDLE_SHA256 = (
    "601f23902b309cca12df3da07c6364214b408f8c91ac53808363bcafb138f224"
)
EXPECTED_SPLIT_FREEZE_MANIFEST_SHA256 = (
    "1a45fff2406e981a12e63b85e40bc6b2bfc39d484cd85aa534f2237b9d561fc3"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_created_at(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("created_at must be a non-empty ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("created_at must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a timezone offset")


def project_relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact must be inside project root: {resolved}") from error


def resolve_binding_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid artifact path in {manifest_path}")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def require_file_binding(manifest_path: Path, binding: Mapping[str, Any]) -> Path:
    path = resolve_binding_path(manifest_path, binding.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = binding.get("sha256")
    actual_sha = sha256_file(path)
    if expected_sha != actual_sha:
        raise ValueError(f"artifact SHA-256 mismatch: {path}")
    expected_bytes = binding.get("byte_count")
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise ValueError(f"artifact byte count mismatch: {path}")
    return path


def artifact_binding(
    path: Path,
    project_root: Path,
    *,
    schema_version: str | None = None,
    record_count: int | None = None,
    base_fact_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": project_relative(path, project_root),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    if base_fact_ids is not None:
        result["base_fact_ids_sha256"] = sha256_value(sorted(base_fact_ids))
    return result


def validate_and_load_g0a(
    project_root: Path, static_manifest_path: Path
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path, List[Dict[str, Any]]]:
    static_manifest_path = static_manifest_path.resolve()
    static_manifest = read_json(static_manifest_path)
    if static_manifest.get("schema_version") != "static-g0a-freeze-manifest-v1":
        raise ValueError("unsupported static G0A freeze manifest schema")
    if static_manifest.get("status") != "g0a_static_stimulus_frozen":
        raise ValueError("static G0A is not frozen")
    if static_manifest.get("static_frozen") is not True:
        raise ValueError("static G0A static_frozen flag is not true")
    artifacts = static_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("static G0A manifest is missing artifacts")
    bundle_binding = artifacts.get("frozen_static_bundle.jsonl")
    split_binding = artifacts.get("split_freeze_manifest.json")
    if not isinstance(bundle_binding, dict) or not isinstance(split_binding, dict):
        raise ValueError("static G0A manifest is missing bundle/split bindings")
    bundle_path = require_file_binding(static_manifest_path, bundle_binding)
    split_manifest_path = require_file_binding(static_manifest_path, split_binding)
    split_manifest = read_json(split_manifest_path)
    if split_manifest.get("schema_version") != "factual-split-freeze-manifest-v2":
        raise ValueError("unsupported split freeze manifest schema")
    if split_manifest.get("split_status") != "frozen":
        raise ValueError("split freeze manifest is not frozen")
    rows = read_jsonl(bundle_path)
    if len(rows) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise ValueError("frozen static bundle must contain exactly 160 facts")

    ids = [str(row.get("base_fact_id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("frozen static bundle has missing or duplicate base_fact_id")
    split_bundle_binding = split_manifest.get("input_bundle")
    if not isinstance(split_bundle_binding, dict):
        raise ValueError("split freeze manifest is missing its input bundle binding")
    if split_bundle_binding.get("schema_version") != "factual-perturbation-input-bundle-v1":
        raise ValueError("split freeze manifest has an unexpected input bundle schema")
    if split_bundle_binding.get("sha256") != sha256_file(bundle_path):
        raise ValueError("split freeze manifest input bundle SHA-256 mismatch")
    if split_bundle_binding.get("record_count") != len(rows):
        raise ValueError("split freeze manifest input bundle record count mismatch")
    if split_bundle_binding.get("base_fact_ids_sha256") != sha256_value(sorted(ids)):
        raise ValueError("split freeze manifest input bundle ID digest mismatch")
    split_counts = Counter(str(row.get("split_assignment") or "") for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unexpected frozen split counts: {dict(split_counts)}")

    component_splits: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        base_fact_id = str(row["base_fact_id"])
        relation = row.get("probe_relation_id")
        component_id = row.get("leakage_component_id")
        if not isinstance(relation, str) or not relation:
            raise ValueError(f"missing frozen probe_relation_id: {base_fact_id}")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError(f"missing frozen leakage_component_id: {base_fact_id}")
        lineage = row.get("postclosure_lineage")
        if not isinstance(lineage, dict):
            raise ValueError(f"missing postclosure lineage: {base_fact_id}")
        if lineage.get("postclosure_split_group_id") != component_id:
            raise ValueError(f"split group/component mismatch: {base_fact_id}")
        component_splits[component_id].add(str(row["split_assignment"]))
        static_mcq = row.get("static_mcq")
        if not isinstance(static_mcq, dict):
            raise ValueError(f"missing static MCQ: {base_fact_id}")
        variants = static_mcq.get("variants")
        inputs = static_mcq.get("behavior_inputs")
        if not isinstance(variants, list) or len(variants) != EXPECTED_VARIANTS_PER_FACT:
            raise ValueError(f"unexpected static variant count: {base_fact_id}")
        if not isinstance(inputs, list) or len(inputs) != EXPECTED_STATIC_INPUTS_PER_FACT:
            raise ValueError(f"unexpected static input count: {base_fact_id}")
    leaking = sorted(key for key, splits in component_splits.items() if len(splits) != 1)
    if leaking:
        raise ValueError(f"leakage component crosses frozen splits: {leaking[0]}")

    development = [row for row in rows if row["split_assignment"] == "development"]
    relation_counts = Counter(str(row["probe_relation_id"]) for row in development)
    if tuple(sorted(relation_counts)) != tuple(sorted(EXPECTED_RELATIONS)):
        raise ValueError(f"unexpected Development relations: {sorted(relation_counts)}")
    if any(relation_counts[relation] != 24 for relation in EXPECTED_RELATIONS):
        raise ValueError(f"Development relations are not 24 each: {dict(relation_counts)}")
    return static_manifest, bundle_path, split_manifest, split_manifest_path, rows


def component_projection(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["leakage_component_id"])].append(row)
    result: List[Dict[str, Any]] = []
    for component_id, members in grouped.items():
        member_ids = sorted(str(row["base_fact_id"]) for row in members)
        relation_counts = Counter(str(row["probe_relation_id"]) for row in members)
        result.append(
            {
                "leakage_component_id": component_id,
                "base_fact_ids": member_ids,
                "fact_count": len(member_ids),
                "relation_counts": dict(sorted(relation_counts.items())),
            }
        )
    return sorted(result, key=lambda row: row["leakage_component_id"])


def assign_development_folds(
    development_rows: Sequence[Mapping[str, Any]], bundle_sha256: str
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    components = component_projection(development_rows)
    non_singletons = [row for row in components if row["fact_count"] > 1]
    singletons = [row for row in components if row["fact_count"] == 1]
    non_singletons.sort(
        key=lambda row: (
            -int(row["fact_count"]),
            sha256_value(
                [
                    FOLD_POLICY_VERSION,
                    bundle_sha256,
                    row["leakage_component_id"],
                    row["base_fact_ids"],
                ]
            ),
        )
    )
    if len(non_singletons) > 12:
        raise ValueError("too many non-singleton components for exact fold search")

    best: Tuple[Tuple[Any, ...], Dict[str, str]] | None = None
    for choices in itertools.product(FOLD_IDS, repeat=len(non_singletons)):
        component_to_fold = {
            str(component["leakage_component_id"]): fold_id
            for component, fold_id in zip(non_singletons, choices)
        }
        relation_by_fold = {
            fold_id: Counter() for fold_id in FOLD_IDS
        }
        facts_by_fold = Counter()
        valid = True
        for component, fold_id in zip(non_singletons, choices):
            facts_by_fold[fold_id] += int(component["fact_count"])
            for relation, count in component["relation_counts"].items():
                relation_by_fold[fold_id][relation] += int(count)
                if relation_by_fold[fold_id][relation] > TARGET_RELATION_COUNT_PER_FOLD:
                    valid = False
            if facts_by_fold[fold_id] > TARGET_FACT_COUNT_PER_FOLD:
                valid = False
        if not valid:
            continue
        excess = {
            fold_id: sum(
                int(component["fact_count"]) - 1
                for component, assigned in zip(non_singletons, choices)
                if assigned == fold_id
            )
            for fold_id in FOLD_IDS
        }
        assignment_projection = sorted(component_to_fold.items())
        score = (
            max(excess.values()) - min(excess.values()),
            sum(value * value for value in excess.values()),
            sha256_value(
                [FOLD_POLICY_VERSION, bundle_sha256, assignment_projection]
            ),
        )
        if best is None or score < best[0]:
            best = (score, component_to_fold)
    if best is None:
        raise ValueError("no feasible non-singleton component fold assignment")
    component_to_fold = dict(best[1])

    relation_by_fold = {fold_id: Counter() for fold_id in FOLD_IDS}
    for component in non_singletons:
        fold_id = component_to_fold[str(component["leakage_component_id"])]
        relation_by_fold[fold_id].update(component["relation_counts"])

    singletons_by_relation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for component in singletons:
        relation_counts = component["relation_counts"]
        if len(relation_counts) != 1 or next(iter(relation_counts.values())) != 1:
            raise ValueError("singleton component has invalid relation projection")
        singletons_by_relation[next(iter(relation_counts))].append(component)
    for relation in EXPECTED_RELATIONS:
        relation_components = sorted(
            singletons_by_relation[relation],
            key=lambda row: sha256_value(
                [
                    FOLD_POLICY_VERSION,
                    bundle_sha256,
                    relation,
                    row["leakage_component_id"],
                    row["base_fact_ids"],
                ]
            ),
        )
        slots: List[Tuple[str, int]] = []
        for fold_id in FOLD_IDS:
            remaining = (
                TARGET_RELATION_COUNT_PER_FOLD
                - int(relation_by_fold[fold_id][relation])
            )
            if remaining < 0:
                raise ValueError("non-singleton assignment exceeded relation quota")
            slots.extend((fold_id, slot_index) for slot_index in range(remaining))
        slots.sort(
            key=lambda slot: sha256_value(
                [FOLD_POLICY_VERSION, bundle_sha256, relation, slot[0], slot[1]]
            )
        )
        if len(slots) != len(relation_components):
            raise ValueError(f"singleton quota mismatch for relation: {relation}")
        for component, (fold_id, _slot_index) in zip(relation_components, slots):
            component_to_fold[str(component["leakage_component_id"])] = fold_id

    assignment: Dict[str, str] = {}
    for component in components:
        fold_id = component_to_fold.get(str(component["leakage_component_id"]))
        if fold_id not in FOLD_IDS:
            raise ValueError("component did not receive a fold")
        for base_fact_id in component["base_fact_ids"]:
            if base_fact_id in assignment:
                raise ValueError(f"duplicate fold assignment: {base_fact_id}")
            assignment[str(base_fact_id)] = fold_id
    return assignment, components


def validate_fold_assignment(
    development_rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, str]
) -> Dict[str, Any]:
    by_id = {str(row["base_fact_id"]): row for row in development_rows}
    if set(assignment) != set(by_id):
        raise ValueError("fold assignment does not cover Development exactly")
    fold_counts = Counter(assignment.values())
    if set(fold_counts) != set(FOLD_IDS):
        raise ValueError("fold assignment has an invalid fold ID set")
    if any(fold_counts[fold_id] != TARGET_FACT_COUNT_PER_FOLD for fold_id in FOLD_IDS):
        raise ValueError(f"fold fact counts are not 24 each: {dict(fold_counts)}")
    relation_counts: Dict[str, Counter[str]] = {
        fold_id: Counter() for fold_id in FOLD_IDS
    }
    component_folds: Dict[str, set[str]] = defaultdict(set)
    for base_fact_id, fold_id in assignment.items():
        row = by_id[base_fact_id]
        relation_counts[fold_id][str(row["probe_relation_id"])] += 1
        component_folds[str(row["leakage_component_id"])].add(fold_id)
    if any(len(folds) != 1 for folds in component_folds.values()):
        raise ValueError("leakage component crosses Development folds")
    for fold_id in FOLD_IDS:
        expected = {relation: TARGET_RELATION_COUNT_PER_FOLD for relation in EXPECTED_RELATIONS}
        if dict(relation_counts[fold_id]) != expected:
            raise ValueError(
                f"fold relation counts are not six each: {fold_id}: "
                f"{dict(relation_counts[fold_id])}"
            )
    return {
        "all_development_ids_assigned_once": True,
        "validation_or_sealed_ids_assigned": 0,
        "leakage_components_atomic": True,
        "fold_fact_counts_exact": True,
        "fold_relation_counts_exact": True,
        "fold_count": len(FOLD_IDS),
        "fact_count_per_fold": TARGET_FACT_COUNT_PER_FOLD,
        "relation_count_per_fold": TARGET_RELATION_COUNT_PER_FOLD,
    }


def build_fold_manifest(
    *,
    project_root: Path,
    static_manifest_path: Path,
    static_manifest: Mapping[str, Any],
    bundle_path: Path,
    split_manifest_path: Path,
    split_manifest: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
    created_at: str,
) -> Dict[str, Any]:
    development = sorted(
        (row for row in all_rows if row["split_assignment"] == "development"),
        key=lambda row: str(row["base_fact_id"]),
    )
    bundle_sha = sha256_file(bundle_path)
    assignment, components = assign_development_folds(development, bundle_sha)
    checks = validate_fold_assignment(development, assignment)
    row_by_id = {str(row["base_fact_id"]): row for row in development}
    assignment_rows = [
        {
            "base_fact_id": base_fact_id,
            "source_id": row_by_id[base_fact_id].get("source_id"),
            "probe_relation_id": row_by_id[base_fact_id]["probe_relation_id"],
            "leakage_component_id": row_by_id[base_fact_id]["leakage_component_id"],
            "development_fold_id": assignment[base_fact_id],
            "source_static_record_sha256": sha256_value(row_by_id[base_fact_id]),
        }
        for base_fact_id in sorted(assignment)
    ]
    folds: List[Dict[str, Any]] = []
    all_ids = set(assignment)
    for fold_id in FOLD_IDS:
        eval_ids = sorted(base_fact_id for base_fact_id, fold in assignment.items() if fold == fold_id)
        train_ids = sorted(all_ids - set(eval_ids))
        eval_components = sorted(
            {str(row_by_id[base_fact_id]["leakage_component_id"]) for base_fact_id in eval_ids}
        )
        train_components = sorted(
            {str(row_by_id[base_fact_id]["leakage_component_id"]) for base_fact_id in train_ids}
        )
        if set(eval_components) & set(train_components):
            raise ValueError(f"train/eval component overlap: {fold_id}")
        relation_counts = Counter(
            str(row_by_id[base_fact_id]["probe_relation_id"]) for base_fact_id in eval_ids
        )
        folds.append(
            {
                "fold_id": fold_id,
                "eval_fact_count": len(eval_ids),
                "train_fact_count": len(train_ids),
                "eval_component_count": len(eval_components),
                "train_component_count": len(train_components),
                "relation_counts": dict(sorted(relation_counts.items())),
                "eval_base_fact_ids": eval_ids,
                "train_base_fact_ids_sha256": sha256_value(train_ids),
                "eval_base_fact_ids_sha256": sha256_value(eval_ids),
                "train_component_ids_sha256": sha256_value(train_components),
                "eval_component_ids_sha256": sha256_value(eval_components),
                "train_eval_fact_overlap_count": 0,
                "train_eval_component_overlap_count": 0,
            }
        )
    input_projection = [
        {
            "base_fact_id": str(row["base_fact_id"]),
            "source_id": row.get("source_id"),
            "probe_relation_id": str(row["probe_relation_id"]),
            "leakage_component_id": str(row["leakage_component_id"]),
            "split_assignment": str(row["split_assignment"]),
        }
        for row in development
    ]
    component_sizes = Counter(int(component["fact_count"]) for component in components)
    script_path = Path(__file__).resolve()
    return {
        "schema_version": FOLD_SCHEMA_VERSION,
        "status": "frozen",
        "created_at": created_at,
        "tool_version": TOOL_VERSION,
        "assignment_policy_version": FOLD_POLICY_VERSION,
        "model_independent": True,
        "applicable_model_identity": None,
        "registration_chronology": {
            "phase_created": "post_p0_pre_g0b",
            "materialized_after_proxy_screen": True,
            "preregistered_for_completed_proxy_run": False,
            "prospective_for_exact_hf_replay": True,
            "prospective_for_pnt_and_repair": True,
            "frozen_before_exact_hf_and_pnt": True,
        },
        "proxy_output_independence": {
            "behavior_outputs_consumed": [],
            "proxy_labels_used": False,
            "proxy_counts_used": False,
            "proxy_based_assignment": False,
            "generator_input_class": "g0a_static_fields_only",
        },
        "upstream_bindings": {
            "static_freeze_manifest": artifact_binding(
                static_manifest_path,
                project_root,
                schema_version=str(static_manifest["schema_version"]),
            ),
            "split_freeze_manifest": artifact_binding(
                split_manifest_path,
                project_root,
                schema_version=str(split_manifest["schema_version"]),
            ),
            "frozen_static_bundle": artifact_binding(
                bundle_path,
                project_root,
                schema_version="factual-perturbation-input-bundle-v1",
                record_count=len(all_rows),
                base_fact_ids=[str(row["base_fact_id"]) for row in all_rows],
            ),
            "split_policy_version": split_manifest.get("split_policy_version"),
        },
        "generator_code": artifact_binding(script_path, project_root),
        "scope": {
            "source_fact_count": len(all_rows),
            "source_split_counts": dict(EXPECTED_SPLIT_COUNTS),
            "assigned_split": "development",
            "assigned_fact_count": len(development),
            "excluded_validation_fact_count": EXPECTED_SPLIT_COUNTS["validation"],
            "excluded_sealed_fact_count": EXPECTED_SPLIT_COUNTS["sealed"],
            "statistical_unit": "base_fact_id",
            "stratification_field": "probe_relation_id",
            "group_atomicity_field": "leakage_component_id",
            "relation_counts": {
                relation: 24 for relation in EXPECTED_RELATIONS
            },
            "development_base_fact_ids_sha256": sha256_value(
                sorted(str(row["base_fact_id"]) for row in development)
            ),
            "development_component_ids_sha256": sha256_value(
                sorted(str(component["leakage_component_id"]) for component in components)
            ),
            "assignment_input_projection_sha256": sha256_value(input_projection),
        },
        "component_summary": {
            "component_count": len(components),
            "component_size_counts": {
                str(size): count for size, count in sorted(component_sizes.items())
            },
            "non_singleton_component_count": sum(
                count for size, count in component_sizes.items() if size > 1
            ),
            "cross_relation_component_count": sum(
                len(component["relation_counts"]) > 1 for component in components
            ),
        },
        "fold_contract": {
            "fold_ids": list(FOLD_IDS),
            "fold_count": len(FOLD_IDS),
            "eval_fact_count_per_fold": TARGET_FACT_COUNT_PER_FOLD,
            "train_fact_count_per_fold": 72,
            "relation_count_per_fold": TARGET_RELATION_COUNT_PER_FOLD,
            "component_atomicity_precedes_balance": True,
            "each_fact_eval_fold_count": 1,
            "each_fact_train_fold_count": 3,
            "vector_source_rule": "train_complement_only",
        },
        "folds": folds,
        "assignments": assignment_rows,
        "assignment_sha256": sha256_value(
            [
                {
                    "base_fact_id": row["base_fact_id"],
                    "development_fold_id": row["development_fold_id"],
                }
                for row in assignment_rows
            ]
        ),
        "checks": checks,
        "authorization_state": {
            "exact_hf_behavior_authorized": False,
            "pnt_authorized": False,
            "validation_authorized": False,
            "sealed_authorized": False,
        },
    }


def build_protocol_manifest(
    *,
    project_root: Path,
    static_manifest_path: Path,
    static_manifest: Mapping[str, Any],
    bundle_path: Path,
    split_manifest_path: Path,
    split_manifest: Mapping[str, Any],
    all_rows: Sequence[Mapping[str, Any]],
    fold_manifest_path: Path,
    fold_manifest_artifact_path: Path,
    fold_manifest: Mapping[str, Any],
    protocol_document_path: Path,
    created_at: str,
    completed_proxy_run_id: str,
) -> Dict[str, Any]:
    development_ids = sorted(
        str(row["base_fact_id"])
        for row in all_rows
        if row["split_assignment"] == "development"
    )
    fold_manifest_binding = artifact_binding(
        fold_manifest_path,
        project_root,
        schema_version=str(fold_manifest["schema_version"]),
        record_count=len(fold_manifest["assignments"]),
        base_fact_ids=[
            str(row["base_fact_id"]) for row in fold_manifest["assignments"]
        ],
    )
    fold_manifest_binding["path"] = project_relative(
        fold_manifest_artifact_path, project_root
    )
    fold_manifest_binding["assignment_sha256"] = fold_manifest[
        "assignment_sha256"
    ]
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_core_status": "frozen",
        "exact_hf_resolution_status": "pending_g0b",
        "created_at": created_at,
        "tool_version": TOOL_VERSION,
        "registration_chronology": {
            "phase_created": "post_p0_pre_g0b",
            "preregistered_for_completed_proxy_run": False,
            "prospective_for_exact_hf_replay": True,
            "prospective_for_pnt_and_repair": True,
            "completed_proxy_run_id": completed_proxy_run_id,
            "completed_proxy_run_used_for_policy_selection": False,
            "historical_timing_limitation": (
                "This supplementary manifest was materialized after P0 and must not "
                "be described as a pre-P0 preregistration."
            ),
        },
        "upstream_bindings": {
            "static_freeze_manifest": artifact_binding(
                static_manifest_path,
                project_root,
                schema_version=str(static_manifest["schema_version"]),
            ),
            "split_freeze_manifest": artifact_binding(
                split_manifest_path,
                project_root,
                schema_version=str(split_manifest["schema_version"]),
            ),
            "frozen_static_bundle": artifact_binding(
                bundle_path,
                project_root,
                schema_version="factual-perturbation-input-bundle-v1",
                record_count=len(all_rows),
                base_fact_ids=[str(row["base_fact_id"]) for row in all_rows],
            ),
            "development_fold_manifest": fold_manifest_binding,
            "authoritative_protocol_document": artifact_binding(
                protocol_document_path, project_root
            ),
        },
        "generator_code": artifact_binding(Path(__file__).resolve(), project_root),
        "proxy_output_independence": {
            "generator_input_allowlist": [
                project_relative(static_manifest_path, project_root),
                project_relative(split_manifest_path, project_root),
                project_relative(bundle_path, project_root),
                project_relative(fold_manifest_artifact_path, project_root),
                project_relative(protocol_document_path, project_root),
            ],
            "proxy_behavior_outputs_read_by_generator": False,
            "proxy_labels_used": False,
            "proxy_counts_used": False,
            "proxy_based_selection": False,
            "proxy_negative_records_excluded": False,
            "validation_sealed_proxy_exposed": False,
            "prohibited_inputs": [
                "runs/**/simulation_results.jsonl",
                "runs/**/preholdout_summary.json",
                "runs/**/three_arm_analysis.json",
                "runs/**/experiment_report.md",
                "runs/**/proxy_representative_probe_*",
            ],
        },
        "cohort_contract": {
            "statistical_unit": "base_fact_id",
            "split_fact_counts": dict(EXPECTED_SPLIT_COUNTS),
            "development_replay_population": "all_96_without_proxy_filtering",
            "development_base_fact_ids_sha256": sha256_value(development_ids),
            "validation_behavior_locked_until": "G2_REPAIR_RULE_FROZEN",
            "sealed_behavior_locked_until": "G3_VALIDATION_CONFIRMED",
        },
        "static_stimulus_contract": {
            "manipulation_family": "explicit_false_assertion_v1",
            "designated_distractors_per_fact": 2,
            "languages": ["en", "zh"],
            "arms": ["original", "neutral", "targeted"],
            "original_shared_across_distractor_variants": True,
            "unique_inputs_per_fact": 10,
            "target_variants_per_fact": 2,
            "both_variants_retained": True,
            "variant_is_nested_measurement_not_independent_fact": True,
            "panel_vote_or_union_label_forbidden": True,
        },
        "exact_hf_replay_contract": {
            "exact_hf_identity_required": True,
            "exact_hf_identity": None,
            "development_fact_count": 96,
            "expected_unique_input_count": 960,
            "mandatory_base_fact_ids_sha256": sha256_value(development_ids),
            "proxy_positive_filter_allowed": False,
            "proxy_labels_role": "post_hoc_migration_comparison_only",
            "render_all_160_before_opening_development": True,
            "expected_render_count_all_splits": 1600,
        },
        "label_contract": {
            "hf_directed_candidate": {
                "all_of": [
                    "zh_original_is_gold",
                    "zh_neutral_for_designated_distractor_is_gold",
                    "zh_targeted_is_designated_distractor",
                ]
            },
            "hf_zh_specific_strict": {
                "requires": "hf_directed_candidate",
                "all_of": [
                    "en_original_is_gold",
                    "en_neutral_for_designated_distractor_is_gold",
                    "en_targeted_is_gold",
                ],
            },
            "hf_resistant_control": {
                "all_of": [
                    "en_original_is_gold",
                    "en_neutral_for_designated_distractor_is_gold",
                    "en_targeted_is_gold",
                    "zh_original_is_gold",
                    "zh_neutral_for_designated_distractor_is_gold",
                    "zh_targeted_is_gold",
                ]
            },
            "other_states": {
                "baseline_failure": "required_original_is_not_gold",
                "neutral_unstable": "original_is_gold_and_neutral_is_not_gold",
                "off_target_failure": (
                    "targeted_is_wrong_and_not_designated_distractor"
                ),
                "shared_language_susceptibility": (
                    "en_targeted_is_designated_distractor"
                ),
                "incomplete": "any_required_arm_is_missing_or_nonterminal",
                "weak_continuous_signal": (
                    "no_flip_but_gold_minus_distractor_margin_declines"
                ),
            },
            "invariants": [
                "hf_zh_specific_strict_implies_hf_directed_candidate",
                "neutral_unstable_is_not_hf_directed_candidate",
                "off_target_failure_is_not_hf_directed_candidate",
                "proxy_label_is_not_exact_hf_label",
            ],
        },
        "terminal_and_retry_policy": {
            "policy_core_status": "frozen",
            "retry_allowed_only_for": [
                "infrastructure_failure",
                "transport_failure",
                "predeclared_parser_failure",
            ],
            "retry_for_wrong_answer_or_non_flip": False,
            "retry_for_off_target_answer": False,
            "prompt_or_decoding_change_during_retry": False,
            "missing_arm_imputation_allowed": False,
            "missing_arm_label": "incomplete",
            "terminal_states": ["completed", "terminal_failure"],
            "exact_max_attempts": None,
            "exact_max_attempts_status": "pending_g0b_runtime_resolution",
        },
        "matching_policy": {
            "construction_timing": (
                "after_exact_hf_labels_before_hidden_state_or_repair_results"
            ),
            "control_label": "hf_resistant_control",
            "hard_conditions": [
                "same_split",
                "same_exact_hf_identity",
                "same_probe_relation_id",
                "same_answer_type",
                "different_leakage_component_id",
            ],
            "preferred_covariates": [
                "answer_token_length",
                "prompt_tier_or_template",
                "original_gold_margin",
                "neutral_gold_margin",
                "prompt_length",
                "context_length",
                "distractor_type",
                "distractor_length",
                "review_quality",
            ],
            "default_control_population": "retain_all_eligible_controls",
            "base_fact_weighting": "equal",
            "within_fact_variant_aggregation": "mean_over_two_variants",
            "unmatched_directed_records_retained_in_funnel": True,
            "exact_distance_or_adjustment": None,
            "exact_distance_or_adjustment_status": "pending_g0b_preregistration",
        },
        "natural_baseline_contract": {
            "required_for_all_development_facts": True,
            "languages": ["en", "zh"],
            "mcq_original_is_not_a_substitute": True,
            "required_metrics": [
                "alias_aware_generation_correctness",
                "complete_answer_sequence_log_probability",
                "answer_token_rank",
                "gold_minus_distractor_log_probability_margin",
            ],
            "labels": [
                "natural_pnt_gap",
                "induced_only_gap",
                "natural_en_failure",
            ],
        },
        "crossfit_contract": {
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "fold_assignment_sha256": fold_manifest["assignment_sha256"],
            "fold_count": 4,
            "train_fact_count_per_fold": 72,
            "held_out_fact_count_per_fold": 24,
            "vector_source": "train_complement_only",
            "held_out_fact_may_source_its_injected_vector": False,
            "validation_or_sealed_may_source_vector": False,
            "final_vector_rebuilt_from_all_development_after_oof_selection": True,
        },
        "vector_and_intervention_contract": {
            "hook_family": "resid_pre",
            "vector_types": ["task_only", "difference_only", "combined"],
            "comparison_conditions": [
                "no_intervention",
                "norm_matched_random",
                "task_only",
                "difference_only",
                "combined",
            ],
            "required_retention_checks": [
                "original",
                "neutral",
                "english",
                "hf_resistant_control",
                "unrelated_facts",
            ],
            "primary_attribution_contrast": (
                "[Targeted-Neutral]_hf_directed-"
                "[Targeted-Neutral]_hf_resistant"
            ),
            "resolved_layer_ids": None,
            "layer_resolver_status": "pending_exact_hf_architecture",
            "resolved_hidden_state_position": None,
            "hidden_state_position_status": "pending_exact_hf_tokenization",
            "scale_grid": None,
            "scale_grid_status": "pending_g0b_preregistration",
        },
        "selection_rule": {
            "ordered_objectives": [
                "satisfy_all_retention_and_control_harm_constraints",
                "maximize_zh_targeted_paired_margin_improvement_on_hf_directed",
                "outperform_no_intervention_and_norm_matched_random",
                "minimize_absolute_scale",
                "use_frozen_layer_order",
                "use_frozen_vector_type_order",
            ],
            "vector_type_tie_break_order": [
                "task_only",
                "difference_only",
                "combined",
            ],
            "retention_thresholds": None,
            "control_harm_threshold": None,
            "threshold_status": "pending_g0b_preregistration",
        },
        "validation_contract": {
            "allowed_confirmation_modes": [
                "induced_confirmation",
                "natural_only_confirmation",
            ],
            "recommended_mode": "induced_confirmation",
            "selected_mode": None,
            "selection_status": "pending_before_g2_freeze",
            "validation_fact_count": 32,
            "sealed_fact_count": 32,
            "minimum_directed_event_count": None,
            "minimum_event_count_status": "pending_before_validation_open",
            "below_minimum_action": "insufficient_evidence",
            "retuning_after_validation_allowed": False,
            "sealed_open_once": True,
        },
        "reporting_contract": {
            "independent_sample_unit": "base_fact_id",
            "report_variant_and_fact_denominators": True,
            "report_n_over_N_and_95_percent_interval": True,
            "report_failures_retries_missing_and_exclusions": True,
            "relation_strata_default": "descriptive_when_small",
            "panel_or_model_union_label_emitted": False,
        },
        "invalidation_policy": {
            "static_content_change": "new_g0a_version_and_full_p0_rerun",
            "fold_or_protocol_core_change": "new_protocol_version_no_result_mixing",
            "exact_hf_identity_change": "new_g0b_and_downstream_run",
            "rule_change_after_validation": (
                "validation_downgrades_to_development_and_new_holdouts_required"
            ),
        },
        "pending_exact_hf_bindings": [
            "exact_hf_model_manifest",
            "exact_hf_render_manifest",
            "answer_tokenization_policy",
            "exact_max_attempts",
            "exact_matching_distance_or_adjustment",
            "resolved_layer_ids",
            "resolved_hidden_state_position",
            "scale_grid",
            "retention_thresholds",
            "control_harm_threshold",
            "validation_confirmation_mode",
            "minimum_directed_event_count",
        ],
        "authorization_state": {
            "g0a_static_content_frozen": True,
            "protocol_core_frozen": True,
            "development_folds_frozen": True,
            "exact_hf_identity_bound": False,
            "exact_hf_render_frozen": False,
            "exact_hf_behavior_authorized": False,
            "pnt_authorized": False,
            "validation_authorized": False,
            "sealed_authorized": False,
        },
    }


def generate(
    *,
    project_root: Path,
    static_manifest_path: Path,
    protocol_document_path: Path,
    output_dir: Path,
    created_at: str,
    completed_proxy_run_id: str,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    if not protocol_document_path.resolve().is_file():
        raise FileNotFoundError(protocol_document_path)
    validate_created_at(created_at)
    if not isinstance(completed_proxy_run_id, str) or not completed_proxy_run_id.strip():
        raise ValueError("completed_proxy_run_id must be non-empty")
    (
        static_manifest,
        bundle_path,
        split_manifest,
        split_manifest_path,
        all_rows,
    ) = validate_and_load_g0a(project_root, static_manifest_path)

    parent = output_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent))
    try:
        fold_path = temporary / "development_fold_manifest.json"
        fold_manifest = build_fold_manifest(
            project_root=project_root,
            static_manifest_path=static_manifest_path,
            static_manifest=static_manifest,
            bundle_path=bundle_path,
            split_manifest_path=split_manifest_path,
            split_manifest=split_manifest,
            all_rows=all_rows,
            created_at=created_at,
        )
        fold_path.write_bytes(pretty_json_bytes(fold_manifest))
        final_fold_path = output_dir.resolve() / fold_path.name
        protocol_manifest = build_protocol_manifest(
            project_root=project_root,
            static_manifest_path=static_manifest_path,
            static_manifest=static_manifest,
            bundle_path=bundle_path,
            split_manifest_path=split_manifest_path,
            split_manifest=split_manifest,
            all_rows=all_rows,
            fold_manifest_path=fold_path,
            fold_manifest_artifact_path=final_fold_path,
            fold_manifest=fold_manifest,
            protocol_document_path=protocol_document_path.resolve(),
            created_at=created_at,
            completed_proxy_run_id=completed_proxy_run_id,
        )
        protocol_path = temporary / "perturbation_protocol_manifest.json"
        protocol_path.write_bytes(pretty_json_bytes(protocol_manifest))
        os.replace(temporary, output_dir.resolve())
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": project_relative(output_dir, project_root),
        "development_fold_manifest": {
            "sha256": sha256_file(output_dir / "development_fold_manifest.json"),
            "assignment_sha256": fold_manifest["assignment_sha256"],
            "assigned_fact_count": len(fold_manifest["assignments"]),
        },
        "perturbation_protocol_manifest": {
            "sha256": sha256_file(output_dir / "perturbation_protocol_manifest.json"),
            "protocol_core_status": protocol_manifest["protocol_core_status"],
            "exact_hf_resolution_status": protocol_manifest[
                "exact_hf_resolution_status"
            ],
        },
    }


def verify_existing(
    *,
    project_root: Path,
    static_manifest_path: Path,
    protocol_document_path: Path,
    output_dir: Path,
    expected_created_at: str | None = None,
    expected_completed_proxy_run_id: str | None = None,
) -> Dict[str, Any]:
    fold_path = output_dir / "development_fold_manifest.json"
    protocol_path = output_dir / "perturbation_protocol_manifest.json"
    fold_manifest = read_json(fold_path)
    protocol_manifest = read_json(protocol_path)
    if fold_manifest.get("schema_version") != FOLD_SCHEMA_VERSION:
        raise ValueError("unexpected development fold manifest schema")
    if protocol_manifest.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unexpected perturbation protocol manifest schema")
    fold_created_at = fold_manifest.get("created_at")
    protocol_created_at = protocol_manifest.get("created_at")
    if fold_created_at != protocol_created_at:
        raise ValueError("fold/protocol created_at mismatch")
    validate_created_at(str(fold_created_at or ""))
    if expected_created_at is not None and fold_created_at != expected_created_at:
        raise ValueError("manifest created_at does not match the expected value")
    completed_proxy_run_id = protocol_manifest.get(
        "registration_chronology", {}
    ).get("completed_proxy_run_id")
    if (
        expected_completed_proxy_run_id is not None
        and completed_proxy_run_id != expected_completed_proxy_run_id
    ):
        raise ValueError("completed proxy run ID does not match the expected value")
    (
        static_manifest,
        bundle_path,
        split_manifest,
        split_manifest_path,
        all_rows,
    ) = validate_and_load_g0a(project_root, static_manifest_path)
    rebuilt_fold = build_fold_manifest(
        project_root=project_root,
        static_manifest_path=static_manifest_path,
        static_manifest=static_manifest,
        bundle_path=bundle_path,
        split_manifest_path=split_manifest_path,
        split_manifest=split_manifest,
        all_rows=all_rows,
        created_at=str(fold_created_at),
    )
    if fold_manifest != rebuilt_fold:
        raise ValueError("development fold manifest does not reproduce")
    fold_binding = protocol_manifest.get("upstream_bindings", {}).get(
        "development_fold_manifest", {}
    )
    if fold_binding.get("sha256") != sha256_file(fold_path):
        raise ValueError("protocol manifest fold SHA-256 mismatch")
    rebuilt_protocol = build_protocol_manifest(
        project_root=project_root,
        static_manifest_path=static_manifest_path,
        static_manifest=static_manifest,
        bundle_path=bundle_path,
        split_manifest_path=split_manifest_path,
        split_manifest=split_manifest,
        all_rows=all_rows,
        fold_manifest_path=fold_path,
        fold_manifest_artifact_path=fold_path,
        fold_manifest=fold_manifest,
        protocol_document_path=protocol_document_path.resolve(),
        created_at=str(protocol_created_at),
        completed_proxy_run_id=str(completed_proxy_run_id or ""),
    )
    if protocol_manifest != rebuilt_protocol:
        raise ValueError("perturbation protocol manifest does not reproduce")
    return {
        "verified": True,
        "output_dir": project_relative(output_dir, project_root),
        "development_fold_manifest_sha256": sha256_file(fold_path),
        "fold_assignment_sha256": fold_manifest["assignment_sha256"],
        "perturbation_protocol_manifest_sha256": sha256_file(protocol_path),
        "protocol_core_status": protocol_manifest["protocol_core_status"],
        "exact_hf_resolution_status": protocol_manifest[
            "exact_hf_resolution_status"
        ],
        "pnt_authorized": protocol_manifest["authorization_state"]["pnt_authorized"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    result.add_argument(
        "--static-freeze-manifest",
        type=Path,
        default=Path(
            "data_processed/factual_perturbation/"
            "static-g0a-160-dual-v8-20260916/frozen-v1/static_freeze_manifest.json"
        ),
    )
    result.add_argument(
        "--protocol-document",
        type=Path,
        default=Path("tasks/task_160_mcq_pnt_protocol.md"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_processed/factual_perturbation/"
            "static-g0a-160-dual-v8-20260916/pre-exact-hf-protocol-v1"
        ),
    )
    result.add_argument("--created-at")
    result.add_argument(
        "--completed-proxy-run-id", default=COMPLETED_PROXY_RUN_ID
    )
    result.add_argument("--verify-only", action="store_true")
    return result


def resolve_from_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def validate_production_cohort_anchor(static_manifest_path: Path) -> None:
    if sha256_file(static_manifest_path) != EXPECTED_STATIC_FREEZE_MANIFEST_SHA256:
        raise ValueError("production static freeze manifest SHA-256 mismatch")
    static_manifest = read_json(static_manifest_path)
    artifacts = static_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("production static freeze manifest is missing artifacts")
    bundle = artifacts.get("frozen_static_bundle.jsonl")
    split = artifacts.get("split_freeze_manifest.json")
    if not isinstance(bundle, dict) or not isinstance(split, dict):
        raise ValueError("production static freeze manifest is missing cohort bindings")
    if bundle.get("sha256") != EXPECTED_FROZEN_STATIC_BUNDLE_SHA256:
        raise ValueError("production frozen bundle SHA-256 mismatch")
    if split.get("sha256") != EXPECTED_SPLIT_FREEZE_MANIFEST_SHA256:
        raise ValueError("production split freeze manifest SHA-256 mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    project_root = args.project_root.resolve()
    static_manifest_path = resolve_from_root(project_root, args.static_freeze_manifest)
    protocol_document_path = resolve_from_root(project_root, args.protocol_document)
    output_dir = resolve_from_root(project_root, args.output_dir)
    validate_production_cohort_anchor(static_manifest_path)
    if args.verify_only:
        result = verify_existing(
            project_root=project_root,
            static_manifest_path=static_manifest_path,
            protocol_document_path=protocol_document_path,
            output_dir=output_dir,
            expected_created_at=args.created_at,
            expected_completed_proxy_run_id=args.completed_proxy_run_id,
        )
    else:
        result = generate(
            project_root=project_root,
            static_manifest_path=static_manifest_path,
            protocol_document_path=protocol_document_path,
            output_dir=output_dir,
            created_at=args.created_at or utc_now(),
            completed_proxy_run_id=args.completed_proxy_run_id,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
