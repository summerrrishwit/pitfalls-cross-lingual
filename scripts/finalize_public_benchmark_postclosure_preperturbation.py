#!/usr/bin/env python3
"""Finalize a provisional pre-perturbation bundle after duplicate closure.

This offline tool binds a pre-closure selection artifact to a resolved duplicate
closure, removes rows that closure excluded or deduplicated, applies the
component-level provisional split, and regenerates unverified distractor
candidates from the retained cohort only.  It never translates, perturbs,
freezes, or calls a model/API.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "public-benchmark-postclosure-preperturbation-finalizer-v3"
INPUT_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
SELECTION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-selection-manifest-v1"
)
CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-candidates-v2"
)
RESOLUTION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-resolution-v2"
)
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-provisional-split-v2"
)
COMPONENT_SCHEMA_VERSION = "public-benchmark-leakage-component-v2"
EXCLUSION_SCHEMA_VERSION = "public-benchmark-cohort-exclusion-v1"
ADJUDICATION_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-adjudication-v1"
)
DISTRACTOR_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-distractor-candidate-v1"
)
ASSIGNMENT_SCHEMA_VERSION = (
    "public-benchmark-postclosure-cohort-assignment-v1"
)
FINALIZATION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-postclosure-preperturbation-finalization-v3"
)
SUMMARY_SCHEMA_VERSION = (
    "public-benchmark-postclosure-preperturbation-summary-v2"
)

SPLITS = ("development", "validation", "sealed")
LEAKAGE_GROUPING_SOURCES = [
    "input_cohort_split_group_id",
    "adjudicated_same_fact",
    "adjudicated_same_leakage_component",
]
INPUT_SPLIT_GROUP_SOURCE_SEMANTICS = (
    "upstream_provisional_normalized_answer_group"
)
INPUT_SPLIT_GROUP_UNION_POLICY = (
    "unconditional_union_before_provisional_resplit_v1"
)
POSTCLOSURE_SPLIT_POLICY_VERSION = (
    "relation-stratified-component-greedy-sha256-v2"
)
SEMANTIC_AWARE_SPLIT_POLICY_VERSION = (
    "relation-stratified-lexical-semantic-component-greedy-sha256-v1"
)
SEMANTIC_CLOSURE_MANIFEST_SCHEMA_VERSION = (
    "static-g0a-semantic-closure-manifest-v1"
)
SEMANTIC_CANDIDATE_SCHEMA_VERSION = (
    "static-g0a-semantic-closure-candidate-v1"
)
SEMANTIC_DECISION_SCHEMA_VERSION = (
    "static-g0a-semantic-closure-decision-v1"
)
COHORT_REPAIR_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-cohort-repair-manifest-v1"
)
REPLACEMENT_SEMANTIC_REVIEW_SCHEMA_VERSION = (
    "public-benchmark-replacement-semantic-review-v1"
)
SEMANTIC_COMPONENT_SCHEMA_VERSION = (
    "public-benchmark-semantic-aware-leakage-component-v1"
)
SEMANTIC_EDGE_RESOLUTION_SCHEMA_VERSION = (
    "public-benchmark-semantic-edge-resolution-v1"
)
SEMANTIC_SPLIT_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-semantic-aware-provisional-split-v1"
)
SEMANTIC_IDENTITY_FIELDS = (
    "subject_en",
    "answer_en",
    "answer_aliases_en",
    "canonical_fact_en",
    "prompt_en",
    "probe_relation_id",
)
REMOVAL_DISPOSITIONS = frozenset(
    {"deduplicated_same_fact", "excluded_by_adjudication"}
)
DISTRACTORS_PER_FACT = 2
ZH_FIELDS = (
    "subject_zh",
    "answer_zh",
    "answer_aliases_zh",
    "canonical_fact_zh",
    "prompt_zh",
    "distractor_candidates_zh",
)
TARGET_FIELDS = (
    "subject_target",
    "answer_target",
    "prompt_target",
    "target_language",
)

SAFETY_CONTRACT = {
    "network_or_model_used": False,
    "model_inference_performed": False,
    "translation_performed": False,
    "perturbation_performed": False,
    "canonical_freeze_performed": False,
    "split_freeze_performed": False,
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


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


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
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, payload)


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
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _resolved_binding_path(binding: Mapping[str, Any], owner_path: Path) -> Path:
    declared = Path(_required_string(binding.get("path"), "binding.path"))
    if not declared.is_absolute():
        declared = owner_path.resolve().parent / declared
    return declared.resolve()


def _resolve_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_path: Optional[Path] = None,
    expected_schema: Optional[str] = None,
    jsonl: bool,
    allow_empty: bool = False,
) -> Tuple[Path, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is invalid")
    path = _resolved_binding_path(binding, owner_path)
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} binding points to the wrong artifact")
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count is stale")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema binding is invalid")
    value = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if jsonl and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record_count is stale")
    return path, value


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label} row {position}.{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = dict(row)
    return result


def _answer_terms(row: Mapping[str, Any], *, base_fact_id: str) -> List[str]:
    answer = _required_string(row.get("answer_en"), f"{base_fact_id}.answer_en")
    aliases = row.get("answer_aliases_en")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"{base_fact_id}.answer_aliases_en must be a non-empty list")
    normalized = [
        normalize_text(_required_string(value, f"{base_fact_id}.answer_aliases_en"))
        for value in aliases
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{base_fact_id}.answer_aliases_en contains duplicates")
    if normalize_text(answer) not in set(normalized):
        raise ValueError(
            f"{base_fact_id}.answer_aliases_en does not contain answer_en"
        )
    return normalized


def _validate_preperturbation_bundle(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    index = _unique_index(rows, "base_fact_id", "preperturbation bundle")
    for base_fact_id, row in index.items():
        if row.get("schema_version") != INPUT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported bundle schema: {base_fact_id}")
        if row.get("bundle_status") != "preperturbation_provisional_not_frozen":
            raise ValueError(f"Input row is not a preperturbation bundle: {base_fact_id}")
        if row.get("canonical_status") != "pending_review":
            raise ValueError(f"Input row canonical review is not pending: {base_fact_id}")
        if row.get("canonical_freeze_status") != "provisional_not_frozen":
            raise ValueError(f"Input row canonical status is not provisional: {base_fact_id}")
        if row.get("split_status") != "provisional_not_frozen":
            raise ValueError(f"Input row split status is not provisional: {base_fact_id}")
        if row.get("split_assignment") not in SPLITS:
            raise ValueError(f"Input row has invalid split: {base_fact_id}")
        _required_string(row.get("split_group_id"), f"{base_fact_id}.split_group_id")
        _required_string(
            row.get("split_policy_version"),
            f"{base_fact_id}.split_policy_version",
        )
        if row.get("human_gold") is not False:
            raise ValueError(f"Input row must have human_gold=false: {base_fact_id}")
        if row.get("perturbation_ready") is not False:
            raise ValueError(f"Input row unexpectedly permits perturbation: {base_fact_id}")
        if row.get("perturbation_status") != "not_run":
            raise ValueError(f"Input row has perturbation results: {base_fact_id}")
        if row.get("path_not_token_experiment_ready") is not False:
            raise ValueError(f"Input row unexpectedly permits PATH_not_token: {base_fact_id}")
        if row.get("sealed_evaluation_status") != "not_run":
            raise ValueError(f"Input row has sealed evaluation results: {base_fact_id}")
        experiment_status = row.get("experiment_status")
        if not isinstance(experiment_status, dict) or not experiment_status:
            raise ValueError(f"Input row experiment_status is invalid: {base_fact_id}")
        if any(value != "not_run" for value in experiment_status.values()):
            raise ValueError(f"Input row already has model experiment results: {base_fact_id}")
        admission = row.get("admission")
        if not isinstance(admission, dict):
            raise ValueError(f"Input row admission is invalid: {base_fact_id}")
        for field in (
            "behavior_screening_ready",
            "hf_bridge_ready",
            "relation_conditioned_mechanism_eligible",
            "path_not_token_experiment_ready",
        ):
            if admission.get(field) is not False:
                raise ValueError(
                    f"Input row admission unexpectedly permits {field}: "
                    f"{base_fact_id}"
                )
        if row.get("translation_status") != "pending_generation_and_review":
            raise ValueError(f"Input row has unsupported translation status: {base_fact_id}")
        for field in (*TARGET_FIELDS, *ZH_FIELDS):
            if row.get(field) is not None:
                raise ValueError(
                    f"Input row already contains translation field {field}: "
                    f"{base_fact_id}"
                )
        _required_string(row.get("probe_relation_id"), f"{base_fact_id}.probe_relation_id")
        _answer_terms(row, base_fact_id=base_fact_id)
    return index


def _validate_selection_manifest(
    selection_manifest_path: Path,
    *,
    bundle_path: Path,
    bundle_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest = read_json(selection_manifest_path)
    if manifest.get("schema_version") != SELECTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported selection manifest schema_version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Selection manifest artifacts are missing")
    _resolve_binding(
        artifacts.get("preperturbation_behavior_input_bundle"),
        owner_path=selection_manifest_path,
        label="selection preperturbation bundle",
        expected_path=bundle_path,
        expected_schema=INPUT_BUNDLE_SCHEMA_VERSION,
        jsonl=True,
    )
    if manifest.get("selected_base_fact_count") != len(bundle_rows):
        raise ValueError("Selection manifest selected_base_fact_count is stale")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict):
        raise ValueError("Selection manifest safety contract is missing")
    for field in (
        "network_or_model_used",
        "model_inference_performed",
        "translation_performed",
        "perturbation_performed",
        "canonical_freeze_performed",
        "split_freeze_performed",
        "path_not_token_experiment_performed",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"Selection manifest safety contract is invalid: {field}")
    return manifest


def _same_binding(first: Any, second: Any, label: str) -> None:
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError(f"{label} binding is missing")
    for field in ("path", "sha256", "byte_count"):
        if first.get(field) != second.get(field):
            raise ValueError(f"{label} binding mismatch: {field}")


def _expected_input_split_group_contract(
    bundle_index: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for base_fact_id, row in bundle_index.items():
        split_group_id = _required_string(
            row.get("split_group_id"), f"{base_fact_id}.split_group_id"
        )
        groups[split_group_id].append(base_fact_id)
    ordered_groups = [
        {
            "split_group_id": split_group_id,
            "ordered_cohort_base_fact_ids": members,
        }
        for split_group_id, members in sorted(groups.items())
    ]
    return dict(groups), {
        "source_field": "split_group_id",
        "source_semantics": INPUT_SPLIT_GROUP_SOURCE_SEMANTICS,
        "union_policy": INPUT_SPLIT_GROUP_UNION_POLICY,
        "group_count": len(groups),
        "multi_member_group_count": sum(
            len(members) > 1 for members in groups.values()
        ),
        "ordered_group_members_sha256": sha256_value(ordered_groups),
    }


class _DisjointSet:
    def __init__(self, identifiers: Iterable[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def find(self, identifier: str) -> str:
        parent = self.parent[identifier]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[identifier] = parent
        return parent

    def union(self, first: str, second: str) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        smaller, larger = sorted((left, right))
        self.parent[larger] = smaller


def _integer_targets(total: int) -> Dict[str, int]:
    ratios = {"development": 0.60, "validation": 0.20, "sealed": 0.20}
    raw = {split: total * ratios[split] for split in SPLITS}
    targets = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(targets.values())
    order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - targets[split]), SPLITS.index(split)),
    )
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def _assign_semantic_components(
    components: List[Dict[str, Any]],
    bundle_index: Mapping[str, Mapping[str, Any]],
    *,
    seed: str,
) -> Dict[str, Any]:
    relation_totals: Counter[str] = Counter()
    for component in components:
        counts: Counter[str] = Counter()
        for base_fact_id in component["retained_cohort_base_fact_ids"]:
            relation = _required_string(
                bundle_index[base_fact_id].get("probe_relation_id"),
                f"{base_fact_id}.probe_relation_id",
            )
            counts[relation] += 1
        component["relation_counts"] = dict(sorted(counts.items()))
        relation_totals.update(counts)

    targets = {
        relation: _integer_targets(total)
        for relation, total in sorted(relation_totals.items())
    }
    coupled = sorted(
        (
            component
            for component in components
            if len(component["retained_cohort_base_fact_ids"]) > 1
        ),
        key=lambda component: str(component["component_id"]),
    )
    singletons = [
        component
        for component in components
        if len(component["retained_cohort_base_fact_ids"]) == 1
    ]
    singleton_by_relation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for component in singletons:
        relation_counts = component["relation_counts"]
        if len(relation_counts) != 1 or next(iter(relation_counts.values())) != 1:
            raise ValueError("A singleton semantic component has invalid relation counts")
        singleton_by_relation[next(iter(relation_counts))].append(component)

    best: Optional[Tuple[int, str, Tuple[str, ...]]] = None
    best_counts: Optional[Dict[str, Dict[str, int]]] = None
    for choices in itertools.product(SPLITS, repeat=len(coupled)):
        current = {
            relation: {split: 0 for split in SPLITS} for relation in targets
        }
        coupled_cost = 0
        feasible = True
        for component, split in zip(coupled, choices):
            for relation, count in component["relation_counts"].items():
                current[relation][split] += int(count)
                if current[relation][split] > targets[relation][split]:
                    feasible = False
                    break
            if not feasible:
                break
            coupled_cost += sum(
                bundle_index[base_fact_id]["split_assignment"] != split
                for base_fact_id in component["retained_cohort_base_fact_ids"]
            )
        if not feasible:
            continue

        singleton_cost = 0
        for relation, relation_targets in targets.items():
            residual = {
                split: relation_targets[split] - current[relation][split]
                for split in SPLITS
            }
            relation_singletons = singleton_by_relation.get(relation, [])
            if min(residual.values()) < 0 or sum(residual.values()) != len(
                relation_singletons
            ):
                feasible = False
                break
            available = Counter(
                str(
                    bundle_index[
                        component["retained_cohort_base_fact_ids"][0]
                    ]["split_assignment"]
                )
                for component in relation_singletons
            )
            maximum_stays = sum(
                min(available[split], residual[split]) for split in SPLITS
            )
            singleton_cost += len(relation_singletons) - maximum_stays
        if not feasible:
            continue
        total_cost = coupled_cost + singleton_cost
        tie_break = sha256_value(
            [
                seed,
                [
                    [str(component["component_id"]), split]
                    for component, split in zip(coupled, choices)
                ],
            ]
        )
        candidate = (total_cost, tie_break, choices)
        if best is None or candidate < best:
            best = candidate
            best_counts = current
    if best is None or best_counts is None:
        raise ValueError(
            "No exact relation-stratified split exists for semantic components"
        )

    for component, split in zip(coupled, best[2]):
        component["provisional_split_assignment"] = split
    current = copy.deepcopy(best_counts)
    for relation, relation_singletons in sorted(singleton_by_relation.items()):
        need = {
            split: targets[relation][split] - current[relation][split]
            for split in SPLITS
        }
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for component in relation_singletons:
            base_fact_id = component["retained_cohort_base_fact_ids"][0]
            by_source[str(bundle_index[base_fact_id]["split_assignment"])].append(
                component
            )
        leftovers: List[Dict[str, Any]] = []
        for split in SPLITS:
            candidates = sorted(
                by_source.get(split, []),
                key=lambda component: (
                    hashlib.sha256(
                        f"{seed}\u241f{component['component_id']}\u241fstay".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    str(component["component_id"]),
                ),
            )
            stay_count = min(len(candidates), need[split])
            for component in candidates[:stay_count]:
                component["provisional_split_assignment"] = split
                current[relation][split] += 1
                need[split] -= 1
            leftovers.extend(candidates[stay_count:])
        leftovers.sort(key=lambda component: str(component["component_id"]))
        for component in leftovers:
            destinations = [split for split in SPLITS if need[split] > 0]
            if not destinations:
                raise ValueError("Semantic singleton reassignment overfilled targets")
            destination = min(
                destinations,
                key=lambda split: (
                    hashlib.sha256(
                        f"{seed}\u241f{component['component_id']}\u241f{split}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    split,
                ),
            )
            component["provisional_split_assignment"] = destination
            current[relation][destination] += 1
            need[destination] -= 1
        if any(need.values()):
            raise ValueError("Semantic singleton reassignment did not fill targets")

    total_abs_deviation = sum(
        abs(current[relation][split] - targets[relation][split])
        for relation in targets
        for split in SPLITS
    )
    realized_change_count = sum(
        bundle_index[base_fact_id]["split_assignment"]
        != component["provisional_split_assignment"]
        for component in components
        for base_fact_id in component["retained_cohort_base_fact_ids"]
    )
    if realized_change_count != best[0]:
        raise ValueError("Semantic split minimization cost is inconsistent")
    return {
        "relation_totals": dict(sorted(relation_totals.items())),
        "target_relation_split_counts": targets,
        "actual_relation_split_counts": current,
        "exact_relation_targets_achieved": total_abs_deviation == 0,
        "total_absolute_relation_count_deviation": total_abs_deviation,
        "assignment_objective": (
            "minimize_input_split_changes_subject_to_exact_relation_targets_v1"
        ),
        "minimum_input_split_change_count": realized_change_count,
    }


def _semantic_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {field: copy.deepcopy(row.get(field)) for field in SEMANTIC_IDENTITY_FIELDS}


def _validate_semantic_rebind_chain(
    *,
    semantic_manifest_path: Path,
    cohort_repair_manifest_path: Path,
    replacement_semantic_review_path: Path,
    current_bundle_path: Path,
    current_bundle_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    semantic_manifest = read_json(semantic_manifest_path)
    if (
        semantic_manifest.get("schema_version")
        != SEMANTIC_CLOSURE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported semantic closure manifest")
    for field in (
        "behavior_blind",
        "candidate_generation_complete",
        "candidate_adjudication_complete",
        "cohort_scope_complete",
        "out_of_cohort_bridge_search_complete",
        "out_of_cohort_bridge_adjudication_complete",
        "source_revision_binding_complete",
    ):
        if semantic_manifest.get(field) is not True:
            raise ValueError(f"Semantic closure manifest is incomplete: {field}")
    if semantic_manifest.get("human_gold") is not False:
        raise ValueError("Semantic closure manifest must set human_gold=false")
    if semantic_manifest.get("unresolved_candidate_count") != 0:
        raise ValueError("Semantic closure manifest has unresolved candidates")

    source_bundle_binding = semantic_manifest.get("input_bundle")
    if not isinstance(source_bundle_binding, dict):
        raise ValueError("Semantic source bundle binding is missing")
    source_bundle_path = _resolved_binding_path(
        source_bundle_binding, semantic_manifest_path
    )
    if (
        source_bundle_binding.get("schema_version")
        != INPUT_BUNDLE_SCHEMA_VERSION
        or source_bundle_binding.get("sha256") != sha256_file(source_bundle_path)
    ):
        raise ValueError("Semantic source bundle binding is stale")
    source_bundle_rows = read_jsonl(source_bundle_path)
    if source_bundle_binding.get("record_count") != len(source_bundle_rows):
        raise ValueError("Semantic source bundle record count is stale")
    if source_bundle_binding.get("base_fact_ids_sha256") != sha256_value(
        sorted(str(row.get("base_fact_id")) for row in source_bundle_rows)
    ):
        raise ValueError("Semantic source bundle identity order is stale")
    candidate_path, candidate_rows = _resolve_binding(
        semantic_manifest.get("candidates"),
        owner_path=semantic_manifest_path,
        label="semantic candidates",
        expected_schema=SEMANTIC_CANDIDATE_SCHEMA_VERSION,
        jsonl=True,
    )
    adjudication_path, adjudication_rows = _resolve_binding(
        semantic_manifest.get("adjudications"),
        owner_path=semantic_manifest_path,
        label="semantic adjudications",
        expected_schema=SEMANTIC_DECISION_SCHEMA_VERSION,
        jsonl=True,
    )
    source_bundle_index = _unique_index(
        source_bundle_rows, "base_fact_id", "semantic source bundle"
    )
    candidate_index = _unique_index(
        candidate_rows, "candidate_id", "semantic candidates"
    )
    adjudication_index = _unique_index(
        adjudication_rows, "candidate_id", "semantic adjudications"
    )
    if set(candidate_index) != set(adjudication_index):
        raise ValueError("Semantic adjudications do not exactly cover candidates")

    decision_counts: Counter[str] = Counter()
    for candidate_id, candidate in candidate_index.items():
        if candidate.get("schema_version") != SEMANTIC_CANDIDATE_SCHEMA_VERSION:
            raise ValueError(f"Invalid semantic candidate schema: {candidate_id}")
        if candidate.get("behavior_blind") is not True:
            raise ValueError(f"Semantic candidate is not behavior blind: {candidate_id}")
        left_id = _required_string(
            candidate.get("left_base_fact_id"), f"{candidate_id}.left_base_fact_id"
        )
        right_id = _required_string(
            candidate.get("right_base_fact_id"), f"{candidate_id}.right_base_fact_id"
        )
        if left_id == right_id or left_id not in source_bundle_index or right_id not in source_bundle_index:
            raise ValueError(f"Semantic candidate has invalid endpoints: {candidate_id}")
        decision = adjudication_index[candidate_id]
        if decision.get("schema_version") != SEMANTIC_DECISION_SCHEMA_VERSION:
            raise ValueError(f"Invalid semantic decision schema: {candidate_id}")
        if decision.get("candidate_record_sha256") != sha256_value(candidate):
            raise ValueError(f"Stale semantic candidate hash: {candidate_id}")
        if not all(
            (
                decision.get("behavior_blind") is True,
                decision.get("reviewer_type") == "codex_proxy",
                decision.get("human_gold") is False,
                decision.get("terminal_status") == "completed",
            )
        ):
            raise ValueError(f"Invalid semantic decision provenance: {candidate_id}")
        disposition = decision.get("decision")
        if disposition not in {"distinct", "same_fact", "same_leakage_component"}:
            raise ValueError(f"Invalid semantic decision: {candidate_id}")
        decision_counts[str(disposition)] += 1
    if semantic_manifest.get("decision_counts") != dict(sorted(decision_counts.items())):
        raise ValueError("Semantic manifest decision counts are stale")

    repair_manifest = read_json(cohort_repair_manifest_path)
    if (
        repair_manifest.get("schema_version")
        != COHORT_REPAIR_MANIFEST_SCHEMA_VERSION
        or repair_manifest.get("status") != "cohort_repaired_not_frozen"
    ):
        raise ValueError("Unsupported cohort repair manifest")
    repair_artifacts = repair_manifest.get("artifacts")
    if not isinstance(repair_artifacts, dict):
        raise ValueError("Cohort repair artifacts are missing")
    _resolve_binding(
        repair_artifacts.get("preperturbation_behavior_input_bundle"),
        owner_path=cohort_repair_manifest_path,
        label="repaired cohort bundle",
        expected_path=current_bundle_path,
        expected_schema=INPUT_BUNDLE_SCHEMA_VERSION,
        jsonl=True,
    )
    repair = repair_manifest.get("repair")
    if not isinstance(repair, dict):
        raise ValueError("Cohort repair decision is missing")
    removed_id = _required_string(
        repair.get("removed_base_fact_id"), "repair.removed_base_fact_id"
    )
    replacement_id = _required_string(
        repair.get("replacement_base_fact_id"), "repair.replacement_base_fact_id"
    )
    source_ids = set(source_bundle_index)
    current_ids = set(current_bundle_index)
    if source_ids - current_ids != {removed_id} or current_ids - source_ids != {
        replacement_id
    }:
        raise ValueError("Semantic source/current cohort delta does not match repair")
    for base_fact_id in sorted(source_ids & current_ids):
        if _semantic_identity(source_bundle_index[base_fact_id]) != _semantic_identity(
            current_bundle_index[base_fact_id]
        ):
            raise ValueError(
                f"Semantic identity changed for retained row: {base_fact_id}"
            )

    replacement_review = read_json(replacement_semantic_review_path)
    replacement_row = current_bundle_index[replacement_id]
    comparison_ids = [
        base_fact_id for base_fact_id in current_bundle_index if base_fact_id != replacement_id
    ]
    if not all(
        (
            replacement_review.get("schema_version")
            == REPLACEMENT_SEMANTIC_REVIEW_SCHEMA_VERSION,
            replacement_review.get("base_fact_id") == replacement_id,
            replacement_review.get("behavior_blind") is True,
            replacement_review.get("reviewer_type") == "codex_proxy",
            replacement_review.get("human_gold") is False,
            replacement_review.get("terminal_status") == "completed",
            replacement_review.get("decision")
            == "no_same_fact_or_leakage_component_candidate_identified",
            replacement_review.get("comparison_base_fact_count")
            == len(comparison_ids),
            replacement_review.get("comparison_base_fact_ids_sha256")
            == sha256_value(comparison_ids),
            replacement_review.get("replacement_record_sha256")
            == sha256_value(replacement_row),
            replacement_review.get("replacement_semantic_identity_sha256")
            == sha256_value(_semantic_identity(replacement_row)),
        )
    ):
        raise ValueError("Replacement semantic review is stale or incomplete")
    for field in ("review_method", "review_scope", "reviewed_at", "reviewer_id", "rationale"):
        _required_string(replacement_review.get(field), f"replacement_review.{field}")

    active_edges: List[Dict[str, Any]] = []
    retired_edges: List[Dict[str, Any]] = []
    source_cross_split_positive_count = 0
    for candidate_id in candidate_index:
        candidate = candidate_index[candidate_id]
        decision = adjudication_index[candidate_id]
        disposition = str(decision["decision"])
        if disposition not in {"same_fact", "same_leakage_component"}:
            continue
        left_id = str(candidate["left_base_fact_id"])
        right_id = str(candidate["right_base_fact_id"])
        source_cross_split = (
            candidate.get("left_split_assignment")
            != candidate.get("right_split_assignment")
        )
        source_cross_split_positive_count += int(source_cross_split)
        edge = {
            "edge_id": candidate_id,
            "decision": disposition,
            "left_base_fact_id": left_id,
            "right_base_fact_id": right_id,
            "left_source_split_assignment": candidate.get("left_split_assignment"),
            "right_source_split_assignment": candidate.get("right_split_assignment"),
            "source_cross_split": source_cross_split,
            "rationale": decision.get("rationale"),
            "source_candidate_record_sha256": sha256_value(candidate),
            "source_decision_record_sha256": sha256_value(decision),
            "behavior_blind": True,
            "reviewer_type": "codex_proxy",
            "human_gold": False,
        }
        endpoints = {left_id, right_id}
        if endpoints <= current_ids:
            if disposition == "same_fact":
                raise ValueError(
                    f"Repaired cohort still contains semantic same_fact edge: {candidate_id}"
                )
            edge["rebind_disposition"] = "active_semantic_union_edge"
            active_edges.append(edge)
        elif removed_id in endpoints and len(endpoints & current_ids) == 1:
            edge["rebind_disposition"] = "retired_by_cohort_repair"
            edge["retired_base_fact_id"] = removed_id
            edge["replacement_base_fact_id"] = replacement_id
            retired_edges.append(edge)
        else:
            raise ValueError(
                f"Positive semantic edge cannot be rebound after repair: {candidate_id}"
            )
    if semantic_manifest.get("cross_split_positive_count") != source_cross_split_positive_count:
        raise ValueError("Semantic manifest cross-split positive count is stale")

    return {
        "semantic_manifest": semantic_manifest,
        "semantic_manifest_path": semantic_manifest_path,
        "source_bundle_path": source_bundle_path,
        "candidate_path": candidate_path,
        "adjudication_path": adjudication_path,
        "cohort_repair_manifest_path": cohort_repair_manifest_path,
        "replacement_semantic_review_path": replacement_semantic_review_path,
        "active_edges": active_edges,
        "retired_edges": retired_edges,
        "source_positive_edge_count": len(active_edges) + len(retired_edges),
        "source_cross_split_positive_count": source_cross_split_positive_count,
        "removed_base_fact_id": removed_id,
        "replacement_base_fact_id": replacement_id,
    }


def _validate_resolution_chain(
    resolution_manifest_path: Path,
    *,
    bundle_path: Path,
    bundle_index: Mapping[str, Mapping[str, Any]],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Dict[str, Any]],
]:
    resolution = read_json(resolution_manifest_path)
    if resolution.get("schema_version") != RESOLUTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported closure resolution manifest")
    if resolution.get("status") != "bounded_lexical_closure_resolved_non_frozen":
        raise ValueError("Closure resolution is incomplete")
    for field in (
        "formal_bundle_emitted",
        "review_freeze_emitted",
        "split_freeze_emitted",
        "perturbation_authorized",
    ):
        if resolution.get(field) is not False:
            raise ValueError(f"Closure resolution safety contract is invalid: {field}")
    if resolution.get("unresolved_candidate_count") != 0:
        raise ValueError("Closure resolution still has unresolved candidates")

    candidate_path, candidate = _resolve_binding(
        resolution.get("candidate_manifest"),
        owner_path=resolution_manifest_path,
        label="closure candidate manifest",
        jsonl=False,
    )
    if candidate.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported closure candidate manifest")
    if candidate.get("status") != "conservative_candidate_closure_complete":
        raise ValueError("Closure candidate manifest is incomplete")
    candidate_inputs = candidate.get("inputs")
    if not isinstance(candidate_inputs, dict):
        raise ValueError("Closure candidate manifest inputs are missing")
    _resolve_binding(
        candidate_inputs.get("cohort_bundle"),
        owner_path=candidate_path,
        label="closure cohort bundle",
        expected_path=bundle_path,
        expected_schema=INPUT_BUNDLE_SCHEMA_VERSION,
        jsonl=True,
    )
    cohort_metadata = candidate.get("cohort")
    if not isinstance(cohort_metadata, dict):
        raise ValueError("Closure candidate cohort metadata is missing")
    ordered_input_ids = list(bundle_index)
    if cohort_metadata.get("base_fact_count") != len(ordered_input_ids):
        raise ValueError("Closure candidate cohort count is stale")
    if cohort_metadata.get("ordered_base_fact_ids_sha256") != sha256_value(
        ordered_input_ids
    ):
        raise ValueError("Closure candidate cohort identity order is stale")
    if cohort_metadata.get("ordered_row_hashes_sha256") != sha256_value(
        [sha256_value(bundle_index[base_fact_id]) for base_fact_id in ordered_input_ids]
    ):
        raise ValueError("Closure candidate cohort row hashes are stale")
    candidate_safety = candidate.get("safety_contract")
    if not isinstance(candidate_safety, dict) or any(
        candidate_safety.get(field) is not False
        for field in (
            "network_or_model_used",
            "review_freeze_emitted",
            "split_freeze_emitted",
            "perturbation_authorized",
            "human_gold",
        )
    ):
        raise ValueError("Closure candidate safety contract is invalid")

    outputs = resolution.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Closure resolution outputs are missing")
    components_path, components = _resolve_binding(
        outputs.get("components"),
        owner_path=resolution_manifest_path,
        label="closure components",
        expected_schema=COMPONENT_SCHEMA_VERSION,
        jsonl=True,
        allow_empty=True,
    )
    exclusions_path, exclusions = _resolve_binding(
        outputs.get("dedup_exclusions"),
        owner_path=resolution_manifest_path,
        label="closure exclusions",
        expected_schema=EXCLUSION_SCHEMA_VERSION,
        jsonl=True,
        allow_empty=True,
    )
    split_path, split_manifest = _resolve_binding(
        outputs.get("provisional_split_manifest"),
        owner_path=resolution_manifest_path,
        label="closure split manifest",
        expected_schema=SPLIT_MANIFEST_SCHEMA_VERSION,
        jsonl=False,
    )
    _, adjudications = _resolve_binding(
        resolution.get("adjudications"),
        owner_path=resolution_manifest_path,
        label="closure adjudications",
        expected_schema=ADJUDICATION_SCHEMA_VERSION,
        jsonl=True,
        allow_empty=True,
    )
    adjudication_index = _unique_index(
        adjudications, "pair_id", "closure adjudications"
    )
    for pair_id, row in adjudication_index.items():
        if row.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
            raise ValueError(f"Invalid closure adjudication schema: {pair_id}")
    if split_manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported closure split manifest")
    if split_manifest.get("status") != "provisional_recomputed_not_frozen":
        raise ValueError("Closure split manifest is incomplete")
    if split_manifest.get("split_status") != "provisional_not_frozen":
        raise ValueError("Closure split is not provisional")
    split_policy_version = split_manifest.get("split_policy_version")
    if not isinstance(split_policy_version, str) or not split_policy_version.strip():
        raise ValueError(
            "Closure split_policy_version must be a non-empty string"
        )
    if split_policy_version != POSTCLOSURE_SPLIT_POLICY_VERSION:
        raise ValueError("Closure split policy is unsupported")
    if split_manifest.get("unresolved_candidate_count") != 0:
        raise ValueError("Closure split still has unresolved candidates")
    split_safety = split_manifest.get("safety_contract")
    if not isinstance(split_safety, dict) or not all(
        (
            split_safety.get("component_atomicity_enforced") is True,
            split_safety.get("input_split_group_atomicity_enforced") is True,
            split_safety.get("copied_provisional_split_assignments") is False,
            split_safety.get("network_or_model_used") is False,
            split_safety.get("review_freeze_emitted") is False,
            split_safety.get("split_freeze_emitted") is False,
            split_safety.get("perturbation_authorized") is False,
            split_safety.get("human_gold") is False,
        )
    ):
        raise ValueError("Closure split safety contract is invalid")
    input_split_groups, expected_input_group_contract = (
        _expected_input_split_group_contract(bundle_index)
    )
    grouping_contract = split_manifest.get("leakage_grouping_contract")
    if not isinstance(grouping_contract, dict) or set(grouping_contract) != {
        "grouping_sources",
        "input_split_group",
        "same_fact_implies_same_leakage_component",
    }:
        raise ValueError("Closure leakage grouping contract is invalid")
    if grouping_contract.get("grouping_sources") != LEAKAGE_GROUPING_SOURCES:
        raise ValueError("Closure leakage grouping sources are unsupported")
    if grouping_contract.get("input_split_group") != expected_input_group_contract:
        raise ValueError("Closure input split-group contract is stale")
    if grouping_contract.get("same_fact_implies_same_leakage_component") is not True:
        raise ValueError("Closure same_fact leakage grouping contract is invalid")
    if resolution.get("leakage_grouping_contract") != grouping_contract:
        raise ValueError("Resolution/split leakage grouping contracts disagree")
    _same_binding(
        split_manifest.get("candidate_manifest"),
        resolution.get("candidate_manifest"),
        "candidate manifest",
    )
    _same_binding(
        split_manifest.get("adjudications"),
        resolution.get("adjudications"),
        "closure adjudications",
    )
    _same_binding(split_manifest.get("components"), outputs.get("components"), "components")
    _same_binding(
        split_manifest.get("dedup_exclusions"),
        outputs.get("dedup_exclusions"),
        "dedup exclusions",
    )

    exclusion_index = _unique_index(exclusions, "base_fact_id", "closure exclusions")
    removed: Dict[str, Dict[str, Any]] = {}
    for base_fact_id, row in exclusion_index.items():
        if row.get("schema_version") != EXCLUSION_SCHEMA_VERSION:
            raise ValueError(f"Invalid exclusion schema: {base_fact_id}")
        if base_fact_id not in bundle_index:
            raise ValueError(f"Closure exclusion is outside the input cohort: {base_fact_id}")
        disposition = row.get("disposition")
        if disposition not in REMOVAL_DISPOSITIONS:
            raise ValueError(f"Unsupported exclusion disposition: {base_fact_id}")
        representative = row.get("representative_base_fact_id")
        if disposition == "deduplicated_same_fact":
            if not isinstance(representative, str) or representative not in bundle_index:
                raise ValueError(f"Dedup exclusion has invalid representative: {base_fact_id}")
            if representative == base_fact_id:
                raise ValueError(f"Dedup exclusion is self-referential: {base_fact_id}")
        elif representative is not None:
            raise ValueError(f"Explicit exclusion unexpectedly has representative: {base_fact_id}")
        reason_pairs = row.get("reason_pair_ids")
        if not isinstance(reason_pairs, list) or not reason_pairs:
            raise ValueError(f"Closure exclusion lacks reason pairs: {base_fact_id}")
        if any(pair_id not in adjudication_index for pair_id in reason_pairs):
            raise ValueError(f"Closure exclusion cites unknown reason pair: {base_fact_id}")
        reason_decisions = {
            adjudication_index[pair_id].get("decision") for pair_id in reason_pairs
        }
        if (
            disposition == "deduplicated_same_fact"
            and "same_fact" not in reason_decisions
        ):
            raise ValueError(f"Dedup exclusion lacks same_fact evidence: {base_fact_id}")
        if disposition == "excluded_by_adjudication":
            matching_exclusions = [
                adjudication_index[pair_id]
                for pair_id in reason_pairs
                if adjudication_index[pair_id].get("decision") == "exclude_cohort"
                and base_fact_id
                in adjudication_index[pair_id].get(
                    "excluded_cohort_base_fact_ids", []
                )
            ]
            if not matching_exclusions:
                raise ValueError(
                    f"Explicit exclusion lacks exclude_cohort evidence: {base_fact_id}"
                )
        removed[base_fact_id] = row

    component_index = _unique_index(components, "component_id", "closure components")
    assignment_rows = split_manifest.get("component_assignments")
    if not isinstance(assignment_rows, list):
        raise ValueError("Closure split component_assignments are missing")
    assignment_index = _unique_index(
        assignment_rows, "component_id", "split component assignments"
    )
    if set(assignment_index) != set(component_index):
        raise ValueError("Split assignments do not exactly cover closure components")

    final_split_by_id: Dict[str, str] = {}
    component_by_id: Dict[str, str] = {}
    for component_id, component in component_index.items():
        if component.get("schema_version") != COMPONENT_SCHEMA_VERSION:
            raise ValueError(f"Invalid component schema: {component_id}")
        retained = component.get("retained_cohort_base_fact_ids")
        if not isinstance(retained, list) or not retained:
            raise ValueError(f"Closure component has no retained rows: {component_id}")
        if len(retained) != len(set(retained)):
            raise ValueError(f"Closure component repeats a retained row: {component_id}")
        all_cohort = component.get("all_cohort_base_fact_ids")
        if (
            not isinstance(all_cohort, list)
            or len(all_cohort) != len(set(all_cohort))
            or any(base_fact_id not in bundle_index for base_fact_id in all_cohort)
        ):
            raise ValueError(f"Closure component has invalid cohort members: {component_id}")
        if not set(retained) <= set(all_cohort):
            raise ValueError(
                f"Closure component retained rows are absent from all_cohort: {component_id}"
            )
        expected_group_ids = sorted(
            {
                _required_string(
                    bundle_index[base_fact_id].get("split_group_id"),
                    f"{base_fact_id}.split_group_id",
                )
                for base_fact_id in all_cohort
            }
        )
        if component.get("input_split_group_ids") != expected_group_ids:
            raise ValueError(f"Component input split-group IDs are stale: {component_id}")
        expected_supporting_group_ids = [
            split_group_id
            for split_group_id in expected_group_ids
            if len(input_split_groups[split_group_id]) > 1
        ]
        if (
            component.get("supporting_input_split_group_ids")
            != expected_supporting_group_ids
        ):
            raise ValueError(
                f"Component supporting input split groups are stale: {component_id}"
            )
        assignment = assignment_index[component_id]
        split = assignment.get("split_assignment")
        if split not in SPLITS:
            raise ValueError(f"Closure component has invalid final split: {component_id}")
        if component.get("provisional_split_assignment") != split:
            raise ValueError(f"Component/split manifest assignment mismatch: {component_id}")
        if assignment.get("retained_cohort_base_fact_ids") != retained:
            raise ValueError(f"Component retained IDs changed in split manifest: {component_id}")
        relation_counts: Counter[str] = Counter()
        for base_fact_id in retained:
            if not isinstance(base_fact_id, str) or base_fact_id not in bundle_index:
                raise ValueError(f"Component retains an unknown row: {component_id}")
            if base_fact_id in removed:
                raise ValueError(f"Removed row is still retained: {base_fact_id}")
            if base_fact_id in final_split_by_id:
                raise ValueError(f"Retained row appears in multiple components: {base_fact_id}")
            relation = _required_string(
                bundle_index[base_fact_id].get("probe_relation_id"),
                f"{base_fact_id}.probe_relation_id",
            )
            relation_counts[relation] += 1
            final_split_by_id[base_fact_id] = str(split)
            component_by_id[base_fact_id] = component_id
        if component.get("relation_counts") != dict(sorted(relation_counts.items())):
            raise ValueError(f"Component relation_counts are stale: {component_id}")
        if assignment.get("relation_counts") != component.get("relation_counts"):
            raise ValueError(f"Split assignment relation_counts are stale: {component_id}")
    input_ids = set(bundle_index)
    if set(final_split_by_id) | set(removed) != input_ids:
        missing = sorted(input_ids - set(final_split_by_id) - set(removed))
        raise ValueError(f"Closure resolution does not cover the full cohort: {missing[:5]}")
    if set(final_split_by_id) & set(removed):
        raise ValueError("Closure resolution both retains and removes a cohort row")
    for split_group_id, members in input_split_groups.items():
        retained_members = [
            base_fact_id for base_fact_id in members if base_fact_id in final_split_by_id
        ]
        retained_components = {
            component_by_id[base_fact_id] for base_fact_id in retained_members
        }
        retained_splits = {
            final_split_by_id[base_fact_id] for base_fact_id in retained_members
        }
        if len(retained_components) > 1 or len(retained_splits) > 1:
            raise ValueError(
                "Input split_group_id is not atomic after closure: "
                f"{split_group_id}"
            )
    for base_fact_id, row in removed.items():
        representative = row.get("representative_base_fact_id")
        if representative is not None and representative not in final_split_by_id:
            raise ValueError(f"Dedup representative is not retained: {base_fact_id}")

    counts = split_manifest.get("cohort_counts")
    resolution_counts = resolution.get("counts")
    if not isinstance(counts, dict) or counts != resolution_counts:
        raise ValueError("Resolution/split cohort counts disagree")
    expected_counts = {
        "input": len(bundle_index),
        "retained": len(final_split_by_id),
        "deduplicated": sum(
            row["disposition"] == "deduplicated_same_fact" for row in removed.values()
        ),
        "explicitly_excluded": sum(
            row["disposition"] == "excluded_by_adjudication" for row in removed.values()
        ),
        "replacement_required_to_restore_input_size": len(removed),
    }
    if counts != expected_counts:
        raise ValueError("Closure cohort counts are stale")

    actual_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {split: 0 for split in SPLITS}
    )
    for base_fact_id, split in final_split_by_id.items():
        relation = str(bundle_index[base_fact_id]["probe_relation_id"])
        actual_counts[relation][split] += 1
    normalized_actual = {
        relation: dict(values) for relation, values in sorted(actual_counts.items())
    }
    if split_manifest.get("actual_relation_split_counts") != normalized_actual:
        raise ValueError("Closure actual_relation_split_counts are stale")
    relation_totals = {
        relation: sum(values.values()) for relation, values in normalized_actual.items()
    }
    if split_manifest.get("relation_totals") != relation_totals:
        raise ValueError("Closure relation_totals are stale")

    return (
        resolution,
        split_manifest,
        components,
        exclusions,
        final_split_by_id,
        component_by_id,
        removed,
    )


def _merge_semantic_components(
    *,
    lexical_components: Sequence[Mapping[str, Any]],
    bundle_index: Mapping[str, Mapping[str, Any]],
    lexical_split_by_id: Mapping[str, str],
    lexical_component_by_id: Mapping[str, str],
    removed: Mapping[str, Mapping[str, Any]],
    semantic_rebind: Mapping[str, Any],
    split_seed: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Any],
]:
    lexical_index = _unique_index(
        lexical_components, "component_id", "lexical closure components"
    )
    union = _DisjointSet(lexical_index)
    active_edges = list(semantic_rebind["active_edges"])
    runtime_retired_edge_ids = set()
    for edge in active_edges:
        left_id = str(edge["left_base_fact_id"])
        right_id = str(edge["right_base_fact_id"])
        if left_id in removed or right_id in removed:
            runtime_retired_edge_ids.add(str(edge["edge_id"]))
            continue
        if left_id not in lexical_component_by_id or right_id not in lexical_component_by_id:
            raise ValueError(f"Semantic edge endpoint is not retained: {edge['edge_id']}")
        union.union(
            lexical_component_by_id[left_id],
            lexical_component_by_id[right_id],
        )

    lexical_groups: Dict[str, List[str]] = defaultdict(list)
    for component_id in lexical_index:
        lexical_groups[union.find(component_id)].append(component_id)
    input_order = {base_fact_id: index for index, base_fact_id in enumerate(bundle_index)}
    semantic_components: List[Dict[str, Any]] = []
    for lexical_component_ids in lexical_groups.values():
        lexical_component_ids = sorted(lexical_component_ids)
        retained_ids = sorted(
            {
                str(base_fact_id)
                for component_id in lexical_component_ids
                for base_fact_id in lexical_index[component_id][
                    "retained_cohort_base_fact_ids"
                ]
            },
            key=lambda base_fact_id: input_order[base_fact_id],
        )
        semantic_edge_ids = sorted(
            str(edge["edge_id"])
            for edge in active_edges
            if str(edge["edge_id"]) not in runtime_retired_edge_ids
            and lexical_component_by_id[str(edge["left_base_fact_id"])]
            in lexical_component_ids
            and lexical_component_by_id[str(edge["right_base_fact_id"])]
            in lexical_component_ids
        )
        component_id = "semantic_leakage_component_" + sha256_value(
            {
                "lexical_component_ids": lexical_component_ids,
                "semantic_edge_ids": semantic_edge_ids,
            }
        )[:24]
        semantic_components.append(
            {
                "schema_version": SEMANTIC_COMPONENT_SCHEMA_VERSION,
                "component_id": component_id,
                "retained_cohort_base_fact_ids": retained_ids,
                "retained_cohort_base_fact_count": len(retained_ids),
                "lexical_component_ids": lexical_component_ids,
                "supporting_lexical_union_pair_ids": sorted(
                    {
                        str(pair_id)
                        for lexical_component_id in lexical_component_ids
                        for pair_id in lexical_index[lexical_component_id].get(
                            "supporting_union_pair_ids", []
                        )
                    }
                ),
                "supporting_semantic_edge_ids": semantic_edge_ids,
                "input_split_group_ids": sorted(
                    {
                        str(split_group_id)
                        for lexical_component_id in lexical_component_ids
                        for split_group_id in lexical_index[lexical_component_id].get(
                            "input_split_group_ids", []
                        )
                    }
                ),
                "presemantic_lexical_split_assignments": sorted(
                    {
                        lexical_split_by_id[base_fact_id]
                        for base_fact_id in retained_ids
                    }
                ),
            }
        )
    semantic_components.sort(key=lambda row: str(row["component_id"]))
    split_stats = _assign_semantic_components(
        semantic_components,
        bundle_index,
        seed=split_seed,
    )

    final_split_by_id: Dict[str, str] = {}
    final_component_by_id: Dict[str, str] = {}
    for component in semantic_components:
        component_id = str(component["component_id"])
        split = str(component["provisional_split_assignment"])
        for base_fact_id in component["retained_cohort_base_fact_ids"]:
            if base_fact_id in final_split_by_id:
                raise ValueError(
                    f"Semantic component repeats retained row: {base_fact_id}"
                )
            final_split_by_id[base_fact_id] = split
            final_component_by_id[base_fact_id] = component_id
    if set(final_split_by_id) != set(lexical_split_by_id):
        raise ValueError("Semantic components do not exactly cover retained cohort")

    edge_resolutions: List[Dict[str, Any]] = []
    for edge in active_edges:
        edge_id = str(edge["edge_id"])
        left_id = str(edge["left_base_fact_id"])
        right_id = str(edge["right_base_fact_id"])
        if edge_id in runtime_retired_edge_ids:
            status = "retired_by_lexical_resolution"
            final_component_id = None
            final_split = None
            changed_ids: List[str] = []
        else:
            if final_component_by_id[left_id] != final_component_by_id[right_id]:
                raise ValueError(f"Semantic edge was not made component-atomic: {edge_id}")
            if final_split_by_id[left_id] != final_split_by_id[right_id]:
                raise ValueError(f"Semantic edge still crosses final split: {edge_id}")
            status = "resolved_same_component_and_split"
            final_component_id = final_component_by_id[left_id]
            final_split = final_split_by_id[left_id]
            changed_ids = [
                base_fact_id
                for base_fact_id, source_split in (
                    (left_id, edge["left_source_split_assignment"]),
                    (right_id, edge["right_source_split_assignment"]),
                )
                if final_split_by_id[base_fact_id] != source_split
            ]
        edge_resolutions.append(
            {
                "schema_version": SEMANTIC_EDGE_RESOLUTION_SCHEMA_VERSION,
                **copy.deepcopy(edge),
                "resolution_status": status,
                "left_presemantic_lexical_split_assignment": (
                    lexical_split_by_id.get(left_id)
                ),
                "right_presemantic_lexical_split_assignment": (
                    lexical_split_by_id.get(right_id)
                ),
                "final_semantic_component_id": final_component_id,
                "final_split_assignment": final_split,
                "base_fact_ids_changed_from_source_split": changed_ids,
            }
        )
    for edge in semantic_rebind["retired_edges"]:
        present_ids = [
            base_fact_id
            for base_fact_id in (
                str(edge["left_base_fact_id"]),
                str(edge["right_base_fact_id"]),
            )
            if base_fact_id in final_split_by_id
        ]
        edge_resolutions.append(
            {
                "schema_version": SEMANTIC_EDGE_RESOLUTION_SCHEMA_VERSION,
                **copy.deepcopy(edge),
                "resolution_status": "retired_by_cohort_repair",
                "left_presemantic_lexical_split_assignment": (
                    lexical_split_by_id.get(str(edge["left_base_fact_id"]))
                ),
                "right_presemantic_lexical_split_assignment": (
                    lexical_split_by_id.get(str(edge["right_base_fact_id"]))
                ),
                "final_semantic_component_id": (
                    final_component_by_id[present_ids[0]] if present_ids else None
                ),
                "final_split_assignment": (
                    final_split_by_id[present_ids[0]] if present_ids else None
                ),
                "base_fact_ids_changed_from_source_split": [
                    base_fact_id
                    for base_fact_id in present_ids
                    if final_split_by_id[base_fact_id]
                    != (
                        edge["left_source_split_assignment"]
                        if base_fact_id == str(edge["left_base_fact_id"])
                        else edge["right_source_split_assignment"]
                    )
                ],
            }
        )
    edge_resolutions.sort(key=lambda row: str(row["edge_id"]))
    if sum(bool(row["source_cross_split"]) for row in edge_resolutions) != int(
        semantic_rebind["source_cross_split_positive_count"]
    ):
        raise ValueError("Semantic cross-split edge resolution coverage is incomplete")

    return (
        semantic_components,
        edge_resolutions,
        final_split_by_id,
        final_component_by_id,
        split_stats,
    )


def _make_distractors(
    retained_rows: Sequence[Mapping[str, Any]],
    *,
    input_row_hashes: Mapping[str, str],
    component_by_id: Mapping[str, str],
    resolution_sha256: str,
    semantic_partition_evidence_sha256: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    pools: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in retained_rows:
        pools[(str(row["probe_relation_id"]), str(row["split_assignment"]))].append(row)

    by_target: Dict[str, List[Dict[str, Any]]] = {}
    flattened: List[Dict[str, Any]] = []
    for target in retained_rows:
        target_id = str(target["base_fact_id"])
        relation = str(target["probe_relation_id"])
        split = str(target["split_assignment"])
        target_terms = set(_answer_terms(target, base_fact_id=target_id))
        eligible: List[Tuple[str, Mapping[str, Any]]] = []
        for donor in pools[(relation, split)]:
            donor_id = str(donor["base_fact_id"])
            if donor_id == target_id:
                continue
            donor_terms = set(_answer_terms(donor, base_fact_id=donor_id))
            if target_terms.intersection(donor_terms):
                continue
            score = sha256_value(
                [
                    TOOL_VERSION,
                    resolution_sha256,
                    semantic_partition_evidence_sha256,
                    target_id,
                    donor_id,
                    donor.get("answer_en"),
                ]
            )
            eligible.append((score, donor))
        eligible.sort(key=lambda pair: (pair[0], str(pair[1]["base_fact_id"])))
        if len(eligible) < DISTRACTORS_PER_FACT:
            raise ValueError(
                f"Fewer than {DISTRACTORS_PER_FACT} retained same-relation/final-split, "
                f"alias-disjoint distractors for {target_id}"
            )
        selected: List[Dict[str, Any]] = []
        for rank, (score, donor) in enumerate(
            eligible[:DISTRACTORS_PER_FACT], start=1
        ):
            donor_id = str(donor["base_fact_id"])
            candidate = {
                "schema_version": DISTRACTOR_SCHEMA_VERSION,
                "distractor_id": "postclosure_dist_"
                + sha256_value([target_id, donor_id, score])[:20],
                "target_base_fact_id": target_id,
                "donor_base_fact_id": donor_id,
                "probe_relation_id": relation,
                "split_assignment": split,
                "target_leakage_component_id": component_by_id[target_id],
                "donor_leakage_component_id": component_by_id[donor_id],
                "target_preperturbation_record_sha256": input_row_hashes[target_id],
                "donor_preperturbation_record_sha256": input_row_hashes[donor_id],
                "closure_resolution_manifest_sha256": resolution_sha256,
                "semantic_partition_evidence_sha256": (
                    semantic_partition_evidence_sha256
                ),
                "rank": rank,
                "text_en": donor["answer_en"],
                "answer_en": donor["answer_en"],
                "answer_aliases_en": copy.deepcopy(donor["answer_aliases_en"]),
                "selection_score_sha256": score,
                "selection_method": (
                    "deterministic_semantic_aware_same_relation_final_split_"
                    "alias_disjoint_v1"
                    if semantic_partition_evidence_sha256 is not None
                    else "deterministic_postclosure_same_relation_final_split_"
                    "alias_disjoint_v1"
                ),
                "verification_status": "codex_proxy_pending_review",
                "status": "codex_proxy_pending_review",
                "verified": False,
                "human_gold": False,
            }
            selected.append(candidate)
            flattened.append(candidate)
        by_target[target_id] = selected
    return by_target, flattened


def _validate_distractors(
    rows: Sequence[Mapping[str, Any]],
    flattened: Sequence[Mapping[str, Any]],
) -> None:
    row_index = _unique_index(rows, "base_fact_id", "final retained cohort")
    flat_index = _unique_index(flattened, "distractor_id", "final distractors")
    observed: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for distractor_id, candidate in flat_index.items():
        target_id = _required_string(
            candidate.get("target_base_fact_id"), f"{distractor_id}.target_base_fact_id"
        )
        donor_id = _required_string(
            candidate.get("donor_base_fact_id"), f"{distractor_id}.donor_base_fact_id"
        )
        if target_id not in row_index or donor_id not in row_index:
            raise ValueError(f"Distractor references a removed row: {distractor_id}")
        if target_id == donor_id:
            raise ValueError(f"Distractor self-donor: {distractor_id}")
        target = row_index[target_id]
        donor = row_index[donor_id]
        if target.get("probe_relation_id") != donor.get("probe_relation_id"):
            raise ValueError(f"Distractor crosses relation: {distractor_id}")
        if target.get("split_assignment") != donor.get("split_assignment"):
            raise ValueError(f"Distractor crosses final split: {distractor_id}")
        if candidate.get("probe_relation_id") != target.get("probe_relation_id"):
            raise ValueError(f"Distractor relation metadata is stale: {distractor_id}")
        if candidate.get("split_assignment") != target.get("split_assignment"):
            raise ValueError(f"Distractor split metadata is stale: {distractor_id}")
        if set(_answer_terms(target, base_fact_id=target_id)).intersection(
            _answer_terms(donor, base_fact_id=donor_id)
        ):
            raise ValueError(f"Distractor answer aliases overlap target: {distractor_id}")
        if candidate.get("answer_en") != donor.get("answer_en") or candidate.get(
            "answer_aliases_en"
        ) != donor.get("answer_aliases_en"):
            raise ValueError(f"Distractor donor payload is stale: {distractor_id}")
        if not all(
            (
                candidate.get("verification_status") == "codex_proxy_pending_review",
                candidate.get("status") == "codex_proxy_pending_review",
                candidate.get("verified") is False,
                candidate.get("human_gold") is False,
            )
        ):
            raise ValueError(f"Distractor is not pending/unverified: {distractor_id}")
        observed[target_id].append(candidate)

    for base_fact_id, row in row_index.items():
        if row.get("split_group_id") != row.get("leakage_component_id"):
            raise ValueError(f"Final row split group is not its closure component: {base_fact_id}")
        embedded = row.get("distractor_candidates")
        if not isinstance(embedded, list) or len(embedded) != DISTRACTORS_PER_FACT:
            raise ValueError(f"Final row has wrong distractor count: {base_fact_id}")
        if len(observed[base_fact_id]) != DISTRACTORS_PER_FACT:
            raise ValueError(f"Flattened distractor count is wrong: {base_fact_id}")
        if canonical_json_bytes(embedded) != canonical_json_bytes(observed[base_fact_id]):
            raise ValueError(f"Embedded/flattened distractors disagree: {base_fact_id}")
        donors = [str(value["donor_base_fact_id"]) for value in embedded]
        if len(donors) != len(set(donors)):
            raise ValueError(f"Final row repeats a distractor donor: {base_fact_id}")


def finalize_postclosure_preperturbation(
    *,
    preperturbation_bundle_path: Path,
    selection_manifest_path: Path,
    closure_resolution_manifest_path: Path,
    output_dir: Path,
    semantic_closure_manifest_path: Optional[Path] = None,
    cohort_repair_manifest_path: Optional[Path] = None,
    replacement_semantic_review_path: Optional[Path] = None,
) -> Dict[str, Any]:
    preperturbation_bundle_path = Path(preperturbation_bundle_path).resolve()
    selection_manifest_path = Path(selection_manifest_path).resolve()
    closure_resolution_manifest_path = Path(
        closure_resolution_manifest_path
    ).resolve()
    semantic_paths = (
        semantic_closure_manifest_path,
        cohort_repair_manifest_path,
        replacement_semantic_review_path,
    )
    if any(path is not None for path in semantic_paths) and not all(
        path is not None for path in semantic_paths
    ):
        raise ValueError(
            "semantic closure, cohort repair, and replacement semantic review "
            "must be supplied together"
        )
    if semantic_closure_manifest_path is not None:
        semantic_closure_manifest_path = Path(
            semantic_closure_manifest_path
        ).resolve()
        cohort_repair_manifest_path = Path(cohort_repair_manifest_path).resolve()
        replacement_semantic_review_path = Path(
            replacement_semantic_review_path
        ).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    input_rows = read_jsonl(preperturbation_bundle_path)
    input_index = _validate_preperturbation_bundle(input_rows)
    _validate_selection_manifest(
        selection_manifest_path,
        bundle_path=preperturbation_bundle_path,
        bundle_rows=input_rows,
    )
    (
        _,
        split_manifest,
        components,
        exclusions,
        lexical_split_by_id,
        lexical_component_by_id,
        removed,
    ) = _validate_resolution_chain(
        closure_resolution_manifest_path,
        bundle_path=preperturbation_bundle_path,
        bundle_index=input_index,
    )

    semantic_rebind: Optional[Dict[str, Any]] = None
    semantic_components: List[Dict[str, Any]] = []
    semantic_edge_resolutions: List[Dict[str, Any]] = []
    semantic_split_stats: Optional[Dict[str, Any]] = None
    semantic_partition_evidence_sha: Optional[str] = None
    if semantic_closure_manifest_path is not None:
        assert cohort_repair_manifest_path is not None
        assert replacement_semantic_review_path is not None
        semantic_rebind = _validate_semantic_rebind_chain(
            semantic_manifest_path=semantic_closure_manifest_path,
            cohort_repair_manifest_path=cohort_repair_manifest_path,
            replacement_semantic_review_path=replacement_semantic_review_path,
            current_bundle_path=preperturbation_bundle_path,
            current_bundle_index=input_index,
        )
        semantic_partition_evidence_sha = sha256_value(
            {
                "lexical_resolution_manifest_sha256": sha256_file(
                    closure_resolution_manifest_path
                ),
                "semantic_closure_manifest_sha256": sha256_file(
                    semantic_closure_manifest_path
                ),
                "cohort_repair_manifest_sha256": sha256_file(
                    cohort_repair_manifest_path
                ),
                "replacement_semantic_review_sha256": sha256_file(
                    replacement_semantic_review_path
                ),
                "active_semantic_edges": semantic_rebind["active_edges"],
            }
        )
        split_seed = _required_string(
            split_manifest.get("split_seed"), "lexical split_manifest.split_seed"
        )
        (
            semantic_components,
            semantic_edge_resolutions,
            final_split_by_id,
            component_by_id,
            semantic_split_stats,
        ) = _merge_semantic_components(
            lexical_components=components,
            bundle_index=input_index,
            lexical_split_by_id=lexical_split_by_id,
            lexical_component_by_id=lexical_component_by_id,
            removed=removed,
            semantic_rebind=semantic_rebind,
            split_seed=f"{split_seed}:semantic-aware-v1",
        )
        postclosure_split_policy_version = SEMANTIC_AWARE_SPLIT_POLICY_VERSION
    else:
        final_split_by_id = lexical_split_by_id
        component_by_id = lexical_component_by_id
        postclosure_split_policy_version = str(split_manifest["split_policy_version"])

    input_bundle_sha = sha256_file(preperturbation_bundle_path)
    selection_manifest_sha = sha256_file(selection_manifest_path)
    resolution_manifest_sha = sha256_file(closure_resolution_manifest_path)
    split_manifest_binding = read_json(closure_resolution_manifest_path)["outputs"][
        "provisional_split_manifest"
    ]
    split_manifest_path = _resolved_binding_path(
        split_manifest_binding, closure_resolution_manifest_path
    )
    split_manifest_sha = sha256_file(split_manifest_path)
    input_row_hashes = {
        base_fact_id: sha256_value(row) for base_fact_id, row in input_index.items()
    }

    retained_rows: List[Dict[str, Any]] = []
    original_split_by_id: Dict[str, str] = {}
    for raw_row in input_rows:
        base_fact_id = str(raw_row["base_fact_id"])
        if base_fact_id in removed:
            continue
        row = copy.deepcopy(raw_row)
        original_split = str(row["split_assignment"])
        original_split_group_id = _required_string(
            row.get("split_group_id"), f"{base_fact_id}.split_group_id"
        )
        original_split_policy_version = _required_string(
            row.get("split_policy_version"),
            f"{base_fact_id}.split_policy_version",
        )
        final_split = final_split_by_id[base_fact_id]
        original_split_by_id[base_fact_id] = original_split
        row["bundle_status"] = "postclosure_preperturbation_provisional_not_frozen"
        row["split_assignment"] = final_split
        row["split_group_id"] = component_by_id[base_fact_id]
        row["split_policy_version"] = postclosure_split_policy_version
        row["split_status"] = "provisional_not_frozen"
        row["leakage_component_id"] = component_by_id[base_fact_id]
        row["distractor_candidates"] = []
        row["perturbation_ready"] = False
        row["perturbation_status"] = "not_run"
        row["path_not_token_experiment_ready"] = False
        admission = copy.deepcopy(row["admission"])
        blockers = admission.get("blocking_reasons")
        if not isinstance(blockers, list) or any(
            not isinstance(value, str) or not value for value in blockers
        ):
            raise ValueError(f"Input row admission blockers are invalid: {base_fact_id}")
        if removed and "duplicate_replacement_pending" not in blockers:
            blockers.append("duplicate_replacement_pending")
        admission["blocking_reasons"] = blockers
        admission["behavior_screening_ready"] = False
        admission["hf_bridge_ready"] = False
        admission["relation_conditioned_mechanism_eligible"] = False
        admission["path_not_token_experiment_ready"] = False
        row["admission"] = admission
        row["postclosure_lineage"] = {
            "tool_version": TOOL_VERSION,
            "input_preperturbation_bundle_sha256": input_bundle_sha,
            "input_preperturbation_record_sha256": input_row_hashes[base_fact_id],
            "selection_manifest_sha256": selection_manifest_sha,
            "closure_resolution_manifest_sha256": resolution_manifest_sha,
            "provisional_split_manifest_sha256": split_manifest_sha,
            "leakage_component_id": component_by_id[base_fact_id],
            "lexical_leakage_component_id": lexical_component_by_id[base_fact_id],
            "semantic_component_union_applied": semantic_rebind is not None,
            "semantic_partition_evidence_sha256": semantic_partition_evidence_sha,
            "preclosure_split_assignment": original_split,
            "preclosure_split_group_id": original_split_group_id,
            "preclosure_split_policy_version": original_split_policy_version,
            "postclosure_split_assignment": final_split,
            "postclosure_split_group_id": component_by_id[base_fact_id],
            "postclosure_split_policy_version": (
                postclosure_split_policy_version
            ),
            "duplicate_closure_applied": True,
            "semantic_positive_edges_applied": (
                len(semantic_rebind["active_edges"])
                if semantic_rebind is not None
                else 0
            ),
            "distractors_regenerated_after_final_split": True,
            "canonical_or_split_freeze_performed": False,
        }
        retained_rows.append(row)

    distractors_by_target, distractor_rows = _make_distractors(
        retained_rows,
        input_row_hashes=input_row_hashes,
        component_by_id=component_by_id,
        resolution_sha256=resolution_manifest_sha,
        semantic_partition_evidence_sha256=semantic_partition_evidence_sha,
    )
    for row in retained_rows:
        row["distractor_candidates"] = copy.deepcopy(
            distractors_by_target[str(row["base_fact_id"])]
        )
    _validate_distractors(retained_rows, distractor_rows)

    final_row_hashes = {
        str(row["base_fact_id"]): sha256_value(row) for row in retained_rows
    }
    assignment_rows: List[Dict[str, Any]] = []
    for cohort_index, raw_row in enumerate(input_rows):
        base_fact_id = str(raw_row["base_fact_id"])
        exclusion = removed.get(base_fact_id)
        if exclusion is None:
            disposition = "retained"
            representative = base_fact_id
            component_id: Optional[str] = component_by_id[base_fact_id]
            lexical_component_id: Optional[str] = lexical_component_by_id[
                base_fact_id
            ]
            final_split: Optional[str] = final_split_by_id[base_fact_id]
            reason_pair_ids: List[str] = []
            final_record_sha256: Optional[str] = final_row_hashes[base_fact_id]
        else:
            disposition = str(exclusion["disposition"])
            representative = exclusion.get("representative_base_fact_id")
            component_id = None
            lexical_component_id = None
            final_split = None
            reason_pair_ids = copy.deepcopy(exclusion["reason_pair_ids"])
            final_record_sha256 = None
        assignment_rows.append(
            {
                "schema_version": ASSIGNMENT_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "input_cohort_index": cohort_index,
                "input_record_sha256": input_row_hashes[base_fact_id],
                "preclosure_split_assignment": raw_row["split_assignment"],
                "preclosure_split_group_id": raw_row.get("split_group_id"),
                "disposition": disposition,
                "representative_base_fact_id": representative,
                "lexical_leakage_component_id": lexical_component_id,
                "leakage_component_id": component_id,
                "postclosure_split_assignment": final_split,
                "postclosure_split_group_id": component_id,
                "reason_pair_ids": reason_pair_ids,
                "final_record_sha256": final_record_sha256,
            }
        )

    retained_counts: Counter[str] = Counter(
        f"{row['probe_relation_id']}/{row['split_assignment']}"
        for row in retained_rows
    )
    removed_counts: Counter[str] = Counter(
        row["disposition"] for row in removed.values()
    )
    split_changes = sum(
        original_split_by_id[base_fact_id] != final_split_by_id[base_fact_id]
        for base_fact_id in final_split_by_id
    )
    semantic_split_changes = sum(
        lexical_split_by_id[base_fact_id] != final_split_by_id[base_fact_id]
        for base_fact_id in final_split_by_id
    )
    semantic_summary = {
        "semantic_component_union_applied": semantic_rebind is not None,
        "lexical_component_count": len(components),
        "semantic_aware_component_count": (
            len(semantic_components) if semantic_rebind is not None else len(components)
        ),
        "active_semantic_positive_edge_count": (
            len(semantic_rebind["active_edges"])
            if semantic_rebind is not None
            else 0
        ),
        "retired_semantic_positive_edge_count": (
            len(semantic_rebind["retired_edges"])
            if semantic_rebind is not None
            else 0
        ),
        "source_cross_split_semantic_positive_edge_count": (
            semantic_rebind["source_cross_split_positive_count"]
            if semantic_rebind is not None
            else 0
        ),
        "semantic_aware_input_split_change_count": split_changes,
        "lexical_draft_to_semantic_split_change_count": semantic_split_changes,
        "all_active_semantic_edges_component_atomic": semantic_rebind is not None,
        "all_source_cross_split_semantic_edges_resolved_or_retired": (
            semantic_rebind is not None
        ),
        "exact_relation_targets_achieved_after_semantic_union": (
            semantic_split_stats["exact_relation_targets_achieved"]
            if semantic_split_stats is not None
            else split_manifest.get("exact_relation_targets_achieved")
        ),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": "postclosure_preperturbation_bundle_materialized_not_frozen",
        "input_base_fact_count": len(input_rows),
        "retained_base_fact_count": len(retained_rows),
        "removed_base_fact_count": len(removed),
        "removed_disposition_counts": dict(sorted(removed_counts.items())),
        "replacement_required_to_restore_input_size": len(removed),
        "postclosure_split_change_count": split_changes,
        "split_policy_version": postclosure_split_policy_version,
        "retained_counts_by_relation_split": dict(sorted(retained_counts.items())),
        "distractor_candidate_count": len(distractor_rows),
        "distractors_per_fact": DISTRACTORS_PER_FACT,
        "all_donors_retained": True,
        "all_donors_same_relation_and_final_split": True,
        "all_donor_aliases_disjoint_from_targets": True,
        "input_split_group_atomicity_preserved": True,
        "all_retained_rows_use_validated_split_policy": True,
        **semantic_summary,
        "safety_contract": SAFETY_CONTRACT,
        "next_required_stage": (
            "replacement_selection_and_closure_rerun"
            if removed
            else "independent_distractor_review_then_explicit_freeze_decision"
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        filenames = {
            "bundle": "postclosure_preperturbation_behavior_input_bundle.jsonl",
            "distractors": "distractor_candidates.jsonl",
            "assignments": "postclosure_assignments.jsonl",
            "summary": "summary.json",
            "manifest": "finalization_manifest.json",
            "semantic_components": "semantic_aware_leakage_components.jsonl",
            "semantic_edge_resolutions": "semantic_edge_resolutions.jsonl",
            "semantic_split_manifest": "semantic_aware_split_manifest.json",
        }
        write_jsonl(temporary_dir / filenames["bundle"], retained_rows)
        write_jsonl(temporary_dir / filenames["distractors"], distractor_rows)
        write_jsonl(temporary_dir / filenames["assignments"], assignment_rows)
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

        semantic_artifact_bindings: Dict[str, Any] = {}
        semantic_input_bindings: Dict[str, Any] = {}
        if semantic_rebind is not None:
            assert semantic_closure_manifest_path is not None
            assert cohort_repair_manifest_path is not None
            assert replacement_semantic_review_path is not None
            assert semantic_split_stats is not None
            write_jsonl(
                temporary_dir / filenames["semantic_components"],
                semantic_components,
            )
            write_jsonl(
                temporary_dir / filenames["semantic_edge_resolutions"],
                semantic_edge_resolutions,
            )
            component_binding = output_binding(
                "semantic_components",
                SEMANTIC_COMPONENT_SCHEMA_VERSION,
                len(semantic_components),
            )
            edge_resolution_binding = output_binding(
                "semantic_edge_resolutions",
                SEMANTIC_EDGE_RESOLUTION_SCHEMA_VERSION,
                len(semantic_edge_resolutions),
            )
            semantic_input_bindings = {
                "semantic_closure_manifest": _file_binding(
                    semantic_closure_manifest_path,
                    schema_version=SEMANTIC_CLOSURE_MANIFEST_SCHEMA_VERSION,
                ),
                "cohort_repair_manifest": _file_binding(
                    cohort_repair_manifest_path,
                    schema_version=COHORT_REPAIR_MANIFEST_SCHEMA_VERSION,
                ),
                "replacement_semantic_review": _file_binding(
                    replacement_semantic_review_path,
                    schema_version=REPLACEMENT_SEMANTIC_REVIEW_SCHEMA_VERSION,
                ),
            }
            semantic_split_output = {
                "schema_version": SEMANTIC_SPLIT_MANIFEST_SCHEMA_VERSION,
                "tool_version": TOOL_VERSION,
                "status": (
                    "semantic_positive_edges_component_closed_"
                    "provisional_not_frozen"
                ),
                "split_status": "provisional_not_frozen",
                "split_policy_version": SEMANTIC_AWARE_SPLIT_POLICY_VERSION,
                "semantic_partition_evidence_sha256": (
                    semantic_partition_evidence_sha
                ),
                "inputs": {
                    "lexical_closure_resolution_manifest": _file_binding(
                        closure_resolution_manifest_path,
                        schema_version=RESOLUTION_MANIFEST_SCHEMA_VERSION,
                    ),
                    **semantic_input_bindings,
                },
                "source_semantic_scope": semantic_rebind["semantic_manifest"].get(
                    "semantic_paraphrase_closure_scope"
                ),
                "semantic_paraphrase_closure_complete_within_declared_scope": True,
                "semantic_paraphrase_recall_guaranteed": False,
                "source_positive_edge_count": semantic_rebind[
                    "source_positive_edge_count"
                ],
                "active_positive_edge_count": len(
                    semantic_rebind["active_edges"]
                ),
                "retired_positive_edge_count": len(
                    semantic_rebind["retired_edges"]
                ),
                "source_cross_split_positive_edge_count": semantic_rebind[
                    "source_cross_split_positive_count"
                ],
                "source_cross_split_positive_edges_resolved_or_retired": sum(
                    bool(row["source_cross_split"])
                    for row in semantic_edge_resolutions
                ),
                "lexical_component_count": len(components),
                "semantic_aware_component_count": len(semantic_components),
                **copy.deepcopy(semantic_split_stats),
                "component_assignments": [
                    {
                        "component_id": row["component_id"],
                        "split_assignment": row[
                            "provisional_split_assignment"
                        ],
                        "retained_cohort_base_fact_ids": row[
                            "retained_cohort_base_fact_ids"
                        ],
                        "relation_counts": row["relation_counts"],
                        "supporting_semantic_edge_ids": row[
                            "supporting_semantic_edge_ids"
                        ],
                    }
                    for row in semantic_components
                ],
                "artifacts": {
                    "semantic_aware_components": component_binding,
                    "semantic_edge_resolutions": edge_resolution_binding,
                },
                "safety_contract": {
                    "component_atomicity_enforced": True,
                    "semantic_positive_edges_applied": True,
                    "copied_provisional_split_assignments": False,
                    "network_or_model_used": False,
                    "review_freeze_emitted": False,
                    "split_freeze_emitted": False,
                    "perturbation_authorized": False,
                    "human_gold": False,
                },
            }
            write_json(
                temporary_dir / filenames["semantic_split_manifest"],
                semantic_split_output,
            )
            semantic_artifact_bindings = {
                "semantic_aware_leakage_components": component_binding,
                "semantic_edge_resolutions": edge_resolution_binding,
                "semantic_aware_split_manifest": output_binding(
                    "semantic_split_manifest",
                    SEMANTIC_SPLIT_MANIFEST_SCHEMA_VERSION,
                ),
            }

        manifest = {
            "schema_version": FINALIZATION_MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "postclosure_preperturbation_bundle_materialized_not_frozen",
            "inputs": {
                "preperturbation_bundle": _file_binding(
                    preperturbation_bundle_path,
                    schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
                    record_count=len(input_rows),
                ),
                "selection_manifest": _file_binding(selection_manifest_path),
                "closure_resolution_manifest": _file_binding(
                    closure_resolution_manifest_path,
                    schema_version=RESOLUTION_MANIFEST_SCHEMA_VERSION,
                ),
                "provisional_split_manifest": _file_binding(
                    split_manifest_path,
                    schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
                ),
                **semantic_input_bindings,
            },
            "closure_counts": copy.deepcopy(split_manifest["cohort_counts"]),
            "split_policy_version": postclosure_split_policy_version,
            "consistency_checks": {
                "selection_manifest_binds_input_bundle": True,
                "closure_candidate_manifest_binds_input_bundle": True,
                "closure_covers_every_input_base_fact_id": True,
                "component_assignments_are_atomic": True,
                "input_split_group_atomicity_preserved": True,
                "all_retained_rows_use_validated_split_policy": True,
                "removed_rows_absent_from_output_bundle": True,
                "distractors_regenerated_after_final_split": True,
                "all_donors_retained": True,
                "all_donors_same_relation_and_final_split": True,
                "all_donor_aliases_disjoint_from_targets": True,
                "semantic_positive_edges_component_atomic": (
                    semantic_rebind is not None
                ),
                "source_cross_split_semantic_edges_resolved_or_retired": (
                    semantic_rebind is not None
                ),
            },
            "artifacts": {
                "postclosure_preperturbation_behavior_input_bundle": output_binding(
                    "bundle", INPUT_BUNDLE_SCHEMA_VERSION, len(retained_rows)
                ),
                "distractor_candidates": output_binding(
                    "distractors", DISTRACTOR_SCHEMA_VERSION, len(distractor_rows)
                ),
                "postclosure_assignments": output_binding(
                    "assignments", ASSIGNMENT_SCHEMA_VERSION, len(assignment_rows)
                ),
                "summary": output_binding("summary", SUMMARY_SCHEMA_VERSION),
                **semantic_artifact_bindings,
            },
            "safety_contract": SAFETY_CONTRACT,
        }
        write_json(temporary_dir / filenames["manifest"], manifest)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "finalization_manifest.json"),
        "manifest_sha256": sha256_file(output_dir / "finalization_manifest.json"),
        "summary_path": str(output_dir / "summary.json"),
        "behavior_bundle_path": str(
            output_dir / "postclosure_preperturbation_behavior_input_bundle.jsonl"
        ),
        "retained_base_fact_count": len(retained_rows),
        "removed_base_fact_count": len(removed),
        "distractor_candidate_count": len(distractor_rows),
        "semantic_component_union_applied": semantic_rebind is not None,
        "semantic_aware_component_count": (
            len(semantic_components) if semantic_rebind is not None else len(components)
        ),
        "source_cross_split_semantic_positive_edge_count": (
            semantic_rebind["source_cross_split_positive_count"]
            if semantic_rebind is not None
            else 0
        ),
        "perturbation_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preperturbation-bundle", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--closure-resolution-manifest", type=Path, required=True)
    parser.add_argument("--semantic-closure-manifest", type=Path)
    parser.add_argument("--cohort-repair-manifest", type=Path)
    parser.add_argument("--replacement-semantic-review", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = finalize_postclosure_preperturbation(
            preperturbation_bundle_path=args.preperturbation_bundle,
            selection_manifest_path=args.selection_manifest,
            closure_resolution_manifest_path=args.closure_resolution_manifest,
            output_dir=args.output_dir,
            semantic_closure_manifest_path=args.semantic_closure_manifest,
            cohort_repair_manifest_path=args.cohort_repair_manifest,
            replacement_semantic_review_path=args.replacement_semantic_review,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
