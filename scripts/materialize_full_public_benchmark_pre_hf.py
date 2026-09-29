#!/usr/bin/env python3
"""Materialize the complete provisional public-benchmark pool up to the pre-HF boundary.

This is an offline-only materializer.  It retains every provisional base fact,
assigns an explicitly provisional relation partition, builds leakage components
and Development/Validation/Sealed splits, audits registered multilingual data,
and stages dual-distractor/neutral candidates.  It never loads a tokenizer or
model, runs behavior, promotes a fact to human gold, or freezes a formal split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "factual_triples"
    / "public-benchmarks-full-v1"
    / "provisional"
    / "qwen3.7-plus-v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "full-8969-pre-hf-v1"
DEFAULT_LANGUAGE_REGISTRY = (
    PROJECT_ROOT / "configs" / "public_benchmark_multilingual_registry_v1.json"
)
DEFAULT_FROZEN_SPLIT_BUNDLE = (
    PROJECT_ROOT
    / "data_processed"
    / "factual_perturbation"
    / "static-g0a-160-dual-v8-20260916"
    / "frozen-v1"
    / "frozen_static_bundle.jsonl"
)

TOOL_VERSION = "public-benchmark-full-pre-hf-materializer-v1"
FULL_FACT_SCHEMA_VERSION = "public-benchmark-full-provisional-base-fact-v1"
RELATION_ASSIGNMENT_SCHEMA_VERSION = "public-benchmark-relation-partition-assignment-v1"
RELATION_INVENTORY_SCHEMA_VERSION = "public-benchmark-full-relation-inventory-v1"
COMPONENT_SCHEMA_VERSION = "public-benchmark-full-leakage-component-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "public-benchmark-full-provisional-split-manifest-v1"
LANGUAGE_AUDIT_SCHEMA_VERSION = "public-benchmark-multilingual-dataset-audit-v1"
MULTILINGUAL_AVAILABILITY_SCHEMA_VERSION = "public-benchmark-multilingual-availability-v1"
MULTILINGUAL_ROW_SCHEMA_VERSION = "public-benchmark-multilingual-preperturbation-row-v1"
DISTRACTOR_SCHEMA_VERSION = "public-benchmark-dual-distractor-candidate-v1"
NEUTRAL_SCHEMA_VERSION = "public-benchmark-neutral-reference-candidate-v1"
STATIC_CONTRACT_SCHEMA_VERSION = "public-benchmark-pre-hf-static-contract-v1"
SHORTFALL_SCHEMA_VERSION = "public-benchmark-pre-hf-candidate-shortfall-v1"
INTEGRITY_AUDIT_SCHEMA_VERSION = "public-benchmark-full-pre-hf-integrity-audit-v1"
GATE_SCHEMA_VERSION = "public-benchmark-pre-hf-gate-manifest-v1"
SUMMARY_SCHEMA_VERSION = "public-benchmark-full-pre-hf-summary-v1"

SPLITS = ("development", "validation", "sealed")
SPLIT_RATIOS = {"development": 0.6, "validation": 0.2, "sealed": 0.2}
DEFAULT_SPLIT_SEED = "public-benchmark-full-8969-component-split-v1"
SPLIT_POLICY_VERSION = "relation-source-answer-component-greedy-sha256-v1"
DISTRACTOR_POLICY_VERSION = "same-split-relation-family-dual-distractor-sha256-v1"
NEUTRAL_POLICY_VERSION = "same-split-different-family-neutral-reference-sha256-v1"

SOURCE_ARTIFACTS = {
    "provisional_triples": ("provisional_triples.jsonl", "provisional-factual-triple-v1"),
    "base_fact_clusters": ("base_fact_clusters.jsonl", "provisional-base-fact-cluster-v1"),
    "behavior_input_bundle": ("behavior_input_bundle.jsonl", "factual-perturbation-input-bundle-v1"),
    "review_queue": ("review_queue.jsonl", "provisional-semantic-review-v1"),
}


# These are broad, conservative partition candidates.  They are deliberately
# separate from relation_normalized and probe_relation_id.
RELATION_FAMILY_RULES = (
    (
        "definition_or_meaning",
        r"\b(definition|meaning|means|defined as|refers to|term for|word for|name for definition)\b",
    ),
    (
        "alias_or_identity",
        r"\b(alias|also known as|also called|common name|nickname|full form|stands for|identity)\b",
    ),
    (
        "classification_or_type",
        r"\b(classification|type of|kind of|category|class|species|genus|family|kingdom)\b",
    ),
    (
        "location_or_habitat",
        r"\b(location|located|found in|country|city|state|capital|habitat|native to|filming location|place)\b",
    ),
    (
        "origin_or_source",
        r"\b(origin|originated|derived from|source region|place of origin|etymolog)\b",
    ),
    (
        "part_whole_or_membership",
        r"\b(part of|belongs to|member of|contains|component|constituent|has part|included in)\b",
    ),
    (
        "composition_or_material",
        r"\b(made of|composed of|composition|material|consists of|chemical formula)\b",
    ),
    (
        "function_goal_or_use",
        r"\b(function|purpose|used for|role|goal|objective|aim|serves to|intended)\b",
    ),
    (
        "cause_effect_or_process",
        r"\b(cause|caused by|effect|results in|results from|leads to|formed by|process|mechanism)\b",
    ),
    (
        "creator_author_or_producer",
        r"\b(author|written by|creator|created by|inventor|invented by|developer|developed by|producer|director|founded by)\b",
    ),
    (
        "performer_or_cast",
        r"\b(singer|sung by|performer|performed by|portrayed by|played by|actor|voice actor|cast)\b",
    ),
    (
        "temporal",
        r"\b(date|year|time|duration|period|release date|release year|born|died|age|founded in)\b",
    ),
    (
        "quantity_or_measurement",
        r"\b(number|amount|count|percentage|rate|measurement|height|length|width|mass|weight|temperature|distance|unit)\b",
    ),
    (
        "attribute_or_state",
        r"\b(attribute|property|characteristic|feature|color|shape|status|state|charge|description)\b",
    ),
    (
        "comparison_or_order",
        r"\b(largest|smallest|highest|lowest|first|last|next|successor|predecessor|greater|lesser|rank)\b",
    ),
    (
        "language_or_symbol",
        r"\b(language|translation|symbol|represents|abbreviation|acronym|element symbol)\b",
    ),
)

ANSWER_BUCKET_PATTERNS = (
    ("person_agent", r"\b(person|actor|author|scientist|politician|artist|musician|athlete|human|individual|singer)\b"),
    ("location", r"\b(location|country|city|state|place|region|continent|lake|river|mountain|island|building|venue)\b"),
    ("temporal", r"\b(date|year|time|duration|period|age|century|month|day)\b"),
    ("numeric_measurement", r"\b(number|quantity|percentage|measurement|value|distance|length|mass|weight|temperature|rate|score|integer)\b"),
    ("organization_group", r"\b(organization|company|institution|team|band|group|government|agency|party)\b"),
    ("work_title", r"\b(work|book|film|movie|song|album|series|episode|poem|play|novel|document|title|publication)\b"),
    ("definition_concept", r"\b(definition|meaning|concept|term|theory|principle|process|method|phenomenon|idea)\b"),
    ("biological_entity", r"\b(organism|animal|plant|species|protein|gene|cell|body part|disease)\b"),
    ("language_text", r"\b(language|word|phrase|name|text|symbol|letter|character|string)\b"),
    ("boolean_state", r"\b(boolean|yes/no|truth value|state|status)\b"),
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
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
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, payload)


def normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def normalize_match_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def slug(value: Any) -> str:
    text = normalize_match_text(value).replace(" ", "_")
    return text[:64] or "unknown"


def stable_id(prefix: str, parts: Sequence[Any], length: int = 24) -> str:
    return f"{prefix}_{sha256_value(list(parts))[:length]}"


def _index_unique(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> Dict[str, Mapping[str, Any]]:
    output: Dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = str(row.get(field) or "").strip()
        if not key:
            raise ValueError(f"Missing {field} in {label} row {row_number}")
        if key in output:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        output[key] = row
    return output


def _artifact(path: Path, schema_version: str, record_count: Optional[int] = None) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def load_and_validate_sources(source_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    source_dir = source_dir.resolve()
    summary_path = source_dir / "summary.json"
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("Source summary must be a JSON object")
    rows: Dict[str, List[Dict[str, Any]]] = {}
    bindings: Dict[str, Any] = {}
    for role, (filename, expected_schema) in SOURCE_ARTIFACTS.items():
        path = source_dir / filename
        artifact_rows = read_jsonl(path)
        for row_number, row in enumerate(artifact_rows, start=1):
            if row.get("schema_version") != expected_schema:
                raise ValueError(
                    f"Unsupported {role} schema at row {row_number}: {row.get('schema_version')!r}"
                )
            if row.get("canonical_status") != "pending_review":
                raise ValueError(f"{role} row {row_number} is not pending_review")
            if row.get("human_gold") is not False:
                raise ValueError(f"{role} row {row_number} must have human_gold=false")
            if row.get("evidence_tier") != "provisional_single_model":
                raise ValueError(f"{role} row {row_number} has unsupported evidence_tier")
        expected_sha = (summary.get("artifacts") or {}).get(filename, {}).get("sha256")
        actual_sha = sha256_file(path)
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"Source summary SHA mismatch for {filename}")
        rows[role] = artifact_rows
        bindings[role] = _artifact(path, expected_schema, len(artifact_rows))

    clusters = _index_unique(rows["base_fact_clusters"], "base_fact_id", "base_fact_clusters")
    behavior = _index_unique(rows["behavior_input_bundle"], "base_fact_id", "behavior_input_bundle")
    review = _index_unique(rows["review_queue"], "base_fact_id", "review_queue")
    provisional_by_fact: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    provisional_candidates = _index_unique(
        rows["provisional_triples"], "candidate_id", "provisional_triples"
    )
    for row in rows["provisional_triples"]:
        provisional_by_fact[str(row.get("base_fact_id") or "")].append(row)
    universes = [set(clusters), set(behavior), set(review), set(provisional_by_fact)]
    if any(universe != universes[0] for universe in universes[1:]):
        raise ValueError(
            "Source artifact base_fact_id universes differ: "
            + ", ".join(str(len(universe)) for universe in universes)
        )
    expected_count = (summary.get("counts") or {}).get("base_fact_clusters")
    if expected_count is not None and expected_count != len(clusters):
        raise ValueError("Source summary base_fact count mismatch")

    for base_fact_id in sorted(clusters):
        cluster = clusters[base_fact_id]
        bundle = behavior[base_fact_id]
        members = cluster.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Cluster has no members: {base_fact_id}")
        member_ids = {str(item.get("candidate_id") or "") for item in members}
        fact_candidate_ids = {str(item.get("candidate_id") or "") for item in provisional_by_fact[base_fact_id]}
        if member_ids != fact_candidate_ids:
            raise ValueError(f"Cluster member mismatch: {base_fact_id}")
        representative = str(cluster.get("representative_candidate_id") or "")
        if representative not in provisional_candidates:
            raise ValueError(f"Unknown representative candidate: {base_fact_id}")
        if bundle.get("candidate_id") != representative:
            raise ValueError(f"Behavior row does not bind representative: {base_fact_id}")
        if cluster.get("split_group_id") != bundle.get("split_group_id"):
            raise ValueError(f"Input split group mismatch: {base_fact_id}")

    bindings["source_summary"] = _artifact(
        summary_path,
        str(summary.get("schema_version") or "single-model-provisional-summary-v2"),
    )
    return rows, {"summary": summary, "bindings": bindings}


def answer_type_bucket(value: Any) -> str:
    normalized = normalize_text(value)
    for bucket, pattern in ANSWER_BUCKET_PATTERNS:
        if re.search(pattern, normalized):
            return bucket
    return "other"


def assign_relation(row: Mapping[str, Any]) -> Dict[str, Any]:
    base_fact_id = str(row["base_fact_id"])
    candidate = str(row.get("probe_relation_candidate") or "").strip()
    raw = normalize_text(row.get("relation_raw"))
    if candidate:
        family_id = candidate
        status = "existing_probe_relation_candidate_partition"
        basis = "source_probe_relation_candidate"
        confidence: Optional[float] = None
    else:
        family_id = ""
        basis = ""
        confidence = None
        for rule_id, pattern in RELATION_FAMILY_RULES:
            if re.search(pattern, raw):
                family_id = rule_id
                status = "broad_rule_candidate_partition"
                basis = pattern
                confidence = 0.7
                break
        if not family_id:
            bucket = answer_type_bucket(row.get("answer_type"))
            family_id = f"unresolved_by_answer_type__{bucket}"
            status = "unresolved_partition_only"
            basis = "answer_type_bucket_fallback"
    return {
        "schema_version": RELATION_ASSIGNMENT_SCHEMA_VERSION,
        "base_fact_id": base_fact_id,
        "relation_signature_id": row.get("relation_signature_id"),
        "relation_raw": row.get("relation_raw"),
        "relation_normalized": row.get("relation_normalized"),
        "probe_relation_candidate": row.get("probe_relation_candidate"),
        "probe_relation_id": row.get("probe_relation_id"),
        "relation_family_candidate": family_id,
        "relation_partition_id": family_id,
        "relation_partition_status": status,
        "relation_partition_basis": basis,
        "relation_partition_confidence": confidence,
        "answer_type_bucket": answer_type_bucket(row.get("answer_type")),
        "formal_relation_normalization_performed": False,
        "mechanism_relation_eligible": bool(row.get("probe_relation_id")),
    }


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


def _union_groups(union_find: UnionFind, groups: Mapping[str, Sequence[str]]) -> None:
    for members in groups.values():
        ordered = sorted(set(members))
        for member in ordered[1:]:
            union_find.union(ordered[0], member)


def load_frozen_split_constraints(path: Optional[Path], available_ids: Sequence[str]) -> Dict[str, Any]:
    if path is None:
        return {"rows": [], "binding": None, "split_by_fact": {}, "component_by_fact": {}}
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    split_by_fact: Dict[str, str] = {}
    component_by_fact: Dict[str, str] = {}
    available = set(available_ids)
    for row_number, row in enumerate(rows, start=1):
        base_fact_id = str(row.get("base_fact_id") or "")
        split = str(row.get("split_assignment") or "")
        component = str(row.get("leakage_component_id") or row.get("split_group_id") or "")
        if base_fact_id not in available:
            raise ValueError(f"Frozen split fact is outside the full pool: {base_fact_id}")
        if base_fact_id in split_by_fact:
            raise ValueError(f"Duplicate frozen split fact: {base_fact_id}")
        if split not in SPLITS:
            raise ValueError(f"Unsupported frozen split at row {row_number}: {split}")
        if not component:
            raise ValueError(f"Missing frozen leakage component at row {row_number}")
        split_by_fact[base_fact_id] = split
        component_by_fact[base_fact_id] = component
    return {
        "rows": rows,
        "binding": _artifact(path, str(rows[0].get("schema_version") or "unknown"), len(rows)),
        "split_by_fact": split_by_fact,
        "component_by_fact": component_by_fact,
    }


def build_components(
    behavior_rows: Sequence[Mapping[str, Any]],
    relation_by_fact: Mapping[str, Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    ids = [str(row["base_fact_id"]) for row in behavior_rows]
    union_find = UnionFind(ids)
    source_groups: MutableMapping[str, List[str]] = defaultdict(list)
    for row in behavior_rows:
        group_id = str(row.get("split_group_id") or "")
        if not group_id:
            raise ValueError(f"Missing source split_group_id: {row['base_fact_id']}")
        source_groups[group_id].append(str(row["base_fact_id"]))
    _union_groups(union_find, source_groups)

    frozen_groups: MutableMapping[str, List[str]] = defaultdict(list)
    for base_fact_id, component_id in frozen.get("component_by_fact", {}).items():
        frozen_groups[str(component_id)].append(str(base_fact_id))
    _union_groups(union_find, frozen_groups)

    members_by_root: MutableMapping[str, List[str]] = defaultdict(list)
    for base_fact_id in ids:
        members_by_root[union_find.find(base_fact_id)].append(base_fact_id)
    behavior_by_id = {str(row["base_fact_id"]): row for row in behavior_rows}
    components: List[Dict[str, Any]] = []
    component_by_fact: Dict[str, str] = {}
    frozen_split_by_fact = frozen.get("split_by_fact", {})
    for members in sorted((sorted(value) for value in members_by_root.values()), key=lambda x: x[0]):
        component_id = stable_id("full_leakage_component", members)
        pinned = {frozen_split_by_fact[item] for item in members if item in frozen_split_by_fact}
        if len(pinned) > 1:
            raise ValueError(
                f"Frozen split conflict after leakage union in {component_id}: {sorted(pinned)}"
            )
        relation_counts = Counter(
            str(relation_by_fact[item]["relation_partition_id"]) for item in members
        )
        source_counts = Counter(str(behavior_by_id[item].get("source_dataset") or "unknown") for item in members)
        answer_counts = Counter(str(relation_by_fact[item]["answer_type_bucket"]) for item in members)
        source_split_groups = sorted(
            {str(behavior_by_id[item].get("split_group_id") or "") for item in members}
        )
        frozen_semantic_groups = sorted(
            {
                str(frozen["component_by_fact"][item])
                for item in members
                if item in frozen.get("component_by_fact", {})
            }
        )
        component = {
            "schema_version": COMPONENT_SCHEMA_VERSION,
            "leakage_component_id": component_id,
            "member_count": len(members),
            "base_fact_ids": members,
            "source_split_group_ids": source_split_groups,
            "frozen_semantic_component_ids": frozen_semantic_groups,
            "relation_partition_counts": dict(sorted(relation_counts.items())),
            "source_dataset_counts": dict(sorted(source_counts.items())),
            "answer_type_bucket_counts": dict(sorted(answer_counts.items())),
            "pinned_split_assignment": next(iter(pinned)) if pinned else None,
            "pinned_fact_count": sum(item in frozen_split_by_fact for item in members),
            "component_semantics": [
                "source_normalized_answer_group",
                *( ["preserved_frozen_160_semantic_component"] if frozen_semantic_groups else [] ),
            ],
        }
        components.append(component)
        for member in members:
            component_by_fact[member] = component_id
    return components, component_by_fact


def integer_split_targets(total: int) -> Dict[str, int]:
    raw = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
    targets = {split: int(math.floor(raw[split])) for split in SPLITS}
    remaining = total - sum(targets.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - targets[split]), SPLITS.index(split)))
    for split in order[:remaining]:
        targets[split] += 1
    return targets


def _delta_cost(current: int, addition: int, target: float) -> float:
    scale = max(1.0, target)
    before = ((current - target) / scale) ** 2
    after = ((current + addition - target) / scale) ** 2
    return after - before


def assign_component_splits(
    components: Sequence[MutableMapping[str, Any]],
    split_seed: str,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    total = sum(int(component["member_count"]) for component in components)
    targets = integer_split_targets(total)
    global_counts = Counter({split: 0 for split in SPLITS})
    feature_names = (
        "relation_partition_counts",
        "source_dataset_counts",
        "answer_type_bucket_counts",
    )
    feature_totals: Dict[str, Counter] = {name: Counter() for name in feature_names}
    feature_counts: Dict[str, Dict[str, Counter]] = {
        name: {split: Counter() for split in SPLITS} for name in feature_names
    }
    for component in components:
        for name in feature_names:
            feature_totals[name].update(component[name])

    assignments: Dict[str, str] = {}

    def apply(component: Mapping[str, Any], split: str) -> None:
        component_id = str(component["leakage_component_id"])
        assignments[component_id] = split
        global_counts[split] += int(component["member_count"])
        for name in feature_names:
            feature_counts[name][split].update(component[name])

    pinned_components = [component for component in components if component.get("pinned_split_assignment")]
    for component in sorted(pinned_components, key=lambda item: str(item["leakage_component_id"])):
        apply(component, str(component["pinned_split_assignment"]))

    unpinned = [component for component in components if not component.get("pinned_split_assignment")]
    unpinned.sort(
        key=lambda item: (
            -int(item["member_count"]),
            sha256_value([split_seed, item["leakage_component_id"]]),
        )
    )
    feature_weights = {
        "relation_partition_counts": 3.0,
        "source_dataset_counts": 1.5,
        "answer_type_bucket_counts": 1.5,
    }
    for component in unpinned:
        scores: List[Tuple[float, str, str]] = []
        size = int(component["member_count"])
        for split in SPLITS:
            score = 12.0 * _delta_cost(global_counts[split], size, float(targets[split]))
            for name in feature_names:
                for feature, addition in component[name].items():
                    target = feature_totals[name][feature] * SPLIT_RATIOS[split]
                    score += feature_weights[name] * _delta_cost(
                        feature_counts[name][split][feature], int(addition), target
                    )
            overflow = max(0, global_counts[split] + size - targets[split])
            score += 25.0 * (overflow / max(1, targets[split])) ** 2
            tie = sha256_value([split_seed, component["leakage_component_id"], split])
            scores.append((score, tie, split))
        apply(component, min(scores)[2])

    for component in components:
        component["split_assignment"] = assignments[str(component["leakage_component_id"])]
        component["split_policy_version"] = SPLIT_POLICY_VERSION

    distributions: Dict[str, Dict[str, Dict[str, int]]] = {}
    for name in feature_names:
        distributions[name] = {
            feature: {
                split: feature_counts[name][split][feature]
                for split in SPLITS
            }
            for feature in sorted(feature_totals[name])
        }
    audit = {
        "target_base_fact_counts": targets,
        "actual_base_fact_counts": {split: global_counts[split] for split in SPLITS},
        "absolute_target_deviation": {
            split: abs(global_counts[split] - targets[split]) for split in SPLITS
        },
        "component_counts": {
            split: sum(assignments[str(component["leakage_component_id"])] == split for component in components)
            for split in SPLITS
        },
        "pinned_component_count": len(pinned_components),
        "pinned_fact_count": sum(int(component["pinned_fact_count"]) for component in components),
        "stratification_distributions": distributions,
    }
    return assignments, audit


def load_language_registry(path: Path) -> Dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Language registry must be a JSON object")
    if value.get("schema_version") != "public-benchmark-multilingual-registry-v1":
        raise ValueError("Unsupported language registry schema")
    languages = value.get("languages")
    if not isinstance(languages, list) or not languages:
        raise ValueError("Language registry has no languages")
    codes = [str(item.get("language_code") or "") for item in languages]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("Language registry codes must be non-empty and unique")
    if value.get("source_language") not in codes:
        raise ValueError("Language registry source language is missing")
    return value


def audit_language_datasets(
    registry: Mapping[str, Any],
    registry_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, List[Tuple[int, Mapping[str, Any]]]]]:
    indexes: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
    language_audits: List[Dict[str, Any]] = []
    for language in registry["languages"]:
        code = str(language["language_code"])
        dataset_path_value = language.get("dataset_path")
        if not dataset_path_value:
            language_audits.append(
                {
                    **language,
                    "dataset_status": "source_bundle_fields",
                    "record_count": None,
                    "sha256": None,
                    "translation_field_complete_count": None,
                }
            )
            indexes[code] = []
            continue
        dataset_path = Path(str(dataset_path_value))
        if not dataset_path.is_absolute():
            dataset_path = (PROJECT_ROOT / dataset_path).resolve()
        raw = read_json(dataset_path)
        if not isinstance(raw, list):
            raise ValueError(f"Language dataset must be a JSON list: {dataset_path}")
        indexed: List[Tuple[int, Mapping[str, Any]]] = []
        complete = 0
        for row_number, row in enumerate(raw):
            if not isinstance(row, dict):
                raise ValueError(f"Language dataset row is not an object: {dataset_path}:{row_number}")
            indexed.append((row_number, row))
            if row.get("transori") and row.get("transanswer") and isinstance(row.get("transchoices"), list):
                complete += 1
        indexes[code] = indexed
        language_audits.append(
            {
                **language,
                "dataset_path": str(dataset_path),
                "dataset_status": "external_independent_dataset_not_assumed_aligned",
                "record_count": len(raw),
                "sha256": sha256_file(dataset_path),
                "byte_count": dataset_path.stat().st_size,
                "translation_field_complete_count": complete,
            }
        )
    audit = {
        "schema_version": LANGUAGE_AUDIT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "registry": _artifact(
            registry_path.resolve(),
            str(registry["schema_version"]),
        ),
        "registry_id": registry.get("registry_id"),
        "source_language": registry.get("source_language"),
        "alignment_policy": registry.get("alignment_policy"),
        "languages": language_audits,
    }
    return audit, indexes


def build_multilingual_rows(
    behavior_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    language_audit: MutableMapping[str, Any],
    dataset_rows: Mapping[str, Sequence[Tuple[int, Mapping[str, Any]]]],
    split_by_fact: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    source_language = str(registry["source_language"])
    question_indexes: Dict[str, MutableMapping[str, List[Tuple[int, Mapping[str, Any]]]]] = {}
    for language in registry["languages"]:
        code = str(language["language_code"])
        index: MutableMapping[str, List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for row_number, row in dataset_rows.get(code, []):
            index[normalize_match_text(row.get("oriquestion"))].append((row_number, row))
        question_indexes[code] = index

    availability_rows: List[Dict[str, Any]] = []
    materialized_rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, Counter] = {str(item["language_code"]): Counter() for item in registry["languages"]}
    aligned_ids: Dict[str, set] = {str(item["language_code"]): set() for item in registry["languages"]}
    for source in sorted(behavior_rows, key=lambda row: str(row["base_fact_id"])):
        base_fact_id = str(source["base_fact_id"])
        aliases = {
            normalize_match_text(value)
            for value in [source.get("answer_en"), *(source.get("answer_aliases_en") or [])]
            if normalize_match_text(value)
        }
        question_key = normalize_match_text(source.get("source_question_en"))
        per_language: Dict[str, Dict[str, Any]] = {}
        for language in registry["languages"]:
            code = str(language["language_code"])
            if code == source_language:
                availability = {
                    "availability_status": "source_available_provisional",
                    "translation_status": "not_applicable_source_language",
                    "external_dataset_row_index": None,
                    "external_record_sha256": None,
                }
                materialized_rows.append(
                    {
                        "schema_version": MULTILINGUAL_ROW_SCHEMA_VERSION,
                        "base_fact_id": base_fact_id,
                        "language_code": code,
                        "split_assignment": split_by_fact[base_fact_id],
                        "question": source.get("source_question_en"),
                        "answer": source.get("answer_en"),
                        "answer_aliases": source.get("answer_aliases_en") or [],
                        "choices": source.get("source_choices_en") or [],
                        "prompt": source.get("prompt_en"),
                        "translation_status": "not_applicable_source_language",
                        "semantic_review_status": "pending_base_fact_review",
                        "behavior_authorized": False,
                    }
                )
                aligned_ids[code].add(base_fact_id)
            else:
                matches = list(question_indexes[code].get(question_key, []))
                answer_matches = [
                    item
                    for item in matches
                    if normalize_match_text(item[1].get("answer")) in aliases
                ]
                if len(answer_matches) == 1:
                    row_number, external = answer_matches[0]
                    translated_complete = bool(
                        external.get("transori")
                        and external.get("transanswer")
                        and isinstance(external.get("transchoices"), list)
                    )
                    status = (
                        "aligned_external_translation_candidate_unreviewed"
                        if translated_complete
                        else "aligned_external_translation_incomplete"
                    )
                    external_sha = sha256_value(external)
                    availability = {
                        "availability_status": status,
                        "translation_status": "candidate_unreviewed" if translated_complete else "incomplete",
                        "external_dataset_row_index": row_number,
                        "external_record_sha256": external_sha,
                    }
                    aligned_ids[code].add(base_fact_id)
                    if translated_complete:
                        materialized_rows.append(
                            {
                                "schema_version": MULTILINGUAL_ROW_SCHEMA_VERSION,
                                "base_fact_id": base_fact_id,
                                "language_code": code,
                                "split_assignment": split_by_fact[base_fact_id],
                                "question": external.get("transori"),
                                "answer": external.get("transanswer"),
                                "answer_aliases": [external.get("transanswer")],
                                "choices": external.get("transchoices") or [],
                                "prompt": None,
                                "translation_status": "candidate_unreviewed",
                                "semantic_review_status": "pending_translation_equivalence_review",
                                "external_dataset_row_index": row_number,
                                "external_record_sha256": external_sha,
                                "behavior_authorized": False,
                            }
                        )
                elif len(answer_matches) > 1:
                    availability = {
                        "availability_status": "ambiguous_exact_alignment",
                        "translation_status": "not_materialized",
                        "external_dataset_row_index": None,
                        "external_record_sha256": None,
                        "candidate_count": len(answer_matches),
                    }
                elif matches:
                    availability = {
                        "availability_status": "question_match_answer_conflict",
                        "translation_status": "not_materialized",
                        "external_dataset_row_index": None,
                        "external_record_sha256": None,
                        "candidate_count": len(matches),
                    }
                else:
                    availability = {
                        "availability_status": "pending_translation",
                        "translation_status": "not_run",
                        "external_dataset_row_index": None,
                        "external_record_sha256": None,
                    }
            per_language[code] = availability
            status_counts[code][availability["availability_status"]] += 1
        availability_rows.append(
            {
                "schema_version": MULTILINGUAL_AVAILABILITY_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "split_assignment": split_by_fact[base_fact_id],
                "languages": per_language,
                "all_languages_share_base_fact_split": True,
            }
        )

    audit_by_code = {str(item["language_code"]): item for item in language_audit["languages"]}
    for code in status_counts:
        audit_by_code[code]["base_fact_alignment_status_counts"] = dict(sorted(status_counts[code].items()))
        audit_by_code[code]["aligned_base_fact_count"] = len(aligned_ids[code])
        audit_by_code[code]["same_fact_claim_requires_exact_alignment"] = True
    summary = {
        "language_count": len(registry["languages"]),
        "availability_record_count": len(availability_rows),
        "materialized_language_row_count": len(materialized_rows),
        "status_counts_by_language": {
            code: dict(sorted(counts.items())) for code, counts in sorted(status_counts.items())
        },
        "aligned_base_fact_counts": {
            code: len(values) for code, values in sorted(aligned_ids.items())
        },
        "cross_language_split_mismatch_count": 0,
    }
    return availability_rows, materialized_rows, summary


def select_static_candidates(
    behavior_rows: Sequence[Mapping[str, Any]],
    relation_by_fact: Mapping[str, Mapping[str, Any]],
    component_by_fact: Mapping[str, str],
    split_by_fact: Mapping[str, str],
    *,
    excluded_candidate_pairs: Optional[Set[Tuple[str, str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Select two candidates of each kind without weakening the base policy.

    ``excluded_candidate_pairs`` is an optional, evidence-backed deny-list of
    ``(candidate_kind, target_base_fact_id, source_base_fact_id)`` tuples.  It
    is intentionally pair-scoped: rejecting or deferring one use of a source
    fact must not assert that the source fact itself is false or remove it as a
    donor for unrelated targets.  Omitting the argument preserves the original
    v1 selection byte-for-byte.
    """

    blocked_pairs = set(excluded_candidate_pairs or ())
    invalid_kinds = sorted({kind for kind, _, _ in blocked_pairs} - {"distractor", "neutral"})
    if invalid_kinds:
        raise ValueError(f"Unsupported excluded candidate kinds: {invalid_kinds}")
    by_id = {str(row["base_fact_id"]): row for row in behavior_rows}
    typed_pool: MutableMapping[Tuple[str, str, str], List[str]] = defaultdict(list)
    family_pool: MutableMapping[Tuple[str, str], List[str]] = defaultdict(list)
    split_pool: MutableMapping[str, List[str]] = defaultdict(list)
    for base_fact_id in sorted(by_id):
        relation = relation_by_fact[base_fact_id]
        split = split_by_fact[base_fact_id]
        family = str(relation["relation_partition_id"])
        bucket = str(relation["answer_type_bucket"])
        typed_pool[(split, family, bucket)].append(base_fact_id)
        family_pool[(split, family)].append(base_fact_id)
        split_pool[split].append(base_fact_id)

    distractor_rows: List[Dict[str, Any]] = []
    neutral_rows: List[Dict[str, Any]] = []
    distractor_ids: Dict[str, List[str]] = defaultdict(list)
    neutral_ids: Dict[str, List[str]] = defaultdict(list)
    distractor_count_by_fact: Counter = Counter()
    neutral_count_by_fact: Counter = Counter()
    for base_fact_id in sorted(by_id):
        source = by_id[base_fact_id]
        relation = relation_by_fact[base_fact_id]
        split = split_by_fact[base_fact_id]
        family = str(relation["relation_partition_id"])
        bucket = str(relation["answer_type_bucket"])
        source_answer = normalize_text(source.get("answer_en"))
        source_subject = normalize_text(source.get("subject_en"))
        candidates: List[Tuple[str, str]] = []
        seen = set()
        for tier, pool in (
            ("same_family_same_answer_type_bucket", typed_pool[(split, family, bucket)]),
            ("same_family_relaxed_answer_type", family_pool[(split, family)]),
        ):
            for candidate_id in pool:
                if candidate_id not in seen:
                    candidates.append((candidate_id, tier))
                    seen.add(candidate_id)
        candidates.sort(
            key=lambda item: sha256_value(
                [DISTRACTOR_POLICY_VERSION, base_fact_id, item[0], item[1]]
            )
        )
        selected_answers = set()
        for candidate_id, tier in candidates:
            if len(distractor_ids[base_fact_id]) >= 2:
                break
            if ("distractor", base_fact_id, candidate_id) in blocked_pairs:
                continue
            if candidate_id == base_fact_id:
                continue
            if component_by_fact[candidate_id] == component_by_fact[base_fact_id]:
                continue
            candidate = by_id[candidate_id]
            answer = normalize_text(candidate.get("answer_en"))
            if not answer or answer == source_answer or answer in selected_answers:
                continue
            selected_answers.add(answer)
            distractor_id = stable_id(
                "full_distractor",
                [base_fact_id, candidate_id, len(distractor_ids[base_fact_id]) + 1],
            )
            distractor_ids[base_fact_id].append(distractor_id)
            distractor_rows.append(
                {
                    "schema_version": DISTRACTOR_SCHEMA_VERSION,
                    "distractor_id": distractor_id,
                    "base_fact_id": base_fact_id,
                    "slot": len(distractor_ids[base_fact_id]),
                    "source_base_fact_id": candidate_id,
                    "distractor_text_en": candidate.get("answer_en"),
                    "relation_partition_id": family,
                    "answer_type_match_tier": tier,
                    "split_assignment": split,
                    "same_split": True,
                    "same_leakage_component": False,
                    "verified": False,
                    "review_status": "pending_factual_uniqueness_and_semantic_review",
                }
            )
        distractor_count_by_fact[len(distractor_ids[base_fact_id])] += 1

        neutral_candidates = []
        for candidate_id in split_pool[split]:
            if ("neutral", base_fact_id, candidate_id) in blocked_pairs:
                continue
            if candidate_id == base_fact_id:
                continue
            candidate_relation = relation_by_fact[candidate_id]
            if candidate_relation["relation_partition_id"] == family:
                continue
            if component_by_fact[candidate_id] == component_by_fact[base_fact_id]:
                continue
            candidate = by_id[candidate_id]
            if normalize_text(candidate.get("answer_en")) == source_answer:
                continue
            if normalize_text(candidate.get("subject_en")) == source_subject:
                continue
            neutral_candidates.append(candidate_id)
        neutral_candidates.sort(
            key=lambda candidate_id: sha256_value(
                [NEUTRAL_POLICY_VERSION, base_fact_id, candidate_id]
            )
        )
        for candidate_id in neutral_candidates[:2]:
            neutral_id = stable_id(
                "full_neutral",
                [base_fact_id, candidate_id, len(neutral_ids[base_fact_id]) + 1],
            )
            neutral_ids[base_fact_id].append(neutral_id)
            candidate = by_id[candidate_id]
            neutral_rows.append(
                {
                    "schema_version": NEUTRAL_SCHEMA_VERSION,
                    "neutral_candidate_id": neutral_id,
                    "base_fact_id": base_fact_id,
                    "slot": len(neutral_ids[base_fact_id]),
                    "source_base_fact_id": candidate_id,
                    "neutral_context_candidate_en": candidate.get("canonical_fact_en"),
                    "source_relation_partition_id": relation_by_fact[candidate_id]["relation_partition_id"],
                    "target_relation_partition_id": family,
                    "split_assignment": split,
                    "same_split": True,
                    "same_leakage_component": False,
                    "verified_unrelated": False,
                    "review_status": "pending_unrelatedness_and_length_review",
                }
            )
        neutral_count_by_fact[len(neutral_ids[base_fact_id])] += 1

    summary = {
        "distractor_policy_version": DISTRACTOR_POLICY_VERSION,
        "neutral_policy_version": NEUTRAL_POLICY_VERSION,
        "distractor_candidate_count": len(distractor_rows),
        "neutral_candidate_count": len(neutral_rows),
        "distractor_count_distribution": {
            str(count): facts for count, facts in sorted(distractor_count_by_fact.items())
        },
        "neutral_count_distribution": {
            str(count): facts for count, facts in sorted(neutral_count_by_fact.items())
        },
        "dual_distractor_fact_count": distractor_count_by_fact[2],
        "dual_neutral_fact_count": neutral_count_by_fact[2],
        "verified_distractor_count": 0,
        "verified_neutral_count": 0,
    }
    return distractor_rows, neutral_rows, {**summary, "distractor_ids": distractor_ids, "neutral_ids": neutral_ids}


def build_relation_inventory(
    assignments: Sequence[Mapping[str, Any]],
    behavior_by_fact: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    families: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    signatures = set()
    primary_signatures = set()
    status_counts = Counter()
    for assignment in assignments:
        family = str(assignment["relation_partition_id"])
        families[family].append(assignment)
        primary_signature = str(assignment.get("relation_signature_id") or "")
        if primary_signature:
            primary_signatures.add(primary_signature)
        for signature_id in assignment.get("relation_signature_ids") or [primary_signature]:
            if signature_id:
                signatures.add(str(signature_id))
        status_counts[str(assignment["relation_partition_status"])] += 1
    entries = []
    for family in sorted(families):
        rows = families[family]
        raw_counts = Counter(str(item.get("relation_raw") or "") for item in rows)
        source_counts = Counter(
            str(behavior_by_fact[str(item["base_fact_id"])].get("source_dataset") or "unknown")
            for item in rows
        )
        entries.append(
            {
                "relation_partition_id": family,
                "base_fact_count": len(rows),
                "signature_count": len({str(item.get("relation_signature_id") or "") for item in rows}),
                "assignment_status_counts": dict(
                    sorted(Counter(str(item["relation_partition_status"]) for item in rows).items())
                ),
                "source_dataset_counts": dict(sorted(source_counts.items())),
                "top_relation_raw": [
                    {"relation_raw": raw, "base_fact_count": count}
                    for raw, count in sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
                ],
                "example_base_fact_ids": sorted(str(item["base_fact_id"]) for item in rows)[:10],
                "formal_probe_relation_id": False,
            }
        )
    return {
        "schema_version": RELATION_INVENTORY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "base_fact_count": len(assignments),
        "relation_signature_count": len(signatures),
        "primary_relation_signature_count": len(primary_signatures),
        "relation_partition_count": len(entries),
        "relation_partition_status_counts": dict(sorted(status_counts.items())),
        "formal_relation_normalization_performed": False,
        "probe_relation_freeze_performed": False,
        "entries": entries,
    }


def build_full_fact_rows(
    behavior_rows: Sequence[Mapping[str, Any]],
    relation_by_fact: Mapping[str, Mapping[str, Any]],
    component_by_fact: Mapping[str, str],
    split_by_fact: Mapping[str, str],
) -> List[Dict[str, Any]]:
    output = []
    for source in sorted(behavior_rows, key=lambda row: str(row["base_fact_id"])):
        base_fact_id = str(source["base_fact_id"])
        relation = relation_by_fact[base_fact_id]
        output.append(
            {
                "schema_version": FULL_FACT_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "candidate_id": source.get("candidate_id"),
                "source_pool_id": source.get("source_pool_id"),
                "source_dataset": source.get("source_dataset"),
                "source_subset": source.get("source_subset"),
                "source_id": source.get("source_id"),
                "source_format": source.get("source_format"),
                "source_question_en": source.get("source_question_en"),
                "source_choices_en": source.get("source_choices_en") or [],
                "subject_en": source.get("subject_en"),
                "answer_en": source.get("answer_en"),
                "answer_aliases_en": source.get("answer_aliases_en") or [],
                "answer_type": source.get("answer_type"),
                "canonical_fact_en": source.get("canonical_fact_en"),
                "prompt_en": source.get("prompt_en"),
                "prompt_quality_tier": source.get("prompt_quality_tier"),
                "prompt_risk_flags": source.get("prompt_risk_flags") or [],
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "relation_raw": source.get("relation_raw"),
                "relation_signature_id": source.get("relation_signature_id"),
                "relation_signature_ids": relation.get("relation_signature_ids") or [source.get("relation_signature_id")],
                "relation_normalized": source.get("relation_normalized"),
                "probe_relation_candidate": source.get("probe_relation_candidate"),
                "probe_relation_id": source.get("probe_relation_id"),
                "relation_partition_id": relation["relation_partition_id"],
                "relation_partition_status": relation["relation_partition_status"],
                "answer_type_bucket": relation["answer_type_bucket"],
                "leakage_component_id": component_by_fact[base_fact_id],
                "split_assignment": split_by_fact[base_fact_id],
                "split_policy_version": SPLIT_POLICY_VERSION,
                "split_status": "provisional_not_frozen",
                "translation_status": "source_language_only_or_pending",
                "distractor_status": "candidate_generation_only",
                "hf_model_execution_status": "not_run",
                "hf_tokenizer_execution_status": "not_run",
            }
        )
    return output


def output_binding(path: Path, schema_version: str, record_count: Optional[int] = None) -> Dict[str, Any]:
    value = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def audit_written_outputs(
    *,
    paths: Mapping[str, Path],
    source_dir: Path,
    frozen_split_bundle_path: Optional[Path],
) -> Dict[str, Any]:
    """Re-read materialized rows and fail closed on structural or leakage drift."""

    source_behavior = read_jsonl(source_dir.resolve() / "behavior_input_bundle.jsonl")
    source_by_id = _index_unique(source_behavior, "base_fact_id", "source behavior")
    full_rows = read_jsonl(paths["full_base_facts"])
    relation_rows = read_jsonl(paths["relation_assignments"])
    component_rows = read_jsonl(paths["leakage_components"])
    availability_rows = read_jsonl(paths["multilingual_availability"])
    multilingual_rows = read_jsonl(paths["multilingual_preperturbation_rows"])
    distractor_rows = read_jsonl(paths["distractor_candidates"], allow_empty=True)
    neutral_rows = read_jsonl(paths["neutral_reference_candidates"], allow_empty=True)
    static_rows = read_jsonl(paths["pre_hf_static_contract"])
    shortfall_rows = read_jsonl(paths["candidate_shortfalls"], allow_empty=True)

    full_by_id = _index_unique(full_rows, "base_fact_id", "full_base_facts")
    relation_by_id = _index_unique(relation_rows, "base_fact_id", "relation_assignments")
    availability_by_id = _index_unique(
        availability_rows, "base_fact_id", "multilingual_availability"
    )
    static_by_id = _index_unique(static_rows, "base_fact_id", "pre_hf_static_contract")
    universe = set(source_by_id)
    named_universes = {
        "full_base_facts": set(full_by_id),
        "relation_assignments": set(relation_by_id),
        "multilingual_availability": set(availability_by_id),
        "pre_hf_static_contract": set(static_by_id),
    }
    for label, candidate_universe in named_universes.items():
        if candidate_universe != universe:
            raise ValueError(f"Integrity audit universe mismatch: {label}")

    component_by_fact: Dict[str, str] = {}
    component_split_by_id: Dict[str, str] = {}
    for component in component_rows:
        component_id = str(component.get("leakage_component_id") or "")
        split = str(component.get("split_assignment") or "")
        if not component_id or split not in SPLITS:
            raise ValueError("Integrity audit found invalid leakage component")
        if component_id in component_split_by_id:
            raise ValueError(f"Duplicate leakage component ID: {component_id}")
        component_split_by_id[component_id] = split
        members = component.get("base_fact_ids")
        if not isinstance(members, list) or len(members) != component.get("member_count"):
            raise ValueError(f"Invalid component members: {component_id}")
        for base_fact_id in members:
            base_fact_id = str(base_fact_id)
            if base_fact_id in component_by_fact:
                raise ValueError(f"Fact occurs in multiple leakage components: {base_fact_id}")
            component_by_fact[base_fact_id] = component_id
            if full_by_id[base_fact_id]["split_assignment"] != split:
                raise ValueError(f"Component split mismatch: {base_fact_id}")
    if set(component_by_fact) != universe:
        raise ValueError("Leakage components do not cover the full universe exactly once")

    source_group_to_components: MutableMapping[str, set] = defaultdict(set)
    source_group_to_splits: MutableMapping[str, set] = defaultdict(set)
    for base_fact_id, source in source_by_id.items():
        source_group = str(source.get("split_group_id") or "")
        source_group_to_components[source_group].add(component_by_fact[base_fact_id])
        source_group_to_splits[source_group].add(full_by_id[base_fact_id]["split_assignment"])
    source_group_component_violations = sum(len(values) != 1 for values in source_group_to_components.values())
    source_group_split_violations = sum(len(values) != 1 for values in source_group_to_splits.values())
    if source_group_component_violations or source_group_split_violations:
        raise ValueError("Source normalized-answer group crosses components or splits")

    frozen_mismatches = 0
    frozen_count = 0
    if frozen_split_bundle_path is not None:
        frozen_rows = read_jsonl(frozen_split_bundle_path.resolve())
        frozen_count = len(frozen_rows)
        for frozen in frozen_rows:
            base_fact_id = str(frozen["base_fact_id"])
            if full_by_id[base_fact_id]["split_assignment"] != frozen["split_assignment"]:
                frozen_mismatches += 1
    if frozen_mismatches:
        raise ValueError("Frozen 160 split constraint mismatch")

    distractor_by_fact: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    distractor_ids = set()
    for row in distractor_rows:
        distractor_id = str(row.get("distractor_id") or "")
        base_fact_id = str(row.get("base_fact_id") or "")
        source_base_fact_id = str(row.get("source_base_fact_id") or "")
        if distractor_id in distractor_ids or not distractor_id:
            raise ValueError(f"Duplicate or empty distractor ID: {distractor_id}")
        distractor_ids.add(distractor_id)
        if base_fact_id not in universe or source_base_fact_id not in universe:
            raise ValueError(f"Distractor references unknown fact: {distractor_id}")
        if full_by_id[base_fact_id]["split_assignment"] != full_by_id[source_base_fact_id]["split_assignment"]:
            raise ValueError(f"Distractor crosses splits: {distractor_id}")
        if component_by_fact[base_fact_id] == component_by_fact[source_base_fact_id]:
            raise ValueError(f"Distractor shares leakage component: {distractor_id}")
        if relation_by_id[base_fact_id]["relation_partition_id"] != relation_by_id[source_base_fact_id]["relation_partition_id"]:
            raise ValueError(f"Distractor crosses relation partitions: {distractor_id}")
        if row.get("verified") is not False:
            raise ValueError(f"Unreviewed distractor was promoted: {distractor_id}")
        distractor_by_fact[base_fact_id].append(row)
    for base_fact_id, rows in distractor_by_fact.items():
        if len(rows) > 2 or len({normalize_text(row.get("distractor_text_en")) for row in rows}) != len(rows):
            raise ValueError(f"Invalid dual distractors for {base_fact_id}")

    neutral_by_fact: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    neutral_ids = set()
    for row in neutral_rows:
        neutral_id = str(row.get("neutral_candidate_id") or "")
        base_fact_id = str(row.get("base_fact_id") or "")
        source_base_fact_id = str(row.get("source_base_fact_id") or "")
        if neutral_id in neutral_ids or not neutral_id:
            raise ValueError(f"Duplicate or empty neutral candidate ID: {neutral_id}")
        neutral_ids.add(neutral_id)
        if base_fact_id not in universe or source_base_fact_id not in universe:
            raise ValueError(f"Neutral candidate references unknown fact: {neutral_id}")
        if full_by_id[base_fact_id]["split_assignment"] != full_by_id[source_base_fact_id]["split_assignment"]:
            raise ValueError(f"Neutral candidate crosses splits: {neutral_id}")
        if component_by_fact[base_fact_id] == component_by_fact[source_base_fact_id]:
            raise ValueError(f"Neutral candidate shares leakage component: {neutral_id}")
        if relation_by_id[base_fact_id]["relation_partition_id"] == relation_by_id[source_base_fact_id]["relation_partition_id"]:
            raise ValueError(f"Neutral candidate shares relation partition: {neutral_id}")
        if row.get("verified_unrelated") is not False:
            raise ValueError(f"Unreviewed neutral candidate was promoted: {neutral_id}")
        neutral_by_fact[base_fact_id].append(row)
    if any(len(neutral_by_fact[base_fact_id]) > 2 for base_fact_id in universe):
        raise ValueError("A fact has more than two neutral reference candidates")

    registry_audit = read_json(paths["language_registry"])
    language_codes = {
        str(item["language_code"]) for item in registry_audit.get("languages", [])
    }
    multilingual_keys = set()
    for row in multilingual_rows:
        key = (str(row["base_fact_id"]), str(row["language_code"]))
        if key in multilingual_keys:
            raise ValueError(f"Duplicate multilingual row: {key}")
        multilingual_keys.add(key)
        if row["split_assignment"] != full_by_id[key[0]]["split_assignment"]:
            raise ValueError(f"Multilingual row split mismatch: {key}")
    for base_fact_id, availability in availability_by_id.items():
        if set(availability.get("languages", {})) != language_codes:
            raise ValueError(f"Language universe mismatch: {base_fact_id}")
        if availability["split_assignment"] != full_by_id[base_fact_id]["split_assignment"]:
            raise ValueError(f"Multilingual availability split mismatch: {base_fact_id}")

    for base_fact_id, static in static_by_id.items():
        expected_distractors = [row["distractor_id"] for row in sorted(distractor_by_fact[base_fact_id], key=lambda row: row["slot"])]
        expected_neutral = [row["neutral_candidate_id"] for row in sorted(neutral_by_fact[base_fact_id], key=lambda row: row["slot"])]
        if static.get("distractor_candidate_ids") != expected_distractors:
            raise ValueError(f"Static contract distractor references mismatch: {base_fact_id}")
        if static.get("neutral_reference_candidate_ids") != expected_neutral:
            raise ValueError(f"Static contract neutral references mismatch: {base_fact_id}")
        expected_complete = len(expected_distractors) == 2 and len(expected_neutral) == 2
        if static.get("candidate_static_contract_complete") is not expected_complete:
            raise ValueError(f"Static contract completeness mismatch: {base_fact_id}")
        if static.get("hf_behavior_authorized") is not False:
            raise ValueError(f"Static contract improperly authorizes HF behavior: {base_fact_id}")

    expected_shortfall_ids = {
        base_fact_id
        for base_fact_id, static in static_by_id.items()
        if not static["candidate_static_contract_complete"]
    }
    supplied_shortfall_ids = {
        str(row.get("base_fact_id") or "") for row in shortfall_rows
    }
    if len(supplied_shortfall_ids) != len(shortfall_rows) or supplied_shortfall_ids != expected_shortfall_ids:
        raise ValueError("Candidate shortfall rows do not exactly match incomplete static contracts")

    split_counts = Counter(row["split_assignment"] for row in full_rows)
    return {
        "schema_version": INTEGRITY_AUDIT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "all_checks_passed": True,
        "checks": {
            "source_and_output_base_fact_universes_equal": True,
            "base_fact_ids_unique_in_core_outputs": True,
            "leakage_components_cover_each_fact_once": True,
            "source_normalized_answer_groups_do_not_cross_components": True,
            "source_normalized_answer_groups_do_not_cross_splits": True,
            "frozen_160_split_constraints_preserved": True,
            "distractors_are_same_split_same_relation_different_component": True,
            "neutral_references_are_same_split_different_relation_different_component": True,
            "multilingual_rows_inherit_base_fact_split": True,
            "static_contract_references_are_exact": True,
            "candidate_shortfalls_are_exhaustive": True,
            "no_static_contract_authorizes_hf_behavior": True,
        },
        "counts": {
            "source_base_facts": len(universe),
            "full_base_facts": len(full_rows),
            "relation_assignments": len(relation_rows),
            "leakage_components": len(component_rows),
            "multilingual_availability_records": len(availability_rows),
            "multilingual_preperturbation_rows": len(multilingual_rows),
            "distractor_candidates": len(distractor_rows),
            "neutral_reference_candidates": len(neutral_rows),
            "static_contract_records": len(static_rows),
            "candidate_shortfalls": len(shortfall_rows),
            "frozen_split_constraints": frozen_count,
        },
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "violations": {
            "source_group_component": source_group_component_violations,
            "source_group_split": source_group_split_violations,
            "frozen_split": frozen_mismatches,
            "cross_language_split": 0,
            "cross_component_distractor": 0,
            "cross_split_distractor": 0,
            "cross_component_neutral": 0,
            "cross_split_neutral": 0,
        },
    }


def materialize(
    *,
    source_dir: Path,
    output_dir: Path,
    language_registry_path: Path,
    frozen_split_bundle_path: Optional[Path],
    split_seed: str = DEFAULT_SPLIT_SEED,
) -> Dict[str, Any]:
    source_rows, source_meta = load_and_validate_sources(source_dir)
    behavior_rows = source_rows["behavior_input_bundle"]
    behavior_by_fact = _index_unique(behavior_rows, "base_fact_id", "behavior_input_bundle")
    cluster_by_fact = _index_unique(
        source_rows["base_fact_clusters"], "base_fact_id", "base_fact_clusters"
    )
    base_fact_ids = sorted(behavior_by_fact)

    relation_assignments = [assign_relation(behavior_by_fact[base_fact_id]) for base_fact_id in base_fact_ids]
    for assignment in relation_assignments:
        base_fact_id = str(assignment["base_fact_id"])
        assignment["relation_signature_ids"] = list(
            cluster_by_fact[base_fact_id].get("relation_signature_ids")
            or [assignment.get("relation_signature_id")]
        )
    relation_by_fact = {str(row["base_fact_id"]): row for row in relation_assignments}
    relation_inventory = build_relation_inventory(relation_assignments, behavior_by_fact)

    frozen = load_frozen_split_constraints(frozen_split_bundle_path, base_fact_ids)
    components, component_by_fact = build_components(behavior_rows, relation_by_fact, frozen)
    component_assignments, split_audit = assign_component_splits(components, split_seed)
    split_by_fact: Dict[str, str] = {}
    for component in components:
        split = component_assignments[str(component["leakage_component_id"])]
        for base_fact_id in component["base_fact_ids"]:
            split_by_fact[str(base_fact_id)] = split
    if set(split_by_fact) != set(base_fact_ids):
        raise ValueError("Split assignment does not cover the complete base-fact universe")
    frozen_mismatches = [
        base_fact_id
        for base_fact_id, split in frozen["split_by_fact"].items()
        if split_by_fact[base_fact_id] != split
    ]
    if frozen_mismatches:
        raise ValueError(f"Frozen split assignments changed: {frozen_mismatches[:10]}")

    component_sizes = sorted(int(component["member_count"]) for component in components)
    component_size_summary = {
        "maximum": max(component_sizes),
        "median": component_sizes[len(component_sizes) // 2],
        "p95_nearest_rank": component_sizes[max(0, math.ceil(0.95 * len(component_sizes)) - 1)],
        "singleton_component_count": sum(size == 1 for size in component_sizes),
        "multi_fact_component_count": sum(size > 1 for size in component_sizes),
        "at_least_10_fact_component_count": sum(size >= 10 for size in component_sizes),
    }

    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "split_seed": split_seed,
        "split_ratios": SPLIT_RATIOS,
        "split_status": "provisional_not_frozen",
        "formal_split_freeze_performed": False,
        "base_fact_count": len(base_fact_ids),
        "leakage_component_count": len(components),
        "leakage_grouping_sources": [
            "source_normalized_answer_group",
            "preserved_frozen_160_semantic_components",
        ],
        "semantic_paraphrase_closure_complete_for_full_pool": False,
        "historical_exposure_complete_for_full_pool": False,
        "frozen_split_constraint": frozen["binding"],
        "frozen_split_mismatch_count": 0,
        "cross_component_split_violation_count": 0,
        "component_size_summary": component_size_summary,
        **split_audit,
    }

    registry = load_language_registry(language_registry_path)
    language_audit, language_dataset_rows = audit_language_datasets(
        registry, language_registry_path
    )
    multilingual_availability, multilingual_rows, multilingual_summary = build_multilingual_rows(
        behavior_rows,
        registry,
        language_audit,
        language_dataset_rows,
        split_by_fact,
    )

    distractors, neutral_candidates, static_selection = select_static_candidates(
        behavior_rows,
        relation_by_fact,
        component_by_fact,
        split_by_fact,
    )
    distractor_ids = static_selection.pop("distractor_ids")
    neutral_ids = static_selection.pop("neutral_ids")
    availability_by_fact = {
        str(row["base_fact_id"]): row for row in multilingual_availability
    }
    static_contract_rows: List[Dict[str, Any]] = []
    for base_fact_id in base_fact_ids:
        available_codes = sorted(
            code
            for code, state in availability_by_fact[base_fact_id]["languages"].items()
            if state["availability_status"]
            in {
                "source_available_provisional",
                "aligned_external_translation_candidate_unreviewed",
            }
        )
        pending_codes = sorted(
            code
            for code, state in availability_by_fact[base_fact_id]["languages"].items()
            if state["availability_status"] == "pending_translation"
        )
        dual_complete = len(distractor_ids[base_fact_id]) == 2 and len(neutral_ids[base_fact_id]) == 2
        static_contract_rows.append(
            {
                "schema_version": STATIC_CONTRACT_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "split_assignment": split_by_fact[base_fact_id],
                "relation_partition_id": relation_by_fact[base_fact_id]["relation_partition_id"],
                "leakage_component_id": component_by_fact[base_fact_id],
                "available_language_codes": available_codes,
                "pending_translation_language_codes": pending_codes,
                "distractor_candidate_ids": list(distractor_ids[base_fact_id]),
                "neutral_reference_candidate_ids": list(neutral_ids[base_fact_id]),
                "planned_arm_contract": {
                    "design": "original_plus_dual_distractor_neutral_targeted",
                    "arm_count_per_available_language": 5,
                    "arms": [
                        "original",
                        "neutral_distractor_1",
                        "targeted_distractor_1",
                        "neutral_distractor_2",
                        "targeted_distractor_2",
                    ],
                },
                "candidate_static_contract_complete": dual_complete,
                "reviewed_static_contract_complete": False,
                "targeted_and_neutral_text_materialized": False,
                "targeted_and_neutral_text_status": "pending_semantic_construction_and_review",
                "hf_behavior_authorized": False,
            }
        )

    candidate_shortfall_rows = []
    for row in static_contract_rows:
        if row["candidate_static_contract_complete"]:
            continue
        reasons = []
        if len(row["distractor_candidate_ids"]) < 2:
            reasons.append("fewer_than_two_same_split_relation_partition_distractors")
        if len(row["neutral_reference_candidate_ids"]) < 2:
            reasons.append("fewer_than_two_same_split_different_relation_neutral_references")
        candidate_shortfall_rows.append(
            {
                "schema_version": SHORTFALL_SCHEMA_VERSION,
                "base_fact_id": row["base_fact_id"],
                "split_assignment": row["split_assignment"],
                "relation_partition_id": row["relation_partition_id"],
                "distractor_candidate_count": len(row["distractor_candidate_ids"]),
                "neutral_reference_candidate_count": len(row["neutral_reference_candidate_ids"]),
                "shortfall_reasons": reasons,
                "fact_excluded_from_full_pool": False,
                "hf_behavior_authorized": False,
            }
        )

    full_fact_rows = build_full_fact_rows(
        behavior_rows, relation_by_fact, component_by_fact, split_by_fact
    )

    output_dir = output_dir.resolve()
    paths = {
        "full_base_facts": output_dir / "full_base_facts.jsonl",
        "relation_assignments": output_dir / "relation_assignments.jsonl",
        "relation_inventory": output_dir / "relation_inventory.json",
        "leakage_components": output_dir / "leakage_components.jsonl",
        "split_manifest": output_dir / "split_manifest.json",
        "language_registry": output_dir / "language_registry.json",
        "multilingual_availability": output_dir / "multilingual_availability.jsonl",
        "multilingual_preperturbation_rows": output_dir / "multilingual_preperturbation_rows.jsonl",
        "distractor_candidates": output_dir / "distractor_candidates.jsonl",
        "neutral_reference_candidates": output_dir / "neutral_reference_candidates.jsonl",
        "pre_hf_static_contract": output_dir / "pre_hf_static_contract.jsonl",
        "candidate_shortfalls": output_dir / "candidate_shortfalls.jsonl",
        "integrity_audit": output_dir / "integrity_audit.json",
        "pre_hf_gate_manifest": output_dir / "pre_hf_gate_manifest.json",
        "summary": output_dir / "summary.json",
    }
    write_jsonl(paths["full_base_facts"], full_fact_rows)
    write_jsonl(paths["relation_assignments"], relation_assignments)
    write_json(paths["relation_inventory"], relation_inventory)
    write_jsonl(paths["leakage_components"], components)
    write_json(paths["split_manifest"], split_manifest)
    write_json(paths["language_registry"], language_audit)
    write_jsonl(paths["multilingual_availability"], multilingual_availability)
    write_jsonl(paths["multilingual_preperturbation_rows"], multilingual_rows)
    write_jsonl(paths["distractor_candidates"], distractors)
    write_jsonl(paths["neutral_reference_candidates"], neutral_candidates)
    write_jsonl(paths["pre_hf_static_contract"], static_contract_rows)
    write_jsonl(paths["candidate_shortfalls"], candidate_shortfall_rows)

    integrity_audit = audit_written_outputs(
        paths=paths,
        source_dir=source_dir,
        frozen_split_bundle_path=frozen_split_bundle_path,
    )
    write_json(paths["integrity_audit"], integrity_audit)

    output_artifacts = {
        "full_base_facts": output_binding(paths["full_base_facts"], FULL_FACT_SCHEMA_VERSION, len(full_fact_rows)),
        "relation_assignments": output_binding(paths["relation_assignments"], RELATION_ASSIGNMENT_SCHEMA_VERSION, len(relation_assignments)),
        "relation_inventory": output_binding(paths["relation_inventory"], RELATION_INVENTORY_SCHEMA_VERSION),
        "leakage_components": output_binding(paths["leakage_components"], COMPONENT_SCHEMA_VERSION, len(components)),
        "split_manifest": output_binding(paths["split_manifest"], SPLIT_MANIFEST_SCHEMA_VERSION),
        "language_registry": output_binding(paths["language_registry"], LANGUAGE_AUDIT_SCHEMA_VERSION),
        "multilingual_availability": output_binding(paths["multilingual_availability"], MULTILINGUAL_AVAILABILITY_SCHEMA_VERSION, len(multilingual_availability)),
        "multilingual_preperturbation_rows": output_binding(paths["multilingual_preperturbation_rows"], MULTILINGUAL_ROW_SCHEMA_VERSION, len(multilingual_rows)),
        "distractor_candidates": output_binding(paths["distractor_candidates"], DISTRACTOR_SCHEMA_VERSION, len(distractors)),
        "neutral_reference_candidates": output_binding(paths["neutral_reference_candidates"], NEUTRAL_SCHEMA_VERSION, len(neutral_candidates)),
        "pre_hf_static_contract": output_binding(paths["pre_hf_static_contract"], STATIC_CONTRACT_SCHEMA_VERSION, len(static_contract_rows)),
        "candidate_shortfalls": output_binding(paths["candidate_shortfalls"], SHORTFALL_SCHEMA_VERSION, len(candidate_shortfall_rows)),
        "integrity_audit": output_binding(paths["integrity_audit"], INTEGRITY_AUDIT_SCHEMA_VERSION),
    }
    unresolved_count = sum(
        row["relation_partition_status"] == "unresolved_partition_only"
        for row in relation_assignments
    )
    pending_translation_count = sum(
        state["availability_status"] == "pending_translation"
        for row in multilingual_availability
        for state in row["languages"].values()
    )
    incomplete_static_count = sum(
        not row["candidate_static_contract_complete"] for row in static_contract_rows
    )
    gate = {
        "schema_version": GATE_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "stage": "complete_provisional_pool_offline_pre_hf",
        "pre_hf_boundary_reached": True,
        "requested_base_fact_universe_complete": len(base_fact_ids) == 8969,
        "base_fact_count": len(base_fact_ids),
        "source_artifacts": source_meta["bindings"],
        "output_artifacts": output_artifacts,
        "offline_completion": {
            "all_base_facts_retained": True,
            "relation_partition_assigned_to_every_fact": len(relation_assignments) == len(base_fact_ids),
            "leakage_component_assigned_to_every_fact": len(component_by_fact) == len(base_fact_ids),
            "development_validation_sealed_assigned_to_every_fact": len(split_by_fact) == len(base_fact_ids),
            "all_registered_languages_accounted_for": True,
            "independent_structural_integrity_audit_passed": integrity_audit["all_checks_passed"],
            "dual_distractor_candidates_generated_where_pool_allows": True,
            "neutral_reference_candidates_generated_where_pool_allows": True,
        },
        "execution_evidence": {
            "hf_model_execution": False,
            "hf_tokenizer_execution": False,
            "behavior_output_count": 0,
            "development_behavior_exposure_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
            "hidden_state_collection_count": 0,
            "intervention_count": 0,
        },
        "authorization": {
            "canonical_freeze_authorized": False,
            "formal_split_freeze_authorized": False,
            "hf_behavior_authorized": False,
            "pnt_mechanism_authorized": False,
            "repair_claim_authorized": False,
        },
        "evidence_boundary": {
            "canonical_status": "pending_review",
            "evidence_tier": "provisional_single_model",
            "human_gold": False,
            "relation_partition_is_formal_normalization": False,
            "external_language_dataset_alignment_is_assumed": False,
            "distractor_candidates_are_verified": False,
            "semantic_paraphrase_closure_complete": False,
            "historical_exposure_complete": False,
        },
        "remaining_blockers_before_formal_hf_behavior": {
            "base_fact_semantic_review_pending_count": len(base_fact_ids),
            "formal_relation_normalization_pending_count": len(base_fact_ids),
            "unresolved_relation_partition_count": unresolved_count,
            "target_translation_pending_fact_language_count": pending_translation_count,
            "verified_distractor_pending_count": len(distractors),
            "verified_neutral_pending_count": len(neutral_candidates),
            "candidate_static_contract_incomplete_fact_count": incomplete_static_count,
            "full_pool_semantic_closure_complete": False,
            "full_pool_historical_exposure_complete": False,
            "split_freeze_complete": False,
            "exact_hf_checkpoint_and_tokenizer_bound": False,
        },
        "next_stage_requires": [
            "fact_and_relation_review_or_explicit_provisional-use policy",
            "translation generation and equivalence review for selected target languages",
            "dual distractor factual-uniqueness and semantic review",
            "neutral unrelatedness and length review",
            "full-pool semantic leakage closure and split freeze",
            "then bind one exact HF checkpoint and tokenizer",
        ],
    }
    write_json(paths["pre_hf_gate_manifest"], gate)
    output_artifacts_with_gate = {
        **output_artifacts,
        "pre_hf_gate_manifest": output_binding(paths["pre_hf_gate_manifest"], GATE_SCHEMA_VERSION),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": "completed_provisional_offline_pre_hf",
        "output_dir": str(output_dir),
        "counts": {
            "input_provisional_triples": len(source_rows["provisional_triples"]),
            "base_facts": len(base_fact_ids),
            "relation_signatures": relation_inventory["relation_signature_count"],
            "primary_relation_signatures": relation_inventory["primary_relation_signature_count"],
            "relation_partitions": relation_inventory["relation_partition_count"],
            "unresolved_relation_partitions": unresolved_count,
            "leakage_components": len(components),
            "multilingual_availability_records": len(multilingual_availability),
            "multilingual_preperturbation_rows": len(multilingual_rows),
            "distractor_candidates": len(distractors),
            "neutral_reference_candidates": len(neutral_candidates),
            "static_contract_records": len(static_contract_rows),
            "candidate_shortfalls": len(candidate_shortfall_rows),
        },
        "split": {
            "policy_version": SPLIT_POLICY_VERSION,
            "base_fact_counts": split_audit["actual_base_fact_counts"],
            "component_counts": split_audit["component_counts"],
            "component_size_summary": component_size_summary,
            "frozen_160_assignment_mismatch_count": 0,
            "cross_component_split_violation_count": 0,
        },
        "relations": {
            "status_counts": relation_inventory["relation_partition_status_counts"],
            "formal_relation_normalization_performed": False,
            "probe_relation_freeze_performed": False,
        },
        "multilingual": multilingual_summary,
        "static_candidates": static_selection,
        "execution": gate["execution_evidence"],
        "formal_hf_behavior_authorized": False,
        "artifacts": output_artifacts_with_gate,
    }
    write_json(paths["summary"], summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    result.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    result.add_argument("--language-registry", default=str(DEFAULT_LANGUAGE_REGISTRY))
    result.add_argument("--frozen-split-bundle", default=str(DEFAULT_FROZEN_SPLIT_BUNDLE))
    result.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    result.add_argument(
        "--do-not-preserve-frozen-160-splits",
        action="store_true",
        help="Do not use the existing frozen 160-fact split as a hard constraint.",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    frozen_path = None if args.do_not_preserve_frozen_160_splits else Path(args.frozen_split_bundle)
    summary = materialize(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        language_registry_path=Path(args.language_registry),
        frozen_split_bundle_path=frozen_path,
        split_seed=str(args.split_seed),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
