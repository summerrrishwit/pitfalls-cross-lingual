#!/usr/bin/env python3
"""Materialize a fresh, pending semantic-closure v2 review packet.

The packet is bound to the current static-G0A staging and resolved rows.  It
combines the active positive pairs preserved by the persistent v3 lineage with
every mechanical cross-group risk pair found on the resolved rows.  Historical
v3 verdicts are retained only as advisory provenance; every newly emitted
decision remains ``pending/defer``.

The five persistent inputs do not contain the complete legacy v1 candidate
universe (notably its historical ``distinct`` candidates), nor do they prove a
fresh full-pool/out-of-cohort search after cohort repair.  Consequently this
producer deliberately emits an incomplete pending manifest.  Its output cannot
pass the static-G0A finalizer until exhaustive candidate generation and fresh
adjudication are completed by a later stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_static_g0a as g0a  # noqa: E402
from scripts import materialize_public_benchmark_duplicate_closure as lexical_closure  # noqa: E402


TOOL_VERSION = "static-g0a-semantic-closure-materializer-v1"
POLICY_VERSION = "resolved-row-semantic-closure-review-v2-pending"
SOURCE_EDGE_SCHEMA = "public-benchmark-semantic-edge-resolution-v1"
SOURCE_SPLIT_SCHEMA = "public-benchmark-semantic-aware-provisional-split-v1"

CANDIDATES_NAME = "semantic_closure_candidates.jsonl"
ADJUDICATIONS_NAME = "semantic_closure_adjudications.template.jsonl"
MANIFEST_NAME = "semantic_closure_manifest.pending.json"
FULL_POOL_REVIEW_NAME = "full_pool_lexical_review.template.jsonl"
FULL_POOL_REVIEW_TEMPLATE_SCHEMA = (
    "public-benchmark-duplicate-closure-review-shard-template-v1"
)

ACTIVE_EDGE_DISPOSITION = "active_semantic_union_edge"
ACTIVE_EDGE_STATUS = "resolved_same_component_and_split"
POSITIVE_RELATIONSHIPS = frozenset({"same_fact", "same_leakage_component"})


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _pair(left: str, right: str) -> Tuple[str, str]:
    if left == right:
        raise ValueError(f"semantic pair repeats one endpoint: {left}")
    return tuple(sorted((left, right)))


def _candidate_id(left: str, right: str) -> str:
    # Pair-only identity remains stable when endpoint content is repaired.  The
    # candidate-record hash still invalidates any decision bound to old content.
    digest = g0a.sha256_value([left, right])
    return f"semantic_pair_{digest[:24]}"


def _endpoint_view(row: Mapping[str, Any]) -> Dict[str, Any]:
    fields = (
        "base_fact_id",
        "candidate_id",
        "probe_relation_id",
        "subject_en",
        "subject_zh",
        "prompt_en",
        "prompt_zh",
        "canonical_fact_en",
        "canonical_fact_zh",
        "answer_en",
        "answer_zh",
        "answer_aliases_en",
        "answer_aliases_zh",
        "split_group_id",
        "split_assignment",
    )
    return {field: copy.deepcopy(row.get(field)) for field in fields}


def _snapshot(path: Path, *, schema_version: str) -> Dict[str, Any]:
    return g0a._binding(Path(path).resolve(), schema_version=schema_version)


def _assert_unchanged(bindings: Sequence[Mapping[str, Any]]) -> None:
    for binding in bindings:
        path = Path(str(binding["path"]))
        if (
            not path.is_file()
            or g0a.sha256_file(path) != binding["sha256"]
            or path.stat().st_size != binding["byte_count"]
        ):
            raise RuntimeError(f"input changed while materializing closure packet: {path}")


def _validate_identity(
    actual: Any,
    expected: Mapping[str, Any],
    label: str,
    *,
    fields: Sequence[str] = ("sha256", "schema_version", "record_count"),
) -> None:
    g0a._validate_binding_identity(actual, expected, label, fields=fields)


def _load_and_validate_inputs(
    *,
    staging_manifest_path: Path,
    resolved_manifest_path: Path,
    v3_finalization_manifest_path: Path,
    v3_semantic_edges_path: Path,
    v3_split_manifest_path: Path,
) -> Dict[str, Any]:
    staging_manifest_path = Path(staging_manifest_path).resolve()
    resolved_manifest_path = Path(resolved_manifest_path).resolve()
    v3_finalization_manifest_path = Path(v3_finalization_manifest_path).resolve()
    v3_semantic_edges_path = Path(v3_semantic_edges_path).resolve()
    v3_split_manifest_path = Path(v3_split_manifest_path).resolve()

    required_snapshots = [
        _snapshot(
            staging_manifest_path, schema_version=g0a.STAGING_MANIFEST_SCHEMA
        ),
        _snapshot(
            resolved_manifest_path,
            schema_version=g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
        ),
        _snapshot(
            v3_finalization_manifest_path,
            schema_version=g0a.EXPECTED_SOURCE_MANIFEST_SCHEMA,
        ),
        _snapshot(v3_semantic_edges_path, schema_version=SOURCE_EDGE_SCHEMA),
        _snapshot(v3_split_manifest_path, schema_version=SOURCE_SPLIT_SCHEMA),
    ]

    staging_manifest, staging_rows, staging_by_id = g0a._load_staging(
        staging_manifest_path
    )
    if len(staging_rows) != g0a.EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"staging contains {len(staging_rows)} rows, expected {g0a.EXPECTED_RECORD_COUNT}"
        )

    resolved_manifest = g0a.read_json(resolved_manifest_path)
    if resolved_manifest.get("schema_version") != g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA:
        raise ValueError("resolved draft manifest schema_version is unsupported")
    if resolved_manifest.get("status") != "resolved_static_draft_not_frozen":
        raise ValueError("resolved draft manifest status is invalid")
    for field in ("static_frozen", "behavior_authorized", "simulation_authorized"):
        if resolved_manifest.get(field) is not False:
            raise ValueError(f"resolved draft manifest must keep {field}=false")
    g0a._validate_file_binding(
        resolved_manifest.get("staging_manifest"),
        owner_path=resolved_manifest_path,
        label="resolved draft staging manifest",
        expected_path=staging_manifest_path,
        expected_schema=g0a.STAGING_MANIFEST_SCHEMA,
    )
    _validate_identity(
        resolved_manifest.get("source_bundle"),
        staging_manifest["source_bundle"],
        "resolved draft source_bundle",
        fields=("sha256", "schema_version", "record_count", "base_fact_ids_sha256"),
    )
    _validate_identity(
        resolved_manifest.get("effective_staging_bundle"),
        staging_manifest["effective_staging_bundle"],
        "resolved draft effective_staging_bundle",
        fields=("sha256", "schema_version", "record_count", "base_fact_ids_sha256"),
    )
    resolved_rows_path, resolved_rows = g0a._validate_file_binding(
        resolved_manifest.get("resolved_rows"),
        owner_path=resolved_manifest_path,
        label="resolved static draft rows",
        expected_schema=g0a.FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    resolved_by_id: Dict[str, Dict[str, Any]] = {}
    for row in resolved_rows:
        base_fact_id = g0a.formal_admission.explicit_base_fact_id(
            row, "resolved static draft row"
        )
        if base_fact_id in resolved_by_id:
            raise ValueError(f"duplicate resolved base_fact_id: {base_fact_id}")
        resolved_by_id[base_fact_id] = row
    if set(resolved_by_id) != set(staging_by_id):
        raise ValueError("resolved rows do not exactly cover the staging cohort")
    if resolved_manifest.get("base_fact_count") != len(resolved_rows):
        raise ValueError("resolved draft base_fact_count is stale")
    if resolved_manifest["resolved_rows"].get(
        "base_fact_ids_sha256"
    ) != g0a.sha256_value(sorted(resolved_by_id)):
        raise ValueError("resolved draft base_fact_ids_sha256 is stale")

    finalization = g0a.read_json(v3_finalization_manifest_path)
    if finalization.get("schema_version") != g0a.EXPECTED_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("v3 finalization manifest schema_version is unsupported")
    if finalization.get("status") != g0a.EXPECTED_SOURCE_STATUS:
        raise ValueError("v3 finalization manifest status is invalid")
    final_artifacts = _required_mapping(
        finalization.get("artifacts"), "v3 finalization artifacts"
    )
    source_binding = _required_mapping(
        final_artifacts.get("postclosure_preperturbation_behavior_input_bundle"),
        "v3 finalization source bundle",
    )
    source_path, source_rows = g0a._validate_file_binding(
        source_binding,
        owner_path=v3_finalization_manifest_path,
        label="v3 finalization source bundle",
        expected_schema=g0a.FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    if source_path != Path(staging_manifest["source_bundle"]["path"]).resolve():
        raise ValueError("v3 finalization source bundle points to the wrong artifact")
    _validate_identity(
        source_binding,
        staging_manifest["source_bundle"],
        "v3 finalization source bundle",
    )
    if len(source_rows) != len(resolved_rows):
        raise ValueError("v3 finalization source bundle count differs from resolved cohort")

    _, edge_rows = g0a._validate_file_binding(
        final_artifacts.get("semantic_edge_resolutions"),
        owner_path=v3_finalization_manifest_path,
        label="v3 semantic edge resolutions",
        expected_path=v3_semantic_edges_path,
        expected_schema=SOURCE_EDGE_SCHEMA,
        jsonl=True,
    )
    g0a._validate_file_binding(
        final_artifacts.get("semantic_aware_split_manifest"),
        owner_path=v3_finalization_manifest_path,
        label="v3 semantic-aware split manifest",
        expected_path=v3_split_manifest_path,
        expected_schema=SOURCE_SPLIT_SCHEMA,
    )

    split_manifest = g0a.read_json(v3_split_manifest_path)
    if split_manifest.get("schema_version") != SOURCE_SPLIT_SCHEMA:
        raise ValueError("v3 semantic-aware split manifest schema_version is unsupported")
    if split_manifest.get("status") != (
        "semantic_positive_edges_component_closed_provisional_not_frozen"
    ):
        raise ValueError("v3 semantic-aware split manifest status is invalid")
    split_artifacts = _required_mapping(
        split_manifest.get("artifacts"), "v3 split artifacts"
    )
    g0a._validate_file_binding(
        split_artifacts.get("semantic_edge_resolutions"),
        owner_path=v3_split_manifest_path,
        label="v3 split semantic edge resolutions",
        expected_path=v3_semantic_edges_path,
        expected_schema=SOURCE_EDGE_SCHEMA,
        jsonl=True,
    )

    active_edges: List[Dict[str, Any]] = []
    retired_edges: List[Dict[str, Any]] = []
    seen_edge_ids = set()
    for index, edge in enumerate(edge_rows, start=1):
        if edge.get("schema_version") != SOURCE_EDGE_SCHEMA:
            raise ValueError(f"v3 semantic edge row {index} schema_version is unsupported")
        edge_id = g0a._required_string(edge.get("edge_id"), "v3 semantic edge_id")
        if edge_id in seen_edge_ids:
            raise ValueError(f"duplicate v3 semantic edge_id: {edge_id}")
        seen_edge_ids.add(edge_id)
        disposition = edge.get("rebind_disposition")
        if disposition == "retired_by_cohort_repair":
            retired_edges.append(edge)
            continue
        if disposition != ACTIVE_EDGE_DISPOSITION:
            raise ValueError(f"unsupported v3 semantic edge disposition: {edge_id}")
        if edge.get("resolution_status") != ACTIVE_EDGE_STATUS:
            raise ValueError(f"active v3 semantic edge has invalid status: {edge_id}")
        if edge.get("decision") not in POSITIVE_RELATIONSHIPS:
            raise ValueError(f"active v3 semantic edge is not positive: {edge_id}")
        left = g0a._required_string(edge.get("left_base_fact_id"), f"{edge_id}.left")
        right = g0a._required_string(edge.get("right_base_fact_id"), f"{edge_id}.right")
        if not set(_pair(left, right)) <= set(resolved_by_id):
            raise ValueError(f"active v3 semantic edge is outside current 160: {edge_id}")
        active_edges.append(edge)

    expected_active = _required_int(
        split_manifest.get("active_positive_edge_count"),
        "v3 active_positive_edge_count",
    )
    expected_retired = _required_int(
        split_manifest.get("retired_positive_edge_count"),
        "v3 retired_positive_edge_count",
    )
    if len(active_edges) != expected_active or len(retired_edges) != expected_retired:
        raise ValueError("v3 semantic edge counts disagree with split manifest")
    if split_manifest.get("semantic_paraphrase_recall_guaranteed") is not False:
        raise ValueError("v3 split manifest unexpectedly claims semantic recall")

    transitive_snapshots = [
        _snapshot(
            Path(staging_manifest["staging_records"]["path"]),
            schema_version=g0a.STAGING_RECORD_SCHEMA,
        ),
        _snapshot(resolved_rows_path, schema_version=g0a.FINAL_BUNDLE_SCHEMA),
        _snapshot(source_path, schema_version=g0a.FINAL_BUNDLE_SCHEMA),
    ]
    legacy_reference = copy.deepcopy(
        _required_mapping(
            _required_mapping(split_manifest.get("inputs"), "v3 split inputs").get(
                "semantic_closure_manifest"
            ),
            "v3 legacy semantic closure reference",
        )
    )
    return {
        "staging_manifest": staging_manifest,
        "staging_by_id": staging_by_id,
        "resolved_manifest": resolved_manifest,
        "resolved_by_id": resolved_by_id,
        "active_edges": active_edges,
        "retired_edges": retired_edges,
        "split_manifest": split_manifest,
        "legacy_reference": legacy_reference,
        "required_snapshots": required_snapshots,
        "transitive_snapshots": transitive_snapshots,
    }


def _build_candidates(
    *,
    staging_by_id: Mapping[str, Mapping[str, Any]],
    resolved_by_id: Mapping[str, Mapping[str, Any]],
    active_edges: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    mechanical_reasons = g0a.formal_admission._mechanical_cross_group_risk_reasons(
        list(resolved_by_id.values())
    )
    evidence_by_pair: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for edge in active_edges:
        pair = _pair(
            g0a._required_string(edge.get("left_base_fact_id"), "edge left"),
            g0a._required_string(edge.get("right_base_fact_id"), "edge right"),
        )
        evidence_by_pair.setdefault(pair, []).append(edge)

    all_pairs = set(mechanical_reasons) | set(evidence_by_pair)
    rows: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for left, right in sorted(all_pairs):
        if left not in staging_by_id or right not in staging_by_id:
            raise ValueError(f"candidate pair is absent from staging rows: {left}|{right}")
        if left not in resolved_by_id or right not in resolved_by_id:
            raise ValueError(f"candidate pair is absent from resolved rows: {left}|{right}")
        reasons: List[str] = []
        if (left, right) in evidence_by_pair:
            reasons.append("v3_active_positive_edge")
            source_counts["v3_active_positive_edge"] += 1
        if (left, right) in mechanical_reasons:
            reasons.append("resolved_mechanical_cross_group_risk")
            source_counts["resolved_mechanical_cross_group_risk"] += 1

        advisory = []
        for edge in sorted(
            evidence_by_pair.get((left, right), []), key=lambda item: str(item["edge_id"])
        ):
            advisory.append(
                {
                    "source_edge_id": edge["edge_id"],
                    "source_edge_record_sha256": g0a.sha256_value(edge),
                    "historical_advisory_relationship": edge["decision"],
                    "historical_advisory_rationale": edge.get("rationale"),
                    "source_candidate_record_sha256": edge.get(
                        "source_candidate_record_sha256"
                    ),
                    "source_decision_record_sha256": edge.get(
                        "source_decision_record_sha256"
                    ),
                    "source_reviewer_type": edge.get("reviewer_type"),
                    "source_human_gold": edge.get("human_gold"),
                    "advisory_only": True,
                    "reused_as_current_decision": False,
                }
            )

        left_staging = staging_by_id[left]
        right_staging = staging_by_id[right]
        left_resolved = resolved_by_id[left]
        right_resolved = resolved_by_id[right]
        rows.append(
            {
                "schema_version": g0a.SEMANTIC_CANDIDATE_SCHEMA,
                "candidate_id": _candidate_id(left, right),
                "left_base_fact_id": left,
                "right_base_fact_id": right,
                "left_source_record_sha256": left_staging["source_record_sha256"],
                "right_source_record_sha256": right_staging["source_record_sha256"],
                "left_effective_source_record_sha256": left_staging[
                    "effective_source_record_sha256"
                ],
                "right_effective_source_record_sha256": right_staging[
                    "effective_source_record_sha256"
                ],
                "left_input_record_sha256": g0a.sha256_value(left_resolved),
                "right_input_record_sha256": g0a.sha256_value(right_resolved),
                "left": _endpoint_view(left_resolved),
                "right": _endpoint_view(right_resolved),
                "generation_reasons": reasons,
                "mechanical_risk_reasons": sorted(
                    mechanical_reasons.get((left, right), set())
                ),
                "v3_advisory_evidence": advisory,
                "requires_fresh_adjudication": True,
                "review_blinded_to_behavior_results": True,
            }
        )
    rows.sort(key=lambda row: row["candidate_id"])
    if not rows:
        raise ValueError("semantic closure candidate packet cannot be empty")
    return rows, dict(sorted(source_counts.items()))


def _build_templates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "schema_version": g0a.SEMANTIC_DECISION_SCHEMA,
            "candidate_id": row["candidate_id"],
            "candidate_record_sha256": g0a.sha256_value(row),
            "terminal_status": "pending",
            "decision": "defer",
            "rationale": "",
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "review_blinded_to_behavior_results": True,
        }
        for row in candidates
    ]


def _build_full_pool_templates(
    pairs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Create fillable shards accepted by the existing adjudication materializer."""

    templates: List[Dict[str, Any]] = []
    seen = set()
    for index, pair in enumerate(pairs, start=1):
        pair_id = g0a._required_string(pair.get("pair_id"), f"fresh pair {index}.pair_id")
        if pair_id in seen:
            raise ValueError(f"duplicate fresh lexical pair_id: {pair_id}")
        seen.add(pair_id)
        endpoints: Dict[str, Mapping[str, Any]] = {}
        for side in ("left", "right"):
            endpoint = _required_mapping(pair.get(side), f"{pair_id}.{side}")
            provenance = _required_mapping(
                endpoint.get("provenance"), f"{pair_id}.{side}.provenance"
            )
            endpoints[side] = provenance
        templates.append(
            {
                "pair_id": pair_id,
                "decision": "defer",
                "rationale": "",
                "reviewer_id": "",
                "confidence": "",
            }
        )
    templates.sort(key=lambda row: row["pair_id"])
    return templates


def _load_fresh_lexical_closure(
    manifest_path: Path,
    *,
    resolved_manifest: Mapping[str, Any],
    resolved_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest, cohort_rows, pairs = lexical_closure._validate_candidate_manifest(
        manifest_path
    )
    inputs = _required_mapping(manifest.get("inputs"), "fresh closure inputs")
    cohort_binding = _required_mapping(
        inputs.get("cohort_bundle"), "fresh closure cohort bundle"
    )
    _validate_identity(
        cohort_binding,
        _required_mapping(
            resolved_manifest.get("resolved_rows"), "resolved rows binding"
        ),
        "fresh closure resolved cohort",
        fields=("sha256", "schema_version", "record_count"),
    )
    if Path(str(cohort_binding["path"])).resolve() != Path(
        str(resolved_manifest["resolved_rows"]["path"])
    ).resolve():
        raise ValueError("fresh closure cohort does not point to resolved rows")
    if cohort_rows != list(resolved_by_id.values()):
        raise ValueError("fresh closure cohort rows differ from resolved rows")

    closure = _required_mapping(manifest.get("closure"), "fresh closure statistics")
    if closure.get("closure_pair_count") != len(pairs):
        raise ValueError("fresh closure pair count is stale")
    pair_binding = _required_mapping(
        closure.get("candidate_pairs"), "fresh closure candidate pairs"
    )
    pair_path = Path(str(pair_binding["path"])).resolve()
    templates = _build_full_pool_templates(pairs)

    scope_counts = Counter(str(pair.get("audit_scope")) for pair in pairs)
    cohort_ids = set(resolved_by_id)
    endpoint_scope_counts: Counter[str] = Counter()
    current_cohort_pairs = set()
    for pair in pairs:
        endpoint_ids: List[Optional[str]] = []
        comparison_present = False
        for side in ("left", "right"):
            provenance = pair[side]["provenance"]
            if provenance.get("input_role") == "comparison_canonical":
                comparison_present = True
                endpoint_ids.append(None)
            else:
                endpoint_ids.append(str(provenance.get("base_fact_id")))
        if comparison_present:
            endpoint_scope_counts["contains_comparison_canonical"] += 1
        else:
            cohort_endpoint_count = sum(value in cohort_ids for value in endpoint_ids)
            endpoint_scope_counts[
                {0: "out_of_cohort_to_out_of_cohort", 1: "cohort_to_out_of_cohort", 2: "cohort_to_cohort"}[
                    cohort_endpoint_count
                ]
            ] += 1
            if cohort_endpoint_count == 2:
                current_cohort_pairs.add(tuple(sorted(str(value) for value in endpoint_ids)))

    snapshots = [_snapshot(manifest_path, schema_version=lexical_closure.CANDIDATE_MANIFEST_SCHEMA_VERSION)]
    for binding, schema in (
        (inputs["authoritative_source_manifest"], lexical_closure.SOURCE_UNIVERSE_MANIFEST_SCHEMA_VERSION),
        (inputs["full_pool"], lexical_closure.INPUT_BUNDLE_SCHEMA_VERSION),
        (inputs["comparison_canonical"], str(inputs["comparison_canonical"].get("schema_version") or "comparison-canonical-unspecified")),
        (manifest["reconstruction"], lexical_closure.INPUT_BUNDLE_SCHEMA_VERSION),
        (pair_binding, lexical_closure.AUDIT_PAIR_SCHEMA_VERSION),
    ):
        snapshots.append(_snapshot(Path(str(binding["path"])), schema_version=schema))
    return {
        "manifest": manifest,
        "manifest_binding": snapshots[0],
        "pair_binding": copy.deepcopy(pair_binding),
        "pairs": pairs,
        "templates": templates,
        "scope_counts": dict(sorted(scope_counts.items())),
        "endpoint_scope_counts": dict(sorted(endpoint_scope_counts.items())),
        "current_cohort_pairs": current_cohort_pairs,
        "snapshots": snapshots,
    }


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(g0a.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def materialize_semantic_closure(
    *,
    staging_manifest_path: Path,
    resolved_manifest_path: Path,
    v3_finalization_manifest_path: Path,
    v3_semantic_edges_path: Path,
    v3_split_manifest_path: Path,
    output_dir: Path,
    fresh_lexical_candidate_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    loaded = _load_and_validate_inputs(
        staging_manifest_path=staging_manifest_path,
        resolved_manifest_path=resolved_manifest_path,
        v3_finalization_manifest_path=v3_finalization_manifest_path,
        v3_semantic_edges_path=v3_semantic_edges_path,
        v3_split_manifest_path=v3_split_manifest_path,
    )
    fresh_lexical: Optional[Dict[str, Any]] = None
    if fresh_lexical_candidate_manifest_path is not None:
        fresh_lexical = _load_fresh_lexical_closure(
            fresh_lexical_candidate_manifest_path,
            resolved_manifest=loaded["resolved_manifest"],
            resolved_by_id=loaded["resolved_by_id"],
        )
    candidates, source_counts = _build_candidates(
        staging_by_id=loaded["staging_by_id"],
        resolved_by_id=loaded["resolved_by_id"],
        active_edges=loaded["active_edges"],
    )
    templates = _build_templates(candidates)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent))
    )
    try:
        candidates_path = temporary_dir / CANDIDATES_NAME
        adjudications_path = temporary_dir / ADJUDICATIONS_NAME
        full_pool_review_path = temporary_dir / FULL_POOL_REVIEW_NAME
        manifest_path = temporary_dir / MANIFEST_NAME
        candidates_path.write_bytes(_jsonl_payload(candidates))
        adjudications_path.write_bytes(_jsonl_payload(templates))
        if fresh_lexical is not None:
            full_pool_review_path.write_bytes(
                _jsonl_payload(fresh_lexical["templates"])
            )

        staging_manifest = loaded["staging_manifest"]
        resolved_manifest = loaded["resolved_manifest"]
        split_manifest = loaded["split_manifest"]
        legacy_reference = loaded["legacy_reference"]
        blockers = [
            "fresh_semantic_adjudications_pending",
            "candidate_generation_not_exhaustive",
            "legacy_v1_distinct_candidate_universe_not_reconstructed",
            "replacement_all_pairs_comparison_not_reexecuted",
            "semantic_near_duplicate_recall_not_guaranteed",
        ]
        if fresh_lexical is None:
            blockers.extend(
                [
                    "fresh_full_pool_lexical_generation_missing",
                    "full_pool_out_of_cohort_bridge_search_not_reexecuted",
                ]
            )
        else:
            blockers.append("fresh_full_pool_lexical_adjudications_pending")
        manifest = {
            "schema_version": g0a.SEMANTIC_CLOSURE_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "created_at": g0a.utc_now(),
            "status": "pending_incomplete_candidate_generation_and_adjudication",
            "policy_version": POLICY_VERSION,
            "input_bundle": copy.deepcopy(staging_manifest["source_bundle"]),
            "effective_staging_bundle": copy.deepcopy(
                staging_manifest["effective_staging_bundle"]
            ),
            "resolved_static_draft": copy.deepcopy(resolved_manifest["resolved_rows"]),
            "staging_manifest": copy.deepcopy(loaded["required_snapshots"][0]),
            "resolved_static_draft_manifest": copy.deepcopy(
                loaded["required_snapshots"][1]
            ),
            "upstream_v3": {
                "finalization_manifest": copy.deepcopy(
                    loaded["required_snapshots"][2]
                ),
                "semantic_edge_resolutions": copy.deepcopy(
                    loaded["required_snapshots"][3]
                ),
                "semantic_aware_split_manifest": copy.deepcopy(
                    loaded["required_snapshots"][4]
                ),
                "source_semantic_scope": split_manifest.get("source_semantic_scope"),
                "semantic_paraphrase_recall_guaranteed": split_manifest.get(
                    "semantic_paraphrase_recall_guaranteed"
                ),
                "active_positive_edge_count": len(loaded["active_edges"]),
                "retired_positive_edge_count": len(loaded["retired_edges"]),
                "legacy_v1_semantic_closure_reference": {
                    **legacy_reference,
                    "reference_copied_from_v3_manifest_only": True,
                    "artifact_read_by_this_tool": False,
                    "candidate_rows_reused": False,
                    "verdicts_reused_as_current_decisions": False,
                },
            },
            "coverage": {
                "current_resolved_cohort_size": len(loaded["resolved_by_id"]),
                "v3_active_positive_pairs_in_current_cohort": len(
                    loaded["active_edges"]
                ),
                "resolved_mechanical_cross_group_pair_count": len(
                    g0a.formal_admission._mechanical_cross_group_risk_pairs(
                        list(loaded["resolved_by_id"].values())
                    )
                ),
                "materialized_pair_count": len(candidates),
                "candidate_source_counts": source_counts,
                "legacy_distinct_candidate_rows_available_in_bound_v3_edges": 0,
                "full_pool_search_reexecuted": False,
                "out_of_cohort_bridge_search_reexecuted": False,
                "replacement_record_all_other_current_rows_comparison_reexecuted": False,
            },
            "retired_edge_lineage": [
                {
                    "source_edge_id": edge["edge_id"],
                    "source_edge_record_sha256": g0a.sha256_value(edge),
                    "left_base_fact_id": edge["left_base_fact_id"],
                    "right_base_fact_id": edge["right_base_fact_id"],
                    "left_in_current_resolved_cohort": edge["left_base_fact_id"]
                    in loaded["resolved_by_id"],
                    "right_in_current_resolved_cohort": edge["right_base_fact_id"]
                    in loaded["resolved_by_id"],
                    "historical_advisory_relationship": edge["decision"],
                    "source_candidate_record_sha256": edge.get(
                        "source_candidate_record_sha256"
                    ),
                    "source_decision_record_sha256": edge.get(
                        "source_decision_record_sha256"
                    ),
                    "retired_from_current_candidates": True,
                    "advisory_only": True,
                }
                for edge in sorted(
                    loaded["retired_edges"], key=lambda item: str(item["edge_id"])
                )
            ],
            "cohort_replacement_lineage": [
                {
                    "replacement_base_fact_id": base_fact_id,
                    "replaces_base_fact_id": lineage["replaces_base_fact_id"],
                    "same_fact_representative_id": lineage.get(
                        "same_fact_representative_id"
                    ),
                    "replacement_input_record_sha256": g0a.sha256_value(row),
                    "all_other_current_rows_comparison_reexecuted": False,
                }
                for base_fact_id, row in sorted(loaded["resolved_by_id"].items())
                for lineage in [row.get("cohort_repair_lineage")]
                if isinstance(lineage, dict)
                and lineage.get("disposition") == "replacement"
                and isinstance(lineage.get("replaces_base_fact_id"), str)
            ],
            "candidates": g0a._binding(
                candidates_path,
                schema_version=g0a.SEMANTIC_CANDIDATE_SCHEMA,
                record_count=len(candidates),
                id_values=[row["candidate_id"] for row in candidates],
                id_digest_field="candidate_ids_sha256",
            ),
            "adjudications": g0a._binding(
                adjudications_path,
                schema_version=g0a.SEMANTIC_DECISION_SCHEMA,
                record_count=len(templates),
                id_values=[row["candidate_id"] for row in templates],
                id_digest_field="candidate_ids_sha256",
            ),
            "review_blinded_to_behavior_results": True,
            "behavior_result_count": 0,
            "semantic_paraphrase_closure_complete": False,
            "candidate_generation_complete": False,
            "cohort_scope_complete": False,
            "out_of_cohort_bridge_search_complete": False,
            "unresolved_candidate_count": len(templates),
            "known_blockers": blockers,
            "static_frozen": False,
            "behavior_authorized": False,
            "simulation_authorized": False,
        }
        if fresh_lexical is not None:
            active_pair_keys = {
                _pair(
                    str(edge["left_base_fact_id"]),
                    str(edge["right_base_fact_id"]),
                )
                for edge in loaded["active_edges"]
            }
            fresh_overlap_count = len(
                active_pair_keys & fresh_lexical["current_cohort_pairs"]
            )
            fresh_binding = g0a._binding(
                full_pool_review_path,
                schema_version=FULL_POOL_REVIEW_TEMPLATE_SCHEMA,
                record_count=len(fresh_lexical["templates"]),
                id_values=[row["pair_id"] for row in fresh_lexical["templates"]],
                id_digest_field="pair_ids_sha256",
            )
            fresh_binding["path"] = str(output_dir / FULL_POOL_REVIEW_NAME)
            manifest["fresh_full_pool_lexical_closure"] = {
                "candidate_manifest": copy.deepcopy(
                    fresh_lexical["manifest_binding"]
                ),
                "candidate_pairs": copy.deepcopy(fresh_lexical["pair_binding"]),
                "review_shard_template": fresh_binding,
                "candidate_pair_count": len(fresh_lexical["pairs"]),
                "unresolved_candidate_count": len(fresh_lexical["templates"]),
                "audit_scope_counts": fresh_lexical["scope_counts"],
                "endpoint_scope_counts": fresh_lexical[
                    "endpoint_scope_counts"
                ],
                "overlap_with_v3_active_positive_pair_count": fresh_overlap_count,
                "combined_unique_pair_count": (
                    len(fresh_lexical["pairs"])
                    + len(active_pair_keys)
                    - fresh_overlap_count
                ),
                "full_pool_lexical_search_reexecuted": True,
                "out_of_cohort_lexical_bridge_search_reexecuted": True,
                "comparison_canonical_lexical_search_reexecuted": True,
                "semantic_equivalence_decisions_performed": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "old_verdicts_reused": False,
            }
            manifest["coverage"].update(
                {
                    "full_pool_search_reexecuted": True,
                    "out_of_cohort_bridge_search_reexecuted": True,
                    "fresh_full_pool_search_kind": "bounded_deterministic_lexical_candidate_generation",
                    "fresh_full_pool_candidate_pair_count": len(
                        fresh_lexical["pairs"]
                    ),
                }
            )
        # The staged paths must become the final paths before the manifest is
        # serialized; hashes are content hashes and remain unchanged on rename.
        manifest["candidates"]["path"] = str(output_dir / CANDIDATES_NAME)
        manifest["adjudications"]["path"] = str(output_dir / ADJUDICATIONS_NAME)
        manifest_path.write_bytes(_json_payload(manifest))

        _assert_unchanged(
            [
                *loaded["required_snapshots"],
                *loaded["transitive_snapshots"],
                *(fresh_lexical["snapshots"] if fresh_lexical is not None else []),
            ]
        )
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        os.rename(temporary_dir, output_dir)
        temporary_dir = None
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    result = {
        "status": manifest["status"],
        "semantic_closure_manifest": str(output_dir / MANIFEST_NAME),
        "semantic_closure_candidates": str(output_dir / CANDIDATES_NAME),
        "semantic_closure_adjudications_template": str(
            output_dir / ADJUDICATIONS_NAME
        ),
        "candidate_count": len(candidates),
        "unresolved_candidate_count": len(templates),
        "known_blockers": blockers,
        "behavior_authorized": False,
    }
    if fresh_lexical is not None:
        result.update(
            {
                "fresh_full_pool_candidate_count": len(fresh_lexical["pairs"]),
                "fresh_full_pool_unresolved_candidate_count": len(
                    fresh_lexical["templates"]
                ),
                "fresh_full_pool_review_template": str(
                    output_dir / FULL_POOL_REVIEW_NAME
                ),
            }
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--resolved-manifest", type=Path, required=True)
    parser.add_argument("--v3-finalization-manifest", type=Path, required=True)
    parser.add_argument("--v3-semantic-edges", type=Path, required=True)
    parser.add_argument("--v3-split-manifest", type=Path, required=True)
    parser.add_argument("--fresh-lexical-candidate-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_semantic_closure(
        staging_manifest_path=args.staging_manifest,
        resolved_manifest_path=args.resolved_manifest,
        v3_finalization_manifest_path=args.v3_finalization_manifest,
        v3_semantic_edges_path=args.v3_semantic_edges,
        v3_split_manifest_path=args.v3_split_manifest,
        output_dir=args.output_dir,
        fresh_lexical_candidate_manifest_path=args.fresh_lexical_candidate_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
