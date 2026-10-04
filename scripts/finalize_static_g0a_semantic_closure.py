#!/usr/bin/env python3
"""Finalize the bounded static-G0A semantic-closure evidence.

This command is deliberately narrower than a claim of universal semantic
duplicate recall.  It validates and deterministically replays the bound
full-pool lexical candidate protocol, requires terminal review of every
cohort-reachable candidate, applies only explicit structural corrections,
projects reviewed positive connectivity back to the current 160-row cohort,
and merges that projection with the fresh current-cohort semantic review.

The existing split is never recomputed or rewritten by this command.  Any
positive component that crosses an existing split (or split group) fails
closed.  The emitted v2 manifest can be consumed by ``manage_static_g0a.py
finalize`` but still does not itself freeze G0A or authorize behavior calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_static_g0a as g0a  # noqa: E402
from scripts import materialize_public_benchmark_duplicate_adjudications as duplicate_adjudications  # noqa: E402
from scripts import materialize_public_benchmark_duplicate_closure as duplicate_closure  # noqa: E402
from scripts import materialize_static_g0a_semantic_closure as pending_closure  # noqa: E402


TOOL_VERSION = "static-g0a-semantic-closure-finalizer-v1"
POLICY_VERSION = "bounded-full-pool-lexical-plus-current-semantic-closure-v2"
STATUS = "complete"
CLAIM_SCOPE = (
    "all candidates emitted by the frozen deterministic cohort-seeded full-pool "
    "lexical protocol, including reachable out-of-cohort bridge nodes, plus the "
    "bound current-cohort semantic candidate set, were terminally adjudicated"
)

FINAL_MANIFEST_NAME = "semantic_closure_manifest.json"
FINAL_CANDIDATES_NAME = "semantic_closure_candidates.jsonl"
FINAL_DECISIONS_NAME = "semantic_closure_adjudications.jsonl"
CORRECTED_REVIEW_NAME = "full_pool_lexical_review.corrected.jsonl"
CORRECTION_AUDIT_NAME = "full_pool_structural_corrections_applied.jsonl"
DUPLICATE_ADJUDICATIONS_NAME = "duplicate_closure_adjudications.jsonl"
REPLACEMENT_SCREEN_NAME = "replacement_all_pairs_screen.jsonl"
MERGE_AUDIT_NAME = "semantic_closure_merge_audit.json"
RESOLUTION_DIR_NAME = "full_pool_resolution"

CORRECTION_AUDIT_SCHEMA = "static-g0a-full-pool-structural-correction-v1"
REPLACEMENT_SCREEN_SCHEMA = "static-g0a-replacement-all-pairs-screen-v1"
MERGE_AUDIT_SCHEMA = "static-g0a-semantic-closure-merge-audit-v1"
CORRECTED_REVIEW_SCHEMA = (
    "public-benchmark-duplicate-closure-review-shard-template-v1"
)
POSITIVE_RELATIONSHIPS = frozenset({"same_fact", "same_leakage_component"})
TERMINAL_RELATIONSHIPS = frozenset({*POSITIVE_RELATIONSHIPS, "distinct"})


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _pair(left: str, right: str) -> Tuple[str, str]:
    if left == right:
        raise ValueError(f"pair repeats one endpoint: {left}")
    return tuple(sorted((left, right)))


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(g0a.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _published_binding(
    staged_path: Path,
    published_path: Path,
    *,
    schema_version: str,
    record_count: Optional[int] = None,
    id_values: Optional[Sequence[str]] = None,
    id_digest_field: Optional[str] = None,
) -> Dict[str, Any]:
    binding = g0a._binding(
        staged_path,
        schema_version=schema_version,
        record_count=record_count,
        id_values=id_values,
        id_digest_field=id_digest_field,
    )
    binding["path"] = str(Path(published_path).resolve())
    return binding


def _snapshot(path: Path, *, schema_version: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    result = {
        "path": str(path),
        "sha256": g0a.sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _assert_unchanged(bindings: Iterable[Mapping[str, Any]]) -> None:
    seen: Set[str] = set()
    for binding in bindings:
        path = Path(_required_string(binding.get("path"), "snapshot.path"))
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            raise RuntimeError(f"input disappeared during semantic finalization: {path}")
        if path.stat().st_size != binding.get("byte_count"):
            raise RuntimeError(f"input size changed during semantic finalization: {path}")
        if g0a.sha256_file(path) != binding.get("sha256"):
            raise RuntimeError(f"input hash changed during semantic finalization: {path}")


def _validate_pending_packet(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = g0a.read_json(manifest_path)
    if manifest.get("schema_version") != g0a.SEMANTIC_CLOSURE_MANIFEST_SCHEMA:
        raise ValueError("pending semantic closure manifest schema is unsupported")
    if manifest.get("status") != "pending_incomplete_candidate_generation_and_adjudication":
        raise ValueError("semantic closure input is not the expected pending packet")
    if manifest.get("review_blinded_to_behavior_results") is not True:
        raise ValueError("pending semantic review is not behavior blind")
    if manifest.get("behavior_result_count") != 0:
        raise ValueError("pending semantic review contains behavior results")
    for field in (
        "semantic_paraphrase_closure_complete",
        "candidate_generation_complete",
        "cohort_scope_complete",
        "out_of_cohort_bridge_search_complete",
        "static_frozen",
        "behavior_authorized",
        "simulation_authorized",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"pending manifest must keep {field}=false")

    staging_manifest_path, _ = g0a._validate_file_binding(
        manifest.get("staging_manifest"),
        owner_path=manifest_path,
        label="pending staging manifest",
        expected_schema=g0a.STAGING_MANIFEST_SCHEMA,
    )
    staging_manifest, staging_rows, staging_by_id = g0a._load_staging(
        staging_manifest_path
    )
    resolved_manifest_path, _ = g0a._validate_file_binding(
        manifest.get("resolved_static_draft_manifest"),
        owner_path=manifest_path,
        label="pending resolved draft manifest",
        expected_schema=g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
    )
    resolved_manifest = g0a.read_json(resolved_manifest_path)
    resolved_path, resolved_rows = g0a._validate_file_binding(
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
    if len(resolved_rows) != g0a.EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"resolved cohort has {len(resolved_rows)} rows, expected {g0a.EXPECTED_RECORD_COUNT}"
        )
    if set(resolved_by_id) != set(staging_by_id):
        raise ValueError("resolved and staging cohorts differ")
    g0a._validate_binding_identity(
        manifest.get("input_bundle"),
        staging_manifest["source_bundle"],
        "pending input_bundle",
    )
    g0a._validate_binding_identity(
        manifest.get("effective_staging_bundle"),
        staging_manifest["effective_staging_bundle"],
        "pending effective_staging_bundle",
    )
    g0a._validate_binding_identity(
        manifest.get("resolved_static_draft"),
        resolved_manifest["resolved_rows"],
        "pending resolved_static_draft",
    )

    candidates_path, semantic_candidates = g0a._validate_file_binding(
        manifest.get("candidates"),
        owner_path=manifest_path,
        label="pending semantic candidates",
        expected_schema=g0a.SEMANTIC_CANDIDATE_SCHEMA,
        jsonl=True,
    )
    template_path, semantic_template = g0a._validate_file_binding(
        manifest.get("adjudications"),
        owner_path=manifest_path,
        label="pending semantic adjudication template",
        expected_schema=g0a.SEMANTIC_DECISION_SCHEMA,
        jsonl=True,
    )
    if manifest.get("unresolved_candidate_count") != len(semantic_candidates):
        raise ValueError("pending semantic unresolved count is stale")
    if len(semantic_template) != len(semantic_candidates):
        raise ValueError("pending semantic template does not cover candidates")

    fresh = _required_mapping(
        manifest.get("fresh_full_pool_lexical_closure"),
        "pending fresh_full_pool_lexical_closure",
    )
    candidate_manifest_path, _ = g0a._validate_file_binding(
        fresh.get("candidate_manifest"),
        owner_path=manifest_path,
        label="fresh full-pool candidate manifest",
        expected_schema=duplicate_closure.CANDIDATE_MANIFEST_SCHEMA_VERSION,
    )
    fresh_manifest, cohort_rows, fresh_pairs = duplicate_closure._validate_candidate_manifest(
        candidate_manifest_path
    )
    if cohort_rows != resolved_rows:
        raise ValueError("fresh full-pool closure is not bound to resolved rows")
    pair_binding = _required_mapping(
        _required_mapping(fresh_manifest.get("closure"), "fresh closure").get(
            "candidate_pairs"
        ),
        "fresh candidate-pair binding",
    )
    g0a._validate_binding_identity(
        fresh.get("candidate_pairs"),
        pair_binding,
        "pending fresh candidate pairs",
        fields=("sha256", "schema_version", "record_count"),
    )
    if fresh.get("candidate_pair_count") != len(fresh_pairs):
        raise ValueError("pending fresh candidate count is stale")
    if fresh.get("semantic_near_duplicate_recall_guaranteed") is not False:
        raise ValueError("pending packet unexpectedly claims semantic recall")

    snapshots = [
        _snapshot(manifest_path, schema_version=g0a.SEMANTIC_CLOSURE_MANIFEST_SCHEMA),
        _snapshot(staging_manifest_path, schema_version=g0a.STAGING_MANIFEST_SCHEMA),
        _snapshot(
            resolved_manifest_path,
            schema_version=g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
        ),
        _snapshot(resolved_path, schema_version=g0a.FINAL_BUNDLE_SCHEMA),
        _snapshot(candidates_path, schema_version=g0a.SEMANTIC_CANDIDATE_SCHEMA),
        _snapshot(template_path, schema_version=g0a.SEMANTIC_DECISION_SCHEMA),
        _snapshot(
            candidate_manifest_path,
            schema_version=duplicate_closure.CANDIDATE_MANIFEST_SCHEMA_VERSION,
        ),
        _snapshot(
            Path(str(pair_binding["path"])),
            schema_version=duplicate_closure.AUDIT_PAIR_SCHEMA_VERSION,
        ),
    ]
    for _, binding in duplicate_adjudications._iter_file_bindings(
        fresh_manifest, label="fresh candidate manifest"
    ):
        snapshots.append(dict(binding))
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "staging_manifest": staging_manifest,
        "staging_by_id": staging_by_id,
        "resolved_manifest": resolved_manifest,
        "resolved_rows": resolved_rows,
        "resolved_by_id": resolved_by_id,
        "semantic_candidates": semantic_candidates,
        "fresh_manifest": fresh_manifest,
        "fresh_candidate_manifest_path": candidate_manifest_path,
        "fresh_pairs": fresh_pairs,
        "snapshots": snapshots,
    }


def _validate_semantic_decisions(
    candidates: Sequence[Mapping[str, Any]], decisions_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    decisions_path = Path(decisions_path).resolve()
    decisions = g0a.read_jsonl(decisions_path)
    candidate_by_id: Dict[str, Mapping[str, Any]] = {}
    pair_seen: Set[Tuple[str, str]] = set()
    for index, candidate in enumerate(candidates, start=1):
        if candidate.get("schema_version") != g0a.SEMANTIC_CANDIDATE_SCHEMA:
            raise ValueError(f"semantic candidate {index} schema is unsupported")
        candidate_id = _required_string(candidate.get("candidate_id"), "candidate_id")
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate semantic candidate_id: {candidate_id}")
        pair = _pair(
            _required_string(candidate.get("left_base_fact_id"), f"{candidate_id}.left"),
            _required_string(candidate.get("right_base_fact_id"), f"{candidate_id}.right"),
        )
        if pair in pair_seen:
            raise ValueError(f"duplicate semantic pair: {pair}")
        pair_seen.add(pair)
        candidate_by_id[candidate_id] = candidate

    decision_by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(decisions, start=1):
        if row.get("schema_version") != g0a.SEMANTIC_DECISION_SCHEMA:
            raise ValueError(f"semantic decision {index} schema is unsupported")
        candidate_id = _required_string(row.get("candidate_id"), "decision candidate_id")
        if candidate_id in decision_by_id or candidate_id not in candidate_by_id:
            raise ValueError(f"semantic decision coverage is invalid: {candidate_id}")
        if row.get("candidate_record_sha256") != g0a.sha256_value(
            candidate_by_id[candidate_id]
        ):
            raise ValueError(f"semantic decision has stale candidate hash: {candidate_id}")
        if row.get("terminal_status") != "completed":
            raise ValueError(f"semantic decision is not terminal: {candidate_id}")
        if row.get("decision") not in TERMINAL_RELATIONSHIPS:
            raise ValueError(f"semantic decision is unresolved: {candidate_id}")
        if row.get("reviewer_type") not in {"human", "codex_proxy"}:
            raise ValueError(f"semantic reviewer_type is invalid: {candidate_id}")
        if row.get("reviewer_type") == "codex_proxy" and row.get("human_gold") is not False:
            raise ValueError(f"semantic Codex decision must have human_gold=false: {candidate_id}")
        if row.get("review_blinded_to_behavior_results") is not True:
            raise ValueError(f"semantic decision is not behavior blind: {candidate_id}")
        _required_string(row.get("rationale"), f"{candidate_id}.rationale")
        decision_by_id[candidate_id] = dict(row)
    if set(decision_by_id) != set(candidate_by_id):
        missing = sorted(set(candidate_by_id) - set(decision_by_id))
        raise ValueError(f"semantic decisions do not exactly cover candidates: {missing[:5]}")
    return decisions, decision_by_id


def _load_review_shards_with_provenance(
    paths: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one full-pool review shard is required")
    rows: List[Dict[str, Any]] = []
    bindings: List[Dict[str, Any]] = []
    provenance: Dict[str, Dict[str, Any]] = {}
    resolved_paths = [Path(path).resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("full-pool review shard paths are repeated")
    for path in sorted(resolved_paths, key=str):
        shard_rows, binding = duplicate_adjudications._read_shard_snapshot(path)
        for row in shard_rows:
            pair_id = str(row["pair_id"])
            if pair_id in provenance:
                raise ValueError(f"duplicate full-pool review decision: {pair_id}")
            provenance[pair_id] = {
                "source_shard": copy.deepcopy(binding),
                "source_review_record_sha256": g0a.sha256_value(row),
            }
            rows.append(row)
        bindings.append(binding)
    return rows, bindings, provenance


def _cohort_endpoint_ids(
    pair_row: Mapping[str, Any], cohort_ids: Set[str]
) -> Set[str]:
    output: Set[str] = set()
    for side in ("left", "right"):
        provenance = _required_mapping(
            _required_mapping(pair_row.get(side), f"pair.{side}").get("provenance"),
            f"pair.{side}.provenance",
        )
        if (
            provenance.get("input_role") == "current_bundle"
            and provenance.get("base_fact_id") in cohort_ids
        ):
            output.add(str(provenance["base_fact_id"]))
    return output


def _apply_structural_corrections(
    *,
    candidate_pairs: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    review_provenance: Mapping[str, Mapping[str, Any]],
    correction_path: Path,
    cohort_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    correction_path = Path(correction_path).resolve()
    correction = g0a.read_json(correction_path)
    if set(correction) != {"reason", "corrections"}:
        raise ValueError("structural correction file must contain only reason/corrections")
    correction_reason = _required_string(correction.get("reason"), "correction reason")
    corrections = _required_mapping(correction.get("corrections"), "corrections")
    correction_rationales = {
        _required_string(pair_id, "correction pair_id"): _required_string(
            rationale, f"correction rationale {pair_id}"
        )
        for pair_id, rationale in corrections.items()
    }
    pair_by_id = {
        _required_string(row.get("pair_id"), "fresh pair_id"): row
        for row in candidate_pairs
    }
    review_by_id = {
        _required_string(row.get("pair_id"), "review pair_id"): dict(row)
        for row in review_rows
    }
    if len(pair_by_id) != len(candidate_pairs):
        raise ValueError("fresh candidate pairs repeat pair_id")
    if len(review_by_id) != len(review_rows):
        raise ValueError("full-pool review rows repeat pair_id")
    missing = sorted(set(pair_by_id) - set(review_by_id))
    extra = sorted(set(review_by_id) - set(pair_by_id))
    if missing or extra:
        raise ValueError(
            "full-pool review must exactly cover candidates: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    structurally_invalid = {
        pair_id
        for pair_id, review in review_by_id.items()
        if review.get("decision") == "exclude_cohort"
        and not _cohort_endpoint_ids(pair_by_id[pair_id], cohort_ids)
    }
    already_applied = {
        pair_id
        for pair_id, rationale in correction_rationales.items()
        if pair_id in review_by_id
        and pair_id in pair_by_id
        and not _cohort_endpoint_ids(pair_by_id[pair_id], cohort_ids)
        and review_by_id[pair_id].get("decision") == "distinct"
        and review_by_id[pair_id].get("rationale") == rationale
        and "structural" in str(review_by_id[pair_id].get("reviewer_id", ""))
    }
    accepted_corrections = structurally_invalid | already_applied
    if set(correction_rationales) != accepted_corrections:
        missing_corrections = sorted(structurally_invalid - set(correction_rationales))
        extra_corrections = sorted(set(correction_rationales) - accepted_corrections)
        raise ValueError(
            "structural corrections must exactly cover out-of-cohort exclude_cohort "
            "rows or match explicitly corrected shard rows: "
            f"missing={missing_corrections[:5]}, extra={extra_corrections[:5]}"
        )

    correction_binding = _snapshot(correction_path)
    corrected: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    for pair in candidate_pairs:
        pair_id = str(pair["pair_id"])
        source = review_by_id[pair_id]
        output = dict(source)
        if pair_id in correction_rationales:
            correction_application_status = (
                "applied_by_finalizer"
                if pair_id in structurally_invalid
                else "already_applied_in_source_shard_and_revalidated"
            )
            output["decision"] = "distinct"
            output["rationale"] = correction_rationales[pair_id]
            source_provenance = _required_mapping(
                review_provenance.get(pair_id), f"review provenance {pair_id}"
            )
            audit_rows.append(
                {
                    "schema_version": CORRECTION_AUDIT_SCHEMA,
                    "pair_id": pair_id,
                    "candidate_row_sha256": g0a.sha256_value(pair),
                    "source_review_shard": copy.deepcopy(
                        source_provenance["source_shard"]
                    ),
                    "source_review_record_sha256": source_provenance[
                        "source_review_record_sha256"
                    ],
                    "source_decision": source["decision"],
                    "source_rationale": source["rationale"],
                    "correction_file": copy.deepcopy(correction_binding),
                    "correction_reason": correction_reason,
                    "corrected_decision": "distinct",
                    "corrected_rationale": output["rationale"],
                    "correction_application_status": correction_application_status,
                    "structural_issue_reported_by_correction": (
                        "exclude_cohort_without_current_cohort_endpoint"
                    ),
                    "structural_rule": (
                        "exclude_cohort_is_illegal_without_a_bound_current_cohort_endpoint; "
                        "out-of-cohort fact-quality concerns are non-edges for current-cohort closure"
                    ),
                    "semantic_equivalence_silently_rewritten": False,
                }
            )
        corrected.append(output)
    if any(row.get("decision") == "exclude_cohort" for row in corrected):
        unresolved = [
            str(row["pair_id"])
            for row in corrected
            if row.get("decision") == "exclude_cohort"
        ]
        raise ValueError(
            "current-cohort exclusions require cohort repair, not semantic finalization: "
            + ",".join(unresolved[:5])
        )
    return corrected, audit_rows, {
        "correction_file": correction_binding,
        "source_decision_counts": dict(
            sorted(Counter(str(row["decision"]) for row in review_rows).items())
        ),
        "corrected_decision_counts": dict(
            sorted(Counter(str(row["decision"]) for row in corrected).items())
        ),
        "structural_correction_count": len(audit_rows),
    }


def _endpoint_node(pair_row: Mapping[str, Any], side: str) -> str:
    provenance = _required_mapping(
        _required_mapping(pair_row.get(side), f"pair.{side}").get("provenance"),
        f"pair.{side}.provenance",
    )
    role = _required_string(provenance.get("input_role"), f"pair.{side}.input_role")
    if role == "current_bundle":
        identifier = _required_string(
            provenance.get("base_fact_id"), f"pair.{side}.base_fact_id"
        )
    elif role == "comparison_canonical":
        identifier = _required_string(
            provenance.get("audit_record_id"), f"pair.{side}.audit_record_id"
        )
    else:
        raise ValueError(f"unsupported endpoint input_role: {role}")
    return f"{role}:{identifier}"


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _project_full_pool_connectivity(
    *,
    candidate_pairs: Sequence[Mapping[str, Any]],
    adjudications: Sequence[Mapping[str, Any]],
    cohort_ids: Set[str],
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], str], List[Dict[str, Any]]]:
    pair_by_id = {str(row["pair_id"]): row for row in candidate_pairs}
    decision_by_id = {str(row["pair_id"]): row for row in adjudications}
    if set(pair_by_id) != set(decision_by_id):
        raise ValueError("full-pool adjudications do not exactly cover candidate pairs")
    leakage = _DisjointSet()
    same_fact = _DisjointSet()
    positive_pair_nodes: Dict[str, Tuple[str, str]] = {}
    direct_current: Dict[Tuple[str, str], str] = {}
    for pair_id in sorted(pair_by_id):
        pair_row = pair_by_id[pair_id]
        decision = str(decision_by_id[pair_id]["decision"])
        left, right = _endpoint_node(pair_row, "left"), _endpoint_node(pair_row, "right")
        leakage.find(left)
        leakage.find(right)
        same_fact.find(left)
        same_fact.find(right)
        direct_ids: List[str] = []
        for side in ("left", "right"):
            provenance = pair_row[side]["provenance"]
            if (
                provenance.get("input_role") == "current_bundle"
                and provenance.get("base_fact_id") in cohort_ids
            ):
                direct_ids.append(str(provenance["base_fact_id"]))
        if len(direct_ids) == 2:
            key = _pair(*direct_ids)
            if key in direct_current:
                raise ValueError(f"full-pool candidates repeat a current pair: {key}")
            direct_current[key] = decision
        if decision in POSITIVE_RELATIONSHIPS:
            leakage.union(left, right)
            positive_pair_nodes[pair_id] = (left, right)
            if decision == "same_fact":
                same_fact.union(left, right)

    members_by_root: Dict[str, List[str]] = defaultdict(list)
    for base_fact_id in sorted(cohort_ids):
        node = f"current_bundle:{base_fact_id}"
        members_by_root[leakage.find(node)].append(base_fact_id)
    projected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    component_rows: List[Dict[str, Any]] = []
    for root, members in sorted(members_by_root.items()):
        if len(members) < 2:
            continue
        member_nodes = {f"current_bundle:{value}" for value in members}
        component_support = sorted(
            pair_id
            for pair_id, (left, right) in positive_pair_nodes.items()
            if leakage.find(left) == root and leakage.find(right) == root
        )
        component_id = "bounded_semantic_component_" + g0a.sha256_value(members)[:24]
        component_rows.append(
            {
                "component_id": component_id,
                "current_cohort_base_fact_ids": members,
                "supporting_full_pool_positive_pair_ids": component_support,
            }
        )
        for left, right in combinations(members, 2):
            relationship = (
                "same_fact"
                if same_fact.find(f"current_bundle:{left}")
                == same_fact.find(f"current_bundle:{right}")
                else "same_leakage_component"
            )
            projected[(left, right)] = {
                "relationship": relationship,
                "component_id": component_id,
                "supporting_full_pool_positive_pair_ids": component_support,
            }
    return projected, direct_current, component_rows


def _validate_split_integrity(
    *,
    resolved_by_id: Mapping[str, Mapping[str, Any]],
    positive_pairs: Iterable[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    union = _DisjointSet()
    for base_fact_id in resolved_by_id:
        union.find(base_fact_id)
    for left, right in positive_pairs:
        union.union(left, right)
    components: Dict[str, List[str]] = defaultdict(list)
    for base_fact_id in sorted(resolved_by_id):
        components[union.find(base_fact_id)].append(base_fact_id)
    output: List[Dict[str, Any]] = []
    for members in components.values():
        if len(members) < 2:
            continue
        splits = sorted({str(resolved_by_id[value].get("split_assignment")) for value in members})
        groups = sorted({str(resolved_by_id[value].get("split_group_id")) for value in members})
        if len(splits) != 1:
            raise ValueError(
                "positive semantic component crosses existing split assignments: "
                + ",".join(members)
            )
        if len(groups) != 1:
            raise ValueError(
                "positive semantic component crosses existing split groups: "
                + ",".join(members)
            )
        output.append(
            {
                "component_id": "current_semantic_component_"
                + g0a.sha256_value(members)[:24],
                "base_fact_ids": members,
                "split_assignment": splits[0],
                "split_group_id": groups[0],
            }
        )
    output.sort(key=lambda row: row["component_id"])
    return output


def _merge_current_semantic_evidence(
    *,
    semantic_candidates: Sequence[Mapping[str, Any]],
    semantic_decision_by_id: Mapping[str, Mapping[str, Any]],
    projected_pairs: Mapping[Tuple[str, str], Mapping[str, Any]],
    direct_current_reviews: Mapping[Tuple[str, str], str],
    staging_by_id: Mapping[str, Mapping[str, Any]],
    resolved_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    semantic_by_pair: Dict[Tuple[str, str], Tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for candidate in semantic_candidates:
        candidate_id = str(candidate["candidate_id"])
        pair = _pair(
            str(candidate["left_base_fact_id"]),
            str(candidate["right_base_fact_id"]),
        )
        semantic_by_pair[pair] = (candidate, semantic_decision_by_id[candidate_id])

    overlap = sorted(set(semantic_by_pair) & set(direct_current_reviews))
    overlap_rows: List[Dict[str, Any]] = []
    for pair in overlap:
        semantic_decision = str(semantic_by_pair[pair][1]["decision"])
        full_pool_decision = str(direct_current_reviews[pair])
        if semantic_decision != full_pool_decision:
            raise ValueError(
                "overlapping full-pool/current semantic reviews disagree for "
                f"{pair}: {full_pool_decision} != {semantic_decision}"
            )
        overlap_rows.append(
            {
                "left_base_fact_id": pair[0],
                "right_base_fact_id": pair[1],
                "full_pool_decision": full_pool_decision,
                "current_semantic_decision": semantic_decision,
                "consistent": True,
            }
        )

    candidates = [copy.deepcopy(dict(row)) for row in semantic_candidates]
    decisions = [
        copy.deepcopy(dict(semantic_decision_by_id[str(row["candidate_id"])]))
        for row in semantic_candidates
    ]
    synthesized = []
    for pair, projection in sorted(projected_pairs.items()):
        if pair in semantic_by_pair:
            existing = str(semantic_by_pair[pair][1]["decision"])
            if existing != projection["relationship"]:
                raise ValueError(
                    "projected connectivity conflicts with current semantic review for "
                    f"{pair}: {projection['relationship']} != {existing}"
                )
            continue
        left, right = pair
        candidate_id = pending_closure._candidate_id(left, right)
        candidate = {
            "schema_version": g0a.SEMANTIC_CANDIDATE_SCHEMA,
            "candidate_id": candidate_id,
            "left_base_fact_id": left,
            "right_base_fact_id": right,
            "left_source_record_sha256": staging_by_id[left]["source_record_sha256"],
            "right_source_record_sha256": staging_by_id[right]["source_record_sha256"],
            "left_effective_source_record_sha256": staging_by_id[left][
                "effective_source_record_sha256"
            ],
            "right_effective_source_record_sha256": staging_by_id[right][
                "effective_source_record_sha256"
            ],
            "left_input_record_sha256": g0a.sha256_value(resolved_by_id[left]),
            "right_input_record_sha256": g0a.sha256_value(resolved_by_id[right]),
            "left": pending_closure._endpoint_view(resolved_by_id[left]),
            "right": pending_closure._endpoint_view(resolved_by_id[right]),
            "generation_reasons": [
                "fresh_full_pool_positive_connectivity_projection"
            ],
            "bounded_projection_component_id": projection["component_id"],
            "supporting_full_pool_positive_pair_ids": projection[
                "supporting_full_pool_positive_pair_ids"
            ],
            "requires_fresh_adjudication": False,
            "derived_from_terminal_reviewed_positive_edges": True,
            "review_blinded_to_behavior_results": True,
        }
        decision = {
            "schema_version": g0a.SEMANTIC_DECISION_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_record_sha256": g0a.sha256_value(candidate),
            "terminal_status": "completed",
            "decision": projection["relationship"],
            "rationale": (
                "Derived by transitive projection of terminally reviewed positive "
                "edges in the bounded full-pool lexical closure."
            ),
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "review_blinded_to_behavior_results": True,
            "derivation": "deterministic_positive_connectivity_projection_v1",
            "supporting_full_pool_positive_pair_ids": projection[
                "supporting_full_pool_positive_pair_ids"
            ],
        }
        candidates.append(candidate)
        decisions.append(decision)
        synthesized.append(candidate_id)

    order = sorted(range(len(candidates)), key=lambda index: str(candidates[index]["candidate_id"]))
    candidates = [candidates[index] for index in order]
    decision_lookup = {str(row["candidate_id"]): row for row in decisions}
    decisions = [decision_lookup[str(row["candidate_id"])] for row in candidates]
    if len(decision_lookup) != len(decisions):
        raise ValueError("merged semantic decisions repeat candidate_id")
    return candidates, decisions, {
        "overlap_count": len(overlap_rows),
        "overlaps": overlap_rows,
        "projected_pair_count": len(projected_pairs),
        "synthesized_projection_candidate_count": len(synthesized),
        "synthesized_projection_candidate_ids": synthesized,
    }


def _near_qualifies(metrics: Mapping[str, Any], threshold: float) -> bool:
    return bool(
        metrics["length_ratio"] >= 0.50
        and (
            metrics["decision_score"] >= threshold
            or (
                metrics["token_containment"] >= 0.90
                and metrics["length_ratio"] >= 0.60
            )
        )
    )


def _replacement_all_pairs_screen(
    *,
    resolved_rows: Sequence[Mapping[str, Any]],
    fresh_pairs: Sequence[Mapping[str, Any]],
    lexical_policy: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    audit_tool = duplicate_closure._load_audit_module()
    resolved_by_id = {str(row["base_fact_id"]): row for row in resolved_rows}
    replacements = [
        row
        for row in resolved_rows
        if isinstance(row.get("cohort_repair_lineage"), dict)
        and row["cohort_repair_lineage"].get("disposition") == "replacement"
    ]
    emitted_by_pair: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for pair_row in fresh_pairs:
        ids: List[str] = []
        for side in ("left", "right"):
            provenance = pair_row[side]["provenance"]
            if provenance.get("input_role") != "current_bundle":
                break
            ids.append(str(provenance.get("base_fact_id")))
        if len(ids) == 2:
            emitted_by_pair[_pair(*ids)].append(str(pair_row["pair_id"]))

    question_threshold = float(lexical_policy["question_near_threshold"])
    fact_threshold = float(lexical_policy["canonical_fact_near_threshold"])
    rows: List[Dict[str, Any]] = []
    missing_candidates: List[Tuple[str, str]] = []
    for replacement in sorted(replacements, key=lambda row: str(row["base_fact_id"])):
        replacement_id = str(replacement["base_fact_id"])
        replacement_question = audit_tool.normalize_lexical(
            audit_tool.first_string(
                replacement, ("source_question_en", "source_question", "question")
            )
        )
        replacement_fact = audit_tool.normalize_lexical(
            audit_tool.first_string(
                replacement, ("canonical_fact_en", "canonical_fact")
            )
        )
        replacement_answer = audit_tool.first_string(
            replacement, ("answer_en", "answer", "source_answer")
        )
        replacement_aliases = {
            audit_tool.normalize_answer(value)
            for value in audit_tool.extract_aliases(replacement, replacement_answer)
            if audit_tool.normalize_answer(value)
        }
        for other_id, other in sorted(resolved_by_id.items()):
            if other_id == replacement_id:
                continue
            other_question = audit_tool.normalize_lexical(
                audit_tool.first_string(
                    other, ("source_question_en", "source_question", "question")
                )
            )
            other_fact = audit_tool.normalize_lexical(
                audit_tool.first_string(other, ("canonical_fact_en", "canonical_fact"))
            )
            other_answer = audit_tool.first_string(
                other, ("answer_en", "answer", "source_answer")
            )
            other_aliases = {
                audit_tool.normalize_answer(value)
                for value in audit_tool.extract_aliases(other, other_answer)
                if audit_tool.normalize_answer(value)
            }
            question_metrics = audit_tool.lexical_similarity(
                replacement_question, other_question
            )
            fact_metrics = audit_tool.lexical_similarity(replacement_fact, other_fact)
            shared_aliases = sorted(replacement_aliases & other_aliases)
            match_types = []
            if shared_aliases:
                match_types.append("answer_alias_exact")
            if replacement_question and replacement_question == other_question:
                match_types.append("question_exact")
            elif _near_qualifies(question_metrics, question_threshold):
                match_types.append("question_near_lexical")
            if replacement_fact and replacement_fact == other_fact:
                match_types.append("canonical_fact_exact")
            elif _near_qualifies(fact_metrics, fact_threshold):
                match_types.append("canonical_fact_near_lexical")
            pair = _pair(replacement_id, other_id)
            emitted = sorted(emitted_by_pair.get(pair, []))
            if match_types and not emitted:
                missing_candidates.append(pair)
            rows.append(
                {
                    "schema_version": REPLACEMENT_SCREEN_SCHEMA,
                    "replacement_base_fact_id": replacement_id,
                    "other_base_fact_id": other_id,
                    "replacement_input_record_sha256": g0a.sha256_value(replacement),
                    "other_input_record_sha256": g0a.sha256_value(other),
                    "shared_answer_aliases_normalized": shared_aliases,
                    "question_metrics": question_metrics,
                    "canonical_fact_metrics": fact_metrics,
                    "bounded_direct_match_types": match_types,
                    "bounded_direct_candidate_qualifies": bool(match_types),
                    "emitted_full_pool_candidate_pair_ids": emitted,
                    "screening_policy": {
                        "answer_normalization_version": lexical_policy[
                            "answer_normalization_version"
                        ],
                        "text_normalization_version": lexical_policy[
                            "text_normalization_version"
                        ],
                        "question_near_threshold": question_threshold,
                        "canonical_fact_near_threshold": fact_threshold,
                        "all_current_cohort_pairs_enumerated": True,
                        "minhash_blocking_bypassed_for_replacement_screen": True,
                    },
                }
            )
    if missing_candidates:
        raise ValueError(
            "replacement direct lexical screen found pairs absent from fresh closure: "
            + ",".join(f"{left}|{right}" for left, right in missing_candidates[:5])
        )
    expected = len(replacements) * (len(resolved_rows) - 1)
    if len(rows) != expected:
        raise ValueError("replacement all-pairs screen coverage is incomplete")
    return rows, {
        "replacement_base_fact_ids": sorted(str(row["base_fact_id"]) for row in replacements),
        "replacement_count": len(replacements),
        "other_current_row_count_per_replacement": len(resolved_rows) - 1,
        "screened_pair_count": len(rows),
        "bounded_direct_candidate_count": sum(
            bool(row["bounded_direct_candidate_qualifies"]) for row in rows
        ),
        "all_replacement_to_other_current_pairs_screened": True,
        "semantic_near_duplicate_recall_guaranteed": False,
    }


def _rewrite_resolution_for_publish(
    *,
    staged_resolution_dir: Path,
    published_resolution_dir: Path,
    staged_adjudications_path: Path,
    published_adjudications_path: Path,
) -> Dict[str, Any]:
    components_path = staged_resolution_dir / "leakage_components.jsonl"
    exclusions_path = staged_resolution_dir / "dedup_exclusions.jsonl"
    split_path = staged_resolution_dir / "provisional_split_manifest.json"
    resolution_path = staged_resolution_dir / "resolution_manifest.json"
    split = g0a.read_json(split_path)
    resolution = g0a.read_json(resolution_path)
    adjudication_binding = _published_binding(
        staged_adjudications_path,
        published_adjudications_path,
        schema_version=duplicate_closure.ADJUDICATION_SCHEMA_VERSION,
        record_count=len(g0a.read_jsonl(staged_adjudications_path)),
    )
    component_rows = g0a.read_jsonl(components_path, allow_empty=True)
    exclusion_rows = g0a.read_jsonl(exclusions_path, allow_empty=True)
    component_binding = _published_binding(
        components_path,
        published_resolution_dir / components_path.name,
        schema_version=duplicate_closure.COMPONENT_SCHEMA_VERSION,
        record_count=len(component_rows),
    )
    exclusion_binding = _published_binding(
        exclusions_path,
        published_resolution_dir / exclusions_path.name,
        schema_version=duplicate_closure.EXCLUSION_SCHEMA_VERSION,
        record_count=len(exclusion_rows),
    )
    split["adjudications"] = adjudication_binding
    split["components"] = component_binding
    split["dedup_exclusions"] = exclusion_binding
    split["downstream_use"] = "diagnostic_resolution_only_existing_split_preserved"
    split["provisional_assignments_adopted"] = False
    split_path.write_bytes(_json_payload(split))
    split_binding = _published_binding(
        split_path,
        published_resolution_dir / split_path.name,
        schema_version=duplicate_closure.SPLIT_MANIFEST_SCHEMA_VERSION,
    )
    resolution["adjudications"] = adjudication_binding
    resolution["outputs"]["components"] = component_binding
    resolution["outputs"]["dedup_exclusions"] = exclusion_binding
    resolution["outputs"]["provisional_split_manifest"] = split_binding
    resolution["downstream_use"] = "diagnostic_resolution_only_existing_split_preserved"
    resolution["provisional_assignments_adopted"] = False
    resolution_path.write_bytes(_json_payload(resolution))
    return {
        "resolution": resolution,
        "components": component_rows,
        "exclusions": exclusion_rows,
        "bindings": {
            "resolution_manifest": _published_binding(
                resolution_path,
                published_resolution_dir / resolution_path.name,
                schema_version=duplicate_closure.RESOLUTION_MANIFEST_SCHEMA_VERSION,
            ),
            "provisional_split_manifest": split_binding,
            "components": component_binding,
            "dedup_exclusions": exclusion_binding,
        },
    }


def finalize_semantic_closure(
    *,
    pending_manifest_path: Path,
    semantic_decisions_path: Path,
    full_pool_review_shard_paths: Sequence[Path],
    structural_corrections_path: Path,
    output_dir: Path,
    reviewed_at: str,
    review_method: str = "fresh-bounded-full-pool-codex-semantic-review-v1",
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    reviewed_at = duplicate_adjudications._validate_reviewed_at(reviewed_at)
    review_method = _required_string(review_method, "review_method")
    loaded = _validate_pending_packet(pending_manifest_path)
    semantic_decisions_path = Path(semantic_decisions_path).resolve()
    semantic_decisions, semantic_decision_by_id = _validate_semantic_decisions(
        loaded["semantic_candidates"], semantic_decisions_path
    )
    review_rows, review_bindings, review_provenance = (
        _load_review_shards_with_provenance(full_pool_review_shard_paths)
    )
    corrected_reviews, correction_audit, correction_stats = (
        _apply_structural_corrections(
            candidate_pairs=loaded["fresh_pairs"],
            review_rows=review_rows,
            review_provenance=review_provenance,
            correction_path=structural_corrections_path,
            cohort_ids=set(loaded["resolved_by_id"]),
        )
    )
    full_pool_adjudications = duplicate_adjudications._materialize_rows(
        candidate_pairs=loaded["fresh_pairs"],
        cohort_rows=loaded["resolved_rows"],
        review_rows=corrected_reviews,
        reviewed_at=reviewed_at,
        review_method=review_method,
    )
    projected, direct_current, projected_components = _project_full_pool_connectivity(
        candidate_pairs=loaded["fresh_pairs"],
        adjudications=full_pool_adjudications,
        cohort_ids=set(loaded["resolved_by_id"]),
    )
    final_candidates, final_decisions, merge_stats = _merge_current_semantic_evidence(
        semantic_candidates=loaded["semantic_candidates"],
        semantic_decision_by_id=semantic_decision_by_id,
        projected_pairs=projected,
        direct_current_reviews=direct_current,
        staging_by_id=loaded["staging_by_id"],
        resolved_by_id=loaded["resolved_by_id"],
    )
    semantic_pair_decisions = {
        _pair(str(candidate["left_base_fact_id"]), str(candidate["right_base_fact_id"])): str(
            decision["decision"]
        )
        for candidate, decision in zip(final_candidates, final_decisions)
    }
    positive_pairs = [
        pair
        for pair, decision in semantic_pair_decisions.items()
        if decision in POSITIVE_RELATIONSHIPS
    ]
    split_components = _validate_split_integrity(
        resolved_by_id=loaded["resolved_by_id"], positive_pairs=positive_pairs
    )
    mechanical_pairs = g0a.formal_admission._mechanical_cross_group_risk_pairs(
        loaded["resolved_rows"]
    )
    missing_mechanical = mechanical_pairs - set(semantic_pair_decisions)
    if missing_mechanical:
        raise ValueError(
            "merged current semantic candidates omit mechanical pairs: "
            + ",".join(f"{left}|{right}" for left, right in sorted(missing_mechanical)[:5])
        )

    lexical_policy = _required_mapping(
        _required_mapping(
            loaded["fresh_manifest"].get("lexical_audit"), "fresh lexical audit"
        ).get("policy"),
        "fresh lexical policy",
    )
    replacement_screen, replacement_stats = _replacement_all_pairs_screen(
        resolved_rows=loaded["resolved_rows"],
        fresh_pairs=loaded["fresh_pairs"],
        lexical_policy=lexical_policy,
    )
    expected_replacements = sorted(
        str(row.get("replacement_base_fact_id"))
        for row in loaded["manifest"].get("cohort_replacement_lineage", [])
    )
    if expected_replacements != replacement_stats["replacement_base_fact_ids"]:
        raise ValueError("replacement coverage does not match pending lineage")

    source_unique_pair_count = (
        len(loaded["fresh_pairs"])
        + len(loaded["semantic_candidates"])
        - merge_stats["overlap_count"]
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent))
    )
    published = False
    try:
        corrected_path = temporary_dir / CORRECTED_REVIEW_NAME
        correction_audit_path = temporary_dir / CORRECTION_AUDIT_NAME
        duplicate_adjudications_path = temporary_dir / DUPLICATE_ADJUDICATIONS_NAME
        replacement_screen_path = temporary_dir / REPLACEMENT_SCREEN_NAME
        final_candidates_path = temporary_dir / FINAL_CANDIDATES_NAME
        final_decisions_path = temporary_dir / FINAL_DECISIONS_NAME
        merge_audit_path = temporary_dir / MERGE_AUDIT_NAME
        final_manifest_path = temporary_dir / FINAL_MANIFEST_NAME
        resolution_dir = temporary_dir / RESOLUTION_DIR_NAME

        corrected_path.write_bytes(_jsonl_payload(corrected_reviews))
        correction_audit_path.write_bytes(_jsonl_payload(correction_audit))
        duplicate_adjudications_path.write_bytes(
            _jsonl_payload(full_pool_adjudications)
        )
        duplicate_closure._validate_adjudications(
            duplicate_adjudications_path,
            loaded["fresh_pairs"],
            set(loaded["resolved_by_id"]),
        )
        duplicate_closure.resolve_candidates(
            candidate_manifest_path=loaded["fresh_candidate_manifest_path"],
            adjudications_path=duplicate_adjudications_path,
            output_dir=resolution_dir,
        )
        resolution = _rewrite_resolution_for_publish(
            staged_resolution_dir=resolution_dir,
            published_resolution_dir=output_dir / RESOLUTION_DIR_NAME,
            staged_adjudications_path=duplicate_adjudications_path,
            published_adjudications_path=output_dir / DUPLICATE_ADJUDICATIONS_NAME,
        )
        counts = _required_mapping(
            resolution["resolution"].get("counts"), "full-pool resolution counts"
        )
        if (
            counts.get("input") != len(loaded["resolved_rows"])
            or counts.get("retained") != len(loaded["resolved_rows"])
            or counts.get("deduplicated") != 0
            or counts.get("explicitly_excluded") != 0
            or resolution["exclusions"]
        ):
            raise ValueError(
                "full-pool resolution would mutate the current 160-row cohort; "
                "cohort repair is required before finalization"
            )

        replacement_screen_path.write_bytes(_jsonl_payload(replacement_screen))
        final_candidates_path.write_bytes(_jsonl_payload(final_candidates))
        final_decisions_path.write_bytes(_jsonl_payload(final_decisions))
        merge_audit = {
            "schema_version": MERGE_AUDIT_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "bounded_semantic_evidence_merged",
            "source_pair_counts": {
                "full_pool_lexical_closure_candidates": len(loaded["fresh_pairs"]),
                "current_semantic_candidates": len(loaded["semantic_candidates"]),
                "direct_pair_overlap": merge_stats["overlap_count"],
                "combined_unique_source_pairs": source_unique_pair_count,
            },
            "overlap_consistency": merge_stats["overlaps"],
            "full_pool_positive_projection": {
                "projected_current_pair_count": len(projected),
                "projected_current_components": projected_components,
            },
            "current_positive_components": split_components,
            "existing_split_preserved": True,
            "provisional_resolution_split_assignments_adopted": False,
            "cross_existing_split_positive_component_count": 0,
            "cross_existing_split_group_positive_component_count": 0,
            "final_current_candidate_count": len(final_candidates),
            "final_current_decision_count": len(final_decisions),
            "synthesized_projection_candidate_count": merge_stats[
                "synthesized_projection_candidate_count"
            ],
            "semantic_near_duplicate_recall_guaranteed": False,
        }
        merge_audit_path.write_bytes(_json_payload(merge_audit))

        final_candidate_binding = _published_binding(
            final_candidates_path,
            output_dir / FINAL_CANDIDATES_NAME,
            schema_version=g0a.SEMANTIC_CANDIDATE_SCHEMA,
            record_count=len(final_candidates),
            id_values=[str(row["candidate_id"]) for row in final_candidates],
            id_digest_field="candidate_ids_sha256",
        )
        final_decision_binding = _published_binding(
            final_decisions_path,
            output_dir / FINAL_DECISIONS_NAME,
            schema_version=g0a.SEMANTIC_DECISION_SCHEMA,
            record_count=len(final_decisions),
            id_values=[str(row["candidate_id"]) for row in final_decisions],
            id_digest_field="candidate_ids_sha256",
        )
        final_manifest = {
            "schema_version": g0a.SEMANTIC_CLOSURE_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "created_at": g0a.utc_now(),
            "status": STATUS,
            "policy_version": POLICY_VERSION,
            "input_bundle": copy.deepcopy(loaded["manifest"]["input_bundle"]),
            "effective_staging_bundle": copy.deepcopy(
                loaded["manifest"]["effective_staging_bundle"]
            ),
            "resolved_static_draft": copy.deepcopy(
                loaded["manifest"]["resolved_static_draft"]
            ),
            "pending_packet": _snapshot(
                loaded["manifest_path"],
                schema_version=g0a.SEMANTIC_CLOSURE_MANIFEST_SCHEMA,
            ),
            "candidates": final_candidate_binding,
            "adjudications": final_decision_binding,
            "bounded_protocol": {
                "definition": CLAIM_SCOPE,
                "protocol_frozen_by_candidate_manifest_sha256": g0a.sha256_file(
                    loaded["fresh_candidate_manifest_path"]
                ),
                "full_pool_record_count": loaded["fresh_manifest"][
                    "full_pool_authority"
                ]["expected_record_count"],
                "cohort_seed_count": len(loaded["resolved_rows"]),
                "selection_policy": loaded["fresh_manifest"]["closure"][
                    "selection_policy"
                ],
                "full_pool_lexical_candidate_generation_replayed": True,
                "cohort_reachable_candidate_count": len(loaded["fresh_pairs"]),
                "cohort_reachable_candidates_terminally_adjudicated": True,
                "current_semantic_candidates_terminally_adjudicated": True,
                "out_of_cohort_bridge_nodes_included": True,
                "comparison_canonical_nodes_included": True,
                "unbounded_semantic_search_claimed": False,
            },
            "full_pool_lexical_closure": {
                "candidate_manifest": _snapshot(
                    loaded["fresh_candidate_manifest_path"],
                    schema_version=duplicate_closure.CANDIDATE_MANIFEST_SCHEMA_VERSION,
                ),
                "candidate_pairs": copy.deepcopy(
                    loaded["fresh_manifest"]["closure"]["candidate_pairs"]
                ),
                "raw_review_shards": review_bindings,
                "structural_corrections": correction_stats["correction_file"],
                "correction_audit": _published_binding(
                    correction_audit_path,
                    output_dir / CORRECTION_AUDIT_NAME,
                    schema_version=CORRECTION_AUDIT_SCHEMA,
                    record_count=len(correction_audit),
                    id_values=[str(row["pair_id"]) for row in correction_audit],
                    id_digest_field="pair_ids_sha256",
                ),
                "corrected_review": _published_binding(
                    corrected_path,
                    output_dir / CORRECTED_REVIEW_NAME,
                    schema_version=CORRECTED_REVIEW_SCHEMA,
                    record_count=len(corrected_reviews),
                    id_values=[str(row["pair_id"]) for row in corrected_reviews],
                    id_digest_field="pair_ids_sha256",
                ),
                "duplicate_closure_adjudications": _published_binding(
                    duplicate_adjudications_path,
                    output_dir / DUPLICATE_ADJUDICATIONS_NAME,
                    schema_version=duplicate_closure.ADJUDICATION_SCHEMA_VERSION,
                    record_count=len(full_pool_adjudications),
                    id_values=[str(row["pair_id"]) for row in full_pool_adjudications],
                    id_digest_field="pair_ids_sha256",
                ),
                "duplicate_closure_resolution": resolution["bindings"],
                "source_decision_counts": correction_stats["source_decision_counts"],
                "corrected_decision_counts": correction_stats[
                    "corrected_decision_counts"
                ],
                "structural_correction_count": correction_stats[
                    "structural_correction_count"
                ],
                "unresolved_candidate_count": 0,
                "full_pool_lexical_search_complete_under_bounded_protocol": True,
                "out_of_cohort_bridge_search_complete_under_bounded_protocol": True,
                "semantic_near_duplicate_recall_guaranteed": False,
            },
            "current_semantic_review": {
                "source_candidates": copy.deepcopy(loaded["manifest"]["candidates"]),
                "source_adjudications": _snapshot(
                    semantic_decisions_path,
                    schema_version=g0a.SEMANTIC_DECISION_SCHEMA,
                ),
                "candidate_count": len(loaded["semantic_candidates"]),
                "unresolved_candidate_count": 0,
            },
            "merge": {
                "audit": _published_binding(
                    merge_audit_path,
                    output_dir / MERGE_AUDIT_NAME,
                    schema_version=MERGE_AUDIT_SCHEMA,
                ),
                "source_full_pool_pair_count": len(loaded["fresh_pairs"]),
                "source_current_semantic_pair_count": len(
                    loaded["semantic_candidates"]
                ),
                "overlap_pair_count": merge_stats["overlap_count"],
                "overlap_decisions_consistent": True,
                "combined_unique_source_pair_count": source_unique_pair_count,
                "positive_components_cross_existing_split": False,
                "positive_components_cross_existing_split_group": False,
                "existing_split_rewritten": False,
                "provisional_resolution_split_adopted": False,
            },
            "replacement_all_pairs_coverage": {
                **replacement_stats,
                "screen": _published_binding(
                    replacement_screen_path,
                    output_dir / REPLACEMENT_SCREEN_NAME,
                    schema_version=REPLACEMENT_SCREEN_SCHEMA,
                    record_count=len(replacement_screen),
                ),
                "fresh_reconstructed_full_pool_overlay_validated": True,
            },
            "review_blinded_to_behavior_results": True,
            "behavior_result_count": 0,
            "semantic_paraphrase_closure_complete": True,
            "semantic_paraphrase_closure_definition": CLAIM_SCOPE,
            "semantic_near_duplicate_recall_guaranteed": False,
            "candidate_generation_complete": True,
            "cohort_scope_complete": True,
            "out_of_cohort_bridge_search_complete": True,
            "unresolved_candidate_count": 0,
            "known_blockers": [],
            "human_gold": False,
            "static_frozen": False,
            "behavior_authorized": False,
            "simulation_authorized": False,
        }
        final_manifest_path.write_bytes(_json_payload(final_manifest))

        input_snapshots = [
            *loaded["snapshots"],
            _snapshot(semantic_decisions_path, schema_version=g0a.SEMANTIC_DECISION_SCHEMA),
            *review_bindings,
            correction_stats["correction_file"],
        ]
        _assert_unchanged(input_snapshots)
        duplicate_adjudications._publish_directory_no_replace(
            temporary_dir, output_dir
        )
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    return {
        "status": STATUS,
        "semantic_closure_manifest": str(output_dir / FINAL_MANIFEST_NAME),
        "semantic_closure_manifest_sha256": g0a.sha256_file(
            output_dir / FINAL_MANIFEST_NAME
        ),
        "full_pool_candidate_count": len(loaded["fresh_pairs"]),
        "current_semantic_candidate_count": len(loaded["semantic_candidates"]),
        "overlap_pair_count": merge_stats["overlap_count"],
        "combined_unique_source_pair_count": source_unique_pair_count,
        "final_current_candidate_count": len(final_candidates),
        "replacement_screened_pair_count": len(replacement_screen),
        "semantic_near_duplicate_recall_guaranteed": False,
        "static_frozen": False,
        "behavior_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending-manifest", type=Path, required=True)
    parser.add_argument("--semantic-decisions", type=Path, required=True)
    parser.add_argument(
        "--full-pool-review-shard", type=Path, action="append", required=True
    )
    parser.add_argument("--structural-corrections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument(
        "--review-method",
        default="fresh-bounded-full-pool-codex-semantic-review-v1",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = finalize_semantic_closure(
        pending_manifest_path=args.pending_manifest,
        semantic_decisions_path=args.semantic_decisions,
        full_pool_review_shard_paths=args.full_pool_review_shard,
        structural_corrections_path=args.structural_corrections,
        output_dir=args.output_dir,
        reviewed_at=args.reviewed_at,
        review_method=args.review_method,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
