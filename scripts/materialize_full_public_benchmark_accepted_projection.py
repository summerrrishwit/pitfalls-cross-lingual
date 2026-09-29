#!/usr/bin/env python3
"""Materialize one accepted candidate of each kind per retained target.

This is a deliberately narrow, offline projection over a completed full-set
candidate review.  It does not regenerate candidates or recompute split and
component structure.  A target is retained only when it has at least one
accepted distractor and one accepted Neutral candidate whose donors are also
retained targets.  That condition is evaluated to a monotone fixed point.

The selected candidate rows and retained fact rows are copied unchanged from
the source post-review package.  The source split manifest and complete
component artifact are copied byte-for-byte as structural snapshots.  A
quarantine record is a projection-availability disposition only; it is not a
claim that the underlying fact is false and it is never human gold.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "full-public-benchmark-accepted-candidate-projection-v1"
MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-full-accepted-candidate-projection-manifest-v1"
)
STATUS = "accepted_only_min1_per_kind_materialized_provisional_offline"
POLICY_VERSION = "accepted-only-min1-per-kind-v1"
QUARANTINE_SCHEMA_VERSION = (
    "public-benchmark-full-accepted-projection-quarantine-v1"
)
COMPONENT_SCHEMA_VERSION = "public-benchmark-full-leakage-component-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-full-provisional-split-manifest-v1"
)
STRUCTURAL_SNAPSHOT_SCOPE = "source_r09_support_cohort"
CANDIDATE_KIND_ORDER = ("distractor", "neutral")


def _load_sibling(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


candidate_review = _load_sibling(
    "_accepted_projection_candidate_review",
    "run_full_public_benchmark_candidate_review.py",
)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _resolve_path(binding: Mapping[str, Any], owner_path: Path, label: str) -> Path:
    raw = binding.get("path") or binding.get("filename")
    candidate = Path(_required_string(raw, f"{label}.path"))
    if not candidate.is_absolute():
        candidate = owner_path.parent / candidate
    return candidate.resolve()


def _input_binding(
    path: Path,
    *,
    schema_version: str,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    value: Dict[str, Any] = {
        "path": str(path),
        "sha256": candidate_review.sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def _output_binding(
    path: Path,
    *,
    schema_version: str,
    record_count: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    value: Dict[str, Any] = {
        "filename": path.name,
        "sha256": candidate_review.sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        value["record_count"] = record_count
    return value


def _verify_input_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: str,
    explicit_path: Optional[Path] = None,
    record_count: Optional[int] = None,
) -> Path:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    path = _resolve_path(binding, owner_path, label)
    if explicit_path is not None and path != Path(explicit_path).resolve():
        raise ValueError(f"{label} binding differs from the supplied path")
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != candidate_review.sha256_file(path):
        raise ValueError(f"{label} SHA-256 mismatch")
    if binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count mismatch")
    if binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version mismatch")
    if record_count is not None and binding.get("record_count") != record_count:
        raise ValueError(f"{label} record_count mismatch")
    return path


def _verify_json_output(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: str,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} output binding is missing")
    path = _resolve_path(binding, owner_path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != candidate_review.sha256_file(path):
        raise ValueError(f"{label} output SHA-256 mismatch")
    if binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} output byte_count mismatch")
    if binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} output schema_version mismatch")
    value = candidate_review.read_json(path)
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"{label} payload schema_version mismatch")
    return path, value, dict(binding)


def _verify_jsonl_output(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: str,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} output binding is missing")
    path = _resolve_path(binding, owner_path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != candidate_review.sha256_file(path):
        raise ValueError(f"{label} output SHA-256 mismatch")
    if binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} output byte_count mismatch")
    if binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} output schema_version mismatch")
    rows = candidate_review.read_jsonl(path, allow_empty=True)
    if binding.get("record_count") != len(rows):
        raise ValueError(f"{label} output record_count mismatch")
    for row_number, row in enumerate(rows, start=1):
        if row.get("schema_version") != expected_schema:
            raise ValueError(f"{label} row schema mismatch at row {row_number}")
    return path, rows, dict(binding)


def _index_unique(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label}[{row_number}].{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = dict(row)
    return result


def _load_structural_snapshots(
    *,
    postreview_manifest_path: Path,
    postreview_manifest: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    outputs = postreview_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Postreview outputs are missing")
    component_path, components, component_binding = (
        candidate_review._verify_jsonl_binding(
            manifest_path=postreview_manifest_path,
            binding=outputs.get("leakage_components"),
            explicit_path=None,
            label="source leakage_components",
            expected_schema=COMPONENT_SCHEMA_VERSION,
            allow_empty=False,
        )
    )
    split_binding = outputs.get("split_manifest")
    if not isinstance(split_binding, dict):
        raise ValueError("Source split_manifest binding is missing")
    split_path = _resolve_path(
        split_binding, postreview_manifest_path, "source split_manifest"
    )
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if split_binding.get("sha256") != candidate_review.sha256_file(split_path):
        raise ValueError("Source split_manifest SHA-256 mismatch")
    if split_binding.get("byte_count") != split_path.stat().st_size:
        raise ValueError("Source split_manifest byte_count mismatch")
    if split_binding.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Source split_manifest schema_version mismatch")
    split_manifest = candidate_review.read_json(split_path)
    if split_manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Source split_manifest payload schema_version mismatch")

    fact_index = _index_unique(facts, "base_fact_id", "source facts")
    component_index = _index_unique(
        components, "leakage_component_id", "source components"
    )
    member_owner: Dict[str, str] = {}
    for component_id, component in component_index.items():
        if component.get("schema_version") != COMPONENT_SCHEMA_VERSION:
            raise ValueError(f"Source component schema is invalid: {component_id}")
        members = component.get("base_fact_ids")
        if not isinstance(members, list) or component.get("member_count") != len(members):
            raise ValueError(f"Source component membership is invalid: {component_id}")
        for raw_base_fact_id in members:
            base_fact_id = _required_string(
                raw_base_fact_id, f"{component_id}.base_fact_ids"
            )
            if base_fact_id not in fact_index or base_fact_id in member_owner:
                raise ValueError(
                    f"Source component coverage is invalid: {base_fact_id}"
                )
            fact = fact_index[base_fact_id]
            if fact.get("leakage_component_id") != component_id:
                raise ValueError(
                    f"Source fact/component identity is stale: {base_fact_id}"
                )
            if fact.get("split_assignment") != component.get("split_assignment"):
                raise ValueError(
                    f"Source fact/component split is stale: {base_fact_id}"
                )
            member_owner[base_fact_id] = component_id
    if set(member_owner) != set(fact_index):
        raise ValueError("Source components do not exactly cover source facts")
    if split_manifest.get("base_fact_count") != len(facts):
        raise ValueError("Source split_manifest base_fact_count is stale")
    if split_manifest.get("leakage_component_count") != len(components):
        raise ValueError("Source split_manifest leakage_component_count is stale")
    if split_manifest.get("split_status") != "provisional_not_frozen":
        raise ValueError("Source split is not provisional")
    if split_manifest.get("formal_split_freeze_performed") is not False:
        raise ValueError("Source split was already frozen")
    actual_split_counts = dict(
        sorted(Counter(str(row["split_assignment"]) for row in facts).items())
    )
    if split_manifest.get("actual_base_fact_counts") != actual_split_counts:
        raise ValueError("Source split_manifest assignment counts are stale")
    return {
        "component_path": component_path,
        "components": components,
        "component_binding": component_binding,
        "split_path": split_path,
        "split_manifest": split_manifest,
        "split_binding": dict(split_binding),
    }


def _load_source_context(
    *,
    postreview_manifest_path: Path,
    candidate_review_manifest_path: Path,
    candidate_adjudications_path: Path,
) -> Dict[str, Any]:
    postreview_manifest_path = Path(postreview_manifest_path).resolve()
    candidate_review_manifest_path = Path(candidate_review_manifest_path).resolve()
    candidate_adjudications_path = Path(candidate_adjudications_path).resolve()

    postreview_manifest, manifest_kind, input_bindings, units = (
        candidate_review.load_review_units(
            candidate_manifest_path=postreview_manifest_path
        )
    )
    if manifest_kind != "postreview_rebuild":
        raise ValueError("Accepted projection requires a postreview rebuild manifest")
    review_bundle = candidate_review.load_review_evidence_manifest(
        candidate_review_manifest_path
    )
    if review_bundle["candidate_manifest_path"] != postreview_manifest_path:
        raise ValueError("Candidate review is not bound to the supplied postreview manifest")
    if review_bundle["adjudications_path"] != candidate_adjudications_path:
        raise ValueError("Candidate adjudications differ from the review manifest binding")
    review_manifest = review_bundle["manifest"]
    adjudications = [
        review_bundle["adjudication_by_id"][unit.candidate_id] for unit in units
    ]
    if not all(
        (
            review_manifest.get("status") == "completed",
            review_manifest.get("selection_is_full_candidate_set") is True,
            review_manifest.get("candidate_review_complete_for_full_set") is True,
            review_manifest.get("selected_candidate_count") == len(units),
            review_manifest.get("full_candidate_count") == len(units),
            review_manifest.get("terminal_status_counts") == {"completed": len(units)},
        )
    ):
        raise ValueError("Candidate review is not a completed full-set review")
    if set(review_bundle["adjudication_by_id"]) != {
        unit.candidate_id for unit in units
    }:
        raise ValueError("Candidate review adjudications do not exactly cover the full set")

    decision_counts = dict(
        sorted(Counter(str(row.get("overall_decision")) for row in adjudications).items())
    )
    if review_manifest.get("overall_decision_counts") != decision_counts:
        raise ValueError("Candidate review decision counts are stale")
    unit_by_id = {unit.candidate_id: unit for unit in units}
    for adjudication in adjudications:
        candidate_id = _required_string(
            adjudication.get("candidate_id"), "adjudication.candidate_id"
        )
        unit = unit_by_id[candidate_id]
        if any(
            (
                adjudication.get("candidate_kind") != unit.candidate_kind,
                adjudication.get("target_base_fact_id")
                != unit.target_base_fact_id,
                adjudication.get("source_base_fact_id")
                != unit.source_base_fact_id,
                adjudication.get("candidate_row_sha256")
                != unit.candidate_row_sha256,
                adjudication.get("target_behavior_consumed") is not False,
                adjudication.get("behavior_blind") is not True,
                adjudication.get("human_gold") is not False,
                adjudication.get("response_model_identity_status") != "matched",
            )
        ):
            raise ValueError(f"Candidate adjudication identity is stale: {candidate_id}")

    facts_path = Path(input_bindings["full_base_facts"]["path"])
    distractor_path = Path(input_bindings["distractor_candidates"]["path"])
    neutral_path = Path(input_bindings["neutral_reference_candidates"]["path"])
    facts = candidate_review.read_jsonl(facts_path)
    distractors = candidate_review.read_jsonl(distractor_path, allow_empty=True)
    neutrals = candidate_review.read_jsonl(neutral_path, allow_empty=True)
    structural = _load_structural_snapshots(
        postreview_manifest_path=postreview_manifest_path,
        postreview_manifest=postreview_manifest,
        facts=facts,
    )
    return {
        "postreview_manifest_path": postreview_manifest_path,
        "postreview_manifest": postreview_manifest,
        "candidate_review_manifest_path": candidate_review_manifest_path,
        "candidate_review_manifest": review_manifest,
        "candidate_adjudications_path": candidate_adjudications_path,
        "review_bundle": review_bundle,
        "input_bindings": input_bindings,
        "units": units,
        "facts": facts,
        "distractors": distractors,
        "neutrals": neutrals,
        "adjudications": adjudications,
        "decision_counts": decision_counts,
        **structural,
    }


def _candidate_id(kind: str, row: Mapping[str, Any]) -> str:
    field = "distractor_id" if kind == "distractor" else "neutral_candidate_id"
    return _required_string(row.get(field), f"{kind}.{field}")


def _build_projection_plan(source: Mapping[str, Any]) -> Dict[str, Any]:
    facts = source["facts"]
    fact_ids = [str(row["base_fact_id"]) for row in facts]
    fact_index = _index_unique(facts, "base_fact_id", "source facts")
    unit_by_id = {unit.candidate_id: unit for unit in source["units"]}
    adjudication_by_id = source["review_bundle"]["adjudication_by_id"]

    rows_by_id: Dict[str, Dict[str, Any]] = {}
    kind_by_id: Dict[str, str] = {}
    for kind, rows in (
        ("distractor", source["distractors"]),
        ("neutral", source["neutrals"]),
    ):
        for row in rows:
            candidate_id = _candidate_id(kind, row)
            if candidate_id in rows_by_id:
                raise ValueError(f"Duplicate candidate ID: {candidate_id}")
            rows_by_id[candidate_id] = dict(row)
            kind_by_id[candidate_id] = kind
    if set(rows_by_id) != set(unit_by_id):
        raise ValueError("Source candidate rows differ from the full review unit set")

    accepted: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        kind: defaultdict(list) for kind in CANDIDATE_KIND_ORDER
    }
    for candidate_id, row in rows_by_id.items():
        kind = kind_by_id[candidate_id]
        unit = unit_by_id[candidate_id]
        if candidate_review.sha256_value(row) != unit.candidate_row_sha256:
            raise ValueError(f"Source candidate row hash is stale: {candidate_id}")
        decision = adjudication_by_id[candidate_id]
        if decision.get("overall_decision") == "accept":
            accepted[kind][unit.target_base_fact_id].append(row)
    for kind in CANDIDATE_KIND_ORDER:
        for rows in accepted[kind].values():
            rows.sort(key=lambda row: (int(row["slot"]), _candidate_id(kind, row)))

    retained = set(fact_ids)
    quarantine_by_id: Dict[str, Dict[str, Any]] = {}
    removal_round_count = 0
    while True:
        round_number = removal_round_count + 1
        removals: List[Tuple[str, Dict[str, List[Dict[str, Any]]], List[str]]] = []
        for base_fact_id in fact_ids:
            if base_fact_id not in retained:
                continue
            eligible = {
                kind: [
                    row
                    for row in accepted[kind].get(base_fact_id, [])
                    if str(row["source_base_fact_id"]) in retained
                ]
                for kind in CANDIDATE_KIND_ORDER
            }
            missing = [kind for kind in CANDIDATE_KIND_ORDER if not eligible[kind]]
            if missing:
                removals.append((base_fact_id, eligible, missing))
        if not removals:
            break
        removal_round_count += 1
        for base_fact_id, eligible, missing in removals:
            reasons = {
                kind: (
                    "no_accepted_candidate_for_kind"
                    if not accepted[kind].get(base_fact_id)
                    else "all_accepted_candidate_donors_outside_projection_cohort"
                )
                for kind in missing
            }
            quarantine_by_id[base_fact_id] = {
                "schema_version": QUARANTINE_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "disposition": "quarantine_from_accepted_projection",
                "quarantine_stage": "accepted_candidate_projection_fixed_point",
                "fixed_point_round": round_number,
                "missing_candidate_kinds": missing,
                "reason_by_kind": reasons,
                "accepted_candidate_counts_by_kind_at_source_review": {
                    kind: len(accepted[kind].get(base_fact_id, []))
                    for kind in CANDIDATE_KIND_ORDER
                },
                "eligible_accepted_candidate_counts_by_kind_at_removal": {
                    kind: len(eligible[kind]) for kind in CANDIDATE_KIND_ORDER
                },
                "accepted_donor_base_fact_ids_by_kind_at_source_review": {
                    kind: sorted(
                        {
                            str(row["source_base_fact_id"])
                            for row in accepted[kind].get(base_fact_id, [])
                        }
                    )
                    for kind in CANDIDATE_KIND_ORDER
                },
                "eligible_donor_base_fact_ids_by_kind_at_removal": {
                    kind: sorted(
                        {str(row["source_base_fact_id"]) for row in eligible[kind]}
                    )
                    for kind in CANDIDATE_KIND_ORDER
                },
                "factual_falsehood_asserted": False,
                "human_gold": False,
            }
        retained.difference_update(base_fact_id for base_fact_id, _, _ in removals)

    if not retained:
        raise ValueError("Accepted projection fixed point retained no target facts")
    projected_facts = [
        copy.deepcopy(fact_index[base_fact_id])
        for base_fact_id in fact_ids
        if base_fact_id in retained
    ]
    selected_by_kind: Dict[str, List[Dict[str, Any]]] = {
        kind: [] for kind in CANDIDATE_KIND_ORDER
    }
    selected_adjudications: List[Dict[str, Any]] = []
    selected_candidate_ids: List[str] = []
    for base_fact_id in fact_ids:
        if base_fact_id not in retained:
            continue
        for kind in CANDIDATE_KIND_ORDER:
            eligible = [
                row
                for row in accepted[kind].get(base_fact_id, [])
                if str(row["source_base_fact_id"]) in retained
            ]
            eligible.sort(
                key=lambda row: (int(row["slot"]), _candidate_id(kind, row))
            )
            if not eligible:
                raise AssertionError("Fixed point did not preserve candidate availability")
            selected = copy.deepcopy(eligible[0])
            candidate_id = _candidate_id(kind, selected)
            selected_by_kind[kind].append(selected)
            selected_candidate_ids.append(candidate_id)
            selected_adjudications.append(
                copy.deepcopy(adjudication_by_id[candidate_id])
            )
    quarantine = [
        quarantine_by_id[base_fact_id]
        for base_fact_id in fact_ids
        if base_fact_id in quarantine_by_id
    ]
    return {
        "source_fact_ids": fact_ids,
        "retained_set": retained,
        "projected_facts": projected_facts,
        "projected_target_ids": [str(row["base_fact_id"]) for row in projected_facts],
        "selected_distractors": selected_by_kind["distractor"],
        "selected_neutrals": selected_by_kind["neutral"],
        "selected_candidate_ids": selected_candidate_ids,
        "selected_adjudications": selected_adjudications,
        "quarantine": quarantine,
        "quarantined_ids": [str(row["base_fact_id"]) for row in quarantine],
        "fixed_point_removal_round_count": removal_round_count,
    }


def _counts(source: Mapping[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    decisions = Counter(
        str(row["overall_decision"]) for row in source["adjudications"]
    )
    return {
        "source_base_facts": len(source["facts"]),
        "projected_base_facts": len(plan["projected_facts"]),
        "quarantined_base_facts": len(plan["quarantine"]),
        "source_distractor_candidates": len(source["distractors"]),
        "source_neutral_reference_candidates": len(source["neutrals"]),
        "source_candidate_total": len(source["units"]),
        "selected_distractor_candidates": len(plan["selected_distractors"]),
        "selected_neutral_reference_candidates": len(plan["selected_neutrals"]),
        "selected_candidate_total": len(plan["selected_candidate_ids"]),
        "source_review_accept": decisions.get("accept", 0),
        "source_review_reject": decisions.get("reject", 0),
        "source_review_defer": decisions.get("defer", 0),
        "fixed_point_removal_rounds": plan["fixed_point_removal_round_count"],
        "source_leakage_components": len(source["components"]),
    }


def _selection_contract(source: Mapping[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "cardinality_per_target": {"distractor": 1, "neutral": 1},
        "candidate_ordering": ["slot_ascending", "candidate_id_ascending"],
        "accepted_candidates_only": True,
        "source_review_full_set_complete": True,
        "source_review_all_accept": all(
            row.get("overall_decision") == "accept"
            for row in source["adjudications"]
        ),
        "source_review_nonaccept_allowed": True,
        "selected_projection_all_accept": True,
        "donor_must_be_in_projection_target_cohort": True,
        "fixed_point_mode": "monotone_simultaneous_target_removal_until_stable",
        "fixed_point_removal_round_count": plan["fixed_point_removal_round_count"],
        "candidate_rows_rewritten": False,
        "split_recomputed": False,
        "components_recomputed": False,
        "structural_snapshot_scope": STRUCTURAL_SNAPSHOT_SCOPE,
        "structural_snapshots_byte_identical_to_source": True,
        "quarantine_asserts_factual_falsehood": False,
        "human_gold": False,
    }


def _source_review_contract(
    source: Mapping[str, Any], plan: Mapping[str, Any]
) -> Dict[str, Any]:
    manifest = source["candidate_review_manifest"]
    decision_counts = dict(
        sorted(Counter(row["overall_decision"] for row in source["adjudications"]).items())
    )
    return {
        "review_status": "completed",
        "review_schema_version": candidate_review.RUN_MANIFEST_SCHEMA,
        "review_tool_version": candidate_review.TOOL_VERSION,
        "full_candidate_count": len(source["units"]),
        "selection_is_full_candidate_set": True,
        "candidate_review_complete_for_full_set": True,
        "terminal_status_counts": {"completed": len(source["units"])},
        "overall_decision_counts": decision_counts,
        "source_review_all_accept": all(
            row["overall_decision"] == "accept" for row in source["adjudications"]
        ),
        "source_review_nonaccept_allowed": True,
        "selected_projection_all_accept": True,
        "selected_candidate_accept_evidence_count": len(
            plan["selected_adjudications"]
        ),
        "selected_candidate_accept_evidence_ids_sha256": candidate_review.sha256_value(
            plan["selected_candidate_ids"]
        ),
        "review_run_contract_sha256": manifest["run_contract_sha256"],
        "behavior_blind": True,
        "human_gold": False,
    }


def _lineage_digests(
    source: Mapping[str, Any], plan: Mapping[str, Any]
) -> Dict[str, str]:
    projected_facts = plan["projected_facts"]
    selected_adjudications_by_id = {
        str(row["candidate_id"]): row for row in plan["selected_adjudications"]
    }
    selected_rows_by_id = {
        _candidate_id("distractor", row): row for row in plan["selected_distractors"]
    }
    selected_rows_by_id.update(
        {_candidate_id("neutral", row): row for row in plan["selected_neutrals"]}
    )
    selected_candidate_ids = plan["selected_candidate_ids"]
    return {
        "ordered_source_base_fact_ids_sha256": candidate_review.sha256_value(
            plan["source_fact_ids"]
        ),
        "ordered_projected_base_fact_ids_sha256": candidate_review.sha256_value(
            plan["projected_target_ids"]
        ),
        "ordered_quarantined_base_fact_ids_sha256": candidate_review.sha256_value(
            plan["quarantined_ids"]
        ),
        "ordered_selected_distractor_ids_sha256": candidate_review.sha256_value(
            [_candidate_id("distractor", row) for row in plan["selected_distractors"]]
        ),
        "ordered_selected_neutral_candidate_ids_sha256": candidate_review.sha256_value(
            [_candidate_id("neutral", row) for row in plan["selected_neutrals"]]
        ),
        "ordered_selected_candidate_ids_sha256": candidate_review.sha256_value(
            selected_candidate_ids
        ),
        "ordered_selected_candidate_row_hashes_sha256": candidate_review.sha256_value(
            [
                [candidate_id, candidate_review.sha256_value(selected_rows_by_id[candidate_id])]
                for candidate_id in selected_candidate_ids
            ]
        ),
        "ordered_selected_accept_adjudication_row_hashes_sha256": candidate_review.sha256_value(
            [
                [
                    candidate_id,
                    candidate_review.sha256_value(
                        selected_adjudications_by_id[candidate_id]
                    ),
                ]
                for candidate_id in selected_candidate_ids
            ]
        ),
        "ordered_projected_split_assignments_sha256": candidate_review.sha256_value(
            [
                [row["base_fact_id"], row["split_assignment"]]
                for row in projected_facts
            ]
        ),
        "ordered_projected_component_assignments_sha256": candidate_review.sha256_value(
            [
                [row["base_fact_id"], row["leakage_component_id"]]
                for row in projected_facts
            ]
        ),
        "ordered_quarantine_rounds_sha256": candidate_review.sha256_value(
            [
                [row["base_fact_id"], row["fixed_point_round"]]
                for row in plan["quarantine"]
            ]
        ),
    }


def _safety_contract() -> Dict[str, Any]:
    return {
        "output_is_provisional": True,
        "canonical_freeze_emitted": False,
        "review_freeze_emitted": False,
        "split_freeze_emitted": False,
        "candidate_rows_formally_promoted": False,
        "translation_review_complete": False,
        "historical_exposure_complete_for_full_pool": False,
        "hf_checkpoint_bound": False,
        "hf_tokenizer_bound": False,
        "hf_model_executed": False,
        "hf_tokenizer_executed": False,
        "behavior_executed": False,
        "validation_exposed": False,
        "sealed_exposed": False,
        "perturbation_authorized": False,
        "path_not_token_authorized": False,
        "candidate_policy_relaxed": False,
        "quarantine_asserts_factual_falsehood": False,
        "human_gold": False,
    }


def _manifest_payload(
    *,
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> Dict[str, Any]:
    input_bindings = {
        "postreview_manifest": _input_binding(
            source["postreview_manifest_path"],
            schema_version=str(source["postreview_manifest"]["schema_version"]),
        ),
        "candidate_review_manifest": _input_binding(
            source["candidate_review_manifest_path"],
            schema_version=candidate_review.RUN_MANIFEST_SCHEMA,
        ),
        "candidate_review_adjudications": _input_binding(
            source["candidate_adjudications_path"],
            schema_version=candidate_review.ADJUDICATION_SCHEMA,
            record_count=len(source["adjudications"]),
        ),
    }
    lineage = _lineage_digests(source, plan)
    projection_id = "accepted_projection_" + candidate_review.sha256_value(
        {
            "policy_version": POLICY_VERSION,
            "inputs": input_bindings,
            "projected_ids": lineage["ordered_projected_base_fact_ids_sha256"],
            "candidate_ids": lineage["ordered_selected_candidate_ids_sha256"],
            "quarantine_ids": lineage["ordered_quarantined_base_fact_ids_sha256"],
        }
    )[:24]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": STATUS,
        "projection_id": projection_id,
        "inputs": input_bindings,
        "outputs": copy.deepcopy(dict(outputs)),
        "counts": _counts(source, plan),
        "selection_contract": _selection_contract(source, plan),
        "source_review_contract": _source_review_contract(source, plan),
        "lineage_digests": lineage,
        "safety_contract": _safety_contract(),
        "limitations": [
            "The selected candidates carry independent proxy-review acceptance, not human gold.",
            "Non-accept source candidates are allowed and are omitted rather than repeatedly regenerated.",
            "Quarantine records express candidate availability only and do not assert that a fact is false.",
            "The split manifest and complete component artifact remain byte-identical source-cohort structural snapshots; projected target membership is defined by full_base_facts.",
            "No exact HF checkpoint/tokenizer was bound or executed and no behavior, Validation, or Sealed data was exposed.",
        ],
        "next_required_steps": [
            "translate and review zh for the accepted projection",
            "complete the historical-exposure contract and scope-owner attestation",
            "freeze the exact reviewed projection and split",
            "only then bind an exact HF checkpoint and tokenizer",
        ],
    }


def materialize(
    *,
    postreview_manifest_path: Path,
    candidate_review_manifest_path: Path,
    candidate_adjudications_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    source = _load_source_context(
        postreview_manifest_path=postreview_manifest_path,
        candidate_review_manifest_path=candidate_review_manifest_path,
        candidate_adjudications_path=candidate_adjudications_path,
    )
    plan = _build_projection_plan(source)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        paths = {
            "full_base_facts": staged_dir / "full_base_facts.jsonl",
            "distractor_candidates": staged_dir / "distractor_candidates.jsonl",
            "neutral_reference_candidates": staged_dir
            / "neutral_reference_candidates.jsonl",
            "leakage_components": staged_dir / "leakage_components.jsonl",
            "split_manifest": staged_dir / "split_manifest.json",
            "quarantine": staged_dir / "quarantine.jsonl",
            "manifest": staged_dir / "accepted_projection_manifest.json",
        }
        candidate_review.write_jsonl(paths["full_base_facts"], plan["projected_facts"])
        candidate_review.write_jsonl(
            paths["distractor_candidates"], plan["selected_distractors"]
        )
        candidate_review.write_jsonl(
            paths["neutral_reference_candidates"], plan["selected_neutrals"]
        )
        candidate_review.write_jsonl(paths["quarantine"], plan["quarantine"])
        shutil.copyfile(source["component_path"], paths["leakage_components"])
        shutil.copyfile(source["split_path"], paths["split_manifest"])
        outputs = {
            "full_base_facts": _output_binding(
                paths["full_base_facts"],
                schema_version=candidate_review.FULL_FACT_SCHEMA,
                record_count=len(plan["projected_facts"]),
            ),
            "distractor_candidates": _output_binding(
                paths["distractor_candidates"],
                schema_version=candidate_review.DISTRACTOR_SCHEMA,
                record_count=len(plan["selected_distractors"]),
            ),
            "neutral_reference_candidates": _output_binding(
                paths["neutral_reference_candidates"],
                schema_version=candidate_review.NEUTRAL_SCHEMA,
                record_count=len(plan["selected_neutrals"]),
            ),
            "leakage_components": _output_binding(
                paths["leakage_components"],
                schema_version=COMPONENT_SCHEMA_VERSION,
                record_count=len(source["components"]),
            ),
            "split_manifest": _output_binding(
                paths["split_manifest"], schema_version=SPLIT_MANIFEST_SCHEMA_VERSION
            ),
            "quarantine": _output_binding(
                paths["quarantine"],
                schema_version=QUARANTINE_SCHEMA_VERSION,
                record_count=len(plan["quarantine"]),
            ),
        }
        manifest = _manifest_payload(source=source, plan=plan, outputs=outputs)
        candidate_review.write_json(paths["manifest"], manifest)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        staged_dir.replace(output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged_dir, ignore_errors=True)

    manifest_path = output_dir / "accepted_projection_manifest.json"
    return {
        "status": STATUS,
        "projection_id": manifest["projection_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": candidate_review.sha256_file(manifest_path),
        "projected_base_fact_count": len(plan["projected_facts"]),
        "quarantined_base_fact_count": len(plan["quarantine"]),
        "selected_candidate_count": len(plan["selected_candidate_ids"]),
        "source_review_all_accept": manifest["selection_contract"][
            "source_review_all_accept"
        ],
        "selected_projection_all_accept": True,
        "human_gold": False,
    }


def load_projection_context(
    projection_manifest_path: Path,
    *,
    expected_postreview_manifest_path: Optional[Path] = None,
    expected_review_manifest_path: Optional[Path] = None,
    expected_adjudications_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fail-closed validation entrypoint shared by zh and freezer consumers."""

    projection_manifest_path = Path(projection_manifest_path).resolve()
    manifest = candidate_review.read_json(projection_manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Accepted projection manifest schema_version is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION:
        raise ValueError("Accepted projection tool_version is unsupported")
    if manifest.get("status") != STATUS:
        raise ValueError("Accepted projection status is invalid")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Accepted projection inputs/outputs are missing")

    postreview_schema = inputs.get("postreview_manifest", {}).get("schema_version")
    if postreview_schema not in candidate_review.SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS:
        raise ValueError("Accepted projection postreview schema is unsupported")
    postreview_path = _verify_input_binding(
        inputs.get("postreview_manifest"),
        owner_path=projection_manifest_path,
        label="accepted projection postreview manifest",
        expected_schema=str(postreview_schema),
        explicit_path=expected_postreview_manifest_path,
    )
    review_path = _verify_input_binding(
        inputs.get("candidate_review_manifest"),
        owner_path=projection_manifest_path,
        label="accepted projection candidate review manifest",
        expected_schema=candidate_review.RUN_MANIFEST_SCHEMA,
        explicit_path=expected_review_manifest_path,
    )
    adjudication_count = inputs.get("candidate_review_adjudications", {}).get(
        "record_count"
    )
    if isinstance(adjudication_count, bool) or not isinstance(adjudication_count, int):
        raise ValueError("Accepted projection adjudication record_count is invalid")
    adjudications_path = _verify_input_binding(
        inputs.get("candidate_review_adjudications"),
        owner_path=projection_manifest_path,
        label="accepted projection candidate adjudications",
        expected_schema=candidate_review.ADJUDICATION_SCHEMA,
        explicit_path=expected_adjudications_path,
        record_count=adjudication_count,
    )
    source = _load_source_context(
        postreview_manifest_path=postreview_path,
        candidate_review_manifest_path=review_path,
        candidate_adjudications_path=adjudications_path,
    )
    if adjudication_count != len(source["adjudications"]):
        raise ValueError("Accepted projection adjudication count is stale")
    plan = _build_projection_plan(source)

    output_specs = {
        "full_base_facts": (candidate_review.FULL_FACT_SCHEMA, True),
        "distractor_candidates": (candidate_review.DISTRACTOR_SCHEMA, True),
        "neutral_reference_candidates": (candidate_review.NEUTRAL_SCHEMA, True),
        "leakage_components": (COMPONENT_SCHEMA_VERSION, True),
        "split_manifest": (SPLIT_MANIFEST_SCHEMA_VERSION, False),
        "quarantine": (QUARANTINE_SCHEMA_VERSION, True),
    }
    output_paths: Dict[str, Path] = {}
    output_values: Dict[str, Any] = {}
    output_bindings: Dict[str, Dict[str, Any]] = {}
    for label, (schema, is_jsonl) in output_specs.items():
        if is_jsonl:
            path, value, binding = _verify_jsonl_output(
                outputs.get(label),
                owner_path=projection_manifest_path,
                label=label,
                expected_schema=schema,
            )
        else:
            path, value, binding = _verify_json_output(
                outputs.get(label),
                owner_path=projection_manifest_path,
                label=label,
                expected_schema=schema,
            )
        output_paths[label] = path
        output_values[label] = value
        output_bindings[label] = binding

    expected_values = {
        "full_base_facts": plan["projected_facts"],
        "distractor_candidates": plan["selected_distractors"],
        "neutral_reference_candidates": plan["selected_neutrals"],
        "leakage_components": source["components"],
        "split_manifest": source["split_manifest"],
        "quarantine": plan["quarantine"],
    }
    for label, expected in expected_values.items():
        if output_values[label] != expected:
            raise ValueError(f"Accepted projection output differs from replay: {label}")
    if candidate_review.sha256_file(output_paths["leakage_components"]) != candidate_review.sha256_file(
        source["component_path"]
    ) or candidate_review.sha256_file(output_paths["split_manifest"]) != candidate_review.sha256_file(
        source["split_path"]
    ):
        raise ValueError("Accepted projection structural snapshots are not byte-identical")

    expected_manifest = _manifest_payload(
        source=source, plan=plan, outputs=output_bindings
    )
    if manifest != expected_manifest:
        raise ValueError("Accepted projection manifest differs from deterministic replay")
    return {
        "manifest_path": projection_manifest_path,
        "manifest": manifest,
        "postreview_manifest_path": postreview_path,
        "candidate_review_manifest_path": review_path,
        "candidate_review_adjudications_path": adjudications_path,
        "source_review_bundle": source["review_bundle"],
        "output_paths": output_paths,
        "output_bindings": output_bindings,
        "full_base_facts": output_values["full_base_facts"],
        "distractor_candidates": output_values["distractor_candidates"],
        "neutral_reference_candidates": output_values[
            "neutral_reference_candidates"
        ],
        "leakage_components": output_values["leakage_components"],
        "split_manifest": output_values["split_manifest"],
        "quarantine": output_values["quarantine"],
        "projected_target_ids": plan["projected_target_ids"],
        "selected_candidate_ids": plan["selected_candidate_ids"],
        "selected_adjudications": plan["selected_adjudications"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postreview-manifest", type=Path, required=True)
    parser.add_argument("--candidate-review-manifest", type=Path, required=True)
    parser.add_argument("--candidate-adjudications", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize(
        postreview_manifest_path=args.postreview_manifest,
        candidate_review_manifest_path=args.candidate_review_manifest,
        candidate_adjudications_path=args.candidate_adjudications,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
