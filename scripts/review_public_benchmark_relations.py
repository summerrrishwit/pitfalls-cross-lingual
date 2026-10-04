#!/usr/bin/env python3
"""Prepare and apply a hash-bound review of provisional relation signatures.

This tool deliberately stops before factual perturbation.  Signature approval
only establishes that a provisional relation label and direction are suitable
for item-level review.  It does not validate a factual triple, freeze a split,
or authorize behavior/PATH experiments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
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
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs" / "public_benchmark_probe_relation_review_v1.json"
)

TOOL_VERSION = "public-benchmark-relation-review-v1"
POLICY_SCHEMA_VERSION = "probe-relation-review-policy-v1"
EXPORT_MANIFEST_SCHEMA_VERSION = "probe-relation-review-export-manifest-v1"
SIGNATURE_ITEM_SCHEMA_VERSION = "probe-relation-signature-review-item-v1"
REVIEW_PLAN_SCHEMA_VERSION = "probe-relation-signature-review-plan-v1"
DECISION_SCHEMA_VERSION = "probe-relation-signature-decision-v1"
FRAME_SCHEMA_VERSION = "probe-relation-fact-review-frame-v1"
APPLY_MANIFEST_SCHEMA_VERSION = "probe-relation-review-apply-manifest-v1"
REVIEWED_INVENTORY_SCHEMA_VERSION = "reviewed-probe-relation-inventory-v1"
OVERLAY_SCHEMA_VERSION = "probe-relation-signature-overlay-v1"
SUMMARY_SCHEMA_VERSION = "probe-relation-review-summary-v1"

DIRECTIONAL_STATUS = "pending_semantic_and_balance_review"
UNRESOLVED_STATUS = "pending_direction_semantic_and_balance_review"
SPLITS = ("development", "validation", "sealed")
DECISIONS = frozenset({"approve_mapping", "reject_mapping", "defer"})


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {resolved}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {resolved}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, payload)


def write_text(path: Path, value: str) -> None:
    _atomic_write(Path(path), value.encode("utf-8"))


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _index(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        key = _required_string(row.get(field), f"{label} row {position}.{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = row
    return result


def _artifact(path: Path, *, schema_version: str, record_count: int) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
        "record_count": record_count,
        "schema_version": schema_version,
    }


def _validate_policy(policy: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported relation review policy schema_version")
    if policy.get("status") != "pre_perturbation_offline_only":
        raise ValueError("Relation review policy must remain pre-perturbation only")
    if policy.get("human_gold") is not False:
        raise ValueError("Relation review policy must not claim human gold")
    if policy.get("broad_relation_normalized_required") is not False:
        raise ValueError("Broad relation normalization must not gate the probe review")
    selection = policy.get("selection_contract")
    if selection != {
        "prompt_ready_required": True,
        "prompt_risk_flags_must_be_empty": True,
        "one_base_fact_per_split_group": True,
        "model_outputs_must_not_be_used": True,
        "selection_precedes_fact_review": True,
        "shortfall_policy": "fail_closed",
    }:
        raise ValueError("Unsupported relation review selection contract")
    relations = policy.get("relations")
    if not isinstance(relations, list) or not relations:
        raise ValueError("Relation review policy requires relations")
    result: Dict[str, Dict[str, Any]] = {}
    for position, raw in enumerate(relations, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"relations[{position}] must be an object")
        relation_id = _required_string(
            raw.get("probe_relation_id"), f"relations[{position}].probe_relation_id"
        )
        if relation_id in result:
            raise ValueError(f"Duplicate probe_relation_id: {relation_id}")
        direction = _required_string(raw.get("direction"), f"relations[{position}].direction")
        template = _required_string(
            raw.get("prompt_template_en"), f"relations[{position}].prompt_template_en"
        )
        if template.count("{subject}") != 1:
            raise ValueError(f"Prompt template must contain one {{subject}}: {relation_id}")
        frame_counts = raw.get("review_frame_split_counts")
        target_counts = raw.get("target_accept_split_counts")
        for label, counts in (
            ("review_frame_split_counts", frame_counts),
            ("target_accept_split_counts", target_counts),
        ):
            if not isinstance(counts, dict) or set(counts) != set(SPLITS):
                raise ValueError(f"{relation_id}.{label} must define exactly {SPLITS}")
            if any(type(counts[split]) is not int or counts[split] <= 0 for split in SPLITS):
                raise ValueError(f"{relation_id}.{label} values must be positive integers")
        if any(target_counts[split] > frame_counts[split] for split in SPLITS):
            raise ValueError(f"Target counts exceed review frame for {relation_id}")
        result[relation_id] = dict(raw)
    unresolved = policy.get("unresolved_direction_policy")
    if not isinstance(unresolved, dict):
        raise ValueError("Policy lacks unresolved_direction_policy")
    if unresolved.get("eligible_for_balanced_frame") is not False:
        raise ValueError("Direction-unresolved signatures cannot enter the balanced frame")
    return result


def _signature_review_items(
    provisional_rows: Sequence[Mapping[str, Any]],
    *,
    bundle_sha256: str,
    source_candidate_policy_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for position, row in enumerate(provisional_rows, start=1):
        if row.get("schema_version") != "provisional-factual-triple-v1":
            raise ValueError(f"Unsupported provisional triple schema at row {position}")
        signature_id = _required_string(
            row.get("relation_signature_id"), f"provisional row {position}.relation_signature_id"
        )
        groups[signature_id].append(row)

    directional: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for signature_id, members in groups.items():
        members = sorted(members, key=lambda row: str(row.get("candidate_id")))
        candidates = {row.get("probe_relation_candidate") for row in members}
        directions = {row.get("probe_relation_candidate_direction") for row in members}
        statuses = {row.get("probe_relation_status") for row in members}
        if len(candidates) != 1 or len(directions) != 1 or len(statuses) != 1:
            raise ValueError(f"Conflicting provisional relation state: {signature_id}")
        candidate = next(iter(candidates))
        direction = next(iter(directions))
        status = next(iter(statuses))
        if candidate is None:
            continue
        if status not in {DIRECTIONAL_STATUS, UNRESOLVED_STATUS}:
            raise ValueError(f"Unexpected candidate status for {signature_id}: {status}")
        first = members[0]
        base_fact_ids = sorted({str(row["base_fact_id"]) for row in members})
        member_bindings = [
            {
                "candidate_id": str(row["candidate_id"]),
                "base_fact_id": str(row["base_fact_id"]),
                "record_sha256": sha256_value(row),
            }
            for row in members
        ]
        item = {
            "schema_version": SIGNATURE_ITEM_SCHEMA_VERSION,
            "review_track": (
                "direction_unresolved" if direction == "unresolved" else "directional"
            ),
            "signature_id": signature_id,
            "source_candidate_policy_id": source_candidate_policy_id,
            "source_bundle_sha256": bundle_sha256,
            "relation_raw": first.get("relation_raw"),
            "subject_type": first.get("subject_type"),
            "answer_type": first.get("answer_type"),
            "probe_relation_candidate": candidate,
            "probe_relation_candidate_direction": direction,
            "member_record_count": len(members),
            "base_fact_count": len(base_fact_ids),
            "member_base_fact_ids_sha256": sha256_value(base_fact_ids),
            "member_bindings_sha256": sha256_value(member_bindings),
            "source_dataset_counts": dict(
                sorted(Counter(str(row.get("source_dataset")) for row in members).items())
            ),
            "strict_completion_count": sum(
                row.get("prompt_quality_tier") == "strict_factual_completion"
                for row in members
            ),
            "examples": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "base_fact_id": row.get("base_fact_id"),
                    "source_dataset": row.get("source_dataset"),
                    "source_question": row.get("source_question"),
                    "subject": row.get("subject"),
                    "answer": row.get("answer"),
                    "canonical_fact": row.get("canonical_fact"),
                }
                for row in members[:5]
            ],
            "review_contract": {
                "signature_mapping_only": True,
                "item_level_fact_review_still_required": True,
                "broad_relation_normalized_required": False,
                "missing_or_defer_keeps_probe_relation_id_null": True,
                "human_gold": False,
            },
        }
        (unresolved if direction == "unresolved" else directional).append(item)
    key = lambda row: (
        str(row["probe_relation_candidate"]),
        -int(row["base_fact_count"]),
        str(row["signature_id"]),
    )
    directional.sort(key=key)
    unresolved.sort(key=key)
    return directional, unresolved


def _frame_score(seed: str, relation_id: str, split: str, base_fact_id: str) -> str:
    payload = "\u241f".join((seed, relation_id, split, base_fact_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_review_frame(
    bundle_rows: Sequence[Mapping[str, Any]],
    relation_specs: Mapping[str, Mapping[str, Any]],
    seed: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for relation_id, spec in relation_specs.items():
        expected_direction = spec["direction"]
        frame_counts = spec["review_frame_split_counts"]
        for split in SPLITS:
            eligible: List[Mapping[str, Any]] = []
            for row in bundle_rows:
                if row.get("probe_relation_candidate") != relation_id:
                    continue
                if row.get("probe_relation_candidate_direction") != expected_direction:
                    continue
                if row.get("split_assignment") != split:
                    continue
                if row.get("prompt_ready") is not True:
                    continue
                flags = row.get("prompt_risk_flags")
                if not isinstance(flags, list) or flags:
                    continue
                eligible.append(row)
            eligible.sort(
                key=lambda row: _frame_score(
                    seed, relation_id, split, str(row.get("base_fact_id"))
                )
            )
            unique_groups: List[Mapping[str, Any]] = []
            seen_groups = set()
            for row in eligible:
                split_group_id = _required_string(
                    row.get("split_group_id"), "behavior row.split_group_id"
                )
                if split_group_id in seen_groups:
                    continue
                seen_groups.add(split_group_id)
                unique_groups.append(row)
            required = int(frame_counts[split])
            if len(unique_groups) < required:
                raise ValueError(
                    f"Insufficient eligible rows for {relation_id}/{split}: "
                    f"required {required}, found {len(unique_groups)}"
                )
            for rank, row in enumerate(unique_groups[:required], start=1):
                subject = _required_string(row.get("subject_en"), "behavior row.subject_en")
                prompt = str(spec["prompt_template_en"]).format(subject=subject)
                answer = _required_string(row.get("answer_en"), "behavior row.answer_en")
                if answer.casefold() in prompt.casefold():
                    raise ValueError(
                        f"Relation prompt leaks the answer for {row.get('base_fact_id')}"
                    )
                selected.append(
                    {
                        "schema_version": FRAME_SCHEMA_VERSION,
                        "base_fact_id": row.get("base_fact_id"),
                        "candidate_id": row.get("candidate_id"),
                        "source_record_sha256": sha256_value(row),
                        "relation_signature_id": row.get("relation_signature_id"),
                        "probe_relation_candidate": relation_id,
                        "probe_relation_candidate_direction": expected_direction,
                        "proposed_probe_prompt_en": prompt,
                        "source_prompt_en": row.get("prompt_en"),
                        "subject_en": subject,
                        "answer_en": answer,
                        "answer_aliases_en": row.get("answer_aliases_en"),
                        "canonical_fact_en": row.get("canonical_fact_en"),
                        "source_question_en": row.get("source_question_en"),
                        "source_dataset": row.get("source_dataset"),
                        "source_format": row.get("source_format"),
                        "split_group_id": row.get("split_group_id"),
                        "split_assignment": split,
                        "split_status": row.get("split_status"),
                        "frame_rank_within_relation_split": rank,
                        "target_accept_count_for_split": int(
                            spec["target_accept_split_counts"][split]
                        ),
                        "initial_target_slot": rank
                        <= int(spec["target_accept_split_counts"][split]),
                        "selection_score_sha256": _frame_score(
                            seed, relation_id, split, str(row.get("base_fact_id"))
                        ),
                        "selection_before_fact_review": True,
                        "fact_review_decision": None,
                        "probe_relation_id": None,
                        "canonical_status": row.get("canonical_status"),
                        "perturbation_ready": False,
                        "blocking_reasons": [
                            "signature_review_pending",
                            "item_level_fact_review_pending",
                            "canonical_freeze_pending",
                            "split_freeze_pending",
                            "target_translation_pending",
                            "distractor_review_pending",
                        ],
                    }
                )
    selected.sort(
        key=lambda row: (
            str(row["probe_relation_candidate"]),
            SPLITS.index(str(row["split_assignment"])),
            int(row["frame_rank_within_relation_split"]),
        )
    )
    return selected


def export_review(
    *, source_dir: Path, policy_path: Path, output_dir: Path
) -> Dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    policy_path = Path(policy_path).resolve()
    output_dir = Path(output_dir).resolve()
    paths = {
        "behavior_bundle": source_dir / "behavior_input_bundle.jsonl",
        "provisional_triples": source_dir / "provisional_triples.jsonl",
        "relation_inventory": source_dir / "relation_inventory.json",
    }
    policy = read_json(policy_path)
    relation_specs = _validate_policy(policy)
    bundle_rows = read_jsonl(paths["behavior_bundle"])
    provisional_rows = read_jsonl(paths["provisional_triples"])
    inventory = read_json(paths["relation_inventory"])
    if {row.get("schema_version") for row in bundle_rows} != {
        "factual-perturbation-input-bundle-v1"
    }:
        raise ValueError("Behavior bundle has unsupported schema versions")
    if inventory.get("schema_version") != "provisional-relation-inventory-v1":
        raise ValueError("Relation inventory has unsupported schema_version")
    if inventory.get("probe_relation_candidate_policy") != policy.get(
        "source_candidate_policy_id"
    ):
        raise ValueError("Relation inventory candidate policy does not match review policy")
    _index(bundle_rows, "base_fact_id", "behavior bundle")
    _index(provisional_rows, "candidate_id", "provisional triples")
    bundle_sha = sha256_file(paths["behavior_bundle"])
    directional, unresolved = _signature_review_items(
        provisional_rows,
        bundle_sha256=bundle_sha,
        source_candidate_policy_id=str(policy["source_candidate_policy_id"]),
    )
    inventory_index = _index(inventory.get("entries", []), "signature_id", "relation inventory")
    for item in directional + unresolved:
        signature_id = str(item["signature_id"])
        if signature_id not in inventory_index:
            raise ValueError(f"Signature missing from relation inventory: {signature_id}")
    frame = _build_review_frame(
        bundle_rows, relation_specs, _required_string(policy.get("selection_seed"), "selection_seed")
    )

    directional_path = output_dir / "directional_signature_review_items.jsonl"
    unresolved_path = output_dir / "direction_unresolved_signature_review_items.jsonl"
    directional_template_path = output_dir / "directional_signature_decisions_template.jsonl"
    unresolved_template_path = output_dir / "direction_unresolved_signature_decisions_template.jsonl"
    frame_path = output_dir / "predeclared_probe_review_frame.jsonl"
    frame_ids_path = output_dir / "predeclared_probe_review_frame_base_fact_ids.json"
    initial_batch = [row for row in frame if row["initial_target_slot"] is True]
    initial_batch_path = output_dir / "initial_probe_review_batch.jsonl"
    initial_ids_path = output_dir / "initial_probe_review_batch_base_fact_ids.json"
    write_jsonl(directional_path, directional)
    write_jsonl(unresolved_path, unresolved)

    def template(item: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "schema_version": DECISION_SCHEMA_VERSION,
            "signature_id": item["signature_id"],
            "signature_review_item_sha256": sha256_value(item),
            "source_bundle_sha256": bundle_sha,
            "member_base_fact_ids_sha256": item["member_base_fact_ids_sha256"],
            "relation_raw": item["relation_raw"],
            "subject_type": item["subject_type"],
            "answer_type": item["answer_type"],
            "decision": None,
            "relation_normalized": None,
            "probe_relation_id": None,
            "probe_direction": None,
            "confidence": None,
            "reason": None,
            "reviewer_type": None,
            "reviewer_id": None,
            "review_method": None,
            "reviewed_at": None,
            "human_gold": False,
        }

    write_jsonl(directional_template_path, [template(item) for item in directional])
    write_jsonl(unresolved_template_path, [template(item) for item in unresolved])
    write_jsonl(frame_path, frame)
    write_json(frame_ids_path, [str(row["base_fact_id"]) for row in frame])
    write_jsonl(initial_batch_path, initial_batch)
    write_json(initial_ids_path, [str(row["base_fact_id"]) for row in initial_batch])
    artifacts = {
        "directional_signature_review_items": _artifact(
            directional_path,
            schema_version=SIGNATURE_ITEM_SCHEMA_VERSION,
            record_count=len(directional),
        ),
        "direction_unresolved_signature_review_items": _artifact(
            unresolved_path,
            schema_version=SIGNATURE_ITEM_SCHEMA_VERSION,
            record_count=len(unresolved),
        ),
        "directional_signature_decisions_template": _artifact(
            directional_template_path,
            schema_version=DECISION_SCHEMA_VERSION,
            record_count=len(directional),
        ),
        "direction_unresolved_signature_decisions_template": _artifact(
            unresolved_template_path,
            schema_version=DECISION_SCHEMA_VERSION,
            record_count=len(unresolved),
        ),
        "predeclared_probe_review_frame": _artifact(
            frame_path, schema_version=FRAME_SCHEMA_VERSION, record_count=len(frame)
        ),
        "predeclared_probe_review_frame_base_fact_ids": {
            "path": str(frame_ids_path),
            "sha256": sha256_file(frame_ids_path),
            "record_count": len(frame),
            "schema_version": "json-array-of-base-fact-id-v1",
        },
        "initial_probe_review_batch": _artifact(
            initial_batch_path,
            schema_version=FRAME_SCHEMA_VERSION,
            record_count=len(initial_batch),
        ),
        "initial_probe_review_batch_base_fact_ids": {
            "path": str(initial_ids_path),
            "sha256": sha256_file(initial_ids_path),
            "record_count": len(initial_batch),
            "schema_version": "json-array-of-base-fact-id-v1",
        },
    }
    manifest = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": "pre_perturbation_review_materialized",
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "policy_id": policy["policy_id"],
            "source_candidate_policy_id": policy["source_candidate_policy_id"],
        },
        "sources": {
            "behavior_bundle": _artifact(
                paths["behavior_bundle"],
                schema_version="factual-perturbation-input-bundle-v1",
                record_count=len(bundle_rows),
            ),
            "provisional_triples": _artifact(
                paths["provisional_triples"],
                schema_version="provisional-factual-triple-v1",
                record_count=len(provisional_rows),
            ),
            "relation_inventory": _artifact(
                paths["relation_inventory"],
                schema_version="provisional-relation-inventory-v1",
                record_count=int(inventory.get("signature_count", 0)),
            ),
        },
        "counts": {
            "directional_signatures": len(directional),
            "direction_unresolved_signatures": len(unresolved),
            "predeclared_probe_review_frame": len(frame),
            "initial_probe_review_batch": len(initial_batch),
            "target_accept_count": sum(
                sum(int(value) for value in spec["target_accept_split_counts"].values())
                for spec in relation_specs.values()
            ),
        },
        "artifacts": artifacts,
        "safety_contract": {
            "signature_approval_is_not_fact_approval": True,
            "probe_relation_freeze_performed": False,
            "canonical_freeze_performed": False,
            "split_freeze_performed": False,
            "perturbation_performed": False,
            "model_or_network_used": False,
            "unresolved_direction_excluded_from_frame": True,
        },
    }
    manifest_path = output_dir / "relation_review_export_manifest.json"
    write_json(manifest_path, manifest)
    report_path = output_dir / "PRE_PERTURBATION_STATUS.md"
    write_text(
        report_path,
        "\n".join(
            [
                "# Qwen public-benchmark relation review v1",
                "",
                "This package is offline preparation only. It does not contain model outputs or perturbations.",
                "",
                f"- Source base facts: {len(bundle_rows):,}",
                f"- Directional signatures queued: {len(directional):,}",
                f"- Direction-unresolved signatures queued separately: {len(unresolved):,}",
                f"- Predeclared fact-review frame: {len(frame):,}",
                f"- Initial target slots: {len(initial_batch):,}",
                "- Broad `relation_normalized`: intentionally not required for this first probe cohort",
                "- Current state: signature review and item-level fact review required",
                "- Perturbation/model execution: not run",
                "",
            ]
        ),
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        **manifest["counts"],
    }


def _load_bound_artifact(
    manifest_path: Path, binding: Mapping[str, Any]
) -> Tuple[Path, List[Dict[str, Any]]]:
    path = Path(_required_string(binding.get("path"), "artifact.path"))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    else:
        path = path.resolve()
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"Artifact SHA mismatch: {path}")
    rows = read_jsonl(path)
    if len(rows) != binding.get("record_count"):
        raise ValueError(f"Artifact record_count mismatch: {path}")
    if {row.get("schema_version") for row in rows} != {binding.get("schema_version")}:
        raise ValueError(f"Artifact schema mismatch: {path}")
    return path, rows


def _load_export(manifest_path: Path) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != EXPORT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported relation review export manifest")
    if manifest.get("tool_version") != TOOL_VERSION:
        raise ValueError("Unsupported relation review tool version")
    source = manifest.get("sources", {}).get("behavior_bundle")
    if not isinstance(source, dict):
        raise ValueError("Export manifest is missing behavior bundle binding")
    source_path = Path(_required_string(source.get("path"), "behavior_bundle.path"))
    if sha256_file(source_path) != source.get("sha256"):
        raise ValueError("Behavior bundle changed after relation review export")
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for key in (
        "directional_signature_review_items",
        "direction_unresolved_signature_review_items",
        "predeclared_probe_review_frame",
    ):
        binding = manifest.get("artifacts", {}).get(key)
        if not isinstance(binding, dict):
            raise ValueError(f"Export manifest is missing artifact binding: {key}")
        _, loaded[key] = _load_bound_artifact(manifest_path, binding)
    return manifest, loaded


def _reviewer_fields(raw: Mapping[str, Any], label: str) -> Dict[str, Any]:
    reviewer_type = raw.get("reviewer_type")
    if reviewer_type not in {"human", "codex_proxy"}:
        raise ValueError(f"{label}.reviewer_type must be human or codex_proxy")
    human_gold = raw.get("human_gold")
    if type(human_gold) is not bool:
        raise ValueError(f"{label}.human_gold must be boolean")
    if reviewer_type == "codex_proxy" and human_gold is not False:
        raise ValueError(f"{label} codex_proxy cannot claim human gold")
    reviewed_at = _required_string(raw.get("reviewed_at"), f"{label}.reviewed_at")
    try:
        parsed = dt.datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}.reviewed_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label}.reviewed_at must include a timezone")
    return {
        "reviewer_type": reviewer_type,
        "reviewer_id": _required_string(raw.get("reviewer_id"), f"{label}.reviewer_id"),
        "review_method": _required_string(
            raw.get("review_method"), f"{label}.review_method"
        ),
        "reviewed_at": reviewed_at,
        "human_gold": human_gold,
    }


def materialize_decisions(
    *, manifest_path: Path, review_plan_path: Path, output_dir: Path
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest, artifacts = _load_export(manifest_path)
    plan = read_json(Path(review_plan_path))
    if plan.get("schema_version") != REVIEW_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported relation review plan schema_version")
    if plan.get("export_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Review plan is not bound to the export manifest")
    family_reviews = plan.get("directional_family_reviews")
    if not isinstance(family_reviews, dict):
        raise ValueError("Review plan lacks directional_family_reviews")
    unresolved_review = plan.get("direction_unresolved_review")
    if not isinstance(unresolved_review, dict):
        raise ValueError("Review plan lacks direction_unresolved_review")
    expected_families = {
        str(item["probe_relation_candidate"])
        for item in artifacts["directional_signature_review_items"]
    }
    if set(family_reviews) != expected_families:
        raise ValueError(
            "Review plan family coverage mismatch; "
            f"missing={sorted(expected_families - set(family_reviews))[:5]} "
            f"extra={sorted(set(family_reviews) - expected_families)[:5]}"
        )

    def decision_rows(
        items: Sequence[Mapping[str, Any]], *, unresolved: bool
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for item in items:
            family = str(item["probe_relation_candidate"])
            review = unresolved_review if unresolved else family_reviews.get(family)
            if not isinstance(review, dict):
                raise ValueError(f"Review plan does not cover relation family: {family}")
            if review.get("reviewed_all_signature_items") is not True:
                raise ValueError(f"Review plan did not attest full signature review: {family}")
            reviewer = _reviewer_fields(review, f"review {family}")
            overrides = review.get("signature_overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"signature_overrides must be an object: {family}")
            override = overrides.get(str(item["signature_id"]), {})
            if not isinstance(override, dict):
                raise ValueError(f"Invalid signature override: {item['signature_id']}")
            decision = override.get("decision", review.get("default_decision"))
            if decision not in DECISIONS:
                raise ValueError(f"Invalid decision for {item['signature_id']}: {decision}")
            reason = override.get("reason", review.get("default_reason"))
            confidence = override.get("confidence", review.get("default_confidence"))
            if confidence not in {"high", "medium", "low"}:
                raise ValueError(f"Invalid confidence for {item['signature_id']}")
            reason = _required_string(reason, f"decision {item['signature_id']}.reason")
            if unresolved and decision == "approve_mapping":
                raise ValueError("Direction-unresolved signatures cannot be bulk approved")
            probe_relation_id = (
                item["probe_relation_candidate"] if decision == "approve_mapping" else None
            )
            probe_direction = (
                item["probe_relation_candidate_direction"]
                if decision == "approve_mapping"
                else None
            )
            output.append(
                {
                    "schema_version": DECISION_SCHEMA_VERSION,
                    "signature_id": item["signature_id"],
                    "signature_review_item_sha256": sha256_value(item),
                    "source_bundle_sha256": item["source_bundle_sha256"],
                    "member_base_fact_ids_sha256": item[
                        "member_base_fact_ids_sha256"
                    ],
                    "relation_raw": item["relation_raw"],
                    "subject_type": item["subject_type"],
                    "answer_type": item["answer_type"],
                    "decision": decision,
                    "relation_normalized": None,
                    "probe_relation_id": probe_relation_id,
                    "probe_direction": probe_direction,
                    "confidence": confidence,
                    "reason": reason,
                    **reviewer,
                }
            )
        return output

    directional = decision_rows(
        artifacts["directional_signature_review_items"], unresolved=False
    )
    unresolved = decision_rows(
        artifacts["direction_unresolved_signature_review_items"], unresolved=True
    )
    directional_ids = {
        str(item["signature_id"])
        for item in artifacts["directional_signature_review_items"]
    }
    for family, review in family_reviews.items():
        extra = set(review.get("signature_overrides", {})) - directional_ids
        if extra:
            raise ValueError(f"Unknown signature overrides for {family}: {sorted(extra)[:5]}")
    unresolved_ids = {
        str(item["signature_id"])
        for item in artifacts["direction_unresolved_signature_review_items"]
    }
    extra_unresolved = set(unresolved_review.get("signature_overrides", {})) - unresolved_ids
    if extra_unresolved:
        raise ValueError(
            f"Unknown direction-unresolved overrides: {sorted(extra_unresolved)[:5]}"
        )
    output_dir = Path(output_dir).resolve()
    directional_path = output_dir / "directional_signature_decisions.jsonl"
    unresolved_path = output_dir / "direction_unresolved_signature_decisions.jsonl"
    write_jsonl(directional_path, directional)
    write_jsonl(unresolved_path, unresolved)
    result = {
        "schema_version": "probe-relation-signature-decision-materialization-v1",
        "tool_version": TOOL_VERSION,
        "export_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "review_plan": {
            "path": str(Path(review_plan_path).resolve()),
            "sha256": sha256_file(Path(review_plan_path).resolve()),
        },
        "artifacts": {
            "directional_signature_decisions": _artifact(
                directional_path,
                schema_version=DECISION_SCHEMA_VERSION,
                record_count=len(directional),
            ),
            "direction_unresolved_signature_decisions": _artifact(
                unresolved_path,
                schema_version=DECISION_SCHEMA_VERSION,
                record_count=len(unresolved),
            ),
        },
        "decision_counts": dict(
            sorted(Counter(row["decision"] for row in directional + unresolved).items())
        ),
        "human_gold": all(row["human_gold"] is True for row in directional + unresolved),
        "scope_limit": "signature_mapping_only_not_item_level_fact_review",
    }
    result_path = output_dir / "relation_decision_materialization_manifest.json"
    write_json(result_path, result)
    return {
        "manifest_path": str(result_path),
        "manifest_sha256": sha256_file(result_path),
        "decision_counts": result["decision_counts"],
    }


def _validate_decisions(
    decisions: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]]
) -> Dict[str, Mapping[str, Any]]:
    item_index = _index(items, "signature_id", "signature review items")
    decision_index = _index(decisions, "signature_id", "signature decisions")
    if set(decision_index) != set(item_index):
        missing = sorted(set(item_index) - set(decision_index))
        extra = sorted(set(decision_index) - set(item_index))
        raise ValueError(f"Decision coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    for signature_id, decision in decision_index.items():
        item = item_index[signature_id]
        if decision.get("schema_version") != DECISION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported decision schema: {signature_id}")
        for field in (
            "source_bundle_sha256",
            "member_base_fact_ids_sha256",
            "relation_raw",
            "subject_type",
            "answer_type",
        ):
            expected = (
                item[field]
                if field != "source_bundle_sha256"
                else item["source_bundle_sha256"]
            )
            if decision.get(field) != expected:
                raise ValueError(f"Stale decision {field}: {signature_id}")
        if decision.get("signature_review_item_sha256") != sha256_value(item):
            raise ValueError(f"Stale signature review item hash: {signature_id}")
        value = decision.get("decision")
        if value not in DECISIONS:
            raise ValueError(f"Invalid decision: {signature_id}")
        _reviewer_fields(decision, f"decision {signature_id}")
        if decision.get("relation_normalized") is not None:
            raise ValueError("Broad relation_normalized is intentionally not assigned here")
        if value == "approve_mapping":
            if item.get("review_track") != "directional":
                raise ValueError(f"Cannot approve unresolved direction: {signature_id}")
            if decision.get("probe_relation_id") != item.get("probe_relation_candidate"):
                raise ValueError(f"Approved probe relation does not match candidate: {signature_id}")
            if decision.get("probe_direction") != item.get(
                "probe_relation_candidate_direction"
            ):
                raise ValueError(f"Approved direction does not match candidate: {signature_id}")
        elif decision.get("probe_relation_id") is not None or decision.get(
            "probe_direction"
        ) is not None:
            raise ValueError(f"Non-approved decision must keep probe fields null: {signature_id}")
    return decision_index


def apply_decisions(
    *,
    manifest_path: Path,
    directional_decisions_path: Path,
    unresolved_decisions_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest, artifacts = _load_export(manifest_path)
    directional_decisions = read_jsonl(Path(directional_decisions_path))
    unresolved_decisions = read_jsonl(Path(unresolved_decisions_path))
    directional_index = _validate_decisions(
        directional_decisions, artifacts["directional_signature_review_items"]
    )
    unresolved_index = _validate_decisions(
        unresolved_decisions,
        artifacts["direction_unresolved_signature_review_items"],
    )
    decisions = {**directional_index, **unresolved_index}
    relation_inventory_path = Path(manifest["sources"]["relation_inventory"]["path"])
    if sha256_file(relation_inventory_path) != manifest["sources"]["relation_inventory"]["sha256"]:
        raise ValueError("Relation inventory changed after export")
    inventory = read_json(relation_inventory_path)
    reviewed_entries: List[Dict[str, Any]] = []
    for entry in inventory.get("entries", []):
        updated = dict(entry)
        signature_id = str(entry["signature_id"])
        decision = decisions.get(signature_id)
        if decision is None:
            updated["relation_review_status"] = "out_of_probe_taxonomy"
            updated["probe_relation_id"] = None
            updated["probe_relation_status"] = "unassigned"
            updated["relation_review_decision_sha256"] = None
        else:
            outcome = str(decision["decision"])
            updated["relation_review_status"] = outcome
            updated["relation_review_decision_sha256"] = sha256_value(decision)
            updated["relation_review_provenance"] = {
                field: decision[field]
                for field in (
                    "reviewer_type",
                    "reviewer_id",
                    "review_method",
                    "reviewed_at",
                    "human_gold",
                    "confidence",
                    "reason",
                )
            }
            if outcome == "approve_mapping":
                updated["probe_relation_id"] = decision["probe_relation_id"]
                updated["probe_relation_status"] = (
                    "signature_approved_pending_item_review"
                )
            elif outcome == "reject_mapping":
                updated["probe_relation_id"] = None
                updated["probe_relation_status"] = "signature_mapping_rejected"
            else:
                updated["probe_relation_id"] = None
                updated["probe_relation_status"] = "signature_review_deferred"
        reviewed_entries.append(updated)

    reviewed_inventory = {
        "schema_version": REVIEWED_INVENTORY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "source_inventory": {
            "path": str(relation_inventory_path.resolve()),
            "sha256": sha256_file(relation_inventory_path),
        },
        "export_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "directional_decisions": {
            "path": str(Path(directional_decisions_path).resolve()),
            "sha256": sha256_file(Path(directional_decisions_path).resolve()),
        },
        "direction_unresolved_decisions": {
            "path": str(Path(unresolved_decisions_path).resolve()),
            "sha256": sha256_file(Path(unresolved_decisions_path).resolve()),
        },
        "broad_relation_normalized_status": "not_required_for_first_probe_cohort",
        "probe_relation_freeze_status": "signature_only_item_review_pending",
        "human_gold": all(row["human_gold"] is True for row in decisions.values()),
        "signature_count": len(reviewed_entries),
        "entries": reviewed_entries,
    }

    behavior_path = Path(manifest["sources"]["behavior_bundle"]["path"])
    bundle_rows = read_jsonl(behavior_path)
    overlays: List[Dict[str, Any]] = []
    for row in bundle_rows:
        signature_id = str(row["relation_signature_id"])
        decision = decisions.get(signature_id)
        approved = decision is not None and decision["decision"] == "approve_mapping"
        overlays.append(
            {
                "schema_version": OVERLAY_SCHEMA_VERSION,
                "base_fact_id": row["base_fact_id"],
                "source_record_sha256": sha256_value(row),
                "relation_signature_id": signature_id,
                "relation_raw": row["relation_raw"],
                "relation_normalized": None,
                "normalization_status": "not_required_for_first_probe_cohort",
                "probe_relation_candidate": row.get("probe_relation_candidate"),
                "probe_relation_id": decision.get("probe_relation_id") if approved else None,
                "probe_relation_status": (
                    "signature_approved_pending_item_review"
                    if approved
                    else (
                        "signature_mapping_rejected"
                        if decision is not None and decision["decision"] == "reject_mapping"
                        else "not_signature_approved"
                    )
                ),
                "relation_review_decision_sha256": (
                    sha256_value(decision) if decision is not None else None
                ),
                "canonical_status": row.get("canonical_status"),
                "split_status": row.get("split_status"),
                "perturbation_ready": False,
            }
        )

    frame_by_id = _index(
        artifacts["predeclared_probe_review_frame"],
        "base_fact_id",
        "predeclared probe review frame",
    )
    approved_frame: List[Dict[str, Any]] = []
    for frame_row in artifacts["predeclared_probe_review_frame"]:
        signature_id = str(frame_row["relation_signature_id"])
        decision = decisions.get(signature_id)
        if decision is None or decision["decision"] != "approve_mapping":
            continue
        row = dict(frame_by_id[str(frame_row["base_fact_id"])])
        row["probe_relation_id"] = decision["probe_relation_id"]
        row["signature_review_decision_sha256"] = sha256_value(decision)
        row["blocking_reasons"] = [
            reason for reason in row["blocking_reasons"] if reason != "signature_review_pending"
        ]
        row["signature_review_status"] = "approved"
        approved_frame.append(row)
    policy_path = Path(manifest["policy"]["path"])
    if sha256_file(policy_path) != manifest["policy"]["sha256"]:
        raise ValueError("Relation review policy changed after export")
    relation_specs = _validate_policy(read_json(policy_path))
    item_review_batch: List[Dict[str, Any]] = []
    for relation_id, spec in relation_specs.items():
        for split in SPLITS:
            candidates = sorted(
                (
                    row
                    for row in approved_frame
                    if row["probe_relation_id"] == relation_id
                    and row["split_assignment"] == split
                ),
                key=lambda row: int(row["frame_rank_within_relation_split"]),
            )
            required = int(spec["target_accept_split_counts"][split])
            if len(candidates) < required:
                raise ValueError(
                    f"Signature review leaves too few item-review candidates for "
                    f"{relation_id}/{split}: required {required}, found {len(candidates)}"
                )
            for post_signature_rank, row in enumerate(candidates[:required], start=1):
                selected = dict(row)
                selected["post_signature_rank_within_relation_split"] = (
                    post_signature_rank
                )
                selected["selected_for_item_review_batch"] = True
                item_review_batch.append(selected)
    item_review_batch.sort(
        key=lambda row: (
            str(row["probe_relation_id"]),
            SPLITS.index(str(row["split_assignment"])),
            int(row["post_signature_rank_within_relation_split"]),
        )
    )

    output_dir = Path(output_dir).resolve()
    inventory_path = output_dir / "reviewed_relation_inventory.json"
    overlay_path = output_dir / "relation_signature_overlay.jsonl"
    approved_frame_path = output_dir / "signature_approved_probe_review_frame.jsonl"
    approved_ids_path = output_dir / "signature_approved_probe_review_frame_base_fact_ids.json"
    item_review_batch_path = output_dir / "signature_approved_item_review_batch.jsonl"
    item_review_batch_ids_path = (
        output_dir / "signature_approved_item_review_batch_base_fact_ids.json"
    )
    write_json(inventory_path, reviewed_inventory)
    write_jsonl(overlay_path, overlays)
    write_jsonl(approved_frame_path, approved_frame)
    write_json(approved_ids_path, [str(row["base_fact_id"]) for row in approved_frame])
    write_jsonl(item_review_batch_path, item_review_batch)
    write_json(
        item_review_batch_ids_path,
        [str(row["base_fact_id"]) for row in item_review_batch],
    )
    decision_counts = Counter(row["decision"] for row in decisions.values())
    approved_frame_counts = Counter(
        str(row["probe_relation_id"]) for row in approved_frame
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": "signature_review_applied_fact_review_pending",
        "export_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "counts": {
            "source_base_facts": len(bundle_rows),
            "reviewed_signatures": len(decisions),
            "decision_counts": dict(sorted(decision_counts.items())),
            "signature_approved_base_facts": sum(
                overlay["probe_relation_id"] is not None for overlay in overlays
            ),
            "predeclared_review_frame": len(
                artifacts["predeclared_probe_review_frame"]
            ),
            "signature_approved_review_frame": len(approved_frame),
            "signature_approved_item_review_batch": len(item_review_batch),
            "signature_approved_review_frame_by_relation": dict(
                sorted(approved_frame_counts.items())
            ),
        },
        "artifacts": {
            "reviewed_relation_inventory": _artifact(
                inventory_path,
                schema_version=REVIEWED_INVENTORY_SCHEMA_VERSION,
                record_count=len(reviewed_entries),
            ),
            "relation_signature_overlay": _artifact(
                overlay_path,
                schema_version=OVERLAY_SCHEMA_VERSION,
                record_count=len(overlays),
            ),
            "signature_approved_probe_review_frame": _artifact(
                approved_frame_path,
                schema_version=FRAME_SCHEMA_VERSION,
                record_count=len(approved_frame),
            ),
            "signature_approved_probe_review_frame_base_fact_ids": {
                "path": str(approved_ids_path),
                "sha256": sha256_file(approved_ids_path),
                "record_count": len(approved_frame),
                "schema_version": "json-array-of-base-fact-id-v1",
            },
            "signature_approved_item_review_batch": _artifact(
                item_review_batch_path,
                schema_version=FRAME_SCHEMA_VERSION,
                record_count=len(item_review_batch),
            ),
            "signature_approved_item_review_batch_base_fact_ids": {
                "path": str(item_review_batch_ids_path),
                "sha256": sha256_file(item_review_batch_ids_path),
                "record_count": len(item_review_batch),
                "schema_version": "json-array-of-base-fact-id-v1",
            },
        },
        "remaining_blockers_before_perturbation": [
            "item_level_fact_review",
            "revision_and_rereview_if_needed",
            "semantic_near_duplicate_closure",
            "authoritative_historical_exposure_inventory",
            "canonical_freeze",
            "split_freeze",
            "target_translation",
            "distractor_review_or_construction",
        ],
        "not_performed": [
            "probe_relation_freeze",
            "canonical_freeze",
            "split_freeze",
            "translation",
            "distractor_generation",
            "perturbation",
            "behavior_screening",
            "hf_model_execution",
        ],
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    report_path = output_dir / "PRE_PERTURBATION_STATUS.md"
    write_text(
        report_path,
        "\n".join(
            [
                "# Qwen public-benchmark relation review v1",
                "",
                "Signature-level review has been applied. This is not a factual or split freeze.",
                "",
                f"- Reviewed signatures: {len(decisions):,}",
                f"- Approved source base facts: {summary['counts']['signature_approved_base_facts']:,}",
                f"- Approved predeclared review-frame facts: {len(approved_frame):,}",
                f"- Signature-approved item-review batch: {len(item_review_batch):,}",
                "- `relation_normalized`: intentionally left null",
                "- `probe_relation_id`: signature-approved only; item-level review remains required",
                "- Perturbation/model execution: not run",
                "",
                "## Remaining gates",
                "",
                *[
                    f"- {blocker}"
                    for blocker in summary["remaining_blockers_before_perturbation"]
                ],
                "",
            ]
        ),
    )
    apply_manifest = {
        "schema_version": APPLY_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "export_manifest": summary["export_manifest"],
        "directional_decisions": reviewed_inventory["directional_decisions"],
        "direction_unresolved_decisions": reviewed_inventory[
            "direction_unresolved_decisions"
        ],
        "artifacts": summary["artifacts"],
        "safety_contract": {
            "source_artifacts_modified": False,
            "signature_approval_is_not_fact_approval": True,
            "probe_relation_freeze_performed": False,
            "formal_bundle_emitted": False,
            "perturbation_performed": False,
        },
    }
    apply_manifest_path = output_dir / "relation_review_apply_manifest.json"
    write_json(apply_manifest_path, apply_manifest)
    return {
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest_path": str(apply_manifest_path),
        "manifest_sha256": sha256_file(apply_manifest_path),
        **summary["counts"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export signature review and a predeclared frame")
    export.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    export.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    export.add_argument("--output-dir", type=Path, required=True)

    materialize = commands.add_parser(
        "materialize-decisions", help="Bind a completed review plan to signature items"
    )
    materialize.add_argument("--export-manifest", type=Path, required=True)
    materialize.add_argument("--review-plan", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)

    apply = commands.add_parser(
        "apply", help="Validate complete decisions and emit a non-formal relation overlay"
    )
    apply.add_argument("--export-manifest", type=Path, required=True)
    apply.add_argument("--directional-decisions", type=Path, required=True)
    apply.add_argument("--direction-unresolved-decisions", type=Path, required=True)
    apply.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "export":
        result = export_review(
            source_dir=args.source_dir,
            policy_path=args.policy,
            output_dir=args.output_dir,
        )
    elif args.command == "materialize-decisions":
        result = materialize_decisions(
            manifest_path=args.export_manifest,
            review_plan_path=args.review_plan,
            output_dir=args.output_dir,
        )
    else:
        result = apply_decisions(
            manifest_path=args.export_manifest,
            directional_decisions_path=args.directional_decisions,
            unresolved_decisions_path=args.direction_unresolved_decisions,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
