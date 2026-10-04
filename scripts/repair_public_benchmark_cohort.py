#!/usr/bin/env python3
"""Repair the 160-item public-benchmark cohort without overwriting v2.

The tool is intentionally offline.  It performs one explicit, SHA-bound
replacement, applies an independently reviewed semantic revision to the
replacement, rebuilds the pre-closure distractor graph, and emits a selection
manifest that can be passed to the duplicate-closure pipeline.

It does not call a model, translate text, freeze a split, or authorize a
behavior/PATH experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


TOOL_VERSION = "public-benchmark-cohort-repair-v1"
PLAN_SCHEMA_VERSION = "public-benchmark-cohort-repair-plan-v1"
MANIFEST_SCHEMA_VERSION = "public-benchmark-cohort-repair-manifest-v1"
SELECTION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-preperturbation-selection-manifest-v1"
)
SELECTED_IDS_SCHEMA_VERSION = "public-benchmark-preperturbation-selected-ids-v1"
SUMMARY_SCHEMA_VERSION = "public-benchmark-cohort-repair-summary-v1"
LINEAGE_SCHEMA_VERSION = "public-benchmark-cohort-repair-lineage-v1"
REREVIEW_SCHEMA_VERSION = "public-benchmark-preperturbation-rereview-v1"
INPUT_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
DISTRACTOR_SCHEMA_VERSION = "public-benchmark-preperturbation-distractor-candidate-v1"
SPLITS = ("development", "validation", "sealed")
DISTRACTORS_PER_FACT = 2


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in JSONL: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).write_bytes(
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    published_path: Optional[Path] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str((published_path or path).resolve()),
        "sha256": sha256_file(path),
        "byte_count": Path(path).stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _validate_bound_input(path: Path, binding: Mapping[str, Any], label: str) -> None:
    if Path(path).resolve() != Path(_required_string(binding.get("path"), label)).resolve():
        raise ValueError(f"{label} path mismatch")
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")


def _index(rows: Sequence[Mapping[str, Any]], field: str, label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for position, raw in enumerate(rows, start=1):
        row = dict(raw)
        key = _required_string(row.get(field), f"{label} row {position}.{field}")
        if key in result:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        result[key] = row
    return result


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _answer_terms(row: Mapping[str, Any]) -> set[str]:
    answer = _required_string(row.get("answer_en"), "answer_en")
    aliases = row.get("answer_aliases_en")
    if not isinstance(aliases, list) or not aliases:
        raise ValueError(f"{row.get('base_fact_id')} has no answer_aliases_en")
    terms = {_normalize(value) for value in aliases if str(value).strip()}
    if _normalize(answer) not in terms:
        raise ValueError(f"{row.get('base_fact_id')} answer_en is absent from aliases")
    return terms


def _derive_prompt(row: Mapping[str, Any]) -> str:
    fact = _required_string(row.get("canonical_fact_en"), "canonical_fact_en")
    answer = _required_string(row.get("answer_en"), "answer_en")
    fact_without_punctuation = fact.rstrip(".!?。！？").rstrip()
    if not fact_without_punctuation.casefold().endswith(answer.casefold()):
        raise ValueError("Revised canonical_fact_en must end with answer_en")
    prompt = fact_without_punctuation[: -len(answer)].rstrip()
    if not prompt:
        raise ValueError("Revised prompt would be empty")
    return prompt


def _make_preclosure_distractors(
    rows: Sequence[Mapping[str, Any]], policy_id: str
) -> tuple[Dict[str, list[Dict[str, Any]]], list[Dict[str, Any]]]:
    pools: Dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pools[(str(row["probe_relation_id"]), str(row["split_assignment"]))].append(row)
    by_target: Dict[str, list[Dict[str, Any]]] = {}
    flattened: list[Dict[str, Any]] = []
    for target in rows:
        target_id = str(target["base_fact_id"])
        relation = str(target["probe_relation_id"])
        split = str(target["split_assignment"])
        target_terms = _answer_terms(target)
        eligible = []
        for donor in pools[(relation, split)]:
            donor_id = str(donor["base_fact_id"])
            if donor_id == target_id or target_terms.intersection(_answer_terms(donor)):
                continue
            score = sha256_value([policy_id, target_id, donor_id, donor.get("answer_en")])
            eligible.append((score, donor_id, donor))
        eligible.sort(key=lambda item: (item[0], item[1]))
        if len(eligible) < DISTRACTORS_PER_FACT:
            raise ValueError(f"Insufficient distractor donors for {target_id}")
        candidates: list[Dict[str, Any]] = []
        for rank, (score, donor_id, donor) in enumerate(
            eligible[:DISTRACTORS_PER_FACT], start=1
        ):
            candidate = {
                "schema_version": DISTRACTOR_SCHEMA_VERSION,
                "distractor_id": "pre_dist_"
                + sha256_value([target_id, donor_id, score])[:20],
                "target_base_fact_id": target_id,
                "donor_base_fact_id": donor_id,
                "probe_relation_id": relation,
                "split_assignment": split,
                "rank": rank,
                "text_en": donor["answer_en"],
                "answer_en": donor["answer_en"],
                "answer_aliases_en": copy.deepcopy(donor["answer_aliases_en"]),
                "selection_score_sha256": score,
                "selection_method": "deterministic_same_relation_split_alias_disjoint_v1",
                "verification_status": "codex_proxy_pending_review",
                "status": "codex_proxy_pending_review",
                "verified": False,
                "human_gold": False,
            }
            candidates.append(candidate)
            flattened.append(candidate)
        by_target[target_id] = candidates
    return by_target, flattened


def _replacement_row(
    source: Mapping[str, Any],
    frame: Mapping[str, Any],
    decision: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    source_bundle_sha256: str,
    frame_sha256: str,
    decision_file_sha256: str,
    plan_sha256: str,
) -> Dict[str, Any]:
    replacement = plan["replacement"]
    row = copy.deepcopy(dict(source))
    for field, value in replacement["updates"].items():
        row[field] = copy.deepcopy(value)
    row["canonical_fact"] = row["canonical_fact_en"]
    row["base_id"] = row["base_fact_id"]
    row.update(
        {
            "bundle_status": "preperturbation_provisional_not_frozen",
            "canonical_status": "pending_review",
            "canonical_freeze_status": "provisional_not_frozen",
            "evidence_tier": "provisional_single_model",
            "human_gold": False,
            "probe_relation_id": frame["probe_relation_id"],
            "probe_relation_status": "item_reviewed_preperturbation_pending_freeze",
            "relation_normalized": None,
            "normalization_status": "not_required_for_first_probe_cohort",
            "prompt_en": _derive_prompt(row),
            "prompt_template_id": "canonical_fact_answer_suffix_en_provisional_v1",
            "prompt_version": "public-benchmark-preperturbation-open-completion-v1",
            "prompt_quality_tier": "strict_factual_completion",
            "prompt_derivation_method": "canonical_fact_en_terminal_answer_or_accepted_alias_suffix_split",
            "prompt_ready": True,
            "split_assignment": frame["split_assignment"],
            "split_group_id": frame["split_group_id"],
            "split_status": "provisional_not_frozen",
            "distractor_candidates": [],
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
                "source_record_sha256": sha256_value(source),
                "frame_artifact_sha256": frame_sha256,
                "frame_record_sha256": sha256_value(frame),
                "review_decision_artifact_sha256": decision_file_sha256,
                "review_decision_record_sha256": sha256_value(decision),
                "repair_plan_sha256": plan_sha256,
                "source_review_outcome": decision["decision"],
                "cohort_eligibility_disposition": "retain_revise",
                "revision_applied": True,
                "revision_trigger": "review_requested_revision",
                "independent_rereview_decision": replacement["rereview"]["decision"],
            },
        }
    )
    for field in ("answer_en", "canonical_fact_en", "subject_en"):
        _required_string(row.get(field), f"replacement.{field}")
    _answer_terms(row)
    return row


def repair_cohort(
    *,
    plan_path: Path,
    selected_bundle_path: Path,
    selection_manifest_path: Path,
    full_pool_path: Path,
    frame_path: Path,
    replacement_decisions_path: Path,
    relation_policy_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    selected_bundle_path = Path(selected_bundle_path).resolve()
    selection_manifest_path = Path(selection_manifest_path).resolve()
    full_pool_path = Path(full_pool_path).resolve()
    frame_path = Path(frame_path).resolve()
    replacement_decisions_path = Path(replacement_decisions_path).resolve()
    relation_policy_path = Path(relation_policy_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    plan = read_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported cohort repair plan")
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Repair plan inputs are missing")
    for key, path in (
        ("selected_bundle", selected_bundle_path),
        ("selection_manifest", selection_manifest_path),
        ("full_pool", full_pool_path),
        ("review_frame", frame_path),
        ("replacement_decisions", replacement_decisions_path),
        ("relation_policy", relation_policy_path),
    ):
        _validate_bound_input(path, inputs[key], f"inputs.{key}")

    source_rows = read_jsonl(selected_bundle_path)
    if len(source_rows) != 160:
        raise ValueError(f"Expected 160 selected rows, found {len(source_rows)}")
    source_index = _index(source_rows, "base_fact_id", "selected cohort")
    full_rows = read_jsonl(full_pool_path)
    full_index = _index(full_rows, "base_fact_id", "full pool")
    frame_index = _index(read_jsonl(frame_path), "base_fact_id", "review frame")
    decision_index = _index(
        read_jsonl(replacement_decisions_path), "base_fact_id", "replacement decisions"
    )

    removal = plan.get("removal")
    replacement = plan.get("replacement")
    if not isinstance(removal, dict) or not isinstance(replacement, dict):
        raise ValueError("Repair plan removal/replacement is invalid")
    remove_id = _required_string(removal.get("base_fact_id"), "removal.base_fact_id")
    keep_id = _required_string(removal.get("same_fact_representative_id"), "removal.same_fact_representative_id")
    add_id = _required_string(replacement.get("base_fact_id"), "replacement.base_fact_id")
    if remove_id not in source_index or keep_id not in source_index:
        raise ValueError("Removal or same-fact representative is absent from selected cohort")
    if add_id in source_index or add_id not in full_index or add_id not in frame_index:
        raise ValueError("Replacement membership is invalid")
    decision = decision_index.get(add_id)
    if decision is None or decision.get("decision") != "revise":
        raise ValueError("Replacement must have an explicit prior revise decision")
    if replacement.get("rereview", {}).get("decision") != "accept":
        raise ValueError("Replacement revision requires independent accept rereview")
    if replacement["rereview"].get("reviewer_id") == decision.get("review_provenance", {}).get("reviewer_id"):
        raise ValueError("Replacement rereviewer must differ from original reviewer")

    new_row = _replacement_row(
        full_index[add_id],
        frame_index[add_id],
        decision,
        plan,
        source_bundle_sha256=sha256_file(full_pool_path),
        frame_sha256=sha256_file(frame_path),
        decision_file_sha256=sha256_file(replacement_decisions_path),
        plan_sha256=sha256_file(plan_path),
    )
    if new_row["probe_relation_id"] != source_index[remove_id]["probe_relation_id"]:
        raise ValueError("Replacement relation does not match removal relation")
    if new_row["split_assignment"] != source_index[remove_id]["split_assignment"]:
        raise ValueError("Replacement pre-closure split does not match removal split")

    repaired_rows: list[Dict[str, Any]] = []
    for source in source_rows:
        source_id = str(source["base_fact_id"])
        row = copy.deepcopy(new_row if source_id == remove_id else source)
        row["cohort_repair_lineage"] = {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "source_selected_bundle_sha256": sha256_file(selected_bundle_path),
            "source_record_sha256": sha256_value(source) if source_id != remove_id else None,
            "repair_plan_sha256": sha256_file(plan_path),
            "disposition": "replacement" if source_id == remove_id else "retained",
            "replaces_base_fact_id": remove_id if source_id == remove_id else None,
            "same_fact_representative_id": keep_id if source_id == remove_id else None,
        }
        repaired_rows.append(row)

    repaired_index = _index(repaired_rows, "base_fact_id", "repaired cohort")
    if len(repaired_index) != 160 or remove_id in repaired_index or add_id not in repaired_index:
        raise ValueError("Repair did not preserve a 160-row unique cohort")
    counts = Counter(
        f"{row['probe_relation_id']}/{row['split_assignment']}" for row in repaired_rows
    )
    expected_counts = plan.get("expected_preclosure_relation_split_counts")
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(f"Pre-closure relation/split quota mismatch: {dict(counts)}")

    policy = read_json(relation_policy_path)
    policy_id = _required_string(policy.get("policy_id"), "relation policy.policy_id")
    by_target, distractor_rows = _make_preclosure_distractors(repaired_rows, policy_id)
    for row in repaired_rows:
        row["distractor_candidates"] = copy.deepcopy(by_target[row["base_fact_id"]])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        bundle_path = temporary / "preperturbation_behavior_input_bundle.jsonl"
        distractor_path = temporary / "distractor_candidates.jsonl"
        ids_path = temporary / "selected_base_fact_ids.json"
        summary_path = temporary / "summary.json"
        lineage_path = temporary / "cohort_repair_lineage.jsonl"
        rereview_path = temporary / "replacement_rereview.jsonl"
        manifest_path = temporary / "cohort_repair_manifest.json"
        selection_path = temporary / "selection_manifest.json"
        write_jsonl(bundle_path, repaired_rows)
        write_jsonl(distractor_path, distractor_rows)
        selected_ids = [row["base_fact_id"] for row in repaired_rows]
        selected_by_group: Dict[str, list[str]] = defaultdict(list)
        for row in repaired_rows:
            selected_by_group[f"{row['probe_relation_id']}/{row['split_assignment']}"].append(row["base_fact_id"])
        write_json(
            ids_path,
            {
                "schema_version": SELECTED_IDS_SCHEMA_VERSION,
                "relation_order": policy.get("relation_order"),
                "split_order": list(SPLITS),
                "quotas": plan["preclosure_quotas"],
                "selected_base_fact_count": len(selected_ids),
                "selected_base_fact_ids": selected_ids,
                "selected_base_fact_ids_by_relation_split": dict(selected_by_group),
            },
        )
        lineage_rows = [row["cohort_repair_lineage"] | {"base_fact_id": row["base_fact_id"]} for row in repaired_rows]
        write_jsonl(lineage_path, lineage_rows)
        write_jsonl(
            rereview_path,
            [
                {
                    "schema_version": REREVIEW_SCHEMA_VERSION,
                    "source_base_fact_id": add_id,
                    "trigger": "review_requested_revision",
                    "revised_record_sha256": sha256_value(new_row),
                    **copy.deepcopy(replacement["rereview"]),
                    "human_gold": False,
                    "evidence": copy.deepcopy(replacement["evidence"]),
                }
            ],
        )
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "repaired_preperturbation_bundle_materialized_not_frozen",
            "input_base_fact_count": len(source_rows),
            "output_base_fact_count": len(repaired_rows),
            "removed_base_fact_id": remove_id,
            "same_fact_representative_id": keep_id,
            "replacement_base_fact_id": add_id,
            "relation_split_counts": dict(sorted(counts.items())),
            "distractor_candidate_count": len(distractor_rows),
            "distractors_per_fact": DISTRACTORS_PER_FACT,
            "model_or_network_used": False,
            "canonical_freeze_performed": False,
            "split_freeze_performed": False,
            "perturbation_authorized": False,
        }
        write_json(summary_path, summary)

        def out_binding(path: Path, schema: Optional[str] = None, count: Optional[int] = None) -> Dict[str, Any]:
            return _binding(
                path,
                schema_version=schema,
                record_count=count,
                published_path=output_dir / path.name,
            )

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "cohort_repaired_not_frozen",
            "inputs": {
                "repair_plan": _binding(plan_path, schema_version=PLAN_SCHEMA_VERSION),
                "selected_bundle": _binding(selected_bundle_path, schema_version=INPUT_BUNDLE_SCHEMA_VERSION, record_count=len(source_rows)),
                "selection_manifest": _binding(selection_manifest_path, schema_version=SELECTION_MANIFEST_SCHEMA_VERSION),
                "full_pool": _binding(full_pool_path, schema_version=INPUT_BUNDLE_SCHEMA_VERSION, record_count=len(full_rows)),
                "review_frame": _binding(frame_path, record_count=len(frame_index)),
                "replacement_decisions": _binding(replacement_decisions_path, record_count=len(decision_index)),
                "relation_policy": _binding(relation_policy_path),
            },
            "repair": {
                "removed_base_fact_id": remove_id,
                "same_fact_representative_id": keep_id,
                "replacement_base_fact_id": add_id,
                "replacement_revision_sha256": sha256_value(replacement["updates"]),
                "replacement_rereview": copy.deepcopy(replacement["rereview"]),
            },
            "checks": {
                "record_count_preserved": True,
                "unique_base_fact_ids": True,
                "preclosure_relation_split_quotas_preserved": True,
                "replacement_prior_review_was_revise": True,
                "replacement_independently_rereviewed_accept": True,
                "distractors_rebuilt": True,
                "all_distractors_same_relation_and_preclosure_split": True,
                "all_distractor_aliases_disjoint": True,
            },
            "artifacts": {
                "preperturbation_behavior_input_bundle": out_binding(bundle_path, INPUT_BUNDLE_SCHEMA_VERSION, len(repaired_rows)),
                "distractor_candidates": out_binding(distractor_path, DISTRACTOR_SCHEMA_VERSION, len(distractor_rows)),
                "selected_base_fact_ids": out_binding(ids_path, SELECTED_IDS_SCHEMA_VERSION),
                "cohort_repair_lineage": out_binding(lineage_path, LINEAGE_SCHEMA_VERSION, len(lineage_rows)),
                "replacement_rereview": out_binding(rereview_path, REREVIEW_SCHEMA_VERSION, 1),
                "summary": out_binding(summary_path, SUMMARY_SCHEMA_VERSION),
            },
            "safety_contract": {
                "network_or_model_used": False,
                "model_inference_performed": False,
                "translation_performed": False,
                "perturbation_performed": False,
                "canonical_freeze_performed": False,
                "split_freeze_performed": False,
                "path_not_token_experiment_performed": False,
            },
        }
        write_json(manifest_path, manifest)
        selection_manifest = copy.deepcopy(read_json(selection_manifest_path))
        selection_manifest.update(
            {
                "tool_version": TOOL_VERSION,
                "selected_base_fact_count": len(repaired_rows),
                "selected_counts_by_relation_split": dict(sorted(counts.items())),
                "distractor_candidate_count": len(distractor_rows),
                "repair_manifest": out_binding(manifest_path, MANIFEST_SCHEMA_VERSION),
            }
        )
        selection_manifest["artifacts"] = {
            "preperturbation_behavior_input_bundle": out_binding(bundle_path, INPUT_BUNDLE_SCHEMA_VERSION, len(repaired_rows)),
            "distractor_candidates": out_binding(distractor_path, DISTRACTOR_SCHEMA_VERSION, len(distractor_rows)),
            "selected_base_fact_ids": out_binding(ids_path, SELECTED_IDS_SCHEMA_VERSION),
            "cohort_repair_lineage": out_binding(lineage_path, LINEAGE_SCHEMA_VERSION, len(lineage_rows)),
            "replacement_rereview": out_binding(rereview_path, REREVIEW_SCHEMA_VERSION, 1),
            "summary": out_binding(summary_path, SUMMARY_SCHEMA_VERSION),
        }
        write_json(selection_path, selection_manifest)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "bundle_path": str(output_dir / "preperturbation_behavior_input_bundle.jsonl"),
        "selection_manifest_path": str(output_dir / "selection_manifest.json"),
        "repair_manifest_path": str(output_dir / "cohort_repair_manifest.json"),
        "record_count": len(repaired_rows),
        "distractor_candidate_count": len(distractor_rows),
        "perturbation_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selected-bundle", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--full-pool", type=Path, required=True)
    parser.add_argument("--review-frame", type=Path, required=True)
    parser.add_argument("--replacement-decisions", type=Path, required=True)
    parser.add_argument("--relation-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = repair_cohort(
        plan_path=args.plan,
        selected_bundle_path=args.selected_bundle,
        selection_manifest_path=args.selection_manifest,
        full_pool_path=args.full_pool,
        frame_path=args.review_frame,
        replacement_decisions_path=args.replacement_decisions,
        relation_policy_path=args.relation_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
