#!/usr/bin/env python3
"""Audit exact-HF Development behavior and freeze directed/control matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_binding(path: Path, schema_version: str) -> Dict[str, Any]:
    return {
        "path": path.name,
        "schema_version": schema_version,
        "byte_count": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0b-dir", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--proxy-run-dir", type=Path, required=True)
    return parser.parse_args()


def has_top_score_tie(row: Dict[str, Any]) -> bool:
    scores = sorted(
        float(value["sequence_logprob"])
        for value in row["choice_scores"].values()
    )
    return scores[-1] == scores[-2]


def main() -> int:
    args = parse_args()
    g0b_dir = args.g0b_dir.resolve()
    behavior_dir = args.behavior_dir.resolve()
    proxy_dir = args.proxy_run_dir.resolve()
    model_manifest_path = g0b_dir / "exact_hf_model_manifest.json"
    render_manifest_path = g0b_dir / "exact_hf_render_manifest.json"
    gate_path = g0b_dir / "g0b_gate_manifest.json"
    renders_path = g0b_dir / "exact_hf_rendered_inputs.jsonl"
    queue_path = g0b_dir / "hf_replay_queue.jsonl"
    results_path = behavior_dir / "exact_hf_development_behavior_results.jsonl"
    labels_path = behavior_dir / "exact_hf_defect_control_labels.jsonl"
    summary_path = behavior_dir / "exact_hf_development_summary.json"
    proxy_results_path = proxy_dir / "simulation_results.jsonl"

    model_manifest = read_json(model_manifest_path)
    render_manifest = read_json(render_manifest_path)
    gate = read_json(gate_path)
    summary = read_json(summary_path)
    renders = read_jsonl(renders_path)
    queue = read_jsonl(queue_path)
    results = read_jsonl(results_path)
    labels = read_jsonl(labels_path)
    proxy_results = read_jsonl(proxy_results_path)

    if gate.get("gate_passed") is not True or summary.get("status") != "complete":
        raise ValueError("G0B or exact-HF Development behavior is incomplete")
    if len(results) != 960 or len({row["render_id"] for row in results}) != 960:
        raise ValueError("exact-HF results are not 960 unique rows")
    if len(labels) != 192:
        raise ValueError("exact-HF labels are not 192 variants")
    if Counter(row["split_assignment"] for row in results) != Counter({"development": 960}):
        raise ValueError("exact-HF behavior exposed a non-Development split")
    if not all(
        row.get("terminal_status") == "completed"
        and row.get("choice") in {"A", "B", "C"}
        and all(math.isfinite(float(value["sequence_logprob"])) for value in row["choice_scores"].values())
        for row in results
    ):
        raise ValueError("exact-HF result has a nonterminal or invalid score")
    if not all(not row["hf_zh_specific_strict"] or row["hf_directed_candidate"] for row in labels):
        raise ValueError("strict labels are not nested inside directed labels")
    if not all(not (row["hf_directed_candidate"] and row["hf_resistant_control"]) for row in labels):
        raise ValueError("directed and resistant labels overlap")
    if summary["artifacts"]["results"]["sha256"] != sha256_bytes(results_path.read_bytes()):
        raise ValueError("result SHA does not match summary")
    if summary["artifacts"]["labels"]["sha256"] != sha256_bytes(labels_path.read_bytes()):
        raise ValueError("label SHA does not match summary")

    result_by_fact = {}
    for row in results:
        result_by_fact.setdefault(row["base_fact_id"], row)
    originals = {
        (row["base_fact_id"], row["language"]): row
        for row in results
        if row["variant"] == "original"
    }
    by_variant = {
        (row["base_fact_id"], row.get("distractor_id"), row["language"], row["variant"]): row
        for row in results
        if row["variant"] != "original"
    }

    def required_arms(label: Dict[str, Any], languages: Iterable[str]) -> List[Dict[str, Any]]:
        rows_for_label = []
        for language in languages:
            rows_for_label.append(originals[(label["base_fact_id"], language)])
            for variant in ("neutral", "targeted"):
                rows_for_label.append(
                    by_variant[(label["base_fact_id"], label["distractor_id"], language, variant)]
                )
        return rows_for_label

    directed = [
        row for row in labels
        if row["hf_directed_candidate"]
        and not any(has_top_score_tie(arm) for arm in required_arms(row, ("zh",)))
    ]
    resistant_raw = [row for row in labels if row["hf_resistant_control"]]
    resistant = [
        row for row in resistant_raw
        if not any(has_top_score_tie(arm) for arm in required_arms(row, ("en", "zh")))
    ]
    tied_directed_ids = sorted(
        f"{row['base_fact_id']}:{row['distractor_id']}"
        for row in labels
        if row["hf_directed_candidate"] and row not in directed
    )
    tied_resistant_ids = sorted(
        f"{row['base_fact_id']}:{row['distractor_id']}"
        for row in resistant_raw
        if row not in resistant
    )
    match_rows = []
    for target in directed:
        target_component = result_by_fact[target["base_fact_id"]]["leakage_component_id"]
        eligible = [
            control
            for control in resistant
            if control["probe_relation_id"] == target["probe_relation_id"]
            and control["answer_type"] == target["answer_type"]
            and result_by_fact[control["base_fact_id"]]["leakage_component_id"] != target_component
        ]
        match_rows.append({
            "target_id": f"{target['base_fact_id']}:{target['distractor_id']}",
            "target_base_fact_id": target["base_fact_id"],
            "target_distractor_id": target["distractor_id"],
            "probe_relation_id": target["probe_relation_id"],
            "answer_type": target["answer_type"],
            "eligible_control_ids": sorted(
                f"{control['base_fact_id']}:{control['distractor_id']}" for control in eligible
            ),
            "eligible_control_base_fact_ids": sorted({control["base_fact_id"] for control in eligible}),
            "eligible_control_count": len(eligible),
            "matched": bool(eligible),
        })
    matching_manifest = {
        "schema_version": "factual-perturbation-exact-hf-matching-manifest-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "statistical_unit": "base_fact_id",
        "target_label": "hf_directed_candidate_without_exact_top_score_ties",
        "control_label": "hf_resistant_control_without_exact_top_score_ties",
        "hard_conditions": [
            "same_split",
            "same_exact_hf_identity",
            "same_probe_relation_id",
            "same_answer_type",
            "different_leakage_component_id",
            "no_exact_top_sequence_score_tie_in_required_arms",
        ],
        "selection": "retain_all_eligible_controls",
        "within_fact_variant_weight": "1_over_successful_variant_count",
        "directed_variant_count": len(directed),
        "directed_base_fact_count": len({row["base_fact_id"] for row in directed}),
        "resistant_variant_count": len(resistant),
        "resistant_base_fact_count": len({row["base_fact_id"] for row in resistant}),
        "raw_directed_variant_count": sum(row["hf_directed_candidate"] for row in labels),
        "raw_resistant_variant_count": len(resistant_raw),
        "numerically_tied_directed_variant_ids": tied_directed_ids,
        "numerically_tied_resistant_variant_ids": tied_resistant_ids,
        "matched_directed_variant_count": sum(row["matched"] for row in match_rows),
        "unmatched_directed_variant_count": sum(not row["matched"] for row in match_rows),
        "matches": match_rows,
        "inputs": {
            "exact_hf_model_manifest_sha256": sha256_bytes(model_manifest_path.read_bytes()),
            "g0b_gate_manifest_sha256": sha256_bytes(gate_path.read_bytes()),
            "exact_hf_labels_sha256": sha256_bytes(labels_path.read_bytes()),
        },
    }
    matching_path = behavior_dir / "matching_manifest.json"
    write_json(matching_path, matching_manifest)

    proxy_by_stimulus = {row["static_stimulus_id"]: row for row in proxy_results}
    comparable = [row for row in results if row["stimulus_id"] in proxy_by_stimulus]
    agreement = sum(proxy_by_stimulus[row["stimulus_id"]]["choice"] == row["choice"] for row in comparable)
    artifacts = {
        "exact_hf_model_manifest": file_binding(model_manifest_path, model_manifest["schema_version"]),
        "exact_hf_render_manifest": file_binding(render_manifest_path, render_manifest["schema_version"]),
        "g0b_gate_manifest": file_binding(gate_path, gate["schema_version"]),
        "hf_replay_queue": file_binding(queue_path, queue[0]["schema_version"]),
        "exact_hf_behavior_results": file_binding(results_path, results[0]["schema_version"]),
        "exact_hf_labels": file_binding(labels_path, labels[0]["schema_version"]),
        "matching_manifest": file_binding(matching_path, matching_manifest["schema_version"]),
    }
    run_manifest = {
        "schema_version": "factual-perturbation-exact-hf-behavior-run-manifest-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "g1_exact_hf_behavior_frozen",
        "checkpoint_revision": model_manifest["checkpoint_revision"],
        "scoring_policy": model_manifest["decoding_and_scoring"],
        "result_count": len(results),
        "unique_base_fact_count": len({row["base_fact_id"] for row in results}),
        "label_count": len(labels),
        "proxy_comparison": {
            "purpose": "post_hoc_migration_check_only",
            "comparable_count": len(comparable),
            "choice_agreement_count": agreement,
            "choice_agreement_rate": round(agreement / len(comparable), 6),
            "proxy_labels_used_for_selection": False,
        },
        "artifacts": artifacts,
        "validation_called": False,
        "sealed_called": False,
        "exact_hf_evidence": True,
    }
    run_manifest_path = behavior_dir / "exact_hf_behavior_run_manifest.json"
    write_json(run_manifest_path, run_manifest)

    minimum = int(gate["resolved_protocol"]["minimum_directed_event_count"]["count"])
    directed_facts = len({row["base_fact_id"] for row in directed})
    g1_gate = {
        "schema_version": "factual-perturbation-g1-gate-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "g1_exact_hf_behavior_frozen",
        "gate_passed": directed_facts >= minimum and len(results) == 960,
        "blocking_reasons": [] if directed_facts >= minimum else ["insufficient_hf_directed_base_facts"],
        "minimum_directed_base_fact_count": minimum,
        "observed_directed_base_fact_count": directed_facts,
        "numerically_tied_directed_variant_count": len(tied_directed_ids),
        "numerically_tied_resistant_variant_count": len(tied_resistant_ids),
        "matched_directed_variant_count": matching_manifest["matched_directed_variant_count"],
        "unmatched_directed_variant_count": matching_manifest["unmatched_directed_variant_count"],
        "behavior_run_manifest": file_binding(run_manifest_path, run_manifest["schema_version"]),
        "next_stage": "development_natural_baseline_and_pnt_attribution",
        "natural_baseline_authorized": directed_facts >= minimum,
        "development_pnt_authorized": False,
        "validation_authorized": False,
        "sealed_authorized": False,
    }
    write_json(behavior_dir / "g1_gate_manifest.json", g1_gate)
    print(json.dumps({
        "g1_gate_passed": g1_gate["gate_passed"],
        "directed_variants": len(directed),
        "directed_facts": directed_facts,
        "resistant_variants": len(resistant),
        "matched_directed_variants": matching_manifest["matched_directed_variant_count"],
        "unmatched_directed_variants": matching_manifest["unmatched_directed_variant_count"],
        "proxy_exact_choice_agreement_rate": run_manifest["proxy_comparison"]["choice_agreement_rate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
