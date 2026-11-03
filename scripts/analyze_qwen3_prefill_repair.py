#!/usr/bin/env python3
"""V5 diagnostics and unchanged retention gates, with explicit missing bounds."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_qwen3_open_repair import eligibility, fact_mean, input_maps, metrics, resolve
from qwen3_open_repair_common import read_json, read_jsonl, sha, write_json


def diagnose(config: dict[str, Any]) -> None:
    root, out = Path(config["remote_root"]), Path(config["output_dir"])
    files = sorted(out.glob("generation-oracle-*.jsonl"))
    resolved, queue = resolve(root, out, files)
    baseline = input_maps(root)[0]
    complete = read_json(out/"diagnostic_execution_complete.json") if (out/"diagnostic_execution_complete.json").exists() else None
    by_layer = {}
    for layer in config["diagnostic_layers"]:
        records = resolved.get(f"oracle-l{layer}-neutral_to_targeted", [])
        positive = [r for r in records if r["positive"]]
        reverse = resolved.get(f"oracle-l{layer}-targeted_to_neutral", [])
        lower = [(r["base_fact_id"], float(r["generation_correct"] is True)) for r in positive]
        upper = [(r["base_fact_id"], float(r["generation_correct"] is not False)) for r in positive]
        margin = [(r["base_fact_id"], r["gold_margin"]-baseline[r["input_id"]]["gold_minus_designated_distractor_margin"])
                  for r in positive]
        by_layer[str(layer)] = {"positive_variants": len(positive),
            "positive_facts": len({r["base_fact_id"] for r in positive}),
            "restored_variants_known": sum(r["generation_correct"] is True for r in positive),
            "positive_missing_verdicts": sum(r["generation_correct"] is None for r in positive),
            "fact_recovery_lower_bound": fact_mean(lower), "fact_recovery_upper_bound": fact_mean(upper),
            "positive_margin_gain": fact_mean(margin),
            "reverse_positive_variants_known_wrong": sum(r["positive"] and r["generation_correct"] is False for r in reverse),
            "resistant_forward_known_harmed_variants": sum(r["resistant"] and r["generation_correct"] is False for r in records),
            "resistant_reverse_known_harmed_variants": sum(r["resistant"] and r["generation_correct"] is False for r in reverse)}
    selected = None
    status = "pending_execution"
    if complete:
        candidates = sorted(config["diagnostic_layers"], key=lambda layer: (
            -(by_layer[str(layer)]["fact_recovery_lower_bound"] or 0),
            -(by_layer[str(layer)]["positive_margin_gain"] or 0), layer))
        best = candidates[0]
        result = by_layer[str(best)]
        missing = any(v["positive_missing_verdicts"] for v in by_layer.values())
        if missing:
            # Missing labels can bound recovery, but cannot silently become
            # negatives or change which layer wins the prospective ranking.
            robust = all((result["fact_recovery_lower_bound"] or 0) >
                         (by_layer[str(other)]["fact_recovery_upper_bound"] or 0)
                         for other in candidates[1:])
            status = "local_recovery_supported_layer_robust_to_missing" if robust and result["restored_variants_known"] else "pending_primary_semantic_review"
            selected = best if status.startswith("local_recovery_supported") else None
        elif result["restored_variants_known"]:
            status, selected = "local_recovery_supported", best
        else:
            status = "no_local_generated_recovery_at_predeclared_coordinates"
    decision = {"status": status, "selected_layer": selected, "layers": by_layer,
                "pending_unique_reviews": len(queue), "validation_used": 0, "sealed_used": 0,
                "claim_boundary": "Same-fact oracle state substitution is diagnostic only; it is not a deployable or source-independent repair."}
    write_json(out/"local_recovery_summary.json", decision)
    print(decision, flush=True)


def select(config: dict[str, Any], config_path: Path, random: bool) -> None:
    root, out = Path(config["remote_root"]), Path(config["output_dir"])
    specs = read_json(out/"coarse_selection.json")["candidates"]
    files = [out/f"generation-{s['id']}.jsonl" for s in specs]
    if random:
        files += sorted(out.glob("generation-random-*.jsonl"))
    resolved, queue = resolve(root, out, [p for p in files if p.exists()])
    base, _, cb, excluded, positives, resistant = input_maps(root)
    summaries = {name: metrics(rows, base, cb, excluded, positives, resistant) for name, rows in resolved.items()}
    for name,value in summaries.items():
        value["ineligibility_reasons"] = eligibility(value, config["retention"])
        by_id = {r["input_id"]: r for r in resolved[name]}
        value["directed_unresolved_inputs"] = sum(
            by_id.get(key, {}).get("generation_correct") is None for key,b in base.items()
            if b["language"] == "zh" and b["variant"] == "targeted"
            and (b["base_fact_id"], b["designated_distractor_id"]) in positives)
        if value["directed_unresolved_inputs"]:
            value["ineligibility_reasons"].append("directed_generation_judgments_unresolved")
    write_json(out/"generation_metrics.json", summaries)
    names = {s["id"] for s in specs}
    rejected = {name for name, value in summaries.items() if name in names and (
        value["control_newly_harmed_facts"] or len(value["resistant_harmed"])/value["resistant_facts"] > config["retention"]["resistant_harm_rate_max"])}
    active_keys = {r["review_key"] for name, rows in resolved.items() if name not in rejected for r in rows if r["generation_correct"] is None}
    active = [r for r in queue if r["review_key"] in active_keys]
    write_json(out/"active_semantic_review_queue.json", active)
    eligible = [s for s in specs if s["id"] in summaries and not summaries[s["id"]]["ineligibility_reasons"]]
    eligible.sort(key=lambda s: (-summaries[s["id"]]["directed_generation_gain"],
        abs(s["parts"][0]["scale"]), s["parts"][0]["kind"] != "paired", s["id"]))
    complete = all(s["id"] in summaries and summaries[s["id"]]["inputs"] == 532 for s in specs)
    status = "pending_generation" if not complete else "pending_semantic_review" if active else "generation_eligible_pending_random" if eligible else "G2_failed_no_eligible_generation_candidate"
    missing_specs = [s["id"] for s in specs if s["id"] not in rejected and s["id"] in summaries and summaries[s["id"]]["unresolved_paired_inputs"]]
    if complete and not active and not eligible and missing_specs:
        status = "G2_not_confirmable_missing_primary_judgments"
    decision = {"status": status, "eligible_specs": eligible, "definitively_rejected_by_retention": sorted(rejected),
                "pending_unique_reviews": len(active), "unreviewed_unique_outputs_in_failed_candidates": len(queue)-len(active),
                "not_confirmable_due_to_missing_judgments": missing_specs, "summary": summaries,
                "validation_used": 0, "sealed_used": 0}
    write_json(out/"generation_selection.json", decision)
    if random and complete and not active:
        chosen = None
        comparisons = []
        pending_comparisons = []
        for spec in eligible:
            controls = [summaries.get(f"random-{spec['id']}-r{i}") for i in range(config["random_replicates"])]
            if any(c is None or c["inputs"] != 532 or c["unresolved_paired_inputs"] or c["directed_unresolved_inputs"] for c in controls):
                pending_comparisons.append(spec["id"])
                continue
            random_mean = sum(c["directed_generation_gain"] for c in controls)/len(controls)
            observed = summaries[spec["id"]]["directed_generation_gain"]
            passed = observed > random_mean and observed > 0
            comparisons.append({"id": spec["id"], "observed": observed, "random_mean": random_mean, "passed": passed})
            if passed and chosen is None:
                chosen = spec
        final_status = "oof_selected_pending_final_vector" if chosen else "G2_not_confirmable_random_missing_judgments" if pending_comparisons else "G2_failed_random_comparison"
        write_json(out/"selected_rule.json", {"status": final_status,
            "spec": chosen, "comparisons": comparisons, "protocol_sha256": sha(config_path),
            "random_comparisons_pending_judgments_or_execution": pending_comparisons,
            "generation_metrics_sha256": sha(out/"generation_metrics.json"), "validation_used": 0, "sealed_used": 0})
    print({"status": status, "eligible": [s["id"] for s in eligible], "pending_unique_reviews": len(active),
           "hard_rejected": sorted(rejected)}, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["diagnose", "select"], required=True)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.phase == "diagnose":
        diagnose(config)
    else:
        select(config, args.config, args.random)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
