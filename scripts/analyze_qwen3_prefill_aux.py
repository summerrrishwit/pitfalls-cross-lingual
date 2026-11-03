#!/usr/bin/env python3
"""Descriptive timing/source-size diagnostics on the fixed 113-input panel."""
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_qwen3_open_repair import input_maps, metrics, resolve
from qwen3_open_repair_common import read_json, read_jsonl, sha, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export-random", action="store_true")
    args = parser.parse_args()
    cfg = read_json(args.config)
    root, out = Path(cfg["remote_root"]), Path(cfg["output_dir"])
    files = sorted(out.glob("timing-*.jsonl"))+sorted(out.glob("learning-*.jsonl"))
    if args.export_random:
        files += sorted(out.glob("generation-random-*.jsonl"))
    resolved, pending = resolve(root, out, files)
    baseline, _, cb, excluded, positives, resistant = input_maps(root)
    timing = read_json(out/"timing_complete.json") if (out/"timing_complete.json").exists() else None
    if not timing:
        print({"status": "timing_execution_pending", "reviews": len(pending)}, flush=True)
        return 0
    panel_records = resolved["timing-every-token-"+timing["spec"]["id"]]
    ids = {r["input_id"] for r in panel_records}
    assert len(ids) == timing["panel_inputs"] == 113
    panel_baseline = {key: value for key, value in baseline.items() if key in ids}
    # Retention calculations here refer only to this fixed panel. They are not
    # substituted for the full-Development G2 metrics.
    results = {}
    for name, records in resolved.items():
        if name.startswith("random-"):
            continue
        if len(records) != len(ids) or {r["input_id"] for r in records} != ids:
            results[name] = {"status": "incomplete_execution", "inputs": len(records)}
            continue
        result = metrics(records, panel_baseline, cb, excluded, positives, resistant)
        results[name] = {"status": "complete" if not result["unresolved_paired_inputs"] else "semantic_missing",
            "inputs": len(records), "directed_generation_gain": result["directed_generation_gain"],
            "directed_margin_gain": result["directed_margin_gain"],
            "retention_on_fixed_panel": result["retention"], "resistant_harmed_facts": result["resistant_harmed"],
            "resistant_unresolved_facts": result["resistant_unresolved"],
            "unresolved_paired_inputs": result["unresolved_paired_inputs"], "missing_verdicts": result["missing_verdicts_total"]}
    learning = read_json(out/"learning_complete.json") if (out/"learning_complete.json").exists() else None
    reference_spec = learning["spec"] if learning else timing["spec"]
    reference = read_jsonl(out/"labeled"/f"generation-{reference_spec['id']}.jsonl")
    subset = [r for r in reference if r["input_id"] in ids]
    original = metrics(subset, panel_baseline, cb, excluded, positives, resistant)
    results["prefill-n72-"+reference_spec["id"]] = {"status": "complete", "inputs": len(subset),
        "directed_generation_gain": original["directed_generation_gain"], "directed_margin_gain": original["directed_margin_gain"],
        "retention_on_fixed_panel": original["retention"], "resistant_harmed_facts": original["resistant_harmed"],
        "unresolved_paired_inputs": original["unresolved_paired_inputs"]}
    summary = {"scope": "fixed_113_input_25_fact_Development_diagnostic_panel_not_G2_or_independent_confirmation",
               "timing_reference": timing["spec"], "learning_reference": reference_spec,
               "learning_execution_complete": learning is not None, "trials": results,
               "pending_unique_review_exports": len(pending), "protocol_sha256": sha(args.config),
               "code_sha256": sha(Path(__file__)), "validation_used": 0, "sealed_used": 0}
    write_json(out/"auxiliary_diagnostic_summary.json", summary)
    print({"trials": len(results), "pending_unique_reviews": len(pending), "learning_complete": learning is not None}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
