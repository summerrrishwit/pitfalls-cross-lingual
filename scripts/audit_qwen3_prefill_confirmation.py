#!/usr/bin/env python3
"""Read-only checks on completed frozen prefill confirmation artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from qwen3_open_repair_common import read_json, read_jsonl, sha, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "sealed"], required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    root, out = Path(cfg["remote_root"]), Path(cfg["output_dir"])
    rule = read_json(out/"repair_rule_frozen.json")
    assert rule["protocol_sha256"] == sha(args.config)
    for name, want in rule["code_sha256"].items():
        assert sha(root/"scripts"/name) == want, name
    for name, want in rule["cache_sha256"].items():
        assert sha(root/name) == want, name
    assert sha(out/"final_vectors.pt") == rule["final_vectors_sha256"]
    assert sha(out/"semantic_review_decisions.jsonl") == rule["development_semantic_decisions_sha256"]
    assert sha(out/"generation_metrics.json") == rule["generation_metrics_sha256"]
    directory = out/args.split
    execution = read_json(directory/"execution_complete.json")
    confirmation = read_json(directory/"confirmation.json")
    assert execution["frozen_rule_sha256"] == confirmation["frozen_rule_sha256"] == sha(out/"repair_rule_frozen.json")
    specs = {r["option_free_id"]: r for r in read_jsonl(directory/"inputs.jsonl")}
    assert len(specs) == 320 and len({r["base_fact_id"] for r in specs.values()}) == 32
    base_cfg = read_json(root/"run_config.json")
    old = Path(base_cfg["old_run_dir"])
    assigned = {r["base_fact_id"] for r in read_jsonl(old/"input/frozen_static_bundle.jsonl") if r["split_assignment"] == args.split}
    assert assigned == {r["base_fact_id"] for r in specs.values()}
    no_op = read_jsonl(directory/"generation-noop.jsonl")
    assert len(no_op) == 320 and {r["input_id"] for r in no_op} == set(specs)
    zh = {key for key, r in specs.items() if r["language"] == "zh"}
    controls = {r["control_input_id"] for r in read_jsonl(root/"inputs/controls/unrelated_fact_controls.jsonl") if r["language"] == "zh"}
    assert len(zh) == 160 and len(controls) == 13
    decisions = {r["review_key"]: r for r in read_jsonl(directory/"semantic_review_decisions.jsonl")}
    file_bindings = {}
    for name in ["noop", "repair", *[f"random-r{i}" for i in range(5)]]:
        path = directory/f"generation-{name}.jsonl"
        rows = read_jsonl(path)
        labeled = read_jsonl(directory/f"labeled-{name}.jsonl")
        expected = set(specs) if name == "noop" else zh|controls
        assert len(rows) == len(expected) and len(labeled) == len(rows)
        assert {r["input_id"] for r in rows} == expected
        for raw, label in zip(rows, labeled):
            assert raw["input_id"] == label["input_id"] and raw["generation"] == label["generation"]
            if name != "noop":
                assert raw["hook_scope"] == "prefill"
            decision = decisions.get(raw["review_key"])
            if decision:
                assert decision["reviewer_type"] == "codex_proxy" and decision["human_gold"] is False
                assert decision["generation_sha256"] == hashlib.sha256(raw["generation"].encode()).hexdigest()
                assert label["generation_correct"] == decision["semantic_correct"]
            elif raw["generation_correct"] is not None:
                assert label["generation_correct"] == raw["generation_correct"]
        file_bindings[path.name] = sha(path)
        file_bindings[f"labeled-{name}.jsonl"] = sha(directory/f"labeled-{name}.jsonl")
    mcqs = read_jsonl(directory/"mcq_results.jsonl")
    assert len(mcqs) == len({r["render_id"] for r in mcqs}) == 320
    for name in ("confirmation.json", "inputs.jsonl", "mcq_results.jsonl", "semantic_review_decisions.jsonl", "review_export.jsonl", "execution_complete.json"):
        file_bindings[name] = sha(directory/name)
    audit = {"status": "passed", "scope": "completed_saved_artifacts_independent_of_confirmation_analyzer",
             "split": args.split, "frozen_rule_sha256": sha(out/"repair_rule_frozen.json"),
             "frozen_code_files_verified": len(rule["code_sha256"]), "assigned_facts": 32,
             "no_op_inputs": 320, "intervention_inputs_per_arm": 173, "random_replicates": 5,
             "mcq_inputs": 320, "confirmation_status": confirmation["status"],
             "file_sha256": file_bindings, "audit_code_sha256": sha(Path(__file__)),
             "sealed_opened": (out/"sealed/opened.json").exists()}
    if args.split == "validation" and confirmation["status"] != "G3_validation_confirmed":
        assert not audit["sealed_opened"]
    write_json(directory/"confirmation_integrity_audit.json", audit)
    print({k: v for k, v in audit.items() if k != "file_sha256"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
