#!/usr/bin/env python3
"""Freeze and independently confirm a G2-passing prefill-only v5b rule."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from qwen3_open_repair_common import append, injection_vectors, read_json, read_jsonl, sha, write_json
from run_qwen3_open_independent import CODE_FILES, materialize
from run_qwen3_prefill_repair import PrefillEngine


EXTRA_CODE = ("run_qwen3_prefill_repair.py", "run_qwen3_prefill_followup.py",
              "analyze_qwen3_prefill_repair.py", "run_qwen3_prefill_independent.py")


def code_identity() -> dict[str, str]:
    return {name: sha(Path(__file__).parent/name) for name in (*CODE_FILES, *EXTRA_CODE)}


def cache_identity(root: Path) -> dict[str, str]:
    paths = [root/"results/open_results_labeled.jsonl", root/"results/unrelated_control_semantic_review.jsonl"]
    paths += [root/version/name for version in ("repair-v3", "repair-v4")
              for name in ("review_export.jsonl", "semantic_review_decisions.jsonl")]
    return {str(path.relative_to(root)): sha(path) for path in paths}


def freeze(config_path: Path) -> None:
    cfg = read_json(config_path)
    out = Path(cfg["output_dir"])
    selected = read_json(out/"selected_rule.json")
    if selected["status"] != "oof_selected_pending_final_vector":
        raise ValueError("G2 selection did not pass")
    if selected["protocol_sha256"] != sha(config_path):
        raise ValueError("G2 selection belongs to another protocol")
    if selected["generation_metrics_sha256"] != sha(out/"generation_metrics.json"):
        raise ValueError("G2 metrics changed after random selection")
    decision = read_json(out/"generation_selection.json")
    if decision["status"] != "generation_eligible_pending_random" or selected["spec"] not in decision["eligible_specs"]:
        raise ValueError("selected rule is no longer generation-eligible")
    root = Path(cfg["remote_root"])
    preflight = read_json(out/"final_source_preflight.json")
    if preflight["status"] != "passed" or preflight["source_facts"] != 96 or preflight["validation_used"] or preflight["sealed_used"]:
        raise ValueError("final source/donor preflight did not pass")
    if preflight["source_specs_sha256"] != sha(root/"inputs/g2a/vector_source_prompt_specs.jsonl"):
        raise ValueError("final source specifications changed")
    final = read_json(out/"final_vector_manifest.json")
    if final["icl_source_count"] != 480 or final["paired_source_facts"] != 96:
        raise ValueError("final full-Development source incomplete")
    if set(final["partitions"]) != {"final-development-all"} or final["partitions"]["final-development-all"]["source_facts"] != 96:
        raise ValueError("invalid final source partition")
    if sha(out/"final_vectors.pt") != final["vectors_sha256"]:
        raise ValueError("final source vector changed")
    if final["validation_facts_used"] or final["sealed_facts_used"]:
        raise ValueError("holdout entered final source")
    if any((out/split/"opened.json").exists() for split in ("validation", "sealed")):
        raise ValueError("cannot freeze after holdout exposure")
    trace_path = root/"repair-v3/open_trajectory_summary.json"
    trace = read_json(trace_path)
    rule = {"status": "g2_repair_rule_frozen", "spec": selected["spec"], "hook_scope": "prefill",
            "protocol_sha256": sha(config_path), "code_sha256": code_identity(),
            "cache_sha256": cache_identity(root), "final_vectors_sha256": sha(out/"final_vectors.pt"),
            "final_vector_manifest_sha256": sha(out/"final_vector_manifest.json"),
            "final_source_preflight_sha256": sha(out/"final_source_preflight.json"),
            "selected_rule_sha256": sha(out/"selected_rule.json"),
            "generation_metrics_sha256": sha(out/"generation_metrics.json"),
            "development_semantic_decisions_sha256": sha(out/"semantic_review_decisions.jsonl"),
            "development_review_export_sha256": sha(out/"review_export.jsonl"),
            "v5_run_identity_sha256": sha(out/"v5_run_identity.json"),
            "local_recovery_summary_sha256": sha(out/"local_recovery_summary.json"),
            "development_trajectory_summary_sha256": sha(trace_path),
            "layer_contrast_reference": {str(part["layer"]): trace["directed_minus_resistant_contrast"][part["layer"]]
                                         for part in selected["spec"]["parts"]},
            "trigger": "all_ZH_conditions; no_EN_injection", "validation_used_at_freeze": 0, "sealed_used_at_freeze": 0}
    path = out/"repair_rule_frozen.json"
    if path.exists() and read_json(path) != rule:
        raise ValueError("refusing changed frozen prefill rule")
    write_json(path, rule)
    print("PREFILL_RULE_FROZEN", rule["spec"], flush=True)


def independent(config_path: Path, split: str) -> None:
    cfg = read_json(config_path)
    root, out = Path(cfg["remote_root"]), Path(cfg["output_dir"])
    rule = read_json(out/"repair_rule_frozen.json")
    if rule["status"] != "g2_repair_rule_frozen" or rule["hook_scope"] != "prefill":
        raise ValueError("prefill G2 rule not frozen")
    if rule["protocol_sha256"] != sha(config_path) or rule["code_sha256"] != code_identity():
        raise ValueError("frozen code or configuration changed")
    if rule["cache_sha256"] != cache_identity(root) or rule["final_vectors_sha256"] != sha(out/"final_vectors.pt"):
        raise ValueError("frozen cache/vector changed")
    if rule["development_semantic_decisions_sha256"] != sha(out/"semantic_review_decisions.jsonl") or rule["generation_metrics_sha256"] != sha(out/"generation_metrics.json"):
        raise ValueError("frozen Development adjudication/selection changed")
    if split == "sealed":
        validation = read_json(out/"validation/confirmation.json")
        if validation["status"] != "G3_validation_confirmed" or validation["frozen_rule_sha256"] != sha(out/"repair_rule_frozen.json"):
            raise ValueError("Validation did not confirm this frozen rule")
    engine = PrefillEngine(config_path, "prefill-"+split)
    lock = (out/f".{split}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = out/split
    directory.mkdir(exist_ok=True)
    opened = directory/"opened.json"
    identity = {"frozen_rule_sha256": sha(out/"repair_rule_frozen.json"), "split": split,
                "policy": "single_open_checkpoint_resume_only"}
    if opened.exists() and read_json(opened) != identity:
        raise ValueError("independent split opened under another rule")
    write_json(opened, identity)
    inputs = materialize(engine, split)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n" for row in inputs)
    input_path = directory/"inputs.jsonl"
    if input_path.exists() and input_path.read_text() != text:
        raise ValueError("independent inputs changed")
    input_path.write_text(text)
    from run_qwen3_exact_hf_behavior import score_batch
    old = Path(engine.base_config["old_run_dir"])/"output-v3"
    renders_path = old/"exact_hf_rendered_inputs.jsonl"
    if sha(renders_path) != read_json(old/"exact_hf_render_manifest.json")["rendered_inputs"]["sha256"]:
        raise ValueError("MCQ renders changed")
    renders = [r for r in read_jsonl(renders_path) if r["split_assignment"] == split]
    if len(renders) != 320:
        raise ValueError("MCQ independent cohort incomplete")
    path = directory/"mcq_results.jsonl"
    done = {r["render_id"] for r in read_jsonl(path)}
    for row in renders:
        if row["render_id"] not in done:
            append(path, score_batch(engine.model, engine.torch, [row], engine.manifest["tokenizer"]["pad_token_id"])[0])
    path = directory/"generation-noop.jsonl"
    done = {r["input_id"] for r in read_jsonl(path)}
    for row in inputs:
        if row["option_free_id"] not in done:
            append(path, engine.run_generation(row, {}))
    from collect_qwen3_open_vectors import collect_chunks
    collect_chunks(engine, inputs, directory/"open-trajectories", True)
    bank = engine.torch.load(out/"final_vectors.pt", map_location="cpu", weights_only=True)
    zh = [r for r in inputs if r["language"] == "zh"]+[r for r in engine.controls if r["language"] == "zh"]
    for replicate in [None, *range(cfg["random_replicates"])]:
        name = "repair" if replicate is None else f"random-r{replicate}"
        path = directory/f"generation-{name}.jsonl"
        done = {r["input_id"] for r in read_jsonl(path)}
        vectors = injection_vectors(engine.torch, bank, "final-development-all", rule["spec"], cfg["protocol_version"], replicate)
        for row in zh:
            key = row.get("option_free_id", row.get("control_input_id"))
            if key not in done:
                record = engine.run_generation(row, vectors)
                record.update(condition=name, hook_scope="prefill")
                append(path, record)
        print(split, name, len(zh), flush=True)
    write_json(directory/"execution_complete.json", {"status": "complete_pending_blinded_review", "facts": 32,
        "open_noop_inputs": 320, "mcq_inputs": 320, "ZH_inputs_per_intervention": 160,
        "ZH_controls_per_intervention": 13, "random_replicates": 5, "frozen_rule_sha256": sha(out/"repair_rule_frozen.json")})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["freeze", "validation", "sealed"], required=True)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.config)
    else:
        independent(args.config, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
