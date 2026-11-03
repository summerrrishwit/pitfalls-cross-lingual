#!/usr/bin/env python3
"""Freeze the selected rule and run untouched independent partitions once."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import Engine, append, injection_vectors, read_json, read_jsonl, render, sha, suffix, write_json


CODE_FILES=("qwen3_open_repair_common.py","run_qwen3_option_free_remote.py","run_qwen3_open_independent.py",
            "analyze_qwen3_independent.py","analyze_qwen3_open_repair.py","analyze_qwen3_open_trajectories.py",
            "prepare_qwen3_g2a_offline_contract.py","run_qwen3_exact_hf_behavior.py","collect_qwen3_open_vectors.py",
            "run_qwen3_open_repair_search.py")


def frozen_code() -> dict[str,str]:
    return {name:sha(Path(__file__).parent/name) for name in CODE_FILES}


def freeze(config_path: Path) -> None:
    cfg=read_json(config_path);out=Path(cfg["output_dir"])
    selected=read_json(out/"selected_rule.json")
    if selected["status"]!="oof_selected_pending_final_vector":raise ValueError("G2 selection did not pass")
    root=Path(cfg["remote_root"])
    vectors=read_json(out/"final_vector_manifest.json")
    if vectors["source_count"]!=480 or set(vectors["partitions"])!={"final-development-all"}:raise ValueError("final vector source invalid")
    if sha(out/"final_vectors.pt")!=vectors["vectors_sha256"]:raise ValueError("final vector SHA invalid")
    record={"status":"g2_repair_rule_frozen","spec":selected["spec"],"protocol_sha256":sha(config_path),
            "selected_rule_sha256":sha(out/"selected_rule.json"),"final_vectors_sha256":sha(out/"final_vectors.pt"),
            "final_vector_manifest_sha256":sha(out/"final_vector_manifest.json"),"code_sha256":frozen_code(),
            "generation_metrics_sha256":sha(out/"generation_metrics.json"),
            "development_semantic_cache_sha256":sha(root/"results/open_results_labeled.jsonl"),
            "unrelated_control_semantics_sha256":sha(root/"results/unrelated_control_semantic_review.jsonl"),
            "trajectory_summary_sha256":sha(out/"open_trajectory_summary.json"),
            "layer_contrast_reference":{str(p["layer"]):read_json(out/"open_trajectory_summary.json")["directed_minus_resistant_contrast"][p["layer"]] for p in selected["spec"]["parts"]},
            "trigger":"all_ZH_conditions; EN_has_no_injection","validation_used_at_freeze":0,"sealed_used_at_freeze":0}
    path=out/"repair_rule_frozen.json"
    if path.exists() and read_json(path)!=record:raise ValueError("refusing to overwrite a different frozen rule")
    write_json(path,record);print("G2_REPAIR_RULE_FROZEN",record["spec"],flush=True)


def materialize(engine: Engine, split: str) -> list[dict[str,Any]]:
    from prepare_qwen3_g2a_offline_contract import answer_groups, option_free_prompt
    bundle=read_jsonl(Path(engine.base_config["old_run_dir"])/"input/frozen_static_bundle.jsonl")
    selected=[r for r in bundle if r["split_assignment"]==split]
    if len(selected)!=32:raise ValueError("independent cohort must contain all assigned 32 facts")
    result=[]
    for fact in selected:
        for stimulus in fact["static_mcq"]["behavior_inputs"]:
            content=option_free_prompt(fact,stimulus)
            rendered,tokens=render(engine.tokenizer,content)
            groups=answer_groups(fact)
            if len(tokens)+engine.config["max_new_tokens"]>engine.base_config["max_prompt_tokens"]:raise ValueError("independent prompt too long")
            for g in groups:
                for alias in g[f"aliases_{stimulus['language']}"]:
                    suffix(engine.tokenizer,rendered,tokens,alias)
            identity=[engine.config["protocol_version"],split,stimulus["stimulus_id"]]
            result.append({"option_free_id":hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24],
                "base_fact_id":fact["base_fact_id"],"split_assignment":split,"source_id":fact["source_id"],
                "leakage_component_id":fact["leakage_component_id"],"probe_relation_id":fact["probe_relation_id"],
                "language":stimulus["language"],"variant":stimulus["arm"],
                "designated_distractor_id":stimulus.get("distractor_id"),"factual_completion_prompt":stimulus["prompt"],
                "answer_groups":groups,"user_prompt":content,"rendered_prompt":rendered,"prompt_token_ids":tokens,
                "source_stimulus_id":stimulus["stimulus_id"]})
    if len(result)!=320:raise ValueError("independent input coverage must be 320")
    return result


def independent(config_path: Path, split: str) -> None:
    cfg=read_json(config_path);out=Path(cfg["output_dir"]);rule=read_json(out/"repair_rule_frozen.json")
    if rule["status"]!="g2_repair_rule_frozen" or rule["protocol_sha256"]!=sha(config_path) or rule["code_sha256"]!=frozen_code():raise ValueError("frozen rule/code changed before independent evaluation")
    if sha(out/"final_vectors.pt")!=rule["final_vectors_sha256"]:raise ValueError("final vectors changed")
    root=Path(cfg["remote_root"])
    if sha(root/"results/open_results_labeled.jsonl")!=rule["development_semantic_cache_sha256"] or sha(root/"results/unrelated_control_semantic_review.jsonl")!=rule["unrelated_control_semantics_sha256"]:raise ValueError("frozen semantic cache changed")
    if split=="sealed":
        validation=read_json(out/"validation/confirmation.json")
        if validation["status"]!="G3_validation_confirmed" or validation["frozen_rule_sha256"]!=sha(out/"repair_rule_frozen.json"):raise ValueError("Validation did not confirm this frozen rule")
    engine=Engine(config_path,split)
    lock=(out/f".{split}.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    destination=out/split;destination.mkdir(exist_ok=True)
    opened=destination/"opened.json"
    identity={"frozen_rule_sha256":sha(out/"repair_rule_frozen.json"),"split":split,
              "policy":"single_open_with_infrastructure_checkpoint_resume_only"}
    if opened.exists() and read_json(opened)!=identity:raise ValueError("independent split already opened under a different rule")
    write_json(opened,identity)
    inputs=materialize(engine,split)
    spec_path=destination/"inputs.jsonl"
    serialized="".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in inputs)
    if spec_path.exists() and spec_path.read_text()!=serialized:raise ValueError("independent inputs changed during resume")
    spec_path.write_text(serialized)
    bank=engine.torch.load(out/"final_vectors.pt",map_location="cpu",weights_only=True)
    # MCQs retain the frozen three-completion batch and are phenotype-only.
    from run_qwen3_exact_hf_behavior import score_batch
    old=Path(engine.base_config["old_run_dir"])/"output-v3"
    renders_path=old/"exact_hf_rendered_inputs.jsonl"
    if sha(renders_path)!=read_json(old/"exact_hf_render_manifest.json")["rendered_inputs"]["sha256"]:raise ValueError("frozen MCQ renders changed")
    renders=[r for r in read_jsonl(renders_path) if r["split_assignment"]==split]
    if len(renders)!=320:raise ValueError("independent MCQ coverage must be 320")
    path=destination/"mcq_results.jsonl";done={r["render_id"] for r in read_jsonl(path)}
    for r in renders:
        if r["render_id"] in done:continue
        append(path,score_batch(engine.model,engine.torch,[r],engine.manifest["tokenizer"]["pad_token_id"])[0])
    no_op_path=destination/"generation-noop.jsonl";done={r["input_id"] for r in read_jsonl(no_op_path)}
    for row in inputs:
        if row["option_free_id"] in done:continue
        append(no_op_path,engine.run_generation(row,{}))
        done.add(row["option_free_id"])
        if len(done)%20==0:
            write_json(destination/"progress.json",{"stage":"no_op","completed":len(done),"expected":320})
    # Descriptive trajectories are measured at the predeclared coordinates.
    from collect_qwen3_open_vectors import collect_chunks
    collect_chunks(engine,inputs,destination/"open-trajectories",True)
    zh=[r for r in inputs if r["language"]=="zh"]+[r for r in engine.controls if r["language"]=="zh"]
    for replicate in [None,*range(cfg["random_replicates"])]:
        name="repair" if replicate is None else f"random-r{replicate}"
        path=destination/f"generation-{name}.jsonl";done={r["input_id"] for r in read_jsonl(path)}
        vectors=injection_vectors(engine.torch,bank,"final-development-all",rule["spec"],cfg["protocol_version"],replicate)
        for row in zh:
            if row.get("option_free_id",row.get("control_input_id")) in done:continue
            record=engine.run_generation(row,vectors);record["condition"]=name
            append(path,record);done.add(record["input_id"])
            if len(done)%20==0:
                write_json(destination/"progress.json",{"stage":name,"completed":len(done),"expected":len(zh)})
        print(split,name,len(done),flush=True)
    write_json(destination/"execution_complete.json",{"status":"complete_pending_blinded_semantic_review","facts":32,"open_noop_inputs":320,"mcq_inputs":320,
        "ZH_inputs_per_intervention":160,"ZH_controls_per_intervention":13,"random_replicates":5,"frozen_rule_sha256":sha(out/"repair_rule_frozen.json")})


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--phase",choices=["freeze","validation","sealed"],required=True)
    args=parser.parse_args()
    if args.phase=="freeze":freeze(args.config)
    else:independent(args.config,args.phase)
    return 0


if __name__=="__main__":raise SystemExit(main())
