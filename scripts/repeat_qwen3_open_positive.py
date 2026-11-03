#!/usr/bin/env python3
"""Confirm exact-runtime reproducibility of all positive open-answer triples."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket

from run_qwen3_option_free_remote import append, generate, prepare, read_json, read_jsonl, sha, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if socket.gethostname() != config["remote_hostname"]:
        raise ValueError("remote execution required")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForCausalLM
    out = Path(config["output_dir"]) / "positive-repeat"
    out.mkdir(exist_ok=True)
    tokenizer, inputs, manifest = prepare(config, out)
    parent = Path(config["output_dir"])
    labels = read_jsonl(parent / "open_variant_labels.jsonl")
    outputs = {r["option_free_id"]: r for r in read_jsonl(parent / "open_results_labeled.jsonl")}
    selected = set()
    for row in labels:
        if row["open_induced_directed"] is not True:
            continue
        languages = ("en", "zh") if row["open_language_specific"] is True else ("zh",)
        for language in languages:
            for arm in ("original", "neutral", "targeted"):
                selected.add((row["base_fact_id"],language,arm,None if arm=="original" else row["distractor_id"]))
    chosen = [r for r in inputs if (r["base_fact_id"],r["language"],r["variant"],r["designated_distractor_id"]) in selected]
    if len(chosen) != len(selected):
        raise ValueError("repeat input selection incomplete")
    identity = {"config_sha256":sha(args.config),"script_sha256":sha(Path(__file__)),
                "label_sha256":sha(parent/"open_variant_labels.jsonl"),
                "output_sha256":sha(parent/"open_results_labeled.jsonl")}
    if (out/"identity.json").exists() and read_json(out/"identity.json") != identity:
        raise ValueError("repeat identity changed")
    write_json(out/"identity.json",identity)
    runtime = manifest["decoding_and_scoring"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = runtime["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    model = AutoModelForCausalLM.from_pretrained(config["model_path"],dtype=torch.bfloat16,
            local_files_only=True,attn_implementation=runtime["attention_implementation"]).eval().to("cuda")
    path=out/"generation_reproduction.jsonl"
    done={r["option_free_id"] for r in read_jsonl(path)}
    for row in chosen:
        if row["option_free_id"] in done:
            continue
        result=generate(model,torch,tokenizer,row["prompt_token_ids"],config["max_new_tokens"])
        previous=outputs[row["option_free_id"]]
        append(path,{"option_free_id":row["option_free_id"],"base_fact_id":row["base_fact_id"],
                     "language":row["language"],"variant":row["variant"],
                     "previous_generation":previous["generation"],**result,
                     "exact_generation_match":result["generation"]==previous["generation"]})
    repeated=read_jsonl(path)
    summary={"status":"passed" if len(repeated)==len(chosen) and all(r["exact_generation_match"] for r in repeated) else "failed",
             "inputs":len(repeated),"expected":len(chosen),"facts":len({r["base_fact_id"] for r in repeated}),
             "exact_matches":sum(r["exact_generation_match"] for r in repeated),
             "same_runtime_deterministic_repetition_only":True,
             "validation_facts_used":0,"sealed_facts_used":0}
    write_json(out/"reproduction_summary.json",summary)
    print(summary,flush=True)
    return 0 if summary["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
