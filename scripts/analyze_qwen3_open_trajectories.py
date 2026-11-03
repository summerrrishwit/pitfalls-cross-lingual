#!/usr/bin/env python3
"""Descriptive layer contrast; explicitly flag unidentifiable first-token cases."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import read_json, read_jsonl, write_json


def aggregate(entries: dict[str,Any], variants: list[dict[str,Any]], baseline: dict[str,Any]) -> dict[str,Any]:
    lookup={(r["base_fact_id"],r["language"],r["variant"],r["designated_distractor_id"]):r for r in baseline.values()}
    groups={"directed":defaultdict(list),"resistant":defaultdict(list)}
    excluded=[];language_weak=set()
    for label in variants:
        fid,target=label["base_fact_id"],label["distractor_id"]
        triple=[lookup[(fid,"zh",arm,None if arm=="original" else target)] for arm in ("original","neutral","targeted")]
        category="directed" if label["open_induced_directed"] is True else "resistant" if all(r["generation_correct"] is True for r in triple) else None
        if category is None:continue
        neutral,targeted=[entries[r["option_free_id"]]["first_token_lens"] for r in triple[1:]]
        if set(neutral["gold:zh"]["token_ids"]) & set(neutral[f"{target}:zh"]["token_ids"]):
            excluded.append({"base_fact_id":fid,"distractor_id":target,"reason":"gold_and_distractor_share_first_token"});continue
        values=[]
        for layer in range(36):
            nm=neutral["gold:zh"]["best_first_token_logprob"][layer]-neutral[f"{target}:zh"]["best_first_token_logprob"][layer]
            tm=targeted["gold:zh"]["best_first_token_logprob"][layer]-targeted[f"{target}:zh"]["best_first_token_logprob"][layer]
            values.append(tm-nm)
        groups[category][fid].append(values)
        if set(neutral["gold:en"]["token_ids"]) & set(neutral["gold:zh"]["token_ids"]):language_weak.add(fid)
    means={}
    for group,facts in groups.items():
        vectors=[[sum(v[layer] for v in values)/len(values) for layer in range(36)] for values in facts.values()]
        means[group]={"facts":len(facts),"variants":sum(len(v) for v in facts.values()),"targeted_minus_neutral":[sum(v[layer] for v in vectors)/len(vectors) for layer in range(36)] if vectors else [None]*36}
    contrast=[a-b if a is not None and b is not None else None for a,b in zip(means["directed"]["targeted_minus_neutral"],means["resistant"]["targeted_minus_neutral"])]
    return {"status":"complete","groups":means,"directed_minus_resistant_contrast":contrast,
            "shared_first_token_variants_excluded":excluded,"EN_ZH_shared_first_token_facts":sorted(language_weak),
            "claim_boundary":"Descriptive first-answer-token lens only. Full-answer probabilities and generated answers are separate; this does not establish a recall/translation causal stage."}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();cfg=read_json(args.config);root=Path(cfg["remote_root"]);out=Path(cfg["output_dir"])
    entries={}
    for path in sorted((out/"open-trajectories").glob("chunk-*.json")):
        for r in read_json(path):entries[r["id"]]=r
    if len(entries)!=960:raise ValueError("incomplete open trajectory collection")
    baseline={r["option_free_id"]:r for r in read_jsonl(root/"results/open_results_labeled.jsonl")}
    labels=read_jsonl(root/"results/open_variant_labels.jsonl")
    report=aggregate(entries,labels,baseline)
    write_json(out/"open_trajectory_summary.json",report)
    print({"groups":{k:{x:v[x] for x in ("facts","variants")} for k,v in report["groups"].items()},"excluded_first_token_overlap":len(report["shared_first_token_variants_excluded"]),"contrast_at_candidate_layers":{str(l):report["directed_minus_resistant_contrast"][l] for l in cfg["candidate_layers"]}},flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
