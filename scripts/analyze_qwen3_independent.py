#!/usr/bin/env python3
"""Apply frozen independent confirmation gates after blinded semantic review."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from analyze_qwen3_open_repair import fact_mean, metrics
from analyze_qwen3_open_trajectories import aggregate
from qwen3_open_repair_common import read_json, read_jsonl, sha, write_json


def resolve(directory: Path, root: Path) -> tuple[dict[str,Any],list[Any]]:
    inputs={r["option_free_id"]:r for r in read_jsonl(directory/"inputs.jsonl")}
    for row in read_jsonl(root/"inputs/controls/unrelated_fact_controls.jsonl"):
        row["answer_groups"]=[{"option_id":"gold",f"aliases_{row['language']}":row["answer_aliases"]}]
        inputs[row["control_input_id"]]=row
    files=sorted(directory.glob("generation-*.jsonl"));raw={p.stem.removeprefix("generation-"):read_jsonl(p) for p in files}
    known={r["review_key"]:r for records in raw.values() for r in records if r["generation_correct"] is not None}
    decisions={r["review_key"]:r for r in read_jsonl(directory/"semantic_review_decisions.jsonl")}
    export=read_jsonl(directory/"review_export.jsonl");exported={r["review_key"] for r in export};pending={}
    for name,records in raw.items():
        for r in records:
            d=decisions.get(r["review_key"])
            if d:
                if d.get("human_gold") is not False or d.get("reviewer_type")!="codex_proxy" or d.get("generation_sha256")!=hashlib.sha256(r["generation"].encode()).hexdigest():raise ValueError("invalid independent semantic review binding")
                r.update(generation_correct=d["semantic_correct"],generated_answer_group=d.get("answer_group"),correctness_basis="explicit_blinded_codex_proxy")
            elif r["generation_correct"] is None and r["review_key"] in known:
                prior=known[r["review_key"]];r.update(generation_correct=prior["generation_correct"],generated_answer_group=prior["generated_answer_group"],correctness_basis="identical_generation_known_verdict")
            elif r["generation_correct"] is None:
                source=inputs[r["input_id"]]
                entry={"review_key":r["review_key"],"base_fact_id":source.get("base_fact_id",source.get("control_id")),
                       "language":r["language"],"question":source.get("factual_completion_prompt",source.get("prompt")),
                       "generation":r["generation"],"answer_groups":source["answer_groups"]}
                pending[r["review_key"]]=entry
                if r["review_key"] not in exported:export.append({"index":len(export),**entry});exported.add(r["review_key"])
        (directory/f"labeled-{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
    (directory/"review_export.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in export))
    write_json(directory/"semantic_review_queue.json",list(pending.values()))
    return raw,list(pending.values())


def conjunction(values: list[bool|None]) -> bool|None:
    return False if False in values else None if None in values else True


def labels_and_maps(directory: Path, no_op: list[dict[str,Any]]) -> tuple[Any,...]:
    specs={r["option_free_id"]:r for r in read_jsonl(directory/"inputs.jsonl")}
    baseline={r["input_id"]:{**specs[r["input_id"]],**r,"gold_minus_designated_distractor_margin":r.get("gold_margin")} for r in no_op}
    lookup={(r["base_fact_id"],r["language"],r["variant"],r["designated_distractor_id"]):r for r in baseline.values()}
    pairs=sorted({(r["base_fact_id"],r["designated_distractor_id"]) for r in baseline.values() if r["variant"]=="targeted"})
    variants=[];positives=set();resistant={}
    for fid,target in pairs:
        zh=[lookup[(fid,"zh",a,None if a=="original" else target)] for a in ("original","neutral","targeted")]
        directed=conjunction([zh[0]["generation_correct"],zh[1]["generation_correct"],None if zh[2]["generation_correct"] is None else not zh[2]["generation_correct"],None if zh[2]["generation_correct"] is None and zh[2]["generated_answer_group"] is None else zh[2]["generated_answer_group"]==target])
        variants.append({"base_fact_id":fid,"distractor_id":target,"open_induced_directed":directed})
        if directed is True:positives.add((fid,target))
        if all(r["generation_correct"] is True for r in zh):resistant.setdefault(fid,set()).update(r["input_id"] for r in zh)
    excluded={r["base_fact_id"] for r in baseline.values() if r["variant"]=="original" and r["generation_correct"] is None}
    return baseline,variants,positives,resistant,excluded


def bootstrap(records: list[dict[str,Any]], baseline: dict[str,Any]) -> dict[str,Any]:
    by_id={r["input_id"]:r for r in records if r["base_fact_id"] is not None};by_fact=defaultdict(list);component={}
    for key,b in baseline.items():
        if b["language"]!="zh" or b["variant"]!="targeted" or b["generation_correct"] is None:continue
        r=by_id.get(key)
        if r is None or r["generation_correct"] is None:continue
        by_fact[b["base_fact_id"]].append(float(r["generation_correct"])-float(b["generation_correct"]))
        component[b["base_fact_id"]]=b["leakage_component_id"]
    blocks=defaultdict(list)
    for fid,values in by_fact.items():blocks[component[fid]].append(sum(values)/len(values))
    keys=sorted(blocks);rng=random.Random(20261016);samples=[]
    if not keys:return {"facts":0,"interval":None}
    for _ in range(2000):
        values=[v for _ in keys for v in blocks[rng.choice(keys)]];samples.append(sum(values)/len(values))
    samples.sort()
    return {"facts":len(by_fact),"components":len(keys),"resamples":2000,"paired_accuracy_gain_95pct_interval":[samples[49],samples[1949]],"pilot_only":True}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--split",choices=["validation","sealed"],required=True)
    args=parser.parse_args();cfg=read_json(args.config);root=Path(cfg["remote_root"]);out=Path(cfg["output_dir"]);directory=out/args.split
    rule=read_json(out/"repair_rule_frozen.json")
    raw,queue=resolve(directory,root)
    if "noop" not in raw or len(raw["noop"])!=320:raise ValueError("independent no-op incomplete")
    baseline,variants,positives,resistant,excluded=labels_and_maps(directory,raw["noop"])
    cb={r["control_input_id"]:r for r in read_jsonl(root/"results/unrelated_control_semantic_review.jsonl")}
    summaries={name:metrics(values,baseline,cb,excluded,positives,resistant) for name,values in raw.items() if name!="noop"}
    entries={r["id"]:r for p in sorted((directory/"open-trajectories").glob("chunk-*.json")) for r in read_json(p)}
    trace=aggregate(entries,variants,baseline);write_json(directory/"trajectory_summary.json",trace)
    reasons=[];observed=summaries.get("repair")
    if queue:reasons.append("pending_blinded_semantic_review")
    if len({r["base_fact_id"] for r in baseline.values()})!=32:reasons.append("not_all_assigned_facts_evaluated")
    if len({fid for fid,target in positives})<cfg["independent_gate"]["min_induced_facts_for_directional_pilot_confirmation"]:reasons.append("too_few_induced_facts_for_confirmation")
    if observed is None or observed["cohort_inputs"]!=160 or observed["inputs"]!=173:reasons.append("repair_execution_incomplete")
    if observed:
        if observed["unresolved_paired_inputs"]:reasons.append("unresolved_primary_pairs")
        for arm in ("original","neutral"):
            gain=observed["retention"][arm]["paired_accuracy_gain"]
            if gain is None or gain < -cfg["retention"][f"{arm}_accuracy_drop_max"]:reasons.append(f"{arm}_retention_failed")
        if observed["resistant_unresolved"] or observed["resistant_facts"]==0:reasons.append("resistant_retention_not_assessable")
        elif len(observed["resistant_harmed"])/observed["resistant_facts"]>cfg["retention"]["resistant_harm_rate_max"]:reasons.append("resistant_harm_failed")
        if observed["control_newly_harmed_facts"] or observed["control_unresolved_on_baseline_correct"]:reasons.append("unrelated_fact_retention_failed_or_unresolved")
        if (observed["retention"]["targeted"]["paired_accuracy_gain"] or 0)<=0:reasons.append("all_fact_targeted_generation_not_improved")
        if (observed["directed_generation_gain"] or 0)<=0:reasons.append("directed_generation_not_improved")
        cohort={r["input_id"]:r for r in raw["repair"] if r["base_fact_id"] is not None}
        margins=[(b["base_fact_id"],cohort[k]["gold_margin"]-b["gold_minus_designated_distractor_margin"]) for k,b in baseline.items() if b["language"]=="zh" and b["variant"]=="targeted" and k in cohort]
        all_margin=fact_mean(margins)
        if all_margin is None or all_margin<=0:reasons.append("all_fact_targeted_margin_not_improved")
        randoms=[summaries.get(f"random-r{i}") for i in range(cfg["random_replicates"])]
        if any(r is None or r["cohort_inputs"]!=160 or r["unresolved_paired_inputs"] for r in randoms):reasons.append("random_control_incomplete_or_unresolved");random_mean=None
        else:
            random_mean=sum((r["directed_generation_gain"] or 0) for r in randoms)/len(randoms)
            if (observed["directed_generation_gain"] or 0)<=random_mean:reasons.append("repair_not_above_random_mean")
        for layer,reference in rule["layer_contrast_reference"].items():
            actual=trace["directed_minus_resistant_contrast"][int(layer)]
            if reference is None or actual is None or reference*actual<=0:reasons.append(f"attribution_contrast_not_confirmed_layer_{layer}")
    else:all_margin=None;random_mean=None
    status=("G3_validation_confirmed" if args.split=="validation" else "G4_validated_repair") if not reasons else ("G3_validation_failed_or_insufficient" if args.split=="validation" else "G4_failed_or_insufficient")
    record={"status":status,"failure_or_insufficiency_reasons":reasons,"frozen_rule_sha256":sha(out/"repair_rule_frozen.json"),
            "assigned_facts":32,"induced_facts":len({fid for fid,target in positives}),"semantic_pending_unique":len(queue),
            "excluded_unresolved_original_facts":sorted(excluded),"metrics":summaries,"random_mean_directed_gain":random_mean,
            "all_fact_targeted_margin_gain":all_margin,"uncertainty":bootstrap(raw.get("repair",[]),baseline),
            "english_policy":"all_EN_conditions_measured_noop; repair_trigger_disabled_for_EN",
            "human_gold":False,"pilot_only":True,"rule_adjusted_from_independent_outputs":False}
    write_json(directory/"confirmation.json",record)
    print(json.dumps({k:v for k,v in record.items() if k!="metrics"},ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
