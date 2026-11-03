#!/usr/bin/env python3
"""Fact-level retention-first OOF rule selection; no model inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import read_json, read_jsonl, sha, write_json


def fact_mean(values: list[tuple[str,float]]) -> float | None:
    grouped=defaultdict(list)
    for key,value in values:grouped[key].append(value)
    return sum(sum(v)/len(v) for v in grouped.values())/len(grouped) if grouped else None


def input_maps(root: Path) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],set[str],set[tuple[str,str]],dict[str,set[str]]]:
    baseline={r["option_free_id"]:r for r in read_jsonl(root/"results/open_results_labeled.jsonl")}
    controls={r["control_input_id"]:r for r in read_jsonl(root/"inputs/controls/unrelated_fact_controls.jsonl")}
    cb={r["control_input_id"]:r for r in read_jsonl(root/"results/unrelated_control_semantic_review.jsonl")}
    excluded={r["base_fact_id"] for r in baseline.values() if r["variant"]=="original" and r["generation_correct"] is None}
    labels=read_jsonl(root/"results/open_variant_labels.jsonl")
    positives={(r["base_fact_id"],r["distractor_id"]) for r in labels if r["open_induced_directed"] is True}
    lookup={(r["base_fact_id"],r["language"],r["variant"],r["designated_distractor_id"]):r for r in baseline.values()}
    resistant=defaultdict(set)
    for r in labels:
        fid,target=r["base_fact_id"],r["distractor_id"]
        triple=[lookup[(fid,"zh",arm,None if arm=="original" else target)] for arm in ("original","neutral","targeted")]
        if all(t["generation_correct"] is True for t in triple):
            resistant[fid].update(t["option_free_id"] for t in triple)
    return baseline,controls,cb,excluded,positives,dict(resistant)


def resolve(root: Path, out: Path, files: list[Path]) -> tuple[dict[str,list[dict[str,Any]]],list[dict[str,Any]]]:
    baseline,controls,*_=input_maps(root)
    decisions={r["review_key"]:r for r in read_jsonl(out/"semantic_review_decisions.jsonl")}
    export=read_jsonl(out/"review_export.jsonl")
    exported={r["review_key"] for r in export}
    unresolved={};resolved={}
    for path in files:
        results=read_jsonl(path)
        for r in results:
            if r["generation_correct"] is not None:continue
            decision=decisions.get(r["review_key"])
            if decision:
                if decision.get("human_gold") is not False or decision.get("reviewer_type")!="codex_proxy" or decision.get("generation_sha256")!=hashlib.sha256(r["generation"].encode()).hexdigest():
                    raise ValueError("semantic decision provenance/text binding invalid")
                r["generation_correct"]=decision["semantic_correct"]
                r["generated_answer_group"]=decision.get("answer_group")
                r["correctness_basis"]="explicit_codex_proxy_review"
                continue
            source=baseline.get(r["input_id"],controls.get(r["input_id"]))
            if source is None:raise ValueError("unknown input ID")
            entry={"review_key":r["review_key"],"base_fact_id":source.get("base_fact_id",source.get("control_id")),
                   "language":r["language"],"question":source.get("factual_completion_prompt",source.get("prompt")),
                   "generation":r["generation"],"answer_groups":source.get("answer_groups",[{"option_id":"gold","aliases_en":source["answer_aliases"] if r["language"]=="en" else [],"aliases_zh":source["answer_aliases"] if r["language"]=="zh" else []}]) if "answer_groups" not in source else source["answer_groups"]}
            unresolved[r["review_key"]]=entry
            if r["review_key"] not in exported:
                export.append({"index":len(export),**entry});exported.add(r["review_key"])
        resolved[path.stem.removeprefix("generation-")]=results
        destination=out/"labeled"/path.name;destination.parent.mkdir(exist_ok=True)
        destination.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results))
    (out/"review_export.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in export))
    write_json(out/"semantic_review_queue.json",list(unresolved.values()))
    return resolved,list(unresolved.values())


def metrics(records: list[dict[str,Any]], baseline: dict[str,Any], cb: dict[str,Any],
            excluded: set[str], positives: set[tuple[str,str]], resistant: dict[str,set[str]]) -> dict[str,Any]:
    cohort={r["input_id"]:r for r in records if r["base_fact_id"] is not None}
    retention={};unresolved=[]
    for arm in ("original","neutral","targeted"):
        before=[];after=[];changes=[]
        for key,b in baseline.items():
            if b["language"]!="zh" or b["variant"]!=arm or b["base_fact_id"] in excluded or b["generation_correct"] is None:continue
            r=cohort.get(key)
            if r is None or r["generation_correct"] is None:
                unresolved.append(key);continue
            fid=b["base_fact_id"]
            before.append((fid,float(b["generation_correct"])));after.append((fid,float(r["generation_correct"])))
            changes.append((fid,float(r["generation_correct"])-float(b["generation_correct"])))
        retention[arm]={"before":fact_mean(before),"after":fact_mean(after),"paired_accuracy_gain":fact_mean(changes),"paired_facts":len({x[0] for x in changes})}
    directed=[];margins=[]
    for key,b in baseline.items():
        if b["language"]!="zh" or b["variant"]!="targeted" or (b["base_fact_id"],b["designated_distractor_id"]) not in positives:continue
        r=cohort.get(key)
        if r and r["generation_correct"] is not None:
            directed.append((b["base_fact_id"],float(r["generation_correct"])-float(b["generation_correct"])))
        if r and r.get("gold_margin") is not None:
            margins.append((b["base_fact_id"],r["gold_margin"]-b["gold_minus_designated_distractor_margin"]))
    harmed=[];resistant_unknown=[]
    for fid,ids in resistant.items():
        if fid in excluded:continue
        if any(cohort.get(key,{}).get("generation_correct") is False for key in ids):harmed.append(fid)
        elif any(cohort.get(key,{}).get("generation_correct") is None for key in ids):resistant_unknown.append(fid)
    control_harmed=set();control_unknown=set()
    for r in records:
        if r["base_fact_id"] is not None:continue
        if cb[r["input_id"]]["semantic_correct"] is True:
            if r["generation_correct"] is False:control_harmed.add(r["control_id"])
            elif r["generation_correct"] is None:control_unknown.add(r["control_id"])
    return {"inputs":len(records),"cohort_inputs":len(cohort),"retention":retention,
            "directed_generation_gain":fact_mean(directed),"directed_facts":len({r[0] for r in directed}),
            "directed_margin_gain":fact_mean(margins),"unresolved_paired_inputs":len(set(unresolved)),
            "resistant_facts":len([fid for fid in resistant if fid not in excluded]),"resistant_harmed":harmed,
            "resistant_unresolved":resistant_unknown,"control_newly_harmed_facts":sorted(control_harmed),
            "control_unresolved_on_baseline_correct":sorted(control_unknown),
            "missing_verdicts_total":sum(r["generation_correct"] is None for r in records)}


def eligibility(result: dict[str,Any], constraints: dict[str,Any]) -> list[str]:
    reasons=[]
    if result["cohort_inputs"]!=480 or result["inputs"]!=532:reasons.append("incomplete_development_or_controls")
    if result["unresolved_paired_inputs"]:reasons.append("unresolved_primary_pairs")
    for arm in ("original","neutral"):
        gain=result["retention"][arm]["paired_accuracy_gain"]
        if gain is None or gain < -constraints[f"{arm}_accuracy_drop_max"]:reasons.append(f"{arm}_retention_failed")
    if result["resistant_unresolved"]:reasons.append("resistant_retention_unresolved")
    denominator=result["resistant_facts"]
    if denominator==0 or len(result["resistant_harmed"])/denominator>constraints["resistant_harm_rate_max"]:reasons.append("resistant_harm_failed")
    if result["control_newly_harmed_facts"]:reasons.append("unrelated_control_harm")
    if result["control_unresolved_on_baseline_correct"]:reasons.append("unrelated_control_retention_unresolved")
    if (result["directed_generation_gain"] or 0)<=0:reasons.append("no_directed_generation_gain")
    if (result["retention"]["targeted"]["paired_accuracy_gain"] or 0)<=0:reasons.append("no_all_fact_targeted_generation_gain")
    if (result["directed_margin_gain"] or 0)<=0:reasons.append("no_positive_margin_gain")
    return reasons


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--random",action="store_true")
    args=parser.parse_args();config=read_json(args.config)
    root,out=Path(config["remote_root"]),Path(config["output_dir"])
    specs=read_json(out/"coarse_selection.json")["candidates"]
    files=[out/f"generation-{r['id']}.jsonl" for r in specs]
    if args.random:files+=sorted(out.glob("generation-random-*.jsonl"))
    files=[p for p in files if p.exists()]
    resolved,queue=resolve(root,out,files)
    baseline,controls,cb,excluded,positives,resistant=input_maps(root)
    summaries={name:metrics(records,baseline,cb,excluded,positives,resistant) for name,records in resolved.items()}
    for name,value in summaries.items():
        value["ineligibility_reasons"]=eligibility(value,config["retention"])
    write_json(out/"generation_metrics.json",summaries)
    candidate_names={s["id"] for s in specs}
    hard_rejected={name for name,value in summaries.items() if name in candidate_names and (value["control_newly_harmed_facts"] or
                   (value["resistant_facts"] and len(value["resistant_harmed"])/value["resistant_facts"]>config["retention"]["resistant_harm_rate_max"]))}
    active_keys={r["review_key"] for name,records in resolved.items() if name not in hard_rejected for r in records if r["generation_correct"] is None}
    active_queue=[r for r in queue if r["review_key"] in active_keys]
    write_json(out/"active_semantic_review_queue.json",active_queue)
    eligible=[r for r in specs if r["id"] in summaries and not summaries[r["id"]]["ineligibility_reasons"]]
    eligible.sort(key=lambda r:(-summaries[r["id"]]["directed_generation_gain"],sum(abs(p["scale"]) for p in r["parts"]),min(p["layer"] for p in r["parts"]),
                                2 if len(r["parts"])>1 else 0 if r["parts"][0]["kind"]=="task" else 1,r["id"]))
    complete=all(r["id"] in summaries and summaries[r["id"]]["cohort_inputs"]==480 and summaries[r["id"]]["inputs"]==532 for r in specs)
    decision={"status":"pending_generation" if not complete else "pending_semantic_review" if active_queue else "generation_eligible_pending_random" if eligible else "G2_failed_no_eligible_generation_candidate",
              "eligible_specs":eligible,"pending_unique_reviews":len(active_queue),
              "definitively_rejected_by_retention":sorted(hard_rejected),"unreviewed_outputs_retained_in_failed_candidates":len(queue)-len(active_queue),
              "pruning_policy":"Known zero-harm/resistant-harm failures cannot be rescued by other missing judgments; no failed outcome is imputed negative.",
              "original_deferred_facts_excluded_from_selection":sorted(excluded),
              "summary":{k:{"directed_gain":v["directed_generation_gain"],"all_targeted_gain":v["retention"]["targeted"]["paired_accuracy_gain"],"reasons":v["ineligibility_reasons"]} for k,v in summaries.items()},
              "validation_used":0,"sealed_used":0}
    if complete and not active_queue and eligible:decision["status"]="generation_eligible_pending_random"
    decision["not_confirmable_due_to_missing_judgments"] = sorted(
        name for name,value in summaries.items() if name in candidate_names and name not in hard_rejected
        and any(reason in value["ineligibility_reasons"] for reason in (
            "unresolved_primary_pairs", "resistant_retention_unresolved", "unrelated_control_retention_unresolved")))
    if complete and not active_queue and not eligible and decision["not_confirmable_due_to_missing_judgments"]:
        decision["status"]="G2_not_confirmable_missing_primary_judgments"
    write_json(out/"generation_selection.json",decision)
    if args.random and complete and not active_queue:
        selected=None;comparisons=[]
        for spec in eligible:
            randoms=[summaries.get(f"random-{spec['id']}-r{i}") for i in range(config["random_replicates"])]
            if any(r is None or r["cohort_inputs"]!=480 or r["unresolved_paired_inputs"] for r in randoms):continue
            mean=sum(r["directed_generation_gain"] for r in randoms)/len(randoms)
            observed=summaries[spec["id"]]["directed_generation_gain"]
            passed=observed>mean and observed>0
            comparisons.append({"config_id":spec["id"],"observed_directed_gain":observed,"random_mean_directed_gain":mean,"passed":passed})
            if passed and selected is None:selected=spec
        final={"status":"oof_selected_pending_final_vector" if selected else "G2_failed_random_comparison","spec":selected,"comparisons":comparisons,
               "protocol_sha256":sha(args.config),"generation_metrics_sha256":sha(out/"generation_metrics.json"),
               "validation_used":0,"sealed_used":0}
        write_json(out/"selected_rule.json",final)
    print(json.dumps(decision,ensure_ascii=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
