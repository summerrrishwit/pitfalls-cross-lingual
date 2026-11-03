#!/usr/bin/env python3
"""Predeclared bounded OOF score screen and full Development generation trials."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import Engine, append, choice_specs, injection_vectors, read_json, read_jsonl, row_key, sha, write_json


def load_bank(engine: Engine) -> dict[str, Any]:
    path=engine.out/"oof_vectors.pt"
    manifest=read_json(engine.out/"oof_vector_manifest.json")
    if sha(path)!=manifest["vectors_sha256"]:
        raise ValueError("OOF vectors changed")
    if read_json(engine.out/"no_op_injection_audit.json")["status"]!="passed":
        raise ValueError("no-op coordinate audit not passed")
    return engine.torch.load(path,map_location="cpu",weights_only=True)


def screen(engine: Engine) -> None:
    bank=load_bank(engine)
    rows=[r for r in engine.inputs if r["language"]=="zh"]
    positives={(r["base_fact_id"],r["distractor_id"]) for r in read_jsonl(engine.root/"results/open_variant_labels.jsonl") if r["open_induced_directed"] is True}
    all_scores=[]
    for index,spec in enumerate(choice_specs(engine.config)):
        path=engine.out/f"screen-{spec['id']}.jsonl"
        saved=read_jsonl(path);done={r["input_id"] for r in saved}
        for row in rows:
            if row_key(row) in done:continue
            fold=row["development_fold_id"]
            vectors=injection_vectors(engine.torch,bank,fold,spec,engine.config["protocol_version"])
            score=engine.sequence_scores(row,vectors,all_groups=row["variant"]=="original")
            result={"input_id":row_key(row),"base_fact_id":row["base_fact_id"],"variant":row["variant"],"distractor_id":row["designated_distractor_id"],"evaluation_partition":fold,**score}
            append(path,result);saved.append(result)
            if len(saved)%40==0:
                write_json(engine.out/"search_progress.json",{"stage":"screen","config_index":index+1,"configs":12,"config_id":spec["id"],"rows":len(saved),"expected":480})
        by_fact=defaultdict(list)
        for r in saved:
            if r["variant"]=="targeted" and (r["base_fact_id"],r["distractor_id"]) in positives:
                baseline=engine.baseline[r["input_id"]]["gold_minus_designated_distractor_margin"]
                by_fact[r["base_fact_id"]].append(r["gold_margin"]-baseline)
        mean=sum(sum(v)/len(v) for v in by_fact.values())/len(by_fact)
        all_scores.append({"spec":spec,"positive_fact_weighted_margin_gain":mean,"positive_facts":len(by_fact),"inputs":len(saved),"results_sha256":sha(path)})
        print("screen_complete",spec["id"],mean,flush=True)
        write_json(engine.out/"coarse_screen_summary.json",all_scores)
    chosen=[]
    for family in ("task","difference"):
        candidates=[r for r in all_scores if r["spec"]["parts"][0]["kind"]==family]
        candidates.sort(key=lambda r:(-r["positive_fact_weighted_margin_gain"],abs(r["spec"]["parts"][0]["scale"]),r["spec"]["parts"][0]["layer"]))
        chosen.append(candidates[0]["spec"])
    chosen.append({"id":"combined","parts":[*chosen[0]["parts"],*chosen[1]["parts"]]})
    write_json(engine.out/"coarse_selection.json",{"status":"fixed_before_candidate_generation","candidates":chosen,"screen_config_sha256":read_json(engine.out/"screen-identity.json")["config_sha256"]})


def generation(engine: Engine, random: bool = False) -> None:
    bank=load_bank(engine)
    if random:
        eligible=read_json(engine.out/"generation_selection.json")
        if eligible["status"]!="generation_eligible_pending_random":
            raise ValueError("no generation-eligible candidate for random controls")
        specs=eligible["eligible_specs"]
    else:specs=read_json(engine.out/"coarse_selection.json")["candidates"]
    rows=[(r["development_fold_id"],r) for r in engine.inputs if r["language"]=="zh"]
    for fold in sorted(bank):
        rows.extend((fold,r) for r in engine.controls if r["language"]=="zh")
    for spec in specs:
        replicates=range(engine.config["random_replicates"]) if random else [None]
        for replicate in replicates:
            label=f"random-{spec['id']}-r{replicate}" if random else spec["id"]
            path=engine.out/f"generation-{label}.jsonl"
            previous=read_jsonl(path);done={(r["evaluation_partition"],r["input_id"]) for r in previous}
            if len(done)!=len(previous):raise ValueError("duplicate generation checkpoint")
            for fold,row in rows:
                if (fold,row_key(row)) in done:continue
                vectors=injection_vectors(engine.torch,bank,fold,spec,engine.config["protocol_version"],replicate)
                result=engine.run_generation(row,vectors)
                result.update(config_id=spec["id"],random_replicate=replicate,evaluation_partition=fold)
                append(path,result);done.add((fold,row_key(row)))
                if len(done)%20==0:
                    progress={"stage":"random_generation" if random else "candidate_generation","config":label,"completed":len(done),"expected":len(rows)}
                    write_json(engine.out/"search_progress.json",progress);print(progress,flush=True)
            print("generation_complete",label,len(done),flush=True)
    write_json(engine.out/("random_generation_complete.json" if random else "candidate_generation_complete.json"),{"status":"complete","specs":specs,"inputs_per_config":len(rows),"validation_used":0,"sealed_used":0})


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--phase",choices=["screen","generate","random"],required=True)
    args=parser.parse_args();engine=Engine(args.config,args.phase)
    lock=(engine.out/".search.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.phase=="screen":screen(engine)
    else:generation(engine,args.phase=="random")
    return 0


if __name__=="__main__":raise SystemExit(main())
