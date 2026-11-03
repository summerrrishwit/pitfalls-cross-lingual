#!/usr/bin/env python3
"""Collect open-answer trajectories and leak-free OOF source vectors remotely."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import Engine, append, read_json, read_jsonl, render, sha, suffix, write_json


def collect_chunks(engine: Engine, specs: list[dict[str, Any]], directory: Path, is_open: bool) -> Any:
    torch=engine.torch
    directory.mkdir(exist_ok=True)
    all_states=[]
    chunk_size=32
    for start in range(0,len(specs),chunk_size):
        batch=specs[start:start+chunk_size]
        ids=[r["option_free_id"] if is_open else r["vector_source_id"] for r in batch]
        path=directory/f"chunk-{start:04d}.pt"
        metrics_path=directory/f"chunk-{start:04d}.json"
        if path.exists() and metrics_path.exists():
            saved=torch.load(path,map_location="cpu",weights_only=True)
            if saved["ids"]!=ids:
                raise ValueError("cached states belong to different source IDs")
            all_states.append(saved["states"])
            continue
        states=[];metadata=[]
        for row in batch:
            tokens=row["prompt_token_ids"] if is_open else render(engine.tokenizer,row["prompt"])[1]
            activation=engine.states(tokens)
            states.append(activation.cpu().to(torch.bfloat16))
            entry={"id":row["option_free_id"] if is_open else row["vector_source_id"],
                   "base_fact_id":row["base_fact_id"],"prompt_tokens":len(tokens)}
            if is_open:
                entry.update(language=row["language"],variant=row["variant"],target=row["designated_distractor_id"])
                with torch.inference_mode():
                    logits=engine.model.lm_head(engine.model.model.norm(activation)).float()
                    logp=torch.log_softmax(logits,dim=-1)
                    candidates={}
                    for group in row["answer_groups"]:
                        for language in ("en","zh"):
                            first_ids=sorted({suffix(engine.tokenizer,row["rendered_prompt"],tokens,a)[0] for a in group[f"aliases_{language}"]})
                            ids_tensor=torch.tensor(first_ids,device="cuda")
                            values=logp[:,ids_tensor]
                            best=values.argmax(dim=-1)
                            selected=ids_tensor[best]
                            ranks=(logits>logits[torch.arange(36,device="cuda"),selected,None]).sum(dim=-1)+1
                            candidates[f"{group['option_id']}:{language}"]={"best_first_token_logprob":values.max(dim=-1).values.cpu().tolist(),"best_first_token_rank":ranks.cpu().tolist(),"token_ids":first_ids}
                    entry["first_token_lens"]=candidates
                    entry["diagnostic_only"]="first-token lens; multi-token complete answer scores remain separate"
                    del logits,logp,values
            metadata.append(entry)
        tensor=torch.stack(states)
        temporary=path.with_suffix(".tmp")
        torch.save({"ids":ids,"states":tensor},temporary);temporary.replace(path)
        write_json(metrics_path,metadata)
        all_states.append(tensor)
        write_json(directory/"progress.json",{"completed":min(start+chunk_size,len(specs)),"expected":len(specs)})
        print(directory.name,min(start+chunk_size,len(specs)),"/",len(specs),flush=True)
    return torch.cat(all_states)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--final",action="store_true")
    args=parser.parse_args();engine=Engine(args.config,"final-vectors" if args.final else "collect")
    if args.final:
        rule=read_json(engine.out/"selected_rule.json")
        if rule.get("status")!="oof_selected_pending_final_vector":
            raise ValueError("final vector source cannot be consumed before OOF rule selection")
    else:
        engine.no_op_audit()
        collect_chunks(engine,engine.inputs,engine.out/"open-trajectories",True)
    source=read_jsonl(engine.root/"inputs/g2a/vector_source_prompt_specs.jsonl")
    source=[r for r in source if (r["source_partition_id"]=="final-development-all")==args.final]
    if len(source)!=(480 if args.final else 1440):
        raise ValueError("source coverage changed")
    for row in source:
        excluded=row["excluded_evaluation_fold_id"]
        if excluded and (engine.folds[row["base_fact_id"]]==excluded or any(engine.folds[d]==excluded for d in row["icl_donor_base_fact_ids"])):
            raise ValueError("evaluation fold entered vector source/donors")
        if any(engine.components[d]==row["leakage_component_id"] or engine.relations[d]!=row["probe_relation_id"] for d in row["icl_donor_base_fact_ids"]):
            raise ValueError("donor relation/component rule violated")
    tensor=collect_chunks(engine,source,engine.out/("final-source-states" if args.final else "oof-source-states"),False)
    indices=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(source):indices[r["source_partition_id"]][r["kind"]].append(i)
    bank={};checks={}
    for partition,kinds in indices.items():
        count=96 if args.final else 72
        if any(len(v)!=count for v in kinds.values()) or len(kinds)!=5:
            raise ValueError("source means are not fact balanced")
        means={k:tensor[v].float().mean(dim=0) for k,v in kinds.items()}
        bank[partition]={"task":(means["task_icl_recall_en"]+means["task_icl_recall_zh"])/2,
                         "difference":means["translation_en_to_zh"]-(means["fact_recall_reference_en"]+means["fact_recall_reference_zh"])/2}
        checks[partition]={"source_facts":count,"prompt_kinds":dict(Counter(r["kind"] for r in source if r["source_partition_id"]==partition)),
                           "source_fact_ids":sorted({r["base_fact_id"] for r in source if r["source_partition_id"]==partition})}
    path=engine.out/("final_vectors.pt" if args.final else "oof_vectors.pt")
    engine.torch.save(bank,path)
    write_json(engine.out/("final_vector_manifest.json" if args.final else "oof_vector_manifest.json"),{
        "status":"complete","coordinate":"zero-based decoder layer resid_pre at last rendered prompt token",
        "source_count":len(source),"partitions":checks,"vectors_sha256":sha(path),"source_specs_sha256":sha(engine.root/"inputs/g2a/vector_source_prompt_specs.jsonl"),
        "validation_facts_used":0,"sealed_facts_used":0})
    print("vector_collection_complete",str(path),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
