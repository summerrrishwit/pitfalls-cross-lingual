#!/usr/bin/env python3
"""Export source-complete Chinese candidate coverage; never certify formal eligibility."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from run_zh_candidate_pipeline import verify,read,rows,digest,write_rows,atomic,now


def export(root: Path) -> dict:
    manifest=verify(root);sources={r['record_id']:r for r in rows(root/'input.jsonl')};norm={};translated={};reviewed={};failed={}
    for p in sorted((root/'batches').glob('batch_*.json')):
        b=read(p)
        if b['status']=='failed':
            batch_index=int(p.stem.split('_')[1]);batch_size=manifest.get('batch_size',2)
            ids=list(sources)[batch_index*batch_size:(batch_index+1)*batch_size]
            if digest([sources[rid] for rid in ids])!=b['input_sha256']:raise ValueError('failed_checkpoint_scope_mismatch')
            failed.update({rid:('technical_translation_failed' if b.get('normalized') else 'technical_normalization_failed') for rid in ids})
        for item in b.get('normalized',[]):
            rid=item['record_id']
            if rid not in sources or item['input_sha256']!=digest(sources[rid]):raise ValueError('source_binding_mismatch')
            if rid in norm:raise ValueError('duplicate_normalization')
            norm[rid]=item
        for item in b.get('translated',[]):
            rid=item['record_id']
            if rid in translated or rid not in norm or item['english_sha256']!=norm[rid]['english_sha256'] or digest(item['english'])!=item['english_sha256'] or digest(item['translation'])!=item['translation_sha256']:raise ValueError('translation_binding_mismatch')
            translated[rid]=item
    for p in sorted((root/'review_packets').glob('packet_*/final_manifest.json')):
        m=read(p);final=p.parent/'final.jsonl'
        dispatch=next((q for q in [p.parent/'dispatch.json',root/'dispatch.json'] if q.exists() and hashlib.sha256(q.read_bytes()).hexdigest()==m['dispatch_sha256']),None)
        if hashlib.sha256(final.read_bytes()).hexdigest()!=m['final_sha256'] or dispatch is None:raise ValueError('final_binding_mismatch')
        for binding in m['real_agent_bindings']:
            if hashlib.sha256(Path(binding['output_path']).read_bytes()).hexdigest()!=binding['output_sha256']:raise ValueError('review_output_changed')
        for item in rows(final):
            rid=item['record_id']
            if rid in reviewed or rid not in translated or item['english_sha256']!=translated[rid]['english_sha256'] or item['translation_sha256']!=translated[rid]['translation_sha256']:raise ValueError('review_binding_mismatch')
            if item.get('formal_experiment_eligible') is not False or item.get('human_gold') is not False:raise ValueError('invalid_promotion')
            reviewed[rid]=item
    coverage=[]
    for rid,source in sources.items():
        n=norm.get(rid);rev=reviewed.get(rid)
        status=rev['status'] if rev else 'pending_translation_review' if rid in translated else 'normalization_'+n['decision'] if n and n['decision']!='ready' else failed[rid] if rid in failed else 'pending_translation' if n else 'pending_normalization'
        coverage.append({'record_id':rid,'source_sha256':digest(source),'status':status,'formal_experiment_eligible':False,'human_gold':False})
    output=root/'exports';output.mkdir(exist_ok=True)
    write_rows(output/'source_coverage.jsonl',coverage)
    write_rows(output/'normalized_candidates.jsonl',[n for n in norm.values() if n['decision']=='ready'])
    write_rows(output/'accepted_chinese_candidates.jsonl',[r for r in reviewed.values() if r['status']=='zh_translation_accepted'])
    counts=Counter(x['status'] for x in coverage)
    specs={}
    for n in norm.values():
        if n['decision']=='ready':
            spec=n['english']['relation_spec'];specs.setdefault(spec['id'],{})[digest(spec)]=spec
    atomic(output/'proposed_relation_catalog.json',{'relations':specs,'conflicting_ids':[k for k,v in specs.items() if len(v)>1],'formal_membership_frozen':False})
    summary={'at':now(),'source_count':len(sources),'normalized_count':len(norm),'translated_count':len(translated),'reviewed_count':len(reviewed),'source_status_counts':dict(counts),'counts_are_not_deduplicated_facts_or_net_new160':True,'new_target_behavior_calls':0,'formal_experiment_eligible':False,'human_gold':False,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.jsonl')}}
    atomic(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();print(json.dumps(export(a.run_dir.resolve()),ensure_ascii=False))
