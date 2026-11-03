#!/usr/bin/env python3
"""Chinese-first Qwen normalization/translation; agents review translation only."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = 'zh-candidate-pipeline-v1'
CHECKS = {'translation_equivalent', 'natural', 'answer_not_leaked', 'identity_preserved'}
ZH_FIELDS = {'subject_zh', 'prompt_zh', 'canonical_fact_zh', 'answer_zh', 'answer_aliases_zh'}
NORMALIZE = '''Using ONLY the supplied original QA and old extractions, normalize a reusable factual candidate.
The correct source answer is given; do not answer from memory or use wrong choices as facts.
Keep an existing answer exactly; if it conflicts with source, defer. If absent, copy a concise literal
substring from source_answer. Preserve uncertainty and qualifiers. Do not invent dates/markets/versions.
Use an existing relation ID only when its definition/direction fits exactly. If no listed relation fits,
propose a precise new snake_case ID with explicit definition and direction; never force a broad category.
Insufficient source support must defer.
Return JSON {records:[{record_id,decision,subject,relation_id,relation_definition,direction,answer,canonical_fact,prompt_en,evidence_quotes,reason}]}.
decision is ready|defer|reject. For ready, all text fields are nonempty, evidence_quotes contains literal
substrings of source_question/source_answer, and prompt_en is option-free and does not contain its answer.
For defer/reject all candidate fields are null and evidence_quotes is []; reason is nonempty.
Old reviews are not gold. There is NO downstream independent relationship review in this pipeline.'''
TRANSLATE = '''Translate all supplied English fields faithfully into natural Simplified Chinese.
Do not correct or strengthen English facts or introduce unsupported specificity. Preserve entity identity,
relation direction, qualifiers, answer identity and alias order. prompt_zh must not reveal the answer.
Return JSON {records:[{record_id,subject_zh,prompt_zh,canonical_fact_zh,answer_zh,answer_aliases_zh}]}.
Exactly one row per record_id; aliases length/order must match English. No extra fields.'''
REVIEW = '''Review EN->ZH translation ONLY, not factual truth or relation admission. Read the complete
English and Chinese fields of EVERY packet row. Review in bounded groups (at most5 rows per tool output),
without truncation. No outside lookup. Check all fields, entity/answer/alias identity and order, qualifiers,
meaning, usable natural Chinese and no answer leaked in prompt. Minor stylistic preferences alone do not
justify rejection. Copy record_id,input_sha256,english_sha256,translation_sha256 exactly.
Return one JSONL row per packet item: copied hashes, verdict(accept|reject|defer), checks(object with
translation_equivalent,natural,answer_not_leaked,identity_preserved booleans), reason(nonempty),issues(list
of strings),human_gold:false. accept iff all4 checks true. Use defer for genuinely uncertain translation
cases, reject for clear material errors. Do NOT revise the English, certify its relation/fact, or add facts.
Luna reviews every translated candidate. Only translation uncertainty is routed to a different Sol agent;
Sol must also copy review_sha256 and may accept/reject/defer. Sol is model adjudication, not human gold.'''


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp=f.name
    os.replace(tmp,path)


def write_rows(path: Path, value: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        f.write(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in value));tmp=f.name
    os.replace(tmp,path)


def indexed(value: Any, inputs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(value,dict) or set(value)!={'records'} or not isinstance(value['records'],list):
        raise ValueError('invalid_records_schema')
    result={}
    for row in value['records']:
        if not isinstance(row,dict) or not isinstance(row.get('record_id'),str) or row['record_id'] in result:
            raise ValueError('duplicate_or_invalid_record_id')
        result[row['record_id']]=row
    if set(result)!={r['record_id'] for r in inputs}:raise ValueError('record_id_coverage_mismatch')
    return result


def normalize(value: Any, inputs: list[dict[str, Any]], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    by=indexed(value,inputs);known={r['id']:r for r in catalog['relations']};out=[]
    fields={'subject','relation_id','relation_definition','direction','answer','canonical_fact','prompt_en'}
    expected={'record_id','decision','reason','evidence_quotes'}|fields
    for source in inputs:
        row=by[source['record_id']]
        if set(row)!=expected or row['decision'] not in {'ready','defer','reject'} or not isinstance(row['reason'],str) or not row['reason'].strip():
            raise ValueError('invalid_normalization_schema')
        if row['decision']=='ready':
            if any(not isinstance(row[k],str) or not row[k].strip() for k in fields):raise ValueError('missing_candidate_field')
            if not re.fullmatch('[a-z][a-z0-9_]{2,79}',row['relation_id']):raise ValueError('invalid_relation_id')
            spec=known.get(row['relation_id'])
            if spec and (row['relation_definition']!=spec['definition'] or row['direction']!=spec['direction']):raise ValueError('known_relation_spec_mismatch')
            if source.get('answer') and row['answer']!=source['answer']:raise ValueError('existing_answer_changed')
            if not source.get('answer') and row['answer'] not in source['source_answer']:raise ValueError('answer_not_in_source')
            quotes=row['evidence_quotes']
            if not isinstance(quotes,list) or not quotes or any(not isinstance(q,str) or not q or not any(q in str(source.get(k) or '') for k in ['source_question','source_answer']) for q in quotes):
                raise ValueError('unsupported_evidence_quote')
            if re.search(r'(?<!\w)'+re.escape(row['answer'])+r'(?!\w)',row['prompt_en'],re.I):raise ValueError('answer_leaked_in_prompt')
            aliases=list(dict.fromkeys([row['answer'],*([x for x in source.get('answer_aliases_en',[]) if isinstance(x,str) and x.strip()])]))
            candidate={k:row[k] for k in fields};candidate['answer_aliases_en']=aliases
            candidate['relation_spec']=spec or {'id':row['relation_id'],'definition':row['relation_definition'],'direction':row['direction']}
            candidate['catalog_known_relation']=spec is not None
            row={**row,'english':candidate,'english_sha256':digest(candidate)}
        elif any(row[k] is not None for k in fields) or row['evidence_quotes']!=[]:raise ValueError('nonready_has_candidate')
        out.append({**row,'input_sha256':digest(source),'human_gold':False,'independent_relation_review':False})
    return out


def translate(value: Any, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by=indexed(value,inputs);out=[]
    for item in inputs:
        row=by[item['record_id']]
        if set(row)!=ZH_FIELDS|{'record_id'} or any(not isinstance(row[k],str) or not row[k].strip() for k in ZH_FIELDS-{'answer_aliases_zh'}):raise ValueError('invalid_translation_schema')
        aliases=row['answer_aliases_zh']
        if not isinstance(aliases,list) or len(aliases)!=len(item['english']['answer_aliases_en']) or any(not isinstance(x,str) or not x.strip() for x in aliases):raise ValueError('alias_count_or_order_contract')
        if row['answer_zh'].casefold() in row['prompt_zh'].casefold():raise ValueError('chinese_answer_leak')
        out.append({**item,'translation':row,'translation_sha256':digest(row)})
    return out


def compact(source: dict[str, Any]) -> dict[str, Any]:
    fields=['record_id','source_question','source_answer','subject','subject_type','relation_raw','answer','answer_type','canonical_fact','prompt_en','answer_aliases_en']
    result={k:source.get(k) for k in fields}
    result['model_annotations']={k:{f:v.get(f) for f in ['extraction_status','exclusion_reason','subject','relation_raw','answer','canonical_fact']} for k,v in source.get('model_annotations',{}).items()}
    return result


def prepare(root: Path, env_file: Path, concurrency: int, mode: str) -> None:
    if (root/'manifest.json').exists():raise ValueError('already_prepared')
    if concurrency not in {8,16}:raise ValueError('concurrency_must_be_8_or_16')
    inputs=rows(root/'input.jsonl')
    if len(inputs)!=7250 or len({r['record_id'] for r in inputs})!=len(inputs):raise ValueError('source_universe_mismatch')
    sys.path.insert(0,str(Path(__file__).parent));from probe_relation_expansion_api import load_route
    _,base,model=load_route(env_file)
    runtime=root/'runtime';runtime.mkdir()
    for name in [Path(__file__).name,'probe_relation_expansion_api.py']:
        shutil.copyfile(Path(__file__).parent/name,runtime/name)
    bindings={str(p):sha(p) for p in [root/'input.jsonl',root/'relation_catalog.json',*runtime.glob('*.py')]}
    m={'version':VERSION,'created_at':now(),'bindings':bindings,'source_count':len(inputs),'env_file':str(env_file.resolve()),'requested_model':model,'expected_response_model':'Qwen-3.8-Max','route_sha256':digest([base.rstrip('/'),model]),'concurrency':concurrency,'batch_size':2,'review_packet_size':20,'execution_mode':mode,'runtime':str(runtime),'python':sys.executable,'normalization_independent_review':False,'review_requested_model':'gpt-6-luna','adjudicator_requested_model':'gpt-6.1-sol','translation_scope':'core_fact_fields_only_no_generated_neutral_or_targeted_context','automatic_semantic_repairs':0,'prompt_hashes':{'normalization':digest(NORMALIZE),'translation':digest(TRANSLATE),'review':digest(REVIEW)},'human_gold':False,'formal_experiment_eligible':False}
    atomic(root/'manifest.json',m);atomic(root/'status.json',{'state':'prepared','input_count':len(inputs),'updated_at':now()})
    atomic(root/'formal_entry_gates.json',{'candidate_relation_catalog_bound':True,'source_universe_bound':True,'formal_relation_membership_frozen':False,'canonical_and_alias_freeze_complete':False,'final_rendered_input_review_complete':False,'historical_exposure_closed':False,'split_and_leakage_components_frozen':False,'target_model_and_scoring_frozen':False,'formal_experiment_eligible':False,'pnt_eligible':False})


def verify(root: Path) -> dict[str, Any]:
    m=read(root/'manifest.json')
    if any(sha(Path(p))!=h for p,h in m['bindings'].items()):raise ValueError('frozen_binding_changed')
    return m


def pack(root: Path, flush: bool=False, verified_manifest: dict[str, Any]|None=None) -> None:
    m=verified_manifest or verify(root);translated=[];claimed=set()
    for p in sorted((root/'review_packets').glob('packet_*/manifest.json')):
        pm=read(p)
        if sha(p.parent/'input.jsonl')!=pm['input_file_sha256']:raise ValueError('review_packet_changed')
        claimed.update(r['record_id'] for r in rows(p.parent/'input.jsonl'))
    for p in sorted((root/'batches').glob('batch_*.json')):
        v=read(p)
        for item in v.get('translated',[]):
            if digest(item['english'])!=item['english_sha256'] or digest(item['translation'])!=item['translation_sha256']:raise ValueError('checkpoint_payload_hash_changed')
            if item['record_id'] not in claimed:translated.append(item)
    size=m['review_packet_size']
    count=len(list((root/'review_packets').glob('packet_*/manifest.json')))
    for i in range(0,len(translated),size):
        chunk=translated[i:i+size]
        if len(chunk)<size and not flush:break
        target=root/'review_packets'/f'packet_{count:04d}';target.mkdir(parents=True,exist_ok=False)
        write_rows(target/'input.jsonl',chunk);(target/'instructions.md').write_text(REVIEW+'\n')
        atomic(target/'manifest.json',{'record_count':len(chunk),'input_file_sha256':sha(target/'input.jsonl'),'instructions_sha256':sha(target/'instructions.md'),'run_manifest_sha256':sha(root/'manifest.json'),'review_scope':'translation_only','human_gold':False})
        count+=1


def progress(root: Path, state: str|None=None) -> dict[str, Any]:
    counts=Counter();n=0;t=0;decisions=Counter()
    for p in (root/'batches').glob('batch_*.json'):
        v=read(p);counts[v['status']]+=1;n+=len(v.get('normalized',[]));t+=len(v.get('translated',[]));decisions.update(x['decision'] for x in v.get('normalized',[]))
    outcomes=Counter()
    for p in (root/'review_packets').glob('packet_*/final.jsonl'):outcomes.update(v['status'] for v in rows(p))
    packets=len(list((root/'review_packets').glob('packet_*/manifest.json')))
    prior=read(root/'status.json') if (root/'status.json').exists() else {}
    value={'updated_at':now(),'state':state or prior.get('state','prepared'),'input_count':read(root/'manifest.json')['source_count'],'normalized_records':n,'normalization_decisions':dict(decisions),'translated_records':t,'batch_status_counts':dict(counts),'review_packets':packets,'reviewed_outcomes':dict(outcomes),'human_gold':False,'formal_experiment_eligible':False}
    atomic(root/'status.json',value);return value



BACKOFF_SCHEDULE = (60, 120, 300)


def transport_failure(detail):
    status = detail.get('http_status')
    return (detail.get('error_code') == 'gateway_zero_usage_error_response'
            or bool(detail.get('gateway_error_hint'))
            or detail.get('error_type') in {'APIConnectionError', 'APITimeoutError', 'InternalServerError', 'RateLimitError'}
            or isinstance(status, int) and (status == 429 or status >= 500))


def verify_overlay(root):
    authority = read(root/'active_worker_runtime.json')
    if authority['parent_manifest_sha256'] != sha(root/'manifest.json'):
        raise ValueError('worker_parent_manifest_changed')
    if Path(authority['entrypoint']).resolve() != Path(__file__).resolve():
        raise ValueError('worker_entrypoint_changed')
    for name, expected in authority['bindings'].items():
        if sha(Path(name)) != expected:
            raise ValueError('worker_runtime_binding_changed')
    return authority


def tail_retry_batches(root, source, manifest):
    claimed = {row['record_id'] for p in (root/'review_packets').glob('packet_*/input.jsonl') for row in rows(p)}
    selected = []
    for index, start in enumerate(range(0, len(source), manifest['batch_size'])):
        path = root/'batches'/f'batch_{index:05d}.json'
        if not path.exists():
            continue
        value = read(path)
        attempts = value.get('attempts', [])
        if (value.get('status') != 'failed' or not attempts or not transport_failure(attempts[-1])
                or value.get('recovery_history') or value.get('transport_retry_epoch')):
            continue
        batch = source[start:start+manifest['batch_size']]
        if value['input_sha256'] != digest(batch) or value.get('translated'):
            raise ValueError('tail_retry_checkpoint_changed')
        if claimed & {row['record_id'] for row in batch}:
            raise ValueError('tail_retry_reviewed_record')
        selected.append((index, batch, path, value))
    # Validate the whole selection before any requeue mutation.
    result = []
    for index, batch, path, value in selected:
        archive = root/'transport_retry_history'/'epoch_1'/path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise ValueError('tail_retry_archive_already_exists')
        shutil.copyfile(path, archive)
        if sha(archive) != sha(path):
            raise ValueError('tail_retry_archive_copy_failed')
        value['transport_retry_epoch'] = 1
        value['recovery_history'] = [{'recovery_id': 'transport_tail_epoch_1',
                                      'archive': str(archive), 'sha256': sha(archive), 'previous_status': 'failed'}]
        value['status'] = 'normalized' if value.get('normalized') else 'recovery_pending'
        atomic(path, value)
        result.append((index, batch))
    return result


def work(root: Path) -> None:
    verify_overlay(root)
    m=verify(root);lock=(root/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'fatal.json').exists() or (root/'STOP').exists():raise ValueError('stopped_run_requires_explicit_new_recovery')
    atomic(root/'worker.json',{'pid':os.getpid(),'started_at':now()})
    sys.path.insert(0,str(Path(__file__).parent));from probe_relation_expansion_api import load_route,safe_error,safe_model_label,gateway_error_hint
    from openai import OpenAI
    from qwen_stream_transport import stream_completion
    key,base,model=load_route(Path(m['env_file']))
    if digest([base.rstrip('/'),model])!=m['route_sha256']:raise ValueError('configured_route_changed')
    source=rows(root/'input.jsonl');catalog=read(root/'relation_catalog.json');stop=threading.Event();log_lock=threading.Lock();publish_lock=threading.Lock();health_lock=threading.Lock();failures=0;pause_until=0.;backoff_level=0
    def fatal(code: str,detail: dict[str, Any]) -> None:
        stop.set()
        with log_lock:
            if not (root/'fatal.json').exists():atomic(root/'fatal.json',{**detail,'at':now(),'error_code':code})
    def exposure(v: dict[str, Any]) -> None:
        with log_lock:
            with (root/'exposure.jsonl').open('a') as f:f.write(json.dumps({'at':now(),**v},ensure_ascii=False)+'\n')
    def call(stage: str, batch: list[dict[str, Any]], prompt: str, validator: Any) -> tuple[Any,list[dict[str, Any]]]:
        nonlocal failures,pause_until,backoff_level
        attempts=[];correction=''
        for attempt in range(3):
            while True:
                with health_lock:remaining=pause_until-time.monotonic()
                if remaining<=0:break
                if stop.wait(min(remaining,1)) or (root/'STOP').exists():return None,attempts
            if stop.is_set() or (root/'STOP').exists():return None,attempts
            detail={'stage':stage,'record_ids':[r['record_id'] for r in batch],'attempt':attempt+1,'prompt_sha256':digest(prompt+correction),'requested_model':model,'worker_policy':'transport_stream_v1'}
            exposure({**detail,'event':'request_started','target_behavior_exposure':False})
            start=time.monotonic()
            try:
                with OpenAI(api_key=key,base_url=base,max_retries=0,timeout=300) as c:
                    response=stream_completion(c,expected_model=m['expected_response_model'],model=model,messages=[{'role':'user','content':prompt+correction}],temperature=0,max_tokens=8192,response_format={'type':'json_object'},extra_body={'enable_thinking':False})
                detail.update(response.streaming_diagnostics)
                content=response.choices[0].message.content or ''
                detail.update(response_model=safe_model_label(response.model,key,base),usage=response.usage.model_dump() if response.usage else {},content_sha256=hashlib.sha256(content.encode()).hexdigest())
                hint=gateway_error_hint(content,detail['usage'],key,base)
                if hint:detail['gateway_error_hint']=hint;raise RuntimeError('gateway_zero_usage_error_response')
                if response.model!=m['expected_response_model']:fatal('response_model_identity_mismatch',detail);raise RuntimeError('response_model_identity_mismatch')
                with health_lock:failures=0;backoff_level=0
                result=validator(json.loads(content))
                detail.update(status='completed',latency_seconds=round(time.monotonic()-start,3));attempts.append(detail);exposure({**detail,'event':'request_finished'});return result,attempts
            except Exception as error:
                detail.update(status='failed',**safe_error(error),latency_seconds=round(time.monotonic()-start,3))
                if isinstance(error,ValueError) and re.fullmatch('[a-z_]+',str(error)):
                    detail['validation_code']=str(error);correction='\nPrevious output failed '+str(error)+'. Return the exact schema and literal source evidence.'
                attempts.append(detail);exposure({**detail,'event':'request_finished'})
                if isinstance(error,RuntimeError) and str(error)=='response_model_identity_mismatch':fatal('response_model_identity_mismatch',detail)
                status=getattr(error,'status_code',None)
                if status in {401,403,404}:fatal('api_access_or_model_unavailable',detail)
                technical=isinstance(error,RuntimeError) or type(error).__name__ in {'APIConnectionError','APITimeoutError','InternalServerError','RateLimitError'} or isinstance(status,int) and (status==429 or status>=500)
                if isinstance(error,RuntimeError) and str(error) not in {'gateway_zero_usage_error_response','response_model_identity_mismatch'}:fatal('unexpected_runtime_error',detail)
                if technical:
                    delay=0
                    with health_lock:
                        failures+=1
                        if failures>=16:
                            delay=BACKOFF_SCHEDULE[min(backoff_level,len(BACKOFF_SCHEDULE)-1)]
                            backoff_level+=1;failures=0;pause_until=max(pause_until,time.monotonic()+delay)
                    if delay:exposure({'event':'transport_backoff','seconds':delay,'target_behavior_exposure':False})
                if stop.is_set() or status==400:return None,attempts
                if attempt<2:time.sleep(2*2**attempt)
        return None,attempts
    def batch_work(index: int,batch: list[dict[str, Any]]) -> None:
        path=root/'batches'/f'batch_{index:05d}.json'
        value=read(path) if path.exists() else {'input_sha256':digest(batch),'normalized':[],'translated':[],'attempts':[]}
        if value['input_sha256']!=digest(batch):raise ValueError('checkpoint_source_changed')
        if value.get('status') in {'completed','failed'}:return
        if stop.is_set() or (root/'STOP').exists():return
        if not value['normalized']:
            prompt=NORMALIZE+'\nCATALOG:'+json.dumps(catalog,ensure_ascii=False)+'\nINPUT:'+json.dumps([compact(r) for r in batch],ensure_ascii=False)
            norm,attempts=call('normalize',batch,prompt,lambda v:normalize(v,batch,catalog));value['attempts']+=attempts
            if norm is None:value['status']='failed';atomic(path,value);return
            value['normalized']=norm;value['status']='normalized';atomic(path,value)
        items=[{k:r[k] for k in ['record_id','input_sha256','english','english_sha256']} for r in value['normalized'] if r['decision']=='ready']
        if items:
            prompt=TRANSLATE+'\nINPUT:'+json.dumps(items,ensure_ascii=False)
            zh,attempts=call('translate',items,prompt,lambda v:translate(v,items));value['attempts']+=attempts
            if zh is None:value['status']='failed';atomic(path,value);return
            value['translated']=zh
        value['status']='completed';atomic(path,value)
    progress(root,'running_qwen')
    def run_pass(batches):
        with ThreadPoolExecutor(max_workers=m['concurrency']) as executor:
            futures=[executor.submit(batch_work,i,batch) for i,batch in batches]
            for future in as_completed(futures):
                future.result()
                if stop.is_set() or (root/'STOP').exists():continue
                with publish_lock:
                    if m['execution_mode']=='pipeline':pack(root,verified_manifest=m)
                    progress(root,'running_qwen')
    run_pass([(i,source[j:j+m['batch_size']]) for i,j in enumerate(range(0,len(source),m['batch_size']))])
    if not stop.is_set() and not (root/'STOP').exists():
        verify_overlay(root);verify(root)
        retries=tail_retry_batches(root,source,m)
        exposure({'event':'transport_tail_retry_started','batch_count':len(retries),'target_behavior_exposure':False})
        run_pass(retries)
    pack(root,flush=True,verified_manifest=m)
    progress(root,'blocked_api' if stop.is_set() else 'stopped_by_request' if (root/'STOP').exists() else 'qwen_finished_reviews_pending')


def validate_review(row: dict[str, Any], item: dict[str, Any], review_hash: str|None=None) -> None:
    keys=['record_id','input_sha256','english_sha256','translation_sha256']
    if any(row.get(k)!=item[k] for k in keys):raise ValueError('review_binding_mismatch')
    if row.get('human_gold') is not False or row.get('verdict') not in {'accept','reject','defer'} or not isinstance(row.get('reason'),str) or not row['reason'].strip():raise ValueError('invalid_review')
    checks=row.get('checks')
    if not isinstance(checks,dict) or set(checks)!=CHECKS or any(type(v) is not bool for v in checks.values()):raise ValueError('invalid_review_checks')
    if (row['verdict']=='accept')!=all(checks.values()):raise ValueError('review_decision_inconsistent')
    if not isinstance(row.get('issues'),list) or any(not isinstance(x,str) for x in row['issues']):raise ValueError('invalid_review_issues')
    if review_hash is not None and row.get('review_sha256')!=review_hash:raise ValueError('sol_review_binding_mismatch')


def finalize(root: Path,packet: Path,dispatch: Path) -> None:
    verify(root);pm=read(packet/'manifest.json')
    if sha(packet/'input.jsonl')!=pm['input_file_sha256'] or sha(packet/'instructions.md')!=pm['instructions_sha256'] or pm['run_manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('packet_binding_changed')
    items=rows(packet/'input.jsonl');by={r['record_id']:r for r in items};ledger=read(dispatch)
    if ledger.get('packet_manifest_sha256')!=sha(packet/'manifest.json'):raise ValueError('dispatch_packet_mismatch')
    reviews={};sol={};output_bindings=[]
    for d in ledger['dispatches']:
        role=d.get('role');expected='gpt-6-luna' if role=='reviewer' else 'gpt-6.1-sol' if role=='adjudicator' else None
        if not expected or d.get('requested_model')!=expected or not d.get('agent_id'):raise ValueError('invalid_actual_agent_dispatch')
        p=Path(d['output_path'])
        if sha(p)!=d['output_sha256']:raise ValueError('agent_output_changed')
        output_bindings.append(d)
        for row in rows(p):
            target=reviews if role=='reviewer' else sol
            if row['record_id'] not in by or row['record_id'] in target:raise ValueError('duplicate_or_unassigned_review')
            target[row['record_id']]=row
    if set(reviews)!=set(by):raise ValueError('incomplete_luna_review')
    if not set(sol)<={rid for rid,r in reviews.items() if r['verdict']=='defer'}:raise ValueError('unrequested_sol_adjudication')
    luna_ids={d['agent_id'] for d in output_bindings if d['role']=='reviewer'}
    if any(d['agent_id'] in luna_ids for d in output_bindings if d['role']=='adjudicator'):raise ValueError('sol_not_independent_agent')
    output=[];difficult=[]
    for rid,item in by.items():
        rev=reviews[rid];validate_review(rev,item);ad=sol.get(rid)
        if ad:validate_review(ad,item,digest(rev))
        effective=ad or rev
        if rev['verdict']=='defer' and ad is None:
            status='pending_sol_translation';difficult.append({**item,'review':rev,'review_sha256':digest(rev)})
        else:status={'accept':'zh_translation_accepted','reject':'zh_translation_rejected','defer':'zh_translation_deferred'}[effective['verdict']]
        output.append({**item,'status':status,'review':rev,'adjudication':ad,'normalization_independent_review':False,'formal_experiment_eligible':False,'human_gold':False})
    write_rows(packet/'final.jsonl',output);write_rows(packet/'sol_inputs.jsonl',difficult)
    atomic(packet/'final_manifest.json',{'final_sha256':sha(packet/'final.jsonl'),'dispatch_sha256':sha(dispatch),'real_agent_bindings':output_bindings,'counts':dict(Counter(r['status'] for r in output)),'formal_experiment_eligible':False,'human_gold':False})
    progress(root)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','launch','work','pack','status','finalize']);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--env-file',type=Path);parser.add_argument('--concurrency',type=int,default=8);parser.add_argument('--mode',choices=['pipeline','bulk'],default='pipeline');parser.add_argument('--packet',type=Path);parser.add_argument('--dispatch',type=Path);args=parser.parse_args();root=args.run_dir.resolve()
    if args.command=='prepare':prepare(root,args.env_file,args.concurrency,args.mode)
    elif args.command=='work':
        try:work(root)
        except Exception as e:
            atomic(root/'worker_error.json',{'at':now(),'error_type':type(e).__name__,'error_code':str(e) if re.fullmatch('[a-z_]+',str(e)) else None});raise
    elif args.command=='launch':
        m=verify(root)
        if (root/'fatal.json').exists() or (root/'STOP').exists():raise ValueError('run_stopped')
        with (root/'launch.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (root/'worker.json').exists():
                try:os.kill(read(root/'worker.json')['pid'],0)
                except ProcessLookupError:pass
                else:raise ValueError('worker_already_exists')
            with (root/'worker.log').open('a') as log:
                child=subprocess.Popen([m['python'],str(Path(m['runtime'])/Path(__file__).name),'work','--run-dir',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
            atomic(root/'worker.json',{'pid':child.pid,'launched_at':now()});print(json.dumps({'state':'launched','worker_pid':child.pid}))
    elif args.command=='pack':pack(root,flush=True)
    elif args.command=='finalize':finalize(root,args.packet,args.dispatch)
    else:print(json.dumps(progress(root),ensure_ascii=False))


if __name__=='__main__':main()
