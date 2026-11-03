"""User-authorized, checkpoint-preserving streaming transport deployment."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
RUN=HERE.parent/'zh_normalize_translate_20261023_v1'
ARCHIVE=RUN/'operational_history/streaming_transport_20261024_v1'
sys.path.insert(0,str(REPO/'scripts'))
from recover_zh_candidate_pipeline import read, sha, worker_entrypoint, health, module, recover

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def verify_current():
    m=module(RUN/'runtime/run_zh_candidate_pipeline.py','frozen_stream_upgrade').verify(RUN)
    for p,h in read(RUN/'operations_tools.json')['files'].items():
        if sha(Path(p))!=h:raise ValueError('operations_binding_changed')
    entry=worker_entrypoint(RUN,m)
    return m,entry

def stop_old():
    m,entry=verify_current();info=read(RUN/'worker.json');pid=info['pid']
    expected=[m['python'],str(entry),'work','--run-dir',str(RUN)]
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    if p.returncode!=0 or shlex.split(p.stdout.strip())!=expected:raise ValueError('worker_identity_changed')
    if (RUN/'STOP').exists():raise ValueError('existing_user_stop')
    if (RUN/'fatal.json').exists() or (RUN/'worker_error.json').exists():raise ValueError('unexpected_worker_error')
    plan={'at':datetime.now(timezone.utc).isoformat(),'old_pid':pid,'old_command':expected,
          'purpose':'explicit_user_interface_repair_streaming_transport','operator_stop_not_user_cancellation':True}
    atomic(HERE/'controlled_stop.json',plan)
    with (RUN/'STOP').open('x') as f:f.write(json.dumps(plan)+'\n')
    shutil.copyfile(RUN/'STOP',HERE/'operator_STOP.json')
    print(json.dumps({'state':'draining_existing_worker','pid':pid,'signal_sent':False}))

def activate():
    m,oldentry=verify_current();plan=read(HERE/'controlled_stop.json');pid=plan['old_pid']
    if read(RUN/'worker.json')['pid']!=pid:raise ValueError('worker_changed')
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    if p.returncode!=1 or p.stdout.strip():raise ValueError('old_worker_still_alive')
    if sha(RUN/'STOP')!=sha(HERE/'operator_STOP.json'):raise ValueError('stop_marker_changed')
    if read(RUN/'status.json')['state']!='stopped_by_request':raise ValueError('worker_did_not_drain')
    candidate=RUN/'runtime_transport_stream_v1'
    evidence_manifest={**m,'transport_probe_runtime':str(candidate),'transport_probe_min_success_ratio':.75,
                       'transport_helper_sha256':sha(candidate/'qwen_stream_transport.py')}
    route=module(RUN/'runtime/probe_relation_expansion_api.py','stream_upgrade_route')
    _,base,model=route.load_route(Path(m['env_file']))
    if module(RUN/'runtime/run_zh_candidate_pipeline.py','digest_stream_upgrade').digest([base.rstrip('/'),model])!=m['route_sha256']:
        raise ValueError('route_changed')
    reports=[HERE/'stream_final_probe/report.json']
    evidence=health(reports,evidence_manifest,base,model)
    # Bound current data before changing the active operational authority.
    protected={str(p):sha(p) for p in (RUN/'review_packets').glob('packet_*/*') if p.is_file()}
    checkpoints={str(p):sha(p) for p in (RUN/'batches').glob('batch_*.json')}
    exposure=(RUN/'exposure.jsonl').read_bytes()
    before={'at':datetime.now(timezone.utc).isoformat(),'worker':read(RUN/'worker.json'),'status':read(RUN/'status.json'),
            'protected_review_files':protected,'batch_checkpoints':checkpoints,'exposure_prefix_bytes':len(exposure),
            'exposure_prefix_sha256':hashlib.sha256(exposure).hexdigest(),'probe_evidence':evidence}
    atomic(HERE/'before_activation.json',before)
    authority=read(RUN/'active_worker_runtime.json')
    authority.update(version='zh-transport-stream-v1',response_transport='sse',
        user_authorization='User2026-10-24 explicitly requested fixing the Qwen interface; matched synthetic probes support streaming transport.',
        parent_runtime_authority_sha256=sha(RUN/'active_worker_runtime.json'),
        entrypoint=str(candidate/'run_zh_candidate_pipeline.py'),
        transport_probe_runtime=str(candidate),
        bindings={str(p):sha(p) for p in candidate.glob('*.py')},
        generation_request_semantics_unchanged=True,output_tokens_unchanged=8192)
    atomic(RUN/'active_worker_runtime.json',authority)
    policy=read(RUN/'operations_policy.json');policy['version']='zh-operations-v4-stream-20261024'
    policy['response_transport']='sse_complete_identity_usage_required'
    policy['authorization_streaming']='Explicit user API repair; original v3 continuation and bounded tail retry remain unchanged.'
    atomic(RUN/'operations_policy.json',policy)
    bindings=read(RUN/'operations_tools.json');bindings.update(version='zh-operations-tools-v4-stream-20261024',tests_passed=59)
    bindings['files']={p:sha(Path(p)) for p in bindings['files']}
    atomic(RUN/'operations_tools.json',bindings)
    # Remove only this repair's own byte-identical maintenance marker.
    shutil.copyfile(RUN/'STOP',ARCHIVE/'operator_STOP.json');(RUN/'STOP').unlink()
    api=module(RUN/'runtime/run_zh_candidate_pipeline.py','stream_upgrade_progress')
    api.progress(RUN,'stopped_for_transport_upgrade')
    result=recover(RUN,'stream_transport_20261024_v1',reports,apply=True,limit=32)
    atomic(HERE/'activation_result.json',result)
    for name,h in protected.items():
        if sha(Path(name))!=h:raise ValueError('review_changed_during_upgrade')
    for name,h in checkpoints.items():
        if sha(Path(name))!=h:raise ValueError('checkpoint_changed_during_upgrade')
    if hashlib.sha256((RUN/'exposure.jsonl').read_bytes()[:len(exposure)]).hexdigest()!=before['exposure_prefix_sha256']:
        raise ValueError('exposure_prefix_changed')
    print(json.dumps({'state':result['state'],'worker_pid':result['worker_pid'],'old_pid':pid,'review_files_preserved':True,
                      'checkpoint_bytes_preserved':True,'exposure_prefix_preserved':True}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stop','activate']);args=parser.parse_args()
    if args.action=='stop':stop_old()
    else:activate()
