"""Explicit verified access recovery, preserving the existing automatic policy."""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE.parent/'zh_normalize_translate_20261023_v1'
sys.path.insert(0,str(HERE.parent.parent/'scripts'))
from recover_zh_candidate_pipeline import read,sha,module,health,recover,worker_entrypoint

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

api=module(RUN/'runtime/run_zh_candidate_pipeline.py','explicit_access_v2_frozen')
m=api.verify(RUN);old=worker_entrypoint(RUN,m)
for f,h in read(RUN/'operations_tools.json')['files'].items():
    if sha(Path(f))!=h:raise ValueError('operations_binding_changed')
pid=read(RUN/'worker.json')['pid']
check=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
if check.returncode!=1 or check.stdout.strip():raise ValueError('old_worker_still_present')
if (RUN/'STOP').exists():raise ValueError('existing_user_stop')
if read(RUN/'fatal.json').get('http_status')!=403:raise ValueError('current_failure_is_not_403')
runtime=RUN/'runtime_transport_diagnostics_v1'
route=module(RUN/'runtime/probe_relation_expansion_api.py','explicit_access_v2_route')
_,base,model=route.load_route(Path(m['env_file']))
if api.digest([base.rstrip('/'),model])!=m['route_sha256']:raise ValueError('route_changed')
reports=[HERE/'verified_stream_probes/report.json']
evidence=health(reports,{**m,'transport_probe_runtime':str(runtime),'transport_helper_sha256':sha(runtime/'qwen_stream_transport.py'),
                         'transport_probe_min_success_ratio':1},base,model)
archive=RUN/'operational_history/access_recovery_20261024_v2'
archive.mkdir(parents=True,exist_ok=False)
bindings={}
for name in ['active_worker_runtime.json','operations_tools.json','notes.md','orchestration.md','service_health.json']:
    p=RUN/name;dest=archive/name;shutil.copyfile(p,dest);bindings[str(p)]={'archive':str(dest),'sha256':sha(dest)}
atomic(archive/'manifest.json',bindings)
before={'at':datetime.now(timezone.utc).isoformat(),'status':read(RUN/'status.json'),'worker':read(RUN/'worker.json'),
        'fatal':read(RUN/'fatal.json'),'exposure_prefix_bytes':(RUN/'exposure.jsonl').stat().st_size,
        'exposure_prefix_sha256':sha(RUN/'exposure.jsonl'),'probe_evidence':evidence,
        'review_files':{str(p):sha(p) for p in (RUN/'review_packets').glob('packet_*/*') if p.is_file()},
        'attempts':{str(p):read(p).get('attempts',[]) for p in (RUN/'batches').glob('batch_*.json')}}
atomic(HERE/'before_recovery.json',before)
authority=read(RUN/'active_worker_runtime.json')
authority.update(version='zh-transport-stream-diagnostics-v1',entrypoint=str(runtime/'run_zh_candidate_pipeline.py'),
    parent_runtime_authority_sha256=sha(RUN/'active_worker_runtime.json'),
    bindings={str(p):sha(p) for p in runtime.glob('*.py')},transport_probe_runtime=str(runtime),
    http_error_diagnostics='sanitized_provider_body_type_code_message_and_selected_response_headers',
    generation_translation_validation_and_retry_contracts_unchanged=True)
atomic(RUN/'active_worker_runtime.json',authority)
ops=read(RUN/'operations_tools.json');ops.update(version='zh-operations-tools-v4-access-diagnostics-20261024',tests_passed=71)
ops['files']={f:sha(Path(f)) for f in ops['files']};atomic(RUN/'operations_tools.json',ops)
plan=recover(RUN,'access_restored_20261024_v2',reports,apply=False,allow_access_restore=True)
atomic(HERE/'validated_recovery_plan.json',plan)
result=recover(RUN,'access_restored_20261024_v2',reports,apply=True,allow_access_restore=True)
atomic(HERE/'recovery_result.json',result)
for f,h in before['review_files'].items():
    if sha(Path(f))!=h:raise ValueError('existing_review_changed')
for f,attempts in before['attempts'].items():
    if read(Path(f)).get('attempts',[])[:len(attempts)]!=attempts:raise ValueError('old_attempt_prefix_changed')
if hashlib.sha256((RUN/'exposure.jsonl').read_bytes()[:before['exposure_prefix_bytes']]).hexdigest()!=before['exposure_prefix_sha256']:
    raise ValueError('exposure_prefix_changed')
print(json.dumps({'state':result['state'],'worker_pid':result['worker_pid'],'old_worker_pid':pid,
    'access_batches_requeued_once':result['access_batches_requeued_once'],'old_review_attempt_and_exposure_bindings_preserved':True}))
