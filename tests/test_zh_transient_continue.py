import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_zh_candidate_pipeline as original

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_zh_resilient_runtime import build
from recover_zh_candidate_pipeline import health, worker_entrypoint, inventory


def setup_run(tmp_path, count=14):
    entry = tmp_path/'runtime_v2'/'run_zh_candidate_pipeline.py'
    entry.parent.mkdir()
    entry.write_text(build((ROOT/'scripts/run_zh_candidate_pipeline.py').read_text()))
    spec = importlib.util.spec_from_file_location('resilient_test', entry)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    src, cat, _ = original.fixture()
    source = [{**src, 'record_id': f'source:{n}'} for n in range(count)]
    api.write_rows(tmp_path/'input.jsonl', source)
    api.atomic(tmp_path/'relation_catalog.json', cat)
    api.atomic(tmp_path/'manifest.json', {
        'bindings': {str(tmp_path/'input.jsonl'): api.sha(tmp_path/'input.jsonl')},
        'source_count': count, 'review_packet_size': 20, 'env_file': 'unused',
        'requested_model': 'qwen-3.8-max', 'expected_response_model': 'Qwen-3.8-Max',
        'route_sha256': api.digest(['route', 'qwen-3.8-max']),
        'concurrency': 1, 'batch_size': 2, 'execution_mode': 'pipeline', 'runtime': str(entry.parent), 'python': sys.executable})
    api.atomic(tmp_path/'active_worker_runtime.json', {
        'parent_manifest_sha256': api.sha(tmp_path/'manifest.json'), 'entrypoint': str(entry),
        'bindings': {str(entry): api.sha(entry)}, 'transport_probe_min_success_ratio': .75})
    api.BACKOFF_SCHEDULE = (0, 0, 0)
    return api, source


def mock_client(monkeypatch, fail):
    calls = []
    class Client:
        def __init__(self, **kwargs): self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def create(self, **kwargs):
            prompt = kwargs['messages'][0]['content']
            inputs, _ = json.JSONDecoder().raw_decode(prompt.split('INPUT:', 1)[1])
            stage = 'normalize' if 'CATALOG:' in prompt else 'translate'
            calls.append((stage, inputs[0]['record_id']))
            response_model, content = fail(stage, inputs, calls)
            if content is None:
                if stage == 'normalize':
                    records = [{**original.fixture()[2], 'record_id': v['record_id']} for v in inputs]
                else:
                    records = [{**original.translated()['translation'], 'record_id': v['record_id']} for v in inputs]
                content = json.dumps({'records': records})
            return SimpleNamespace(model=response_model, usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=Client))
    monkeypatch.setitem(sys.modules, 'probe_relation_expansion_api', SimpleNamespace(
        load_route=lambda _: ('private', 'route', 'qwen-3.8-max'),
        safe_error=lambda e: {'error_type': type(e).__name__, 'error_code': str(e)},
        safe_model_label=lambda v,*_:v, gateway_error_hint=lambda content,*_: 'network unstable' if content=='gateway-error' else None))
    return calls


def test_transport_failures_continue_other_sources_then_retry_tail(tmp_path, monkeypatch):
    api, source = setup_run(tmp_path)
    attempts = {}
    def fail(stage, inputs, calls):
        rid = inputs[0]['record_id']
        attempts[(stage,rid)] = attempts.get((stage,rid),0)+1
        if stage=='normalize' and int(rid.split(':')[1]) < 12 and attempts[(stage,rid)]<=3:
            return 'qwen-3.8-max','gateway-error'
        return 'Qwen-3.8-Max',None
    calls = mock_client(monkeypatch, fail)
    monkeypatch.setattr(api.time, 'sleep', lambda _:None)
    api.work(tmp_path)
    assert not (tmp_path/'fatal.json').exists()
    assert api.read(tmp_path/'status.json')['translated_records']==14
    assert calls.index(('normalize','source:12')) < [i for i,c in enumerate(calls) if c==('normalize','source:0')][3]
    for n in range(6):
        value=api.read(tmp_path/'batches'/f'batch_{n:05d}.json')
        old=api.read(tmp_path/'transport_retry_history/epoch_1'/f'batch_{n:05d}.json')
        assert value['attempts'][:len(old['attempts'])]==old['attempts']
        assert value['transport_retry_epoch']==1 and value['status']=='completed'


def test_tail_retries_are_bounded_and_do_not_reopen_schema_or_exhausted(tmp_path, monkeypatch):
    api, source=setup_run(tmp_path,2)
    calls=mock_client(monkeypatch, lambda *args:('qwen-3.8-max','gateway-error'))
    monkeypatch.setattr(api.time,'sleep',lambda _:None)
    api.work(tmp_path)
    assert len(calls)==6
    assert api.read(tmp_path/'batches/batch_00000.json')['status']=='failed'
    assert api.tail_retry_batches(tmp_path,source,api.read(tmp_path/'manifest.json'))==[]
    value=api.read(tmp_path/'batches/batch_00000.json');value.pop('recovery_history');value.pop('transport_retry_epoch')
    value['attempts']=[{'error_type':'ValueError','validation_code':'invalid_translation_schema'}]
    api.atomic(tmp_path/'batches/batch_00000.json',value)
    assert api.tail_retry_batches(tmp_path,source,api.read(tmp_path/'manifest.json'))==[]


@pytest.mark.parametrize('kind',['identity','auth','unknown'])
def test_non_transport_fatals_still_stop_all_work(tmp_path,monkeypatch,kind):
    api,_=setup_run(tmp_path)
    class Access(Exception): status_code=401
    def fail(*args):
        if kind=='auth':raise Access('unauthorized')
        if kind=='unknown':raise RuntimeError('unexpected broken internal operation')
        return 'wrong-model','{}'
    calls=mock_client(monkeypatch,fail)
    api.work(tmp_path)
    assert len(calls)==1
    assert api.read(tmp_path/'status.json')['state']=='blocked_api'
    assert api.read(tmp_path/'fatal.json')['error_code'] in {'response_model_identity_mismatch','api_access_or_model_unavailable','unexpected_runtime_error'}


def test_overlay_hash_drift_refuses_before_calls(tmp_path):
    api,_=setup_run(tmp_path)
    with Path(api.__file__).open('a') as f:f.write('\n# changed\n')
    with pytest.raises(ValueError,match='worker_runtime_binding_changed'):api.work(tmp_path)
    with pytest.raises(ValueError,match='worker_runtime_binding_changed'):worker_entrypoint(tmp_path,api.read(tmp_path/'manifest.json'))


def test_large_failure_queue_does_not_block_untouched_work_in_tail_mode(tmp_path):
    api,source=setup_run(tmp_path,6)
    for index,start in enumerate(range(0,6,2)):
        api.atomic(tmp_path/'batches'/f'batch_{index:05d}.json',{
            'input_sha256':api.digest(source[start:start+2]),'normalized':[],'translated':[],
            'status':'failed','attempts':[{'error_type':'APIConnectionError'}]})
    manifest=api.read(tmp_path/'manifest.json')
    with pytest.raises(ValueError,match='failed_batch_recovery_limit_exceeded'):inventory(tmp_path,api,manifest,1)
    selected,_=inventory(tmp_path,api,{**manifest,'defer_technical_retries_to_tail':True},1)
    assert len(selected)==1
    assert all(api.read(p)['status']=='failed' for p in (tmp_path/'batches').glob('*.json'))


def test_partial_probe_allows_transport_failure_but_not_bad_identity(tmp_path):
    import hashlib
    from datetime import datetime,timezone
    probe=tmp_path/'probe_relation_expansion_api.py';probe.write_text('probe')
    manifest={'runtime':str(tmp_path),'expected_response_model':'Qwen-3.8-Max','concurrency':8,'transport_probe_min_success_ratio':.75}
    reports=[]
    for c in [1,8]:
        folder=tmp_path/str(c);folder.mkdir()
        requests=[{'request_index':i,'status':'completed','response_model':'Qwen-3.8-Max','usage':{'total_tokens':100},'semantic_case_matches':4} for i in range(c)]
        if c==8:requests[-1]={'request_index':7,'status':'failed','error_type':'APIConnectionError'}
        original.p.write_rows(folder/f'requests_c{c}.jsonl',requests)
        report={'created_at':datetime.now(timezone.utc).isoformat(),'route_sha256':hashlib.sha256(b'route\nqwen-3.8-max').hexdigest(),'requested_model':'qwen-3.8-max','probe_source_sha256':original.p.sha(probe),'observed_response_model':'Qwen-3.8-Max','input_policy':'synthetic_static_cases_only','retries':0,'levels':[{'concurrency':c,'requests':c,'completed':c if c==1 else 7,'failed':0 if c==1 else 1,'peak_in_flight':c,'bounded_probe_passed':c==1}]}
        original.p.atomic(folder/'report.json',report);reports.append(folder/'report.json')
    health(reports,manifest,'route','qwen-3.8-max')
    requests[0]['response_model']='wrong';original.p.write_rows(folder/'requests_c8.jsonl',requests)
    with pytest.raises(ValueError,match='probe_request_evidence_failed'):health(reports,manifest,'route','qwen-3.8-max')
