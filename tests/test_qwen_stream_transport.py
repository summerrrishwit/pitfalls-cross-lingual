import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from qwen_stream_transport import stream_completion
from build_zh_streaming_runtime import build_worker, build_probe


def chunk(text=None, finish=None, model='Qwen-3.8-Max', tokens=None):
    return NS(model=model, usage=None if tokens is None else NS(model_dump=lambda: {'total_tokens':tokens}),
              choices=[] if text is None and finish is None else [NS(index=0,delta=NS(content=text),finish_reason=finish)])


class Stream:
    def __init__(self, chunks): self.chunks=chunks;self.closed=False
    def __enter__(self):return iter(self.chunks)
    def __exit__(self,*args):self.closed=True


def client(chunks):
    stream=Stream(chunks);requests=[]
    def create(**kwargs):requests.append(kwargs);return stream
    return NS(chat=NS(completions=NS(create=create))),stream,requests


def test_complete_response_preserves_payload_and_usage():
    c,s,requests=client([chunk('{"ok":'),chunk('true}',finish='stop'),chunk(tokens=8)])
    kwargs={'model':'qwen-3.8-max','messages':[{'role':'user','content':'original'}], 'max_tokens':8192,
            'response_format':{'type':'json_object'},'extra_body':{'enable_thinking':False}}
    r=stream_completion(c,expected_model='Qwen-3.8-Max',**kwargs)
    assert r.choices[0].message.content=='{"ok":true}' and r.usage.model_dump()=={'total_tokens':8}
    assert requests==[{**kwargs,'stream':True,'stream_options':{'include_usage':True}}]
    assert s.closed and r.streaming_diagnostics['finish_reason']=='stop'


@pytest.mark.parametrize('chunks',[
    [chunk('{"ok":true}')],
    [chunk('{"ok":true}',finish='stop')],
    [chunk(tokens=8)],
    [chunk('{"ok":true}',finish='stop',model=None),chunk(tokens=8,model=None)],
])
def test_missing_end_usage_or_identity_is_never_a_candidate(chunks):
    from openai import APIConnectionError
    c,s,_=client(chunks)
    with pytest.raises(APIConnectionError):stream_completion(c,expected_model='Qwen-3.8-Max')
    assert s.closed


def test_identity_drift_in_any_chunk_fails_closed():
    c,s,_=client([chunk('{'),chunk('}',finish='stop',model='wrong-model'),chunk(tokens=8)])
    with pytest.raises(RuntimeError,match='response_model_identity_mismatch'):
        stream_completion(c,expected_model='Qwen-3.8-Max')
    assert s.closed


def test_zero_usage_and_truncated_json_are_not_accepted():
    c,_,_=client([chunk('"gateway failure"',finish='stop',model='qwen-3.8-max'),chunk(tokens=0,model='qwen-3.8-max')])
    with pytest.raises(RuntimeError,match='gateway_zero_usage_error_response'):
        stream_completion(c,expected_model='Qwen-3.8-Max')
    c,_,_=client([chunk('{"ok":',finish='length'),chunk(tokens=8)])
    with pytest.raises(ValueError,match='stream_output_truncated'):
        stream_completion(c,expected_model='Qwen-3.8-Max')


def test_midstream_connection_failure_preserved():
    from openai import APIConnectionError
    import httpx
    def interrupted():
        yield chunk('{')
        raise APIConnectionError(request=httpx.Request('POST','https://redacted.invalid'))
    c,s,_=client(interrupted())
    with pytest.raises(APIConnectionError):stream_completion(c,expected_model='Qwen-3.8-Max')
    assert s.closed


@pytest.mark.parametrize('kind,status',[('server_error',503),('authentication_error',401),('rate_limit_error',429)])
def test_sse_error_keeps_existing_transport_and_access_policy(kind,status):
    from openai import APIStatusError
    c,s,_=client([NS(error={'type':kind},model=None,usage=None,choices=[])])
    with pytest.raises(APIStatusError) as error:
        stream_completion(c,expected_model='Qwen-3.8-Max')
    assert error.value.status_code==status and s.closed


def test_unknown_sse_error_fails_closed():
    c,s,_=client([NS(error={'type':'unrecognized'},model=None,usage=None,choices=[])])
    with pytest.raises(RuntimeError,match='unexpected_stream_error'):
        stream_completion(c,expected_model='Qwen-3.8-Max')


def test_overlay_preserves_frozen_prompts_schema_and_retry_loop():
    import ast
    root=Path(__file__).resolve().parents[1]
    parent=root/'analysis_output/zh_normalize_translate_20261023_v1/runtime_transport_continue_v1'
    before=(parent/'run_zh_candidate_pipeline.py').read_text();after=build_worker(before)
    for name in ['NORMALIZE','TRANSLATE','REVIEW']:
        def assignment(source):
            return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert assignment(before)==assignment(after)
    for name in ['normalize','translate','validate_review','tail_retry_batches','finalize']:
        def source_function(source):
            return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))
        assert source_function(before)==source_function(after)
    assert 'for attempt in range(3)' in after and 'max_tokens=8192' in after
    compile(build_probe((parent/'probe_relation_expansion_api.py').read_text()),'<probe>','exec')


def test_recovery_probe_requires_the_bound_stream_helper(tmp_path):
    import hashlib,json
    from datetime import datetime,timezone
    from recover_zh_candidate_pipeline import health,sha
    frozen=tmp_path/'original';frozen.mkdir();active=tmp_path/'stream';active.mkdir()
    (frozen/'probe_relation_expansion_api.py').write_text('original')
    probe=active/'probe_relation_expansion_api.py';probe.write_text('stream')
    helper=active/'qwen_stream_transport.py';helper.write_text('assembler')
    request={'request_index':0,'status':'completed','response_model':'Qwen-3.8-Max',
             'usage':{'total_tokens':10},'semantic_case_matches':4}
    (tmp_path/'requests_c1.jsonl').write_text(json.dumps(request)+'\n')
    report={'created_at':datetime.now(timezone.utc).isoformat(),'requested_model':'qwen-3.8-max',
            'route_sha256':hashlib.sha256(b'route\nqwen-3.8-max').hexdigest(),
            'probe_source_sha256':sha(probe),'transport_helper_sha256':sha(helper),
            'observed_response_model':'Qwen-3.8-Max','input_policy':'synthetic_static_cases_only','retries':0,
            'levels':[{'concurrency':1,'requests':1,'completed':1,'failed':0,'peak_in_flight':1,'bounded_probe_passed':True}]}
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    manifest={'runtime':str(frozen),'transport_probe_runtime':str(active),'transport_helper_sha256':sha(helper),
              'expected_response_model':'Qwen-3.8-Max','concurrency':1}
    health([path],manifest,'route','qwen-3.8-max')
    report['transport_helper_sha256']='different';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='probe_identity_or_route_mismatch'):
        health([path],manifest,'route','qwen-3.8-max')
