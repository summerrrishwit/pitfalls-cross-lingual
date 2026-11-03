import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sanitized_api_error import api_error_diagnostics
from build_zh_error_diagnostics_runtime import build_worker


def test_redacts_credentials_and_route_from_body_and_headers():
    error=SimpleNamespace(status_code=403,body={'error':{'type':'permission_error','message':'blocked secret-value at https://private.invalid/v1 (private.invalid) sk-other-secret'}},
        response=SimpleNamespace(headers={'server':'nginx','x-request-id':'req1','authorization':'secret-value'}))
    value=api_error_diagnostics(error,'secret-value','https://private.invalid/v1');text=json.dumps(value)
    assert 'secret-value' not in text and 'private.invalid' not in text and 'sk-other-secret' not in text
    assert value['provider_error_type']=='permission_error' and value['provider_error_headers']['x-request-id']=='req1'
    assert 'authorization' not in text


def test_long_html_body_is_bounded_and_hashed():
    value=api_error_diagnostics(SimpleNamespace(status_code=403,body='x'*5000,response=None),'key','https://private.invalid')
    assert len(value['provider_error_body_redacted'])==2400
    assert value['provider_error_body_truncated'] and len(value['provider_error_body_sha256'])==64


def test_local_non_http_exception_does_not_log_exception_string():
    assert api_error_diagnostics(ValueError('secret-value'),'secret-value','https://private.invalid')=={}


def test_missing_body_and_response_stays_empty():
    assert api_error_diagnostics(SimpleNamespace(status_code=403,body=None,response=None),'key','https://private.invalid')=={}


def test_response_text_fallback_is_redacted():
    value=api_error_diagnostics(SimpleNamespace(status_code=403,body=None,response=SimpleNamespace(text='key-denied',headers={})),
        'key-denied','https://private.invalid')
    assert value['provider_error_body_redacted']=='[redacted]'


def test_generated_runtime_changes_only_logging_and_import():
    root=Path(__file__).resolve().parents[1]
    source=(root/'analysis_output/zh_normalize_translate_20261023_v1/runtime_transport_stream_v1/run_zh_candidate_pipeline.py').read_text()
    generated=build_worker(source)
    expected=source.replace('    from qwen_stream_transport import stream_completion\n',
        '    from qwen_stream_transport import stream_completion\n    from sanitized_api_error import api_error_diagnostics\n',1)
    expected=expected.replace("                detail.update(status='failed',**safe_error(error),latency_seconds=round(time.monotonic()-start,3))\n",
        "                detail.update(status='failed',**safe_error(error),latency_seconds=round(time.monotonic()-start,3))\n                detail.update(api_error_diagnostics(error,key,base))\n",1)
    assert generated==expected
    for name in ['NORMALIZE','TRANSLATE','REVIEW']:
        def get(text):return next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert get(source)==get(generated)
