import ast
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_zh_gateway403_runtime import HELPER,build
ns={};exec(HELPER,ns);classify=ns['transient_gateway403']
SIGNATURE={'http_status':403,'provider_error_type':'server_error','provider_error_message':'网络连接不稳定，请稍后重试。','provider_error_body_sha256':'250cd3b0b0faeb04c78fa83c7a231e1b6744205930c47b248a837848abc77867'}
def test_only_observed_provider_error_is_transient():
    assert classify(SIGNATURE)
    for key in SIGNATURE:
        d=dict(SIGNATURE);d.pop(key);assert not classify(d)
@pytest.mark.parametrize('change',[{'http_status':401},{'provider_error_type':'permission_denied'},{'provider_error_message':'权限不足'},{'provider_error_message':'余额不足'},{'provider_error_message':'invalid_api_key'},{'provider_error_body_sha256':'changed'}])
def test_access_denial_or_unrecognized_body_never_bypassed(change):
    assert not classify({**SIGNATURE,**change})
def test_overlay_preserves_generation_review_and_batch_logic():
    path=Path(__file__).resolve().parents[1]/'analysis_output/zh_normalize_translate_20261023_v1/runtime_transport_diagnostics_v1/run_zh_candidate_pipeline.py'
    before=path.read_text();after=build(before)
    def defs(s):return {n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=defs(before),defs(after)
    for name in a.keys()-{'work','transport_failure'}:assert a[name]==b[name]
    assert 'range(3)' in after and 'if failures>=16:' in after
    with pytest.raises(ValueError):build(after)
