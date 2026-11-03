"""Versioned transport patch for the observed misclassified gateway403."""
from pathlib import Path
import shutil

HELPER = '''def transient_gateway403(detail):
    # Narrow provider signature only. Unknown403 and real access denial stay fatal.
    return (detail.get('http_status') == 403
            and detail.get('provider_error_type') == 'server_error'
            and detail.get('provider_error_message', '').strip() == '网络连接不稳定，请稍后重试。'
            and detail.get('provider_error_body_sha256') == '250cd3b0b0faeb04c78fa83c7a231e1b6744205930c47b248a837848abc77867')


'''

def build(source):
    changes = [
        ('def transport_failure(detail):', HELPER + 'def transport_failure(detail):'),
        ("    return (detail.get('error_code') == 'gateway_zero_usage_error_response'", "    return (transient_gateway403(detail) or detail.get('error_code') == 'gateway_zero_usage_error_response'"),
        ("                if status in {401,403,404}:fatal('api_access_or_model_unavailable',detail)", "                gateway403=transient_gateway403(detail)\n                if status in {401,403,404} and not gateway403:fatal('api_access_or_model_unavailable',detail)"),
        ('                technical=isinstance(error,RuntimeError)', '                technical=gateway403 or isinstance(error,RuntimeError)'),
    ]
    for old,new in changes:
        if source.count(old)!=1: raise ValueError('unexpected_parent_source_shape')
        source=source.replace(old,new,1)
    compile(source,'<gateway403-overlay>','exec')
    return source

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('target',type=Path);a=p.parse_args()
    a.target.mkdir(exist_ok=False)
    for f in a.parent.glob('*.py'):shutil.copy2(f,a.target/f.name)
    (a.target/'run_zh_candidate_pipeline.py').write_text(build((a.parent/'run_zh_candidate_pipeline.py').read_text()))
    shutil.copy2(__file__,a.target/Path(__file__).name)
