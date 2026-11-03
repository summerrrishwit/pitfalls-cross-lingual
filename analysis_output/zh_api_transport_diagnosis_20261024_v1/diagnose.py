"""Bounded same-route synthetic diagnostic; never prints credentials or URLs."""
import hashlib
import json
import os
import re
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
values = dotenv_values('/Users/qiqing/learn_resource/mutil-language/pitfalls-cross-lingual/utils/.env')
key, base, model = [str(values[x]).strip() for x in ['AILYUN_API_KEY', 'AILYUN_BASE_URL', 'AILYUN_MODEL']]
parts = urlsplit(base)
def clean(text):
    text = str(text)
    for secret in [key, base, parts.hostname]:
        if secret:
            text = text.replace(secret, '[redacted]')
    return re.sub(r'https?://\S+|sk-[\w-]+', '[redacted]', text)[:1200]

report = {'at': datetime.now(timezone.utc).isoformat(), 'requested_model': model,
          'route_sha256': hashlib.sha256(json.dumps([base.rstrip('/'), model], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
          'synthetic_only': True, 'human_gold': False,
          'proxy_environment_present': {k: bool(os.getenv(k)) for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy']}}
started = time.monotonic()
try:
    addresses = socket.getaddrinfo(parts.hostname, parts.port or 443, type=socket.SOCK_STREAM)
    report['dns'] = {'ok': True, 'address_count': len(addresses), 'seconds': round(time.monotonic()-started, 3)}
    started = time.monotonic()
    with socket.create_connection((parts.hostname, parts.port or 443), timeout=10) as raw:
        with ssl.create_default_context().wrap_socket(raw, server_hostname=parts.hostname) as tls:
            report['direct_tls'] = {'ok': True, 'version': tls.version(), 'seconds': round(time.monotonic()-started,3)}
except Exception as error:
    report['connection_error'] = {'type': type(error).__name__, 'message': clean(error)}
payload = {'model': model, 'messages': [{'role':'user','content':'Return only the JSON object {"ok":true}.'}],
           'temperature': 0, 'max_tokens': 64, 'response_format': {'type':'json_object'}, 'enable_thinking': False}
started = time.monotonic()
try:
    with httpx.Client(timeout=httpx.Timeout(35, connect=10)) as client:
        response = client.post(base.rstrip('/')+'/chat/completions', headers={'Authorization':'Bearer '+key}, json=payload)
        report['minimal_request'] = {'status': response.status_code, 'seconds': round(time.monotonic()-started,3),
              'server': clean(response.headers.get('server','')), 'content_type': response.headers.get('content-type')}
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code == 200 and isinstance(body,dict) and body.get('choices'):
            content = body['choices'][0].get('message',{}).get('content') or ''
            usage = body.get('usage') or {}
            report['minimal_request'].update(response_model=clean(body.get('model')), usage=usage,
                content=clean(content), identity_matches=body.get('model')=='Qwen-3.8-Max',
                valid=body.get('model')=='Qwen-3.8-Max' and usage.get('total_tokens',0)>0 and json.loads(content)=={'ok':True})
        else:
            report['minimal_request']['error_body'] = clean(response.text)
except Exception as error:
    report['minimal_request'] = {'error_type':type(error).__name__, 'message':clean(error), 'seconds':round(time.monotonic()-started,3)}
(ROOT/'connectivity_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
