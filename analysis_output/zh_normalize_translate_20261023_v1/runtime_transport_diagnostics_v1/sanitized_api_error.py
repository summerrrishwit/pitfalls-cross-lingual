"""Sanitized provider diagnostics without changing request/retry decisions."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit


def api_error_diagnostics(error: Exception, key: str, base: str) -> dict[str, Any]:
    if not isinstance(getattr(error, 'status_code', None), int):
        return {}
    secrets = [key, base, urlsplit(base).hostname or '']

    def clean(value: Any) -> str:
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, '[redacted]')
        return re.sub(r'https?://\S+|sk-[\w-]+', '[redacted]', text, flags=re.IGNORECASE)

    body = getattr(error, 'body', None)
    response = getattr(error, 'response', None)
    if body is None and response is not None:
        try:
            body = response.text
        except RuntimeError:
            body = None
    out: dict[str, Any] = {}
    if body is not None:
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True) if isinstance(body, (dict, list)) else str(body)
        out['provider_error_body_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        sanitized = clean(raw)
        out['provider_error_body_redacted'] = sanitized[:2400]
        out['provider_error_body_truncated'] = len(sanitized) > 2400
        if isinstance(body, dict):
            details = body.get('error') if isinstance(body.get('error'), dict) else body
            for field in ['type', 'code', 'message']:
                if isinstance(details.get(field), (str, int, float, bool)):
                    out['provider_error_' + field] = clean(details[field])[:1200]
    if response is not None:
        headers = getattr(response, 'headers', {})
        out['provider_error_headers'] = {name: clean(headers[name])[:256] for name in
            ['server', 'content-type', 'cf-mitigated', 'cf-ray', 'retry-after', 'x-request-id', 'request-id']
            if name in headers}
    return out
