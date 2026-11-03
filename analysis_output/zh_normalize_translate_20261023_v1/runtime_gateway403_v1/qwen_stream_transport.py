"""Assemble complete Qwen SSE responses before any production validation."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any


def stream_completion(client: Any, *, expected_model: str, **request: Any) -> Any:
    """Keep request semantics; require stable identity, a final stop, and usage.

    Partial streams never become candidates. Connection failures use the existing
    bounded technical retry path; identity or unexpected stream errors fail closed.
    """
    import httpx
    from openai import APIConnectionError, APIStatusError

    def incomplete() -> None:
        raise APIConnectionError(message='incomplete_qwen_stream',
                                 request=httpx.Request('POST', 'https://redacted.invalid'))

    started = time.monotonic()
    content: list[str] = []
    models: set[str] = set()
    usage: dict[str, Any] | None = None
    finish_reason = None
    first_chunk_seconds = None
    count = 0
    with client.chat.completions.create(
            **request, stream=True, stream_options={'include_usage': True}) as stream:
        for chunk in stream:
            count += 1
            if first_chunk_seconds is None:
                first_chunk_seconds = round(time.monotonic() - started, 3)
            if time.monotonic() - started > 300:
                incomplete()
            error = getattr(chunk, 'error', None)
            if error:
                error_type = error.get('type') if isinstance(error, dict) else None
                code = error.get('code') if isinstance(error, dict) else None
                status = code if type(code) is int and 400 <= code <= 599 else {
                    'server_error': 503, 'overloaded_error': 503,
                    'rate_limit_error': 429, 'timeout_error': 504,
                    'connection_error': 502, 'authentication_error': 401,
                    'permission_error': 403}.get(error_type)
                if status is None:
                    raise RuntimeError('unexpected_stream_error')
                raise APIStatusError('qwen_stream_error', response=httpx.Response(status,
                    request=httpx.Request('POST', 'https://redacted.invalid')), body=None)
            label = getattr(chunk, 'model', None)
            if label:
                models.add(label)
            current_usage = getattr(chunk, 'usage', None)
            if current_usage is not None:
                usage = current_usage.model_dump()
            for choice in chunk.choices:
                if choice.index != 0:
                    raise RuntimeError('unexpected_stream_choices')
                delta = choice.delta.content
                if delta:
                    if finish_reason is not None:
                        raise RuntimeError('stream_content_after_finish')
                    content.append(delta)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
    # Some gateways return their own lowercase model label with a zero-usage
    # error wrapper. Like the original client, classify that as a failed request
    # before treating any label as evidence of a model identity.
    if usage is not None and usage.get('total_tokens', 0) <= 0:
        raise RuntimeError('gateway_zero_usage_error_response')
    if models and models != {expected_model}:
        raise RuntimeError('response_model_identity_mismatch')
    if finish_reason is None or models != {expected_model} or usage is None:
        incomplete()
    if finish_reason != 'stop':
        raise ValueError('stream_output_truncated' if finish_reason == 'length' else 'stream_nonstop_finish')
    return SimpleNamespace(model=expected_model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=''.join(content)))],
        usage=SimpleNamespace(model_dump=lambda: dict(usage)),
        streaming_diagnostics={'stream': True, 'stream_chunks': count,
                               'first_chunk_seconds': first_chunk_seconds,
                               'finish_reason': finish_reason})
