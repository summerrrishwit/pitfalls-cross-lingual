#!/usr/bin/env python3
"""Safely probe non-SenseNova model profiles declared in the project .env.

The output intentionally excludes API keys and endpoint URLs. A successful
catalog lookup is discovery evidence; a successful minimal generation is the
basic compatibility check for the declared model.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import dotenv_values
from openai import OpenAI


PROFILES = {
    "openai": {
        "protocol": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
    },
    "aliyun": {
        "protocol": "openai",
        "api_key_env": "AILYUN_API_KEY",
        "base_url_env": "AILYUN_BASE_URL",
        "model_env": "AILYUN_MODEL",
    },
    "test": {
        "protocol": "openai",
        "api_key_env": "TEST_API_KEY",
        "base_url_env": "TEST_BASE_URL",
        "model_env": "TEST_MODEL",
    },
    "claude": {
        "protocol": "anthropic",
        "api_key_env": "CC_API_KEY",
        "base_url_env": "CC_BASE_URL",
        "model_env": "CC_MODEL_ID",
    },
    "deepseek": {
        "protocol": "openai",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "model_env": "DEEPSEEK_MODEL",
    },
}


def safe_error(error: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {"error_type": type(error).__name__}
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        result["http_status"] = status_code
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error") if isinstance(body.get("error"), dict) else body
        for field in ("type", "code", "param"):
            value = nested.get(field)
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[f"provider_{field}"] = value
    return result


def api_root(base_url: str) -> str:
    """Return a URL rooted at /v1 without exposing it in reports."""
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/messages", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def response_has_text(response: Any) -> bool:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    message = getattr(choices[0], "message", None)
    if message is None:
        return False
    return bool(
        (getattr(message, "content", None) or "").strip()
        or (getattr(message, "reasoning_content", None) or "").strip()
    )


def probe_openai(profile: str, api_key: str, base_url: str, model: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile": profile,
        "protocol": "openai_compatible",
        "declared_model": model,
    }
    client = OpenAI(
        api_key=api_key,
        base_url=api_root(base_url),
        timeout=45.0,
        max_retries=0,
    )
    try:
        page = client.models.list()
        models = sorted({item.id for item in page.data if isinstance(item.id, str)})
        result["catalog"] = {
            "status": "ok",
            "model_count": len(models),
            "declared_model_listed": model in models,
            "models": models,
        }
    except Exception as error:  # provider-specific error classes vary
        result["catalog"] = {"status": "failed", **safe_error(error)}

    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    }
    if model.startswith("gpt-"):
        request["max_completion_tokens"] = 16
    else:
        request["max_tokens"] = 16
    if model.startswith("qwen"):
        request["extra_body"] = {"enable_thinking": False}
    try:
        response = client.chat.completions.create(**request)
        result["minimal_generation"] = {
            "status": "ok",
            "response_has_text": response_has_text(response),
            "response_model": getattr(response, "model", None),
        }
    except Exception as error:
        result["minimal_generation"] = {"status": "failed", **safe_error(error)}
    return result


def anthropic_url(base_url: str, resource: str) -> str:
    return f"{api_root(base_url).rstrip('/')}/{resource.lstrip('/')}"


def safe_http_error(response: httpx.Response) -> dict[str, Any]:
    result: dict[str, Any] = {"http_status": response.status_code}
    try:
        body = response.json()
    except ValueError:
        return result
    if isinstance(body, dict):
        nested = body.get("error") if isinstance(body.get("error"), dict) else body
        for field in ("type", "code", "param"):
            value = nested.get(field)
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[f"provider_{field}"] = value
    return result


def probe_anthropic(profile: str, api_key: str, base_url: str, model: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "profile": profile,
        "protocol": "anthropic",
        "declared_model": model,
    }
    headers = {
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=45.0) as client:
        try:
            response = client.get(anthropic_url(base_url, "models"), headers=headers)
            if response.is_success:
                body = response.json()
                data = body.get("data", []) if isinstance(body, dict) else []
                models = sorted(
                    {
                        item["id"]
                        for item in data
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    }
                )
                result["catalog"] = {
                    "status": "ok",
                    "model_count": len(models),
                    "declared_model_listed": model in models,
                    "models": models,
                }
            else:
                result["catalog"] = {"status": "failed", **safe_http_error(response)}
        except Exception as error:
            result["catalog"] = {"status": "failed", **safe_error(error)}

        try:
            response = client.post(
                anthropic_url(base_url, "messages"),
                headers=headers,
                json={
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                },
            )
            if response.is_success:
                body = response.json()
                content = body.get("content", []) if isinstance(body, dict) else []
                has_text = any(
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and bool(str(item.get("text", "")).strip())
                    for item in content
                )
                result["minimal_generation"] = {
                    "status": "ok",
                    "response_has_text": has_text,
                    "response_model": body.get("model") if isinstance(body, dict) else None,
                }
            else:
                result["minimal_generation"] = {
                    "status": "failed",
                    **safe_http_error(response),
                }
        except Exception as error:
            result["minimal_generation"] = {"status": "failed", **safe_error(error)}
    return result


def probe_profile(name: str, spec: dict[str, str], values: dict[str, Any]) -> dict[str, Any]:
    api_key = str(values.get(spec["api_key_env"]) or "").strip()
    base_url = str(values.get(spec["base_url_env"]) or "").strip()
    model = str(values.get(spec["model_env"]) or "").strip()
    missing = [
        env_name
        for env_name, value in (
            (spec["api_key_env"], api_key),
            (spec["base_url_env"], base_url),
            (spec["model_env"], model),
        )
        if not value
    ]
    if missing:
        return {
            "profile": name,
            "protocol": spec["protocol"],
            "status": "not_configured",
            "missing_env": missing,
        }
    if spec["protocol"] == "anthropic":
        return probe_anthropic(name, api_key, base_url, model)
    return probe_openai(name, api_key, base_url, model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    values = dotenv_values(args.env_file)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(PROFILES)) as executor:
        futures = {
            executor.submit(probe_profile, name, spec, values): name
            for name, spec in PROFILES.items()
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["profile"])

    report = {
        "schema_version": "configured-model-api-probe-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sense_nova_excluded": True,
        "secrets_or_endpoints_included": False,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "sense_nova_excluded": True,
        "profiles": [
            {
                "profile": item["profile"],
                "declared_model": item.get("declared_model"),
                "catalog_status": item.get("catalog", {}).get("status"),
                "catalog_model_count": item.get("catalog", {}).get("model_count"),
                "declared_model_listed": item.get("catalog", {}).get("declared_model_listed"),
                "minimal_generation": item.get("minimal_generation"),
            }
            for item in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
