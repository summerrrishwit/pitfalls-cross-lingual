#!/usr/bin/env python3
"""Run bounded, role-specific structured probes for the Chinese MVP model stack."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from openai import OpenAI

from probe_configured_models import (
    PROFILES,
    anthropic_url,
    api_root,
    safe_error,
    safe_http_error,
)


CASES = [
    {
        "case_id": "translation_qwen38max",
        "profile": "aliyun",
        "model": "qwen3.8-max",
        "role": "translation",
        "expected_keys": ["translation", "preserves_answer_suffix"],
        "prompt": (
            "Translate the factual-recall prompt into natural Simplified Chinese while keeping the answer "
            "as the missing suffix. Return JSON only with keys translation and preserves_answer_suffix. "
            "Prompt: The main religion in Thailand is"
        ),
    },
    {
        "case_id": "translation_review_gpt55",
        "profile": "openai",
        "model": "gpt-5.5",
        "role": "translation_review",
        "expected_keys": ["decision", "issues"],
        "prompt": (
            "Review whether the Chinese prompt is a faithful, natural answer-suffix translation. "
            "English: The main religion in Thailand is. Chinese: 泰国的主要宗教是. Answer: Buddhism/佛教. "
            "Return JSON only: {\"decision\":\"accept|reject\",\"issues\":[]}."
        ),
    },
    {
        "case_id": "perturbation_generation_gpt56sol",
        "profile": "openai",
        "model": "gpt-5.6-sol",
        "role": "perturbation_generation",
        "expected_keys": ["english_perturbation", "chinese_perturbation"],
        "prompt": (
            "Create one subtle bilingual contextual perturbation that raises the salience of distractor Islam "
            "without stating anything false and without changing the answer Buddhism. Base prompt: The main "
            "religion in Thailand is / 泰国的主要宗教是. Return JSON only with keys english_perturbation and "
            "chinese_perturbation."
        ),
    },
    {
        "case_id": "perturbation_validation_qwen38max",
        "profile": "aliyun",
        "model": "qwen3.8-max",
        "role": "perturbation_validation",
        "expected_keys": ["decision", "checks"],
        "prompt": (
            "Audit this bilingual factual perturbation. Fact: Thailand-main religion-Buddhism. Distractor: Islam. "
            "English: Despite the presence of Islam, the main religion in Thailand is. "
            "Chinese: 尽管泰国也有穆斯林，泰国的主要宗教是. Return JSON only with decision accept|reject and "
            "checks containing booleans ground_truth_preserved, no_explicit_falsehood, relation_preserved, "
            "bilingual_equivalent, naturalness."
        ),
    },
    {
        "case_id": "simulation_qwen37plus",
        "profile": "aliyun",
        "model": "qwen3.7-plus",
        "role": "simulation",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "simulation_deepseek_v4_flash",
        "profile": "deepseek",
        "model": "bailian/deepseek-v4-flash-0731",
        "role": "simulation",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "simulation_hy4_preview",
        "profile": "test",
        "model": "hy4-preview",
        "role": "simulation",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "simulation_gemini37_flash",
        "profile": "test",
        "model": "gemini-3.7-flash",
        "role": "simulation",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "simulation_qwen36_27b",
        "profile": "aliyun",
        "model": "qwen3.6-27b",
        "role": "simulation",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "simulation_deepseek_v32",
        "profile": "deepseek",
        "model": "bailian/deepseek-v3.2",
        "role": "simulation",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "simulation_gpt5mini",
        "profile": "openai",
        "model": "gpt-5-mini",
        "role": "simulation",
        "expected_keys": ["choice"],
        "max_output_tokens": 256,
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
    {
        "case_id": "holdout_claude_opus48",
        "profile": "claude",
        "model": "claude-opus-4-8",
        "role": "holdout",
        "expected_keys": ["choice"],
        "prompt": (
            "Answer the multiple-choice question. What is the main religion in Thailand? "
            "A. Buddhism B. Islam C. Christianity D. Hinduism. Return JSON only: {\"choice\":\"A|B|C|D\"}."
        ),
    },
]

MINIMAL_CASES = [
    {
        "case_id": f"minimal_{profile}_{model.replace('/', '_')}",
        "profile": profile,
        "model": model,
        "role": "minimal",
        "expected_keys": ["ok"],
        "max_output_tokens": 256,
        "prompt": 'Return JSON only: {"ok":true}',
    }
    for profile, model in (
        ("aliyun", "qwen3.8-max"),
        ("openai", "gpt-5.5"),
        ("openai", "gpt-5.6-sol"),
        ("aliyun", "qwen3.7-plus"),
        ("deepseek", "bailian/deepseek-v4-flash-0731"),
        ("test", "hy4-preview"),
        ("test", "gemini-3.7-flash"),
        ("aliyun", "qwen3.6-27b"),
        ("deepseek", "bailian/deepseek-v3.2"),
        ("openai", "gpt-5-mini"),
        ("claude", "claude-opus-4-8"),
    )
]


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def result_from_text(case: dict[str, Any], text: str, response_model: str | None) -> dict[str, Any]:
    parsed = parse_json_object(text)
    expected = case["expected_keys"]
    return {
        "case_id": case["case_id"],
        "profile": case["profile"],
        "model": case["model"],
        "role": case["role"],
        "status": "ok",
        "response_has_text": bool(text.strip()),
        "json_object": parsed is not None,
        "required_keys_present": parsed is not None and all(key in parsed for key in expected),
        "returned_keys": sorted(parsed) if parsed is not None else [],
        "response_model": response_model,
    }


def openai_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    return (getattr(message, "content", None) or "").strip()


def run_openai_case(case: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    spec = PROFILES[case["profile"]]
    api_key = str(values.get(spec["api_key_env"]) or "").strip()
    base_url = str(values.get(spec["base_url_env"]) or "").strip()
    if not api_key or not base_url:
        return {**{key: case[key] for key in ("case_id", "profile", "model", "role")}, "status": "not_configured"}
    client = OpenAI(api_key=api_key, base_url=api_root(base_url), timeout=180.0, max_retries=0)
    request: dict[str, Any] = {
        "model": case["model"],
        "messages": [{"role": "user", "content": case["prompt"]}],
    }
    if case["model"].startswith("gpt-"):
        request["max_completion_tokens"] = case.get("max_output_tokens", 768)
        request["reasoning_effort"] = "low"
    else:
        request["max_tokens"] = case.get("max_output_tokens", 512)
    if case["model"].startswith("qwen") or "deepseek" in case["model"]:
        request["extra_body"] = {"enable_thinking": False}
    try:
        response = client.chat.completions.create(**request)
        return result_from_text(case, openai_text(response), getattr(response, "model", None))
    except Exception as error:
        return {
            **{key: case[key] for key in ("case_id", "profile", "model", "role")},
            "status": "failed",
            **safe_error(error),
        }


def run_anthropic_case(case: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    spec = PROFILES[case["profile"]]
    api_key = str(values.get(spec["api_key_env"]) or "").strip()
    base_url = str(values.get(spec["base_url_env"]) or "").strip()
    if not api_key or not base_url:
        return {**{key: case[key] for key in ("case_id", "profile", "model", "role")}, "status": "not_configured"}
    headers = {
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                anthropic_url(base_url, "messages"),
                headers=headers,
                json={
                    "model": case["model"],
                    "max_tokens": case.get("max_output_tokens", 512),
                    "messages": [{"role": "user", "content": case["prompt"]}],
                },
            )
        if not response.is_success:
            return {
                **{key: case[key] for key in ("case_id", "profile", "model", "role")},
                "status": "failed",
                **safe_http_error(response),
            }
        body = response.json()
        text = "".join(
            str(item.get("text", ""))
            for item in body.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
        return result_from_text(case, text, body.get("model"))
    except Exception as error:
        return {
            **{key: case[key] for key in ("case_id", "profile", "model", "role")},
            "status": "failed",
            **safe_error(error),
        }


def run_case(case: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    if PROFILES[case["profile"]]["protocol"] == "anthropic":
        return run_anthropic_case(case, values)
    return run_openai_case(case, values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite",
        choices=("representative", "minimal", "all"),
        default="representative",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run only the named case; may be supplied more than once.",
    )
    args = parser.parse_args()
    values = dotenv_values(args.env_file)

    cases = CASES
    if args.suite == "minimal":
        cases = MINIMAL_CASES
    elif args.suite == "all":
        cases = MINIMAL_CASES + CASES
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in cases}
        if missing:
            parser.error(f"unknown case id(s): {', '.join(sorted(missing))}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_case, case, values): case["case_id"] for case in cases}
        for future in as_completed(futures):
            results.append(future.result())
    order = {case["case_id"]: index for index, case in enumerate(cases)}
    results.sort(key=lambda item: order[item["case_id"]])

    report = {
        "schema_version": "model-role-api-probe-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sense_nova_excluded": True,
        "secrets_or_endpoints_included": False,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
