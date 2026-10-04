"""Resumable Chinese factual-perturbation experiment runtime.

The runtime keeps model calls auditable without persisting credentials or endpoint URLs.
Every stage has a single JSONL checkpoint writer and terminal records are reused on resume.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import dotenv_values
from openai import OpenAI

from factual_pitfalls import formal_admission
from factual_pitfalls.route_identity import normalize_model_protocol


SCHEMA_VERSION = "factual-perturbation-runtime-v1"
INPUT_BUNDLE_SCHEMA_VERSION = formal_admission.INPUT_BUNDLE_SCHEMA_VERSION
LEGACY_INPUT_BUNDLE_SCHEMA_VERSION = formal_admission.LEGACY_INPUT_BUNDLE_SCHEMA_VERSION
FORMAL_INPUT_BUNDLE_STATUSES = {"frozen", "reviewed_frozen"}
FORMAL_INPUT_EVIDENCE_TIERS = formal_admission.FORMAL_EVIDENCE_TIERS
INPUT_SPLIT_ASSIGNMENTS = {"development", "validation", "sealed"}
TRANSLATION_PROMPT_VERSION = "factual-translation-zh-v1"
TRANSLATION_REVIEW_PROMPT_VERSION = "factual-translation-review-zh-v1"
TRANSLATION_REPAIR_PROMPT_VERSION = "factual-translation-repair-zh-v2"
TRANSLATION_REPAIR_REVIEW_PROMPT_VERSION = "factual-translation-repair-review-zh-v1"
DISTRACTOR_REVIEW_PROMPT_VERSION = "factual-distractor-review-zh-v1"
DISTRACTOR_GENERATION_PROMPT_VERSION = "factual-distractor-generation-bilingual-v1"
PERTURBATION_PROMPT_VERSION = "factual-perturbation-bilingual-v1"
PERTURBATION_REVIEW_PROMPT_VERSION = "factual-perturbation-review-bilingual-v3"
PERTURBATION_SENSITIVITY_PROMPT_VERSION = "factual-perturbation-bilingual-sensitivity-v1"
PERTURBATION_SENSITIVITY_REVIEW_PROMPT_VERSION = "factual-perturbation-review-bilingual-sensitivity-v1"
EXPLICIT_FALSE_ASSERTION_PROMPT_VERSION = "factual-perturbation-explicit-false-assertion-v1"
EXPLICIT_FALSE_ASSERTION_REVIEW_PROMPT_VERSION = (
    "factual-perturbation-review-explicit-false-assertion-v1"
)
SIMULATION_PROMPT_VERSION = "factual-simulation-three-arm-v2"
SIMULATION_MULTI_PROMPT_VERSION = "factual-simulation-multi-option-three-arm-v1"
HOLDOUT_PROMPT_VERSION = "factual-holdout-two-choice-v1"
CANDIDATE_FREEZE_MANIFEST_SCHEMA_VERSION = (
    "factual-perturbation-candidate-freeze-manifest-v2"
)
CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION = (
    "factual-perturbation-frozen-candidate-v2"
)
STATIC_G0A_FREEZE_MANIFEST_SCHEMA_VERSION = "static-g0a-freeze-manifest-v1"
STATIC_G0A_MCQ_SCHEMA_VERSION = "static-g0a-dual-distractor-mcq-v1"
STATIC_PROXY_BEHAVIOR_INPUT_SCHEMA_VERSION = (
    "factual-perturbation-static-proxy-behavior-input-v1"
)
STATIC_PROXY_EXECUTION_MODE = "g0a-frozen-static-proxy-v1"
STATIC_SINGLE_QWEN3_EXECUTION_MODE = "g0a-frozen-static-qwen3-8b-behavior-v1"
STATIC_PROXY_REPRESENTATIVE_PROBE_RESULT_SCHEMA_VERSION = (
    "factual-perturbation-static-proxy-representative-probe-result-v2"
)
STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_SCHEMA_VERSION = (
    "factual-perturbation-static-proxy-representative-probe-manifest-v2"
)
STATIC_PROXY_ROUTE_RUNTIME_IDENTITY_SCHEMA_VERSION = (
    "factual-perturbation-static-proxy-route-runtime-identity-v1"
)
STATIC_PROXY_RESPONSE_MODEL_COMPATIBILITY_POLICY = (
    "frozen-provider-aware-response-model-v1"
)
STATIC_PROXY_REPRESENTATIVE_PROBE_SELECTION_POLICY = (
    "sha256-min-zh-targeted-static-stimulus-v1"
)
STATIC_PROXY_ROSTER_VERSION = "static-proxy-four-model-v9"
STATIC_SINGLE_QWEN3_ROSTER_VERSION = "static-qwen3-8b-single-model-v1"
STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME = (
    "proxy_representative_probe_results.jsonl"
)
STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME = (
    "proxy_representative_probe_manifest.json"
)
STATIC_PROXY_MODEL_ROSTER = (
    ("qwen3.6-27b", "aliyun"),
    ("gemini-3.7-flash", "test_gateway"),
    ("gemma3:12b", "ollama_local"),
    ("llama3.1:8b", "ollama_local"),
)
STATIC_SINGLE_QWEN3_MODEL_ROSTER = (("qwen3-8b", "qwen3_vllm"),)
STATIC_BEHAVIOR_EXECUTION_MODES = {
    STATIC_PROXY_EXECUTION_MODE,
    STATIC_SINGLE_QWEN3_EXECUTION_MODE,
}
STATIC_PROXY_EXPECTED_SPLIT_COUNTS = {
    "development": 96,
    "validation": 32,
    "sealed": 32,
}
STATIC_PROXY_STATIC_FACT_COUNT = 160
STATIC_PROXY_STIMULI_PER_FACT = 10
STATIC_PROXY_VARIANTS_PER_FACT = 2
STATIC_PROXY_EXPECTED_CALLS_PER_MODEL = (
    STATIC_PROXY_EXPECTED_SPLIT_COUNTS["development"]
    * STATIC_PROXY_STIMULI_PER_FACT
)
STATIC_PROXY_EXPECTED_CALL_COUNT = (
    STATIC_PROXY_EXPECTED_CALLS_PER_MODEL * len(STATIC_PROXY_MODEL_ROSTER)
)
STATIC_PROXY_MODEL_AGGREGATION = "none"
SIMULATION_STATIC_G0A_PROMPT_VERSION = "factual-simulation-static-g0a-mcq-v1"


def _static_behavior_contract(execution_mode: Any) -> Dict[str, Any]:
    """Return the immutable roster/count contract for a static behavior mode."""

    if execution_mode == STATIC_PROXY_EXECUTION_MODE:
        roster = STATIC_PROXY_MODEL_ROSTER
        roster_version = STATIC_PROXY_ROSTER_VERSION
        config_key = "static_proxy"
        roster_error = "static proxy behavior run requires the frozen four-model roster"
    elif execution_mode == STATIC_SINGLE_QWEN3_EXECUTION_MODE:
        roster = STATIC_SINGLE_QWEN3_MODEL_ROSTER
        roster_version = STATIC_SINGLE_QWEN3_ROSTER_VERSION
        config_key = "static_single_model"
        roster_error = (
            "static single-model behavior run requires the frozen Qwen3-8B roster"
        )
    else:
        raise ValueError("unsupported frozen static behavior execution mode")
    expected_calls_per_model = STATIC_PROXY_EXPECTED_CALLS_PER_MODEL
    return {
        "execution_mode": execution_mode,
        "config_key": config_key,
        "model_roster": roster,
        "roster_version": roster_version,
        "expected_model_count": len(roster),
        "expected_calls_per_model": expected_calls_per_model,
        "expected_simulation_call_count": expected_calls_per_model * len(roster),
        "model_aggregation": STATIC_PROXY_MODEL_AGGREGATION,
        "roster_error": roster_error,
        "required_max_workers": (
            8 if execution_mode == STATIC_SINGLE_QWEN3_EXECUTION_MODE else None
        ),
        "required_provider_max_concurrency": (
            {"qwen3_vllm": 8}
            if execution_mode == STATIC_SINGLE_QWEN3_EXECUTION_MODE
            else None
        ),
        "required_model_max_retries": (
            1 if execution_mode == STATIC_SINGLE_QWEN3_EXECUTION_MODE else None
        ),
    }

SIMULATION_LANGUAGES = ("en", "zh")
SIMULATION_VARIANTS = ("original", "neutral", "targeted")
SHARED_SIMULATION_VARIANTS = ("original", "neutral")
TRUTHFUL_SALIENCE_MANIPULATION_FAMILY = "truthful_distractor_salience_v1"
EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY = "explicit_false_assertion_v1"
SUPPORTED_MANIPULATION_FAMILIES = {
    TRUTHFUL_SALIENCE_MANIPULATION_FAMILY,
    EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _runtime_sha256() -> str:
    """Fingerprint both the runtime and its formal-admission policy module."""

    return sha256_value(
        {
            "perturbation.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "formal_admission.py": hashlib.sha256(
                Path(formal_admission.__file__).read_bytes()
            ).hexdigest(),
        }
    )


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_jsonl_snapshot(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse and fingerprint the exact same JSONL byte snapshot."""

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL snapshot: {path}: {exc}") from exc
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(rows),
    }


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("response_not_json_object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response_not_json_object")
    return value


def api_root(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/messages", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not path.endswith("/v1"):
        path = f"{path}/v1"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


class StaticProxyIdentityError(RuntimeError):
    """A fail-closed static-proxy route, runtime, or model identity failure."""


class ResponseModelMissing(StaticProxyIdentityError):
    """The provider response omitted the frozen model identity."""


class ResponseModelMismatch(StaticProxyIdentityError):
    """The provider response model differed from the frozen identity."""


def _normalized_endpoint_url(value: str) -> str:
    """Canonicalize an endpoint only for hashing; callers must never persist it."""

    parsed = urlparse(str(value).strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise StaticProxyIdentityError("static_proxy_endpoint_identity_invalid")
    if parsed.username is not None or parsed.password is not None:
        raise StaticProxyIdentityError(
            "static_proxy_endpoint_identity_must_not_embed_credentials"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise StaticProxyIdentityError(
            "static_proxy_endpoint_identity_invalid_port"
        ) from error
    hostname = parsed.hostname.lower()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or ""
    return urlunparse(
        (parsed.scheme.lower(), netloc, path, "", "", "")
    )


def _static_proxy_response_model_identity(
    requested_model: str,
    provider_profile: str,
    response_model: Optional[str],
) -> Dict[str, Any]:
    """Return a conservative, provider-aware response-model compatibility verdict."""

    requested = str(requested_model).strip()
    observed = str(response_model).strip() if response_model is not None else ""
    requested_folded = requested.casefold()
    observed_folded = observed.casefold()
    allowed = {requested_folded}
    # The DeepSeek route is intentionally requested through a Bailian namespace,
    # while that route may report the unqualified underlying model identifier.
    if provider_profile == "deepseek" and requested_folded.startswith("bailian/"):
        allowed.add(requested_folded.split("/", 1)[1])
    # Google-compatible endpoints sometimes add the documented `models/` resource
    # prefix. No other prefix/suffix or fuzzy match is accepted.
    if provider_profile == "test_gateway":
        allowed.add(f"models/{requested_folded}")
    compatible = bool(observed_folded) and observed_folded in allowed
    return {
        "policy": STATIC_PROXY_RESPONSE_MODEL_COMPATIBILITY_POLICY,
        "requested_model": requested,
        "provider_profile": provider_profile,
        "response_model": response_model,
        "compatible": compatible,
        "reason": (
            "compatible"
            if compatible
            else ("response_model_missing" if not observed_folded else "response_model_drift")
        ),
    }


def _require_static_proxy_response_model_identity(
    requested_model: str,
    provider_profile: str,
    response_model: Optional[str],
) -> Dict[str, Any]:
    evidence = _static_proxy_response_model_identity(
        requested_model, provider_profile, response_model
    )
    if evidence["compatible"] is not True:
        raise StaticProxyIdentityError(
            f"static_proxy_{evidence['reason']}:{provider_profile}:{requested_model}"
        )
    return evidence


def safe_error(error: Exception) -> Dict[str, Any]:
    result: Dict[str, Any] = {"error_type": type(error).__name__}
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


def validate_translation(value: Dict[str, Any], distractor_ids: Sequence[str]) -> None:
    for key in ("prompt_zh", "answer_zh", "answer_aliases_zh", "distractors"):
        if key not in value:
            raise ValueError(f"missing_{key}")
    if not isinstance(value["prompt_zh"], str) or not value["prompt_zh"].strip():
        raise ValueError("invalid_prompt_zh")
    if not isinstance(value["answer_zh"], str) or not value["answer_zh"].strip():
        raise ValueError("invalid_answer_zh")
    if not isinstance(value["answer_aliases_zh"], list) or not all(
        isinstance(item, str) for item in value["answer_aliases_zh"]
    ):
        raise ValueError("invalid_answer_aliases_zh")
    if not isinstance(value["distractors"], list):
        raise ValueError("invalid_distractors")
    found = {item.get("distractor_id") for item in value["distractors"] if isinstance(item, dict)}
    if found != set(distractor_ids):
        raise ValueError("distractor_translation_id_mismatch")
    if any(not str(item.get("text_zh") or "").strip() for item in value["distractors"]):
        raise ValueError("invalid_distractor_translation")


def validate_decision(value: Dict[str, Any], checks: Sequence[str] = ()) -> None:
    if value.get("decision") not in {"accept", "reject"}:
        raise ValueError("invalid_decision")
    if not isinstance(value.get("issues", []), list):
        raise ValueError("invalid_issues")
    if checks:
        actual = value.get("checks")
        if not isinstance(actual, dict) or any(not isinstance(actual.get(key), bool) for key in checks):
            raise ValueError("invalid_checks")


def validate_generated_distractors(value: Dict[str, Any], expected_count: int) -> None:
    distractors = value.get("distractors")
    if not isinstance(distractors, list) or len(distractors) != expected_count:
        raise ValueError("invalid_generated_distractor_count")
    pairs = []
    for distractor in distractors:
        if not isinstance(distractor, dict):
            raise ValueError("invalid_generated_distractor")
        text_en = distractor.get("text_en")
        text_zh = distractor.get("text_zh")
        if not isinstance(text_en, str) or not text_en.strip():
            raise ValueError("invalid_generated_distractor_en")
        if not isinstance(text_zh, str) or not text_zh.strip():
            raise ValueError("invalid_generated_distractor_zh")
        pairs.append((text_en.strip().casefold(), text_zh.strip()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate_generated_distractors")


def validate_perturbations(
    value: Dict[str, Any],
    expected_count: int,
    strength_levels: Sequence[str] = (),
    matched_neutral: bool = False,
) -> None:
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        raise ValueError("invalid_candidate_count")
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("invalid_candidate")
        required = ["english_context", "chinese_context"]
        if matched_neutral:
            required.extend(("neutral_english_context", "neutral_chinese_context"))
        if strength_levels:
            required.append("strength")
        for key in required:
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"invalid_{key}")
    if strength_levels:
        observed = [str(item.get("strength") or "") for item in candidates]
        if observed != list(strength_levels):
            raise ValueError("invalid_strength_levels")


def validate_choice(value: Dict[str, Any], allowed_choices: Sequence[str] = ("A", "B")) -> None:
    choice = str(value.get("choice") or "").strip().upper()
    allowed = set(allowed_choices)
    if choice and choice[0] in allowed:
        value["choice"] = choice[0]
    if value.get("choice") not in allowed:
        raise ValueError("invalid_choice")


@dataclass
class ModelResult:
    terminal_status: str
    parsed_response: Optional[Dict[str, Any]]
    raw_response: Optional[str]
    response_model: Optional[str]
    usage: Dict[str, int]
    latency_ms: int
    attempt_count: int
    attempts: List[Dict[str, Any]]


class ModelRouter:
    def __init__(self, config: Dict[str, Any], env_path: Path, event_log: Path):
        self.config = config
        self.env_path = Path(env_path)
        self.values = dotenv_values(self.env_path)
        self.event_log = event_log
        self.timeout = float(config["execution"].get("timeout_seconds", 180))
        self.max_retries = int(config["execution"].get("max_retries", 3))
        self.static_proxy_runtime_identity_guard: Optional[Any] = None
        self._static_proxy_identity_failure: Optional[str] = None
        self._static_proxy_identity_failure_lock = threading.Lock()
        limits = config["execution"].get("provider_max_concurrency", {})
        self.profile_semaphores = {
            name: threading.BoundedSemaphore(
                max(1, int(limits.get(name, config["execution"].get("max_workers", 4))))
            )
            for name in config["provider_profiles"]
        }

    @property
    def static_proxy_identity_failure(self) -> Optional[str]:
        with self._static_proxy_identity_failure_lock:
            return self._static_proxy_identity_failure

    def _remember_static_proxy_identity_failure(self, error: Exception) -> None:
        with self._static_proxy_identity_failure_lock:
            if self._static_proxy_identity_failure is None:
                self._static_proxy_identity_failure = type(error).__name__

    def _profile(self, model_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
        profile_name = model_spec["provider_profile"]
        profile = dict(self.config["provider_profiles"][profile_name])
        try:
            profile["protocol"] = normalize_model_protocol(profile.get("protocol"))
        except ValueError as error:
            raise RuntimeError(
                f"unsupported_provider_protocol:{profile_name}"
            ) from error
        base_url_env = profile.get("base_url_env")
        base = str(
            (self.values.get(base_url_env) if base_url_env else None)
            or profile.get("base_url")
            or ""
        ).strip()
        if not base:
            raise RuntimeError(f"provider_not_configured:{profile_name}")

        authentication = str(profile.get("authentication") or "api_key")
        if authentication == "none":
            parsed = urlparse(base)
            if profile.get("protocol") == "anthropic" or parsed.hostname not in {
                "127.0.0.1", "localhost", "::1",
            }:
                raise RuntimeError(f"unauthenticated_provider_requires_loopback:{profile_name}")
            # The OpenAI client requires a non-empty value even when the local
            # compatible server ignores Authorization. This sentinel is not a
            # credential and is never persisted in run artifacts.
            key = "local-no-auth"
        elif authentication == "api_key":
            api_key_env = profile.get("api_key_env")
            key = str(self.values.get(api_key_env) if api_key_env else "").strip()
            if not key:
                raise RuntimeError(f"provider_not_configured:{profile_name}")
        else:
            raise RuntimeError(f"unsupported_provider_authentication:{profile_name}")
        return profile, key, base

    @staticmethod
    def _openai_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        return str(
            getattr(message, "content", None)
            or getattr(message, "reasoning_content", None)
            or ""
        ).strip()

    @staticmethod
    def _usage(value: Any) -> Dict[str, int]:
        if value is None:
            return {}
        result = {}
        aliases = {
            "input_tokens": ("prompt_tokens", "input_tokens"),
            "output_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
        }
        for target, names in aliases.items():
            for name in names:
                number = getattr(value, name, None)
                if isinstance(number, int):
                    result[target] = number
                    break
        return result

    def _one_call(self, model_spec: Dict[str, Any], prompt: str) -> Tuple[str, Optional[str], Dict[str, int]]:
        profile, api_key, base_url = self._profile(model_spec)
        with self.profile_semaphores[model_spec["provider_profile"]]:
            return self._one_call_unlocked(profile, api_key, base_url, model_spec, prompt)

    def _one_call_unlocked(self, profile: Dict[str, Any], api_key: str, base_url: str, model_spec: Dict[str, Any], prompt: str) -> Tuple[str, Optional[str], Dict[str, int]]:
        model = model_spec["model"]
        timeout = float(
            model_spec.get(
                "timeout_seconds",
                profile.get("timeout_seconds", self.timeout),
            )
        )
        if profile["protocol"] == "anthropic":
            headers = {
                "x-api-key": api_key,
                "authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{api_root(base_url).rstrip('/')}/messages",
                    headers=headers,
                    json={
                        "model": model,
                        "max_tokens": model_spec.get("max_output_tokens", 512),
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
            response.raise_for_status()
            body = response.json()
            text = "".join(
                str(item.get("text", ""))
                for item in body.get("content", [])
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
            usage_body = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            usage = {
                key: int(value)
                for key, value in {
                    "input_tokens": usage_body.get("input_tokens"),
                    "output_tokens": usage_body.get("output_tokens"),
                }.items()
                if isinstance(value, int)
            }
            if usage:
                usage["total_tokens"] = sum(usage.values())
            return text, body.get("model"), usage

        client = OpenAI(api_key=api_key, base_url=api_root(base_url), timeout=timeout, max_retries=0)
        request: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model.startswith("gpt-"):
            request["max_completion_tokens"] = model_spec.get("max_output_tokens", 768)
            if model_spec.get("reasoning_effort"):
                request["reasoning_effort"] = model_spec["reasoning_effort"]
        else:
            request["max_tokens"] = model_spec.get("max_output_tokens", 512)
        if model_spec.get("disable_thinking") is True:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        elif model.startswith("qwen") or "deepseek" in model:
            request["extra_body"] = {"enable_thinking": False}
        if model_spec.get("strict_mcq_choice_schema") is True:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mcq_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": ["A", "B", "C"],
                            }
                        },
                        "required": ["choice"],
                        "additionalProperties": False,
                    },
                },
            }
        elif model_spec.get("json_mode") or profile.get("json_mode"):
            request["response_format"] = {"type": "json_object"}
        if "temperature" in model_spec and not model.startswith("gpt-"):
            request["temperature"] = model_spec["temperature"]
        response = client.chat.completions.create(**request)
        return self._openai_text(response), getattr(response, "model", None), self._usage(getattr(response, "usage", None))

    def request_json(
        self,
        stage: str,
        item_id: str,
        model_spec: Dict[str, Any],
        prompt: str,
        validator: Callable[[Dict[str, Any]], None],
    ) -> ModelResult:
        attempts: List[Dict[str, Any]] = []
        started_all = time.monotonic()
        last_text: Optional[str] = None
        last_response_model: Optional[str] = None
        last_usage: Dict[str, int] = {}
        max_retries = int(model_spec.get("max_retries", self.max_retries))
        for attempt in range(1, max_retries + 2):
            started = time.monotonic()
            try:
                if self.static_proxy_identity_failure is not None:
                    raise StaticProxyIdentityError(
                        "static_proxy_identity_previously_failed"
                    )
                if self.static_proxy_runtime_identity_guard is not None:
                    self.static_proxy_runtime_identity_guard.assert_current_route(
                        model_spec
                    )
                text, response_model, usage = self._one_call(model_spec, prompt)
                last_text, last_response_model, last_usage = text, response_model, usage
                expected_response_model = model_spec.get("expected_response_model")
                if expected_response_model is not None:
                    expected_response_model = str(expected_response_model).strip()
                    if not expected_response_model:
                        raise StaticProxyIdentityError(
                            "expected_response_model_must_be_nonempty"
                        )
                    if response_model is None or not str(response_model).strip():
                        raise ResponseModelMissing(
                            f"response_model_missing:{model_spec['provider_profile']}:{model_spec['model']}"
                        )
                    if str(response_model) != expected_response_model:
                        raise ResponseModelMismatch(
                            f"response_model_mismatch:{model_spec['provider_profile']}:{model_spec['model']}"
                        )
                elif model_spec.get("require_response_model_identity") is True:
                    _require_static_proxy_response_model_identity(
                        str(model_spec["model"]),
                        str(model_spec["provider_profile"]),
                        response_model,
                    )
                parsed = parse_json_object(text)
                validator(parsed)
                attempts.append({"attempt": attempt, "status": "ok", "latency_ms": round((time.monotonic() - started) * 1000)})
                result = ModelResult("completed", parsed, text, response_model, usage, round((time.monotonic() - started_all) * 1000), attempt, attempts)
                self._event(stage, item_id, model_spec, result)
                return result
            except Exception as error:
                attempts.append({"attempt": attempt, "status": "failed", "latency_ms": round((time.monotonic() - started) * 1000), **safe_error(error)})
                if isinstance(error, StaticProxyIdentityError):
                    self._remember_static_proxy_identity_failure(error)
                    break
                if attempt <= max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        result = ModelResult("failed", None, last_text, last_response_model, last_usage, round((time.monotonic() - started_all) * 1000), len(attempts), attempts)
        self._event(stage, item_id, model_spec, result)
        return result

    def _event(self, stage: str, item_id: str, model_spec: Dict[str, Any], result: ModelResult) -> None:
        event = {
            "timestamp": utc_now(),
            "stage": stage,
            "item_id": item_id,
            "provider_profile": model_spec["provider_profile"],
            "model": model_spec["model"],
            "terminal_status": result.terminal_status,
            "latency_ms": result.latency_ms,
            "attempt_count": result.attempt_count,
            "attempts": result.attempts,
            "credentials_or_endpoints_included": False,
        }
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def model_record(result: ModelResult, model_spec: Dict[str, Any], prompt_version: str) -> Dict[str, Any]:
    return {
        "terminal_status": result.terminal_status,
        "model": model_spec["model"],
        "provider_profile": model_spec["provider_profile"],
        "response_model": result.response_model,
        "prompt_version": prompt_version,
        "parsed_response": result.parsed_response,
        "raw_response": result.raw_response,
        "usage": result.usage,
        "latency_ms": result.latency_ms,
        "attempt_count": result.attempt_count,
        "attempts": result.attempts,
        "created_at": utc_now(),
    }


def _static_proxy_route_identity_record(
    router: ModelRouter, model_spec: Dict[str, Any]
) -> Dict[str, Any]:
    profile, _api_key, base_url = router._profile(model_spec)
    protocol = str(profile.get("protocol") or "")
    normalized_base = _normalized_endpoint_url(api_root(base_url))
    route_suffix = "messages" if protocol == "anthropic" else "chat/completions"
    normalized_route = _normalized_endpoint_url(
        f"{normalized_base.rstrip('/')}/{route_suffix}"
    )
    record = {
        "model": str(model_spec["model"]),
        "provider_profile": str(model_spec["provider_profile"]),
        "protocol": protocol,
        "base_url_sha256": hashlib.sha256(
            normalized_base.encode("utf-8")
        ).hexdigest(),
        "request_route_sha256": hashlib.sha256(
            normalized_route.encode("utf-8")
        ).hexdigest(),
        "credentials_or_endpoints_included": False,
    }
    record["route_identity_sha256"] = sha256_value(record)
    return record


def _ollama_api_root(base_url: str) -> str:
    normalized = _normalized_endpoint_url(api_root(base_url))
    parsed = urlparse(normalized)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


def _ollama_runtime_identity(
    base_url: str, requested_model: str, timeout_seconds: float
) -> Dict[str, Any]:
    """Read local Ollama attestation metadata without persisting its endpoint."""

    root = _ollama_api_root(base_url)
    timeout = max(1.0, min(float(timeout_seconds), 10.0))
    try:
        with httpx.Client(timeout=timeout) as client:
            version_response = client.get(f"{root}/api/version")
            version_response.raise_for_status()
            version_body = version_response.json()
            tags_response = client.get(f"{root}/api/tags")
            tags_response.raise_for_status()
            tags_body = tags_response.json()
            server_version = str(
                version_body.get("version") if isinstance(version_body, dict) else ""
            ).strip()
            models = tags_body.get("models") if isinstance(tags_body, dict) else None
            if not server_version or not isinstance(models, list):
                raise StaticProxyIdentityError(
                    "ollama_runtime_identity_incomplete"
                )
            matching = [
                item
                for item in models
                if isinstance(item, dict)
                and requested_model
                in {
                    str(item.get("name") or "").strip(),
                    str(item.get("model") or "").strip(),
                }
            ]
            digests = {
                str(item.get("digest") or "").strip()
                for item in matching
                if str(item.get("digest") or "").strip()
            }
            if len(digests) != 1:
                raise StaticProxyIdentityError(
                    f"ollama_installed_model_digest_unavailable:{requested_model}"
                )
            installed_digest = next(iter(digests))

            template_evidence: Dict[str, Any]
            try:
                show_response = client.post(
                    f"{root}/api/show",
                    json={"model": requested_model, "verbose": True},
                )
                show_response.raise_for_status()
                show_body = show_response.json()
                template = (
                    show_body.get("template") if isinstance(show_body, dict) else None
                )
                if isinstance(template, str) and template:
                    template_evidence = {
                        "status": "observed",
                        "sha256": hashlib.sha256(
                            template.encode("utf-8")
                        ).hexdigest(),
                        "source": "ollama_api_show",
                        "limitation": None,
                    }
                else:
                    template_evidence = {
                        "status": "unknown",
                        "sha256": None,
                        "source": "ollama_api_show",
                        "limitation": "ollama_api_show_template_missing",
                    }
            except Exception:
                template_evidence = {
                    "status": "unknown",
                    "sha256": None,
                    "source": "ollama_api_show",
                    "limitation": "ollama_api_show_unavailable",
                }
    except StaticProxyIdentityError:
        raise
    except Exception as error:
        raise StaticProxyIdentityError(
            f"ollama_runtime_identity_unavailable:{requested_model}"
        ) from error

    return {
        "kind": "ollama",
        "attestation_status": "observed",
        "server_version": server_version,
        "server_version_sha256": hashlib.sha256(
            server_version.encode("utf-8")
        ).hexdigest(),
        "installed_model_digest": installed_digest,
        "installed_model_digest_sha256": hashlib.sha256(
            installed_digest.encode("utf-8")
        ).hexdigest(),
        "template": template_evidence,
    }


def _static_proxy_model_route_runtime_identity(
    router: ModelRouter, model_spec: Dict[str, Any]
) -> Dict[str, Any]:
    route = _static_proxy_route_identity_record(router, model_spec)
    profile, _api_key, base_url = router._profile(model_spec)
    if str(model_spec["provider_profile"]) == "ollama_local":
        runtime = _ollama_runtime_identity(
            base_url,
            str(model_spec["model"]),
            float(model_spec.get("timeout_seconds", router.timeout)),
        )
    else:
        runtime = {
            "kind": "provider_managed",
            "attestation_status": "not_available",
            "server_version": None,
            "limitation": "provider_does_not_expose_stable_runtime_attestation",
        }
    record = {**route, "runtime": runtime}
    record["identity_sha256"] = sha256_value(record)
    return record


def _static_proxy_route_runtime_identity_snapshot(
    router: ModelRouter, model_specs: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    records = [
        _static_proxy_model_route_runtime_identity(router, model_spec)
        for model_spec in model_specs
    ]
    snapshot = {
        "schema_version": STATIC_PROXY_ROUTE_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "record_count": len(records),
        "records": records,
        "identity_sha256_by_model": {
            str(record["model"]): record["identity_sha256"] for record in records
        },
        "route_runtime_identity_set_sha256": sha256_value(records),
        "credentials_or_endpoints_included": False,
    }
    roster = tuple(
        (str(spec["model"]), str(spec["provider_profile"]))
        for spec in model_specs
    )
    _validate_static_proxy_route_runtime_identity_snapshot(snapshot, roster)
    return snapshot


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_static_proxy_route_runtime_identity_snapshot(
    snapshot: Any,
    expected_roster: Sequence[Tuple[str, str]] = STATIC_PROXY_MODEL_ROSTER,
) -> Dict[str, Dict[str, Any]]:
    expected_roster = tuple(expected_roster)
    if not isinstance(snapshot, dict):
        raise ValueError("static proxy route/runtime identity snapshot is missing")
    if (
        snapshot.get("schema_version")
        != STATIC_PROXY_ROUTE_RUNTIME_IDENTITY_SCHEMA_VERSION
        or snapshot.get("record_count") != len(expected_roster)
        or snapshot.get("credentials_or_endpoints_included") is not False
    ):
        raise ValueError("static proxy route/runtime identity snapshot is invalid")
    serialized = canonical_json(snapshot).casefold()
    if "http://" in serialized or "https://" in serialized:
        raise ValueError("static proxy route/runtime identity leaked an endpoint")
    pending_values = [snapshot]
    forbidden_plaintext_keys = {"api_key", "base_url", "request_route", "endpoint_url"}
    while pending_values:
        value = pending_values.pop()
        if isinstance(value, dict):
            if forbidden_plaintext_keys & {str(key).casefold() for key in value}:
                raise ValueError(
                    "static proxy route/runtime identity contains a plaintext secret field"
                )
            pending_values.extend(value.values())
        elif isinstance(value, list):
            pending_values.extend(value)
    records = snapshot.get("records")
    if not isinstance(records, list) or len(records) != len(expected_roster):
        raise ValueError("static proxy route/runtime identity roster is incomplete")
    observed_roster = [
        (record.get("model"), record.get("provider_profile"))
        for record in records
        if isinstance(record, dict)
    ]
    if observed_roster != list(expected_roster):
        raise ValueError("static proxy route/runtime identity roster changed")
    by_model: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("static proxy route/runtime identity record is invalid")
        if (
            record.get("credentials_or_endpoints_included") is not False
            or not _is_sha256(record.get("base_url_sha256"))
            or not _is_sha256(record.get("request_route_sha256"))
            or not _is_sha256(record.get("route_identity_sha256"))
            or not _is_sha256(record.get("identity_sha256"))
        ):
            raise ValueError("static proxy route/runtime identity hashes are invalid")
        route_fields = {
            key: record[key]
            for key in (
                "model",
                "provider_profile",
                "protocol",
                "base_url_sha256",
                "request_route_sha256",
                "credentials_or_endpoints_included",
            )
        }
        if record["route_identity_sha256"] != sha256_value(route_fields):
            raise ValueError("static proxy route identity digest changed")
        identity_fields = dict(record)
        identity_sha256 = identity_fields.pop("identity_sha256")
        if identity_sha256 != sha256_value(identity_fields):
            raise ValueError("static proxy route/runtime identity digest changed")
        runtime = record.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("static proxy runtime identity is invalid")
        if record["provider_profile"] == "ollama_local":
            template = runtime.get("template")
            if (
                runtime.get("kind") != "ollama"
                or runtime.get("attestation_status") != "observed"
                or not isinstance(runtime.get("server_version"), str)
                or not runtime["server_version"]
                or not _is_sha256(runtime.get("server_version_sha256"))
                or not isinstance(runtime.get("installed_model_digest"), str)
                or not runtime["installed_model_digest"]
                or not _is_sha256(runtime.get("installed_model_digest_sha256"))
                or not isinstance(template, dict)
                or template.get("status") not in {"observed", "unknown"}
            ):
                raise ValueError("static proxy Ollama runtime identity is incomplete")
            if runtime["server_version_sha256"] != hashlib.sha256(
                runtime["server_version"].encode("utf-8")
            ).hexdigest() or runtime["installed_model_digest_sha256"] != hashlib.sha256(
                runtime["installed_model_digest"].encode("utf-8")
            ).hexdigest():
                raise ValueError("static proxy Ollama runtime digest changed")
            if template["status"] == "observed":
                if not _is_sha256(template.get("sha256")) or template.get("limitation") is not None:
                    raise ValueError("static proxy Ollama template identity is invalid")
            elif template.get("sha256") is not None or not template.get("limitation"):
                raise ValueError("static proxy Ollama template limitation is missing")
        elif (
            runtime.get("kind") != "provider_managed"
            or runtime.get("attestation_status") != "not_available"
            or runtime.get("server_version") is not None
            or not runtime.get("limitation")
        ):
            raise ValueError("static proxy managed-provider limitation is missing")
        by_model[str(record["model"])] = record
    expected_by_model = {
        model: record["identity_sha256"] for model, record in by_model.items()
    }
    if snapshot.get("identity_sha256_by_model") != expected_by_model:
        raise ValueError("static proxy route/runtime per-model digest changed")
    if snapshot.get("route_runtime_identity_set_sha256") != sha256_value(records):
        raise ValueError("static proxy route/runtime identity set digest changed")
    return by_model


class _StaticProxyRuntimeIdentityGuard:
    """Cheap per-attempt route/config check against a probe-time runtime snapshot."""

    def __init__(
        self,
        router: ModelRouter,
        expected_snapshot: Dict[str, Any],
        config_sha256: str,
        expected_roster: Sequence[Tuple[str, str]] = STATIC_PROXY_MODEL_ROSTER,
    ):
        self.router = router
        self.expected_by_model = (
            _validate_static_proxy_route_runtime_identity_snapshot(
                expected_snapshot, expected_roster
            )
        )
        self.config_sha256 = config_sha256
        self._lock = threading.Lock()
        self.drift_reason: Optional[str] = None

    def assert_current_route(self, model_spec: Dict[str, Any]) -> None:
        with self._lock:
            if self.drift_reason is not None:
                raise StaticProxyIdentityError(self.drift_reason)
            model = str(model_spec.get("model") or "")
            expected = self.expected_by_model.get(model)
            if expected is None:
                self.drift_reason = f"static_proxy_unfrozen_model:{model}"
                raise StaticProxyIdentityError(self.drift_reason)
            if sha256_value(self.router.config) != self.config_sha256:
                self.drift_reason = "static_proxy_config_identity_drift"
                raise StaticProxyIdentityError(self.drift_reason)
            # The router snapshots its dotenv values at construction, so every
            # attempt is forced to the same in-memory route. Ollama network
            # metadata is compared once at full-run startup to avoid thousands
            # of metadata requests; response_model is checked after every call.
            current = _static_proxy_route_identity_record(self.router, model_spec)
            if current["route_identity_sha256"] != expected["route_identity_sha256"]:
                self.drift_reason = (
                    f"static_proxy_route_identity_drift:{expected['provider_profile']}:{model}"
                )
                raise StaticProxyIdentityError(self.drift_reason)


def stage_run(
    items: Sequence[Dict[str, Any]],
    output_path: Path,
    key: str,
    worker: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_workers: int,
    retry_terminal_failures: bool = True,
) -> List[Dict[str, Any]]:
    existing_rows = read_jsonl(output_path)
    existing = {row[key]: row for row in existing_rows}
    pending = [
        item for item in items
        if item[key] not in existing
        or (
            retry_terminal_failures
            and existing[item[key]].get("terminal_status") != "completed"
        )
    ]
    if max_workers <= 1:
        for item in pending:
            previous = existing.get(item[key])
            current = worker(item)
            if previous and previous.get("terminal_status") != "completed":
                current["attempts"] = [
                    *(previous.get("attempts") or []),
                    *(current.get("attempts") or []),
                ]
                current["attempt_count"] = (
                    int(previous.get("attempt_count") or 0)
                    + int(current.get("attempt_count") or 0)
                )
                current["resume_run_count"] = int(previous.get("resume_run_count") or 0) + 1
            existing[item[key]] = current
            write_jsonl(output_path, [existing[item[key]] for item in items if item[key] in existing])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                previous = existing.get(item[key])
                current = future.result()
                if previous and previous.get("terminal_status") != "completed":
                    current["attempts"] = [*(previous.get("attempts") or []), *(current.get("attempts") or [])]
                    current["attempt_count"] = int(previous.get("attempt_count") or 0) + int(current.get("attempt_count") or 0)
                    current["resume_run_count"] = int(previous.get("resume_run_count") or 0) + 1
                existing[item[key]] = current
                write_jsonl(output_path, [existing[value[key]] for value in items if value[key] in existing])
    return [existing[item[key]] for item in items if item[key] in existing]


def _ranked(rows: Sequence[Dict[str, Any]], seed: int, salt: str) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: sha256_value([seed, salt, row["source_id"]]))


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _artifact_path_label(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _resolve_input_bundle_path(
    config: Dict[str, Any],
    project_root: Path,
    override: Optional[Path] = None,
) -> Path:
    if override is not None:
        return _project_path(project_root, override)
    inputs = config.get("inputs") or {}
    configured = inputs.get("input_bundle")
    if configured:
        return _project_path(project_root, configured)
    legacy_canonical = inputs.get("canonical_triples")
    if legacy_canonical:
        canonical_path = _project_path(project_root, legacy_canonical)
        expected_canonical = (
            project_root / formal_admission.KNOWN_LEGACY_CANONICAL_RELATIVE_PATH
        ).resolve()
        if canonical_path != expected_canonical:
            raise ValueError(
                "legacy canonical_triples routing is limited to the pinned historical artifact"
            )
        return canonical_path.with_name("factual_triples_with_distractors.jsonl")
    raise ValueError(
        "missing input bundle: configure inputs.input_bundle or legacy inputs.canonical_triples"
    )


def _optional_configured_input_path(
    inputs: Dict[str, Any], project_root: Path, field: str
) -> Optional[Path]:
    value = inputs.get(field)
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"inputs.{field} must be a non-empty path when configured")
    return _project_path(project_root, value)


def _load_input_bundle(
    path: Path,
    *,
    allow_legacy_schema: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"input bundle does not exist: {path}")
    raw = path.read_bytes()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"input bundle is not valid UTF-8: {path}: {exc}") from exc
    if path.suffix == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input bundle: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object: {path}")
        schema_version = payload.get("schema_version")
        if schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported input bundle schema_version: {schema_version}")
        rows = payload.get("records")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("input bundle records must be an array of objects")
        bundle_id = payload.get("bundle_id") or path.stem
        bundle_format = "json"
    elif path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
        declared_versions = {
            row.get("schema_version") for row in rows if row.get("schema_version") is not None
        }
        if declared_versions:
            if declared_versions != {INPUT_BUNDLE_SCHEMA_VERSION} or any(
                row.get("schema_version") is None for row in rows
            ):
                raise ValueError(
                    f"mixed or unsupported input bundle schema versions: {sorted(str(v) for v in declared_versions)}"
                )
            schema_version = INPUT_BUNDLE_SCHEMA_VERSION
        else:
            if not allow_legacy_schema:
                raise ValueError(
                    "explicit input bundle JSONL rows must declare schema_version="
                    f"{INPUT_BUNDLE_SCHEMA_VERSION}"
                )
            schema_version = LEGACY_INPUT_BUNDLE_SCHEMA_VERSION
        bundle_id = path.stem
        bundle_format = "jsonl"
    else:
        raise ValueError("input bundle must be a .json or .jsonl file")

    source_ids = [row.get("source_id") for row in rows]
    if any(not isinstance(source_id, str) or not source_id.strip() for source_id in source_ids):
        raise ValueError("every input bundle record must have a non-empty source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("input bundle contains duplicate source_id values")
    if schema_version == INPUT_BUNDLE_SCHEMA_VERSION:
        base_fact_ids = [
            _input_base_fact_id(row, schema_version) for row in rows
        ]
        if len(base_fact_ids) != len(set(base_fact_ids)):
            raise ValueError("explicit input bundle contains duplicate base_fact_id values")
    return rows, {
        "bundle_id": str(bundle_id),
        "schema_version": schema_version,
        "format": bundle_format,
        "record_count": len(rows),
        "sha256": input_sha256,
    }


def _input_base_fact_id(row: Dict[str, Any], schema_version: str) -> str:
    if schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
        return str(row["source_id"])
    return formal_admission.explicit_base_fact_id(
        row, f"explicit input bundle row {row.get('source_id')}"
    )


def _input_canonical_status(row: Dict[str, Any], schema_version: str) -> str:
    if schema_version == INPUT_BUNDLE_SCHEMA_VERSION:
        status = row.get("canonical_status")
        if status not in {"frozen", "pending_review", "rejected"}:
            raise ValueError(
                f"explicit input bundle row {row.get('source_id')} has invalid canonical_status: {status}"
            )
        return str(status)
    decision = row.get("canonical_decision")
    if decision == "keep":
        return "frozen"
    if decision == "pending_review":
        return "pending_review"
    return "rejected"


def _input_prompt_ready(row: Dict[str, Any], schema_version: str) -> bool:
    if schema_version == INPUT_BUNDLE_SCHEMA_VERSION:
        admission = row.get("admission")
        return (
            row.get("prompt_ready") is True
            and isinstance(admission, dict)
            and admission.get("prompt_ready") is True
        )
    return row.get("factual_prompt_status") == "generated"


def _input_review_blockers(
    row: Dict[str, Any],
    schema_version: str,
    external_manifest_blockers: Sequence[str] = (),
) -> List[str]:
    """Keep a renamed provisional row from becoming formal evidence."""
    if schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
        return []
    blockers: List[str] = []
    if row.get("bundle_status") not in FORMAL_INPUT_BUNDLE_STATUSES:
        blockers.append("bundle_not_review_frozen")
    admission = row.get("admission")
    if not isinstance(admission, dict) or admission.get("semantic_review_complete") is not True:
        blockers.append("semantic_review_incomplete")
    if row.get("evidence_tier") not in FORMAL_INPUT_EVIDENCE_TIERS:
        blockers.append("evidence_tier_not_formal")
    if row.get("split_status") != "frozen":
        blockers.append("split_not_frozen")
    if row.get("split_assignment") not in INPUT_SPLIT_ASSIGNMENTS:
        blockers.append("split_assignment_missing_or_invalid")
    if not isinstance(row.get("split_group_id"), str) or not row["split_group_id"].strip():
        blockers.append("split_group_id_missing")
    if (
        not isinstance(row.get("split_policy_version"), str)
        or not row["split_policy_version"].strip()
    ):
        blockers.append("split_policy_version_missing")
    if row.get("canonical_status") == "frozen":
        blockers.extend(external_manifest_blockers)
    return blockers


def _input_split_allowed(
    row: Dict[str, Any], schema_version: str, allowed_splits: Sequence[str]
) -> bool:
    if schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
        return True
    return row.get("split_assignment") in set(allowed_splits)


def _configured_allowed_splits(config: Dict[str, Any]) -> List[str]:
    configured_splits = (config.get("inputs") or {}).get(
        "allowed_splits", ["development"]
    )
    if (
        not isinstance(configured_splits, list)
        or not configured_splits
        or any(split not in INPUT_SPLIT_ASSIGNMENTS for split in configured_splits)
    ):
        raise ValueError(
            "inputs.allowed_splits must be a non-empty subset of "
            f"{sorted(INPUT_SPLIT_ASSIGNMENTS)}"
        )
    return sorted(set(configured_splits))


def _validate_input_split_groups(rows: Sequence[Dict[str, Any]], schema_version: str) -> None:
    if schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
        return
    group_contracts: Dict[str, Tuple[str, str]] = {}
    policy_versions = set()
    for row in rows:
        group_id = row.get("split_group_id")
        assignment = row.get("split_assignment")
        policy_version = row.get("split_policy_version")
        if (
            not isinstance(group_id, str)
            or not group_id.strip()
            or assignment not in INPUT_SPLIT_ASSIGNMENTS
            or not isinstance(policy_version, str)
            or not policy_version.strip()
        ):
            continue
        contract = (str(assignment), policy_version.strip())
        policy_versions.add(policy_version.strip())
        previous = group_contracts.setdefault(group_id.strip(), contract)
        if previous != contract:
            raise ValueError(
                f"split group {group_id!r} has inconsistent assignment or policy version"
            )
    if len(policy_versions) > 1:
        raise ValueError("explicit input bundle must use exactly one split_policy_version")


def _unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = " ".join(text.casefold().split())
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return sorted(output, key=lambda value: (value.casefold(), value))


def _answer_aliases_en(row: Dict[str, Any], answer: str) -> List[str]:
    source_snapshot = row.get("source_snapshot") or {}
    source_record = source_snapshot.get("record") or {}
    upstream_metadata = source_record.get("upstream_metadata") or {}
    return _unique_strings([
        answer,
        row.get("source_answer"),
        *(row.get("answer_aliases_en") or []),
        *(row.get("answer_aliases") or []),
        *(upstream_metadata.get("answer_aliases") or []),
    ])


def _prepared_distractors(row: Dict[str, Any], max_distractors: int) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    explicit = row.get("distractors")
    candidates = explicit if isinstance(explicit, list) else row.get("distractor_candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"input bundle distractors must be an array: {row.get('source_id')}")
    for candidate in candidates[:max_distractors]:
        if not isinstance(candidate, dict):
            raise ValueError(f"input bundle distractor must be an object: {row.get('source_id')}")
        source_choice_index = candidate.get("source_choice_index")
        distractor_id = candidate.get("distractor_id")
        if not distractor_id and source_choice_index is not None:
            distractor_id = f"{row['source_id']}_source_wrong_{source_choice_index}"
        text_en = (
            candidate.get("text_en")
            or candidate.get("answer_en")
            or candidate.get("distractor_answer")
        )
        if not isinstance(distractor_id, str) or not distractor_id.strip():
            raise ValueError(f"input bundle distractor is missing distractor_id: {row.get('source_id')}")
        if not isinstance(text_en, str) or not text_en.strip():
            raise ValueError(f"input bundle distractor is missing text_en: {distractor_id}")
        prepared.append({
            **candidate,
            "distractor_id": distractor_id,
            "text_en": text_en.strip(),
            "source_choice_index": source_choice_index,
            "source": candidate.get("source") or candidate.get("distractor_source") or "input_bundle",
        })
    distractor_ids = [row["distractor_id"] for row in prepared]
    if len(distractor_ids) != len(set(distractor_ids)):
        raise ValueError(f"duplicate distractor_id for source {row.get('source_id')}")
    return prepared


def _select_target_distractors(
    accepted_distractors: Sequence[Dict[str, Any]],
    per_source: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Select perturbation targets without discarding multi-option foils."""
    if per_source <= 0:
        return list(accepted_distractors)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in accepted_distractors:
        grouped[row["source_id"]].append(row)
    return [
        row
        for source_id in sorted(grouped)
        for row in sorted(
            grouped[source_id],
            key=lambda value: (
                value.get("distractor_source") == "generated_replacement",
                sha256_value([seed, "distractor-selection", value["item_id"]]),
            ),
        )[:per_source]
    ]


def stratified_sample(rows: Sequence[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source_dataset"], row["prompt_quality_tier"], row.get("answer_type") or "unknown")].append(row)
    for key in groups:
        groups[key] = _ranked(groups[key], seed, "|".join(key))
    by_dataset: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for key in groups:
        by_dataset[key[0]].append(key)
    for dataset in by_dataset:
        by_dataset[dataset].sort(key=lambda key: sha256_value([seed, list(key)]))
    datasets = sorted(by_dataset, key=lambda value: sha256_value([seed, value]))
    selected: List[Dict[str, Any]] = []
    offsets = Counter()
    while len(selected) < limit:
        progressed = False
        for dataset in datasets:
            keys = by_dataset[dataset]
            for _ in range(len(keys)):
                key = keys[offsets[(dataset, "key")] % len(keys)]
                offsets[(dataset, "key")] += 1
                index = offsets[key]
                if index < len(groups[key]):
                    selected.append(groups[key][index])
                    offsets[key] += 1
                    progressed = True
                    break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def prepare_manifest(
    config: Dict[str, Any],
    project_root: Path,
    run_dir: Path,
    limit: int,
    run_id: str,
    exclude_source_ids: Sequence[str] = (),
    input_bundle_path: Optional[Path] = None,
) -> Dict[str, Any]:
    prompt_path = _resolve_input_bundle_path(config, project_root, input_bundle_path)
    inputs = config.get("inputs") or {}
    legacy_schema_allowed = input_bundle_path is None and not inputs.get("input_bundle")
    input_rows, input_bundle = _load_input_bundle(
        prompt_path,
        allow_legacy_schema=legacy_schema_allowed,
    )
    _validate_input_split_groups(input_rows, input_bundle["schema_version"])
    allowed_input_splits = _configured_allowed_splits(config)
    input_sha256 = input_bundle["sha256"]
    review_freeze_manifest_path = _optional_configured_input_path(
        inputs, project_root, "review_freeze_manifest"
    )
    split_freeze_manifest_path = _optional_configured_input_path(
        inputs, project_root, "split_freeze_manifest"
    )
    external_formal_admission = formal_admission.validate_external_formal_admission(
        input_bundle_path=prompt_path,
        input_schema_version=input_bundle["schema_version"],
        records=input_rows,
        input_bundle_sha256=input_sha256,
        review_freeze_manifest_path=review_freeze_manifest_path,
        split_freeze_manifest_path=split_freeze_manifest_path,
        legacy_project_root=(
            project_root
            if input_bundle["schema_version"] == LEGACY_INPUT_BUNDLE_SCHEMA_VERSION
            else None
        ),
    )
    current_runtime_sha256 = _runtime_sha256()
    previous = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else None
    if previous:
        previous_runtime_sha256 = previous.get("runtime_sha256")
        if not previous_runtime_sha256:
            raise ValueError("run manifest is missing runtime fingerprint; use a new run_id")
        if previous_runtime_sha256 != current_runtime_sha256:
            raise ValueError("run manifest runtime fingerprint does not match current runtime")
        previous_input_sha256 = previous.get("input_bundle_sha256") or previous.get("prompt_input_sha256")
        if not previous_input_sha256:
            raise ValueError("run manifest is missing input bundle fingerprint; use a new run_id")
        if previous_input_sha256 != input_sha256:
            raise ValueError("run manifest input bundle fingerprint does not match requested input")
        previous_admission = previous.get("external_formal_admission") or {}
        previous_fingerprint = previous_admission.get("admission_fingerprint_sha256")
        current_fingerprint = external_formal_admission["admission_fingerprint_sha256"]
        if not previous_fingerprint:
            raise ValueError("run manifest is missing formal admission fingerprint; use a new run_id")
        if previous_fingerprint != current_fingerprint:
            raise ValueError("run manifest formal admission fingerprint does not match requested input")

    status_by_source = {
        row["source_id"]: _input_canonical_status(row, input_bundle["schema_version"])
        for row in input_rows
    }
    review_blockers_by_source = {
        row["source_id"]: _input_review_blockers(
            row,
            input_bundle["schema_version"],
            external_formal_admission["blockers"],
        )
        for row in input_rows
    }
    pending_review_rows = [
        {
            **row,
            "input_admission_status": (
                "pending_review_excluded_from_formal_prepare"
                if status_by_source[row["source_id"]] == "pending_review"
                else "frozen_missing_explicit_review_evidence"
            ),
            "input_admission_blockers": review_blockers_by_source[row["source_id"]],
        }
        for row in input_rows
        if status_by_source[row["source_id"]] == "pending_review"
        or (
            status_by_source[row["source_id"]] == "frozen"
            and bool(review_blockers_by_source[row["source_id"]])
        )
    ]
    input_exclusions = [
        {
            **row,
            "input_admission_status": (
                "rejected_before_formal_prepare"
                if status_by_source[row["source_id"]] == "rejected"
                else (
                    "frozen_but_prompt_not_ready"
                    if not _input_prompt_ready(row, input_bundle["schema_version"])
                    else "split_not_allowed_for_prepare"
                )
            ),
        }
        for row in input_rows
        if status_by_source[row["source_id"]] == "rejected"
        or (
            status_by_source[row["source_id"]] == "frozen"
            and not review_blockers_by_source[row["source_id"]]
            and (
                not _input_prompt_ready(row, input_bundle["schema_version"])
                or not _input_split_allowed(
                    row, input_bundle["schema_version"], allowed_input_splits
                )
            )
        )
    ]
    review_queue_path = run_dir / "input_review_queue.jsonl"
    exclusion_path = run_dir / "input_exclusions.jsonl"
    write_jsonl(review_queue_path, pending_review_rows)
    write_jsonl(exclusion_path, input_exclusions)

    formal_rows = [
        row for row in input_rows
        if status_by_source[row["source_id"]] == "frozen"
        and not review_blockers_by_source[row["source_id"]]
        and _input_prompt_ready(row, input_bundle["schema_version"])
        and _input_split_allowed(row, input_bundle["schema_version"], allowed_input_splits)
    ]
    if input_bundle["schema_version"] == INPUT_BUNDLE_SCHEMA_VERSION:
        formal_split_policy_versions = {
            str(row["split_policy_version"]).strip() for row in formal_rows
        }
        if len(formal_split_policy_versions) > 1:
            raise ValueError(
                "formal input rows must use exactly one split_policy_version"
            )
    excluded = set(exclude_source_ids)
    rows = [
        row for row in formal_rows
        if row.get("source_id") not in excluded
    ]
    selected = stratified_sample(rows, limit, int(config["pilot"]["seed"]))
    prepared = []
    external_admission_manifest_record = {
        **external_formal_admission,
        "input_bundle": {
            **external_formal_admission["input_bundle"],
            "path": _artifact_path_label(project_root, prompt_path),
        },
        "review_freeze_manifest": {
            **external_formal_admission["review_freeze_manifest"],
            "path": (
                _artifact_path_label(project_root, review_freeze_manifest_path)
                if review_freeze_manifest_path is not None
                else None
            ),
        },
        "split_freeze_manifest": {
            **external_formal_admission["split_freeze_manifest"],
            "path": (
                _artifact_path_label(project_root, split_freeze_manifest_path)
                if split_freeze_manifest_path is not None
                else None
            ),
        },
    }
    max_distractors = int(config["pilot"]["max_distractors_per_triple"])
    for row in selected:
        answer_en = str(
            row.get("answer_en") or row.get("prompt_expected_answer") or row.get("answer") or ""
        ).strip()
        prompt_en = str(row.get("prompt_en") or "").strip()
        canonical_fact_value = row.get("canonical_fact")
        if input_bundle["schema_version"] == INPUT_BUNDLE_SCHEMA_VERSION:
            canonical_fact_value = canonical_fact_value or row.get("canonical_fact_en")
        canonical_fact = str(canonical_fact_value or "").strip()
        if not answer_en or not prompt_en or not canonical_fact:
            raise ValueError(f"frozen prompt-ready input is incomplete: {row['source_id']}")
        distractors = _prepared_distractors(row, max_distractors)
        source_snapshot = row.get("source_snapshot") or {}
        source_record = source_snapshot.get("record") or {}
        base_fact_id = _input_base_fact_id(row, input_bundle["schema_version"])
        prepared.append({
            "base_fact_id": base_fact_id,
            "base_id": base_fact_id,
            "source_id": row["source_id"],
            "source_pool_id": row.get("source_pool_id"),
            "source_dataset": row["source_dataset"],
            "source_format": row.get("source_format") or source_record.get("source_format"),
            "evidence_tier": row.get("evidence_tier") or "legacy_reviewed_canonical",
            "canonical_status": "frozen",
            "split_group_id": row.get("split_group_id"),
            "split_assignment": row.get("split_assignment"),
            "split_status": row.get("split_status"),
            "split_policy_version": row.get("split_policy_version"),
            "sealed_evaluation_status": row.get("sealed_evaluation_status"),
            "answer_type": row.get("answer_type"),
            "normalization_status": row.get("normalization_status"),
            "relation_raw": row.get("relation_raw"),
            "relation_normalized": row.get("relation_normalized"),
            "probe_relation_id": row.get("probe_relation_id"),
            "prompt_quality_tier": row["prompt_quality_tier"],
            "prompt_en": prompt_en,
            "answer_en": answer_en,
            "answer_aliases_en": _answer_aliases_en(row, answer_en),
            "canonical_fact": canonical_fact,
            "distractors": distractors,
            "input_record_sha256": sha256_value(row),
            "formal_admission_evidence": {
                "review_freeze_manifest": external_admission_manifest_record[
                    "review_freeze_manifest"
                ],
                "split_freeze_manifest": external_admission_manifest_record[
                    "split_freeze_manifest"
                ],
            },
        })
    config_sha = sha256_value(config)
    runtime_sha = current_runtime_sha256
    amendments = list(previous.get("config_amendments", [])) if previous else []
    if previous and previous.get("config_sha256") != config_sha:
        amendments.append({
            "changed_at": utc_now(),
            "from_config_sha256": previous.get("config_sha256"),
            "to_config_sha256": config_sha,
            "reason": "preflight remediation after a recorded failed gate; completed checkpoints retained",
        })
    simulation_role = config["model_roles"]["simulation"]
    perturbation_generation_role = config["model_roles"].get(
        "perturbation_generation", {}
    )
    manipulation_family = _configured_manipulation_family(config)
    neutral_mode = simulation_role.get("neutral_context_mode", "generic_shared")
    shared_variants = ["original"] if neutral_mode == "matched_per_candidate" else list(SHARED_SIMULATION_VARIANTS)
    designated_distractors_per_base_fact = int(
        perturbation_generation_role.get("target_distractors_per_source", 0)
    )
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        if designated_distractors_per_base_fact != 2:
            raise ValueError("explicit_false_assertion_v1_requires_two_designated_distractors")
        if not perturbation_generation_role.get("matched_neutral"):
            raise ValueError("explicit_false_assertion_v1_requires_matched_neutral")
        if neutral_mode != "matched_per_candidate":
            raise ValueError("explicit_false_assertion_v1_requires_candidate_specific_neutral")
        if simulation_role.get("endpoint", "binary") != "multi_option":
            raise ValueError("explicit_false_assertion_v1_requires_multi_option_endpoint")
        if int(simulation_role.get("max_options", 3)) != 3:
            raise ValueError("explicit_false_assertion_v1_requires_three_options")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": config_sha,
        "runtime_sha256": runtime_sha,
        "config_amendments": amendments,
        "prompt_input": _artifact_path_label(project_root, prompt_path),
        "prompt_input_sha256": input_sha256,
        "input_bundle": _artifact_path_label(project_root, prompt_path),
        "input_bundle_id": input_bundle["bundle_id"],
        "input_bundle_format": input_bundle["format"],
        "input_bundle_schema_version": input_bundle["schema_version"],
        "input_bundle_sha256": input_sha256,
        "input_bundle_record_count": input_bundle["record_count"],
        "external_formal_admission": external_admission_manifest_record,
        "review_freeze_manifest": external_admission_manifest_record[
            "review_freeze_manifest"
        ],
        "split_freeze_manifest": external_admission_manifest_record[
            "split_freeze_manifest"
        ],
        "input_admission_policy": {
            "formal_prepare_requires_canonical_status": "frozen",
            "explicit_prompt_ready_rule": (
                "top-level prompt_ready=true and admission.prompt_ready=true"
            ),
            "explicit_frozen_review_requirements": {
                "bundle_status": sorted(FORMAL_INPUT_BUNDLE_STATUSES),
                "semantic_review_complete": True,
                "evidence_tier": sorted(FORMAL_INPUT_EVIDENCE_TIERS),
                "split_status": "frozen",
                "split_assignment": sorted(INPUT_SPLIT_ASSIGNMENTS),
                "split_group_id": "non_empty",
                "split_policy_version": "non_empty_and_single_version",
                "external_review_freeze_manifest": "required_and_exact_bundle_bound",
                "external_split_freeze_manifest": "required_and_exact_bundle_bound",
            },
            "allowed_splits_for_this_prepare": allowed_input_splits,
            "legacy_compatibility_rule": "canonical_decision=keep is treated as frozen",
            "pending_review_action": "write_review_queue_and_exclude_from_formal_prepare",
        },
        "input_admission_counts": {
            "frozen": sum(status == "frozen" for status in status_by_source.values()),
            "pending_review": sum(
                status == "pending_review" for status in status_by_source.values()
            ),
            "frozen_review_incomplete": sum(
                status_by_source[row["source_id"]] == "frozen"
                and bool(review_blockers_by_source[row["source_id"]])
                for row in input_rows
            ),
            "rejected": sum(status == "rejected" for status in status_by_source.values()),
            "frozen_prompt_ready": len(formal_rows),
            "frozen_prompt_not_ready": sum(
                row["input_admission_status"] == "frozen_but_prompt_not_ready"
                for row in input_exclusions
            ),
            "split_not_allowed": sum(
                row["input_admission_status"] == "split_not_allowed_for_prepare"
                for row in input_exclusions
            ),
            "eligible_after_explicit_exclusions": len(rows),
        },
        "input_review_queue": _artifact_path_label(project_root, review_queue_path),
        "input_review_queue_sha256": hashlib.sha256(review_queue_path.read_bytes()).hexdigest(),
        "input_exclusions": _artifact_path_label(project_root, exclusion_path),
        "input_exclusions_sha256": hashlib.sha256(exclusion_path.read_bytes()).hexdigest(),
        "seed": config["pilot"]["seed"],
        "requested_limit": limit,
        "selected_count": len(prepared),
        "selected_source_ids_sha256": sha256_value(
            sorted(row["source_id"] for row in prepared)
        ),
        "excluded_source_count": len(excluded),
        "excluded_source_ids_sha256": sha256_value(sorted(excluded)),
        "stratify_by": config["pilot"]["stratify_by"],
        "paths_not_taken_enabled": False,
        "holdout_enabled": False,
        "minimum_translation_acceptance": config.get("preflight", {}).get("minimum_translation_acceptance", 0.5),
        "currency_cost_gate": config.get("preflight", {}).get("currency_cost_gate"),
        "simulation_models": [model["model"] for model in config["model_roles"]["simulation"]["models"]],
        "simulation_languages": list(SIMULATION_LANGUAGES),
        "simulation_variants": list(SIMULATION_VARIANTS),
        "shared_simulation_variants": shared_variants,
        "neutral_context_mode": neutral_mode,
        "simulation_endpoint": simulation_role.get("endpoint", "binary"),
        "simulation_max_options": int(simulation_role.get("max_options", 3)),
        "manipulation_family": manipulation_family,
        "designated_distractors_per_base_fact": designated_distractors_per_base_fact,
        "statistical_unit": "base_fact_id",
        "simulation_requires_codex_proxy_review": bool(
            config.get("preflight", {}).get("semantic_review", {}).get("reviewer_type") == "codex_proxy"
        ),
        "translation_repair_enabled": bool(
            config["model_roles"]["translation"].get("repair_rejected", False)
        ),
        "distractor_replenishment_enabled": bool(
            config["model_roles"].get("distractor_generation", {}).get("enabled", False)
        ),
        "perturbation_auto_judge_hard_gate": bool(
            config["model_roles"]["perturbation_validation"].get("hard_gate", True)
        ),
        "records": prepared,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def prepare_proxy_rerun(
    config: Dict[str, Any],
    project_root: Path,
    source_run_dir: Path,
    run_dir: Path,
    run_id: str,
    review_path: Path,
    max_candidates_per_source: Optional[int] = None,
    strength_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a three-arm rerun by reusing frozen upstream checkpoints and a Codex allowlist."""
    if (run_dir / "run_manifest.json").exists():
        raise ValueError(f"target_run_already_prepared:{run_id}")
    source_manifest = read_json(source_run_dir / "run_manifest.json")
    review = read_json(review_path)
    if review.get("reviewer_type") != "codex_proxy" or review.get("not_human_gold") is not True:
        raise ValueError("invalid_codex_proxy_review_identity")
    if review.get("run_id") != source_manifest.get("run_id"):
        raise ValueError("codex_proxy_review_run_mismatch")
    manipulation_family = str(
        source_manifest.get("manipulation_family")
        or TRUTHFUL_SALIENCE_MANIPULATION_FAMILY
    )
    allowed_input_splits = _configured_allowed_splits(config)
    source_records = list(source_manifest.get("records") or [])
    if not source_records:
        raise ValueError("proxy_rerun_source_manifest_has_no_records")
    invalid_split_records = [
        row for row in source_records
        if row.get("split_assignment") not in INPUT_SPLIT_ASSIGNMENTS
    ]
    if invalid_split_records:
        raise ValueError("proxy_rerun_source_records_missing_frozen_split_assignment")
    selected_records = [
        row for row in source_records
        if row.get("split_assignment") in set(allowed_input_splits)
    ]
    if not selected_records:
        raise ValueError("proxy_rerun_allowed_splits_selected_no_records")
    selected_source_ids = {row["source_id"] for row in selected_records}
    excluded_split_source_ids = sorted(
        row["source_id"] for row in source_records
        if row["source_id"] not in selected_source_ids
    )
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        if allowed_input_splits != ["development"]:
            raise ValueError(
                "explicit_false_assertion_proxy_screen_requires_development_only"
            )
        if (
            review.get("review_contract_version")
            != EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            or review.get("manipulation_family")
            != EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            or review.get("adjudicator_type") != "codex"
            or review.get("review_status") != "codex_adjudicated"
            or review.get("human_gold") is not False
            or review.get("review_blinded_to_simulation_results") is not True
        ):
            raise ValueError("invalid_explicit_false_assertion_codex_review_contract")
    source_perturbations = read_jsonl(source_run_dir / "perturbations.jsonl")
    review_rows = review.get("perturbation_reviews", [])
    accepted_ids = sorted(
        row["item_id"] for row in review_rows
        if row.get("decision") == "accept"
        and (
            manipulation_family != EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            or (
                isinstance(row.get("checks"), dict)
                and all(
                    row["checks"].get(check) is True
                    for check in EXPLICIT_FALSE_ASSERTION_CHECKS
                )
            )
        )
    )
    available_ids = {row.get("candidate_id") for row in source_perturbations}
    if not accepted_ids or not set(accepted_ids).issubset(available_ids):
        raise ValueError("invalid_codex_proxy_perturbation_allowlist")
    accepted_before_sampling = len(accepted_ids)
    row_by_id = {row["candidate_id"]: row for row in source_perturbations}
    accepted_ids = [
        candidate_id for candidate_id in accepted_ids
        if row_by_id[candidate_id].get("source_id") in selected_source_ids
    ]
    if not accepted_ids:
        raise ValueError("proxy_rerun_allowed_splits_selected_no_candidates")
    accepted_after_split_filter = len(accepted_ids)
    if strength_filter is not None:
        accepted_ids = [
            candidate_id for candidate_id in accepted_ids
            if row_by_id[candidate_id].get("candidate", {}).get("strength") == strength_filter
        ]
        if not accepted_ids:
            raise ValueError("strength_filter_selected_no_candidates")

    simulation_role = config["model_roles"]["simulation"]
    configured_manipulation_family = _configured_manipulation_family(config)
    neutral_mode = simulation_role.get("neutral_context_mode", "generic_shared")
    shared_variants = (
        ["original"]
        if neutral_mode == "matched_per_candidate"
        else list(SHARED_SIMULATION_VARIANTS)
    )
    requested_design = {
        "simulation_endpoint": simulation_role.get("endpoint", "binary"),
        "simulation_max_options": int(simulation_role.get("max_options", 3)),
        "neutral_context_mode": neutral_mode,
        "shared_simulation_variants": shared_variants,
    }
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        if configured_manipulation_family != manipulation_family:
            raise ValueError("explicit_false_assertion_proxy_manipulation_family_mismatch")
        target_distractor_count = int(
            (config.get("model_roles") or {}).get(
                "perturbation_generation", {}
            ).get("target_distractors_per_source", 0)
        )
        if target_distractor_count != 2:
            raise ValueError(
                "explicit_false_assertion_proxy_design_mismatch:"
                "designated_distractors_per_base_fact"
            )
        if source_manifest.get("designated_distractors_per_base_fact") != 2:
            raise ValueError(
                "explicit_false_assertion_source_design_mismatch:"
                "designated_distractors_per_base_fact"
            )
        required_design = {
            "simulation_endpoint": "multi_option",
            "simulation_max_options": 3,
            "neutral_context_mode": "matched_per_candidate",
            "shared_simulation_variants": ["original"],
        }
        for field, required in required_design.items():
            if source_manifest.get(field) != required:
                raise ValueError(
                    f"explicit_false_assertion_source_design_mismatch:{field}"
                )
            if requested_design[field] != required:
                raise ValueError(
                    f"explicit_false_assertion_proxy_design_mismatch:{field}"
                )
    endpoint_exclusions: List[Dict[str, str]] = []
    if simulation_role.get("endpoint", "binary") == "multi_option":
        accepted_distractor_rows = [
            row for row in read_jsonl(source_run_dir / "verified_distractors.jsonl")
            if row.get("terminal_status") == "completed"
            and row.get("parsed_response", {}).get("decision") == "accept"
            and all(row.get("parsed_response", {}).get("checks", {}).values())
        ]
        accepted_distractor_ids = {row["item_id"] for row in accepted_distractor_rows}
        accepted_distractor_count_by_source = Counter(
            row["source_id"] for row in accepted_distractor_rows
        )
        endpoint_eligible_ids = []
        for candidate_id in accepted_ids:
            candidate = row_by_id[candidate_id]
            reasons = []
            if candidate.get("distractor_id") not in accepted_distractor_ids:
                reasons.append("target_distractor_not_auto_accepted")
            if accepted_distractor_count_by_source[candidate["source_id"]] < 2:
                reasons.append("fewer_than_two_auto_accepted_distractors")
            if reasons:
                endpoint_exclusions.append({
                    "candidate_id": candidate_id,
                    "source_id": candidate["source_id"],
                    "reason": ";".join(reasons),
                })
            else:
                endpoint_eligible_ids.append(candidate_id)
        accepted_ids = endpoint_eligible_ids
        if not accepted_ids:
            raise ValueError("multi_option_endpoint_selected_no_candidates")

    accepted_after_endpoint_filter = len(accepted_ids)
    design_exclusions: List[Dict[str, str]] = []
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        required_distractors = 2
        if max_candidates_per_source is not None and max_candidates_per_source < 2:
            raise ValueError("dual_distractor_review_requires_at_least_two_candidates_per_source")
        candidates_by_source_and_distractor: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for candidate_id in accepted_ids:
            candidate = row_by_id[candidate_id]
            candidates_by_source_and_distractor[
                (candidate["source_id"], candidate["distractor_id"])
            ].append(candidate_id)
        selected_by_source: Dict[str, List[str]] = defaultdict(list)
        for (source_id, distractor_id), candidate_ids in sorted(
            candidates_by_source_and_distractor.items()
        ):
            selected_by_source[source_id].append(
                min(
                    candidate_ids,
                    key=lambda value: sha256_value(
                        [source_manifest.get("seed", 0), "proxy-rerun", source_id, distractor_id, value]
                    ),
                )
            )
        accepted_ids = []
        for source_id, candidate_ids in sorted(selected_by_source.items()):
            if len(candidate_ids) < required_distractors:
                design_exclusions.extend({
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "reason": "fewer_than_two_codex_adjudicated_designated_distractors",
                } for candidate_id in candidate_ids)
                continue
            ranked = sorted(
                candidate_ids,
                key=lambda value: sha256_value(
                    [source_manifest.get("seed", 0), "dual-distractor", source_id, value]
                ),
            )[:required_distractors]
            accepted_ids.extend(ranked)
        accepted_ids.sort()
        if not accepted_ids:
            raise ValueError("dual_distractor_review_selected_no_complete_base_facts")

    accepted_after_design_filter = len(accepted_ids)
    if max_candidates_per_source is not None:
        if max_candidates_per_source < 1:
            raise ValueError("max_candidates_per_source_must_be_positive")
        grouped: Dict[str, List[str]] = defaultdict(list)
        for candidate_id in accepted_ids:
            grouped[row_by_id[candidate_id]["source_id"]].append(candidate_id)
        accepted_ids = sorted(
            candidate_id
            for source_id, candidate_ids in grouped.items()
            for candidate_id in sorted(
                candidate_ids,
                key=lambda value: sha256_value([source_manifest["seed"], "proxy-rerun", source_id, value]),
            )[:max_candidates_per_source]
        )

    checkpoint_names = (
        "translations.jsonl", "translation_reviews.jsonl", "verified_distractors.jsonl",
        "perturbation_generations.jsonl", "perturbations.jsonl",
    )
    reused_artifacts = {}
    for name in checkpoint_names:
        source_path = source_run_dir / name
        source_rows = read_jsonl(source_path)
        if any(not row.get("source_id") for row in source_rows):
            raise ValueError(f"proxy_rerun_checkpoint_missing_source_id:{name}")
        rows = [
            row for row in source_rows
            if row.get("source_id") in selected_source_ids
        ]
        write_jsonl(run_dir / name, rows)
        reused_artifacts[name] = {
            "row_count": len(rows),
            "source_row_count": len(source_rows),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }

    source_input_admission_policy = dict(
        source_manifest.get("input_admission_policy") or {}
    )
    source_input_admission_policy["allowed_splits_for_this_prepare"] = (
        allowed_input_splits
    )
    manifest = {
        **source_manifest,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": sha256_value(config),
        "runtime_sha256": _runtime_sha256(),
        "config_amendments": [
            *(source_manifest.get("config_amendments") or []),
            {
                "changed_at": utc_now(),
                "from_config_sha256": source_manifest.get("config_sha256"),
                "to_config_sha256": sha256_value(config),
                "reason": "three-arm design remediation and Codex proxy-review filtering",
            },
        ],
        "simulation_models": [model["model"] for model in config["model_roles"]["simulation"]["models"]],
        "simulation_languages": list(SIMULATION_LANGUAGES),
        "simulation_variants": list(SIMULATION_VARIANTS),
        **requested_design,
        "records": selected_records,
        "selected_count": len(selected_records),
        "selected_source_ids_sha256": sha256_value(sorted(selected_source_ids)),
        "input_admission_policy": source_input_admission_policy,
        "proxy_split_filter": {
            "allowed_splits": allowed_input_splits,
            "source_record_count": len(source_records),
            "selected_record_count": len(selected_records),
            "excluded_source_count": len(excluded_split_source_ids),
            "excluded_source_ids_sha256": sha256_value(excluded_split_source_ids),
        },
        "simulation_requires_codex_proxy_review": True,
        "endpoint_excluded_candidate_ids": [
            row["candidate_id"] for row in endpoint_exclusions
        ],
        "endpoint_exclusions": endpoint_exclusions,
        "design_exclusions": design_exclusions,
        "resumed_from_run_id": source_manifest["run_id"],
        "reused_upstream_artifacts": reused_artifacts,
        "codex_proxy_review": {
            "review_version": review["review_version"],
            "reviewer_type": "codex_proxy",
            "not_human_gold": True,
            "adjudicator_type": review.get("adjudicator_type", "codex"),
            "review_status": review.get("review_status", "codex_adjudicated"),
            "human_gold": False,
            "review_contract_version": review.get("review_contract_version"),
            "manipulation_family": manipulation_family,
            "review_blinded_to_simulation_results": review.get(
                "review_blinded_to_simulation_results", True
            ),
            "source_path": str(review_path.relative_to(project_root)),
            "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "accepted_perturbation_ids": accepted_ids,
            "accepted_before_sampling": accepted_before_sampling,
            "accepted_after_split_filter": accepted_after_split_filter,
            "accepted_after_endpoint_filter": accepted_after_endpoint_filter,
            "accepted_after_design_filter": accepted_after_design_filter,
            "selection_policy": {
                "independent_unit": "base_fact_id",
                "max_candidates_per_source": max_candidates_per_source,
                "strength_filter": strength_filter,
                "ranking": "sha256(seed,proxy-rerun,source_id,candidate_id)",
            },
        },
        "paths_not_taken_enabled": False,
        "holdout_enabled": False,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _resolve_static_freeze_artifact(
    static_manifest_path: Path,
    static_manifest: Dict[str, Any],
    artifact_name: str,
) -> Tuple[Path, Dict[str, Any]]:
    artifacts = static_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("static G0A freeze manifest is missing artifacts")
    binding = artifacts.get(artifact_name)
    if not isinstance(binding, dict):
        raise ValueError(f"static G0A freeze manifest is missing {artifact_name}")
    value = binding.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"static G0A artifact has invalid path: {artifact_name}")
    path = Path(value)
    if not path.is_absolute():
        path = static_manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if binding.get("sha256") != actual_sha256:
        raise ValueError(f"static G0A artifact SHA-256 mismatch: {artifact_name}")
    if binding.get("byte_count") is not None and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"static G0A artifact byte count mismatch: {artifact_name}")
    return path, dict(binding)


def _static_behavior_identity_policy(
    roster: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    return {
        "schema_version": STATIC_PROXY_ROUTE_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "capture_stage": "representative_probe",
        "full_run_startup_revalidation_required": True,
        "per_attempt_route_and_config_revalidation_required": True,
        "per_response_model_identity_required": True,
        "ollama_server_and_model_digest_required": any(
            provider == "ollama_local" for _, provider in roster
        ),
        "ollama_template_unknown_with_limitation_allowed": True,
        "credentials_or_endpoints_included": False,
    }


def _validate_static_behavior_config(
    config: Dict[str, Any], execution_mode: str
) -> List[Dict[str, Any]]:
    contract = _static_behavior_contract(execution_mode)
    expected_contract = {
        "roster_version": contract["roster_version"],
        "expected_model_count": contract["expected_model_count"],
        "expected_calls_per_model": contract["expected_calls_per_model"],
        "expected_simulation_call_count": contract[
            "expected_simulation_call_count"
        ],
        "model_aggregation": contract["model_aggregation"],
    }
    if config.get(contract["config_key"]) != expected_contract:
        raise ValueError(
            "static behavior config does not declare the current roster contract"
        )
    if _configured_allowed_splits(config) != ["development"]:
        raise ValueError("static proxy behavior run requires development-only input")
    roles = config.get("model_roles") or {}
    generation = roles.get("perturbation_generation") or {}
    simulation = roles.get("simulation") or {}
    validation = roles.get("perturbation_validation") or {}
    if _configured_manipulation_family(config) != EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        raise ValueError("static proxy behavior run requires explicit_false_assertion_v1")
    if int(generation.get("target_distractors_per_source", 0)) != 2:
        raise ValueError("static proxy behavior run requires two designated distractors")
    if int(generation.get("candidates_per_distractor", 0)) != 1:
        raise ValueError("static proxy behavior run requires one frozen context per distractor")
    if generation.get("matched_neutral") is not True:
        raise ValueError("static proxy behavior run requires matched neutral contexts")
    if simulation.get("endpoint") != "multi_option" or int(simulation.get("max_options", 0)) != 3:
        raise ValueError("static proxy behavior run requires the frozen three-option endpoint")
    if simulation.get("neutral_context_mode") != "matched_per_candidate":
        raise ValueError("static proxy behavior run requires candidate-specific neutral contexts")
    required_checks = set(validation.get("required_checks") or ())
    if not set(EXPLICIT_FALSE_ASSERTION_CHECKS) <= required_checks:
        raise ValueError("static proxy behavior config is missing explicit-false review checks")
    model_specs = simulation.get("models")
    if not isinstance(model_specs, list):
        raise ValueError("static proxy behavior run requires a model list")
    roster = tuple(
        (str(spec.get("model")), str(spec.get("provider_profile")))
        for spec in model_specs
        if isinstance(spec, dict)
    )
    if roster != contract["model_roster"] or len(roster) != len(model_specs):
        raise ValueError(contract["roster_error"])
    required_workers = contract["required_max_workers"]
    if required_workers is not None:
        execution = config.get("execution") or {}
        if int(execution.get("max_workers", 0)) != required_workers:
            raise ValueError("static Qwen3-8B behavior run requires max_workers=8")
        configured_provider_limits = execution.get("provider_max_concurrency") or {}
        if any(
            int(configured_provider_limits.get(provider, 0)) != expected
            for provider, expected in contract[
                "required_provider_max_concurrency"
            ].items()
        ):
            raise ValueError(
                "static Qwen3-8B behavior run requires qwen3_vllm concurrency=8"
            )
        if any(
            int(spec.get("max_retries", -1))
            != contract["required_model_max_retries"]
            for spec in model_specs
        ):
            raise ValueError("static Qwen3-8B behavior run requires max_retries=1")
    return [dict(spec) for spec in model_specs]


def _validate_static_proxy_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _validate_static_behavior_config(config, STATIC_PROXY_EXECUTION_MODE)


def _validate_static_single_qwen3_config(
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return _validate_static_behavior_config(
        config, STATIC_SINGLE_QWEN3_EXECUTION_MODE
    )


def _validate_static_mcq_row(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_fact_id = formal_admission.explicit_base_fact_id(row, "static G0A row")
    if row.get("split_assignment") not in INPUT_SPLIT_ASSIGNMENTS:
        raise ValueError(f"static G0A row has invalid split: {base_fact_id}")
    if (
        row.get("canonical_status") != "frozen"
        or row.get("static_stimulus_status") != "frozen"
        or row.get("static_generation_complete") is not True
        or row.get("behavior_authorized") is not False
        or row.get("simulation_authorized") is not False
        or row.get("human_gold") is not False
    ):
        raise ValueError(f"static G0A row is not a frozen behavior-blind stimulus: {base_fact_id}")
    adjudication = row.get("codex_adjudication")
    if (
        not isinstance(adjudication, dict)
        or adjudication.get("reviewer_type") != "codex_proxy"
        or adjudication.get("human_gold") is not False
        or adjudication.get("decision") not in {"accept", "revise"}
    ):
        raise ValueError(f"static G0A row has invalid Codex adjudication: {base_fact_id}")
    mcq = row.get("static_mcq")
    if (
        not isinstance(mcq, dict)
        or mcq.get("schema_version") != STATIC_G0A_MCQ_SCHEMA_VERSION
        or mcq.get("manipulation_family") != EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
        or mcq.get("option_order_policy") != "base-fact-option-sha256-v1"
    ):
        raise ValueError(f"static G0A row has invalid MCQ contract: {base_fact_id}")
    options = mcq.get("options")
    if not isinstance(options, list) or len(options) != 3:
        raise ValueError(f"static G0A row must contain exactly three options: {base_fact_id}")
    if [option.get("choice") for option in options if isinstance(option, dict)] != ["A", "B", "C"]:
        raise ValueError(f"static G0A option order is invalid: {base_fact_id}")
    if sum(option.get("kind") == "gold" for option in options) != 1:
        raise ValueError(f"static G0A row must contain exactly one gold option: {base_fact_id}")
    option_ids = [option.get("option_id") for option in options]
    if any(not isinstance(value, str) or not value for value in option_ids) or len(set(option_ids)) != 3:
        raise ValueError(f"static G0A option identifiers are invalid: {base_fact_id}")
    variants = mcq.get("variants")
    if not isinstance(variants, list) or len(variants) != STATIC_PROXY_VARIANTS_PER_FACT:
        raise ValueError(f"static G0A row must contain two variants: {base_fact_id}")
    variant_by_id: Dict[str, Dict[str, Any]] = {}
    candidate_ids = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError(f"static G0A variant is invalid: {base_fact_id}")
        distractor_id = variant.get("distractor_id")
        candidate_id = variant.get("selected_candidate_id")
        if (
            not isinstance(distractor_id, str)
            or distractor_id not in option_ids
            or not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or distractor_id in variant_by_id
            or candidate_id in candidate_ids
        ):
            raise ValueError(f"static G0A variant identity is invalid: {base_fact_id}")
        if (
            variant.get("reviewer_type") != "codex_proxy"
            or variant.get("human_gold") is not False
            or variant.get("verified") is not True
            or variant.get("verification_status") != "codex_adjudicated"
        ):
            raise ValueError(f"static G0A variant review identity is invalid: {base_fact_id}:{distractor_id}")
        for field in (
            "targeted_context_en",
            "targeted_context_zh",
            "neutral_context_en",
            "neutral_context_zh",
        ):
            if not isinstance(variant.get(field), str) or not variant[field].strip():
                raise ValueError(f"static G0A variant is missing {field}: {base_fact_id}:{distractor_id}")
        variant_by_id[distractor_id] = dict(variant)
        candidate_ids.add(candidate_id)
    distractor_option_ids = {
        str(option["option_id"]) for option in options if option.get("kind") == "distractor"
    }
    if set(variant_by_id) != distractor_option_ids or len(distractor_option_ids) != 2:
        raise ValueError(f"static G0A variants do not match both distractor options: {base_fact_id}")
    for option in options:
        if option.get("kind") == "gold":
            expected_en, expected_zh = row.get("answer_en"), row.get("answer_zh")
        else:
            variant = variant_by_id[str(option["option_id"])]
            expected_en, expected_zh = variant.get("text_en"), variant.get("text_zh")
        if option.get("text_en") != expected_en or option.get("text_zh") != expected_zh:
            raise ValueError(f"static G0A option text is inconsistent: {base_fact_id}")
    stimuli = mcq.get("behavior_inputs")
    if not isinstance(stimuli, list) or len(stimuli) != STATIC_PROXY_STIMULI_PER_FACT:
        raise ValueError(f"static G0A row must contain ten behavior inputs: {base_fact_id}")
    expected = {
        (language, "original", None)
        for language in SIMULATION_LANGUAGES
    } | {
        (language, arm, distractor_id)
        for distractor_id in variant_by_id
        for language in SIMULATION_LANGUAGES
        for arm in ("neutral", "targeted")
    }
    observed = set()
    stimulus_ids = set()
    for stimulus in stimuli:
        if not isinstance(stimulus, dict):
            raise ValueError(f"static G0A behavior input is invalid: {base_fact_id}")
        language = stimulus.get("language")
        arm = stimulus.get("arm")
        distractor_id = stimulus.get("distractor_id")
        identity = (language, arm, distractor_id)
        if identity in observed or identity not in expected:
            raise ValueError(f"static G0A behavior input coverage is invalid: {base_fact_id}")
        observed.add(identity)
        stimulus_id = stimulus.get("stimulus_id")
        if not isinstance(stimulus_id, str) or not stimulus_id or stimulus_id in stimulus_ids:
            raise ValueError(f"static G0A stimulus_id is invalid: {base_fact_id}")
        stimulus_ids.add(stimulus_id)
        if stimulus.get("options") != options:
            raise ValueError(f"static G0A behavior input changed frozen option order: {stimulus_id}")
        expected_prompt = row["prompt_en" if language == "en" else "prompt_zh"]
        if stimulus.get("prompt") != expected_prompt:
            raise ValueError(f"static G0A behavior input changed frozen prompt: {stimulus_id}")
        if arm == "original":
            if stimulus.get("context") != "":
                raise ValueError(f"static G0A Original must have empty context: {stimulus_id}")
        else:
            expected_context = variant_by_id[str(distractor_id)][f"{arm}_context_{language}"]
            if stimulus.get("context") != expected_context:
                raise ValueError(f"static G0A behavior input changed frozen context: {stimulus_id}")
    if observed != expected:
        raise ValueError(f"static G0A behavior input coverage is incomplete: {base_fact_id}")
    return [dict(option) for option in options], [dict(stimulus) for stimulus in stimuli]


def _prepare_static_behavior_run(
    config: Dict[str, Any],
    project_root: Path,
    static_freeze_manifest_path: Path,
    run_dir: Path,
    run_id: str,
    execution_mode: str,
) -> Dict[str, Any]:
    """Bind and expand the exact frozen G0A stimuli for Development-only proxy behavior."""

    contract = _static_behavior_contract(execution_mode)
    roster = contract["model_roster"]
    run_dir = Path(run_dir).resolve()
    target_manifest_path = run_dir / "run_manifest.json"
    if target_manifest_path.exists():
        raise ValueError(f"target_run_already_prepared:{run_id}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"target_run_directory_not_empty:{run_id}")
    model_specs = _validate_static_behavior_config(config, execution_mode)
    static_freeze_manifest_path = Path(static_freeze_manifest_path).resolve()
    static_manifest = read_json(static_freeze_manifest_path)
    if (
        static_manifest.get("schema_version") != STATIC_G0A_FREEZE_MANIFEST_SCHEMA_VERSION
        or static_manifest.get("status") != "g0a_static_stimulus_frozen"
        or static_manifest.get("static_frozen") is not True
        or static_manifest.get("behavior_authorized") is not False
        or static_manifest.get("simulation_authorized") is not False
        or static_manifest.get("reviewer_type") != "codex_proxy"
        or static_manifest.get("human_gold") is not False
    ):
        raise ValueError("static G0A freeze manifest is not an eligible frozen proxy source")
    bundle_path, bundle_artifact = _resolve_static_freeze_artifact(
        static_freeze_manifest_path, static_manifest, "frozen_static_bundle.jsonl"
    )
    review_manifest_path, _ = _resolve_static_freeze_artifact(
        static_freeze_manifest_path, static_manifest, "review_freeze_manifest.json"
    )
    split_manifest_path, _ = _resolve_static_freeze_artifact(
        static_freeze_manifest_path, static_manifest, "split_freeze_manifest.json"
    )
    bundle_rows, bundle_snapshot = _read_jsonl_snapshot(bundle_path)
    bundle_identity = formal_admission.bundle_identity(
        bundle_path,
        INPUT_BUNDLE_SCHEMA_VERSION,
        bundle_rows,
        input_bundle_sha256=bundle_snapshot["sha256"],
    )
    frozen_binding = static_manifest.get("frozen_bundle")
    if not isinstance(frozen_binding, dict):
        raise ValueError("static G0A freeze manifest is missing frozen_bundle")
    for field in ("sha256", "schema_version", "record_count", "base_fact_ids_sha256"):
        if frozen_binding.get(field) != bundle_identity.get(field):
            raise ValueError(f"static G0A frozen_bundle.{field} mismatch")
    if bundle_artifact.get("sha256") != bundle_identity["sha256"]:
        raise ValueError("static G0A artifact and frozen_bundle SHA-256 disagree")
    admission = formal_admission.validate_external_formal_admission(
        input_bundle_path=bundle_path,
        input_schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
        records=bundle_rows,
        input_bundle_sha256=bundle_identity["sha256"],
        review_freeze_manifest_path=review_manifest_path,
        split_freeze_manifest_path=split_manifest_path,
    )
    if admission.get("authorized") is not True:
        raise ValueError(f"static G0A formal admission is not authorized: {admission.get('blockers')}")
    split_counts = Counter(str(row.get("split_assignment")) for row in bundle_rows)
    if len(bundle_rows) != STATIC_PROXY_STATIC_FACT_COUNT or dict(split_counts) != STATIC_PROXY_EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "static G0A proxy source must contain the frozen 96/32/32 split"
        )
    if (
        static_manifest.get("base_fact_count") != STATIC_PROXY_STATIC_FACT_COUNT
        or static_manifest.get("variant_count")
        != STATIC_PROXY_VARIANTS_PER_FACT * STATIC_PROXY_STATIC_FACT_COUNT
        or static_manifest.get("unique_behavior_input_count")
        != STATIC_PROXY_STIMULI_PER_FACT * STATIC_PROXY_STATIC_FACT_COUNT
    ):
        raise ValueError("static G0A freeze manifest count contract mismatch")

    development_rows = sorted(
        (row for row in bundle_rows if row.get("split_assignment") == "development"),
        key=lambda row: formal_admission.explicit_base_fact_id(row, "static G0A row"),
    )
    if len(development_rows) != STATIC_PROXY_EXPECTED_SPLIT_COUNTS["development"]:
        raise ValueError("static G0A Development selection must contain exactly 96 facts")
    expanded_inputs: List[Dict[str, Any]] = []
    perturbations: List[Dict[str, Any]] = []
    seen_candidate_ids = set()
    for row in development_rows:
        base_fact_id = formal_admission.explicit_base_fact_id(row, "static G0A row")
        source_id = str(row["source_id"])
        options, stimuli = _validate_static_mcq_row(row)
        mcq = row["static_mcq"]
        variant_by_id = {
            str(variant["distractor_id"]): variant for variant in mcq["variants"]
        }
        candidate_ids = sorted(str(variant["selected_candidate_id"]) for variant in mcq["variants"])
        for variant in mcq["variants"]:
            candidate_id = str(variant["selected_candidate_id"])
            if candidate_id in seen_candidate_ids:
                raise ValueError(f"duplicate frozen selected_candidate_id: {candidate_id}")
            seen_candidate_ids.add(candidate_id)
            perturbations.append(
                {
                    "candidate_id": candidate_id,
                    "base_fact_id": base_fact_id,
                    "source_id": source_id,
                    "distractor_id": variant["distractor_id"],
                    "manipulation_family": EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY,
                    "candidate": {
                        "english_context": variant["targeted_context_en"],
                        "chinese_context": variant["targeted_context_zh"],
                        "neutral_english_context": variant["neutral_context_en"],
                        "neutral_chinese_context": variant["neutral_context_zh"],
                    },
                    "terminal_status": "completed",
                    "reviewer_type": "codex_proxy",
                    "review_status": "codex_adjudicated",
                    "human_gold": False,
                    "source_static_row_sha256": sha256_value(row),
                }
            )
        for model_spec in model_specs:
            for stimulus in stimuli:
                language = str(stimulus["language"])
                option_mapping = {
                    str(option["choice"]): str(option[f"text_{language}"])
                    for option in stimulus["options"]
                }
                option_ids = [str(option["option_id"]) for option in stimulus["options"]]
                correct_choice = next(
                    str(option["choice"])
                    for option in stimulus["options"]
                    if option.get("kind") == "gold"
                )
                distractor_id = stimulus.get("distractor_id")
                target_choice = (
                    next(
                        str(option["choice"])
                        for option in stimulus["options"]
                        if option.get("option_id") == distractor_id
                    )
                    if distractor_id is not None
                    else None
                )
                candidate_id = (
                    str(variant_by_id[str(distractor_id)]["selected_candidate_id"])
                    if distractor_id is not None
                    else None
                )
                simulation_id = sha256_value(
                    [
                        bundle_identity["sha256"],
                        stimulus["stimulus_id"],
                        model_spec["provider_profile"],
                        model_spec["model"],
                    ]
                )[:24]
                expanded_inputs.append(
                    {
                        "schema_version": STATIC_PROXY_BEHAVIOR_INPUT_SCHEMA_VERSION,
                        "simulation_id": simulation_id,
                        "static_stimulus_id": stimulus["stimulus_id"],
                        "static_behavior_input_sha256": sha256_value(stimulus),
                        "source_static_row_sha256": sha256_value(row),
                        "frozen_bundle_sha256": bundle_identity["sha256"],
                        "candidate_id": candidate_id,
                        "candidate_ids": candidate_ids if candidate_id is None else [candidate_id],
                        "base_fact_id": base_fact_id,
                        "source_id": source_id,
                        "split_assignment": "development",
                        "distractor_id": distractor_id,
                        "designated_distractor_ids": sorted(variant_by_id),
                        "model_spec": dict(model_spec),
                        "language": language,
                        "variant": stimulus["arm"],
                        "simulation_scope": (
                            "shared_base_fact_baseline"
                            if stimulus["arm"] == "original"
                            else "candidate"
                        ),
                        "options": option_mapping,
                        "option_ids": option_ids,
                        "option_count": len(option_ids),
                        "endpoint": "multi_option",
                        "correct_choice": correct_choice,
                        "target_choice": target_choice,
                        "prompt": _simulation_prompt(
                            str(stimulus["prompt"]),
                            str(stimulus["context"]),
                            option_mapping,
                            language,
                        ),
                    }
                )
    expected_stimuli = contract["expected_calls_per_model"]
    expected_calls = expected_stimuli * len(model_specs)
    simulation_ids = [row["simulation_id"] for row in expanded_inputs]
    if len(expanded_inputs) != expected_calls or len(simulation_ids) != len(set(simulation_ids)):
        raise ValueError(
            "static proxy behavior expansion did not produce "
            f"{contract['expected_simulation_call_count']:,} unique calls"
        )
    calls_by_model = Counter(row["model_spec"]["model"] for row in expanded_inputs)
    if any(calls_by_model.get(model, 0) != expected_stimuli for model, _ in roster):
        raise ValueError("static proxy behavior expansion did not produce 960 calls per model")
    behavior_inputs_path = run_dir / "proxy_behavior_inputs.jsonl"
    perturbations_path = run_dir / "perturbations.jsonl"
    simulation_results_path = run_dir / "simulation_results.jsonl"
    write_jsonl(behavior_inputs_path, expanded_inputs)
    write_jsonl(perturbations_path, perturbations)
    write_jsonl(simulation_results_path, [])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_mode": execution_mode,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": sha256_value(config),
        "runtime_sha256": _runtime_sha256(),
        "proxy_roster_version": contract["roster_version"],
        "expected_model_count": contract["expected_model_count"],
        "expected_calls_per_model": contract["expected_calls_per_model"],
        "expected_simulation_call_count": contract[
            "expected_simulation_call_count"
        ],
        "model_aggregation": contract["model_aggregation"],
        "g0a_static_freeze_manifest": {
            "path": str(static_freeze_manifest_path),
            "sha256": hashlib.sha256(static_freeze_manifest_path.read_bytes()).hexdigest(),
            "schema_version": STATIC_G0A_FREEZE_MANIFEST_SCHEMA_VERSION,
        },
        "input_bundle": _artifact_path_label(project_root, bundle_path),
        "input_bundle_schema_version": INPUT_BUNDLE_SCHEMA_VERSION,
        "input_bundle_sha256": bundle_identity["sha256"],
        "input_bundle_record_count": len(bundle_rows),
        "external_formal_admission": admission,
        "review_freeze_manifest": admission["review_freeze_manifest"],
        "split_freeze_manifest": admission["split_freeze_manifest"],
        "records": development_rows,
        "selected_count": len(development_rows),
        "selected_source_ids_sha256": sha256_value(
            sorted(str(row["source_id"]) for row in development_rows)
        ),
        "proxy_split_filter": {
            "allowed_splits": ["development"],
            "source_split_counts": dict(split_counts),
            "selected_record_count": len(development_rows),
            "validation_behavior_input_count": 0,
            "sealed_behavior_input_count": 0,
        },
        "prebuilt_proxy_behavior_inputs": {
            "path": behavior_inputs_path.name,
            "sha256": hashlib.sha256(behavior_inputs_path.read_bytes()).hexdigest(),
            "schema_version": STATIC_PROXY_BEHAVIOR_INPUT_SCHEMA_VERSION,
            "record_count": len(expanded_inputs),
            "simulation_ids_sha256": sha256_value(sorted(simulation_ids)),
            "static_stimulus_count": expected_stimuli,
            "calls_per_model": {
                model: calls_by_model[model] for model, _ in roster
            },
        },
        "simulation_models": [spec["model"] for spec in model_specs],
        "simulation_languages": list(SIMULATION_LANGUAGES),
        "simulation_variants": list(SIMULATION_VARIANTS),
        "shared_simulation_variants": ["original"],
        "neutral_context_mode": "matched_per_candidate",
        "simulation_endpoint": "multi_option",
        "simulation_max_options": 3,
        "manipulation_family": EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY,
        "designated_distractors_per_base_fact": STATIC_PROXY_VARIANTS_PER_FACT,
        "statistical_unit": "base_fact_id",
        "codex_proxy_review": {
            "reviewer_type": "codex_proxy",
            "not_human_gold": True,
            "adjudicator_type": "codex",
            "review_status": "codex_adjudicated",
            "human_gold": False,
            "review_blinded_to_simulation_results": True,
            "accepted_perturbation_ids": sorted(seen_candidate_ids),
            "source_static_freeze_sha256": hashlib.sha256(
                static_freeze_manifest_path.read_bytes()
            ).hexdigest(),
        },
        "static_generation_authorized": False,
        "proxy_behavior_authorized": True,
        "behavior_authorized": True,
        "simulation_authorized": True,
        "paths_not_taken_enabled": False,
        "holdout_enabled": False,
        "exact_hf_evidence": False,
        "pnt_authorized": False,
        "pnt_eligible": False,
    }
    if execution_mode == STATIC_SINGLE_QWEN3_EXECUTION_MODE:
        manifest["diagnostic_only"] = True
        manifest["behavior_evidence_class"] = "single_model_api_behavior"
        manifest["single_model_behavior_scope"] = {
            "purpose": "development_only_behavioral_defect_validation",
            "prior_proxy_behavior_results_consumed": False,
            "prior_proxy_labels_consumed": False,
            "validation_or_sealed_consumed": False,
            "exact_hf_checkpoint_identity_frozen": False,
        }
        manifest["execution_limits"] = {
            "max_workers": int(config["execution"]["max_workers"]),
            "provider_max_concurrency": {
                provider: int(
                    config["execution"]["provider_max_concurrency"][provider]
                )
                for _, provider in roster
            },
            "max_retries_by_model": {
                str(spec["model"]): int(spec["max_retries"])
                for spec in model_specs
            },
        }
    _, representative_selection = _select_static_proxy_representative_inputs(
        manifest, expanded_inputs
    )
    manifest["representative_probe"] = {
        "required_before_full_run": True,
        "selection": representative_selection,
        "expected_model_count": contract["expected_model_count"],
        "results_path": STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME,
        "manifest_path": STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME,
    }
    manifest["proxy_route_runtime_identity_policy"] = (
        _static_behavior_identity_policy(roster)
    )
    write_json(target_manifest_path, manifest)
    return manifest


def prepare_static_proxy_run(
    config: Dict[str, Any],
    project_root: Path,
    static_freeze_manifest_path: Path,
    run_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    return _prepare_static_behavior_run(
        config,
        project_root,
        static_freeze_manifest_path,
        run_dir,
        run_id,
        STATIC_PROXY_EXECUTION_MODE,
    )


def prepare_static_single_qwen3_run(
    config: Dict[str, Any],
    project_root: Path,
    static_freeze_manifest_path: Path,
    run_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Prepare all 960 frozen Development stimuli for Qwen3-8B behavior only."""

    return _prepare_static_behavior_run(
        config,
        project_root,
        static_freeze_manifest_path,
        run_dir,
        run_id,
        STATIC_SINGLE_QWEN3_EXECUTION_MODE,
    )


def prepare_simulation_rerun(
    config: Dict[str, Any],
    project_root: Path,
    source_run_dir: Path,
    run_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Reuse a frozen candidate set while replacing only the Simulation panel."""
    source_manifest_path = source_run_dir / "run_manifest.json"
    source_manifest = read_json(source_manifest_path)
    review = source_manifest.get("codex_proxy_review") or {}
    accepted_ids = sorted(review.get("accepted_perturbation_ids") or [])
    if (
        review.get("reviewer_type") != "codex_proxy"
        or review.get("not_human_gold") is not True
        or not accepted_ids
    ):
        raise ValueError("source_run_has_no_frozen_codex_candidate_set")

    source_perturbations = read_jsonl(source_run_dir / "perturbations.jsonl")
    available_ids = {row.get("candidate_id") for row in source_perturbations}
    if not set(accepted_ids).issubset(available_ids):
        raise ValueError("source_run_missing_frozen_candidates")

    simulation_role = config["model_roles"]["simulation"]
    neutral_mode = simulation_role.get("neutral_context_mode", "generic_shared")
    shared_variants = [
        "original"
    ] if neutral_mode == "matched_per_candidate" else list(SHARED_SIMULATION_VARIANTS)
    requested_design = {
        "simulation_endpoint": simulation_role.get("endpoint", "binary"),
        "simulation_max_options": int(simulation_role.get("max_options", 3)),
        "neutral_context_mode": neutral_mode,
        "shared_simulation_variants": shared_variants,
    }
    for field, requested in requested_design.items():
        if source_manifest.get(field) != requested:
            raise ValueError(f"simulation_rerun_design_mismatch:{field}")

    target_manifest_path = run_dir / "run_manifest.json"
    if target_manifest_path.exists():
        manifest = read_json(target_manifest_path)
        if manifest.get("simulation_rerun", {}).get("source_run_id") != source_manifest["run_id"]:
            raise ValueError("simulation_rerun_source_mismatch")
        requested_models = [model["model"] for model in simulation_role["models"]]
        if manifest.get("simulation_models") != requested_models:
            raise ValueError("simulation_rerun_model_panel_mismatch")
        for field, requested in requested_design.items():
            if manifest.get(field) != requested:
                raise ValueError(f"simulation_rerun_design_mismatch:{field}")
        previous_hash = manifest.get("config_sha256")
        next_hash = sha256_value(config)
        if previous_hash != next_hash:
            manifest.setdefault("config_amendments", []).append({
                "changed_at": utc_now(),
                "from_config_sha256": previous_hash,
                "to_config_sha256": next_hash,
                "reason": "update Simulation request settings; completed checkpoints retained",
            })
        manifest["config_version"] = config["config_version"]
        manifest["config_sha256"] = next_hash
        manifest["runtime_sha256"] = _runtime_sha256()
        write_json(target_manifest_path, manifest)
        return manifest

    checkpoint_names = (
        "translations.jsonl", "translation_reviews.jsonl", "verified_distractors.jsonl",
        "perturbation_generations.jsonl", "perturbations.jsonl",
    )
    reused_artifacts = {}
    for name in checkpoint_names:
        source_path = source_run_dir / name
        rows = read_jsonl(source_path)
        write_jsonl(run_dir / name, rows)
        reused_artifacts[name] = {
            "row_count": len(rows),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }

    manifest = {
        **source_manifest,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": sha256_value(config),
        "runtime_sha256": _runtime_sha256(),
        "config_amendments": [
            *(source_manifest.get("config_amendments") or []),
            {
                "changed_at": utc_now(),
                "from_config_sha256": source_manifest.get("config_sha256"),
                "to_config_sha256": sha256_value(config),
                "reason": "replace only the Simulation model panel over frozen candidate records",
            },
        ],
        "simulation_models": [model["model"] for model in simulation_role["models"]],
        **requested_design,
        "simulation_languages": list(SIMULATION_LANGUAGES),
        "simulation_variants": list(SIMULATION_VARIANTS),
        "resumed_from_run_id": source_manifest["run_id"],
        "reused_upstream_artifacts": reused_artifacts,
        "simulation_rerun": {
            "source_run_id": source_manifest["run_id"],
            "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            "frozen_candidate_ids": accepted_ids,
        },
        "paths_not_taken_enabled": False,
        "holdout_enabled": False,
    }
    write_json(target_manifest_path, manifest)
    return manifest


def export_codex_review_packet(run_dir: Path) -> Dict[str, Any]:
    """Export a Simulation-blind source bundle for final Codex proxy adjudication."""
    manifest = read_json(run_dir / "run_manifest.json")
    manipulation_family = str(
        manifest.get("manipulation_family")
        or TRUTHFUL_SALIENCE_MANIPULATION_FAMILY
    )
    translations = {
        row["source_id"]: row["parsed_response"]
        for row in read_jsonl(run_dir / "translations.jsonl")
        if row.get("terminal_status") == "completed" and row.get("parsed_response")
    }
    accepted_distractors = [
        row for row in read_jsonl(run_dir / "verified_distractors.jsonl")
        if row.get("terminal_status") == "completed"
        and row.get("parsed_response", {}).get("decision") == "accept"
        and all(row.get("parsed_response", {}).get("checks", {}).values())
    ]
    distractors_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    accepted_distractor_ids = set()
    for row in accepted_distractors:
        distractors_by_source[row["source_id"]].append(row)
        accepted_distractor_ids.add(row["item_id"])
    candidates = [
        row for row in read_jsonl(run_dir / "perturbations.jsonl")
        if row.get("terminal_status") == "completed"
        and row.get("distractor_id") in accepted_distractor_ids
        and len(distractors_by_source[row["source_id"]]) >= 2
    ]
    if manifest.get("perturbation_auto_judge_hard_gate", True):
        candidates = [
            row for row in candidates
            if row.get("parsed_response", {}).get("decision") == "accept"
            and all(row.get("parsed_response", {}).get("checks", {}).values())
        ]
    candidates_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_source[row["source_id"]].append(row)
    source_by_id = {row["source_id"]: row for row in manifest["records"]}
    packets = []
    for source_id in sorted(candidates_by_source):
        source = source_by_id[source_id]
        packets.append({
            "source_id": source_id,
            "base_fact_id": source.get("base_fact_id") or source_id,
            "source": {
                key: source.get(key)
                for key in (
                    "source_dataset", "answer_type", "prompt_en", "answer_en",
                    "canonical_fact", "relation_raw", "relation_normalized",
                )
            },
            "translation": translations.get(source_id),
            "accepted_distractors": [
                {
                    "distractor_id": row["item_id"],
                    "text_en": row["distractor_en"],
                    "text_zh": row["distractor_zh"],
                    "automatic_review": row.get("parsed_response"),
                }
                for row in sorted(distractors_by_source[source_id], key=lambda value: value["item_id"])
            ],
            "candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "distractor_id": row["distractor_id"],
                    "candidate": row["candidate"],
                    "automatic_review": row.get("parsed_response"),
                }
                for row in sorted(candidates_by_source[source_id], key=lambda value: value["candidate_id"])
            ],
        })
    packet = {
        "schema_version": "factual-perturbation-codex-review-packet-v1",
        "run_id": manifest["run_id"],
        "created_at": utc_now(),
        "reviewer_type": "codex_proxy",
        "adjudicator_type": "codex",
        "review_status": "codex_adjudication_pending",
        "human_gold": False,
        "not_human_gold": True,
        "review_blinded_to_simulation_results": True,
        "review_contract_version": manipulation_family,
        "manipulation_family": manipulation_family,
        "allowed_decisions": ["accept", "reject", "defer"],
        "automatic_judge_is_advisory": not manifest.get("perturbation_auto_judge_hard_gate", True),
        "selection_policy": (
            "Select exactly one valid Targeted/Neutral pair for each of the two designated distractors "
            "within an eligible base_fact_id before any Simulation call."
            if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            else "Select at most one valid candidate per source before any Simulation call."
        ),
        "statistical_unit": "base_fact_id",
        "required_checks": {
            "source": [
                "canonical_fact_usable", "answer_unique", "three_semantically_distinct_options",
                "same_answer_type_or_plausible_foil",
            ],
            "candidate": (
                list(EXPLICIT_FALSE_ASSERTION_CHECKS)
                if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                else [
                    "truth_preserved", "bilingual_equivalent", "target_distractor_salience",
                    "matched_neutral_has_no_answer_or_distractor_cue", "l2_strength",
                    "no_prompt_duplication", "natural",
                ]
            ),
        },
        "records": packets,
    }
    output_path = run_dir / "codex_proxy_review_input_v5.json"
    write_json(output_path, packet)
    return {
        "run_id": manifest["run_id"],
        "output_path": str(output_path),
        "source_count": len(packets),
        "candidate_count": sum(len(row["candidates"]) for row in packets),
        "automatic_judge_is_advisory": packet["automatic_judge_is_advisory"],
        "review_blinded_to_simulation_results": True,
        "review_contract_version": manipulation_family,
        "statistical_unit": "base_fact_id",
    }


def prepare_strength_rerun(
    config: Dict[str, Any],
    project_root: Path,
    source_run_dir: Path,
    run_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Reuse reviewed items/distractors but regenerate matched-neutral strength pairs."""
    source_manifest = read_json(source_run_dir / "run_manifest.json")
    target_manifest_path = run_dir / "run_manifest.json"
    if target_manifest_path.exists():
        manifest = read_json(target_manifest_path)
        if manifest.get("resumed_from_run_id") != source_manifest.get("run_id"):
            raise ValueError("strength_rerun_source_mismatch")
        previous_hash = manifest.get("config_sha256")
        next_hash = sha256_value(config)
        if previous_hash != next_hash:
            manifest.setdefault("config_amendments", []).append({
                "changed_at": utc_now(),
                "from_config_sha256": previous_hash,
                "to_config_sha256": next_hash,
                "reason": "increase output budget after matched-pair structured responses were truncated",
            })
        manifest["config_version"] = config["config_version"]
        manifest["config_sha256"] = next_hash
        manifest["runtime_sha256"] = _runtime_sha256()
        write_json(target_manifest_path, manifest)
        return manifest
    review = source_manifest.get("codex_proxy_review") or {}
    accepted_candidate_ids = set(review.get("accepted_perturbation_ids") or [])
    if not accepted_candidate_ids:
        raise ValueError("source_run_has_no_codex_accepted_candidates")
    source_perturbations = {
        row["candidate_id"]: row for row in read_jsonl(source_run_dir / "perturbations.jsonl")
        if row.get("candidate_id") in accepted_candidate_ids
    }
    if set(source_perturbations) != accepted_candidate_ids:
        raise ValueError("source_run_missing_accepted_perturbations")
    selected_sources = {row["source_id"] for row in source_perturbations.values()}
    selected_distractors = {row["distractor_id"] for row in source_perturbations.values()}

    filters = {
        "translations.jsonl": lambda row: row.get("source_id") in selected_sources,
        "translation_reviews.jsonl": lambda row: row.get("source_id") in selected_sources,
        "verified_distractors.jsonl": lambda row: row.get("item_id") in selected_distractors,
    }
    reused_artifacts = {}
    for name, keep in filters.items():
        source_path = source_run_dir / name
        rows = [row for row in read_jsonl(source_path) if keep(row)]
        write_jsonl(run_dir / name, rows)
        reused_artifacts[name] = {
            "row_count": len(rows),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }

    neutral_mode = config["model_roles"]["simulation"].get("neutral_context_mode", "generic_shared")
    if neutral_mode != "matched_per_candidate":
        raise ValueError("strength_rerun_requires_matched_per_candidate_neutral")
    manifest = {
        **source_manifest,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": sha256_value(config),
        "runtime_sha256": _runtime_sha256(),
        "config_amendments": [
            *(source_manifest.get("config_amendments") or []),
            {
                "changed_at": utc_now(),
                "from_config_sha256": source_manifest.get("config_sha256"),
                "to_config_sha256": sha256_value(config),
                "reason": "regenerate frozen distractors with l1/l2 targeted contexts and matched neutral controls",
            },
        ],
        "selected_count": len(selected_sources),
        "records": [row for row in source_manifest["records"] if row["source_id"] in selected_sources],
        "simulation_models": [model["model"] for model in config["model_roles"]["simulation"]["models"]],
        "simulation_languages": list(SIMULATION_LANGUAGES),
        "simulation_variants": list(SIMULATION_VARIANTS),
        "shared_simulation_variants": ["original"],
        "neutral_context_mode": neutral_mode,
        "simulation_endpoint": config["model_roles"]["simulation"].get("endpoint", "binary"),
        "simulation_max_options": int(config["model_roles"]["simulation"].get("max_options", 3)),
        "simulation_requires_codex_proxy_review": True,
        "resumed_from_run_id": source_manifest["run_id"],
        "reused_upstream_artifacts": reused_artifacts,
        "source_candidate_ids": sorted(accepted_candidate_ids),
        "codex_proxy_review": None,
        "paths_not_taken_enabled": False,
        "holdout_enabled": False,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _translation_source_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    """Project only candidates eligible for translation into model prompts."""

    return {
        "prompt_en": item["prompt_en"],
        "answer_en": item["answer_en"],
        "answer_aliases_en": item["answer_aliases_en"],
        "distractors": [{"distractor_id": d["distractor_id"], "text_en": d["text_en"]} for d in item["distractors"]],
    }


def _translation_prompt(item: Dict[str, Any]) -> str:
    payload = _translation_source_payload(item)
    return (
        "Translate this factual-recall item into natural Simplified Chinese. Preserve the incomplete/open-answer "
        "prompt form, factual meaning, answer identity, and distractor identity. Do not answer inside prompt_zh. "
        "Return JSON only with prompt_zh, answer_zh, answer_aliases_zh (array), and distractors "
        "(array of objects with distractor_id and text_zh). Input: " + canonical_json(payload)
    )


def _translation_review_prompt(item: Dict[str, Any], translation: Dict[str, Any]) -> str:
    payload = {"source": _translation_source_payload(item), "translation": translation}
    return (
        "Independently review this English/Chinese factual-recall translation. Accept only if the Chinese prompt "
        "is natural, does not reveal the answer, preserves answer identity and prompt form, and every distractor "
        "translation preserves its identity. Return JSON only with decision (accept|reject), issues (array), and "
        "checks containing prompt_faithful, answer_faithful, distractors_faithful, answer_not_leaked, naturalness "
        "as booleans. Input: " + canonical_json(payload)
    )


def _translation_repair_prompt(
    item: Dict[str, Any], translation: Dict[str, Any], review: Dict[str, Any]
) -> str:
    payload = {
        "source": _translation_source_payload(item),
        "translation": translation,
        "review": review,
    }
    return (
        "Repair this Simplified-Chinese factual-recall translation using the independent review. Correct only the "
        "reported fidelity, leakage, or naturalness problems while preserving the English prompt form, answer "
        "identity, distractor identities, and every distractor_id. The concatenation of prompt_zh and answer_zh "
        "must form one natural, complete factual statement. Keep every relation-bearing phrase or queried term "
        "from the original prompt in prompt_zh; never move such content into answer_zh. Keep prompt_zh as an "
        "incomplete completion stem rather than rewriting it as a question. If translating the queried term itself "
        "into Chinese would directly reveal its definition or answer, retain that term in English in prompt_zh. "
        "Return JSON only with prompt_zh, answer_zh, "
        "answer_aliases_zh (array), and distractors (array of objects with distractor_id and text_zh). Input: "
        + canonical_json(payload)
    )


TRANSLATION_CHECKS = ("prompt_faithful", "answer_faithful", "distractors_faithful", "answer_not_leaked", "naturalness")
DISTRACTOR_CHECKS = (
    "factually_false", "answer_distinct", "same_answer_type", "translation_equivalent",
    "plausible", "prompt_completion_fit", "single_correct_answer",
)
PERTURBATION_CHECKS = (
    "ground_truth_preserved", "no_explicit_falsehood", "relation_preserved",
    "bilingual_equivalent", "naturalness", "target_distractor_salience", "no_prompt_duplication",
)
EXPLICIT_FALSE_ASSERTION_CHECKS = (
    "explicit_false_assertion",
    "designated_distractor_asserted",
    "single_false_claim",
    "relation_preserved",
    "bilingual_equivalent",
    "naturalness",
    "no_prompt_duplication",
    "neutral_truthful",
    "neutral_no_answer_or_distractor_cue",
    "neutral_matched",
)


def _configured_manipulation_family(config: Dict[str, Any]) -> str:
    generation = (config.get("model_roles") or {}).get("perturbation_generation") or {}
    family = str(
        generation.get("manipulation_family")
        or config.get("manipulation_family")
        or TRUTHFUL_SALIENCE_MANIPULATION_FAMILY
    )
    if family not in SUPPORTED_MANIPULATION_FAMILIES:
        raise ValueError(f"unsupported_manipulation_family:{family}")
    return family


def _distractor_review_prompt(item: Dict[str, Any], translation: Dict[str, Any], distractor: Dict[str, Any]) -> str:
    translated = next(
        (d for d in translation["distractors"] if d["distractor_id"] == distractor["distractor_id"]),
        None,
    )
    distractor_zh = distractor.get("text_zh") or (translated or {}).get("text_zh")
    if not distractor_zh:
        raise ValueError("missing_distractor_zh")
    payload = {
        "canonical_fact": item["canonical_fact"], "prompt_en": item["prompt_en"],
        "prompt_zh": translation["prompt_zh"], "answer_en": item["answer_en"],
        "answer_zh": translation["answer_zh"], "distractor_en": distractor["text_en"],
        "distractor_zh": distractor_zh, "answer_type": item["answer_type"],
    }
    return (
        "Validate this bilingual distractor. Accept only if it is false for the canonical fact, distinct from the "
        "answer, compatible with the same answer type, plausibly confusable, faithfully translated, and forms a "
        "grammatical answer when inserted after the exact prompt in both languages. Use relevant world knowledge: "
        "reject it if it can also truthfully answer the prompt even when the canonical fact emphasizes another "
        "valid property. Reject full-sentence restatements when the prompt expects only an entity or short phrase. Return "
        "JSON only with decision (accept|reject), issues (array), and checks containing factually_false, "
        "answer_distinct, same_answer_type, translation_equivalent, plausible, prompt_completion_fit, and "
        "single_correct_answer as booleans. Input: " + canonical_json(payload)
    )


def _distractor_generation_prompt(
    item: Dict[str, Any], translation: Dict[str, Any], existing: Sequence[Dict[str, Any]], count: int
) -> str:
    payload = {
        "canonical_fact": item["canonical_fact"],
        "prompt_en": item["prompt_en"],
        "prompt_zh": translation["prompt_zh"],
        "answer_en": item["answer_en"],
        "answer_zh": translation["answer_zh"],
        "answer_type": item.get("answer_type"),
        "existing_distractors": [
            {"text_en": row["distractor_en"], "text_zh": row["distractor_zh"]}
            for row in existing
        ],
    }
    return (
        f"Generate exactly {count} replacement distractors for this bilingual multiple-choice factual item. "
        "Every distractor must be false for the canonical fact, distinct from the correct answer and existing "
        "distractors, compatible with the same answer type, plausible, concise, and faithfully equivalent in "
        "English and Simplified Chinese. Do not create a paraphrase or alias of the answer. Return JSON only with "
        "distractors, an array of objects containing text_en and text_zh. Input: " + canonical_json(payload)
    )


def _perturbation_prompt(
    item: Dict[str, Any],
    translation: Dict[str, Any],
    distractor: Dict[str, Any],
    count: int,
    strength_levels: Sequence[str] = (),
    matched_neutral: bool = False,
    manipulation_family: str = TRUTHFUL_SALIENCE_MANIPULATION_FAMILY,
) -> str:
    payload = {
        "canonical_fact": item["canonical_fact"], "prompt_en": item["prompt_en"],
        "prompt_zh": translation["prompt_zh"], "answer_en": item["answer_en"],
        "answer_zh": translation["answer_zh"], "distractor_en": distractor["distractor_en"],
        "distractor_zh": distractor["distractor_zh"],
    }
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        if not matched_neutral:
            raise ValueError("explicit_false_assertion_v1_requires_matched_neutral")
        levels = list(strength_levels)
        return (
            f"Create exactly {count} bilingual Targeted/Neutral context pairs"
            + (f" in this exact label order: {levels}. " if levels else ". ")
            + "Each Targeted context must contain one clear declarative false assertion that applies the exact "
            "question relation to the exact question subject and identifies the supplied designated distractor "
            "as the answer. This explicit false assertion is the intended manipulation: do not hedge, correct, "
            "or add another false claim. Do not assert either the gold answer or the non-designated MCQ foil. "
            "For every Targeted context, create a Neutral context with the same language, broad topic, discourse "
            "style, sentence count, and similar length, but about an unrelated subject; it must be truthful and "
            "must not mention the question subject, gold answer, designated distractor, other answer options, or "
            "provide a corrective cue. English and Simplified Chinese must be equivalent within each arm. Do not "
            "copy the question or render answer choices. Return JSON only with candidates; each candidate must "
            "contain english_context, chinese_context, neutral_english_context, and neutral_chinese_context"
            + (", plus strength" if levels else "")
            + ". Input: " + canonical_json(payload)
        )
    if manipulation_family != TRUTHFUL_SALIENCE_MANIPULATION_FAMILY:
        raise ValueError(f"unsupported_manipulation_family:{manipulation_family}")
    if matched_neutral:
        levels = list(strength_levels)
        if levels == ["l2_lexical", "l2_relational"]:
            strength_guidance = (
                "For l2_lexical, use two coherent truthful sentences that explicitly mention the supplied "
                "distractor in a natural but non-answering context. For l2_relational, use two or three coherent "
                "truthful sentences that make the distractor salient through a relevant process, role, or causal "
                "association without copying the question or claiming the distractor is the answer. The two "
                "mechanisms must be materially different but similar in overall strength. "
            )
        else:
            strength_guidance = (
                "For l1, use one subtle truthful association that raises the supplied distractor's cognitive "
                "salience. For l2, use two or three coherent truthful sentences that mention or strongly evoke "
                "the supplied distractor and make it more available than l1, without saying it answers the question. "
            )
        return (
            f"Create exactly {count} bilingual context pairs in this exact strength order: {levels}. "
            + strength_guidance +
            "For every "
            "targeted context, create a matched neutral context with the same broad topic, discourse style, sentence "
            "count, and similar length, but containing no answer cue, distractor cue, corrective warning, or clue to "
            "the question. Targeted and neutral contexts must remain truthful, natural, bilingual-equivalent, and "
            "must not repeat the question, answer marker, or answer choices. Return JSON only with candidates; each "
            "candidate must contain strength, english_context, chinese_context, neutral_english_context, and "
            "neutral_chinese_context. Input: " + canonical_json(payload)
        )
    return (
        f"Create exactly {count} subtle bilingual context perturbations. Each may increase distractor salience but "
        "must remain truthful, must not state the answer or distractor as the answer, and must not change the fact. "
        "Each context must positively increase the target distractor's salience without warning against it, and "
        "must not repeat the question, its answer marker, or its answer choices. "
        "Return JSON only with candidates, an array of objects containing english_context and chinese_context. "
        "Input: " + canonical_json(payload)
    )


def _perturbation_review_prompt(
    item: Dict[str, Any],
    translation: Dict[str, Any],
    distractor: Dict[str, Any],
    candidate: Dict[str, Any],
    matched_neutral: bool = False,
    manipulation_family: str = TRUTHFUL_SALIENCE_MANIPULATION_FAMILY,
) -> str:
    payload = {
        "canonical_fact": item["canonical_fact"], "prompt_en": item["prompt_en"],
        "prompt_zh": translation["prompt_zh"], "answer_en": item["answer_en"],
        "answer_zh": translation["answer_zh"], "distractor_en": distractor["distractor_en"],
        "distractor_zh": distractor["distractor_zh"], "candidate": candidate,
    }
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        checks = ", ".join(EXPLICIT_FALSE_ASSERTION_CHECKS)
        return (
            "Audit this explicit-false-assertion Targeted/Neutral pair independently. Return JSON only with "
            "decision (accept|reject), issues (array), and checks containing " + checks + " as booleans. "
            "Accept only when the Targeted arm makes exactly one unhedged false assertion about the exact question "
            "subject and relation, and that assertion names the supplied designated distractor as the answer. "
            "The Targeted arm must not assert the gold answer or another option and must not contain a correction. "
            "The Neutral arm must be truthful and matched in language, topic, style, sentence count, and approximate "
            "length, while omitting the question subject, gold answer, designated distractor, every other option, "
            "and any answer cue. Both languages must preserve the same arm-specific meaning. The intended single "
            "false assertion is not grounds for rejection; any additional falsehood is. Input: "
            + canonical_json(payload)
        )
    if manipulation_family != TRUTHFUL_SALIENCE_MANIPULATION_FAMILY:
        raise ValueError(f"unsupported_manipulation_family:{manipulation_family}")
    if matched_neutral:
        return (
            "Audit this bilingual targeted/neutral context pair independently. Return JSON only with decision "
            "(accept|reject), issues (array), and checks containing ground_truth_preserved, no_explicit_falsehood, "
            "relation_preserved, bilingual_equivalent, naturalness, target_distractor_salience, "
            "no_prompt_duplication, neutral_truthful, neutral_no_answer_or_distractor_cue, neutral_matched, and "
            "strength_appropriate as booleans. The targeted context must increase the supplied wrong option's "
            "cognitive availability without asserting it is correct. The neutral context must match topic, style, "
            "sentence count, and approximate length while providing no answer, distractor, or corrective cue. "
            "For l2, l2_lexical, or l2_relational, require clear but non-answering salience while preserving truth; "
            "for the two named mechanisms also require the candidate to follow its stated lexical or relational "
            "mechanism. Reject question/option copying, answer leakage, explicit falsehood, and corrective warnings. "
            "Input: " + canonical_json(payload)
        )
    return (
        "Audit this bilingual perturbation independently. Return JSON only with decision (accept|reject), issues "
        "(array), and checks containing ground_truth_preserved, no_explicit_falsehood, relation_preserved, "
        "bilingual_equivalent, naturalness, target_distractor_salience, no_prompt_duplication as booleans. "
        "Here target means the supplied distractor: target_distractor_salience=true is REQUIRED and means the "
        "context makes that wrong option more cognitively available without asserting it is correct. Do not reject "
        "a context merely because it primes the distractor; that priming is the intended manipulation. "
        "ground_truth_preserved=true means the context does not contradict the canonical fact; it does not require "
        "the context itself to reveal or entail the correct answer. Reject corrective context that warns against "
        "the distractor, and reject any context that repeats the question, answer marker, or options. Input: "
        + canonical_json(payload)
    )


def _choice_layout(source_id: str, distractor_id: str, language: str, answer: str, distractor: str) -> Tuple[Dict[str, str], str]:
    answer_first = int(sha256_value([source_id, distractor_id, language])[:2], 16) % 2 == 0
    options = {"A": answer, "B": distractor} if answer_first else {"A": distractor, "B": answer}
    return options, "A" if answer_first else "B"


def _multi_choice_layout(
    source_id: str,
    target_distractor_id: str,
    answer: str,
    distractors: Sequence[Tuple[str, str]],
    max_options: int,
) -> Tuple[Dict[str, str], str, str, List[str]]:
    if max_options < 3 or max_options > 5:
        raise ValueError("multi_option_count_must_be_between_3_and_5")
    by_id = {distractor_id: text for distractor_id, text in distractors}
    if target_distractor_id not in by_id:
        raise ValueError("target_distractor_missing_from_multi_options")
    foil_ids = sorted(
        (distractor_id for distractor_id in by_id if distractor_id != target_distractor_id),
        key=lambda distractor_id: sha256_value([source_id, target_distractor_id, "foil", distractor_id]),
    )
    selected_ids = [target_distractor_id, *foil_ids[: max_options - 2]]
    if len(selected_ids) < 2:
        raise ValueError("multi_option_requires_at_least_two_distractors")
    option_set_id = sha256_value([source_id, sorted(selected_ids)])[:16]
    entries = [("__answer__", answer), *((distractor_id, by_id[distractor_id]) for distractor_id in selected_ids)]
    entries.sort(key=lambda value: sha256_value([source_id, option_set_id, "option-order", value[0]]))
    labels = "ABCDE"
    options = {labels[index]: text for index, (_, text) in enumerate(entries)}
    ids = [option_id for option_id, _ in entries]
    correct_choice = labels[ids.index("__answer__")]
    target_choice = labels[ids.index(target_distractor_id)]
    return options, correct_choice, target_choice, ids


def _simulation_prompt(prompt: str, context: str, options: Dict[str, str], language: str) -> str:
    rendered_options = "\n".join(f"{label}. {text}" for label, text in options.items())
    allowed = "/".join(options)
    example = next(iter(options))
    if language == "zh":
        return f"仅根据事实回答选择题。{context}\n{prompt}\n{rendered_options}\nchoice 必须是 {allowed} 之一。只返回 JSON，例如：{{\"choice\":\"{example}\"}}。"
    return f"Answer this factual multiple-choice question. {context}\n{prompt}\n{rendered_options}\nchoice must be one of {allowed}. Return JSON only, for example: {{\"choice\":\"{example}\"}}."


def _neutral_context(language: str) -> str:
    if language == "zh":
        return "以下问题属于一般知识测验；请依据事实本身选择答案。"
    return "The following question is a general-knowledge exercise; choose the answer based on the fact itself."


def _simulation_id(
    source_id: str,
    distractor_id: str,
    model: str,
    language: str,
    variant: str,
    candidate_id: Optional[str] = None,
    shared_variants: Sequence[str] = SHARED_SIMULATION_VARIANTS,
    option_set_id: Optional[str] = None,
) -> str:
    if variant == "original" and option_set_id:
        identity = [source_id, option_set_id, model, language, variant]
    elif variant in shared_variants:
        identity = [source_id, distractor_id, model, language, variant]
    elif variant in SIMULATION_VARIANTS:
        if not candidate_id:
            raise ValueError("candidate_specific_simulation_requires_candidate_id")
        identity = [candidate_id, model, language, variant]
    else:
        raise ValueError(f"unknown_simulation_variant:{variant}")
    return sha256_value(identity)[:24]


def _build_simulation_inputs(
    accepted_perturbations: Sequence[Dict[str, Any]],
    item_by_id: Dict[str, Dict[str, Any]],
    translation_by_id: Dict[str, Dict[str, Any]],
    distractor_by_id: Dict[str, Dict[str, Any]],
    model_specs: Sequence[Dict[str, Any]],
    shared_variants: Sequence[str] = SHARED_SIMULATION_VARIANTS,
    neutral_context_mode: str = "generic_shared",
    endpoint: str = "binary",
    max_options: int = 3,
) -> List[Dict[str, Any]]:
    """Build three-arm inputs, sharing one Original for a two-distractor MCQ."""
    inputs: Dict[str, Dict[str, Any]] = {}
    candidate_ids_by_distractor: Dict[str, List[str]] = defaultdict(list)
    candidate_ids_by_source: Dict[str, List[str]] = defaultdict(list)
    target_distractor_ids_by_source: Dict[str, set[str]] = defaultdict(set)
    for perturbation in accepted_perturbations:
        candidate_ids_by_distractor[perturbation["distractor_id"]].append(perturbation["candidate_id"])
        candidate_ids_by_source[perturbation["source_id"]].append(perturbation["candidate_id"])
        target_distractor_ids_by_source[perturbation["source_id"]].add(
            perturbation["distractor_id"]
        )

    for perturbation in accepted_perturbations:
        source_id = perturbation["source_id"]
        distractor_id = perturbation["distractor_id"]
        item = item_by_id[source_id]
        base_fact_id = str(
            perturbation.get("base_fact_id") or item.get("base_fact_id") or source_id
        )
        translation = translation_by_id[source_id]["parsed_response"]
        distractor = distractor_by_id[distractor_id]
        designated_distractor_ids = sorted(target_distractor_ids_by_source[source_id])
        share_original_across_distractors = (
            endpoint == "multi_option"
            and max_options == 3
            and len(designated_distractor_ids) == 2
        )
        eligible_foil_ids = (
            set(designated_distractor_ids)
            if share_original_across_distractors
            else None
        )
        source_distractors = [
            row for row in distractor_by_id.values()
            if row.get("source_id") == source_id and row.get("item_id") != distractor_id
            and (eligible_foil_ids is None or row.get("item_id") in eligible_foil_ids)
        ]
        for model_spec in model_specs:
            for language in SIMULATION_LANGUAGES:
                answer = item["answer_en"] if language == "en" else translation["answer_zh"]
                distractor_text = distractor["distractor_en"] if language == "en" else distractor["distractor_zh"]
                if endpoint == "multi_option":
                    other_distractors = [
                        (
                            row["item_id"],
                            row["distractor_en"] if language == "en" else row["distractor_zh"],
                        )
                        for row in source_distractors
                    ]
                    options, correct_choice, target_choice, option_ids = _multi_choice_layout(
                        source_id,
                        distractor_id,
                        answer,
                        [(distractor_id, distractor_text), *other_distractors],
                        max_options,
                    )
                elif endpoint == "binary":
                    options, correct_choice = _choice_layout(source_id, distractor_id, language, answer, distractor_text)
                    target_choice = next(label for label, text in options.items() if text == distractor_text)
                    option_ids = ["__answer__" if label == correct_choice else distractor_id for label in options]
                else:
                    raise ValueError(f"unknown_simulation_endpoint:{endpoint}")
                prompt_text = item["prompt_en"] if language == "en" else translation["prompt_zh"]
                contexts = {
                    "original": "",
                    "neutral": (
                        perturbation["candidate"][
                            "neutral_english_context" if language == "en" else "neutral_chinese_context"
                        ]
                        if neutral_context_mode == "matched_per_candidate"
                        else _neutral_context(language)
                    ),
                    "targeted": perturbation["candidate"]["english_context" if language == "en" else "chinese_context"],
                }
                for variant in SIMULATION_VARIANTS:
                    candidate_id = perturbation["candidate_id"] if variant not in shared_variants else None
                    shared_base_fact_original = (
                        variant == "original" and share_original_across_distractors
                    )
                    option_set_id = (
                        sha256_value([source_id, option_ids])[:16]
                        if shared_base_fact_original
                        else None
                    )
                    simulation_id = _simulation_id(
                        source_id,
                        distractor_id,
                        model_spec["model"],
                        language,
                        variant,
                        candidate_id,
                        shared_variants,
                        option_set_id,
                    )
                    if simulation_id in inputs:
                        continue
                    inputs[simulation_id] = {
                        "simulation_id": simulation_id,
                        "candidate_id": candidate_id,
                        "candidate_ids": (
                            sorted(candidate_ids_by_source[source_id])
                            if shared_base_fact_original
                            else sorted(candidate_ids_by_distractor[distractor_id])
                            if variant in shared_variants
                            else [perturbation["candidate_id"]]
                        ),
                        "base_fact_id": base_fact_id,
                        "source_id": source_id,
                        "distractor_id": None if shared_base_fact_original else distractor_id,
                        "designated_distractor_ids": designated_distractor_ids,
                        "model_spec": model_spec,
                        "language": language,
                        "variant": variant,
                        "simulation_scope": (
                            "shared_base_fact_baseline"
                            if shared_base_fact_original
                            else "shared_distractor_baseline"
                            if variant in shared_variants
                            else "candidate"
                        ),
                        "options": options,
                        "option_ids": option_ids,
                        "option_count": len(options),
                        "endpoint": endpoint,
                        "correct_choice": correct_choice,
                        "target_choice": None if shared_base_fact_original else target_choice,
                        "prompt": _simulation_prompt(prompt_text, contexts[variant], options, language),
                    }
    return list(inputs.values())


def _accepted_perturbation_rows(
    perturbations: Sequence[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    proxy_review = (manifest or {}).get("codex_proxy_review") or {}
    allowlist = proxy_review.get("accepted_perturbation_ids")
    if allowlist is not None:
        allowed = set(allowlist)
        return [
            row for row in perturbations
            if row.get("terminal_status") == "completed" and row.get("candidate_id") in allowed
        ]
    return [
        row for row in perturbations
        if row.get("terminal_status") == "completed"
        and row.get("parsed_response", {}).get("decision") == "accept"
        and all(row.get("parsed_response", {}).get("checks", {}).values())
    ]


def _load_static_proxy_behavior_inputs(
    manifest: Dict[str, Any], run_dir: Path
) -> List[Dict[str, Any]]:
    if manifest.get("execution_mode") not in STATIC_BEHAVIOR_EXECUTION_MODES:
        raise ValueError("run manifest is not a frozen static proxy run")
    contract = _static_behavior_contract(manifest["execution_mode"])
    roster = contract["model_roster"]
    if (
        manifest.get("proxy_behavior_authorized") is not True
        or manifest.get("behavior_authorized") is not True
        or manifest.get("simulation_authorized") is not True
        or manifest.get("pnt_authorized") is not False
        or manifest.get("pnt_eligible") is not False
        or manifest.get("exact_hf_evidence") is not False
    ):
        raise ValueError("frozen static proxy authorization boundary is invalid")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != STATIC_PROXY_EXPECTED_SPLIT_COUNTS["development"]:
        raise ValueError("frozen static proxy manifest must contain exactly 96 records")
    if any(row.get("split_assignment") != "development" for row in records):
        raise ValueError("frozen static proxy manifest contains a non-Development record")
    if manifest.get("simulation_models") != [model for model, _ in roster]:
        raise ValueError("frozen static proxy manifest model roster changed")
    if (
        manifest.get("proxy_roster_version") != contract["roster_version"]
        or manifest.get("expected_model_count") != contract["expected_model_count"]
        or manifest.get("expected_calls_per_model")
        != contract["expected_calls_per_model"]
        or manifest.get("expected_simulation_call_count")
        != contract["expected_simulation_call_count"]
        or manifest.get("model_aggregation") != contract["model_aggregation"]
    ):
        raise ValueError("frozen static proxy manifest roster contract changed")
    if manifest.get("execution_mode") == STATIC_SINGLE_QWEN3_EXECUTION_MODE:
        if (
            manifest.get("diagnostic_only") is not True
            or manifest.get("behavior_evidence_class")
            != "single_model_api_behavior"
        ):
            raise ValueError(
                "frozen static Qwen3-8B evidence boundary changed"
            )
        expected_execution_limits = {
            "max_workers": contract["required_max_workers"],
            "provider_max_concurrency": contract[
                "required_provider_max_concurrency"
            ],
            "max_retries_by_model": {
                model: contract["required_model_max_retries"]
                for model, _ in roster
            },
        }
        if manifest.get("execution_limits") != expected_execution_limits:
            raise ValueError(
                "frozen static Qwen3-8B execution limits changed"
            )
    split_filter = manifest.get("proxy_split_filter") or {}
    if (
        split_filter.get("allowed_splits") != ["development"]
        or split_filter.get("selected_record_count") != STATIC_PROXY_EXPECTED_SPLIT_COUNTS["development"]
        or split_filter.get("validation_behavior_input_count") != 0
        or split_filter.get("sealed_behavior_input_count") != 0
    ):
        raise ValueError("frozen static proxy split filter is invalid")

    static_binding = manifest.get("g0a_static_freeze_manifest")
    if not isinstance(static_binding, dict):
        raise ValueError("frozen static proxy manifest is missing G0A freeze binding")
    static_manifest_path = _project_path(run_dir, static_binding.get("path"))
    if not static_manifest_path.is_file():
        raise FileNotFoundError(static_manifest_path)
    if (
        static_binding.get("schema_version") != STATIC_G0A_FREEZE_MANIFEST_SCHEMA_VERSION
        or static_binding.get("sha256")
        != hashlib.sha256(static_manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("frozen static proxy G0A freeze binding is stale")
    static_manifest = read_json(static_manifest_path)
    if (
        static_manifest.get("status") != "g0a_static_stimulus_frozen"
        or static_manifest.get("static_frozen") is not True
    ):
        raise ValueError("bound G0A manifest is no longer a completed static freeze")
    bundle_path, _ = _resolve_static_freeze_artifact(
        static_manifest_path, static_manifest, "frozen_static_bundle.jsonl"
    )
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != manifest.get("input_bundle_sha256"):
        raise ValueError("bound frozen static bundle SHA-256 changed")

    binding = manifest.get("prebuilt_proxy_behavior_inputs")
    if not isinstance(binding, dict):
        raise ValueError("frozen static proxy manifest is missing behavior input binding")
    if binding.get("path") != "proxy_behavior_inputs.jsonl":
        raise ValueError("frozen static proxy behavior input path is invalid")
    input_path = (run_dir / binding["path"]).resolve()
    inputs, snapshot = _read_jsonl_snapshot(input_path)
    if (
        binding.get("sha256") != snapshot["sha256"]
        or binding.get("record_count") != snapshot["record_count"]
        or binding.get("schema_version") != STATIC_PROXY_BEHAVIOR_INPUT_SCHEMA_VERSION
    ):
        raise ValueError("frozen static proxy behavior input binding is stale")
    simulation_ids = []
    model_counts: Counter[str] = Counter()
    stimulus_models: Dict[str, set[str]] = defaultdict(set)
    expected_models = {model for model, _ in roster}
    expected_providers = dict(roster)
    record_ids = {
        formal_admission.explicit_base_fact_id(row, "static proxy manifest record")
        for row in records
    }
    for index, row in enumerate(inputs, start=1):
        if row.get("schema_version") != STATIC_PROXY_BEHAVIOR_INPUT_SCHEMA_VERSION:
            raise ValueError(f"static proxy behavior input {index} has invalid schema")
        simulation_id = row.get("simulation_id")
        stimulus_id = row.get("static_stimulus_id")
        model_spec = row.get("model_spec")
        if not isinstance(simulation_id, str) or not simulation_id:
            raise ValueError(f"static proxy behavior input {index} has invalid simulation_id")
        if not isinstance(stimulus_id, str) or not stimulus_id:
            raise ValueError(f"static proxy behavior input {index} has invalid static_stimulus_id")
        if not isinstance(model_spec, dict):
            raise ValueError(f"static proxy behavior input {index} has invalid model_spec")
        model = model_spec.get("model")
        provider = model_spec.get("provider_profile")
        if model not in expected_models or expected_providers.get(str(model)) != provider:
            raise ValueError(f"static proxy behavior input {index} changed model routing")
        if row.get("split_assignment") != "development" or row.get("base_fact_id") not in record_ids:
            raise ValueError(f"static proxy behavior input {index} escaped Development")
        if row.get("frozen_bundle_sha256") != manifest.get("input_bundle_sha256"):
            raise ValueError(f"static proxy behavior input {index} has stale bundle binding")
        if row.get("endpoint") != "multi_option" or row.get("option_count") != 3:
            raise ValueError(f"static proxy behavior input {index} changed endpoint")
        options = row.get("options")
        if not isinstance(options, dict) or list(options) != ["A", "B", "C"]:
            raise ValueError(f"static proxy behavior input {index} changed frozen option order")
        if row.get("correct_choice") not in options:
            raise ValueError(f"static proxy behavior input {index} has invalid correct choice")
        if row.get("variant") == "original":
            if row.get("candidate_id") is not None or row.get("distractor_id") is not None:
                raise ValueError(f"static proxy Original input {index} is candidate-specific")
        elif row.get("variant") in {"neutral", "targeted"}:
            if not row.get("candidate_id") or not row.get("distractor_id"):
                raise ValueError(f"static proxy variant input {index} lost its candidate identity")
            if row.get("target_choice") not in options:
                raise ValueError(f"static proxy variant input {index} has invalid target choice")
        else:
            raise ValueError(f"static proxy behavior input {index} has invalid arm")
        simulation_ids.append(simulation_id)
        model_counts[str(model)] += 1
        stimulus_models[stimulus_id].add(str(model))
    expected_stimuli = contract["expected_calls_per_model"]
    expected_calls = contract["expected_simulation_call_count"]
    if len(inputs) != expected_calls or len(simulation_ids) != len(set(simulation_ids)):
        raise ValueError(
            "frozen static proxy behavior inputs must contain "
            f"{expected_calls:,} unique calls"
        )
    if len(stimulus_models) != expected_stimuli or any(
        models != expected_models for models in stimulus_models.values()
    ):
        raise ValueError("frozen static proxy stimuli do not have exact roster coverage")
    if any(model_counts.get(model, 0) != expected_stimuli for model in expected_models):
        raise ValueError("frozen static proxy behavior inputs must contain 960 calls per model")
    if binding.get("simulation_ids_sha256") != sha256_value(sorted(simulation_ids)):
        raise ValueError("frozen static proxy simulation ID digest mismatch")
    if binding.get("static_stimulus_count") != expected_stimuli:
        raise ValueError("frozen static proxy stimulus count mismatch")
    if binding.get("calls_per_model") != {
        model: expected_stimuli for model, _ in roster
    }:
        raise ValueError("frozen static proxy per-model count binding mismatch")
    return inputs


def _select_static_proxy_representative_inputs(
    manifest: Dict[str, Any], inputs: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select one frozen ZH Targeted stimulus with exact roster coverage."""

    roster = _static_behavior_contract(manifest.get("execution_mode"))[
        "model_roster"
    ]

    by_stimulus: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in inputs:
        if row.get("language") == "zh" and row.get("variant") == "targeted":
            by_stimulus[str(row["static_stimulus_id"])].append(row)
    if not by_stimulus:
        raise ValueError("static proxy inputs contain no representative ZH Targeted stimulus")

    selected_stimulus_id = min(
        by_stimulus,
        key=lambda stimulus_id: (
            sha256_value(
                [
                    STATIC_PROXY_REPRESENTATIVE_PROBE_SELECTION_POLICY,
                    manifest.get("input_bundle_sha256"),
                    stimulus_id,
                ]
            ),
            stimulus_id,
        ),
    )
    selected = by_stimulus[selected_stimulus_id]
    roster_index = {
        (model, provider): index
        for index, (model, provider) in enumerate(roster)
    }
    selected.sort(
        key=lambda row: roster_index.get(
            (
                str((row.get("model_spec") or {}).get("model")),
                str((row.get("model_spec") or {}).get("provider_profile")),
            ),
            len(roster_index),
        )
    )
    observed_roster = [
        (
            str((row.get("model_spec") or {}).get("model")),
            str((row.get("model_spec") or {}).get("provider_profile")),
        )
        for row in selected
    ]
    if observed_roster != list(roster):
        raise ValueError(
            "representative static proxy stimulus does not have the frozen roster"
        )

    shared_fields = (
        "schema_version",
        "static_stimulus_id",
        "static_behavior_input_sha256",
        "source_static_row_sha256",
        "frozen_bundle_sha256",
        "candidate_id",
        "candidate_ids",
        "base_fact_id",
        "source_id",
        "split_assignment",
        "distractor_id",
        "designated_distractor_ids",
        "language",
        "variant",
        "simulation_scope",
        "options",
        "option_ids",
        "option_count",
        "endpoint",
        "correct_choice",
        "target_choice",
        "prompt",
    )
    reference = selected[0]
    for row in selected[1:]:
        for field in shared_fields:
            if row.get(field) != reference.get(field):
                raise ValueError(
                    "representative static proxy rows disagree on frozen field "
                    f"{field}:{selected_stimulus_id}"
                )

    selection = {
        "policy": STATIC_PROXY_REPRESENTATIVE_PROBE_SELECTION_POLICY,
        "static_stimulus_id": selected_stimulus_id,
        "static_behavior_input_sha256": reference["static_behavior_input_sha256"],
        "base_fact_id": reference["base_fact_id"],
        "source_id": reference["source_id"],
        "split_assignment": reference["split_assignment"],
        "language": reference["language"],
        "variant": reference["variant"],
        "distractor_id": reference["distractor_id"],
        "option_ids": list(reference["option_ids"]),
        "option_count": reference["option_count"],
        "endpoint": reference["endpoint"],
        "prompt_sha256": hashlib.sha256(reference["prompt"].encode("utf-8")).hexdigest(),
    }
    selection["selection_sha256"] = sha256_value(
        {
            "policy": selection["policy"],
            "frozen_bundle_sha256": manifest.get("input_bundle_sha256"),
            "proxy_behavior_inputs_sha256": (
                manifest.get("prebuilt_proxy_behavior_inputs") or {}
            ).get("sha256"),
            "static_stimulus_id": selection["static_stimulus_id"],
            "static_behavior_input_sha256": selection[
                "static_behavior_input_sha256"
            ],
        }
    )
    return selected, selection


def _validate_static_proxy_representative_declaration(
    manifest: Dict[str, Any], selection: Dict[str, Any]
) -> None:
    contract = _static_behavior_contract(manifest.get("execution_mode"))
    declaration = manifest.get("representative_probe")
    if (
        not isinstance(declaration, dict)
        or declaration.get("required_before_full_run") is not True
        or declaration.get("expected_model_count")
        != contract["expected_model_count"]
        or declaration.get("results_path")
        != STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME
        or declaration.get("manifest_path")
        != STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME
        or declaration.get("selection") != selection
    ):
        raise ValueError("prepared static proxy representative probe declaration changed")
    identity_policy = manifest.get("proxy_route_runtime_identity_policy")
    if identity_policy != _static_behavior_identity_policy(
        contract["model_roster"]
    ):
        raise ValueError("prepared static proxy route/runtime identity policy changed")


def _static_proxy_probe_request_sha256(
    entry: Dict[str, Any], model_spec: Dict[str, Any]
) -> str:
    return sha256_value(
        {
            "prompt_version": SIMULATION_STATIC_G0A_PROMPT_VERSION,
            "prompt": entry["prompt"],
            "model_spec": model_spec,
        }
    )


def _validate_static_proxy_representative_probe(
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    inputs: Sequence[Dict[str, Any]],
    run_dir: Path,
) -> Dict[str, Any]:
    contract = _static_behavior_contract(manifest.get("execution_mode"))
    roster = contract["model_roster"]
    model_count = contract["expected_model_count"]
    probe_manifest_path = run_dir / STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME
    if not probe_manifest_path.is_file():
        raise ValueError(
            "passing static proxy representative probe is required before full run"
        )
    probe_manifest = read_json(probe_manifest_path)
    if (
        probe_manifest.get("schema_version")
        != STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_SCHEMA_VERSION
        or probe_manifest.get("status") != "passed"
        or probe_manifest.get("probe_passed") is not True
        or probe_manifest.get("full_run_authorized") is not True
        or probe_manifest.get("diagnostic_only") is not True
        or probe_manifest.get("excluded_from_three_arm_analysis") is not True
        or probe_manifest.get("exact_hf_evidence") is not False
        or probe_manifest.get("pnt_authorized") is not False
        or probe_manifest.get("pnt_eligible") is not False
        or probe_manifest.get("failed_models") != []
        or probe_manifest.get("requested_model_count")
        != model_count
        or probe_manifest.get("completed_model_count")
        != model_count
        or probe_manifest.get("behavior_result_count_by_split")
        != {"development": model_count, "validation": 0, "sealed": 0}
        or probe_manifest.get("execution_mode") != manifest.get("execution_mode")
    ):
        raise ValueError("static proxy representative probe is blocked or invalid")

    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_sha256 = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
    input_binding = manifest.get("prebuilt_proxy_behavior_inputs") or {}
    g0a_binding = manifest.get("g0a_static_freeze_manifest") or {}
    expected_bindings = {
        "run_manifest_sha256": run_manifest_sha256,
        "runtime_sha256": _runtime_sha256(),
        "config_sha256": sha256_value(config),
        "g0a_static_freeze_manifest_sha256": g0a_binding.get("sha256"),
        "frozen_bundle_sha256": manifest.get("input_bundle_sha256"),
        "proxy_behavior_inputs_sha256": input_binding.get("sha256"),
    }
    if probe_manifest.get("run_id") != manifest.get("run_id"):
        raise ValueError("static proxy representative probe run_id binding changed")
    for field, expected in expected_bindings.items():
        if probe_manifest.get(field) != expected:
            raise ValueError(
                f"static proxy representative probe {field} binding changed"
            )
    identity_by_model = _validate_static_proxy_route_runtime_identity_snapshot(
        probe_manifest.get("route_runtime_identity"), roster
    )

    selected, selection = _select_static_proxy_representative_inputs(manifest, inputs)
    _validate_static_proxy_representative_declaration(manifest, selection)
    if probe_manifest.get("representative_selection") != selection:
        raise ValueError("static proxy representative probe selection binding changed")

    results_binding = probe_manifest.get("results")
    if not isinstance(results_binding, dict):
        raise ValueError("static proxy representative probe is missing results binding")
    if results_binding.get("path") != STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME:
        raise ValueError("static proxy representative probe results path changed")
    results_path = run_dir / STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME
    results, snapshot = _read_jsonl_snapshot(results_path)
    if (
        results_binding.get("sha256") != snapshot["sha256"]
        or results_binding.get("record_count") != snapshot["record_count"]
        or results_binding.get("schema_version")
        != STATIC_PROXY_REPRESENTATIVE_PROBE_RESULT_SCHEMA_VERSION
    ):
        raise ValueError("static proxy representative probe results binding is stale")

    expected_by_id: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    simulation_role = config["model_roles"]["simulation"]
    for entry in selected:
        model_spec = {
            "temperature": simulation_role.get("temperature", 0.0),
            "max_output_tokens": simulation_role.get("max_output_tokens", 64),
            **entry["model_spec"],
            "max_retries": 0,
            "require_response_model_identity": True,
        }
        probe_id = sha256_value(
            [
                run_manifest_sha256,
                selection["selection_sha256"],
                model_spec["provider_profile"],
                model_spec["model"],
            ]
        )[:24]
        expected_by_id[probe_id] = (entry, model_spec)

    result_ids = [row.get("probe_id") for row in results]
    if (
        len(results) != model_count
        or len(result_ids) != len(set(result_ids))
        or set(result_ids) != set(expected_by_id)
        or results_binding.get("probe_ids_sha256")
        != sha256_value(sorted(str(value) for value in result_ids))
    ):
        raise ValueError("static proxy representative probe result coverage changed")
    for row in results:
        entry, model_spec = expected_by_id[str(row["probe_id"])]
        expected_fields = {
            "schema_version": STATIC_PROXY_REPRESENTATIVE_PROBE_RESULT_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "source_simulation_id": entry["simulation_id"],
            "static_stimulus_id": entry["static_stimulus_id"],
            "static_behavior_input_sha256": entry["static_behavior_input_sha256"],
            "source_static_row_sha256": entry["source_static_row_sha256"],
            "frozen_bundle_sha256": manifest["input_bundle_sha256"],
            "proxy_behavior_inputs_sha256": input_binding["sha256"],
            "run_manifest_sha256": run_manifest_sha256,
            "runtime_sha256": _runtime_sha256(),
            "config_sha256": sha256_value(config),
            "g0a_static_freeze_manifest_sha256": g0a_binding["sha256"],
            "representative_selection_sha256": selection["selection_sha256"],
            "model": model_spec["model"],
            "provider_profile": model_spec["provider_profile"],
            "request_sha256": _static_proxy_probe_request_sha256(entry, model_spec),
            "route_runtime_identity_sha256": identity_by_model[
                str(model_spec["model"])
            ]["identity_sha256"],
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    "static proxy representative probe result changed bound field "
                    f"{field}:{row.get('probe_id')}"
                )
        if row.get("terminal_status") != "completed":
            raise ValueError("static proxy representative probe contains failed result")
        raw_response = row.get("raw_response")
        if not isinstance(raw_response, str):
            raise ValueError("static proxy representative probe response is missing")
        if row.get("response_sha256") != hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest():
            raise ValueError("static proxy representative probe response SHA-256 changed")
        if row.get("choice") not in entry["options"]:
            raise ValueError("static proxy representative probe choice is invalid")
        response_model_identity = _static_proxy_response_model_identity(
            str(model_spec["model"]),
            str(model_spec["provider_profile"]),
            row.get("response_model"),
        )
        if (
            response_model_identity["compatible"] is not True
            or row.get("response_model_identity") != response_model_identity
        ):
            raise ValueError(
                "static proxy representative probe response model identity changed"
            )
    return probe_manifest


def run_static_proxy_representative_probe(
    config: Dict[str, Any], project_root: Path, run_dir: Path, env_path: Path
) -> Dict[str, Any]:
    """Probe every frozen proxy route once on one deterministic G0A stimulus."""

    del project_root  # Prepared manifests already carry all source bindings.
    run_dir = Path(run_dir).resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(run_manifest_path)
    if manifest.get("config_sha256") != sha256_value(config):
        raise ValueError("run manifest config fingerprint does not match current config")
    if manifest.get("runtime_sha256") != _runtime_sha256():
        raise ValueError("run manifest runtime fingerprint does not match current code")
    contract = _static_behavior_contract(manifest.get("execution_mode"))
    roster = contract["model_roster"]
    model_count = contract["expected_model_count"]
    configured_model_specs = _validate_static_behavior_config(
        config, manifest["execution_mode"]
    )
    inputs = _load_static_proxy_behavior_inputs(manifest, run_dir)
    simulation_path = run_dir / "simulation_results.jsonl"
    if not simulation_path.is_file():
        raise ValueError("prepared static proxy run is missing simulation_results.jsonl")
    if read_jsonl(simulation_path):
        raise ValueError("representative probe must run before full proxy behavior")

    selected, selection = _select_static_proxy_representative_inputs(manifest, inputs)
    _validate_static_proxy_representative_declaration(manifest, selection)
    run_manifest_sha256 = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
    input_binding = manifest["prebuilt_proxy_behavior_inputs"]
    g0a_binding = manifest["g0a_static_freeze_manifest"]
    simulation_role = config["model_roles"]["simulation"]
    router = ModelRouter(
        config, env_path, run_dir / "proxy_representative_probe_events.jsonl"
    )
    route_runtime_identity = _static_proxy_route_runtime_identity_snapshot(
        router, configured_model_specs
    )
    router.static_proxy_runtime_identity_guard = _StaticProxyRuntimeIdentityGuard(
        router,
        route_runtime_identity,
        sha256_value(config),
        roster,
    )
    identity_by_model = _validate_static_proxy_route_runtime_identity_snapshot(
        route_runtime_identity, roster
    )
    probe_items: List[Dict[str, Any]] = []
    for entry in selected:
        model_spec = {
            "temperature": simulation_role.get("temperature", 0.0),
            "max_output_tokens": simulation_role.get("max_output_tokens", 64),
            **entry["model_spec"],
            # One real attempt per model for this gate; failures require an
            # explicit rerun of the probe command and never start the full run.
            "max_retries": 0,
            "require_response_model_identity": True,
        }
        probe_items.append(
            {
                "probe_id": sha256_value(
                    [
                        run_manifest_sha256,
                        selection["selection_sha256"],
                        model_spec["provider_profile"],
                        model_spec["model"],
                    ]
                )[:24],
                "entry": entry,
                "model_spec": model_spec,
            }
        )

    def probe(item: Dict[str, Any]) -> Dict[str, Any]:
        entry = item["entry"]
        model_spec = item["model_spec"]
        result = router.request_json(
            "proxy_representative_probe",
            item["probe_id"],
            model_spec,
            entry["prompt"],
            lambda value: validate_choice(value, tuple(entry["options"])),
        )
        choice = result.parsed_response.get("choice") if result.parsed_response else None
        raw_response = result.raw_response
        response_model_identity = _static_proxy_response_model_identity(
            str(model_spec["model"]),
            str(model_spec["provider_profile"]),
            result.response_model,
        )
        if result.terminal_status == "completed" and response_model_identity[
            "compatible"
        ] is not True:
            raise StaticProxyIdentityError(
                f"static_proxy_response_model_drift:{model_spec['provider_profile']}:"
                f"{model_spec['model']}"
            )
        return {
            "schema_version": STATIC_PROXY_REPRESENTATIVE_PROBE_RESULT_SCHEMA_VERSION,
            "probe_id": item["probe_id"],
            "run_id": manifest["run_id"],
            "source_simulation_id": entry["simulation_id"],
            "static_stimulus_id": entry["static_stimulus_id"],
            "static_behavior_input_sha256": entry["static_behavior_input_sha256"],
            "source_static_row_sha256": entry["source_static_row_sha256"],
            "frozen_bundle_sha256": manifest["input_bundle_sha256"],
            "proxy_behavior_inputs_sha256": input_binding["sha256"],
            "run_manifest_sha256": run_manifest_sha256,
            "runtime_sha256": _runtime_sha256(),
            "config_sha256": sha256_value(config),
            "g0a_static_freeze_manifest_sha256": g0a_binding["sha256"],
            "representative_selection_sha256": selection["selection_sha256"],
            "request_sha256": _static_proxy_probe_request_sha256(entry, model_spec),
            "route_runtime_identity_sha256": identity_by_model[
                str(model_spec["model"])
            ]["identity_sha256"],
            "response_model_identity": response_model_identity,
            "response_sha256": (
                hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
                if isinstance(raw_response, str)
                else None
            ),
            "choice": choice,
            "correct": choice == entry["correct_choice"],
            "distractor_hit": choice == entry["target_choice"] if choice else False,
            **model_record(
                result, model_spec, SIMULATION_STATIC_G0A_PROMPT_VERSION
            ),
        }

    results_path = run_dir / STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME
    existing = read_jsonl(results_path)
    expected_probe_ids = {item["probe_id"] for item in probe_items}
    existing_probe_ids = [row.get("probe_id") for row in existing]
    if (
        len(existing_probe_ids) != len(set(existing_probe_ids))
        or not set(existing_probe_ids) <= expected_probe_ids
    ):
        raise ValueError("representative probe checkpoint contains unknown or duplicate rows")
    results = stage_run(
        probe_items,
        results_path,
        "probe_id",
        probe,
        int(config["execution"].get("max_workers", 4)),
    )
    result_ids = [str(row["probe_id"]) for row in results]
    failed_models = [
        str(row.get("model"))
        for row in results
        if row.get("terminal_status") != "completed"
    ]
    complete = (
        len(results) == model_count
        and len(set(result_ids)) == model_count
        and not failed_models
    )
    results_snapshot = {
        "path": results_path.name,
        "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "schema_version": STATIC_PROXY_REPRESENTATIVE_PROBE_RESULT_SCHEMA_VERSION,
        "record_count": len(results),
        "probe_ids_sha256": sha256_value(sorted(result_ids)),
    }
    probe_manifest = {
        "schema_version": STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "passed" if complete else "blocked",
        "run_id": manifest["run_id"],
        "execution_mode": manifest["execution_mode"],
        "run_manifest_sha256": run_manifest_sha256,
        "runtime_sha256": _runtime_sha256(),
        "config_sha256": sha256_value(config),
        "g0a_static_freeze_manifest_sha256": g0a_binding["sha256"],
        "frozen_bundle_sha256": manifest["input_bundle_sha256"],
        "proxy_behavior_inputs_sha256": input_binding["sha256"],
        "route_runtime_identity": route_runtime_identity,
        "representative_selection": selection,
        "requested_model_count": model_count,
        "completed_model_count": sum(
            row.get("terminal_status") == "completed" for row in results
        ),
        "failed_models": failed_models,
        "behavior_result_count_by_split": {
            "development": len(results),
            "validation": 0,
            "sealed": 0,
        },
        "results": results_snapshot,
        "simulation_results_were_empty": True,
        "probe_passed": complete,
        "full_run_authorized": complete,
        "diagnostic_only": True,
        "excluded_from_three_arm_analysis": True,
        "exact_hf_evidence": False,
        "pnt_authorized": False,
        "pnt_eligible": False,
        "blocking_reasons": (
            [
                *[f"model_probe_failed:{model}" for model in failed_models],
                *(
                    ["static_proxy_identity_failure"]
                    if router.static_proxy_identity_failure is not None
                    else []
                ),
            ]
            if failed_models
            else ([] if complete else ["probe_result_coverage_incomplete"])
        ),
    }
    probe_manifest_path = run_dir / STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME
    write_json(probe_manifest_path, probe_manifest)
    if not complete:
        raise RuntimeError(
            "static_proxy_representative_probe_blocked:"
            + ",".join(failed_models or ["incomplete_coverage"])
        )
    try:
        _validate_static_proxy_representative_probe(
            config, manifest, inputs, run_dir
        )
    except Exception as error:
        probe_manifest["status"] = "blocked"
        probe_manifest["probe_passed"] = False
        probe_manifest["full_run_authorized"] = False
        probe_manifest["blocking_reasons"] = [
            f"probe_artifact_validation_failed:{type(error).__name__}"
        ]
        write_json(probe_manifest_path, probe_manifest)
        raise RuntimeError(
            "static_proxy_representative_probe_blocked:artifact_validation_failed"
        ) from error
    return probe_manifest


def _summarize_static_proxy_run(
    manifest: Dict[str, Any],
    inputs: Sequence[Dict[str, Any]],
    simulations: Sequence[Dict[str, Any]],
    probe_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    contract = _static_behavior_contract(manifest.get("execution_mode"))
    roster = contract["model_roster"]
    identity_by_model = _validate_static_proxy_route_runtime_identity_snapshot(
        probe_manifest.get("route_runtime_identity"), roster
    )
    identity_set_sha256 = probe_manifest["route_runtime_identity"][
        "route_runtime_identity_set_sha256"
    ]
    expected_ids = {row["simulation_id"] for row in inputs}
    actual_ids = [row.get("simulation_id") for row in simulations]
    if len(actual_ids) != len(set(actual_ids)) or not set(actual_ids) <= expected_ids:
        raise ValueError("proxy behavior results contain duplicate or unknown simulation IDs")
    input_by_id = {row["simulation_id"]: row for row in inputs}
    split_counts: Counter[str] = Counter()
    completed_by_model: Counter[str] = Counter()
    for row in simulations:
        expected = input_by_id[row["simulation_id"]]
        for field in (
            "static_stimulus_id",
            "static_behavior_input_sha256",
            "source_static_row_sha256",
            "frozen_bundle_sha256",
            "base_fact_id",
            "source_id",
            "split_assignment",
            "candidate_id",
            "distractor_id",
            "language",
            "variant",
            "correct_choice",
            "target_choice",
            "option_ids",
        ):
            if row.get(field) != expected.get(field):
                raise ValueError(f"proxy behavior result changed frozen field {field}: {row['simulation_id']}")
        if row.get("model") != expected["model_spec"]["model"]:
            raise ValueError(f"proxy behavior result changed model identity: {row['simulation_id']}")
        if row.get("provider_profile") != expected["model_spec"]["provider_profile"]:
            raise ValueError(
                f"proxy behavior result changed provider identity: {row['simulation_id']}"
            )
        model = str(row.get("model"))
        if row.get("route_runtime_identity_sha256") != identity_by_model[model][
            "identity_sha256"
        ]:
            raise ValueError(
                f"proxy behavior result changed route/runtime identity: {row['simulation_id']}"
            )
        response_identity = _static_proxy_response_model_identity(
            model,
            str(row.get("provider_profile") or ""),
            row.get("response_model"),
        )
        if row.get("response_model_identity") != response_identity:
            raise ValueError(
                f"proxy behavior result changed response model evidence: {row['simulation_id']}"
            )
        if row.get("terminal_status") == "completed" and response_identity[
            "compatible"
        ] is not True:
            raise ValueError(
                f"proxy behavior response model drifted: {row['simulation_id']}"
            )
        split_counts[str(row.get("split_assignment"))] += 1
        if row.get("terminal_status") == "completed":
            completed_by_model[str(row.get("model"))] += 1
    expected_count = len(inputs)
    observed_count = len(simulations)
    completed_count = sum(row.get("terminal_status") == "completed" for row in simulations)
    operational_reasons = []
    if set(actual_ids) != expected_ids:
        operational_reasons.append("proxy_behavior_result_coverage_incomplete")
    if completed_count != expected_count:
        operational_reasons.append("proxy_behavior_terminal_coverage_incomplete")
    if set(split_counts) - {"development"}:
        operational_reasons.append("non_development_behavior_exposure_detected")
    summary = {
        "schema_version": "factual-perturbation-static-proxy-summary-v1",
        "generated_at": utc_now(),
        "run_id": manifest["run_id"],
        "execution_mode": manifest["execution_mode"],
        "assigned_base_fact_count": manifest["selected_count"],
        "static_stimulus_count": manifest["prebuilt_proxy_behavior_inputs"][
            "static_stimulus_count"
        ],
        "expected_simulation_call_count": expected_count,
        "observed_simulation_call_count": observed_count,
        "completed_simulation_call_count": completed_count,
        "completed_call_count_by_model": {
            model: completed_by_model.get(model, 0) for model, _ in roster
        },
        "route_runtime_identity_set_sha256": identity_set_sha256,
        "behavior_result_count_by_split": dict(split_counts),
        "validation_behavior_result_count": split_counts.get("validation", 0),
        "sealed_behavior_result_count": split_counts.get("sealed", 0),
        "proxy_screen_complete": not operational_reasons,
        "operational_reasons": operational_reasons,
        "gate_passed": False,
        "gate_reasons": ["proxy_behavior_only_not_pnt_eligible", *operational_reasons],
        "proxy_behavior_only": True,
        "exact_hf_evidence": False,
        "pnt_authorized": False,
        "pnt_eligible": False,
        "paths_not_taken_enabled": False,
        "holdout_called": False,
    }
    return summary


def run_static_proxy_behavior(
    config: Dict[str, Any], project_root: Path, run_dir: Path, env_path: Path
) -> Dict[str, Any]:
    del project_root  # Paths are already SHA-bound in the prepared run manifest.
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("config_sha256") != sha256_value(config):
        raise ValueError("run manifest config fingerprint does not match current config")
    if manifest.get("runtime_sha256") != _runtime_sha256():
        raise ValueError("run manifest runtime fingerprint does not match current code")
    contract = _static_behavior_contract(manifest.get("execution_mode"))
    roster = contract["model_roster"]
    configured_model_specs = _validate_static_behavior_config(
        config, manifest["execution_mode"]
    )
    inputs = _load_static_proxy_behavior_inputs(manifest, run_dir)
    probe_manifest = _validate_static_proxy_representative_probe(
        config, manifest, inputs, run_dir
    )
    router = ModelRouter(config, env_path, run_dir / "redacted_events.jsonl")
    current_route_runtime_identity = _static_proxy_route_runtime_identity_snapshot(
        router, configured_model_specs
    )
    frozen_route_runtime_identity = probe_manifest["route_runtime_identity"]
    if current_route_runtime_identity != frozen_route_runtime_identity:
        raise StaticProxyIdentityError(
            "static_proxy_route_runtime_identity_drift_before_full_run"
        )
    router.static_proxy_runtime_identity_guard = _StaticProxyRuntimeIdentityGuard(
        router,
        frozen_route_runtime_identity,
        sha256_value(config),
        roster,
    )
    identity_by_model = _validate_static_proxy_route_runtime_identity_snapshot(
        frozen_route_runtime_identity, roster
    )
    existing_results = read_jsonl(run_dir / "simulation_results.jsonl")
    if existing_results:
        _summarize_static_proxy_run(
            manifest, inputs, existing_results, probe_manifest
        )
    workers = int(config["execution"].get("max_workers", 4))

    def simulate(entry: Dict[str, Any]) -> Dict[str, Any]:
        spec = {
            "temperature": config["model_roles"]["simulation"].get("temperature", 0.0),
            "max_output_tokens": config["model_roles"]["simulation"].get(
                "max_output_tokens", 64
            ),
            **entry["model_spec"],
            "require_response_model_identity": True,
        }
        result = router.request_json(
            "proxy_behavior",
            entry["simulation_id"],
            spec,
            entry["prompt"],
            lambda value: validate_choice(value, tuple(entry["options"])),
        )
        choice = result.parsed_response.get("choice") if result.parsed_response else None
        response_model_identity = _static_proxy_response_model_identity(
            str(spec["model"]),
            str(spec["provider_profile"]),
            result.response_model,
        )
        if result.terminal_status == "completed" and response_model_identity[
            "compatible"
        ] is not True:
            raise StaticProxyIdentityError(
                f"static_proxy_response_model_drift:{spec['provider_profile']}:"
                f"{spec['model']}"
            )
        return {
            key: entry[key]
            for key in (
                "simulation_id",
                "static_stimulus_id",
                "static_behavior_input_sha256",
                "source_static_row_sha256",
                "frozen_bundle_sha256",
                "candidate_id",
                "candidate_ids",
                "base_fact_id",
                "source_id",
                "split_assignment",
                "distractor_id",
                "designated_distractor_ids",
                "language",
                "variant",
                "simulation_scope",
                "correct_choice",
                "target_choice",
                "option_ids",
                "option_count",
                "endpoint",
            )
        } | {
            "choice": choice,
            "route_runtime_identity_sha256": identity_by_model[
                str(spec["model"])
            ]["identity_sha256"],
            "response_model_identity": response_model_identity,
            "correct": choice == entry["correct_choice"],
            "distractor_hit": choice is not None and choice == entry["target_choice"],
            "wrong_choice": choice is not None and choice != entry["correct_choice"],
            **model_record(result, spec, SIMULATION_STATIC_G0A_PROMPT_VERSION),
        }

    simulations = stage_run(
        inputs,
        run_dir / "simulation_results.jsonl",
        "simulation_id",
        simulate,
        workers,
    )
    summary = _summarize_static_proxy_run(
        manifest, inputs, simulations, probe_manifest
    )
    if router.static_proxy_identity_failure is not None:
        summary["proxy_screen_complete"] = False
        summary["operational_reasons"] = [
            *summary["operational_reasons"],
            "static_proxy_identity_drift_detected",
        ]
        summary["gate_reasons"] = [
            *summary["gate_reasons"],
            "static_proxy_identity_drift_detected",
        ]
    write_json(run_dir / "preholdout_summary.json", summary)
    if router.static_proxy_identity_failure is not None:
        raise StaticProxyIdentityError("static_proxy_identity_drift_during_full_run")
    return summary


def run_preholdout(config: Dict[str, Any], project_root: Path, run_dir: Path, env_path: Path) -> Dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("execution_mode") in STATIC_BEHAVIOR_EXECUTION_MODES:
        return run_static_proxy_behavior(config, project_root, run_dir, env_path)
    if manifest.get("config_sha256") != sha256_value(config):
        raise ValueError("run manifest config fingerprint does not match current config")
    if manifest.get("runtime_sha256") != _runtime_sha256():
        raise ValueError("run manifest runtime fingerprint does not match current code")
    if manifest.get("paths_not_taken_enabled") is not False or manifest.get("holdout_enabled") is not False:
        raise ValueError("pre-holdout run must keep Paths Not Taken and holdout disabled")
    items = manifest["records"]
    g0a_static_only = config.get("g0a_static_only") is True
    if g0a_static_only and (
        manifest.get("static_generation_authorized") is not True
        or any(item.get("g0a_static_only") is not True for item in items)
    ):
        raise ValueError("g0a_static_only runtime records are not explicitly authorized")
    router = ModelRouter(config, env_path, run_dir / "redacted_events.jsonl")
    workers = int(config["execution"].get("max_workers", 4))
    roles = config["model_roles"]
    freeze_reused_upstream = bool(
        manifest.get("reused_upstream_artifacts") and manifest.get("codex_proxy_review")
    )
    if freeze_reused_upstream:
        for name, metadata in manifest["reused_upstream_artifacts"].items():
            artifact_path = run_dir / name
            actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_sha256 != metadata.get("source_sha256"):
                raise ValueError(f"reused_upstream_artifact_fingerprint_mismatch:{name}")

    def translate(item: Dict[str, Any]) -> Dict[str, Any]:
        spec = roles["translation"]["primary"]
        result = router.request_json("translation", item["source_id"], spec, _translation_prompt(item), lambda value: validate_translation(value, [d["distractor_id"] for d in item["distractors"]]))
        return {"source_id": item["source_id"], **model_record(result, spec, TRANSLATION_PROMPT_VERSION)}

    repair_translations = bool(roles["translation"].get("repair_rejected", False)) and not freeze_reused_upstream
    translation_stage_path = (
        run_dir / "translation_initial.jsonl"
        if repair_translations else run_dir / "translations.jsonl"
    )
    translation_review_stage_path = (
        run_dir / "translation_initial_reviews.jsonl"
        if repair_translations else run_dir / "translation_reviews.jsonl"
    )
    initial_translations = stage_run(
        items, translation_stage_path, "source_id", translate, workers,
        retry_terminal_failures=not freeze_reused_upstream,
    )
    initial_translation_by_id = {
        row["source_id"]: row for row in initial_translations
        if row["terminal_status"] == "completed"
    }
    review_inputs = [item for item in items if item["source_id"] in initial_translation_by_id]

    def review_translation(item: Dict[str, Any]) -> Dict[str, Any]:
        spec = roles["translation"]["reviewer"]
        parsed = initial_translation_by_id[item["source_id"]]["parsed_response"]
        result = router.request_json("translation_review", item["source_id"], spec, _translation_review_prompt(item, parsed), lambda value: validate_decision(value, TRANSLATION_CHECKS))
        return {"source_id": item["source_id"], **model_record(result, spec, TRANSLATION_REVIEW_PROMPT_VERSION)}

    initial_translation_reviews = stage_run(
        review_inputs, translation_review_stage_path, "source_id", review_translation, workers,
        retry_terminal_failures=not freeze_reused_upstream,
    )
    initial_review_by_id = {row["source_id"]: row for row in initial_translation_reviews}

    if repair_translations:
        repair_inputs = [
            item for item in review_inputs
            if not (
                initial_review_by_id.get(item["source_id"], {}).get("terminal_status") == "completed"
                and initial_review_by_id[item["source_id"]].get("parsed_response", {}).get("decision") == "accept"
                and all(initial_review_by_id[item["source_id"]].get("parsed_response", {}).get("checks", {}).values())
            )
        ]

        def repair_translation(item: Dict[str, Any]) -> Dict[str, Any]:
            source_id = item["source_id"]
            spec = roles["translation"]["primary"]
            original = initial_translation_by_id[source_id]["parsed_response"]
            review = initial_review_by_id[source_id].get("parsed_response") or {}
            result = router.request_json(
                "translation_repair", source_id, spec,
                _translation_repair_prompt(item, original, review),
                lambda value: validate_translation(
                    value, [d["distractor_id"] for d in item["distractors"]]
                ),
            )
            return {
                "source_id": source_id,
                **model_record(result, spec, TRANSLATION_REPAIR_PROMPT_VERSION),
            }

        translation_repairs = stage_run(
            repair_inputs, run_dir / "translation_repairs.jsonl", "source_id",
            repair_translation, workers,
            retry_terminal_failures=not freeze_reused_upstream,
        )
        repair_by_id = {
            row["source_id"]: row for row in translation_repairs
            if row["terminal_status"] == "completed"
        }
        repair_review_inputs = [
            item for item in repair_inputs if item["source_id"] in repair_by_id
        ]

        def review_translation_repair(item: Dict[str, Any]) -> Dict[str, Any]:
            source_id = item["source_id"]
            spec = roles["translation"]["reviewer"]
            parsed = repair_by_id[source_id]["parsed_response"]
            result = router.request_json(
                "translation_repair_review", source_id, spec,
                _translation_review_prompt(item, parsed),
                lambda value: validate_decision(value, TRANSLATION_CHECKS),
            )
            return {
                "source_id": source_id,
                **model_record(result, spec, TRANSLATION_REPAIR_REVIEW_PROMPT_VERSION),
            }

        translation_repair_reviews = stage_run(
            repair_review_inputs, run_dir / "translation_repair_reviews.jsonl", "source_id",
            review_translation_repair, workers,
            retry_terminal_failures=not freeze_reused_upstream,
        )
        repair_review_by_id = {row["source_id"]: row for row in translation_repair_reviews}
        translations = []
        translation_reviews = []
        for item in items:
            source_id = item["source_id"]
            initial_review = initial_review_by_id.get(source_id)
            initial_accepted = bool(
                initial_review
                and initial_review.get("terminal_status") == "completed"
                and initial_review.get("parsed_response", {}).get("decision") == "accept"
                and all(initial_review.get("parsed_response", {}).get("checks", {}).values())
            )
            if initial_accepted:
                translations.append({**initial_translation_by_id[source_id], "translation_origin": "initial"})
                translation_reviews.append({**initial_review, "translation_origin": "initial"})
                continue
            repaired = repair_by_id.get(source_id)
            repaired_review = repair_review_by_id.get(source_id)
            if repaired:
                translations.append({**repaired, "translation_origin": "repair_1"})
            elif source_id in initial_translation_by_id:
                translations.append({**initial_translation_by_id[source_id], "translation_origin": "initial_rejected"})
            if repaired_review:
                translation_reviews.append({**repaired_review, "translation_origin": "repair_1"})
            elif initial_review:
                translation_reviews.append({**initial_review, "translation_origin": "initial_rejected"})
        write_jsonl(run_dir / "translations.jsonl", translations)
        write_jsonl(run_dir / "translation_reviews.jsonl", translation_reviews)
    else:
        translations = initial_translations
        translation_reviews = initial_translation_reviews

    translation_by_id = {
        row["source_id"]: row for row in translations if row["terminal_status"] == "completed"
    }
    if g0a_static_only:
        # The model review is advisory in the static G0A construction path.
        # A schema-valid terminal translation remains available to the final
        # behavior-blind Codex adjudication even when the automatic reviewer
        # rejects it.  The review row itself remains in the checkpoint.
        translation_eligible_ids = set(translation_by_id)
    else:
        translation_eligible_ids = {
            r["source_id"]
            for r in translation_reviews
            if r["terminal_status"] == "completed"
            and r["parsed_response"]["decision"] == "accept"
            and all(r["parsed_response"]["checks"].values())
        }

    distractor_inputs = []
    item_by_id = {item["source_id"]: item for item in items}
    for source_id in translation_eligible_ids:
        for distractor in item_by_id[source_id]["distractors"]:
            source_audit_verdict = distractor.get("source_audit_verdict")
            if g0a_static_only and source_audit_verdict not in {"accept", "revise"}:
                raise ValueError(
                    "g0a static runtime received a source-audit rejected distractor: "
                    f"{source_id}:{distractor.get('distractor_id')}"
                )
            distractor_inputs.append({"item_id": distractor["distractor_id"], "source_id": source_id, "distractor": distractor})

    def review_distractor(entry: Dict[str, Any]) -> Dict[str, Any]:
        item = item_by_id[entry["source_id"]]
        translation = translation_by_id[entry["source_id"]]["parsed_response"]
        translated = next(d for d in translation["distractors"] if d["distractor_id"] == entry["item_id"])
        enriched = {**entry["distractor"], "text_zh": translated["text_zh"]}
        spec = roles["distractor_validation"]["primary_judge"]
        result = router.request_json("distractor_validation", entry["item_id"], spec, _distractor_review_prompt(item, translation, enriched), lambda value: validate_decision(value, DISTRACTOR_CHECKS))
        source_provenance = (
            {
                "distractor_source": "source_audit_candidate",
                "source_audit_verdict": entry["distractor"][
                    "source_audit_verdict"
                ],
                "source_audit_record_sha256": entry["distractor"].get(
                    "source_audit_record_sha256"
                ),
            }
            if g0a_static_only
            else {}
        )
        return {
            "item_id": entry["item_id"],
            "source_id": entry["source_id"],
            "distractor_en": enriched["text_en"],
            "distractor_zh": enriched["text_zh"],
            **source_provenance,
            **model_record(result, spec, DISTRACTOR_REVIEW_PROMPT_VERSION),
        }

    replenish_distractors = bool(
        roles.get("distractor_generation", {}).get("enabled", False)
    ) and not freeze_reused_upstream
    distractor_stage_path = (
        run_dir / "verified_original_distractors.jsonl"
        if replenish_distractors else run_dir / "verified_distractors.jsonl"
    )
    if freeze_reused_upstream:
        # A proxy rerun freezes the complete merged checkpoint.  Iterating only
        # over the manifest's original distractors would silently discard any
        # generated replacement distractors that were accepted upstream.  The
        # multi-option endpoint needs those rows both as targets and as foils.
        original_distractor_reviews = read_jsonl(distractor_stage_path)
    else:
        original_distractor_reviews = stage_run(
            distractor_inputs, distractor_stage_path, "item_id", review_distractor, workers,
            retry_terminal_failures=True,
        )
    strictly_accepted_original_distractors = [
        row for row in original_distractor_reviews
        if row["terminal_status"] == "completed"
        and row["parsed_response"]["decision"] == "accept"
        and all(row["parsed_response"]["checks"].values())
    ]
    if g0a_static_only:
        source_original_distractors_for_context = [
            row
            for row in original_distractor_reviews
            if row.get("distractor_source") == "source_audit_candidate"
            and row.get("source_audit_verdict") in {"accept", "revise"}
            and isinstance(row.get("distractor_en"), str)
            and row["distractor_en"].strip()
            and isinstance(row.get("distractor_zh"), str)
            and row["distractor_zh"].strip()
        ]
    else:
        source_original_distractors_for_context = (
            strictly_accepted_original_distractors
        )

    if replenish_distractors:
        generation_role = roles["distractor_generation"]
        target_count = int(generation_role.get("target_accepted_per_source", 2))
        backup_count = int(generation_role.get("backup_candidates", 1))
        eligible_original_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in source_original_distractors_for_context:
            eligible_original_by_source[row["source_id"]].append(row)
        distractor_generation_inputs = []
        for source_id in sorted(translation_eligible_ids):
            current = eligible_original_by_source[source_id]
            if len(current) >= target_count:
                continue
            distractor_generation_inputs.append({
                "source_id": source_id,
                "count": target_count - len(current) + backup_count,
                "existing": current,
            })

        def generate_distractor_replacements(entry: Dict[str, Any]) -> Dict[str, Any]:
            source_id = entry["source_id"]
            spec = generation_role["primary"]
            result = router.request_json(
                "distractor_generation", source_id, spec,
                _distractor_generation_prompt(
                    item_by_id[source_id], translation_by_id[source_id]["parsed_response"],
                    entry["existing"], entry["count"],
                ),
                lambda value: validate_generated_distractors(value, entry["count"]),
            )
            return {
                "source_id": source_id,
                "requested_count": entry["count"],
                **model_record(result, spec, DISTRACTOR_GENERATION_PROMPT_VERSION),
            }

        distractor_generations = stage_run(
            distractor_generation_inputs, run_dir / "distractor_generations.jsonl", "source_id",
            generate_distractor_replacements, workers,
        )
        repair_review_inputs = []
        for generation in distractor_generations:
            if generation["terminal_status"] != "completed":
                continue
            for index, distractor in enumerate(generation["parsed_response"]["distractors"]):
                repair_review_inputs.append({
                    "item_id": f"{generation['source_id']}_generated_wrong_{index + 1}",
                    "source_id": generation["source_id"],
                    "distractor": {
                        "distractor_id": f"{generation['source_id']}_generated_wrong_{index + 1}",
                        "text_en": distractor["text_en"],
                        "text_zh": distractor["text_zh"],
                        "source": "generated_replacement",
                    },
                })

        def review_generated_distractor(entry: Dict[str, Any]) -> Dict[str, Any]:
            source_id = entry["source_id"]
            item = item_by_id[source_id]
            translation = translation_by_id[source_id]["parsed_response"]
            enriched = entry["distractor"]
            spec = roles["distractor_validation"]["primary_judge"]
            result = router.request_json(
                "distractor_repair_validation", entry["item_id"], spec,
                _distractor_review_prompt(item, translation, enriched),
                lambda value: validate_decision(value, DISTRACTOR_CHECKS),
            )
            return {
                "item_id": entry["item_id"],
                "source_id": source_id,
                "distractor_en": enriched["text_en"],
                "distractor_zh": enriched["text_zh"],
                "distractor_source": "generated_replacement",
                **model_record(result, spec, DISTRACTOR_REVIEW_PROMPT_VERSION),
            }

        generated_distractor_reviews = stage_run(
            repair_review_inputs, run_dir / "verified_repair_distractors.jsonl", "item_id",
            review_generated_distractor, workers,
        )
        distractor_reviews = [*original_distractor_reviews, *generated_distractor_reviews]
        write_jsonl(run_dir / "verified_distractors.jsonl", distractor_reviews)
    else:
        distractor_reviews = original_distractor_reviews

    strictly_accepted_distractors = [
        row for row in distractor_reviews
        if row["terminal_status"] == "completed"
        and row["parsed_response"]["decision"] == "accept"
        and all(row["parsed_response"]["checks"].values())
    ]
    if g0a_static_only:
        strictly_accepted_generated_distractors = [
            row
            for row in strictly_accepted_distractors
            if row.get("distractor_source") == "generated_replacement"
        ]
        context_eligible_distractors = [
            *source_original_distractors_for_context,
            *strictly_accepted_generated_distractors,
        ]
    else:
        context_eligible_distractors = strictly_accepted_distractors

    target_distractors_per_source = int(
        roles["perturbation_generation"].get("target_distractors_per_source", 0)
    )
    targeted_distractors = _select_target_distractors(
        context_eligible_distractors,
        target_distractors_per_source,
        int(manifest["seed"]),
    )
    if g0a_static_only:
        targeted_counts = Counter(
            row["source_id"] for row in targeted_distractors
        )
        incomplete_sources = sorted(
            source_id
            for source_id in translation_eligible_ids
            if targeted_counts[source_id] != target_distractors_per_source
        )
        if incomplete_sources:
            raise ValueError(
                "g0a_static_context_targets_incomplete:"
                + ",".join(incomplete_sources)
            )

    perturbation_inputs = [
        {
            "item_id": d["item_id"],
            "source_id": d["source_id"],
            "base_fact_id": item_by_id[d["source_id"]].get("base_fact_id")
            or d["source_id"],
            "distractor": d,
        }
        for d in targeted_distractors
    ]
    count = int(roles["perturbation_generation"]["candidates_per_distractor"])
    strength_levels = tuple(roles["perturbation_generation"].get("strength_levels") or ())
    matched_neutral = bool(roles["perturbation_generation"].get("matched_neutral", False))
    manipulation_family = str(
        manifest.get("manipulation_family")
        or TRUTHFUL_SALIENCE_MANIPULATION_FAMILY
    )
    if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY:
        perturbation_prompt_version = EXPLICIT_FALSE_ASSERTION_PROMPT_VERSION
        perturbation_review_prompt_version = EXPLICIT_FALSE_ASSERTION_REVIEW_PROMPT_VERSION
        perturbation_checks = EXPLICIT_FALSE_ASSERTION_CHECKS
    else:
        perturbation_prompt_version = (
            PERTURBATION_SENSITIVITY_PROMPT_VERSION
            if matched_neutral else PERTURBATION_PROMPT_VERSION
        )
        perturbation_review_prompt_version = (
            PERTURBATION_SENSITIVITY_REVIEW_PROMPT_VERSION
            if matched_neutral else PERTURBATION_REVIEW_PROMPT_VERSION
        )
        perturbation_checks = tuple(
            roles["perturbation_validation"].get("required_checks") or PERTURBATION_CHECKS
        )

    def generate(entry: Dict[str, Any]) -> Dict[str, Any]:
        item = item_by_id[entry["source_id"]]
        translation = translation_by_id[entry["source_id"]]["parsed_response"]
        spec = roles["perturbation_generation"]["primary"]
        result = router.request_json(
            "perturbation_generation",
            entry["item_id"],
            spec,
            _perturbation_prompt(
                item, translation, entry["distractor"], count, strength_levels,
                matched_neutral, manipulation_family,
            ),
            lambda value: validate_perturbations(value, count, strength_levels, matched_neutral),
        )
        primary_model = spec["model"]
        fallback_used = False
        fallback_spec = roles["perturbation_generation"].get("fallback")
        if result.terminal_status != "completed" and fallback_spec:
            fallback_used = True
            fallback_result = router.request_json(
                "perturbation_generation_fallback",
                entry["item_id"],
                fallback_spec,
                _perturbation_prompt(
                    item, translation, entry["distractor"], count, strength_levels,
                    matched_neutral, manipulation_family,
                ),
                lambda value: validate_perturbations(value, count, strength_levels, matched_neutral),
            )
            combined_usage = Counter(result.usage)
            combined_usage.update(fallback_result.usage)
            result = ModelResult(
                terminal_status=fallback_result.terminal_status,
                parsed_response=fallback_result.parsed_response,
                raw_response=fallback_result.raw_response,
                response_model=fallback_result.response_model,
                usage=dict(combined_usage),
                latency_ms=result.latency_ms + fallback_result.latency_ms,
                attempt_count=result.attempt_count + fallback_result.attempt_count,
                attempts=[
                    *({**attempt, "route_model": primary_model} for attempt in result.attempts),
                    *({**attempt, "route_model": fallback_spec["model"]} for attempt in fallback_result.attempts),
                ],
            )
            spec = fallback_spec
        return {
            "item_id": entry["item_id"],
            "source_id": entry["source_id"],
            "base_fact_id": entry["base_fact_id"],
            "manipulation_family": manipulation_family,
            "primary_model": primary_model,
            "fallback_used": fallback_used,
            **model_record(result, spec, perturbation_prompt_version),
        }

    generations = stage_run(
        perturbation_inputs, run_dir / "perturbation_generations.jsonl", "item_id", generate, workers,
        retry_terminal_failures=not freeze_reused_upstream,
    )
    validation_inputs = []
    # The target cap controls perturbation generation only. Keep every
    # accepted distractor here so a multi-option Simulation still has foils.
    distractor_by_id = {d["item_id"]: d for d in context_eligible_distractors}
    for generation in generations:
        if generation["terminal_status"] != "completed":
            continue
        for index, candidate in enumerate(generation["parsed_response"]["candidates"]):
            validation_inputs.append({
                "candidate_id": f"{generation['item_id']}_p{index + 1}",
                "source_id": generation["source_id"],
                "base_fact_id": generation.get("base_fact_id") or generation["source_id"],
                "distractor_id": generation["item_id"],
                "manipulation_family": manipulation_family,
                "candidate": candidate,
            })

    def validate_perturbation(entry: Dict[str, Any]) -> Dict[str, Any]:
        item = item_by_id[entry["source_id"]]
        translation = translation_by_id[entry["source_id"]]["parsed_response"]
        spec = roles["perturbation_validation"]["primary_judge"]
        result = router.request_json(
            "perturbation_validation",
            entry["candidate_id"],
            spec,
            _perturbation_review_prompt(
                item,
                translation,
                distractor_by_id[entry["distractor_id"]],
                entry["candidate"],
                matched_neutral,
                manipulation_family,
            ),
            lambda value: validate_decision(value, perturbation_checks),
        )
        return {**entry, **model_record(result, spec, perturbation_review_prompt_version)}

    perturbations = stage_run(
        validation_inputs, run_dir / "perturbations.jsonl", "candidate_id", validate_perturbation, workers,
        retry_terminal_failures=not freeze_reused_upstream,
    )
    accepted_perturbations = _accepted_perturbation_rows(perturbations, manifest)

    if manifest.get("simulation_requires_codex_proxy_review") and not manifest.get("codex_proxy_review"):
        summary = summarize_run(
            manifest, translations, translation_reviews, distractor_reviews,
            generations, perturbations, read_jsonl(run_dir / "simulation_results.jsonl"),
        )
        write_json(run_dir / "preholdout_summary.json", summary)
        return summary

    simulation_inputs = _build_simulation_inputs(
        accepted_perturbations,
        item_by_id,
        translation_by_id,
        distractor_by_id,
        roles["simulation"]["models"],
        tuple(manifest.get("shared_simulation_variants") or SHARED_SIMULATION_VARIANTS),
        str(manifest.get("neutral_context_mode") or "generic_shared"),
        str(manifest.get("simulation_endpoint") or "binary"),
        int(manifest.get("simulation_max_options") or 3),
    )
    if roles["simulation"].get("batch_by_model", False):
        model_order = {
            (spec["provider_profile"], spec["model"]): index
            for index, spec in enumerate(roles["simulation"]["models"])
        }
        simulation_inputs.sort(key=lambda entry: (
            model_order[(
                entry["model_spec"]["provider_profile"],
                entry["model_spec"]["model"],
            )],
            entry["source_id"],
            entry.get("distractor_id") or "",
            entry["language"],
            entry["variant"],
            entry.get("candidate_id") or "",
        ))

    def simulate(entry: Dict[str, Any]) -> Dict[str, Any]:
        spec = {"temperature": roles["simulation"].get("temperature", 0.0), "max_output_tokens": roles["simulation"].get("max_output_tokens", 64), **entry["model_spec"]}
        result = router.request_json(
            "simulation",
            entry["simulation_id"],
            spec,
            entry["prompt"],
            lambda value: validate_choice(value, tuple(entry["options"])),
        )
        choice = result.parsed_response.get("choice") if result.parsed_response else None
        return {
            key: entry[key]
            for key in (
                "simulation_id", "candidate_id", "candidate_ids", "base_fact_id", "source_id",
                "distractor_id", "designated_distractor_ids", "language", "variant",
                "simulation_scope", "correct_choice",
            )
        } | {
            "target_choice": entry["target_choice"],
            "option_ids": entry["option_ids"],
            "option_count": entry["option_count"],
            "endpoint": entry["endpoint"],
            "choice": choice,
            "correct": choice == entry["correct_choice"],
            "distractor_hit": choice is not None and choice == entry["target_choice"],
            "wrong_choice": choice is not None and choice != entry["correct_choice"],
            **model_record(
                result,
                spec,
                SIMULATION_MULTI_PROMPT_VERSION
                if entry["endpoint"] == "multi_option"
                else SIMULATION_PROMPT_VERSION,
            ),
        }

    simulations = stage_run(simulation_inputs, run_dir / "simulation_results.jsonl", "simulation_id", simulate, workers)
    summary = summarize_run(manifest, translations, translation_reviews, distractor_reviews, generations, perturbations, simulations)
    write_json(run_dir / "preholdout_summary.json", summary)
    return summary


def detect_baseline_inconsistencies(
    simulations: Sequence[Dict[str, Any]],
    shared_variants: Sequence[str] = SHARED_SIMULATION_VARIANTS,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in simulations:
        if row.get("variant") not in shared_variants or row.get("terminal_status") != "completed":
            continue
        key = (
            str(row.get("source_id")), str(row.get("distractor_id")), str(row.get("model")),
            str(row.get("language")), str(row.get("variant")),
        )
        grouped[key].append(row)
    inconsistencies = []
    for key, rows in grouped.items():
        outcomes = {(row.get("choice"), bool(row.get("correct"))) for row in rows}
        if len(outcomes) <= 1:
            continue
        inconsistencies.append({
            "source_id": key[0], "distractor_id": key[1], "model": key[2],
            "language": key[3], "variant": key[4],
            "observed_outcomes": [
                {"choice": choice, "correct": correct}
                for choice, correct in sorted(outcomes, key=lambda value: str(value))
            ],
            "simulation_ids": sorted(str(row.get("simulation_id")) for row in rows),
            "candidate_ids": sorted(str(row.get("candidate_id")) for row in rows if row.get("candidate_id")),
        })
    return inconsistencies


def _candidate_simulation_rows(
    perturbation: Dict[str, Any],
    simulations: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    shared_variants: Sequence[str] = SHARED_SIMULATION_VARIANTS,
) -> List[Dict[str, Any]]:
    if "neutral" not in variants:
        return [row for row in simulations if row.get("candidate_id") == perturbation["candidate_id"]]
    selected = []
    for row in simulations:
        if row.get("variant") not in shared_variants and row.get("candidate_id") == perturbation["candidate_id"]:
            selected.append(row)
        elif (
            row.get("variant") == "original"
            and row.get("simulation_scope") == "shared_base_fact_baseline"
            and row.get("source_id") == perturbation["source_id"]
        ):
            selected.append(row)
        elif (
            row.get("variant") in shared_variants
            and row.get("source_id") == perturbation["source_id"]
            and row.get("distractor_id") == perturbation["distractor_id"]
        ):
            selected.append(row)
    return selected


def _fully_simulated_candidate_ids(
    manifest: Dict[str, Any],
    accepted_perturbations: Sequence[Dict[str, Any]],
    simulations: Sequence[Dict[str, Any]],
) -> Tuple[set[str], int]:
    variants = tuple(manifest.get("simulation_variants") or ("original", "perturbed"))
    shared_variants = tuple(manifest.get("shared_simulation_variants") or SHARED_SIMULATION_VARIANTS)
    languages = tuple(manifest.get("simulation_languages") or SIMULATION_LANGUAGES)
    models = tuple(manifest.get("simulation_models") or sorted({str(row.get("model")) for row in simulations if row.get("model")}))
    expected = {(model, language, variant) for model in models for language in languages for variant in variants}
    fully_simulated = set()
    for perturbation in accepted_perturbations:
        rows = _candidate_simulation_rows(perturbation, simulations, variants, shared_variants)
        completed_keys = {
            (str(row.get("model")), str(row.get("language")), str(row.get("variant")))
            for row in rows if row.get("terminal_status") == "completed"
        }
        if completed_keys == expected:
            fully_simulated.add(perturbation["candidate_id"])
    return fully_simulated, len(expected)


def summarize_run(manifest: Dict[str, Any], translations: Sequence[Dict[str, Any]], reviews: Sequence[Dict[str, Any]], distractors: Sequence[Dict[str, Any]], generations: Sequence[Dict[str, Any]], perturbations: Sequence[Dict[str, Any]], simulations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def completed(rows: Sequence[Dict[str, Any]]) -> int:
        return sum(row.get("terminal_status") == "completed" for row in rows)
    accepted_reviews = {r["source_id"] for r in reviews if r.get("terminal_status") == "completed" and r.get("parsed_response", {}).get("decision") == "accept" and all(r.get("parsed_response", {}).get("checks", {}).values())}
    accepted_distractors = {r["item_id"] for r in distractors if r.get("terminal_status") == "completed" and r.get("parsed_response", {}).get("decision") == "accept" and all(r.get("parsed_response", {}).get("checks", {}).values())}
    accepted_perturbation_rows = _accepted_perturbation_rows(perturbations, manifest)
    accepted_perturbations = {r["candidate_id"] for r in accepted_perturbation_rows}
    fully_simulated, expected_outcomes_per_candidate = _fully_simulated_candidate_ids(
        manifest, accepted_perturbation_rows, simulations
    )
    baseline_inconsistencies = detect_baseline_inconsistencies(
        simulations, tuple(manifest.get("shared_simulation_variants") or SHARED_SIMULATION_VARIANTS)
    )
    usage = Counter()
    latency = []
    attempts = 0
    failed_calls = 0
    for row in [*translations, *reviews, *distractors, *generations, *perturbations, *simulations]:
        usage.update(row.get("usage") or {})
        if isinstance(row.get("latency_ms"), int):
            latency.append(row["latency_ms"])
        attempts += int(row.get("attempt_count") or 0)
        failed_calls += row.get("terminal_status") != "completed"
    gate_reasons = []
    if completed(translations) != manifest["selected_count"]:
        gate_reasons.append("translation_incomplete")
    completed_reviews = completed(reviews)
    if completed_reviews != completed(translations):
        gate_reasons.append("translation_review_incomplete")
    minimum_translation_acceptance = float(manifest.get("minimum_translation_acceptance", 0.5))
    if manifest["selected_count"] and len(accepted_reviews) / manifest["selected_count"] < minimum_translation_acceptance:
        gate_reasons.append("translation_review_acceptance_below_threshold")
    if not accepted_distractors:
        gate_reasons.append("no_verified_distractors")
    if not accepted_perturbations:
        gate_reasons.append("no_verified_perturbations")
    if accepted_perturbations != fully_simulated:
        gate_reasons.append("incomplete_simulation_panel")
    if manifest.get("simulation_requires_codex_proxy_review") and not manifest.get("codex_proxy_review"):
        gate_reasons.append("codex_proxy_review_required_before_simulation")
    if baseline_inconsistencies:
        gate_reasons.append("baseline_response_inconsistent")
    if manifest.get("currency_cost_gate") is None:
        gate_reasons.append("currency_cost_not_auditable")
    model_metrics: Dict[str, Dict[str, Any]] = {}
    all_model_rows = [*translations, *reviews, *distractors, *generations, *perturbations, *simulations]
    for model in sorted({str(row.get("model")) for row in all_model_rows if row.get("model")}):
        model_rows = [row for row in all_model_rows if row.get("model") == model]
        model_latencies = [row["latency_ms"] for row in model_rows if isinstance(row.get("latency_ms"), int)]
        model_usage = Counter()
        for row in model_rows:
            model_usage.update(row.get("usage") or {})
        complete_count = sum(row.get("terminal_status") == "completed" for row in model_rows)
        model_metrics[model] = {
            "call_count": len(model_rows),
            "completed_count": complete_count,
            "failure_rate": round((len(model_rows) - complete_count) / len(model_rows), 6),
            "retry_count": sum(max(0, int(row.get("attempt_count") or 0) - 1) for row in model_rows),
            "latency_ms_mean": round(sum(model_latencies) / len(model_latencies), 2) if model_latencies else None,
            "latency_ms_max": max(model_latencies) if model_latencies else None,
            "usage": dict(model_usage),
        }
    simulation_usage = Counter()
    for row in simulations:
        simulation_usage.update(row.get("usage") or {})
    simulation_latencies = [row["latency_ms"] for row in simulations if isinstance(row.get("latency_ms"), int)]
    simulation_attempts = sum(int(row.get("attempt_count") or 0) for row in simulations)
    base_fact_id_by_source = {
        row["source_id"]: str(row.get("base_fact_id") or row["source_id"])
        for row in manifest.get("records", [])
        if row.get("source_id")
    }
    assigned_base_fact_ids = set(base_fact_id_by_source.values())
    calibration_base_fact_ids = {
        str(
            row.get("base_fact_id")
            or base_fact_id_by_source.get(row["source_id"])
            or row["source_id"]
        )
        for row in accepted_perturbation_rows
    }
    target_variants = {
        (
            str(
                row.get("base_fact_id")
                or base_fact_id_by_source.get(row["source_id"])
                or row["source_id"]
            ),
            row["distractor_id"],
        )
        for row in accepted_perturbation_rows
    }
    return {
        "schema_version": "factual-perturbation-run-summary-v1",
        "generated_at": utc_now(), "run_id": manifest["run_id"],
        "base_triple_count": manifest["selected_count"],
        "statistical_unit": "base_fact_id",
        "base_fact_count": len(assigned_base_fact_ids),
        "calibration_base_fact_count": len(calibration_base_fact_ids),
        "target_variant_count": len(target_variants),
        "manipulation_family": manifest.get(
            "manipulation_family", TRUTHFUL_SALIENCE_MANIPULATION_FAMILY
        ),
        "calibration_base_triple_count": len({
            row["source_id"] for row in accepted_perturbation_rows
        }),
        "counts": {
            "translations_completed": completed(translations), "translations_review_accepted": len(accepted_reviews),
            "distractors_reviewed": len(distractors), "distractors_accepted": len(accepted_distractors),
            "perturbation_groups_completed": completed(generations), "perturbations_reviewed": len(perturbations),
            "perturbations_accepted": len(accepted_perturbations), "simulation_calls": len(simulations),
            "simulation_calls_completed": completed(simulations), "fully_simulated_candidates": len(fully_simulated),
            "expected_simulation_outcomes_per_candidate": expected_outcomes_per_candidate,
            "baseline_inconsistency_count": len(baseline_inconsistencies),
            "endpoint_excluded_candidate_count": len(
                manifest.get("endpoint_excluded_candidate_ids") or []
            ),
            "failed_terminal_calls": failed_calls, "total_attempts": attempts,
        },
        "usage": dict(usage),
        "latency_ms": {"count": len(latency), "mean": round(sum(latency) / len(latency), 2) if latency else None, "max": max(latency) if latency else None},
        "simulation_only": {
            "physical_call_count": len(simulations),
            "completed_count": completed(simulations),
            "attempt_count": simulation_attempts,
            "retry_count": max(0, simulation_attempts - len(simulations)),
            "usage": dict(simulation_usage),
            "latency_ms_mean": round(sum(simulation_latencies) / len(simulation_latencies), 2) if simulation_latencies else None,
            "latency_ms_max": max(simulation_latencies) if simulation_latencies else None,
        },
        "cost": {"estimated_usd": None, "reason": "provider_price_table_not_configured; token usage retained for billing reconciliation"},
        "model_metrics": model_metrics,
        "baseline_inconsistencies": baseline_inconsistencies,
        "gate_passed": not gate_reasons,
        "gate_reasons": gate_reasons,
        "paths_not_taken_enabled": False,
        "holdout_called": False,
    }


def audit_run(run_dir: Path) -> Dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("execution_mode") in STATIC_BEHAVIOR_EXECUTION_MODES:
        inputs = _load_static_proxy_behavior_inputs(manifest, run_dir)
        probe_manifest = read_json(
            run_dir / STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME
        )
        summary = _summarize_static_proxy_run(
            manifest,
            inputs,
            read_jsonl(run_dir / "simulation_results.jsonl"),
            probe_manifest,
        )
        write_json(run_dir / "preholdout_summary.json", summary)
        return summary
    summary = summarize_run(
        manifest,
        read_jsonl(run_dir / "translations.jsonl"),
        read_jsonl(run_dir / "translation_reviews.jsonl"),
        read_jsonl(run_dir / "verified_distractors.jsonl"),
        read_jsonl(run_dir / "perturbation_generations.jsonl"),
        read_jsonl(run_dir / "perturbations.jsonl"),
        read_jsonl(run_dir / "simulation_results.jsonl"),
    )
    write_json(run_dir / "preholdout_summary.json", summary)
    return summary


def analyze_three_arm(run_dir: Path) -> Dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("execution_mode") in STATIC_BEHAVIOR_EXECUTION_MODES:
        _load_static_proxy_behavior_inputs(manifest, run_dir)
    manipulation_family = manifest.get(
        "manipulation_family", TRUTHFUL_SALIENCE_MANIPULATION_FAMILY
    )
    variants = tuple(manifest.get("simulation_variants") or ())
    if variants != SIMULATION_VARIANTS:
        raise ValueError("three_arm_analysis_requires_original_neutral_targeted")
    perturbations = _accepted_perturbation_rows(read_jsonl(run_dir / "perturbations.jsonl"), manifest)
    simulations = read_jsonl(run_dir / "simulation_results.jsonl")
    models = tuple(manifest["simulation_models"])
    shared_variants = tuple(manifest.get("shared_simulation_variants") or SHARED_SIMULATION_VARIANTS)
    model_count = len(models)
    base_fact_id_by_source = {
        row["source_id"]: str(row.get("base_fact_id") or row["source_id"])
        for row in manifest.get("records", [])
        if row.get("source_id")
    }
    candidate_results = []
    control_instabilities: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for perturbation in perturbations:
        rows = _candidate_simulation_rows(perturbation, simulations, variants, shared_variants)
        by_key = {
            (row.get("model"), row.get("language"), row.get("variant")): row
            for row in rows if row.get("terminal_status") == "completed"
        }
        expected = {(model, language, variant) for model in models for language in SIMULATION_LANGUAGES for variant in variants}
        accuracy = {}
        distractor_hit_rate = {}
        coverage = {}
        for language in SIMULATION_LANGUAGES:
            for variant in variants:
                arm_rows = [
                    by_key[(model, language, variant)]
                    for model in models if (model, language, variant) in by_key
                ]
                name = f"{language}_{variant}"
                coverage[name] = len(arm_rows)
                accuracy[name] = (
                    round(sum(bool(row.get("correct")) for row in arm_rows) / model_count, 6)
                    if len(arm_rows) == model_count else None
                )
                distractor_hit_rate[name] = (
                    round(sum(bool(row.get("distractor_hit")) for row in arm_rows) / model_count, 6)
                    if len(arm_rows) == model_count else None
                )
        targeted_zh_wrong_models = []
        targeted_zh_flips = []
        targeted_zh_nontarget_wrong_models = []
        strict_targeted_zh_flip_models = []
        proxy_directed_candidate_models = []
        proxy_zh_specific_strict_models = []
        neutral_unstable_models = []
        off_target_failure_models = []
        shared_language_susceptibility_models = []
        incomplete_models = []
        model_statuses: Dict[str, Dict[str, Any]] = {}
        for model in models:
            model_expected = {
                (model, language, variant)
                for language in SIMULATION_LANGUAGES
                for variant in variants
            }
            model_complete = all(key in by_key for key in model_expected)
            original = by_key.get((model, "zh", "original"), {})
            neutral = by_key.get((model, "zh", "neutral"), {})
            targeted = by_key.get((model, "zh", "targeted"), {})
            neutral_unstable_languages = [
                language for language in SIMULATION_LANGUAGES
                if by_key.get((model, language, "original"), {}).get("correct") is True
                and by_key.get((model, language, "neutral"), {}).get("correct") is False
            ] if model_complete else []
            off_target_failure_languages = [
                language for language in SIMULATION_LANGUAGES
                if by_key.get((model, language, "targeted"), {}).get("correct") is False
                and by_key.get((model, language, "targeted"), {}).get(
                    "distractor_hit"
                ) is not True
            ] if model_complete else []
            shared_language_susceptibility = False
            proxy_directed_candidate = False
            proxy_zh_specific_strict = False
            if original.get("correct") is True and neutral.get("correct") is True and targeted.get("correct") is False:
                targeted_zh_wrong_models.append(model)
                if targeted.get("distractor_hit") is True:
                    targeted_zh_flips.append(model)
                    proxy_directed_candidate = model_complete
                    if proxy_directed_candidate:
                        proxy_directed_candidate_models.append(model)
                    english_controls_pass = all(
                        by_key.get((model, "en", variant), {}).get("correct") is True
                        for variant in SIMULATION_VARIANTS
                    )
                    english_baselines_pass = all(
                        by_key.get((model, "en", variant), {}).get("correct") is True
                        for variant in ("original", "neutral")
                    )
                    shared_language_susceptibility = bool(
                        model_complete
                        and english_baselines_pass
                        and by_key.get((model, "en", "targeted"), {}).get(
                            "distractor_hit"
                        ) is True
                    )
                    if model_complete and english_controls_pass:
                        proxy_zh_specific_strict = True
                        strict_targeted_zh_flip_models.append(model)
                        proxy_zh_specific_strict_models.append(model)
                else:
                    targeted_zh_nontarget_wrong_models.append(model)
            if not model_complete:
                incomplete_models.append(model)
            if neutral_unstable_languages:
                neutral_unstable_models.append(model)
            if off_target_failure_languages:
                off_target_failure_models.append(model)
            if shared_language_susceptibility:
                shared_language_susceptibility_models.append(model)
            model_statuses[model] = {
                "complete": model_complete,
                "evaluable": model_complete,
                "proxy_directed_candidate": proxy_directed_candidate,
                "proxy_zh_specific_strict": proxy_zh_specific_strict,
                "neutral_unstable": bool(neutral_unstable_languages),
                "neutral_unstable_languages": neutral_unstable_languages,
                "off_target_failure": bool(off_target_failure_languages),
                "off_target_failure_languages": off_target_failure_languages,
                "shared_language_susceptibility": shared_language_susceptibility,
            }
        for model in models:
            for language in SIMULATION_LANGUAGES:
                original = by_key.get((model, language, "original"), {})
                neutral = by_key.get((model, language, "neutral"), {})
                if original.get("terminal_status") == "completed" and neutral.get("terminal_status") == "completed" and original.get("choice") != neutral.get("choice"):
                    key = (perturbation["source_id"], perturbation["distractor_id"], model, language)
                    control_instabilities[key] = {
                        "source_id": key[0], "distractor_id": key[1], "model": key[2], "language": key[3],
                        "original_choice": original.get("choice"), "neutral_choice": neutral.get("choice"),
                    }
        complete = set(by_key) == expected
        base_fact_id = str(
            perturbation.get("base_fact_id")
            or base_fact_id_by_source.get(perturbation["source_id"])
            or perturbation["source_id"]
        )
        candidate_results.append({
            "candidate_id": perturbation["candidate_id"],
            "base_fact_id": base_fact_id,
            "source_id": perturbation["source_id"],
            "distractor_id": perturbation["distractor_id"],
            "complete": complete,
            "coverage": coverage,
            "accuracy": accuracy,
            "distractor_hit_rate": distractor_hit_rate,
            "targeted_zh_wrong_models": targeted_zh_wrong_models,
            "targeted_zh_flip_models": targeted_zh_flips,
            "targeted_zh_nontarget_wrong_models": targeted_zh_nontarget_wrong_models,
            "strict_targeted_zh_flip_models": strict_targeted_zh_flip_models,
            "proxy_directed_candidate_models": proxy_directed_candidate_models,
            "proxy_zh_specific_strict_models": proxy_zh_specific_strict_models,
            "neutral_unstable_models": neutral_unstable_models,
            "off_target_failure_models": off_target_failure_models,
            "shared_language_susceptibility_models": (
                shared_language_susceptibility_models
            ),
            "incomplete_models": incomplete_models,
            "model_statuses": model_statuses,
            "paths_not_taken_candidate": bool(
                manipulation_family != EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                and complete
                and strict_targeted_zh_flip_models
            ),
        })
    naive_calls = len(perturbations) * len(models) * len(SIMULATION_LANGUAGES) * len(variants)
    strict_signal_count_by_model = Counter(
        model
        for row in candidate_results
        for model in row["strict_targeted_zh_flip_models"]
    )
    candidates_by_base_fact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_results:
        candidates_by_base_fact[row["base_fact_id"]].append(row)
    manifest_records_by_base_fact: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest.get("records", []):
        base_fact_id = str(row.get("base_fact_id") or row.get("source_id") or "")
        if base_fact_id:
            manifest_records_by_base_fact[base_fact_id].append(row)
    assigned_base_fact_ids = set(manifest_records_by_base_fact) or set(
        candidates_by_base_fact
    )
    configured_variants_per_fact = int(
        manifest.get("designated_distractors_per_base_fact") or 0
    )
    if (
        manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
        and configured_variants_per_fact != 2
    ):
        raise ValueError(
            "explicit_false_assertion_analysis_requires_two_designated_distractors"
        )
    base_fact_results = []
    for base_fact_id in sorted(assigned_base_fact_ids):
        rows = candidates_by_base_fact.get(base_fact_id, [])
        designated_distractor_ids = sorted({row["distractor_id"] for row in rows})
        expected_variant_count = (
            configured_variants_per_fact
            if configured_variants_per_fact > 0
            else len(designated_distractor_ids)
        )
        model_labels = {}
        for model in models:
            directed_ids = sorted({
                row["distractor_id"]
                for row in rows
                if row["model_statuses"][model]["proxy_directed_candidate"]
            })
            zh_specific_ids = sorted({
                row["distractor_id"]
                for row in rows
                if row["model_statuses"][model]["proxy_zh_specific_strict"]
            })
            evaluable_ids = sorted({
                row["distractor_id"]
                for row in rows
                if row["model_statuses"][model]["evaluable"]
            })
            neutral_unstable_ids = sorted({
                row["distractor_id"]
                for row in rows
                if row["model_statuses"][model]["neutral_unstable"]
            })
            off_target_failure_ids = sorted({
                row["distractor_id"]
                for row in rows
                if row["model_statuses"][model]["off_target_failure"]
            })
            shared_language_susceptibility_ids = sorted({
                row["distractor_id"]
                for row in rows
                if row["model_statuses"][model][
                    "shared_language_susceptibility"
                ]
            })
            ten_input_terminal = bool(
                expected_variant_count == 2
                and len(rows) == expected_variant_count
                and len(designated_distractor_ids) == expected_variant_count
                and len(evaluable_ids) == expected_variant_count
            )
            incomplete_variant_count = max(
                0, expected_variant_count - len(evaluable_ids)
            )
            model_labels[model] = {
                "ten_input_terminal": ten_input_terminal,
                "evaluable_variant_count": len(evaluable_ids),
                "evaluable_distractor_ids": evaluable_ids,
                "incomplete": incomplete_variant_count > 0,
                "incomplete_variant_count": incomplete_variant_count,
                "proxy_directed_candidate": bool(directed_ids),
                "proxy_directed_variant_count": len(directed_ids),
                "proxy_directed_distractor_ids": directed_ids,
                "proxy_directed_any": bool(directed_ids),
                "proxy_directed_both": bool(
                    expected_variant_count == 2
                    and len(directed_ids) == expected_variant_count
                ),
                "proxy_zh_specific_strict": bool(zh_specific_ids),
                "proxy_zh_specific_strict_variant_count": len(zh_specific_ids),
                "proxy_zh_specific_strict_distractor_ids": zh_specific_ids,
                "proxy_zh_specific_strict_any": bool(zh_specific_ids),
                "proxy_zh_specific_strict_both": bool(
                    expected_variant_count == 2
                    and len(zh_specific_ids) == expected_variant_count
                ),
                "neutral_unstable": bool(neutral_unstable_ids),
                "neutral_unstable_variant_count": len(neutral_unstable_ids),
                "neutral_unstable_distractor_ids": neutral_unstable_ids,
                "off_target_failure": bool(off_target_failure_ids),
                "off_target_failure_variant_count": len(off_target_failure_ids),
                "off_target_failure_distractor_ids": off_target_failure_ids,
                "shared_language_susceptibility": bool(
                    shared_language_susceptibility_ids
                ),
                "shared_language_susceptibility_variant_count": len(
                    shared_language_susceptibility_ids
                ),
                "shared_language_susceptibility_distractor_ids": (
                    shared_language_susceptibility_ids
                ),
            }
        source_ids = sorted({row["source_id"] for row in rows})
        if not source_ids:
            source_ids = sorted({
                str(row.get("source_id"))
                for row in manifest_records_by_base_fact.get(base_fact_id, [])
                if row.get("source_id")
            })
        base_fact_results.append({
            "base_fact_id": base_fact_id,
            "source_ids": source_ids,
            "designated_distractor_ids": designated_distractor_ids,
            "target_variant_count": len(designated_distractor_ids),
            "expected_target_variant_count": expected_variant_count,
            "candidate_count": len(rows),
            "complete": bool(
                models
                and all(
                    model_labels[model]["ten_input_terminal"] for model in models
                )
            ),
            "model_labels": model_labels,
        })
    assigned_variant_count = sum(
        row["expected_target_variant_count"] for row in base_fact_results
    )
    proxy_funnel_by_model = {}
    for model in models:
        labels = [row["model_labels"][model] for row in base_fact_results]
        proxy_funnel_by_model[model] = {
            "N_assigned_facts": len(base_fact_results),
            "N_assigned_variants": assigned_variant_count,
            "N_10_input_terminal_facts": sum(
                label["ten_input_terminal"] for label in labels
            ),
            "N_evaluable_variants": sum(
                label["evaluable_variant_count"] for label in labels
            ),
            "N_proxy_directed_candidate_variants": sum(
                label["proxy_directed_variant_count"] for label in labels
            ),
            "N_proxy_zh_specific_strict_variants": sum(
                label["proxy_zh_specific_strict_variant_count"] for label in labels
            ),
            "N_neutral_unstable": sum(
                label["neutral_unstable_variant_count"] for label in labels
            ),
            "N_off_target_failure": sum(
                label["off_target_failure_variant_count"] for label in labels
            ),
            "N_shared_language_susceptibility": sum(
                label["shared_language_susceptibility_variant_count"]
                for label in labels
            ),
            "N_incomplete": sum(
                label["incomplete_variant_count"] for label in labels
            ),
            "base_fact_level": {
                "N_proxy_directed_candidate_any": sum(
                    label["proxy_directed_any"] for label in labels
                ),
                "N_proxy_directed_candidate_both": sum(
                    label["proxy_directed_both"] for label in labels
                ),
                "N_proxy_zh_specific_strict_any": sum(
                    label["proxy_zh_specific_strict_any"] for label in labels
                ),
                "N_proxy_zh_specific_strict_both": sum(
                    label["proxy_zh_specific_strict_both"] for label in labels
                ),
                "N_neutral_unstable_any": sum(
                    label["neutral_unstable"] for label in labels
                ),
                "N_off_target_failure_any": sum(
                    label["off_target_failure"] for label in labels
                ),
                "N_shared_language_susceptibility_any": sum(
                    label["shared_language_susceptibility"] for label in labels
                ),
                "N_incomplete_facts": sum(
                    label["incomplete"] for label in labels
                ),
            },
        }
    proxy_directed_variant_count_by_model = {
        model: proxy_funnel_by_model[model][
            "N_proxy_directed_candidate_variants"
        ]
        for model in models
    }
    proxy_zh_specific_strict_variant_count_by_model = {
        model: proxy_funnel_by_model[model][
            "N_proxy_zh_specific_strict_variants"
        ]
        for model in models
    }
    proxy_directed_base_fact_count_by_model = {
        model: proxy_funnel_by_model[model]["base_fact_level"][
            "N_proxy_directed_candidate_any"
        ]
        for model in models
    }
    proxy_zh_specific_strict_base_fact_count_by_model = {
        model: proxy_funnel_by_model[model]["base_fact_level"][
            "N_proxy_zh_specific_strict_any"
        ]
        for model in models
    }
    model_arm_metrics: Dict[str, Dict[str, Any]] = {}
    for model in models:
        model_arm_metrics[model] = {}
        for language in SIMULATION_LANGUAGES:
            for variant in variants:
                arm_rows = [
                    row for row in simulations
                    if row.get("model") == model
                    and row.get("language") == language
                    and row.get("variant") == variant
                ]
                completed_rows = [
                    row for row in arm_rows if row.get("terminal_status") == "completed"
                ]
                model_arm_metrics[model][f"{language}_{variant}"] = {
                    "call_count": len(arm_rows),
                    "completed_count": len(completed_rows),
                    "accuracy_completed": (
                        round(
                            sum(bool(row.get("correct")) for row in completed_rows)
                            / len(completed_rows),
                            6,
                        )
                        if completed_rows else None
                    ),
                    "distractor_hit_rate_completed": (
                        round(
                            sum(bool(row.get("distractor_hit")) for row in completed_rows)
                            / len(completed_rows),
                            6,
                        )
                        if completed_rows else None
                    ),
                }
    result = {
        "schema_version": "factual-perturbation-three-arm-analysis-v1",
        "generated_at": utc_now(),
        "analysis_runtime_sha256": _runtime_sha256(),
        "run_id": manifest["run_id"],
        "manipulation_family": manipulation_family,
        "statistical_unit": "base_fact_id",
        "panel_union_reported": False,
        "proxy_behavior_only": (
            manifest.get("execution_mode") in STATIC_BEHAVIOR_EXECUTION_MODES
        ),
        "exact_hf_evidence": False,
        "pnt_authorized": False,
        "pnt_eligible": False,
        "reviewer_type": manifest.get("codex_proxy_review", {}).get("reviewer_type"),
        "not_human_gold": manifest.get("codex_proxy_review", {}).get("not_human_gold"),
        "candidate_count": len(perturbations),
        "assigned_base_fact_count": len(assigned_base_fact_ids),
        "assigned_target_variant_count": assigned_variant_count,
        "analyzed_base_fact_count": len(base_fact_results),
        "complete_base_fact_count": sum(row["complete"] for row in base_fact_results),
        "target_variant_count": len({
            (row["base_fact_id"], row["distractor_id"])
            for row in candidate_results
        }),
        "complete_candidate_count": sum(row["complete"] for row in candidate_results),
        "paths_not_taken_candidate_count": sum(row["paths_not_taken_candidate"] for row in candidate_results),
        "strict_signal_count_by_model": {
            model: strict_signal_count_by_model.get(model, 0) for model in models
        },
        "proxy_directed_candidate_variant_count_by_model": (
            proxy_directed_variant_count_by_model
        ),
        "proxy_directed_candidate_base_fact_count_by_model": (
            proxy_directed_base_fact_count_by_model
        ),
        "proxy_zh_specific_strict_variant_count_by_model": (
            proxy_zh_specific_strict_variant_count_by_model
        ),
        "proxy_zh_specific_strict_base_fact_count_by_model": (
            proxy_zh_specific_strict_base_fact_count_by_model
        ),
        "proxy_funnel_by_model": proxy_funnel_by_model,
        "model_arm_metrics": model_arm_metrics,
        "physical_simulation_calls": len(simulations),
        "naive_unshared_call_count": naive_calls,
        "shared_baseline_calls_saved": naive_calls - len(simulations),
        "baseline_inconsistencies": detect_baseline_inconsistencies(simulations, shared_variants),
        "neutral_control_instabilities": list(control_instabilities.values()),
        "candidate_results": candidate_results,
        "base_fact_results": base_fact_results,
        "conclusion": (
            "Explicit-false proxy labels are reported separately for each model; no panel or model-union label is produced."
            if manipulation_family == EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            else
            "At least one target-distractor-specific Chinese flip survived the shared-baseline, neutral-control rerun."
            if any(row["paths_not_taken_candidate"] for row in candidate_results)
            else "No target-distractor-specific Chinese flip survived the shared-baseline, neutral-control rerun."
        ),
    }
    write_json(run_dir / "three_arm_analysis.json", result)
    return result


def freeze_candidates(run_dir: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest = read_json(run_dir / "run_manifest.json")
    perturbations = {
        row["candidate_id"]: row
        for row in _accepted_perturbation_rows(read_jsonl(run_dir / "perturbations.jsonl"), manifest)
    }
    simulations = read_jsonl(run_dir / "simulation_results.jsonl")
    variants = tuple(manifest.get("simulation_variants") or ("original", "perturbed"))
    shared_variants = tuple(manifest.get("shared_simulation_variants") or SHARED_SIMULATION_VARIANTS)
    model_count = len(manifest.get("simulation_models") or config["model_roles"]["simulation"]["models"])
    expected_count = model_count * len(SIMULATION_LANGUAGES) * len(variants)
    base_fact_id_by_source = {
        row["source_id"]: str(row.get("base_fact_id") or row["source_id"])
        for row in manifest.get("records", [])
        if row.get("source_id")
    }
    frozen = []
    for candidate_id, perturbation in perturbations.items():
        rows = _candidate_simulation_rows(perturbation, simulations, variants, shared_variants)
        completed_keys = {
            (row.get("model"), row.get("language"), row.get("variant"))
            for row in rows if row.get("terminal_status") == "completed"
        }
        if len(completed_keys) != expected_count:
            continue
        acc = {
            (language, variant): sum(
                bool(r["correct"])
                for r in rows
                if r["language"] == language and r["variant"] == variant and r.get("terminal_status") == "completed"
            ) / model_count
            for language in SIMULATION_LANGUAGES for variant in variants
        }
        treatment_variant = "targeted" if "targeted" in variants else "perturbed"
        english_ok = acc[("en", treatment_variant)] >= float(config["scoring"]["english_perturbed_accuracy_required"])
        if "neutral" in variants:
            control_floor = min(acc[("zh", "original")], acc[("zh", "neutral")])
            induced = control_floor == 1.0 and acc[("zh", treatment_variant)] < 1.0
            amplified = control_floor < 1.0 and acc[("zh", treatment_variant)] < control_floor
        else:
            induced = acc[("zh", "original")] == 1.0 and acc[("zh", treatment_variant)] < 1.0
            amplified = acc[("zh", "original")] < 1.0 and acc[("zh", treatment_variant)] < acc[("zh", "original")]
        if english_ok and (induced or amplified):
            frozen.append({
                "schema_version": CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
                "candidate_id": candidate_id, "source_id": perturbation["source_id"],
                "base_fact_id": str(
                    perturbation.get("base_fact_id")
                    or base_fact_id_by_source.get(perturbation["source_id"])
                    or perturbation["source_id"]
                ),
                "distractor_id": perturbation["distractor_id"], "candidate": perturbation["candidate"],
                "badcase_type": "induced" if induced else "amplified",
                "accuracy": {f"{k[0]}_{k[1]}": v for k, v in acc.items()},
                "freeze_version": CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
            })
    frozen_path = run_dir / "frozen_candidates.jsonl"
    write_jsonl(frozen_path, frozen)
    candidate_ids_sha256 = sha256_value([row["candidate_id"] for row in frozen])
    write_json(run_dir / "candidate_freeze_manifest.json", {
        "schema_version": CANDIDATE_FREEZE_MANIFEST_SCHEMA_VERSION, "created_at": utc_now(),
        "candidate_count": len(frozen), "candidate_ids_sha256": candidate_ids_sha256,
        "frozen_candidates": {
            "path": frozen_path.name,
            "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "record_count": len(frozen),
            "schema_version": CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
            "candidate_ids_sha256": candidate_ids_sha256,
        },
        "holdout_eligible": True, "paths_not_taken_enabled": False,
    })
    return frozen


def run_holdout(config: Dict[str, Any], run_dir: Path, env_path: Path) -> Dict[str, Any]:
    freeze_manifest = read_json(run_dir / "candidate_freeze_manifest.json")
    if freeze_manifest.get("schema_version") != CANDIDATE_FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported candidate freeze manifest schema")
    if freeze_manifest.get("holdout_eligible") is not True:
        raise ValueError("candidate freeze is not holdout eligible")
    if freeze_manifest.get("paths_not_taken_enabled") is not False:
        raise ValueError("candidate freeze must keep Paths Not Taken disabled")
    frozen_binding = freeze_manifest.get("frozen_candidates")
    if not isinstance(frozen_binding, dict):
        raise ValueError("candidate freeze manifest is missing frozen_candidates binding")
    if frozen_binding.get("path") != "frozen_candidates.jsonl":
        raise ValueError("candidate freeze artifact path is invalid")
    if frozen_binding.get("schema_version") != CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION:
        raise ValueError("candidate freeze artifact schema is invalid")
    manifest = read_json(run_dir / "run_manifest.json")
    manifest_runtime_sha256 = manifest.get("runtime_sha256")
    if not manifest_runtime_sha256:
        raise ValueError("run manifest is missing runtime fingerprint; use a new run_id")
    if manifest_runtime_sha256 != _runtime_sha256():
        raise ValueError("run manifest runtime fingerprint does not match current code")
    if manifest.get("config_sha256") != sha256_value(config):
        raise ValueError("run manifest config fingerprint does not match current config")
    frozen, frozen_snapshot = _read_jsonl_snapshot(
        run_dir / "frozen_candidates.jsonl"
    )
    if frozen_binding.get("sha256") != frozen_snapshot["sha256"]:
        raise ValueError("candidate freeze artifact SHA-256 mismatch")
    if (
        type(frozen_binding.get("record_count")) is not int
        or frozen_binding["record_count"] != frozen_snapshot["record_count"]
        or type(freeze_manifest.get("candidate_count")) is not int
        or freeze_manifest["candidate_count"] != frozen_snapshot["record_count"]
    ):
        raise ValueError("candidate freeze artifact record count mismatch")
    candidate_ids: List[str] = []
    variants = tuple(manifest.get("simulation_variants") or ("original", "perturbed"))
    expected_accuracy_keys = {
        f"{language}_{variant}"
        for language in SIMULATION_LANGUAGES
        for variant in variants
    }
    for index, row in enumerate(frozen, start=1):
        if row.get("schema_version") != CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"candidate freeze row {index} has invalid schema")
        if row.get("freeze_version") != CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION:
            raise ValueError(f"candidate freeze row {index} has invalid freeze_version")
        for field in ("candidate_id", "source_id", "distractor_id"):
            identifier = row.get(field)
            if (
                not isinstance(identifier, str)
                or not identifier.strip()
                or identifier != identifier.strip()
            ):
                raise ValueError(f"candidate freeze row {index} has invalid {field}")
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate freeze row {index} has invalid candidate object")
        for field in ("english_context", "chinese_context"):
            context = candidate.get(field)
            if not isinstance(context, str) or not context.strip():
                raise ValueError(
                    f"candidate freeze row {index} has invalid candidate.{field}"
                )
        if row.get("badcase_type") not in {"induced", "amplified"}:
            raise ValueError(f"candidate freeze row {index} has invalid badcase_type")
        accuracy = row.get("accuracy")
        if not isinstance(accuracy, dict):
            raise ValueError(f"candidate freeze row {index} has invalid accuracy object")
        if set(accuracy) != expected_accuracy_keys:
            raise ValueError(f"candidate freeze row {index} has invalid accuracy keys")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in accuracy.values()
        ):
            raise ValueError(f"candidate freeze row {index} has invalid accuracy value")
        candidate_ids.append(row["candidate_id"])
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate freeze artifact contains duplicate candidate_id values")
    candidate_ids_sha256 = sha256_value(candidate_ids)
    if (
        frozen_binding.get("candidate_ids_sha256") != candidate_ids_sha256
        or freeze_manifest.get("candidate_ids_sha256") != candidate_ids_sha256
    ):
        raise ValueError("candidate freeze candidate ID digest mismatch")
    item_by_id = {r["source_id"]: r for r in manifest["records"]}
    translations = {r["source_id"]: r["parsed_response"] for r in read_jsonl(run_dir / "translations.jsonl") if r.get("terminal_status") == "completed"}
    distractors = {r["item_id"]: r for r in read_jsonl(run_dir / "verified_distractors.jsonl") if r.get("terminal_status") == "completed"}
    router = ModelRouter(config, env_path, run_dir / "redacted_events.jsonl")
    spec = config["model_roles"]["non_simulation_holdout"]["models"][0]
    inputs = []
    for candidate in frozen:
        item = item_by_id[candidate["source_id"]]
        translation = translations[candidate["source_id"]]
        distractor = distractors[candidate["distractor_id"]]
        for language in ("en", "zh"):
            answer = item["answer_en"] if language == "en" else translation["answer_zh"]
            distractor_text = distractor["distractor_en"] if language == "en" else distractor["distractor_zh"]
            options, correct_choice = _choice_layout(item["source_id"], candidate["distractor_id"], language, answer, distractor_text)
            context = candidate["candidate"]["english_context" if language == "en" else "chinese_context"]
            prompt_text = item["prompt_en"] if language == "en" else translation["prompt_zh"]
            inputs.append({"holdout_id": sha256_value([candidate["candidate_id"], language, spec["model"]])[:24], "candidate_id": candidate["candidate_id"], "source_id": item["source_id"], "language": language, "correct_choice": correct_choice, "prompt": _simulation_prompt(prompt_text, context, options, language)})

    def evaluate(entry: Dict[str, Any]) -> Dict[str, Any]:
        result = router.request_json("holdout", entry["holdout_id"], spec, entry["prompt"], validate_choice)
        choice = result.parsed_response.get("choice") if result.parsed_response else None
        return {key: entry[key] for key in ("holdout_id", "candidate_id", "source_id", "language", "correct_choice")} | {"choice": choice, "correct": choice == entry["correct_choice"], **model_record(result, spec, HOLDOUT_PROMPT_VERSION)}

    results = stage_run(inputs, run_dir / "holdout_results.jsonl", "holdout_id", evaluate, int(config["execution"].get("max_workers", 4)))
    summary = {"schema_version": "factual-perturbation-holdout-summary-v1", "candidate_count": len(frozen), "call_count": len(results), "completed_count": sum(r["terminal_status"] == "completed" for r in results), "english_accuracy": _accuracy(results, "en"), "chinese_accuracy": _accuracy(results, "zh"), "freeze_manifest_sha256": sha256_value(freeze_manifest)}
    write_json(run_dir / "holdout_summary.json", summary)
    return summary


def _accuracy(rows: Sequence[Dict[str, Any]], language: str) -> Optional[float]:
    selected = [r for r in rows if r.get("language") == language and r.get("terminal_status") == "completed"]
    return round(sum(bool(r.get("correct")) for r in selected) / len(selected), 6) if selected else None


def build_report(run_dir: Path) -> str:
    summary = read_json(run_dir / "preholdout_summary.json")
    three_arm = read_json(run_dir / "three_arm_analysis.json") if (run_dir / "three_arm_analysis.json").exists() else None
    freeze = read_json(run_dir / "candidate_freeze_manifest.json") if (run_dir / "candidate_freeze_manifest.json").exists() else None
    holdout = read_json(run_dir / "holdout_summary.json") if (run_dir / "holdout_summary.json").exists() else None
    if summary.get("execution_mode") in STATIC_BEHAVIOR_EXECUTION_MODES:
        lines = [
            f"# Development proxy behavior report: {summary['run_id']}",
            "",
            f"- Proxy screen complete: `{str(summary['proxy_screen_complete']).lower()}`",
            f"- Assigned Development facts: {summary['assigned_base_fact_count']}",
            f"- Frozen stimuli: {summary['static_stimulus_count']}",
            (
                "- Completed behavior calls: "
                f"{summary['completed_simulation_call_count']}/"
                f"{summary['expected_simulation_call_count']}"
            ),
            (
                "- Completed calls by model: `"
                + json.dumps(summary["completed_call_count_by_model"], sort_keys=True)
                + "`"
            ),
            f"- Validation behavior rows: {summary['validation_behavior_result_count']}",
            f"- Sealed behavior rows: {summary['sealed_behavior_result_count']}",
            "- Evidence class: proxy behavior only; not exact-HF evidence.",
            "- PNT eligible: `false`",
            f"- Operational reasons: `{json.dumps(summary['operational_reasons'])}`",
        ]
        if three_arm:
            lines.extend(
                [
                    "",
                    "## Per-model proxy labels",
                    "",
                    (
                        "- Directed variant counts: `"
                        + json.dumps(
                            three_arm.get(
                                "proxy_directed_candidate_variant_count_by_model", {}
                            ),
                            sort_keys=True,
                        )
                        + "`"
                    ),
                    (
                        "- ZH-specific strict variant counts: `"
                        + json.dumps(
                            three_arm.get(
                                "proxy_zh_specific_strict_variant_count_by_model", {}
                            ),
                            sort_keys=True,
                        )
                        + "`"
                    ),
                    "- Panel/union label emitted: `false`",
                    "- PNT eligible: `false`",
                ]
            )
        report = "\n".join(lines) + "\n"
        (run_dir / "experiment_report.md").write_text(report, encoding="utf-8")
        return report
    counts = summary["counts"]
    lines = [
        f"# Factual perturbation report: {summary['run_id']}", "",
        f"- Gate passed: `{str(summary['gate_passed']).lower()}`", f"- Base triples: {summary['base_triple_count']}",
        f"- Translation reviews accepted: {counts['translations_review_accepted']}",
        f"- Distractors accepted: {counts['distractors_accepted']}", f"- Perturbations accepted: {counts['perturbations_accepted']}",
        f"- Fully simulated candidates: {counts['fully_simulated_candidates']}", f"- Failed terminal calls: {counts['failed_terminal_calls']}",
        f"- Expected outcomes per candidate: {counts.get('expected_simulation_outcomes_per_candidate')}",
        f"- Baseline inconsistencies: {counts.get('baseline_inconsistency_count', 0)}",
        f"- Total attempts: {counts['total_attempts']}", f"- Mean/max latency ms: {summary['latency_ms']['mean']} / {summary['latency_ms']['max']}",
        f"- Token usage: `{json.dumps(summary['usage'], sort_keys=True)}`", f"- Cost: {summary['cost']['reason']}",
        f"- Gate reasons: `{json.dumps(summary['gate_reasons'])}`", "",
        "## Simulation-only execution", "",
        f"- Physical calls: {summary.get('simulation_only', {}).get('physical_call_count')}",
        f"- Completed: {summary.get('simulation_only', {}).get('completed_count')}",
        f"- Attempts/retries: {summary.get('simulation_only', {}).get('attempt_count')} / {summary.get('simulation_only', {}).get('retry_count')}",
        f"- Token usage: `{json.dumps(summary.get('simulation_only', {}).get('usage', {}), sort_keys=True)}`", "",
        "## Per-model reliability", "",
        "| Model | Calls | Completed | Failure rate | Retries | Mean latency ms | Max latency ms |", "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {model} | {metrics['call_count']} | {metrics['completed_count']} | {metrics['failure_rate']:.4f} | {metrics['retry_count']} | {metrics['latency_ms_mean']} | {metrics['latency_ms_max']} |"
            for model, metrics in summary.get("model_metrics", {}).items()
        ], "",
        "## Baseline integrity", "",
        f"- Inconsistent shared baselines: {len(summary.get('baseline_inconsistencies', []))}",
        "## Isolation", "", "- Paths Not Taken remained disabled.", "- No SenseNova model was called.",
        "- Credentials and endpoint URLs are absent from artifacts and event logs.",
    ]
    if three_arm:
        lines.extend([
            "", "## Three-arm analysis", "",
            f"- Complete candidates: {three_arm['complete_candidate_count']}/{three_arm['candidate_count']}",
            f"- Paths Not Taken candidates: {three_arm['paths_not_taken_candidate_count']}",
            f"- Strict signal count by model: `{json.dumps(three_arm.get('strict_signal_count_by_model', {}), sort_keys=True)}`",
            f"- Statistical unit: {three_arm.get('statistical_unit', 'base_fact_id')}",
            f"- Analyzed base facts / target variants: {three_arm.get('analyzed_base_fact_count')} / {three_arm.get('target_variant_count')}",
            f"- Proxy directed base-fact count by model: `{json.dumps(three_arm.get('proxy_directed_candidate_base_fact_count_by_model', {}), sort_keys=True)}`",
            f"- Proxy ZH-specific strict base-fact count by model: `{json.dumps(three_arm.get('proxy_zh_specific_strict_base_fact_count_by_model', {}), sort_keys=True)}`",
            f"- Shared baseline calls saved: {three_arm['shared_baseline_calls_saved']}",
            f"- Neutral-control instabilities: {len(three_arm['neutral_control_instabilities'])}",
            f"- Conclusion: {three_arm['conclusion']}",
        ])
        model_arm_metrics = three_arm.get("model_arm_metrics") or {}
        if model_arm_metrics:
            lines.extend([
                "", "### Model-level arm scores", "",
                "| Model | Arm | Completed | Accuracy | Distractor-hit rate |",
                "|---|---|---:|---:|---:|",
                *[
                    f"| {model} | {arm} | {metrics['completed_count']}/{metrics['call_count']} | "
                    f"{metrics['accuracy_completed']} | {metrics['distractor_hit_rate_completed']} |"
                    for model, arms in model_arm_metrics.items()
                    for arm, metrics in arms.items()
                ],
            ])
        proxy_funnel = three_arm.get("proxy_funnel_by_model") or {}
        if proxy_funnel:
            lines.extend([
                "", "### Per-model proxy funnel", "",
                (
                    "| Model | Assigned facts | Assigned variants | 10-input terminal facts | "
                    "Evaluable variants | Directed | ZH-specific strict | Neutral unstable | "
                    "Off-target | Shared-language | Incomplete |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {model} | {funnel['N_assigned_facts']} | "
                    f"{funnel['N_assigned_variants']} | "
                    f"{funnel['N_10_input_terminal_facts']} | "
                    f"{funnel['N_evaluable_variants']} | "
                    f"{funnel['N_proxy_directed_candidate_variants']} | "
                    f"{funnel['N_proxy_zh_specific_strict_variants']} | "
                    f"{funnel['N_neutral_unstable']} | "
                    f"{funnel['N_off_target_failure']} | "
                    f"{funnel['N_shared_language_susceptibility']} | "
                    f"{funnel['N_incomplete']} |"
                    for model, funnel in proxy_funnel.items()
                ],
                "", "### Per-model base-fact aggregation", "",
                (
                    "| Model | Directed any/both | ZH-specific any/both | "
                    "Neutral unstable any | Off-target any | Shared-language any | "
                    "Incomplete facts |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {model} | "
                    f"{funnel['base_fact_level']['N_proxy_directed_candidate_any']}/"
                    f"{funnel['base_fact_level']['N_proxy_directed_candidate_both']} | "
                    f"{funnel['base_fact_level']['N_proxy_zh_specific_strict_any']}/"
                    f"{funnel['base_fact_level']['N_proxy_zh_specific_strict_both']} | "
                    f"{funnel['base_fact_level']['N_neutral_unstable_any']} | "
                    f"{funnel['base_fact_level']['N_off_target_failure_any']} | "
                    f"{funnel['base_fact_level']['N_shared_language_susceptibility_any']} | "
                    f"{funnel['base_fact_level']['N_incomplete_facts']} |"
                    for model, funnel in proxy_funnel.items()
                ],
            ])
    if freeze:
        lines.extend(["", "## Candidate freeze", "", f"- Frozen candidates: {freeze['candidate_count']}", f"- Candidate ID digest: `{freeze['candidate_ids_sha256']}`"])
    if holdout:
        lines.extend(["", "## Claude Opus 4.8 holdout", "", f"- Completed calls: {holdout['completed_count']}/{holdout['call_count']}", f"- English accuracy: {holdout['english_accuracy']}", f"- Chinese accuracy: {holdout['chinese_accuracy']}"])
    report = "\n".join(lines) + "\n"
    (run_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    return report
