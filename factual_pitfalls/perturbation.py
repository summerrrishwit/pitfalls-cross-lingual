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


SCHEMA_VERSION = "factual-perturbation-runtime-v1"
TRANSLATION_PROMPT_VERSION = "factual-translation-zh-v1"
TRANSLATION_REVIEW_PROMPT_VERSION = "factual-translation-review-zh-v1"
TRANSLATION_REPAIR_PROMPT_VERSION = "factual-translation-repair-zh-v1"
TRANSLATION_REPAIR_REVIEW_PROMPT_VERSION = "factual-translation-repair-review-zh-v1"
DISTRACTOR_REVIEW_PROMPT_VERSION = "factual-distractor-review-zh-v1"
DISTRACTOR_GENERATION_PROMPT_VERSION = "factual-distractor-generation-bilingual-v1"
PERTURBATION_PROMPT_VERSION = "factual-perturbation-bilingual-v1"
PERTURBATION_REVIEW_PROMPT_VERSION = "factual-perturbation-review-bilingual-v3"
PERTURBATION_SENSITIVITY_PROMPT_VERSION = "factual-perturbation-bilingual-sensitivity-v1"
PERTURBATION_SENSITIVITY_REVIEW_PROMPT_VERSION = "factual-perturbation-review-bilingual-sensitivity-v1"
SIMULATION_PROMPT_VERSION = "factual-simulation-three-arm-v2"
SIMULATION_MULTI_PROMPT_VERSION = "factual-simulation-multi-option-three-arm-v1"
HOLDOUT_PROMPT_VERSION = "factual-holdout-two-choice-v1"

SIMULATION_LANGUAGES = ("en", "zh")
SIMULATION_VARIANTS = ("original", "neutral", "targeted")
SHARED_SIMULATION_VARIANTS = ("original", "neutral")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
            required.extend(("neutral_english_context", "neutral_chinese_context", "strength"))
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
        self.values = dotenv_values(env_path)
        self.event_log = event_log
        self.timeout = float(config["execution"].get("timeout_seconds", 180))
        self.max_retries = int(config["execution"].get("max_retries", 3))
        limits = config["execution"].get("provider_max_concurrency", {})
        self.profile_semaphores = {
            name: threading.BoundedSemaphore(
                max(1, int(limits.get(name, config["execution"].get("max_workers", 4))))
            )
            for name in config["provider_profiles"]
        }

    def _profile(self, model_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
        profile_name = model_spec["provider_profile"]
        profile = self.config["provider_profiles"][profile_name]
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
        if model.startswith("qwen") or "deepseek" in model:
            request["extra_body"] = {"enable_thinking": False}
        if model_spec.get("json_mode") or profile.get("json_mode"):
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
                text, response_model, usage = self._one_call(model_spec, prompt)
                last_text, last_response_model, last_usage = text, response_model, usage
                parsed = parse_json_object(text)
                validator(parsed)
                attempts.append({"attempt": attempt, "status": "ok", "latency_ms": round((time.monotonic() - started) * 1000)})
                result = ModelResult("completed", parsed, text, response_model, usage, round((time.monotonic() - started_all) * 1000), attempt, attempts)
                self._event(stage, item_id, model_spec, result)
                return result
            except Exception as error:
                attempts.append({"attempt": attempt, "status": "failed", "latency_ms": round((time.monotonic() - started) * 1000), **safe_error(error)})
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
) -> Dict[str, Any]:
    prompt_path = project_root / "data_processed/factual_triples/triple-full-v1/canonical-v2/factual_triples_with_distractors.jsonl"
    excluded = set(exclude_source_ids)
    rows = [
        row for row in read_jsonl(prompt_path)
        if row.get("factual_prompt_status") == "generated" and row.get("source_id") not in excluded
    ]
    selected = stratified_sample(rows, limit, int(config["pilot"]["seed"]))
    prepared = []
    max_distractors = int(config["pilot"]["max_distractors_per_triple"])
    for row in selected:
        distractors = []
        for candidate in row.get("distractor_candidates", [])[:max_distractors]:
            distractors.append({
                "distractor_id": f"{row['source_id']}_source_wrong_{candidate['source_choice_index']}",
                "text_en": candidate["distractor_answer"],
                "source_choice_index": candidate["source_choice_index"],
                "source": "original_wrong_option",
            })
        prepared.append({
            "source_id": row["source_id"],
            "source_dataset": row["source_dataset"],
            "answer_type": row.get("answer_type"),
            "normalization_status": row.get("normalization_status"),
            "relation_raw": row.get("relation_raw"),
            "relation_normalized": row.get("relation_normalized"),
            "prompt_quality_tier": row["prompt_quality_tier"],
            "prompt_en": row["prompt_en"],
            "answer_en": row["prompt_expected_answer"],
            "answer_aliases_en": sorted({row["prompt_expected_answer"], row.get("source_answer") or row["prompt_expected_answer"]}),
            "canonical_fact": row["canonical_fact"],
            "distractors": distractors,
        })
    previous = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else None
    config_sha = sha256_value(config)
    runtime_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    amendments = list(previous.get("config_amendments", [])) if previous else []
    if previous and previous.get("config_sha256") != config_sha:
        amendments.append({
            "changed_at": utc_now(),
            "from_config_sha256": previous.get("config_sha256"),
            "to_config_sha256": config_sha,
            "reason": "preflight remediation after a recorded failed gate; completed checkpoints retained",
        })
    simulation_role = config["model_roles"]["simulation"]
    neutral_mode = simulation_role.get("neutral_context_mode", "generic_shared")
    shared_variants = ["original"] if neutral_mode == "matched_per_candidate" else list(SHARED_SIMULATION_VARIANTS)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": config_sha,
        "runtime_sha256": runtime_sha,
        "config_amendments": amendments,
        "prompt_input": str(prompt_path.relative_to(project_root)),
        "prompt_input_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
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
    accepted_ids = sorted(
        row["item_id"] for row in review.get("perturbation_reviews", [])
        if row.get("decision") == "accept"
    )
    source_perturbations = read_jsonl(source_run_dir / "perturbations.jsonl")
    available_ids = {row.get("candidate_id") for row in source_perturbations}
    if not accepted_ids or not set(accepted_ids).issubset(available_ids):
        raise ValueError("invalid_codex_proxy_perturbation_allowlist")
    accepted_before_sampling = len(accepted_ids)
    if strength_filter is not None:
        row_by_id = {row["candidate_id"]: row for row in source_perturbations}
        accepted_ids = [
            candidate_id for candidate_id in accepted_ids
            if row_by_id[candidate_id].get("candidate", {}).get("strength") == strength_filter
        ]
        if not accepted_ids:
            raise ValueError("strength_filter_selected_no_candidates")

    simulation_role = config["model_roles"]["simulation"]
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
        row_by_id = {row["candidate_id"]: row for row in source_perturbations}
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
    if max_candidates_per_source is not None:
        if max_candidates_per_source < 1:
            raise ValueError("max_candidates_per_source_must_be_positive")
        row_by_id = {row["candidate_id"]: row for row in source_perturbations}
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
        rows = read_jsonl(source_path)
        write_jsonl(run_dir / name, rows)
        reused_artifacts[name] = {
            "row_count": len(rows),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }

    neutral_mode = simulation_role.get("neutral_context_mode", "generic_shared")
    shared_variants = ["original"] if neutral_mode == "matched_per_candidate" else list(SHARED_SIMULATION_VARIANTS)
    manifest = {
        **source_manifest,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": sha256_value(config),
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        "shared_simulation_variants": shared_variants,
        "neutral_context_mode": neutral_mode,
        "simulation_endpoint": simulation_role.get("endpoint", "binary"),
        "simulation_max_options": int(simulation_role.get("max_options", 3)),
        "simulation_requires_codex_proxy_review": True,
        "endpoint_excluded_candidate_ids": [
            row["candidate_id"] for row in endpoint_exclusions
        ],
        "endpoint_exclusions": endpoint_exclusions,
        "resumed_from_run_id": source_manifest["run_id"],
        "reused_upstream_artifacts": reused_artifacts,
        "codex_proxy_review": {
            "review_version": review["review_version"],
            "reviewer_type": "codex_proxy",
            "not_human_gold": True,
            "source_path": str(review_path.relative_to(project_root)),
            "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "accepted_perturbation_ids": accepted_ids,
            "accepted_before_sampling": accepted_before_sampling,
            "accepted_after_endpoint_filter": accepted_after_endpoint_filter,
            "selection_policy": {
                "independent_unit": "source_id",
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
        manifest["runtime_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        "not_human_gold": True,
        "review_blinded_to_simulation_results": True,
        "automatic_judge_is_advisory": not manifest.get("perturbation_auto_judge_hard_gate", True),
        "selection_policy": "Select at most one valid candidate per source before any Simulation call.",
        "required_checks": {
            "source": [
                "canonical_fact_usable", "answer_unique", "three_semantically_distinct_options",
                "same_answer_type_or_plausible_foil",
            ],
            "candidate": [
                "truth_preserved", "bilingual_equivalent", "target_distractor_salience",
                "matched_neutral_has_no_answer_or_distractor_cue", "l2_strength",
                "no_prompt_duplication", "natural",
            ],
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
        manifest["runtime_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
        "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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


def _translation_prompt(item: Dict[str, Any]) -> str:
    payload = {
        "prompt_en": item["prompt_en"],
        "answer_en": item["answer_en"],
        "answer_aliases_en": item["answer_aliases_en"],
        "distractors": [{"distractor_id": d["distractor_id"], "text_en": d["text_en"]} for d in item["distractors"]],
    }
    return (
        "Translate this factual-recall item into natural Simplified Chinese. Preserve the incomplete/open-answer "
        "prompt form, factual meaning, answer identity, and distractor identity. Do not answer inside prompt_zh. "
        "Return JSON only with prompt_zh, answer_zh, answer_aliases_zh (array), and distractors "
        "(array of objects with distractor_id and text_zh). Input: " + canonical_json(payload)
    )


def _translation_review_prompt(item: Dict[str, Any], translation: Dict[str, Any]) -> str:
    payload = {"source": item, "translation": translation}
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
    payload = {"source": item, "translation": translation, "review": review}
    return (
        "Repair this Simplified-Chinese factual-recall translation using the independent review. Correct only the "
        "reported fidelity, leakage, or naturalness problems while preserving the English prompt form, answer "
        "identity, distractor identities, and every distractor_id. Return JSON only with prompt_zh, answer_zh, "
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
) -> str:
    payload = {
        "canonical_fact": item["canonical_fact"], "prompt_en": item["prompt_en"],
        "prompt_zh": translation["prompt_zh"], "answer_en": item["answer_en"],
        "answer_zh": translation["answer_zh"], "distractor_en": distractor["distractor_en"],
        "distractor_zh": distractor["distractor_zh"],
    }
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
) -> str:
    payload = {
        "canonical_fact": item["canonical_fact"], "prompt_en": item["prompt_en"],
        "prompt_zh": translation["prompt_zh"], "answer_en": item["answer_en"],
        "answer_zh": translation["answer_zh"], "distractor_en": distractor["distractor_en"],
        "distractor_zh": distractor["distractor_zh"], "candidate": candidate,
    }
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
    entries = [("__answer__", answer), *((distractor_id, by_id[distractor_id]) for distractor_id in selected_ids)]
    entries.sort(key=lambda value: sha256_value([source_id, target_distractor_id, "option-order", value[0]]))
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
) -> str:
    if variant in shared_variants:
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
    """Build three-arm inputs while sharing original/neutral baselines per distractor."""
    inputs: Dict[str, Dict[str, Any]] = {}
    candidate_ids_by_distractor: Dict[str, List[str]] = defaultdict(list)
    for perturbation in accepted_perturbations:
        candidate_ids_by_distractor[perturbation["distractor_id"]].append(perturbation["candidate_id"])

    for perturbation in accepted_perturbations:
        source_id = perturbation["source_id"]
        distractor_id = perturbation["distractor_id"]
        item = item_by_id[source_id]
        translation = translation_by_id[source_id]["parsed_response"]
        distractor = distractor_by_id[distractor_id]
        source_distractors = [
            row for row in distractor_by_id.values()
            if row.get("source_id") == source_id and row.get("item_id") != distractor_id
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
                    simulation_id = _simulation_id(
                        source_id,
                        distractor_id,
                        model_spec["model"],
                        language,
                        variant,
                        candidate_id,
                        shared_variants,
                    )
                    if simulation_id in inputs:
                        continue
                    inputs[simulation_id] = {
                        "simulation_id": simulation_id,
                        "candidate_id": candidate_id,
                        "candidate_ids": (
                            sorted(candidate_ids_by_distractor[distractor_id])
                            if variant in shared_variants
                            else [perturbation["candidate_id"]]
                        ),
                        "source_id": source_id,
                        "distractor_id": distractor_id,
                        "model_spec": model_spec,
                        "language": language,
                        "variant": variant,
                        "simulation_scope": (
                            "shared_distractor_baseline"
                            if variant in shared_variants
                            else "candidate"
                        ),
                        "options": options,
                        "option_ids": option_ids,
                        "option_count": len(options),
                        "endpoint": endpoint,
                        "correct_choice": correct_choice,
                        "target_choice": target_choice,
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


def run_preholdout(config: Dict[str, Any], project_root: Path, run_dir: Path, env_path: Path) -> Dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("config_sha256") != sha256_value(config):
        raise ValueError("run manifest config fingerprint does not match current config")
    if manifest.get("runtime_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("run manifest runtime fingerprint does not match current code")
    if manifest.get("paths_not_taken_enabled") is not False or manifest.get("holdout_enabled") is not False:
        raise ValueError("pre-holdout run must keep Paths Not Taken and holdout disabled")
    items = manifest["records"]
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
    accepted_ids = {r["source_id"] for r in translation_reviews if r["terminal_status"] == "completed" and r["parsed_response"]["decision"] == "accept" and all(r["parsed_response"]["checks"].values())}

    distractor_inputs = []
    item_by_id = {item["source_id"]: item for item in items}
    for source_id in accepted_ids:
        for distractor in item_by_id[source_id]["distractors"]:
            distractor_inputs.append({"item_id": distractor["distractor_id"], "source_id": source_id, "distractor": distractor})

    def review_distractor(entry: Dict[str, Any]) -> Dict[str, Any]:
        item = item_by_id[entry["source_id"]]
        translation = translation_by_id[entry["source_id"]]["parsed_response"]
        translated = next(d for d in translation["distractors"] if d["distractor_id"] == entry["item_id"])
        enriched = {**entry["distractor"], "text_zh": translated["text_zh"]}
        spec = roles["distractor_validation"]["primary_judge"]
        result = router.request_json("distractor_validation", entry["item_id"], spec, _distractor_review_prompt(item, translation, enriched), lambda value: validate_decision(value, DISTRACTOR_CHECKS))
        return {"item_id": entry["item_id"], "source_id": entry["source_id"], "distractor_en": enriched["text_en"], "distractor_zh": enriched["text_zh"], **model_record(result, spec, DISTRACTOR_REVIEW_PROMPT_VERSION)}

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
    accepted_original_distractors = [
        row for row in original_distractor_reviews
        if row["terminal_status"] == "completed"
        and row["parsed_response"]["decision"] == "accept"
        and all(row["parsed_response"]["checks"].values())
    ]

    if replenish_distractors:
        generation_role = roles["distractor_generation"]
        target_count = int(generation_role.get("target_accepted_per_source", 2))
        backup_count = int(generation_role.get("backup_candidates", 1))
        accepted_original_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in accepted_original_distractors:
            accepted_original_by_source[row["source_id"]].append(row)
        distractor_generation_inputs = []
        for source_id in sorted(accepted_ids):
            current = accepted_original_by_source[source_id]
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

    accepted_distractors = [
        row for row in distractor_reviews
        if row["terminal_status"] == "completed"
        and row["parsed_response"]["decision"] == "accept"
        and all(row["parsed_response"]["checks"].values())
    ]

    target_distractors_per_source = int(
        roles["perturbation_generation"].get("target_distractors_per_source", 0)
    )
    targeted_distractors = _select_target_distractors(
        accepted_distractors,
        target_distractors_per_source,
        int(manifest["seed"]),
    )

    perturbation_inputs = [
        {"item_id": d["item_id"], "source_id": d["source_id"], "distractor": d}
        for d in targeted_distractors
    ]
    count = int(roles["perturbation_generation"]["candidates_per_distractor"])
    strength_levels = tuple(roles["perturbation_generation"].get("strength_levels") or ())
    matched_neutral = bool(roles["perturbation_generation"].get("matched_neutral", False))
    perturbation_prompt_version = (
        PERTURBATION_SENSITIVITY_PROMPT_VERSION if matched_neutral else PERTURBATION_PROMPT_VERSION
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
                item, translation, entry["distractor"], count, strength_levels, matched_neutral
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
                    item, translation, entry["distractor"], count, strength_levels, matched_neutral
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
    distractor_by_id = {d["item_id"]: d for d in accepted_distractors}
    for generation in generations:
        if generation["terminal_status"] != "completed":
            continue
        for index, candidate in enumerate(generation["parsed_response"]["candidates"]):
            validation_inputs.append({"candidate_id": f"{generation['item_id']}_p{index + 1}", "source_id": generation["source_id"], "distractor_id": generation["item_id"], "candidate": candidate})

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
            entry["distractor_id"],
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
                "simulation_id", "candidate_id", "candidate_ids", "source_id", "distractor_id",
                "language", "variant", "simulation_scope", "correct_choice",
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
    return {
        "schema_version": "factual-perturbation-run-summary-v1",
        "generated_at": utc_now(), "run_id": manifest["run_id"],
        "base_triple_count": manifest["selected_count"],
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
    variants = tuple(manifest.get("simulation_variants") or ())
    if variants != SIMULATION_VARIANTS:
        raise ValueError("three_arm_analysis_requires_original_neutral_targeted")
    perturbations = _accepted_perturbation_rows(read_jsonl(run_dir / "perturbations.jsonl"), manifest)
    simulations = read_jsonl(run_dir / "simulation_results.jsonl")
    models = tuple(manifest["simulation_models"])
    shared_variants = tuple(manifest.get("shared_simulation_variants") or SHARED_SIMULATION_VARIANTS)
    model_count = len(models)
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
        for model in models:
            original = by_key.get((model, "zh", "original"), {})
            neutral = by_key.get((model, "zh", "neutral"), {})
            targeted = by_key.get((model, "zh", "targeted"), {})
            if original.get("correct") is True and neutral.get("correct") is True and targeted.get("correct") is False:
                targeted_zh_wrong_models.append(model)
                if targeted.get("distractor_hit") is True:
                    targeted_zh_flips.append(model)
                    english_controls_pass = all(
                        by_key.get((model, "en", variant), {}).get("correct") is True
                        for variant in SIMULATION_VARIANTS
                    )
                    if english_controls_pass:
                        strict_targeted_zh_flip_models.append(model)
                else:
                    targeted_zh_nontarget_wrong_models.append(model)
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
        candidate_results.append({
            "candidate_id": perturbation["candidate_id"],
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
            "paths_not_taken_candidate": bool(complete and strict_targeted_zh_flip_models),
        })
    naive_calls = len(perturbations) * len(models) * len(SIMULATION_LANGUAGES) * len(variants)
    strict_signal_count_by_model = Counter(
        model
        for row in candidate_results
        for model in row["strict_targeted_zh_flip_models"]
    )
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
        "analysis_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": manifest["run_id"],
        "reviewer_type": manifest.get("codex_proxy_review", {}).get("reviewer_type"),
        "not_human_gold": manifest.get("codex_proxy_review", {}).get("not_human_gold"),
        "candidate_count": len(perturbations),
        "complete_candidate_count": sum(row["complete"] for row in candidate_results),
        "paths_not_taken_candidate_count": sum(row["paths_not_taken_candidate"] for row in candidate_results),
        "strict_signal_count_by_model": {
            model: strict_signal_count_by_model.get(model, 0) for model in models
        },
        "model_arm_metrics": model_arm_metrics,
        "physical_simulation_calls": len(simulations),
        "naive_unshared_call_count": naive_calls,
        "shared_baseline_calls_saved": naive_calls - len(simulations),
        "baseline_inconsistencies": detect_baseline_inconsistencies(simulations, shared_variants),
        "neutral_control_instabilities": list(control_instabilities.values()),
        "candidate_results": candidate_results,
        "conclusion": (
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
                "candidate_id": candidate_id, "source_id": perturbation["source_id"],
                "distractor_id": perturbation["distractor_id"], "candidate": perturbation["candidate"],
                "badcase_type": "induced" if induced else "amplified",
                "accuracy": {f"{k[0]}_{k[1]}": v for k, v in acc.items()},
                "freeze_version": "factual-perturbation-candidate-freeze-v1",
            })
    write_jsonl(run_dir / "frozen_candidates.jsonl", frozen)
    write_json(run_dir / "candidate_freeze_manifest.json", {
        "schema_version": "factual-perturbation-candidate-freeze-v1", "created_at": utc_now(),
        "candidate_count": len(frozen), "candidate_ids_sha256": sha256_value([r["candidate_id"] for r in frozen]),
        "holdout_eligible": True, "paths_not_taken_enabled": False,
    })
    return frozen


def run_holdout(config: Dict[str, Any], run_dir: Path, env_path: Path) -> Dict[str, Any]:
    freeze_manifest = read_json(run_dir / "candidate_freeze_manifest.json")
    frozen = read_jsonl(run_dir / "frozen_candidates.jsonl")
    if not freeze_manifest.get("holdout_eligible"):
        raise ValueError("candidate freeze is not holdout eligible")
    manifest = read_json(run_dir / "run_manifest.json")
    if manifest.get("config_sha256") != sha256_value(config):
        raise ValueError("run manifest config fingerprint does not match current config")
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
    if freeze:
        lines.extend(["", "## Candidate freeze", "", f"- Frozen candidates: {freeze['candidate_count']}", f"- Candidate ID digest: `{freeze['candidate_ids_sha256']}`"])
    if holdout:
        lines.extend(["", "## Claude Opus 4.8 holdout", "", f"- Completed calls: {holdout['completed_count']}/{holdout['call_count']}", f"- English accuracy: {holdout['english_accuracy']}", f"- Chinese accuracy: {holdout['chinese_accuracy']}"])
    report = "\n".join(lines) + "\n"
    (run_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    return report
