"""Resumable raw-source to cross-lingual-factual-pitfall pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


AUDIT_MODEL = "qwen3.7-plus"
PROMPT_VERSION = "raw-english-factual-audit-v1"
ROLE_NAMES = {
    "factual_audit_model",
    "perturbation_generator_model",
    "translation_model",
    "screen_models",
    "answer_extract_model",
}


class RawSourceSnapshot(TypedDict):
    dataset: str
    source_path: str
    original_index: int
    record: Dict[str, Any]


class RawManifest(TypedDict):
    manifest_version: str
    run_id: str
    created_at: str
    config: Dict[str, Any]
    source_datasets: Dict[str, Any]
    candidates: List[Dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL in {path} line {line_number}: {error}") from error
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    temporary.replace(path)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    sources = config.get("source_datasets")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source_datasets must be a non-empty object")
    if not isinstance(config.get("raw_sample_size"), int) or config["raw_sample_size"] <= 0:
        raise ValueError("raw_sample_size must be a positive integer")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if not isinstance(config.get("target_languages"), list) or not config["target_languages"]:
        raise ValueError("target_languages must be a non-empty list")
    roles = config.get("roles")
    if not isinstance(roles, dict) or set(roles) != ROLE_NAMES:
        raise ValueError(f"roles must contain exactly {sorted(ROLE_NAMES)}")
    if roles["factual_audit_model"] != AUDIT_MODEL:
        raise ValueError(f"roles.factual_audit_model must be {AUDIT_MODEL!r}")
    for role in ("perturbation_generator_model", "translation_model", "answer_extract_model"):
        if not isinstance(roles[role], str) or not roles[role].strip():
            raise ValueError(f"roles.{role} must be a non-empty model ID")
    if not isinstance(roles["screen_models"], list) or not roles["screen_models"] or not all(
        isinstance(model, str) and model.strip() for model in roles["screen_models"]
    ):
        raise ValueError("roles.screen_models must be a non-empty list of model IDs")
    supported_models = config.get("supported_models")
    if not isinstance(supported_models, list) or not supported_models or not all(isinstance(model, str) and model.strip() for model in supported_models):
        raise ValueError("supported_models must be a non-empty list of provider-validated model IDs")
    configured_models = [roles["factual_audit_model"], roles["perturbation_generator_model"], roles["translation_model"], roles["answer_extract_model"], *roles["screen_models"]]
    unsupported = sorted(set(configured_models) - set(supported_models))
    if unsupported:
        raise ValueError(f"roles contain unsupported model IDs: {unsupported}")
    thresholds = config.get("weakness_thresholds", {})
    for key in ("min_rate_ori", "max_rate_trans", "min_pitfall_score"):
        if not isinstance(thresholds.get(key), (int, float)):
            raise ValueError(f"weakness_thresholds.{key} must be numeric")
    runtime = config.get("runtime", {})
    for key in ("temperature", "timeout_seconds"):
        if not isinstance(runtime.get(key), (int, float)):
            raise ValueError(f"runtime.{key} must be numeric")
    if not isinstance(runtime.get("max_retries"), int) or runtime["max_retries"] < 0:
        raise ValueError("runtime.max_retries must be a non-negative integer")


def raw_schema_error(record: Any) -> Optional[str]:
    if not isinstance(record, dict):
        return "record_must_be_object"
    for field in ("question", "answer", "source"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            return f"missing_or_empty_{field}"
    if not isinstance(record.get("choices"), list) or not record["choices"]:
        return "missing_or_empty_choices"
    if not all(isinstance(choice, str) and choice.strip() for choice in record["choices"]):
        return "invalid_choice"
    if record["answer"] not in record["choices"]:
        return "answer_not_in_choices"
    return None


def raw_stratum(dataset: str, record: Dict[str, Any]) -> str:
    subject = str(record.get("subject") or "__no_subject__")
    return f"{dataset}::{subject}"


def proportional_allocation(groups: Dict[str, List[Dict[str, Any]]], desired: int) -> Dict[str, int]:
    total = sum(len(values) for values in groups.values())
    desired = min(desired, total)
    if not total:
        return {key: 0 for key in groups}
    allocation = {key: int(desired * len(values) / total) for key, values in groups.items()}
    remaining = desired - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (-(desired * len(groups[key]) / total - allocation[key]), key),
    )
    for key in order:
        if remaining <= 0:
            break
        allocation[key] += 1
        remaining -= 1
    return allocation


def build_raw_manifest(config: Dict[str, Any], project_root: Path, run_id: str) -> RawManifest:
    all_candidates: List[Dict[str, Any]] = []
    source_details: Dict[str, Any] = {}
    seen_questions: Dict[str, Tuple[str, int]] = {}
    for dataset, relative_path in config["source_datasets"].items():
        source_path = (project_root / relative_path).resolve()
        with source_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{source_path} must contain a JSON array")
        exclusions = []
        eligible_count = 0
        for original_index, record in enumerate(records):
            error = raw_schema_error(record)
            if error:
                exclusions.append({"original_index": original_index, "reason": error})
                continue
            eligible_count += 1
            normalized = normalize_text(record["question"])
            if normalized in seen_questions:
                duplicate_dataset, duplicate_index = seen_questions[normalized]
                exclusions.append(
                    {
                        "original_index": original_index,
                        "reason": "duplicate_english_question",
                        "duplicate_of": {"dataset": duplicate_dataset, "original_index": duplicate_index},
                    }
                )
                continue
            seen_questions[normalized] = (dataset, original_index)
            all_candidates.append(
                {
                    "candidate_id": f"raw_{dataset}_{original_index:06d}",
                    "stratum": raw_stratum(dataset, record),
                    "source_snapshot": {
                        "dataset": dataset,
                        "source_path": str(source_path),
                        "original_index": original_index,
                        "record": copy.deepcopy(record),
                    },
                }
            )
        source_details[dataset] = {
            "source_path": str(source_path),
            "source_fingerprint_sha256": file_fingerprint(source_path),
            "source_record_count": len(records),
            "schema_eligible_count": eligible_count,
            "unique_eligible_count": 0,
            "exclusions": exclusions,
        }

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in all_candidates:
        groups[candidate["stratum"]].append(candidate)
    allocation = proportional_allocation(groups, config["raw_sample_size"])
    selected = []
    stratum_details = {}
    for stratum in sorted(groups):
        values = sorted(groups[stratum], key=lambda item: item["source_snapshot"]["original_index"])
        seed = int(hashlib.sha256(f"{config['seed']}:{stratum}".encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(values)
        chosen = values[: allocation[stratum]]
        selected.extend(chosen)
        stratum_details[stratum] = {
            "eligible_count": len(values),
            "allocated_count": allocation[stratum],
            "selected_count": len(chosen),
            "selected_original_indices": [item["source_snapshot"]["original_index"] for item in chosen],
        }
    selected.sort(key=lambda item: (item["stratum"], item["source_snapshot"]["original_index"]))
    for candidate in all_candidates:
        source_details[candidate["source_snapshot"]["dataset"]]["unique_eligible_count"] += 1
    return {
        "manifest_version": "raw-source-v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "sampling_config": {
            "source_datasets": copy.deepcopy(config["source_datasets"]),
            "raw_sample_size": config["raw_sample_size"],
            "seed": config["seed"],
            "target_languages": copy.deepcopy(config["target_languages"]),
        },
        "source_datasets": {
            **source_details,
            "allocation": {
                "requested_count": config["raw_sample_size"],
                "available_count": len(all_candidates),
                "actual_count": len(selected),
                "shortfall": max(0, config["raw_sample_size"] - len(selected)),
                "strata": stratum_details,
            },
        },
        "candidates": selected,
    }


def raw_audit_prompt(candidate: Dict[str, Any]) -> str:
    record = candidate["source_snapshot"]["record"]
    return f'''You are curating an English factual-recall dataset.
Review ONLY the raw English QA below. Do not evaluate translations, generated perturbations, model performance, or any cross-lingual metric.

Accept only a question with a single stable, objective, directly retrievable factual answer that can be expressed as a subject-relation-answer triple and a natural English completion prompt. Reject or flag questions requiring the options, subjective judgment, material ambiguity, or multi-step reasoning.

Return exactly one JSON object and no Markdown:
{{
  "decision": "accept" | "reject" | "needs_review",
  "is_factual_recall": true | false,
  "factual_type": "geography" | "history" | "science" | "culture" | "entity_attribute" | "other" | "unknown",
  "answer_unique": true | false,
  "has_material_ambiguity": true | false,
  "option_dependent": true | false,
  "requires_multistep_reasoning": true | false,
  "reason": "short explanation",
  "subject_en": "required for accept",
  "relation_en": "required for accept",
  "answer_en": "required for accept",
  "prompt_en": "required for accept"
}}

Source: {record.get("source", "")}
Subject: {record.get("subject", "")}
English question: {record["question"]}
English choices: {json.dumps(record["choices"], ensure_ascii=False)}
Canonical English answer: {record["answer"]}
'''


def generation_prompt(review: Dict[str, Any], wrong_option: str) -> str:
    record = review["source_snapshot"]["record"]
    return f'''Create one subtle English distraction for a factual multiple-choice question.
It must be contextually related to the specified incorrect option, but it MUST NOT change, contradict, or reveal the canonical answer. It must not alter choices or require a new answer.

Return exactly one JSON object and no Markdown:
{{"distraction": "...", "insertion": "prefix" | "suffix", "answer_preserved": true}}

Question: {record["question"]}
Choices: {json.dumps(record["choices"], ensure_ascii=False)}
Canonical answer: {record["answer"]}
Incorrect-option target: {wrong_option}
'''


def translation_prompt(generation: Dict[str, Any], target_language: str) -> str:
    return f'''Translate this enhanced English multiple-choice item into {target_language}.
Preserve the question meaning, all choices, and the canonical answer. The translated canonical answer must exactly match one translated choice.

Return exactly one JSON object and no Markdown:
{{"question": "...", "choices": ["..."], "answer": "..."}}

English question: {generation["enhanced_question"]}
English choices: {json.dumps(generation["choices"], ensure_ascii=False)}
English answer: {generation["answer" ]}
'''


def answer_prompt(question: str, choices: List[str]) -> str:
    return f"""Answer the following multiple-choice question. Return the best answer; explanations are allowed.\n\nQuestion: {question}\nChoices: {json.dumps(choices, ensure_ascii=False)}"""


def answer_extract_prompt(raw_answer: str, choices: List[str]) -> str:
    return f'''Extract exactly one answer from the allowed choices. Return exactly one JSON object and no Markdown: {{"final_answer": "one allowed choice"}}.

Allowed choices: {json.dumps(choices, ensure_ascii=False)}
Model answer: {raw_answer}
'''


def extract_json_object(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    return parsed


def validate_audit_response(response: Dict[str, Any], canonical_answer: str) -> Tuple[str, List[str]]:
    errors = []
    decision = response.get("decision")
    if decision not in {"accept", "reject", "needs_review"}:
        return "needs_review", ["invalid_decision"]
    for field in ("is_factual_recall", "answer_unique", "has_material_ambiguity", "option_dependent", "requires_multistep_reasoning"):
        if not isinstance(response.get(field), bool):
            errors.append(f"invalid_{field}")
    for field in ("factual_type", "reason"):
        if not isinstance(response.get(field), str) or not response[field].strip():
            errors.append(f"invalid_{field}")
    if decision == "accept":
        expected = {"is_factual_recall": True, "answer_unique": True, "has_material_ambiguity": False, "option_dependent": False, "requires_multistep_reasoning": False}
        for field, value in expected.items():
            if response.get(field) is not value:
                errors.append(f"contradictory_accept_{field}")
        for field in ("subject_en", "relation_en", "answer_en", "prompt_en"):
            if not isinstance(response.get(field), str) or not response[field].strip():
                errors.append(f"invalid_{field}")
        if response.get("answer_en", "").strip() != canonical_answer.strip():
            errors.append("answer_en_does_not_match_canonical")
    return ("needs_review" if errors else decision), errors


def validate_generation_response(response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(response.get("distraction"), str) or not response["distraction"].strip():
        errors.append("invalid_distraction")
    if response.get("insertion") not in {"prefix", "suffix"}:
        errors.append("invalid_insertion")
    if response.get("answer_preserved") is not True:
        errors.append("answer_not_preserved")
    return not errors, errors


def validate_translation_response(response: Dict[str, Any], expected_choice_count: int) -> Tuple[bool, List[str]]:
    errors = []
    if not isinstance(response.get("question"), str) or not response["question"].strip():
        errors.append("invalid_question")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != expected_choice_count or not all(isinstance(choice, str) and choice.strip() for choice in choices):
        errors.append("invalid_choices")
    if not isinstance(response.get("answer"), str) or response.get("answer") not in (choices or []):
        errors.append("answer_not_in_choices")
    return not errors, errors


def _load_environment(project_root: Path) -> None:
    load_dotenv(project_root / "utils" / ".env", override=False)


def make_client(project_root: Path, timeout_seconds: float) -> OpenAI:
    _load_environment(project_root)
    if os.getenv("OPENAI_BASE_URL"):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), timeout=timeout_seconds, max_retries=0)
    return OpenAI(
        api_key=os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("BAILIAN_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=timeout_seconds,
        max_retries=0,
    )


def transient_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


def call_model(client: OpenAI, model: str, prompt: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
    attempts = 0
    for attempts in range(1, runtime["max_retries"] + 2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful dataset pipeline component. Follow the requested output format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=runtime["temperature"],
            )
            content = response.choices[0].message.content
            if not content:
                return {"raw_response": None, "attempt_count": attempts, "error": {"category": "empty_response", "message": "empty model response"}}
            return {"raw_response": content, "attempt_count": attempts, "error": None}
        except Exception as error:
            retryable = transient_error(error)
            if retryable and attempts <= runtime["max_retries"]:
                time.sleep(min(2 ** (attempts - 1), 4))
                continue
            return {"raw_response": None, "attempt_count": attempts, "error": {"category": "transient_failure" if retryable else "permanent_failure", "message": str(error)}}
    return {"raw_response": None, "attempt_count": attempts, "error": {"category": "unknown_failure", "message": "no call attempt"}}


def audit_candidate(candidate: Dict[str, Any], config: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    configured = os.getenv("FACTUAL_AUDIT_MODEL", config["roles"]["factual_audit_model"])
    if configured != AUDIT_MODEL:
        raise ValueError(f"FACTUAL_AUDIT_MODEL must be {AUDIT_MODEL!r}")
    call = call_model(client, AUDIT_MODEL, raw_audit_prompt(candidate), config["runtime"])
    record = {"review_id": candidate["candidate_id"], "candidate_id": candidate["candidate_id"], "source_snapshot": candidate["source_snapshot"], "audit": {"model": AUDIT_MODEL, "role": "FACTUAL_AUDIT_MODEL", "prompt_version": PROMPT_VERSION, "temperature": config["runtime"]["temperature"], **call}, "created_at": utc_now()}
    if call["error"]:
        return {**record, "terminal_status": "audit_failed", "decision": "needs_review", "parsed_response": None, "validation_errors": ["audit_request_failed"]}
    try:
        parsed = extract_json_object(call["raw_response"])
        decision, errors = validate_audit_response(parsed, candidate["source_snapshot"]["record"]["answer"])
        return {**record, "terminal_status": "completed", "decision": decision, "parsed_response": parsed, "validation_errors": errors}
    except (json.JSONDecodeError, ValueError) as error:
        return {**record, "terminal_status": "completed", "decision": "needs_review", "parsed_response": None, "validation_errors": [f"invalid_response_json: {error}"]}


def generate_candidate(review: Dict[str, Any], wrong_option: str, wrong_option_index: int, config: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    call = call_model(client, config["roles"]["perturbation_generator_model"], generation_prompt(review, wrong_option), config["runtime"])
    base = {"generation_id": f"{review['candidate_id']}_wrong_{wrong_option_index}", "candidate_id": review["candidate_id"], "source_snapshot": review["source_snapshot"], "factual_review": review, "wrong_option": wrong_option, "generator": {"model": config["roles"]["perturbation_generator_model"], "prompt_version": "answer-preserving-perturbation-v1", "temperature": config["runtime"]["temperature"], **call}, "created_at": utc_now()}
    if call["error"]:
        return {**base, "terminal_status": "generation_failed", "generation_validation_errors": ["generation_request_failed"]}
    try:
        parsed = extract_json_object(call["raw_response"])
        valid, errors = validate_generation_response(parsed)
        record = review["source_snapshot"]["record"]
        distraction = parsed.get("distraction", "")
        enhanced = f"{distraction} {record['question']}" if parsed.get("insertion") == "prefix" else f"{record['question']} {distraction}"
        return {**base, "terminal_status": "completed" if valid else "generation_failed", "generation_response": parsed, "generation_validation_errors": errors, "enhanced_question": enhanced, "choices": record["choices"], "answer": record["answer"]}
    except (json.JSONDecodeError, ValueError) as error:
        return {**base, "terminal_status": "generation_failed", "generation_validation_errors": [f"invalid_generation_json: {error}"]}


def translate_candidate(generation: Dict[str, Any], target_language: str, config: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    call = call_model(client, config["roles"]["translation_model"], translation_prompt(generation, target_language), config["runtime"])
    base = {"translation_id": f"{generation['generation_id']}_{target_language.lower()}", "generation_id": generation["generation_id"], "candidate_id": generation["candidate_id"], "target_language": target_language, "generation": generation, "translator": {"model": config["roles"]["translation_model"], "prompt_version": "generated-candidate-translation-v1", "temperature": config["runtime"]["temperature"], **call}, "created_at": utc_now()}
    if call["error"]:
        return {**base, "terminal_status": "translation_failed", "translation_validation_errors": ["translation_request_failed"]}
    try:
        parsed = extract_json_object(call["raw_response"])
        valid, errors = validate_translation_response(parsed, len(generation["choices"]))
        return {**base, "terminal_status": "completed" if valid else "translation_failed", "translated": parsed if valid else None, "translation_validation_errors": errors}
    except (json.JSONDecodeError, ValueError) as error:
        return {**base, "terminal_status": "translation_failed", "translated": None, "translation_validation_errors": [f"invalid_translation_json: {error}"]}


def screen_question(question: str, choices: List[str], answer: str, model: str, extractor_model: str, config: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    answer_call = call_model(client, model, answer_prompt(question, choices), config["runtime"])
    result = {"model": model, "temperature": config["runtime"]["temperature"], "raw_response": answer_call["raw_response"], "answer_attempt_count": answer_call["attempt_count"], "extraction": None, "correct": False, "score": 0.0}
    if answer_call["error"]:
        result["status"] = "answer_request_failed"
        result["error"] = answer_call["error"]
        return result
    extract_call = call_model(client, extractor_model, answer_extract_prompt(answer_call["raw_response"], choices), config["runtime"])
    result["extraction"] = {"model": extractor_model, **extract_call}
    if extract_call["error"]:
        result["status"] = "extraction_request_failed"
        result["error"] = extract_call["error"]
        return result
    try:
        final_answer = extract_json_object(extract_call["raw_response"]).get("final_answer")
    except (json.JSONDecodeError, ValueError):
        final_answer = None
    if final_answer not in choices:
        result["status"] = "out_of_choices"
        result["extracted_answer"] = final_answer
        return result
    result.update({"status": "success", "extracted_answer": final_answer, "correct": final_answer == answer, "score": 1.0 if final_answer == answer else 0.0})
    return result


def screen_translation(translation: Dict[str, Any], config: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    generation = translation["generation"]
    models = config["roles"]["screen_models"]
    extractor = config["roles"]["answer_extract_model"]
    english = {model: screen_question(generation["enhanced_question"], generation["choices"], generation["answer"], model, extractor, config, client) for model in models}
    translated = translation["translated"]
    target = {model: screen_question(translated["question"], translated["choices"], translated["answer"], model, extractor, config, client) for model in models}
    rate_ori = sum(item["score"] for item in english.values()) / len(models)
    rate_trans = sum(item["score"] for item in target.values()) / len(models)
    pitfall_score = rate_ori - rate_trans
    thresholds = config["weakness_thresholds"]
    retained = rate_ori >= thresholds["min_rate_ori"] and rate_trans <= thresholds["max_rate_trans"] and pitfall_score >= thresholds["min_pitfall_score"]
    return {"screen_id": translation["translation_id"], "translation_id": translation["translation_id"], "target_language": translation["target_language"], "screen_models": models, "english": english, "target": target, "rate_ori": rate_ori, "rate_trans": rate_trans, "pitfall_score": pitfall_score, "retained": retained, "terminal_status": "completed", "translation": translation, "created_at": utc_now()}


def stage_summary(records: Iterable[Dict[str, Any]], stage: str) -> Dict[str, Any]:
    values = list(records)
    return {"stage": stage, "record_count": len(values), "by_terminal_status": dict(Counter(record.get("terminal_status", "unknown") for record in values)), "by_target_language": dict(Counter(record.get("target_language", "") for record in values if record.get("target_language"))), "retained_count": sum(1 for record in values if record.get("retained") is True)}
