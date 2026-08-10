"""Atomic factual-triple extraction and relation normalization pipeline."""

from __future__ import annotations

import copy
import difflib
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

from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    APITimeoutError as AnthropicAPITimeoutError,
    Anthropic,
    Omit,
    RateLimitError as AnthropicRateLimitError,
)
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


TRIPLE_LABEL_MODELS = ("sensenova-6.7-flash-lite", "qwen3.7-plus")
TRIPLE_EXTRACTION_MODEL = TRIPLE_LABEL_MODELS[0]
RELATION_NORMALIZATION_MODEL = "qwen3.7-plus"
SENSENOVA_BASE_URL = "https://token.sensenova.cn"
QWEN_OPENAI_BASE_URL = "https://ctapi.csxdtx.com:16000/v1"
BAILIAN_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_BAILIAN_API_MODEL = "qwen3.7-plus-2026-05-26"
TRIPLE_EXTRACTION_PROMPT_VERSION = "atomic-factual-triple-v6"
RELATION_INVENTORY_VERSION = "relation-inventory-v1"
CODEX_REFERENCE_VERSION = "codex-factual-triple-reference-v1"
CALIBRATION_ERROR_THRESHOLD = 0.03

EXCLUSION_REASONS = {
    "not_factual",
    "option_dependent",
    "negative_question",
    "multi_step_reasoning",
    "calculation",
    "causal_reasoning",
    "contextual_inference",
    "scenario_classification",
    "clue_solving",
    "typical_action_or_location",
    "subjective_or_normative",
    "ambiguous_subject",
    "ambiguous_relation",
    "non_unique_answer",
    "time_sensitive",
    "malformed_question",
    "long_context_comprehension",
    "answer_not_suitable",
    "other",
}

TRIPLE_FIELDS = (
    "subject",
    "subject_type",
    "relation_raw",
    "answer",
    "answer_type",
    "canonical_fact",
)

ROLE_NAMES = {"triple_label_models", "relation_normalization_model"}


class RawSourceSnapshot(TypedDict):
    dataset: str
    source_path: str
    original_index: int
    record: Dict[str, Any]


class RawManifest(TypedDict):
    manifest_version: str
    run_id: str
    created_at: str
    sampling_config: Dict[str, Any]
    source_datasets: Dict[str, Any]
    candidates: List[Dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
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
    raw_sample_size = config.get("raw_sample_size")
    if raw_sample_size != "all" and (not isinstance(raw_sample_size, int) or raw_sample_size <= 0):
        raise ValueError('raw_sample_size must be a positive integer or "all"')
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if config.get("sampling_strategy") not in {"balanced_sources", "proportional"}:
        raise ValueError("sampling_strategy must be 'balanced_sources' or 'proportional'")

    roles = config.get("roles")
    if not isinstance(roles, dict) or set(roles) != ROLE_NAMES:
        raise ValueError(f"roles must contain exactly {sorted(ROLE_NAMES)}")
    if roles["triple_label_models"] != list(TRIPLE_LABEL_MODELS):
        raise ValueError(f"roles.triple_label_models must be {list(TRIPLE_LABEL_MODELS)!r}")
    if roles["relation_normalization_model"] != RELATION_NORMALIZATION_MODEL:
        raise ValueError(f"roles.relation_normalization_model must be {RELATION_NORMALIZATION_MODEL!r}")

    supported_models = config.get("supported_models")
    if not isinstance(supported_models, list) or not supported_models or not all(
        isinstance(model, str) and model.strip() for model in supported_models
    ):
        raise ValueError("supported_models must be a non-empty list of provider-validated model IDs")
    configured_models = set(roles["triple_label_models"]) | {roles["relation_normalization_model"]}
    unsupported = sorted(configured_models - set(supported_models))
    if unsupported:
        raise ValueError(f"roles contain unsupported model IDs: {unsupported}")

    runtime = config.get("runtime", {})
    for key in ("temperature", "timeout_seconds"):
        if not isinstance(runtime.get(key), (int, float)):
            raise ValueError(f"runtime.{key} must be numeric")
    if not isinstance(runtime.get("max_tokens", 4096), int) or runtime.get("max_tokens", 4096) <= 0:
        raise ValueError("runtime.max_tokens must be a positive integer")
    if not isinstance(runtime.get("max_retries"), int) or runtime["max_retries"] < 0:
        raise ValueError("runtime.max_retries must be a non-negative integer")
    if not isinstance(config.get("output_root"), str) or not config["output_root"].strip():
        raise ValueError("output_root must be a non-empty path string")


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
    source_subset = str(record.get("subject") or "__no_subject__")
    return f"{dataset}::{source_subset}"


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


def balanced_capacity_allocation(capacities: Dict[str, int], desired: int) -> Dict[str, int]:
    """Allocate a pilot evenly across sources and redistribute source shortfalls."""
    allocation = {key: 0 for key in capacities}
    desired = min(desired, sum(capacities.values()))
    keys = sorted(capacities)
    while sum(allocation.values()) < desired:
        progressed = False
        for key in keys:
            if allocation[key] < capacities[key] and sum(allocation.values()) < desired:
                allocation[key] += 1
                progressed = True
        if not progressed:
            break
    return allocation


def interleave_candidates_by_dataset(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a deterministic dataset round-robin order for bounded pilot runs."""
    queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        queues[candidate["source_snapshot"]["dataset"]].append(candidate)
    for dataset in queues:
        queues[dataset].sort(key=lambda item: (item["stratum"], item["source_snapshot"]["original_index"]))
    offsets = {dataset: 0 for dataset in queues}
    ordered = []
    while any(offsets[dataset] < len(queues[dataset]) for dataset in queues):
        for dataset in sorted(queues):
            offset = offsets[dataset]
            if offset < len(queues[dataset]):
                ordered.append(queues[dataset][offset])
                offsets[dataset] += 1
    return ordered


def build_raw_manifest(
    config: Dict[str, Any],
    project_root: Path,
    run_id: str,
    excluded_candidate_ids: Optional[Iterable[str]] = None,
) -> RawManifest:
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

    excluded_ids = set(excluded_candidate_ids or [])
    available_candidates = []
    for candidate in all_candidates:
        if candidate["candidate_id"] in excluded_ids:
            source_details[candidate["source_snapshot"]["dataset"]]["exclusions"].append(
                {
                    "original_index": candidate["source_snapshot"]["original_index"],
                    "reason": "excluded_by_prior_manifest",
                    "candidate_id": candidate["candidate_id"],
                }
            )
        else:
            available_candidates.append(candidate)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dataset_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in available_candidates:
        groups[candidate["stratum"]].append(candidate)
        dataset_groups[candidate["source_snapshot"]["dataset"]].append(candidate)
    full_population = config["raw_sample_size"] == "all"
    desired_count = len(available_candidates) if full_population else config["raw_sample_size"]
    if full_population or config["sampling_strategy"] == "proportional":
        allocation = proportional_allocation(groups, desired_count)
        dataset_allocation = {
            dataset: sum(
                count for stratum, count in allocation.items() if stratum.startswith(f"{dataset}::")
            )
            for dataset in dataset_groups
        }
    else:
        dataset_allocation = balanced_capacity_allocation(
            {dataset: len(values) for dataset, values in dataset_groups.items()}, desired_count
        )
        allocation = {stratum: 0 for stratum in groups}
        for dataset, dataset_quota in dataset_allocation.items():
            source_strata = {
                stratum: values
                for stratum, values in groups.items()
                if stratum.startswith(f"{dataset}::")
            }
            allocation.update(proportional_allocation(source_strata, dataset_quota))
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
    selected = interleave_candidates_by_dataset(selected)
    for candidate in available_candidates:
        source_details[candidate["source_snapshot"]["dataset"]]["unique_eligible_count"] += 1
    return {
        "manifest_version": "raw-source-v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "sampling_config": {
            "source_datasets": copy.deepcopy(config["source_datasets"]),
            "raw_sample_size": config["raw_sample_size"],
            "seed": config["seed"],
            "sampling_strategy": config["sampling_strategy"],
            "excluded_candidate_count": len(excluded_ids),
        },
        "source_datasets": {
            **source_details,
            "allocation": {
                "requested_count": config["raw_sample_size"],
                "available_count": len(available_candidates),
                "actual_count": len(selected),
                "shortfall": max(0, desired_count - len(selected)),
                "selection_mode": "full_population" if full_population else "stratified_sample",
                "dataset_allocations": dataset_allocation,
                "strata": stratum_details,
            },
        },
        "candidates": selected,
    }


def triple_extraction_prompt(candidate: Dict[str, Any]) -> str:
    snapshot = candidate["source_snapshot"]
    record = snapshot["record"]
    source_subset = record.get("subject") or ""
    return f'''You are a high-precision annotator constructing an atomic factual-triple dataset from English QA sources.

Your task is NOT to answer the question and NOT to review answer choices. The canonical answer is supplied by the source dataset. Determine whether the English question and canonical answer express exactly one atomic fact that can be represented as:

(subject, relation_raw, answer)

Atomic fact requirements:
1. The subject is one uniquely identifiable entity, concept, substance, process, event, work, place, or factual object.
2. relation_raw describes one stable factual property or relationship from subject to answer.
3. The answer is a concise object copied from or explicitly stated inside the provided canonical answer. A full-sentence source answer may be normalized to the entity or short phrase that directly answers the question.
4. The fact is directly retrievable without answer choices, multi-step reasoning, calculation, scenario interpretation, or prediction of a typical action/location.
5. The question provides enough scope for one sufficiently unique answer.
6. The fact is not materially time-sensitive unless the question includes an explicit time anchor.
7. The final answer is a concise entity, category, attribute, date, quantity, or short noun phrase. A source sentence is allowed only when it explicitly contains such an answer; explanations, procedures, recommendations, and multi-clause propositions are not triple objects.
8. Test uniqueness without choices: if the same subject and relation have multiple independently correct objects, reject the item even when only one appears among the MCQ choices.
9. "Which of these/following" is extractable only when the question itself names a complete comparison universe and the answer is globally unique in that universe. An unspecified option set, a disputed "best/most reliable" choice, or a relation with several correct objects is option_dependent or non_unique_answer.
10. Distinguish direct properties from causal explanation: "compounds decompose when heated" and "climate factors determine plant types" state one relation and are extractable; asking why an event occurs or interpreting a scenario's cause remains causal_reasoning.
11. A full-sentence source answer may be shortened only when the question explicitly asks for an entity, person, place, date, quantity, category, or attribute. Do not turn an explanatory proposition requested by "which statement/best explains/why" into a triple object.

Triple direction is mandatory:
- Orient the triple as subject --relation_raw--> the concise grounded answer.
- Never reverse the triple merely to obtain a more familiar relation.
- The subject must not equal, alias, or simply repeat the canonical answer.
- The subject may be a named anchor or one concise defining class/description that expresses a single definition.
- Accept single-definition and direct classification questions such as a device definition, scientific term definition, anatomical name, or reaction category. Use a stable relation such as "term for definition" or "classification".
- Reject multi-clue identity riddles that combine several independent biographical, geographic, or narrative hints merely to guess a person/place. Those are clue_solving.
- Example: "Which country uses the yen?" with canonical answer "Japan" must use subject "Yen" and answer "Japan". It must not become subject "Japan" and answer "Yen".

Use extraction_status="not_extractable" for option-dependent or negative questions, multi-step reasoning, calculation, causal reasoning, scenario classification, clue solving, typical actions or locations, subjective judgments, ambiguity, non-unique answers, unanchored changing facts, malformed premises, long-context comprehension, or any item not expressible as one atomic fact.

Output rules:
- relation_raw must be a short, self-contained English noun phrase or relational phrase describing the direction from subject to answer.
- relation_raw must name the factual relationship itself. Do not use vague scaffolding such as "described by", "related to", "associated with", "answer to", or wording that depends on the original question.
- relation_raw must remain interpretable when read only with subject and answer.
- Do not create relation_normalized.
- Do not generate prompt_en.
- Prefer an exact copy. If the source answer is a sentence, copy only the shortest substring that directly answers the question. Do not use outside knowledge to invent or expand an answer.
- For not_extractable, every triple field must be null.
- Return exactly one JSON object and no Markdown.

Boundary examples:

Example 1 — extractable direct relation:
Question: Who is the creator of the comic strip 'The Far Side'?
Canonical answer: Gary Larson
Result: subject="The Far Side", relation_raw="creator", answer="Gary Larson".

Example 2 — one direct definition is extractable:
Question: What do we call a simple machine that consists of a rope and grooved wheel?
Canonical answer: pulley
Result: subject="simple machine consisting of a rope and grooved wheel", relation_raw="term for definition", answer="pulley".

Example 3 — comparison set can be the subject, but the answer cannot be copied into the subject:
Question: Which planet has the shortest orbit around the Sun?
Canonical answer: Mercury
Result: subject="Solar System planets", relation_raw="member with the shortest orbit around the Sun", answer="Mercury".

Example 4 — explicit named phrase can have a definition:
Question: What is meant by the phrase 'empire by invitation'?
Canonical answer: Voluntary reliance on an external power for security
Result: subject="empire by invitation", relation_raw="meaning", answer="Voluntary reliance on an external power for security".

Example 5 — causal question is not an atomic retrieval triple for this dataset:
Question: What occurs when competing species evolve different adaptations?
Canonical answer: specialization
Result: not_extractable, exclusion_reason="causal_reasoning".

Example 6 — direct scientific classification is extractable:
Question: What type of reactions absorb heat from their surroundings?
Canonical answer: endothermic
Result: subject="reactions that absorb heat from their surroundings", relation_raw="classification", answer="endothermic".

Example 7 — sentence answer is not a triple object:
Question: Which statement describes the formation of a delta?
Canonical answer: A mass of sediment is deposited at the mouth of a river.
Result: not_extractable, exclusion_reason="answer_not_suitable".

Example 8 — normalize an entity explicitly contained in a sentence answer:
Question: In what country was Barack Obama born?
Canonical answer: Obama was born in Hawaii, which is a US state
Result: subject="Barack Obama", relation_raw="country of birth", answer="US".

Example 9 — reject an option-dependent non-unique fact:
Question: Which element is included in every protein molecule?
Canonical answer: carbon
Result: not_extractable, exclusion_reason="non_unique_answer", because hydrogen, oxygen, and nitrogen are also generally present and only the choices make carbon unique.

Example 10 — a quoted sentence may contain a short entity answer:
Question: Who said, "Only the dead have seen the end of war?"
Canonical answer: George Santayana wrote "Only the dead have seen the end of war"
Result: subject="Only the dead have seen the end of war", relation_raw="quotation author", answer="George Santayana".

Example 11 — multi-hint identity resolution is not a definition:
Question: This Jewish American studied at NYU, lived in New York City, and became famous for a controversial act. His name is Bernie what?
Canonical answer: Goetz
Result: not_extractable, exclusion_reason="clue_solving".

Example 12 — missing-list completion is not a reusable unique relation:
Question: The five human senses are taste, touch, vision, hearing and one more. What is it?
Canonical answer: smell
Result: not_extractable, exclusion_reason="non_unique_answer", because "the five senses include" has several objects and uniqueness depends on the prompt-specific partial list.

Example 13 — an anchored date does not repair missing survey provenance:
Question: As of 2019, about what percentage of Italians say free media is very important?
Canonical answer: 56%
Result: not_extractable, exclusion_reason="ambiguous_relation", because no survey source or measurement frame makes the percentage unique.

Example 14 — condition is a direct property, not causal reasoning:
Question: Most mercury compounds decompose when they are?
Canonical answer: heated
Result: subject="most mercury compounds", relation_raw="decompose when", answer="heated".

Example 15 — direct stated ecological relation is extractable:
Question: Temperature and precipitation determine the types of what that can grow in an area?
Canonical answer: plants
Result: subject="temperature and precipitation", relation_raw="determine types of organisms that can grow in an area", answer="plants".

Example 16 — named theoretical constructs may be a clear subject:
Question: The needs of Murray and Maslow are similar to what from operant conditioning theory?
Canonical answer: Reinforcers
Result: subject="needs in Murray and Maslow's theories", relation_raw="analogous concept in operant conditioning", answer="Reinforcers".

Example 17 — scenario-to-principle classification is excluded:
Question: Matching Grand Canyon rock layers on opposite sides were deposited simultaneously; this is an example of what?
Canonical answer: lateral continuity
Result: not_extractable, exclusion_reason="scenario_classification".

Example 18 — incomplete comparison universe is option dependent:
Question: Which of the following has a warmer average surface temperature than Earth?
Canonical answer: Venus
Result: not_extractable, exclusion_reason="option_dependent", because the compared set is absent and more than one Solar System body can satisfy the relation.

Example 19 — disputed best-property/tool choices are excluded:
Question: Which property is most reliable for identifying one mineral from another?
Canonical answer: hardness
Result: not_extractable, exclusion_reason="option_dependent".

Example 20 — advice is not a factual object:
Question: How long should you wait between eating and swimming?
Canonical answer: You can swim immediately after eating
Result: not_extractable, exclusion_reason="subjective_or_normative".

Example 21 — explanatory statement remains excluded:
Question: Which best explains the sound heard through a radio?
Canonical answer: Sounds are waves that travel through matter.
Result: not_extractable, exclusion_reason="answer_not_suitable".

Example 22 — ambiguous option-set superlative is excluded:
Question: Which of these atomic structures has the least amount of mass?
Canonical answer: an electron
Result: not_extractable, exclusion_reason="option_dependent", because "these" supplies an unstated comparison set and "atomic structures" is not a precise universe.

Example 23 — scenario causal superlative is excluded:
Question: A city has vehicle-exhaust smog. Which weather condition contributes most to smog formation?
Canonical answer: temperature inversion
Result: not_extractable, exclusion_reason="causal_reasoning".

Example 24 — vague subject and best-model selection are excluded:
Question: Where do recent rocks point towards on Earth?
Canonical answer: north magnetic pole
Result: not_extractable, exclusion_reason="ambiguous_subject".
Question: Which type of model best shows underwater ocean ridges?
Canonical answer: topographic map
Result: not_extractable, exclusion_reason="option_dependent".

Allowed exclusion_reason values:
{json.dumps(sorted(EXCLUSION_REASONS), ensure_ascii=False)}

Return this exact schema:
{{
  "schema_version": "{TRIPLE_EXTRACTION_PROMPT_VERSION}",
  "extraction_status": "extracted" | "not_extractable",
  "exclusion_reason": null | "one allowed value",
  "subject": "string or null",
  "subject_type": "short lowercase type or null",
  "relation_raw": "short English relational phrase or null",
  "answer": "concise answer copied from or explicitly contained in the canonical answer, or null",
  "answer_type": "short lowercase type or null",
  "canonical_fact": "one declarative sentence or null",
  "extraction_confidence": 0.0
}}

Source ID: {candidate["candidate_id"]}
Source dataset: {snapshot["dataset"]}
Source subset: {source_subset}
English question: {record["question"]}
Canonical English answer: {record["answer"]}
'''


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    return parsed


def validate_triple_extraction_response(response: Dict[str, Any], canonical_answer: str) -> List[str]:
    errors = []
    if response.get("schema_version") != TRIPLE_EXTRACTION_PROMPT_VERSION:
        errors.append("invalid_schema_version")
    status = response.get("extraction_status")
    if status not in {"extracted", "not_extractable"}:
        errors.append("invalid_extraction_status")

    confidence = response.get("extraction_confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        errors.append("invalid_extraction_confidence")

    missing_fields = [field for field in TRIPLE_FIELDS if field not in response]
    errors.extend(f"missing_{field}" for field in missing_fields)

    if status == "extracted":
        if response.get("exclusion_reason") is not None:
            errors.append("extracted_has_exclusion_reason")
        for field in TRIPLE_FIELDS:
            if not isinstance(response.get(field), str) or not response[field].strip():
                errors.append(f"invalid_{field}")
        if isinstance(response.get("answer"), str) and not answer_is_grounded_in_source(
            response["answer"], canonical_answer
        ):
            errors.append("answer_not_grounded_in_source")
        if (
            isinstance(response.get("subject"), str)
            and isinstance(response.get("answer"), str)
            and normalize_text(response["subject"]) == normalize_text(response["answer"])
        ):
            errors.append("subject_matches_answer")
    elif status == "not_extractable":
        if response.get("exclusion_reason") not in EXCLUSION_REASONS:
            errors.append("invalid_exclusion_reason")
        for field in TRIPLE_FIELDS:
            if response.get(field) is not None:
                errors.append(f"not_extractable_has_{field}")
    return errors


def source_answer_is_unsuitable(answer: str) -> bool:
    """Conservative rule for sentence-like or overly long triple objects."""
    words = re.findall(r"[\w'-]+", answer, flags=re.UNICODE)
    stripped = answer.strip()
    return (
        len(words) > 12
        or (len(words) > 5 and bool(re.search(r"[.!?][\"')\]]?$", stripped)))
        or (len(words) > 8 and (":" in stripped or ";" in stripped))
        or (len(words) > 5 and bool(re.match(r"^(?:it|they|he|she|you|we|there)\b", stripped, re.IGNORECASE)))
    )


def answer_is_grounded_in_source(answer: str, source_answer: str) -> bool:
    """Return whether the normalized answer is copied from the source answer."""
    normalized_answer = normalize_text(answer).strip(" .,:;!?\"'()[]")
    normalized_source = normalize_text(source_answer).strip(" .,:;!?\"'()[]")
    if not normalized_answer:
        return False
    if normalized_answer in normalized_source:
        return True
    without_article = re.sub(r"^(?:a|an|the)\s+", "", normalized_answer)
    if without_article and without_article in normalized_source:
        return True
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "for"}
    answer_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized_answer) if token not in stopwords}
    source_tokens = {token for token in re.findall(r"[a-z0-9]+", normalized_source) if token not in stopwords}
    return bool(answer_tokens and answer_tokens <= source_tokens)


def adjust_extraction_response_grounding(
    response: Dict[str, Any], source_answer: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply transparent local repairs while retaining the raw model response."""
    adjusted = copy.deepcopy(response)
    changes: List[str] = []
    if adjusted.get("extraction_status") != "extracted" or not isinstance(adjusted.get("answer"), str):
        return adjusted, changes
    answer = adjusted["answer"]
    if answer_is_grounded_in_source(answer, source_answer):
        return adjusted, changes
    similarity = difflib.SequenceMatcher(
        None, normalize_text(answer), normalize_text(source_answer)
    ).ratio()
    if not source_answer_is_unsuitable(source_answer) and similarity >= 0.85:
        adjusted["answer"] = source_answer
        changes.append("answer_near_match_replaced_with_source_answer")
        return adjusted, changes
    if source_answer_is_unsuitable(source_answer):
        adjusted.update(
            {
                "extraction_status": "not_extractable",
                "exclusion_reason": "answer_not_suitable",
                **{field: None for field in TRIPLE_FIELDS},
            }
        )
        changes.append("ungrounded_sentence_answer_downgraded_to_not_extractable")
    return adjusted, changes


def _load_environment(project_root: Path) -> None:
    load_dotenv(project_root / "utils" / ".env", override=False)


def make_extraction_client(project_root: Path, timeout_seconds: float, model: str) -> Any:
    _load_environment(project_root)
    if model == "sensenova-6.7-flash-lite":
        api_key = os.getenv("SENSENOVA_API_KEY")
        if not api_key:
            raise ValueError("SENSENOVA_API_KEY is required for SenseNova triple extraction")
        return Anthropic(
            api_key="",
            auth_token=api_key,
            base_url=os.getenv("SENSENOVA_BASE_URL") or SENSENOVA_BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
            default_headers={"Authorization": f"Bearer {api_key}", "X-Api-Key": Omit()},
        )
    if model == "qwen3.7-plus":
        provider = (os.getenv("QWEN_EXTRACTION_PROVIDER") or "bailian").strip().lower()
        if provider == "bailian":
            api_key = os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            base_url = (
                os.getenv("BAILIAN_BASE_URL")
                or os.getenv("DASHSCOPE_BASE_URL")
                or BAILIAN_OPENAI_BASE_URL
            )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or QWEN_OPENAI_BASE_URL
        else:
            raise ValueError(f"Unsupported Qwen extraction provider: {provider}")
        if not api_key:
            raise ValueError(f"API key is required for Qwen extraction provider: {provider}")
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
    raise ValueError(f"Unsupported triple label model: {model}")


def make_normalization_client(project_root: Path, timeout_seconds: float) -> OpenAI:
    _load_environment(project_root)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("BAILIAN_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise ValueError("OPENAI_API_KEY or BAILIAN_API_KEY/DASHSCOPE_API_KEY is required for relation normalization")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)


def transient_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            AnthropicAPIConnectionError,
            AnthropicAPITimeoutError,
            AnthropicRateLimitError,
        ),
    ):
        return True
    return isinstance(error, (APIStatusError, AnthropicAPIStatusError)) and error.status_code >= 500


def call_extraction_model(client: Any, model: str, prompt: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
    attempts = 0
    provider = (os.getenv("QWEN_EXTRACTION_PROVIDER") or "bailian").strip().lower()
    api_model = QWEN_BAILIAN_API_MODEL if model == "qwen3.7-plus" and provider == "bailian" else model
    for attempts in range(1, runtime["max_retries"] + 2):
        try:
            system = "You are a careful atomic factual-triple extraction component. Follow the JSON contract exactly."
            if model == "sensenova-6.7-flash-lite":
                response = client.messages.create(
                    model=model,
                    max_tokens=runtime.get("max_tokens", 4096),
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=runtime["temperature"],
                )
                content = "".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                ).strip()
            elif model == "qwen3.7-plus":
                request = {
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": runtime["temperature"],
                    "max_tokens": runtime.get("max_tokens", 4096),
                }
                if provider == "bailian":
                    request["extra_body"] = {"enable_thinking": False}
                response = client.chat.completions.create(**request)
                content = (response.choices[0].message.content or "").strip()
            else:
                raise ValueError(f"Unsupported triple label model: {model}")
            if not content:
                return {
                    "api_model": api_model,
                    "raw_response": None,
                    "attempt_count": attempts,
                    "error": {"category": "empty_response", "message": "empty model response"},
                }
            return {"api_model": api_model, "raw_response": content, "attempt_count": attempts, "error": None}
        except Exception as error:
            retryable = transient_error(error)
            if retryable and attempts <= runtime["max_retries"]:
                time.sleep(min(2 ** (attempts - 1), 4))
                continue
            return {
                "api_model": api_model,
                "raw_response": None,
                "attempt_count": attempts,
                "error": {
                    "category": "transient_failure" if retryable else "permanent_failure",
                    "message": str(error),
                },
            }
    return {
        "api_model": api_model,
        "raw_response": None,
        "attempt_count": attempts,
        "error": {"category": "unknown_failure", "message": "no call attempt"},
    }


def _source_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = candidate["source_snapshot"]
    record = snapshot["record"]
    return {
        "source_id": candidate["candidate_id"],
        "source_dataset": snapshot["dataset"],
        "source_subset": record.get("subject"),
        "source_path": snapshot["source_path"],
        "source_original_index": snapshot["original_index"],
        "source_question": record["question"],
        "source_choices": copy.deepcopy(record["choices"]),
        "source_answer": record["answer"],
    }


def extract_triple_candidate(
    candidate: Dict[str, Any], config: Dict[str, Any], client: Any, model: str = TRIPLE_EXTRACTION_MODEL
) -> Dict[str, Any]:
    if model not in config["roles"]["triple_label_models"] or model not in TRIPLE_LABEL_MODELS:
        raise ValueError(f"Unsupported configured triple label model: {model}")
    base = {
        **_source_fields(candidate),
        "candidate_id": candidate["candidate_id"],
        "source_snapshot": candidate["source_snapshot"],
        "created_at": utc_now(),
    }
    empty_triple = {field: None for field in TRIPLE_FIELDS}
    call = call_extraction_model(
        client,
        model,
        triple_extraction_prompt(candidate),
        config["runtime"],
    )
    base = {
        **base,
        "extraction": {
            "model": model,
            "role": "TRIPLE_LABEL_MODEL",
            "prompt_version": TRIPLE_EXTRACTION_PROMPT_VERSION,
            "temperature": config["runtime"]["temperature"],
            **call,
        },
    }
    if call["error"]:
        return {
            **base,
            "terminal_status": "extraction_failed",
            "extraction_status": None,
            "exclusion_reason": None,
            **empty_triple,
            "extraction_confidence": None,
            "parsed_response": None,
            "validation_errors": ["extraction_request_failed"],
        }
    try:
        parsed = extract_json_object(call["raw_response"])
        parsed, program_adjustments = adjust_extraction_response_grounding(parsed, base["source_answer"])
        base["extraction"]["program_adjustments"] = program_adjustments
        errors = validate_triple_extraction_response(parsed, base["source_answer"])
        return {
            **base,
            "terminal_status": "validation_failed" if errors else "completed",
            "extraction_status": parsed.get("extraction_status"),
            "exclusion_reason": parsed.get("exclusion_reason"),
            **{field: parsed.get(field) for field in TRIPLE_FIELDS},
            "extraction_confidence": parsed.get("extraction_confidence"),
            "parsed_response": parsed,
            "validation_errors": errors,
        }
    except (json.JSONDecodeError, ValueError) as error:
        return {
            **base,
            "terminal_status": "validation_failed",
            "extraction_status": None,
            "exclusion_reason": None,
            **empty_triple,
            "extraction_confidence": None,
            "parsed_response": None,
            "validation_errors": [f"invalid_response_json: {error}"],
        }


def relation_signature_id(relation_raw: str, subject_type: str, answer_type: str) -> str:
    payload = "\u241f".join(
        (normalize_text(relation_raw), normalize_text(subject_type), normalize_text(answer_type))
    )
    return f"rel_sig_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def build_relation_inventory(records: Iterable[Dict[str, Any]], max_examples: int = 5) -> Dict[str, Any]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    records = list(records)
    groups: Dict[str, Dict[str, Any]] = {}
    extracted_count = 0
    for record in records:
        if (
            record.get("terminal_status") != "completed"
            or record.get("extraction_status") != "extracted"
            or record.get("validation_errors")
        ):
            continue
        extracted_count += 1
        signature_id = relation_signature_id(
            record["relation_raw"], record["subject_type"], record["answer_type"]
        )
        entry = groups.setdefault(
            signature_id,
            {
                "signature_id": signature_id,
                "relation_raw": record["relation_raw"],
                "relation_raw_normalized": normalize_text(record["relation_raw"]),
                "subject_type": record["subject_type"],
                "answer_type": record["answer_type"],
                "direction": f"{record['subject_type']} -> {record['answer_type']}",
                "count": 0,
                "examples": [],
            },
        )
        entry["count"] += 1
        if len(entry["examples"]) < max_examples:
            entry["examples"].append(
                {
                    "source_id": record["source_id"],
                    "source_dataset": record["source_dataset"],
                    "subject": record["subject"],
                    "answer": record["answer"],
                    "canonical_fact": record["canonical_fact"],
                }
            )
    entries = sorted(groups.values(), key=lambda item: (-item["count"], item["signature_id"]))
    return {
        "inventory_version": RELATION_INVENTORY_VERSION,
        "created_at": utc_now(),
        "source_record_count": len(records),
        "extracted_record_count": extracted_count,
        "signature_count": len(entries),
        "max_examples_per_signature": max_examples,
        "entries": entries,
    }


def validate_taxonomy(taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    version = taxonomy.get("taxonomy_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("taxonomy_version must be a non-empty string")
    relations = taxonomy.get("relations")
    if not isinstance(relations, list):
        raise ValueError("taxonomy.relations must be a list")
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError(f"taxonomy relation {index} must be an object")
        for field in ("id", "definition", "subject_type", "answer_type", "direction"):
            if not isinstance(relation.get(field), str) or not relation[field].strip():
                raise ValueError(f"taxonomy relation {index} has invalid {field}")
        relation_id = relation["id"]
        if relation_id in by_id:
            raise ValueError(f"duplicate taxonomy relation id: {relation_id}")
        expected_direction = f"{relation['subject_type']} -> {relation['answer_type']}"
        if relation["direction"] != expected_direction:
            raise ValueError(
                f"taxonomy relation {relation_id} direction must be {expected_direction!r}"
            )
        by_id[relation_id] = relation
    return by_id


def validate_relation_mapping(
    mapping: Dict[str, Any], inventory: Dict[str, Any], taxonomy: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    relations = validate_taxonomy(taxonomy)
    if mapping.get("taxonomy_version") != taxonomy["taxonomy_version"]:
        raise ValueError("mapping taxonomy_version does not match taxonomy")
    if mapping.get("inventory_version") != inventory.get("inventory_version"):
        raise ValueError("mapping inventory_version does not match inventory")
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise ValueError("inventory.entries must be a list")
    inventory_by_id = {entry["signature_id"]: entry for entry in entries}
    mappings = mapping.get("mappings")
    if not isinstance(mappings, list):
        raise ValueError("mapping.mappings must be a list")
    by_signature: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(mappings):
        if not isinstance(item, dict):
            raise ValueError(f"mapping item {index} must be an object")
        signature_id = item.get("signature_id")
        if signature_id not in inventory_by_id:
            raise ValueError(f"mapping references unknown signature_id: {signature_id}")
        if signature_id in by_signature:
            raise ValueError(f"duplicate mapping for signature_id: {signature_id}")
        status = item.get("normalization_status")
        if status not in {"mapped", "out_of_taxonomy", "ambiguous"}:
            raise ValueError(f"invalid normalization_status for {signature_id}")
        relation_id = item.get("relation_normalized")
        if status == "mapped":
            if relation_id not in relations:
                raise ValueError(f"mapping references unknown relation: {relation_id}")
            relation = relations[relation_id]
            if item.get("subject_type_normalized") != relation["subject_type"]:
                raise ValueError(f"mapping normalized subject type mismatch for {signature_id} -> {relation_id}")
            if item.get("answer_type_normalized") != relation["answer_type"]:
                raise ValueError(f"mapping normalized answer type mismatch for {signature_id} -> {relation_id}")
        elif relation_id is not None:
            raise ValueError(f"unresolved mapping {signature_id} must have null relation_normalized")
        elif item.get("subject_type_normalized") is not None or item.get("answer_type_normalized") is not None:
            raise ValueError(f"unresolved mapping {signature_id} must have null normalized types")
        by_signature[signature_id] = item
    return by_signature


def apply_relation_mapping(
    records: Iterable[Dict[str, Any]],
    inventory: Dict[str, Any],
    taxonomy: Dict[str, Any],
    mapping: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mapping_by_signature = validate_relation_mapping(mapping, inventory, taxonomy)
    normalized = []
    for record in records:
        if record.get("terminal_status") != "completed" or record.get("extraction_status") != "extracted":
            normalized.append(
                {
                    **record,
                    "taxonomy_version": taxonomy["taxonomy_version"],
                    "normalization_status": "not_applicable",
                    "relation_normalized": None,
                }
            )
            continue
        signature_id = relation_signature_id(
            record["relation_raw"], record["subject_type"], record["answer_type"]
        )
        item = mapping_by_signature.get(signature_id)
        normalized.append(
            {
                **record,
                "relation_signature_id": signature_id,
                "taxonomy_version": taxonomy["taxonomy_version"],
                "normalization_status": item["normalization_status"] if item else "out_of_taxonomy",
                "relation_normalized": item.get("relation_normalized") if item else None,
                "subject_type_normalized": item.get("subject_type_normalized") if item else None,
                "answer_type_normalized": item.get("answer_type_normalized") if item else None,
                "normalization_reason": item.get("normalization_reason") if item else "no explicit mapping",
                "normalization_confidence": item.get("normalization_confidence") if item else None,
            }
        )
    return normalized


def model_slug(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-.")
    if not slug:
        raise ValueError("model must produce a non-empty path slug")
    return slug


def extraction_summary(records: Iterable[Dict[str, Any]], model: str = TRIPLE_EXTRACTION_MODEL) -> Dict[str, Any]:
    records = list(records)
    return {
        "stage": "extract",
        "record_count": len(records),
        "terminal_status_counts": dict(Counter(record.get("terminal_status", "unknown") for record in records)),
        "extraction_status_counts": dict(Counter(record.get("extraction_status", "unknown") for record in records)),
        "exclusion_reason_counts": dict(
            Counter(record.get("exclusion_reason") for record in records if record.get("exclusion_reason"))
        ),
        "validation_error_count": sum(bool(record.get("validation_errors")) for record in records),
        "model": model,
        "prompt_version": TRIPLE_EXTRACTION_PROMPT_VERSION,
        "created_at": utc_now(),
    }


def build_codex_reference(
    base_records: Iterable[Dict[str, Any]], policy: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if policy.get("reference_version") != CODEX_REFERENCE_VERSION:
        raise ValueError("invalid Codex reference version")
    overrides = {item["candidate_id"]: item for item in policy.get("overrides", [])}
    annotations = {item["candidate_id"]: item for item in policy.get("annotations", [])}
    records = []
    for base in base_records:
        candidate_id = base["candidate_id"]
        decision = overrides.get(candidate_id, {})
        annotation = annotations.get(candidate_id, {})
        status = decision.get("extraction_status", base.get("extraction_status"))
        if status not in {"extracted", "not_extractable"}:
            raise ValueError(f"invalid reference status for {candidate_id}")
        expected = decision.get("expected_triple")
        if expected is None and status == "extracted" and base.get("extraction_status") == "extracted":
            expected = {field: base.get(field) for field in TRIPLE_FIELDS}
        records.append(
            {
                "reference_version": CODEX_REFERENCE_VERSION,
                "candidate_id": candidate_id,
                **{key: base.get(key) for key in (
                    "source_id", "source_dataset", "source_subset", "source_path",
                    "source_original_index", "source_question", "source_choices", "source_answer",
                )},
                "extraction_status": status,
                "expected_triple": expected if status == "extracted" else None,
                "codex_review": decision.get("codex_review") or annotation.get("codex_review") or "agrees_with_v4",
                "review_tier": decision.get("review_tier") or annotation.get("review_tier") or "clear",
            }
        )
    unknown = sorted((set(overrides) | set(annotations)) - {row["candidate_id"] for row in records})
    if unknown:
        raise ValueError(f"reference policy contains unknown candidate IDs: {unknown}")
    return records


def evaluate_model_against_codex_reference(
    reference_records: Iterable[Dict[str, Any]],
    model_records: Iterable[Dict[str, Any]],
    model: str,
    threshold: float = CALIBRATION_ERROR_THRESHOLD,
) -> Dict[str, Any]:
    reference = {row["candidate_id"]: row for row in reference_records}
    observed = {row["candidate_id"]: row for row in model_records}
    disagreements = []
    incomplete = []
    field_mismatches = []
    for candidate_id, gold in reference.items():
        row = observed.get(candidate_id)
        if row is None or row.get("terminal_status") != "completed":
            incomplete.append(
                {
                    "candidate_id": candidate_id,
                    "reference_status": gold["extraction_status"],
                    "observed_terminal_status": row.get("terminal_status") if row else "missing",
                }
            )
            continue
        if row.get("extraction_status") != gold["extraction_status"]:
            disagreements.append(
                {
                    "candidate_id": candidate_id,
                    "source_dataset": gold.get("source_dataset"),
                    "source_original_index": gold.get("source_original_index"),
                    "source_question": gold.get("source_question"),
                    "source_answer": gold.get("source_answer"),
                    "reference_status": gold["extraction_status"],
                    "model_status": row.get("extraction_status"),
                    "model_exclusion_reason": row.get("exclusion_reason"),
                    "model_triple": {field: row.get(field) for field in TRIPLE_FIELDS},
                }
            )
        elif gold["extraction_status"] == "extracted" and gold.get("expected_triple"):
            expected = gold["expected_triple"]
            differing = [
                field for field in ("subject", "relation_raw", "answer")
                if normalize_text(str(row.get(field) or "")) != normalize_text(str(expected.get(field) or ""))
            ]
            if differing:
                field_mismatches.append(
                    {
                        "candidate_id": candidate_id,
                        "differing_fields": differing,
                        "expected_triple": expected,
                        "model_triple": {field: row.get(field) for field in TRIPLE_FIELDS},
                    }
                )
    unexpected = sorted(set(observed) - set(reference))
    error_count = len(disagreements) + len(incomplete)
    reference_count = len(reference)
    error_rate = error_count / reference_count if reference_count else 1.0
    return {
        "evaluation_version": "codex-calibration-evaluation-v1",
        "reference_version": CODEX_REFERENCE_VERSION,
        "model": model,
        "prompt_version": TRIPLE_EXTRACTION_PROMPT_VERSION,
        "reference_count": reference_count,
        "completed_reference_count": reference_count - len(incomplete),
        "label_disagreement_count": len(disagreements),
        "incomplete_count": len(incomplete),
        "label_error_count": error_count,
        "label_error_rate": error_rate,
        "threshold": threshold,
        "gate_passed": error_rate <= threshold and not incomplete and not unexpected,
        "disagreements": disagreements,
        "incomplete": incomplete,
        "unexpected_candidate_ids": unexpected,
        "extracted_field_mismatch_count": len(field_mismatches),
        "field_mismatches": field_mismatches,
        "created_at": utc_now(),
    }


def summarize_dual_model_records(
    records_by_model: Dict[str, Iterable[Dict[str, Any]]]
) -> Dict[str, Any]:
    models = list(records_by_model)
    if len(models) != 2:
        raise ValueError("dual-model summary requires exactly two models")
    indexed = {model: {row["candidate_id"]: row for row in rows} for model, rows in records_by_model.items()}
    candidate_ids = sorted(set().union(*(set(rows) for rows in indexed.values())))
    agreement = Counter()
    disagreements = []
    for candidate_id in candidate_ids:
        rows = {model: indexed[model].get(candidate_id) for model in models}
        statuses = {
            model: (row.get("extraction_status") if row and row.get("terminal_status") == "completed" else None)
            for model, row in rows.items()
        }
        if len(set(statuses.values())) == 1 and None not in statuses.values():
            agreement[f"consensus_{next(iter(statuses.values()))}"] += 1
        else:
            agreement["requires_codex_review"] += 1
            source = next((row for row in rows.values() if row), {})
            disagreements.append(
                {
                    "candidate_id": candidate_id,
                    "source_dataset": source.get("source_dataset"),
                    "source_original_index": source.get("source_original_index"),
                    "source_question": source.get("source_question"),
                    "source_answer": source.get("source_answer"),
                    "model_results": {
                        model: {
                            "terminal_status": rows[model].get("terminal_status") if rows[model] else "missing",
                            "extraction_status": statuses[model],
                            "exclusion_reason": rows[model].get("exclusion_reason") if rows[model] else None,
                            "triple": {field: rows[model].get(field) for field in TRIPLE_FIELDS} if rows[model] else None,
                        }
                        for model in models
                    },
                }
            )
    return {
        "summary_version": "codex-dual-model-summary-v1",
        "models": models,
        "prompt_version": TRIPLE_EXTRACTION_PROMPT_VERSION,
        "candidate_count": len(candidate_ids),
        "agreement_counts": dict(agreement),
        "model_summaries": {
            model: extraction_summary(indexed[model].values(), model=model) for model in models
        },
        "disagreements": disagreements,
        "created_at": utc_now(),
    }
