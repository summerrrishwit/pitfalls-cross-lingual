#!/usr/bin/env python3
"""Build review-only artifacts from one model's factual-triple extraction.

This adapter is intentionally separate from the dual-model canonical pipeline.
Its outputs are provisional: every fact remains pending semantic review, no
``probe_relation_id`` is frozen, and no behavior or mechanism experiment is
declared ready.  The adapter is deterministic for fixed input and policy bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs" / "path_not_token_public_benchmark_relation_candidates_v2.json"
)

ADAPTER_VERSION = "single-model-provisional-adapter-v2"
PROVISIONAL_TRIPLE_SCHEMA_VERSION = "provisional-factual-triple-v1"
BASE_CLUSTER_SCHEMA_VERSION = "provisional-base-fact-cluster-v1"
RELATION_INVENTORY_VERSION = "provisional-relation-inventory-v1"
REVIEW_QUEUE_SCHEMA_VERSION = "provisional-semantic-review-v1"
REVIEW_SAMPLE_SCHEMA_VERSION = "provisional-semantic-review-sample-v1"
BEHAVIOR_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "provisional-base-fact-split-manifest-v1"
OVERLAP_AUDIT_SCHEMA_VERSION = "provisional-comparison-overlap-audit-v1"
SUMMARY_VERSION = "single-model-provisional-summary-v2"
FACT_KEY_VERSION = "normalized-subject-relation-answer-v1"
PROMPT_VERSION = "provisional-factual-prompt-v1"
DEFAULT_REVIEW_SAMPLE_SIZE = 400
DEFAULT_REVIEW_SAMPLE_SEED = "public-benchmark-provisional-review-v1"
DEFAULT_RANDOM_REVIEW_SAMPLE_SIZE = 400
DEFAULT_RANDOM_REVIEW_SAMPLE_SEED = "public-benchmark-provisional-random-review-v1"
DEFAULT_RISK_REVIEW_SAMPLE_SIZE = 200
DEFAULT_RISK_REVIEW_SAMPLE_SEED = "public-benchmark-provisional-prompt-risk-review-v1"
OVERLAP_EXAMPLE_LIMIT = 10
SPLIT_POLICY_VERSION = "normalized-answer-group-sha256-v1"
DEFAULT_SPLIT_SEED = "public-benchmark-provisional-split-v1"
SPLIT_RATIOS = (
    ("development", 0.60),
    ("validation", 0.20),
    ("sealed", 0.20),
)

MCQ_FALLBACK_STRUCTURE_RISK = "mcq_fallback_structure_risk"
MCQ_FALLBACK_EXPLICIT_CHOICE_REFERENCE = (
    "mcq_fallback_explicit_choice_reference"
)
MCQ_FALLBACK_BLANK = "mcq_fallback_blank"
MCQ_FALLBACK_NEGATIVE_OR_EXCEPTION = "mcq_fallback_negative_or_exception"
MCQ_FALLBACK_INCOMPLETE = "mcq_fallback_incomplete"
MCQ_FALLBACK_DETAIL_FLAGS = (
    MCQ_FALLBACK_EXPLICIT_CHOICE_REFERENCE,
    MCQ_FALLBACK_BLANK,
    MCQ_FALLBACK_NEGATIVE_OR_EXCEPTION,
    MCQ_FALLBACK_INCOMPLETE,
)

EXPLICIT_CHOICE_REFERENCE_RE = re.compile(
    r"\b(?:which|one) of the following\b"
    r"|\bfollowing (?:terms|statements|describes|is|person)\b"
    r"|\bbest described as\b"
    r"|\bchoose\b",
    flags=re.IGNORECASE,
)
BLANK_RE = re.compile(r"_{2,}|\bblank\b", flags=re.IGNORECASE)
NEGATIVE_OR_EXCEPTION_RE = re.compile(
    r"\b(?:not|except|incorrect|false)\b|\bleast likely\b",
    flags=re.IGNORECASE,
)
INCOMPLETE_RE = re.compile(
    r":\s*$"
    r"|\b(?:is|are|was|were|refers to|includes|called|has|contain|means)\s*$",
    flags=re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"Input JSONL is empty: {path}")
    return records


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(path)


def unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = normalize_text(text)
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def stable_id(prefix: str, parts: Sequence[Any], length: int = 20) -> str:
    payload = "\u241f".join(normalize_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def relation_signature_id(relation_raw: str, subject_type: str, answer_type: str) -> str:
    return stable_id("rel_sig", (relation_raw, subject_type, answer_type), length=16)


def base_fact_id(subject: str, relation_raw: str, answer: str) -> str:
    return stable_id("pbf", (subject, relation_raw, answer))


def validate_and_compile_policy(policy: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if policy.get("schema_version") not in {
        "probe-relation-candidate-policy-v1",
        "probe-relation-candidate-policy-v2",
    }:
        raise ValueError("Unsupported probe relation candidate policy schema")
    if policy.get("status") != "provisional" or policy.get("human_gold") is not False:
        raise ValueError("Probe relation candidate policy must remain provisional and non-human-gold")
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Probe relation candidate policy requires a non-empty rules list")

    compiled: List[Dict[str, Any]] = []
    candidate_ids = set()
    relation_owners: Dict[str, str] = {}
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"Policy rule {index} is not an object")
        candidate = str(rule.get("probe_relation_candidate") or "").strip()
        direction = str(rule.get("direction") or "").strip()
        raw_values = rule.get("relation_raw")
        if not candidate or candidate in candidate_ids:
            raise ValueError(f"Invalid or duplicate probe relation candidate: {candidate!r}")
        if not direction or not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"Incomplete probe relation candidate rule: {candidate}")
        if direction == "unresolved" and "direction_unresolved" not in candidate:
            raise ValueError(
                "Unresolved direction must be explicit in the candidate id: "
                f"{candidate}"
            )
        normalized_relations = {normalize_text(value) for value in raw_values if normalize_text(value)}
        if len(normalized_relations) != len(raw_values):
            raise ValueError(f"Duplicate or empty relation_raw alias in rule: {candidate}")
        for relation in normalized_relations:
            owner = relation_owners.get(relation)
            if owner is not None:
                raise ValueError(
                    f"relation_raw alias {relation!r} belongs to both {owner} and {candidate}"
                )
            relation_owners[relation] = candidate
        subject_types = rule.get("subject_types")
        answer_types = rule.get("answer_types")
        if subject_types is not None and not isinstance(subject_types, list):
            raise ValueError(f"subject_types must be a list or null: {candidate}")
        if answer_types is not None and not isinstance(answer_types, list):
            raise ValueError(f"answer_types must be a list or null: {candidate}")
        compiled.append(
            {
                "probe_relation_candidate": candidate,
                "direction": direction,
                "relation_raw": normalized_relations,
                "subject_types": (
                    {normalize_text(value) for value in subject_types} if subject_types is not None else None
                ),
                "answer_types": (
                    {normalize_text(value) for value in answer_types} if answer_types is not None else None
                ),
            }
        )
        candidate_ids.add(candidate)
    return compiled


def match_probe_relation_candidate(
    record: Mapping[str, Any], compiled_policy: Sequence[Mapping[str, Any]]
) -> Optional[Dict[str, str]]:
    relation = normalize_text(record.get("relation_raw"))
    subject_type = normalize_text(record.get("subject_type"))
    answer_type = normalize_text(record.get("answer_type"))
    matches = []
    for rule in compiled_policy:
        if relation not in rule["relation_raw"]:
            continue
        if rule["subject_types"] is not None and subject_type not in rule["subject_types"]:
            continue
        if rule["answer_types"] is not None and answer_type not in rule["answer_types"]:
            continue
        matches.append(rule)
    if len(matches) > 1:
        names = [str(item["probe_relation_candidate"]) for item in matches]
        raise ValueError(f"Record matches multiple probe relation candidate rules: {names}")
    if not matches:
        return None
    return {
        "probe_relation_candidate": str(matches[0]["probe_relation_candidate"]),
        "probe_relation_candidate_direction": str(matches[0]["direction"]),
    }


def filter_reason(record: Mapping[str, Any]) -> Optional[str]:
    if record.get("terminal_status") != "completed":
        return "terminal_status_not_completed"
    if record.get("extraction_status") != "extracted":
        return "extraction_status_not_extracted"
    validation_errors = record.get("validation_errors")
    if not isinstance(validation_errors, list):
        return "validation_errors_not_a_list"
    if validation_errors:
        return "validation_errors_present"
    return None


def validate_input_identity(records: Sequence[Mapping[str, Any]], expected_model: str) -> None:
    seen_candidates = set()
    observed_models = set()
    for index, record in enumerate(records, start=1):
        candidate_id = str(record.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError(f"Input row {index} is missing candidate_id")
        if candidate_id in seen_candidates:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_candidates.add(candidate_id)
        extraction = record.get("extraction")
        if not isinstance(extraction, dict):
            raise ValueError(f"Input row {candidate_id} is missing extraction provenance")
        model = str(extraction.get("model") or extraction.get("api_model") or "").strip()
        if not model:
            raise ValueError(f"Input row {candidate_id} is missing extraction model")
        observed_models.add(model)
    if observed_models != {expected_model}:
        raise ValueError(
            f"Expected exactly source model {expected_model!r}, observed {sorted(observed_models)}"
        )


def validate_eligible_record(record: Mapping[str, Any]) -> None:
    required_strings = (
        "candidate_id",
        "source_id",
        "source_dataset",
        "source_question",
        "source_answer",
        "subject",
        "subject_type",
        "relation_raw",
        "answer",
        "answer_type",
        "canonical_fact",
    )
    missing = [name for name in required_strings if not str(record.get(name) or "").strip()]
    if missing:
        raise ValueError(f"Eligible record {record.get('candidate_id')} has empty fields: {missing}")
    if not isinstance(record.get("source_choices"), list):
        raise ValueError(f"Eligible record {record.get('candidate_id')} has non-list source_choices")


def canonical_fact_completion(canonical_fact: str, answer: str) -> Optional[str]:
    fact = re.sub(r"[\s.!?]+$", "", canonical_fact.strip())
    match = re.search(rf"\b{re.escape(answer.strip())}$", fact, flags=re.IGNORECASE)
    if not match:
        return None
    prompt = fact[: match.start()].rstrip(" :,-")
    return prompt if prompt else None


def build_prompt(record: Mapping[str, Any]) -> Dict[str, str]:
    strict_prompt = canonical_fact_completion(str(record["canonical_fact"]), str(record["answer"]))
    if strict_prompt is not None:
        return {
            "prompt_en": strict_prompt,
            "prompt_quality_tier": "strict_factual_completion",
            "prompt_template_id": "canonical_fact_answer_suffix_en_provisional_v1",
        }
    return {
        "prompt_en": f"Factual question: {str(record['source_question']).strip()}\nAnswer:",
        "prompt_quality_tier": "open_answer_fallback",
        "prompt_template_id": "source_question_open_answer_en_provisional_v1",
    }


def source_format(record: Mapping[str, Any]) -> str:
    snapshot = record.get("source_snapshot")
    if isinstance(snapshot, dict):
        source_record = snapshot.get("record")
        if isinstance(source_record, dict):
            value = str(source_record.get("source_format") or "").strip()
            if value:
                return value
    return "multiple_choice" if record.get("source_choices") else "open_qa"


def prompt_risk_flags(
    record: Mapping[str, Any], prompt: Mapping[str, str]
) -> List[str]:
    """Return conservative, programmatic review hints for generated prompts.

    These flags are deliberately limited to MCQ records whose canonical fact
    could not provide an answer-suffix completion.  They are review-routing
    hints only: no flag rejects a fact, changes ``canonical_status``, or makes
    a semantic judgment about correctness.
    """

    if (
        source_format(record) != "multiple_choice"
        or prompt.get("prompt_quality_tier") != "open_answer_fallback"
    ):
        return []

    question = str(record.get("source_question") or "").strip()
    details: List[str] = []
    if EXPLICIT_CHOICE_REFERENCE_RE.search(question):
        details.append(MCQ_FALLBACK_EXPLICIT_CHOICE_REFERENCE)
    if BLANK_RE.search(question):
        details.append(MCQ_FALLBACK_BLANK)
    if NEGATIVE_OR_EXCEPTION_RE.search(question):
        details.append(MCQ_FALLBACK_NEGATIVE_OR_EXCEPTION)
    if INCOMPLETE_RE.search(question):
        details.append(MCQ_FALLBACK_INCOMPLETE)
    if not details:
        return []
    return [MCQ_FALLBACK_STRUCTURE_RISK, *details]


def source_answer_aliases(record: Mapping[str, Any]) -> List[str]:
    upstream_aliases: List[Any] = []
    snapshot = record.get("source_snapshot")
    if isinstance(snapshot, dict) and isinstance(snapshot.get("record"), dict):
        metadata = snapshot["record"].get("upstream_metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("answer_aliases"), list):
            upstream_aliases = metadata["answer_aliases"]
    return unique_strings([record.get("answer"), record.get("source_answer"), *upstream_aliases])


def record_provenance(
    record: Mapping[str, Any], input_path: Path, input_sha256: str
) -> Dict[str, Any]:
    extraction = record["extraction"]
    retry_history = record.get("retry_history")
    local_adjudication = record.get("local_adjudication")
    return {
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "input_record_sha256": sha256_value(record),
        "source_model": extraction.get("model"),
        "api_model": extraction.get("api_model"),
        "extraction_prompt_version": extraction.get("prompt_version"),
        "attempt_count": extraction.get("attempt_count"),
        "retry_history_present": isinstance(retry_history, list) and bool(retry_history),
        "retry_history_count": len(retry_history) if isinstance(retry_history, list) else 0,
        "local_adjudication_applied": isinstance(local_adjudication, dict),
        "local_adjudication_version": (
            local_adjudication.get("adjudication_version")
            if isinstance(local_adjudication, dict)
            else None
        ),
    }


def probe_relation_candidate_status(
    candidate: Optional[str], direction: Optional[str]
) -> str:
    if candidate is None:
        return "unassigned"
    if direction == "unresolved":
        return "pending_direction_semantic_and_balance_review"
    return "pending_semantic_and_balance_review"


def make_provisional_record(
    record: Mapping[str, Any],
    input_path: Path,
    input_sha256: str,
    policy_id: str,
    compiled_policy: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    validate_eligible_record(record)
    prompt = build_prompt(record)
    risk_flags = prompt_risk_flags(record, prompt)
    candidate = match_probe_relation_candidate(record, compiled_policy)
    relation_candidate = candidate["probe_relation_candidate"] if candidate else None
    relation_direction = candidate["probe_relation_candidate_direction"] if candidate else None
    fact_id = base_fact_id(record["subject"], record["relation_raw"], record["answer"])
    signature_id = relation_signature_id(
        str(record["relation_raw"]), str(record["subject_type"]), str(record["answer_type"])
    )
    return {
        "schema_version": PROVISIONAL_TRIPLE_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "provisional_id": f"provisional_{record['candidate_id']}",
        "base_fact_id": fact_id,
        "base_fact_key_version": FACT_KEY_VERSION,
        "candidate_id": record["candidate_id"],
        "source_id": record["source_id"],
        "source_dataset": record["source_dataset"],
        "source_subset": record.get("source_subset"),
        "source_path": record.get("source_path"),
        "source_original_index": record.get("source_original_index"),
        "source_snapshot": record.get("source_snapshot"),
        "source_format": source_format(record),
        "source_question": record["source_question"],
        "source_choices": list(record["source_choices"]),
        "source_answer": record["source_answer"],
        "subject": record["subject"],
        "subject_type": record["subject_type"],
        "relation_raw": record["relation_raw"],
        "answer": record["answer"],
        "answer_type": record["answer_type"],
        "answer_aliases_en": source_answer_aliases(record),
        "canonical_fact": record["canonical_fact"],
        "terminal_status": record["terminal_status"],
        "extraction_status": record["extraction_status"],
        "extraction_confidence": record.get("extraction_confidence"),
        "validation_errors": list(record["validation_errors"]),
        "created_at": record.get("created_at"),
        "extraction": record["extraction"],
        "parsed_response": record.get("parsed_response"),
        "retry_history": record.get("retry_history"),
        "local_adjudication": record.get("local_adjudication"),
        "relation_signature_id": signature_id,
        "relation_normalized": None,
        "relation_normalization_status": "pending_review",
        "probe_relation_candidate": relation_candidate,
        "probe_relation_candidate_direction": relation_direction,
        "probe_relation_candidate_policy": policy_id,
        "probe_relation_id": None,
        "probe_relation_status": probe_relation_candidate_status(
            relation_candidate, relation_direction
        ),
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "provisional_status": "provisional_single_model",
        "human_gold": False,
        "prompt_version": PROMPT_VERSION,
        **prompt,
        "prompt_risk_flags": risk_flags,
        "prompt_risk_assessment": "programmatic_review_hint_only",
        "prompt_ready": True,
        "provenance": record_provenance(record, input_path, input_sha256),
    }


def cluster_records(
    provisional_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in provisional_records:
        grouped[record["base_fact_id"]].append(record)

    clusters: List[Dict[str, Any]] = []
    representatives: Dict[str, Dict[str, Any]] = {}
    for fact_id, members in grouped.items():
        members = sorted(members, key=lambda item: str(item["candidate_id"]))
        representative = min(
            members,
            key=lambda item: (
                item["prompt_quality_tier"] != "strict_factual_completion",
                item["source_format"] != "multiple_choice",
                str(item["candidate_id"]),
            ),
        )
        representatives[fact_id] = representative
        candidate_values = {item["probe_relation_candidate"] for item in members}
        direction_values = {
            item.get("probe_relation_candidate_direction") for item in members
        }
        cluster_candidate = (
            next(iter(candidate_values))
            if len(candidate_values) == 1 and None not in candidate_values
            else None
        )
        cluster_direction = (
            next(iter(direction_values))
            if len(direction_values) == 1 and None not in direction_values
            else None
        )
        if len(candidate_values) > 1:
            cluster_status = "candidate_rule_conflict"
            cluster_candidate = None
            cluster_direction = None
        elif cluster_candidate is not None and len(direction_values) > 1:
            cluster_status = "candidate_direction_conflict"
            cluster_direction = None
        else:
            cluster_status = probe_relation_candidate_status(
                cluster_candidate, cluster_direction
            )
        member_prompt_risk_flags = sorted(
            {
                str(flag)
                for item in members
                for flag in item.get("prompt_risk_flags", [])
            }
        )
        clusters.append(
            {
                "schema_version": BASE_CLUSTER_SCHEMA_VERSION,
                "base_fact_id": fact_id,
                "base_fact_key_version": FACT_KEY_VERSION,
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "probe_relation_candidate": cluster_candidate,
                "probe_relation_candidate_direction": cluster_direction,
                "probe_relation_id": None,
                "probe_relation_status": cluster_status,
                "representative_candidate_id": representative["candidate_id"],
                "subject": representative["subject"],
                "relation_raw": representative["relation_raw"],
                "answer": representative["answer"],
                "member_count": len(members),
                "duplicate_status": "duplicate_cluster" if len(members) > 1 else "unique",
                "prompt_risk_flags": list(representative.get("prompt_risk_flags", [])),
                "member_prompt_risk_flags": member_prompt_risk_flags,
                "relation_signature_ids": sorted(
                    {item["relation_signature_id"] for item in members}
                ),
                "source_datasets": sorted({str(item["source_dataset"]) for item in members}),
                "members": [
                    {
                        "candidate_id": item["candidate_id"],
                        "source_id": item["source_id"],
                        "source_dataset": item["source_dataset"],
                        "input_record_sha256": item["provenance"]["input_record_sha256"],
                    }
                    for item in members
                ],
            }
        )
    clusters.sort(key=lambda item: str(item["base_fact_id"]))
    return clusters, representatives


def split_assignment_for_answer(answer_normalized: str, seed: str) -> Tuple[str, str]:
    assignment_hash = hashlib.sha256(
        f"{seed}\u241f{answer_normalized}".encode("utf-8")
    ).hexdigest()
    score = int(assignment_hash[:16], 16) / float(16**16)
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS:
        cumulative += ratio
        if score < cumulative:
            return name, assignment_hash
    return SPLIT_RATIOS[-1][0], assignment_hash


def build_provisional_split_manifest(
    clusters: Sequence[Mapping[str, Any]],
    representatives: Mapping[str, Mapping[str, Any]],
    input_sha256: str,
    seed: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    assignments: Dict[str, Dict[str, str]] = {}
    for cluster in clusters:
        fact_id = str(cluster["base_fact_id"])
        representative = representatives[fact_id]
        answer_normalized = normalize_text(representative["answer"])
        group_id = stable_id("split_group", (answer_normalized,))
        assignment, assignment_hash = split_assignment_for_answer(answer_normalized, seed)
        group = grouped.setdefault(
            group_id,
            {
                "split_group_id": group_id,
                "answer_key_sha256": hashlib.sha256(
                    answer_normalized.encode("utf-8")
                ).hexdigest(),
                "assignment_hash": assignment_hash,
                "split_assignment": assignment,
                "base_fact_ids": [],
            },
        )
        if group["answer_key_sha256"] != hashlib.sha256(
            answer_normalized.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"Split group hash collision: {group_id}")
        group["base_fact_ids"].append(fact_id)
        assignments[fact_id] = {
            "split_group_id": group_id,
            "split_assignment": assignment,
            "split_status": "provisional_not_frozen",
            "split_policy_version": SPLIT_POLICY_VERSION,
        }

    groups = []
    for group in grouped.values():
        group["base_fact_ids"].sort()
        group["base_fact_count"] = len(group["base_fact_ids"])
        groups.append(group)
    groups.sort(key=lambda item: str(item["split_group_id"]))
    group_counts = sorted_counter(str(item["split_assignment"]) for item in groups)
    base_counts = sorted_counter(
        str(assignments[str(cluster["base_fact_id"])]["split_assignment"])
        for cluster in clusters
    )
    return (
        {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "split_status": "provisional_not_frozen",
            "seed": seed,
            "grouping_key": "normalized_answer",
            "assignment_method": "sha256_threshold_v1",
            "ratios": {name: ratio for name, ratio in SPLIT_RATIOS},
            "input_sha256": input_sha256,
            "canonical_status": "pending_review",
            "human_gold": False,
            "split_group_count": len(groups),
            "base_fact_count": len(clusters),
            "split_group_counts": group_counts,
            "base_fact_counts": base_counts,
            "sealed_evaluation_status": "not_run",
            "groups": groups,
        },
        assignments,
    )


def overlap_key(record: Mapping[str, Any], fields: Sequence[str]) -> Optional[str]:
    parts = [normalize_text(record.get(field)) for field in fields]
    if any(not part for part in parts):
        return None
    return "\u241f".join(parts)


def build_comparison_overlap_audit(
    provisional_records: Sequence[Mapping[str, Any]],
    current_input_sha256: str,
    comparison_path: Path,
) -> Dict[str, Any]:
    comparison_path = comparison_path.resolve()
    comparison_records = read_jsonl(comparison_path)
    metric_fields = {
        "source_id": ("source_id",),
        "normalized_question": ("source_question",),
        "normalized_question_answer": ("source_question", "source_answer"),
        "normalized_subject_answer": ("subject", "answer"),
        "normalized_raw_triple": ("subject", "relation_raw", "answer"),
        "normalized_canonical_fact": ("canonical_fact",),
    }
    metrics: Dict[str, Dict[str, Any]] = {}
    exact_overlap_counts: Dict[str, int] = {}
    unique_key_overlap_counts: Dict[str, int] = {}
    for name, fields in metric_fields.items():
        current_keys = [overlap_key(record, fields) for record in provisional_records]
        comparison_keys = [overlap_key(record, fields) for record in comparison_records]
        current_valid = [key for key in current_keys if key is not None]
        comparison_valid = [key for key in comparison_keys if key is not None]
        current_unique = set(current_valid)
        comparison_unique = set(comparison_valid)
        current_overlap_count = sum(key in comparison_unique for key in current_valid)
        unique_overlap_count = len(current_unique & comparison_unique)
        comparison_by_key: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for comparison_record, key in zip(comparison_records, comparison_keys):
            if key is not None:
                comparison_by_key[key].append(comparison_record)
        for matches in comparison_by_key.values():
            matches.sort(
                key=lambda item: (
                    str(item.get("source_id") or ""),
                    str(item.get("candidate_id") or ""),
                )
            )
        matched_examples = []
        for current_record, key in sorted(
            zip(provisional_records, current_keys),
            key=lambda pair: str(pair[0].get("candidate_id") or ""),
        ):
            if key is None or key not in comparison_by_key:
                continue
            for comparison_record in comparison_by_key[key]:
                matched_examples.append(
                    {
                        "match_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                        "current_candidate_id": current_record.get("candidate_id"),
                        "current_source_id": current_record.get("source_id"),
                        "current_base_fact_id": current_record.get("base_fact_id"),
                        "comparison_candidate_id": comparison_record.get("candidate_id"),
                        "comparison_source_id": comparison_record.get("source_id"),
                    }
                )
                if len(matched_examples) >= OVERLAP_EXAMPLE_LIMIT:
                    break
            if len(matched_examples) >= OVERLAP_EXAMPLE_LIMIT:
                break
        exact_overlap_counts[name] = current_overlap_count
        unique_key_overlap_counts[name] = unique_overlap_count
        metrics[name] = {
            "fields": list(fields),
            "normalization": "NFKC_casefold_trim_collapse_whitespace",
            "current_valid_record_count": len(current_valid),
            "comparison_valid_record_count": len(comparison_valid),
            "current_unique_key_count": len(current_unique),
            "comparison_unique_key_count": len(comparison_unique),
            "current_record_overlap_count": current_overlap_count,
            "unique_key_overlap_count": unique_overlap_count,
            "matched_example_limit": OVERLAP_EXAMPLE_LIMIT,
            "matched_examples": matched_examples,
        }
    return {
        "schema_version": OVERLAP_AUDIT_SCHEMA_VERSION,
        "audit_status": "completed_exact_normalized_match_only",
        "audit_scope": "eligible_provisional_records_vs_comparison_canonical_records",
        "current_input_sha256": current_input_sha256,
        "current_record_count": len(provisional_records),
        "comparison": {
            "path": str(comparison_path),
            "sha256": sha256_file(comparison_path),
            "record_count": len(comparison_records),
        },
        "exact_overlap_counts": exact_overlap_counts,
        "unique_key_overlap_counts": unique_key_overlap_counts,
        "metrics": metrics,
        "limitations": {
            "exact_normalized_string_audit_only": True,
            "semantic_near_duplicate_clustering": "not_performed",
            "paraphrase_detection": "not_performed",
        },
    }


def build_relation_inventory(
    provisional_records: Sequence[Mapping[str, Any]],
    input_sha256: str,
    policy: Mapping[str, Any],
    policy_sha256: str,
    max_examples: int,
) -> Dict[str, Any]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in provisional_records:
        groups[str(record["relation_signature_id"])].append(record)

    entries = []
    for signature_id, records in groups.items():
        records = sorted(records, key=lambda item: str(item["candidate_id"]))
        first = records[0]
        candidate_values = {item["probe_relation_candidate"] for item in records}
        direction_values = {
            item.get("probe_relation_candidate_direction") for item in records
        }
        relation_candidate = (
            next(iter(candidate_values))
            if len(candidate_values) == 1 and None not in candidate_values
            else None
        )
        relation_candidate_direction = (
            next(iter(direction_values))
            if len(direction_values) == 1 and None not in direction_values
            else None
        )
        if len(candidate_values) > 1:
            relation_candidate_status = "candidate_rule_conflict"
            relation_candidate_direction = None
        elif relation_candidate is not None and len(direction_values) > 1:
            relation_candidate_status = "candidate_direction_conflict"
            relation_candidate_direction = None
        else:
            relation_candidate_status = probe_relation_candidate_status(
                relation_candidate, relation_candidate_direction
            )
        entries.append(
            {
                "signature_id": signature_id,
                "relation_raw": first["relation_raw"],
                "relation_raw_normalized": normalize_text(first["relation_raw"]),
                "subject_type": first["subject_type"],
                "answer_type": first["answer_type"],
                "direction": f"{first['subject_type']} -> {first['answer_type']}",
                "record_count": len(records),
                "base_fact_count": len({item["base_fact_id"] for item in records}),
                "strict_completion_count": sum(
                    item["prompt_quality_tier"] == "strict_factual_completion"
                    for item in records
                ),
                "source_dataset_counts": dict(
                    sorted(Counter(str(item["source_dataset"]) for item in records).items())
                ),
                "canonical_status": "pending_review",
                "human_gold": False,
                "probe_relation_candidate": relation_candidate,
                "probe_relation_candidate_direction": relation_candidate_direction,
                "probe_relation_id": None,
                "probe_relation_status": relation_candidate_status,
                "examples": [
                    {
                        "base_fact_id": item["base_fact_id"],
                        "candidate_id": item["candidate_id"],
                        "source_id": item["source_id"],
                        "source_dataset": item["source_dataset"],
                        "subject": item["subject"],
                        "answer": item["answer"],
                        "canonical_fact": item["canonical_fact"],
                    }
                    for item in records[:max_examples]
                ],
            }
        )
    entries.sort(key=lambda item: (-int(item["record_count"]), str(item["signature_id"])))
    return {
        "schema_version": RELATION_INVENTORY_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "inventory_status": "provisional_single_model",
        "canonical_status": "pending_review",
        "human_gold": False,
        "input_sha256": input_sha256,
        "probe_relation_candidate_policy": policy["policy_id"],
        "probe_relation_candidate_policy_sha256": policy_sha256,
        "probe_relation_id_status": "not_frozen",
        "record_count": len(provisional_records),
        "base_fact_count": len({item["base_fact_id"] for item in provisional_records}),
        "signature_count": len(entries),
        "max_examples_per_signature": max_examples,
        "entries": entries,
    }


def review_reasons(cluster: Mapping[str, Any], representative: Mapping[str, Any]) -> List[str]:
    reasons = ["single_model_extraction_requires_independent_semantic_review"]
    if int(cluster["member_count"]) > 1:
        reasons.append("duplicate_source_records_clustered")
    if representative["prompt_quality_tier"] == "open_answer_fallback":
        reasons.append("canonical_fact_not_answer_suffix")
    if len(str(representative["relation_raw"]).split()) > 4:
        reasons.append("relation_phrase_long")
    if len(re.findall(r"[\w'-]+", str(representative["answer"]), flags=re.UNICODE)) > 12:
        reasons.append("answer_phrase_long")
    subject = normalize_text(representative["subject"])
    answer = normalize_text(representative["answer"])
    if len(answer) >= 4 and re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", subject):
        reasons.append("answer_phrase_occurs_in_subject")
    if cluster["probe_relation_candidate"] is not None:
        reasons.append("probe_relation_candidate_requires_semantic_and_balance_review")
    if cluster.get("probe_relation_candidate_direction") == "unresolved":
        reasons.append("probe_relation_candidate_direction_unresolved")
    if representative.get("prompt_risk_flags"):
        reasons.append("mcq_fallback_structure_risk_requires_review")
    return reasons


def build_review_queue(
    clusters: Sequence[Mapping[str, Any]],
    representatives: Mapping[str, Mapping[str, Any]],
    policy_id: str,
    split_assignments: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    queue = []
    for cluster in clusters:
        representative = representatives[str(cluster["base_fact_id"])]
        split = split_assignments[str(cluster["base_fact_id"])]
        queue.append(
            {
                "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
                "review_id": f"review_{cluster['base_fact_id']}",
                "base_fact_id": cluster["base_fact_id"],
                "canonical_status": "pending_review",
                "review_case": "provisional_single_model",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "source_model": representative["provenance"]["source_model"],
                "source_datasets": cluster["source_datasets"],
                "source_format": representative["source_format"],
                "prompt_quality_tier": representative["prompt_quality_tier"],
                "prompt_risk_flags": list(representative.get("prompt_risk_flags", [])),
                "prompt_risk_assessment": representative["prompt_risk_assessment"],
                "probe_relation_candidate": cluster["probe_relation_candidate"],
                "probe_relation_candidate_direction": cluster[
                    "probe_relation_candidate_direction"
                ],
                "probe_relation_candidate_policy": policy_id,
                "probe_relation_id": None,
                "probe_relation_status": cluster["probe_relation_status"],
                "split_group_id": split["split_group_id"],
                "split_assignment": split["split_assignment"],
                "split_status": split["split_status"],
                "split_policy_version": split["split_policy_version"],
                "review_reasons": review_reasons(cluster, representative),
                "member_candidate_ids": [item["candidate_id"] for item in cluster["members"]],
                "subject": representative["subject"],
                "subject_type": representative["subject_type"],
                "relation_raw": representative["relation_raw"],
                "answer": representative["answer"],
                "answer_type": representative["answer_type"],
                "canonical_fact": representative["canonical_fact"],
                "source_question": representative["source_question"],
                "requested_decision": "accept_reject_or_revise",
                "review_decision": None,
                "reviewer": None,
            }
        )
    queue.sort(key=lambda item: str(item["base_fact_id"]))
    return queue


def deterministic_review_sample(
    review_queue: Sequence[Mapping[str, Any]], target_size: int, seed: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if target_size < 0:
        raise ValueError("review sample size cannot be negative")
    buckets: Dict[Tuple[str, str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for item in review_queue:
        stratum = (
            "+".join(str(value) for value in item["source_datasets"]),
            str(item["source_format"]),
            str(item["prompt_quality_tier"]),
            str(item["probe_relation_candidate"] or "unmapped"),
            str(item["split_assignment"]),
        )
        buckets[stratum].append(item)
    separator = "\u241f"
    for stratum, items in buckets.items():
        items.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}{separator}{separator.join(stratum)}{separator}{item['base_fact_id']}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
    strata = sorted(
        buckets,
        key=lambda key: hashlib.sha256(
            f"{seed}{separator}{separator.join(key)}".encode("utf-8")
        ).hexdigest(),
    )
    target = min(target_size, len(review_queue))
    positions = {stratum: 0 for stratum in strata}
    selected: List[Dict[str, Any]] = []
    while len(selected) < target:
        progressed = False
        for stratum in strata:
            position = positions[stratum]
            if position >= len(buckets[stratum]):
                continue
            source = buckets[stratum][position]
            positions[stratum] += 1
            selected.append(
                {
                    **source,
                    "schema_version": REVIEW_SAMPLE_SCHEMA_VERSION,
                    "sampling": {
                        "method": "deterministic_balanced_round_robin_v1",
                        "seed": seed,
                        "sample_status": "selected_pending_review",
                        "population_estimation_supported": False,
                        "stratum": {
                            "source_datasets": list(source["source_datasets"]),
                            "source_format": source["source_format"],
                            "prompt_quality_tier": source["prompt_quality_tier"],
                            "probe_relation_candidate": source["probe_relation_candidate"],
                            "split_assignment": source["split_assignment"],
                        },
                        "sample_index": len(selected),
                    },
                }
            )
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    metadata = {
        "method": "deterministic_balanced_round_robin_v1",
        "seed": seed,
        "requested_size": target_size,
        "actual_size": len(selected),
        "stratum_count": len(strata),
        "sampling_weights_provided": False,
        "intended_use": "coverage_oriented_semantic_review_not_population_estimation",
        "population_estimation_supported": False,
        "stratification_fields": [
            "source_datasets",
            "source_format",
            "prompt_quality_tier",
            "probe_relation_candidate",
            "split_assignment",
        ],
    }
    return selected, metadata


def deterministic_random_review_sample(
    review_queue: Sequence[Mapping[str, Any]], target_size: int, seed: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select an equal-probability, reproducible sample from the full queue."""

    if target_size < 0:
        raise ValueError("random review sample size cannot be negative")
    separator = "\u241f"
    ranked = sorted(
        review_queue,
        key=lambda item: (
            hashlib.sha256(
                f"{seed}{separator}{item['base_fact_id']}".encode("utf-8")
            ).hexdigest(),
            str(item["base_fact_id"]),
        ),
    )
    target = min(target_size, len(ranked))
    inclusion_probability = target / len(ranked) if ranked else 0.0
    analysis_weight = (1.0 / inclusion_probability) if inclusion_probability else None
    selected = [
        {
            **source,
            "schema_version": REVIEW_SAMPLE_SCHEMA_VERSION,
            "sampling": {
                "method": "deterministic_uniform_hash_rank_without_replacement_v1",
                "seed": seed,
                "sample_status": "selected_pending_review",
                "sample_index": index,
                "selection_frame": "full_review_queue",
                "inclusion_probability": inclusion_probability,
                "analysis_weight": analysis_weight,
                "population_estimation_supported": True,
            },
        }
        for index, source in enumerate(ranked[:target])
    ]
    metadata = {
        "method": "deterministic_uniform_hash_rank_without_replacement_v1",
        "seed": seed,
        "requested_size": target_size,
        "actual_size": len(selected),
        "population_size": len(review_queue),
        "selection_frame": "full_review_queue",
        "inclusion_probability": inclusion_probability,
        "sampling_weights_provided": bool(selected),
        "intended_use": "overall_review_outcome_population_estimation",
        "population_estimation_supported": bool(selected),
        "population_estimation_conditions": [
            "seed_precommitted_before_review_outcomes",
            "all_selected_items_receive_the_same_review_protocol",
            "nonresponse_is_reported_and_handled",
        ],
    }
    return selected, metadata


def deterministic_prompt_risk_review_sample(
    review_queue: Sequence[Mapping[str, Any]], target_size: int, seed: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select a reproducible, coverage-oriented sample of flagged prompts."""

    if target_size < 0:
        raise ValueError("risk review sample size cannot be negative")
    separator = "\u241f"
    risk_frame = [
        item
        for item in review_queue
        if any(flag in MCQ_FALLBACK_DETAIL_FLAGS for flag in item["prompt_risk_flags"])
    ]
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in risk_frame:
        for flag in item["prompt_risk_flags"]:
            if flag in MCQ_FALLBACK_DETAIL_FLAGS:
                buckets[str(flag)].append(item)
    for flag, items in buckets.items():
        items.sort(
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}{separator}{flag}{separator}{item['base_fact_id']}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                str(item["base_fact_id"]),
            )
        )
    flags = sorted(
        buckets,
        key=lambda flag: hashlib.sha256(
            f"{seed}{separator}{flag}".encode("utf-8")
        ).hexdigest(),
    )
    target = min(target_size, len(risk_frame))
    positions = {flag: 0 for flag in flags}
    selected_sources: List[Mapping[str, Any]] = []
    selected_ids = set()
    while len(selected_sources) < target:
        progressed = False
        for flag in flags:
            items = buckets[flag]
            while positions[flag] < len(items):
                source = items[positions[flag]]
                positions[flag] += 1
                fact_id = str(source["base_fact_id"])
                if fact_id in selected_ids:
                    continue
                selected_sources.append(source)
                selected_ids.add(fact_id)
                progressed = True
                break
            if len(selected_sources) >= target:
                break
        if not progressed:
            break
    selected = [
        {
            **source,
            "schema_version": REVIEW_SAMPLE_SCHEMA_VERSION,
            "sampling": {
                "method": "deterministic_prompt_risk_round_robin_v1",
                "seed": seed,
                "sample_status": "selected_pending_review",
                "sample_index": index,
                "selection_frame": "prompt_risk_flagged_review_queue",
                "matched_prompt_risk_flags": [
                    flag
                    for flag in source["prompt_risk_flags"]
                    if flag in MCQ_FALLBACK_DETAIL_FLAGS
                ],
                "population_estimation_supported": False,
            },
        }
        for index, source in enumerate(selected_sources)
    ]
    metadata = {
        "method": "deterministic_prompt_risk_round_robin_v1",
        "seed": seed,
        "requested_size": target_size,
        "actual_size": len(selected),
        "population_size": len(review_queue),
        "risk_frame_size": len(risk_frame),
        "risk_flag_counts": sorted_counter(
            flag
            for item in risk_frame
            for flag in item["prompt_risk_flags"]
            if flag in MCQ_FALLBACK_DETAIL_FLAGS
        ),
        "selection_frame": "prompt_risk_flagged_review_queue",
        "sampling_weights_provided": False,
        "intended_use": "targeted_prompt_structure_risk_review_not_population_estimation",
        "population_estimation_supported": False,
    }
    return selected, metadata


def distractor_candidates(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    aliases = {normalize_text(value) for value in record["answer_aliases_en"]}
    seen = set()
    output = []
    for index, choice in enumerate(record["source_choices"]):
        text = str(choice or "").strip()
        normalized = normalize_text(text)
        if not text or normalized in aliases or normalized in seen:
            continue
        seen.add(normalized)
        output.append(
            {
                "distractor_id": stable_id(
                    "dist", (record["base_fact_id"], str(index), normalized), length=16
                ),
                "source_choice_index": index,
                "text_en": text,
                "answer_en": text,
                "verification_status": "pending_review",
                "verified": False,
            }
        )
    return output


def build_behavior_bundle(
    clusters: Sequence[Mapping[str, Any]],
    representatives: Mapping[str, Mapping[str, Any]],
    policy_id: str,
    source_pool_id: str,
    split_assignments: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    output = []
    for cluster in clusters:
        representative = representatives[str(cluster["base_fact_id"])]
        split = split_assignments[str(cluster["base_fact_id"])]
        distractors = distractor_candidates(representative)
        blocking_reasons = ["canonical_pending_review", "target_translation_missing"]
        if not distractors:
            blocking_reasons.append("verified_distractor_missing")
        else:
            blocking_reasons.append("distractors_pending_review")
        if cluster["probe_relation_candidate"] is None:
            blocking_reasons.append("probe_relation_candidate_unassigned")
        else:
            blocking_reasons.append("probe_relation_candidate_not_frozen")
        if cluster.get("probe_relation_candidate_direction") == "unresolved":
            blocking_reasons.append("probe_relation_candidate_direction_unresolved")
        output.append(
            {
                "schema_version": BEHAVIOR_BUNDLE_SCHEMA_VERSION,
                "bundle_status": "draft_pending_review",
                "base_fact_id": cluster["base_fact_id"],
                "base_id": cluster["base_fact_id"],
                "candidate_id": representative["candidate_id"],
                "source_id": representative["source_id"],
                "source_pool_id": source_pool_id,
                "source_dataset": representative["source_dataset"],
                "source_subset": representative["source_subset"],
                "source_format": representative["source_format"],
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "subject_en": representative["subject"],
                "relation_raw": representative["relation_raw"],
                "relation_normalized": None,
                "normalization_status": "pending_review",
                "relation_signature_id": representative["relation_signature_id"],
                "probe_relation_candidate": cluster["probe_relation_candidate"],
                "probe_relation_candidate_direction": cluster[
                    "probe_relation_candidate_direction"
                ],
                "probe_relation_candidate_policy": policy_id,
                "probe_relation_id": None,
                "probe_relation_status": cluster["probe_relation_status"],
                "split_group_id": split["split_group_id"],
                "split_assignment": split["split_assignment"],
                "split_status": split["split_status"],
                "split_policy_version": split["split_policy_version"],
                "sealed_evaluation_status": "not_run",
                "answer_en": representative["answer"],
                "answer_type": representative["answer_type"],
                "answer_aliases_en": representative["answer_aliases_en"],
                "canonical_fact": representative["canonical_fact"],
                "canonical_fact_en": representative["canonical_fact"],
                "source_question_en": representative["source_question"],
                "source_choices_en": representative["source_choices"],
                "prompt_en": representative["prompt_en"],
                "prompt_version": representative["prompt_version"],
                "prompt_template_id": representative["prompt_template_id"],
                "prompt_quality_tier": representative["prompt_quality_tier"],
                "prompt_risk_flags": list(representative.get("prompt_risk_flags", [])),
                "prompt_risk_assessment": representative["prompt_risk_assessment"],
                "factual_prompt_status": "generated",
                "prompt_ready": True,
                "target_language": None,
                "subject_target": None,
                "relation_target": None,
                "answer_target": None,
                "prompt_target": None,
                "translation_status": "not_run",
                "distractor_candidates": distractors,
                "admission": {
                    "prompt_ready": True,
                    "semantic_review_complete": False,
                    "behavior_screening_ready": False,
                    "hf_bridge_ready": False,
                    "relation_conditioned_mechanism_eligible": False,
                    "path_not_token_experiment_ready": False,
                    "blocking_reasons": blocking_reasons,
                },
                "experiment_status": {
                    "ollama_behavior_screening": "not_run",
                    "hf_exact_checkpoint_reproduction": "pending_gpu",
                    "answer_tokenization": "not_run",
                    "hidden_state_collection": "not_run",
                    "activation_patching": "not_run",
                    "vector_intervention": "not_run",
                    "repair_validation": "not_run",
                },
                "provenance": representative["provenance"],
            }
        )
    output.sort(key=lambda item: str(item["base_id"]))
    return output


def sorted_counter(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(values).items()))


def prepare(
    input_path: Path,
    output_dir: Path,
    policy_path: Path = DEFAULT_POLICY,
    expected_model: str = "qwen3.7-plus",
    source_pool_id: str = "public-benchmarks-full-v1",
    max_examples: int = 5,
    review_sample_size: int = DEFAULT_REVIEW_SAMPLE_SIZE,
    review_sample_seed: str = DEFAULT_REVIEW_SAMPLE_SEED,
    random_review_sample_size: int = DEFAULT_RANDOM_REVIEW_SAMPLE_SIZE,
    random_review_sample_seed: str = DEFAULT_RANDOM_REVIEW_SAMPLE_SEED,
    risk_review_sample_size: int = DEFAULT_RISK_REVIEW_SAMPLE_SIZE,
    risk_review_sample_seed: str = DEFAULT_RISK_REVIEW_SAMPLE_SEED,
    split_seed: str = DEFAULT_SPLIT_SEED,
    comparison_canonical_path: Optional[Path] = None,
) -> Dict[str, Any]:
    input_path = input_path.resolve()
    policy_path = policy_path.resolve()
    output_dir = output_dir.resolve()
    records = read_jsonl(input_path)
    validate_input_identity(records, expected_model)
    policy = read_json(policy_path)
    compiled_policy = validate_and_compile_policy(policy)
    input_sha256 = sha256_file(input_path)
    policy_sha256 = sha256_file(policy_path)

    filter_counts: Counter[str] = Counter()
    eligible = []
    for record in records:
        reason = filter_reason(record)
        if reason is not None:
            filter_counts[reason] += 1
            continue
        eligible.append(record)
    if not eligible:
        raise ValueError("No completed, extracted, validation-clean records remain after filtering")

    provisional = [
        make_provisional_record(
            record,
            input_path,
            input_sha256,
            str(policy["policy_id"]),
            compiled_policy,
        )
        for record in eligible
    ]
    provisional.sort(key=lambda item: str(item["candidate_id"]))
    clusters, representatives = cluster_records(provisional)
    split_manifest, split_assignments = build_provisional_split_manifest(
        clusters, representatives, input_sha256, split_seed
    )
    for cluster in clusters:
        cluster.update(split_assignments[str(cluster["base_fact_id"])])
    relation_inventory = build_relation_inventory(
        provisional,
        input_sha256,
        policy,
        policy_sha256,
        max_examples,
    )
    review_queue = build_review_queue(
        clusters,
        representatives,
        str(policy["policy_id"]),
        split_assignments,
    )
    review_sample, review_sample_metadata = deterministic_review_sample(
        review_queue, review_sample_size, review_sample_seed
    )
    random_review_sample, random_review_sample_metadata = (
        deterministic_random_review_sample(
            review_queue, random_review_sample_size, random_review_sample_seed
        )
    )
    risk_review_sample, risk_review_sample_metadata = (
        deterministic_prompt_risk_review_sample(
            review_queue, risk_review_sample_size, risk_review_sample_seed
        )
    )
    behavior_bundle = build_behavior_bundle(
        clusters,
        representatives,
        str(policy["policy_id"]),
        source_pool_id,
        split_assignments,
    )
    overlap_audit = (
        build_comparison_overlap_audit(
            provisional,
            input_sha256,
            comparison_canonical_path,
        )
        if comparison_canonical_path is not None
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_values: List[Tuple[str, Any, bool]] = [
        ("provisional_triples.jsonl", provisional, True),
        ("base_fact_clusters.jsonl", clusters, True),
        ("relation_inventory.json", relation_inventory, False),
        ("review_queue.jsonl", review_queue, True),
        ("review_sample.jsonl", review_sample, True),
        ("review_sample_overall_random.jsonl", random_review_sample, True),
        ("review_sample_prompt_risk.jsonl", risk_review_sample, True),
        ("behavior_input_bundle.jsonl", behavior_bundle, True),
        ("split_manifest.json", split_manifest, False),
    ]
    if overlap_audit is not None:
        artifact_values.append(("overlap_audit.json", overlap_audit, False))
    for name, value, is_jsonl in artifact_values:
        path = output_dir / name
        if is_jsonl:
            write_jsonl(path, value)
        else:
            write_json(path, value)

    candidate_record_counts = sorted_counter(
        str(item["probe_relation_candidate"] or "unmapped") for item in provisional
    )
    candidate_base_counts = sorted_counter(
        str(item["probe_relation_candidate"] or "unmapped") for item in clusters
    )
    duplicate_clusters = [item for item in clusters if int(item["member_count"]) > 1]
    summary: Dict[str, Any] = {
        "schema_version": SUMMARY_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "source_pool_id": source_pool_id,
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "byte_count": input_path.stat().st_size,
            "record_count": len(records),
            "expected_source_model": expected_model,
            "observed_extraction_prompt_versions": sorted(
                {
                    str(item["extraction"].get("prompt_version"))
                    for item in records
                    if item["extraction"].get("prompt_version") is not None
                }
            ),
        },
        "policy": {
            "path": str(policy_path),
            "sha256": policy_sha256,
            "policy_id": policy["policy_id"],
            "schema_version": policy["schema_version"],
            "rule_count": len(compiled_policy),
            "status": "provisional",
        },
        "selection": {
            "required_terminal_status": "completed",
            "required_extraction_status": "extracted",
            "requires_empty_validation_errors": True,
            "eligible_record_count": len(provisional),
            "filtered_record_count": len(records) - len(provisional),
            "filter_reason_counts": dict(sorted(filter_counts.items())),
        },
        "counts": {
            "provisional_triples": len(provisional),
            "base_fact_clusters": len(clusters),
            "duplicate_base_fact_clusters": len(duplicate_clusters),
            "duplicate_record_excess": sum(int(item["member_count"]) - 1 for item in duplicate_clusters),
            "relation_signatures": relation_inventory["signature_count"],
            "review_queue": len(review_queue),
            "review_sample": len(review_sample),
            "review_sample_overall_random": len(random_review_sample),
            "review_sample_prompt_risk": len(risk_review_sample),
            "behavior_input_bundle": len(behavior_bundle),
        },
        "eligible_record_distributions": {
            "source_dataset": sorted_counter(str(item["source_dataset"]) for item in provisional),
            "source_format": sorted_counter(str(item["source_format"]) for item in provisional),
            "prompt_quality_tier": sorted_counter(
                str(item["prompt_quality_tier"]) for item in provisional
            ),
            "probe_relation_candidate": candidate_record_counts,
        },
        "base_fact_distributions": {
            "prompt_quality_tier": sorted_counter(
                str(item["prompt_quality_tier"]) for item in behavior_bundle
            ),
            "probe_relation_candidate": candidate_base_counts,
            "prompt_risk_flags": sorted_counter(
                str(flag)
                for item in review_queue
                for flag in item["prompt_risk_flags"]
            ),
        },
        "review_sampling": {
            "coverage_oriented_sample": {
                "artifact": "review_sample.jsonl",
                **review_sample_metadata,
            },
            "overall_random_sample": {
                "artifact": "review_sample_overall_random.jsonl",
                **random_review_sample_metadata,
            },
            "prompt_risk_targeted_sample": {
                "artifact": "review_sample_prompt_risk.jsonl",
                **risk_review_sample_metadata,
            },
            "population_estimation_guidance": {
                "supported_sample": "overall_random_sample",
                "not_supported_samples": [
                    "coverage_oriented_sample",
                    "prompt_risk_targeted_sample",
                ],
            },
        },
        "provisional_split": {
            "manifest": "split_manifest.json",
            "split_policy_version": SPLIT_POLICY_VERSION,
            "split_status": "provisional_not_frozen",
            "seed": split_seed,
            "ratios": {name: ratio for name, ratio in SPLIT_RATIOS},
            "grouping_key": "normalized_answer",
            "split_group_count": split_manifest["split_group_count"],
            "split_group_counts": split_manifest["split_group_counts"],
            "base_fact_counts": split_manifest["base_fact_counts"],
            "sealed_evaluation_status": "not_run",
        },
        "comparison_overlap_audit": (
            {
                "status": overlap_audit["audit_status"],
                "artifact": "overlap_audit.json",
                "comparison_path": overlap_audit["comparison"]["path"],
                "comparison_sha256": overlap_audit["comparison"]["sha256"],
                "exact_overlap_counts": overlap_audit["exact_overlap_counts"],
                "semantic_near_duplicate_clustering": "not_performed",
            }
            if overlap_audit is not None
            else {
                "status": "not_run_comparison_not_provided",
                "artifact": None,
                "semantic_near_duplicate_clustering": "not_performed",
            }
        ),
        "status_contract": {
            "canonical_status": "pending_review",
            "evidence_tier": "provisional_single_model",
            "provisional_status": "provisional_single_model",
            "human_gold": False,
            "probe_relation_id": None,
            "probe_relation_status_for_matched_candidates": "pending_semantic_and_balance_review",
            "unresolved_candidate_direction_status": (
                "pending_direction_semantic_and_balance_review"
            ),
            "prompt_is_not_gated_by_probe_relation_candidate": True,
            "prompt_risk_flags_are_review_hints_only": True,
            "behavior_bundle_status": "draft_pending_review",
            "split_status": "provisional_not_frozen",
            "sealed_evaluation_status": "not_run",
        },
        "not_performed": [
            "independent_semantic_review",
            "semantic_near_duplicate_clustering",
            "canonical_freeze",
            "probe_relation_freeze",
            "split_freeze",
            "target_translation",
            "distractor_validation",
            "behavior_screening",
            "hf_exact_checkpoint_reproduction",
            "answer_tokenization",
            "hidden_state_collection",
            "activation_patching",
            "vector_intervention",
            "repair_validation",
            "sealed_evaluation",
        ],
        "artifacts": {},
    }
    for name, _, _ in artifact_values:
        summary["artifacts"][name] = {
            "sha256": sha256_file(output_dir / name),
        }
    write_json(output_dir / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, help="Single-model triple_extractions.jsonl")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--policy", default=str(DEFAULT_POLICY))
    result.add_argument("--expected-model", default="qwen3.7-plus")
    result.add_argument("--source-pool-id", default="public-benchmarks-full-v1")
    result.add_argument("--max-examples", type=int, default=5)
    result.add_argument("--review-sample-size", type=int, default=DEFAULT_REVIEW_SAMPLE_SIZE)
    result.add_argument("--review-sample-seed", default=DEFAULT_REVIEW_SAMPLE_SEED)
    result.add_argument(
        "--random-review-sample-size",
        type=int,
        default=DEFAULT_RANDOM_REVIEW_SAMPLE_SIZE,
    )
    result.add_argument(
        "--random-review-sample-seed",
        default=DEFAULT_RANDOM_REVIEW_SAMPLE_SEED,
    )
    result.add_argument(
        "--risk-review-sample-size",
        type=int,
        default=DEFAULT_RISK_REVIEW_SAMPLE_SIZE,
    )
    result.add_argument(
        "--risk-review-sample-seed",
        default=DEFAULT_RISK_REVIEW_SAMPLE_SEED,
    )
    result.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    result.add_argument(
        "--comparison-canonical",
        help="Optional prior canonical_triples.jsonl for normalized exact-overlap audit",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    summary = prepare(
        Path(args.input),
        Path(args.output_dir),
        Path(args.policy),
        expected_model=args.expected_model,
        source_pool_id=args.source_pool_id,
        max_examples=args.max_examples,
        review_sample_size=args.review_sample_size,
        review_sample_seed=args.review_sample_seed,
        random_review_sample_size=args.random_review_sample_size,
        random_review_sample_seed=args.random_review_sample_seed,
        risk_review_sample_size=args.risk_review_sample_size,
        risk_review_sample_seed=args.risk_review_sample_seed,
        split_seed=args.split_seed,
        comparison_canonical_path=(
            Path(args.comparison_canonical) if args.comparison_canonical else None
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
