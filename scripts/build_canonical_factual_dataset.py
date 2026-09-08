#!/usr/bin/env python3
"""Build the reviewed canonical factual-triple dataset and downstream candidates.

The script never mutates model extraction checkpoints.  It consumes the two
model JSONL files, writes direct dual-model agreements, routes single-model
extractions and dual-model answer conflicts through an explicit Codex
adjudication artifact, builds the global relation inventory, applies a
Codex-curated taxonomy, generates factual completion prompts, and constructs
source-option distractor candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CANONICAL_POLICY_VERSION = "codex-adjudicated-canonical-v2"
CANONICAL_REVIEW_PROMPT_VERSION = "canonical-triple-adjudication-v1"
TAXONOMY_VERSION = "relation-taxonomy-v1"
MAPPING_VERSION = "relation-mapping-v1"
PROMPT_VERSION = "canonical-fact-factual-prompt-v1"
DISTRACTOR_VERSION = "source-wrong-option-v1"
PREFERRED_MODEL = "sensenova-6.7-flash-lite"
SECONDARY_MODEL = "qwen3.7-plus"


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def relation_signature_id(relation_raw: str, subject_type: str, answer_type: str) -> str:
    payload = "\u241f".join(
        (normalize_text(relation_raw), normalize_text(subject_type), normalize_text(answer_type))
    )
    return f"rel_sig_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def build_relation_inventory(
    records: Iterable[Dict[str, Any]], max_examples: int = 5
) -> Dict[str, Any]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    records = list(records)
    groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
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
        "inventory_version": "relation-inventory-v1",
        "source_record_count": len(records),
        "extracted_record_count": len(records),
        "signature_count": len(entries),
        "max_examples_per_signature": max_examples,
        "entries": entries,
    }


def apply_relation_mapping(
    records: Iterable[Dict[str, Any]],
    inventory: Dict[str, Any],
    taxonomy: Dict[str, Any],
    mapping: Dict[str, Any],
) -> List[Dict[str, Any]]:
    relation_by_id = {item["id"]: item for item in taxonomy["relations"]}
    inventory_ids = {item["signature_id"] for item in inventory["entries"]}
    mapping_by_signature = {}
    for item in mapping["mappings"]:
        signature_id = item["signature_id"]
        if signature_id not in inventory_ids:
            raise ValueError(f"mapping references unknown signature: {signature_id}")
        if signature_id in mapping_by_signature:
            raise ValueError(f"duplicate mapping for signature: {signature_id}")
        status = item["normalization_status"]
        relation_id = item.get("relation_normalized")
        if status == "mapped":
            if relation_id not in relation_by_id:
                raise ValueError(f"mapping references unknown relation: {relation_id}")
            relation = relation_by_id[relation_id]
            if item.get("subject_type_normalized") != relation["subject_type"]:
                raise ValueError(f"normalized subject type mismatch: {signature_id}")
            if item.get("answer_type_normalized") != relation["answer_type"]:
                raise ValueError(f"normalized answer type mismatch: {signature_id}")
        elif status not in {"ambiguous", "out_of_taxonomy"}:
            raise ValueError(f"invalid normalization status: {status}")
        elif relation_id is not None:
            raise ValueError(f"unresolved mapping must have null relation: {signature_id}")
        mapping_by_signature[signature_id] = item

    normalized = []
    for record in records:
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
                "normalization_review_status": item.get("review_status") if item else None,
            }
        )
    return normalized


def _load_candidate_mapping_module():
    path = PROJECT_ROOT / "scripts" / "build_candidate_relation_mapping.py"
    spec = importlib.util.spec_from_file_location("candidate_relation_mapping", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load relation mapping helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate_mapping = _load_candidate_mapping_module()


PROMPT_TEMPLATES: Dict[str, str] = {
    "description_named_term": "The conventional term for {subject} is",
    "term_definition": "The definition of {subject} is",
    "description_named_entity": "The entity matching the description {subject} is",
    "entity_alias": "An alternative name for {subject} is",
    "abbreviation_expansion": "The expansion of {subject} is",
    "entity_antonym": "The opposite of {subject} is",
    "entity_classification": "The class or type of {subject} is",
    "entity_location": "The location associated with {subject} is",
    "organism_habitat": "The natural habitat or range of {subject} is",
    "entity_origin": "The origin of {subject} is",
    "part_of": "The larger whole containing {subject} is",
    "has_part": "A component of {subject} is",
    "made_of": "The material or main constituent of {subject} is",
    "entity_function": "The characteristic function of {subject} is",
    "activity_goal": "The intended goal of {subject} is",
    "activity_requirement": "A prerequisite for {subject} is",
    "cause_effect": "An effect caused by {subject} is",
    "effect_cause": "The cause of {subject} is",
    "producer_product": "A product produced by {subject} is",
    "entity_formation_agent": "The agent or process that forms {subject} is",
    "entity_attribute": "A characteristic property of {subject} is",
    "entity_measurement": "The measured value associated with {subject} is",
    "measurement_target": "The quantity measured by {subject} is",
    "quantity_unit": "The unit used to measure {subject} is",
    "entity_time": "The time associated with {subject} is",
    "work_creator": "The creator of {subject} is",
    "description_example": "An example matching {subject} is",
    "symbol_meaning": "The meaning represented by {subject} is",
    "entity_controller": "The controller or regulator of {subject} is",
    "entity_protection": "The material, structure, or agent protecting {subject} is",
    "entity_transformation": "The result of transforming {subject} is",
    "sequence_successor": "The stage or entity following {subject} is",
    "entity_capability": "A capability of {subject} is",
    "entity_energy_form": "The form of energy associated with {subject} is",
    "entity_instrument": "The instrument used in relation to {subject} is",
    "entity_determinant": "The factor that determines {subject} is",
    "entity_language": "The language associated with {subject} is",
    "element_symbol": "The chemical symbol of {subject} is",
    "person_spouse": "The spouse of {subject} is",
    "work_genre": "The genre of {subject} is",
    "process_location": "The location where {subject} occurs is",
}


EXTRA_RELATIONS: Sequence[Dict[str, Any]] = (
    {"id": "entity_energy_form", "definition": "the form or type of energy associated with an entity or process", "subject_type": "entity", "answer_type": "energy_form", "direction": "entity -> energy_form", "examples": []},
    {"id": "entity_instrument", "definition": "an instrument or tool used to observe, measure, or act on an entity", "subject_type": "entity", "answer_type": "instrument", "direction": "entity -> instrument", "examples": []},
    {"id": "entity_determinant", "definition": "a factor that determines or controls an entity property", "subject_type": "entity_property", "answer_type": "determining_factor", "direction": "entity_property -> determining_factor", "examples": []},
    {"id": "entity_language", "definition": "a language associated with a place, group, person, or work", "subject_type": "entity", "answer_type": "language", "direction": "entity -> language", "examples": []},
    {"id": "element_symbol", "definition": "the chemical symbol of an element", "subject_type": "element", "answer_type": "symbol", "direction": "element -> symbol", "examples": []},
    {"id": "person_spouse", "definition": "the spouse of a person", "subject_type": "person", "answer_type": "person", "direction": "person -> person", "examples": []},
    {"id": "work_genre", "definition": "the genre or artistic category of a creative work", "subject_type": "creative_work", "answer_type": "genre", "direction": "creative_work -> genre", "examples": []},
    {"id": "process_location", "definition": "the location where a process occurs", "subject_type": "process", "answer_type": "location", "direction": "process -> location", "examples": []},
)


CODEX_PRIORITY_RULES: Sequence[Tuple[str, str, float]] = (
    ("entity_transformation", r"\benergy transformation\b", 0.96),
    ("entity_energy_form", r"\b(?:type|form|kind) of energy\b|\benergy (?:required|released|produced|carried|present|kept|measured)\b", 0.93),
    ("entity_instrument", r"\b(?:instrument used|tool used|used to monitor|used to observe|analyzed using)\b", 0.94),
    ("description_named_entity", r"\b(?:body system|organ system|organisms?|celestial object|source) (?:that|which|responsible for)\b", 0.92),
    ("entity_determinant", r"\b(?:determined by|determines|property that determines|factor that determines)\b", 0.94),
    ("effect_cause", r"\b(?:force that keeps|kept in orbit by|force that causes|responsible for)\b", 0.92),
    ("entity_language", r"\b(?:main language|official language|language spoken|written in language)\b", 0.95),
    ("element_symbol", r"\b(?:chemical symbol|element symbol|symbol of element)\b", 0.97),
    ("person_spouse", r"^spouse$|\bmarried to\b", 0.98),
    ("work_genre", r"^genre$|\bgenre of\b", 0.96),
    ("process_location", r"\b(?:process location|where .+ occurs|occurs in|takes place in|begins in)\b", 0.90),
)


CODEX_SEARCH_RULES: Sequence[Tuple[str, str, float]] = (
    ("abbreviation_expansion", r"\b(?:stands for|full form|expansion|abbreviation|acronym)\b", 0.96),
    ("entity_alias", r"\b(?:synonym|nickname|another name|also called|also known|common name|commonly called|official name|alternative name)\b", 0.94),
    ("description_named_term", r"\b(?:term for|term that|scientific term|technical term|word for|process name|group name|referred to as)\b", 0.93),
    ("term_definition", r"\b(?:definition|meaning|means|refers to|defined as)\b", 0.93),
    ("entity_antonym", r"\b(?:opposite|antonym)\b", 0.97),
    ("entity_classification", r"^(?:type|types|kind|form|class|classification|category|genre|species|state of matter|member of class)\b", 0.91),
    ("organism_habitat", r"\b(?:habitat|native to|native range|required habitat|areas .+ found)\b", 0.92),
    ("entity_location", r"\b(?:located|location|found in|occurs in|country|continent|hemisphere|region|coastal state|city|lobe|surface of|orbit)\b", 0.88),
    ("entity_origin", r"\b(?:origin|originated|derived from|source region|etymological)\b", 0.91),
    ("made_of", r"\b(?:made of|made from|composed of|consists of|constructed of|mainly composed|predominantly made|building blocks|particles that make up)\b", 0.92),
    ("part_of", r"\b(?:part of|member of|belongs to|contained in|subdivision|located in lobe)\b", 0.89),
    ("has_part", r"\b(?:contains|comprises|comprise|has component|components|structures at|divided into|number of ribs)\b", 0.89),
    ("entity_function", r"\b(?:function|purpose|role|used to|used for|serves as|serves to|enables|tool used|uses to|responsible for)\b", 0.90),
    ("entity_capability", r"\b(?:capable of|can perform|ability|unable to|cannot)\b", 0.87),
    ("activity_goal", r"\b(?:goal|objective|aim|intended outcome)\b", 0.92),
    ("activity_requirement", r"\b(?:requires|required|necessary|prerequisite|depends on|needed to|must include)\b", 0.90),
    ("effect_cause", r"\b(?:caused by|cause of|results from|attributed to|kept in orbit by|held together by|determined by)\b", 0.91),
    ("cause_effect", r"\b(?:causes|results in|leads to|effect on|affects|damages|stimulates|influences)\b", 0.89),
    ("producer_product", r"\b(?:produces|product|output|waste product|formed substance|releases|secretes|yields|forms)\b", 0.89),
    ("measurement_target", r"\b(?:measures|used to measure|indicates|measurement target)\b", 0.91),
    ("quantity_unit", r"\b(?:unit of measurement|si unit|measured in|unit used|unit most often|unit of time)\b", 0.94),
    ("entity_measurement", r"\b(?:number of|amount|percentage|rate|height|length|width|mass|weight|temperature|boiling point|freezing point|dimensions|numerical value)\b", 0.87),
    ("entity_time", r"\b(?:date|year|duration|time-frame|season when|occurs on day|occurred)\b", 0.88),
    ("work_creator", r"\b(?:author|creator|developer|director|artist|inventor|discoverer|written by|created by|opened by|attributed to)\b", 0.91),
    ("description_example", r"\b(?:example|examples|instance)\b", 0.90),
    ("symbol_meaning", r"\b(?:symbol|represents|represent|symbolizes|birthstone)\b", 0.88),
    ("entity_controller", r"\b(?:controlled by|regulated by|governed by|determines|controller|regulates)\b", 0.90),
    ("entity_protection", r"\b(?:protected by|protective|covering|covered by|covers)\b", 0.89),
    ("entity_transformation", r"\b(?:becomes|develops into|evolved into|transforms into|converted into|changes into|transferred to|energy transformation)\b", 0.90),
    ("sequence_successor", r"\b(?:comes after|followed by|next stage|successor|replaced by|replaced on|stages in order)\b", 0.89),
    ("entity_attribute", r"\b(?:attribute|property|characteristic|color|shape|texture|viscosity|pH|status|legality|title|charge|wavelength|energy type)\b", 0.84),
)


def codex_classify(entry: Dict[str, Any]) -> Tuple[str, Optional[str], float, str]:
    raw = entry["relation_raw_normalized"].strip().lower()
    for candidate_id, pattern, candidate_confidence in CODEX_PRIORITY_RULES:
        if re.search(pattern, raw):
            return (
                "mapped",
                candidate_id,
                candidate_confidence,
                f"Codex global priority review matched semantic pattern: {pattern}",
            )
    status, relation_id, confidence, reason = candidate_mapping.classify(entry)
    if status == "mapped":
        return status, relation_id, confidence, reason
    for candidate_id, pattern, candidate_confidence in CODEX_SEARCH_RULES:
        if re.search(pattern, raw):
            return (
                "mapped",
                candidate_id,
                candidate_confidence,
                f"Codex global semantic review matched conservative pattern: {pattern}",
            )
    return status, relation_id, confidence, reason


def comparable_text(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    value = normalize_text(value)
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def compact_annotation(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "terminal_status": record.get("terminal_status"),
        "extraction_status": record.get("extraction_status"),
        "exclusion_reason": record.get("exclusion_reason"),
        "subject": record.get("subject"),
        "subject_type": record.get("subject_type"),
        "relation_raw": record.get("relation_raw"),
        "answer": record.get("answer"),
        "answer_type": record.get("answer_type"),
        "canonical_fact": record.get("canonical_fact"),
        "extraction_confidence": record.get("extraction_confidence"),
        "validation_errors": record.get("validation_errors", []),
    }


def valid_extracted(record: Dict[str, Any]) -> bool:
    return (
        record.get("terminal_status") == "completed"
        and record.get("extraction_status") == "extracted"
        and not record.get("validation_errors")
    )


def review_case_for_pair(preferred: Dict[str, Any], secondary: Dict[str, Any]) -> Optional[str]:
    preferred_ok = valid_extracted(preferred)
    secondary_ok = valid_extracted(secondary)
    if preferred_ok != secondary_ok:
        return "single_model_extraction"
    if not preferred_ok:
        return None
    preferred_answer = comparable_text(preferred.get("answer"))
    secondary_answer = comparable_text(secondary.get("answer"))
    if not preferred_answer or preferred_answer != secondary_answer:
        return "dual_model_answer_conflict"
    return None


def validate_adjudication(
    adjudication: Dict[str, Any],
    source_id: str,
    review_case: str,
    allowed_models: Sequence[str],
) -> None:
    if adjudication.get("source_id") != source_id:
        raise ValueError(f"adjudication source_id mismatch: {source_id}")
    if adjudication.get("review_case") != review_case:
        raise ValueError(f"adjudication review_case mismatch: {source_id}")
    if adjudication.get("review_prompt_version") != CANONICAL_REVIEW_PROMPT_VERSION:
        raise ValueError(f"adjudication prompt version mismatch: {source_id}")
    decision = adjudication.get("codex_decision")
    if decision not in {"accept", "reject"}:
        raise ValueError(f"invalid adjudication decision: {source_id}")
    selected_model = adjudication.get("selected_model")
    if decision == "accept" and selected_model not in allowed_models:
        raise ValueError(f"accepted adjudication has invalid selected_model: {source_id}")
    if decision == "reject" and selected_model is not None:
        raise ValueError(f"rejected adjudication must not select a model: {source_id}")
    if not isinstance(adjudication.get("reason_code"), str) or not adjudication["reason_code"]:
        raise ValueError(f"adjudication reason_code is required: {source_id}")
    if not isinstance(adjudication.get("rationale"), str) or not adjudication["rationale"]:
        raise ValueError(f"adjudication rationale is required: {source_id}")


def verify_shared_provenance(first: Dict[str, Any], second: Dict[str, Any]) -> None:
    for field in (
        "source_id",
        "source_dataset",
        "source_subset",
        "source_question",
        "source_choices",
        "source_answer",
    ):
        if first.get(field) != second.get(field):
            raise ValueError(f"model provenance mismatch for {first.get('source_id')}: {field}")


def canonicalize_pair(
    preferred: Dict[str, Any],
    secondary: Dict[str, Any],
    adjudication: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    verify_shared_provenance(preferred, secondary)
    preferred_ok = valid_extracted(preferred)
    secondary_ok = valid_extracted(secondary)
    annotations = {
        PREFERRED_MODEL: compact_annotation(preferred),
        SECONDARY_MODEL: compact_annotation(secondary),
    }
    base = {
        key: preferred.get(key)
        for key in (
            "source_id",
            "source_dataset",
            "source_subset",
            "source_path",
            "source_original_index",
            "source_question",
            "source_choices",
            "source_answer",
            "candidate_id",
            "source_snapshot",
        )
    }
    decision: Dict[str, Any] = {
        **base,
        "canonical_policy_version": CANONICAL_POLICY_VERSION,
        "model_annotations": annotations,
    }

    if not preferred_ok and not secondary_ok:
        return {
            **decision,
            "canonical_decision": "exclude",
            "canonical_reason": "both_models_not_extracted_or_invalid",
        }, None
    review_case = review_case_for_pair(preferred, secondary)
    if review_case is not None:
        extracting_models = [
            model
            for model, ok in ((PREFERRED_MODEL, preferred_ok), (SECONDARY_MODEL, secondary_ok))
            if ok
        ]
        pending = {
            **decision,
            "canonical_decision": "pending_review",
            "canonical_reason": f"{review_case}_requires_codex_review",
            "review_case": review_case,
            "eligible_selected_models": extracting_models,
            "review_status": "pending_codex_adjudication",
        }
        if adjudication is None:
            return pending, None

        validate_adjudication(adjudication, preferred["source_id"], review_case, extracting_models)
        reviewed = {
            **pending,
            "codex_adjudication": adjudication,
            "review_status": "codex_adjudicated",
        }
        if adjudication["codex_decision"] == "reject":
            return {
                **reviewed,
                "canonical_decision": "exclude",
                "canonical_reason": f"codex_rejected_{review_case}",
            }, None

        selected_model = adjudication["selected_model"]
        selected = preferred if selected_model == PREFERRED_MODEL else secondary
        canonical = {
            **base,
            "terminal_status": "completed",
            "extraction_status": "extracted",
            "exclusion_reason": None,
            "subject": selected["subject"],
            "subject_type": selected["subject_type"],
            "relation_raw": selected["relation_raw"],
            "answer": selected["answer"],
            "answer_type": selected["answer_type"],
            "canonical_fact": selected["canonical_fact"],
            "extraction_confidence": float(selected.get("extraction_confidence", 0.0)),
            "validation_errors": [],
            "canonical_policy_version": CANONICAL_POLICY_VERSION,
            "canonical_decision": "keep",
            "canonical_reason": f"codex_accepted_{review_case}",
            "canonical_source_model": selected_model,
            "field_conflicts": [
                field
                for field in ("subject", "subject_type", "relation_raw", "answer", "answer_type", "canonical_fact")
                if comparable_text(preferred.get(field)) != comparable_text(secondary.get(field))
            ],
            "review_case": review_case,
            "review_status": "codex_adjudicated",
            "codex_adjudication": adjudication,
            "model_annotations": annotations,
        }
        return {
            **reviewed,
            "canonical_decision": "keep",
            "canonical_reason": f"codex_accepted_{review_case}",
            "canonical_source_model": selected_model,
            "field_conflicts": canonical["field_conflicts"],
        }, canonical

    conflicts = [
        field
        for field in ("subject", "subject_type", "relation_raw", "answer_type", "canonical_fact")
        if comparable_text(preferred.get(field)) != comparable_text(secondary.get(field))
    ]
    canonical = {
        **base,
        "terminal_status": "completed",
        "extraction_status": "extracted",
        "exclusion_reason": None,
        "subject": preferred["subject"],
        "subject_type": preferred["subject_type"],
        "relation_raw": preferred["relation_raw"],
        "answer": preferred["answer"],
        "answer_type": preferred["answer_type"],
        "canonical_fact": preferred["canonical_fact"],
        "extraction_confidence": min(
            float(preferred.get("extraction_confidence", 0.0)),
            float(secondary.get("extraction_confidence", 0.0)),
        ),
        "validation_errors": [],
        "canonical_policy_version": CANONICAL_POLICY_VERSION,
        "canonical_decision": "keep",
        "canonical_reason": "dual_model_extraction_and_answer_consensus",
        "canonical_source_model": PREFERRED_MODEL,
        "field_conflicts": conflicts,
        "model_annotations": annotations,
    }
    return {
        **decision,
        "canonical_decision": "keep",
        "canonical_reason": "dual_model_extraction_and_answer_consensus",
        "canonical_source_model": PREFERRED_MODEL,
        "field_conflicts": conflicts,
    }, canonical


def determine_canonical_triples(
    preferred_records: Iterable[Dict[str, Any]],
    secondary_records: Iterable[Dict[str, Any]],
    adjudications: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    preferred_by_id = {record["source_id"]: record for record in preferred_records}
    secondary_by_id = {record["source_id"]: record for record in secondary_records}
    all_ids = sorted(set(preferred_by_id) | set(secondary_by_id))
    decisions: List[Dict[str, Any]] = []
    canonical: List[Dict[str, Any]] = []
    for source_id in all_ids:
        preferred = preferred_by_id.get(source_id)
        secondary = secondary_by_id.get(source_id)
        if preferred is None or secondary is None:
            present = preferred or secondary
            decisions.append(
                {
                    "source_id": source_id,
                    "source_dataset": present.get("source_dataset") if present else None,
                    "canonical_policy_version": CANONICAL_POLICY_VERSION,
                    "canonical_decision": "exclude",
                    "canonical_reason": "missing_model_record",
                }
            )
            continue
        decision, record = canonicalize_pair(
            preferred,
            secondary,
            (adjudications or {}).get(source_id),
        )
        decisions.append(decision)
        if record is not None:
            canonical.append(record)

    reason_counts = Counter(item["canonical_reason"] for item in decisions)
    field_conflict_counts = Counter(
        conflict for item in canonical for conflict in item.get("field_conflicts", [])
    )
    summary = {
        "canonical_policy_version": CANONICAL_POLICY_VERSION,
        "source_record_count": len(all_ids),
        "canonical_triple_count": len(canonical),
        "excluded_record_count": sum(
            item["canonical_decision"] == "exclude" for item in decisions
        ),
        "decision_reason_counts": dict(reason_counts),
        "canonical_field_conflict_counts": dict(field_conflict_counts),
        "preferred_field_model": PREFERRED_MODEL,
        "secondary_model": SECONDARY_MODEL,
        "pending_review_count": sum(
            item["canonical_decision"] == "pending_review" for item in decisions
        ),
        "codex_accepted_review_count": sum(
            item["canonical_reason"].startswith("codex_accepted_") for item in decisions
        ),
        "codex_rejected_review_count": sum(
            item["canonical_reason"].startswith("codex_rejected_") for item in decisions
        ),
    }
    return decisions, canonical, summary


def build_canonical_review_queue(decisions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    queue = []
    for decision in decisions:
        if decision.get("canonical_decision") != "pending_review":
            continue
        queue.append(
            {
                "source_id": decision["source_id"],
                "source_dataset": decision.get("source_dataset"),
                "source_subset": decision.get("source_subset"),
                "source_question": decision.get("source_question"),
                "source_choices": decision.get("source_choices"),
                "source_answer": decision.get("source_answer"),
                "review_case": decision["review_case"],
                "eligible_selected_models": decision["eligible_selected_models"],
                "model_annotations": decision["model_annotations"],
                "required_output": {
                    "source_id": decision["source_id"],
                    "review_case": decision["review_case"],
                    "codex_decision": "accept|reject",
                    "selected_model": "eligible model when accepted, otherwise null",
                    "reason_code": "short_machine_readable_code",
                    "rationale": "brief item-specific justification",
                    "review_prompt_version": CANONICAL_REVIEW_PROMPT_VERSION,
                },
            }
        )
    return queue


def load_adjudications(path: Path) -> Dict[str, Dict[str, Any]]:
    records = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("adjudication source_id is required")
        if source_id in by_id:
            raise ValueError(f"duplicate adjudication source_id: {source_id}")
        by_id[source_id] = record
    return by_id


def build_taxonomy_and_mapping(
    inventory: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    taxonomy = {
        "taxonomy_version": TAXONOMY_VERSION,
        "source_inventory": "relation_inventory.json",
        "created_by": "codex-global-relation-review-v1",
        "relations": [],
    }
    for item in candidate_mapping.RELATIONS:
        relation_item = dict(item)
        relation_item["prompt_template_en"] = PROMPT_TEMPLATES[item["id"]]
        taxonomy["relations"].append(relation_item)
    for item in EXTRA_RELATIONS:
        relation_item = dict(item)
        relation_item["prompt_template_en"] = PROMPT_TEMPLATES[item["id"]]
        taxonomy["relations"].append(relation_item)

    mappings = []
    relation_by_id = {item["id"]: item for item in taxonomy["relations"]}
    for entry in inventory["entries"]:
        status, relation_id, confidence, reason = codex_classify(entry)
        relation = relation_by_id.get(relation_id) if relation_id else None
        mappings.append(
            {
                "signature_id": entry["signature_id"],
                "normalization_status": status,
                "relation_normalized": relation_id,
                "subject_type_normalized": relation["subject_type"] if relation else None,
                "answer_type_normalized": relation["answer_type"] if relation else None,
                "normalization_reason": reason,
                "normalization_confidence": confidence if relation else None,
                "review_status": "codex_mapped" if relation else "codex_reviewed_unresolved",
            }
        )
    mapping = {
        "mapping_version": MAPPING_VERSION,
        "inventory_version": inventory["inventory_version"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "created_by": "codex-global-relation-review-v1",
        "mappings": mappings,
    }
    candidate_mapping.populate_examples(taxonomy, inventory, mapping)
    summary = candidate_mapping.summarize(inventory, mapping)
    review_queue = candidate_mapping.build_review_queue(inventory, mapping)
    review_queue["review_policy"] = (
        "Every unresolved signature was reviewed by the conservative Codex mapping policy; "
        "unresolved labels remain null rather than being forced into an incompatible relation."
    )
    return taxonomy, mapping, summary, review_queue


def taxonomy_by_id(taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["id"]: item for item in taxonomy["relations"]}


def canonical_fact_completion(canonical_fact: str, answer: str) -> Optional[str]:
    fact = canonical_fact.strip()
    fact = re.sub(r"[\s.!?]+$", "", fact)
    match = re.search(rf"\b{re.escape(answer.strip())}$", fact, flags=re.IGNORECASE)
    if not match:
        return None
    prompt = fact[: match.start()].rstrip(" :,-")
    return prompt if prompt else None


def generate_factual_prompts(
    records: Iterable[Dict[str, Any]], taxonomy: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    relations = taxonomy_by_id(taxonomy)
    outputs: List[Dict[str, Any]] = []
    for record in records:
        relation_id = record.get("relation_normalized")
        relation = relations.get(relation_id)
        if record.get("normalization_status") != "mapped" or relation is None:
            outputs.append(
                {
                    **record,
                    "factual_prompt_version": PROMPT_VERSION,
                    "factual_prompt_status": "not_generated_unresolved_relation",
                    "prompt_en": None,
                    "prompt_template_id": None,
                }
            )
            continue
        prompt = canonical_fact_completion(record["canonical_fact"], record["answer"])
        prompt_source = "canonical_fact_answer_suffix"
        prompt_template_id = f"{relation_id}_canonical_fact_suffix_v1"
        prompt_quality_tier = "strict_factual_completion"
        if prompt is None:
            question = record["source_question"].strip()
            prompt = f"Factual question: {question}\nAnswer:"
            prompt_source = "source_question_open_answer_fallback"
            prompt_template_id = "source_question_open_answer_en_v1"
            prompt_quality_tier = "open_answer_fallback"
        outputs.append(
            {
                **record,
                "factual_prompt_version": PROMPT_VERSION,
                "factual_prompt_status": "generated",
                "prompt_en": prompt,
                "prompt_template_id": prompt_template_id,
                "prompt_source": prompt_source,
                "prompt_quality_tier": prompt_quality_tier,
                "prompt_expected_answer": record["answer"],
            }
        )
    counts = Counter(item["factual_prompt_status"] for item in outputs)
    return outputs, {
        "factual_prompt_version": PROMPT_VERSION,
        "record_count": len(outputs),
        "status_counts": dict(counts),
        "prompt_source_counts": dict(
            Counter(item.get("prompt_source") for item in outputs if item.get("prompt_source"))
        ),
        "prompt_quality_tier_counts": dict(
            Counter(
                item.get("prompt_quality_tier")
                for item in outputs
                if item.get("prompt_quality_tier")
            )
        ),
    }


def option_similarity(answer: str, option: str) -> float:
    answer_norm = normalize_text(answer)
    option_norm = normalize_text(option)
    return round(SequenceMatcher(None, answer_norm, option_norm).ratio(), 6)


def build_distractor_candidates(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    expanded: List[Dict[str, Any]] = []
    for record in records:
        source_choices = record.get("source_choices") or []
        source_answer_key = comparable_text(record.get("source_answer"))
        answer_key = comparable_text(record.get("answer"))
        candidates = []
        source_wrong_options = []
        for index, choice in enumerate(source_choices):
            choice_key = comparable_text(choice)
            if not choice_key or choice_key in {source_answer_key, answer_key}:
                continue
            source_wrong_options.append(choice)
            candidate = {
                "distractor_answer": choice,
                "distractor_source": "original_wrong_option",
                "source_choice_index": index,
                "answer_similarity": option_similarity(record["answer"], choice),
                "type_compatibility": "source_option_parallelism_unverified",
                "distractor_fact_status": "source_label_assumed_false_unverified",
                "distractor_verified": False,
            }
            candidates.append(candidate)
        candidates.sort(key=lambda item: (-item["answer_similarity"], item["source_choice_index"]))
        selected = candidates[0] if candidates else None
        grouped_record = {
            **record,
            "distractor_version": DISTRACTOR_VERSION,
            "source_wrong_options": source_wrong_options,
            "distractor_candidates": candidates,
            "selected_distractor_answer": selected["distractor_answer"] if selected else None,
            "selected_distractor_rule": "highest_answer_text_similarity" if selected else None,
            "distractor_construction_status": "constructed_unverified" if candidates else "no_candidate",
        }
        grouped.append(grouped_record)
        for rank, candidate in enumerate(candidates, start=1):
            expanded.append(
                {
                    "distractor_id": f"{record['source_id']}_source_wrong_{candidate['source_choice_index']}",
                    "source_id": record["source_id"],
                    "source_dataset": record["source_dataset"],
                    "subject": record["subject"],
                    "relation_raw": record["relation_raw"],
                    "relation_normalized": record.get("relation_normalized"),
                    "answer": record["answer"],
                    "source_answer": record["source_answer"],
                    "prompt_en": record.get("prompt_en"),
                    "candidate_rank": rank,
                    **candidate,
                }
            )
    return grouped, expanded, {
        "distractor_version": DISTRACTOR_VERSION,
        "record_count": len(grouped),
        "records_with_candidates": sum(bool(item["distractor_candidates"]) for item in grouped),
        "candidate_count": len(expanded),
        "verified_candidate_count": 0,
    }


def validate_pipeline_outputs(
    preferred_records: Iterable[Dict[str, Any]],
    canonical: Sequence[Dict[str, Any]],
    inventory: Dict[str, Any],
    mapping: Dict[str, Any],
    normalized: Sequence[Dict[str, Any]],
    prompts: Sequence[Dict[str, Any]],
    distractors: Sequence[Dict[str, Any]],
    expanded_distractors: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    errors: List[str] = []
    source_by_id = {item["source_id"]: item for item in preferred_records}
    canonical_ids = [item["source_id"] for item in canonical]
    if len(canonical_ids) != len(set(canonical_ids)):
        errors.append("duplicate canonical source_id")
    for record in canonical:
        source = source_by_id.get(record["source_id"])
        if source is None:
            errors.append(f"canonical record missing source: {record['source_id']}")
        elif record["source_choices"] != source["source_choices"]:
            errors.append(f"source_choices changed: {record['source_id']}")

    if inventory.get("extracted_record_count") != len(canonical):
        errors.append("inventory extracted_record_count mismatch")
    if len(mapping.get("mappings", [])) != inventory.get("signature_count"):
        errors.append("mapping does not cover every inventory signature")
    if [item["source_id"] for item in normalized] != canonical_ids:
        errors.append("normalized record order or IDs differ from canonical triples")
    if [item["source_id"] for item in prompts] != canonical_ids:
        errors.append("prompt record order or IDs differ from canonical triples")

    prompt_by_id = {item["source_id"]: item for item in prompts}
    for record in normalized:
        mapped = record["normalization_status"] == "mapped"
        if mapped != bool(record.get("relation_normalized")):
            errors.append(f"normalization status/relation mismatch: {record['source_id']}")
        prompt = prompt_by_id[record["source_id"]]
        generated = prompt["factual_prompt_status"] == "generated"
        if generated != mapped:
            errors.append(f"prompt status does not follow mapping: {record['source_id']}")
        if generated and (not prompt.get("prompt_en") or not prompt.get("prompt_expected_answer")):
            errors.append(f"generated prompt is incomplete: {record['source_id']}")

    generated_ids = {
        item["source_id"] for item in prompts if item["factual_prompt_status"] == "generated"
    }
    distractor_ids = {item["source_id"] for item in distractors}
    if distractor_ids != generated_ids:
        errors.append("distractor records do not exactly match prompt-ready records")
    expanded_by_source = Counter(item["source_id"] for item in expanded_distractors)
    for record in distractors:
        choices = record["source_choices"]
        expected_wrong = [
            choice
            for choice in choices
            if comparable_text(choice)
            not in {comparable_text(record["source_answer"]), comparable_text(record["answer"])}
        ]
        if record["source_wrong_options"] != expected_wrong:
            errors.append(f"source wrong-option order/content mismatch: {record['source_id']}")
        if len(record["distractor_candidates"]) != expanded_by_source[record["source_id"]]:
            errors.append(f"expanded distractor count mismatch: {record['source_id']}")
        for candidate in record["distractor_candidates"]:
            if candidate["distractor_answer"] not in choices:
                errors.append(f"distractor absent from source choices: {record['source_id']}")
            if comparable_text(candidate["distractor_answer"]) in {
                comparable_text(record["source_answer"]),
                comparable_text(record["answer"]),
            }:
                errors.append(f"answer leaked into distractors: {record['source_id']}")

    result = {
        "validation_version": "canonical-factual-pipeline-validation-v1",
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors[:100],
        "counts": {
            "canonical_triples": len(canonical),
            "inventory_signatures": inventory.get("signature_count"),
            "mapping_decisions": len(mapping.get("mappings", [])),
            "normalized_records": len(normalized),
            "prompt_records": len(prompts),
            "distractor_records": len(distractors),
            "expanded_distractor_candidates": len(expanded_distractors),
        },
    }
    if errors:
        raise ValueError(f"pipeline output validation failed: {errors[:5]}")
    return result


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", required=True)
    result.add_argument("--output-dir", default="canonical-v2")
    result.add_argument("--preferred-model", default=PREFERRED_MODEL)
    result.add_argument("--secondary-model", default=SECONDARY_MODEL)
    result.add_argument("--input-name", default="triple_extractions.jsonl")
    result.add_argument("--max-examples", type=int, default=5)
    result.add_argument(
        "--adjudication",
        help="Codex JSONL decisions for every queued single-model extraction and answer conflict",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = run_dir / output_dir
    preferred_path = run_dir / "models" / args.preferred_model / args.input_name
    secondary_path = run_dir / "models" / args.secondary_model / args.input_name

    preferred_records = read_jsonl(preferred_path)
    secondary_records = read_jsonl(secondary_path)
    baseline_decisions, _, _ = determine_canonical_triples(preferred_records, secondary_records)
    review_queue = build_canonical_review_queue(baseline_decisions)
    expected_review_ids = {item["source_id"] for item in review_queue}
    adjudications = load_adjudications(Path(args.adjudication).resolve()) if args.adjudication else None
    if adjudications is not None:
        supplied_ids = set(adjudications)
        if supplied_ids != expected_review_ids:
            missing = sorted(expected_review_ids - supplied_ids)[:10]
            extra = sorted(supplied_ids - expected_review_ids)[:10]
            raise ValueError(
                f"adjudication coverage mismatch: missing={missing}, extra={extra}, "
                f"expected={len(expected_review_ids)}, supplied={len(supplied_ids)}"
            )
    decisions, canonical, canonical_summary = determine_canonical_triples(
        preferred_records, secondary_records, adjudications
    )
    write_jsonl(output_dir / "canonical_decisions.jsonl", decisions)
    write_jsonl(output_dir / "canonical_review_queue.jsonl", review_queue)
    write_jsonl(output_dir / "canonical_triples.jsonl", canonical)
    write_json(output_dir / "canonical_summary.json", canonical_summary)

    if canonical_summary["pending_review_count"]:
        print(json.dumps({
            "output_dir": str(output_dir),
            "status": "pending_codex_adjudication",
            "review_queue": str(output_dir / "canonical_review_queue.jsonl"),
            "canonical": canonical_summary,
        }, ensure_ascii=False, indent=2))
        return 2

    inventory = build_relation_inventory(canonical, max_examples=args.max_examples)
    write_json(output_dir / "relation_inventory.json", inventory)
    taxonomy, mapping, mapping_summary, review_queue = build_taxonomy_and_mapping(inventory)
    write_json(output_dir / "relation_taxonomy_v1.json", taxonomy)
    write_json(output_dir / "relation_mapping_v1.json", mapping)
    write_json(output_dir / "relation_mapping_summary.json", mapping_summary)
    write_json(output_dir / "relation_review_queue_v1.json", review_queue)

    normalized = apply_relation_mapping(canonical, inventory, taxonomy, mapping)
    write_jsonl(output_dir / "triples_normalized.jsonl", normalized)
    prompts, prompt_summary = generate_factual_prompts(normalized, taxonomy)
    write_jsonl(output_dir / "factual_prompts.jsonl", prompts)
    write_json(output_dir / "factual_prompt_summary.json", prompt_summary)

    prompt_ready = [item for item in prompts if item["factual_prompt_status"] == "generated"]
    distractors, expanded, distractor_summary = build_distractor_candidates(prompt_ready)
    write_jsonl(output_dir / "factual_triples_with_distractors.jsonl", distractors)
    write_jsonl(output_dir / "distractor_candidates.jsonl", expanded)
    write_json(output_dir / "distractor_summary.json", distractor_summary)
    validation_summary = validate_pipeline_outputs(
        preferred_records,
        canonical,
        inventory,
        mapping,
        normalized,
        prompts,
        distractors,
        expanded,
    )
    write_json(output_dir / "pipeline_validation_summary.json", validation_summary)

    print(json.dumps({
        "output_dir": str(output_dir),
        "canonical": canonical_summary,
        "mapping": mapping_summary,
        "prompts": prompt_summary,
        "distractors": distractor_summary,
        "validation": validation_summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
