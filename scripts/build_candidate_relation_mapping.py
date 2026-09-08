#!/usr/bin/env python3
"""Build a conservative candidate taxonomy and mapping from a relation inventory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TAXONOMY_VERSION = "relation-taxonomy-candidate-v1"
MAPPING_VERSION = "relation-mapping-candidate-v1"


def relation(
    relation_id: str,
    definition: str,
    subject_type: str,
    answer_type: str,
) -> Dict[str, Any]:
    return {
        "id": relation_id,
        "definition": definition,
        "subject_type": subject_type,
        "answer_type": answer_type,
        "direction": f"{subject_type} -> {answer_type}",
        "examples": [],
    }


RELATIONS = [
    relation("description_named_term", "the conventional term satisfying a description", "description", "term"),
    relation("term_definition", "the definition or meaning of a named term or expression", "term", "definition"),
    relation("description_named_entity", "the entity satisfying an identifying description", "description", "entity"),
    relation("entity_alias", "an alternative, common, or informal name for an entity", "entity", "name"),
    relation("abbreviation_expansion", "the expansion represented by an abbreviation or initial", "abbreviation", "expansion"),
    relation("entity_antonym", "an entity or concept with the opposite meaning or state", "entity", "opposite_entity"),
    relation("entity_classification", "the class, type, or category of an entity", "entity", "class"),
    relation("entity_location", "the location containing or associated with an entity", "entity", "location"),
    relation("organism_habitat", "the habitat or native range of an organism", "organism", "location"),
    relation("entity_origin", "the geographic, historical, or etymological origin of an entity", "entity", "origin"),
    relation("part_of", "the whole or system containing a named part", "part", "whole"),
    relation("has_part", "a component or member contained by a whole", "whole", "part"),
    relation("made_of", "the material or constituent from which an entity is made", "entity", "material"),
    relation("entity_function", "the characteristic function or intended purpose of an entity", "entity", "function"),
    relation("activity_goal", "the intended goal or objective of an activity", "activity", "goal"),
    relation("activity_requirement", "a prerequisite or resource required by an activity or process", "activity", "prerequisite"),
    relation("cause_effect", "the effect caused or enabled by a cause", "cause", "effect"),
    relation("effect_cause", "the cause or agent responsible for an effect", "effect", "cause"),
    relation("producer_product", "the entity, substance, or outcome produced by a producer or process", "producer", "product"),
    relation("entity_formation_agent", "the agent or process that forms an entity", "entity", "formation_agent"),
    relation("entity_attribute", "a characteristic, state, or qualitative property of an entity", "entity", "attribute"),
    relation("entity_measurement", "a quantitative measurement or value associated with an entity", "entity", "measurement"),
    relation("measurement_target", "the quantity or attribute measured by an instrument or unit", "measuring_entity", "quantity"),
    relation("quantity_unit", "the unit used to express a physical or abstract quantity", "quantity", "unit"),
    relation("entity_time", "a date, duration, period, or other temporal value associated with an entity", "entity", "time"),
    relation("work_creator", "the person or organization that created, authored, developed, or invented an entity", "created_entity", "creator"),
    relation("description_example", "an entity that is an example satisfying a description or class", "description", "example"),
    relation("symbol_meaning", "the meaning or entity represented by a symbol", "symbol", "meaning"),
    relation("entity_controller", "the controller, regulator, or determinant of an entity or process", "entity", "controller"),
    relation("entity_protection", "the material, structure, or agent protecting an entity", "entity", "protector"),
    relation("entity_transformation", "the state, form, or entity resulting from a transformation", "entity", "transformed_entity"),
    relation("sequence_successor", "the entity or stage that follows another in an ordered sequence", "entity", "successor"),
    relation("entity_capability", "an action or behavior an entity is capable of performing", "entity", "capability"),
]


RULES: List[Tuple[str, str, float]] = [
    ("abbreviation_expansion", r"^(stands for|expansion|full form|abbreviation for|acronym for)$", 0.99),
    ("description_named_term", r"^(term for definition|term for|scientific term|technical term|word for|name for definition)$", 0.99),
    ("term_definition", r"^(definition|meaning|means|defined as|refers to|definition of|meaning of)$", 0.98),
    ("entity_alias", r"^(synonym|common name|also called|also known as|commonly called|nickname|nickname for|alternative name|another name)$", 0.98),
    ("entity_antonym", r"^(opposite|antonym|opposite of)$", 0.99),
    ("entity_classification", r"^(type|type of|types|types of|kind of|is a|is a type of|form of|category|taxonomic classification)$", 0.92),
    ("organism_habitat", r"^(habitat|native to|native range|lives in|natural habitat|native hemisphere)$", 0.97),
    ("entity_location", r"^(located in|location|found in|occurs in|country|located in the state|place of occurrence|geographic location)$", 0.95),
    ("entity_origin", r"^(origin|place of origin|etymological origin|country of origin|source region|originated in|derived from)$", 0.94),
    ("part_of", r"^(part of|part of system|part of cycle|body system|member of|belongs to system|contained in)$", 0.96),
    ("has_part", r"^(contains|contain|include|includes|has component|components|typical contents|constituent|constituents)$", 0.94),
    ("made_of", r"^(made of|mainly composed of|composition|material|consists of|constituent material)$", 0.96),
    ("entity_function", r"^(purpose|function|main function|primary function|shared function|role|used for|serves to|primary activity)$", 0.96),
    ("activity_goal", r"^(goal|objective|aim|intended outcome)$", 0.96),
    ("activity_requirement", r"^(requires|required for|needs|necessary for|depends on|prerequisite)$", 0.95),
    ("effect_cause", r"^(caused by|cause of|formed by|produced by|results from|source of)$", 0.96),
    ("cause_effect", r"^(causes|cause|can cause|results in|leads to|can lead to|effect of|affects|produces effect)$", 0.96),
    ("producer_product", r"^(produces|product|output|yields|creates|forms|sound produced)$", 0.92),
    ("entity_formation_agent", r"^(formation agent|deposition agent|formed by|created by process)$", 0.94),
    ("quantity_unit", r"^(unit of measurement|si unit|unit|measured in)$", 0.98),
    ("measurement_target", r"^(measures|measure|measurement target)$", 0.95),
    ("entity_measurement", r"^(measurement|amount|number|percentage|rate|height|length|width|mass|weight|temperature|freezing point|distance|number of legs)$", 0.93),
    ("entity_time", r"^(date|year|time|duration|period|start time|end time|founded in|born in|died in|occurs after|age)$", 0.94),
    ("work_creator", r"^(author|creator|developer|inventor|discoverer|quotation author|written by|created by|developed by|invented by|founded by)$", 0.97),
    ("description_example", r"^(example|example of|instance|instance of)$", 0.90),
    ("symbol_meaning", r"^(represents|symbolizes|symbol for|indicates|indicate)$", 0.90),
    ("entity_controller", r"^(controlled by|regulated by|governed by|determined by|controlled through)$", 0.95),
    ("entity_protection", r"^(protected by|protective covering|covering|protection)$", 0.96),
    ("entity_transformation", r"^(becomes|evolved into|transforms into|converted into|changes into|energy transformation|converts .+ into)$", 0.94),
    ("sequence_successor", r"^(comes after|followed by|next stage|successor)$", 0.93),
    ("entity_capability", r"^(can|can perform|capable of|behavior|action|typical action)$", 0.88),
    ("entity_attribute", r"^(attribute|property|characteristic|feature|color|shape|status|electrical charge|state|description)$", 0.90),
    ("description_named_entity", r"^(name|identity|is the|called)$", 0.86),
]


AMBIGUOUS_EXACT = {
    "is",
    "are",
    "has",
    "source",
    "form",
    "mechanism",
    "effect",
    "relation",
    "associated with",
    "related to",
    "considered to be",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def classify(entry: Dict[str, Any]) -> Tuple[str, Optional[str], float, str]:
    raw = entry["relation_raw_normalized"].strip().lower()
    if raw == "classification":
        answer_type = entry["answer_type"].strip().lower()
        if re.search(r"class|category|type|rank|group|kingdom|phylum|family|order|genus|species", answer_type):
            return "mapped", "entity_classification", 0.90, "Classification label has an explicit class-like answer type."
        return "ambiguous", None, 0.0, "Classification label is overloaded and its answer type is not explicitly class-like."
    for relation_id, pattern, confidence in RULES:
        if re.fullmatch(pattern, raw):
            return "mapped", relation_id, confidence, f"Matched conservative candidate rule: {pattern}"
    if raw in AMBIGUOUS_EXACT or len(raw.split()) <= 1:
        return "ambiguous", None, 0.0, "Relation label is broad or overloaded; manual semantic review required."
    return "out_of_taxonomy", None, 0.0, "No conservative candidate rule matched this relation signature."


def build_mapping(inventory: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    relation_by_id = {item["id"]: item for item in taxonomy["relations"]}
    mappings = []
    for entry in inventory["entries"]:
        status, relation_id, confidence, reason = classify(entry)
        relation_item = relation_by_id.get(relation_id) if relation_id else None
        mappings.append(
            {
                "signature_id": entry["signature_id"],
                "normalization_status": status,
                "relation_normalized": relation_id,
                "subject_type_normalized": relation_item["subject_type"] if relation_item else None,
                "answer_type_normalized": relation_item["answer_type"] if relation_item else None,
                "normalization_reason": reason,
                "normalization_confidence": confidence if relation_item else None,
            }
        )
    return {
        "mapping_version": MAPPING_VERSION,
        "inventory_version": inventory["inventory_version"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "created_by": "codex-conservative-rule-candidate",
        "mappings": mappings,
    }


def populate_examples(
    taxonomy: Dict[str, Any], inventory: Dict[str, Any], mappings: Dict[str, Any], max_examples: int = 5
) -> None:
    relation_by_id = {item["id"]: item for item in taxonomy["relations"]}
    inventory_by_id = {item["signature_id"]: item for item in inventory["entries"]}
    for mapping in mappings["mappings"]:
        relation_id = mapping["relation_normalized"]
        if not relation_id:
            continue
        examples = relation_by_id[relation_id]["examples"]
        for example in inventory_by_id[mapping["signature_id"]]["examples"]:
            pair = [example["subject"], example["answer"]]
            if pair not in examples and len(examples) < max_examples:
                examples.append(pair)


def summarize(inventory: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    inventory_by_id = {item["signature_id"]: item for item in inventory["entries"]}
    signature_counts: Dict[str, int] = {}
    record_counts: Dict[str, int] = {}
    relation_counts: Dict[str, int] = {}
    for item in mapping["mappings"]:
        status = item["normalization_status"]
        count = inventory_by_id[item["signature_id"]]["count"]
        signature_counts[status] = signature_counts.get(status, 0) + 1
        record_counts[status] = record_counts.get(status, 0) + count
        relation_id = item.get("relation_normalized")
        if relation_id:
            relation_counts[relation_id] = relation_counts.get(relation_id, 0) + count
    total_signatures = len(inventory_by_id)
    total_records = sum(item["count"] for item in inventory["entries"])
    return {
        "signature_status_counts": signature_counts,
        "record_status_counts": record_counts,
        "mapped_signature_coverage": signature_counts.get("mapped", 0) / total_signatures,
        "mapped_record_coverage": record_counts.get("mapped", 0) / total_records,
        "mapped_relation_record_counts": dict(
            sorted(relation_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ),
    }


def build_review_queue(inventory: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    inventory_by_id = {item["signature_id"]: item for item in inventory["entries"]}
    unresolved = []
    for item in mapping["mappings"]:
        if item["normalization_status"] == "mapped":
            continue
        entry = inventory_by_id[item["signature_id"]]
        unresolved.append(
            {
                **entry,
                "normalization_status": item["normalization_status"],
                "normalization_reason": item["normalization_reason"],
            }
        )
    unresolved.sort(
        key=lambda item: (
            0 if item["normalization_status"] == "ambiguous" else 1,
            -item["count"],
            item["signature_id"],
        )
    )
    return {
        "mapping_version": mapping["mapping_version"],
        "inventory_version": inventory["inventory_version"],
        "unresolved_signature_count": len(unresolved),
        "unresolved_record_count": sum(item["count"] for item in unresolved),
        "entries": unresolved,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--inventory", required=True)
    result.add_argument("--taxonomy-output", required=True)
    result.add_argument("--mapping-output", required=True)
    result.add_argument("--summary-output", required=True)
    result.add_argument("--review-output", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    inventory_path = Path(args.inventory)
    inventory = load_json(inventory_path)
    taxonomy = {
        "taxonomy_version": TAXONOMY_VERSION,
        "created_by": "codex-conservative-rule-candidate",
        "source_inventory": inventory_path.name,
        "relations": RELATIONS,
    }
    mapping = build_mapping(inventory, taxonomy)
    populate_examples(taxonomy, inventory, mapping)
    summary = summarize(inventory, mapping)
    review_queue = build_review_queue(inventory, mapping)
    write_json(Path(args.taxonomy_output), taxonomy)
    write_json(Path(args.mapping_output), mapping)
    write_json(Path(args.summary_output), summary)
    write_json(Path(args.review_output), review_queue)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
