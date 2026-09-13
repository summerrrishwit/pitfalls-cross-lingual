#!/usr/bin/env python3
"""Prepare legacy Chinese perturbations for a later Paths Not Taken experiment.

This is an offline, deterministic preparation step.  It does not tokenize
answers, run a model, collect hidden states, construct task vectors, or perform
an intervention.  Only questions matched to frozen canonical factual triples
are admitted.  Multiple perturbations of one source question remain grouped
under one independent base fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCHEMA_VERSION = "path-not-token-preparation-base-v1"
PERTURBATION_SCHEMA_VERSION = "path-not-token-preparation-perturbation-v1"
EXCLUSION_SCHEMA_VERSION = "path-not-token-preparation-exclusion-v1"
SUMMARY_VERSION = "path-not-token-preparation-summary-v1"
PROBE_RELATION_VERSION = "probe-relation-v1-pending"


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def group_source_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    required = {
        "question", "choices", "answer", "source", "oriquestion", "transori",
        "transchoices", "transanswer", "transquestion", "rate_ori", "rate_trans",
        "category",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Chinese.json row {index} is missing fields: {missing}")
        groups[normalize_text(row["oriquestion"])].append((index, row))
    for question_key, group in groups.items():
        answers = {normalize_text(row["answer"]) for _, row in group}
        target_answers = {normalize_text(row["transanswer"]) for _, row in group}
        if len(answers) != 1 or len(target_answers) != 1:
            raise ValueError(f"Conflicting answers for source question: {question_key}")
    return groups


def index_unique(records: Sequence[Dict[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = normalize_text(record.get(key_name))
        if not key:
            raise ValueError(f"Missing {key_name}")
        if key in output:
            raise ValueError(f"Duplicate {key_name}: {record.get(key_name)}")
        output[key] = record
    return output


def format_open_prompt(question: str, language: str) -> str:
    marker = "答案：" if language == "zh" else "Answer:"
    return f"{question.strip()}\n{marker}"


def join_segments(values: Iterable[Any]) -> str:
    return "\n\n".join(str(value).strip() for value in values if str(value or "").strip())


def base_id(source_id: str) -> str:
    return f"pnt_zh_{source_id}"


def variant_id(source_id: str, source_row_index: int) -> str:
    return f"{base_id(source_id)}_row_{source_row_index:06d}"


def _legacy_metrics(group: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    pair_counts = Counter(
        (float(row["rate_ori"]), float(row["rate_trans"])) for _, row in group
    )
    rates_en = [pair[0] for pair in pair_counts for _ in range(pair_counts[pair])]
    rates_zh = [pair[1] for pair in pair_counts for _ in range(pair_counts[pair])]
    return {
        "rate_pair_counts": [
            {"rate_en": pair[0], "rate_zh": pair[1], "count": count}
            for pair, count in sorted(pair_counts.items())
        ],
        "rate_en_mean": sum(rates_en) / len(rates_en),
        "rate_zh_mean": sum(rates_zh) / len(rates_zh),
        "pitfall_score_mean": (sum(rates_en) - sum(rates_zh)) / len(rates_en),
        "all_variants_are_cross_lingual_pitfalls": all(
            rate_en > rate_zh for rate_en, rate_zh in pair_counts
        ),
        "status": "legacy_generation_model_aggregate_not_current_mechanism_model_baseline",
    }


def _base_record(
    canonical: Dict[str, Any],
    review: Dict[str, Any],
    group: Sequence[Tuple[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    first = group[0][1]
    source_id = canonical["source_id"]
    answer_en = str(canonical["answer"]).strip()
    answer_zh = str(review["answer_zh"]).strip()
    return {
        "schema_version": BASE_SCHEMA_VERSION,
        "id": base_id(source_id),
        "language": "Chinese",
        "language_code": "zh",
        "source_id": source_id,
        "source_dataset": canonical["source_dataset"],
        "source_subset": canonical.get("source_subset"),
        "category": first["category"],
        "source_row_indices": [index for index, _ in group],
        "perturbation_count": len(group),
        "subject_en": canonical["subject"],
        "relation_en": canonical["relation_raw"],
        "answer_en": answer_en,
        "answer_aliases_en": unique_strings(
            [answer_en, canonical.get("source_answer"), first.get("answer")]
        ),
        "subject_target": review["subject_zh"],
        "relation_target": review["relation_zh"],
        "answer_target": answer_zh,
        "answer_aliases_target": unique_strings(
            [answer_zh, *(review.get("answer_aliases_zh") or []), first.get("transanswer")]
        ),
        "canonical_fact_en": canonical["canonical_fact"],
        "relation_signature_id": canonical.get("relation_signature_id"),
        "relation_normalized": canonical.get("relation_normalized"),
        "relation_normalization_status": canonical.get("normalization_status"),
        "probe_relation_id": None,
        "probe_relation_version": PROBE_RELATION_VERSION,
        "probe_relation_status": "pending_balanced_inventory_review",
        "prompt_en": canonical["prompt_en"],
        "prompt_target": review["completion_prompt_zh"],
        "prompt_format": "factual_completion",
        "open_question_en": first["oriquestion"].strip(),
        "open_question_target": first["transori"].strip(),
        "open_prompt_en": format_open_prompt(first["oriquestion"], "en"),
        "open_prompt_target": format_open_prompt(first["transori"], "zh"),
        "original_question_en": first["oriquestion"],
        "original_question_target": first["transori"],
        "choices_en": first["choices"],
        "choices_target": first["transchoices"],
        "factual_type": first["category"],
        "is_factual_recall": True,
        "legacy_metrics": _legacy_metrics(group),
        "admission": {
            "general_factual_recall_candidate": True,
            "prompt_ready": True,
            "relation_conditioned_mechanism_eligible": False,
            "path_not_token_experiment_ready": False,
            "blocking_reasons": [
                "probe_relation_not_frozen",
                "target_model_and_tokenizer_not_selected",
                "answer_tokenization_not_computed",
                "open_prompt_baseline_not_run",
            ],
        },
        "answer_tokenization": {
            "status": "pending_target_model_and_tokenizer",
            "model_id": None,
            "tokenizer_revision": None,
            "answer_token_ids_en": None,
            "answer_token_ids_target": None,
        },
        "mechanism_analysis": {
            "status": "not_run",
            "hidden_states_collected": False,
            "intervention_run": False,
        },
        "provenance": {
            "canonical_policy_version": canonical.get("canonical_policy_version"),
            "canonical_decision": canonical.get("canonical_decision"),
            "canonical_source_model": canonical.get("canonical_source_model"),
            "factual_prompt_version": canonical.get("factual_prompt_version"),
            "target_translation_review_version": review.get("review_version"),
            "target_translation_reviewer_type": review.get("reviewer_type"),
            "target_translation_not_human_gold": review.get("not_human_gold"),
        },
    }


def _perturbation_record(
    base: Dict[str, Any], source_row_index: int, row: Dict[str, Any]
) -> Dict[str, Any]:
    formatted_en = join_segments([row.get("prequestion"), row["oriquestion"], row.get("sufquestion")])
    formatted_zh = join_segments([row.get("transpre"), row["transori"], row.get("transsuf")])
    return {
        "schema_version": PERTURBATION_SCHEMA_VERSION,
        "id": variant_id(base["source_id"], source_row_index),
        "base_id": base["id"],
        "source_id": base["source_id"],
        "source_row_index": source_row_index,
        "source_dataset": base["source_dataset"],
        "category": base["category"],
        "subject_en": base["subject_en"],
        "relation_en": base["relation_en"],
        "subject_target": base["subject_target"],
        "relation_target": base["relation_target"],
        "probe_relation_id": base["probe_relation_id"],
        "answer_en": base["answer_en"],
        "answer_aliases_en": base["answer_aliases_en"],
        "answer_target": base["answer_target"],
        "answer_aliases_target": base["answer_aliases_target"],
        "clean_open_prompt_en": base["open_prompt_en"],
        "clean_open_prompt_target": base["open_prompt_target"],
        "perturbed_open_prompt_en": format_open_prompt(formatted_en, "en"),
        "perturbed_open_prompt_target": format_open_prompt(formatted_zh, "zh"),
        "perturbation_context_en": {
            "prefix": row.get("prequestion") or "",
            "suffix": row.get("sufquestion") or "",
        },
        "perturbation_context_target": {
            "prefix": row.get("transpre") or "",
            "suffix": row.get("transsuf") or "",
        },
        "source_perturbed_question_en_raw": row["question"],
        "source_perturbed_question_target_raw": row["transquestion"],
        "choices_in_prompt": False,
        "source_choices_en": row["choices"],
        "source_choices_target": row["transchoices"],
        "legacy_metrics": {
            "rate_en": float(row["rate_ori"]),
            "rate_zh": float(row["rate_trans"]),
            "pitfall_score": float(row["rate_ori"]) - float(row["rate_trans"]),
            "status": "legacy_generation_model_aggregate_not_current_mechanism_model_baseline",
        },
        "experimental_status": "prepared_not_evaluated",
    }


def _exclusion_record(
    group: Sequence[Tuple[int, Dict[str, Any]]],
    extractions: Mapping[str, Mapping[str, Dict[str, Any]]],
    decisions_by_id: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    first_index, first = group[0]
    question_key = normalize_text(first["oriquestion"])
    annotations: Dict[str, Any] = {}
    source_id = None
    for model_name, by_question in extractions.items():
        record = by_question.get(question_key)
        if record:
            source_id = source_id or record.get("source_id")
            annotations[model_name] = {
                "extraction_status": record.get("extraction_status"),
                "exclusion_reason": record.get("exclusion_reason"),
                "subject": record.get("subject"),
                "relation_raw": record.get("relation_raw"),
                "answer": record.get("answer"),
            }
    decision = decisions_by_id.get(str(source_id)) if source_id else None
    digest = hashlib.sha256(question_key.encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": EXCLUSION_SCHEMA_VERSION,
        "id": f"pnt_zh_excluded_{digest}",
        "source_id": source_id,
        "source_dataset": first["source"],
        "category": first["category"],
        "source_row_indices": [index for index, _ in group],
        "perturbation_count": len(group),
        "original_question_en": first["oriquestion"],
        "original_question_target": first["transori"],
        "answer_en": first["answer"],
        "answer_target": first["transanswer"],
        "admission_status": "excluded_not_frozen_canonical_atomic_fact",
        "model_extraction_annotations": annotations,
        "canonical_decision": decision.get("canonical_decision") if decision else None,
        "canonical_reason": decision.get("canonical_reason") if decision else None,
        "codex_adjudication": decision.get("codex_adjudication") if decision else None,
        "first_source_row_index": first_index,
    }


def build_records(
    chinese_rows: Sequence[Dict[str, Any]],
    canonical_prompts: Sequence[Dict[str, Any]],
    review_document: Dict[str, Any],
    model_extractions: Mapping[str, Sequence[Dict[str, Any]]],
    canonical_decisions: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups = group_source_rows(chinese_rows)
    canonical_by_question = {
        normalize_text(record["source_question"]): record
        for record in canonical_prompts
        if record.get("factual_prompt_status") == "generated"
    }
    review_by_id = index_unique(review_document["records"], "source_id")
    extraction_indexes = {
        model_name: {
            normalize_text(record["source_question"]): record for record in records
        }
        for model_name, records in model_extractions.items()
    }
    decisions_by_id = {record["source_id"]: record for record in canonical_decisions}

    bases: List[Dict[str, Any]] = []
    variants: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    used_review_ids = set()
    for question_key, group in groups.items():
        canonical = canonical_by_question.get(question_key)
        if canonical is None:
            exclusions.append(_exclusion_record(group, extraction_indexes, decisions_by_id))
            continue
        review = review_by_id.get(normalize_text(canonical["source_id"]))
        if review is None:
            raise ValueError(f"Missing Chinese review for {canonical['source_id']}")
        review = {
            **review,
            "review_version": review_document["review_version"],
            "reviewer_type": review_document["reviewer_type"],
            "not_human_gold": review_document["not_human_gold"],
        }
        used_review_ids.add(normalize_text(canonical["source_id"]))
        base = _base_record(canonical, review, group)
        bases.append(base)
        variants.extend(_perturbation_record(base, index, row) for index, row in group)

    unused_review_ids = sorted(set(review_by_id) - used_review_ids)
    if unused_review_ids:
        raise ValueError(f"Unused Chinese review records: {unused_review_ids}")
    bases.sort(key=lambda record: record["source_id"])
    variants.sort(key=lambda record: record["source_row_index"])
    exclusions.sort(key=lambda record: record["first_source_row_index"])
    return bases, variants, exclusions


def build_summary(
    chinese_path: Path,
    canonical_path: Path,
    review_path: Path,
    bases: Sequence[Dict[str, Any]],
    variants: Sequence[Dict[str, Any]],
    exclusions: Sequence[Dict[str, Any]],
    total_source_rows: int,
) -> Dict[str, Any]:
    return {
        "summary_version": SUMMARY_VERSION,
        "scope": "offline preparation only; no PATH_not_token model experiment was run",
        "input_files": {
            str(chinese_path.relative_to(PROJECT_ROOT)): sha256_file(chinese_path),
            str(canonical_path.relative_to(PROJECT_ROOT)): sha256_file(canonical_path),
            str(review_path.relative_to(PROJECT_ROOT)): sha256_file(review_path),
        },
        "counts": {
            "source_row_count": total_source_rows,
            "unique_base_question_count": len(bases) + len(exclusions),
            "accepted_base_fact_count": len(bases),
            "accepted_perturbation_instance_count": len(variants),
            "excluded_base_question_count": len(exclusions),
            "excluded_source_row_count": total_source_rows - len(variants),
        },
        "accepted_base_counts_by_source": dict(
            sorted(Counter(record["source_dataset"] for record in bases).items())
        ),
        "relation_normalization_status_counts": dict(
            sorted(Counter(record["relation_normalization_status"] for record in bases).items())
        ),
        "probe_relation_status_counts": dict(
            sorted(Counter(record["probe_relation_status"] for record in bases).items())
        ),
        "prompt_contract": {
            "choices_embedded": False,
            "formats": ["factual_completion", "open_question_with_answer_marker"],
            "languages": ["en", "zh"],
            "independent_unit": "base_fact",
            "perturbations_are_repeated_variants": True,
        },
        "not_performed": [
            "probe_relation_inventory_freeze",
            "target_model_selection",
            "answer_tokenization",
            "open_prompt_baseline",
            "hidden_state_collection",
            "task_vector_construction",
            "intervention",
        ],
    }


def build_relation_inventory(bases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in bases:
        groups[record["relation_signature_id"]].append(record)
    relations = []
    for signature_id, records in groups.items():
        first = records[0]
        relations.append(
            {
                "relation_signature_id": signature_id,
                "relation_en": first["relation_en"],
                "relation_target": first["relation_target"],
                "relation_normalized": first["relation_normalized"],
                "relation_normalization_status": first["relation_normalization_status"],
                "probe_relation_id": None,
                "probe_relation_version": PROBE_RELATION_VERSION,
                "probe_relation_status": "pending_balanced_inventory_review",
                "base_fact_count": len(records),
                "perturbation_count": sum(record["perturbation_count"] for record in records),
                "base_ids": [record["id"] for record in records],
                "direction_examples": [
                    {
                        "subject_en": record["subject_en"],
                        "answer_en": record["answer_en"],
                        "subject_target": record["subject_target"],
                        "answer_target": record["answer_target"],
                    }
                    for record in records
                ],
            }
        )
    relations.sort(key=lambda record: record["relation_signature_id"])
    return {
        "inventory_version": "path-not-token-probe-relation-inventory-v1",
        "source_base_schema_version": BASE_SCHEMA_VERSION,
        "relation_signature_count": len(relations),
        "frozen_probe_relation_count": 0,
        "relation_conditioned_experiment_ready": False,
        "blocking_reason": "all candidate relation signatures require balanced inventory review",
        "relations": relations,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    return "\n".join(
        [
            "# Chinese PATH_not_token preparation v1",
            "",
            "This directory contains deterministic preprocessing artifacts only. No model, hidden-state, task-vector, or intervention experiment was run.",
            "",
            "## Counts",
            "",
            f"- Source rows: {counts['source_row_count']}",
            f"- Unique source questions: {counts['unique_base_question_count']}",
            f"- Frozen canonical base facts: {counts['accepted_base_fact_count']}",
            f"- Retained perturbation instances: {counts['accepted_perturbation_instance_count']}",
            f"- Excluded source questions: {counts['excluded_base_question_count']}",
            f"- Excluded source rows: {counts['excluded_source_row_count']}",
            "",
            "## Files",
            "",
            "- `base_facts.jsonl`: one row per independent canonical fact, with bilingual completion and open-question prompts.",
            "- `perturbation_instances.jsonl`: all retained legacy perturbations linked to their base fact; choices are provenance only and are not embedded in prompts.",
            "- `relation_inventory.json`: extracted relation signatures and their current unfrozen probe-relation status.",
            "- `excluded_base_questions.jsonl`: source questions that did not pass the frozen canonical atomic-fact gate.",
            "- `summary.json`: counts, hashes, prompt contract, and explicit not-run stages.",
            "",
            "## Boundary",
            "",
            "`probe_relation_id` remains null and answer-token fields remain pending. The records are ready for downstream tokenization and baseline screening, but they are not yet admitted to a relation-conditioned PATH_not_token mechanism experiment.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese", type=Path, default=PROJECT_ROOT / "data/Chinese.json")
    parser.add_argument(
        "--canonical-prompts",
        type=Path,
        default=PROJECT_ROOT / "data_processed/factual_triples/triple-full-v1/canonical-v2/factual_prompts.jsonl",
    )
    parser.add_argument(
        "--canonical-decisions",
        type=Path,
        default=PROJECT_ROOT / "data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_decisions.jsonl",
    )
    parser.add_argument(
        "--qwen-extractions",
        type=Path,
        default=PROJECT_ROOT / "data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/triple_extractions.jsonl",
    )
    parser.add_argument(
        "--sensenova-extractions",
        type=Path,
        default=PROJECT_ROOT / "data_processed/factual_triples/triple-full-v1/models/sensenova-6.7-flash-lite/triple_extractions.jsonl",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=PROJECT_ROOT / "configs/path_not_token_chinese_v1_review.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_processed/path_not_token/chinese-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chinese_rows = read_json(args.chinese)
    if not isinstance(chinese_rows, list):
        raise ValueError("Chinese.json must contain a JSON array")
    bases, variants, exclusions = build_records(
        chinese_rows,
        read_jsonl(args.canonical_prompts),
        read_json(args.review),
        {
            "qwen3.7-plus": read_jsonl(args.qwen_extractions),
            "sensenova-6.7-flash-lite": read_jsonl(args.sensenova_extractions),
        },
        read_jsonl(args.canonical_decisions),
    )
    write_jsonl(args.output_dir / "base_facts.jsonl", bases)
    write_jsonl(args.output_dir / "perturbation_instances.jsonl", variants)
    write_jsonl(args.output_dir / "excluded_base_questions.jsonl", exclusions)
    write_json(args.output_dir / "relation_inventory.json", build_relation_inventory(bases))
    summary = build_summary(
        args.chinese,
        args.canonical_prompts,
        args.review,
        bases,
        variants,
        exclusions,
        len(chinese_rows),
    )
    summary["output_files"] = {
        name: sha256_file(args.output_dir / name)
        for name in (
            "base_facts.jsonl",
            "perturbation_instances.jsonl",
            "relation_inventory.json",
            "excluded_base_questions.jsonl",
        )
    }
    write_json(args.output_dir / "summary.json", summary)
    report_path = args.output_dir / "README.md"
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
