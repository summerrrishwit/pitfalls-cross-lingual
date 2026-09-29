#!/usr/bin/env python3
"""Freeze behavior-blind bilingual unrelated-fact regression controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


CONTRACT_VERSION = "qwen3-unrelated-fact-controls-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def binding(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return "".join(character for character in text if character.isalnum())


def deduplicate(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def natural_prompt(prompt: str, language: str) -> str:
    instruction = (
        "请补全以下事实，只输出缺失的答案，不要解释：\n"
        if language == "zh"
        else "Complete the following factual statement with only the missing answer:\n"
    )
    return instruction + prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-facts", type=Path, required=True)
    parser.add_argument("--frozen-cohort", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--g0b-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def materialize(
    source_facts_path: Path,
    frozen_cohort_path: Path,
    review_path: Path,
    g0b_gate_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    paths = {
        "source_facts": source_facts_path.resolve(),
        "frozen_cohort": frozen_cohort_path.resolve(),
        "codex_unrelatedness_review": review_path.resolve(),
        "g0b_gate": g0b_gate_path.resolve(),
    }
    sources = read_jsonl(paths["source_facts"])
    cohort = read_jsonl(paths["frozen_cohort"])
    review = read_json(paths["codex_unrelatedness_review"])
    g0b = read_json(paths["g0b_gate"])
    if len(sources) != 13 or len({row["id"] for row in sources}) != 13:
        raise ValueError("expected the complete 13-fact Chinese-v1 control source")
    if len(cohort) != 160 or len({row["base_fact_id"] for row in cohort}) != 160:
        raise ValueError("expected the frozen 160-fact cohort")
    if g0b.get("gate_passed") is not True:
        raise ValueError("G0B must be frozen before binding the control threshold")
    if review.get("reviewer_type") != "codex_proxy" or review.get("human_gold") is not False:
        raise ValueError("unrelatedness review must remain Codex proxy evidence")
    decisions = {
        str(row["control_id"]): row
        for row in review.get("decisions", [])
    }
    if len(decisions) != len(review.get("decisions", [])):
        raise ValueError("duplicate control review decision")
    if set(decisions) != {str(row["id"]) for row in sources}:
        raise ValueError("review decisions do not cover all source controls")
    if any(row.get("decision") != "accept_as_unrelated_control" for row in decisions.values()):
        raise ValueError("every frozen control must pass bounded unrelatedness review")

    cohort_source_ids = {str(row["source_id"]) for row in cohort}
    cohort_prompt_en = {normalize(row["prompt_en"]) for row in cohort}
    cohort_prompt_zh = {normalize(row["prompt_zh"]) for row in cohort}
    cohort_answer_en = {normalize(row["answer_en"]) for row in cohort}
    cohort_answer_zh = {normalize(row["answer_zh"]) for row in cohort}
    rows = []
    for source in sorted(sources, key=lambda row: str(row["id"])):
        control_id = str(source["id"])
        admission = source.get("admission", {})
        provenance = source.get("provenance", {})
        if admission.get("general_factual_recall_candidate") is not True or admission.get("prompt_ready") is not True:
            raise ValueError(f"control source is not prompt-ready factual recall: {control_id}")
        if provenance.get("canonical_decision") != "keep":
            raise ValueError(f"control source lacks a keep decision: {control_id}")
        if provenance.get("target_translation_reviewer_type") != "codex_proxy":
            raise ValueError(f"control source lacks Codex translation review: {control_id}")
        if str(source["source_id"]) in cohort_source_ids:
            raise ValueError(f"control source ID overlaps the frozen cohort: {control_id}")
        prompt_pairs = {
            "en": (str(source["prompt_en"]), str(source["answer_en"]), source.get("answer_aliases_en", [])),
            "zh": (str(source["prompt_target"]), str(source["answer_target"]), source.get("answer_aliases_target", [])),
        }
        if normalize(prompt_pairs["en"][0]) in cohort_prompt_en or normalize(prompt_pairs["zh"][0]) in cohort_prompt_zh:
            raise ValueError(f"control prompt overlaps the frozen cohort: {control_id}")
        if normalize(prompt_pairs["en"][1]) in cohort_answer_en or normalize(prompt_pairs["zh"][1]) in cohort_answer_zh:
            raise ValueError(f"control answer overlaps the frozen cohort: {control_id}")
        for language, (prompt, answer, aliases) in prompt_pairs.items():
            user_prompt = natural_prompt(prompt, language)
            rows.append({
                "schema_version": "qwen3-unrelated-fact-control-input-v1",
                "control_input_id": sha256_bytes(
                    canonical_json([CONTRACT_VERSION, control_id, language]).encode("utf-8")
                )[:24],
                "control_id": control_id,
                "source_id": source["source_id"],
                "language": language,
                "prompt": prompt,
                "user_prompt": user_prompt,
                "user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
                "answer": answer,
                "answer_aliases": deduplicate([answer, *aliases]),
                "role": "unrelated_fact_regression_only",
                "current_160_cohort_member": False,
                "vector_source": False,
                "behavior_selection_used": False,
                "review_decision": decisions[control_id]["decision"],
                "chat_render_contract": {
                    "messages": [{"role": "user", "content": user_prompt}],
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                    "status": "pending_exact_tokenizer_render",
                },
            })
    if len(rows) != 26:
        raise ValueError("expected 26 bilingual unrelated-control inputs")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    controls_path = output_dir / "unrelated_fact_controls.jsonl"
    write_jsonl_atomic(controls_path, rows)
    threshold = g0b["resolved_protocol"]["retention_thresholds"][
        "unrelated_fact_accuracy_drop_max"
    ]
    manifest = {
        "schema_version": "qwen3-unrelated-fact-control-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "behavior_blind_unrelated_control_frozen_pending_exact_tokenizer_audit",
        "scope": "regression_control_only_not_cohort_or_vector_source",
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "counts": {
            "control_facts": 13,
            "control_inputs": 26,
            "language_counts": dict(sorted(Counter(row["language"] for row in rows).items())),
            "current_cohort_facts_used": 0,
            "validation_facts_used": 0,
            "sealed_facts_used": 0,
        },
        "selection_contract": {
            "rule": review["selection_rule"],
            "current_Qwen3_behavior_used": False,
            "legacy_behavior_metadata_used": False,
            "bounded_semantic_review": review["review_scope"],
            "exact_source_prompt_answer_overlap_count": 0,
            "vector_source": False,
            "repair_rule_selection_role": "retention_constraint_only",
        },
        "evaluation_contract": {
            "languages": ["en", "zh"],
            "metric": "paired_alias_aware_accuracy_drop_from_no_intervention",
            "accuracy_drop_max": threshold,
            "small_n_policy": (
                "Report all 13 paired fact outcomes and the exact count; with n=13, the 0.02 "
                "threshold permits zero newly harmed facts. Do not claim population-level safety."
            ),
        },
        "authorization_state": {
            "text_control_set_frozen": True,
            "exact_tokenizer_render_authorized": True,
            "model_execution_authorized": False,
            "vector_source_authorized": False,
            "validation_authorized": False,
            "sealed_authorized": False,
        },
        "inputs": {name: binding(path) for name, path in paths.items()},
        "artifacts": {"unrelated_fact_controls": binding(controls_path)},
        "limitations": [
            "This is a 13-fact bounded regression panel, not an expansion of the formal 160-fact cohort.",
            "Its source translations and unrelatedness decisions are Codex proxy review, not human gold.",
            "No Qwen3 behavior was used to select these controls, and no Qwen3 control baseline has been run.",
        ],
    }
    manifest_path = output_dir / "unrelated_fact_control_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = materialize(
        args.source_facts,
        args.frozen_cohort,
        args.review,
        args.g0b_gate,
        args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
