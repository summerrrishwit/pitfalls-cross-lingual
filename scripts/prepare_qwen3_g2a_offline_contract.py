#!/usr/bin/env python3
"""Materialize the Development-only, pre-tokenization G2A PNT contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CONTRACT_VERSION = "qwen3-g2a-option-free-pnt-contract-v1"
FOLD_IDS = tuple(f"dev-fold-{index:02d}" for index in range(1, 5))
LANGUAGES = ("en", "zh")
CHOICES = ("A", "B", "C")
VECTOR_SOURCE_KINDS = (
    "task_icl_recall_en",
    "task_icl_recall_zh",
    "translation_en_to_zh",
    "fact_recall_reference_en",
    "fact_recall_reference_zh",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


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


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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


def normalize_alias(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return "".join(character for character in value if character.isalnum())


def deduplicate(values: Iterable[Any]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def natural_prompt(fact: Mapping[str, Any], language: str) -> str:
    instruction = (
        "请补全以下事实，只输出缺失的答案，不要解释：\n"
        if language == "zh"
        else "Complete the following factual statement with only the missing answer:\n"
    )
    return instruction + str(fact[f"prompt_{language}"])


def option_free_prompt(fact: Mapping[str, Any], stimulus: Mapping[str, Any]) -> str:
    language = str(stimulus["language"])
    context = str(stimulus["context"]).strip()
    base = natural_prompt(fact, language)
    if not context:
        return base
    prefix = f"背景信息：\n{context}\n\n" if language == "zh" else f"Context:\n{context}\n\n"
    return prefix + base


def answer_groups(fact: Mapping[str, Any]) -> List[Dict[str, Any]]:
    variant_by_id = {
        str(variant["distractor_id"]): variant
        for variant in fact["static_mcq"]["variants"]
    }
    groups = []
    normalized_groups = []
    for option in fact["static_mcq"]["options"]:
        option_id = str(option["option_id"])
        kind = str(option["kind"])
        if kind == "gold":
            aliases_en = deduplicate(
                [fact["answer_en"], *fact.get("answer_aliases_en", [])]
            )
            aliases_zh = deduplicate(
                [fact["answer_zh"], *fact.get("answer_aliases_zh", [])]
            )
        else:
            variant = variant_by_id[option_id]
            aliases_en = deduplicate(
                [option["text_en"], *variant.get("answer_aliases_en", [])]
            )
            aliases_zh = deduplicate(
                [option["text_zh"], *variant.get("answer_aliases_zh", [])]
            )
        groups.append({
            "option_id": option_id,
            "choice": str(option["choice"]),
            "kind": kind,
            "aliases_en": aliases_en,
            "aliases_zh": aliases_zh,
        })
        normalized_groups.append({
            normalize_alias(value)
            for value in [*aliases_en, *aliases_zh]
            if normalize_alias(value)
        })
    for left in range(len(normalized_groups)):
        for right in range(left + 1, len(normalized_groups)):
            if normalized_groups[left] & normalized_groups[right]:
                raise ValueError(
                    f"answer aliases overlap between option groups: {fact['base_fact_id']}"
                )
    return groups


def icl_prompt(
    target: Mapping[str, Any],
    donors: Sequence[Mapping[str, Any]],
    language: str,
) -> str:
    instruction = (
        "请完成每个事实，只输出缺失的答案。"
        if language == "zh"
        else "Complete each factual statement with only the missing answer."
    )
    blocks = [instruction]
    for donor in donors:
        blocks.append(
            f"Q: {donor[f'prompt_{language}']}\nA: {donor[f'answer_{language}']}"
        )
    blocks.append(f"Q: {target[f'prompt_{language}']}\nA:")
    return "\n\n".join(blocks)


def translation_prompt(fact: Mapping[str, Any]) -> str:
    return (
        "Translate the following factual answer from English into Chinese. "
        "Return only the translated answer.\n"
        f"English: {fact['answer_en']}\nChinese:"
    )


def deterministic_donors(
    target: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    partition_id: str,
    language: str,
) -> List[Mapping[str, Any]]:
    eligible = [
        row
        for row in source_rows
        if row["base_fact_id"] != target["base_fact_id"]
        and row["probe_relation_id"] == target["probe_relation_id"]
        and row["leakage_component_id"] != target["leakage_component_id"]
    ]
    eligible.sort(
        key=lambda row: sha256_value(
            [
                CONTRACT_VERSION,
                partition_id,
                language,
                target["base_fact_id"],
                row["base_fact_id"],
            ]
        )
    )
    if len(eligible) < 5:
        raise ValueError(
            f"fewer than five eligible ICL donors: {partition_id}:{target['base_fact_id']}:{language}"
        )
    return eligible[:5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--static-freeze-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--g0b-gate", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--attribution-audit", type=Path, required=True)
    parser.add_argument("--natural-summary", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--unrelated-control-manifest", type=Path, required=True)
    parser.add_argument("--protocol-document", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        key: value.resolve()
        for key, value in {
            "input_bundle": args.input_bundle,
            "static_freeze_manifest": args.static_freeze_manifest,
            "fold_manifest": args.fold_manifest,
            "g0b_gate": args.g0b_gate,
            "model_manifest": args.model_manifest,
            "attribution_audit": args.attribution_audit,
            "natural_summary": args.natural_summary,
            "semantic_review": args.semantic_review,
            "unrelated_control_manifest": args.unrelated_control_manifest,
            "protocol_document": args.protocol_document,
        }.items()
    }
    static_freeze = read_json(paths["static_freeze_manifest"])
    folds = read_json(paths["fold_manifest"])
    g0b = read_json(paths["g0b_gate"])
    model = read_json(paths["model_manifest"])
    attribution = read_json(paths["attribution_audit"])
    natural = read_json(paths["natural_summary"])
    semantic = read_json(paths["semantic_review"])
    unrelated = read_json(paths["unrelated_control_manifest"])
    if static_freeze.get("status") != "g0a_static_stimulus_frozen":
        raise ValueError("G0A static bundle is not frozen")
    if static_freeze["frozen_bundle"]["sha256"] != sha256_file(paths["input_bundle"]):
        raise ValueError("input bundle changed after G0A")
    if folds.get("status") != "frozen" or folds.get("fold_contract", {}).get("fold_count") != 4:
        raise ValueError("Development four-fold assignment is not frozen")
    if g0b.get("gate_passed") is not True or model.get("status") != "exact_hf_identity_frozen":
        raise ValueError("exact-HF identity/render contract is not frozen")
    if g0b["exact_hf_model_manifest"]["sha256"] != sha256_file(paths["model_manifest"]):
        raise ValueError("G0B model binding changed")
    if attribution.get("status") != "passed_frozen_runtime_reproduction":
        raise ValueError("MCQ attribution stability audit has not passed")
    if natural.get("status") != "complete" or natural.get("result_count") != 192:
        raise ValueError("Development natural baseline is incomplete")
    if semantic.get("status") not in {"complete", "complete_with_deferrals"}:
        raise ValueError("natural semantic review is incomplete")
    if unrelated.get("status") != "behavior_blind_unrelated_control_frozen_pending_exact_tokenizer_audit":
        raise ValueError("behavior-blind unrelated-fact control set is not frozen")
    if unrelated.get("counts", {}).get("control_facts") != 13:
        raise ValueError("unexpected unrelated-fact control count")
    if unrelated.get("authorization_state", {}).get("model_execution_authorized") is not False:
        raise ValueError("unrelated-fact controls must remain behavior-blind")

    all_rows = read_jsonl(paths["input_bundle"])
    development = [row for row in all_rows if row.get("split_assignment") == "development"]
    if len(all_rows) != 160 or len(development) != 96:
        raise ValueError("expected 160 total facts and 96 Development facts")
    if len({row["base_fact_id"] for row in development}) != 96:
        raise ValueError("Development base_fact_id values are not unique")
    fact_by_id = {row["base_fact_id"]: row for row in development}
    assignment_by_id = {
        row["base_fact_id"]: row["development_fold_id"]
        for row in folds["assignments"]
    }
    if set(assignment_by_id) != set(fact_by_id):
        raise ValueError("fold assignment IDs differ from Development IDs")

    option_free_rows = []
    for fact in sorted(development, key=lambda row: row["base_fact_id"]):
        groups = answer_groups(fact)
        for stimulus in fact["static_mcq"]["behavior_inputs"]:
            language = str(stimulus["language"])
            user_prompt = option_free_prompt(fact, stimulus)
            option_free_rows.append({
                "schema_version": "qwen3-option-free-three-arm-input-v1",
                "option_free_id": sha256_value(
                    [CONTRACT_VERSION, stimulus["stimulus_id"], "option-free"]
                )[:24],
                "base_fact_id": fact["base_fact_id"],
                "source_id": fact["source_id"],
                "split_assignment": "development",
                "development_fold_id": assignment_by_id[fact["base_fact_id"]],
                "leakage_component_id": fact["leakage_component_id"],
                "probe_relation_id": fact["probe_relation_id"],
                "answer_type": fact["answer_type"],
                "source_stimulus_id": stimulus["stimulus_id"],
                "language": language,
                "variant": stimulus["arm"],
                "designated_distractor_id": stimulus.get("distractor_id"),
                "context": stimulus["context"],
                "factual_completion_prompt": stimulus["prompt"],
                "user_prompt": user_prompt,
                "user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
                "answer_groups": groups,
                "gold_option_id": "gold",
                "target_option_id": (
                    stimulus.get("distractor_id")
                    if stimulus["arm"] == "targeted"
                    else None
                ),
                "chat_render_contract": {
                    "messages": [{"role": "user", "content": user_prompt}],
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                    "status": "pending_exact_tokenizer_render",
                },
                "evaluation_contract": {
                    "generation": "greedy_temperature_0",
                    "candidate_scoring": "complete_answer_sequence_log_probability_over_all_frozen_aliases",
                    "targeted_open_transfer": "generated_or_top_scored_designated_distractor_alias",
                    "statistical_unit": "base_fact_id",
                },
            })
    if len(option_free_rows) != 960:
        raise ValueError("option-free coverage is not exactly 960 Development inputs")

    partitions: Dict[str, List[Dict[str, Any]]] = {}
    for fold_id in FOLD_IDS:
        partitions[fold_id] = [
            row
            for row in development
            if assignment_by_id[row["base_fact_id"]] != fold_id
        ]
    partitions["final-development-all"] = list(development)

    vector_rows = []
    for partition_id, source_rows in partitions.items():
        source_ids = {row["base_fact_id"] for row in source_rows}
        expected_count = 96 if partition_id == "final-development-all" else 72
        if len(source_ids) != expected_count:
            raise ValueError(f"invalid source count for {partition_id}")
        for fact in sorted(source_rows, key=lambda row: row["base_fact_id"]):
            for language in LANGUAGES:
                donors = deterministic_donors(fact, source_rows, partition_id, language)
                prompt = icl_prompt(fact, donors, language)
                vector_rows.append({
                    "schema_version": "qwen3-pnt-vector-source-prompt-v1",
                    "vector_source_id": sha256_value(
                        [CONTRACT_VERSION, partition_id, fact["base_fact_id"], language, "task_icl_recall"]
                    )[:24],
                    "source_partition_id": partition_id,
                    "excluded_evaluation_fold_id": (
                        partition_id if partition_id in FOLD_IDS else None
                    ),
                    "base_fact_id": fact["base_fact_id"],
                    "leakage_component_id": fact["leakage_component_id"],
                    "probe_relation_id": fact["probe_relation_id"],
                    "kind": f"task_icl_recall_{language}",
                    "language": language,
                    "icl_shot_count": 5,
                    "icl_donor_base_fact_ids": [row["base_fact_id"] for row in donors],
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "expected_answer": fact[f"answer_{language}"],
                    "hook_position": "last_rendered_prompt_token_before_answer_continuation",
                    "render_status": "pending_exact_tokenizer_render",
                })
            translation = translation_prompt(fact)
            vector_rows.append({
                "schema_version": "qwen3-pnt-vector-source-prompt-v1",
                "vector_source_id": sha256_value(
                    [CONTRACT_VERSION, partition_id, fact["base_fact_id"], "translation_en_to_zh"]
                )[:24],
                "source_partition_id": partition_id,
                "excluded_evaluation_fold_id": (
                    partition_id if partition_id in FOLD_IDS else None
                ),
                "base_fact_id": fact["base_fact_id"],
                "leakage_component_id": fact["leakage_component_id"],
                "probe_relation_id": fact["probe_relation_id"],
                "kind": "translation_en_to_zh",
                "language": "zh",
                "icl_shot_count": 0,
                "icl_donor_base_fact_ids": [],
                "prompt": translation,
                "prompt_sha256": sha256_bytes(translation.encode("utf-8")),
                "expected_answer": fact["answer_zh"],
                "hook_position": "last_rendered_prompt_token_before_answer_continuation",
                "render_status": "pending_exact_tokenizer_render",
            })
            for language in LANGUAGES:
                prompt = natural_prompt(fact, language)
                vector_rows.append({
                    "schema_version": "qwen3-pnt-vector-source-prompt-v1",
                    "vector_source_id": sha256_value(
                        [CONTRACT_VERSION, partition_id, fact["base_fact_id"], language, "fact_recall_reference"]
                    )[:24],
                    "source_partition_id": partition_id,
                    "excluded_evaluation_fold_id": (
                        partition_id if partition_id in FOLD_IDS else None
                    ),
                    "base_fact_id": fact["base_fact_id"],
                    "leakage_component_id": fact["leakage_component_id"],
                    "probe_relation_id": fact["probe_relation_id"],
                    "kind": f"fact_recall_reference_{language}",
                    "language": language,
                    "icl_shot_count": 0,
                    "icl_donor_base_fact_ids": [],
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "expected_answer": fact[f"answer_{language}"],
                    "hook_position": "last_rendered_prompt_token_before_answer_continuation",
                    "render_status": "pending_exact_tokenizer_render",
                })
    if len(vector_rows) != 1920:
        raise ValueError("vector-source coverage is not exactly 1,920 prompts")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    option_path = output_dir / "option_free_three_arm_inputs.jsonl"
    vector_path = output_dir / "vector_source_prompt_specs.jsonl"
    write_jsonl_atomic(option_path, option_free_rows)
    write_jsonl_atomic(vector_path, vector_rows)

    fold_source_counts = Counter(row["source_partition_id"] for row in vector_rows)
    kind_counts = Counter(row["kind"] for row in vector_rows)
    option_counts = Counter(
        (row["language"], row["variant"])
        for row in option_free_rows
    )
    scale_grid = list(g0b["resolved_protocol"]["scale_grid"])
    manifest = {
        "schema_version": "qwen3-g2a-offline-contract-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "offline_contract_prepared_pending_exact_tokenizer_audit",
        "scope": "development_only_option_free_and_vector_source_preparation",
        "claim_boundary": (
            "This package freezes text-level prompts and estimators only. It contains no new model "
            "outputs, hidden states, vectors, interventions, Validation, or Sealed measurements."
        ),
        "pnt_reference": {
            "repository": "https://github.com/mlu108/paths_not_taken.git",
            "commit": "0318194d41e8175c134a7fd83394b328bec11544",
            "adaptation": (
                "Use the reference five-shot mean task-vector and translation-minus-recall ideas, "
                "but express every vector in the frozen Qwen3 resid_pre coordinate system."
            ),
        },
        "counts": {
            "development_facts": 96,
            "validation_facts_used": 0,
            "sealed_facts_used": 0,
            "option_free_inputs": len(option_free_rows),
            "option_free_arm_counts": {
                f"{language}_{variant}": count
                for (language, variant), count in sorted(option_counts.items())
            },
            "vector_source_prompts": len(vector_rows),
            "vector_source_kind_counts": dict(sorted(kind_counts.items())),
            "vector_source_partition_counts": dict(sorted(fold_source_counts.items())),
        },
        "prompt_contract": {
            "chat_messages": "one user message",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "original_prompt_identity": "byte-identical user content to natural-baseline-v4",
            "neutral_targeted_difference": "context text only within the same language and distractor variant",
            "options_in_prompt": False,
            "target_answer_in_targeted_context": True,
            "candidate_scoring": "complete sequence log probability for all frozen aliases of gold and both distractors",
        },
        "vector_estimators": {
            "coordinate_system": "resid_pre[layer_id] at the last rendered prompt token",
            "candidate_layer_ids": list(range(1, 36)),
            "accumulation_dtype": "float32",
            "stored_dtype": "float32",
            "task_vector": (
                "For each source partition and layer, average the EN and ZH five-shot factual-recall "
                "states within each base_fact_id, then average the 72 fact means. Donors are five "
                "deterministically ordered, same-relation, different-component source facts."
            ),
            "translation_difference_vector": (
                "For each source partition and layer, subtract the fact-balanced mean of EN and ZH "
                "zero-shot factual-recall states from the mean EN-to-ZH explicit-translation state."
            ),
            "behavior_label_filtering": False,
            "variant_weighting": "within-fact mean before across-fact aggregation",
            "final_vector_source": "all 96 Development facts only after the repair rule is frozen",
        },
        "oof_contract": {
            "outer_folds": list(FOLD_IDS),
            "source_facts_per_fold": 72,
            "evaluation_facts_per_fold": 24,
            "no_evaluation_fact_in_its_vector_source": True,
            "layer_grid": list(range(1, 36)),
            "scale_grid": scale_grid,
            "conditions": [
                "no_intervention",
                "norm_matched_random",
                "task_only",
                "difference_only",
                "combined",
            ],
            "combined_policy": (
                "Combine the task-only and difference-only layer/scale settings selected from pooled "
                "out-of-fold Development results; do not search a Cartesian combined grid."
            ),
            "trigger": (
                "Apply a selected intervention to every Chinese factual-completion condition in the "
                "held-out fold, never only to observed failures; English is measured without repair."
            ),
            "generation_hook": (
                "Add the vector at resid_pre[layer_id] to the current last token at every autoregressive step."
            ),
            "random_control": (
                "Five deterministic Gaussian directions per fold/layer, each normalized to the source "
                "vector L2 norm; seeds are SHA-256-derived from contract, fold, vector type, layer, and replicate."
            ),
            "selection_order": [
                "satisfy Original, Neutral, English, resistant-control, and unrelated-control retention",
                "maximize fact-weighted ZH Targeted gold-minus-designated-distractor margin improvement",
                "outperform no-intervention and the mean norm-matched-random control",
                "break ties by smaller absolute scale, then lower layer id, then vector type task/difference/combined",
            ],
        },
        "retention_thresholds": {
            "original_accuracy_drop_max": g0b["resolved_protocol"]["retention_thresholds"]["original_accuracy_drop_max"],
            "neutral_accuracy_drop_max": g0b["resolved_protocol"]["retention_thresholds"]["neutral_accuracy_drop_max"],
            "english_accuracy_drop_max": g0b["resolved_protocol"]["retention_thresholds"]["english_accuracy_drop_max"],
            "unrelated_fact_accuracy_drop_max": g0b["resolved_protocol"]["retention_thresholds"]["unrelated_fact_accuracy_drop_max"],
            "control_harm_threshold": g0b["resolved_protocol"]["control_harm_threshold"],
        },
        "natural_semantic_deferral_policy": {
            "source_review_status": semantic["status"],
            "deferred_record_count": int(semantic.get("decision_counts", {}).get("defer", 0)),
            "treatment": (
                "Treat deferred EN/ZH natural-semantic judgments as missing for binary natural-gap "
                "strata and repair-rule selection; retain their facts for MCQ-conditioned analyses."
            ),
            "human_gold": False,
        },
        "unrelated_fact_control": {
            "status": unrelated["status"],
            "control_fact_count": unrelated["counts"]["control_facts"],
            "control_input_count": unrelated["counts"]["control_inputs"],
            "current_Qwen3_behavior_used": unrelated["selection_contract"]["current_Qwen3_behavior_used"],
            "vector_source": unrelated["selection_contract"]["vector_source"],
            "accuracy_drop_max": unrelated["evaluation_contract"]["accuracy_drop_max"],
            "small_n_policy": unrelated["evaluation_contract"]["small_n_policy"],
        },
        "open_items": [
            "render_and_tokenize_all_prompts_with_the_frozen_Qwen3_tokenizer",
            "audit_prompt_lengths_and_answer_token_prefixes_against_max_model_length",
        ],
        "authorization_state": {
            "offline_prompt_materialization_complete": True,
            "exact_tokenizer_render_authorized": True,
            "vector_hidden_state_collection_authorized": False,
            "development_vector_intervention_authorized": False,
            "validation_authorized": False,
            "sealed_authorized": False,
        },
        "inputs": {key: binding(path) for key, path in paths.items()},
        "artifacts": {
            "option_free_three_arm_inputs": binding(option_path),
            "vector_source_prompt_specs": binding(vector_path),
        },
    }
    manifest_path = output_dir / "g2a_offline_contract_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
