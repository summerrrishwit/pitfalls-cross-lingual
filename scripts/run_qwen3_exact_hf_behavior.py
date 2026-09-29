#!/usr/bin/env python3
"""Score the frozen Development MCQs with one exact local HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCHEMA_RESULT = "factual-perturbation-exact-hf-behavior-result-v1"
SCHEMA_LABEL = "factual-perturbation-exact-hf-defect-control-label-v1"
SCHEMA_SUMMARY = "factual-perturbation-exact-hf-development-summary-v1"
CHOICES = ("A", "B", "C")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--g0b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def validate_g0b(g0b_dir: Path, model_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gate_path = g0b_dir / "g0b_gate_manifest.json"
    model_manifest_path = g0b_dir / "exact_hf_model_manifest.json"
    render_manifest_path = g0b_dir / "exact_hf_render_manifest.json"
    rendered_path = g0b_dir / "exact_hf_rendered_inputs.jsonl"
    queue_path = g0b_dir / "hf_replay_queue.jsonl"
    gate = read_json(gate_path)
    model_manifest = read_json(model_manifest_path)
    render_manifest = read_json(render_manifest_path)
    queue = read_jsonl(queue_path)
    renders = read_jsonl(rendered_path)
    if gate.get("gate_passed") is not True or gate.get("status") != "g0b_exact_hf_render_frozen":
        raise ValueError("G0B gate is not passed")
    bindings = (
        (model_manifest_path, gate["exact_hf_model_manifest"]["sha256"]),
        (render_manifest_path, gate["exact_hf_render_manifest"]["sha256"]),
        (queue_path, gate["hf_replay_queue"]["sha256"]),
        (rendered_path, render_manifest["rendered_inputs"]["sha256"]),
    )
    for path, expected in bindings:
        if file_sha256(path) != expected:
            raise ValueError(f"G0B binding changed: {path.name}")
    if model_path.name != model_manifest.get("checkpoint_revision"):
        raise ValueError("checkpoint revision changed")
    for item in model_manifest.get("model_files", []):
        path = model_path / item["name"]
        if not path.exists() or path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"checkpoint file changed: {item['name']}")
    by_id = {row["render_id"]: row for row in renders}
    selected = []
    for queued in queue:
        if queued.get("split_assignment") != "development" or queued.get("mandatory") is not True or queued.get("proxy_filter_applied") is not False:
            raise ValueError("invalid Development replay queue row")
        row = by_id.get(queued["render_id"])
        if row is None:
            raise ValueError("replay queue references unknown render")
        selected.append(row)
    if len(selected) != 960 or len({row["render_id"] for row in selected}) != 960:
        raise ValueError("exact-HF Development queue must contain 960 unique rows")
    return model_manifest, selected


def common_prefix_length(values: Sequence[Sequence[int]]) -> int:
    result = 0
    for tokens in zip(*values):
        if len(set(tokens)) != 1:
            break
        result += 1
    return result


def score_batch(model: Any, torch: Any, rows: Sequence[Dict[str, Any]], pad_token_id: int) -> List[Dict[str, Any]]:
    sequences: List[List[int]] = []
    metadata: List[Tuple[int, str, int, int]] = []
    divergence_by_row = []
    for row_index, row in enumerate(rows):
        prompt_ids = [int(value) for value in row["prompt_token_ids"]]
        completion_ids = [row["choice_completions"][choice]["token_ids"] for choice in CHOICES]
        divergence = common_prefix_length(completion_ids)
        if divergence >= min(len(value) for value in completion_ids):
            raise ValueError(f"choice completions never diverge: {row['render_id']}")
        divergence_by_row.append(divergence)
        for choice in CHOICES:
            suffix = [int(value) for value in row["choice_completions"][choice]["token_ids"]]
            sequences.append(prompt_ids + suffix)
            metadata.append((row_index, choice, len(prompt_ids), len(suffix)))
    maximum = max(len(value) for value in sequences)
    input_ids = torch.full((len(sequences), maximum), pad_token_id, dtype=torch.long, device="cuda")
    attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long, device="cuda")
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device="cuda")
        attention_mask[index, : len(sequence)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
    scores: List[Dict[str, Dict[str, Any]]] = [dict() for _ in rows]
    for sequence_index, (row_index, choice, prompt_length, suffix_length) in enumerate(metadata):
        suffix = sequences[sequence_index][prompt_length: prompt_length + suffix_length]
        positions = torch.arange(prompt_length - 1, prompt_length + suffix_length - 1, device="cuda")
        token_ids = torch.tensor(suffix, dtype=torch.long, device="cuda")
        token_log_probs = log_probs[sequence_index, positions, token_ids]
        divergence = divergence_by_row[row_index]
        divergent_position = prompt_length + divergence - 1
        divergent_token = suffix[divergence]
        divergent_logits = logits[sequence_index, divergent_position]
        rank = int((divergent_logits > divergent_logits[divergent_token]).sum().item()) + 1
        scores[row_index][choice] = {
            "sequence_logprob": round(float(token_log_probs.sum().item()), 8),
            "mean_token_logprob": round(float(token_log_probs.mean().item()), 8),
            "token_count": suffix_length,
            "first_divergent_token_id": divergent_token,
            "first_divergent_token_rank": rank,
        }
    results = []
    for row, choice_scores in zip(rows, scores):
        chosen = max(CHOICES, key=lambda choice: (choice_scores[choice]["sequence_logprob"], -CHOICES.index(choice)))
        gold = row["correct_choice"]
        distractors = [choice for choice in CHOICES if choice != gold]
        best_distractor = max(distractors, key=lambda choice: choice_scores[choice]["sequence_logprob"])
        target = row.get("target_choice")
        results.append({
            "schema_version": SCHEMA_RESULT,
            "render_id": row["render_id"],
            "base_fact_id": row["base_fact_id"],
            "source_id": row["source_id"],
            "split_assignment": row["split_assignment"],
            "leakage_component_id": row["leakage_component_id"],
            "probe_relation_id": row["probe_relation_id"],
            "answer_type": row["answer_type"],
            "stimulus_id": row["stimulus_id"],
            "language": row["language"],
            "variant": row["variant"],
            "distractor_id": row.get("distractor_id"),
            "correct_choice": gold,
            "target_choice": target,
            "choice": chosen,
            "correct": chosen == gold,
            "distractor_hit": target is not None and chosen == target,
            "choice_scores": choice_scores,
            "gold_minus_best_distractor_logprob_margin": round(
                choice_scores[gold]["sequence_logprob"] - choice_scores[best_distractor]["sequence_logprob"], 8
            ),
            "target_minus_gold_logprob_margin": (
                round(choice_scores[target]["sequence_logprob"] - choice_scores[gold]["sequence_logprob"], 8)
                if target is not None else None
            ),
            "terminal_status": "completed",
        })
    del logits, log_probs, input_ids, attention_mask
    return results


def build_labels(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    originals = {(row["base_fact_id"], row["language"]): row for row in results if row["variant"] == "original"}
    by_variant = {(row["base_fact_id"], row.get("distractor_id"), row["language"], row["variant"]): row for row in results if row["variant"] != "original"}
    pairs = sorted({(row["base_fact_id"], row["distractor_id"]) for row in results if row["variant"] == "targeted"})
    labels = []
    for base_fact_id, distractor_id in pairs:
        arm = {}
        for language in ("en", "zh"):
            arm[(language, "original")] = originals[(base_fact_id, language)]
            for variant in ("neutral", "targeted"):
                arm[(language, variant)] = by_variant[(base_fact_id, distractor_id, language, variant)]
        zh_directed = (
            arm[("zh", "original")]["correct"]
            and arm[("zh", "neutral")]["correct"]
            and arm[("zh", "targeted")]["distractor_hit"]
        )
        en_all_correct = all(arm[("en", variant)]["correct"] for variant in ("original", "neutral", "targeted"))
        resistant = all(arm[(language, variant)]["correct"] for language in ("en", "zh") for variant in ("original", "neutral", "targeted"))
        neutral_unstable = any(
            arm[(language, "original")]["correct"] and not arm[(language, "neutral")]["correct"]
            for language in ("en", "zh")
        )
        off_target = any(
            not arm[(language, "targeted")]["correct"] and not arm[(language, "targeted")]["distractor_hit"]
            for language in ("en", "zh")
        )
        labels.append({
            "schema_version": SCHEMA_LABEL,
            "base_fact_id": base_fact_id,
            "source_id": arm[("zh", "targeted")]["source_id"],
            "distractor_id": distractor_id,
            "probe_relation_id": arm[("zh", "targeted")]["probe_relation_id"],
            "answer_type": arm[("zh", "targeted")]["answer_type"],
            "hf_directed_candidate": bool(zh_directed),
            "hf_zh_specific_strict": bool(zh_directed and en_all_correct),
            "hf_resistant_control": bool(resistant),
            "neutral_unstable": bool(neutral_unstable),
            "off_target_failure": bool(off_target),
            "shared_language_susceptibility": bool(
                zh_directed
                and arm[("en", "original")]["correct"]
                and arm[("en", "neutral")]["correct"]
                and arm[("en", "targeted")]["distractor_hit"]
            ),
            "zh_targeted_minus_neutral_gold_margin_change": round(
                arm[("zh", "targeted")]["gold_minus_best_distractor_logprob_margin"]
                - arm[("zh", "neutral")]["gold_minus_best_distractor_logprob_margin"],
                8,
            ),
        })
    return labels


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    model_manifest, rows = validate_g0b(args.g0b_dir.resolve(), args.model_path.resolve())
    scoring_contract = model_manifest.get("decoding_and_scoring", {})
    expected_batch_size = scoring_contract.get("scoring_batch_size")
    if expected_batch_size is None or args.batch_size != int(expected_batch_size):
        raise ValueError(
            f"batch size {args.batch_size} does not match frozen scoring batch size "
            f"{expected_batch_size}"
        )
    if scoring_contract.get("torch_deterministic_algorithms") is not True:
        raise ValueError("exact-HF scoring requires a frozen deterministic runtime")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(
        scoring_contract["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(scoring_contract["cudnn_allow_tf32"])
    torch.backends.cudnn.benchmark = bool(scoring_contract["cudnn_benchmark"])
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "exact_hf_development_behavior_results.jsonl"
    existing = {row["render_id"]: row for row in read_jsonl(results_path)} if results_path.exists() else {}
    pending = [row for row in rows if row["render_id"] not in existing]
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path.resolve()),
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=str(scoring_contract["attention_implementation"]),
    ).eval().to("cuda")
    pad_token_id = int(model_manifest["tokenizer"]["pad_token_id"])
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset: offset + args.batch_size]
        for result in score_batch(model, torch, batch, pad_token_id):
            existing[result["render_id"]] = result
        ordered = [existing[row["render_id"]] for row in rows if row["render_id"] in existing]
        write_jsonl_atomic(results_path, ordered)
        if offset == 0 or (offset // args.batch_size + 1) % 25 == 0 or offset + args.batch_size >= len(pending):
            print(json.dumps({"completed": len(ordered), "expected": len(rows)}), flush=True)
    ordered = [existing[row["render_id"]] for row in rows]
    if args.limit is None and len(ordered) != 960:
        raise ValueError("exact-HF behavior run is incomplete")
    labels = build_labels(ordered) if args.limit is None else []
    labels_path = output_dir / "exact_hf_defect_control_labels.jsonl"
    if labels:
        write_jsonl_atomic(labels_path, labels)
    arm_metrics = {}
    for language in ("en", "zh"):
        for variant in ("original", "neutral", "targeted"):
            subset = [row for row in ordered if row["language"] == language and row["variant"] == variant]
            arm_metrics[f"{language}_{variant}"] = {
                "count": len(subset),
                "accuracy": round(sum(row["correct"] for row in subset) / len(subset), 6) if subset else None,
                "distractor_hit_rate": round(sum(row["distractor_hit"] for row in subset) / len(subset), 6) if subset else None,
            }
    summary = {
        "schema_version": SCHEMA_SUMMARY,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if args.limit is None else "diagnostic_limit",
        "checkpoint_revision": model_manifest["checkpoint_revision"],
        "result_count": len(ordered),
        "unique_base_fact_count": len({row["base_fact_id"] for row in ordered}),
        "split_counts": dict(Counter(row["split_assignment"] for row in ordered)),
        "all_terminal": all(row["terminal_status"] == "completed" for row in ordered),
        "arm_metrics": arm_metrics,
        "label_counts": {
            "variant_count": len(labels),
            "hf_directed_candidate_variants": sum(row["hf_directed_candidate"] for row in labels),
            "hf_zh_specific_strict_variants": sum(row["hf_zh_specific_strict"] for row in labels),
            "hf_resistant_control_variants": sum(row["hf_resistant_control"] for row in labels),
            "neutral_unstable_variants": sum(row["neutral_unstable"] for row in labels),
            "off_target_failure_variants": sum(row["off_target_failure"] for row in labels),
            "shared_language_susceptibility_variants": sum(row["shared_language_susceptibility"] for row in labels),
            "hf_directed_candidate_facts": len({row["base_fact_id"] for row in labels if row["hf_directed_candidate"]}),
            "hf_zh_specific_strict_facts": len({row["base_fact_id"] for row in labels if row["hf_zh_specific_strict"]}),
            "hf_resistant_control_facts": len({row["base_fact_id"] for row in labels if row["hf_resistant_control"]}),
        } if labels else {},
        "artifacts": {
            "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
            "labels": {"path": labels_path.name, "sha256": file_sha256(labels_path)} if labels else None,
        },
        "exact_hf_evidence": args.limit is None,
        "pnt_authorized": False,
        "validation_called": False,
        "sealed_called": False,
    }
    write_json(output_dir / "exact_hf_development_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
