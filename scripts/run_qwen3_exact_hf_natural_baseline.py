#!/usr/bin/env python3
"""Run the frozen exact-HF natural EN/ZH open-completion baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value)


def alias_correct(generation: str, aliases: Sequence[str]) -> bool:
    normalized = normalize_answer(generation)
    return any(
        normalize_answer(alias) in normalized
        for alias in aliases
        if normalize_answer(alias)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--g0b-dir", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def continuation_ids(tokenizer: Any, rendered_prompt: str, prompt_ids: Sequence[int], text: str) -> List[int]:
    full = tokenizer(rendered_prompt + text, add_special_tokens=False)["input_ids"]
    if list(full[: len(prompt_ids)]) != list(prompt_ids):
        raise ValueError("natural answer tokenization prefix mismatch")
    result = [int(value) for value in full[len(prompt_ids):]]
    if not result:
        raise ValueError("empty natural answer tokenization")
    return result


def sequence_logprob(model: Any, torch: Any, prompt_ids: Sequence[int], suffix_ids: Sequence[int]) -> Dict[str, Any]:
    full = list(prompt_ids) + list(suffix_ids)
    input_ids = torch.tensor([full], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    positions = torch.arange(len(prompt_ids) - 1, len(full) - 1, device="cuda")
    tokens = torch.tensor(suffix_ids, dtype=torch.long, device="cuda")
    values = log_probs[positions, tokens]
    first_logits = logits[len(prompt_ids) - 1]
    first_token = int(suffix_ids[0])
    rank = int((first_logits > first_logits[first_token]).sum().item()) + 1
    return {
        "sequence_logprob": round(float(values.sum().item()), 8),
        "mean_token_logprob": round(float(values.mean().item()), 8),
        "token_count": len(suffix_ids),
        "first_token_id": first_token,
        "first_token_rank": rank,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    g0b_dir = args.g0b_dir.resolve()
    behavior_dir = args.behavior_dir.resolve()
    model_manifest_path = g0b_dir / "exact_hf_model_manifest.json"
    g0b_gate_path = g0b_dir / "g0b_gate_manifest.json"
    g1_gate_path = behavior_dir / "g1_gate_manifest.json"
    model_manifest = read_json(model_manifest_path)
    g0b_gate = read_json(g0b_gate_path)
    g1_gate = read_json(g1_gate_path)
    if g0b_gate.get("gate_passed") is not True or g1_gate.get("natural_baseline_authorized") is not True:
        raise ValueError("natural baseline is not authorized")
    if args.model_path.resolve().name != model_manifest["checkpoint_revision"]:
        raise ValueError("checkpoint revision changed")
    scoring_contract = model_manifest.get("decoding_and_scoring", {})
    if scoring_contract.get("scoring_batch_size") != 1:
        raise ValueError("natural baseline requires frozen single-example scoring")
    if scoring_contract.get("torch_deterministic_algorithms") is not True:
        raise ValueError("natural baseline requires a frozen deterministic runtime")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(
        scoring_contract["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(scoring_contract["cudnn_allow_tf32"])
    torch.backends.cudnn.benchmark = bool(scoring_contract["cudnn_benchmark"])
    bundle = [row for row in read_jsonl(args.input_bundle) if row.get("split_assignment") == "development"]
    if len(bundle) != 96:
        raise ValueError("natural baseline requires all 96 Development facts")
    if args.limit is not None:
        bundle = bundle[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path.resolve()), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path.resolve()),
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=str(scoring_contract["attention_implementation"]),
    ).eval().to("cuda")
    rows = []
    for fact_index, fact in enumerate(bundle, 1):
        for language in ("en", "zh"):
            prompt = str(fact[f"prompt_{language}"])
            instruction = (
                "请补全以下事实，只输出缺失的答案，不要解释：\n"
                if language == "zh"
                else "Complete the following factual statement with only the missing answer:\n"
            )
            messages = [{"role": "user", "content": instruction + prompt}]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            if isinstance(prompt_ids, Mapping):
                prompt_ids = prompt_ids["input_ids"]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            prompt_ids = [int(value) for value in prompt_ids]
            aliases = list(dict.fromkeys([
                str(fact[f"answer_{language}"]),
                *[str(value) for value in fact.get(f"answer_aliases_{language}", [])],
            ]))
            distractors = list(dict.fromkeys(
                str(option[f"text_{language}"])
                for option in fact["static_mcq"]["options"]
                if option.get("kind") == "distractor"
            ))
            gold_scores = {
                alias: sequence_logprob(
                    model, torch, prompt_ids,
                    continuation_ids(tokenizer, rendered, prompt_ids, alias),
                )
                for alias in aliases
            }
            distractor_scores = {
                value: sequence_logprob(
                    model, torch, prompt_ids,
                    continuation_ids(tokenizer, rendered, prompt_ids, value),
                )
                for value in distractors
            }
            best_gold = max(gold_scores, key=lambda value: gold_scores[value]["sequence_logprob"])
            best_distractor = max(distractor_scores, key=lambda value: distractor_scores[value]["sequence_logprob"])
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                generated = model.generate(
                    input_tensor,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )[0, len(prompt_ids):]
            generation = tokenizer.decode(generated, skip_special_tokens=True).strip()
            rows.append({
                "schema_version": "factual-perturbation-exact-hf-natural-baseline-v2",
                "base_fact_id": fact["base_fact_id"],
                "source_id": fact["source_id"],
                "split_assignment": "development",
                "leakage_component_id": fact["leakage_component_id"],
                "probe_relation_id": fact["probe_relation_id"],
                "answer_type": fact["answer_type"],
                "language": language,
                "prompt": prompt,
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "answer_aliases": aliases,
                "generation": generation,
                "alias_aware_generation_correct": alias_correct(generation, aliases),
                "gold_scores": gold_scores,
                "distractor_scores": distractor_scores,
                "best_gold_alias": best_gold,
                "best_distractor": best_distractor,
                "gold_minus_distractor_logprob_margin": round(
                    gold_scores[best_gold]["sequence_logprob"]
                    - distractor_scores[best_distractor]["sequence_logprob"],
                    8,
                ),
                "terminal_status": "completed",
            })
        if fact_index == 1 or fact_index % 10 == 0 or fact_index == len(bundle):
            print(json.dumps({"completed_facts": fact_index, "expected_facts": len(bundle)}), flush=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "natural_open_completion_baseline.jsonl"
    write_jsonl_atomic(results_path, rows)
    by_fact = {}
    for row in rows:
        by_fact.setdefault(row["base_fact_id"], {})[row["language"]] = row
    exact_labels = read_jsonl(behavior_dir / "exact_hf_defect_control_labels.jsonl")
    directed_facts = {row["base_fact_id"] for row in exact_labels if row["hf_directed_candidate"]}
    fact_labels = []
    for base_fact_id, language_rows in sorted(by_fact.items()):
        en = language_rows["en"]
        zh = language_rows["zh"]
        en_correct = en["alias_aware_generation_correct"]
        zh_correct = zh["alias_aware_generation_correct"]
        fact_labels.append({
            "base_fact_id": base_fact_id,
            "natural_en_correct": en_correct,
            "natural_zh_correct": zh_correct,
            "natural_pnt_gap": bool(en_correct and not zh_correct),
            "induced_only_gap": bool(en_correct and zh_correct and base_fact_id in directed_facts),
            "natural_en_failure": not en_correct,
            "en_minus_zh_gold_margin": round(
                en["gold_minus_distractor_logprob_margin"]
                - zh["gold_minus_distractor_logprob_margin"],
                8,
            ),
        })
    labels_path = output_dir / "natural_baseline_labels.jsonl"
    write_jsonl_atomic(labels_path, fact_labels)
    summary = {
        "schema_version": "factual-perturbation-exact-hf-natural-baseline-summary-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if args.limit is None else "diagnostic_limit",
        "checkpoint_revision": model_manifest["checkpoint_revision"],
        "fact_count": len(by_fact),
        "result_count": len(rows),
        "language_counts": dict(Counter(row["language"] for row in rows)),
        "generation_accuracy": {
            language: round(
                sum(row["alias_aware_generation_correct"] for row in rows if row["language"] == language)
                / sum(row["language"] == language for row in rows),
                6,
            )
            for language in ("en", "zh")
        },
        "automatic_generation_match_rule": "normalized_gold_alias_substring_v2",
        "semantic_review_status": "pending_codex_review",
        "label_counts": {
            "natural_pnt_gap": sum(row["natural_pnt_gap"] for row in fact_labels),
            "induced_only_gap": sum(row["induced_only_gap"] for row in fact_labels),
            "natural_en_failure": sum(row["natural_en_failure"] for row in fact_labels),
        },
        "artifacts": {
            "results": {"path": results_path.name, "sha256": file_sha256(results_path)},
            "labels": {"path": labels_path.name, "sha256": file_sha256(labels_path)},
            "exact_hf_model_manifest_sha256": file_sha256(model_manifest_path),
            "g0b_gate_manifest_sha256": file_sha256(g0b_gate_path),
            "g1_gate_manifest_sha256": file_sha256(g1_gate_path),
        },
        "validation_called": False,
        "sealed_called": False,
        "pnt_intervention_run": False,
    }
    write_json(output_dir / "natural_baseline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
