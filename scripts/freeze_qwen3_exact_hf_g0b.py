#!/usr/bin/env python3
"""Freeze Qwen3 exact-HF identity and all 1,600 tokenizer-rendered MCQ inputs.

Run this script in the environment that owns the immutable HF snapshot.  It is
deliberately behavior-blind: it reads only G0A/protocol artifacts and never
reads proxy or target-model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCHEMA_MODEL = "factual-perturbation-exact-hf-model-manifest-v3"
SCHEMA_RENDER_ROW = "factual-perturbation-exact-hf-render-v3"
SCHEMA_RENDER_MANIFEST = "factual-perturbation-exact-hf-render-manifest-v3"
SCHEMA_REPLAY = "factual-perturbation-exact-hf-replay-queue-v3"
SCHEMA_GATE = "factual-perturbation-g0b-gate-v3"
EXPECTED_SPLITS = {"development": 96, "validation": 32, "sealed": 32}
EXPECTED_INPUTS_PER_FACT = 10
CHOICES = ("A", "B", "C")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def file_binding(
    path: Path,
    schema_version: Optional[str] = None,
    path_label: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "path": path_label or str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256_bytes(path.read_bytes()),
    }
    if schema_version:
        result["schema_version"] = schema_version
    return result


def simulation_prompt(prompt: str, context: str, options: Dict[str, str], language: str) -> str:
    rendered_options = "\n".join(f"{label}. {text}" for label, text in options.items())
    allowed = "/".join(options)
    example = next(iter(options))
    if language == "zh":
        return f"仅根据事实回答选择题。{context}\n{prompt}\n{rendered_options}\nchoice 必须是 {allowed} 之一。只返回 JSON，例如：{{\"choice\":\"{example}\"}}。"
    return f"Answer this factual multiple-choice question. {context}\n{prompt}\n{rendered_options}\nchoice must be one of {allowed}. Return JSON only, for example: {{\"choice\":\"{example}\"}}."


def common_prefix_length(values: List[List[int]]) -> int:
    result = 0
    for tokens in zip(*values):
        if len(set(tokens)) != 1:
            break
        result += 1
    return result


def choice_decision_metadata(
    prompt_ids: List[int], completion_token_ids: Dict[str, List[int]]
) -> Dict[str, Any]:
    """Resolve the input position whose next-token logits first distinguish A/B/C."""
    if tuple(completion_token_ids) != CHOICES:
        raise ValueError("choice completion labels are not exactly A/B/C")
    values = [completion_token_ids[choice] for choice in CHOICES]
    if any(not value for value in values):
        raise ValueError("empty choice completion tokenization")
    prefix_length = common_prefix_length(values)
    if prefix_length >= min(len(value) for value in values):
        raise ValueError("choice completions never diverge")
    prefix_ids = values[0][:prefix_length]
    choice_token_ids = {
        choice: completion_token_ids[choice][prefix_length]
        for choice in CHOICES
    }
    if len(set(choice_token_ids.values())) != len(CHOICES):
        raise ValueError("first choice-specific token is not unique for A/B/C")
    decision_input_ids = prompt_ids + prefix_ids
    if not decision_input_ids:
        raise ValueError("choice decision input is empty")
    return {
        "choice_common_prefix_token_ids": prefix_ids,
        "choice_common_prefix_token_count": prefix_length,
        "choice_token_ids": choice_token_ids,
        "choice_decision_input_ids": decision_input_ids,
        "choice_decision_position": len(decision_input_ids) - 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--static-freeze-manifest", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-bundle-label")
    parser.add_argument("--static-freeze-manifest-label")
    parser.add_argument("--protocol-manifest-label")
    parser.add_argument("--host-label", default="autodl")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--scoring-batch-size", type=int, default=1)
    parser.add_argument("--attention-implementation", choices=("eager",), default="eager")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from transformers import AutoConfig, AutoTokenizer
    import torch
    import transformers

    model_path = args.model_path.resolve()
    if args.scoring_batch_size != 1:
        raise ValueError("formal exact-HF scoring batch size is frozen to 1")
    if model_path.name != args.revision:
        raise ValueError("model snapshot directory does not match revision")
    bundle = read_jsonl(args.input_bundle)
    static_manifest = read_json(args.static_freeze_manifest)
    protocol = read_json(args.protocol_manifest)
    if Counter(str(row.get("split_assignment")) for row in bundle) != Counter(EXPECTED_SPLITS):
        raise ValueError("input bundle is not the frozen 96/32/32 population")
    if protocol.get("exact_hf_replay_contract", {}).get("render_all_160_before_opening_development") is not True:
        raise ValueError("protocol does not require all-split rendering")

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    chat_template = tokenizer.chat_template or ""
    if not chat_template:
        raise ValueError("tokenizer has no chat template")

    model_files = []
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if not (path.is_file() or path.is_symlink()):
            continue
        model_files.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_bytes(path.read_bytes()),
        })
    weight_files = [row for row in model_files if row["name"].endswith(".safetensors")]
    if not weight_files:
        raise ValueError("no safetensors weight shards found")

    created_at = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_manifest = {
        "schema_version": SCHEMA_MODEL,
        "created_at": created_at,
        "status": "exact_hf_identity_frozen",
        "model_repo_id": args.model_repo_id,
        "checkpoint_revision": args.revision,
        "snapshot_path": str(model_path),
        "execution_host_label": args.host_label,
        "weight_format": "safetensors",
        "weight_precision": "bfloat16",
        "model_files": model_files,
        "model_files_sha256": sha256_value(model_files),
        "architecture": {
            "model_type": config.model_type,
            "architectures": list(config.architectures or []),
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "vocab_size": int(config.vocab_size),
        },
        "tokenizer": {
            "repo_id": args.model_repo_id,
            "revision": args.revision,
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
        },
        "software": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "decoding_and_scoring": {
            "behavior_choice_policy": "argmax_complete_json_choice_sequence_log_probability_v1",
            "candidate_completions": {choice: canonical_json({"choice": choice}) for choice in CHOICES},
            "length_normalization": "none_equal_semantic_choices",
            "tie_break": list(CHOICES),
            "generation_used_for_primary_choice": False,
            "temperature": 0.0,
            "scoring_batch_size": args.scoring_batch_size,
            "attention_implementation": args.attention_implementation,
            "torch_deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
        "answer_tokenization_policy": "tokenize_full_render_plus_complete_json_choice_then_remove_exact_prompt_prefix_v1",
        "hook_contract": {
            "family": "resid_pre",
            "module_pattern": "model.layers.{layer}",
            "capture": "forward_pre_hook_input_0",
            "layer_ids": list(range(int(config.num_hidden_layers))),
            "hidden_state_position": "last_token_before_first_choice_specific_token",
        },
    }
    model_manifest_path = output_dir / "exact_hf_model_manifest.json"
    write_json(model_manifest_path, model_manifest)

    rendered_rows = []
    for row in sorted(bundle, key=lambda item: str(item["base_fact_id"])):
        behavior_inputs = row.get("static_mcq", {}).get("behavior_inputs")
        if not isinstance(behavior_inputs, list) or len(behavior_inputs) != EXPECTED_INPUTS_PER_FACT:
            raise ValueError(f"invalid behavior input count: {row.get('base_fact_id')}")
        for stimulus in behavior_inputs:
            language = str(stimulus["language"])
            options = {
                str(option["choice"]): str(option[f"text_{language}"])
                for option in stimulus["options"]
            }
            if tuple(options) != CHOICES:
                raise ValueError(f"invalid option labels: {stimulus['stimulus_id']}")
            user_prompt = simulation_prompt(
                str(stimulus["prompt"]), str(stimulus["context"]), options, language
            )
            messages = [{"role": "user", "content": user_prompt}]
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(prompt_ids, Mapping):
                prompt_ids = prompt_ids["input_ids"]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            prompt_ids = [int(value) for value in prompt_ids]
            completions = {}
            for choice in CHOICES:
                completion_text = canonical_json({"choice": choice})
                full_ids = tokenizer(
                    rendered_prompt + completion_text,
                    add_special_tokens=False,
                )["input_ids"]
                if list(full_ids[: len(prompt_ids)]) != prompt_ids:
                    raise ValueError(f"answer tokenization prefix mismatch: {stimulus['stimulus_id']}:{choice}")
                continuation_ids = [int(value) for value in full_ids[len(prompt_ids):]]
                if not continuation_ids:
                    raise ValueError(f"empty choice tokenization: {stimulus['stimulus_id']}:{choice}")
                completions[choice] = {
                    "text": completion_text,
                    "token_ids": continuation_ids,
                    "token_count": len(continuation_ids),
                }
            try:
                decision_metadata = choice_decision_metadata(
                    prompt_ids,
                    {
                        choice: completions[choice]["token_ids"]
                        for choice in CHOICES
                    },
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid choice decision tokenization: {stimulus['stimulus_id']}: {exc}"
                ) from exc
            if len(prompt_ids) + max(value["token_count"] for value in completions.values()) > args.max_model_len:
                raise ValueError(f"render exceeds max model length: {stimulus['stimulus_id']}")
            correct_choice = next(
                str(option["choice"])
                for option in stimulus["options"]
                if option.get("kind") == "gold"
            )
            target_choice = next(
                (str(option["choice"]) for option in stimulus["options"] if option.get("option_id") == stimulus.get("distractor_id")),
                None,
            )
            render_id = sha256_value([args.revision, stimulus["stimulus_id"], "exact-hf-render-v3"])[:24]
            rendered_rows.append({
                "schema_version": SCHEMA_RENDER_ROW,
                "render_id": render_id,
                "base_fact_id": row["base_fact_id"],
                "source_id": row["source_id"],
                "split_assignment": row["split_assignment"],
                "leakage_component_id": row["leakage_component_id"],
                "probe_relation_id": row["probe_relation_id"],
                "answer_type": row["answer_type"],
                "stimulus_id": stimulus["stimulus_id"],
                "language": language,
                "variant": stimulus["arm"],
                "distractor_id": stimulus.get("distractor_id"),
                "correct_choice": correct_choice,
                "target_choice": target_choice,
                "option_ids": [str(option["option_id"]) for option in stimulus["options"]],
                "user_prompt": user_prompt,
                "user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
                "rendered_prompt": rendered_prompt,
                "rendered_prompt_sha256": sha256_bytes(rendered_prompt.encode("utf-8")),
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "choice_completions": completions,
                **decision_metadata,
            })

    split_input_counts = Counter(row["split_assignment"] for row in rendered_rows)
    expected_split_inputs = {key: value * EXPECTED_INPUTS_PER_FACT for key, value in EXPECTED_SPLITS.items()}
    render_ids = [row["render_id"] for row in rendered_rows]
    if len(rendered_rows) != 1600 or len(render_ids) != len(set(render_ids)):
        raise ValueError("rendered input coverage is not exactly 1,600 unique rows")
    if dict(split_input_counts) != expected_split_inputs:
        raise ValueError("rendered split counts do not match 960/320/320")
    rendered_path = output_dir / "exact_hf_rendered_inputs.jsonl"
    write_jsonl(rendered_path, rendered_rows)

    replay_rows = [
        {
            "schema_version": SCHEMA_REPLAY,
            "queue_index": index,
            "render_id": row["render_id"],
            "base_fact_id": row["base_fact_id"],
            "stimulus_id": row["stimulus_id"],
            "split_assignment": row["split_assignment"],
            "mandatory": True,
            "proxy_filter_applied": False,
            "status": "pending",
        }
        for index, row in enumerate(
            (row for row in rendered_rows if row["split_assignment"] == "development"),
            1,
        )
    ]
    if len(replay_rows) != 960:
        raise ValueError("Development replay queue is not exactly 960 rows")
    replay_path = output_dir / "hf_replay_queue.jsonl"
    write_jsonl(replay_path, replay_rows)

    token_lengths = [row["prompt_token_count"] for row in rendered_rows]
    render_manifest = {
        "schema_version": SCHEMA_RENDER_MANIFEST,
        "created_at": created_at,
        "status": "exact_hf_render_frozen",
        "model_manifest": file_binding(model_manifest_path, SCHEMA_MODEL),
        "input_bundle": file_binding(
            args.input_bundle, path_label=args.input_bundle_label
        ),
        "static_freeze_manifest": file_binding(
            args.static_freeze_manifest,
            path_label=args.static_freeze_manifest_label,
        ),
        "protocol_manifest": file_binding(
            args.protocol_manifest, path_label=args.protocol_manifest_label
        ),
        "rendered_inputs": {
            **file_binding(
                rendered_path, SCHEMA_RENDER_ROW, rendered_path.name
            ),
            "record_count": len(rendered_rows),
            "render_ids_sha256": sha256_value(sorted(render_ids)),
        },
        "split_input_counts": dict(split_input_counts),
        "base_fact_count": len(bundle),
        "inputs_per_fact": EXPECTED_INPUTS_PER_FACT,
        "token_audit": {
            "max_model_len": args.max_model_len,
            "minimum_prompt_tokens": min(token_lengths),
            "maximum_prompt_tokens": max(token_lengths),
            "over_limit_count": 0,
            "terminal_count": len(rendered_rows),
        },
        "render_contract": {
            "messages": [{"role": "user"}],
            "add_generation_prompt": True,
            "enable_thinking": False,
            "prompt_builder": "factual-simulation-static-g0a-mcq-v1",
            "choice_decision_position": "last_token_before_first_choice_specific_token",
        },
    }
    render_manifest_path = output_dir / "exact_hf_render_manifest.json"
    write_json(render_manifest_path, render_manifest)

    gate = {
        "schema_version": SCHEMA_GATE,
        "created_at": created_at,
        "status": "g0b_exact_hf_render_frozen",
        "gate_passed": True,
        "blocking_reasons": [],
        "exact_hf_model_manifest": file_binding(
            model_manifest_path, SCHEMA_MODEL, model_manifest_path.name
        ),
        "exact_hf_render_manifest": file_binding(
            render_manifest_path,
            SCHEMA_RENDER_MANIFEST,
            render_manifest_path.name,
        ),
        "hf_replay_queue": {
            **file_binding(replay_path, SCHEMA_REPLAY, replay_path.name),
            "record_count": len(replay_rows),
            "mandatory_base_fact_count": len({row["base_fact_id"] for row in replay_rows}),
            "proxy_filter_applied": False,
        },
        "resolved_protocol": {
            "answer_tokenization_policy": model_manifest["answer_tokenization_policy"],
            "exact_max_attempts": 2,
            "matching_distance_or_adjustment": "retain_all_exact_relation_answer_type_resistant_controls_with_prespecified_covariate_adjustment_v1",
            "resolved_layer_ids": model_manifest["hook_contract"]["layer_ids"],
            "resolved_hidden_state_position": model_manifest["hook_contract"]["hidden_state_position"],
            "scale_grid": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
            "retention_thresholds": {
                "original_accuracy_drop_max": 0.05,
                "neutral_accuracy_drop_max": 0.05,
                "english_accuracy_drop_max": 0.05,
                "unrelated_fact_accuracy_drop_max": 0.02,
            },
            "control_harm_threshold": 0.05,
            "validation_confirmation_mode": "induced_confirmation",
            "minimum_directed_event_count": {"unit": "base_fact_id", "count": 10},
        },
        "authorization_state": {
            "development_exact_hf_behavior_authorized": True,
            "validation_authorized": False,
            "sealed_authorized": False,
            "pnt_authorized": False,
        },
    }
    write_json(output_dir / "g0b_gate_manifest.json", gate)
    print(json.dumps({
        "gate_passed": True,
        "render_count": len(rendered_rows),
        "split_input_counts": dict(split_input_counts),
        "development_replay_count": len(replay_rows),
        "max_prompt_tokens": max(token_lengths),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
