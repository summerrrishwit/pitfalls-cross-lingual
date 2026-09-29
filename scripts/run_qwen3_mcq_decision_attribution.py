#!/usr/bin/env python3
"""Collect Development MCQ decision states using the frozen behavior-forward shape."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


CHOICES = ("A", "B", "C")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_model_snapshot(model_path: Path, model_manifest: Dict[str, Any]) -> None:
    if model_path.name != model_manifest.get("checkpoint_revision"):
        raise ValueError("checkpoint revision changed")
    for item in model_manifest.get("model_files", []):
        path = model_path / item["name"]
        if (
            not path.exists()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"checkpoint file changed: {item['name']}")


def build_behavior_choice_batch(row: Dict[str, Any], torch: Any, pad_token_id: int) -> Any:
    prompt_ids = [int(value) for value in row["prompt_token_ids"]]
    sequences = [
        prompt_ids + [int(value) for value in row["choice_completions"][choice]["token_ids"]]
        for choice in CHOICES
    ]
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(CHOICES), maximum),
        pad_token_id,
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = torch.zeros(
        (len(CHOICES), maximum),
        dtype=torch.long,
        device="cuda",
    )
    for index, sequence in enumerate(sequences):
        input_ids[index, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device="cuda",
        )
        attention_mask[index, : len(sequence)] = 1
    expected_prefix = [int(value) for value in row["choice_decision_input_ids"]]
    decision_position = int(row["choice_decision_position"])
    for choice, sequence in zip(CHOICES, sequences):
        if sequence[: decision_position + 1] != expected_prefix:
            raise ValueError(
                f"behavior choice prefix differs before decision token: {row['render_id']}:{choice}"
            )
    return input_ids, attention_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--g0b-dir", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--attribution-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    g0b_dir = args.g0b_dir.resolve()
    behavior_dir = args.behavior_dir.resolve()
    model_manifest_path = g0b_dir / "exact_hf_model_manifest.json"
    model_manifest = read_json(model_manifest_path)
    runtime = model_manifest["decoding_and_scoring"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(runtime["cublas_workspace_config"])
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    g0b_gate_path = g0b_dir / "g0b_gate_manifest.json"
    render_manifest_path = g0b_dir / "exact_hf_render_manifest.json"
    rendered_path = g0b_dir / "exact_hf_rendered_inputs.jsonl"
    behavior_manifest_path = behavior_dir / "exact_hf_behavior_run_manifest.json"
    behavior_results_path = behavior_dir / "exact_hf_development_behavior_results.jsonl"
    gate = read_json(args.attribution_gate)
    g0b_gate = read_json(g0b_gate_path)
    render_manifest = read_json(render_manifest_path)
    behavior_manifest = read_json(behavior_manifest_path)
    behavior_results = read_jsonl(behavior_results_path)
    renders = [
        row for row in read_jsonl(rendered_path)
        if row.get("split_assignment") == "development"
    ]

    if gate.get("gate_passed") is not True or gate.get("authorization_state", {}).get(
        "development_mcq_attribution_authorized"
    ) is not True:
        raise ValueError("Development MCQ attribution is not authorized")
    gate_bindings = (
        ("g0b_gate", g0b_gate_path),
        ("model_manifest", model_manifest_path),
        ("behavior_manifest", behavior_manifest_path),
    )
    for key, path in gate_bindings:
        if gate["inputs"][key]["sha256"] != sha256_file(path):
            raise ValueError(f"attribution gate binding changed: {key}")
    if g0b_gate.get("gate_passed") is not True:
        raise ValueError("G0B gate is not passed")
    if g0b_gate["exact_hf_model_manifest"]["sha256"] != sha256_file(model_manifest_path):
        raise ValueError("G0B model-manifest binding changed")
    if g0b_gate["exact_hf_render_manifest"]["sha256"] != sha256_file(render_manifest_path):
        raise ValueError("G0B render-manifest binding changed")
    if render_manifest["rendered_inputs"]["sha256"] != sha256_file(rendered_path):
        raise ValueError("rendered inputs changed after G0B")
    if behavior_manifest["artifacts"]["exact_hf_behavior_results"]["sha256"] != sha256_file(
        behavior_results_path
    ):
        raise ValueError("behavior results changed after G1")
    if behavior_manifest.get("result_count") != 960:
        raise ValueError("behavior manifest does not declare 960 results")
    verify_model_snapshot(args.model_path.resolve(), model_manifest)
    expected_software = model_manifest.get("software", {})
    if torch.__version__ != expected_software.get("torch"):
        raise ValueError(f"torch version changed: {torch.__version__}")
    if transformers.__version__ != expected_software.get("transformers"):
        raise ValueError(f"transformers version changed: {transformers.__version__}")
    if len(renders) != 960 or len(behavior_results) != 960:
        raise ValueError("Development attribution requires all 960 rendered inputs and results")
    behavior_by_id = {row["render_id"]: row for row in behavior_results}
    if set(behavior_by_id) != {row["render_id"] for row in renders}:
        raise ValueError("behavior/render ID sets differ")
    for row in renders:
        if row.get("choice_decision_position") != len(row.get("choice_decision_input_ids", [])) - 1:
            raise ValueError(f"invalid decision position: {row['render_id']}")
        if len(set(row.get("choice_token_ids", {}).values())) != 3:
            raise ValueError(f"non-unique choice tokens: {row['render_id']}")
        for choice in CHOICES:
            behavior_token_id = behavior_by_id[row["render_id"]]["choice_scores"][choice][
                "first_divergent_token_id"
            ]
            if int(row["choice_token_ids"][choice]) != int(behavior_token_id):
                raise ValueError(
                    f"choice-token mismatch for {row['render_id']}:{choice}: "
                    f"{row['choice_token_ids'][choice]} != {behavior_token_id}"
                )
    if args.limit is not None:
        renders = renders[: args.limit]

    if runtime.get("scoring_batch_size") != 1 or runtime.get("torch_deterministic_algorithms") is not True:
        raise ValueError("attribution requires the frozen deterministic single-example runtime")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["cudnn_allow_tf32"])
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path.resolve()),
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=str(runtime["attention_implementation"]),
    ).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path.resolve()), local_files_only=True)
    layer_count = int(model_manifest["architecture"]["num_hidden_layers"])
    hidden_size = int(model_manifest["architecture"]["hidden_size"])
    pad_token_id = int(model_manifest["tokenizer"]["pad_token_id"])
    activations = torch.empty((len(renders), layer_count, hidden_size), dtype=torch.bfloat16, device="cpu")
    metric_rows = []
    index_rows = []
    final_rank_mismatch_count = 0
    final_choice_order_mismatch_count = 0
    with torch.inference_mode():
        for row_index, row in enumerate(renders):
            input_ids, attention_mask = build_behavior_choice_batch(row, torch, pad_token_id)
            decision_position = int(row["choice_decision_position"])
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if len(output.hidden_states) != layer_count + 1:
                raise ValueError("unexpected hidden-state count")
            for layer in range(layer_count):
                decision_states = output.hidden_states[layer][:, decision_position, :]
                if not (
                    torch.equal(decision_states[0], decision_states[1])
                    and torch.equal(decision_states[0], decision_states[2])
                ):
                    raise ValueError(
                        f"identical pre-choice prefixes produced different states: "
                        f"{row['render_id']}:layer-{layer}"
                    )
            resid_pre = torch.stack(
                [output.hidden_states[layer][0, decision_position, :] for layer in range(layer_count)]
            )
            activations[row_index].copy_(resid_pre.to(device="cpu", dtype=torch.bfloat16))
            lens_logits = model.lm_head(model.model.norm(resid_pre)).float()
            lens_log_probs = torch.log_softmax(lens_logits, dim=-1)
            layer_metrics = []
            for layer_id in range(layer_count):
                choice_metrics = {}
                for choice in CHOICES:
                    token_id = int(row["choice_token_ids"][choice])
                    token_logit = lens_logits[layer_id, token_id]
                    choice_metrics[choice] = {
                        "token_id": token_id,
                        "token": tokenizer.decode([token_id]),
                        "logit": round(float(token_logit.item()), 8),
                        "logprob": round(float(lens_log_probs[layer_id, token_id].item()), 8),
                        "rank": int((lens_logits[layer_id] > token_logit).sum().item()) + 1,
                    }
                target = row.get("target_choice")
                layer_metrics.append({
                    "layer_id": layer_id,
                    "choice_metrics": choice_metrics,
                    "gold_minus_target_logit_margin": (
                        round(
                            choice_metrics[row["correct_choice"]]["logit"]
                            - choice_metrics[target]["logit"],
                            8,
                        )
                        if target is not None else None
                    ),
                })
            final_logits = output.logits[0, decision_position, :].float()
            final_ranks = {}
            final_choice_logits = {}
            expected_ranks = {}
            for choice in CHOICES:
                token_id = int(row["choice_token_ids"][choice])
                final_choice_logits[choice] = round(float(final_logits[token_id].item()), 8)
                final_ranks[choice] = int((final_logits > final_logits[token_id]).sum().item()) + 1
                expected_ranks[choice] = int(
                    behavior_by_id[row["render_id"]]["choice_scores"][choice]["first_divergent_token_rank"]
                )
                if final_ranks[choice] != expected_ranks[choice]:
                    final_rank_mismatch_count += 1
            attribution_choice_order = sorted(
                CHOICES,
                key=lambda choice: (-float(final_logits[int(row["choice_token_ids"][choice])].item()), CHOICES.index(choice)),
            )
            behavior_choice_order = sorted(
                CHOICES,
                key=lambda choice: (expected_ranks[choice], CHOICES.index(choice)),
            )
            choice_order_matches = attribution_choice_order == behavior_choice_order
            if not choice_order_matches:
                final_choice_order_mismatch_count += 1
            if final_ranks != expected_ranks:
                raise ValueError(
                    f"behavior-shape final ranks differ for {row['render_id']}: "
                    f"{final_ranks} != {expected_ranks}"
                )
            metric_rows.append({
                "schema_version": "qwen3-mcq-decision-logit-lens-v2",
                "activation_row_index": row_index,
                "render_id": row["render_id"],
                "base_fact_id": row["base_fact_id"],
                "stimulus_id": row["stimulus_id"],
                "language": row["language"],
                "variant": row["variant"],
                "distractor_id": row.get("distractor_id"),
                "correct_choice": row["correct_choice"],
                "target_choice": row.get("target_choice"),
                "behavior_full_sequence_choice": behavior_by_id[row["render_id"]]["choice"],
                "choice_decision_position": row["choice_decision_position"],
                "layer_metrics": layer_metrics,
                "final_model_choice_token_logits": final_choice_logits,
                "final_model_choice_token_ranks": final_ranks,
                "behavior_first_divergent_token_ranks": expected_ranks,
                "attribution_forward_choice_order": attribution_choice_order,
                "behavior_forward_choice_order": behavior_choice_order,
                "choice_order_matches_behavior_forward": choice_order_matches,
                "terminal_status": "completed",
            })
            index_rows.append({
                "activation_row_index": row_index,
                "render_id": row["render_id"],
                "stimulus_id": row["stimulus_id"],
                "base_fact_id": row["base_fact_id"],
            })
            if row_index == 0 or (row_index + 1) % 25 == 0 or row_index + 1 == len(renders):
                print(json.dumps({"completed": row_index + 1, "expected": len(renders)}), flush=True)
            del (
                output,
                resid_pre,
                lens_logits,
                lens_log_probs,
                final_logits,
                input_ids,
                attention_mask,
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "mcq_decision_logit_lens.jsonl"
    index_path = output_dir / "mcq_decision_activation_index.jsonl"
    activations_path = output_dir / "mcq_decision_resid_pre.pt"
    write_jsonl(metrics_path, metric_rows)
    write_jsonl(index_path, index_rows)
    activations_temporary = activations_path.with_suffix(activations_path.suffix + ".tmp")
    torch.save(activations, activations_temporary)
    activations_temporary.replace(activations_path)
    manifest = {
        "schema_version": "qwen3-mcq-decision-attribution-manifest-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if args.limit is None else "diagnostic_limit",
        "scope": "development_mcq_decision_position",
        "claim_boundary": (
            "These measurements characterize MCQ choice computation. They are not, by themselves, "
            "evidence for answer-text recall or translation stages from the original PNT method."
        ),
        "fact_count": len({row["base_fact_id"] for row in renders}),
        "render_count": len(renders),
        "split_counts": dict(Counter(row["split_assignment"] for row in renders)),
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "activation_dtype": "bfloat16",
        "activation_shape": list(activations.shape),
        "runtime_verification": {
            "model_snapshot_verified_against_manifest": True,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "attention_implementation": runtime["attention_implementation"],
            "torch_deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "forward_parity": {
            "exact_global_rank_mismatch_count": final_rank_mismatch_count,
            "choice_order_mismatch_render_count": final_choice_order_mismatch_count,
            "interpretation": (
                "Attribution exactly replays the behavior scorer's one-MCQ by three-completion "
                "batch shape and requires exact final-token rank parity."
            ),
        },
        "hidden_state_contract": (
            "hidden_states[layer_id][:,choice_decision_position,:] equals resid_pre for "
            "model.layers[layer_id]; the three identical pre-choice prefixes must match exactly"
        ),
        "logit_lens_contract": "lm_head(final_norm(resid_pre))",
        "inputs": {
            "model_manifest_sha256": sha256_file(model_manifest_path),
            "render_manifest_sha256": sha256_file(render_manifest_path),
            "rendered_inputs_sha256": sha256_file(rendered_path),
            "behavior_results_sha256": sha256_file(behavior_results_path),
            "attribution_gate_sha256": sha256_file(args.attribution_gate),
        },
        "artifacts": {
            "metrics": {"path": metrics_path.name, "sha256": sha256_file(metrics_path)},
            "activation_index": {"path": index_path.name, "sha256": sha256_file(index_path)},
            "resid_pre": {"path": activations_path.name, "sha256": sha256_file(activations_path)},
        },
        "validation_called": False,
        "sealed_called": False,
        "vector_intervention_run": False,
    }
    write_json(output_dir / "mcq_decision_attribution_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
