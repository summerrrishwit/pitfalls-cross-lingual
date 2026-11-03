#!/usr/bin/env python3
"""Resume the frozen Development option-free experiment on the remote GPU only.

Audit every prompt before inference; reuse Original only after exact render and
fresh generation reproduction. Each output is flushed for interruption recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
import unicodedata
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def normalized(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def generation_labels(generation: str, groups: list[dict[str, Any]], language: str,
                      semantic: dict[str, Any] | None = None) -> dict[str, Any]:
    """Exact answers are deterministic; all other outputs need semantic review.

    Cross-language exact aliases establish answer identity separately from output
    language. Substring hits are diagnostic only, never final correctness.
    """
    answer = normalized(generation)
    hits, partial = [], []
    for group in groups:
        aliases = {normalized(a) for lang in ("en", "zh") for a in group.get(f"aliases_{lang}", [])}
        aliases.discard("")
        if answer and answer in aliases:
            hits.append(group["option_id"])
        if answer and any(a in answer for a in aliases):
            partial.append(group["option_id"])
    if semantic is not None:
        correct = semantic["semantic_correct"]
        review = "existing_codex_proxy" if correct is not None else "existing_deferred"
    elif len(hits) == 1:
        correct, review = hits[0] == "gold", "exact_frozen_alias"
    elif not answer:
        correct, review = False, "empty_output"
    else:
        correct, review = None, "pending_semantic_review"
    return {"generation_correct": correct, "correctness_basis": review,
            "generated_answer_group": hits[0] if len(hits) == 1 else None,
            "diagnostic_substring_groups": partial, "human_gold": False}


def render(tokenizer: Any, content: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": content}]
    kwargs = dict(add_generation_prompt=True, enable_thinking=False)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return rendered, [int(t) for t in tokens]


def suffix(tokenizer: Any, rendered: str, tokens: list[int], answer: str) -> list[int]:
    full = tokenizer(rendered + answer, add_special_tokens=False)["input_ids"]
    if full[:len(tokens)] != tokens or len(full) == len(tokens):
        raise ValueError("answer continuation prefix mismatch")
    return full[len(tokens):]


def prepare(config: dict[str, Any], out: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoConfig, AutoTokenizer
    import torch
    staging, old = Path(config["staging_dir"]), Path(config["old_run_dir"])
    model_path = Path(config["model_path"])
    contract = read_json(staging / "g2a/g2a_offline_contract_manifest.json")
    model_manifest = read_json(old / "output-v3/exact_hf_model_manifest.json")
    for key, path in {
        "model_manifest": old / "output-v3/exact_hf_model_manifest.json",
        "g0b_gate": old / "output-v3/g0b_gate_manifest.json",
        "input_bundle": old / "input/frozen_static_bundle.jsonl",
        "fold_manifest": staging / "development_fold_manifest.json",
        "semantic_review": staging / "semantic/natural_semantic_review_manifest.json",
    }.items():
        if sha(path) != contract["inputs"][key]["sha256"]:
            raise ValueError(f"changed frozen input: {key}")
    for key in ("option_free_three_arm_inputs", "vector_source_prompt_specs"):
        if sha(staging / "g2a" / f"{key}.jsonl") != contract["artifacts"][key]["sha256"]:
            raise ValueError(f"changed G2A artifact: {key}")
    controls_manifest = read_json(staging / "controls/unrelated_fact_control_manifest.json")
    controls_path = staging / "controls/unrelated_fact_controls.jsonl"
    if sha(controls_path) != controls_manifest["artifacts"]["unrelated_fact_controls"]["sha256"]:
        raise ValueError("changed unrelated controls")
    if sha(staging / "controls/unrelated_fact_control_manifest.json") != contract["inputs"]["unrelated_control_manifest"]["sha256"]:
        raise ValueError("changed control manifest")
    semantic_manifest = read_json(staging / "semantic/natural_semantic_review_manifest.json")
    if sha(staging / "semantic/natural_semantic_adjudications.jsonl") != semantic_manifest["artifacts"]["adjudications"]["sha256"]:
        raise ValueError("changed semantic adjudications")
    natural_summary = read_json(old / "natural-baseline-v4/natural_baseline_summary.json")
    if sha(old / "natural-baseline-v4/natural_baseline_summary.json") != contract["inputs"]["natural_summary"]["sha256"]:
        raise ValueError("changed natural summary")
    if sha(old / "natural-baseline-v4/natural_open_completion_baseline.jsonl") != natural_summary["artifacts"]["results"]["sha256"]:
        raise ValueError("changed natural baseline results")
    if read_json(old / "output-v3/g0b_gate_manifest.json")["gate_passed"] is not True:
        raise ValueError("G0B gate not passed")
    if model_path.name != model_manifest["checkpoint_revision"]:
        raise ValueError("wrong model revision")
    actual = {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers")}
    if actual != model_manifest["software"]:
        raise ValueError(f"runtime differs from frozen environment: {actual}")
    # Verify the actual checkpoint bytes, not just the snapshot directory name.
    for entry in model_manifest["model_files"]:
        if sha(model_path / entry["name"]) != entry["sha256"]:
            raise ValueError(f"changed checkpoint file: {entry['name']}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    rows = read_jsonl(staging / "g2a/option_free_three_arm_inputs.jsonl")
    vectors = read_jsonl(staging / "g2a/vector_source_prompt_specs.jsonl")
    controls = read_jsonl(controls_path)
    natural = {(r["base_fact_id"], r["language"]): r for r in read_jsonl(old / "natural-baseline-v4/natural_open_completion_baseline.jsonl")}
    if len(rows) != 960 or len({r["base_fact_id"] for r in rows}) != 96 or len(vectors) != 1920 or len(controls) != 26:
        raise ValueError("incomplete input coverage")
    if any(r["split_assignment"] != "development" for r in rows):
        raise ValueError("holdout input in Development queue")
    development_ids = {r["base_fact_id"] for r in rows}
    for row in vectors:
        if row["base_fact_id"] not in development_ids or not set(row["icl_donor_base_fact_ids"]).issubset(development_ids):
            raise ValueError("non-Development vector source/donor")
    audits, rendered_rows = [], []
    max_length = int(model_config.max_position_embeddings)
    for kind, collection in (("option_free", rows), ("vector_source", vectors), ("unrelated_control", controls)):
        for row in collection:
            content = row["prompt"] if kind == "vector_source" else row["user_prompt"]
            if text_sha(content) != row["prompt_sha256" if kind == "vector_source" else "user_prompt_sha256"]:
                raise ValueError("changed prompt text")
            rendered, tokens = render(tokenizer, content)
            if len(tokens) + config["max_new_tokens"] > min(max_length, config["max_prompt_tokens"]):
                raise ValueError("prompt too long for declared run")
            if kind == "option_free":
                groups = row["answer_groups"]
                answers = [a for g in groups for a in g[f"aliases_{row['language']}"]]
            else:
                answers = [row["expected_answer"]] if kind == "vector_source" else row["answer_aliases"]
            for answer in answers:
                continuation = suffix(tokenizer, rendered, tokens, answer)
                if len(tokens) + len(continuation) > max_length:
                    raise ValueError("answer too long")
            if kind == "option_free" and row["variant"] == "original":
                previous = natural[(row["base_fact_id"], row["language"])]
                if previous["prompt_token_ids"] != tokens or previous["rendered_prompt_sha256"] != text_sha(rendered):
                    raise ValueError("Original differs from natural baseline")
            input_id = row.get("option_free_id", row.get("vector_source_id", row.get("control_input_id")))
            audits.append({"input_id": input_id, "kind": kind, "prompt_tokens": len(tokens),
                           "rendered_prompt_sha256": text_sha(rendered), "answer_prefix_passed": True})
            if kind == "option_free":
                rendered_rows.append({**row, "prompt_token_ids": tokens, "rendered_prompt": rendered,
                                      "rendered_prompt_sha256": text_sha(rendered)})
    with (out / "tokenizer_audit_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {"status": "passed", "counts": dict(Counter(r["kind"] for r in audits)),
             "max_prompt_tokens": max(r["prompt_tokens"] for r in audits),
             "original_exact_render_matches": 192, "validation_facts_used": 0, "sealed_facts_used": 0,
             "model_manifest_sha256": sha(old / "output-v3/exact_hf_model_manifest.json"),
             "contract_sha256": sha(staging / "g2a/g2a_offline_contract_manifest.json"),
             "rows_sha256": sha(out / "tokenizer_audit_rows.jsonl")}
    write_json(out / "tokenizer_audit.json", audit)
    return tokenizer, rendered_rows, model_manifest


def score_aliases(model: Any, torch: Any, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    tokens, rendered = row["prompt_token_ids"], row["rendered_prompt"]
    for group in row["answer_groups"]:
        values = {}
        for alias in group[f"aliases_{row['language']}"]:
            tail = suffix(tokenizer, rendered, tokens, alias)
            ids = torch.tensor([tokens + tail], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False).logits[0].float()
                # Score the complete answer, without length normalization.
                relevant = logits[len(tokens)-1:len(tokens)+len(tail)-1]
                logp = torch.log_softmax(relevant, dim=-1)
                idx = torch.tensor(tail, device="cuda")
                chosen = logp[torch.arange(len(tail), device="cuda"), idx]
                values[alias] = {"sequence_logprob": round(chosen.sum().item(), 8),
                                 "token_count": len(tail)}
            del ids, logits, relevant, logp, idx, chosen
        result[group["option_id"]] = values
    best = {key: max(v["sequence_logprob"] for v in values.values()) for key, values in result.items()}
    target = row.get("designated_distractor_id")
    return {"alias_scores": result, "group_sequence_scores": best,
            "top_scored_answer_group": max(best, key=best.get),
            "gold_minus_designated_distractor_margin": best["gold"] - best[target] if target else None}


def generate(model: Any, torch: Any, tokenizer: Any, tokens: list[int], budget: int) -> dict[str, Any]:
    ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        result = model.generate(ids, do_sample=False, max_new_tokens=budget,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id)[0, len(tokens):].tolist()
    return {"generation": tokenizer.decode(result, skip_special_tokens=True).strip(),
            "generated_token_ids": result, "generated_token_count": len(result),
            "generation_budget_exhausted": len(result) == budget and result[-1] != tokenizer.eos_token_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if config["remote_only"] and socket.gethostname() != config["remote_hostname"]:
        raise ValueError("this experiment is authorized only on the specified remote machine")
    out = Path(config["output_dir"])
    if args.smoke:
        out = out / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForCausalLM
    tokenizer, rows, manifest = prepare(config, out)
    print(json.dumps({"event": "tokenizer_audit_passed", "rows": len(rows)}, ensure_ascii=False), flush=True)
    if args.audit_only:
        return 0
    runtime = manifest["decoding_and_scoring"]
    torch.use_deterministic_algorithms(runtime["torch_deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    identity = {"config_sha256": sha(args.config), "script_sha256": sha(Path(__file__)),
                "mode": "smoke" if args.smoke else "development"}
    identity_path = out / "run_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("cannot resume changed code/configuration into the same result directory")
    write_json(identity_path, identity)
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], dtype=torch.bfloat16,
             local_files_only=True, attn_implementation=runtime["attention_implementation"]).eval().to("cuda")
    old, staging = Path(config["old_run_dir"]), Path(config["staging_dir"])
    natural = {(r["base_fact_id"], r["language"]): r for r in read_jsonl(old / "natural-baseline-v4/natural_open_completion_baseline.jsonl")}
    semantics = {(r["base_fact_id"], r["language"]): r for r in read_jsonl(staging / "semantic/natural_semantic_adjudications.jsonl")}
    # A deterministic eight-Original sample, selected without looking at outcomes.
    originals = sorted((r for r in rows if r["variant"] == "original"), key=lambda r: text_sha(r["option_free_id"]))[:8]
    reproduction = []
    for row in originals:
        previous = natural[(row["base_fact_id"], row["language"])]
        fresh = generate(model, torch, tokenizer, row["prompt_token_ids"], config["max_new_tokens"])
        same = fresh["generation"] == previous["generation"]
        reproduction.append({"option_free_id": row["option_free_id"], "old": previous["generation"],
                             "fresh": fresh["generation"], "exact_generation_match": same})
    write_json(out / "original_reproduction.json", {"sample": reproduction, "all_passed": all(r["exact_generation_match"] for r in reproduction)})
    if not all(r["exact_generation_match"] for r in reproduction):
        raise ValueError("Original reproduction failed; do not reuse old generation")
    if args.smoke:
        fact_ids = sorted({r["base_fact_id"] for r in rows}, key=text_sha)[:1]
        rows = [r for r in rows if r["base_fact_id"] in fact_ids]
    result_path = out / "open_results.jsonl"
    existing = read_jsonl(result_path)
    done = {r["option_free_id"] for r in existing}
    if len(done) != len(existing) or not done.issubset({r["option_free_id"] for r in rows}):
        raise ValueError("duplicate or foreign result in resume checkpoint")
    reused = sum(r["execution_kind"] == "reused_original" for r in existing)
    for row in rows:
        if row["option_free_id"] in done:
            continue
        current = {k: v for k, v in row.items() if k not in ("rendered_prompt", "chat_render_contract", "evaluation_contract")}
        current["terminal_status"] = "completed"
        if row["variant"] == "original":
            previous = natural[(row["base_fact_id"], row["language"])]
            current.update(generation=previous["generation"], execution_kind="reused_original",
                           original_source_sha256=sha(old / "natural-baseline-v4/natural_open_completion_baseline.jsonl"))
            reused += 1
        else:
            current.update(generate(model, torch, tokenizer, row["prompt_token_ids"], config["max_new_tokens"]))
            current["execution_kind"] = "fresh_generation"
        semantic = semantics.get((row["base_fact_id"], row["language"])) if row["variant"] == "original" else None
        current.update(generation_labels(current["generation"], row["answer_groups"], row["language"], semantic))
        # Score ALL aliases for every row, including Original; old scores omitted some foil aliases.
        current.update(score_aliases(model, torch, tokenizer, row))
        append(result_path, current)
        done.add(row["option_free_id"])
        progress = {"status": "running", "completed": len(done), "expected": len(rows),
                    "reused_original_generations": reused, "elapsed_seconds": round(time.time()-started, 1),
                    "validation_facts_used": 0, "sealed_facts_used": 0}
        write_json(out / "progress.json", progress)
        if len(done) % 10 == 0:
            print(json.dumps(progress), flush=True)
    # The control baseline is independent of cohort/folds/vector estimation.
    control_path = out / "unrelated_control_baseline.jsonl"
    control_done = {r["control_input_id"] for r in read_jsonl(control_path)}
    for row in read_jsonl(staging / "controls/unrelated_fact_controls.jsonl"):
        if row["control_input_id"] in control_done or args.smoke:
            continue
        _, tokens = render(tokenizer, row["user_prompt"])
        result = generate(model, torch, tokenizer, tokens, config["max_new_tokens"])
        groups = [{"option_id": "gold", f"aliases_{row['language']}": row["answer_aliases"]}]
        append(control_path, {**row, **result, **generation_labels(result["generation"], groups, row["language"]),
                              "prompt_token_ids": tokens, "terminal_status": "completed"})
    finished = {"status": "complete", "completed": len(done), "expected": len(rows),
                "reused_original_generations": reused, "fresh_generations": len(done)-reused,
                "control_inputs": len(read_jsonl(control_path)), "elapsed_seconds": round(time.time()-started, 1),
                "results_sha256": sha(result_path), "validation_facts_used": 0, "sealed_facts_used": 0,
                "intervention_run": False, "completed_at": datetime.now(timezone.utc).isoformat()}
    write_json(out / "progress.json", finished)
    print(json.dumps(finished), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
