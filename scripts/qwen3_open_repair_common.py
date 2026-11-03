"""Shared exact-runtime utilities for the remote-only open-answer repair study."""
from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any

from run_qwen3_option_free_remote import (
    append, generate, generation_labels, normalized, prepare, read_json,
    read_jsonl, render, sha, suffix, text_sha, write_json,
)


class ResidualInjection(AbstractContextManager):
    """Add to resid_pre at every position that predicts an answer token.

    For full teacher-forced input this means prompt[-1] and continuation
    positions; for cached decoding the one-token input gets one addition.
    No parameter or source activation is modified in place.
    """
    def __init__(self, model: Any, vectors: dict[int, Any], prompt_length: int):
        self.model, self.vectors, self.prompt_length = model, vectors, prompt_length
        self.handles: list[Any] = []

    def __enter__(self) -> ResidualInjection:
        for layer_id, vector in self.vectors.items():
            def hook(module: Any, args: tuple[Any, ...], v: Any = vector) -> tuple[Any, ...]:
                hidden = args[0].clone()
                start = min(self.prompt_length - 1, hidden.shape[1] - 1)
                hidden[:, start:, :] += v.to(device=hidden.device, dtype=hidden.dtype)
                return (hidden, *args[1:])
            self.handles.append(self.model.model.layers[layer_id].register_forward_pre_hook(hook))
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def choice_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"id": f"{kind}-l{layer}-s{scale:g}", "parts": [{"kind": kind, "layer": layer, "scale": scale}]}
            for kind in ("task", "difference") for layer in config["candidate_layers"]
            for scale in config["candidate_scales"]]


def injection_vectors(torch: Any, bank: dict[str, Any], partition: str,
                      spec: dict[str, Any], protocol: str, replicate: int | None = None) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for part in spec["parts"]:
        layer = part["layer"]
        vector = bank[partition][part["kind"]][layer].float() * part["scale"]
        result[layer] = result.get(layer, torch.zeros_like(vector)) + vector
    if replicate is not None:
        random_result = {}
        for layer, vector in result.items():
            material = [protocol, partition, spec["id"], layer, replicate]
            seed = int(hashlib.sha256(json.dumps(material).encode()).hexdigest()[:16], 16) % (2**63-1)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            direction = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
            random_result[layer] = direction * (vector.norm() / direction.norm())
        result = random_result
    return result


def row_key(row: dict[str, Any]) -> str:
    return row.get("option_free_id", row.get("control_input_id"))


def review_key(row: dict[str, Any], generation: str) -> str:
    identity = row.get("base_fact_id", row.get("control_id"))
    return hashlib.sha256(json.dumps([identity, row["language"], generation], ensure_ascii=False).encode()).hexdigest()[:24]


class Engine:
    def __init__(self, config_path: Path, stage: str):
        self.config = read_json(config_path)
        self.root = Path(self.config["remote_root"])
        self.out = Path(self.config["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.base_config = read_json(self.root / "run_config.json")
        if socket.gethostname() != self.base_config["remote_hostname"]:
            raise ValueError("all experiment execution must run on the authorized SSH host")
        for key, value in {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
            os.environ[key] = value
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        audit_dir = self.out / f"{stage}-audit"
        audit_dir.mkdir(exist_ok=True)
        self.tokenizer, self.inputs, self.manifest = prepare(self.base_config, audit_dir)
        runtime = self.manifest["decoding_and_scoring"]
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = runtime["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
        self.model = AutoModelForCausalLM.from_pretrained(self.base_config["model_path"], dtype=torch.bfloat16,
            local_files_only=True, attn_implementation=runtime["attention_implementation"]).eval().to("cuda")
        self.folds = {r["base_fact_id"]:r["development_fold_id"] for r in self.inputs}
        self.components = {r["base_fact_id"]:r["leakage_component_id"] for r in self.inputs}
        self.relations = {r["base_fact_id"]:r["probe_relation_id"] for r in self.inputs}
        self.baseline = {r["option_free_id"]:r for r in read_jsonl(self.root / "results/open_results_labeled.jsonl")}
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        for r in self.baseline.values():
            self.cache[(r["base_fact_id"], r["language"], r["generation"])] = {
                "generation_correct":r["generation_correct"], "generated_answer_group":r["generated_answer_group"],
                "correctness_basis":"identical_Development_generation_prior_semantic_label"}
        self.controls = read_jsonl(self.root / "inputs/controls/unrelated_fact_controls.jsonl")
        self.control_baseline = {r["control_input_id"]:r for r in read_jsonl(self.root / "results/unrelated_control_baseline.jsonl")}
        self.control_semantics = {r["control_input_id"]:r for r in read_jsonl(self.root / "results/unrelated_control_semantic_review.jsonl")}
        for r in self.controls:
            rendered, tokens = render(self.tokenizer, r["user_prompt"])
            r.update(prompt_token_ids=tokens, rendered_prompt=rendered,
                     answer_groups=[{"option_id":"gold", f"aliases_{r['language']}":r["answer_aliases"]}])
            b = self.control_baseline[r["control_input_id"]]
            self.cache[(r["control_id"],r["language"],b["generation"])] = {
                "generation_correct":self.control_semantics[r["control_input_id"]]["semantic_correct"],
                "generated_answer_group": "gold" if self.control_semantics[r["control_input_id"]]["semantic_correct"] else None,
                "correctness_basis":"identical_control_generation_prior_semantics"}
        stage_identity = {"config_sha256":sha(config_path),"common_code_sha256":sha(Path(__file__)),
                          "base_run_identity":read_json(self.root/"results/run_identity.json"),"stage":stage}
        path = self.out / f"{stage}-identity.json"
        if path.exists() and read_json(path) != stage_identity:
            raise ValueError("stage identity changed; use a new version instead of mixing results")
        write_json(path,stage_identity)

    def label(self, row: dict[str, Any], generation: str) -> dict[str, Any]:
        identity = row.get("base_fact_id",row.get("control_id"))
        cached = self.cache.get((identity,row["language"],generation))
        if cached is not None:
            return dict(cached)
        result = generation_labels(generation,row["answer_groups"],row["language"])
        if result["generation_correct"] is not None:
            return result
        # Remove a literal repetition of the fixed question, then require the
        # COMPLETE remaining answer to match an alias. Never accept substring.
        question = normalized(row.get("factual_completion_prompt",row.get("prompt","")))
        answer = normalized(generation)
        if question and answer.startswith(question):
            remainder = answer[len(question):]
            matched = []
            for g in row["answer_groups"]:
                aliases = {normalized(a) for l in ("en","zh") for a in g.get(f"aliases_{l}",[])}
                if remainder and remainder in aliases:
                    matched.append(g["option_id"])
            if len(matched)==1:
                result.update(generation_correct=matched[0]=="gold", generated_answer_group=matched[0],
                              correctness_basis="literal_question_prefix_then_exact_complete_frozen_alias")
        return result

    def sequence_scores(self, row: dict[str, Any], vectors: dict[int, Any], all_groups: bool = False) -> dict[str, Any]:
        torch, tokens = self.torch, row["prompt_token_ids"]
        scores = {}
        target = row.get("designated_distractor_id")
        for group in row["answer_groups"]:
            if not all_groups and group["option_id"] not in {"gold",target}:
                continue
            aliases = {}
            for alias in group[f"aliases_{row['language']}"]:
                tail = suffix(self.tokenizer,row["rendered_prompt"],tokens,alias)
                ids = torch.tensor([tokens+tail],device="cuda")
                with ResidualInjection(self.model,vectors,len(tokens)), torch.inference_mode():
                    output = self.model(input_ids=ids,use_cache=False)
                    logits = output.logits[0,len(tokens)-1:len(tokens)+len(tail)-1].float()
                    logp = torch.log_softmax(logits,dim=-1)
                    values = logp[torch.arange(len(tail),device="cuda"),torch.tensor(tail,device="cuda")]
                    aliases[alias] = float(values.sum().item())
                del output,logits,logp,values,ids
            scores[group["option_id"]] = max(aliases.values())
        return {"group_sequence_scores":scores,"gold_margin":scores["gold"]-scores[target] if target else None}

    def run_generation(self, row: dict[str, Any], vectors: dict[int, Any], score: bool = True) -> dict[str, Any]:
        with ResidualInjection(self.model,vectors,len(row["prompt_token_ids"])):
            result = generate(self.model,self.torch,self.tokenizer,row["prompt_token_ids"],self.config["max_new_tokens"])
        result.update(self.label(row,result["generation"]))
        if score and row.get("designated_distractor_id"):
            result.update(self.sequence_scores(row,vectors))
        return {"input_id":row_key(row),"base_fact_id":row.get("base_fact_id"),
                "control_id":row.get("control_id"),"language":row["language"],"variant":row.get("variant","control"),
                "designated_distractor_id":row.get("designated_distractor_id"),
                "review_key":review_key(row,result["generation"]),"terminal_status":"completed",**result}

    def states(self, tokens: list[int]) -> Any:
        with self.torch.inference_mode():
            output = self.model.model(input_ids=self.torch.tensor([tokens],device="cuda"),
                                      use_cache=False,output_hidden_states=True,return_dict=True)
            states = self.torch.stack([h[0,-1] for h in output.hidden_states[:-1]])
        if states.shape != (36,4096):
            raise ValueError("unexpected Qwen3 resid_pre coordinate shape")
        return states

    def no_op_audit(self) -> None:
        sample = sorted((r for r in self.inputs if r["language"]=="zh"),key=lambda r:text_sha(r["option_free_id"]))[:4]
        results=[]
        for row in sample:
            zero = {self.config["candidate_layers"][0]:self.torch.zeros(4096)}
            generated = self.run_generation(row,zero,score=False)
            expected = self.baseline[row["option_free_id"]]
            match = generated["generation"]==expected["generation"]
            delta = 0.0
            if row["designated_distractor_id"]:
                score = self.sequence_scores(row,zero)
                delta = abs(score["gold_margin"]-expected["gold_minus_designated_distractor_margin"])
            results.append({"input_id":row_key(row),"exact_generation_match":match,"absolute_margin_delta":delta})
        passed = all(r["exact_generation_match"] and r["absolute_margin_delta"]<1e-5 for r in results)
        write_json(self.out/"no_op_injection_audit.json",{"status":"passed" if passed else "failed","checks":results})
        if not passed:
            raise ValueError("no-op hook changes frozen baseline")
