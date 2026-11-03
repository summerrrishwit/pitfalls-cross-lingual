#!/usr/bin/env python3
"""Remote-only v5 local-recovery diagnosis and fold-isolated prefill repair."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import AbstractContextManager
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import (
    Engine, append, injection_vectors, read_json, read_jsonl,
    row_key, sha, write_json,
)


class PrefillIntervention(AbstractContextManager):
    """Change only prompt[-1], including equivalent teacher-forced scoring."""
    all_answer_positions = False
    def __init__(self, model: Any, vectors: dict[int, Any], prompt_length: int,
                 replace: bool = False):
        if prompt_length < 2:
            raise ValueError("prefill and cached one-token decoding must be distinguishable")
        self.model, self.vectors, self.prompt_length = model, vectors, prompt_length
        self.replace = replace
        self.handles: list[Any] = []

    def __enter__(self) -> PrefillIntervention:
        for layer, vector in self.vectors.items():
            def hook(module: Any, args: tuple[Any, ...], v: Any = vector) -> tuple[Any, ...]:
                if args[0].shape[1] < self.prompt_length and not self.all_answer_positions:
                    return args
                hidden = args[0].clone()
                position = min(self.prompt_length - 1, hidden.shape[1] - 1)
                selection = slice(position, None) if self.all_answer_positions else position
                value = v.to(device=hidden.device).float()
                if self.replace:
                    hidden[:, selection, :] = value.to(hidden.dtype)
                else:
                    hidden[:, selection, :] = (hidden[:, selection, :].float()+value).to(hidden.dtype)
                return (hidden, *args[1:])
            self.handles.append(self.model.model.layers[layer].register_forward_pre_hook(hook))
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AnswerPositionIntervention(PrefillIntervention):
    all_answer_positions = True


class PrefillEngine(Engine):
    scope = "prefill"

    def __init__(self, config_path: Path, stage: str):
        super().__init__(config_path, stage)
        # Exact fact/language/output decisions remain reusable, including their
        # provenance. None stays missing. Latest explicit recheck takes priority.
        for version in ("repair-v3", "repair-v4"):
            directory = self.root/version
            exports = {r["review_key"]: r for r in read_jsonl(directory/"review_export.jsonl")}
            decisions = {r["review_key"]: r for r in read_jsonl(directory/"semantic_review_decisions.jsonl")}
            for key, decision in decisions.items():
                entry = exports.get(key)
                if entry is None:
                    # Some old seeded decisions never needed an export row.
                    # They cannot populate the text-bound cache without it.
                    continue
                if decision["human_gold"] is not False or decision["reviewer_type"] != "codex_proxy":
                    raise ValueError("unbound prior semantics")
                if hashlib.sha256(entry["generation"].encode()).hexdigest() != decision["generation_sha256"]:
                    raise ValueError("prior semantic text changed")
                self.cache[(entry["base_fact_id"], entry["language"], entry["generation"])] = {
                    "generation_correct": decision["semantic_correct"],
                    "generated_answer_group": decision.get("answer_group"),
                    "correctness_basis": "bound_prior_explicit_codex_proxy_review"}
        bound = [self.root/"repair-v3/collect-identity.json",
                 self.root/"inputs/development_fold_manifest.json",
                 self.root/"inputs/g2a/vector_source_prompt_specs.jsonl"]
        for version in ("repair-v3", "repair-v4"):
            bound += [self.root/version/"review_export.jsonl", self.root/version/"semantic_review_decisions.jsonl"]
        bound += sorted((self.root/"repair-v3/open-trajectories").glob("chunk-*.pt"))
        bound += sorted((self.root/"repair-v3/oof-source-states").glob("chunk-*.pt"))
        identity = {"config_sha256": sha(config_path), "v5_code_sha256": sha(Path(__file__)),
                    "sources_sha256": {str(p.relative_to(self.root)): sha(p) for p in bound}}
        target = self.out/"v5_run_identity.json"
        if target.exists() and read_json(target) != identity:
            raise ValueError("v5 code/config/source identity changed; use a new version")
        write_json(target, identity)

    def intervention(self, row: dict[str, Any], vectors: dict[int, Any]) -> Any:
        cls = PrefillIntervention if self.scope == "prefill" else AnswerPositionIntervention
        return cls(self.model, vectors, len(row["prompt_token_ids"]))

    def sequence_scores(self, row: dict[str, Any], vectors: dict[int, Any],
                        all_groups: bool = False) -> dict[str, Any]:
        with self.intervention(row, vectors):
            return super().sequence_scores(row, {}, all_groups)

    def run_generation(self, row: dict[str, Any], vectors: dict[int, Any],
                       score: bool = True) -> dict[str, Any]:
        with self.intervention(row, vectors):
            result = super().run_generation(row, {}, score=False)
        if score and row.get("designated_distractor_id"):
            result.update(self.sequence_scores(row, vectors))
        return result


def cached_states(engine: PrefillEngine, folder: str) -> dict[str, Any]:
    states = {}
    for path in sorted((engine.root/"repair-v3"/folder).glob("chunk-*.pt")):
        data = engine.torch.load(path, map_location="cpu", weights_only=True)
        for identity, activation in zip(data["ids"], data["states"]):
            if identity in states:
                raise ValueError("duplicate activation ID")
            states[identity] = activation
    if not states:
        raise ValueError("cached source activation bank missing")
    return states


def paired_panel(engine: PrefillEngine) -> list[dict[str, Any]]:
    lookup = {(r["base_fact_id"], r["language"], r["variant"], r["designated_distractor_id"]): r
              for r in engine.inputs}
    panel = []
    for label in read_jsonl(engine.root/"results/open_variant_labels.jsonl"):
        fid, target = label["base_fact_id"], label["distractor_id"]
        original = lookup[(fid, "zh", "original", None)]
        neutral = lookup[(fid, "zh", "neutral", target)]
        targeted = lookup[(fid, "zh", "targeted", target)]
        resistant = all(engine.baseline[r["option_free_id"]]["generation_correct"] is True
                        for r in (original, neutral, targeted))
        positive = label["open_induced_directed"] is True
        if positive or resistant:
            panel.append({"positive": positive, "resistant": resistant,
                          "neutral": neutral, "targeted": targeted})
    return panel


def diagnose(engine: PrefillEngine) -> None:
    states = cached_states(engine, "open-trajectories")
    assert set(states) == {r["option_free_id"] for r in engine.inputs}
    panel = paired_panel(engine)
    positives = [r for r in panel if r["positive"]]
    assert len(positives) == 25
    chosen = min(positives, key=lambda r: hashlib.sha256(r["targeted"]["option_free_id"].encode()).hexdigest())
    audit_path = engine.out/"self_patch_audit.json"
    if not audit_path.exists():
        checks = []
        for layer in engine.config["diagnostic_layers"]:
            row = chosen["targeted"]
            value = states[row["option_free_id"]][layer].float()
            result = engine.run_generation(row, {layer: value-value})
            before = engine.baseline[row["option_free_id"]]
            delta = abs(result["gold_margin"]-before["gold_minus_designated_distractor_margin"])
            checks.append({"layer": layer, "input_id": row["option_free_id"],
                           "generation_exact_match": result["generation"] == before["generation"],
                           "margin_delta": delta})
        if not all(r["generation_exact_match"] and r["margin_delta"] < 1e-5 for r in checks):
            write_json(audit_path, {"status": "failed", "checks": checks})
            raise ValueError("self-patch changes baseline")
        write_json(audit_path, {"status": "passed", "checks": checks})
    elif read_json(audit_path)["status"] != "passed":
        raise ValueError("self-patch audit failed")
    for layer in engine.config["diagnostic_layers"]:
        for mode in engine.config["diagnostic_modes"]:
            path = engine.out/f"generation-oracle-l{layer}-{mode}.jsonl"
            done = {r["input_id"] for r in read_jsonl(path)}
            for pair in panel:
                source, target = (pair["neutral"], pair["targeted"]) if mode == "neutral_to_targeted" else (pair["targeted"], pair["neutral"])
                if row_key(target) in done:
                    continue
                delta = {layer: states[row_key(source)][layer].float()-states[row_key(target)][layer].float()}
                result = engine.run_generation(target, delta)
                result.update(layer=layer, mode=mode, positive=pair["positive"], resistant=pair["resistant"],
                              source_input_id=row_key(source), evaluation_partition=target["development_fold_id"])
                append(path, result)
                done.add(row_key(target))
                if len(done) % 10 == 0:
                    progress = {"stage": "local_recovery", "layer": layer, "mode": mode,
                                "completed": len(done), "expected": len(panel)}
                    write_json(engine.out/"progress.json", progress)
                    print(progress, flush=True)
    write_json(engine.out/"diagnostic_execution_complete.json", {
        "status": "complete_pending_semantic_review", "layers": engine.config["diagnostic_layers"],
        "panel_variants": len(panel), "positive_variants": 25, "validation_used": 0, "sealed_used": 0})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["diagnose"], required=True)
    args = parser.parse_args()
    engine = PrefillEngine(args.config, "v5-"+args.phase)
    lock = (engine.out/".v5.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    diagnose(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
