#!/usr/bin/env python3
"""Finite paired/centered direction tests after v5b local recovery."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
from pathlib import Path
from typing import Any

from qwen3_open_repair_common import append, injection_vectors, read_json, read_jsonl, row_key, sha, write_json
from run_qwen3_prefill_repair import PrefillEngine, cached_states, paired_panel


def source_order(engine: PrefillEngine, identities: list[str], seed: int) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for fid in identities:
        groups[engine.relations[fid]].append(fid)
    for relation in groups:
        groups[relation].sort(key=lambda fid: hashlib.sha256(f"v5-source-size-{seed}-{fid}".encode()).hexdigest())
    relations = sorted(groups, key=lambda value: hashlib.sha256(f"v5-relation-{seed}-{value}".encode()).hexdigest())
    return [groups[relation][index] for index in range(max(map(len, groups.values())))
            for relation in relations if index < len(groups[relation])]


def source_directions(engine: PrefillEngine, layer: int, final: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = engine.torch
    open_states = cached_states(engine, "open-trajectories")
    specs = [r for r in read_jsonl(engine.root/"inputs/g2a/vector_source_prompt_specs.jsonl")
             if (r["source_partition_id"] == "final-development-all") == final]
    if final:
        from collect_qwen3_open_vectors import collect_chunks
        tensor = collect_chunks(engine, specs, engine.out/"final-source-states", False)
        source_states = {r["vector_source_id"]: state for r, state in zip(specs, tensor)}
    else:
        source_states = cached_states(engine, "oof-source-states")
    assert len(specs) == (480 if final else 1440)
    source_lookup = {(r["source_partition_id"], r["kind"], r["base_fact_id"]):
                     source_states[r["vector_source_id"]][layer].float() for r in specs}
    variants: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in engine.inputs:
        if row["language"] == "zh" and row["variant"] in ("neutral", "targeted"):
            variants[row["base_fact_id"]][row["designated_distractor_id"]][row["variant"]] = row
    partitions = ["final-development-all"] if final else sorted(set(engine.folds.values()))
    directions, norms = {}, {}
    for partition in partitions:
        ids = sorted(engine.folds) if final else sorted(fid for fid, fold in engine.folds.items() if fold != partition)
        assert len(ids) == (96 if final else 72)
        directions[partition] = {"paired": {}, "centered_icl": {}}
        norms[partition] = {}
        for fid in ids:
            deltas, reference_norms = [], []
            assert len(variants[fid]) == 2
            for arms in variants[fid].values():
                neutral = open_states[row_key(arms["neutral"])][layer].float()
                targeted = open_states[row_key(arms["targeted"])][layer].float()
                deltas.append(neutral-targeted)
                reference_norms.append(float(neutral.norm()))
            directions[partition]["paired"][fid] = torch.stack(deltas).mean(dim=0)
            directions[partition]["centered_icl"][fid] = (
                source_lookup[(partition, "task_icl_recall_en", fid)] +
                source_lookup[(partition, "task_icl_recall_zh", fid)] -
                source_lookup[(partition, "fact_recall_reference_en", fid)] -
                source_lookup[(partition, "fact_recall_reference_zh", fid)])/2
            norms[partition][fid] = sum(reference_norms)/len(reference_norms)
    return directions, norms


def estimate_bank(engine: PrefillEngine, layer: int, directions: dict[str, Any], norms: dict[str, Any],
                  size: int | None = None, seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = engine.torch
    bank, manifest = {}, {}
    for partition, families in directions.items():
        identities = sorted(families["paired"])
        chosen = source_order(engine, identities, seed)[:size] if size else identities
        assert len(chosen) == (size or len(identities))
        if partition != "final-development-all":
            assert all(engine.folds[fid] != partition for fid in chosen)
        median_norm = float(torch.quantile(torch.tensor([norms[partition][fid] for fid in chosen]), .5))
        bank[partition], manifest[partition] = {}, {"source_fact_ids": chosen, "source_facts": len(chosen),
                                                   "neutral_reference_median_l2": median_norm, "raw_direction_norm": {}}
        for family, facts in families.items():
            mean = torch.stack([facts[fid] for fid in chosen]).mean(dim=0)
            norm = float(mean.norm())
            if norm < 1e-8:
                raise ValueError("degenerate paired/centered source direction")
            vector = torch.zeros(36, 4096)
            vector[layer] = mean*(median_norm/norm)
            bank[partition][family] = vector
            manifest[partition]["raw_direction_norm"][family] = norm
    return bank, manifest


def diagnostic_rows(engine: PrefillEngine) -> list[tuple[str, dict[str, Any]]]:
    selected = {row_key(p[arm]) for p in paired_panel(engine) for arm in ("neutral", "targeted")}
    facts = {p["neutral"]["base_fact_id"] for p in paired_panel(engine)}
    return [(r["development_fold_id"], r) for r in engine.inputs if r["language"] == "zh" and
            (row_key(r) in selected or (r["base_fact_id"] in facts and r["variant"] == "original"))]


def run_trial(engine: PrefillEngine, bank: dict[str, Any], spec: dict[str, Any], name: str,
              rows: list[tuple[str, dict[str, Any]]], replicate: int | None = None,
              scope: str = "prefill", sign: int = 1, prefix: str = "generation") -> None:
    engine.scope = scope
    path = engine.out/f"{prefix}-{name}.jsonl"
    saved = read_jsonl(path)
    done = {(r["evaluation_partition"], r["input_id"]) for r in saved}
    if len(saved) != len(done):
        raise ValueError("duplicate checkpoint")
    for partition, row in rows:
        if (partition, row_key(row)) in done:
            continue
        vectors = injection_vectors(engine.torch, bank, partition, spec, engine.config["protocol_version"], replicate)
        vectors = {layer: value*sign for layer, value in vectors.items()}
        result = engine.run_generation(row, vectors)
        result.update(config_id=spec["id"], random_replicate=replicate, evaluation_partition=partition,
                      hook_scope=scope, direction_sign=sign)
        append(path, result)
        done.add((partition, row_key(row)))
        if len(done) % 20 == 0:
            progress = {"stage": prefix, "config": name, "completed": len(done), "expected": len(rows)}
            write_json(engine.out/"followup_progress.json", progress)
            print(progress, flush=True)
    print("trial_complete", name, len(done), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=["generate", "timing", "random", "learning", "final"], required=True)
    args = parser.parse_args()
    cfg = read_json(args.config)
    out = Path(cfg["output_dir"])
    diagnostic = read_json(out/"local_recovery_summary.json")
    if not diagnostic["status"].startswith("local_recovery_supported"):
        raise ValueError("local generated recovery/layer choice not established")
    layer = diagnostic["selected_layer"]
    if args.phase in ("random", "learning"):
        selection = read_json(out/"generation_selection.json")
        if selection["status"] != "generation_eligible_pending_random":
            raise ValueError("no generation-eligible prefill rule")
    if args.phase == "final" and read_json(out/"selected_rule.json")["status"] != "oof_selected_pending_final_vector":
        raise ValueError("G2 did not pass")
    engine = PrefillEngine(args.config, "v5b-"+args.phase)
    lock = (out/".v5.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {"config_sha256": sha(args.config), "followup_code_sha256": sha(Path(__file__)),
                "local_recovery_summary_sha256": sha(out/"local_recovery_summary.json")}
    identity_path = out/"followup_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("followup estimator/selection changed")
    write_json(identity_path, identity)
    if args.phase == "final":
        directions, norms = source_directions(engine, layer, True)
        bank, manifest = estimate_bank(engine, layer, directions, norms)
        engine.torch.save(bank, out/"final_vectors.pt")
        write_json(out/"final_vector_manifest.json", {"status": "complete", "partitions": manifest,
            "icl_source_count": 480, "paired_source_facts": 96, "vectors_sha256": sha(out/"final_vectors.pt"),
            "validation_facts_used": 0, "sealed_facts_used": 0})
        return 0
    bank_path = out/"oof_vectors.pt"
    if not bank_path.exists():
        directions, norms = source_directions(engine, layer)
        bank, manifest = estimate_bank(engine, layer, directions, norms)
        engine.torch.save(bank, bank_path)
        write_json(out/"oof_vector_manifest.json", {"status": "complete", "layer": layer, "partitions": manifest,
            "vectors_sha256": sha(bank_path), "validation_facts_used": 0, "sealed_facts_used": 0})
    manifest = read_json(out/"oof_vector_manifest.json")
    if sha(bank_path) != manifest["vectors_sha256"]:
        raise ValueError("source bank changed")
    bank = engine.torch.load(bank_path, map_location="cpu", weights_only=True)
    specs = [{"id": f"{family}-l{layer}-rho{rho:g}-prefill", "parts": [{"kind": family, "layer": layer, "scale": rho}]}
             for family in ("paired", "centered_icl") for rho in cfg["candidate_scales"]]
    rows = [(r["development_fold_id"], r) for r in engine.inputs if r["language"] == "zh"]
    rows += [(partition, r) for partition in sorted(bank) for r in engine.controls if r["language"] == "zh"]
    if args.phase == "generate":
        write_json(out/"coarse_selection.json", {"status": "fixed_before_candidate_generation", "candidates": specs,
                                                "diagnostic_layer": layer})
        for spec in specs:
            run_trial(engine, bank, spec, spec["id"], rows)
        write_json(out/"candidate_generation_complete.json", {"status": "complete", "configs": len(specs),
            "inputs_per_config": len(rows), "validation_used": 0, "sealed_used": 0})
    elif args.phase == "random":
        for spec in selection["eligible_specs"]:
            for replicate in range(cfg["random_replicates"]):
                run_trial(engine, bank, spec, f"random-{spec['id']}-r{replicate}", rows, replicate)
        write_json(out/"random_generation_complete.json", {"status": "complete", "replicates": cfg["random_replicates"]})
    else:
        metrics = read_json(out/"generation_metrics.json")
        ranked = sorted(specs, key=lambda s: (-(metrics[s["id"]]["directed_generation_gain"] or 0),
            abs(s["parts"][0]["scale"]), s["parts"][0]["kind"] != "paired"))
        spec = ranked[0] if args.phase == "timing" else selection["eligible_specs"][0]
        panel = diagnostic_rows(engine)
        if args.phase == "timing":
            run_trial(engine, bank, spec, "every-token-"+spec["id"], panel, scope="every_token", prefix="timing")
            run_trial(engine, bank, spec, "negative-"+spec["id"], panel, sign=-1, prefix="timing")
            write_json(out/"timing_complete.json", {"status": "complete", "spec": spec, "panel_inputs": len(panel)})
        else:
            directions, norms = source_directions(engine, layer)
            for seed in (0, 1):
                for size in (18, 36):
                    subset, subset_manifest = estimate_bank(engine, layer, directions, norms, size, seed)
                    name = f"n{size}-seed{seed}-{spec['id']}"
                    write_json(out/f"learning-{name}-sources.json", subset_manifest)
                    run_trial(engine, subset, spec, name, panel, prefix="learning")
            write_json(out/"learning_complete.json", {"status": "complete", "spec": spec,
                "sizes": [18, 36, 72], "seeds": [0, 1], "n72": "reuse_full_candidate_evaluation", "panel_inputs": len(panel)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
