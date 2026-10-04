#!/usr/bin/env python3
"""Authorize Development-only MCQ decision-position attribution after G1."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "byte_count": path.stat().st_size, "sha256": sha256_file(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0b-dir", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--semantic-review-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--runtime-stability-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    g0b = read_json(args.g0b_dir / "g0b_gate_manifest.json")
    model = read_json(args.g0b_dir / "exact_hf_model_manifest.json")
    g1 = read_json(args.behavior_dir / "g1_gate_manifest.json")
    behavior = read_json(args.behavior_dir / "exact_hf_behavior_run_manifest.json")
    natural = read_json(args.natural_dir / "natural_baseline_summary.json")
    semantic = read_json(args.semantic_review_dir / "natural_semantic_review_manifest.json")
    folds = read_json(args.fold_manifest)
    stability = read_json(args.runtime_stability_audit)

    failures = []
    if g0b.get("schema_version") != "factual-perturbation-g0b-gate-v3" or g0b.get("gate_passed") is not True:
        failures.append("g0b_v3_not_passed")
    if model.get("hook_contract", {}).get("hidden_state_position") != "last_token_before_first_choice_specific_token":
        failures.append("mcq_choice_decision_position_not_frozen")
    runtime = model.get("decoding_and_scoring", {})
    if not (
        runtime.get("scoring_batch_size") == 1
        and runtime.get("attention_implementation") == "eager"
        and runtime.get("torch_deterministic_algorithms") is True
    ):
        failures.append("deterministic_runtime_not_frozen")
    if g1.get("gate_passed") is not True or g1.get("observed_directed_base_fact_count", 0) < 10:
        failures.append("g1_behavior_gate_not_passed")
    if behavior.get("validation_called") is not False or behavior.get("sealed_called") is not False:
        failures.append("holdout_behavior_exposed")
    if natural.get("status") != "complete" or natural.get("result_count") != 192:
        failures.append("natural_baseline_incomplete")
    if natural.get("validation_called") is not False or natural.get("sealed_called") is not False:
        failures.append("natural_holdout_exposed")
    if semantic.get("status") not in {"complete", "complete_with_deferrals"}:
        failures.append("semantic_review_incomplete")
    if semantic.get("reviewer_type") != "codex_proxy" or semantic.get("human_gold") is not False:
        failures.append("semantic_review_provenance_invalid")
    if folds.get("status") != "frozen" or folds.get("fold_contract", {}).get("fold_count") != 4:
        failures.append("development_folds_not_frozen")
    if stability.get("status") != "passed_frozen_runtime_reproduction_with_cross_runtime_sensitivity":
        failures.append("runtime_reproduction_not_passed")
    if stability.get("canonical", {}).get("result_sha256") != sha256_file(
        args.behavior_dir / "exact_hf_development_behavior_results.jsonl"
    ):
        failures.append("runtime_audit_not_bound_to_behavior")

    manifest = {
        "schema_version": "qwen3-development-attribution-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_attribution_authorized" if not failures else "blocked",
        "gate_passed": not failures,
        "blocking_reasons": failures,
        "scope": "development_mcq_decision_position_attribution_only",
        "claim_boundary": (
            "MCQ choice-position Logit Lens and resid_pre evidence does not by itself establish "
            "the original PNT answer-recall/translation mechanism."
        ),
        "authorization_state": {
            "development_mcq_attribution_authorized": not failures,
            "development_vector_intervention_authorized": False,
            "validation_authorized": False,
            "sealed_authorized": False,
        },
        "remaining_blockers": [
            "freeze_option_free_recall_and_translation_prompt_contract",
            "freeze_task_and_difference_vector_estimators",
            "freeze_out_of_fold_intervention_grid_and_endpoints",
        ],
        "inputs": {
            "g0b_gate": binding(args.g0b_dir / "g0b_gate_manifest.json"),
            "model_manifest": binding(args.g0b_dir / "exact_hf_model_manifest.json"),
            "g1_gate": binding(args.behavior_dir / "g1_gate_manifest.json"),
            "behavior_manifest": binding(args.behavior_dir / "exact_hf_behavior_run_manifest.json"),
            "natural_summary": binding(args.natural_dir / "natural_baseline_summary.json"),
            "semantic_review": binding(args.semantic_review_dir / "natural_semantic_review_manifest.json"),
            "fold_manifest": binding(args.fold_manifest),
            "runtime_stability_audit": binding(args.runtime_stability_audit),
        },
        "counts": {
            "development_facts": 96,
            "directed_facts": g1["observed_directed_base_fact_count"],
            "matched_directed_variants": g1["matched_directed_variant_count"],
            "unmatched_directed_variants": g1["unmatched_directed_variant_count"],
            "semantic_review_deferrals": semantic["decision_counts"].get("defer", 0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
