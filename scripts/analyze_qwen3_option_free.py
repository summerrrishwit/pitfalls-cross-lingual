#!/usr/bin/env python3
"""Analyze open-answer defects; MCQ labels are optional comparison metadata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True)+"\n" for v in values), encoding="utf-8")


def conjunction(values: list[bool | None]) -> bool | None:
    return False if False in values else None if None in values else True


def any_known(values: list[bool | None]) -> bool | None:
    return True if True in values else None if None in values else False


def defect(original: dict[str, Any], neutral: dict[str, Any], targeted: dict[str, Any], directed: bool) -> bool | None:
    incorrect = None if targeted["generation_correct"] is None else not targeted["generation_correct"]
    conditions = [original["generation_correct"], neutral["generation_correct"], incorrect]
    if directed:
        group = targeted["generated_answer_group"]
        conditions.append(group == targeted["designated_distractor_id"] if group is not None else (None if incorrect is None else False))
    return conjunction(conditions)


def validate_inputs(inputs: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Validate paired arms without assuming a cohort size or two distractors."""
    if not inputs or len({r["option_free_id"] for r in inputs}) != len(inputs):
        raise ValueError("empty inputs or duplicate input IDs")
    keys: set[tuple[str, str, str, str | None]] = set()
    targets: dict[str, set[str]] = defaultdict(set)
    for row in inputs:
        if row.get("split_assignment") != "development":
            raise ValueError("discovery analysis requires Development inputs")
        fact, language, arm = row["base_fact_id"], row["language"], row["variant"]
        target = row["designated_distractor_id"]
        if language not in ("en", "zh") or arm not in ("original", "neutral", "targeted"):
            raise ValueError("this analyzer requires EN/ZH open three-arm inputs")
        if (arm == "original" and target is not None) or (arm != "original" and not target):
            raise ValueError("invalid arm/distractor identity")
        key = (fact, language, arm, target)
        if key in keys:
            raise ValueError("duplicate logical input; use a new run for a revised candidate")
        keys.add(key)
        if arm == "targeted":
            targets[fact].add(target)
    if set(targets) != {r["base_fact_id"] for r in inputs}:
        raise ValueError("fact has no targeted input")
    for fact, distractors in targets.items():
        expected = {(fact, language, "original", None) for language in ("en", "zh")}
        expected.update((fact, language, arm, target) for target in distractors
                        for language in ("en", "zh") for arm in ("neutral", "targeted"))
        if {k for k in keys if k[0] == fact} != expected:
            raise ValueError("input lacks paired Original/Neutral/Targeted coverage")
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.run_dir
    inputs = rows(root / "inputs/g2a/option_free_three_arm_inputs.jsonl")
    targets_by_fact = validate_inputs(inputs)
    outputs = rows(root / "results/open_results.jsonl")
    review_path = root / "results/semantic_review_decisions.jsonl"
    decisions = {r["review_key"]: r for r in rows(review_path)}
    prior = {(r["base_fact_id"], r["language"], r["generation"]): r for r in rows(root / "inputs/semantic/natural_semantic_adjudications.jsonl")}
    by_id = {r["option_free_id"]: r for r in inputs}
    if len({r["option_free_id"] for r in outputs}) != len(outputs):
        raise ValueError("duplicate output IDs")
    queue = {}
    for row in outputs:
        if row["option_free_id"] not in by_id:
            raise ValueError("output ID outside input scope")
        source = by_id[row["option_free_id"]]
        for k in ("base_fact_id", "language", "variant", "user_prompt_sha256", "development_fold_id"):
            if row[k] != source[k]:
                raise ValueError(f"output/source mismatch {k}")
        key = hashlib.sha256(json.dumps([row["base_fact_id"],row["language"],row["generation"]],ensure_ascii=False).encode()).hexdigest()[:24]
        row["review_key"] = key
        if row["generation_correct"] is None:
            decision = decisions.get(key)
            previous = prior.get((row["base_fact_id"], row["language"], row["generation"]))
            if decision:
                if decision.get("human_gold") is not False or decision.get("reviewer_type") != "codex_proxy":
                    raise ValueError("semantic review provenance missing")
                if decision.get("generation_sha256") != hashlib.sha256(row["generation"].encode()).hexdigest():
                    raise ValueError("semantic decision does not bind this generation")
                row["generation_correct"] = decision["semantic_correct"]
                if decision.get("answer_group"):
                    row["generated_answer_group"] = decision["answer_group"]
                row["correctness_basis"] = "codex_proxy_review"
            elif previous and previous["semantic_correct"] is not None:
                row["generation_correct"] = previous["semantic_correct"]
                row["correctness_basis"] = "identical_fact_language_generation_prior_proxy"
            else:
                queue[key] = {"review_key": key, "base_fact_id": row["base_fact_id"], "language": row["language"],
                              "generation": row["generation"], "question": row["factual_completion_prompt"],
                              "answer_groups": row["answer_groups"], "budget_exhausted": row.get("generation_budget_exhausted",False)}
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    jsonl(result_dir / "semantic_review_queue.jsonl", list(queue.values()))
    jsonl(result_dir / "open_results_labeled.jsonl", outputs)
    grouped = defaultdict(dict)
    for row in outputs:
        grouped[row["base_fact_id"]][(row["language"],row["variant"],row["designated_distractor_id"])] = row
    mcq = {(r["base_fact_id"],r["distractor_id"]):r for r in rows(root / "inputs/mcq_labels.jsonl")}
    variants, facts = [], []
    for fact_id, target_ids in sorted(targets_by_fact.items()):
        data = grouped[fact_id]
        target_ids = sorted(target_ids)
        fact_variants = []
        for target in target_ids:
            required = [(l,a,None if a == "original" else target) for l in ("en","zh") for a in ("original","neutral","targeted")]
            if not all(k in data for k in required):
                continue
            zo,zn,zt = [data[("zh",a,None if a == "original" else target)] for a in ("original","neutral","targeted")]
            en = [data[("en",a,None if a == "original" else target)] for a in ("original","neutral","targeted")]
            directed = defect(zo,zn,zt,True)
            record = {"base_fact_id":fact_id,"distractor_id":target,"probe_relation_id":zt["probe_relation_id"],
                      "development_fold_id":zt["development_fold_id"],"leakage_component_id":zt["leakage_component_id"],
                      "open_induced_directed":directed,"open_induced_any_error":defect(zo,zn,zt,False),
                      "open_language_specific":conjunction([directed]+[r["generation_correct"] for r in en]),
                      "zh_original_correct":zo["generation_correct"],"zh_neutral_correct":zn["generation_correct"],
                      "zh_targeted_correct":zt["generation_correct"],"zh_targeted_group":zt["generated_answer_group"],
                      "zh_baseline_failure":None if zo["generation_correct"] is None else not zo["generation_correct"],
                      "zh_neutral_unstable":conjunction([zo["generation_correct"],None if zn["generation_correct"] is None else not zn["generation_correct"]]),
                      "open_off_target_failure":conjunction([zo["generation_correct"],zn["generation_correct"],None if zt["generation_correct"] is None else not zt["generation_correct"],zt["generated_answer_group"] != target if zt["generated_answer_group"] is not None else None]),
                      "scored_preference_transfer":conjunction([zo["top_scored_answer_group"] == "gold",zn["top_scored_answer_group"] == "gold",zt["top_scored_answer_group"] == target]),
                      "targeted_minus_neutral_margin":zt["gold_minus_designated_distractor_margin"]-zn["gold_minus_designated_distractor_margin"],
                      "mcq_directed":mcq.get((fact_id,target), {}).get("hf_directed_candidate"),
                      "mcq_zh_specific":mcq.get((fact_id,target), {}).get("hf_zh_specific_strict")}
            variants.append(record); fact_variants.append(record)
        if fact_variants:
            incomplete = len(fact_variants) != len(target_ids)
            facts.append({"base_fact_id":fact_id,"leakage_component_id":fact_variants[0]["leakage_component_id"],
                          "candidate_coverage_complete":not incomplete,
                          "mcq_directed":any_known([mcq.get((fact_id,t),{}).get("hf_directed_candidate") for t in target_ids]),
                          **{key:any_known([r[key] for r in fact_variants] + ([None] if incomplete else []))
                             for key in ("open_induced_directed","open_induced_any_error","open_language_specific","scored_preference_transfer")}})
        else:
            source = next(r for r in inputs if r["base_fact_id"] == fact_id)
            facts.append({"base_fact_id":fact_id,"leakage_component_id":source["leakage_component_id"],
                          "candidate_coverage_complete":False,
                          "mcq_directed":any_known([mcq.get((fact_id,t),{}).get("hf_directed_candidate") for t in target_ids]),
                          **{key:None for key in ("open_induced_directed","open_induced_any_error","open_language_specific","scored_preference_transfer")}})
    jsonl(result_dir / "open_variant_labels.jsonl", variants)
    jsonl(result_dir / "open_fact_labels.jsonl", facts)
    metrics = {}
    for language in ("en","zh"):
        for arm in ("original","neutral","targeted"):
            subset = [r for r in outputs if r["language"] == language and r["variant"] == arm]
            fact_values = defaultdict(list)
            for row in subset:
                fact_values[row["base_fact_id"]].append(row["generation_correct"])
            complete = [sum(v)/len(v) for v in fact_values.values() if None not in v]
            metrics[f"{language}_{arm}"] = {"inputs":len(subset),"correct":sum(r["generation_correct"] is True for r in subset),
                                            "incorrect":sum(r["generation_correct"] is False for r in subset),
                                            "unresolved":sum(r["generation_correct"] is None for r in subset),
                                            "fully_adjudicated_facts":len(complete),
                                            "fact_weighted_accuracy_on_adjudicated":sum(complete)/len(complete) if complete else None}
    table = Counter((str(r["mcq_directed"]),str(r["open_induced_directed"])) for r in facts if r["mcq_directed"] is not None)
    missing = sum(r["generation_correct"] is None for r in outputs)
    complete_facts = sum(r["candidate_coverage_complete"] for r in facts)
    summary = {"status":"complete" if len(outputs)==len(inputs) and complete_facts==len(targets_by_fact) else "partial",
               "semantic_status":"pending_review" if queue else "complete_with_deferrals" if missing else "complete",
               "missing_semantic_judgments":missing,
               "scoring_version":"open_generation_semantics_v3_optional_mcq_variable_candidates",
               "inference_results_sha256":hashlib.sha256((result_dir/"open_results.jsonl").read_bytes()).hexdigest(),
               "semantic_decisions_sha256":hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path.exists() else None,
               "expected_inputs":len(inputs),"assigned_facts":len(targets_by_fact),
               "completed_inputs":len(outputs),"facts_with_complete_variants":complete_facts,
               "facts_with_both_variants":sum(len(targets_by_fact[r["base_fact_id"]]) == 2 and r["candidate_coverage_complete"] for r in facts),
               "mcq_comparison_facts":sum(r["mcq_directed"] is not None for r in facts),"variants":len(variants),
               "semantic_review_unique_outputs":len(queue),"generation_metrics":metrics,
               "fact_counts":{key:dict(Counter(str(r[key]) for r in facts)) for key in ("open_induced_directed","open_induced_any_error","open_language_specific","scored_preference_transfer")},
               "mcq_by_open_directed_fact_table":[{"mcq_directed":m,"open_directed":o,"facts":n} for (m,o),n in sorted(table.items())],
               "budget_exhausted_outputs":sum(r.get("generation_budget_exhausted",False) for r in outputs),
               "validation_facts_used":0,"sealed_facts_used":0,"intervention_run":False,
               "limitations":["Unresolved judgments are missing, never automatic negatives.",
                              "Generation and candidate preference are separate endpoints.",
                              "Variant count is not independent sample size; counts aggregate by fact.",
                              "Codex proxy semantic review is not human gold."]}
    write(result_dir / "bridge_summary.json", summary)
    lines=["# Qwen3-8B Development open-answer bridge", "",f"Status: {summary['status']}; semantic review: {summary['semantic_status']}.",
           f"Completed {len(outputs)}/{len(inputs)} inputs; {complete_facts}/{len(targets_by_fact)} complete facts; {len(queue)} unique outputs pending review.","",
           "| Endpoint (facts) | True | False | Unresolved |","|---|---:|---:|---:|"]
    for key, counts in summary["fact_counts"].items():
        lines.append(f"| {key} | {counts.get('True',0)} | {counts.get('False',0)} | {counts.get('None',0)} |")
    lines += ["","Validation/Sealed remain unused. No intervention has run. Candidate-score transfer is auxiliary, not generated-answer success."]
    (result_dir/"bridge_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "generation_metrics"},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
