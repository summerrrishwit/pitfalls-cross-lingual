#!/usr/bin/env python3
"""Apply an auditable Codex semantic review to Qwen3 natural generations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-dir", type=Path, required=True)
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    natural_dir = args.natural_dir.resolve()
    behavior_dir = args.behavior_dir.resolve()
    decisions_path = args.decisions.resolve()
    results_path = natural_dir / "natural_open_completion_baseline.jsonl"
    summary_path = natural_dir / "natural_baseline_summary.json"
    labels_path = behavior_dir / "exact_hf_defect_control_labels.jsonl"
    results = read_jsonl(results_path)
    summary = read_json(summary_path)
    decisions = read_json(decisions_path)
    labels = read_jsonl(labels_path)

    if len(results) != 192 or Counter(row["language"] for row in results) != Counter(en=96, zh=96):
        raise ValueError("natural baseline is not complete 96 x EN/ZH")
    if summary.get("status") != "complete" or summary.get("semantic_review_status") != "pending_codex_review":
        raise ValueError("natural baseline is not awaiting semantic review")
    if summary["artifacts"]["results"]["sha256"] != sha256_bytes(results_path.read_bytes()):
        raise ValueError("natural result binding changed")
    if decisions.get("reviewer_type") != "codex_proxy" or decisions.get("human_gold") is not False:
        raise ValueError("semantic decisions must remain non-human Codex proxy judgments")

    keys = {f"{row['base_fact_id']}:{row['language']}" for row in results}
    correct_overrides = set(decisions.get("semantic_correct_overrides", []))
    defer_overrides = dict(decisions.get("defer_overrides", {}))
    if not correct_overrides <= keys or not set(defer_overrides) <= keys:
        raise ValueError("semantic decisions reference an unknown natural result")
    if correct_overrides & set(defer_overrides):
        raise ValueError("semantic correct and defer overrides overlap")

    reviewed = []
    for row in results:
        key = f"{row['base_fact_id']}:{row['language']}"
        automatic = bool(row["alias_aware_generation_correct"])
        if automatic:
            decision, semantic_correct, reason = "accept", True, "gold_alias_surface_match"
        elif key in correct_overrides:
            decision, semantic_correct, reason = "accept", True, "codex_semantic_paraphrase_match"
        elif key in defer_overrides:
            decision, semantic_correct, reason = "defer", None, defer_overrides[key]
        else:
            decision, semantic_correct, reason = "reject", False, "generation_not_semantically_equivalent_to_gold"
        reviewed.append({
            "schema_version": "qwen3-8b-natural-semantic-adjudication-v1",
            "review_id": sha256_bytes(key.encode("utf-8"))[:24],
            "base_fact_id": row["base_fact_id"],
            "source_id": row["source_id"],
            "language": row["language"],
            "automatic_alias_match": automatic,
            "generation": row["generation"],
            "answer_aliases": row["answer_aliases"],
            "decision": decision,
            "semantic_correct": semantic_correct,
            "reason": reason,
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "terminal_status": "completed",
        })

    directed_facts = {
        row["base_fact_id"] for row in labels if row["hf_directed_candidate"]
    }
    by_fact: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in reviewed:
        by_fact.setdefault(row["base_fact_id"], {})[row["language"]] = row
    fact_labels = []
    for base_fact_id, language_rows in sorted(by_fact.items()):
        en = language_rows["en"]["semantic_correct"]
        zh = language_rows["zh"]["semantic_correct"]
        eligible = en is not None and zh is not None
        fact_labels.append({
            "schema_version": "qwen3-8b-natural-semantic-fact-label-v1",
            "base_fact_id": base_fact_id,
            "semantic_review_complete": eligible,
            "natural_en_correct": en,
            "natural_zh_correct": zh,
            "natural_pnt_gap": bool(eligible and en and not zh),
            "induced_only_gap": bool(eligible and en and zh and base_fact_id in directed_facts),
            "natural_en_failure": bool(eligible and not en),
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = output_dir / "natural_semantic_adjudications.jsonl"
    fact_labels_path = output_dir / "natural_semantic_fact_labels.jsonl"
    write_jsonl(reviewed_path, reviewed)
    write_jsonl(fact_labels_path, fact_labels)
    decision_counts = Counter(row["decision"] for row in reviewed)
    language_correct = {
        language: sum(row["semantic_correct"] is True for row in reviewed if row["language"] == language)
        for language in ("en", "zh")
    }
    manifest = {
        "schema_version": "qwen3-8b-natural-semantic-review-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_with_deferrals" if decision_counts["defer"] else "complete",
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "decision_counts": dict(decision_counts),
        "semantic_correct_counts": language_correct,
        "fact_label_counts": {
            "natural_pnt_gap": sum(row["natural_pnt_gap"] for row in fact_labels),
            "induced_only_gap": sum(row["induced_only_gap"] for row in fact_labels),
            "natural_en_failure": sum(row["natural_en_failure"] for row in fact_labels),
            "semantic_review_incomplete": sum(not row["semantic_review_complete"] for row in fact_labels),
        },
        "inputs": {
            "natural_results_sha256": sha256_bytes(results_path.read_bytes()),
            "natural_summary_sha256": sha256_bytes(summary_path.read_bytes()),
            "exact_hf_labels_sha256": sha256_bytes(labels_path.read_bytes()),
            "review_decisions_sha256": sha256_bytes(decisions_path.read_bytes()),
        },
        "artifacts": {
            "adjudications": {"path": reviewed_path.name, "sha256": sha256_bytes(reviewed_path.read_bytes())},
            "fact_labels": {"path": fact_labels_path.name, "sha256": sha256_bytes(fact_labels_path.read_bytes())},
        },
        "limitations": [
            "Codex semantic adjudication is proxy review, not independent human gold.",
            "Deferred records are excluded from binary natural-gap strata but remain available for MCQ-conditioned analysis.",
        ],
    }
    write_json(output_dir / "natural_semantic_review_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
