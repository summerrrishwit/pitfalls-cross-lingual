#!/usr/bin/env python3
"""Materialize the frozen Codex canonical-triple adjudication JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


REVIEW_PROMPT_VERSION = "canonical-triple-adjudication-v1"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_ids(path: Path) -> set[str]:
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return values


def rejection_annotation(row: Dict[str, Any]) -> Dict[str, Any]:
    eligible = set(row["eligible_selected_models"])
    for model, annotation in row["model_annotations"].items():
        if model not in eligible:
            return annotation
    raise ValueError(f"single-model row has no rejecting annotation: {row['source_id']}")


def accepted_single(row: Dict[str, Any]) -> Dict[str, Any]:
    selected_model = row["eligible_selected_models"][0]
    fact = row["model_annotations"][selected_model].get("canonical_fact") or "the proposed atomic fact"
    return {
        "source_id": row["source_id"],
        "review_case": row["review_case"],
        "codex_decision": "accept",
        "selected_model": selected_model,
        "reason_code": "atomic_fact_supported",
        "rationale": f"The proposed triple directly expresses one stable fact supported by the canonical answer: {fact}",
        "review_prompt_version": REVIEW_PROMPT_VERSION,
    }


def rejected_single(row: Dict[str, Any]) -> Dict[str, Any]:
    reason = rejection_annotation(row).get("exclusion_reason") or "atomic_contract_violation"
    phrase = str(reason).replace("_", " ")
    return {
        "source_id": row["source_id"],
        "review_case": row["review_case"],
        "codex_decision": "reject",
        "selected_model": None,
        "reason_code": "atomic_contract_violation",
        "paired_model_exclusion_reason": reason,
        "rationale": (
            "Independent Codex review found that this proposed relation does not satisfy the "
            f"frozen atomic-triple criteria; this aligns with the paired model category: {phrase}."
        ),
        "review_prompt_version": REVIEW_PROMPT_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--accepted-ids", required=True)
    parser.add_argument("--conflict-decisions", required=True)
    parser.add_argument("--prior-adjudication")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queue_path = Path(args.queue).resolve()
    accept_path = Path(args.accepted_ids).resolve()
    conflict_path = Path(args.conflict_decisions).resolve()
    output_path = Path(args.output).resolve()
    queue = read_jsonl(queue_path)
    accepts = accepted_ids(accept_path)
    single_ids = {row["source_id"] for row in queue if row["review_case"] == "single_model_extraction"}
    if not accepts <= single_ids:
        raise ValueError(f"accepted IDs outside the single-model queue: {sorted(accepts - single_ids)[:10]}")

    conflict_records = {
        row["source_id"]: row
        for row in read_jsonl(conflict_path)
        if row.get("review_case") == "dual_model_answer_conflict"
    }
    conflict_ids = {
        row["source_id"] for row in queue if row["review_case"] == "dual_model_answer_conflict"
    }
    if set(conflict_records) != conflict_ids:
        raise ValueError("conflict decision coverage does not exactly match the conflict queue")

    prior_by_id: Dict[str, Dict[str, Any]] = {}
    if args.prior_adjudication:
        prior = json.loads(Path(args.prior_adjudication).read_text(encoding="utf-8"))
        prior_by_id = {item["candidate_id"]: item for item in prior.get("decisions", [])}

    records = []
    reused_prior = 0
    for row in queue:
        source_id = row["source_id"]
        if row["review_case"] == "dual_model_answer_conflict":
            records.append(conflict_records[source_id])
            continue
        prior = prior_by_id.get(source_id)
        if prior is not None:
            prior_accept = prior["final_status"] == "extracted"
            if prior_accept != (source_id in accepts):
                raise ValueError(f"current decision conflicts with prior Codex adjudication: {source_id}")
            record = accepted_single(row) if prior_accept else rejected_single(row)
            record["rationale"] = prior["reason"]
            record["review_provenance"] = "reused_codex_dual-model-adjudication-v1"
            records.append(record)
            reused_prior += 1
            continue
        records.append(accepted_single(row) if source_id in accepts else rejected_single(row))

    if len(records) != len(queue) or len({item["source_id"] for item in records}) != len(queue):
        raise ValueError("adjudication output does not conserve unique queue IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    summary = {
        "adjudication_version": "codex-canonical-triple-adjudication-v1",
        "review_prompt_version": REVIEW_PROMPT_VERSION,
        "queue_path": str(queue_path),
        "queue_sha256": file_sha256(queue_path),
        "accepted_ids_path": str(accept_path),
        "accepted_ids_sha256": file_sha256(accept_path),
        "record_count": len(records),
        "decision_counts": dict(Counter(item["codex_decision"] for item in records)),
        "review_case_counts": dict(Counter(item["review_case"] for item in records)),
        "accepted_by_dataset": dict(
            Counter(
                next(row["source_dataset"] for row in queue if row["source_id"] == item["source_id"])
                for item in records
                if item["codex_decision"] == "accept"
            )
        ),
        "reused_prior_codex_decision_count": reused_prior,
        "coverage_complete": True,
    }
    write_json(output_path.with_name("canonical_adjudication_codex_v1_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
