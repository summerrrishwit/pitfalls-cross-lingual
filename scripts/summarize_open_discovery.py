#!/usr/bin/env python3
"""Validate bounded Development discovery ledgers and report independent-fact yield.

This offline checker makes no model requests. A request ledger has one row per
physical attempt, including failures/retries; batch requests list every fact ID.
Candidate rows record semantic attempts, including rejected and failed attempts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return value


def summarize(policy: dict[str, Any], candidates: list[dict[str, Any]],
              requests: list[dict[str, Any]]) -> dict[str, Any]:
    if policy.get("schema_version") != "open-factual-discovery-policy-v1":
        raise ValueError("unsupported discovery policy")
    for name in ("max_facts", "max_candidates_per_fact", "max_generator_strategy_versions",
                 "max_model_request_attempts", "max_total_tokens"):
        if type(policy.get(name)) is not int or policy[name] < 1:
            raise ValueError(f"invalid policy limit {name}")
    if type(policy.get("max_feedback_revisions_per_fact")) is not int or not 0 <= policy["max_feedback_revisions_per_fact"] <= 1:
        raise ValueError("at most one feedback revision is supported")
    if policy["max_candidates_per_fact"] > 1 + policy["max_feedback_revisions_per_fact"]:
        raise ValueError("candidate limit exceeds bounded feedback policy")
    if policy["max_generator_strategy_versions"] > 2:
        raise ValueError("at most an initial strategy and one revision are supported")
    if policy.get("mcq_admission_required") is not False or policy.get("holdout_feedback_allowed") is not False:
        raise ValueError("MCQ admission and holdout feedback must be disabled")
    if policy.get("stop_after_first_reproducible_defect") is not True:
        raise ValueError("stop-on-success policy must be enabled")

    def unique(items: list[dict[str, Any]], key: str) -> set[str]:
        ids = [row.get(key) for row in items]
        if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"missing or duplicate {key}")
        return set(ids)

    candidate_ids = unique(candidates, "candidate_id")
    unique(requests, "request_id")
    facts: set[str] = set()
    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    versions: set[str] = set()
    families: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    reviewed, discovered, reproduced = set(), set(), set()
    known_tokens = 0
    unknown_token_requests = 0
    known_cost = 0.0
    priced_requests = 0
    roles: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    requested_candidates: set[str] = set()
    for row in candidates:
        if row.get("split_assignment") != "development":
            raise ValueError("discovery candidates must be Development only")
        fact = row.get("base_fact_id")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("candidate base_fact_id is required")
        for name in ("generator_strategy_version", "manipulation_family"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"candidate {name} is required")
        for name in ("static_review_passed", "original_correct", "neutral_correct",
                     "targeted_correct", "targeted_distractor_hit", "replication_passed"):
            if row.get(name) is not None and type(row[name]) is not bool:
                raise ValueError(f"{name} must be a boolean or null")
        if type(row.get("attempt_number")) is not int or row["attempt_number"] < 1:
            raise ValueError("invalid attempt_number")
        if row.get("terminal_status") not in {"completed", "static_rejected", "technical_failed", "pending"}:
            raise ValueError("invalid candidate terminal_status")
        if row.get("error_kind") not in {"factual", "refusal", "format", "none", "unresolved"}:
            raise ValueError("invalid error_kind")
        facts.add(fact)
        by_fact[fact].append(row)
        versions.add(row["generator_strategy_version"])
        families[row["manipulation_family"]] += 1
        if row.get("exclusion_reason"):
            exclusions[row["exclusion_reason"]] += 1
        if row.get("static_review_passed") is True:
            reviewed.add(fact)
        positive = (row["terminal_status"] == "completed" and row.get("static_review_passed") is True
                    and row.get("original_correct") is True and row.get("neutral_correct") is True
                    and row.get("targeted_correct") is False and row.get("error_kind") == "factual")
        row_success = positive and row.get("replication_passed") is True
        if row.get("replication_passed") is True and not positive:
            raise ValueError("defect replication cannot pass without a reviewed open defect")
        if positive:
            discovered.add(fact)
        if row_success:
            reproduced.add(fact)

    for fact, attempts in by_fact.items():
        ordered = sorted(attempts, key=lambda row: row["attempt_number"])
        if len(ordered) > policy["max_candidates_per_fact"]:
            raise ValueError(f"candidate budget exceeded for {fact}")
        if [row["attempt_number"] for row in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("candidate attempts must be contiguous from one")
        for earlier in ordered[:-1]:
            if earlier["terminal_status"] not in {"completed", "static_rejected"}:
                raise ValueError("feedback requires a terminal semantic outcome, not pending/technical failure")
            if earlier["terminal_status"] == "completed" and any(earlier.get(k) is None for k in ("original_correct", "neutral_correct", "targeted_correct", "static_review_passed")):
                raise ValueError("feedback cannot treat unresolved judgments as failures")
            if earlier.get("original_correct") is False:
                raise ValueError("baseline failure is not a reason to revise a perturbation")
            success = (earlier["terminal_status"] == "completed" and earlier.get("static_review_passed") is True
                       and earlier.get("original_correct") is True and earlier.get("neutral_correct") is True
                       and earlier.get("targeted_correct") is False and earlier.get("error_kind") == "factual"
                       and earlier.get("replication_passed") is True)
            if success:
                raise ValueError("candidate generated after reproducible success")
    if len(versions) > policy["max_generator_strategy_versions"]:
        raise ValueError("generator strategy revision budget exceeded")

    for row in requests:
        if row.get("split_assignment") != "development":
            raise ValueError("discovery requests must be Development only")
        ids = row.get("base_fact_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(v, str) or not v.strip() for v in ids):
            raise ValueError("request base_fact_ids are required")
        facts.update(ids)
        references = row.get("candidate_ids", [])
        if not isinstance(references, list) or any(value not in candidate_ids for value in references):
            raise ValueError("request refers to unknown candidate IDs")
        requested_candidates.update(references)
        if row.get("role") not in {"baseline", "generator", "reviewer", "target", "replication", "mcq_aux"}:
            raise ValueError("invalid request role")
        if row.get("terminal_status") not in {"completed", "failed", "in_flight"}:
            raise ValueError("invalid request terminal_status")
        roles[row["role"]] += 1
        statuses[row["terminal_status"]] += 1
        tokens = row.get("total_tokens")
        if tokens is None:
            unknown_token_requests += 1
        elif type(tokens) is not int or tokens < 0:
            raise ValueError("invalid total_tokens")
        else:
            known_tokens += tokens
        cost = row.get("cost_usd")
        if cost is not None:
            known_cost += nonnegative_number(cost, "cost_usd")
            priced_requests += 1
    if candidate_ids - requested_candidates:
        raise ValueError("candidate attempts lack request accounting")
    if len(facts) > policy["max_facts"] or len(requests) > policy["max_model_request_attempts"] or known_tokens > policy["max_total_tokens"]:
        raise ValueError("discovery fact/request/token budget exceeded")

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    neutral_eligible = [r for r in candidates if r.get("original_correct") is True and r.get("neutral_correct") is not None]
    directed = {r["base_fact_id"] for r in candidates if r["base_fact_id"] in reproduced
                and r.get("static_review_passed") is True and r.get("original_correct") is True
                and r.get("neutral_correct") is True and r.get("targeted_correct") is False
                and r.get("targeted_distractor_hit") is True and r.get("replication_passed") is True
                and r.get("error_kind") == "factual" and r["terminal_status"] == "completed"}
    total_tokens = None if unknown_token_requests else known_tokens
    total_cost = known_cost if requests and priced_requests == len(requests) else None
    review_counts = Counter(str(r.get("static_review_passed")) for r in candidates)
    excluded = sum(r.get("static_review_passed") is False or bool(r.get("exclusion_reason")) for r in candidates)
    return {
        "schema_version": "open-discovery-efficiency-summary-v1",
        "request_budget_status": "within_recorded_limit",
        "token_budget_status": "unverified_missing_usage" if unknown_token_requests else "within_recorded_limit",
        "assigned_or_attempted_facts": len(facts), "candidate_attempts": len(candidates),
        "reviewed_facts": len(reviewed), "open_defect_facts": len(discovered),
        "reproducible_defect_facts": len(reproduced), "reproducible_directed_facts": len(directed),
        "reproducible_facts_per_100_requests": 100 * len(reproduced) / len(requests) if requests else None,
        "reproducible_fact_yield": ratio(len(reproduced), len(facts)),
        "neutral_stability_candidate_counts": {"stable": sum(r["neutral_correct"] for r in neutral_eligible), "assessed": len(neutral_eligible)},
        "neutral_stability_on_assessed_candidates": ratio(sum(r["neutral_correct"] for r in neutral_eligible), len(neutral_eligible)),
        "static_review_candidate_counts": dict(review_counts),
        "static_review_exclusion_rate_on_adjudicated": ratio(review_counts["False"], review_counts["False"] + review_counts["True"]),
        "excluded_candidate_attempts": excluded, "excluded_candidate_attempt_rate": ratio(excluded, len(candidates)),
        "candidate_terminal_counts": dict(Counter(r["terminal_status"] for r in candidates)),
        "exclusion_reason_counts": dict(exclusions), "manipulation_family_candidate_counts": dict(families),
        "request_attempts": len(requests), "request_role_counts": dict(roles),
        "request_terminal_counts": dict(statuses), "total_tokens": total_tokens,
        "known_tokens": known_tokens, "unknown_token_requests": unknown_token_requests,
        "reproducible_facts_per_million_tokens": 1_000_000 * len(reproduced) / total_tokens if total_tokens else None,
        "total_cost_usd": total_cost, "known_cost_usd": known_cost,
        "cost_usd_per_reproducible_fact": total_cost / len(reproduced) if total_cost is not None and reproduced else None,
        "mcq_admission_required": False, "new_model_execution_performed": False,
        "limitations": ["Counters validate supplied ledgers, not completeness of external provider logs.",
                        "Missing usage/pricing is unavailable, never a zero budget.",
                        "Candidate/variant/language counts are not independent-fact sample sizes."]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.policy.read_text()), read_rows(args.candidates), read_rows(args.requests))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
