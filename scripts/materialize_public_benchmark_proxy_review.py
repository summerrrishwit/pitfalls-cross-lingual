#!/usr/bin/env python3
"""Materialize complete Codex-proxy fact-review decisions from an audited plan.

The generated rows are inputs to ``review_public_benchmark_bundle.py apply``.
They remain non-human, staging-only evidence and never promote or freeze facts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "public-benchmark-proxy-review-materializer-v1"
PLAN_SCHEMA_VERSION = "public-benchmark-proxy-review-plan-v1"
MANIFEST_SCHEMA_VERSION = "public-benchmark-proxy-review-materialization-v1"
ALLOWED_DECISIONS = frozenset({"accept", "reject", "defer", "revise"})
ALLOWED_TARGET_DECISIONS = frozenset({"accept", "reject"})


def _load_review_tool() -> Any:
    script_path = Path(__file__).resolve().with_name("review_public_benchmark_bundle.py")
    spec = importlib.util.spec_from_file_location("public_benchmark_review_tool", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load review tool: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review_tool = _load_review_tool()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _review_metadata(review: Mapping[str, Any], label: str) -> Dict[str, str]:
    if review.get("reviewer_type") != "codex_proxy":
        raise ValueError(f"{label}.reviewer_type must be codex_proxy")
    if review.get("human_gold") is not False:
        raise ValueError(f"{label} must keep human_gold=false")
    reviewed_at = _required_string(review.get("reviewed_at"), f"{label}.reviewed_at")
    try:
        parsed = dt.datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}.reviewed_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label}.reviewed_at must include a timezone")
    return {
        "reviewer_type": "codex_proxy",
        "reviewer_id": _required_string(review.get("reviewer_id"), f"{label}.reviewer_id"),
        "review_method": _required_string(
            review.get("review_method"), f"{label}.review_method"
        ),
        "reviewed_at": reviewed_at,
    }


def _target_decision_map(
    override: Mapping[str, Any],
    *,
    field: str,
    allowed_ids: set[str],
    base_fact_id: str,
) -> Dict[str, str]:
    value = override.get(field, {})
    if not isinstance(value, dict):
        raise ValueError(f"{base_fact_id}.{field} must be an object")
    unknown = sorted(set(value) - allowed_ids)
    if unknown:
        raise ValueError(f"Unknown {field} targets for {base_fact_id}: {unknown}")
    invalid = sorted(
        target_id
        for target_id, decision in value.items()
        if decision not in ALLOWED_TARGET_DECISIONS
    )
    if invalid:
        raise ValueError(
            f"{base_fact_id}.{field} decisions must be accept or reject: {invalid}"
        )
    return {str(target_id): str(decision) for target_id, decision in value.items()}


def _load_export(
    export_manifest_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    export_manifest_path = Path(export_manifest_path).resolve()
    export_manifest = read_json(export_manifest_path)
    scope_binding = export_manifest.get("scope_manifest")
    if not isinstance(scope_binding, dict):
        raise ValueError("Review export manifest is missing scope_manifest")
    scope_path = Path(_required_string(scope_binding.get("path"), "scope_manifest.path"))
    scope = read_json(scope_path)
    review_tool._validate_scope_manifest(scope)
    items_path, templates_path = review_tool._validate_export_manifest(
        export_manifest, scope, scope_path
    )
    items = review_tool.read_jsonl(items_path)
    templates = review_tool.read_jsonl(templates_path)
    if len(items) != len(templates):
        raise ValueError("Review items and templates have different lengths")
    for item, template in zip(items, templates):
        if item.get("base_fact_id") != template.get("base_fact_id"):
            raise ValueError("Review item/template ordering mismatch")
        if template.get("review_item_sha256") != review_tool.sha256_value(item):
            raise ValueError(f"Stale review decision template: {item.get('base_fact_id')}")
    return export_manifest, items, templates


def materialize(
    *, export_manifest_path: Path, review_plan_path: Path, output_dir: Path
) -> Dict[str, Any]:
    export_manifest_path = Path(export_manifest_path).resolve()
    review_plan_path = Path(review_plan_path).resolve()
    export_manifest, items, templates = _load_export(export_manifest_path)
    plan = read_json(review_plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported proxy review plan schema_version")
    if plan.get("export_manifest_sha256") != sha256_file(export_manifest_path):
        raise ValueError("Proxy review plan is not bound to the export manifest")
    family_reviews = plan.get("family_reviews")
    if not isinstance(family_reviews, dict):
        raise ValueError("Proxy review plan lacks family_reviews")
    item_families = {
        str(item["review_context"]["behavior_row"].get("probe_relation_candidate"))
        for item in items
    }
    if set(family_reviews) != item_families:
        raise ValueError(
            "Proxy review family coverage mismatch; "
            f"missing={sorted(item_families - set(family_reviews))} "
            f"extra={sorted(set(family_reviews) - item_families)}"
        )
    item_by_id = {str(item["base_fact_id"]): item for item in items}
    all_ids = set(item_by_id)
    for family, review in family_reviews.items():
        if not isinstance(review, dict) or review.get("reviewed_all_items") is not True:
            raise ValueError(f"Family review is incomplete: {family}")
        overrides = review.get("item_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"item_overrides must be an object: {family}")
        unknown = set(overrides) - all_ids
        if unknown:
            raise ValueError(f"Unknown item overrides for {family}: {sorted(unknown)[:5]}")
        wrong_family = sorted(
            base_fact_id
            for base_fact_id in overrides
            if str(
                item_by_id[base_fact_id]["review_context"]["behavior_row"].get(
                    "probe_relation_candidate"
                )
            )
            != family
        )
        if wrong_family:
            raise ValueError(
                f"Item overrides for {family} target another family: {wrong_family[:5]}"
            )

    decisions: List[Dict[str, Any]] = []
    for item, template in zip(items, templates):
        base_fact_id = str(item["base_fact_id"])
        family = str(item["review_context"]["behavior_row"].get("probe_relation_candidate"))
        review = family_reviews[family]
        metadata = _review_metadata(review, f"family review {family}")
        override = review.get("item_overrides", {}).get(base_fact_id, {})
        if not isinstance(override, dict):
            raise ValueError(f"Invalid item override: {base_fact_id}")
        decision = override.get("decision", review.get("default_decision"))
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Invalid decision for {base_fact_id}: {decision}")
        reason = _required_string(
            override.get("reason", review.get("default_reason")),
            f"decision {base_fact_id}.reason",
        )
        output = dict(template)
        output["decision"] = decision
        output["notes"] = reason
        output["human_gold"] = False
        output["review_provenance"] = metadata
        member_targets = item["members"]
        alias_targets = item["review_targets"]["aliases"]
        distractor_targets = item["review_targets"]["distractors"]
        member_overrides = _target_decision_map(
            override,
            field="member_decisions",
            allowed_ids={str(target["candidate_id"]) for target in member_targets},
            base_fact_id=base_fact_id,
        )
        alias_overrides = _target_decision_map(
            override,
            field="alias_decisions",
            allowed_ids={str(target["alias_id"]) for target in alias_targets},
            base_fact_id=base_fact_id,
        )
        distractor_overrides = _target_decision_map(
            override,
            field="distractor_decisions",
            allowed_ids={str(target["distractor_id"]) for target in distractor_targets},
            base_fact_id=base_fact_id,
        )
        member_default = "reject" if decision == "reject" else "accept"
        output["member_reviews"] = [
            {
                "candidate_id": target["candidate_id"],
                "triple_record_sha256": target["triple_record_sha256"],
                "decision": member_overrides.get(
                    str(target["candidate_id"]), member_default
                ),
            }
            for target in member_targets
        ]
        if decision == "accept":
            output["alias_reviews"] = [
                {
                    **target,
                    "decision": alias_overrides.get(str(target["alias_id"]), "accept"),
                }
                for target in alias_targets
            ]
            output["distractor_reviews"] = [
                {
                    **target,
                    "decision": distractor_overrides.get(
                        str(target["distractor_id"]), "accept"
                    ),
                }
                for target in distractor_targets
            ]
        else:
            if alias_overrides or distractor_overrides:
                raise ValueError(
                    f"Target-level alias/distractor decisions require accept: {base_fact_id}"
                )
            output["alias_reviews"] = list(template.get("alias_reviews", []))
            output["distractor_reviews"] = list(template.get("distractor_reviews", []))
        decisions.append(output)

    for item, decision in zip(items, decisions):
        review_tool._validate_decision(decision, item)
    output_dir = Path(output_dir).resolve()
    decisions_path = output_dir / "review_decisions_codex_proxy.jsonl"
    write_jsonl(decisions_path, decisions)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "export_manifest": {
            "path": str(export_manifest_path),
            "sha256": sha256_file(export_manifest_path),
        },
        "review_plan": {
            "path": str(review_plan_path),
            "sha256": sha256_file(review_plan_path),
        },
        "decisions": {
            "path": str(decisions_path),
            "sha256": sha256_file(decisions_path),
            "record_count": len(decisions),
            "schema_version": review_tool.DECISION_SCHEMA_VERSION,
        },
        "decision_counts": dict(
            sorted(Counter(row["decision"] for row in decisions).items())
        ),
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "safety_contract": {
            "decision_materialization_only": True,
            "canonical_freeze_performed": False,
            "automatic_promotion_performed": False,
            "perturbation_performed": False,
        },
    }
    manifest_path = output_dir / "proxy_review_materialization_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "decisions_path": str(decisions_path),
        "decisions_sha256": sha256_file(decisions_path),
        "decision_counts": manifest["decision_counts"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--export-manifest", type=Path, required=True)
    result.add_argument("--review-plan", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = materialize(
        export_manifest_path=args.export_manifest,
        review_plan_path=args.review_plan,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
