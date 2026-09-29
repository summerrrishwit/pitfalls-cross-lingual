#!/usr/bin/env python3
"""Materialize an offline zh-accepted secondary projection.

The input candidate projection and zh review are immutable parents.  No model
is called.  Rows quarantined by the terminal zh review are removed as targets,
while the complete parent candidate projection remains the immutable support
cohort.  Surviving facts keep their exact parent fact, selected 1+1 candidate,
translation, component, and split evidence; candidates are never reselected
and structural snapshots are never rewritten.  An excluded target may remain
as a support-only donor, but can never become a behavior or vector target.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TOOL_VERSION = "full-public-benchmark-zh-accepted-secondary-projection-v1"
MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-full-zh-accepted-secondary-projection-manifest-v1"
)
ITEM_SCHEMA_VERSION = "public-benchmark-full-zh-accepted-secondary-projection-item-v1"
EXCLUSION_SCHEMA_VERSION = (
    "public-benchmark-full-zh-accepted-secondary-projection-exclusion-v1"
)
REVIEW_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-full-zh-accepted-secondary-review-manifest-v1"
)
STATUS = "completed_offline_zh_accepted_secondary_projection"
REVIEW_STATUS = "completed_offline_zh_accepted_review_projection"

MANIFEST_NAME = "zh_accepted_projection_manifest.json"
ITEMS_NAME = "zh_accepted_projection_items.jsonl"
EXCLUSIONS_NAME = "zh_accepted_projection_exclusions.jsonl"
REVIEW_MANIFEST_NAME = "zh_accepted_review_manifest.json"
REVIEW_RECORDS_NAME = "zh_translation_review_records.jsonl"
EMPTY_QUARANTINE_NAME = "translation_quarantine.jsonl"


def _load_sibling(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


accepted_projection_tool = _load_sibling(
    "_zh_secondary_accepted_projection",
    "materialize_full_public_benchmark_accepted_projection.py",
)
zh_tool = _load_sibling(
    "_zh_secondary_parent_review", "run_full_public_benchmark_zh_review.py"
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )


def _resolve_binding_path(
    binding: Mapping[str, Any], owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label}.path is missing")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (owner_path.parent / path).resolve()


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(Path(output_path).resolve() if output_path else path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if rows is not None:
        result["record_count"] = len(rows)
        result["schema_versions"] = sorted(
            {
                str(row.get("schema_version"))
                for row in rows
                if row.get("schema_version") is not None
            }
        )
    return result


def _verify_jsonl_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: str,
    allow_empty: bool = False,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    path = _resolve_binding_path(binding, owner_path, label)
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"{label} SHA-256 binding is stale")
    if path.stat().st_size != binding.get("byte_count"):
        raise ValueError(f"{label} byte_count binding is stale")
    rows = read_jsonl(path, allow_empty=allow_empty)
    if binding.get("record_count") != len(rows):
        raise ValueError(f"{label} record_count binding is stale")
    schemas = sorted(
        {
            str(row.get("schema_version"))
            for row in rows
            if row.get("schema_version") is not None
        }
    )
    if rows and schemas != [expected_schema]:
        raise ValueError(f"{label} schema_version is unsupported")
    if binding.get("schema_versions") != schemas:
        raise ValueError(f"{label} schema_versions binding is stale")
    return path, rows, dict(binding)


def _same_binding(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    owner_path: Path,
    label: str,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} binding is missing")
    if _resolve_binding_path(actual, owner_path, label) != Path(
        str(expected["path"])
    ).resolve():
        raise ValueError(f"{label} path differs from the bound parent artifact")
    for field in ("sha256", "byte_count", "record_count", "schema_version"):
        if field in expected and actual.get(field) != expected.get(field):
            raise ValueError(f"{label}.{field} differs from the bound parent artifact")


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(field) or "")
        if not identifier or identifier in result:
            raise ValueError(f"{label} has a missing or duplicate {field}: {identifier}")
        result[identifier] = dict(row)
    return result


def _contract_fingerprint(manifest: Mapping[str, Any]) -> str:
    fields = (
        "schema_version",
        "tool_version",
        "input_bindings",
        "source_fingerprints",
        "route_identity",
        "review_independence",
        "formal_universe_id",
        "formal_universe_status",
        "selected_base_fact_ids_sha256",
        "selected_record_count",
        "universe_record_count",
        "selection_is_full_universe",
        "selected_split_counts",
        "models",
        "candidate_binding",
        "invalidation_contract",
        "language_scope",
        "evidence_boundary",
        "execution_policy",
    )
    payload = {field: manifest.get(field) for field in fields}
    for field in (
        "successor_cohort",
        "accepted_candidate_projection",
        "selection_is_complete_retained_cohort",
        "excluded_base_fact_ids_sha256",
        "selection_is_complete_accepted_projection",
        "accepted_projection_record_count",
        "accepted_projection_quarantined_source_record_count",
    ):
        if field in manifest:
            payload[field] = manifest[field]
    return sha256_value(payload)


def _stage_index(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_schema: str,
    expected_stage: str,
    run_fingerprint: str,
    items_by_id: Mapping[str, Mapping[str, Any]],
    prompt_version: str,
    model_spec: Mapping[str, Any],
    parsed_validator: Callable[[Dict[str, Any], Mapping[str, Any]], None],
    reviewer: bool,
    label: str,
) -> Dict[str, Dict[str, Any]]:
    result = _unique_index(rows, "base_fact_id", label)
    expected_response_model = model_spec.get("expected_response_model")
    if not isinstance(expected_response_model, str) or not expected_response_model:
        raise ValueError(f"{label} expected response model is missing")
    for base_fact_id, row in result.items():
        item = items_by_id.get(base_fact_id)
        if item is None:
            raise ValueError(f"{label} contains an out-of-scope base_fact_id")
        if (
            row.get("run_fingerprint") != run_fingerprint
            or row.get("input_record_sha256") != item.get("input_record_sha256")
        ):
            raise ValueError(f"{label} row is stale: {base_fact_id}")
        try:
            zh_tool._validate_stage_checkpoint_record(
                row,
                item,
                stage=expected_stage,
                schema_version=expected_schema,
                prompt_version=prompt_version,
                model_spec=model_spec,
                expected_response_model=expected_response_model,
                parsed_validator=parsed_validator,
            )
        except ValueError as exc:
            raise ValueError(f"{label} row is invalid: {base_fact_id}: {exc}") from exc
        if row.get("terminal_status") != "completed":
            raise ValueError(f"{label} row is not completed: {base_fact_id}")
        if reviewer and row.get("reviewer_type") != "independent_model_proxy":
            raise ValueError(f"{label} reviewer type is invalid: {base_fact_id}")
    return result


def _load_parent_context(
    *,
    accepted_candidate_projection_manifest_path: Path,
    zh_review_manifest_path: Path,
) -> Dict[str, Any]:
    projection_path = Path(accepted_candidate_projection_manifest_path).resolve()
    zh_manifest_path = Path(zh_review_manifest_path).resolve()
    projection = accepted_projection_tool.load_projection_context(projection_path)
    manifest = read_json(zh_manifest_path)
    if manifest.get("schema_version") != zh_tool.MANIFEST_SCHEMA:
        raise ValueError("parent zh manifest schema_version is unsupported")
    if manifest.get("tool_version") != zh_tool.TOOL_VERSION:
        raise ValueError("parent zh manifest tool_version is unsupported")
    if manifest.get("source_fingerprints") != zh_tool.source_fingerprints():
        raise ValueError("parent zh source fingerprints are stale")
    if manifest.get("run_fingerprint") != _contract_fingerprint(manifest):
        raise ValueError("parent zh run_fingerprint is stale")
    if manifest.get("status") not in {
        "completed_accepted_candidate_projection_zh_proxy_review",
        "completed_with_quarantine_or_limited_scope",
    }:
        raise ValueError("parent zh review is not terminal for the accepted projection")

    input_bindings = manifest.get("input_bindings")
    if not isinstance(input_bindings, dict):
        raise ValueError("parent zh input_bindings are missing")
    formal_manifest_path = _resolve_binding_path(
        input_bindings.get("formal_universe_manifest", {}),
        zh_manifest_path,
        "parent formal universe manifest",
    )
    formal_items_path = _resolve_binding_path(
        input_bindings.get("formal_universe_items", {}),
        zh_manifest_path,
        "parent formal universe items",
    )
    loaded = zh_tool.load_bound_inputs(
        formal_universe_manifest_path=formal_manifest_path,
        formal_universe_items_path=formal_items_path,
        full_base_facts_path=None,
        distractor_candidates_path=None,
        neutral_candidates_path=None,
        split_manifest_path=None,
        accepted_candidate_projection_manifest_path=projection_path,
    )
    for label, expected in loaded["bindings"].items():
        _same_binding(
            input_bindings.get(label),
            expected,
            owner_path=zh_manifest_path,
            label=f"parent zh {label}",
        )
    if manifest.get("candidate_binding") != loaded["candidate_binding"]:
        raise ValueError("parent zh candidate binding is stale")
    if manifest.get("accepted_candidate_projection") != loaded[
        "accepted_candidate_projection"
    ]:
        raise ValueError("parent zh accepted projection summary is stale")

    target_ids = [str(row["base_fact_id"]) for row in loaded["items"]]
    if target_ids != [str(value) for value in projection["projected_target_ids"]]:
        raise ValueError("parent zh target order differs from the candidate projection")
    if not all(
        (
            manifest.get("selection_is_complete_accepted_projection") is True,
            manifest.get("selected_record_count") == len(target_ids),
            manifest.get("accepted_projection_record_count") == len(target_ids),
            manifest.get("selected_base_fact_ids_sha256") == sha256_value(target_ids),
            manifest.get("selected_split_counts") == loaded["split_counts"],
            manifest.get("selection_is_full_universe") is False,
        )
    ):
        raise ValueError("parent zh selected scope is incomplete or stale")

    output_bindings = manifest.get("output_artifacts")
    if not isinstance(output_bindings, dict):
        raise ValueError("parent zh output_artifacts are missing")
    specs = {
        "translation_initial": (zh_tool.GENERATION_SCHEMA, False),
        "translation_initial_reviews": (zh_tool.REVIEW_SCHEMA, False),
        "translation_repairs": (zh_tool.GENERATION_SCHEMA, True),
        "translation_repair_reviews": (zh_tool.REVIEW_SCHEMA, True),
        "zh_translation_review_records": (zh_tool.FINAL_SCHEMA, False),
        "translation_quarantine": (zh_tool.QUARANTINE_SCHEMA, True),
    }
    artifacts: Dict[str, Dict[str, Any]] = {}
    for label, (schema, allow_empty) in specs.items():
        path, rows, binding = _verify_jsonl_binding(
            output_bindings.get(label),
            owner_path=zh_manifest_path,
            label=f"parent zh {label}",
            expected_schema=schema,
            allow_empty=allow_empty,
        )
        artifacts[label] = {"path": path, "rows": rows, "binding": binding}

    items_by_id = {str(row["base_fact_id"]): dict(row) for row in loaded["items"]}
    run_fingerprint = str(manifest["run_fingerprint"])
    models = manifest.get("models")
    if (
        not isinstance(models, dict)
        or not isinstance(models.get("generator"), dict)
        or not isinstance(models.get("reviewer"), dict)
    ):
        raise ValueError("parent zh model identities are missing")
    generator_model = models["generator"]
    reviewer_model = models["reviewer"]
    initial = _stage_index(
        artifacts["translation_initial"]["rows"],
        expected_schema=zh_tool.GENERATION_SCHEMA,
        expected_stage="initial_generation",
        run_fingerprint=run_fingerprint,
        items_by_id=items_by_id,
        prompt_version=zh_tool.PROMPT_VERSION,
        model_spec=generator_model,
        parsed_validator=zh_tool.validate_translation,
        reviewer=False,
        label="parent zh initial",
    )
    initial_reviews = _stage_index(
        artifacts["translation_initial_reviews"]["rows"],
        expected_schema=zh_tool.REVIEW_SCHEMA,
        expected_stage="initial_review",
        run_fingerprint=run_fingerprint,
        items_by_id=items_by_id,
        prompt_version=zh_tool.REVIEW_PROMPT_VERSION,
        model_spec=reviewer_model,
        parsed_validator=zh_tool.validate_review,
        reviewer=True,
        label="parent zh initial reviews",
    )
    repairs = _stage_index(
        artifacts["translation_repairs"]["rows"],
        expected_schema=zh_tool.GENERATION_SCHEMA,
        expected_stage="repair_1",
        run_fingerprint=run_fingerprint,
        items_by_id=items_by_id,
        prompt_version=zh_tool.REPAIR_PROMPT_VERSION,
        model_spec=generator_model,
        parsed_validator=zh_tool.validate_translation,
        reviewer=False,
        label="parent zh repairs",
    )
    repair_reviews = _stage_index(
        artifacts["translation_repair_reviews"]["rows"],
        expected_schema=zh_tool.REVIEW_SCHEMA,
        expected_stage="repair_1_review",
        run_fingerprint=run_fingerprint,
        items_by_id=items_by_id,
        prompt_version=zh_tool.REPAIR_REVIEW_PROMPT_VERSION,
        model_spec=reviewer_model,
        parsed_validator=zh_tool.validate_review,
        reviewer=True,
        label="parent zh repair reviews",
    )
    if set(initial) != set(target_ids) or set(initial_reviews) != set(target_ids):
        raise ValueError("parent zh initial evidence does not cover the projection")
    rejected_initial = {
        base_fact_id
        for base_fact_id, row in initial_reviews.items()
        if row.get("parsed_response", {}).get("decision") != "accept"
    }
    if set(repairs) != rejected_initial or set(repair_reviews) != rejected_initial:
        raise ValueError("parent zh repair evidence does not cover initial rejects")

    for base_fact_id in target_ids:
        item = items_by_id[base_fact_id]
        zh_tool.validate_translation(initial[base_fact_id]["parsed_response"], item)
        zh_tool.validate_review(initial_reviews[base_fact_id]["parsed_response"], item)
        if initial_reviews[base_fact_id].get("translation_record_sha256") != sha256_value(
            initial[base_fact_id]
        ):
            raise ValueError(f"parent zh initial review lineage is stale: {base_fact_id}")
    for base_fact_id in rejected_initial:
        item = items_by_id[base_fact_id]
        zh_tool.validate_translation(repairs[base_fact_id]["parsed_response"], item)
        zh_tool.validate_review(repair_reviews[base_fact_id]["parsed_response"], item)
        if repairs[base_fact_id].get("repaired_translation_record_sha256") != sha256_value(
            initial[base_fact_id]
        ) or repairs[base_fact_id].get("repaired_review_record_sha256") != sha256_value(
            initial_reviews[base_fact_id]
        ):
            raise ValueError(f"parent zh repair lineage is stale: {base_fact_id}")
        if repair_reviews[base_fact_id].get("translation_record_sha256") != sha256_value(
            repairs[base_fact_id]
        ):
            raise ValueError(f"parent zh repair review lineage is stale: {base_fact_id}")

    final_rows = artifacts["zh_translation_review_records"]["rows"]
    if [str(row.get("base_fact_id") or "") for row in final_rows] != target_ids:
        raise ValueError("parent zh final rows do not preserve projection order")
    final_by_id = _unique_index(final_rows, "base_fact_id", "parent zh final rows")
    quarantine_rows = artifacts["translation_quarantine"]["rows"]
    quarantine_by_id = _unique_index(
        quarantine_rows, "base_fact_id", "parent zh quarantine"
    )
    expected_quarantine_ids: List[str] = []
    for base_fact_id in target_ids:
        item = items_by_id[base_fact_id]
        initial_review = initial_reviews[base_fact_id]
        repaired = repairs.get(base_fact_id)
        repaired_review = repair_reviews.get(base_fact_id)
        initial_accepted = initial_review["parsed_response"].get("decision") == "accept"
        repair_accepted = bool(
            repaired_review
            and repaired_review["parsed_response"].get("decision") == "accept"
        )
        accepted = initial_accepted or repair_accepted
        origin = "initial" if initial_accepted else (
            "repair_1" if repair_accepted else "repair_1_rejected"
        )
        generation = initial[base_fact_id] if initial_accepted else repaired
        review = initial_review if initial_accepted else repaired_review
        reasons = [] if accepted else ["repair_review_rejected"]
        row = final_by_id[base_fact_id]
        expected = {
            "schema_version": zh_tool.FINAL_SCHEMA,
            "base_fact_id": base_fact_id,
            "cohort_item_id": item.get("cohort_item_id"),
            "input_record_sha256": item["input_record_sha256"],
            "split_assignment": item["split_assignment"],
            "target_language": "zh",
            "human_gold": False,
            "reviewer_type": "independent_model_proxy",
            "terminal_status": "completed" if accepted else "quarantined",
            "translation_origin": origin,
            "source_fields": item["source"],
            "source_distractors": item["distractors"],
            "source_neutral_candidates": item["neutral_candidates"],
            "translation": generation["parsed_response"] if generation else None,
            "translation_equivalence_review": review["parsed_response"] if review else None,
            "quarantine_reasons": reasons,
            "maximum_translation_repairs_performed": 0 if initial_accepted else 1,
            "formal_claims": {
                "translation_proxy_accepted": accepted,
                "human_gold": False,
                "factual_truth_reviewed": False,
                "distractor_falsehood_reviewed": False,
                "neutral_unrelatedness_reviewed": False,
            },
            "lineage": {
                "initial_generation_record_sha256": sha256_value(initial[base_fact_id]),
                "initial_review_record_sha256": sha256_value(initial_review),
                "repair_record_sha256": sha256_value(repaired) if repaired else None,
                "repair_review_record_sha256": (
                    sha256_value(repaired_review) if repaired_review else None
                ),
            },
        }
        if row != expected:
            raise ValueError(f"parent zh final row is stale: {base_fact_id}")
        if not accepted:
            expected_quarantine_ids.append(base_fact_id)
            expected_quarantine = {
                "schema_version": zh_tool.QUARANTINE_SCHEMA,
                "base_fact_id": base_fact_id,
                "input_record_sha256": item["input_record_sha256"],
                "target_language": "zh",
                "human_gold": False,
                "reasons": reasons,
                "final_record_sha256": sha256_value(row),
            }
            if quarantine_by_id.get(base_fact_id) != expected_quarantine:
                raise ValueError(f"parent zh quarantine row is stale: {base_fact_id}")
    if set(quarantine_by_id) != set(expected_quarantine_ids):
        raise ValueError("parent zh quarantine coverage is stale")

    accepted_count = len(target_ids) - len(expected_quarantine_ids)
    repair_accepted_count = sum(
        row["parsed_response"].get("decision") == "accept"
        for row in repair_reviews.values()
    )
    expected_status = (
        "completed_accepted_candidate_projection_zh_proxy_review"
        if not expected_quarantine_ids
        else "completed_with_quarantine_or_limited_scope"
    )
    counts = manifest.get("counts")
    if manifest.get("status") != expected_status or not isinstance(counts, dict) or counts != {
        "universe_records": int(manifest["universe_record_count"]),
        "selected_records": len(target_ids),
        "proxy_accepted_records": accepted_count,
        "quarantined_records": len(expected_quarantine_ids),
        "initial_review_accepted_records": len(target_ids) - len(rejected_initial),
        "repair_attempted_records": len(repairs),
        "repair_review_accepted_records": repair_accepted_count,
    }:
        raise ValueError("parent zh terminal status or counts are stale")
    claims = manifest.get("completion_claims")
    if not isinstance(claims, dict) or not all(
        (
            claims.get("selected_scope_processing_complete") is True,
            claims.get("full_pool_zh_translation_review_complete") is False,
            claims.get("accepted_candidate_projection_zh_translation_review_complete")
            is (not expected_quarantine_ids),
            claims.get("zh_proxy_review_not_human_gold") is True,
            claims.get("other_15_target_languages_complete") is False,
            claims.get("hf_or_behavior_work_performed") is False,
        )
    ):
        raise ValueError("parent zh completion claims are stale")

    projection_outputs = projection["output_bindings"]
    structural_bindings: Dict[str, Dict[str, Any]] = {}
    for label in ("leakage_components", "split_manifest"):
        binding = dict(projection_outputs[label])
        binding["path"] = str(projection["output_paths"][label])
        structural_bindings[label] = binding
    facts_by_id = _unique_index(projection["full_base_facts"], "base_fact_id", "facts")
    distractors_by_fact = _unique_index(
        projection["distractor_candidates"], "base_fact_id", "distractors"
    )
    neutrals_by_fact = _unique_index(
        projection["neutral_reference_candidates"], "base_fact_id", "neutrals"
    )
    return {
        "projection": projection,
        "projection_manifest_path": projection_path,
        "projection_manifest_binding": _binding(
            projection_path,
            schema_version=accepted_projection_tool.MANIFEST_SCHEMA_VERSION,
        ),
        "zh_manifest": manifest,
        "zh_manifest_path": zh_manifest_path,
        "zh_manifest_binding": _binding(
            zh_manifest_path, schema_version=zh_tool.MANIFEST_SCHEMA
        ),
        "target_ids": target_ids,
        "items_by_id": items_by_id,
        "facts_by_id": facts_by_id,
        "distractors_by_fact": distractors_by_fact,
        "neutrals_by_fact": neutrals_by_fact,
        "final_rows": final_rows,
        "final_by_id": final_by_id,
        "final_binding": artifacts["zh_translation_review_records"]["binding"],
        "quarantine_rows": quarantine_rows,
        "quarantine_by_id": quarantine_by_id,
        "quarantine_ids": expected_quarantine_ids,
        "quarantine_binding": artifacts["translation_quarantine"]["binding"],
        "structural_bindings": structural_bindings,
    }


def _build_plan(parent: Mapping[str, Any]) -> Dict[str, Any]:
    target_ids = list(parent["target_ids"])
    support_ids = set(target_ids)
    retained = set(target_ids)
    excluded: Dict[str, Dict[str, Any]] = {}
    for base_fact_id in parent["quarantine_ids"]:
        final = parent["final_by_id"][base_fact_id]
        quarantine = parent["quarantine_by_id"][base_fact_id]
        excluded[base_fact_id] = {
            "schema_version": EXCLUSION_SCHEMA_VERSION,
            "base_fact_id": base_fact_id,
            "disposition": "exclude_from_zh_accepted_secondary_projection",
            "exclusion_stage": "parent_zh_quarantine_seed",
            "closure_round": 0,
            "reason_codes": list(quarantine["reasons"]),
            "excluded_selected_candidate_kinds": [],
            "excluded_selected_donor_base_fact_ids": [],
            "parent_zh_final_record_sha256": sha256_value(final),
            "parent_zh_quarantine_record_sha256": sha256_value(quarantine),
            "human_gold": False,
        }
    retained.difference_update(excluded)
    if not retained:
        raise ValueError("zh-accepted secondary projection retained no target facts")

    projected_ids = [base_fact_id for base_fact_id in target_ids if base_fact_id in retained]
    exclusion_rows = [excluded[base_fact_id] for base_fact_id in target_ids if base_fact_id in excluded]
    items: List[Dict[str, Any]] = []
    selected_candidate_ids: List[str] = []
    support_only_donor_ids: set[str] = set()
    for base_fact_id in projected_ids:
        fact = parent["facts_by_id"][base_fact_id]
        distractor = parent["distractors_by_fact"][base_fact_id]
        neutral = parent["neutrals_by_fact"][base_fact_id]
        donor_ids = [
            str(distractor["source_base_fact_id"]),
            str(neutral["source_base_fact_id"]),
        ]
        if any(donor_id not in support_ids for donor_id in donor_ids):
            raise ValueError("selected candidate donor is outside the parent support cohort")
        support_only_donor_ids.update(
            donor_id for donor_id in donor_ids if donor_id not in retained
        )
        distractor_id = str(distractor["distractor_id"])
        neutral_id = str(neutral["neutral_candidate_id"])
        selected_candidate_ids.extend((distractor_id, neutral_id))
        items.append(
            {
                "schema_version": ITEM_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "split_assignment": fact["split_assignment"],
                "leakage_component_id": fact["leakage_component_id"],
                "parent_base_fact_row_sha256": sha256_value(fact),
                "selected_distractor_id": distractor_id,
                "selected_distractor_source_base_fact_id": donor_ids[0],
                "selected_distractor_row_sha256": sha256_value(distractor),
                "selected_neutral_candidate_id": neutral_id,
                "selected_neutral_source_base_fact_id": donor_ids[1],
                "selected_neutral_candidate_row_sha256": sha256_value(neutral),
                "parent_zh_final_record_sha256": sha256_value(
                    parent["final_by_id"][base_fact_id]
                ),
                "human_gold": False,
            }
        )
    filtered_rows = [copy.deepcopy(parent["final_by_id"][base_fact_id]) for base_fact_id in projected_ids]
    if any(
        row.get("terminal_status") != "completed"
        or row.get("quarantine_reasons") != []
        or row.get("formal_claims", {}).get("translation_proxy_accepted") is not True
        for row in filtered_rows
    ):
        raise ValueError("projected zh evidence contains a non-accepted row")
    return {
        "projected_ids": projected_ids,
        "excluded_ids": [str(row["base_fact_id"]) for row in exclusion_rows],
        "seed_ids": list(parent["quarantine_ids"]),
        "items": items,
        "exclusions": exclusion_rows,
        "filtered_rows": filtered_rows,
        "selected_candidate_ids": selected_candidate_ids,
        "support_only_donor_ids": [
            base_fact_id
            for base_fact_id in target_ids
            if base_fact_id in support_only_donor_ids
        ],
        "closure_round_count": 0,
    }


def _split_counts(
    projected_ids: Sequence[str], facts_by_id: Mapping[str, Mapping[str, Any]]
) -> Dict[str, int]:
    counts = Counter(str(facts_by_id[base_fact_id]["split_assignment"]) for base_fact_id in projected_ids)
    return {name: counts.get(name, 0) for name in ("development", "validation", "sealed")}


def _review_manifest_payload(
    *,
    parent: Mapping[str, Any],
    plan: Mapping[str, Any],
    projection_id: str,
    filtered_records_binding: Mapping[str, Any],
    empty_quarantine_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": REVIEW_STATUS,
        "projection_id": projection_id,
        "parent_accepted_candidate_projection_manifest": parent[
            "projection_manifest_binding"
        ],
        "parent_zh_review_manifest": parent["zh_manifest_binding"],
        "parent_zh_translation_review_records": parent["final_binding"],
        "parent_zh_translation_quarantine": parent["quarantine_binding"],
        "selection_contract": {
            "source_scope_complete": True,
            "accepted_rows_only": True,
            "parent_quarantine_seeded": True,
            "parent_candidate_projection_is_support_cohort": True,
            "selected_candidate_donor_must_be_in_projected_target_cohort": False,
            "selected_candidate_donor_must_be_in_parent_support_cohort": True,
            "support_only_donors_are_not_behavior_or_vector_targets": True,
            "candidate_reselection_performed": False,
            "model_calls_performed": False,
            "component_recomputed": False,
            "split_recomputed": False,
            "human_gold": False,
        },
        "counts": {
            "parent_selected_records": len(parent["target_ids"]),
            "parent_proxy_accepted_records": len(parent["target_ids"])
            - len(parent["quarantine_ids"]),
            "parent_quarantined_records": len(parent["quarantine_ids"]),
            "projected_accepted_records": len(plan["projected_ids"]),
            "support_only_donor_records": len(plan["support_only_donor_ids"]),
            "dependency_closure_excluded_records": 0,
            "projected_quarantined_records": 0,
        },
        "selected_base_fact_ids_sha256": sha256_value(plan["projected_ids"]),
        "excluded_base_fact_ids_sha256": sha256_value(plan["excluded_ids"]),
        "selected_split_counts": _split_counts(
            plan["projected_ids"], parent["facts_by_id"]
        ),
        "output_artifacts": {
            "zh_translation_review_records": dict(filtered_records_binding),
            "translation_quarantine": dict(empty_quarantine_binding),
        },
        "evidence_boundary": {
            "human_gold": False,
            "offline_projection_only": True,
            "hf_model_execution_count": 0,
            "hf_tokenizer_execution_count": 0,
            "behavior_execution_count": 0,
            "hidden_state_collection_count": 0,
            "intervention_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
        },
    }


def _manifest_payload(
    *,
    parent: Mapping[str, Any],
    plan: Mapping[str, Any],
    projection_id: str,
    items_binding: Mapping[str, Any],
    exclusions_binding: Mapping[str, Any],
    review_manifest_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": STATUS,
        "projection_id": projection_id,
        "inputs": {
            "parent_accepted_candidate_projection_manifest": parent[
                "projection_manifest_binding"
            ],
            "parent_zh_review_manifest": parent["zh_manifest_binding"],
            "parent_zh_translation_review_records": parent["final_binding"],
            "parent_zh_translation_quarantine": parent["quarantine_binding"],
        },
        "selection_contract": {
            "policy_version": "zh-quarantine-filter-parent-support-v1",
            "quarantine_ids_seed_the_projection": True,
            "only_parent_zh_quarantine_targets_are_excluded": True,
            "parent_candidate_projection_is_support_cohort": True,
            "parent_support_cohort_record_count": len(parent["target_ids"]),
            "selected_candidate_donor_must_remain_in_projected_target_cohort": False,
            "selected_candidate_donor_must_remain_in_parent_support_cohort": True,
            "support_only_donors_are_not_behavior_or_vector_targets": True,
            "candidate_reselection_performed": False,
            "model_calls_performed": False,
            "component_recomputed": False,
            "split_recomputed": False,
            "structural_snapshots_modified": False,
            "human_gold": False,
        },
        "structural_snapshots": {
            **copy.deepcopy(parent["structural_bindings"]),
            "reused_by_reference_without_recompute_or_modification": True,
        },
        "counts": {
            "parent_candidate_projection_targets": len(parent["target_ids"]),
            "parent_zh_quarantine_seeds": len(plan["seed_ids"]),
            "dependency_closure_exclusions": 0,
            "support_only_donors": len(plan["support_only_donor_ids"]),
            "total_secondary_exclusions": len(plan["excluded_ids"]),
            "projected_targets": len(plan["projected_ids"]),
            "selected_distractor_candidates": len(plan["projected_ids"]),
            "selected_neutral_candidates": len(plan["projected_ids"]),
            "projected_zh_accepted_records": len(plan["filtered_rows"]),
            "projected_zh_quarantine_records": 0,
            "closure_rounds": plan["closure_round_count"],
        },
        "split_counts": _split_counts(plan["projected_ids"], parent["facts_by_id"]),
        "lineage_digests": {
            "ordered_parent_target_ids_sha256": sha256_value(parent["target_ids"]),
            "ordered_seed_ids_sha256": sha256_value(plan["seed_ids"]),
            "ordered_excluded_ids_sha256": sha256_value(plan["excluded_ids"]),
            "ordered_projected_target_ids_sha256": sha256_value(
                plan["projected_ids"]
            ),
            "ordered_selected_candidate_ids_sha256": sha256_value(
                plan["selected_candidate_ids"]
            ),
            "ordered_support_only_donor_ids_sha256": sha256_value(
                plan["support_only_donor_ids"]
            ),
            "ordered_projection_item_hashes_sha256": sha256_value(
                [sha256_value(row) for row in plan["items"]]
            ),
            "ordered_filtered_zh_record_hashes_sha256": sha256_value(
                [sha256_value(row) for row in plan["filtered_rows"]]
            ),
        },
        "outputs": {
            "items": dict(items_binding),
            "exclusions": dict(exclusions_binding),
            "filtered_zh_review_manifest": dict(review_manifest_binding),
        },
        "safety_contract": {
            "output_is_pre_freeze_projection": True,
            "review_freeze_emitted": False,
            "split_freeze_emitted": False,
            "hf_checkpoint_bound": False,
            "hf_tokenizer_bound": False,
            "hf_model_executed": False,
            "hf_tokenizer_executed": False,
            "behavior_executed": False,
            "validation_exposed": False,
            "sealed_exposed": False,
            "path_not_token_authorized": False,
            "support_only_donors_are_execution_targets": False,
            "human_gold": False,
        },
    }


def _projection_id(parent: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    return "zh_accepted_projection_" + sha256_value(
        {
            "tool_version": TOOL_VERSION,
            "parent_candidate_projection_sha256": parent[
                "projection_manifest_binding"
            ]["sha256"],
            "parent_zh_review_sha256": parent["zh_manifest_binding"]["sha256"],
            "seed_ids": plan["seed_ids"],
            "excluded_ids": plan["excluded_ids"],
            "projected_ids": plan["projected_ids"],
            "selected_candidate_ids": plan["selected_candidate_ids"],
        }
    )[:24]


def materialize(
    *,
    accepted_candidate_projection_manifest_path: Path,
    zh_review_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    parent = _load_parent_context(
        accepted_candidate_projection_manifest_path=Path(
            accepted_candidate_projection_manifest_path
        ),
        zh_review_manifest_path=Path(zh_review_manifest_path),
    )
    plan = _build_plan(parent)
    projection_id = _projection_id(parent, plan)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".tmp", dir=output_dir.parent)
    )
    published = False
    try:
        items_path = staged / ITEMS_NAME
        exclusions_path = staged / EXCLUSIONS_NAME
        filtered_path = staged / REVIEW_RECORDS_NAME
        empty_quarantine_path = staged / EMPTY_QUARANTINE_NAME
        review_manifest_path = staged / REVIEW_MANIFEST_NAME
        manifest_path = staged / MANIFEST_NAME
        _write_jsonl(items_path, plan["items"])
        _write_jsonl(exclusions_path, plan["exclusions"])
        _write_jsonl(filtered_path, plan["filtered_rows"])
        _write_jsonl(empty_quarantine_path, [])
        items_binding = _binding(
            items_path,
            rows=plan["items"],
            output_path=output_dir / ITEMS_NAME,
        )
        exclusions_binding = _binding(
            exclusions_path,
            rows=plan["exclusions"],
            output_path=output_dir / EXCLUSIONS_NAME,
        )
        filtered_binding = _binding(
            filtered_path,
            rows=plan["filtered_rows"],
            output_path=output_dir / REVIEW_RECORDS_NAME,
        )
        empty_quarantine_binding = _binding(
            empty_quarantine_path,
            rows=[],
            output_path=output_dir / EMPTY_QUARANTINE_NAME,
        )
        review_manifest = _review_manifest_payload(
            parent=parent,
            plan=plan,
            projection_id=projection_id,
            filtered_records_binding=filtered_binding,
            empty_quarantine_binding=empty_quarantine_binding,
        )
        _write_json(review_manifest_path, review_manifest)
        review_manifest_binding = _binding(
            review_manifest_path,
            schema_version=REVIEW_MANIFEST_SCHEMA_VERSION,
            output_path=output_dir / REVIEW_MANIFEST_NAME,
        )
        manifest = _manifest_payload(
            parent=parent,
            plan=plan,
            projection_id=projection_id,
            items_binding=items_binding,
            exclusions_binding=exclusions_binding,
            review_manifest_binding=review_manifest_binding,
        )
        _write_json(manifest_path, manifest)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        os.rename(staged, output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged, ignore_errors=True)
    return {
        "status": STATUS,
        "projection_id": projection_id,
        "manifest_path": str(output_dir / MANIFEST_NAME),
        "parent_target_count": len(parent["target_ids"]),
        "parent_zh_quarantine_count": len(plan["seed_ids"]),
        "dependency_closure_exclusion_count": len(plan["excluded_ids"])
        - len(plan["seed_ids"]),
        "support_only_donor_count": len(plan["support_only_donor_ids"]),
        "projected_target_count": len(plan["projected_ids"]),
        "model_call_count": 0,
        "component_recomputed": False,
        "split_recomputed": False,
        "human_gold": False,
    }


def load_projection_context(
    projection_manifest_path: Path,
    *,
    expected_accepted_candidate_projection_manifest_path: Optional[Path] = None,
    expected_zh_review_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    projection_manifest_path = Path(projection_manifest_path).resolve()
    manifest = read_json(projection_manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("zh-accepted projection manifest schema_version is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION or manifest.get("status") != STATUS:
        raise ValueError("zh-accepted projection manifest identity or status is invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("zh-accepted projection inputs are missing")
    parent_projection_path = _resolve_binding_path(
        inputs.get("parent_accepted_candidate_projection_manifest", {}),
        projection_manifest_path,
        "parent accepted candidate projection",
    )
    parent_zh_path = _resolve_binding_path(
        inputs.get("parent_zh_review_manifest", {}),
        projection_manifest_path,
        "parent zh review manifest",
    )
    if expected_accepted_candidate_projection_manifest_path is not None and parent_projection_path != Path(
        expected_accepted_candidate_projection_manifest_path
    ).resolve():
        raise ValueError("zh-accepted projection binds a different candidate projection")
    if expected_zh_review_manifest_path is not None and parent_zh_path != Path(
        expected_zh_review_manifest_path
    ).resolve():
        raise ValueError("zh-accepted projection binds a different zh review")
    parent = _load_parent_context(
        accepted_candidate_projection_manifest_path=parent_projection_path,
        zh_review_manifest_path=parent_zh_path,
    )
    _same_binding(
        inputs.get("parent_accepted_candidate_projection_manifest"),
        parent["projection_manifest_binding"],
        owner_path=projection_manifest_path,
        label="parent accepted candidate projection",
    )
    _same_binding(
        inputs.get("parent_zh_review_manifest"),
        parent["zh_manifest_binding"],
        owner_path=projection_manifest_path,
        label="parent zh review",
    )
    _same_binding(
        inputs.get("parent_zh_translation_review_records"),
        parent["final_binding"],
        owner_path=projection_manifest_path,
        label="parent zh final records",
    )
    _same_binding(
        inputs.get("parent_zh_translation_quarantine"),
        parent["quarantine_binding"],
        owner_path=projection_manifest_path,
        label="parent zh quarantine",
    )
    plan = _build_plan(parent)
    projection_id = _projection_id(parent, plan)
    if manifest.get("projection_id") != projection_id:
        raise ValueError("zh-accepted projection_id is stale")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("zh-accepted projection outputs are missing")
    items_path, items, items_binding = _verify_jsonl_binding(
        outputs.get("items"),
        owner_path=projection_manifest_path,
        label="zh-accepted projection items",
        expected_schema=ITEM_SCHEMA_VERSION,
    )
    exclusions_path, exclusions, exclusions_binding = _verify_jsonl_binding(
        outputs.get("exclusions"),
        owner_path=projection_manifest_path,
        label="zh-accepted projection exclusions",
        expected_schema=EXCLUSION_SCHEMA_VERSION,
        allow_empty=True,
    )
    review_binding = outputs.get("filtered_zh_review_manifest")
    if not isinstance(review_binding, dict):
        raise ValueError("filtered zh review manifest binding is missing")
    review_path = _resolve_binding_path(
        review_binding, projection_manifest_path, "filtered zh review manifest"
    )
    if sha256_file(review_path) != review_binding.get("sha256"):
        raise ValueError("filtered zh review manifest SHA-256 binding is stale")
    if (
        review_path.stat().st_size != review_binding.get("byte_count")
        or review_binding.get("schema_version") != REVIEW_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("filtered zh review manifest binding is stale")
    review_manifest = read_json(review_path)
    if review_manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA_VERSION:
        raise ValueError("filtered zh review manifest schema_version is unsupported")
    review_outputs = review_manifest.get("output_artifacts")
    if not isinstance(review_outputs, dict):
        raise ValueError("filtered zh review outputs are missing")
    filtered_path, filtered_rows, filtered_binding = _verify_jsonl_binding(
        review_outputs.get("zh_translation_review_records"),
        owner_path=review_path,
        label="filtered zh accepted records",
        expected_schema=zh_tool.FINAL_SCHEMA,
    )
    empty_quarantine_path, empty_quarantine, empty_quarantine_binding = _verify_jsonl_binding(
        review_outputs.get("translation_quarantine"),
        owner_path=review_path,
        label="filtered zh quarantine",
        expected_schema=zh_tool.QUARANTINE_SCHEMA,
        allow_empty=True,
    )
    if empty_quarantine:
        raise ValueError("filtered zh accepted projection quarantine must be empty")
    if items != plan["items"] or exclusions != plan["exclusions"] or filtered_rows != plan[
        "filtered_rows"
    ]:
        raise ValueError("zh-accepted projection outputs differ from deterministic replay")
    expected_review = _review_manifest_payload(
        parent=parent,
        plan=plan,
        projection_id=projection_id,
        filtered_records_binding=filtered_binding,
        empty_quarantine_binding=empty_quarantine_binding,
    )
    if review_manifest != expected_review:
        raise ValueError("filtered zh review manifest differs from deterministic replay")
    expected_manifest = _manifest_payload(
        parent=parent,
        plan=plan,
        projection_id=projection_id,
        items_binding=items_binding,
        exclusions_binding=exclusions_binding,
        review_manifest_binding=dict(review_binding),
    )
    if manifest != expected_manifest:
        raise ValueError("zh-accepted projection manifest differs from deterministic replay")
    return {
        "manifest_path": projection_manifest_path,
        "manifest": manifest,
        "manifest_binding": _binding(
            projection_manifest_path, schema_version=MANIFEST_SCHEMA_VERSION
        ),
        "parent": parent,
        "projected_target_ids": plan["projected_ids"],
        "excluded_ids": plan["excluded_ids"],
        "seed_ids": plan["seed_ids"],
        "support_only_donor_ids": plan["support_only_donor_ids"],
        "items": items,
        "items_path": items_path,
        "items_binding": items_binding,
        "exclusions": exclusions,
        "exclusions_path": exclusions_path,
        "exclusions_binding": exclusions_binding,
        "filtered_review_manifest": review_manifest,
        "filtered_review_manifest_path": review_path,
        "filtered_review_manifest_binding": dict(review_binding),
        "filtered_rows": filtered_rows,
        "filtered_rows_path": filtered_path,
        "filtered_rows_binding": filtered_binding,
        "empty_quarantine_path": empty_quarantine_path,
        "empty_quarantine_binding": empty_quarantine_binding,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-candidate-projection-manifest", type=Path, required=True
    )
    parser.add_argument("--zh-review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize(
        accepted_candidate_projection_manifest_path=(
            args.accepted_candidate_projection_manifest
        ),
        zh_review_manifest_path=args.zh_review_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
