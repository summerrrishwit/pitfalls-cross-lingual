#!/usr/bin/env python3
"""Translate the relation-usable part of the 7,663-target projection into five languages.

The runner intentionally performs one semantic generation stage and one independent
proxy-review stage, with no automatic repair loop. Technical retries or batch
splits may repeat transport requests and are counted separately. It is SHA-bound, resumable, and
translation-only: it does not run HF/tokenizer/behavior/hidden-state/intervention
work or change relation, component, candidate, or split assignments.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls import perturbation as runtime  # noqa: E402
from factual_pitfalls.route_identity import (  # noqa: E402
    attach_route_identity_guard,
    build_route_identity_set,
)


def _load_zh_tool() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_full_public_benchmark_zh_review.py"
    spec = importlib.util.spec_from_file_location("_full_zh_review_reuse", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared translation runner helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


zh_tool = _load_zh_tool()

TOOL_VERSION = "full-public-benchmark-multilingual-review-runner-v3"
MANIFEST_SCHEMA = "public-benchmark-full-multilingual-review-run-manifest-v1"
GENERATION_SCHEMA = "public-benchmark-multilingual-translation-generation-v1"
REVIEW_SCHEMA = "public-benchmark-multilingual-translation-independent-review-v1"
FINAL_SCHEMA = "public-benchmark-multilingual-translation-reviewed-record-v1"
QUARANTINE_SCHEMA = "public-benchmark-multilingual-translation-quarantine-v1"
PROMPT_VERSION = "public-benchmark-full-en-multilingual-translation-v1"
REVIEW_PROMPT_VERSION = "public-benchmark-full-en-multilingual-equivalence-review-v1"

PROJECTION_MANIFEST_SCHEMA = (
    "public-benchmark-full-zh-accepted-secondary-projection-manifest-v1"
)
PROJECTION_STATUS = "completed_offline_zh_accepted_secondary_projection"
PROJECTION_ITEM_SCHEMA = "public-benchmark-full-zh-accepted-secondary-projection-item-v1"
SOURCE_REVIEW_MANIFEST_SCHEMA = (
    "public-benchmark-full-zh-accepted-secondary-review-manifest-v1"
)
SOURCE_REVIEW_STATUS = "completed_offline_zh_accepted_review_projection"
SOURCE_RECORD_SCHEMA = "public-benchmark-zh-translation-reviewed-record-v1"
CANDIDATE_PROJECTION_MANIFEST_SCHEMA = (
    "public-benchmark-full-accepted-candidate-projection-manifest-v1"
)
CANDIDATE_PROJECTION_STATUS = (
    "accepted_only_min1_per_kind_materialized_provisional_offline"
)
BASE_FACT_SCHEMA = "public-benchmark-full-provisional-base-fact-v1"
ELIGIBLE_RELATION_PARTITION_STATUSES = frozenset(
    {
        "existing_probe_relation_candidate_partition",
        "broad_rule_candidate_partition",
    }
)
UNRESOLVED_RELATION_PARTITION_STATUS = "unresolved_partition_only"

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "sw": "Swahili",
}
DEFAULT_LANGUAGES = tuple(LANGUAGE_NAMES)
TRANSLATION_FIELDS = {
    "prompt",
    "canonical_fact",
    "answer",
    "answer_aliases",
    "distractors",
    "neutral_candidates",
}
REVIEW_FIELDS = {
    "decision",
    "translation_equivalent",
    "natural",
    "answer_not_leaked",
    "candidate_identity_preserved",
    "issues",
}

canonical_json = zh_tool.canonical_json
sha256_value = zh_tool.sha256_value
sha256_file = zh_tool.sha256_file
read_json = zh_tool.read_json
read_jsonl = zh_tool.read_jsonl
read_jsonl_journal = zh_tool.read_jsonl_journal
write_json = zh_tool.write_json
write_jsonl = zh_tool.write_jsonl
append_jsonl_journal = zh_tool.append_jsonl_journal
artifact_binding = zh_tool.artifact_binding
load_config = zh_tool.load_config
utc_now = zh_tool.utc_now
BatchRequestFailure = zh_tool.BatchRequestFailure


class PipelineStopped(RuntimeError):
    """A resumable operational stop, never a semantic rejection."""

    def __init__(self, summary: Mapping[str, Any]):
        super().__init__(str(summary["reason"]))
        self.summary = dict(summary)


def service_stop_reason(error: Exception) -> Optional[str]:
    metadata = error.metadata if isinstance(error, BatchRequestFailure) else {}
    attempts = metadata.get("attempts", [])
    for entry in [runtime.safe_error(error), *attempts]:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("provider_code") in {
            "insufficient_user_quota", "insufficient_quota", "quota_exceeded",
            "billing_hard_limit_reached", "credit_balance_too_low",
        }:
            return "provider_quota_exhausted"
        if entry.get("http_status") in {401, 402, 403} or entry.get("error_type") in {
            "AuthenticationError", "PermissionDeniedError",
        }:
            return "provider_authentication_or_permission_failed"
        if entry.get("error_type") in {
            "StaticProxyIdentityError", "ResponseModelMissing", "ResponseModelMismatch",
            "RouteIdentityError",
        }:
            return "model_or_route_identity_failed"
    return None


def recovery_source(
    source_path: Path, contract: Mapping[str, Any], output_dir: Path, *, resume: bool,
) -> Dict[str, Any]:
    """Import byte-identical checkpoints only across an execution-only upgrade."""
    source_path = source_path.resolve()
    if source_path.parent == output_dir:
        raise ValueError("recovery must use a new output directory")
    parent = read_json(source_path)
    if parent.get("status") not in {
        "completed_with_quarantine_or_limited_scope", "completed_full_multilingual_proxy_review",
        "interrupted_checkpoint_snapshot",
    }:
        raise ValueError("recovery source must be finalized or an immutable interrupted snapshot")
    # Execution-only upgrades preserve prompt/model/input identities and every
    # inherited checkpoint fingerprint. They do not authorize content changes.
    if parent.get("tool_version") not in {
        "full-public-benchmark-multilingual-review-runner-v1",
        "full-public-benchmark-multilingual-review-runner-v2",
        "full-public-benchmark-multilingual-review-runner-v3",
    }:
        raise ValueError("unsupported recovery source version")
    mutable = {"tool_version", "source_fingerprints", "execution_policy", "selected_targets"}
    for key, value in contract.items():
        if key not in mutable and parent.get(key) != value:
            raise ValueError(f"recovery semantic contract mismatch: {key}")
    for key, value in contract["source_fingerprints"].items():
        if key != "runner_sha256" and parent["source_fingerprints"].get(key) != value:
            raise ValueError(f"recovery shared runtime mismatch: {key}")
    old_selection = parent["selected_targets"]
    if old_selection["sha256"] != contract["selected_targets"]["sha256"]:
        raise ValueError("recovery target inventory mismatch")
    fingerprints = {parent["run_fingerprint"]}
    lineage = parent.get("recovery_lineage")
    if lineage:
        ancestor = lineage["parent_manifest"]
        ancestor_path = Path(ancestor["path"])
        if sha256_file(ancestor_path) != ancestor["sha256"]:
            raise ValueError("recovery ancestor manifest hash mismatch")
        fingerprints.add(lineage["parent_run_fingerprint"])
        fingerprints.update(lineage.get("inherited_run_fingerprints", []))
    bindings = {}
    copies = []
    for key in ("translation_initial", "translation_reviews"):
        binding = parent["output_artifacts"][key]
        path = Path(binding["path"])
        if not path.is_absolute():
            path = source_path.parent / path
        if path.with_suffix(path.suffix + ".journal").exists():
            raise ValueError("recovery source has an uncompacted journal")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"recovery checkpoint hash mismatch: {key}")
        if any(row.get("run_fingerprint") not in fingerprints for row in read_jsonl(path)):
            raise ValueError(f"recovery checkpoint has an unbound ancestor fingerprint: {key}")
        bindings[key] = artifact_binding(path)
        destination = output_dir / path.name
        if not resume:
            if destination.exists():
                raise FileExistsError(f"recovery destination already exists: {destination}")
            copies.append((path, destination))
    for path, destination in copies:
        shutil.copyfile(path, destination)
    return {
        "policy": "byte-identical-checkpoints-execution-only-upgrade-v2",
        "parent_manifest": artifact_binding(source_path),
        "parent_run_fingerprint": parent["run_fingerprint"],
        "inherited_run_fingerprints": sorted(fingerprints),
        "checkpoint_bindings": bindings,
        "successful_generation_reused_without_regeneration": True,
        "semantic_rejections_reused_without_repair": True,
        "prior_exposure": parent.get("translation_model_exposure", {}),
    }


def _source_fingerprints() -> Dict[str, str]:
    return {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "shared_zh_runner_sha256": sha256_file(
            PROJECT_ROOT / "scripts" / "run_full_public_benchmark_zh_review.py"
        ),
        "model_router_runtime_sha256": sha256_file(Path(runtime.__file__).resolve()),
        "route_identity_helper_sha256": sha256_file(
            PROJECT_ROOT / "factual_pitfalls" / "route_identity.py"
        ),
    }


def _require_unique(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> None:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{label} contains a missing {key}")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate {key}: {duplicates[0]}")


def _resolved_binding_path(
    binding: Mapping[str, Any], *, owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} binding has no path")
    path = Path(raw)
    if not path.is_absolute():
        path = owner_path.parent / path
    return path.resolve()


def _verify_binding(
    binding: Any,
    *,
    owner_path: Path,
    supplied_path: Path,
    rows: Optional[Sequence[Mapping[str, Any]]],
    label: str,
) -> Dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    supplied = supplied_path.resolve()
    if _resolved_binding_path(binding, owner_path=owner_path, label=label) != supplied:
        raise ValueError(f"{label} path differs from supplied input")
    actual = artifact_binding(supplied, rows=rows)
    for key in ("sha256", "byte_count", "record_count"):
        if binding.get(key) is not None and binding.get(key) != actual.get(key):
            raise ValueError(f"{label} {key} mismatch")
    return actual


def _validate_source_fields(value: Any, base_fact_id: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "prompt_en",
        "canonical_fact_en",
        "answer_en",
        "answer_aliases_en",
    }:
        raise ValueError(f"invalid English source fields: {base_fact_id}")
    for field in ("prompt_en", "canonical_fact_en", "answer_en"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"empty English source field {field}: {base_fact_id}")
    aliases = value["answer_aliases_en"]
    if not isinstance(aliases, list) or any(
        not isinstance(alias, str) or not alias.strip() for alias in aliases
    ):
        raise ValueError(f"invalid English answer aliases: {base_fact_id}")
    return copy.deepcopy(value)


def _validate_source_candidates(
    value: Any, *, id_key: str, label: str, base_fact_id: str
) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not value or any(
        not isinstance(row, dict) for row in value
    ):
        raise ValueError(f"invalid {label}: {base_fact_id}")
    expected_fields = {id_key, "slot", "text_en"}
    result: List[Dict[str, Any]] = []
    for row in value:
        if set(row) != expected_fields:
            raise ValueError(f"invalid {label} fields: {base_fact_id}")
        if not isinstance(row[id_key], str) or not row[id_key]:
            raise ValueError(f"invalid {label} id: {base_fact_id}")
        if not isinstance(row["slot"], int) or row["slot"] < 1:
            raise ValueError(f"invalid {label} slot: {base_fact_id}")
        if not isinstance(row["text_en"], str) or not row["text_en"].strip():
            raise ValueError(f"empty {label} text: {base_fact_id}")
        result.append(copy.deepcopy(row))
    return result


def load_bound_inputs(
    *,
    candidate_projection_manifest_path: Path,
    candidate_base_facts_path: Path,
    projection_manifest_path: Path,
    projection_items_path: Path,
    source_review_manifest_path: Path,
    source_records_path: Path,
) -> Dict[str, Any]:
    candidate_projection_manifest_path = candidate_projection_manifest_path.resolve()
    candidate_base_facts_path = candidate_base_facts_path.resolve()
    projection_manifest_path = projection_manifest_path.resolve()
    projection_items_path = projection_items_path.resolve()
    source_review_manifest_path = source_review_manifest_path.resolve()
    source_records_path = source_records_path.resolve()
    candidate_projection_manifest = read_json(candidate_projection_manifest_path)
    candidate_base_rows = read_jsonl(candidate_base_facts_path)
    projection_manifest = read_json(projection_manifest_path)
    source_review_manifest = read_json(source_review_manifest_path)
    projection_rows = read_jsonl(projection_items_path)
    source_rows = read_jsonl(source_records_path)
    if (
        candidate_projection_manifest.get("schema_version")
        != CANDIDATE_PROJECTION_MANIFEST_SCHEMA
        or candidate_projection_manifest.get("status")
        != CANDIDATE_PROJECTION_STATUS
    ):
        raise ValueError("unsupported or incomplete accepted-candidate projection")
    if (
        projection_manifest.get("schema_version") != PROJECTION_MANIFEST_SCHEMA
        or projection_manifest.get("status") != PROJECTION_STATUS
    ):
        raise ValueError("unsupported or incomplete zh-accepted projection manifest")
    if (
        source_review_manifest.get("schema_version") != SOURCE_REVIEW_MANIFEST_SCHEMA
        or source_review_manifest.get("status") != SOURCE_REVIEW_STATUS
    ):
        raise ValueError("unsupported or incomplete zh-accepted review manifest")
    candidate_base_binding = _verify_binding(
        candidate_projection_manifest.get("outputs", {}).get("full_base_facts"),
        owner_path=candidate_projection_manifest_path,
        supplied_path=candidate_base_facts_path,
        rows=candidate_base_rows,
        label="accepted-projection base facts",
    )
    projection_binding = _verify_binding(
        projection_manifest.get("outputs", {}).get("items"),
        owner_path=projection_manifest_path,
        supplied_path=projection_items_path,
        rows=projection_rows,
        label="projection items",
    )
    source_binding = _verify_binding(
        source_review_manifest.get("output_artifacts", {}).get(
            "zh_translation_review_records"
        ),
        owner_path=source_review_manifest_path,
        supplied_path=source_records_path,
        rows=source_rows,
        label="zh-accepted source records",
    )
    _require_unique(candidate_base_rows, "base_fact_id", "accepted-projection base facts")
    _require_unique(projection_rows, "base_fact_id", "projection items")
    _require_unique(source_rows, "base_fact_id", "source records")
    projection_ids = [str(row["base_fact_id"]) for row in projection_rows]
    source_ids = [str(row["base_fact_id"]) for row in source_rows]
    if projection_ids != source_ids:
        raise ValueError("projection and source record order or identity differs")
    if len(projection_rows) != 7663:
        raise ValueError("expected the locked 7,663-target projection")
    if projection_manifest.get("counts", {}).get("projected_targets") != len(
        projection_rows
    ):
        raise ValueError("projection target count is stale")
    if source_review_manifest.get("counts", {}).get(
        "projected_accepted_records"
    ) != len(source_rows):
        raise ValueError("source accepted record count is stale")

    candidate_base_by_id = {
        str(row["base_fact_id"]): row for row in candidate_base_rows
    }
    items: List[Dict[str, Any]] = []
    full_split_counts: Counter[str] = Counter()
    selected_split_counts: Counter[str] = Counter()
    relation_status_counts: Counter[str] = Counter()
    for projection, source_row in zip(projection_rows, source_rows):
        base_fact_id = str(projection["base_fact_id"])
        candidate_base = candidate_base_by_id.get(base_fact_id)
        if candidate_base is None:
            raise ValueError(f"projection target missing from candidate base facts: {base_fact_id}")
        if candidate_base.get("schema_version") != BASE_FACT_SCHEMA:
            raise ValueError(f"invalid candidate base-fact schema: {base_fact_id}")
        if projection.get("parent_base_fact_row_sha256") != sha256_value(candidate_base):
            raise ValueError(f"candidate base-fact hash differs from projection: {base_fact_id}")
        if projection.get("schema_version") != PROJECTION_ITEM_SCHEMA:
            raise ValueError(f"invalid projection item schema: {base_fact_id}")
        if source_row.get("schema_version") != SOURCE_RECORD_SCHEMA:
            raise ValueError(f"invalid source record schema: {base_fact_id}")
        if (
            source_row.get("terminal_status") != "completed"
            or source_row.get("target_language") != "zh"
            or source_row.get("human_gold") is not False
            or source_row.get("formal_claims", {}).get(
                "translation_proxy_accepted"
            )
            is not True
        ):
            raise ValueError(f"source record is not proxy-accepted: {base_fact_id}")
        source_row_sha = sha256_value(source_row)
        if projection.get("parent_zh_final_record_sha256") != source_row_sha:
            raise ValueError(f"source row hash differs from projection: {base_fact_id}")
        if projection.get("split_assignment") != source_row.get("split_assignment"):
            raise ValueError(f"source split differs from projection: {base_fact_id}")
        source = _validate_source_fields(source_row.get("source_fields"), base_fact_id)
        distractors = _validate_source_candidates(
            source_row.get("source_distractors"),
            id_key="distractor_id",
            label="source distractors",
            base_fact_id=base_fact_id,
        )
        neutrals = _validate_source_candidates(
            source_row.get("source_neutral_candidates"),
            id_key="neutral_candidate_id",
            label="source neutral candidates",
            base_fact_id=base_fact_id,
        )
        if [row["distractor_id"] for row in distractors] != [
            projection.get("selected_distractor_id")
        ]:
            raise ValueError(f"selected distractor differs from source: {base_fact_id}")
        if [row["neutral_candidate_id"] for row in neutrals] != [
            projection.get("selected_neutral_candidate_id")
        ]:
            raise ValueError(f"selected Neutral differs from source: {base_fact_id}")
        split = str(projection.get("split_assignment") or "")
        if split not in {"development", "validation", "sealed"}:
            raise ValueError(f"invalid split assignment: {base_fact_id}")
        full_split_counts[split] += 1
        relation_status = str(candidate_base.get("relation_partition_status") or "")
        relation_status_counts[relation_status] += 1
        if relation_status == UNRESOLVED_RELATION_PARTITION_STATUS:
            continue
        if relation_status not in ELIGIBLE_RELATION_PARTITION_STATUSES:
            raise ValueError(f"unsupported relation partition status: {base_fact_id}")
        selected_split_counts[split] += 1
        input_identity = {
            "candidate_base_fact_row_sha256": sha256_value(candidate_base),
            "projection_item_sha256": sha256_value(projection),
            "source_record_sha256": source_row_sha,
        }
        items.append(
            {
                "base_fact_id": base_fact_id,
                "cohort_item_id": source_row["cohort_item_id"],
                "split_assignment": split,
                "source": source,
                "distractors": distractors,
                "neutral_candidates": neutrals,
                "projection_item_sha256": input_identity[
                    "projection_item_sha256"
                ],
                "candidate_base_fact_row_sha256": input_identity[
                    "candidate_base_fact_row_sha256"
                ],
                "source_record_sha256": source_row_sha,
                "input_record_sha256": sha256_value(input_identity),
                "relation_partition_id": candidate_base.get("relation_partition_id"),
                "relation_partition_status": relation_status,
            }
        )
    declared_split_counts = source_review_manifest.get("selected_split_counts")
    if dict(sorted(full_split_counts.items())) != dict(sorted(declared_split_counts.items())):
        raise ValueError("source split counts differ from the bound review manifest")
    if len(items) != 4513:
        raise ValueError("expected 4,513 relation-usable translation targets")
    expected_relation_counts = {
        "broad_rule_candidate_partition": 2346,
        "existing_probe_relation_candidate_partition": 2167,
        "unresolved_partition_only": 3150,
    }
    if dict(sorted(relation_status_counts.items())) != expected_relation_counts:
        raise ValueError("relation partition status counts differ from the locked projection")
    ordered_selected_ids_sha256 = sha256_value(
        [item["base_fact_id"] for item in items]
    )
    if (
        ordered_selected_ids_sha256
        != "7221d8f5468c43980d80b4c5e90636165158fe5329dbe263ef3732386f006542"
    ):
        raise ValueError("relation-usable ordered target identity differs from audit")
    return {
        "candidate_projection_manifest": candidate_projection_manifest,
        "projection_manifest": projection_manifest,
        "source_review_manifest": source_review_manifest,
        "items": items,
        "source_projection_record_count": len(projection_rows),
        "source_projection_split_counts": dict(sorted(full_split_counts.items())),
        "selected_split_counts": dict(sorted(selected_split_counts.items())),
        "relation_partition_status_counts": dict(sorted(relation_status_counts.items())),
        "ordered_relation_usable_base_fact_ids_sha256": ordered_selected_ids_sha256,
        "bindings": {
            "candidate_projection_manifest": artifact_binding(
                candidate_projection_manifest_path,
                schema_version=CANDIDATE_PROJECTION_MANIFEST_SCHEMA,
            ),
            "candidate_base_facts": candidate_base_binding,
            "projection_manifest": artifact_binding(
                projection_manifest_path,
                schema_version=PROJECTION_MANIFEST_SCHEMA,
            ),
            "projection_items": projection_binding,
            "source_review_manifest": artifact_binding(
                source_review_manifest_path,
                schema_version=SOURCE_REVIEW_MANIFEST_SCHEMA,
            ),
            "source_records": source_binding,
        },
    }


def _language_specs(codes: Sequence[str]) -> List[Dict[str, str]]:
    normalized = [str(code).strip().lower() for code in codes]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("target language codes must be non-empty and unique")
    if "zh" in normalized:
        raise ValueError("zh is lineage-only and cannot be a target in this runner")
    unknown = sorted(set(normalized) - set(LANGUAGE_NAMES))
    if unknown:
        raise ValueError(f"unsupported target language: {unknown[0]}")
    return [{"code": code, "name": LANGUAGE_NAMES[code]} for code in normalized]


def _model_specs(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    return zh_tool._model_specs(config, args)


def translation_batch_prompt(
    items: Sequence[Mapping[str, Any]], languages: Sequence[Mapping[str, str]]
) -> str:
    payload = {
        "target_languages": list(languages),
        "records": [
            {
                "base_fact_id": item["base_fact_id"],
                "source_fields": item["source"],
                "source_distractors": item["distractors"],
                "source_neutral_candidates": item["neutral_candidates"],
            }
            for item in items
        ],
    }
    return (
        "Translate every English text field into every listed target language. Preserve "
        "factual meaning, answer and alias identity, the incomplete question/Answer prompt "
        "form, distractor identity, and Neutral-context meaning. Do not fact-check, correct, "
        "strengthen, or add the answer to the prompt. Use natural target-language wording and "
        "established localized entity names where appropriate. Return JSON only with exactly "
        "one top-level key records. records must contain exactly one object per base_fact_id; "
        "each object has base_fact_id and translations. translations has exactly the requested "
        "language-code keys. Every language object has exactly prompt, canonical_fact, answer, "
        "answer_aliases, distractors, and neutral_candidates. answer_aliases preserves source "
        "length/order. distractors contain exactly distractor_id and text; neutral_candidates "
        "contain exactly neutral_candidate_id and text, both preserving source order. Input: "
        + canonical_json(payload)
    )


def _validate_translated_candidates(
    actual: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    label: str,
) -> None:
    if not isinstance(actual, list) or any(not isinstance(row, dict) for row in actual):
        raise ValueError(f"invalid_{label}")
    if any(set(row) != {id_key, "text"} for row in actual):
        raise ValueError(f"invalid_{label}_fields")
    if [row.get(id_key) for row in actual] != [row[id_key] for row in expected]:
        raise ValueError(f"{label}_id_or_order_mismatch")
    if any(not isinstance(row.get("text"), str) or not row["text"].strip() for row in actual):
        raise ValueError(f"invalid_{label}_translation")


def validate_translations(
    value: Any, item: Mapping[str, Any], language_codes: Sequence[str]
) -> None:
    if not isinstance(value, dict) or set(value) != set(language_codes):
        raise ValueError("translation_language_keys_mismatch")
    for code in language_codes:
        translation = value[code]
        if not isinstance(translation, dict) or set(translation) != TRANSLATION_FIELDS:
            raise ValueError(f"invalid_translation_fields_{code}")
        for field in ("prompt", "canonical_fact", "answer"):
            if not isinstance(translation[field], str) or not translation[field].strip():
                raise ValueError(f"invalid_{field}_{code}")
        aliases = translation["answer_aliases"]
        if not isinstance(aliases, list) or len(aliases) != len(
            item["source"]["answer_aliases_en"]
        ):
            raise ValueError(f"answer_alias_translation_count_mismatch_{code}")
        if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            raise ValueError(f"invalid_answer_aliases_{code}")
        _validate_translated_candidates(
            translation["distractors"],
            item["distractors"],
            id_key="distractor_id",
            label=f"distractors_{code}",
        )
        _validate_translated_candidates(
            translation["neutral_candidates"],
            item["neutral_candidates"],
            id_key="neutral_candidate_id",
            label=f"neutral_candidates_{code}",
        )


def validate_translation_batch(
    value: Dict[str, Any],
    items: Sequence[Mapping[str, Any]],
    language_codes: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if set(value) != {"records"} or not isinstance(value.get("records"), list):
        raise ValueError("invalid_translation_batch")
    expected = {str(item["base_fact_id"]): item for item in items}
    records = value["records"]
    if any(not isinstance(row, dict) or set(row) != {"base_fact_id", "translations"} for row in records):
        raise ValueError("invalid_translation_batch_record")
    actual_ids = [str(row.get("base_fact_id") or "") for row in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected):
        raise ValueError("translation_batch_base_fact_id_mismatch")
    result: Dict[str, Dict[str, Any]] = {}
    for row in records:
        base_fact_id = str(row["base_fact_id"])
        validate_translations(row["translations"], expected[base_fact_id], language_codes)
        result[base_fact_id] = row["translations"]
    return result


def review_batch_prompt(
    items: Sequence[Mapping[str, Any]],
    translations_by_id: Mapping[str, Mapping[str, Any]],
    languages: Sequence[Mapping[str, str]],
) -> str:
    payload = {
        "target_languages": list(languages),
        "records": [
            {
                "base_fact_id": item["base_fact_id"],
                "source_fields": item["source"],
                "source_distractors": item["distractors"],
                "source_neutral_candidates": item["neutral_candidates"],
                "translations": translations_by_id[str(item["base_fact_id"])],
            }
            for item in items
        ],
    }
    return (
        "Independently review each English-to-target-language translation at the whole-record "
        "level. Review translation equivalence and naturalness only; do not certify factual "
        "truth, distractor falsehood, or Neutral unrelatedness. Confirm that the prompt does "
        "not reveal its answer and candidate identities/order are preserved. Treat a translation "
        "as natural when it is clear and usable to a native reader; do not reject only for minor "
        "stylistic awkwardness, capitalization, or a deliberately untranslated established title. "
        "Reject material meaning, entity, answer-leakage, or usability errors. Return JSON only "
        "with exactly one top-level key records. Each record has base_fact_id and reviews; "
        "reviews has exactly the requested language-code keys. Each language review has exactly "
        "decision (accept|reject), translation_equivalent, natural, answer_not_leaked, "
        "candidate_identity_preserved (booleans), and issues (string array). decision is accept "
        "if and only if all four booleans are true. Input: "
        + canonical_json(payload)
    )


def validate_reviews(value: Any, language_codes: Sequence[str]) -> None:
    if not isinstance(value, dict) or set(value) != set(language_codes):
        raise ValueError("review_language_keys_mismatch")
    for code in language_codes:
        review = value[code]
        if not isinstance(review, dict) or set(review) != REVIEW_FIELDS:
            raise ValueError(f"invalid_review_fields_{code}")
        if review.get("decision") not in {"accept", "reject"}:
            raise ValueError(f"invalid_review_decision_{code}")
        booleans = [
            review.get("translation_equivalent"),
            review.get("natural"),
            review.get("answer_not_leaked"),
            review.get("candidate_identity_preserved"),
        ]
        if any(not isinstance(flag, bool) for flag in booleans):
            raise ValueError(f"invalid_review_boolean_{code}")
        issues = review.get("issues")
        if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
            raise ValueError(f"invalid_review_issues_{code}")
        if (review["decision"] == "accept") != all(booleans):
            raise ValueError(f"review_decision_inconsistent_{code}")


def validate_review_batch(
    value: Dict[str, Any],
    items: Sequence[Mapping[str, Any]],
    language_codes: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    if set(value) != {"records"} or not isinstance(value.get("records"), list):
        raise ValueError("invalid_review_batch")
    expected_ids = {str(item["base_fact_id"]) for item in items}
    records = value["records"]
    if any(not isinstance(row, dict) or set(row) != {"base_fact_id", "reviews"} for row in records):
        raise ValueError("invalid_review_batch_record")
    actual_ids = [str(row.get("base_fact_id") or "") for row in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("review_batch_base_fact_id_mismatch")
    result: Dict[str, Dict[str, Any]] = {}
    for row in records:
        validate_reviews(row["reviews"], language_codes)
        result[str(row["base_fact_id"])] = row["reviews"]
    return result


def _batch_metadata(
    *,
    result: runtime.ModelResult,
    spec: Mapping[str, Any],
    expected_response_model: str,
    prompt: str,
    prompt_version: str,
    item_ids: Sequence[str],
    language_codes: Sequence[str],
) -> Dict[str, Any]:
    return {
        "terminal_status": result.terminal_status,
        "request_model": spec["model"],
        "provider_profile": spec["provider_profile"],
        "expected_response_model": expected_response_model,
        "response_model": result.response_model,
        "response_model_identity_status": (
            "matched"
            if result.response_model == expected_response_model
            and result.terminal_status == "completed"
            else "failed_closed"
        ),
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "batch_id": "multi5_batch_"
        + sha256_value([prompt_version, list(language_codes), *item_ids])[:20],
        "batch_item_count": len(item_ids),
        "batch_base_fact_ids_sha256": sha256_value(list(item_ids)),
        "usage": zh_tool._safe_usage(result.usage),
        "latency_ms": result.latency_ms,
        "attempt_count": result.attempt_count,
        "attempts": zh_tool._safe_attempts(result.attempts),
        "credentials_or_endpoints_included": False,
        "raw_response_stored": False,
        "created_at": utc_now(),
    }


def _should_split(error: Exception) -> bool:
    return zh_tool._should_split_batch_failure(error)


def _failure_record(
    *,
    item: Mapping[str, Any],
    stage: str,
    schema_version: str,
    language_codes: Sequence[str],
    error: Exception,
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "schema_version": schema_version,
        "base_fact_id": item["base_fact_id"],
        "input_record_sha256": item["input_record_sha256"],
        "target_language_codes": list(language_codes),
        "human_gold": False,
        "stage": stage,
        "terminal_status": "failed",
        "parsed_response": None,
        "attempt_count": 0,
        "attempts": [{"attempt": 0, "status": "failed", **runtime.safe_error(error)}],
        "credentials_or_endpoints_included": False,
        "raw_response_stored": False,
        "created_at": utc_now(),
        "batch_fallback_history": list(history),
    }
    if isinstance(error, BatchRequestFailure):
        row.update(error.metadata)
        row["terminal_status"] = "failed"
        row["parsed_response"] = None
        row["batch_fallback_history"] = list(history)
    return row


def run_checkpoint_stage(
    *,
    items: Sequence[Dict[str, Any]],
    output_path: Path,
    stage: str,
    schema_version: str,
    language_codes: Sequence[str],
    run_fingerprint: str,
    batch_worker: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]],
    record_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
    max_workers: int,
    batch_size: int,
    checkpoint_every: int,
    resume: bool,
    retry_failed: bool,
    inherited_run_fingerprints: Sequence[str] = (),
    max_consecutive_failed_batches: int = 3,
    stop_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    if max_consecutive_failed_batches < 1:
        raise ValueError("max_consecutive_failed_batches must be positive")
    allowed_fingerprints = {run_fingerprint, *inherited_run_fingerprints}
    journal_path = output_path.with_suffix(output_path.suffix + ".journal")
    if (output_path.exists() or journal_path.exists()) and not resume:
        raise FileExistsError(f"checkpoint exists; pass --resume: {output_path}")
    compacted = read_jsonl(output_path, missing_ok=True) if resume else []
    journal = read_jsonl_journal(journal_path) if resume else []
    _require_unique(compacted, "base_fact_id", f"{stage} checkpoint")
    _require_unique(journal, "base_fact_id", f"{stage} journal")
    allowed = {str(item["base_fact_id"]): item for item in items}
    for row in [*compacted, *journal]:
        item = allowed.get(str(row.get("base_fact_id") or ""))
        if item is None:
            raise ValueError(f"{stage} checkpoint has an out-of-scope record")
        if row.get("run_fingerprint") not in allowed_fingerprints:
            raise ValueError(f"{stage} checkpoint run fingerprint mismatch")
        if row.get("input_record_sha256") != item["input_record_sha256"]:
            raise ValueError(f"{stage} checkpoint input hash mismatch")
        record_validator(row, item)
    existing = {str(row["base_fact_id"]): dict(row) for row in compacted}
    existing.update({str(row["base_fact_id"]): dict(row) for row in journal})

    def compact() -> None:
        write_jsonl(
            output_path,
            [existing[item["base_fact_id"]] for item in items if item["base_fact_id"] in existing],
        )
        if journal_path.exists():
            journal_path.unlink()

    if resume and journal_path.exists():
        compact()
    pending = [
        item
        for item in items
        if item["base_fact_id"] not in existing
        or (
            retry_failed
            and existing[item["base_fact_id"]].get("terminal_status") == "failed"
        )
    ]

    def failure_summary(batch: Sequence[Dict[str, Any]], error: Exception) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "batch_item_count": len(batch),
            "batch_base_fact_ids_sha256": sha256_value(
                [item["base_fact_id"] for item in batch]
            ),
            **runtime.safe_error(error),
        }
        if isinstance(error, BatchRequestFailure):
            summary.update(
                {
                    key: value
                    for key, value in error.metadata.items()
                    if key
                    in {
                        "batch_id",
                        "request_model",
                        "provider_profile",
                        "response_model",
                        "response_model_identity_status",
                        "attempt_count",
                        "attempts",
                    }
                }
            )
        return summary

    stop_event = threading.Event()
    stop_reasons: List[str] = []
    attempted_ids = set()

    def process_batch(
        batch: Sequence[Dict[str, Any]], history: Sequence[Mapping[str, Any]] = ()
    ) -> List[Dict[str, Any]]:
        if stop_event.is_set():
            return []
        try:
            rows = batch_worker(batch)
            expected_ids = [str(item["base_fact_id"]) for item in batch]
            by_id = {str(row.get("base_fact_id") or ""): row for row in rows}
            if len(rows) != len(batch) or len(by_id) != len(rows) or set(by_id) != set(expected_ids):
                raise ValueError("batch worker returned mismatched records")
            ordered = [dict(by_id[base_fact_id]) for base_fact_id in expected_ids]
            for row in ordered:
                row["run_fingerprint"] = run_fingerprint
                row["batch_fallback_history"] = list(history)
                record_validator(row, allowed[str(row["base_fact_id"])])
            return ordered
        except Exception as error:
            next_history = [*history, failure_summary(batch, error)]
            stop_reason = service_stop_reason(error)
            if stop_reason:
                stop_reasons.append(stop_reason)
                stop_event.set()
            if not stop_event.is_set() and len(batch) > 1 and _should_split(error):
                midpoint = len(batch) // 2
                return [
                    *process_batch(batch[:midpoint], next_history),
                    *process_batch(batch[midpoint:], next_history),
                ]
            return [
                _failure_record(
                    item=item,
                    stage=stage,
                    schema_version=schema_version,
                    language_codes=language_codes,
                    error=error,
                    history=next_history,
                )
                for item in batch
            ]

    appended = 0

    def record(row: Dict[str, Any]) -> None:
        nonlocal appended
        row = dict(row)
        row["run_fingerprint"] = run_fingerprint
        base_fact_id = str(row["base_fact_id"])
        attempted_ids.add(base_fact_id)
        previous = existing.get(base_fact_id)
        if (
            retry_failed
            and previous is not None
            and previous.get("terminal_status") == "failed"
        ):
            retry_history = copy.deepcopy(previous.get("retry_history", []))
            retry_history.append(
                {
                    "failed_record_sha256": sha256_value(previous),
                    "attempt_count": previous.get("attempt_count"),
                    "attempts": copy.deepcopy(previous.get("attempts", [])),
                    "created_at": previous.get("created_at"),
                }
            )
            row["retry_history"] = retry_history
        record_validator(row, allowed[base_fact_id])
        append_jsonl_journal(journal_path, row)
        existing[base_fact_id] = row
        appended += 1
        if appended >= checkpoint_every:
            compact()
            appended = 0
            print(json.dumps({"stage": stage, "attempted_this_invocation": len(attempted_ids),
                              "pending_at_start": len(pending)}, sort_keys=True), flush=True)

    wave_size = max_workers * batch_size
    consecutive_failed_batches = 0

    def record_batch(rows: Sequence[Dict[str, Any]]) -> None:
        nonlocal consecutive_failed_batches
        for row in rows:
            record(row)
        if rows:
            consecutive_failed_batches = (
                consecutive_failed_batches + 1
                if all(row.get("terminal_status") == "failed" for row in rows) else 0
            )
        if consecutive_failed_batches >= max_consecutive_failed_batches:
            stop_reasons.append("consecutive_failed_batches")
            stop_event.set()

    try:
        for start in range(0, len(pending), wave_size):
            if stop_file is not None and stop_file.exists():
                stop_reasons.append("operator_stop_requested")
                stop_event.set()
            if stop_event.is_set():
                break
            wave = pending[start : start + wave_size]
            batches = [wave[index : index + batch_size] for index in range(0, len(wave), batch_size)]
            if max_workers == 1:
                for batch in batches:
                    record_batch(process_batch(batch))
            else:
                with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
                    futures = [executor.submit(process_batch, batch) for batch in batches]
                    for future in as_completed(futures):
                        record_batch(future.result())
    finally:
        compact()
    if stop_event.is_set():
        raise PipelineStopped({
            "reason": stop_reasons[0], "stage": stage,
            "attempted_source_records_this_invocation": len(attempted_ids),
            "unattempted_pending_source_records": len(pending) - len(attempted_ids),
            "checkpoint_records": len(existing),
            "checkpoint": artifact_binding(output_path),
            "in_flight_requests_drained": True,
            "unattempted_records_marked_failed": False,
            "semantic_rejection_inferred": False,
        })
    return [existing[item["base_fact_id"]] for item in items if item["base_fact_id"] in existing]


def _validate_stage_record(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    stage: str,
    schema_version: str,
    language_codes: Sequence[str],
    prompt_version: str,
    model_spec: Mapping[str, Any],
    expected_response_model: str,
    parsed_validator: Callable[[Any], None],
    lineage: Optional[Mapping[str, Any]] = None,
) -> None:
    base_fact_id = str(item["base_fact_id"])
    if row.get("schema_version") != schema_version or row.get("stage") != stage:
        raise ValueError(f"{stage} checkpoint schema mismatch: {base_fact_id}")
    if row.get("base_fact_id") != base_fact_id:
        raise ValueError(f"{stage} checkpoint identity mismatch: {base_fact_id}")
    if row.get("target_language_codes") != list(language_codes) or row.get("human_gold") is not False:
        raise ValueError(f"{stage} checkpoint language/safety mismatch: {base_fact_id}")
    if row.get("credentials_or_endpoints_included") is not False:
        raise ValueError(f"{stage} checkpoint redaction mismatch: {base_fact_id}")
    terminal = row.get("terminal_status")
    if terminal not in {"completed", "failed"}:
        raise ValueError(f"{stage} checkpoint terminal status mismatch: {base_fact_id}")
    if terminal == "failed":
        if row.get("parsed_response") is not None:
            raise ValueError(f"{stage} failed checkpoint has parsed output: {base_fact_id}")
        return
    if (
        row.get("request_model") != model_spec["model"]
        or row.get("provider_profile") != model_spec["provider_profile"]
        or row.get("expected_response_model") != expected_response_model
        or row.get("response_model") != expected_response_model
        or row.get("response_model_identity_status") != "matched"
        or row.get("prompt_version") != prompt_version
    ):
        raise ValueError(f"{stage} checkpoint model/prompt mismatch: {base_fact_id}")
    parsed_validator(row.get("parsed_response"))
    for field, expected in dict(lineage or {}).items():
        if row.get(field) != expected:
            raise ValueError(f"{stage} checkpoint lineage mismatch: {base_fact_id}")


def execute(
    args: argparse.Namespace,
    *,
    router_factory: Callable[..., runtime.ModelRouter] = runtime.ModelRouter,
) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".writer.lock").open("a+") as writer_lock:
        try:
            fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another writer owns output directory: {output_dir}") from error
        return _execute(args, router_factory=router_factory)


def _execute(
    args: argparse.Namespace,
    *,
    router_factory: Callable[..., runtime.ModelRouter],
) -> Dict[str, Any]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    resume_from = getattr(args, "resume_from", None)
    if args.retry_failed and not (args.resume or resume_from):
        raise ValueError("retry_failed requires resume")
    config = load_config(args.config.resolve())
    requested_languages = (
        args.language
        if args.language
        else config.get("scope", {}).get("selected_target_languages", DEFAULT_LANGUAGES)
    )
    language_specs = _language_specs(requested_languages)
    language_codes = [entry["code"] for entry in language_specs]
    if language_codes != list(DEFAULT_LANGUAGES):
        raise ValueError("this run is locked to ar/de/es/ja/sw in that order")
    specs = _model_specs(config, args)
    max_workers = int(
        args.max_workers
        if args.max_workers is not None
        else config.get("execution", {}).get("max_workers", 1)
    )
    generation_workers = getattr(args, "generation_workers", None)
    review_workers = getattr(args, "review_workers", None)
    generation_workers = max_workers if generation_workers is None else generation_workers
    review_workers = max_workers if review_workers is None else review_workers
    if min(max_workers, generation_workers, review_workers, args.batch_size, args.checkpoint_every) < 1:
        raise ValueError("worker, batch, and checkpoint values must be positive")
    limits = config.setdefault("execution", {}).setdefault("provider_max_concurrency", {})
    for profile in {specs["generator"]["provider_profile"], specs["reviewer"]["provider_profile"]}:
        limits[profile] = max(
            workers for role, workers in (("generator", generation_workers), ("reviewer", review_workers))
            if specs[role]["provider_profile"] == profile
        )
    loaded = load_bound_inputs(
        candidate_projection_manifest_path=args.candidate_projection_manifest,
        candidate_base_facts_path=args.candidate_base_facts,
        projection_manifest_path=args.projection_manifest,
        projection_items_path=args.projection_items,
        source_review_manifest_path=args.source_review_manifest,
        source_records_path=args.source_records,
    )
    all_items = loaded["items"]
    selected = all_items[: args.limit] if args.limit is not None else all_items
    selected_ids = [item["base_fact_id"] for item in selected]
    selected_split_counts = dict(
        sorted(Counter(item["split_assignment"] for item in selected).items())
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    managed = (
        output_dir / "selected_translation_targets.jsonl",
        output_dir / "translation_initial.jsonl",
        output_dir / "translation_reviews.jsonl",
        output_dir / "multilingual_translation_review_records.jsonl",
        output_dir / "translation_quarantine.jsonl",
        output_dir / "redacted_events.jsonl",
    )
    if args.resume and any(path.exists() for path in managed) and not manifest_path.is_file():
        raise ValueError("resume requires a run manifest when managed outputs exist")

    selection_rows = [
        {
            "schema_version": "public-benchmark-multilingual-translation-target-v1",
            "base_fact_id": item["base_fact_id"],
            "cohort_item_id": item["cohort_item_id"],
            "split_assignment": item["split_assignment"],
            "relation_partition_id": item["relation_partition_id"],
            "relation_partition_status": item["relation_partition_status"],
            "candidate_base_fact_row_sha256": item[
                "candidate_base_fact_row_sha256"
            ],
            "projection_item_sha256": item["projection_item_sha256"],
            "source_record_sha256": item["source_record_sha256"],
            "input_record_sha256": item["input_record_sha256"],
        }
        for item in selected
    ]
    selection_path = output_dir / "selected_translation_targets.jsonl"
    expected_selection_bytes = b"".join(
        canonical_json(row).encode("utf-8") + b"\n" for row in selection_rows
    )
    if selection_path.exists() and selection_path.read_bytes() != expected_selection_bytes:
        raise ValueError("existing selected target artifact differs from bound selection")
    if not selection_path.exists():
        write_jsonl(selection_path, selection_rows)

    router = router_factory(config, args.env_file.resolve(), output_dir / "redacted_events.jsonl")
    route_identity = build_route_identity_set(
        config=config,
        env_path=args.env_file.resolve(),
        routes=[
            (specs["generator"], args.generator_expected_response_model),
            (specs["reviewer"], args.reviewer_expected_response_model),
        ],
        resolved_env=(
            router.values if isinstance(getattr(router, "values", None), Mapping) else None
        ),
    )
    attach_route_identity_guard(
        router,
        config=config,
        env_path=args.env_file.resolve(),
        identity_set=route_identity,
    )
    contract = {
        "schema_version": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "input_bindings": loaded["bindings"],
        "source_fingerprints": _source_fingerprints(),
        "source_projection_id": loaded["projection_manifest"].get("projection_id"),
        "source_review_projection_id": loaded["source_review_manifest"].get(
            "projection_id"
        ),
        "source_projection_record_count": loaded["source_projection_record_count"],
        "source_projection_split_counts": loaded["source_projection_split_counts"],
        "relation_partition_status_counts": loaded[
            "relation_partition_status_counts"
        ],
        "ordered_relation_usable_base_fact_ids_sha256": loaded[
            "ordered_relation_usable_base_fact_ids_sha256"
        ],
        "relation_usable_policy": {
            "included_relation_partition_statuses": sorted(
                ELIGIBLE_RELATION_PARTITION_STATUSES
            ),
            "excluded_relation_partition_status": UNRESOLVED_RELATION_PARTITION_STATUS,
            "formal_probe_relation_id_claimed": False,
        },
        "relation_usable_record_count": len(all_items),
        "selected_record_count": len(selected),
        "selected_base_fact_ids_sha256": sha256_value(selected_ids),
        "selected_targets": artifact_binding(selection_path, rows=selection_rows),
        "selected_split_counts": selected_split_counts,
        "target_languages": language_specs,
        "route_identity": route_identity,
        "runtime_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "openai_version": importlib.metadata.version("openai"),
            "httpx_version": importlib.metadata.version("httpx"),
            "python_dotenv_version": importlib.metadata.version("python-dotenv"),
        },
        "review_independence": {
            "model_identity_distinct": True,
            "provider_profile_labels_distinct": True,
            "provider_endpoint_distinct": len(
                {
                    record["normalized_request_route_sha256"]
                    for record in route_identity["records"]
                }
            )
            == len(route_identity["records"]),
            "provider_infrastructure_independence_claimed": False,
            "claim_scope": "distinct_model_identity_only",
        },
        "models": {
            "generator": {
                **specs["generator"],
                "expected_response_model": args.generator_expected_response_model,
                "role": "five_language_translation_generation_once",
            },
            "reviewer": {
                **specs["reviewer"],
                "expected_response_model": args.reviewer_expected_response_model,
                "role": "independent_record_language_translation_review_once",
            },
        },
        "execution_policy": {
            "max_workers": max_workers,
            "generation_workers": generation_workers,
            "review_workers": review_workers,
            "provider_max_concurrency": dict(limits),
            "batch_size": args.batch_size,
            "checkpoint_every": args.checkpoint_every,
            "single_writer_journal": True,
            "recursive_binary_split_on_batch_failure": True,
            "resume_supported": True,
            "retry_failed_supported": True,
            "semantic_generation_stages": 1,
            "semantic_review_stages": 1,
            "technical_retry_or_recursive_split_may_repeat_model_requests": True,
            "request_attempts_are_reported_from_redacted_events": True,
            "automatic_repairs": 0,
            "failed_or_rejected_policy": "quarantine_without_iteration",
            "response_model_identity_fail_closed": True,
            "stop_on_quota_auth_permission_or_identity_error": True,
            "max_consecutive_failed_batches": 3,
            "unattempted_records_are_pending_not_rejected": True,
            "operator_stop_file": "STOP_AFTER_WAVE",
        },
        "evidence_boundary": {
            "human_gold": False,
            "reviewer_type": "independent_model_proxy",
            "translation_only": True,
            "factual_truth_review_performed": False,
            "distractor_falsehood_review_performed": False,
            "neutral_unrelatedness_review_performed": False,
            "relation_component_or_split_recomputed": False,
            "hf_model_execution_count": 0,
            "hf_tokenizer_execution_count": 0,
            "behavior_execution_count": 0,
            "hidden_state_collection_count": 0,
            "intervention_count": 0,
            "translation_auxiliary_models_receive_selected_source_text": True,
            "historical_exposure_contract_refresh_required": True,
        },
    }
    if resume_from:
        contract["recovery_lineage"] = recovery_source(
            resume_from, contract, output_dir, resume=args.resume,
        )
    run_fingerprint = sha256_value(contract)
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"run manifest exists; pass --resume: {manifest_path}")
        previous = read_json(manifest_path)
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError("resume run fingerprint mismatch")
    manifest = {
        **contract,
        "run_fingerprint": run_fingerprint,
        "status": "running",
        "invocation_resumed": bool(args.resume),
        "retry_failed_enabled": bool(args.retry_failed or resume_from),
        "started_or_resumed_at": utc_now(),
    }
    write_json(manifest_path, manifest)

    def run_stage(**kwargs: Any) -> List[Dict[str, Any]]:
        try:
            return run_checkpoint_stage(
                **kwargs,
                inherited_run_fingerprints=(
                    contract["recovery_lineage"]["inherited_run_fingerprints"]
                    if resume_from else []
                ),
                stop_file=output_dir / "STOP_AFTER_WAVE",
            )
        except PipelineStopped as error:
            manifest.update({
                "status": ("stopped_after_checkpoint" if error.summary["reason"] == "operator_stop_requested"
                           else "stopped_external_service"), "stopped_at": utc_now(),
                "operational_stop": error.summary,
                "completion_claims": {
                    "selected_scope_processing_complete": False,
                    "full_4513_five_language_proxy_review_complete": False,
                    "hf_tokenizer_behavior_hidden_state_or_intervention_performed": False,
                },
            })
            write_json(manifest_path, manifest)
            raise

    if hasattr(router, "_event"):
        lock = threading.Lock()
        original_event = router._event

        def locked_event(*event_args: Any, **event_kwargs: Any) -> Any:
            with lock:
                return original_event(*event_args, **event_kwargs)

        router._event = locked_event

    def validate_generation_record(row: Mapping[str, Any], item: Mapping[str, Any]) -> None:
        _validate_stage_record(
            row,
            item,
            stage="initial_generation",
            schema_version=GENERATION_SCHEMA,
            language_codes=language_codes,
            prompt_version=PROMPT_VERSION,
            model_spec=specs["generator"],
            expected_response_model=args.generator_expected_response_model,
            parsed_validator=lambda value: validate_translations(
                value, item, language_codes
            ),
        )

    def generate(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        item_ids = [item["base_fact_id"] for item in batch]
        prompt = translation_batch_prompt(batch, language_specs)
        result = router.request_json(
            "full_multilingual_translation_generation",
            "batch_" + sha256_value(item_ids)[:20],
            {
                **specs["generator"],
                "expected_response_model": args.generator_expected_response_model,
            },
            prompt,
            lambda value: validate_translation_batch(value, batch, language_codes),
        )
        metadata = _batch_metadata(
            result=result,
            spec=specs["generator"],
            expected_response_model=args.generator_expected_response_model,
            prompt=prompt,
            prompt_version=PROMPT_VERSION,
            item_ids=item_ids,
            language_codes=language_codes,
        )
        if result.terminal_status != "completed":
            raise BatchRequestFailure(metadata)
        parsed = validate_translation_batch(result.parsed_response or {}, batch, language_codes)
        return [
            {
                "schema_version": GENERATION_SCHEMA,
                "base_fact_id": item["base_fact_id"],
                "input_record_sha256": item["input_record_sha256"],
                "target_language_codes": list(language_codes),
                "human_gold": False,
                "stage": "initial_generation",
                **metadata,
                "parsed_response": parsed[item["base_fact_id"]],
            }
            for item in batch
        ]

    generations = run_stage(
        items=selected,
        output_path=output_dir / "translation_initial.jsonl",
        stage="initial_generation",
        schema_version=GENERATION_SCHEMA,
        language_codes=language_codes,
        run_fingerprint=run_fingerprint,
        batch_worker=generate,
        record_validator=validate_generation_record,
        max_workers=generation_workers,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=bool(args.resume or resume_from),
        retry_failed=bool(args.retry_failed or resume_from),
    )
    generation_by_id = {str(row["base_fact_id"]): row for row in generations}
    review_items = [
        item
        for item in selected
        if generation_by_id[item["base_fact_id"]].get("terminal_status") == "completed"
    ]

    def validate_review_record(row: Mapping[str, Any], item: Mapping[str, Any]) -> None:
        generation = generation_by_id[item["base_fact_id"]]
        _validate_stage_record(
            row,
            item,
            stage="initial_review",
            schema_version=REVIEW_SCHEMA,
            language_codes=language_codes,
            prompt_version=REVIEW_PROMPT_VERSION,
            model_spec=specs["reviewer"],
            expected_response_model=args.reviewer_expected_response_model,
            parsed_validator=lambda value: validate_reviews(value, language_codes),
            lineage={"translation_record_sha256": sha256_value(generation)},
        )

    def review(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        item_ids = [item["base_fact_id"] for item in batch]
        translations = {
            item["base_fact_id"]: generation_by_id[item["base_fact_id"]][
                "parsed_response"
            ]
            for item in batch
        }
        prompt = review_batch_prompt(batch, translations, language_specs)
        result = router.request_json(
            "full_multilingual_translation_review",
            "batch_" + sha256_value(item_ids)[:20],
            {
                **specs["reviewer"],
                "expected_response_model": args.reviewer_expected_response_model,
            },
            prompt,
            lambda value: validate_review_batch(value, batch, language_codes),
        )
        metadata = _batch_metadata(
            result=result,
            spec=specs["reviewer"],
            expected_response_model=args.reviewer_expected_response_model,
            prompt=prompt,
            prompt_version=REVIEW_PROMPT_VERSION,
            item_ids=item_ids,
            language_codes=language_codes,
        )
        if result.terminal_status != "completed":
            raise BatchRequestFailure(metadata)
        parsed = validate_review_batch(result.parsed_response or {}, batch, language_codes)
        return [
            {
                "schema_version": REVIEW_SCHEMA,
                "base_fact_id": item["base_fact_id"],
                "input_record_sha256": item["input_record_sha256"],
                "translation_record_sha256": sha256_value(
                    generation_by_id[item["base_fact_id"]]
                ),
                "target_language_codes": list(language_codes),
                "reviewer_type": "independent_model_proxy",
                "human_gold": False,
                "stage": "initial_review",
                **metadata,
                "parsed_response": parsed[item["base_fact_id"]],
            }
            for item in batch
        ]

    reviews = run_stage(
        items=review_items,
        output_path=output_dir / "translation_reviews.jsonl",
        stage="initial_review",
        schema_version=REVIEW_SCHEMA,
        language_codes=language_codes,
        run_fingerprint=run_fingerprint,
        batch_worker=review,
        record_validator=validate_review_record,
        max_workers=review_workers,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=bool(args.resume or resume_from),
        retry_failed=bool(args.retry_failed or resume_from),
    )
    review_by_id = {str(row["base_fact_id"]): row for row in reviews}

    final_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []
    language_counts = {
        code: {"accepted": 0, "quarantined": 0, "semantic_rejected": 0,
               "operational_failed": 0} for code in language_codes
    }
    for item in selected:
        base_fact_id = item["base_fact_id"]
        generation = generation_by_id[base_fact_id]
        review_row = review_by_id.get(base_fact_id)
        language_results: Dict[str, Any] = {}
        for code in language_codes:
            reasons: List[str] = []
            translation = None
            review_value = None
            if generation.get("terminal_status") != "completed":
                reasons.append("generation_failed")
            else:
                translation = generation["parsed_response"][code]
                if review_row is None or review_row.get("terminal_status") != "completed":
                    reasons.append("independent_review_failed")
                else:
                    review_value = review_row["parsed_response"][code]
                    if review_value.get("decision") != "accept":
                        reasons.append("independent_review_rejected")
            terminal_status = "completed" if not reasons else "quarantined"
            language_counts[code][
                "accepted" if terminal_status == "completed" else "quarantined"
            ] += 1
            if reasons:
                language_counts[code][
                    "semantic_rejected" if reasons == ["independent_review_rejected"]
                    else "operational_failed"
                ] += 1
            language_results[code] = {
                "terminal_status": terminal_status,
                "translation_origin": "initial",
                "translation": translation,
                "translation_equivalence_review": review_value,
                "quarantine_reasons": reasons,
            }
            if reasons:
                quarantine_rows.append(
                    {
                        "schema_version": QUARANTINE_SCHEMA,
                        "base_fact_id": base_fact_id,
                        "input_record_sha256": item["input_record_sha256"],
                        "target_language": code,
                        "human_gold": False,
                        "reasons": reasons,
                        "generation_record_sha256": sha256_value(generation),
                        "review_record_sha256": (
                            sha256_value(review_row) if review_row else None
                        ),
                    }
                )
        final_rows.append(
            {
                "schema_version": FINAL_SCHEMA,
                "base_fact_id": base_fact_id,
                "cohort_item_id": item["cohort_item_id"],
                "input_record_sha256": item["input_record_sha256"],
                "projection_item_sha256": item["projection_item_sha256"],
                "candidate_base_fact_row_sha256": item[
                    "candidate_base_fact_row_sha256"
                ],
                "source_record_sha256": item["source_record_sha256"],
                "split_assignment": item["split_assignment"],
                "relation_partition_id": item["relation_partition_id"],
                "relation_partition_status": item["relation_partition_status"],
                "source_fields": item["source"],
                "source_distractors": item["distractors"],
                "source_neutral_candidates": item["neutral_candidates"],
                "target_language_codes": list(language_codes),
                "human_gold": False,
                "reviewer_type": "independent_model_proxy",
                "terminal_status": (
                    "completed"
                    if all(
                        value["terminal_status"] == "completed"
                        for value in language_results.values()
                    )
                    else "completed_with_language_quarantine"
                ),
                "languages": language_results,
                "formal_claims": {
                    "translation_proxy_review_complete": bool(
                        review_row is not None
                        and review_row.get("terminal_status") == "completed"
                    ),
                    "all_language_translations_proxy_accepted": all(
                        value["terminal_status"] == "completed"
                        for value in language_results.values()
                    ),
                    "human_gold": False,
                    "factual_truth_reviewed": False,
                    "distractor_falsehood_reviewed": False,
                    "neutral_unrelatedness_reviewed": False,
                },
                "lineage": {
                    "generation_record_sha256": sha256_value(generation),
                    "review_record_sha256": (
                        sha256_value(review_row) if review_row else None
                    ),
                },
            }
        )

    final_path = output_dir / "multilingual_translation_review_records.jsonl"
    quarantine_path = output_dir / "translation_quarantine.jsonl"
    write_jsonl(final_path, final_rows)
    write_jsonl(quarantine_path, quarantine_rows)
    pair_count = len(selected) * len(language_codes)
    accepted_pair_count = pair_count - len(quarantine_rows)
    full_scope = len(selected) == len(all_items)
    operational_failures = sum(v["operational_failed"] for v in language_counts.values())
    common_ids = sorted(
        row["base_fact_id"] for row in final_rows
        if row["formal_claims"]["all_language_translations_proxy_accepted"]
    )
    event_rows = read_jsonl(output_dir / "redacted_events.jsonl", missing_ok=True)
    generation_events = [
        row
        for row in event_rows
        if row.get("stage") == "full_multilingual_translation_generation"
    ]
    review_events = [
        row
        for row in event_rows
        if row.get("stage") == "full_multilingual_translation_review"
    ]
    manifest.update(
        {
            "status": (
                "completed_full_multilingual_proxy_review"
                if full_scope and not quarantine_rows
                else "completed_with_quarantine_or_limited_scope"
            ),
            "completed_at": utc_now(),
            "counts": {
                "source_projection_records": loaded["source_projection_record_count"],
                "relation_usable_source_records": len(all_items),
                "selected_records": len(selected),
                "target_languages": len(language_codes),
                "record_language_pairs": pair_count,
                "proxy_accepted_record_language_pairs": accepted_pair_count,
                "quarantined_record_language_pairs": len(quarantine_rows),
                "generation_failed_source_records": sum(
                    row.get("terminal_status") != "completed" for row in generations
                ),
                "review_failed_source_records": sum(
                    row.get("terminal_status") != "completed" for row in reviews
                ),
                "automatic_repairs": 0,
                "by_language": language_counts,
                "common_five_language_accepted_facts": len(common_ids),
                "common_five_language_accepted_ids_sha256": sha256_value(common_ids),
            },
            "completion_claims": {
                "selected_scope_processing_complete": True,
                "full_4513_five_language_proxy_review_complete": bool(
                    full_scope and operational_failures == 0
                ),
                "all_4513_five_language_pairs_accepted": bool(
                    full_scope and accepted_pair_count == pair_count
                ),
                "proxy_review_not_human_gold": True,
                "relation_component_or_split_recomputed": False,
                "hf_tokenizer_behavior_hidden_state_or_intervention_performed": False,
                "historical_exposure_contract_refresh_required": True,
            },
            "translation_model_exposure": {
                "generator_unique_source_records": len(selected),
                "reviewer_unique_source_records": len(review_items),
                "generator_model_request_count": len(generation_events),
                "reviewer_model_request_count": len(review_events),
                "generator_model_attempt_count": sum(
                    int(row.get("attempt_count") or 0) for row in generation_events
                ),
                "reviewer_model_attempt_count": sum(
                    int(row.get("attempt_count") or 0) for row in review_events
                ),
                "generator_source_records_by_split": selected_split_counts,
                "reviewer_source_records_by_split": dict(
                    sorted(
                        Counter(item["split_assignment"] for item in review_items).items()
                    )
                ),
                "this_is_translation_review_exposure_not_behavior_exposure": True,
                "technical_retry_or_recursive_split_can_repeat_source_exposure": True,
            },
            "output_artifacts": {
                "selected_translation_targets": artifact_binding(
                    selection_path, rows=selection_rows
                ),
                "translation_initial": artifact_binding(
                    output_dir / "translation_initial.jsonl", rows=generations
                ),
                "translation_reviews": artifact_binding(
                    output_dir / "translation_reviews.jsonl", rows=reviews
                ),
                "multilingual_translation_review_records": artifact_binding(
                    final_path, rows=final_rows
                ),
                "translation_quarantine": artifact_binding(
                    quarantine_path, rows=quarantine_rows
                ),
            },
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--candidate-projection-manifest", type=Path, required=True)
    parser.add_argument("--candidate-base-facts", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--projection-items", type=Path, required=True)
    parser.add_argument("--source-review-manifest", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--generation-workers", type=int, help="Translation request concurrency override")
    parser.add_argument("--review-workers", type=int, help="Independent review request concurrency override")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path,
                        help="Import finalized or interrupted snapshot checkpoints into a new execution-only run")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--generator-model", default=zh_tool.DEFAULT_GENERATOR_ROUTE)
    parser.add_argument("--generator-provider-profile", default="aliyun")
    parser.add_argument("--generator-max-output-tokens", type=int, default=16384)
    parser.add_argument(
        "--generator-expected-response-model",
        default=zh_tool.DEFAULT_GENERATOR_RESPONSE_MODEL,
    )
    parser.add_argument("--reviewer-model", default=zh_tool.DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--reviewer-provider-profile", default="openai")
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=16384)
    parser.add_argument(
        "--reviewer-expected-response-model",
        default=zh_tool.DEFAULT_REVIEWER_RESPONSE_MODEL,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        manifest = execute(build_parser().parse_args(argv))
    except PipelineStopped as error:
        status = ("stopped_after_checkpoint" if error.summary["reason"] == "operator_stop_requested"
                  else "stopped_external_service")
        print(json.dumps({"status": status, **error.summary},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not any(
        counts["operational_failed"] for counts in manifest["counts"]["by_language"].values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
