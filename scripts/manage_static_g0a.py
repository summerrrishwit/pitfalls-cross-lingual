#!/usr/bin/env python3
"""Stage, review, and freeze the HF-independent G0A static stimulus bundle.

This is deliberately separate from ``prepare_manifest``.  It can stage an
authoritative review-only cohort and run only the translation/distractor/
context-generation portion of ``run_preholdout``.  It never authorizes or runs
Simulation.  Final publication is fail-closed and requires exact, terminal
Codex decisions plus independently materialized semantic-closure and
historical-exposure evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls import formal_admission  # noqa: E402
from factual_pitfalls import perturbation as runtime  # noqa: E402


TOOL_VERSION = "static-g0a-manager-v2"
STAGING_RECORD_SCHEMA = "static-g0a-staging-record-v1"
STAGING_MANIFEST_SCHEMA = "static-g0a-staging-manifest-v1"
STATIC_CODEX_TEMPLATE_SCHEMA = "static-g0a-codex-template-v1"
SOURCE_DECISION_SCHEMA = "static-g0a-source-decision-v1"
SOURCE_DECISION_MANIFEST_SCHEMA = "static-g0a-source-decision-manifest-v1"
CODEX_FACT_AUDIT_SCHEMA = "codex-static-fact-audit-v2"
REVIEW_INPUT_RECORD_SCHEMA = "static-g0a-codex-review-input-v1"
REVIEW_INPUT_MANIFEST_SCHEMA = "static-g0a-codex-review-input-manifest-v1"
CONTEXT_REPAIR_RECORD_SCHEMA = "static-g0a-context-repair-overlay-v1"
CODEX_DECISION_SCHEMA = "static-g0a-codex-decision-v1"
RESOLVED_DRAFT_MANIFEST_SCHEMA = "static-g0a-resolved-draft-manifest-v1"
SEMANTIC_CLOSURE_MANIFEST_SCHEMA = "static-g0a-semantic-closure-manifest-v2"
SEMANTIC_CANDIDATE_SCHEMA = "static-g0a-semantic-closure-candidate-v2"
SEMANTIC_DECISION_SCHEMA = "static-g0a-semantic-closure-decision-v1"
EXPOSURE_MANIFEST_SCHEMA = "static-g0a-historical-exposure-manifest-v2"
EXPOSURE_CHECK_SCHEMA = "static-g0a-historical-exposure-check-v2"
EXPOSURE_REGISTRY_SCHEMA = "static-g0a-historical-exposure-record-v1"
STATIC_FREEZE_MANIFEST_SCHEMA = "static-g0a-freeze-manifest-v1"
FINAL_BUNDLE_SCHEMA = formal_admission.INPUT_BUNDLE_SCHEMA_VERSION

EXPECTED_SOURCE_MANIFEST_SCHEMA = (
    "public-benchmark-postclosure-preperturbation-finalization-v3"
)
EXPECTED_SOURCE_STATUS = "postclosure_preperturbation_bundle_materialized_not_frozen"
EXPECTED_SPLIT_POLICY = (
    "relation-stratified-lexical-semantic-component-greedy-sha256-v1"
)
EXPECTED_SPLIT_COUNTS = {"development": 96, "validation": 32, "sealed": 32}
EXPECTED_RECORD_COUNT = 160
EXPECTED_DISTRACTORS_PER_FACT = 2
MANIPULATION_FAMILY = "explicit_false_assertion_v1"

CONTEXT_FIELDS = (
    "targeted_context_en",
    "targeted_context_zh",
    "neutral_context_en",
    "neutral_context_zh",
)
CONTEXT_REPAIR_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "base_fact_id",
        "source_id",
        "staging_record_sha256",
        "distractor_id",
        "distractor_record_sha256",
        "superseded_candidate_id",
        "superseded_candidate_record_sha256",
        *CONTEXT_FIELDS,
        "reason",
        "terminal_status",
        "reviewer_type",
        "human_gold",
        "review_blinded_to_behavior_results",
    }
)

STATIC_REVIEW_CHECKS = (
    "canonical_fact_supported",
    "unique_answer",
    "relation_preserved",
    "translation_equivalent",
    "aliases_valid",
)
VARIANT_REVIEW_CHECKS = (
    "distractor_factually_false",
    "distractor_answer_distinct",
    "distractor_same_answer_type",
    "distractor_translation_equivalent",
    "single_correct_answer",
    "targeted_explicitly_asserts_designated_distractor",
    "targeted_single_false_claim",
    "targeted_relation_preserved",
    "targeted_bilingual_equivalent",
    "neutral_truthful",
    "neutral_has_no_answer_or_distractor_cue",
    "neutral_matched",
)
REVISION_FIELDS = frozenset(
    {
        "prompt_en",
        "prompt_zh",
        "subject_en",
        "subject_zh",
        "canonical_fact_en",
        "canonical_fact_zh",
        "answer_en",
        "answer_zh",
        "answer_aliases_en",
        "answer_aliases_zh",
    }
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GateBlocked(ValueError):
    """A structurally valid request that cannot pass the G0A gate."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = sorted(set(str(item) for item in blockers))
        super().__init__(";".join(self.blockers))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def read_json(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: List[Dict[str, Any]] = []
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL: {resolved}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {resolved}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {resolved}:{line_number}")
        rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {resolved}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_write(path, payload)


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    id_values: Optional[Sequence[str]] = None,
    id_digest_field: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    if id_values is not None and id_digest_field is not None:
        result[id_digest_field] = sha256_value(sorted(id_values))
    return result


def _resolve_binding_path(binding: Mapping[str, Any], owner_path: Path) -> Path:
    path = Path(_required_string(binding.get("path"), "binding.path"))
    if not path.is_absolute():
        path = owner_path.resolve().parent / path
    return path.resolve()


def _validate_file_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_path: Optional[Path] = None,
    expected_schema: Optional[str] = None,
    jsonl: bool = False,
    allow_empty: bool = False,
) -> Tuple[Path, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = _resolve_binding_path(binding, owner_path)
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} points to the wrong artifact")
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if binding.get("byte_count") is not None and binding["byte_count"] != path.stat().st_size:
        raise ValueError(f"{label} byte_count is stale")
    value: Any = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
    if jsonl and binding.get("record_count") is not None and binding["record_count"] != len(value):
        raise ValueError(f"{label} record_count is stale")
    return path, value


def _validate_binding_identity(
    actual: Any,
    expected: Mapping[str, Any],
    label: str,
    *,
    fields: Sequence[str] = (
        "sha256",
        "schema_version",
        "record_count",
        "base_fact_ids_sha256",
    ),
) -> None:
    """Compare immutable binding metadata without trusting either path spelling."""

    if not isinstance(actual, dict):
        raise ValueError(f"{label} binding must be an object")
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{label}.{field} mismatch")


def _prepare_empty_output_dir(path: Path, label: str) -> Path:
    """Create an output directory, refusing any pre-existing payload.

    Static staging and draft export are cohort-bound operations.  Reusing a
    populated directory would let the runtime resume checkpoints produced for
    different text while matching only their stable IDs.
    """

    resolved = Path(path).resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(f"{label} is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty {label}: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    config = read_json(resolved)
    parent = config.get("extends")
    if not parent:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        project_candidate = (PROJECT_ROOT / parent_path).resolve()
        parent_path = project_candidate if project_candidate.is_file() else (resolved.parent / parent_path).resolve()
    return _deep_merge(load_config(parent_path), config)


def _validate_authoritative_source(
    bundle_path: Path,
    source_manifest_path: Path,
    *,
    expected_record_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    rows = read_jsonl(bundle_path)
    manifest = read_json(source_manifest_path)
    if manifest.get("schema_version") != EXPECTED_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source finalization manifest schema_version is unsupported")
    if manifest.get("status") != EXPECTED_SOURCE_STATUS:
        raise ValueError("source finalization manifest status is not review-only")
    if manifest.get("split_policy_version") != EXPECTED_SPLIT_POLICY:
        raise ValueError("source split_policy_version is unsupported")
    safety = manifest.get("safety_contract") or {}
    required_false = (
        "network_or_model_used",
        "model_inference_performed",
        "canonical_freeze_performed",
        "split_freeze_performed",
        "distractors_are_verified",
    )
    for field in required_false:
        if safety.get(field) is not False:
            raise ValueError(f"source safety_contract.{field} must be false")
    artifacts = manifest.get("artifacts") or {}
    source_binding = artifacts.get("postclosure_preperturbation_behavior_input_bundle")
    _validate_file_binding(
        source_binding,
        owner_path=source_manifest_path,
        label="source behavior bundle",
        expected_path=bundle_path,
        expected_schema=FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    if len(rows) != expected_record_count:
        raise ValueError(
            f"authoritative source must contain {expected_record_count} rows, found {len(rows)}"
        )
    ids: List[str] = []
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        label = f"source row {index}"
        if row.get("schema_version") != FINAL_BUNDLE_SCHEMA:
            raise ValueError(f"{label} schema_version is unsupported")
        base_fact_id = formal_admission.explicit_base_fact_id(row, label)
        ids.append(base_fact_id)
        if row.get("canonical_status") != "pending_review":
            raise ValueError(f"{label} is not pending_review")
        if row.get("human_gold") is not False:
            raise ValueError(f"{label} must have human_gold=false")
        if row.get("split_policy_version") != EXPECTED_SPLIT_POLICY:
            raise ValueError(f"{label} split_policy_version mismatch")
        split = row.get("split_assignment")
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"{label} split_assignment is invalid")
        split_counts[str(split)] += 1
        distractors = row.get("distractor_candidates")
        if not isinstance(distractors, list) or len(distractors) != EXPECTED_DISTRACTORS_PER_FACT:
            raise ValueError(f"{label} must contain exactly two distractor candidates")
        distractor_ids = []
        for distractor in distractors:
            if not isinstance(distractor, dict):
                raise ValueError(f"{label} has an invalid distractor")
            distractor_id = _required_string(
                distractor.get("distractor_id"), f"{label} distractor_id"
            )
            distractor_ids.append(distractor_id)
            if distractor.get("verified") is not False or distractor.get("human_gold") is not False:
                raise ValueError(f"{label} distractor is already represented as verified")
            if distractor.get("target_base_fact_id") != base_fact_id:
                raise ValueError(f"{label} distractor targets a different fact")
            if distractor.get("split_assignment") != split:
                raise ValueError(f"{label} distractor crosses split")
        if len(distractor_ids) != len(set(distractor_ids)):
            raise ValueError(f"{label} contains duplicate distractor_id values")
    if len(ids) != len(set(ids)):
        raise ValueError("authoritative source contains duplicate base_fact_id values")
    if expected_record_count == EXPECTED_RECORD_COUNT and dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"authoritative source split counts mismatch: {dict(split_counts)}")
    identity = {
        "path": str(Path(bundle_path).resolve()),
        "sha256": sha256_file(bundle_path),
        "schema_version": FINAL_BUNDLE_SCHEMA,
        "record_count": len(rows),
        "base_fact_ids_sha256": sha256_value(sorted(ids)),
    }
    return rows, manifest, identity


def _staging_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    base_fact_id = formal_admission.explicit_base_fact_id(row, "source row")
    revision_lineage = row.get("g0a_source_revision") or {}
    distractors = []
    for rank, distractor in enumerate(row["distractor_candidates"], start=1):
        source_review = distractor.get("g0a_source_distractor_review")
        if not isinstance(source_review, dict):
            raise ValueError(
                f"{base_fact_id} distractor is missing source-audit disposition"
            )
        distractors.append(
            {
                "distractor_id": distractor["distractor_id"],
                "rank": rank,
                "text_en": _required_string(
                    distractor.get("text_en") or distractor.get("answer_en"),
                    f"{base_fact_id} distractor text_en",
                ),
                "answer_aliases_en": list(distractor.get("answer_aliases_en") or []),
                "donor_base_fact_id": distractor.get("donor_base_fact_id"),
                "source_record_sha256": sha256_value(distractor),
                "source_audit": copy.deepcopy(source_review),
                "review_status": "pending_codex_adjudication",
                "verified": False,
                "human_gold": False,
            }
        )
    return {
        "schema_version": STAGING_RECORD_SCHEMA,
        "base_fact_id": base_fact_id,
        "source_id": row["source_id"],
        "candidate_id": row.get("candidate_id"),
        "source_record_sha256": row.get("g0a_original_source_record_sha256")
        or revision_lineage.get("original_source_record_sha256")
        or sha256_value(row),
        "effective_source_record_sha256": sha256_value(row),
        "source_revision": copy.deepcopy(revision_lineage) if revision_lineage else None,
        "source_dataset": row.get("source_dataset"),
        "answer_type": row.get("answer_type"),
        "probe_relation_id": row.get("probe_relation_id"),
        "prompt_en": row.get("prompt_en"),
        "subject_en": row.get("subject_en"),
        "answer_en": row.get("answer_en"),
        "answer_aliases_en": list(row.get("answer_aliases_en") or []),
        "canonical_fact_en": row.get("canonical_fact_en") or row.get("canonical_fact"),
        "split_group_id": row.get("split_group_id"),
        "split_assignment": row.get("split_assignment"),
        "split_policy_version": row.get("split_policy_version"),
        "distractors": distractors,
        "manipulation_family": MANIPULATION_FAMILY,
        "designated_distractor_count": 2,
        "context_pair_count_required": 2,
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "static_generation_authorized": True,
        "behavior_authorized": False,
    }


def _static_codex_template(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": STATIC_CODEX_TEMPLATE_SCHEMA,
        "base_fact_id": record["base_fact_id"],
        "staging_record_sha256": sha256_value(record),
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "terminal_status": "pending",
        "decision": "defer",
        "reason": "",
        "checks": {field: None for field in STATIC_REVIEW_CHECKS},
        "revisions": {},
        "proposed_patch": {"set": {}},
        "variants": [
            {
                "distractor_id": item["distractor_id"],
                "selected_candidate_id": None,
                "decision": "defer",
                "checks": {field: None for field in VARIANT_REVIEW_CHECKS},
                "reason": "context pair not generated yet",
            }
            for item in record["distractors"]
            if (item.get("source_audit") or {}).get(
                "translation_validation_eligible"
            )
            is True
        ],
    }


def _runtime_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    retained_distractors = []
    replacement_requests = []
    for item in record["distractors"]:
        source_review = item.get("source_audit") or {}
        if source_review.get("translation_validation_eligible") is True:
            retained_distractors.append(
                {
                    "distractor_id": item["distractor_id"],
                    "text_en": item["text_en"],
                    "answer_aliases_en": item.get("answer_aliases_en") or [],
                    "donor_base_fact_id": item.get("donor_base_fact_id"),
                    "source_audit_verdict": source_review.get("verdict"),
                    "source_audit_record_sha256": source_review.get(
                        "audit_record_sha256"
                    ),
                }
            )
            continue
        if source_review.get("replacement_generation_required") is True:
            # Do not copy the rejected text into the runtime record.  Runtime
            # translation/review prompts embed the whole item, so carrying the
            # text here would accidentally expose a rejected candidate to
            # those stages even if it were absent from ``distractors``.
            replacement_requests.append(
                {
                    "replaces_source_distractor_id": item["distractor_id"],
                    "rank": item["rank"],
                    "source_audit_verdict": source_review.get("verdict"),
                    "source_audit_record_sha256": source_review.get(
                        "audit_record_sha256"
                    ),
                    "action": "generate_then_validate_replacement",
                }
            )
            continue
        raise ValueError(
            "staging distractor has no permitted runtime disposition: "
            f"{record['base_fact_id']}:{item.get('distractor_id')}"
        )
    return {
        "base_fact_id": record["base_fact_id"],
        "base_id": record["base_fact_id"],
        "source_id": record["source_id"],
        "source_dataset": record.get("source_dataset"),
        "evidence_tier": "provisional_single_model",
        "canonical_status": "pending_review",
        "split_group_id": record.get("split_group_id"),
        "split_assignment": record.get("split_assignment"),
        "split_status": "provisional_not_frozen",
        "split_policy_version": record.get("split_policy_version"),
        "answer_type": record.get("answer_type"),
        "probe_relation_id": record.get("probe_relation_id"),
        "prompt_quality_tier": "strict_factual_completion",
        "prompt_en": record.get("prompt_en"),
        "subject_en": record.get("subject_en"),
        "answer_en": record.get("answer_en"),
        "answer_aliases_en": record.get("answer_aliases_en"),
        "canonical_fact": record.get("canonical_fact_en"),
        "distractors": retained_distractors,
        "replacement_generation_requests": replacement_requests,
        "replacement_generation_required_count": len(replacement_requests),
        "input_record_sha256": record["effective_source_record_sha256"],
        "g0a_static_only": True,
        "behavior_authorized": False,
    }


def _prepare_static_runtime_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    prepared = copy.deepcopy(dict(config))
    roles = prepared.get("model_roles")
    if not isinstance(roles, dict):
        raise ValueError("config.model_roles is required")
    generation = roles.get("perturbation_generation")
    simulation = roles.get("simulation")
    validation = roles.get("perturbation_validation")
    if not isinstance(generation, dict) or not isinstance(simulation, dict) or not isinstance(validation, dict):
        raise ValueError("config must define perturbation_generation, perturbation_validation, and simulation")
    if generation.get("manipulation_family") != MANIPULATION_FAMILY:
        raise ValueError("static G0A requires manipulation_family=explicit_false_assertion_v1")
    if int(generation.get("target_distractors_per_source", 0)) != 2:
        raise ValueError("static G0A requires target_distractors_per_source=2")
    if generation.get("matched_neutral") is not True:
        raise ValueError("static G0A requires matched_neutral=true")
    if int(generation.get("candidates_per_distractor", 0)) < 1:
        raise ValueError("static G0A requires at least one context pair per distractor")
    if simulation.get("endpoint") != "multi_option" or int(simulation.get("max_options", 0)) != 3:
        raise ValueError("static G0A requires a three-option multi_option contract")
    if simulation.get("neutral_context_mode") != "matched_per_candidate":
        raise ValueError("static G0A requires matched_per_candidate neutral contexts")
    required_checks = set(validation.get("required_checks") or ())
    if not set(runtime.EXPLICIT_FALSE_ASSERTION_CHECKS) <= required_checks:
        raise ValueError("static G0A perturbation validation is missing explicit-false checks")
    # This derived config is intentionally unusable for behavior calls.
    simulation["models"] = []
    prepared.setdefault("preflight", {}).setdefault("semantic_review", {})[
        "reviewer_type"
    ] = "codex_proxy"
    prepared["config_version"] = f"{prepared.get('config_version', 'unknown')}-g0a-static-upstream"
    prepared["g0a_static_only"] = True
    prepared["behavior_authorized"] = False
    return prepared


def _make_runtime_manifest(
    *,
    config: Mapping[str, Any],
    run_id: str,
    source_identity: Mapping[str, Any],
    source_manifest_path: Path,
    staging_manifest_path: Path,
    staging_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    roles = config["model_roles"]
    simulation = roles["simulation"]
    seed = int(config.get("pilot", {}).get("seed", 0))
    runtime_records = [_runtime_record(row) for row in staging_records]
    replacement_request_count = sum(
        int(row["replacement_generation_required_count"])
        for row in runtime_records
    )
    replenishment_enabled = bool(
        roles.get("distractor_generation", {}).get("enabled", False)
    )
    if replacement_request_count and not replenishment_enabled:
        raise ValueError(
            "source-audit rejected distractors require distractor_generation.enabled=true"
        )
    return {
        "schema_version": runtime.SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "config_version": config["config_version"],
        "config_sha256": runtime.sha256_value(config),
        "runtime_sha256": runtime._runtime_sha256(),
        "config_amendments": [],
        "prompt_input": source_identity["path"],
        "prompt_input_sha256": source_identity["sha256"],
        "input_bundle": source_identity["path"],
        "input_bundle_id": f"static-g0a:{source_identity['sha256'][:16]}",
        "input_bundle_format": "jsonl",
        "input_bundle_schema_version": source_identity["schema_version"],
        "input_bundle_sha256": source_identity["sha256"],
        "input_bundle_record_count": source_identity["record_count"],
        "external_formal_admission": {
            "required": True,
            "authorized": False,
            "blockers": [
                "g0a_static_staging_only",
                "review_freeze_manifest_missing",
                "split_freeze_manifest_missing",
            ],
        },
        "review_freeze_manifest": {"path": None, "sha256": None, "schema_version": None},
        "split_freeze_manifest": {"path": None, "sha256": None, "schema_version": None},
        "g0a_static_staging": _binding(
            staging_manifest_path, schema_version=STAGING_MANIFEST_SCHEMA
        ),
        "source_finalization_manifest": _binding(source_manifest_path),
        "input_admission_policy": {
            "mode": "g0a_static_review_only",
            "formal_prepare_bypassed": False,
            "formal_prepare_modified": False,
            "static_generation_only": True,
            "simulation_forbidden": True,
        },
        "input_admission_counts": {
            "pending_review_staged": len(staging_records),
            "formal_frozen": 0,
        },
        "seed": seed,
        "requested_limit": len(staging_records),
        "selected_count": len(staging_records),
        "selected_source_ids_sha256": sha256_value(
            sorted(str(row["source_id"]) for row in staging_records)
        ),
        "records_sha256": sha256_value(runtime_records),
        "paths_not_taken_enabled": False,
        "holdout_enabled": False,
        "static_generation_authorized": True,
        "behavior_authorized": False,
        "simulation_authorized": False,
        "simulation_models": [],
        "simulation_languages": list(runtime.SIMULATION_LANGUAGES),
        "simulation_variants": list(runtime.SIMULATION_VARIANTS),
        "shared_simulation_variants": ["original"],
        "neutral_context_mode": "matched_per_candidate",
        "simulation_endpoint": "multi_option",
        "simulation_max_options": 3,
        "manipulation_family": MANIPULATION_FAMILY,
        "designated_distractors_per_base_fact": 2,
        "statistical_unit": "base_fact_id",
        "simulation_requires_codex_proxy_review": True,
        "codex_proxy_review": None,
        "automatic_review_authority": {
            "translation": "advisory",
            "source_audit_accepted_distractor": "advisory",
            "generated_replacement_distractor": "hard_gate",
            "perturbation_context": "advisory",
            "final_authority": "codex_proxy",
        },
        "minimum_translation_acceptance": config.get("preflight", {}).get(
            "minimum_translation_acceptance", 0.0
        ),
        "currency_cost_gate": config.get("preflight", {}).get("currency_cost_gate"),
        "translation_repair_enabled": bool(roles["translation"].get("repair_rejected", False)),
        "distractor_replenishment_enabled": bool(
            roles.get("distractor_generation", {}).get("enabled", False)
        ),
        "source_audit_replacement_generation_request_count": (
            replacement_request_count
        ),
        "perturbation_auto_judge_hard_gate": bool(
            roles["perturbation_validation"].get("hard_gate", True)
        ),
        "records": runtime_records,
    }


DISTRACTOR_UPDATE_FIELDS = {
    "target_base_fact_id",
    "distractor_id",
    "donor_base_fact_id",
    "old_text_en",
    "new_text_en",
    "propagation_reason",
}


def _validate_distractor_update_declarations(
    *,
    decisions: Sequence[Mapping[str, Any]],
    original_by_id: Mapping[str, Mapping[str, Any]],
    revised_by_id: Mapping[str, Mapping[str, Any]],
    label: str,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], int]:
    """Validate explicit donor propagation assertions across the full decision set.

    The audit may state the same propagation once on the donor and once on the
    target.  Identical duplicate declarations are retained as provenance but
    collapse to one effective update.  A declaration is never trusted merely
    because its strings look plausible: it must bind an existing target
    distractor to its actual donor, its old text, and the donor's revised
    answer.
    """

    unique: Dict[Tuple[str, str], Dict[str, Any]] = {}
    declaration_count = 0
    for decision in decisions:
        owner_id = _required_string(
            decision.get("base_fact_id"), f"{label} base_fact_id"
        )
        proposed_patch = decision.get("proposed_patch") or {}
        raw_updates = proposed_patch.get("distractor_updates", [])
        if raw_updates is None:
            raw_updates = []
        if not isinstance(raw_updates, list):
            raise ValueError(f"{label} distractor_updates must be a list: {owner_id}")
        if raw_updates and proposed_patch.get("regenerate_dependents") is not True:
            raise ValueError(
                f"{label} distractor_updates require regenerate_dependents=true: {owner_id}"
            )
        for update_index, update in enumerate(raw_updates, start=1):
            declaration_count += 1
            if not isinstance(update, dict) or set(update) != DISTRACTOR_UPDATE_FIELDS:
                raise ValueError(
                    f"{label} distractor update fields are invalid: {owner_id}:{update_index}"
                )
            normalized = {
                field: _required_string(
                    update.get(field), f"{label} {field}: {owner_id}:{update_index}"
                )
                for field in DISTRACTOR_UPDATE_FIELDS
            }
            target_id = normalized["target_base_fact_id"]
            donor_id = normalized["donor_base_fact_id"]
            distractor_id = normalized["distractor_id"]
            if owner_id not in {target_id, donor_id}:
                raise ValueError(
                    f"{label} distractor update is unrelated to its decision row: {owner_id}:{distractor_id}"
                )
            target = original_by_id.get(target_id)
            donor = original_by_id.get(donor_id)
            revised_donor = revised_by_id.get(donor_id)
            if target is None or donor is None or revised_donor is None:
                raise ValueError(
                    f"{label} distractor update references an unknown target or donor: {owner_id}:{distractor_id}"
                )
            matches = [
                item
                for item in (target.get("distractor_candidates") or [])
                if item.get("distractor_id") == distractor_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{label} distractor update target is absent or ambiguous: {target_id}:{distractor_id}"
                )
            source_distractor = matches[0]
            if source_distractor.get("target_base_fact_id") != target_id:
                raise ValueError(
                    f"{label} distractor update target_base_fact_id mismatch: {target_id}:{distractor_id}"
                )
            if source_distractor.get("donor_base_fact_id") != donor_id:
                raise ValueError(
                    f"{label} distractor update donor_base_fact_id mismatch: {target_id}:{distractor_id}"
                )
            source_text = _required_string(
                source_distractor.get("text_en") or source_distractor.get("answer_en"),
                f"{label} source distractor text: {target_id}:{distractor_id}",
            )
            donor_old_answer = _required_string(
                donor.get("answer_en"), f"{label} donor old answer: {donor_id}"
            )
            donor_new_answer = _required_string(
                revised_donor.get("answer_en"), f"{label} donor revised answer: {donor_id}"
            )
            if normalized["old_text_en"] != source_text:
                raise ValueError(
                    f"{label} distractor update old_text_en is stale: {target_id}:{distractor_id}"
                )
            if normalized["old_text_en"] != donor_old_answer:
                raise ValueError(
                    f"{label} distractor update old_text_en disagrees with donor: {target_id}:{distractor_id}"
                )
            if normalized["new_text_en"] != donor_new_answer:
                raise ValueError(
                    f"{label} distractor update new_text_en disagrees with revised donor: {target_id}:{distractor_id}"
                )
            if donor_new_answer == donor_old_answer:
                raise ValueError(
                    f"{label} distractor update does not correspond to a donor answer revision: {target_id}:{distractor_id}"
                )
            key = (target_id, distractor_id)
            existing = unique.get(key)
            if existing is not None and {
                field: existing[field] for field in DISTRACTOR_UPDATE_FIELDS
            } != normalized:
                raise ValueError(
                    f"{label} has conflicting distractor update declarations: {target_id}:{distractor_id}"
                )
            if existing is None:
                unique[key] = {
                    **normalized,
                    "declaration_record_sha256s": [],
                    "declared_by_base_fact_ids": [],
                }
            declaration_sha = sha256_value(
                {
                    "decision_record_sha256": sha256_value(decision),
                    "distractor_update": normalized,
                }
            )
            if declaration_sha not in unique[key]["declaration_record_sha256s"]:
                unique[key]["declaration_record_sha256s"].append(declaration_sha)
            if owner_id not in unique[key]["declared_by_base_fact_ids"]:
                unique[key]["declared_by_base_fact_ids"].append(owner_id)
    for value in unique.values():
        value["declaration_record_sha256s"].sort()
        value["declared_by_base_fact_ids"].sort()
    return unique, declaration_count


def _validated_source_distractor_reviews(
    *,
    decision: Mapping[str, Any],
    original: Mapping[str, Any],
    label: str,
) -> Dict[str, Dict[str, Any]]:
    """Validate and index the exact source-audit verdict for every distractor."""

    base_fact_id = _required_string(
        decision.get("base_fact_id"), f"{label} base_fact_id"
    )
    source_audit = decision.get("source_audit")
    if not isinstance(source_audit, dict):
        raise ValueError(f"{label} source_audit is missing: {base_fact_id}")
    if source_audit.get("schema_version") != CODEX_FACT_AUDIT_SCHEMA:
        raise ValueError(
            f"{label} source_audit schema_version is unsupported: {base_fact_id}"
        )
    audit_record_sha256 = source_audit.get("audit_record_sha256")
    if not _is_sha256(audit_record_sha256):
        raise ValueError(
            f"{label} source_audit record hash is invalid: {base_fact_id}"
        )
    original_distractors = original.get("distractor_candidates")
    if not isinstance(original_distractors, list):
        raise ValueError(f"{label} source distractors are invalid: {base_fact_id}")
    expected = {
        _required_string(
            item.get("distractor_id"),
            f"{label} source distractor_id {base_fact_id}",
        ): (rank, item)
        for rank, item in enumerate(original_distractors, start=1)
        if isinstance(item, dict)
    }
    if len(expected) != len(original_distractors):
        raise ValueError(
            f"{label} source distractor ids are invalid or duplicate: {base_fact_id}"
        )
    reviewed_distractors = source_audit.get("distractors")
    if not isinstance(reviewed_distractors, list) or len(reviewed_distractors) != len(
        expected
    ):
        raise ValueError(
            f"{label} source_audit distractor coverage is invalid: {base_fact_id}"
        )
    indexed: Dict[str, Dict[str, Any]] = {}
    for reviewed in reviewed_distractors:
        if not isinstance(reviewed, dict):
            raise ValueError(
                f"{label} source_audit distractor is invalid: {base_fact_id}"
            )
        distractor_id = _required_string(
            reviewed.get("distractor_id"),
            f"{label} source_audit distractor_id {base_fact_id}",
        )
        if distractor_id in indexed or distractor_id not in expected:
            raise ValueError(
                f"{label} source_audit distractor coverage mismatch: "
                f"{base_fact_id}:{distractor_id}"
            )
        expected_rank, original_distractor = expected[distractor_id]
        if reviewed.get("rank") != expected_rank:
            raise ValueError(
                f"{label} source_audit distractor rank mismatch: "
                f"{base_fact_id}:{distractor_id}"
            )
        original_text = _required_string(
            original_distractor.get("text_en")
            or original_distractor.get("answer_en"),
            f"{label} source distractor text {base_fact_id}:{distractor_id}",
        )
        if reviewed.get("text_en") != original_text:
            raise ValueError(
                f"{label} source_audit distractor text mismatch: "
                f"{base_fact_id}:{distractor_id}"
            )
        verdict = reviewed.get("verdict")
        if verdict not in {"accept", "revise", "reject", "defer"}:
            raise ValueError(
                f"{label} source_audit distractor verdict is invalid: "
                f"{base_fact_id}:{distractor_id}"
            )
        if verdict == "defer":
            raise GateBlocked(
                [f"source_distractor_defer:{base_fact_id}:{distractor_id}"]
            )
        indexed[distractor_id] = {
            "schema_version": CODEX_FACT_AUDIT_SCHEMA,
            "verdict": verdict,
            "reason": _required_string(
                reviewed.get("reason"),
                f"{label} source_audit distractor reason "
                f"{base_fact_id}:{distractor_id}",
            ),
            "rank": expected_rank,
            "reviewed_text_en": original_text,
            "audit_record_sha256": audit_record_sha256,
            "decision_record_sha256": sha256_value(decision),
        }
    if set(indexed) != set(expected):
        raise ValueError(
            f"{label} source_audit omits a distractor: {base_fact_id}"
        )
    return indexed


def _apply_source_decisions(
    rows: Sequence[Mapping[str, Any]],
    decision_path: Path,
    *,
    input_bundle_sha256: str,
    require_full_coverage: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply bounded pre-translation Codex patches without claiming final review."""

    original_by_id = {
        formal_admission.explicit_base_fact_id(row, "source row"): copy.deepcopy(dict(row))
        for row in rows
    }
    revised_by_id = copy.deepcopy(original_by_id)
    for base_fact_id, revised in revised_by_id.items():
        revised["g0a_original_source_record_sha256"] = sha256_value(
            original_by_id[base_fact_id]
        )
    decisions = read_jsonl(decision_path)
    seen = set()
    decision_counts: Counter[str] = Counter()
    distractor_reviews_by_base_fact_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    allowed = {
        "subject_en",
        "prompt_en",
        "canonical_fact_en",
        "answer_en",
        "answer_aliases_en",
    }
    for index, decision in enumerate(decisions, start=1):
        if decision.get("schema_version") != SOURCE_DECISION_SCHEMA:
            raise ValueError(f"source decision row {index} schema_version is unsupported")
        base_fact_id = _required_string(decision.get("base_fact_id"), "source decision base_fact_id")
        if base_fact_id in seen or base_fact_id not in original_by_id:
            raise ValueError(f"source decision coverage is invalid: {base_fact_id}")
        seen.add(base_fact_id)
        original = original_by_id[base_fact_id]
        if decision.get("source_record_sha256") != sha256_value(original):
            raise ValueError(f"source decision has stale source hash: {base_fact_id}")
        if decision.get("input_bundle_sha256") != input_bundle_sha256:
            raise ValueError(f"source decision has stale input bundle hash: {base_fact_id}")
        if decision.get("source_id") != original.get("source_id"):
            raise ValueError(f"source decision source_id mismatch: {base_fact_id}")
        if decision.get("reviewer_type") != "codex_proxy" or decision.get("human_gold") is not False:
            raise ValueError(f"source decision reviewer identity is invalid: {base_fact_id}")
        if decision.get("blind_to_behavior") is not True:
            raise ValueError(f"source decision is not blind to behavior: {base_fact_id}")
        if decision.get("review_status") != "codex_adjudicated":
            raise ValueError(f"source decision review_status is invalid: {base_fact_id}")
        if decision.get("terminal_status") != "completed":
            raise GateBlocked([f"source_decision_not_terminal:{base_fact_id}"])
        disposition = decision.get("decision")
        if disposition not in {"accept", "revise", "reject", "defer"}:
            raise ValueError(f"source decision is invalid: {base_fact_id}")
        decision_counts[str(disposition)] += 1
        if disposition in {"reject", "defer"}:
            raise GateBlocked([f"source_decision_{disposition}:{base_fact_id}"])
        distractor_reviews_by_base_fact_id[base_fact_id] = (
            _validated_source_distractor_reviews(
                decision=decision,
                original=original,
                label="source decision",
            )
        )
        proposed_patch = decision.get("proposed_patch")
        patch_set: Any = {}
        distractor_updates: Any = []
        if proposed_patch is not None:
            if not isinstance(proposed_patch, dict) or set(proposed_patch) - {
                "set",
                "distractor_updates",
                "regenerate_dependents",
            }:
                raise ValueError(f"source proposed_patch structure is invalid: {base_fact_id}")
            patch_set = proposed_patch.get("set", {})
            distractor_updates = proposed_patch.get("distractor_updates", [])
        if patch_set is None:
            patch_set = {}
        if distractor_updates is None:
            distractor_updates = []
        if not isinstance(patch_set, dict) or not set(patch_set) <= allowed:
            raise ValueError(f"source proposed_patch.set has unsupported fields: {base_fact_id}")
        if not isinstance(distractor_updates, list):
            raise ValueError(f"source distractor_updates must be a list: {base_fact_id}")
        if disposition == "accept" and (patch_set or distractor_updates):
            raise ValueError(f"accepted source decision cannot contain a patch: {base_fact_id}")
        if disposition == "revise" and not (patch_set or distractor_updates):
            raise ValueError(f"revised source decision requires a non-empty patch: {base_fact_id}")
        normalized_patch = {
            field: _validate_revision_value(field, value, base_fact_id)
            for field, value in patch_set.items()
        }
        revised = revised_by_id[base_fact_id]
        old_answer = _required_string(revised.get("answer_en"), f"source answer_en {base_fact_id}")
        revised.update(normalized_patch)
        if "canonical_fact_en" in normalized_patch:
            revised["canonical_fact"] = normalized_patch["canonical_fact_en"]
        if "answer_en" in normalized_patch and "answer_aliases_en" not in normalized_patch:
            new_answer = normalized_patch["answer_en"]
            old_aliases = list(revised.get("answer_aliases_en") or [])
            revised["answer_aliases_en"] = [
                new_answer,
                *[
                    value
                    for value in old_aliases
                    if isinstance(value, str)
                    and value.strip()
                    and value.strip().casefold() != old_answer.casefold()
                    and value.strip().casefold() != new_answer.casefold()
                ],
            ]
        revised["g0a_source_revision"] = {
            "schema_version": SOURCE_DECISION_SCHEMA,
            "decision": disposition,
            "decision_record_sha256": sha256_value(decision),
            "original_source_record_sha256": sha256_value(original),
            "revision_fields": sorted(normalized_patch),
            "distractor_update_count": len(distractor_updates),
            "reviewer_type": "codex_proxy",
            "human_gold": False,
        }

    if require_full_coverage and seen != set(original_by_id):
        missing = sorted(set(original_by_id) - seen)
        raise GateBlocked(
            [f"source_decision_missing:{base_fact_id}" for base_fact_id in missing]
        )

    declared_updates, declaration_count = _validate_distractor_update_declarations(
        decisions=decisions,
        original_by_id=original_by_id,
        revised_by_id=revised_by_id,
        label="source decision",
    )
    for base_fact_id, reviews in distractor_reviews_by_base_fact_id.items():
        for distractor_id, review in reviews.items():
            if review["verdict"] == "revise" and (
                base_fact_id,
                distractor_id,
            ) not in declared_updates:
                raise ValueError(
                    "revised source distractor lacks a bound adjudicated update: "
                    f"{base_fact_id}:{distractor_id}"
                )
    answer_by_id = {
        base_fact_id: {
            "old_answer_en": original_by_id[base_fact_id].get("answer_en"),
            "answer_en": revised_by_id[base_fact_id].get("answer_en"),
            "answer_aliases_en": list(revised_by_id[base_fact_id].get("answer_aliases_en") or []),
        }
        for base_fact_id in revised_by_id
    }
    propagated_count = 0
    for base_fact_id, revised in revised_by_id.items():
        distractors = revised.get("distractor_candidates") or []
        for distractor in distractors:
            donor_id = distractor.get("donor_base_fact_id")
            donor = answer_by_id.get(str(donor_id))
            if donor is None or donor["answer_en"] == donor["old_answer_en"]:
                continue
            distractor["text_en"] = donor["answer_en"]
            distractor["answer_en"] = donor["answer_en"]
            distractor["answer_aliases_en"] = donor["answer_aliases_en"]
            distractor["g0a_donor_revision"] = {
                "donor_base_fact_id": donor_id,
                "source_decision_record_sha256": revised_by_id[str(donor_id)][
                    "g0a_source_revision"
                ]["decision_record_sha256"],
                "explicit_distractor_update_declared": (
                    (base_fact_id, distractor.get("distractor_id")) in declared_updates
                ),
                "distractor_update_declaration_record_sha256s": copy.deepcopy(
                    declared_updates.get(
                        (base_fact_id, distractor.get("distractor_id")), {}
                    ).get("declaration_record_sha256s", [])
                ),
            }
            propagated_count += 1
    distractor_verdict_counts: Counter[str] = Counter()
    replacement_generation_required_count = 0
    translation_validation_eligible_count = 0
    for base_fact_id, revised in revised_by_id.items():
        reviews = distractor_reviews_by_base_fact_id[base_fact_id]
        for distractor in revised.get("distractor_candidates") or []:
            distractor_id = _required_string(
                distractor.get("distractor_id"),
                f"effective source distractor_id {base_fact_id}",
            )
            review = reviews[distractor_id]
            verdict = str(review["verdict"])
            effective_text = _required_string(
                distractor.get("text_en") or distractor.get("answer_en"),
                f"effective source distractor text {base_fact_id}:{distractor_id}",
            )
            if verdict == "revise":
                update = declared_updates[(base_fact_id, distractor_id)]
                if effective_text != update["new_text_en"]:
                    raise ValueError(
                        "revised source distractor did not receive adjudicated text: "
                        f"{base_fact_id}:{distractor_id}"
                    )
            eligible = verdict in {"accept", "revise"}
            replacement_required = verdict == "reject"
            distractor["g0a_source_distractor_review"] = {
                **copy.deepcopy(review),
                "effective_text_en": effective_text,
                "translation_validation_eligible": eligible,
                "replacement_generation_required": replacement_required,
                "runtime_action": (
                    "translate_then_validate"
                    if eligible
                    else "generate_then_validate_replacement"
                ),
            }
            distractor_verdict_counts[verdict] += 1
            translation_validation_eligible_count += int(eligible)
            replacement_generation_required_count += int(replacement_required)
    ordered = [revised_by_id[formal_admission.explicit_base_fact_id(row, "source row")] for row in rows]
    return ordered, {
        "path": str(Path(decision_path).resolve()),
        "sha256": sha256_file(decision_path),
        "schema_version": SOURCE_DECISION_SCHEMA,
        "record_count": len(decisions),
        "decision_counts": dict(decision_counts),
        "covered_base_fact_count": len(seen),
        "source_universe_count": len(rows),
        "missing_decisions_are_unchanged_not_accepted": len(rows) - len(seen),
        "donor_reference_update_count": propagated_count,
        "distractor_update_declaration_count": declaration_count,
        "unique_declared_distractor_update_count": len(declared_updates),
        "applied_declared_distractor_update_count": sum(
            1
            for target_id, revised in revised_by_id.items()
            for distractor in (revised.get("distractor_candidates") or [])
            if (target_id, distractor.get("distractor_id")) in declared_updates
            and (distractor.get("g0a_donor_revision") or {}).get(
                "explicit_distractor_update_declared"
            )
            is True
        ),
        "distractor_verdict_counts": dict(distractor_verdict_counts),
        "translation_validation_eligible_distractor_count": (
            translation_validation_eligible_count
        ),
        "replacement_generation_required_count": (
            replacement_generation_required_count
        ),
    }


def adapt_source_audit(
    *,
    input_bundle_path: Path,
    audit_path: Path,
    output_dir: Path,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    """Convert the full Codex fact audit into SHA-bound staging decisions."""

    input_bundle_path = Path(input_bundle_path).resolve()
    audit_path = Path(audit_path).resolve()
    source_rows = read_jsonl(input_bundle_path)
    if len(source_rows) != expected_record_count:
        raise ValueError(
            f"source audit adapter expected {expected_record_count} source rows, found {len(source_rows)}"
        )
    source_sha = sha256_file(input_bundle_path)
    source_by_id = {
        formal_admission.explicit_base_fact_id(row, "source audit input row"): row
        for row in source_rows
    }
    if len(source_by_id) != len(source_rows):
        raise ValueError("source audit input contains duplicate base_fact_id values")
    audit_rows = read_jsonl(audit_path)
    if len(audit_rows) != len(source_rows):
        raise ValueError(
            f"Codex fact audit must cover all {len(source_rows)} rows, found {len(audit_rows)}"
        )
    decisions: List[Dict[str, Any]] = []
    seen = set()
    counts: Counter[str] = Counter()
    distractor_counts: Counter[str] = Counter()
    allowed_patch_fields = {
        "subject_en",
        "prompt_en",
        "canonical_fact_en",
        "answer_en",
        "answer_aliases_en",
    }
    for index, audit in enumerate(audit_rows, start=1):
        if audit.get("schema_version") != CODEX_FACT_AUDIT_SCHEMA:
            raise ValueError(f"fact audit row {index} schema_version is unsupported")
        if audit.get("input_bundle_sha256") != source_sha:
            raise ValueError(f"fact audit row {index} input_bundle_sha256 mismatch")
        base_fact_id = _required_string(audit.get("base_fact_id"), "fact audit base_fact_id")
        if base_fact_id in seen or base_fact_id not in source_by_id:
            raise ValueError(f"fact audit coverage is invalid: {base_fact_id}")
        seen.add(base_fact_id)
        source = source_by_id[base_fact_id]
        for field in ("source_id", "split_assignment", "probe_relation_id"):
            if audit.get(field) != source.get(field):
                raise ValueError(f"fact audit {field} mismatch: {base_fact_id}")
        if audit.get("blind_to_behavior") is not True:
            raise ValueError(f"fact audit is not blind to behavior: {base_fact_id}")
        if audit.get("reviewer_type") != "codex_proxy":
            raise ValueError(f"fact audit reviewer_type is invalid: {base_fact_id}")
        if audit.get("review_status") != "codex_adjudicated":
            raise ValueError(f"fact audit review_status is not terminal: {base_fact_id}")
        if audit.get("human_gold") is not False:
            raise ValueError(f"fact audit must keep human_gold=false: {base_fact_id}")
        disposition = audit.get("decision")
        if disposition not in {"accept", "revise", "reject", "defer"}:
            raise ValueError(f"fact audit decision is invalid: {base_fact_id}")
        counts[str(disposition)] += 1
        proposed_patch = audit.get("proposed_patch") or {"set": {}}
        if not isinstance(proposed_patch, dict) or set(proposed_patch) - {
            "set",
            "distractor_updates",
            "regenerate_dependents",
        }:
            raise ValueError(f"fact audit proposed_patch structure is invalid: {base_fact_id}")
        patch_set = proposed_patch.get("set") or {}
        if not isinstance(patch_set, dict) or not set(patch_set) <= allowed_patch_fields:
            raise ValueError(f"fact audit proposed_patch.set has unsupported fields: {base_fact_id}")
        distractor_updates = proposed_patch.get("distractor_updates", [])
        if distractor_updates is None:
            distractor_updates = []
        if not isinstance(distractor_updates, list):
            raise ValueError(f"fact audit distractor_updates must be a list: {base_fact_id}")
        if disposition == "accept" and (patch_set or distractor_updates):
            raise ValueError(f"accepted fact audit row contains a patch: {base_fact_id}")
        if disposition == "revise" and not (patch_set or distractor_updates):
            raise ValueError(f"revised fact audit row lacks a non-empty patch: {base_fact_id}")
        normalized_patch = {
            field: _validate_revision_value(field, value, base_fact_id)
            for field, value in patch_set.items()
        }
        expected_distractors = {
            item["distractor_id"]: item for item in source.get("distractor_candidates") or []
        }
        audit_distractors = audit.get("distractors")
        if not isinstance(audit_distractors, list) or len(audit_distractors) != 2:
            raise ValueError(f"fact audit must cover two distractors: {base_fact_id}")
        found_distractors = set()
        for reviewed in audit_distractors:
            distractor_id = _required_string(
                reviewed.get("distractor_id"), f"fact audit distractor_id {base_fact_id}"
            )
            if distractor_id in found_distractors or distractor_id not in expected_distractors:
                raise ValueError(f"fact audit distractor coverage mismatch: {base_fact_id}")
            found_distractors.add(distractor_id)
            if reviewed.get("text_en") != expected_distractors[distractor_id].get("text_en"):
                raise ValueError(f"fact audit distractor text mismatch: {base_fact_id}:{distractor_id}")
            verdict = reviewed.get("verdict")
            if verdict not in {"accept", "revise", "reject", "defer"}:
                raise ValueError(f"fact audit distractor verdict is invalid: {base_fact_id}:{distractor_id}")
            distractor_counts[str(verdict)] += 1
        if found_distractors != set(expected_distractors):
            raise ValueError(f"fact audit omits a distractor: {base_fact_id}")
        decisions.append(
            {
                "schema_version": SOURCE_DECISION_SCHEMA,
                "base_fact_id": base_fact_id,
                "source_id": source["source_id"],
                "source_record_sha256": sha256_value(source),
                "input_bundle_sha256": source_sha,
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "blind_to_behavior": True,
                "review_status": "codex_adjudicated",
                "terminal_status": "completed",
                "decision": disposition,
                "issues": copy.deepcopy(audit.get("issues") or []),
                "proposed_patch": {
                    "set": normalized_patch,
                    "regenerate_dependents": bool(proposed_patch.get("regenerate_dependents", False)),
                    "distractor_updates": copy.deepcopy(distractor_updates),
                },
                "source_audit": {
                    "schema_version": CODEX_FACT_AUDIT_SCHEMA,
                    "audit_record_sha256": sha256_value(audit),
                    "audit_file_sha256": sha256_file(audit_path),
                    "audit_revision": audit.get("audit_revision"),
                    "supersedes_audit_sha256": audit.get("supersedes_audit_sha256"),
                    "distractors": copy.deepcopy(audit_distractors),
                },
            }
        )
    if seen != set(source_by_id):
        missing = sorted(set(source_by_id) - seen)
        raise ValueError(f"fact audit misses source rows: {missing[:5]}")
    revised_by_id = copy.deepcopy(source_by_id)
    for decision in decisions:
        revised_by_id[decision["base_fact_id"]].update(
            (decision.get("proposed_patch") or {}).get("set") or {}
        )
    declared_updates, declaration_count = _validate_distractor_update_declarations(
        decisions=decisions,
        original_by_id=source_by_id,
        revised_by_id=revised_by_id,
        label="fact audit",
    )
    decisions.sort(key=lambda row: row["base_fact_id"])
    output_dir = _prepare_empty_output_dir(
        Path(output_dir), "source decisions output directory"
    )
    decisions_path = output_dir / "source_decisions.jsonl"
    write_jsonl(decisions_path, decisions)
    manifest = {
        "schema_version": SOURCE_DECISION_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "full_source_audit_adapted_for_static_staging",
        "input_bundle": {
            "path": str(input_bundle_path),
            "sha256": source_sha,
            "schema_version": FINAL_BUNDLE_SCHEMA,
            "record_count": len(source_rows),
            "base_fact_ids_sha256": sha256_value(sorted(source_by_id)),
        },
        "source_audit": _binding(
            audit_path,
            schema_version=CODEX_FACT_AUDIT_SCHEMA,
            record_count=len(audit_rows),
        ),
        "source_decisions": _binding(
            decisions_path,
            schema_version=SOURCE_DECISION_SCHEMA,
            record_count=len(decisions),
            id_values=[row["base_fact_id"] for row in decisions],
            id_digest_field="base_fact_ids_sha256",
        ),
        "decision_counts": dict(counts),
        "distractor_verdict_counts": dict(distractor_counts),
        "distractor_update_declaration_count": declaration_count,
        "unique_declared_distractor_update_count": len(declared_updates),
        "blind_to_behavior": True,
        "reviewer_type": "codex_proxy",
        "human_gold": False,
    }
    manifest_path = output_dir / "source_decisions_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "source_decisions": str(decisions_path),
        "source_decisions_manifest": str(manifest_path),
        "record_count": len(decisions),
        "decision_counts": dict(counts),
        "distractor_update_declaration_count": declaration_count,
        "unique_declared_distractor_update_count": len(declared_updates),
    }


def _validate_source_decisions_manifest(
    *,
    manifest_path: Path,
    decisions_path: Path,
    source_identity: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate the complete, behavior-blind Codex source adjudication packet."""

    manifest_path = Path(manifest_path).resolve()
    decisions_path = Path(decisions_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SOURCE_DECISION_MANIFEST_SCHEMA:
        raise ValueError("source decisions manifest schema_version is unsupported")
    if manifest.get("status") != "full_source_audit_adapted_for_static_staging":
        raise ValueError("source decisions manifest status is not terminal")
    if manifest.get("blind_to_behavior") is not True:
        raise ValueError("source decisions manifest is not blind to behavior")
    if manifest.get("reviewer_type") != "codex_proxy":
        raise ValueError("source decisions manifest reviewer_type is invalid")
    if manifest.get("human_gold") is not False:
        raise ValueError("source decisions manifest must keep human_gold=false")

    input_path, input_rows = _validate_file_binding(
        manifest.get("input_bundle"),
        owner_path=manifest_path,
        label="source decisions input bundle",
        expected_path=Path(str(source_identity["path"])),
        expected_schema=FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    del input_path
    _validate_binding_identity(
        manifest.get("input_bundle"), source_identity, "source decisions input_bundle"
    )
    if input_rows != list(source_rows):
        raise ValueError("source decisions input bundle rows differ from staged source rows")

    _, decisions = _validate_file_binding(
        manifest.get("source_decisions"),
        owner_path=manifest_path,
        label="source decisions",
        expected_path=decisions_path,
        expected_schema=SOURCE_DECISION_SCHEMA,
        jsonl=True,
    )
    expected_ids = sorted(
        formal_admission.explicit_base_fact_id(row, "source row") for row in source_rows
    )
    actual_ids = sorted(
        _required_string(row.get("base_fact_id"), "source decision base_fact_id")
        for row in decisions
    )
    if actual_ids != expected_ids:
        raise ValueError("source decisions manifest does not cover the exact source universe")
    decisions_binding = manifest["source_decisions"]
    if decisions_binding.get("base_fact_ids_sha256") != sha256_value(expected_ids):
        raise ValueError("source decisions base_fact_ids_sha256 is stale")

    actual_counts = Counter(str(row.get("decision")) for row in decisions)
    if dict(actual_counts) != manifest.get("decision_counts"):
        raise ValueError("source decisions manifest decision_counts is stale")
    if actual_counts.get("reject", 0) or actual_counts.get("defer", 0):
        raise GateBlocked(["source_decisions_contain_reject_or_defer"])
    if set(actual_counts) - {"accept", "revise"}:
        raise ValueError("source decisions manifest contains an invalid decision")

    _, source_audit_rows = _validate_file_binding(
        manifest.get("source_audit"),
        owner_path=manifest_path,
        label="source fact audit",
        expected_schema=CODEX_FACT_AUDIT_SCHEMA,
        jsonl=True,
    )
    source_audit_by_id = {
        _required_string(row.get("base_fact_id"), "source audit base_fact_id"): row
        for row in source_audit_rows
    }
    if len(source_audit_by_id) != len(source_audit_rows) or set(
        source_audit_by_id
    ) != set(expected_ids):
        raise ValueError("source fact audit does not cover the exact source universe")
    source_audit_file_sha256 = manifest["source_audit"]["sha256"]
    for decision in decisions:
        base_fact_id = str(decision["base_fact_id"])
        audit = source_audit_by_id[base_fact_id]
        expected_source_audit = {
            "schema_version": CODEX_FACT_AUDIT_SCHEMA,
            "audit_record_sha256": sha256_value(audit),
            "audit_file_sha256": source_audit_file_sha256,
            "audit_revision": audit.get("audit_revision"),
            "supersedes_audit_sha256": audit.get("supersedes_audit_sha256"),
            "distractors": copy.deepcopy(audit.get("distractors")),
        }
        if decision.get("source_audit") != expected_source_audit:
            raise ValueError(
                f"source decision audit provenance is stale: {base_fact_id}"
            )

    source_by_id = {
        formal_admission.explicit_base_fact_id(row, "source row"): row
        for row in source_rows
    }
    distractor_verdict_counts: Counter[str] = Counter()
    for decision in decisions:
        base_fact_id = _required_string(
            decision.get("base_fact_id"), "source decision base_fact_id"
        )
        reviews = _validated_source_distractor_reviews(
            decision=decision,
            original=source_by_id[base_fact_id],
            label="source decisions manifest",
        )
        distractor_verdict_counts.update(
            str(review["verdict"]) for review in reviews.values()
        )
    if dict(distractor_verdict_counts) != manifest.get("distractor_verdict_counts"):
        raise ValueError(
            "source decisions manifest distractor_verdict_counts is stale"
        )

    return {
        **_binding(manifest_path, schema_version=SOURCE_DECISION_MANIFEST_SCHEMA),
        "input_bundle_sha256": source_identity["sha256"],
        "source_decisions_sha256": sha256_file(decisions_path),
        "record_count": len(decisions),
        "decision_counts": dict(actual_counts),
        "distractor_verdict_counts": dict(distractor_verdict_counts),
        "blind_to_behavior": True,
        "reviewer_type": "codex_proxy",
        "human_gold": False,
    }


def stage_static_g0a(
    *,
    input_bundle_path: Path,
    source_finalization_manifest_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    source_decisions_path: Optional[Path] = None,
    source_decisions_manifest_path: Optional[Path] = None,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    rows, source_manifest, identity = _validate_authoritative_source(
        Path(input_bundle_path),
        Path(source_finalization_manifest_path),
        expected_record_count=expected_record_count,
    )
    if source_decisions_path is None or source_decisions_manifest_path is None:
        missing = []
        if source_decisions_path is None:
            missing.append("source_decisions_missing")
        if source_decisions_manifest_path is None:
            missing.append("source_decisions_manifest_missing")
        raise GateBlocked(missing)
    source_decision_manifest_provenance = _validate_source_decisions_manifest(
        manifest_path=Path(source_decisions_manifest_path),
        decisions_path=Path(source_decisions_path),
        source_identity=identity,
        source_rows=rows,
    )
    rows, source_decision_provenance = _apply_source_decisions(
        rows,
        Path(source_decisions_path),
        input_bundle_sha256=identity["sha256"],
        require_full_coverage=True,
    )
    source_decision_provenance["manifest"] = source_decision_manifest_provenance
    output_dir = _prepare_empty_output_dir(Path(output_dir), "static G0A staging directory")
    staging_records = [_staging_record(row) for row in rows]
    staging_records.sort(key=lambda row: row["base_fact_id"])
    effective_source_rows = sorted(
        (copy.deepcopy(dict(row)) for row in rows),
        key=lambda row: formal_admission.explicit_base_fact_id(
            row, "effective source row"
        ),
    )
    records_path = output_dir / "static_staging_records.jsonl"
    effective_source_path = output_dir / "effective_source_bundle.jsonl"
    template_path = output_dir / "codex_adjudication_template.jsonl"
    write_jsonl(effective_source_path, effective_source_rows)
    write_jsonl(records_path, staging_records)
    write_jsonl(template_path, [_static_codex_template(row) for row in staging_records])
    staging_records_binding = _binding(
        records_path,
        schema_version=STAGING_RECORD_SCHEMA,
        record_count=len(staging_records),
        id_values=[row["base_fact_id"] for row in staging_records],
        id_digest_field="base_fact_ids_sha256",
    )
    known_blockers = [
        "codex_adjudication_incomplete",
        "semantic_paraphrase_closure_evidence_missing",
        "historical_exposure_evidence_missing",
        "review_freeze_manifest_missing",
        "split_freeze_manifest_missing",
    ]
    closure_binding = (source_manifest.get("inputs") or {}).get(
        "closure_resolution_manifest"
    )
    if isinstance(closure_binding, dict):
        try:
            closure_path, closure = _validate_file_binding(
                closure_binding,
                owner_path=Path(source_finalization_manifest_path),
                label="source closure resolution manifest",
            )
            if closure.get("semantic_paraphrase_closure_complete") is not True:
                known_blockers.append("source_semantic_paraphrase_closure_incomplete")
            if closure.get("historical_exposure_complete") is not True:
                known_blockers.append("source_historical_exposure_incomplete")
            closure_provenance = _binding(closure_path, schema_version=closure.get("schema_version"))
        except (ValueError, FileNotFoundError):
            known_blockers.append("source_closure_resolution_evidence_invalid")
            closure_provenance = closure_binding
    else:
        known_blockers.append("source_closure_resolution_evidence_missing")
        closure_provenance = None
    manifest_path = output_dir / "static_staging_manifest.json"
    manifest = {
        "schema_version": STAGING_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "review_only_static_staged_not_frozen",
        "source_bundle": identity,
        "source_finalization_manifest": _binding(
            source_finalization_manifest_path,
            schema_version=source_manifest["schema_version"],
        ),
        "source_closure_resolution_manifest": closure_provenance,
        "source_decisions": source_decision_provenance,
        "source_decisions_manifest": source_decision_manifest_provenance,
        "effective_source_bundle": _binding(
            effective_source_path,
            schema_version=FINAL_BUNDLE_SCHEMA,
            record_count=len(effective_source_rows),
            id_values=[
                formal_admission.explicit_base_fact_id(row, "effective source row")
                for row in effective_source_rows
            ],
            id_digest_field="base_fact_ids_sha256",
        ),
        "staging_records": staging_records_binding,
        "effective_staging_bundle": copy.deepcopy(staging_records_binding),
        "codex_adjudication_template": _binding(
            template_path,
            schema_version=STATIC_CODEX_TEMPLATE_SCHEMA,
            record_count=len(staging_records),
        ),
        "counts": {
            "base_facts": len(staging_records),
            "distractors": sum(len(row["distractors"]) for row in staging_records),
            "translation_validation_eligible_distractors": sum(
                1
                for row in staging_records
                for distractor in row["distractors"]
                if (distractor.get("source_audit") or {}).get(
                    "translation_validation_eligible"
                )
                is True
            ),
            "replacement_generation_requests": sum(
                1
                for row in staging_records
                for distractor in row["distractors"]
                if (distractor.get("source_audit") or {}).get(
                    "replacement_generation_required"
                )
                is True
            ),
            "required_context_pairs": 2 * len(staging_records),
            "required_unique_behavior_inputs": 10 * len(staging_records),
            "split": dict(Counter(row["split_assignment"] for row in staging_records)),
        },
        "manipulation_family": MANIPULATION_FAMILY,
        "designated_distractors_per_base_fact": 2,
        "context_pairs_per_base_fact": 2,
        "statistical_unit": "base_fact_id",
        "reviewer_contract": {
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "blind_to_behavior_results": True,
            "terminal_decisions": ["accept", "revise", "reject"],
            "publishable_decisions": ["accept", "revise"],
            "defer_or_reject_action": "fail_closed",
            "revision_fields": sorted(REVISION_FIELDS),
        },
        "static_generation_authorized": True,
        "behavior_authorized": False,
        "simulation_authorized": False,
        "formal_prepare_modified": False,
        "formal_prepare_authorized": False,
        "known_blockers": sorted(set(known_blockers)),
    }
    write_json(manifest_path, manifest)

    result: Dict[str, Any] = {
        "staging_manifest": str(manifest_path),
        "staging_record_count": len(staging_records),
        "behavior_authorized": False,
        "known_blockers": manifest["known_blockers"],
    }
    if config_path is not None:
        if not run_id:
            raise ValueError("run_id is required when config_path is provided")
        static_config = _prepare_static_runtime_config(load_config(config_path))
        static_config_path = output_dir / "static_upstream_config.json"
        write_json(static_config_path, static_config)
        run_manifest = _make_runtime_manifest(
            config=static_config,
            run_id=run_id,
            source_identity=identity,
            source_manifest_path=Path(source_finalization_manifest_path),
            staging_manifest_path=manifest_path,
            staging_records=staging_records,
        )
        run_manifest_path = output_dir / "run_manifest.json"
        write_json(run_manifest_path, run_manifest)
        result.update(
            {
                "static_upstream_config": str(static_config_path),
                "run_manifest": str(run_manifest_path),
            }
        )
    return result


def _load_staging(
    staging_manifest_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    path = Path(staging_manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != STAGING_MANIFEST_SCHEMA:
        raise ValueError("unsupported static staging manifest schema_version")
    if manifest.get("behavior_authorized") is not False:
        raise ValueError("static staging manifest must keep behavior_authorized=false")
    records_path, rows = _validate_file_binding(
        manifest.get("staging_records"),
        owner_path=path,
        label="static staging records",
        expected_schema=STAGING_RECORD_SCHEMA,
        jsonl=True,
    )
    _validate_binding_identity(
        manifest.get("effective_staging_bundle"),
        manifest["staging_records"],
        "effective staging bundle",
    )
    effective_path = _resolve_binding_path(manifest["effective_staging_bundle"], path)
    if effective_path != records_path:
        raise ValueError("effective staging bundle points to the wrong artifact")
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != STAGING_RECORD_SCHEMA:
            raise ValueError(f"staging row {index} schema_version is unsupported")
        base_fact_id = _required_string(row.get("base_fact_id"), "staging base_fact_id")
        if base_fact_id in by_id:
            raise ValueError(f"duplicate staging base_fact_id: {base_fact_id}")
        if row.get("behavior_authorized") is not False:
            raise ValueError(f"staging row authorizes behavior: {base_fact_id}")
        distractors = row.get("distractors")
        if not isinstance(distractors, list) or len(distractors) != 2:
            raise ValueError(f"staging row must contain two distractors: {base_fact_id}")
        source_distractor_ids = set()
        eligible_count = 0
        replacement_count = 0
        for distractor in distractors:
            if not isinstance(distractor, dict):
                raise ValueError(f"staging row has an invalid distractor: {base_fact_id}")
            distractor_id = _required_string(
                distractor.get("distractor_id"),
                f"staging distractor_id {base_fact_id}",
            )
            if distractor_id in source_distractor_ids:
                raise ValueError(
                    f"staging row repeats a distractor: {base_fact_id}:{distractor_id}"
                )
            source_distractor_ids.add(distractor_id)
            source_review = distractor.get("source_audit")
            if not isinstance(source_review, dict):
                raise ValueError(
                    f"staging distractor lacks source audit: {base_fact_id}:{distractor_id}"
                )
            verdict = source_review.get("verdict")
            if verdict not in {"accept", "revise", "reject"}:
                raise ValueError(
                    f"staging distractor source verdict is invalid: {base_fact_id}:{distractor_id}"
                )
            eligible = source_review.get("translation_validation_eligible") is True
            replacement_required = (
                source_review.get("replacement_generation_required") is True
            )
            if eligible != (verdict in {"accept", "revise"}) or (
                replacement_required != (verdict == "reject")
            ):
                raise ValueError(
                    f"staging distractor source action is inconsistent: {base_fact_id}:{distractor_id}"
                )
            eligible_count += int(eligible)
            replacement_count += int(replacement_required)
        if eligible_count + replacement_count != 2:
            raise ValueError(
                f"staging row source distractor disposition is incomplete: {base_fact_id}"
            )
        by_id[base_fact_id] = row
    expected_count = (manifest.get("counts") or {}).get("base_facts")
    if expected_count != len(rows):
        raise ValueError("staging manifest base fact count mismatch")
    source_binding = manifest.get("source_bundle")
    source_path, source_rows = _validate_file_binding(
        source_binding,
        owner_path=path,
        label="staging source bundle",
        expected_schema=FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    source_identity = {
        "path": str(source_path),
        "sha256": source_binding.get("sha256"),
        "schema_version": source_binding.get("schema_version"),
        "record_count": source_binding.get("record_count"),
        "base_fact_ids_sha256": source_binding.get("base_fact_ids_sha256"),
    }
    _, effective_source_rows = _validate_file_binding(
        manifest.get("effective_source_bundle"),
        owner_path=path,
        label="effective source bundle",
        expected_schema=FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    original_by_id = {
        formal_admission.explicit_base_fact_id(row, "original source row"): row
        for row in source_rows
    }
    effective_by_id = {
        formal_admission.explicit_base_fact_id(row, "effective source row"): row
        for row in effective_source_rows
    }
    if (
        len(original_by_id) != len(source_rows)
        or len(effective_by_id) != len(effective_source_rows)
        or set(original_by_id) != set(by_id)
        or set(effective_by_id) != set(by_id)
    ):
        raise ValueError("source, effective source, and staging universes differ")
    for base_fact_id, staging_row in by_id.items():
        if staging_row.get("source_record_sha256") != sha256_value(
            original_by_id[base_fact_id]
        ):
            raise ValueError(f"staging original source hash is stale: {base_fact_id}")
        if staging_row.get("effective_source_record_sha256") != sha256_value(
            effective_by_id[base_fact_id]
        ):
            raise ValueError(f"staging effective source hash is stale: {base_fact_id}")
        if staging_row != _staging_record(effective_by_id[base_fact_id]):
            raise ValueError(f"staging row differs from effective source projection: {base_fact_id}")
    source_decisions = manifest.get("source_decisions")
    source_decisions_manifest = manifest.get("source_decisions_manifest")
    decisions_path = _resolve_binding_path(source_decisions, path)
    decisions_manifest_path, _ = _validate_file_binding(
        source_decisions_manifest,
        owner_path=path,
        label="source decisions manifest",
        expected_schema=SOURCE_DECISION_MANIFEST_SCHEMA,
    )
    _validate_source_decisions_manifest(
        manifest_path=decisions_manifest_path,
        decisions_path=decisions_path,
        source_identity=source_identity,
        source_rows=source_rows,
    )
    return manifest, rows, by_id


def _validate_runtime_manifest_staging_contract(
    manifest_path: Path,
    *,
    expected_staging_manifest_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], List[Dict[str, Any]]]:
    """Prove that runtime records are the exact projection of staged records."""

    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != runtime.SCHEMA_VERSION:
        raise ValueError("static upstream run manifest schema_version is unsupported")
    if manifest.get("static_generation_authorized") is not True:
        raise ValueError("static upstream run manifest does not authorize static generation")
    for field in ("behavior_authorized", "simulation_authorized"):
        if manifest.get(field) is not False:
            raise ValueError(f"static upstream run manifest must keep {field}=false")
    if manifest.get("simulation_models") != []:
        raise ValueError("static upstream run manifest must have an empty Simulation panel")
    if manifest.get("simulation_requires_codex_proxy_review") is not True:
        raise ValueError("static upstream run manifest is missing the Codex barrier")
    if manifest.get("codex_proxy_review") is not None:
        raise ValueError("static upstream run manifest cannot pre-authorize Codex review")
    if manifest.get("automatic_review_authority") != {
        "translation": "advisory",
        "source_audit_accepted_distractor": "advisory",
        "generated_replacement_distractor": "hard_gate",
        "perturbation_context": "advisory",
        "final_authority": "codex_proxy",
    }:
        raise ValueError("static upstream automatic-review authority contract changed")
    staging_path, _ = _validate_file_binding(
        manifest.get("g0a_static_staging"),
        owner_path=manifest_path,
        label="run manifest G0A staging",
        expected_path=expected_staging_manifest_path,
        expected_schema=STAGING_MANIFEST_SCHEMA,
    )
    staging_manifest, staging_rows, _ = _load_staging(staging_path)
    expected_records = [_runtime_record(row) for row in staging_rows]
    actual_records = manifest.get("records")
    if not isinstance(actual_records, list) or actual_records != expected_records:
        raise ValueError("run manifest records differ from effective staging projection")
    expected_digest = sha256_value(expected_records)
    if manifest.get("records_sha256") != expected_digest:
        raise ValueError("run manifest records_sha256 is stale")
    if manifest.get("selected_count") != len(expected_records):
        raise ValueError("run manifest selected_count is stale")
    expected_source_ids_digest = sha256_value(
        sorted(str(row["source_id"]) for row in staging_rows)
    )
    if manifest.get("selected_source_ids_sha256") != expected_source_ids_digest:
        raise ValueError("run manifest selected_source_ids_sha256 is stale")
    expected_replacement_request_count = sum(
        int(row["replacement_generation_required_count"])
        for row in expected_records
    )
    if (
        manifest.get("source_audit_replacement_generation_request_count")
        != expected_replacement_request_count
    ):
        raise ValueError(
            "run manifest source-audit replacement request count is stale"
        )
    if expected_replacement_request_count and manifest.get(
        "distractor_replenishment_enabled"
    ) is not True:
        raise ValueError(
            "run manifest disables required source-audit distractor replacement"
        )
    source_binding = staging_manifest["source_bundle"]
    for field in ("input_bundle_sha256", "prompt_input_sha256"):
        if manifest.get(field) != source_binding.get("sha256"):
            raise ValueError(f"run manifest {field} is stale")
    if manifest.get("input_bundle_record_count") != source_binding.get("record_count"):
        raise ValueError("run manifest input_bundle_record_count is stale")
    return manifest, staging_path, staging_manifest, staging_rows


def run_static_upstream(
    *,
    staging_dir: Path,
    config_path: Path,
    env_path: Path,
) -> Dict[str, Any]:
    """Run only static upstream stages and prove the Simulation checkpoint is empty."""

    staging_dir = Path(staging_dir).resolve()
    manifest_path = staging_dir / "run_manifest.json"
    manifest, _, _, _ = _validate_runtime_manifest_staging_contract(manifest_path)
    config = read_json(Path(config_path))
    if manifest.get("config_sha256") != runtime.sha256_value(config):
        raise ValueError("static upstream config SHA does not match run manifest")
    if manifest.get("runtime_sha256") != runtime._runtime_sha256():
        raise ValueError("static upstream runtime SHA does not match current code")
    for field in ("behavior_authorized", "simulation_authorized"):
        if manifest.get(field) is not False:
            raise ValueError(f"static upstream run must keep {field}=false")
    if manifest.get("codex_proxy_review"):
        raise ValueError("static upstream run cannot contain a Codex release decision")
    if manifest.get("simulation_requires_codex_proxy_review") is not True:
        raise ValueError("static upstream run is missing the pre-Simulation barrier")
    if (config.get("model_roles") or {}).get("simulation", {}).get("models") != []:
        raise ValueError("static upstream config must have an empty Simulation model list")
    simulation_path = staging_dir / "simulation_results.jsonl"
    if simulation_path.exists() and read_jsonl(simulation_path, allow_empty=True):
        raise ValueError("static upstream directory already contains behavior results")
    summary = runtime.run_preholdout(config, PROJECT_ROOT, staging_dir, Path(env_path))
    simulations = (
        read_jsonl(simulation_path, allow_empty=True) if simulation_path.exists() else []
    )
    if simulations:
        raise RuntimeError("static upstream safety violation: Simulation produced rows")
    stop_manifest = {
        "schema_version": "static-g0a-upstream-stop-manifest-v1",
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "static_upstream_complete_simulation_blocked",
        "run_manifest": _binding(manifest_path, schema_version=runtime.SCHEMA_VERSION),
        "static_upstream_config": _binding(config_path),
        "simulation_result_count": 0,
        "simulation_results": (
            _binding(simulation_path, record_count=0) if simulation_path.exists() else None
        ),
        "behavior_authorized": False,
        "simulation_authorized": False,
        "barrier_reason": "codex_adjudication_and_static_freeze_not_complete",
        "upstream_summary": summary,
    }
    stop_path = staging_dir / "g0a_upstream_stop_manifest.json"
    write_json(stop_path, stop_manifest)
    return {
        "stop_manifest": str(stop_path),
        "status": stop_manifest["status"],
        "behavior_authorized": False,
    }


def _rows_by_key(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    label: str,
    *,
    require_terminal: bool,
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        identifier = _required_string(row.get(key), f"{label} row {index} {key}")
        if identifier in result:
            raise ValueError(f"duplicate {label} {key}: {identifier}")
        if require_terminal and row.get("terminal_status") != "completed":
            raise GateBlocked([f"{label}_not_terminal:{identifier}"])
        result[identifier] = row
    return result


def _completed_by_key(
    rows: Sequence[Mapping[str, Any]], key: str, label: str
) -> Dict[str, Mapping[str, Any]]:
    return _rows_by_key(rows, key, label, require_terminal=True)


def _review_accepts(row: Mapping[str, Any]) -> bool:
    parsed = row.get("parsed_response")
    return bool(
        isinstance(parsed, dict)
        and parsed.get("decision") == "accept"
        and isinstance(parsed.get("checks"), dict)
        and parsed["checks"]
        and all(value is True for value in parsed["checks"].values())
    )


def _artifact_or_blocker(
    path: Path, *, allow_empty: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if not path.is_file():
        return [], f"upstream_artifact_missing:{path.name}"
    try:
        return read_jsonl(path, allow_empty=allow_empty), None
    except (ValueError, FileNotFoundError) as exc:
        return [], f"upstream_artifact_invalid:{path.name}:{type(exc).__name__}"


def _validate_upstream_stop_manifest(
    stop_manifest_path: Path,
    *,
    run_manifest_path: Path,
) -> Dict[str, Any]:
    stop_manifest_path = Path(stop_manifest_path).resolve()
    stop = read_json(stop_manifest_path)
    if stop.get("schema_version") != "static-g0a-upstream-stop-manifest-v1":
        raise ValueError("upstream stop manifest schema_version is unsupported")
    if stop.get("status") != "static_upstream_complete_simulation_blocked":
        raise ValueError("upstream stop manifest status is not terminal")
    if stop.get("simulation_result_count") != 0:
        raise ValueError("upstream stop manifest reports behavior results")
    for field in ("behavior_authorized", "simulation_authorized"):
        if stop.get(field) is not False:
            raise ValueError(f"upstream stop manifest must keep {field}=false")
    _validate_file_binding(
        stop.get("run_manifest"),
        owner_path=stop_manifest_path,
        label="upstream stop run manifest",
        expected_path=run_manifest_path,
        expected_schema=runtime.SCHEMA_VERSION,
    )
    simulation_binding = stop.get("simulation_results")
    if simulation_binding is not None:
        _, simulation_rows = _validate_file_binding(
            simulation_binding,
            owner_path=stop_manifest_path,
            label="upstream stop Simulation results",
            jsonl=True,
            allow_empty=True,
        )
        if simulation_rows:
            raise ValueError("upstream stop Simulation artifact is not empty")
    return stop


def _project_upstream_context_candidate(
    context_row: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the only review-visible projection of an upstream context row."""

    candidate = context_row.get("candidate")
    if not isinstance(candidate, dict):
        return None
    upstream_fields = {
        "targeted_context_en": "english_context",
        "targeted_context_zh": "chinese_context",
        "neutral_context_en": "neutral_english_context",
        "neutral_context_zh": "neutral_chinese_context",
    }
    projected_contexts: Dict[str, str] = {}
    for output_field, source_field in upstream_fields.items():
        value = candidate.get(source_field)
        if not isinstance(value, str) or not value.strip():
            return None
        projected_contexts[output_field] = value
    return {
        "candidate_id": context_row["candidate_id"],
        "candidate_record_sha256": sha256_value(context_row),
        **projected_contexts,
        "strength": candidate.get("strength"),
        "automatic_review": context_row.get("parsed_response"),
    }


def _project_context_repair_candidate(
    repair: Mapping[str, Any],
    superseded: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project a bound repair row without granting it automatic approval."""

    return {
        "candidate_id": repair["candidate_id"],
        "candidate_record_sha256": sha256_value(repair),
        **{field: repair[field] for field in CONTEXT_FIELDS},
        "strength": superseded.get("strength"),
        "automatic_review": None,
        "candidate_source": "codex_behavior_blind_context_repair",
        "superseded_candidate_id": repair["superseded_candidate_id"],
        "superseded_candidate_record_sha256": repair[
            "superseded_candidate_record_sha256"
        ],
    }


def _validate_and_project_context_repairs(
    repair_rows: Sequence[Mapping[str, Any]],
    *,
    staging_by_id: Mapping[str, Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, Any]],
    distractor_by_id: Mapping[str, Mapping[str, Any]],
    perturbation_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Validate repair provenance and return candidates grouped by target variant.

    A repair may add context text only.  Its base fact, distractor, and
    superseded upstream candidate must all resolve to exact bound records.
    Keeping the superseded candidate in the review packet makes the repair an
    auditable alternative rather than an in-place rewrite.
    """

    result: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    seen_candidate_ids = set(perturbation_by_id)
    seen_targets = set()
    for index, repair in enumerate(repair_rows, start=1):
        if set(repair) != set(CONTEXT_REPAIR_FIELDS):
            missing = sorted(set(CONTEXT_REPAIR_FIELDS) - set(repair))
            extra = sorted(set(repair) - set(CONTEXT_REPAIR_FIELDS))
            raise ValueError(
                f"context repair row {index} fields mismatch; missing={missing}; extra={extra}"
            )
        if repair.get("schema_version") != CONTEXT_REPAIR_RECORD_SCHEMA:
            raise ValueError(f"context repair row {index} schema_version is unsupported")
        if repair.get("terminal_status") != "completed":
            raise ValueError(f"context repair row {index} is not terminal")
        if (
            repair.get("reviewer_type") != "codex_proxy"
            or repair.get("human_gold") is not False
            or repair.get("review_blinded_to_behavior_results") is not True
        ):
            raise ValueError(f"context repair row {index} reviewer identity is invalid")

        base_fact_id = _required_string(
            repair.get("base_fact_id"), f"context repair row {index} base_fact_id"
        )
        source_id = _required_string(
            repair.get("source_id"), f"context repair row {index} source_id"
        )
        distractor_id = _required_string(
            repair.get("distractor_id"), f"context repair row {index} distractor_id"
        )
        superseded_id = _required_string(
            repair.get("superseded_candidate_id"),
            f"context repair row {index} superseded_candidate_id",
        )
        candidate_id = _required_string(
            repair.get("candidate_id"), f"context repair row {index} candidate_id"
        )
        _required_string(repair.get("reason"), f"context repair row {index} reason")
        for field in (
            "staging_record_sha256",
            "distractor_record_sha256",
            "superseded_candidate_record_sha256",
        ):
            if not _is_sha256(repair.get(field)):
                raise ValueError(f"context repair row {index} {field} is not SHA-256")
        for field in CONTEXT_FIELDS:
            _required_string(repair.get(field), f"context repair row {index} {field}")

        staging = staging_by_id.get(base_fact_id)
        review = review_by_id.get(base_fact_id)
        if staging is None or review is None:
            raise ValueError(
                f"context repair references unknown base fact: {base_fact_id}"
            )
        if source_id != staging.get("source_id") or source_id != review.get("source_id"):
            raise ValueError(f"context repair source identity mismatch: {base_fact_id}")
        if repair["staging_record_sha256"] != sha256_value(staging):
            raise ValueError(f"context repair has stale staging hash: {base_fact_id}")

        variants = review.get("variants")
        if not isinstance(variants, list):
            raise ValueError(f"review variants are invalid: {base_fact_id}")
        variant_matches = [
            value
            for value in variants
            if isinstance(value, dict) and value.get("distractor_id") == distractor_id
        ]
        if len(variant_matches) != 1:
            raise ValueError(
                f"context repair target variant is absent or ambiguous: {base_fact_id}:{distractor_id}"
            )
        variant = variant_matches[0]
        distractor = distractor_by_id.get(distractor_id)
        if distractor is None or distractor.get("source_id") != source_id:
            raise ValueError(
                f"context repair distractor identity mismatch: {base_fact_id}:{distractor_id}"
            )
        distractor_hash = sha256_value(distractor)
        if (
            repair["distractor_record_sha256"] != distractor_hash
            or variant.get("distractor_record_sha256") != distractor_hash
        ):
            raise ValueError(
                f"context repair has stale distractor hash: {base_fact_id}:{distractor_id}"
            )

        superseded_row = perturbation_by_id.get(superseded_id)
        if (
            superseded_row is None
            or superseded_row.get("source_id") != source_id
            or superseded_row.get("distractor_id") != distractor_id
        ):
            raise ValueError(
                f"context repair superseded candidate identity mismatch: "
                f"{base_fact_id}:{distractor_id}:{superseded_id}"
            )
        superseded_hash = sha256_value(superseded_row)
        if repair["superseded_candidate_record_sha256"] != superseded_hash:
            raise ValueError(
                f"context repair has stale superseded candidate hash: "
                f"{base_fact_id}:{distractor_id}:{superseded_id}"
            )
        superseded = _project_upstream_context_candidate(superseded_row)
        candidates = variant.get("context_candidates")
        if superseded is None or not isinstance(candidates, list):
            raise ValueError(
                f"context repair superseded candidate is not reviewable: "
                f"{base_fact_id}:{distractor_id}:{superseded_id}"
            )
        superseded_matches = [
            value
            for value in candidates
            if isinstance(value, dict) and value.get("candidate_id") == superseded_id
        ]
        if len(superseded_matches) != 1 or superseded_matches[0] != superseded:
            raise ValueError(
                f"context repair superseded candidate does not match upstream: "
                f"{base_fact_id}:{distractor_id}:{superseded_id}"
            )

        target = (base_fact_id, distractor_id, superseded_id)
        if target in seen_targets:
            raise ValueError(
                f"duplicate context repair target: {base_fact_id}:{distractor_id}:{superseded_id}"
            )
        seen_targets.add(target)
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"duplicate context repair candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        if all(repair[field] == superseded[field] for field in CONTEXT_FIELDS):
            raise ValueError(
                f"context repair does not change context: {base_fact_id}:{distractor_id}"
            )
        result[(base_fact_id, distractor_id)].append(
            _project_context_repair_candidate(repair, superseded)
        )
    for candidates in result.values():
        candidates.sort(key=lambda value: value["candidate_id"])
    return result


def _validate_review_context_provenance(
    *,
    staging_by_id: Mapping[str, Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, Any]],
    distractor_by_id: Mapping[str, Mapping[str, Any]],
    perturbation_by_id: Mapping[str, Mapping[str, Any]],
    repair_candidates: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
) -> None:
    """Reconstruct every variant and context from bound source artifacts."""

    source_to_base = {
        row["source_id"]: base_fact_id
        for base_fact_id, row in staging_by_id.items()
    }
    if len(source_to_base) != len(staging_by_id):
        raise ValueError("staging source_id values must be unique")
    context_by_distractor: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for context in perturbation_by_id.values():
        if context.get("source_id") in source_to_base:
            context_by_distractor[str(context.get("distractor_id"))].append(context)

    for base_fact_id, review in review_by_id.items():
        staging = staging_by_id[base_fact_id]
        source_id = staging["source_id"]
        staging_distractor_by_id = {
            value["distractor_id"]: value for value in staging["distractors"]
        }
        source_audit_eligible_ids = {
            distractor_id
            for distractor_id, value in staging_distractor_by_id.items()
            if (value.get("source_audit") or {}).get(
                "translation_validation_eligible"
            )
            is True
        }
        expected_distractors = sorted(
            (
                value
                for value in distractor_by_id.values()
                if value.get("source_id") == source_id
                and context_by_distractor.get(str(value["item_id"]))
                and (
                    str(value["item_id"]) in source_audit_eligible_ids
                    or (
                        value.get("distractor_source")
                        == "generated_replacement"
                        and value.get("terminal_status") == "completed"
                        and _review_accepts(value)
                    )
                )
            ),
            key=lambda value: str(value["item_id"]),
        )
        if len(expected_distractors) != EXPECTED_DISTRACTORS_PER_FACT:
            raise ValueError(
                f"bound upstream distractor count changed: {base_fact_id}"
            )
        variants = review.get("variants")
        if not isinstance(variants, list):
            raise ValueError(f"review variants are invalid: {base_fact_id}")
        actual_variant_ids = [
            value.get("distractor_id") if isinstance(value, dict) else None
            for value in variants
        ]
        expected_variant_ids = [
            str(value["item_id"]) for value in expected_distractors
        ]
        if actual_variant_ids != expected_variant_ids:
            raise ValueError(
                f"review variant identity/order differs from upstream: {base_fact_id}"
            )

        for variant, distractor in zip(variants, expected_distractors):
            distractor_id = str(distractor["item_id"])
            expected_candidates = []
            for context in sorted(
                context_by_distractor[distractor_id],
                key=lambda value: str(value["candidate_id"]),
            ):
                projected = _project_upstream_context_candidate(context)
                if projected is not None:
                    expected_candidates.append(projected)
            expected_candidates.extend(
                copy.deepcopy(
                    list(repair_candidates.get((base_fact_id, distractor_id), []))
                )
            )
            expected_candidates.sort(key=lambda value: value["candidate_id"])
            expected_variant = {
                "distractor_id": distractor_id,
                "donor_base_fact_id": (
                    staging_distractor_by_id.get(distractor_id) or {}
                ).get("donor_base_fact_id"),
                "answer_aliases_en": list(
                    (staging_distractor_by_id.get(distractor_id) or {}).get(
                        "answer_aliases_en"
                    )
                    or []
                ),
                "distractor_record_sha256": sha256_value(distractor),
                "text_en": distractor.get("distractor_en"),
                "text_zh": distractor.get("distractor_zh"),
                "automatic_review": distractor.get("parsed_response"),
                "automatic_review_terminal_status": distractor.get(
                    "terminal_status"
                ),
                "context_candidates": expected_candidates,
            }
            if variant != expected_variant:
                raise ValueError(
                    f"review variant/context differs from bound provenance: "
                    f"{base_fact_id}:{distractor_id}"
                )


def export_codex_review(
    *,
    staging_manifest_path: Path,
    upstream_run_dir: Path,
    output_dir: Path,
    context_repairs_path: Optional[Path] = None,
) -> Dict[str, Any]:
    staging_manifest_path = Path(staging_manifest_path).resolve()
    upstream_run_dir = Path(upstream_run_dir).resolve()
    run_manifest_path = upstream_run_dir / "run_manifest.json"
    run_manifest, _, staging_manifest, staging_rows = (
        _validate_runtime_manifest_staging_contract(
            run_manifest_path,
            expected_staging_manifest_path=staging_manifest_path,
        )
    )
    staging_by_id = {row["base_fact_id"]: row for row in staging_rows}
    if run_manifest.get("behavior_authorized") is not False:
        raise ValueError("upstream run manifest must keep behavior_authorized=false")
    if run_manifest.get("simulation_authorized") is not False:
        raise ValueError("upstream run manifest must keep simulation_authorized=false")
    simulation_path = upstream_run_dir / "simulation_results.jsonl"
    if simulation_path.exists() and read_jsonl(simulation_path, allow_empty=True):
        raise GateBlocked(["behavior_results_present_review_not_blind"])
    stop_manifest_path = upstream_run_dir / "g0a_upstream_stop_manifest.json"
    _validate_upstream_stop_manifest(
        stop_manifest_path,
        run_manifest_path=run_manifest_path,
    )

    artifact_names = (
        "translations.jsonl",
        "translation_reviews.jsonl",
        "verified_distractors.jsonl",
        "perturbations.jsonl",
    )
    artifacts: Dict[str, List[Dict[str, Any]]] = {}
    blockers: List[str] = []
    for name in artifact_names:
        rows, blocker = _artifact_or_blocker(upstream_run_dir / name)
        artifacts[name] = rows
        if blocker:
            blockers.append(blocker)
    if blockers:
        raise GateBlocked(blockers)

    translations = _completed_by_key(
        artifacts["translations.jsonl"], "source_id", "translation"
    )
    translation_reviews = _rows_by_key(
        artifacts["translation_reviews.jsonl"],
        "source_id",
        "translation_review",
        require_terminal=False,
    )
    distractor_rows = _rows_by_key(
        artifacts["verified_distractors.jsonl"],
        "item_id",
        "distractor_review",
        require_terminal=False,
    )
    perturbation_rows = _completed_by_key(
        artifacts["perturbations.jsonl"], "candidate_id", "context_review"
    )

    source_to_base = {row["source_id"]: row["base_fact_id"] for row in staging_rows}
    if len(source_to_base) != len(staging_rows):
        raise ValueError("staging source_id values must be unique")
    context_by_distractor: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in perturbation_rows.values():
        if row.get("source_id") in source_to_base:
            context_by_distractor[str(row.get("distractor_id"))].append(row)

    review_rows: List[Dict[str, Any]] = []
    decision_templates: List[Dict[str, Any]] = []
    blockers = []
    for staging in staging_rows:
        source_id = staging["source_id"]
        base_fact_id = staging["base_fact_id"]
        staging_distractor_by_id = {
            item["distractor_id"]: item for item in staging["distractors"]
        }
        source_audit_eligible_ids = {
            distractor_id
            for distractor_id, item in staging_distractor_by_id.items()
            if (item.get("source_audit") or {}).get(
                "translation_validation_eligible"
            )
            is True
        }
        translation = translations.get(source_id)
        translation_review = translation_reviews.get(source_id)
        if translation is None:
            blockers.append(f"translation_missing:{base_fact_id}")
            continue
        if translation_review is None:
            blockers.append(f"translation_review_missing:{base_fact_id}")
            continue
        parsed_translation = translation.get("parsed_response")
        if not isinstance(parsed_translation, dict):
            blockers.append(f"translation_payload_missing:{base_fact_id}")
            continue
        # Only the two distractors selected upstream receive context pairs, so
        # those rows define the actual static variants.  Source-audit accepted
        # originals keep their automatic review as advisory evidence; generated
        # replacements remain hard-gated by automatic validation.  Unused
        # backups remain auditable but must not make export ambiguous.
        distractors = sorted(
            (
                row
                for row in distractor_rows.values()
                if row.get("source_id") == source_id
                and context_by_distractor.get(str(row["item_id"]))
                and (
                    str(row["item_id"]) in source_audit_eligible_ids
                    or (
                        row.get("distractor_source") == "generated_replacement"
                        and row.get("terminal_status") == "completed"
                        and _review_accepts(row)
                    )
                )
            ),
            key=lambda row: str(row["item_id"]),
        )
        if len(distractors) != 2:
            blockers.append(
                f"accepted_distractor_count:{base_fact_id}:expected_2:actual_{len(distractors)}"
            )
            continue
        variants = []
        for distractor in distractors:
            distractor_id = str(distractor["item_id"])
            contexts = sorted(
                context_by_distractor.get(distractor_id, []),
                key=lambda row: str(row["candidate_id"]),
            )
            usable_contexts = []
            for context_row in contexts:
                projected = _project_upstream_context_candidate(context_row)
                if projected is not None:
                    usable_contexts.append(projected)
            if not usable_contexts:
                blockers.append(f"context_pair_missing:{base_fact_id}:{distractor_id}")
            variants.append(
                {
                    "distractor_id": distractor_id,
                    "donor_base_fact_id": (
                        staging_distractor_by_id.get(distractor_id) or {}
                    ).get("donor_base_fact_id"),
                    "answer_aliases_en": list(
                        (staging_distractor_by_id.get(distractor_id) or {}).get(
                            "answer_aliases_en"
                        )
                        or []
                    ),
                    "distractor_record_sha256": sha256_value(distractor),
                    "text_en": distractor.get("distractor_en"),
                    "text_zh": distractor.get("distractor_zh"),
                    "automatic_review": distractor.get("parsed_response"),
                    "automatic_review_terminal_status": distractor.get(
                        "terminal_status"
                    ),
                    "context_candidates": usable_contexts,
                }
            )
        record = {
            "schema_version": REVIEW_INPUT_RECORD_SCHEMA,
            "base_fact_id": base_fact_id,
            "source_id": source_id,
            "staging_record_sha256": sha256_value(staging),
            "source": {
                field: staging.get(field)
                for field in (
                    "source_dataset",
                    "answer_type",
                    "probe_relation_id",
                    "subject_en",
                    "prompt_en",
                    "answer_en",
                    "answer_aliases_en",
                    "canonical_fact_en",
                    "split_group_id",
                    "split_assignment",
                    "split_policy_version",
                )
            },
            "translation": parsed_translation,
            "translation_record_sha256": sha256_value(translation),
            "translation_review": translation_review.get("parsed_response"),
            "translation_review_terminal_status": translation_review.get(
                "terminal_status"
            ),
            "translation_review_record_sha256": sha256_value(translation_review),
            "variants": variants,
            "manipulation_family": MANIPULATION_FAMILY,
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "review_blinded_to_behavior_results": True,
        }
        review_rows.append(record)
        decision_templates.append(
            {
                "schema_version": CODEX_DECISION_SCHEMA,
                "base_fact_id": base_fact_id,
                "review_input_record_sha256": sha256_value(record),
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "terminal_status": "pending",
                "decision": "defer",
                "reason": "",
                "checks": {field: None for field in STATIC_REVIEW_CHECKS},
                "revisions": {},
                "proposed_patch": {"set": {}},
                "variants": [
                    {
                        "distractor_id": variant["distractor_id"],
                        "selected_candidate_id": None,
                        "decision": "defer",
                        "reason": "",
                        "checks": {field: None for field in VARIANT_REVIEW_CHECKS},
                    }
                    for variant in variants
                ],
            }
        )
    if len(review_rows) != len(staging_rows):
        blockers.append(
            f"review_input_coverage:expected_{len(staging_rows)}:actual_{len(review_rows)}"
        )
    if blockers:
        raise GateBlocked(blockers)

    repair_rows: List[Dict[str, Any]] = []
    resolved_context_repairs_path: Optional[Path] = None
    if context_repairs_path is not None:
        resolved_context_repairs_path = Path(context_repairs_path).resolve()
        repair_rows = read_jsonl(resolved_context_repairs_path)
        review_by_id = {row["base_fact_id"]: row for row in review_rows}
        repair_candidates = _validate_and_project_context_repairs(
            repair_rows,
            staging_by_id=staging_by_id,
            review_by_id=review_by_id,
            distractor_by_id=distractor_rows,
            perturbation_by_id=perturbation_rows,
        )
        for (base_fact_id, distractor_id), candidates in repair_candidates.items():
            review = review_by_id[base_fact_id]
            variant = next(
                value
                for value in review["variants"]
                if value["distractor_id"] == distractor_id
            )
            variant["context_candidates"].extend(candidates)
            variant["context_candidates"].sort(
                key=lambda value: value["candidate_id"]
            )
        templates_by_id = {
            row["base_fact_id"]: row for row in decision_templates
        }
        for base_fact_id, review in review_by_id.items():
            templates_by_id[base_fact_id]["review_input_record_sha256"] = (
                sha256_value(review)
            )

    review_rows.sort(key=lambda row: row["base_fact_id"])
    decision_templates.sort(key=lambda row: row["base_fact_id"])
    output_dir = _prepare_empty_output_dir(
        Path(output_dir), "Codex review output directory"
    )
    review_path = output_dir / "codex_adjudication_input.jsonl"
    decision_path = output_dir / "codex_adjudication_decisions.template.jsonl"
    write_jsonl(review_path, review_rows)
    write_jsonl(decision_path, decision_templates)
    upstream_bindings = {
        name: _binding(upstream_run_dir / name, record_count=len(artifacts[name]))
        for name in artifact_names
    }
    manifest = {
        "schema_version": REVIEW_INPUT_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "codex_review_ready_behavior_blind",
        "staging_manifest": _binding(
            staging_manifest_path, schema_version=STAGING_MANIFEST_SCHEMA
        ),
        "run_manifest": _binding(run_manifest_path, schema_version=runtime.SCHEMA_VERSION),
        "upstream_stop_manifest": _binding(
            stop_manifest_path,
            schema_version="static-g0a-upstream-stop-manifest-v1",
        ),
        "upstream_artifacts": upstream_bindings,
        "review_input": _binding(
            review_path,
            schema_version=REVIEW_INPUT_RECORD_SCHEMA,
            record_count=len(review_rows),
            id_values=[row["base_fact_id"] for row in review_rows],
            id_digest_field="base_fact_ids_sha256",
        ),
        "decision_template": _binding(
            decision_path,
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(decision_templates),
        ),
        "base_fact_count": len(review_rows),
        "variant_count": sum(len(row["variants"]) for row in review_rows),
        "context_candidate_count": sum(
            len(variant["context_candidates"])
            for row in review_rows
            for variant in row["variants"]
        ),
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "review_blinded_to_behavior_results": True,
        "simulation_results_checked": True,
        "simulation_result_count": 0,
        "behavior_authorized": False,
    }
    if resolved_context_repairs_path is not None:
        manifest["context_repair_overlay"] = _binding(
            resolved_context_repairs_path,
            schema_version=CONTEXT_REPAIR_RECORD_SCHEMA,
            record_count=len(repair_rows),
            id_values=[row["candidate_id"] for row in repair_rows],
            id_digest_field="candidate_ids_sha256",
        )
        manifest["context_repair_count"] = len(repair_rows)
    manifest_path = output_dir / "codex_adjudication_input_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "review_input_manifest": str(manifest_path),
        "base_fact_count": len(review_rows),
        "variant_count": manifest["variant_count"],
        "context_repair_count": len(repair_rows),
        "behavior_authorized": False,
    }


def _validate_source_binding(
    actual: Any, expected: Mapping[str, Any], label: str
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} is missing input_bundle binding")
    for field in ("sha256", "schema_version", "record_count", "base_fact_ids_sha256"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{label} input_bundle.{field} mismatch")


def _load_review_input(
    review_input_manifest_path: Path,
    staging_manifest_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    path = Path(review_input_manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != REVIEW_INPUT_MANIFEST_SCHEMA:
        raise ValueError("unsupported Codex review input manifest schema_version")
    if manifest.get("review_blinded_to_behavior_results") is not True:
        raise ValueError("Codex review input is not behavior blind")
    if manifest.get("simulation_results_checked") is not True:
        raise ValueError("Codex review input did not check Simulation results")
    if manifest.get("simulation_result_count") != 0:
        raise ValueError("Codex review input contains behavior exposure")
    if manifest.get("reviewer_type") != "codex_proxy" or manifest.get("human_gold") is not False:
        raise ValueError("Codex review input reviewer identity is invalid")
    if manifest.get("behavior_authorized") is not False:
        raise ValueError("Codex review input must keep behavior_authorized=false")
    _validate_file_binding(
        manifest.get("staging_manifest"),
        owner_path=path,
        label="review input staging manifest",
        expected_path=staging_manifest_path,
        expected_schema=STAGING_MANIFEST_SCHEMA,
    )
    _, staging_rows, staging_by_id = _load_staging(staging_manifest_path)
    run_manifest_path, _ = _validate_file_binding(
        manifest.get("run_manifest"),
        owner_path=path,
        label="review input run manifest",
        expected_schema=runtime.SCHEMA_VERSION,
    )
    _validate_runtime_manifest_staging_contract(
        run_manifest_path,
        expected_staging_manifest_path=staging_manifest_path,
    )
    stop_manifest_path, _ = _validate_file_binding(
        manifest.get("upstream_stop_manifest"),
        owner_path=path,
        label="review input upstream stop manifest",
        expected_schema="static-g0a-upstream-stop-manifest-v1",
    )
    _validate_upstream_stop_manifest(
        stop_manifest_path,
        run_manifest_path=run_manifest_path,
    )
    upstream_bindings = manifest.get("upstream_artifacts")
    if not isinstance(upstream_bindings, dict):
        raise ValueError("review input upstream_artifacts must be an object")
    upstream_rows: Dict[str, List[Dict[str, Any]]] = {}
    for name in (
        "translations.jsonl",
        "translation_reviews.jsonl",
        "verified_distractors.jsonl",
        "perturbations.jsonl",
    ):
        _, artifact_rows = _validate_file_binding(
            upstream_bindings.get(name),
            owner_path=path,
            label=f"review input upstream artifact {name}",
            expected_path=run_manifest_path.parent / name,
            jsonl=True,
        )
        upstream_rows[name] = artifact_rows
    _, rows = _validate_file_binding(
        manifest.get("review_input"),
        owner_path=path,
        label="Codex review input",
        expected_schema=REVIEW_INPUT_RECORD_SCHEMA,
        jsonl=True,
    )
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != REVIEW_INPUT_RECORD_SCHEMA:
            raise ValueError(f"review input row {index} schema_version is unsupported")
        base_fact_id = _required_string(row.get("base_fact_id"), "review input base_fact_id")
        if base_fact_id in by_id:
            raise ValueError(f"duplicate review input base_fact_id: {base_fact_id}")
        staging = staging_by_id.get(base_fact_id)
        if staging is None:
            raise ValueError(f"review input references unknown staging row: {base_fact_id}")
        if row.get("staging_record_sha256") != sha256_value(staging):
            raise ValueError(f"review input has stale staging row hash: {base_fact_id}")
        variants = row.get("variants")
        if not isinstance(variants, list) or len(variants) != 2:
            raise ValueError(f"review input must contain two variants: {base_fact_id}")
        by_id[base_fact_id] = row
    if len(rows) != len(staging_rows) or set(by_id) != set(staging_by_id):
        raise ValueError("review input does not exactly cover staging rows")
    if manifest.get("base_fact_count") != len(rows):
        raise ValueError("review input manifest base_fact_count is stale")
    if manifest.get("variant_count") != sum(
        len(row["variants"]) for row in rows
    ):
        raise ValueError("review input manifest variant_count is stale")
    if manifest.get("context_candidate_count") != sum(
        len(variant["context_candidates"])
        for row in rows
        for variant in row["variants"]
    ):
        raise ValueError("review input manifest context_candidate_count is stale")

    distractor_by_id = _rows_by_key(
        upstream_rows["verified_distractors.jsonl"],
        "item_id",
        "review input distractor",
        require_terminal=False,
    )
    perturbation_by_id = _completed_by_key(
        upstream_rows["perturbations.jsonl"],
        "candidate_id",
        "review input context",
    )
    repair_rows: List[Dict[str, Any]] = []
    repair_binding = manifest.get("context_repair_overlay")
    if repair_binding is None:
        if manifest.get("context_repair_count") not in (None, 0):
            raise ValueError("review input context_repair_count has no overlay")
    else:
        _, repair_rows = _validate_file_binding(
            repair_binding,
            owner_path=path,
            label="review input context repair overlay",
            expected_schema=CONTEXT_REPAIR_RECORD_SCHEMA,
            jsonl=True,
        )
        if manifest.get("context_repair_count") != len(repair_rows):
            raise ValueError("review input context_repair_count is stale")
        repair_ids = [
            _required_string(
                row.get("candidate_id"), "context repair overlay candidate_id"
            )
            for row in repair_rows
        ]
        if repair_binding.get("candidate_ids_sha256") != sha256_value(
            sorted(repair_ids)
        ):
            raise ValueError("context repair overlay candidate_ids_sha256 is stale")
    repair_candidates = _validate_and_project_context_repairs(
        repair_rows,
        staging_by_id=staging_by_id,
        review_by_id=by_id,
        distractor_by_id=distractor_by_id,
        perturbation_by_id=perturbation_by_id,
    )
    _validate_review_context_provenance(
        staging_by_id=staging_by_id,
        review_by_id=by_id,
        distractor_by_id=distractor_by_id,
        perturbation_by_id=perturbation_by_id,
        repair_candidates=repair_candidates,
    )
    return manifest, rows, by_id


def _validate_revision_value(field: str, value: Any, base_fact_id: str) -> Any:
    if field in {"answer_aliases_en", "answer_aliases_zh"}:
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"invalid revision {field}: {base_fact_id}")
        normalized = [item.strip() for item in value]
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError(f"duplicate aliases in revision {field}: {base_fact_id}")
        return normalized
    return _required_string(value, f"revision {field}: {base_fact_id}")


def _validate_codex_decisions(
    decision_path: Path,
    review_by_id: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Counter[str]]:
    decisions = read_jsonl(decision_path)
    by_id: Dict[str, Dict[str, Any]] = {}
    blockers: List[str] = []
    counts: Counter[str] = Counter()
    for index, decision in enumerate(decisions, start=1):
        if decision.get("schema_version") != CODEX_DECISION_SCHEMA:
            raise ValueError(f"Codex decision row {index} schema_version is unsupported")
        base_fact_id = _required_string(decision.get("base_fact_id"), "decision base_fact_id")
        if base_fact_id in by_id:
            raise ValueError(f"duplicate Codex decision: {base_fact_id}")
        review = review_by_id.get(base_fact_id)
        if review is None:
            raise ValueError(f"Codex decision references unknown base fact: {base_fact_id}")
        if decision.get("review_input_record_sha256") != sha256_value(review):
            raise ValueError(f"Codex decision has stale review input hash: {base_fact_id}")
        if decision.get("reviewer_type") != "codex_proxy" or decision.get("human_gold") is not False:
            raise ValueError(f"Codex decision reviewer identity is invalid: {base_fact_id}")
        if decision.get("terminal_status") != "completed":
            blockers.append(f"codex_decision_not_terminal:{base_fact_id}")
        disposition = decision.get("decision")
        if disposition not in {"accept", "revise", "reject", "defer"}:
            raise ValueError(f"Codex decision disposition is invalid: {base_fact_id}")
        counts[str(disposition)] += 1
        if disposition in {"reject", "defer"}:
            blockers.append(f"codex_decision_{disposition}:{base_fact_id}")
        checks = decision.get("checks")
        if not isinstance(checks, dict) or set(checks) != set(STATIC_REVIEW_CHECKS):
            raise ValueError(f"Codex decision checks are incomplete: {base_fact_id}")
        if disposition in {"accept", "revise"} and any(
            checks.get(field) is not True for field in STATIC_REVIEW_CHECKS
        ):
            blockers.append(f"codex_static_check_failed:{base_fact_id}")
        revisions = decision.get("revisions")
        proposed_patch = decision.get("proposed_patch")
        if proposed_patch is not None:
            if not isinstance(proposed_patch, dict) or set(proposed_patch) != {"set"}:
                raise ValueError(f"proposed_patch must contain only set: {base_fact_id}")
            patch_set = proposed_patch.get("set")
            if not isinstance(patch_set, dict):
                raise ValueError(f"proposed_patch.set must be an object: {base_fact_id}")
            if revisions not in (None, {}, patch_set):
                raise ValueError(f"revisions and proposed_patch.set conflict: {base_fact_id}")
            revisions = patch_set
        if not isinstance(revisions, dict) or not set(revisions) <= REVISION_FIELDS:
            raise ValueError(f"Codex revisions contain unsupported fields: {base_fact_id}")
        if disposition == "accept" and revisions:
            raise ValueError(f"accept decision cannot contain revisions: {base_fact_id}")
        if disposition == "revise" and not revisions:
            raise ValueError(f"revise decision must contain revisions: {base_fact_id}")
        decision["revisions"] = {
            field: _validate_revision_value(field, value, base_fact_id)
            for field, value in revisions.items()
        }
        decision["proposed_patch"] = {"set": copy.deepcopy(decision["revisions"])}

        review_variants = {
            variant["distractor_id"]: variant for variant in review["variants"]
        }
        decision_variants = decision.get("variants")
        if not isinstance(decision_variants, list) or len(decision_variants) != 2:
            raise ValueError(f"Codex decision must contain two variants: {base_fact_id}")
        seen = set()
        for variant in decision_variants:
            if not isinstance(variant, dict):
                raise ValueError(f"Codex variant decision is invalid: {base_fact_id}")
            distractor_id = _required_string(
                variant.get("distractor_id"), f"Codex variant distractor_id {base_fact_id}"
            )
            if distractor_id in seen or distractor_id not in review_variants:
                raise ValueError(f"Codex variant coverage mismatch: {base_fact_id}")
            seen.add(distractor_id)
            if variant.get("decision") != "accept":
                blockers.append(f"codex_variant_not_accepted:{base_fact_id}:{distractor_id}")
            variant_checks = variant.get("checks")
            if not isinstance(variant_checks, dict) or set(variant_checks) != set(VARIANT_REVIEW_CHECKS):
                raise ValueError(f"Codex variant checks are incomplete: {base_fact_id}:{distractor_id}")
            if any(variant_checks.get(field) is not True for field in VARIANT_REVIEW_CHECKS):
                blockers.append(f"codex_variant_check_failed:{base_fact_id}:{distractor_id}")
            selected_candidate_id = _required_string(
                variant.get("selected_candidate_id"),
                f"selected_candidate_id {base_fact_id}:{distractor_id}",
            )
            candidates = {
                candidate["candidate_id"]: candidate
                for candidate in review_variants[distractor_id]["context_candidates"]
            }
            if selected_candidate_id not in candidates:
                raise ValueError(
                    f"selected context candidate is absent: {base_fact_id}:{distractor_id}"
                )
        if seen != set(review_variants):
            raise ValueError(f"Codex variant decisions do not cover both distractors: {base_fact_id}")
        by_id[base_fact_id] = decision
    missing = sorted(set(review_by_id) - set(by_id))
    extra = sorted(set(by_id) - set(review_by_id))
    blockers.extend(f"codex_decision_missing:{base_fact_id}" for base_fact_id in missing)
    if extra:
        raise ValueError(f"Codex decisions contain unknown IDs: {extra[:5]}")
    return by_id, blockers, counts


def _load_source_rows(
    staging_manifest: Mapping[str, Any], staging_manifest_path: Path
) -> Dict[str, Dict[str, Any]]:
    source_binding = staging_manifest.get("effective_source_bundle")
    source_path = _resolve_binding_path(source_binding, staging_manifest_path)
    if source_binding.get("sha256") != sha256_file(source_path):
        raise ValueError("staging effective source bundle SHA-256 is stale")
    rows = read_jsonl(source_path)
    by_id = {
        formal_admission.explicit_base_fact_id(row, "source bundle row"): row
        for row in rows
    }
    if len(by_id) != len(rows):
        raise ValueError("effective source bundle has duplicate base_fact_id values")
    return by_id


def _ordered_options(base_fact_id: str, answer_en: str, answer_zh: str, variants: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    values = [
        {"option_id": "gold", "kind": "gold", "text_en": answer_en, "text_zh": answer_zh},
        *[
            {
                "option_id": variant["distractor_id"],
                "kind": "distractor",
                "text_en": variant["text_en"],
                "text_zh": variant["text_zh"],
            }
            for variant in variants
        ],
    ]
    values.sort(key=lambda value: sha256_value([base_fact_id, "option-order-v1", value["option_id"]]))
    for label, value in zip(("A", "B", "C"), values):
        value["choice"] = label
    return values


def _behavior_inputs(
    base_fact_id: str,
    prompt_en: str,
    prompt_zh: str,
    options: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    option_rows = [dict(option) for option in options]
    result = []
    for language, prompt in (("en", prompt_en), ("zh", prompt_zh)):
        result.append(
            {
                "stimulus_id": f"{base_fact_id}:{language}:original",
                "language": language,
                "arm": "original",
                "distractor_id": None,
                "prompt": prompt,
                "context": "",
                "options": option_rows,
            }
        )
    for variant in variants:
        distractor_id = variant["distractor_id"]
        for language, prompt in (("en", prompt_en), ("zh", prompt_zh)):
            for arm in ("neutral", "targeted"):
                result.append(
                    {
                        "stimulus_id": f"{base_fact_id}:{distractor_id}:{language}:{arm}",
                        "language": language,
                        "arm": arm,
                        "distractor_id": distractor_id,
                        "prompt": prompt,
                        "context": variant[f"{arm}_context_{language}"],
                        "options": option_rows,
                    }
                )
    if len(result) != 10:
        raise AssertionError("dual-distractor static design must produce ten inputs")
    return result


def _materialize_final_rows(
    *,
    source_by_id: Mapping[str, Mapping[str, Any]],
    staging_by_id: Mapping[str, Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, Any]],
    decisions_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    def aliases_after_answer_change(
        aliases: Sequence[Any], old_answer: str, new_answer: str
    ) -> List[str]:
        retained = [
            str(alias).strip()
            for alias in aliases
            if isinstance(alias, str)
            and alias.strip()
            and alias.strip().casefold() != old_answer.casefold()
        ]
        values = [new_answer, *retained]
        result = []
        seen = set()
        for value in values:
            normalized = value.casefold()
            if normalized not in seen:
                seen.add(normalized)
                result.append(value)
        return result

    resolved_by_id: Dict[str, Dict[str, Any]] = {}
    for base_fact_id in sorted(review_by_id):
        source = source_by_id[base_fact_id]
        staging = staging_by_id[base_fact_id]
        review = review_by_id[base_fact_id]
        decision = decisions_by_id[base_fact_id]
        if staging.get("effective_source_record_sha256") != sha256_value(source):
            raise ValueError(f"staging effective source hash is stale: {base_fact_id}")
        translation = review["translation"]
        old_answer_en = _required_string(staging.get("answer_en"), f"answer_en {base_fact_id}")
        old_answer_zh = _required_string(translation.get("answer_zh"), f"answer_zh {base_fact_id}")
        values = {
            "subject_en": staging.get("subject_en"),
            "subject_zh": translation.get("subject_zh"),
            "prompt_en": staging.get("prompt_en"),
            "canonical_fact_en": staging.get("canonical_fact_en"),
            "answer_en": old_answer_en,
            "answer_aliases_en": list(staging.get("answer_aliases_en") or []),
            "prompt_zh": translation.get("prompt_zh"),
            "canonical_fact_zh": translation.get("canonical_fact_zh"),
            "answer_zh": old_answer_zh,
            "answer_aliases_zh": list(translation.get("answer_aliases_zh") or []),
        }
        revisions = decision.get("revisions") or {}
        values.update(revisions)
        values["prompt_en"] = _required_string(values["prompt_en"], f"final prompt_en {base_fact_id}")
        values["prompt_zh"] = _required_string(values["prompt_zh"], f"final prompt_zh {base_fact_id}")
        values["canonical_fact_en"] = _required_string(
            values["canonical_fact_en"], f"final canonical_fact_en {base_fact_id}"
        )
        values["answer_en"] = _required_string(values["answer_en"], f"final answer_en {base_fact_id}")
        values["answer_zh"] = _required_string(values["answer_zh"], f"final answer_zh {base_fact_id}")
        if values["answer_en"] != old_answer_en and "answer_aliases_en" not in revisions:
            values["answer_aliases_en"] = aliases_after_answer_change(
                values["answer_aliases_en"], old_answer_en, values["answer_en"]
            )
        if values["answer_zh"] != old_answer_zh and "answer_aliases_zh" not in revisions:
            values["answer_aliases_zh"] = aliases_after_answer_change(
                values["answer_aliases_zh"], old_answer_zh, values["answer_zh"]
            )
        resolved_by_id[base_fact_id] = {
            **values,
            "old_answer_en": old_answer_en,
            "old_answer_zh": old_answer_zh,
            "answer_revised": (
                values["answer_en"] != old_answer_en or values["answer_zh"] != old_answer_zh
            ),
        }

    final_rows = []
    for base_fact_id in sorted(review_by_id):
        source = source_by_id[base_fact_id]
        staging = staging_by_id[base_fact_id]
        review = review_by_id[base_fact_id]
        decision = decisions_by_id[base_fact_id]
        values = resolved_by_id[base_fact_id]
        prompt_en = values["prompt_en"]
        prompt_zh = values["prompt_zh"]
        canonical_fact_en = values["canonical_fact_en"]
        answer_en = values["answer_en"]
        answer_zh = values["answer_zh"]
        review_variants = {row["distractor_id"]: row for row in review["variants"]}
        selected_variants = []
        for variant_decision in decision["variants"]:
            distractor_id = variant_decision["distractor_id"]
            reviewed = review_variants[distractor_id]
            candidate = next(
                item
                for item in reviewed["context_candidates"]
                if item["candidate_id"] == variant_decision["selected_candidate_id"]
            )
            donor_base_fact_id = reviewed.get("donor_base_fact_id")
            donor = resolved_by_id.get(str(donor_base_fact_id))
            old_text_en = _required_string(reviewed.get("text_en"), f"distractor_en {distractor_id}")
            old_text_zh = _required_string(reviewed.get("text_zh"), f"distractor_zh {distractor_id}")
            text_en = donor["answer_en"] if donor is not None else old_text_en
            text_zh = donor["answer_zh"] if donor is not None else old_text_zh
            aliases_en = (
                list(donor["answer_aliases_en"])
                if donor is not None
                else list(reviewed.get("answer_aliases_en") or [text_en])
            )
            aliases_zh = (
                list(donor["answer_aliases_zh"])
                if donor is not None
                else [text_zh]
            )
            propagated = bool(donor and donor["answer_revised"])
            contexts = {
                "targeted_context_en": candidate["targeted_context_en"],
                "targeted_context_zh": candidate["targeted_context_zh"],
                "neutral_context_en": candidate["neutral_context_en"],
                "neutral_context_zh": candidate["neutral_context_zh"],
            }
            if propagated:
                contexts = {
                    key: value.replace(old_text_en, text_en).replace(old_text_zh, text_zh)
                    for key, value in contexts.items()
                }
            selected_variants.append(
                {
                    "distractor_id": distractor_id,
                    "donor_base_fact_id": donor_base_fact_id,
                    "selected_candidate_id": candidate["candidate_id"],
                    "text_en": text_en,
                    "text_zh": text_zh,
                    "answer_aliases_en": aliases_en,
                    "answer_aliases_zh": aliases_zh,
                    **contexts,
                    "donor_answer_revision_propagated": propagated,
                    "context_candidate_record_sha256": candidate["candidate_record_sha256"],
                    "review_decision_sha256": sha256_value(variant_decision),
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                    "verified": True,
                    "verification_status": "codex_adjudicated",
                }
            )
        selected_variants.sort(key=lambda row: row["distractor_id"])
        target_aliases = {
            str(value).strip().casefold()
            for value in [answer_en, *(values["answer_aliases_en"] or [])]
            if str(value).strip()
        }
        for variant in selected_variants:
            distractor_aliases = {
                str(value).strip().casefold()
                for value in [variant["text_en"], *variant["answer_aliases_en"]]
                if str(value).strip()
            }
            if target_aliases & distractor_aliases:
                raise ValueError(
                    f"revised donor makes distractor overlap target answer: {base_fact_id}:{variant['distractor_id']}"
                )
        options = _ordered_options(base_fact_id, answer_en, answer_zh, selected_variants)
        static_mcq = {
            "schema_version": "static-g0a-dual-distractor-mcq-v1",
            "manipulation_family": MANIPULATION_FAMILY,
            "option_order_policy": "base-fact-option-sha256-v1",
            "options": options,
            "variants": selected_variants,
        }
        static_mcq["behavior_inputs"] = _behavior_inputs(
            base_fact_id, prompt_en, prompt_zh, options, selected_variants
        )
        row = copy.deepcopy(dict(source))
        row.update(
            {
                "schema_version": FINAL_BUNDLE_SCHEMA,
                "bundle_status": "reviewed_frozen",
                "canonical_freeze_status": "frozen",
                "canonical_status": "frozen",
                "split_status": "frozen",
                "evidence_tier": "independent_adjudicated",
                "human_gold": False,
                "subject_en": values.get("subject_en"),
                "subject_zh": values.get("subject_zh"),
                "prompt_en": prompt_en,
                "prompt_zh": prompt_zh,
                "canonical_fact": canonical_fact_en,
                "canonical_fact_en": canonical_fact_en,
                "canonical_fact_zh": values.get("canonical_fact_zh"),
                "answer_en": answer_en,
                "answer_zh": answer_zh,
                "answer_aliases_en": values["answer_aliases_en"],
                "answer_aliases_zh": values["answer_aliases_zh"],
                "translation_status": "codex_adjudicated_frozen",
                "distractor_candidates": [
                    {
                        key: variant[key]
                        for key in (
                            "distractor_id",
                            "donor_base_fact_id",
                            "text_en",
                            "text_zh",
                            "answer_aliases_en",
                            "answer_aliases_zh",
                            "donor_answer_revision_propagated",
                            "reviewer_type",
                            "human_gold",
                            "verified",
                            "verification_status",
                        )
                    }
                    for variant in selected_variants
                ],
                "distractor_candidates_zh": [
                    {"distractor_id": variant["distractor_id"], "text_zh": variant["text_zh"]}
                    for variant in selected_variants
                ],
                "static_mcq": static_mcq,
                "static_stimulus_status": "frozen",
                "static_generation_complete": True,
                "behavior_authorized": False,
                "simulation_authorized": False,
                "perturbation_ready": True,
                "perturbation_status": "static_frozen_not_run",
                "path_not_token_experiment_ready": False,
                "codex_adjudication": {
                    "decision": decision["decision"],
                    "decision_record_sha256": sha256_value(decision),
                    "review_input_record_sha256": decision["review_input_record_sha256"],
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                    "revision_fields": sorted((decision.get("revisions") or {}).keys()),
                },
                "admission": {
                    **(row.get("admission") or {}),
                    "prompt_ready": True,
                    "semantic_review_complete": True,
                    "behavior_screening_ready": False,
                    "hf_bridge_ready": False,
                    "path_not_token_experiment_ready": False,
                    "blocking_reasons": [
                        "proxy_behavior_run_not_authorized",
                        "exact_hf_runtime_not_frozen",
                    ],
                },
            }
        )
        final_rows.append(row)
    return final_rows


def _load_resolved_core(
    *,
    staging_manifest_path: Path,
    review_input_manifest_path: Path,
    codex_decisions_path: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Counter[str],
    List[Dict[str, Any]],
]:
    staging_manifest_path = Path(staging_manifest_path).resolve()
    review_input_manifest_path = Path(review_input_manifest_path).resolve()
    codex_decisions_path = Path(codex_decisions_path).resolve()
    staging_manifest, staging_rows, staging_by_id = _load_staging(staging_manifest_path)
    review_manifest, review_rows, review_by_id = _load_review_input(
        review_input_manifest_path, staging_manifest_path
    )
    if set(review_by_id) != set(staging_by_id):
        raise GateBlocked(["codex_review_input_does_not_cover_staging_universe"])
    decisions_by_id, decision_blockers, decision_counts = _validate_codex_decisions(
        codex_decisions_path, review_by_id
    )
    if decision_blockers:
        raise GateBlocked(decision_blockers)
    source_by_id = _load_source_rows(staging_manifest, staging_manifest_path)
    if set(source_by_id) != set(staging_by_id):
        raise GateBlocked(["effective_source_bundle_does_not_cover_staging_universe"])
    final_rows = _materialize_final_rows(
        source_by_id=source_by_id,
        staging_by_id=staging_by_id,
        review_by_id=review_by_id,
        decisions_by_id=decisions_by_id,
    )
    return (
        staging_manifest,
        staging_rows,
        staging_by_id,
        review_manifest,
        review_rows,
        review_by_id,
        decisions_by_id,
        decision_counts,
        final_rows,
    )


def resolve_static_g0a_draft(
    *,
    staging_manifest_path: Path,
    review_input_manifest_path: Path,
    codex_decisions_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Materialize the exact post-Codex rows that closure must adjudicate."""

    (
        staging_manifest,
        _,
        _,
        _,
        _,
        _,
        _,
        decision_counts,
        final_rows,
    ) = _load_resolved_core(
        staging_manifest_path=staging_manifest_path,
        review_input_manifest_path=review_input_manifest_path,
        codex_decisions_path=codex_decisions_path,
    )
    output_dir = _prepare_empty_output_dir(
        Path(output_dir), "resolved static G0A draft directory"
    )
    rows_path = output_dir / "resolved_static_draft.jsonl"
    manifest_path = output_dir / "resolved_static_draft_manifest.json"
    write_jsonl(rows_path, final_rows)
    ids = [
        formal_admission.explicit_base_fact_id(row, "resolved static draft row")
        for row in final_rows
    ]
    rows_binding = _binding(
        rows_path,
        schema_version=FINAL_BUNDLE_SCHEMA,
        record_count=len(final_rows),
        id_values=ids,
        id_digest_field="base_fact_ids_sha256",
    )
    manifest = {
        "schema_version": RESOLVED_DRAFT_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "resolved_static_draft_not_frozen",
        "source_bundle": copy.deepcopy(staging_manifest["source_bundle"]),
        "effective_source_bundle": copy.deepcopy(
            staging_manifest["effective_source_bundle"]
        ),
        "effective_staging_bundle": copy.deepcopy(
            staging_manifest["effective_staging_bundle"]
        ),
        "staging_manifest": _binding(
            Path(staging_manifest_path), schema_version=STAGING_MANIFEST_SCHEMA
        ),
        "review_input_manifest": _binding(
            Path(review_input_manifest_path), schema_version=REVIEW_INPUT_MANIFEST_SCHEMA
        ),
        "codex_decisions": _binding(
            Path(codex_decisions_path),
            schema_version=CODEX_DECISION_SCHEMA,
            record_count=len(final_rows),
            id_values=ids,
            id_digest_field="base_fact_ids_sha256",
        ),
        "decision_counts": dict(decision_counts),
        "resolved_rows": rows_binding,
        "base_fact_count": len(final_rows),
        "static_frozen": False,
        "behavior_authorized": False,
        "simulation_authorized": False,
        "next_gate": "semantic_closure_and_historical_exposure_on_resolved_rows",
    }
    write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "resolved_static_draft_manifest": str(manifest_path),
        "resolved_static_draft": str(rows_path),
        "base_fact_count": len(final_rows),
        "behavior_authorized": False,
    }


def _validate_resolved_static_draft(
    *,
    manifest_path: Path,
    staging_manifest_path: Path,
    review_input_manifest_path: Path,
    codex_decisions_path: Path,
    staging_manifest: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != RESOLVED_DRAFT_MANIFEST_SCHEMA:
        raise ValueError("resolved static draft manifest schema_version is unsupported")
    if manifest.get("status") != "resolved_static_draft_not_frozen":
        raise ValueError("resolved static draft manifest status is invalid")
    for field in ("static_frozen", "behavior_authorized", "simulation_authorized"):
        if manifest.get(field) is not False:
            raise ValueError(f"resolved static draft manifest must keep {field}=false")
    _validate_file_binding(
        manifest.get("staging_manifest"),
        owner_path=manifest_path,
        label="resolved draft staging manifest",
        expected_path=staging_manifest_path,
        expected_schema=STAGING_MANIFEST_SCHEMA,
    )
    _validate_file_binding(
        manifest.get("review_input_manifest"),
        owner_path=manifest_path,
        label="resolved draft review input manifest",
        expected_path=review_input_manifest_path,
        expected_schema=REVIEW_INPUT_MANIFEST_SCHEMA,
    )
    _validate_file_binding(
        manifest.get("codex_decisions"),
        owner_path=manifest_path,
        label="resolved draft Codex decisions",
        expected_path=codex_decisions_path,
        expected_schema=CODEX_DECISION_SCHEMA,
        jsonl=True,
    )
    _validate_binding_identity(
        manifest.get("source_bundle"),
        staging_manifest["source_bundle"],
        "resolved draft source_bundle",
    )
    _validate_binding_identity(
        manifest.get("effective_source_bundle"),
        staging_manifest["effective_source_bundle"],
        "resolved draft effective_source_bundle",
    )
    _validate_binding_identity(
        manifest.get("effective_staging_bundle"),
        staging_manifest["effective_staging_bundle"],
        "resolved draft effective_staging_bundle",
    )
    rows_path, rows = _validate_file_binding(
        manifest.get("resolved_rows"),
        owner_path=manifest_path,
        label="resolved static draft rows",
        expected_schema=FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    if rows != list(final_rows):
        raise ValueError("resolved static draft rows differ from recomputed final rows")
    ids = sorted(
        formal_admission.explicit_base_fact_id(row, "resolved static draft row")
        for row in rows
    )
    if manifest["resolved_rows"].get("base_fact_ids_sha256") != sha256_value(ids):
        raise ValueError("resolved static draft base_fact_ids_sha256 is stale")
    if manifest.get("base_fact_count") != len(rows):
        raise ValueError("resolved static draft base_fact_count is stale")
    return rows_path, manifest, copy.deepcopy(manifest["resolved_rows"])


def _validate_semantic_closure_evidence(
    *,
    manifest_path: Path,
    source_binding: Mapping[str, Any],
    effective_staging_binding: Mapping[str, Any],
    resolved_rows_binding: Mapping[str, Any],
    staging_by_id: Mapping[str, Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SEMANTIC_CLOSURE_MANIFEST_SCHEMA:
        raise ValueError("unsupported semantic closure manifest schema_version")
    _validate_source_binding(manifest.get("input_bundle"), source_binding, "semantic closure")
    _validate_binding_identity(
        manifest.get("effective_staging_bundle"),
        effective_staging_binding,
        "semantic closure effective_staging_bundle",
    )
    _validate_binding_identity(
        manifest.get("resolved_static_draft"),
        resolved_rows_binding,
        "semantic closure resolved_static_draft",
    )
    if manifest.get("review_blinded_to_behavior_results") is not True:
        raise ValueError("semantic closure is not behavior blind")
    if manifest.get("behavior_result_count") != 0:
        raise ValueError("semantic closure contains behavior exposure")
    for field in (
        "semantic_paraphrase_closure_complete",
        "candidate_generation_complete",
        "cohort_scope_complete",
        "out_of_cohort_bridge_search_complete",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"semantic closure {field} must be true")
    if manifest.get("status") != "complete":
        raise ValueError("semantic closure status must be complete")
    if manifest.get("unresolved_candidate_count") != 0:
        raise ValueError("semantic closure has unresolved candidates")
    policy = _required_string(manifest.get("policy_version"), "semantic closure policy_version")
    _, candidates = _validate_file_binding(
        manifest.get("candidates"),
        owner_path=manifest_path,
        label="semantic closure candidates",
        expected_schema=SEMANTIC_CANDIDATE_SCHEMA,
        jsonl=True,
    )
    _, decisions = _validate_file_binding(
        manifest.get("adjudications"),
        owner_path=manifest_path,
        label="semantic closure adjudications",
        expected_schema=SEMANTIC_DECISION_SCHEMA,
        jsonl=True,
    )
    if not candidates:
        raise ValueError("semantic closure candidates cannot be an evidence-free empty artifact")
    final_by_id = {
        formal_admission.explicit_base_fact_id(row, "final row"): row
        for row in final_rows
    }
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    pair_by_id: Dict[str, Tuple[str, str]] = {}
    represented_pairs = set()
    for index, row in enumerate(candidates, start=1):
        if row.get("schema_version") != SEMANTIC_CANDIDATE_SCHEMA:
            raise ValueError(f"semantic candidate row {index} schema_version is unsupported")
        candidate_id = _required_string(row.get("candidate_id"), "semantic candidate_id")
        if candidate_id in candidate_by_id:
            raise ValueError(f"duplicate semantic candidate_id: {candidate_id}")
        left = _required_string(row.get("left_base_fact_id"), f"{candidate_id} left_base_fact_id")
        right = _required_string(row.get("right_base_fact_id"), f"{candidate_id} right_base_fact_id")
        pair = tuple(sorted((left, right)))
        if left == right or not set(pair) <= set(staging_by_id):
            raise ValueError(f"semantic candidate endpoints are invalid: {candidate_id}")
        if pair in represented_pairs:
            raise ValueError(f"duplicate semantic candidate pair: {pair}")
        if row.get("left_source_record_sha256") != staging_by_id[left]["source_record_sha256"]:
            raise ValueError(f"semantic candidate has stale left source hash: {candidate_id}")
        if row.get("right_source_record_sha256") != staging_by_id[right]["source_record_sha256"]:
            raise ValueError(f"semantic candidate has stale right source hash: {candidate_id}")
        if row.get("left_effective_source_record_sha256") != staging_by_id[left][
            "effective_source_record_sha256"
        ]:
            raise ValueError(f"semantic candidate has stale left effective source hash: {candidate_id}")
        if row.get("right_effective_source_record_sha256") != staging_by_id[right][
            "effective_source_record_sha256"
        ]:
            raise ValueError(f"semantic candidate has stale right effective source hash: {candidate_id}")
        if row.get("left_input_record_sha256") != formal_admission.sha256_value(
            final_by_id[left]
        ):
            raise ValueError(f"semantic candidate has stale left resolved row hash: {candidate_id}")
        if row.get("right_input_record_sha256") != formal_admission.sha256_value(
            final_by_id[right]
        ):
            raise ValueError(f"semantic candidate has stale right resolved row hash: {candidate_id}")
        candidate_by_id[candidate_id] = row
        pair_by_id[candidate_id] = pair
        represented_pairs.add(pair)
    decision_by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(decisions, start=1):
        if row.get("schema_version") != SEMANTIC_DECISION_SCHEMA:
            raise ValueError(f"semantic decision row {index} schema_version is unsupported")
        candidate_id = _required_string(row.get("candidate_id"), "semantic decision candidate_id")
        if candidate_id in decision_by_id or candidate_id not in candidate_by_id:
            raise ValueError(f"semantic decision coverage is invalid: {candidate_id}")
        if row.get("candidate_record_sha256") != sha256_value(candidate_by_id[candidate_id]):
            raise ValueError(f"semantic decision has stale candidate hash: {candidate_id}")
        if row.get("terminal_status") != "completed":
            raise ValueError(f"semantic decision is not terminal: {candidate_id}")
        if row.get("decision") not in {"same_fact", "same_leakage_component", "distinct"}:
            raise ValueError(f"semantic decision is unresolved: {candidate_id}")
        if row.get("reviewer_type") not in {"human", "codex_proxy"}:
            raise ValueError(f"semantic reviewer_type is invalid: {candidate_id}")
        if row.get("reviewer_type") == "codex_proxy" and row.get("human_gold") is not False:
            raise ValueError(f"semantic codex decision must have human_gold=false: {candidate_id}")
        decision_by_id[candidate_id] = row
    if set(decision_by_id) != set(candidate_by_id):
        raise ValueError("semantic adjudications do not exactly cover candidates")
    mechanical_pairs = formal_admission._mechanical_cross_group_risk_pairs(final_rows)
    missing_mechanical = mechanical_pairs - represented_pairs
    if missing_mechanical:
        raise ValueError(
            "semantic closure omits mechanical pairs: "
            + ",".join(f"{left}|{right}" for left, right in sorted(missing_mechanical)[:5])
        )
    formal_candidates = []
    formal_decisions = []
    mechanical_reasons = formal_admission._mechanical_cross_group_risk_reasons(final_rows)
    for candidate_id in sorted(candidate_by_id):
        evidence = candidate_by_id[candidate_id]
        decision = decision_by_id[candidate_id]
        left, right = pair_by_id[candidate_id]
        formal_candidate = {
            "schema_version": formal_admission.NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "left_base_fact_id": left,
            "right_base_fact_id": right,
            "left_candidate_id": final_by_id[left]["candidate_id"],
            "right_candidate_id": final_by_id[right]["candidate_id"],
            "left_input_record_sha256": formal_admission.sha256_value(final_by_id[left]),
            "right_input_record_sha256": formal_admission.sha256_value(final_by_id[right]),
            "semantic_evidence_record_sha256": sha256_value(evidence),
            "semantic_policy_version": policy,
        }
        relationship = decision["decision"]
        formal_decision_value = "same_fact" if relationship == "same_fact" else "distinct"
        exact_reasons = mechanical_reasons.get((left, right), set()) & {
            "canonical_fact_exact",
            "prompt_answer_exact",
        }
        if formal_decision_value == "distinct" and exact_reasons:
            raise ValueError(f"exact duplicate was not adjudicated same_fact: {candidate_id}")
        if relationship in {"same_fact", "same_leakage_component"}:
            left_row, right_row = final_by_id[left], final_by_id[right]
            if (
                left_row.get("split_group_id") != right_row.get("split_group_id")
                or left_row.get("split_assignment") != right_row.get("split_assignment")
            ):
                raise ValueError(
                    f"{relationship} pair crosses split groups: {candidate_id}"
                )
        formal_decision = {
            "schema_version": formal_admission.NEAR_DUPLICATE_DECISION_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "candidate_record_sha256": formal_admission.sha256_value(formal_candidate),
            "left_candidate_id": formal_candidate["left_candidate_id"],
            "right_candidate_id": formal_candidate["right_candidate_id"],
            "left_input_record_sha256": formal_candidate["left_input_record_sha256"],
            "right_input_record_sha256": formal_candidate["right_input_record_sha256"],
            "decision": formal_decision_value,
            "semantic_relationship": relationship,
            "semantic_decision_record_sha256": sha256_value(decision),
            "reviewer_type": decision["reviewer_type"],
            "human_gold": decision.get("human_gold", False),
        }
        formal_candidates.append(formal_candidate)
        formal_decisions.append(formal_decision)
    provenance = {
        "manifest": _binding(manifest_path, schema_version=SEMANTIC_CLOSURE_MANIFEST_SCHEMA),
        "policy_version": policy,
        "candidate_count": len(formal_candidates),
        "effective_staging_bundle_sha256": effective_staging_binding["sha256"],
        "resolved_static_draft_sha256": resolved_rows_binding["sha256"],
    }
    return formal_candidates, formal_decisions, provenance


def _validate_exposure_evidence(
    *,
    manifest_path: Path,
    source_binding: Mapping[str, Any],
    effective_staging_binding: Mapping[str, Any],
    resolved_rows_binding: Mapping[str, Any],
    staging_by_id: Mapping[str, Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != EXPOSURE_MANIFEST_SCHEMA:
        raise ValueError("unsupported historical exposure manifest schema_version")
    _validate_source_binding(manifest.get("input_bundle"), source_binding, "historical exposure")
    _validate_binding_identity(
        manifest.get("effective_staging_bundle"),
        effective_staging_binding,
        "historical exposure effective_staging_bundle",
    )
    _validate_binding_identity(
        manifest.get("resolved_static_draft"),
        resolved_rows_binding,
        "historical exposure resolved_static_draft",
    )
    if manifest.get("review_blinded_to_behavior_results") is not True:
        raise ValueError("historical exposure review is not behavior blind")
    if manifest.get("behavior_result_count") != 0:
        raise ValueError("historical exposure evidence contains behavior exposure")
    if manifest.get("status") != "complete" or manifest.get("historical_exposure_complete") is not True:
        raise ValueError("historical exposure evidence is incomplete")
    policy = _required_string(manifest.get("policy_version"), "historical exposure policy_version")
    _, checks = _validate_file_binding(
        manifest.get("checks"),
        owner_path=manifest_path,
        label="historical exposure checks",
        expected_schema=EXPOSURE_CHECK_SCHEMA,
        jsonl=True,
    )
    _, registry = _validate_file_binding(
        manifest.get("registry"),
        owner_path=manifest_path,
        label="historical exposure registry",
        expected_schema=EXPOSURE_REGISTRY_SCHEMA,
        jsonl=True,
        allow_empty=True,
    )
    final_by_id = {
        formal_admission.explicit_base_fact_id(row, "final row"): row for row in final_rows
    }
    sensitive_ids = {
        base_fact_id
        for base_fact_id, row in final_by_id.items()
        if row.get("split_assignment") in {"validation", "sealed"}
    }
    check_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(checks, start=1):
        if row.get("schema_version") != EXPOSURE_CHECK_SCHEMA:
            raise ValueError(f"historical exposure check row {index} schema_version is unsupported")
        base_fact_id = _required_string(row.get("base_fact_id"), "exposure check base_fact_id")
        if base_fact_id in check_by_id or base_fact_id not in sensitive_ids:
            raise ValueError(f"historical exposure check coverage is invalid: {base_fact_id}")
        if row.get("source_record_sha256") != staging_by_id[base_fact_id]["source_record_sha256"]:
            raise ValueError(f"historical exposure check has stale source hash: {base_fact_id}")
        if row.get("effective_source_record_sha256") != staging_by_id[base_fact_id][
            "effective_source_record_sha256"
        ]:
            raise ValueError(f"historical exposure check has stale effective source hash: {base_fact_id}")
        if row.get("input_record_sha256") != formal_admission.sha256_value(
            final_by_id[base_fact_id]
        ):
            raise ValueError(f"historical exposure check has stale resolved row hash: {base_fact_id}")
        if row.get("terminal_status") != "completed":
            raise ValueError(f"historical exposure check is not terminal: {base_fact_id}")
        if row.get("split_assignment") != final_by_id[base_fact_id].get("split_assignment"):
            raise ValueError(f"historical exposure check has wrong split: {base_fact_id}")
        if row.get("historically_exposed") is not False:
            raise ValueError(f"historical exposure is present or unresolved: {base_fact_id}")
        checked_sources = row.get("checked_sources")
        if not isinstance(checked_sources, list) or not checked_sources or any(
            not isinstance(item, str) or not item.strip() for item in checked_sources
        ):
            raise ValueError(f"historical exposure check has no concrete sources: {base_fact_id}")
        if not _is_sha256(row.get("query_fingerprint_sha256")):
            raise ValueError(f"historical exposure check has no query fingerprint: {base_fact_id}")
        _required_string(row.get("checked_at"), f"historical exposure checked_at {base_fact_id}")
        check_by_id[base_fact_id] = row
    if set(check_by_id) != sensitive_ids:
        missing = sorted(sensitive_ids - set(check_by_id))
        raise ValueError(f"historical exposure checks do not cover holdouts: {missing[:5]}")
    formal_registry = []
    exposed_ids = set()
    for index, row in enumerate(registry, start=1):
        if row.get("schema_version") != EXPOSURE_REGISTRY_SCHEMA:
            raise ValueError(f"historical exposure registry row {index} schema_version is unsupported")
        exposure_id = _required_string(row.get("exposure_id"), "exposure_id")
        base_fact_id = _required_string(row.get("base_fact_id"), f"{exposure_id} base_fact_id")
        if base_fact_id not in final_by_id:
            raise ValueError(f"historical exposure registry references unknown fact: {exposure_id}")
        exposed_ids.add(base_fact_id)
        formal_registry.append(
            {
                **row,
                "schema_version": formal_admission.HISTORICAL_EXPOSURE_REGISTRY_SCHEMA_VERSION,
            }
        )
    if exposed_ids & sensitive_ids:
        raise ValueError("historical exposure registry contains validation or sealed facts")
    if not registry and manifest.get("zero_registry_supported_by_per_fact_checks") is not True:
        raise ValueError("empty historical registry lacks explicit per-fact evidence contract")
    formal_checks = [
        {
            "schema_version": formal_admission.HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION,
            "base_fact_id": base_fact_id,
            "input_record_sha256": formal_admission.sha256_value(final_by_id[base_fact_id]),
            "split_assignment": final_by_id[base_fact_id]["split_assignment"],
            "historically_exposed": False,
            "source_check_record_sha256": sha256_value(check_by_id[base_fact_id]),
        }
        for base_fact_id in sorted(sensitive_ids)
    ]
    provenance = {
        "manifest": _binding(manifest_path, schema_version=EXPOSURE_MANIFEST_SCHEMA),
        "policy_version": policy,
        "check_count": len(formal_checks),
        "registry_count": len(formal_registry),
        "effective_staging_bundle_sha256": effective_staging_binding["sha256"],
        "resolved_static_draft_sha256": resolved_rows_binding["sha256"],
    }
    return formal_registry, formal_checks, provenance


def _portable_binding(
    path: Path,
    *,
    schema_version: str,
    record_count: int,
    ids: Sequence[str],
    id_digest_field: str,
) -> Dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "schema_version": schema_version,
        "record_count": record_count,
        id_digest_field: formal_admission.sha256_value(sorted(ids)),
    }


def _blocker_result(
    output_dir: Path,
    blockers: Sequence[str],
    *,
    evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "static-g0a-finalize-blockers-v1",
        "tool_version": TOOL_VERSION,
        "created_at": utc_now(),
        "status": "blocked_fail_closed",
        "static_frozen": False,
        "behavior_authorized": False,
        "frozen_artifacts_emitted": False,
        "blockers": sorted(set(blockers)),
        "evidence": dict(evidence or {}),
    }
    report_path = output_dir / "g0a_finalize_blockers.json"
    write_json(report_path, report)
    return {**report, "blocker_report": str(report_path)}


def finalize_static_g0a(
    *,
    staging_manifest_path: Path,
    review_input_manifest_path: Path,
    codex_decisions_path: Path,
    output_dir: Path,
    resolved_static_draft_manifest_path: Optional[Path] = None,
    semantic_closure_manifest_path: Optional[Path] = None,
    historical_exposure_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    staging_manifest_path = Path(staging_manifest_path).resolve()
    review_input_manifest_path = Path(review_input_manifest_path).resolve()
    codex_decisions_path = Path(codex_decisions_path).resolve()
    blockers: List[str] = []
    evidence: Dict[str, Any] = {}
    try:
        (
            staging_manifest,
            staging_rows,
            staging_by_id,
            review_manifest,
            review_rows,
            review_by_id,
            decisions_by_id,
            decision_counts,
            final_rows,
        ) = _load_resolved_core(
            staging_manifest_path=staging_manifest_path,
            review_input_manifest_path=review_input_manifest_path,
            codex_decisions_path=codex_decisions_path,
        )
        evidence.update(
            {
                "staging_manifest": _binding(staging_manifest_path, schema_version=STAGING_MANIFEST_SCHEMA),
                "review_input_manifest": _binding(
                    review_input_manifest_path, schema_version=REVIEW_INPUT_MANIFEST_SCHEMA
                ),
                "codex_decisions": _binding(
                    codex_decisions_path,
                    schema_version=CODEX_DECISION_SCHEMA,
                    record_count=len(decisions_by_id),
                ),
                "codex_decision_counts": dict(decision_counts),
            }
        )
    except GateBlocked as exc:
        return _blocker_result(output_dir, exc.blockers, evidence=evidence)
    except (ValueError, FileNotFoundError) as exc:
        return _blocker_result(
            output_dir,
            [f"core_static_evidence_invalid:{type(exc).__name__}:{exc}"],
            evidence=evidence,
        )
    resolved_rows_binding: Optional[Dict[str, Any]] = None
    if resolved_static_draft_manifest_path is None:
        blockers.append("resolved_static_draft_manifest_missing")
    else:
        try:
            _, _, resolved_rows_binding = _validate_resolved_static_draft(
                manifest_path=Path(resolved_static_draft_manifest_path),
                staging_manifest_path=staging_manifest_path,
                review_input_manifest_path=review_input_manifest_path,
                codex_decisions_path=codex_decisions_path,
                staging_manifest=staging_manifest,
                final_rows=final_rows,
            )
            evidence["resolved_static_draft"] = _binding(
                Path(resolved_static_draft_manifest_path),
                schema_version=RESOLVED_DRAFT_MANIFEST_SCHEMA,
            )
        except (ValueError, FileNotFoundError) as exc:
            blockers.append(
                f"resolved_static_draft_invalid:{type(exc).__name__}:{exc}"
            )

    source_binding = staging_manifest["source_bundle"]
    effective_staging_binding = staging_manifest["effective_staging_bundle"]
    if semantic_closure_manifest_path is None:
        blockers.append("semantic_paraphrase_closure_evidence_missing")
    elif resolved_rows_binding is None:
        blockers.append("semantic_paraphrase_closure_blocked_by_unresolved_draft")
    else:
        try:
            formal_candidates, formal_near_decisions, semantic_provenance = (
                _validate_semantic_closure_evidence(
                    manifest_path=Path(semantic_closure_manifest_path),
                    source_binding=source_binding,
                    effective_staging_binding=effective_staging_binding,
                    resolved_rows_binding=resolved_rows_binding,
                    staging_by_id=staging_by_id,
                    final_rows=final_rows,
                )
            )
            evidence["semantic_closure"] = semantic_provenance
        except (ValueError, FileNotFoundError) as exc:
            blockers.append(f"semantic_paraphrase_closure_evidence_invalid:{type(exc).__name__}:{exc}")
    if historical_exposure_manifest_path is None:
        blockers.append("historical_exposure_evidence_missing")
    elif resolved_rows_binding is None:
        blockers.append("historical_exposure_blocked_by_unresolved_draft")
    else:
        try:
            formal_registry, formal_exposure_checks, exposure_provenance = (
                _validate_exposure_evidence(
                    manifest_path=Path(historical_exposure_manifest_path),
                    source_binding=source_binding,
                    effective_staging_binding=effective_staging_binding,
                    resolved_rows_binding=resolved_rows_binding,
                    staging_by_id=staging_by_id,
                    final_rows=final_rows,
                )
            )
            evidence["historical_exposure"] = exposure_provenance
        except (ValueError, FileNotFoundError) as exc:
            blockers.append(f"historical_exposure_evidence_invalid:{type(exc).__name__}:{exc}")
    if blockers:
        return _blocker_result(output_dir, blockers, evidence=evidence)

    if output_dir.exists() and any(
        (output_dir / name).exists()
        for name in (
            "frozen_static_bundle.jsonl",
            "review_freeze_manifest.json",
            "split_freeze_manifest.json",
            "static_freeze_manifest.json",
        )
    ):
        raise FileExistsError(f"refusing to overwrite existing frozen G0A artifacts: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent))
    )
    try:
        bundle_path = temporary_dir / "frozen_static_bundle.jsonl"
        scope_path = temporary_dir / "formal_review_scope.jsonl"
        review_decisions_path = temporary_dir / "formal_review_decisions.jsonl"
        near_candidates_path = temporary_dir / "near_duplicate_candidates.jsonl"
        near_decisions_path = temporary_dir / "near_duplicate_adjudications.jsonl"
        registry_path = temporary_dir / "historical_exposure_registry.jsonl"
        exposure_checks_path = temporary_dir / "historical_exposure_checks.jsonl"
        review_manifest_path = temporary_dir / "review_freeze_manifest.json"
        split_manifest_path = temporary_dir / "split_freeze_manifest.json"
        static_manifest_path = temporary_dir / "static_freeze_manifest.json"
        write_jsonl(bundle_path, final_rows)
        bundle_identity = formal_admission.bundle_identity(
            bundle_path, FINAL_BUNDLE_SCHEMA, final_rows
        )
        if bundle_identity["sha256"] != resolved_rows_binding["sha256"]:
            raise ValueError("published frozen bundle differs from resolved static draft")
        input_binding = {
            field: bundle_identity[field]
            for field in ("sha256", "schema_version", "record_count", "base_fact_ids_sha256")
        }
        final_by_id = {
            formal_admission.explicit_base_fact_id(row, "final row"): row for row in final_rows
        }
        ids = sorted(final_by_id)
        scope_rows = [
            {
                "schema_version": formal_admission.REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "input_record_sha256": formal_admission.sha256_value(final_by_id[base_fact_id]),
                "source_staging_record_sha256": sha256_value(staging_by_id[base_fact_id]),
            }
            for base_fact_id in ids
        ]
        formal_review_rows = [
            {
                "schema_version": formal_admission.REVIEW_DECISION_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "input_record_sha256": formal_admission.sha256_value(final_by_id[base_fact_id]),
                "decision": "accept",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "evidence_tier": "independent_adjudicated",
                "member_review_complete": True,
                "alias_review_complete": True,
                "distractor_review_complete": True,
                "source_codex_decision": decisions_by_id[base_fact_id]["decision"],
                "source_codex_decision_sha256": sha256_value(decisions_by_id[base_fact_id]),
            }
            for base_fact_id in ids
        ]
        write_jsonl(scope_path, scope_rows)
        write_jsonl(review_decisions_path, formal_review_rows)
        write_jsonl(near_candidates_path, formal_candidates)
        write_jsonl(near_decisions_path, formal_near_decisions)
        write_jsonl(registry_path, formal_registry)
        write_jsonl(exposure_checks_path, formal_exposure_checks)
        review_manifest = {
            "schema_version": formal_admission.REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION,
            "input_bundle": input_binding,
            "source_universe": {
                "record_count": len(ids),
                "base_fact_ids_sha256": formal_admission.sha256_value(ids),
            },
            "review_scope": _portable_binding(
                scope_path,
                schema_version=formal_admission.REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
                record_count=len(scope_rows),
                ids=ids,
                id_digest_field="base_fact_ids_sha256",
            ),
            "review_decisions": _portable_binding(
                review_decisions_path,
                schema_version=formal_admission.REVIEW_DECISION_SCHEMA_VERSION,
                record_count=len(formal_review_rows),
                ids=ids,
                id_digest_field="base_fact_ids_sha256",
            ),
            "decision_counts": {
                "accept": len(ids),
                "reject": 0,
                "defer": 0,
                "revise": 0,
                "missing": 0,
            },
            "source_codex_decision_counts": dict(decision_counts),
            "review_complete": True,
            "canonical_freeze_authorized": True,
            "reviewer_type": "codex_proxy",
            "human_gold": False,
        }
        write_json(review_manifest_path, review_manifest)
        assignments, _ = formal_admission._split_assignments(final_rows)
        split_manifest = {
            "schema_version": formal_admission.SPLIT_FREEZE_MANIFEST_SCHEMA_VERSION,
            "input_bundle": input_binding,
            "review_freeze_manifest": {
                "path": review_manifest_path.name,
                "sha256": sha256_file(review_manifest_path),
                "schema_version": formal_admission.REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION,
            },
            "split_policy_version": EXPECTED_SPLIT_POLICY,
            "split_assignments_sha256": formal_admission.sha256_value(assignments),
            "split_status": "frozen",
            "near_duplicate_policy_version": semantic_provenance["policy_version"],
            "near_duplicate_review_complete": True,
            "unresolved_near_duplicate_count": 0,
            "near_duplicate_candidates": _portable_binding(
                near_candidates_path,
                schema_version=formal_admission.NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION,
                record_count=len(formal_candidates),
                ids=[row["candidate_id"] for row in formal_candidates],
                id_digest_field="candidate_ids_sha256",
            ),
            "near_duplicate_adjudications": _portable_binding(
                near_decisions_path,
                schema_version=formal_admission.NEAR_DUPLICATE_DECISION_SCHEMA_VERSION,
                record_count=len(formal_near_decisions),
                ids=[row["candidate_id"] for row in formal_near_decisions],
                id_digest_field="candidate_ids_sha256",
            ),
            "historical_exposure_policy_version": exposure_provenance["policy_version"],
            "historical_exposure_check_complete": True,
            "historical_exposure_registry": _portable_binding(
                registry_path,
                schema_version=formal_admission.HISTORICAL_EXPOSURE_REGISTRY_SCHEMA_VERSION,
                record_count=len(formal_registry),
                ids=[row["exposure_id"] for row in formal_registry],
                id_digest_field="exposure_ids_sha256",
            ),
            "historical_exposure_checks": _portable_binding(
                exposure_checks_path,
                schema_version=formal_admission.HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION,
                record_count=len(formal_exposure_checks),
                ids=[row["base_fact_id"] for row in formal_exposure_checks],
                id_digest_field="base_fact_ids_sha256",
            ),
            "unresolved_cross_split_count": 0,
            "validation_exposure_count": 0,
            "sealed_exposure_count": 0,
        }
        write_json(split_manifest_path, split_manifest)
        validated = formal_admission.validate_external_formal_admission(
            input_bundle_path=bundle_path,
            input_schema_version=FINAL_BUNDLE_SCHEMA,
            records=final_rows,
            input_bundle_sha256=bundle_identity["sha256"],
            review_freeze_manifest_path=review_manifest_path,
            split_freeze_manifest_path=split_manifest_path,
        )
        if validated.get("authorized") is not True:
            raise ValueError(f"generated formal manifests did not authorize: {validated.get('blockers')}")
        artifact_paths = [
            bundle_path,
            scope_path,
            review_decisions_path,
            near_candidates_path,
            near_decisions_path,
            registry_path,
            exposure_checks_path,
            review_manifest_path,
            split_manifest_path,
        ]
        static_manifest = {
            "schema_version": STATIC_FREEZE_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "created_at": utc_now(),
            "status": "g0a_static_stimulus_frozen",
            "source_staging_manifest": evidence["staging_manifest"],
            "codex_review_input_manifest": evidence["review_input_manifest"],
            "codex_decisions": evidence["codex_decisions"],
            "codex_decision_counts": dict(decision_counts),
            "resolved_static_draft_evidence": evidence["resolved_static_draft"],
            "semantic_closure_evidence": semantic_provenance,
            "historical_exposure_evidence": exposure_provenance,
            "artifacts": {
                path.name: {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "byte_count": path.stat().st_size,
                }
                for path in artifact_paths
            },
            "frozen_bundle": {
                **input_binding,
                "path": bundle_path.name,
            },
            "formal_admission_fingerprint_sha256": validated[
                "admission_fingerprint_sha256"
            ],
            "base_fact_count": len(final_rows),
            "variant_count": 2 * len(final_rows),
            "unique_behavior_input_count": 10 * len(final_rows),
            "statistical_unit": "base_fact_id",
            "human_gold": False,
            "reviewer_type": "codex_proxy",
            "static_frozen": True,
            "static_generation_authorized": False,
            "behavior_authorized": False,
            "simulation_authorized": False,
            "next_gate": "explicit_proxy_behavior_run_manifest",
        }
        write_json(static_manifest_path, static_manifest)

        if output_dir.exists():
            allowed = {"g0a_finalize_blockers.json"}
            existing = {path.name for path in output_dir.iterdir()}
            if existing - allowed:
                raise FileExistsError(f"output directory is not empty: {output_dir}")
            for name in existing:
                (output_dir / name).unlink()
            output_dir.rmdir()
        os.replace(temporary_dir, output_dir)
        temporary_dir = None  # type: ignore[assignment]
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return {
        "status": "g0a_static_stimulus_frozen",
        "static_freeze_manifest": str(output_dir / "static_freeze_manifest.json"),
        "frozen_bundle": str(output_dir / "frozen_static_bundle.jsonl"),
        "review_freeze_manifest": str(output_dir / "review_freeze_manifest.json"),
        "split_freeze_manifest": str(output_dir / "split_freeze_manifest.json"),
        "base_fact_count": len(final_rows),
        "variant_count": 2 * len(final_rows),
        "behavior_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    adapt = commands.add_parser(
        "adapt-source-audit",
        help="convert codex-static-fact-audit-v2 into SHA-bound source decisions",
    )
    adapt.add_argument("--input-bundle", type=Path, required=True)
    adapt.add_argument("--audit", type=Path, required=True)
    adapt.add_argument("--output-dir", type=Path, required=True)

    stage = commands.add_parser("stage", help="stage an authoritative review-only source bundle")
    stage.add_argument("--input-bundle", type=Path, required=True)
    stage.add_argument("--source-finalization-manifest", type=Path, required=True)
    stage.add_argument("--output-dir", type=Path, required=True)
    stage.add_argument("--config", type=Path)
    stage.add_argument("--run-id")
    stage.add_argument(
        "--source-decisions",
        type=Path,
        required=True,
        help="Complete JSONL of SHA-bound static-g0a-source-decision-v1 rows.",
    )
    stage.add_argument(
        "--source-decisions-manifest",
        type=Path,
        required=True,
        help="Manifest emitted by adapt-source-audit for the exact source bundle.",
    )

    run = commands.add_parser(
        "run-upstream", help="run translation/distractor/context stages and hard-stop before Simulation"
    )
    run.add_argument("--staging-dir", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")

    export = commands.add_parser(
        "export-review", help="export a behavior-blind Codex adjudication packet"
    )
    export.add_argument("--staging-manifest", type=Path, required=True)
    export.add_argument("--upstream-run-dir", type=Path, required=True)
    export.add_argument(
        "--context-repairs",
        type=Path,
        help=(
            "Optional behavior-blind, SHA-bound context repair overlay. "
            "Repairs add candidates and never rewrite upstream artifacts."
        ),
    )
    export.add_argument("--output-dir", type=Path, required=True)

    resolve = commands.add_parser(
        "resolve-draft",
        help="materialize the exact post-Codex rows for closure/exposure review",
    )
    resolve.add_argument("--staging-manifest", type=Path, required=True)
    resolve.add_argument("--review-input-manifest", type=Path, required=True)
    resolve.add_argument("--codex-decisions", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)

    finalize = commands.add_parser(
        "finalize", help="fail-closed publication of the exact-SHA static bundle"
    )
    finalize.add_argument("--staging-manifest", type=Path, required=True)
    finalize.add_argument("--review-input-manifest", type=Path, required=True)
    finalize.add_argument("--codex-decisions", type=Path, required=True)
    finalize.add_argument("--resolved-static-draft-manifest", type=Path, required=True)
    finalize.add_argument("--semantic-closure-manifest", type=Path)
    finalize.add_argument("--historical-exposure-manifest", type=Path)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "adapt-source-audit":
        value = adapt_source_audit(
            input_bundle_path=args.input_bundle,
            audit_path=args.audit,
            output_dir=args.output_dir,
        )
    elif args.command == "stage":
        value = stage_static_g0a(
            input_bundle_path=args.input_bundle,
            source_finalization_manifest_path=args.source_finalization_manifest,
            output_dir=args.output_dir,
            config_path=args.config,
            run_id=args.run_id,
            source_decisions_path=args.source_decisions,
            source_decisions_manifest_path=args.source_decisions_manifest,
        )
    elif args.command == "run-upstream":
        value = run_static_upstream(
            staging_dir=args.staging_dir,
            config_path=args.config,
            env_path=args.env_file,
        )
    elif args.command == "export-review":
        value = export_codex_review(
            staging_manifest_path=args.staging_manifest,
            upstream_run_dir=args.upstream_run_dir,
            output_dir=args.output_dir,
            context_repairs_path=args.context_repairs,
        )
    elif args.command == "resolve-draft":
        value = resolve_static_g0a_draft(
            staging_manifest_path=args.staging_manifest,
            review_input_manifest_path=args.review_input_manifest,
            codex_decisions_path=args.codex_decisions,
            output_dir=args.output_dir,
        )
    else:
        value = finalize_static_g0a(
            staging_manifest_path=args.staging_manifest,
            review_input_manifest_path=args.review_input_manifest,
            codex_decisions_path=args.codex_decisions,
            resolved_static_draft_manifest_path=args.resolved_static_draft_manifest,
            semantic_closure_manifest_path=args.semantic_closure_manifest,
            historical_exposure_manifest_path=args.historical_exposure_manifest,
            output_dir=args.output_dir,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if value.get("status") == "blocked_fail_closed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
