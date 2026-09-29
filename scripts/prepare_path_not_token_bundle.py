#!/usr/bin/env python3
"""Prepare a reviewed behavior bundle for later HF/PATH_not_token work.

This adapter is deliberately offline.  It does not load a model or tokenizer,
run a behavior baseline, collect activations, or perform an intervention.
Frozen canonical facts are written to ``base_facts.jsonl`` only when external
review- and split-freeze manifests bind the exact input bundle.  With
``--allow-provisional``, non-frozen provisional facts are written only to the
separate ``review_only_base_facts.jsonl`` artifact and keep their upstream
canonical status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls import formal_admission


RECORD_SCHEMA_VERSION = "path-not-token-bundle-base-v1"
EXCLUSION_SCHEMA_VERSION = "path-not-token-bundle-exclusion-v1"
SUMMARY_SCHEMA_VERSION = "path-not-token-bundle-summary-v1"
INPUT_BUNDLE_SCHEMA_VERSION = formal_admission.INPUT_BUNDLE_SCHEMA_VERSION
FROZEN_STATUS = "frozen"
PROVISIONAL_STATUSES = frozenset({"pending_review", "provisional_single_model"})
SUPPORTED_CANONICAL_STATUSES = frozenset(
    {FROZEN_STATUS, *PROVISIONAL_STATUSES, "rejected"}
)
FORMAL_BUNDLE_STATUSES = frozenset({"frozen", "reviewed_frozen"})
FORMAL_EVIDENCE_TIERS = formal_admission.FORMAL_EVIDENCE_TIERS
FORMAL_SPLIT_STATUS = "frozen"
SUPPORTED_SPLIT_ASSIGNMENTS = frozenset({"development", "validation", "sealed"})
PROMPT_FORMAT_BY_QUALITY_TIER = {
    "strict_factual_completion": "factual_completion",
    "relation_specific_open_completion": "factual_completion",
    "open_answer_fallback": "open_question_with_answer_marker",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"\s+", " ", text)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_snapshot(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    records: List[Dict[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL snapshot: {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    if not records:
        raise ValueError(f"Input JSONL is empty: {path}")
    return records, hashlib.sha256(raw).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl_snapshot(path)[0]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(path)


def _required_string(record: Mapping[str, Any], field: str, record_label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_label} is missing non-empty {field}")
    return value.strip()


def _optional_string(record: Mapping[str, Any], field: str, record_label: str) -> Optional[str]:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{record_label} has non-string {field}")
    value = value.strip()
    return value or None


def _base_fact_id(record: Mapping[str, Any], row_number: int) -> str:
    return formal_admission.explicit_base_fact_id(record, f"row {row_number}")


def _canonical_status(record: Mapping[str, Any], record_label: str) -> str:
    status = record.get("canonical_status")
    if status not in SUPPORTED_CANONICAL_STATUSES:
        raise ValueError(
            f"{record_label} has invalid canonical_status: {status}; "
            f"expected one of {sorted(SUPPORTED_CANONICAL_STATUSES)}"
        )
    return str(status)


def _evidence_tier(record: Mapping[str, Any], record_label: str) -> str:
    """Read the final field name while tolerating the provisional adapter alias."""
    primary = record.get("evidence_tier")
    compatibility_alias = record.get("evidence_level")
    if primary is not None and compatibility_alias is not None:
        if str(primary).strip() != str(compatibility_alias).strip():
            raise ValueError(f"{record_label} has conflicting evidence_tier and evidence_level")
    value = primary if primary is not None else compatibility_alias
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_label} is missing non-empty evidence_tier")
    return value.strip()


def _source_prompt_ready(record: Mapping[str, Any]) -> bool:
    admission = record.get("admission")
    return (
        record.get("prompt_ready") is True
        and isinstance(admission, dict)
        and admission.get("prompt_ready") is True
    )


def _formal_review_blockers(
    record: Mapping[str, Any],
    evidence_tier: str,
    external_manifest_blockers: Sequence[str] = (),
) -> List[str]:
    blockers: List[str] = []
    if record.get("bundle_status") not in FORMAL_BUNDLE_STATUSES:
        blockers.append("bundle_not_review_frozen")
    admission = record.get("admission")
    if not isinstance(admission, dict) or admission.get("semantic_review_complete") is not True:
        blockers.append("semantic_review_incomplete")
    if not _source_prompt_ready(record):
        blockers.append("prompt_not_ready")
    if evidence_tier not in FORMAL_EVIDENCE_TIERS:
        blockers.append("evidence_tier_not_formal")
    if record.get("split_status") != FORMAL_SPLIT_STATUS:
        blockers.append("split_not_frozen")
    if record.get("split_assignment") not in SUPPORTED_SPLIT_ASSIGNMENTS:
        blockers.append("split_assignment_missing_or_invalid")
    if not isinstance(record.get("split_group_id"), str) or not record["split_group_id"].strip():
        blockers.append("split_group_id_missing")
    if (
        not isinstance(record.get("split_policy_version"), str)
        or not record["split_policy_version"].strip()
    ):
        blockers.append("split_policy_version_missing")
    if record.get("canonical_status") == FROZEN_STATUS:
        blockers.extend(external_manifest_blockers)
    return blockers


def _prompt_format(prompt_quality_tier: str, record_label: str) -> str:
    try:
        return PROMPT_FORMAT_BY_QUALITY_TIER[prompt_quality_tier]
    except KeyError as exc:
        raise ValueError(
            f"{record_label} has unsupported prompt_quality_tier: {prompt_quality_tier}"
        ) from exc


def _answer_aliases(record: Mapping[str, Any], answer_en: str, record_label: str) -> List[str]:
    values = record.get("answer_aliases_en")
    if not isinstance(values, list):
        raise ValueError(f"{record_label} answer_aliases_en must be an array")
    output: List[str] = []
    seen = set()
    for value in [answer_en, *values]:
        if not isinstance(value, str):
            raise ValueError(f"{record_label} answer_aliases_en must contain only strings")
        text = value.strip()
        key = normalize_text(text)
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def _blocking_reasons(
    canonical_status: str,
    probe_relation_id: Optional[str],
    prompt_ready: bool,
    split_status: Any,
    split_assignment: Any,
    translation_ready: bool,
) -> List[str]:
    reasons: List[str] = []
    if canonical_status != FROZEN_STATUS:
        reasons.extend(["canonical_not_frozen", "semantic_review_required"])
    if not prompt_ready:
        reasons.append("prompt_not_ready")
    if split_status != FORMAL_SPLIT_STATUS:
        reasons.append("split_not_frozen")
    if split_assignment not in SUPPORTED_SPLIT_ASSIGNMENTS:
        reasons.append("split_assignment_missing_or_invalid")
    if probe_relation_id is None:
        reasons.append("probe_relation_not_frozen")
    if not translation_ready:
        reasons.append("target_translation_not_frozen")
    reasons.extend(
        [
            "target_model_and_tokenizer_not_selected",
            "answer_tokenization_not_computed",
            "hf_behavior_baseline_not_run",
        ]
    )
    return reasons


def _path_record(
    source: Mapping[str, Any],
    base_fact_id: str,
    input_path: Path,
    input_sha256: str,
    external_formal_admission: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    record_label = f"base fact {base_fact_id}"
    source_id = _required_string(source, "source_id", record_label)
    canonical_status = _canonical_status(source, record_label)
    evidence_tier = _evidence_tier(source, record_label)
    human_gold = source.get("human_gold")
    if type(human_gold) is not bool:
        raise ValueError(f"{record_label} human_gold must be boolean")
    if canonical_status in PROVISIONAL_STATUSES and human_gold is not False:
        raise ValueError(f"{record_label} provisional input must have human_gold=false")
    prompt_en = _required_string(source, "prompt_en", record_label)
    prompt_quality_tier = _required_string(source, "prompt_quality_tier", record_label)
    prompt_format = _prompt_format(prompt_quality_tier, record_label)
    answer_en = _required_string(source, "answer_en", record_label)
    relation_raw = _required_string(source, "relation_raw", record_label)
    relation_normalized = _optional_string(source, "relation_normalized", record_label)
    probe_relation_id = _optional_string(source, "probe_relation_id", record_label)
    probe_relation_candidate = _optional_string(
        source, "probe_relation_candidate", record_label
    )
    probe_relation_candidate_direction = _optional_string(
        source, "probe_relation_candidate_direction", record_label
    )
    probe_relation_candidate_policy = _optional_string(
        source, "probe_relation_candidate_policy", record_label
    )
    provenance = source.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{record_label} provenance must be an object")

    prompt_ready = _source_prompt_ready(source)
    split_status = source.get("split_status")
    split_assignment = source.get("split_assignment")
    target_language = _optional_string(source, "target_language", record_label)
    subject_target = _optional_string(source, "subject_target", record_label)
    answer_target = _optional_string(source, "answer_target", record_label)
    prompt_target = _optional_string(source, "prompt_target", record_label)
    translation_prompt = _optional_string(
        source, "translation_prompt_en_to_target", record_label
    )
    translation_status = source.get("translation_status")
    translation_ready = (
        translation_status in {"reviewed", "frozen", "reviewed_frozen"}
        and target_language is not None
        and subject_target is not None
        and answer_target is not None
        and prompt_target is not None
        and translation_prompt is not None
    )
    review_only = canonical_status != FROZEN_STATUS
    blockers = _blocking_reasons(
        canonical_status,
        probe_relation_id,
        prompt_ready,
        split_status,
        split_assignment,
        translation_ready,
    )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "preparation_status": (
            "review_only_pending_canonical_freeze"
            if review_only
            else "prepared_offline_not_evaluated"
        ),
        "review_only": review_only,
        "base_fact_id": base_fact_id,
        "base_id": base_fact_id,
        "source_id": source_id,
        "source_pool_id": source.get("source_pool_id"),
        "source_dataset": source.get("source_dataset"),
        "source_subset": source.get("source_subset"),
        "source_format": source.get("source_format"),
        "canonical_status": canonical_status,
        "evidence_tier": evidence_tier,
        "human_gold": human_gold,
        "subject_en": source.get("subject_en"),
        "relation_raw": relation_raw,
        "relation_normalized": relation_normalized,
        "normalization_status": source.get("normalization_status"),
        "relation_signature_id": source.get("relation_signature_id"),
        "probe_relation_id": probe_relation_id,
        "probe_relation_candidate": probe_relation_candidate,
        "probe_relation_candidate_direction": probe_relation_candidate_direction,
        "probe_relation_candidate_policy": probe_relation_candidate_policy,
        "probe_relation_status": source.get("probe_relation_status"),
        "split_policy_version": source.get("split_policy_version"),
        "split_status": split_status,
        "split_group_id": source.get("split_group_id"),
        "split_assignment": split_assignment,
        "sealed_evaluation_status": source.get("sealed_evaluation_status"),
        "canonical_fact_en": source.get("canonical_fact_en") or source.get("canonical_fact"),
        "prompt_en": prompt_en,
        "prompt_version": source.get("prompt_version"),
        "prompt_template_id": source.get("prompt_template_id"),
        "prompt_quality_tier": prompt_quality_tier,
        "prompt_format": prompt_format,
        "answer_en": answer_en,
        "answer_aliases_en": _answer_aliases(source, answer_en, record_label),
        "target_language": target_language,
        "subject_target": subject_target,
        "relation_target": source.get("relation_target"),
        "answer_target": answer_target,
        "answer_aliases_target": source.get("answer_aliases_target"),
        "prompt_target": prompt_target,
        "translation_prompt_en_to_target": translation_prompt,
        "translation_status": translation_status,
        "language_alignment": {
            "known_id": base_fact_id,
            "status": "reviewed" if translation_ready else "pending_reviewed_translation",
            "cross_language_split_group_id": source.get("split_group_id"),
        },
        "path_not_token_experiment_ready": False,
        "admission": {
            "canonical_frozen": canonical_status == FROZEN_STATUS,
            "external_freeze_manifests_verified": bool(
                external_formal_admission is not None
                and external_formal_admission.get("authorized") is True
            ),
            "prompt_ready": prompt_ready,
            "review_only": review_only,
            "relation_conditioned_mechanism_eligible": False,
            "target_translation_ready": translation_ready,
            "path_not_token_experiment_ready": False,
            "blocking_reasons": blockers,
        },
        "answer_tokenization": {
            "status": "pending_target_model_and_tokenizer",
            "model_id": None,
            "model_revision": None,
            "tokenizer_id": None,
            "tokenizer_revision": None,
            "answer_token_ids_en": None,
            "answer_token_ids_target": None,
        },
        "mechanism_analysis": {
            "status": "not_run",
            "hidden_states_collected": False,
            "logit_lens_run": False,
            "activation_patching_run": False,
            "vector_intervention_run": False,
        },
        "provenance": {
            "input_bundle_path": str(input_path),
            "input_bundle_sha256": input_sha256,
            "input_record_sha256": sha256_value(source),
            "review_freeze_manifest": (
                external_formal_admission.get("review_freeze_manifest")
                if external_formal_admission is not None
                else {"path": None, "sha256": None, "schema_version": None}
            ),
            "split_freeze_manifest": (
                external_formal_admission.get("split_freeze_manifest")
                if external_formal_admission is not None
                else {"path": None, "sha256": None, "schema_version": None}
            ),
            "upstream": provenance,
        },
    }


def build_records(
    records: Sequence[Mapping[str, Any]],
    input_path: Path,
    input_sha256: str,
    allow_provisional: bool = False,
    review_freeze_manifest_path: Optional[Path] = None,
    split_freeze_manifest_path: Optional[Path] = None,
    _validated_external_formal_admission: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    frozen: List[Dict[str, Any]] = []
    review_only: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    seen_base_fact_ids = set()
    split_group_contracts: Dict[str, Tuple[str, str]] = {}
    input_split_policy_versions = set()
    formal_split_policy_versions = set()
    external_formal_admission = formal_admission.validate_external_formal_admission(
        input_bundle_path=input_path,
        input_schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
        records=records,
        input_bundle_sha256=input_sha256,
        review_freeze_manifest_path=review_freeze_manifest_path,
        split_freeze_manifest_path=split_freeze_manifest_path,
    )
    if (
        _validated_external_formal_admission is not None
        and dict(_validated_external_formal_admission) != external_formal_admission
    ):
        raise ValueError(
            "prevalidated formal admission does not match revalidated evidence"
        )

    for row_number, source in enumerate(records, start=1):
        schema_version = source.get("schema_version")
        if schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"row {row_number} has unsupported schema_version: {schema_version}; "
                f"expected {INPUT_BUNDLE_SCHEMA_VERSION}"
            )
        base_fact_id = _base_fact_id(source, row_number)
        if base_fact_id in seen_base_fact_ids:
            raise ValueError(f"duplicate base_fact_id: {base_fact_id}")
        seen_base_fact_ids.add(base_fact_id)
        record_label = f"base fact {base_fact_id}"
        source_id = _required_string(source, "source_id", record_label)
        status = _canonical_status(source, record_label)
        evidence_tier = _evidence_tier(source, record_label)
        formal_review_blockers = _formal_review_blockers(
            source,
            evidence_tier,
            external_formal_admission["blockers"],
        )
        split_group_id = source.get("split_group_id")
        split_assignment = source.get("split_assignment")
        split_policy_version = source.get("split_policy_version")
        if (
            isinstance(split_group_id, str)
            and split_group_id.strip()
            and split_assignment in SUPPORTED_SPLIT_ASSIGNMENTS
            and isinstance(split_policy_version, str)
            and split_policy_version.strip()
        ):
            split_contract = (str(split_assignment), split_policy_version.strip())
            input_split_policy_versions.add(split_policy_version.strip())
            previous_contract = split_group_contracts.setdefault(
                split_group_id.strip(), split_contract
            )
            if previous_contract != split_contract:
                raise ValueError(
                    f"split group {split_group_id!r} has inconsistent assignment or policy version"
                )

        if status == "rejected":
            exclusion_reason = "canonical_rejected"
        elif status == FROZEN_STATUS and formal_review_blockers:
            exclusion_reason = "frozen_missing_explicit_review_evidence"
        elif status in PROVISIONAL_STATUSES and not allow_provisional:
            exclusion_reason = "provisional_requires_allow_provisional"
        else:
            prepared = _path_record(
                source,
                base_fact_id,
                input_path,
                input_sha256,
                external_formal_admission,
            )
            if status == FROZEN_STATUS:
                frozen.append(prepared)
                formal_split_policy_versions.add(str(source["split_policy_version"]).strip())
            else:
                review_only.append(prepared)
            continue

        exclusions.append(
            {
                "schema_version": EXCLUSION_SCHEMA_VERSION,
                "base_fact_id": base_fact_id,
                "source_id": source_id,
                "canonical_status": status,
                "exclusion_reason": exclusion_reason,
                "review_blockers": formal_review_blockers,
                "input_record_sha256": sha256_value(source),
            }
        )

    if len(input_split_policy_versions) > 1:
        raise ValueError("input bundle must use exactly one split_policy_version")
    if len(formal_split_policy_versions) > 1:
        raise ValueError("formal input rows must use exactly one split_policy_version")
    frozen.sort(key=lambda item: item["base_fact_id"])
    review_only.sort(key=lambda item: item["base_fact_id"])
    exclusions.sort(key=lambda item: item["base_fact_id"])
    return frozen, review_only, exclusions


def build_summary(
    input_path: Path,
    input_sha256: str,
    input_count: int,
    allow_provisional: bool,
    frozen: Sequence[Mapping[str, Any]],
    review_only: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    external_formal_admission: Mapping[str, Any],
) -> Dict[str, Any]:
    status_counts = Counter(
        [record["canonical_status"] for record in frozen]
        + [record["canonical_status"] for record in review_only]
        + [record["canonical_status"] for record in exclusions]
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "scope": "offline HF/PATH_not_token preparation only; no model experiment was run",
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "record_count": input_count,
        },
        "external_formal_admission": external_formal_admission,
        "review_freeze_manifest": external_formal_admission[
            "review_freeze_manifest"
        ],
        "split_freeze_manifest": external_formal_admission[
            "split_freeze_manifest"
        ],
        "allow_provisional": allow_provisional,
        "counts": {
            "input": input_count,
            "frozen_base_facts": len(frozen),
            "review_only_base_facts": len(review_only),
            "excluded": len(exclusions),
            "canonical_status": dict(sorted(status_counts.items())),
        },
        "status_contract": {
            "formal_output_requires_canonical_status": FROZEN_STATUS,
            "formal_output_requires_split_status": FORMAL_SPLIT_STATUS,
            "formal_output_split_assignments": sorted(SUPPORTED_SPLIT_ASSIGNMENTS),
            "formal_output_requires_split_group_id": True,
            "formal_output_requires_single_split_policy_version": True,
            "formal_output_requires_source_prompt_ready": True,
            "formal_output_requires_review_freeze_manifest": True,
            "formal_output_requires_split_freeze_manifest": True,
            "provisional_output_is_separate_review_only_artifact": True,
            "provisional_status_is_preserved": True,
            "answer_tokenization": "pending_target_model_and_tokenizer",
            "mechanism_analysis": "not_run",
            "path_not_token_experiment_ready": False,
        },
        "not_performed": [
            "target_model_selection",
            "answer_tokenization",
            "hf_behavior_baseline",
            "hidden_state_collection",
            "logit_lens",
            "activation_patching",
            "vector_intervention",
            "repair_validation",
        ],
    }


def prepare(
    input_path: Path,
    output_dir: Path,
    allow_provisional: bool = False,
    review_freeze_manifest_path: Optional[Path] = None,
    split_freeze_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    records, input_sha256 = read_jsonl_snapshot(input_path)
    review_freeze_manifest_path = (
        review_freeze_manifest_path.resolve()
        if review_freeze_manifest_path is not None
        else None
    )
    split_freeze_manifest_path = (
        split_freeze_manifest_path.resolve()
        if split_freeze_manifest_path is not None
        else None
    )
    external_formal_admission = formal_admission.validate_external_formal_admission(
        input_bundle_path=input_path,
        input_schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
        records=records,
        input_bundle_sha256=input_sha256,
        review_freeze_manifest_path=review_freeze_manifest_path,
        split_freeze_manifest_path=split_freeze_manifest_path,
    )
    frozen, review_only, exclusions = build_records(
        records,
        input_path,
        input_sha256,
        allow_provisional=allow_provisional,
        review_freeze_manifest_path=review_freeze_manifest_path,
        split_freeze_manifest_path=split_freeze_manifest_path,
        _validated_external_formal_admission=external_formal_admission,
    )
    summary = build_summary(
        input_path,
        input_sha256,
        len(records),
        allow_provisional,
        frozen,
        review_only,
        exclusions,
        external_formal_admission,
    )

    artifact_records = {
        "base_facts.jsonl": frozen,
        "review_only_base_facts.jsonl": review_only,
        "excluded_records.jsonl": exclusions,
    }
    for name, values in artifact_records.items():
        write_jsonl(output_dir / name, values)
    summary["outputs"] = {
        name: {
            "sha256": sha256_file(output_dir / name),
            "record_count": len(values),
        }
        for name, values in artifact_records.items()
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--input-bundle",
        type=Path,
        required=True,
        help="Reviewed or provisional behavior_input_bundle.jsonl",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for offline PATH_not_token records",
    )
    result.add_argument(
        "--review-freeze-manifest",
        type=Path,
        help=(
            "External factual-review-freeze-manifest-v2 bound to the exact input bundle; "
            "required for frozen rows to enter base_facts.jsonl"
        ),
    )
    result.add_argument(
        "--split-freeze-manifest",
        type=Path,
        help=(
            "External factual-split-freeze-manifest-v2 bound to the exact input bundle; "
            "required for frozen rows to enter base_facts.jsonl"
        ),
    )
    result.add_argument(
        "--allow-provisional",
        action="store_true",
        help=(
            "Write pending canonical facts only to review_only_base_facts.jsonl; "
            "never promote them to formal base_facts.jsonl"
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    summary = prepare(
        args.input_bundle,
        args.output_dir,
        allow_provisional=args.allow_provisional,
        review_freeze_manifest_path=args.review_freeze_manifest,
        split_freeze_manifest_path=args.split_freeze_manifest,
    )
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
