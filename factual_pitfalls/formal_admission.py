"""Fail-closed external-manifest admission for reviewed factual bundles.

Formal admission is based on byte snapshots, not row-local status claims. The
review and split manifests must bind the exact input bundle and concrete
decision/audit artifacts. A digest and the JSON it describes are always
derived from the same ``read_bytes`` snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


INPUT_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
LEGACY_INPUT_BUNDLE_SCHEMA_VERSION = "legacy-canonical-factual-prompt-v2"
FORMAL_EVIDENCE_TIERS = frozenset(
    {
        "dual_model_consensus",
        "independent_adjudicated",
        "reviewed_canonical",
        "legacy_reviewed_canonical",
    }
)
REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION = "factual-review-freeze-manifest-v2"
SPLIT_FREEZE_MANIFEST_SCHEMA_VERSION = "factual-split-freeze-manifest-v2"

REVIEW_SCOPE_RECORD_SCHEMA_VERSION = "factual-review-freeze-scope-record-v1"
# These evidence rows intentionally use new schema versions.  The v1 review
# decisions did not bind the promoted evidence tier, and the v1 near-duplicate
# decisions did not bind the candidate row or either endpoint.  Accepting those
# rows as formal evidence would therefore be fail-open compatibility.
REVIEW_DECISION_SCHEMA_VERSION = "factual-review-freeze-decision-v2"
NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION = "factual-near-duplicate-candidate-v2"
NEAR_DUPLICATE_DECISION_SCHEMA_VERSION = "factual-near-duplicate-decision-v2"
HISTORICAL_EXPOSURE_REGISTRY_SCHEMA_VERSION = "factual-historical-exposure-registry-v1"
HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION = "factual-historical-exposure-check-v1"

KNOWN_LEGACY_CANONICAL_RELATIVE_PATH = Path(
    "data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_triples.jsonl"
)
KNOWN_LEGACY_INPUT_RELATIVE_PATH = Path(
    "data_processed/factual_triples/triple-full-v1/canonical-v2/"
    "factual_triples_with_distractors.jsonl"
)
KNOWN_LEGACY_CANONICAL_SHA256 = (
    "4fb7addaf141c63588a9442edfe59a77c3ab402d0dfc2799dd7b3619f0067acc"
)
KNOWN_LEGACY_INPUT_SHA256 = (
    "d98bd77dbc6d80f86898946c010459c9e4377cbca6b0ccf0dd6a7b8517751608"
)
KNOWN_LEGACY_INPUT_RECORD_COUNT = 2450

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _json_from_bytes(raw: bytes, path: Path) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON snapshot: {path}: {exc}") from exc


def _read_json_snapshot(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = resolved.read_bytes()
    value = _json_from_bytes(raw, resolved)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {resolved}")
    return value, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": value.get("schema_version"),
    }


def _read_jsonl_snapshot(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    raw = resolved.read_bytes()
    rows: List[Dict[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 JSONL snapshot: {resolved}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {resolved}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {resolved}:{line_number}")
        rows.append(value)
    return rows, hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    """Compatibility reader; admission itself uses the paired snapshot helper."""

    return _read_json_snapshot(path)[0]


def explicit_base_fact_id(record: Mapping[str, Any], record_label: str) -> str:
    authoritative = record.get("base_fact_id")
    compatibility_alias = record.get("base_id")
    if authoritative is not None and compatibility_alias is not None:
        if str(authoritative).strip() != str(compatibility_alias).strip():
            raise ValueError(
                f"{record_label} has conflicting base_fact_id and base_id"
            )
    value = authoritative if authoritative is not None else compatibility_alias
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_label} is missing non-empty base_fact_id")
    return value.strip()


def stable_base_fact_ids_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the sorted full bundle ID inventory, independent of row ordering."""

    base_fact_ids = [
        explicit_base_fact_id(record, f"input bundle row {index}")
        for index, record in enumerate(records, start=1)
    ]
    if len(base_fact_ids) != len(set(base_fact_ids)):
        raise ValueError("input bundle contains duplicate base_fact_id values")
    return sha256_value(sorted(base_fact_ids))


def _parse_bundle_snapshot(path: Path, raw: bytes) -> Tuple[List[Dict[str, Any]], str]:
    if path.suffix.casefold() == ".json":
        payload = _json_from_bytes(raw, path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected input bundle JSON object: {path}")
        rows = payload.get("records")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Input bundle records must be an array of objects: {path}")
        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, str):
            raise ValueError(f"Input bundle is missing schema_version: {path}")
        return rows, schema_version
    if path.suffix.casefold() != ".jsonl":
        raise ValueError("input bundle must be a .json or .jsonl file")
    rows: List[Dict[str, Any]] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 input bundle: {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        rows.append(value)
    versions = {row.get("schema_version") for row in rows}
    if versions == {INPUT_BUNDLE_SCHEMA_VERSION}:
        return rows, INPUT_BUNDLE_SCHEMA_VERSION
    if versions == {None}:
        return rows, LEGACY_INPUT_BUNDLE_SCHEMA_VERSION
    raise ValueError(f"Mixed or unsupported input bundle schema versions: {path}")


def bundle_identity(
    input_bundle_path: Path,
    input_schema_version: str,
    records: Sequence[Mapping[str, Any]],
    input_bundle_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build identity from one real byte snapshot and verify caller precomputation."""

    path = Path(input_bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if input_bundle_sha256 is not None:
        if not _is_sha256(input_bundle_sha256):
            raise ValueError("input_bundle_sha256 must be a lowercase SHA-256 hex digest")
        if input_bundle_sha256 != actual_sha256:
            raise ValueError("input_bundle_sha256 does not match input bundle bytes")
    snapshot_records, snapshot_schema = _parse_bundle_snapshot(path, raw)
    if snapshot_schema != input_schema_version:
        raise ValueError("input schema_version does not match input bundle bytes")
    if snapshot_records != list(records):
        raise ValueError("input records do not match input bundle bytes")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "schema_version": input_schema_version,
        "record_count": len(records),
        "base_fact_ids_sha256": (
            stable_base_fact_ids_sha256(records)
            if input_schema_version == INPUT_BUNDLE_SCHEMA_VERSION
            else None
        ),
    }


def _missing_manifest_provenance() -> Dict[str, Any]:
    return {"path": None, "sha256": None, "schema_version": None}


def _validate_bundle_binding(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    manifest_label: str,
) -> None:
    binding = manifest.get("input_bundle")
    if not isinstance(binding, dict):
        raise ValueError(f"{manifest_label} is missing input_bundle binding")
    for field in (
        "sha256",
        "schema_version",
        "record_count",
        "base_fact_ids_sha256",
    ):
        actual = binding.get(field)
        expected_value = expected.get(field)
        type_matches = (
            type(actual) is int
            if field == "record_count"
            else isinstance(actual, str)
        )
        if not type_matches or actual != expected_value:
            raise ValueError(
                f"{manifest_label} input_bundle.{field} does not match the exact input bundle"
            )


def _single_split_policy_version(records: Sequence[Mapping[str, Any]]) -> Optional[str]:
    versions = {
        value.strip()
        for record in records
        if record.get("canonical_status") == "frozen"
        for value in [record.get("split_policy_version")]
        if isinstance(value, str) and value.strip()
    }
    if len(versions) > 1:
        raise ValueError("input bundle must use exactly one split_policy_version")
    return next(iter(versions), None)


def _binding_path(binding: Mapping[str, Any], manifest_path: Path, label: str) -> Path:
    value = binding.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.path must be non-empty")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _validate_jsonl_binding(
    binding: Any,
    *,
    manifest_path: Path,
    label: str,
    expected_schema: str,
    id_field: str,
    ids_digest_field: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding must be an object")
    path = _binding_path(binding, manifest_path, label)
    rows, actual_sha256 = _read_jsonl_snapshot(path)
    if binding.get("sha256") != actual_sha256:
        raise ValueError(f"{label}.sha256 does not match artifact bytes")
    if binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label}.schema_version is unsupported")
    if type(binding.get("record_count")) is not int or binding["record_count"] != len(rows):
        raise ValueError(f"{label}.record_count does not match artifact rows")
    identifiers: List[str] = []
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != expected_schema:
            raise ValueError(f"{label} row {index} has unsupported schema_version")
        identifier = row.get(id_field)
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{label} row {index} is missing {id_field}")
        identifiers.append(identifier.strip())
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} contains duplicate {id_field} values")
    expected_ids_sha256 = sha256_value(sorted(identifiers))
    if binding.get(ids_digest_field) != expected_ids_sha256:
        raise ValueError(f"{label}.{ids_digest_field} does not match artifact rows")
    return rows, {
        "path": str(path),
        "sha256": actual_sha256,
        "schema_version": expected_schema,
        "record_count": len(rows),
        ids_digest_field: expected_ids_sha256,
    }


def _record_index(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {
        explicit_base_fact_id(record, f"input bundle row {index}"): record
        for index, record in enumerate(records, start=1)
    }


def _validate_reviewer_identity(row: Mapping[str, Any], label: str) -> None:
    reviewer_type = row.get("reviewer_type")
    if reviewer_type not in {"human", "codex_proxy"}:
        raise ValueError(f"{label} has unsupported reviewer_type")
    if type(row.get("human_gold")) is not bool:
        raise ValueError(f"{label} human_gold must be boolean")
    if reviewer_type == "codex_proxy" and row.get("human_gold") is not False:
        raise ValueError(f"{label} codex_proxy must have human_gold=false")


def _validate_input_review_evidence(
    input_record: Mapping[str, Any],
    decision: Mapping[str, Any],
    label: str,
) -> None:
    """Require row-local evidence claims to match the bound review decision."""

    input_human_gold = input_record.get("human_gold")
    if type(input_human_gold) is not bool:
        raise ValueError(f"{label} input row human_gold must be boolean")
    if input_human_gold is not decision.get("human_gold"):
        raise ValueError(
            f"{label} input row human_gold does not match verified review evidence"
        )

    input_evidence_tier = input_record.get("evidence_tier")
    decision_evidence_tier = decision.get("evidence_tier")
    if input_evidence_tier not in FORMAL_EVIDENCE_TIERS:
        raise ValueError(f"{label} input row evidence_tier is not a canonical formal tier")
    if decision_evidence_tier not in FORMAL_EVIDENCE_TIERS:
        raise ValueError(f"{label} review evidence_tier is not a canonical formal tier")
    if input_evidence_tier != decision_evidence_tier:
        raise ValueError(
            f"{label} input row evidence_tier does not match verified review evidence"
        )


def _validate_review_manifest(
    path: Path,
    expected_bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path).resolve()
    manifest, provenance = _read_json_snapshot(path)
    if manifest.get("schema_version") != REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported review freeze manifest schema_version")
    _validate_bundle_binding(manifest, expected_bundle, "review freeze manifest")
    if manifest.get("review_complete") is not True:
        raise ValueError("review freeze manifest review_complete must be true")
    if manifest.get("canonical_freeze_authorized") is not True:
        raise ValueError(
            "review freeze manifest canonical_freeze_authorized must be true"
        )

    index = _record_index(records)
    all_ids = sorted(index)
    frozen_ids = sorted(
        base_fact_id
        for base_fact_id, record in index.items()
        if record.get("canonical_status") == "frozen"
    )
    rejected_ids = sorted(
        base_fact_id
        for base_fact_id, record in index.items()
        if record.get("canonical_status") == "rejected"
    )
    frozen_id_set = set(frozen_ids)
    nonterminal_ids = sorted(set(all_ids) - set(frozen_ids) - set(rejected_ids))
    if nonterminal_ids:
        raise ValueError(
            "review freeze input contains non-terminal canonical_status rows: "
            + ", ".join(nonterminal_ids[:5])
        )
    universe = manifest.get("source_universe")
    if not isinstance(universe, dict):
        raise ValueError("review freeze manifest is missing source_universe")
    if (
        type(universe.get("record_count")) is not int
        or universe.get("record_count") != len(all_ids)
    ):
        raise ValueError("review freeze manifest source_universe.record_count mismatch")
    if universe.get("base_fact_ids_sha256") != sha256_value(all_ids):
        raise ValueError("review freeze manifest source_universe.base_fact_ids_sha256 mismatch")

    scope_rows, scope_provenance = _validate_jsonl_binding(
        manifest.get("review_scope"),
        manifest_path=path,
        label="review freeze manifest review_scope",
        expected_schema=REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
        id_field="base_fact_id",
        ids_digest_field="base_fact_ids_sha256",
    )
    scope_ids = sorted(str(row["base_fact_id"]).strip() for row in scope_rows)
    if scope_ids != all_ids:
        raise ValueError("review scope does not exactly cover the source universe")
    for row in scope_rows:
        base_fact_id = str(row["base_fact_id"]).strip()
        if row.get("input_record_sha256") != sha256_value(index[base_fact_id]):
            raise ValueError(f"review scope has stale input_record_sha256: {base_fact_id}")

    decision_rows, decision_provenance = _validate_jsonl_binding(
        manifest.get("review_decisions"),
        manifest_path=path,
        label="review freeze manifest review_decisions",
        expected_schema=REVIEW_DECISION_SCHEMA_VERSION,
        id_field="base_fact_id",
        ids_digest_field="base_fact_ids_sha256",
    )
    decision_ids = sorted(str(row["base_fact_id"]).strip() for row in decision_rows)
    if decision_ids != scope_ids:
        raise ValueError("review decisions do not exactly cover review scope")
    for row in decision_rows:
        base_fact_id = str(row["base_fact_id"]).strip()
        expected_decision = "accept" if base_fact_id in frozen_id_set else "reject"
        if row.get("decision") != expected_decision:
            raise ValueError(
                f"review decision does not match terminal canonical_status: {base_fact_id}"
            )
        if row.get("input_record_sha256") != sha256_value(index[base_fact_id]):
            raise ValueError(f"review decision has stale input_record_sha256: {base_fact_id}")
        for field in (
            "member_review_complete",
            "alias_review_complete",
            "distractor_review_complete",
        ):
            if row.get(field) is not True:
                raise ValueError(f"review decision {base_fact_id} has incomplete {field}")
        _validate_reviewer_identity(row, f"review decision {base_fact_id}")
        _validate_input_review_evidence(
            index[base_fact_id], row, f"review decision {base_fact_id}"
        )
    expected_counts = {
        "accept": len(frozen_ids),
        "reject": len(rejected_ids),
        "defer": 0,
        "revise": 0,
        "missing": 0,
    }
    decision_counts = manifest.get("decision_counts")
    if (
        not isinstance(decision_counts, dict)
        or set(decision_counts) != set(expected_counts)
        or any(type(decision_counts.get(field)) is not int for field in expected_counts)
        or decision_counts != expected_counts
    ):
        raise ValueError("review freeze manifest decision_counts mismatch")

    provenance.update(
        {
            "schema_version": REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION,
            "review_scope": scope_provenance,
            "review_decisions": decision_provenance,
        }
    )
    return manifest, provenance


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _answer_aliases(record: Mapping[str, Any]) -> set[str]:
    aliases = record.get("answer_aliases_en")
    values = [record.get("answer_en"), *(aliases if isinstance(aliases, list) else [])]
    return {text for value in values if (text := _normalized_text(value))}


def _mechanical_cross_group_risk_reasons(
    records: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], set[str]]:
    indexed = {
        explicit_base_fact_id(record, f"input bundle row {index}"): record
        for index, record in enumerate(records, start=1)
        if record.get("canonical_status") in {"frozen", "rejected"}
    }
    buckets: Dict[str, Dict[Any, List[str]]] = {
        "canonical_fact_exact": {},
        "prompt_answer_exact": {},
        "answer_alias_exact": {},
    }
    for base_fact_id, record in indexed.items():
        canonical_fact = _normalized_text(
            record.get("canonical_fact_en") or record.get("canonical_fact")
        )
        if canonical_fact:
            buckets["canonical_fact_exact"].setdefault(canonical_fact, []).append(
                base_fact_id
            )
        prompt = _normalized_text(record.get("prompt_en"))
        answer = _normalized_text(record.get("answer_en"))
        if prompt and answer:
            buckets["prompt_answer_exact"].setdefault((prompt, answer), []).append(
                base_fact_id
            )
        for alias in _answer_aliases(record):
            buckets["answer_alias_exact"].setdefault(alias, []).append(base_fact_id)

    risks: Dict[Tuple[str, str], set[str]] = {}
    for reason, keyed_groups in buckets.items():
        for grouped_ids in keyed_groups.values():
            for left_id, right_id in combinations(sorted(set(grouped_ids)), 2):
                left_group = indexed[left_id].get("split_group_id")
                right_group = indexed[right_id].get("split_group_id")
                if (
                    isinstance(left_group, str)
                    and left_group.strip()
                    and left_group == right_group
                ):
                    continue
                risks.setdefault((left_id, right_id), set()).add(reason)
    return risks


def _mechanical_cross_group_risk_pairs(
    records: Sequence[Mapping[str, Any]],
) -> set[Tuple[str, str]]:
    return set(_mechanical_cross_group_risk_reasons(records))


def _split_assignments(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, str]], Dict[str, Mapping[str, Any]]]:
    index = _record_index(records)
    assignments: List[Dict[str, str]] = []
    group_contracts: Dict[str, Tuple[str, str]] = {}
    for base_fact_id, record in index.items():
        if record.get("canonical_status") != "frozen":
            continue
        group_id = record.get("split_group_id")
        assignment = record.get("split_assignment")
        policy = record.get("split_policy_version")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"frozen input row is missing split_group_id: {base_fact_id}")
        if assignment not in {"development", "validation", "sealed"}:
            raise ValueError(f"frozen input row has invalid split_assignment: {base_fact_id}")
        if not isinstance(policy, str) or not policy.strip():
            raise ValueError(f"frozen input row is missing split_policy_version: {base_fact_id}")
        contract = (str(assignment), policy.strip())
        previous = group_contracts.setdefault(group_id.strip(), contract)
        if previous != contract:
            raise ValueError(f"split group {group_id!r} has inconsistent assignment or policy")
        assignments.append(
            {
                "base_fact_id": base_fact_id,
                "split_group_id": group_id.strip(),
                "split_assignment": str(assignment),
            }
        )
    assignments.sort(key=lambda row: row["base_fact_id"])
    return assignments, index


def _validate_manifest_reference(
    reference: Any,
    *,
    manifest_path: Path,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    referenced_path = _binding_path(reference, manifest_path, label)
    if referenced_path != Path(str(expected["path"])).resolve():
        raise ValueError(f"{label}.path does not reference the validated manifest")
    for field in ("sha256", "schema_version"):
        if reference.get(field) != expected.get(field):
            raise ValueError(f"{label}.{field} does not reference the validated manifest")


def _validate_split_manifest(
    path: Path,
    expected_bundle: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    split_policy_version: Optional[str],
    review_provenance: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(path).resolve()
    manifest, provenance = _read_json_snapshot(path)
    if manifest.get("schema_version") != SPLIT_FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported split freeze manifest schema_version")
    _validate_bundle_binding(manifest, expected_bundle, "split freeze manifest")
    _validate_manifest_reference(
        manifest.get("review_freeze_manifest"),
        manifest_path=path,
        expected=review_provenance,
        label="split freeze manifest review_freeze_manifest",
    )
    if split_policy_version is None:
        raise ValueError(
            "input bundle must provide one split_policy_version for split freeze admission"
        )
    if manifest.get("split_policy_version") != split_policy_version:
        raise ValueError(
            "split freeze manifest split_policy_version does not match the input bundle"
        )
    if manifest.get("split_status") != "frozen":
        raise ValueError("split freeze manifest split_status must equal 'frozen'")
    if manifest.get("near_duplicate_review_complete") is not True:
        raise ValueError("split freeze manifest near_duplicate_review_complete must equal True")
    if manifest.get("historical_exposure_check_complete") is not True:
        raise ValueError("split freeze manifest historical_exposure_check_complete must equal True")
    for field in (
        "unresolved_near_duplicate_count",
        "unresolved_cross_split_count",
        "validation_exposure_count",
        "sealed_exposure_count",
    ):
        if type(manifest.get(field)) is not int or manifest.get(field) != 0:
            raise ValueError(f"split freeze manifest {field} must equal 0")
    for field in ("near_duplicate_policy_version", "historical_exposure_policy_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"split freeze manifest {field} must be non-empty")

    assignments, index = _split_assignments(records)
    if manifest.get("split_assignments_sha256") != sha256_value(assignments):
        raise ValueError("split freeze manifest split_assignments_sha256 mismatch")
    frozen_ids = {row["base_fact_id"] for row in assignments}
    terminal_ids = set(index)

    candidate_rows, candidate_provenance = _validate_jsonl_binding(
        manifest.get("near_duplicate_candidates"),
        manifest_path=path,
        label="split freeze manifest near_duplicate_candidates",
        expected_schema=NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION,
        id_field="candidate_id",
        ids_digest_field="candidate_ids_sha256",
    )
    candidate_contracts: Dict[str, Dict[str, Any]] = {}
    represented_pairs: set[Tuple[str, str]] = set()
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"]).strip()
        left = row.get("left_base_fact_id")
        right = row.get("right_base_fact_id")
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError(f"near-duplicate candidate has invalid endpoints: {candidate_id}")
        declared_left = left.strip()
        declared_right = right.strip()
        if not declared_left or not declared_right or declared_left == declared_right:
            raise ValueError(f"near-duplicate candidate has invalid endpoints: {candidate_id}")
        pair = tuple(sorted((declared_left, declared_right)))
        if not set(pair) <= terminal_ids:
            raise ValueError(f"near-duplicate candidate references unknown terminal row: {candidate_id}")
        if pair in represented_pairs:
            raise ValueError(f"duplicate near-duplicate candidate pair: {pair}")
        for side, base_fact_id in (
            ("left", declared_left),
            ("right", declared_right),
        ):
            endpoint = index[base_fact_id]
            endpoint_candidate_id = endpoint.get("candidate_id")
            if not isinstance(endpoint_candidate_id, str) or not endpoint_candidate_id.strip():
                raise ValueError(
                    f"near-duplicate candidate endpoint is missing candidate_id: {candidate_id}"
                )
            if row.get(f"{side}_candidate_id") != endpoint_candidate_id:
                raise ValueError(
                    f"near-duplicate candidate has stale {side} candidate_id: {candidate_id}"
                )
            if row.get(f"{side}_input_record_sha256") != sha256_value(endpoint):
                raise ValueError(f"near-duplicate candidate has stale {side} row hash: {candidate_id}")
        candidate_contracts[candidate_id] = {
            "pair": pair,
            "candidate_record_sha256": sha256_value(row),
            "left_candidate_id": row["left_candidate_id"],
            "right_candidate_id": row["right_candidate_id"],
            "left_input_record_sha256": row["left_input_record_sha256"],
            "right_input_record_sha256": row["right_input_record_sha256"],
        }
        represented_pairs.add(pair)
    mechanical_risk_reasons = _mechanical_cross_group_risk_reasons(records)
    missing_mechanical_pairs = set(mechanical_risk_reasons) - represented_pairs
    if missing_mechanical_pairs:
        raise ValueError("near-duplicate candidates omit mechanically detectable cross-group pairs")

    near_decisions, near_decision_provenance = _validate_jsonl_binding(
        manifest.get("near_duplicate_adjudications"),
        manifest_path=path,
        label="split freeze manifest near_duplicate_adjudications",
        expected_schema=NEAR_DUPLICATE_DECISION_SCHEMA_VERSION,
        id_field="candidate_id",
        ids_digest_field="candidate_ids_sha256",
    )
    if {str(row["candidate_id"]).strip() for row in near_decisions} != set(candidate_contracts):
        raise ValueError("near-duplicate adjudications do not exactly cover candidates")
    for row in near_decisions:
        candidate_id = str(row["candidate_id"]).strip()
        decision = row.get("decision")
        if decision not in {"same_fact", "distinct"}:
            raise ValueError(f"near-duplicate decision is unresolved: {candidate_id}")
        _validate_reviewer_identity(row, f"near-duplicate decision {candidate_id}")
        candidate_contract = candidate_contracts[candidate_id]
        for field in (
            "candidate_record_sha256",
            "left_candidate_id",
            "right_candidate_id",
            "left_input_record_sha256",
            "right_input_record_sha256",
        ):
            if row.get(field) != candidate_contract[field]:
                raise ValueError(
                    f"near-duplicate decision has stale or missing {field}: {candidate_id}"
                )
        pair = candidate_contract["pair"]
        exact_reasons = mechanical_risk_reasons.get(pair, set()) & {
            "canonical_fact_exact",
            "prompt_answer_exact",
        }
        if decision == "distinct" and exact_reasons:
            raise ValueError(
                f"exact duplicate candidate cannot be adjudicated distinct: {candidate_id}"
            )
        if decision == "same_fact":
            left, right = pair
            if left in frozen_ids and right in frozen_ids:
                assignment_by_id = {item["base_fact_id"]: item for item in assignments}
                left_assignment = assignment_by_id[left]
                right_assignment = assignment_by_id[right]
                if (
                    left_assignment["split_group_id"] != right_assignment["split_group_id"]
                    or left_assignment["split_assignment"] != right_assignment["split_assignment"]
                ):
                    raise ValueError(
                        f"same-fact candidate remains split across groups: {candidate_id}"
                    )

    registry_rows, registry_provenance = _validate_jsonl_binding(
        manifest.get("historical_exposure_registry"),
        manifest_path=path,
        label="split freeze manifest historical_exposure_registry",
        expected_schema=HISTORICAL_EXPOSURE_REGISTRY_SCHEMA_VERSION,
        id_field="exposure_id",
        ids_digest_field="exposure_ids_sha256",
    )
    exposed_ids = set()
    for row in registry_rows:
        base_fact_id = row.get("base_fact_id")
        if not isinstance(base_fact_id, str) or not base_fact_id.strip():
            raise ValueError("historical exposure registry row is missing base_fact_id")
        exposed_ids.add(base_fact_id.strip())
    sensitive_ids = {
        row["base_fact_id"]
        for row in assignments
        if row["split_assignment"] in {"validation", "sealed"}
    }
    if exposed_ids & sensitive_ids:
        raise ValueError("historical exposure registry contains validation or sealed facts")
    exposure_checks, exposure_provenance = _validate_jsonl_binding(
        manifest.get("historical_exposure_checks"),
        manifest_path=path,
        label="split freeze manifest historical_exposure_checks",
        expected_schema=HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION,
        id_field="base_fact_id",
        ids_digest_field="base_fact_ids_sha256",
    )
    if {str(row["base_fact_id"]).strip() for row in exposure_checks} != sensitive_ids:
        raise ValueError("historical exposure checks do not exactly cover validation and sealed facts")
    assignment_by_id = {row["base_fact_id"]: row for row in assignments}
    for row in exposure_checks:
        base_fact_id = str(row["base_fact_id"]).strip()
        if row.get("input_record_sha256") != sha256_value(index[base_fact_id]):
            raise ValueError(f"historical exposure check has stale row hash: {base_fact_id}")
        if row.get("split_assignment") != assignment_by_id[base_fact_id]["split_assignment"]:
            raise ValueError(f"historical exposure check has wrong split: {base_fact_id}")
        if row.get("historically_exposed") is not False:
            raise ValueError(f"historical exposure is unresolved or present: {base_fact_id}")

    provenance.update(
        {
            "schema_version": SPLIT_FREEZE_MANIFEST_SCHEMA_VERSION,
            "review_freeze_manifest": {
                field: review_provenance[field] for field in ("path", "sha256", "schema_version")
            },
            "near_duplicate_candidates": candidate_provenance,
            "near_duplicate_adjudications": near_decision_provenance,
            "historical_exposure_registry": registry_provenance,
            "historical_exposure_checks": exposure_provenance,
        }
    )
    return manifest, provenance


def _validate_known_legacy_bundle(
    identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    project_root: Optional[Path],
) -> None:
    if project_root is None:
        raise ValueError("legacy input bundle requires verified historical project root")
    root = Path(project_root).resolve()
    expected_input = (root / KNOWN_LEGACY_INPUT_RELATIVE_PATH).resolve()
    expected_canonical = (root / KNOWN_LEGACY_CANONICAL_RELATIVE_PATH).resolve()
    if Path(str(identity["path"])).resolve() != expected_input:
        raise ValueError("legacy compatibility is limited to the pinned historical input path")
    if identity.get("sha256") != KNOWN_LEGACY_INPUT_SHA256:
        raise ValueError("legacy input bundle SHA-256 does not match the pinned historical artifact")
    if len(records) != KNOWN_LEGACY_INPUT_RECORD_COUNT:
        raise ValueError("legacy input bundle record count does not match the pinned artifact")
    if not expected_canonical.is_file():
        raise FileNotFoundError(expected_canonical)
    if sha256_file(expected_canonical) != KNOWN_LEGACY_CANONICAL_SHA256:
        raise ValueError("legacy canonical SHA-256 does not match the pinned historical artifact")


def _admission_fingerprint(result: Mapping[str, Any]) -> str:
    def without_paths(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_paths(child)
                for key, child in value.items()
                if key != "path"
            }
        if isinstance(value, list):
            return [without_paths(child) for child in value]
        return value

    return sha256_value(
        without_paths({
            "required": result["required"],
            "legacy_compatibility_exempt": result["legacy_compatibility_exempt"],
            "authorized": result["authorized"],
            "blockers": result["blockers"],
            "input_bundle": result["input_bundle"],
            "review_freeze_manifest": result["review_freeze_manifest"],
            "split_freeze_manifest": result["split_freeze_manifest"],
            "split_policy_version": result["split_policy_version"],
        })
    )


def validate_external_formal_admission(
    *,
    input_bundle_path: Path,
    input_schema_version: str,
    records: Sequence[Mapping[str, Any]],
    input_bundle_sha256: Optional[str] = None,
    review_freeze_manifest_path: Optional[Path] = None,
    split_freeze_manifest_path: Optional[Path] = None,
    legacy_project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate exact-bundle review/split evidence or return missing blockers."""

    identity = bundle_identity(
        input_bundle_path,
        input_schema_version,
        records,
        input_bundle_sha256=input_bundle_sha256,
    )
    if input_schema_version != INPUT_BUNDLE_SCHEMA_VERSION:
        if input_schema_version != LEGACY_INPUT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported input bundle schema_version: {input_schema_version}")
        _validate_known_legacy_bundle(identity, records, legacy_project_root)
        result = {
            "required": False,
            "legacy_compatibility_exempt": True,
            "authorized": True,
            "blockers": [],
            "input_bundle": identity,
            "review_freeze_manifest": _missing_manifest_provenance(),
            "split_freeze_manifest": _missing_manifest_provenance(),
            "split_policy_version": None,
        }
        result["admission_fingerprint_sha256"] = _admission_fingerprint(result)
        return result

    blockers: List[str] = []
    split_policy_version = _single_split_policy_version(records)
    review_provenance = _missing_manifest_provenance()
    split_provenance = _missing_manifest_provenance()
    if review_freeze_manifest_path is None:
        blockers.append("review_freeze_manifest_missing")
    else:
        _, review_provenance = _validate_review_manifest(
            Path(review_freeze_manifest_path), identity, records
        )
    if split_freeze_manifest_path is None:
        blockers.append("split_freeze_manifest_missing")
    elif review_freeze_manifest_path is None:
        raise ValueError("split freeze manifest requires a review freeze manifest")
    else:
        _, split_provenance = _validate_split_manifest(
            Path(split_freeze_manifest_path),
            identity,
            records,
            split_policy_version,
            review_provenance,
        )

    result = {
        "required": True,
        "legacy_compatibility_exempt": False,
        "authorized": not blockers,
        "blockers": blockers,
        "input_bundle": identity,
        "review_freeze_manifest": review_provenance,
        "split_freeze_manifest": split_provenance,
        "split_policy_version": split_policy_version,
    }
    result["admission_fingerprint_sha256"] = _admission_fingerprint(result)
    return result
