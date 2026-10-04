#!/usr/bin/env python3
"""Predeclare and validate the full-pool historical-exposure authority contract.

The materialize command binds the immutable 8,969-item universe, scans the
current repository worktree and every blob reachable from local Git refs, and
emits a scope-owner attestation *template*.  It deliberately cannot assert
anything about external directories, separately executed jobs, or hosted
platforms, and it never emits a completed attestation or a freeze artifact.

The validate command is read-only.  It validates either the untouched pending
packet or a separately authored scope-owner attestation.  A valid attestation
is evidence that a future successor-universe declaration may bind; it is not
itself review/split freeze authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "full-public-benchmark-historical-exposure-contract-v2"
POLICY_VERSION = "full-universe-repository-scan-plus-scope-owner-authority-v2"
CONTRACT_SCHEMA_VERSION = "public-benchmark-historical-exposure-authority-contract-v2"
ATTESTATION_SCHEMA_VERSION = "public-benchmark-historical-exposure-scope-owner-attestation-v2"
INVENTORY_SCHEMA_VERSION = "public-benchmark-repository-exposure-inventory-v2"
MATCH_SCHEMA_VERSION = "public-benchmark-repository-exposure-match-v2"

SOURCE_UNIVERSE_SCHEMA_VERSION = "public-benchmark-formal-cohort-universe-v2"
SOURCE_ITEM_SCHEMA_VERSION = "public-benchmark-formal-cohort-item-v1"
SOURCE_CONTRACT_STATUS = "not_predeclared"
EXPECTED_RECORD_COUNT = 8969

CONTRACT_NAME = "historical_exposure_authority_contract.pending.v2.json"
ATTESTATION_TEMPLATE_NAME = "scope_owner_attestation.template.v2.json"
INVENTORY_NAME = "repository_exposure_inventory.v2.jsonl"
MATCHES_NAME = "repository_exposure_matches.v2.jsonl"

PENDING_STATUS = "pending_scope_owner_attestation"
AFFIRMED_STATUS = "affirmed_by_scope_owner"
SCOPE_KINDS = (
    "external_directories",
    "external_platforms",
    "separately_executed_runs",
    "other_behavior_sources",
)
REQUIRED_STATEMENTS = (
    "authority_to_attest_full_scope",
    "entire_formal_universe_reviewed_for_historical_exposure",
    "repository_scan_and_detected_hits_reviewed",
    "all_external_directories_inventoried",
    "all_external_platforms_inventoried",
    "all_separately_executed_runs_inventoried",
    "all_other_behavior_sources_inventoried",
    "known_historical_exposures_fully_disclosed",
    "no_unlisted_historical_behavior_source_known",
)
REQUIRED_LIMITATION_ACKNOWLEDGEMENTS = (
    "repository_scan_is_heuristic_not_semantic_proof",
    "attestation_covers_only_activity_through_attested_through",
    "later_behavior_activity_requires_refresh_before_freeze",
    "attestation_does_not_authorize_review_split_hf_or_behavior",
)

IDENTIFIER_FIELDS = frozenset(
    {
        "base_fact_id",
        "source_base_fact_id",
        "cohort_item_id",
        "candidate_id",
        "candidate_ids",
        "source_id",
    }
)
DIRECT_BEHAVIOR_MARKERS = {
    "simulation_id": b'"simulation_id"',
    "behavior_result": b'"behavior_result"',
    "distractor_hit": b'"distractor_hit"',
    "strict_signal": b'"strict_signal"',
    "strict_signal_ids": b'"strict_signal_ids"',
    "selected_option": b'"selected_option"',
    "predicted_answer": b'"predicted_answer"',
}
VARIANT_MARKERS = (b'"variant"', b'"arm"', b'"condition"')
MODEL_MARKERS = (
    b'"model"',
    b'"model_id"',
    b'"model_name"',
    b'"response_model"',
    b'"checkpoint"',
)
OUTCOME_MARKERS = (
    b'"raw_response"',
    b'"model_response"',
    b'"response_text"',
    b'"selected_option"',
    b'"predicted_answer"',
    b'"correct"',
)
RESULT_LIKE_PATH_RE = re.compile(
    r"(?:^|[/_.-])(?:result|results|prediction|predictions|evaluation|eval|response|responses)(?:$|[/_.-])",
    re.IGNORECASE,
)
NON_BEHAVIOR_PATH_TOKENS = (
    "triple_extraction",
    "translation",
    "fact-review",
    "fact_review",
    "semantic-review",
    "semantic_review",
    "perturbation_generation",
    "adjudication",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ContractValidationError(ValueError):
    """The authority contract or attestation fails closed."""


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
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractValidationError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractValidationError(
                    f"JSONL row {line_number} must be an object: {path}"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise ContractValidationError(f"JSONL file is empty: {path}")
    return rows


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _required_sha256(value: Any, label: str) -> str:
    value = _required_string(value, label)
    if SHA256_RE.fullmatch(value) is None:
        raise ContractValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _parse_timestamp(value: Any, label: str) -> datetime:
    value = _required_string(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError(f"{label} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractValidationError(f"{label} must include a timezone")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], stderr=subprocess.STDOUT
    )


def _binding(
    path: Path,
    *,
    schema_version: str,
    record_count: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(Path(output_path).resolve() if output_path else path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema_version,
    }
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _resolve_binding_path(binding: Mapping[str, Any], owner_path: Path) -> Path:
    raw_path = Path(_required_string(binding.get("path"), "binding.path"))
    return raw_path.resolve() if raw_path.is_absolute() else (owner_path.parent / raw_path).resolve()


def _validate_file_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    schema_version: str,
    jsonl: bool = False,
    allow_empty: bool = False,
) -> Tuple[Path, Any]:
    if not isinstance(binding, dict):
        raise ContractValidationError(f"{label} binding must be an object")
    if binding.get("schema_version") != schema_version:
        raise ContractValidationError(f"{label} schema binding is invalid")
    path = _resolve_binding_path(binding, owner_path)
    if not path.is_file():
        raise ContractValidationError(f"{label} is missing: {path}")
    if path.stat().st_size != binding.get("byte_count"):
        raise ContractValidationError(f"{label} byte_count binding is stale")
    if sha256_file(path) != binding.get("sha256"):
        raise ContractValidationError(f"{label} SHA binding is stale")
    value = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if jsonl and binding.get("record_count") != len(value):
        raise ContractValidationError(f"{label} record_count binding is stale")
    return path, value


def _load_source_universe(
    universe_manifest_path: Path, *, expected_record_count: int
) -> Dict[str, Any]:
    universe_manifest_path = Path(universe_manifest_path).resolve()
    manifest = read_json(universe_manifest_path)
    if manifest.get("schema_version") != SOURCE_UNIVERSE_SCHEMA_VERSION:
        raise ContractValidationError("source universe schema_version is unsupported")
    if manifest.get("historical_exposure_contract_status") != SOURCE_CONTRACT_STATUS:
        raise ContractValidationError(
            "source universe must retain historical_exposure_contract_status=not_predeclared"
        )
    if manifest.get("universe_status") != "declared_immutable_not_reviewed":
        raise ContractValidationError("source universe status is invalid")
    for field in ("review_complete", "canonical_freeze_authorized", "split_freeze_authorized"):
        if manifest.get(field) is not False:
            raise ContractValidationError(f"source universe must keep {field}=false")
    if manifest.get("record_count") != expected_record_count:
        raise ContractValidationError(
            f"source universe has {manifest.get('record_count')} records, expected {expected_record_count}"
        )
    selection = manifest.get("selection_source")
    if not isinstance(selection, dict) or selection.get("mode") != "full_pool":
        raise ContractValidationError("source universe must declare the full pool")

    items_path, items = _validate_file_binding(
        manifest.get("items"),
        owner_path=universe_manifest_path,
        label="source universe items",
        schema_version=SOURCE_ITEM_SCHEMA_VERSION,
        jsonl=True,
    )
    universe_id = _required_string(manifest.get("universe_id"), "source universe_id")
    cohort_item_ids: List[str] = []
    source_base_fact_ids: List[str] = []
    row_hashes: List[str] = []
    for index, row in enumerate(items):
        if row.get("schema_version") != SOURCE_ITEM_SCHEMA_VERSION:
            raise ContractValidationError(f"source universe item {index} has invalid schema")
        if row.get("universe_id") != universe_id or row.get("selection_index") != index:
            raise ContractValidationError(f"source universe item {index} has stale identity")
        cohort_item_ids.append(_required_string(row.get("cohort_item_id"), "cohort_item_id"))
        source_base_fact_ids.append(
            _required_string(row.get("source_base_fact_id"), "source_base_fact_id")
        )
        row_hashes.append(sha256_value(row))
    if len(set(cohort_item_ids)) != len(cohort_item_ids):
        raise ContractValidationError("source universe repeats cohort_item_id")
    if len(set(source_base_fact_ids)) != len(source_base_fact_ids):
        raise ContractValidationError("source universe repeats source_base_fact_id")
    digest_checks = {
        "ordered_cohort_item_ids_sha256": sha256_value(cohort_item_ids),
        "ordered_source_base_fact_ids_sha256": sha256_value(source_base_fact_ids),
        "ordered_item_row_hashes_sha256": sha256_value(row_hashes),
    }
    for field, expected in digest_checks.items():
        if manifest.get(field) != expected:
            raise ContractValidationError(f"source universe {field} is stale")
    return {
        "manifest": manifest,
        "manifest_path": universe_manifest_path,
        "manifest_binding": _binding(
            universe_manifest_path, schema_version=SOURCE_UNIVERSE_SCHEMA_VERSION
        ),
        "items": items,
        "items_path": items_path,
        "items_binding": dict(manifest["items"]),
        "universe_id": universe_id,
        "cohort_item_ids": cohort_item_ids,
        "source_base_fact_ids": source_base_fact_ids,
    }


def _identifier_map(items: Sequence[Mapping[str, Any]]) -> Dict[str, set[str]]:
    result: Dict[str, set[str]] = defaultdict(set)
    for row in items:
        base_fact_id = str(row["source_base_fact_id"])
        candidates: List[Any] = [
            base_fact_id,
            row.get("cohort_item_id"),
            row.get("source_id"),
        ]
        members = row.get("member_bindings")
        if isinstance(members, list):
            candidates.extend(
                member.get("candidate_id")
                for member in members
                if isinstance(member, dict)
            )
        for value in candidates:
            if isinstance(value, str) and value:
                result[value].add(base_fact_id)
    return dict(result)


class _IdentifierMatcher:
    """Match identifiers with the legacy regex's exact selection semantics.

    The previous implementation used a length-descending bytes alternation and
    ``finditer``.  It therefore selected the leftmost match, the longest
    identifier at that start, and then resumed at the end of that non-overlap
    match.  A byte trie preserves that behavior without testing every
    alternative at every candidate position in a large payload.
    """

    __slots__ = ("_transitions", "_terminals")

    def __init__(self, identifiers: Iterable[str]) -> None:
        encoded = sorted(
            ((value.encode("utf-8"), value) for value in identifiers),
            key=lambda item: item[0],
        )
        if not encoded:
            raise ContractValidationError("source universe has no scan identifiers")
        self._transitions: List[Dict[int, int]] = [{}]
        self._terminals: List[Optional[str]] = [None]
        for raw, identifier in encoded:
            node = 0
            for byte in raw:
                child = self._transitions[node].get(byte)
                if child is None:
                    child = len(self._transitions)
                    self._transitions[node][byte] = child
                    self._transitions.append({})
                    self._terminals.append(None)
                node = child
            self._terminals[node] = identifier

    def contains(self, payload: bytes) -> bool:
        """Return whether any identifier occurs, including overlapping ones."""

        root = self._transitions[0]
        transitions = self._transitions
        terminals = self._terminals
        size = len(payload)
        start = 0
        while start < size:
            node = root.get(payload[start])
            if node is None:
                start += 1
                continue
            if terminals[node] is not None:
                return True
            index = start + 1
            while index < size:
                node = transitions[node].get(payload[index])
                if node is None:
                    break
                if terminals[node] is not None:
                    return True
                index += 1
            start += 1
        return False

    def matched_identifiers(self, payload: bytes) -> List[str]:
        """Return sorted unique legacy leftmost-longest non-overlap matches."""

        root = self._transitions[0]
        transitions = self._transitions
        terminals = self._terminals
        matched: set[str] = set()
        size = len(payload)
        start = 0
        while start < size:
            node = root.get(payload[start])
            if node is None:
                start += 1
                continue
            index = start + 1
            selected = terminals[node]
            selected_end = index if selected is not None else -1
            while index < size:
                node = transitions[node].get(payload[index])
                if node is None:
                    break
                index += 1
                terminal = terminals[node]
                if terminal is not None:
                    selected = terminal
                    selected_end = index
            if selected is None:
                start += 1
                continue
            matched.add(selected)
            start = selected_end
        return sorted(matched)


def _identifier_pattern(identifier_map: Mapping[str, set[str]]) -> _IdentifierMatcher:
    return _IdentifierMatcher(identifier_map)


def _identifier_values(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in IDENTIFIER_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value:
            values.add(value)
        elif isinstance(value, list):
            values.update(item for item in value if isinstance(item, str) and item)
    return values


def _matched_identifiers(payload: bytes, pattern: _IdentifierMatcher) -> List[str]:
    return pattern.matched_identifiers(payload)


def _json_record_payloads(
    value: Any, identifier_map: Mapping[str, set[str]]
) -> Iterator[bytes]:
    if isinstance(value, list):
        for item in value:
            yield from _json_record_payloads(item, identifier_map)
        return
    if not isinstance(value, dict):
        return
    direct = _identifier_values(value)
    keyed = {key for key in value if isinstance(key, str) and key in identifier_map}
    if direct.intersection(identifier_map) or keyed:
        yield canonical_json_bytes(value)
        return
    for child in value.values():
        yield from _json_record_payloads(child, identifier_map)


def _behavior_signatures(payload: bytes, relative_paths: Sequence[str]) -> List[str]:
    signatures = [
        name for name, marker in DIRECT_BEHAVIOR_MARKERS.items() if marker in payload
    ]
    has_variant = any(marker in payload for marker in VARIANT_MARKERS)
    has_model = any(marker in payload for marker in MODEL_MARKERS)
    has_outcome = any(marker in payload for marker in OUTCOME_MARKERS)
    if has_variant and has_model and has_outcome:
        signatures.append("variant_model_outcome")
    result_path = any(RESULT_LIKE_PATH_RE.search(path) for path in relative_paths)
    excluded = all(
        any(token in path.casefold() for token in NON_BEHAVIOR_PATH_TOKENS)
        for path in relative_paths
    )
    if result_path and not excluded and has_model and has_outcome:
        signatures.append("result_path_model_outcome")
    return sorted(set(signatures))


def _behavior_evidence(
    payload: bytes,
    *,
    relative_paths: Sequence[str],
    pattern: _IdentifierMatcher,
    identifier_map: Mapping[str, set[str]],
) -> Dict[str, Any]:
    lower_paths = [path.casefold() for path in relative_paths]
    record_payloads: List[bytes] = []
    mode = "unstructured_whole_artifact"
    if any(path.endswith((".jsonl", ".ndjson")) for path in lower_paths):
        mode = "jsonl_record_lines"
        for line in payload.splitlines():
            if not line.strip() or not pattern.contains(line):
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractValidationError(
                    f"identifier-bearing JSONL row is invalid in {relative_paths}"
                ) from exc
            if not isinstance(value, dict):
                raise ContractValidationError(
                    f"identifier-bearing JSONL row is not an object in {relative_paths}"
                )
            record_payloads.append(line)
    elif any(path.endswith(".json") for path in lower_paths):
        mode = "json_structural_records"
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractValidationError(
                f"identifier-bearing JSON is invalid in {relative_paths}"
            ) from exc
        record_payloads.extend(_json_record_payloads(value, identifier_map))
        if not record_payloads and pattern.contains(payload):
            mode = "json_unattributed_whole_artifact"
            record_payloads = [payload]
    else:
        record_payloads = [payload]

    hit_ids: set[str] = set()
    signatures: set[str] = set()
    for record in record_payloads:
        identifiers = _matched_identifiers(record, pattern)
        record_signatures = _behavior_signatures(record, relative_paths)
        if not identifiers or not record_signatures:
            continue
        signatures.update(record_signatures)
        for identifier in identifiers:
            hit_ids.update(identifier_map[identifier])
    return {
        "behavior_hit_source_base_fact_ids": sorted(hit_ids),
        "behavior_signature_names": sorted(signatures),
        "record_association_mode": mode,
        "identifier_bearing_record_count": len(record_payloads),
    }


def _scan_payload(
    payload: bytes,
    *,
    inventory_id: str,
    source_kind: str,
    relative_paths: Sequence[str],
    pattern: _IdentifierMatcher,
    identifier_map: Mapping[str, set[str]],
) -> Dict[str, Any]:
    identifiers = _matched_identifiers(payload, pattern)
    matched_ids = sorted(
        {
            base_fact_id
            for identifier in identifiers
            for base_fact_id in identifier_map[identifier]
        }
    )
    evidence = (
        _behavior_evidence(
            payload,
            relative_paths=relative_paths,
            pattern=pattern,
            identifier_map=identifier_map,
        )
        if identifiers
        else {
            "behavior_hit_source_base_fact_ids": [],
            "behavior_signature_names": [],
            "record_association_mode": "not_applicable_no_identifier_match",
            "identifier_bearing_record_count": 0,
        }
    )
    return {
        "schema_version": MATCH_SCHEMA_VERSION,
        "inventory_id": inventory_id,
        "source_kind": source_kind,
        "relative_paths": list(relative_paths),
        "matched_identifiers": identifiers,
        "matched_source_base_fact_ids": matched_ids,
        **evidence,
        "historical_behavior_exposure_candidate_detected": bool(
            evidence["behavior_hit_source_base_fact_ids"]
        ),
        "scan_status": "completed_exact_identifier_and_behavior_signature_scan",
    }


def _worktree_regular_files(repo_root: Path) -> List[Path]:
    repo_root = Path(repo_root).resolve()
    result: List[Path] = []
    for current_root, directory_names, file_names in os.walk(repo_root):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        directory_names[:] = [name for name in directory_names if name != ".git"]
        for name in directory_names:
            if (current / name).is_symlink():
                raise ContractValidationError(
                    f"repository scan refuses symlinked directory: {current / name}"
                )
        for name in file_names:
            path = current / name
            if path.is_symlink():
                raise ContractValidationError(
                    f"repository scan refuses symlinked file: {path}"
                )
            if path.is_file():
                result.append(path.resolve())
    return sorted(result, key=lambda path: str(path.relative_to(repo_root)))


def _reachable_git_blobs(repo_root: Path) -> List[Dict[str, Any]]:
    commits = sorted(
        line.decode("ascii")
        for line in _git(repo_root, "rev-list", "--all").splitlines()
        if line
    )
    if not commits:
        raise ContractValidationError("repository has no reachable Git commits")
    by_oid: Dict[str, Dict[str, set[str]]] = defaultdict(
        lambda: {"relative_paths": set(), "reachable_commits": set(), "modes": set()}
    )
    for commit in commits:
        for entry in _git(repo_root, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            if object_type != "blob":
                continue
            row = by_oid[object_id]
            row["relative_paths"].add(raw_path.decode("utf-8"))
            row["reachable_commits"].add(commit)
            row["modes"].add(mode)
    return [
        {
            "git_blob_oid": oid,
            "relative_paths": sorted(row["relative_paths"]),
            "reachable_commits": sorted(row["reachable_commits"]),
            "modes": sorted(row["modes"]),
        }
        for oid, row in sorted(by_oid.items())
    ]


def _git_blob_payloads(
    repo_root: Path, sources: Sequence[Mapping[str, Any]]
) -> Iterator[Tuple[Mapping[str, Any], bytes]]:
    process = subprocess.Popen(
        ["git", "-C", str(repo_root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for source in sources:
            process.stdin.write(
                str(source["git_blob_oid"]).encode("ascii") + b"\n"
            )
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n")
            try:
                object_id, object_type, raw_size = header.split()
                size = int(raw_size)
            except (ValueError, TypeError) as exc:
                raise ContractValidationError(
                    f"cannot parse git cat-file header for {source['git_blob_oid']}"
                ) from exc
            if object_id.decode("ascii") != source["git_blob_oid"] or object_type != b"blob":
                raise ContractValidationError("git cat-file returned an unexpected object")
            payload = process.stdout.read(size)
            if len(payload) != size or process.stdout.read(1) != b"\n":
                raise ContractValidationError("git cat-file returned a truncated blob")
            yield source, payload
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            raise subprocess.CalledProcessError(return_code, process.args, stderr=stderr)
    finally:
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def _scan_repository(
    repo_root: Path, identifier_map: Mapping[str, set[str]]
) -> Dict[str, Any]:
    pattern = _identifier_pattern(identifier_map)
    inventory: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    worktree_files = _worktree_regular_files(repo_root)
    for path in worktree_files:
        relative_path = str(path.relative_to(repo_root))
        payload = path.read_bytes()
        content_sha256 = hashlib.sha256(payload).hexdigest()
        inventory_id = "worktree_" + sha256_value(
            {"relative_path": relative_path, "content_sha256": content_sha256}
        )[:24]
        match = _scan_payload(
            payload,
            inventory_id=inventory_id,
            source_kind="current_worktree_file",
            relative_paths=[relative_path],
            pattern=pattern,
            identifier_map=identifier_map,
        )
        inventory.append(
            {
                "schema_version": INVENTORY_SCHEMA_VERSION,
                "inventory_id": inventory_id,
                "source_kind": "current_worktree_file",
                "relative_paths": [relative_path],
                "byte_count": len(payload),
                "content_sha256": content_sha256,
                "identifier_match_count": len(match["matched_identifiers"]),
                "matched_source_base_fact_count": len(
                    match["matched_source_base_fact_ids"]
                ),
                "behavior_candidate_base_fact_count": len(
                    match["behavior_hit_source_base_fact_ids"]
                ),
                "scan_status": match["scan_status"],
            }
        )
        if match["matched_identifiers"]:
            matches.append(match)

    git_blobs = _reachable_git_blobs(repo_root)
    for source, payload in _git_blob_payloads(repo_root, git_blobs):
        content_sha256 = hashlib.sha256(payload).hexdigest()
        inventory_id = "gitblob_" + sha256_value(
            {"git_blob_oid": source["git_blob_oid"], "content_sha256": content_sha256}
        )[:24]
        match = _scan_payload(
            payload,
            inventory_id=inventory_id,
            source_kind="reachable_git_blob",
            relative_paths=source["relative_paths"],
            pattern=pattern,
            identifier_map=identifier_map,
        )
        inventory.append(
            {
                "schema_version": INVENTORY_SCHEMA_VERSION,
                "inventory_id": inventory_id,
                "source_kind": "reachable_git_blob",
                "git_blob_oid": source["git_blob_oid"],
                "relative_paths": source["relative_paths"],
                "reachable_commits": source["reachable_commits"],
                "modes": source["modes"],
                "byte_count": len(payload),
                "content_sha256": content_sha256,
                "identifier_match_count": len(match["matched_identifiers"]),
                "matched_source_base_fact_count": len(
                    match["matched_source_base_fact_ids"]
                ),
                "behavior_candidate_base_fact_count": len(
                    match["behavior_hit_source_base_fact_ids"]
                ),
                "scan_status": match["scan_status"],
            }
        )
        if match["matched_identifiers"]:
            matches.append(match)
    inventory.sort(key=lambda row: str(row["inventory_id"]))
    matches.sort(key=lambda row: str(row["inventory_id"]))
    candidate_ids = sorted(
        {
            value
            for row in matches
            for value in row["behavior_hit_source_base_fact_ids"]
        }
    )
    return {
        "inventory": inventory,
        "matches": matches,
        "worktree_file_count": len(worktree_files),
        "reachable_git_blob_count": len(git_blobs),
        "identifier_matching_artifact_count": len(matches),
        "behavior_candidate_source_base_fact_ids": candidate_ids,
    }


def _challenge(
    *,
    universe: Mapping[str, Any],
    repository_inventory_sha256: str,
    repository_matches_sha256: str,
    repository_snapshot: Mapping[str, Any],
    contract_created_at: str,
) -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "contract_created_at": contract_created_at,
        "source_universe_manifest_sha256": universe["manifest_binding"]["sha256"],
        "source_universe_items_sha256": universe["items_binding"]["sha256"],
        "universe_id": universe["universe_id"],
        "record_count": len(universe["items"]),
        "ordered_source_base_fact_ids_sha256": sha256_value(
            universe["source_base_fact_ids"]
        ),
        "repository_inventory_sha256": repository_inventory_sha256,
        "repository_matches_sha256": repository_matches_sha256,
        "repository_git_head": repository_snapshot["git_head"],
        "repository_refs_sha256": repository_snapshot["refs_sha256"],
        "repository_snapshot_sha256": sha256_value(repository_snapshot),
        "authority_required_from": "scope_owner_explicit_confirmation",
        "required_scope_kinds": list(SCOPE_KINDS),
        "known_exposure_identity": "source_base_fact_id",
    }


def _attestation_template(
    *,
    contract_id: str,
    challenge: Mapping[str, Any],
    behavior_candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    adjudications = []
    for row in behavior_candidates:
        for base_fact_id in row["behavior_hit_source_base_fact_ids"]:
            adjudications.append(
                {
                    "inventory_id": row["inventory_id"],
                    "source_base_fact_id": base_fact_id,
                    "detected_signature_names": row["behavior_signature_names"],
                    "decision": None,
                    "rationale": None,
                }
            )
    adjudications.sort(key=lambda row: (row["inventory_id"], row["source_base_fact_id"]))
    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "status": "pending_user_completion",
        "attestation_source": None,
        "attester_role": None,
        "attested_by": None,
        "authority_basis": None,
        "attested_at": None,
        "attested_through": None,
        "bindings": {
            "contract_id": contract_id,
            "attestation_challenge_sha256": sha256_value(challenge),
            "source_universe_manifest_sha256": challenge[
                "source_universe_manifest_sha256"
            ],
            "source_universe_items_sha256": challenge["source_universe_items_sha256"],
            "repository_inventory_sha256": challenge["repository_inventory_sha256"],
            "repository_matches_sha256": challenge["repository_matches_sha256"],
            "repository_git_head": challenge["repository_git_head"],
            "repository_refs_sha256": challenge["repository_refs_sha256"],
        },
        "statements": {field: None for field in REQUIRED_STATEMENTS},
        "scope_inventory": {
            kind: {"inventory_complete": None, "sources": None}
            for kind in SCOPE_KINDS
        },
        "repository_hit_adjudications": adjudications,
        "known_historical_exposure_source_base_fact_ids": None,
        "limitations_acknowledged": {
            field: None for field in REQUIRED_LIMITATION_ACKNOWLEDGEMENTS
        },
        "completion_instructions": [
            "Copy this template to a new path; do not edit the bound template in place.",
            "Only the scope owner may set status=affirmed_by_scope_owner and identify their authority basis.",
            "Set every statement, inventory_complete field, and limitation acknowledgement explicitly to true.",
            "Use an explicit empty sources list only after confirming that a scope has no sources.",
            "Adjudicate every repository behavior candidate; disclose confirmed and external exposures by source_base_fact_id.",
            "Run validate immediately before successor-universe/freeze binding; later behavior activity requires a refreshed attestation.",
        ],
    }


def materialize_pending_contract(
    *,
    universe_manifest_path: Path,
    repo_root: Path,
    output_dir: Path,
    expected_record_count: int = EXPECTED_RECORD_COUNT,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    repo_root = Path(repo_root).resolve()
    git_toplevel = Path(_git(repo_root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if git_toplevel != repo_root:
        raise ContractValidationError("repo_root is not the Git toplevel")
    universe = _load_source_universe(
        universe_manifest_path, expected_record_count=expected_record_count
    )
    repository_scan = _scan_repository(repo_root, _identifier_map(universe["items"]))
    git_head = _git(repo_root, "rev-parse", "HEAD").decode().strip()
    refs_payload = _git(
        repo_root, "for-each-ref", "--format=%(refname)%00%(objectname)"
    )
    status_payload = _git(repo_root, "status", "--porcelain=v1", "-z")
    repository_snapshot = {
        "repo_root": str(repo_root),
        "git_head": git_head,
        "refs_sha256": hashlib.sha256(refs_payload).hexdigest(),
        "reachable_commit_count": int(
            _git(repo_root, "rev-list", "--all", "--count").decode().strip()
        ),
        "dirty_status_sha256": hashlib.sha256(status_payload).hexdigest(),
        "dirty_entry_count": len([part for part in status_payload.split(b"\0") if part]),
    }
    created_at = _utc_now()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent))
    )
    try:
        staged_inventory = staged_dir / INVENTORY_NAME
        staged_matches = staged_dir / MATCHES_NAME
        staged_template = staged_dir / ATTESTATION_TEMPLATE_NAME
        staged_contract = staged_dir / CONTRACT_NAME
        staged_inventory.write_bytes(_jsonl_payload(repository_scan["inventory"]))
        staged_matches.write_bytes(_jsonl_payload(repository_scan["matches"]))
        inventory_binding = _binding(
            staged_inventory,
            schema_version=INVENTORY_SCHEMA_VERSION,
            record_count=len(repository_scan["inventory"]),
            output_path=output_dir / INVENTORY_NAME,
        )
        matches_binding = _binding(
            staged_matches,
            schema_version=MATCH_SCHEMA_VERSION,
            record_count=len(repository_scan["matches"]),
            output_path=output_dir / MATCHES_NAME,
        )
        challenge = _challenge(
            universe=universe,
            repository_inventory_sha256=inventory_binding["sha256"],
            repository_matches_sha256=matches_binding["sha256"],
            repository_snapshot=repository_snapshot,
            contract_created_at=created_at,
        )
        contract_id = "historical_exposure_contract_" + sha256_value(challenge)[:24]
        behavior_candidate_rows = [
            row
            for row in repository_scan["matches"]
            if row["behavior_hit_source_base_fact_ids"]
        ]
        template = _attestation_template(
            contract_id=contract_id,
            challenge=challenge,
            behavior_candidates=behavior_candidate_rows,
        )
        staged_template.write_bytes(_json_payload(template))
        template_binding = _binding(
            staged_template,
            schema_version=ATTESTATION_SCHEMA_VERSION,
            output_path=output_dir / ATTESTATION_TEMPLATE_NAME,
        )
        blockers = [
            "scope_owner_identity_and_authority_not_confirmed",
            "external_directories_not_attested",
            "external_platforms_not_attested",
            "separately_executed_runs_not_attested",
            "other_behavior_sources_not_attested",
            "known_historical_exposure_registry_not_attested",
            "final_split_not_bound",
        ]
        if behavior_candidate_rows:
            blockers.append("repository_behavior_candidates_not_adjudicated")
        contract = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "policy_version": POLICY_VERSION,
            "contract_id": contract_id,
            "created_at": created_at,
            "status": PENDING_STATUS,
            "source_universe_v1": {
                **universe["manifest_binding"],
                "universe_id": universe["universe_id"],
                "record_count": len(universe["items"]),
                "ordered_source_base_fact_ids_sha256": sha256_value(
                    universe["source_base_fact_ids"]
                ),
                "historical_exposure_contract_status": SOURCE_CONTRACT_STATUS,
            },
            "source_universe_items": universe["items_binding"],
            "predeclaration": {
                "scope": "entire_immutable_formal_universe_before_final_split",
                "source_universe_split_freeze_authorized_at_materialization": False,
                "scope_owner_attestation_required": True,
                "future_successor_universe_must_bind_contract_and_attestation_sha256": True,
                "exposed_items_must_not_enter_validation_or_sealed": True,
                "attestation_must_cover_activity_through_final_freeze": True,
            },
            "repository_snapshot": repository_snapshot,
            "repository_scan": {
                "policy": "exact-universe-identifiers-plus-record-level-behavior-signatures-v2",
                "current_worktree_complete": True,
                "reachable_git_blobs_complete": True,
                "semantic_recall_guaranteed": False,
                "worktree_file_count": repository_scan["worktree_file_count"],
                "reachable_git_blob_count": repository_scan[
                    "reachable_git_blob_count"
                ],
                "identifier_matching_artifact_count": repository_scan[
                    "identifier_matching_artifact_count"
                ],
                "behavior_candidate_base_fact_count": len(
                    repository_scan["behavior_candidate_source_base_fact_ids"]
                ),
                "behavior_candidate_source_base_fact_ids": repository_scan[
                    "behavior_candidate_source_base_fact_ids"
                ],
                "inventory": inventory_binding,
                "matches": matches_binding,
            },
            "attestation_challenge": challenge,
            "attestation_challenge_sha256": sha256_value(challenge),
            "scope_owner_attestation_template": template_binding,
            "authority_boundary": {
                "repository_evidence_generated_by_tool": True,
                "external_scope_claims_generated_by_tool": False,
                "scope_owner_identity_generated_by_tool": False,
                "known_exposure_registry_generated_by_tool": False,
                "template_is_attestation": False,
            },
            "execution_safety": {
                "network_used": False,
                "model_inference_used": False,
                "hf_checkpoint_loaded": False,
                "tokenizer_loaded": False,
                "behavior_run_performed": False,
                "validation_or_sealed_exposure_assessed": False,
            },
            "known_blockers": blockers,
            "historical_exposure_complete": False,
            "review_freeze_authorized": False,
            "split_freeze_authorized": False,
            "exact_hf_authorized": False,
            "behavior_authorized": False,
        }
        staged_contract.write_bytes(_json_payload(contract))
        if os.path.lexists(output_dir):
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        os.rename(staged_dir, output_dir)
        staged_dir = None
    finally:
        if staged_dir is not None and staged_dir.exists():
            shutil.rmtree(staged_dir)

    contract_path = output_dir / CONTRACT_NAME
    return {
        "status": PENDING_STATUS,
        "contract_id": contract_id,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "attestation_template_path": str(output_dir / ATTESTATION_TEMPLATE_NAME),
        "record_count": len(universe["items"]),
        "repository_behavior_candidate_base_fact_count": len(
            repository_scan["behavior_candidate_source_base_fact_ids"]
        ),
        "historical_exposure_complete": False,
        "split_freeze_authorized": False,
        "exact_hf_authorized": False,
    }


def _validate_pending_packet(contract_path: Path) -> Dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    contract = read_json(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractValidationError("authority contract schema_version is unsupported")
    if contract.get("tool_version") != TOOL_VERSION or contract.get("policy_version") != POLICY_VERSION:
        raise ContractValidationError("authority contract tool/policy version is unsupported")
    if contract.get("status") != PENDING_STATUS:
        raise ContractValidationError("authority contract must remain pending")
    for field in (
        "historical_exposure_complete",
        "review_freeze_authorized",
        "split_freeze_authorized",
        "exact_hf_authorized",
        "behavior_authorized",
    ):
        if contract.get(field) is not False:
            raise ContractValidationError(f"pending authority contract must keep {field}=false")
    created_at = _parse_timestamp(contract.get("created_at"), "contract.created_at")

    source_binding = contract.get("source_universe_v1")
    if not isinstance(source_binding, dict):
        raise ContractValidationError("source_universe_v1 binding is missing")
    if source_binding.get("schema_version") != SOURCE_UNIVERSE_SCHEMA_VERSION:
        raise ContractValidationError("source universe schema binding is invalid")
    if source_binding.get("historical_exposure_contract_status") != SOURCE_CONTRACT_STATUS:
        raise ContractValidationError("source universe contract status binding is invalid")
    universe_path = _resolve_binding_path(source_binding, contract_path)
    if universe_path.stat().st_size != source_binding.get("byte_count"):
        raise ContractValidationError("source universe manifest byte_count binding is stale")
    if sha256_file(universe_path) != source_binding.get("sha256"):
        raise ContractValidationError("source universe manifest SHA binding is stale")
    expected_count = source_binding.get("record_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 1:
        raise ContractValidationError("source universe record_count is invalid")
    universe = _load_source_universe(
        universe_path, expected_record_count=expected_count
    )
    if universe["universe_id"] != source_binding.get("universe_id"):
        raise ContractValidationError("source universe_id binding is stale")
    if universe["manifest_binding"]["sha256"] != source_binding.get("sha256"):
        raise ContractValidationError("source universe manifest binding is stale")
    if contract.get("source_universe_items") != universe["items_binding"]:
        raise ContractValidationError("source universe items binding is stale")

    expected_predeclaration = {
        "scope": "entire_immutable_formal_universe_before_final_split",
        "source_universe_split_freeze_authorized_at_materialization": False,
        "scope_owner_attestation_required": True,
        "future_successor_universe_must_bind_contract_and_attestation_sha256": True,
        "exposed_items_must_not_enter_validation_or_sealed": True,
        "attestation_must_cover_activity_through_final_freeze": True,
    }
    if contract.get("predeclaration") != expected_predeclaration:
        raise ContractValidationError("authority predeclaration is invalid")
    expected_authority_boundary = {
        "repository_evidence_generated_by_tool": True,
        "external_scope_claims_generated_by_tool": False,
        "scope_owner_identity_generated_by_tool": False,
        "known_exposure_registry_generated_by_tool": False,
        "template_is_attestation": False,
    }
    if contract.get("authority_boundary") != expected_authority_boundary:
        raise ContractValidationError("authority boundary is invalid")
    expected_execution_safety = {
        "network_used": False,
        "model_inference_used": False,
        "hf_checkpoint_loaded": False,
        "tokenizer_loaded": False,
        "behavior_run_performed": False,
        "validation_or_sealed_exposure_assessed": False,
    }
    if contract.get("execution_safety") != expected_execution_safety:
        raise ContractValidationError("execution safety contract is invalid")

    scan = contract.get("repository_scan")
    if not isinstance(scan, dict):
        raise ContractValidationError("repository_scan is missing")
    if scan.get("policy") != (
        "exact-universe-identifiers-plus-record-level-behavior-signatures-v2"
    ):
        raise ContractValidationError("repository scan policy is invalid")
    for field in ("current_worktree_complete", "reachable_git_blobs_complete"):
        if scan.get(field) is not True:
            raise ContractValidationError(f"repository scan must keep {field}=true")
    if scan.get("semantic_recall_guaranteed") is not False:
        raise ContractValidationError(
            "repository scan cannot claim guaranteed semantic recall"
        )
    inventory_path, inventory = _validate_file_binding(
        scan.get("inventory"),
        owner_path=contract_path,
        label="repository inventory",
        schema_version=INVENTORY_SCHEMA_VERSION,
        jsonl=True,
    )
    matches_path, matches = _validate_file_binding(
        scan.get("matches"),
        owner_path=contract_path,
        label="repository matches",
        schema_version=MATCH_SCHEMA_VERSION,
        jsonl=True,
        allow_empty=True,
    )
    universe_ids = set(universe["source_base_fact_ids"])
    inventory_by_id: Dict[str, Dict[str, Any]] = {}
    expected_match_ids: set[str] = set()
    source_counts = {"current_worktree_file": 0, "reachable_git_blob": 0}
    for index, row in enumerate(inventory, start=1):
        if row.get("schema_version") != INVENTORY_SCHEMA_VERSION:
            raise ContractValidationError(f"repository inventory row {index} has invalid schema")
        inventory_id = _required_string(row.get("inventory_id"), "inventory_id")
        if inventory_id in inventory_by_id:
            raise ContractValidationError(f"duplicate inventory_id: {inventory_id}")
        source_kind = row.get("source_kind")
        if source_kind not in source_counts:
            raise ContractValidationError(f"unsupported repository source_kind: {source_kind}")
        source_counts[source_kind] += 1
        if row.get("scan_status") != (
            "completed_exact_identifier_and_behavior_signature_scan"
        ):
            raise ContractValidationError(
                f"repository inventory scan is incomplete: {inventory_id}"
            )
        for field in (
            "identifier_match_count",
            "matched_source_base_fact_count",
            "behavior_candidate_base_fact_count",
        ):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractValidationError(f"repository inventory {field} is invalid")
        if row["identifier_match_count"]:
            expected_match_ids.add(inventory_id)
        inventory_by_id[inventory_id] = row
    match_ids: set[str] = set()
    candidate_ids: set[str] = set()
    behavior_candidate_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(matches, start=1):
        if row.get("schema_version") != MATCH_SCHEMA_VERSION:
            raise ContractValidationError(f"repository match row {index} has invalid schema")
        inventory_id = _required_string(row.get("inventory_id"), "match.inventory_id")
        if inventory_id in match_ids or inventory_id not in inventory_by_id:
            raise ContractValidationError(f"repository match coverage is invalid: {inventory_id}")
        if row.get("source_kind") != inventory_by_id[inventory_id].get("source_kind"):
            raise ContractValidationError(f"repository match source_kind is stale: {inventory_id}")
        matched_ids = row.get("matched_source_base_fact_ids")
        hit_ids = row.get("behavior_hit_source_base_fact_ids")
        identifiers = row.get("matched_identifiers")
        signatures = row.get("behavior_signature_names")
        if (
            not isinstance(matched_ids, list)
            or not isinstance(hit_ids, list)
            or not isinstance(identifiers, list)
            or not isinstance(signatures, list)
        ):
            raise ContractValidationError(f"repository match IDs are invalid: {inventory_id}")
        for label, values in (
            ("matched_identifiers", identifiers),
            ("matched_source_base_fact_ids", matched_ids),
            ("behavior_hit_source_base_fact_ids", hit_ids),
            ("behavior_signature_names", signatures),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ContractValidationError(
                    f"repository match {label} contains an invalid value: {inventory_id}"
                )
            if values != sorted(set(values)):
                raise ContractValidationError(
                    f"repository match {label} must be sorted and unique: {inventory_id}"
                )
        if not set(matched_ids).issubset(universe_ids) or not set(hit_ids).issubset(set(matched_ids)):
            raise ContractValidationError(f"repository match references unknown facts: {inventory_id}")
        inventory_row = inventory_by_id[inventory_id]
        if inventory_row["identifier_match_count"] != len(identifiers):
            raise ContractValidationError(
                f"repository identifier match count is stale: {inventory_id}"
            )
        if inventory_row["matched_source_base_fact_count"] != len(matched_ids):
            raise ContractValidationError(f"repository match count is stale: {inventory_id}")
        if inventory_row["behavior_candidate_base_fact_count"] != len(hit_ids):
            raise ContractValidationError(f"repository behavior count is stale: {inventory_id}")
        if row.get("historical_behavior_exposure_candidate_detected") is not bool(hit_ids):
            raise ContractValidationError(
                f"repository behavior candidate flag is stale: {inventory_id}"
            )
        if row.get("scan_status") != (
            "completed_exact_identifier_and_behavior_signature_scan"
        ):
            raise ContractValidationError(
                f"repository match scan is incomplete: {inventory_id}"
            )
        candidate_ids.update(hit_ids)
        if hit_ids:
            behavior_candidate_rows.append(row)
        match_ids.add(inventory_id)
    if match_ids != expected_match_ids:
        raise ContractValidationError("repository matches do not cover identifier-bearing inventory")
    if scan.get("worktree_file_count") != source_counts["current_worktree_file"]:
        raise ContractValidationError("repository worktree file count is stale")
    if scan.get("reachable_git_blob_count") != source_counts["reachable_git_blob"]:
        raise ContractValidationError("repository Git blob count is stale")
    if scan.get("identifier_matching_artifact_count") != len(matches):
        raise ContractValidationError("repository identifier match count is stale")
    if scan.get("behavior_candidate_source_base_fact_ids") != sorted(candidate_ids):
        raise ContractValidationError("repository behavior candidate IDs are stale")
    if scan.get("behavior_candidate_base_fact_count") != len(candidate_ids):
        raise ContractValidationError("repository behavior candidate count is stale")

    repository_snapshot = contract.get("repository_snapshot")
    if not isinstance(repository_snapshot, dict) or set(repository_snapshot) != {
        "repo_root",
        "git_head",
        "refs_sha256",
        "reachable_commit_count",
        "dirty_status_sha256",
        "dirty_entry_count",
    }:
        raise ContractValidationError("repository snapshot fields are invalid")
    _required_string(repository_snapshot.get("repo_root"), "repository_snapshot.repo_root")
    _required_string(repository_snapshot.get("git_head"), "repository_snapshot.git_head")
    _required_sha256(repository_snapshot.get("refs_sha256"), "repository_snapshot.refs_sha256")
    _required_sha256(
        repository_snapshot.get("dirty_status_sha256"),
        "repository_snapshot.dirty_status_sha256",
    )
    for field in ("reachable_commit_count", "dirty_entry_count"):
        value = repository_snapshot.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContractValidationError(f"repository_snapshot.{field} is invalid")

    challenge = contract.get("attestation_challenge")
    if not isinstance(challenge, dict):
        raise ContractValidationError("attestation challenge is missing")
    expected_challenge = _challenge(
        universe=universe,
        repository_inventory_sha256=sha256_file(inventory_path),
        repository_matches_sha256=sha256_file(matches_path),
        repository_snapshot=repository_snapshot,
        contract_created_at=contract["created_at"],
    )
    if challenge != expected_challenge:
        raise ContractValidationError("attestation challenge is stale")
    challenge_sha = sha256_value(challenge)
    if contract.get("attestation_challenge_sha256") != challenge_sha:
        raise ContractValidationError("attestation challenge SHA is stale")
    contract_id = "historical_exposure_contract_" + challenge_sha[:24]
    if contract.get("contract_id") != contract_id:
        raise ContractValidationError("contract_id is stale")
    template_path, template = _validate_file_binding(
        contract.get("scope_owner_attestation_template"),
        owner_path=contract_path,
        label="scope-owner attestation template",
        schema_version=ATTESTATION_SCHEMA_VERSION,
    )
    expected_template = _attestation_template(
        contract_id=contract_id,
        challenge=challenge,
        behavior_candidates=behavior_candidate_rows,
    )
    if template != expected_template:
        raise ContractValidationError("scope-owner attestation template was edited or is stale")
    expected_blockers = [
        "scope_owner_identity_and_authority_not_confirmed",
        "external_directories_not_attested",
        "external_platforms_not_attested",
        "separately_executed_runs_not_attested",
        "other_behavior_sources_not_attested",
        "known_historical_exposure_registry_not_attested",
        "final_split_not_bound",
    ]
    if behavior_candidate_rows:
        expected_blockers.append("repository_behavior_candidates_not_adjudicated")
    if contract.get("known_blockers") != expected_blockers:
        raise ContractValidationError("pending authority blockers are stale")
    return {
        "contract": contract,
        "contract_path": contract_path,
        "created_at": created_at,
        "universe": universe,
        "inventory": inventory,
        "matches": matches,
        "behavior_candidate_rows": behavior_candidate_rows,
        "template_path": template_path,
        "template": template,
    }


def _validate_scope_sources(
    value: Any,
    *,
    scope_kind: str,
    universe_ids: set[str],
    contract_created_at: datetime,
) -> set[str]:
    if not isinstance(value, dict) or set(value) != {"inventory_complete", "sources"}:
        raise ContractValidationError(f"scope_inventory.{scope_kind} fields are invalid")
    if value.get("inventory_complete") is not True:
        raise ContractValidationError(
            f"scope_inventory.{scope_kind}.inventory_complete must be explicitly true"
        )
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise ContractValidationError(
            f"scope_inventory.{scope_kind}.sources must be an explicit list"
        )
    source_ids: set[str] = set()
    exposed: set[str] = set()
    required_fields = {
        "source_id",
        "scope_description",
        "review_method",
        "reviewed_through",
        "evidence_reference",
        "evidence_sha256",
        "known_exposed_source_base_fact_ids",
    }
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict) or set(source) != required_fields:
            raise ContractValidationError(
                f"scope_inventory.{scope_kind}.sources[{index}] fields are invalid"
            )
        source_id = _required_string(source.get("source_id"), "external source_id")
        if source_id in source_ids:
            raise ContractValidationError(f"duplicate external source_id: {source_id}")
        source_ids.add(source_id)
        _required_string(source.get("scope_description"), "external scope_description")
        _required_string(source.get("review_method"), "external review_method")
        reviewed_through = _parse_timestamp(
            source.get("reviewed_through"), "external reviewed_through"
        )
        if reviewed_through < contract_created_at:
            raise ContractValidationError(
                f"external source review predates the contract: {source_id}"
            )
        _required_string(source.get("evidence_reference"), "external evidence_reference")
        evidence_sha = source.get("evidence_sha256")
        if evidence_sha is not None:
            _required_sha256(evidence_sha, "external evidence_sha256")
        ids = source.get("known_exposed_source_base_fact_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ContractValidationError(
                f"external exposure IDs are invalid: {source_id}"
            )
        if ids != sorted(set(ids)):
            raise ContractValidationError(
                f"external exposure IDs must be sorted and unique: {source_id}"
            )
        if not set(ids).issubset(universe_ids):
            raise ContractValidationError(
                f"external exposure IDs are outside the universe: {source_id}"
            )
        exposed.update(ids)
    return exposed


def _validate_attestation(packet: Mapping[str, Any], attestation_path: Path) -> Dict[str, Any]:
    attestation_path = Path(attestation_path).resolve()
    value = read_json(attestation_path)
    expected_top_fields = {
        "schema_version",
        "status",
        "attestation_source",
        "attester_role",
        "attested_by",
        "authority_basis",
        "attested_at",
        "attested_through",
        "bindings",
        "statements",
        "scope_inventory",
        "repository_hit_adjudications",
        "known_historical_exposure_source_base_fact_ids",
        "limitations_acknowledged",
        "completion_instructions",
    }
    if set(value) != expected_top_fields:
        raise ContractValidationError("scope-owner attestation fields are invalid")
    if value.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise ContractValidationError("scope-owner attestation schema_version is unsupported")
    if value.get("status") != AFFIRMED_STATUS:
        raise ContractValidationError("scope-owner attestation is not explicitly affirmed")
    if value.get("attestation_source") != "user_explicit_confirmation":
        raise ContractValidationError(
            "attestation_source must be user_explicit_confirmation"
        )
    if value.get("attester_role") != "scope_owner":
        raise ContractValidationError("attester_role must be scope_owner")
    _required_string(value.get("attested_by"), "attested_by")
    _required_string(value.get("authority_basis"), "authority_basis")
    attested_at = _parse_timestamp(value.get("attested_at"), "attested_at")
    attested_through = _parse_timestamp(value.get("attested_through"), "attested_through")
    if attested_at < packet["created_at"] or attested_through < packet["created_at"]:
        raise ContractValidationError("scope-owner attestation predates the contract packet")
    if attested_through > attested_at:
        raise ContractValidationError("attested_through cannot be later than attested_at")
    if attested_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ContractValidationError("attested_at cannot be materially in the future")

    challenge = packet["contract"]["attestation_challenge"]
    expected_bindings = packet["template"]["bindings"]
    if value.get("bindings") != expected_bindings:
        raise ContractValidationError("scope-owner attestation bindings are stale")
    if value.get("completion_instructions") != packet["template"].get(
        "completion_instructions"
    ):
        raise ContractValidationError("scope-owner completion instructions are stale")
    statements = value.get("statements")
    if not isinstance(statements, dict) or set(statements) != set(REQUIRED_STATEMENTS):
        raise ContractValidationError("scope-owner attestation statements are invalid")
    for field in REQUIRED_STATEMENTS:
        if statements.get(field) is not True:
            raise ContractValidationError(f"statement {field} must be explicitly true")
    limitations = value.get("limitations_acknowledged")
    if not isinstance(limitations, dict) or set(limitations) != set(
        REQUIRED_LIMITATION_ACKNOWLEDGEMENTS
    ):
        raise ContractValidationError("limitation acknowledgements are invalid")
    for field in REQUIRED_LIMITATION_ACKNOWLEDGEMENTS:
        if limitations.get(field) is not True:
            raise ContractValidationError(
                f"limitation acknowledgement {field} must be explicitly true"
            )

    universe_ids = set(packet["universe"]["source_base_fact_ids"])
    scope_inventory = value.get("scope_inventory")
    if not isinstance(scope_inventory, dict) or set(scope_inventory) != set(SCOPE_KINDS):
        raise ContractValidationError("scope_inventory is incomplete")
    external_exposures: set[str] = set()
    for scope_kind in SCOPE_KINDS:
        external_exposures.update(
            _validate_scope_sources(
                scope_inventory[scope_kind],
                scope_kind=scope_kind,
                universe_ids=universe_ids,
                contract_created_at=packet["created_at"],
            )
        )

    expected_pairs: Dict[Tuple[str, str], List[str]] = {}
    for row in packet["behavior_candidate_rows"]:
        for base_fact_id in row["behavior_hit_source_base_fact_ids"]:
            expected_pairs[(row["inventory_id"], base_fact_id)] = row[
                "behavior_signature_names"
            ]
    adjudications = value.get("repository_hit_adjudications")
    if not isinstance(adjudications, list):
        raise ContractValidationError("repository_hit_adjudications must be a list")
    actual_pairs: set[Tuple[str, str]] = set()
    repository_exposures: set[str] = set()
    for index, row in enumerate(adjudications, start=1):
        if not isinstance(row, dict) or set(row) != {
            "inventory_id",
            "source_base_fact_id",
            "detected_signature_names",
            "decision",
            "rationale",
        }:
            raise ContractValidationError(
                f"repository hit adjudication {index} fields are invalid"
            )
        pair = (
            _required_string(row.get("inventory_id"), "adjudication.inventory_id"),
            _required_string(
                row.get("source_base_fact_id"), "adjudication.source_base_fact_id"
            ),
        )
        if pair in actual_pairs or pair not in expected_pairs:
            raise ContractValidationError(
                f"repository hit adjudication coverage is invalid: {pair}"
            )
        actual_pairs.add(pair)
        if row.get("detected_signature_names") != expected_pairs[pair]:
            raise ContractValidationError(
                f"repository hit adjudication signatures are stale: {pair}"
            )
        decision = row.get("decision")
        if decision not in {"confirmed_exposure", "false_positive_not_behavior_exposure"}:
            raise ContractValidationError(
                f"repository hit adjudication decision is invalid: {pair}"
            )
        _required_string(row.get("rationale"), "repository hit adjudication rationale")
        if decision == "confirmed_exposure":
            repository_exposures.add(pair[1])
    if actual_pairs != set(expected_pairs):
        raise ContractValidationError("repository behavior candidates are not fully adjudicated")

    declared_exposures = value.get("known_historical_exposure_source_base_fact_ids")
    if not isinstance(declared_exposures, list) or any(
        not isinstance(item, str) for item in declared_exposures
    ):
        raise ContractValidationError(
            "known_historical_exposure_source_base_fact_ids must be an explicit list"
        )
    if declared_exposures != sorted(set(declared_exposures)):
        raise ContractValidationError("known historical exposure IDs must be sorted and unique")
    if not set(declared_exposures).issubset(universe_ids):
        raise ContractValidationError("known historical exposure IDs are outside the universe")
    expected_exposures = sorted(repository_exposures | external_exposures)
    if declared_exposures != expected_exposures:
        raise ContractValidationError(
            "known historical exposures do not equal repository plus external disclosures"
        )
    return {
        "status": "valid_scope_owner_attestation_for_successor_binding",
        "contract_id": packet["contract"]["contract_id"],
        "contract_path": str(packet["contract_path"]),
        "contract_sha256": sha256_file(packet["contract_path"]),
        "attestation_path": str(attestation_path),
        "attestation_sha256": sha256_file(attestation_path),
        "attested_by": value["attested_by"],
        "attested_at": value["attested_at"],
        "attested_through": value["attested_through"],
        "known_historical_exposure_count": len(declared_exposures),
        "known_historical_exposure_source_base_fact_ids": declared_exposures,
        "scope_owner_attestation_valid": True,
        "ready_for_successor_universe_binding": True,
        "historical_exposure_complete": False,
        "review_freeze_authorized": False,
        "split_freeze_authorized": False,
        "exact_hf_authorized": False,
        "behavior_authorized": False,
    }


def validate_contract(
    *, contract_path: Path, scope_owner_attestation_path: Optional[Path] = None
) -> Dict[str, Any]:
    packet = _validate_pending_packet(contract_path)
    if scope_owner_attestation_path is None:
        return {
            "status": "valid_pending_contract_waiting_for_scope_owner",
            "contract_id": packet["contract"]["contract_id"],
            "contract_path": str(packet["contract_path"]),
            "contract_sha256": sha256_file(packet["contract_path"]),
            "attestation_template_path": str(packet["template_path"]),
            "repository_behavior_candidate_base_fact_count": packet["contract"][
                "repository_scan"
            ]["behavior_candidate_base_fact_count"],
            "scope_owner_attestation_valid": False,
            "ready_for_successor_universe_binding": False,
            "historical_exposure_complete": False,
            "review_freeze_authorized": False,
            "split_freeze_authorized": False,
            "exact_hf_authorized": False,
            "behavior_authorized": False,
        }
    return _validate_attestation(packet, scope_owner_attestation_path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize", help="emit a pending v2 packet")
    materialize.add_argument("--universe-manifest", type=Path, required=True)
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate packet and optional attestation")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--scope-owner-attestation", type=Path)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "materialize":
            result = materialize_pending_contract(
                universe_manifest_path=args.universe_manifest,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
            )
        else:
            result = validate_contract(
                contract_path=args.contract,
                scope_owner_attestation_path=args.scope_owner_attestation,
            )
    except (ContractValidationError, FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked_fail_closed",
                    "historical_exposure_complete": False,
                    "split_freeze_authorized": False,
                    "exact_hf_authorized": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
