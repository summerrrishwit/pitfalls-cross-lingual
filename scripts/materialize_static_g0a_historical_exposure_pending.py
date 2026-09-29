#!/usr/bin/env python3
"""Create a fail-closed pending historical-exposure attestation packet.

This tool scans every regular file in the current worktree (excluding ``.git``)
and every unique blob reachable from any local Git ref.  It records a complete,
content-hashed inventory and looks for exact holdout identifiers plus frozen
behavior-result signatures.  It also keeps the narrower per-row inventory of
``simulation_results.jsonl`` files for direct auditability.

Repository scanning cannot attest to external directories, separately run
scripts, or hosted platforms.  Those scopes require an explicit scope-owner
attestation, so this materializer always leaves ``historical_exposure_complete``
false.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_static_g0a as g0a  # noqa: E402


TOOL_VERSION = "static-g0a-historical-exposure-pending-materializer-v2"
POLICY_VERSION = "worktree-and-reachable-git-blob-exposure-scan-v2-pending"
SIMULATION_SCAN_SCHEMA = "static-g0a-local-simulation-exposure-scan-v1"
REPOSITORY_INVENTORY_SCHEMA = "static-g0a-repository-artifact-inventory-v1"
REPOSITORY_SCAN_RESULT_SCHEMA = "static-g0a-repository-exposure-scan-result-v1"
EXPECTED_SIMULATION_FILE_COUNT = 13
EXPECTED_HOLDOUT_COUNT = 64
CHECKS_NAME = "historical_exposure_checks.pending.jsonl"
REGISTRY_NAME = "historical_exposure_registry.pending.jsonl"
SIMULATION_SCANS_NAME = "simulation_results_scans.jsonl"
REPOSITORY_INVENTORY_NAME = "repository_artifact_inventory.jsonl"
REPOSITORY_SCAN_RESULTS_NAME = "repository_exposure_scan_results.jsonl"
MANIFEST_NAME = "historical_exposure_manifest.pending.json"
IDENTIFIER_FIELDS = frozenset({"base_fact_id", "candidate_id", "candidate_ids", "source_id"})
BEHAVIOR_STRONG_MARKERS = {
    "simulation_id": b'"simulation_id"',
    "simulation_scope": b'"simulation_scope"',
    "distractor_hit": b'"distractor_hit"',
    "wrong_choice": b'"wrong_choice"',
    "strict_signal": b'"strict_signal"',
    "strict_signal_ids": b'"strict_signal_ids"',
    "behavior_result": b'"behavior_result"',
}
BEHAVIOR_VARIANT_MARKERS = (b'"variant"', b'"arm"', b'"condition"')
BEHAVIOR_MODEL_MARKERS = (
    b'"model"',
    b'"model_id"',
    b'"model_name"',
    b'"response_model"',
    b'"checkpoint"',
)
BEHAVIOR_OUTCOME_MARKERS = (
    b'"raw_response"',
    b'"model_response"',
    b'"response_text"',
    b'"choice"',
    b'"selected_option"',
    b'"predicted_answer"',
    b'"correct"',
)
RESULT_LIKE_PATH_RE = re.compile(
    r"(?:^|[/_.-])"
    r"(?:result|results|prediction|predictions|evaluation|eval|response|responses)"
    r"(?:$|[/_.-])",
    re.IGNORECASE,
)
NON_BEHAVIOR_RESULT_PATH_TOKENS = (
    "triple_extraction",
    "translation",
    "perturbation_generation",
    "review",
    "adjudication",
)


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(g0a.canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _git(repo_root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], stderr=subprocess.STDOUT
    )


def _identifier_values(row: Mapping[str, Any]) -> set[str]:
    """Read provenance identifiers without inspecting model response fields."""

    values: set[str] = set()
    for field in IDENTIFIER_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value:
            values.add(value)
        elif isinstance(value, list):
            values.update(item for item in value if isinstance(item, str) and item)
    return values


def _identifier_pattern(identifier_to_base_fact_ids: Mapping[str, set[str]]) -> re.Pattern[bytes]:
    encoded = sorted(
        (identifier.encode("utf-8") for identifier in identifier_to_base_fact_ids),
        key=lambda item: (-len(item), item),
    )
    if not encoded:
        raise ValueError("repository exposure scan has no holdout identifiers")
    return re.compile(b"|".join(re.escape(identifier) for identifier in encoded))


def _behavior_signature_names(payload: bytes, relative_paths: Sequence[str]) -> List[str]:
    signatures = [
        name for name, marker in BEHAVIOR_STRONG_MARKERS.items() if marker in payload
    ]
    has_variant = any(marker in payload for marker in BEHAVIOR_VARIANT_MARKERS)
    has_model = any(marker in payload for marker in BEHAVIOR_MODEL_MARKERS)
    has_outcome = any(marker in payload for marker in BEHAVIOR_OUTCOME_MARKERS)
    if has_variant and has_model and has_outcome:
        signatures.append("variant_model_outcome")
    result_like_path = any(RESULT_LIKE_PATH_RE.search(path) for path in relative_paths)
    excluded_generation_path = all(
        any(token in path.lower() for token in NON_BEHAVIOR_RESULT_PATH_TOKENS)
        for path in relative_paths
    )
    if result_like_path and not excluded_generation_path and has_model and has_outcome:
        signatures.append("result_path_model_outcome")
    return sorted(set(signatures))


def _matched_identifiers(
    payload: bytes, identifier_pattern: re.Pattern[bytes]
) -> List[str]:
    return sorted(
        {match.group(0).decode("utf-8") for match in identifier_pattern.finditer(payload)}
    )


def _json_record_payloads(
    value: Any, identifier_to_base_fact_ids: Mapping[str, set[str]]
) -> Iterator[bytes]:
    if isinstance(value, list):
        for item in value:
            yield from _json_record_payloads(item, identifier_to_base_fact_ids)
        return
    if not isinstance(value, dict):
        return
    direct_identifiers = _identifier_values(value)
    keyed_identifiers = {
        key
        for key in value
        if isinstance(key, str) and key in identifier_to_base_fact_ids
    }
    if direct_identifiers & set(identifier_to_base_fact_ids) or keyed_identifiers:
        yield g0a.canonical_json_bytes(value)
        return
    for child in value.values():
        yield from _json_record_payloads(child, identifier_to_base_fact_ids)


def _behavior_hit_evidence(
    payload: bytes,
    *,
    relative_paths: Sequence[str],
    identifier_pattern: re.Pattern[bytes],
    identifier_to_base_fact_ids: Mapping[str, set[str]],
) -> Dict[str, Any]:
    lower_paths = [path.lower() for path in relative_paths]
    is_jsonl = any(
        path.endswith((".jsonl", ".ndjson")) for path in lower_paths
    )
    is_json = any(path.endswith(".json") for path in lower_paths)
    record_payloads: List[bytes] = []
    parsing_mode = "unstructured_whole_artifact"
    if is_jsonl:
        parsing_mode = "jsonl_record_lines"
        for line in payload.splitlines():
            if line.strip() and identifier_pattern.search(line):
                try:
                    parsed_line = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"identifier-bearing JSONL record is invalid in {relative_paths}"
                    ) from exc
                if not isinstance(parsed_line, dict):
                    raise ValueError(
                        f"identifier-bearing JSONL record is not an object in {relative_paths}"
                    )
                record_payloads.append(line)
    elif is_json:
        parsing_mode = "json_structural_records"
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"identifier-bearing JSON artifact is invalid in {relative_paths}"
            ) from exc
        record_payloads.extend(
            _json_record_payloads(parsed, identifier_to_base_fact_ids)
        )
        if not record_payloads and identifier_pattern.search(payload):
            parsing_mode = "json_unattributed_whole_artifact"
            record_payloads = [payload]
    else:
        record_payloads = [payload]

    behavior_hit_ids = set()
    signature_names = set()
    for record_payload in record_payloads:
        identifiers = _matched_identifiers(record_payload, identifier_pattern)
        if not identifiers:
            continue
        signatures = _behavior_signature_names(record_payload, relative_paths)
        if not signatures:
            continue
        signature_names.update(signatures)
        for identifier in identifiers:
            behavior_hit_ids.update(identifier_to_base_fact_ids[identifier])
    return {
        "behavior_hit_base_fact_ids": sorted(behavior_hit_ids),
        "behavior_signature_names": sorted(signature_names),
        "record_association_mode": parsing_mode,
        "identifier_bearing_record_count": len(record_payloads),
    }


def _scan_repository_payload(
    payload: bytes,
    *,
    inventory_id: str,
    source_kind: str,
    relative_paths: Sequence[str],
    identifier_pattern: re.Pattern[bytes],
    identifier_to_base_fact_ids: Mapping[str, set[str]],
) -> Dict[str, Any]:
    matched_identifiers = _matched_identifiers(payload, identifier_pattern)
    matched_base_fact_ids = sorted(
        {
            base_fact_id
            for identifier in matched_identifiers
            for base_fact_id in identifier_to_base_fact_ids[identifier]
        }
    )
    hit_evidence = (
        _behavior_hit_evidence(
            payload,
            relative_paths=relative_paths,
            identifier_pattern=identifier_pattern,
            identifier_to_base_fact_ids=identifier_to_base_fact_ids,
        )
        if matched_identifiers
        else {
            "behavior_hit_base_fact_ids": [],
            "behavior_signature_names": [],
            "record_association_mode": "not_applicable_no_identifier_match",
            "identifier_bearing_record_count": 0,
        }
    )
    signatures = hit_evidence["behavior_signature_names"]
    behavior_hit_ids = hit_evidence["behavior_hit_base_fact_ids"]
    return {
        "schema_version": REPOSITORY_SCAN_RESULT_SCHEMA,
        "inventory_id": inventory_id,
        "source_kind": source_kind,
        "relative_paths": list(relative_paths),
        "matched_identifiers": matched_identifiers,
        "matched_holdout_base_fact_ids": matched_base_fact_ids,
        "behavior_signature_names": signatures,
        "behavior_hit_base_fact_ids": behavior_hit_ids,
        "historical_behavior_exposure_detected": bool(behavior_hit_ids),
        "record_association_mode": hit_evidence["record_association_mode"],
        "identifier_bearing_record_count": hit_evidence[
            "identifier_bearing_record_count"
        ],
        "scan_status": "completed_exact_identifier_and_behavior_signature_scan",
    }


def _worktree_regular_files(repo_root: Path) -> List[Path]:
    repo_root = Path(repo_root).resolve()
    result: List[Path] = []
    for current_root, directory_names, file_names in os.walk(repo_root):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        if current == repo_root:
            directory_names[:] = [name for name in directory_names if name != ".git"]
        for name in list(directory_names):
            path = current / name
            if path.is_symlink():
                raise ValueError(f"worktree scan refuses symlinked directory: {path}")
        for name in file_names:
            path = current / name
            if path.is_symlink():
                raise ValueError(f"worktree scan refuses symlinked file: {path}")
            if path.is_file():
                result.append(path.resolve())
    return sorted(result, key=lambda path: str(path.relative_to(repo_root)))


def _reachable_git_blob_inventory(repo_root: Path) -> List[Dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    commits = sorted(
        value.decode("ascii")
        for value in _git(repo_root, "rev-list", "--all").splitlines()
        if value
    )
    if not commits:
        raise ValueError("repository has no reachable Git commits")
    by_oid: Dict[str, Dict[str, set[str]]] = defaultdict(
        lambda: {"relative_paths": set(), "reachable_commits": set()}
    )
    for commit in commits:
        tree = _git(repo_root, "ls-tree", "-r", "-z", commit)
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            try:
                metadata, raw_path = entry.split(b"\t", 1)
                mode, object_type, object_id = metadata.decode("ascii").split()
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"cannot parse Git tree entry for {commit}") from exc
            if object_type != "blob":
                continue
            relative_path = raw_path.decode("utf-8")
            record = by_oid[object_id]
            record["relative_paths"].add(relative_path)
            record["reachable_commits"].add(commit)
            record.setdefault("modes", set()).add(mode)
    return [
        {
            "git_blob_oid": object_id,
            "relative_paths": sorted(record["relative_paths"]),
            "reachable_commits": sorted(record["reachable_commits"]),
            "modes": sorted(record["modes"]),
        }
        for object_id, record in sorted(by_oid.items())
    ]


def _git_blob_payloads(
    repo_root: Path, sources: Sequence[Mapping[str, Any]]
) -> Iterator[Tuple[Mapping[str, Any], bytes]]:
    repo_root = Path(repo_root).resolve()
    process = subprocess.Popen(
        ["git", "-C", str(repo_root), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            b"".join(
                str(source["git_blob_oid"]).encode("ascii") + b"\n"
                for source in sources
            )
        )
        process.stdin.close()
        for source in sources:
            header = process.stdout.readline().rstrip(b"\n")
            try:
                object_id, object_type, raw_size = header.split()
                size = int(raw_size)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"cannot parse git cat-file header for {source['git_blob_oid']}"
                ) from exc
            if (
                object_id.decode("ascii") != source["git_blob_oid"]
                or object_type != b"blob"
                or size < 0
            ):
                raise ValueError(
                    f"git cat-file returned the wrong object for {source['git_blob_oid']}"
                )
            payload = process.stdout.read(size)
            delimiter = process.stdout.read(1)
            if len(payload) != size or delimiter != b"\n":
                raise ValueError(
                    f"git cat-file returned a truncated blob for {source['git_blob_oid']}"
                )
            yield source, payload
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            raise subprocess.CalledProcessError(
                return_code,
                process.args,
                stderr=stderr,
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def _materialize_repository_inventory(
    *,
    repo_root: Path,
    identifier_to_base_fact_ids: Mapping[str, set[str]],
) -> Dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    pattern = _identifier_pattern(identifier_to_base_fact_ids)
    inventory: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    worktree_files = _worktree_regular_files(repo_root)
    for path in worktree_files:
        relative_path = str(path.relative_to(repo_root))
        payload = path.read_bytes()
        content_sha256 = hashlib.sha256(payload).hexdigest()
        inventory_id = "worktree_" + g0a.sha256_value(
            {"relative_path": relative_path, "content_sha256": content_sha256}
        )[:24]
        result = _scan_repository_payload(
            payload,
            inventory_id=inventory_id,
            source_kind="current_worktree_file",
            relative_paths=[relative_path],
            identifier_pattern=pattern,
            identifier_to_base_fact_ids=identifier_to_base_fact_ids,
        )
        inventory.append(
            {
                "schema_version": REPOSITORY_INVENTORY_SCHEMA,
                "inventory_id": inventory_id,
                "source_kind": "current_worktree_file",
                "relative_paths": [relative_path],
                "byte_count": len(payload),
                "content_sha256": content_sha256,
                "identifier_match_count": len(result["matched_identifiers"]),
                "matched_holdout_base_fact_count": len(
                    result["matched_holdout_base_fact_ids"]
                ),
                "behavior_hit_base_fact_count": len(result["behavior_hit_base_fact_ids"]),
                "scan_status": result["scan_status"],
            }
        )
        if result["matched_identifiers"]:
            results.append(result)

    git_blob_sources = _reachable_git_blob_inventory(repo_root)
    for source, payload in _git_blob_payloads(repo_root, git_blob_sources):
        content_sha256 = hashlib.sha256(payload).hexdigest()
        inventory_id = "gitblob_" + g0a.sha256_value(
            {
                "git_blob_oid": source["git_blob_oid"],
                "content_sha256": content_sha256,
            }
        )[:24]
        result = _scan_repository_payload(
            payload,
            inventory_id=inventory_id,
            source_kind="reachable_git_blob",
            relative_paths=source["relative_paths"],
            identifier_pattern=pattern,
            identifier_to_base_fact_ids=identifier_to_base_fact_ids,
        )
        inventory.append(
            {
                "schema_version": REPOSITORY_INVENTORY_SCHEMA,
                "inventory_id": inventory_id,
                "source_kind": "reachable_git_blob",
                "git_blob_oid": source["git_blob_oid"],
                "relative_paths": source["relative_paths"],
                "reachable_commits": source["reachable_commits"],
                "modes": source["modes"],
                "byte_count": len(payload),
                "content_sha256": content_sha256,
                "identifier_match_count": len(result["matched_identifiers"]),
                "matched_holdout_base_fact_count": len(
                    result["matched_holdout_base_fact_ids"]
                ),
                "behavior_hit_base_fact_count": len(result["behavior_hit_base_fact_ids"]),
                "scan_status": result["scan_status"],
            }
        )
        if result["matched_identifiers"]:
            results.append(result)

    inventory.sort(key=lambda row: row["inventory_id"])
    results.sort(key=lambda row: row["inventory_id"])
    behavior_hit_ids = sorted(
        {
            base_fact_id
            for row in results
            for base_fact_id in row["behavior_hit_base_fact_ids"]
        }
    )
    return {
        "inventory": inventory,
        "results": results,
        "worktree_file_count": len(worktree_files),
        "reachable_git_blob_count": len(git_blob_sources),
        "identifier_matching_artifact_count": len(results),
        "behavior_hit_base_fact_ids": behavior_hit_ids,
    }


def _scan_simulation_file(
    path: Path,
    *,
    identifier_to_base_fact_ids: Mapping[str, set[str]],
    repo_root: Path,
) -> Dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    rows = g0a.read_jsonl(path, allow_empty=True)
    matches: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for identifier in _identifier_values(row):
            for base_fact_id in identifier_to_base_fact_ids.get(identifier, set()):
                matches[base_fact_id].add(identifier)
    relative_path = str(Path(path).resolve().relative_to(repo_root))
    return {
        "schema_version": SIMULATION_SCAN_SCHEMA,
        "relative_path": relative_path,
        "file": g0a._binding(path, record_count=len(rows)),
        "record_count": len(rows),
        "identifier_fields_scanned": sorted(IDENTIFIER_FIELDS),
        "behavior_outcome_fields_inspected": False,
        "matched_holdout_base_fact_count": len(matches),
        "matches": [
            {
                "base_fact_id": base_fact_id,
                "matched_identifiers": sorted(identifiers),
            }
            for base_fact_id, identifiers in sorted(matches.items())
        ],
        "local_file_scan_status": "completed_identifier_metadata_only",
    }


def _load_resolved(
    resolved_manifest_path: Path,
) -> Dict[str, Any]:
    resolved_manifest_path = Path(resolved_manifest_path).resolve()
    manifest = g0a.read_json(resolved_manifest_path)
    if manifest.get("schema_version") != g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA:
        raise ValueError("resolved draft manifest schema_version is unsupported")
    if manifest.get("status") != "resolved_static_draft_not_frozen":
        raise ValueError("resolved draft manifest status is invalid")
    for field in ("static_frozen", "behavior_authorized", "simulation_authorized"):
        if manifest.get(field) is not False:
            raise ValueError(f"resolved draft manifest must keep {field}=false")
    staging_manifest_path, staging_manifest = g0a._validate_file_binding(
        manifest.get("staging_manifest"),
        owner_path=resolved_manifest_path,
        label="resolved draft staging manifest",
        expected_schema=g0a.STAGING_MANIFEST_SCHEMA,
    )
    _, staging_rows, staging_by_id = g0a._load_staging(staging_manifest_path)
    resolved_rows_path, resolved_rows = g0a._validate_file_binding(
        manifest.get("resolved_rows"),
        owner_path=resolved_manifest_path,
        label="resolved static draft rows",
        expected_schema=g0a.FINAL_BUNDLE_SCHEMA,
        jsonl=True,
    )
    resolved_by_id: Dict[str, Dict[str, Any]] = {}
    for row in resolved_rows:
        base_fact_id = g0a.formal_admission.explicit_base_fact_id(
            row, "resolved static draft row"
        )
        if base_fact_id in resolved_by_id:
            raise ValueError(f"duplicate resolved base_fact_id: {base_fact_id}")
        resolved_by_id[base_fact_id] = row
    if set(resolved_by_id) != set(staging_by_id):
        raise ValueError("resolved and staging cohorts differ")
    if len(resolved_rows) != g0a.EXPECTED_RECORD_COUNT:
        raise ValueError("resolved cohort does not contain 160 facts")
    holdout_ids = sorted(
        base_fact_id
        for base_fact_id, row in resolved_by_id.items()
        if row.get("split_assignment") in {"validation", "sealed"}
    )
    if len(holdout_ids) != EXPECTED_HOLDOUT_COUNT:
        raise ValueError(
            f"validation/sealed cohort contains {len(holdout_ids)} facts, "
            f"expected {EXPECTED_HOLDOUT_COUNT}"
        )
    return {
        "manifest": manifest,
        "manifest_path": resolved_manifest_path,
        "manifest_binding": g0a._binding(
            resolved_manifest_path,
            schema_version=g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
        ),
        "resolved_rows_path": resolved_rows_path,
        "resolved_rows": resolved_rows,
        "resolved_by_id": resolved_by_id,
        "staging_manifest": staging_manifest,
        "staging_manifest_path": staging_manifest_path,
        "staging_by_id": staging_by_id,
        "holdout_ids": holdout_ids,
    }


def materialize_pending_exposure(
    *,
    resolved_manifest_path: Path,
    repo_root: Path,
    output_dir: Path,
    expected_simulation_file_count: int = EXPECTED_SIMULATION_FILE_COUNT,
) -> Dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    git_toplevel = Path(
        _git(repo_root, "rev-parse", "--show-toplevel").decode().strip()
    ).resolve()
    if git_toplevel != repo_root:
        raise ValueError("repo_root is not the Git toplevel")
    loaded = _load_resolved(resolved_manifest_path)

    tracked_payload = _git(repo_root, "ls-files", "-z")
    tracked_paths = sorted(
        value.decode("utf-8")
        for value in tracked_payload.split(b"\0")
        if value
    )
    simulation_paths = sorted(
        (
            repo_root / relative
            for relative in tracked_paths
            if Path(relative).name == "simulation_results.jsonl"
        ),
        key=str,
    )
    if len(simulation_paths) != expected_simulation_file_count:
        raise ValueError(
            f"tracked simulation_results file count is {len(simulation_paths)}, "
            f"expected {expected_simulation_file_count}"
        )

    identifier_to_ids: Dict[str, set[str]] = defaultdict(set)
    query_payload_by_id: Dict[str, Dict[str, Any]] = {}
    for base_fact_id in loaded["holdout_ids"]:
        row = loaded["resolved_by_id"][base_fact_id]
        identifiers = sorted(
            {
                value
                for value in (
                    base_fact_id,
                    row.get("candidate_id"),
                    row.get("source_id"),
                )
                if isinstance(value, str) and value
            }
        )
        for identifier in identifiers:
            identifier_to_ids[identifier].add(base_fact_id)
        query_payload_by_id[base_fact_id] = {
            "base_fact_id": base_fact_id,
            "candidate_id": row.get("candidate_id"),
            "source_id": row.get("source_id"),
            "resolved_input_record_sha256": g0a.sha256_value(row),
        }

    scans = [
        _scan_simulation_file(
            path,
            identifier_to_base_fact_ids=identifier_to_ids,
            repo_root=repo_root,
        )
        for path in simulation_paths
    ]
    repository_scan = _materialize_repository_inventory(
        repo_root=repo_root,
        identifier_to_base_fact_ids=identifier_to_ids,
    )
    matches_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for scan in scans:
        for match in scan["matches"]:
            matches_by_id[match["base_fact_id"]].append(
                {
                    "relative_path": scan["relative_path"],
                    "matched_identifiers": match["matched_identifiers"],
                }
            )

    repository_behavior_hits_by_id: Dict[str, List[str]] = defaultdict(list)
    for result in repository_scan["results"]:
        for base_fact_id in result["behavior_hit_base_fact_ids"]:
            repository_behavior_hits_by_id[base_fact_id].append(result["inventory_id"])

    checked_sources = [str(path.relative_to(repo_root)) for path in simulation_paths]
    checks: List[Dict[str, Any]] = []
    for base_fact_id in loaded["holdout_ids"]:
        row = loaded["resolved_by_id"][base_fact_id]
        staging = loaded["staging_by_id"][base_fact_id]
        local_hits = matches_by_id.get(base_fact_id, [])
        repository_behavior_hits = sorted(
            repository_behavior_hits_by_id.get(base_fact_id, [])
        )
        exposed = bool(local_hits or repository_behavior_hits)
        checks.append(
            {
                "schema_version": g0a.EXPOSURE_CHECK_SCHEMA,
                "base_fact_id": base_fact_id,
                "source_record_sha256": staging["source_record_sha256"],
                "effective_source_record_sha256": staging[
                    "effective_source_record_sha256"
                ],
                "input_record_sha256": g0a.sha256_value(row),
                "split_assignment": row["split_assignment"],
                "query_fingerprint_sha256": g0a.sha256_value(
                    query_payload_by_id[base_fact_id]
                ),
                "checked_sources": checked_sources,
                "local_simulation_identifier_hits": local_hits,
                "repository_behavior_hit_inventory_ids": repository_behavior_hits,
                "historically_exposed": True if exposed else None,
                "terminal_status": (
                    "completed_exposed_blocked"
                    if exposed
                    else "pending_external_scope_attestation"
                ),
                "reviewer_type": "codex_proxy",
                "human_gold": False,
            }
        )

    head = _git(repo_root, "rev-parse", "HEAD").decode().strip()
    status_payload = _git(repo_root, "status", "--porcelain=v1", "-z")
    commit_count = int(_git(repo_root, "rev-list", "--all", "--count").decode().strip())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir: Optional[Path] = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.publish.", dir=str(output_dir.parent))
    )
    try:
        checks_path = temporary_dir / CHECKS_NAME
        registry_path = temporary_dir / REGISTRY_NAME
        scans_path = temporary_dir / SIMULATION_SCANS_NAME
        repository_inventory_path = temporary_dir / REPOSITORY_INVENTORY_NAME
        repository_results_path = temporary_dir / REPOSITORY_SCAN_RESULTS_NAME
        manifest_path = temporary_dir / MANIFEST_NAME
        checks_path.write_bytes(_jsonl_payload(checks))
        registry_path.write_bytes(b"")
        scans_path.write_bytes(_jsonl_payload(scans))
        repository_inventory_path.write_bytes(
            _jsonl_payload(repository_scan["inventory"])
        )
        repository_results_path.write_bytes(_jsonl_payload(repository_scan["results"]))
        behavior_hit_ids = repository_scan["behavior_hit_base_fact_ids"]
        status = (
            "blocked_historical_exposure_detected"
            if behavior_hit_ids
            else "pending_external_scope_attestation"
        )
        known_blockers = [
            "external_directories_not_attested",
            "external_script_run_history_not_attested",
            "external_platform_run_history_not_attested",
        ]
        if behavior_hit_ids:
            known_blockers.append("repository_historical_behavior_exposure_detected")
        manifest = {
            "schema_version": g0a.EXPOSURE_MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "created_at": g0a.utc_now(),
            "status": status,
            "policy_version": POLICY_VERSION,
            "input_bundle": copy.deepcopy(loaded["staging_manifest"]["source_bundle"]),
            "effective_staging_bundle": copy.deepcopy(
                loaded["staging_manifest"]["effective_staging_bundle"]
            ),
            "resolved_static_draft": copy.deepcopy(
                loaded["manifest"]["resolved_rows"]
            ),
            "resolved_static_draft_manifest": copy.deepcopy(
                loaded["manifest_binding"]
            ),
            "repository_scope": {
                "repo_root": str(repo_root),
                "git_head": head,
                "reachable_commit_count": commit_count,
                "tracked_file_count": len(tracked_paths),
                "tracked_paths_sha256": g0a.sha256_value(tracked_paths),
                "dirty_entry_count": len([x for x in status_payload.split(b"\0") if x]),
                "dirty_status_sha256": g0a.sha256_value(
                    status_payload.decode("utf-8", errors="surrogateescape")
                ),
                "current_worktree_simulation_identifier_scan_complete": True,
                "current_worktree_all_artifact_content_scan_complete": True,
                "reachable_git_history_content_scan_complete": True,
                "external_directories_attested": False,
                "external_script_run_history_attested": False,
                "external_platform_run_history_attested": False,
            },
            "simulation_results_scan": {
                "file_count": len(scans),
                "record_count": sum(scan["record_count"] for scan in scans),
                "matched_holdout_base_fact_count": len(matches_by_id),
                "identifier_metadata_only": True,
                "behavior_outcome_fields_inspected": False,
                "scans": g0a._binding(
                    scans_path,
                    schema_version=SIMULATION_SCAN_SCHEMA,
                    record_count=len(scans),
                ),
            },
            "repository_artifact_scan": {
                "policy": "exact-holdout-identifiers-plus-behavior-signature-v1",
                "current_worktree_complete": True,
                "reachable_git_blobs_complete": True,
                "worktree_file_count": repository_scan["worktree_file_count"],
                "reachable_git_blob_count": repository_scan[
                    "reachable_git_blob_count"
                ],
                "identifier_matching_artifact_count": repository_scan[
                    "identifier_matching_artifact_count"
                ],
                "behavior_hit_base_fact_count": len(behavior_hit_ids),
                "behavior_hit_base_fact_ids": behavior_hit_ids,
                "inventory": g0a._binding(
                    repository_inventory_path,
                    schema_version=REPOSITORY_INVENTORY_SCHEMA,
                    record_count=len(repository_scan["inventory"]),
                    id_values=[
                        row["inventory_id"] for row in repository_scan["inventory"]
                    ],
                    id_digest_field="inventory_ids_sha256",
                ),
                "results": g0a._binding(
                    repository_results_path,
                    schema_version=REPOSITORY_SCAN_RESULT_SCHEMA,
                    record_count=len(repository_scan["results"]),
                    id_values=[row["inventory_id"] for row in repository_scan["results"]],
                    id_digest_field="inventory_ids_sha256",
                ),
            },
            "checks": g0a._binding(
                checks_path,
                schema_version=g0a.EXPOSURE_CHECK_SCHEMA,
                record_count=len(checks),
                id_values=[row["base_fact_id"] for row in checks],
                id_digest_field="base_fact_ids_sha256",
            ),
            "registry": g0a._binding(
                registry_path,
                schema_version=g0a.EXPOSURE_REGISTRY_SCHEMA,
                record_count=0,
            ),
            "holdout_check_count": len(checks),
            "unresolved_check_count": sum(
                row["terminal_status"] != "completed" for row in checks
            ),
            "review_blinded_to_behavior_results": True,
            "behavior_result_count": 0,
            "zero_registry_supported_by_per_fact_checks": False,
            "historical_exposure_complete": False,
            "known_blockers": known_blockers,
            "static_frozen": False,
            "behavior_authorized": False,
            "simulation_authorized": False,
        }
        for key, filename in (
            ("checks", CHECKS_NAME),
            ("registry", REGISTRY_NAME),
        ):
            manifest[key]["path"] = str(output_dir / filename)
        manifest["simulation_results_scan"]["scans"]["path"] = str(
            output_dir / SIMULATION_SCANS_NAME
        )
        manifest["repository_artifact_scan"]["inventory"]["path"] = str(
            output_dir / REPOSITORY_INVENTORY_NAME
        )
        manifest["repository_artifact_scan"]["results"]["path"] = str(
            output_dir / REPOSITORY_SCAN_RESULTS_NAME
        )
        manifest_path.write_bytes(_json_payload(manifest))
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
        os.rename(temporary_dir, output_dir)
        temporary_dir = None
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return {
        "status": manifest["status"],
        "historical_exposure_manifest": str(output_dir / MANIFEST_NAME),
        "holdout_check_count": len(checks),
        "simulation_results_file_count": len(scans),
        "local_identifier_hit_count": len(matches_by_id),
        "repository_behavior_hit_base_fact_count": len(
            repository_scan["behavior_hit_base_fact_ids"]
        ),
        "historical_exposure_complete": False,
        "behavior_authorized": False,
        "known_blockers": manifest["known_blockers"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--resolved-manifest", type=Path, required=True)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--expected-simulation-file-count",
        type=int,
        default=EXPECTED_SIMULATION_FILE_COUNT,
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = materialize_pending_exposure(
        resolved_manifest_path=args.resolved_manifest,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        expected_simulation_file_count=args.expected_simulation_file_count,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
