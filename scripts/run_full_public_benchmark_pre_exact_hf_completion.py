#!/usr/bin/env python3
"""Wait for a candidate review, audit cross-round stability, and stop safely.

This tool is intentionally audit-only.  It never launches a candidate retry,
resolution, successor rebuild, another review round, zh translation, an
attestation step, a freezer, HF/tokenizer work, or behavior/mechanism work.

Repeated review-until-acceptance is an optional-stopping procedure when an
unchanged candidate can receive different decisions across rounds.  Therefore
the named terminal review is compared with its immediately preceding round and
the workflow always stops at ``protocol_review_required`` after emitting a
SHA-bound stability report.  A running review can optionally be watched until
its atomic terminal manifest appears; it is never resumed by this tool.  If an
operator has independently stopped an interrupted writer,
``--acknowledge-interrupted-review`` permits a read-only partial diagnostic
over the durable checkpoint and journal.  That diagnostic is always
incomplete, always stops at protocol review, and never authorizes downstream
work.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
CANDIDATE_REVIEW_SCRIPT = SCRIPT_DIR / "run_full_public_benchmark_candidate_review.py"
RESOLUTION_SCRIPT = SCRIPT_DIR / "resolve_full_public_benchmark_candidate_review.py"

TOOL_VERSION = "full-public-benchmark-candidate-stability-gate-v1"
STATE_SCHEMA = "public-benchmark-candidate-stability-state-v1"
JOURNAL_SCHEMA = "public-benchmark-candidate-stability-event-v1"
REPORT_SCHEMA = "candidate-cross-round-stability-report-v1"
CANDIDATE_MANIFEST_SCHEMA = "public-benchmark-full-candidate-review-run-manifest-v1"
HISTORICAL_CANDIDATE_TOOL_VERSION = (
    "full-public-benchmark-candidate-review-runner-v2"
)
CURRENT_CANDIDATE_TOOL_VERSION = "full-public-benchmark-candidate-review-runner-v3"
SUPPORTED_CANDIDATE_TOOL_VERSIONS = frozenset(
    {HISTORICAL_CANDIDATE_TOOL_VERSION, CURRENT_CANDIDATE_TOOL_VERSION}
)
# Backward-compatible public name for tests/callers that mean the current
# runner.  Historical v2 is accepted only by the explicit frozen validator.
CANDIDATE_TOOL_VERSION = CURRENT_CANDIDATE_TOOL_VERSION

HISTORICAL_CHECKPOINT_SCHEMA = "public-benchmark-full-candidate-review-checkpoint-v1"
HISTORICAL_ADJUDICATION_SCHEMA = "public-benchmark-full-candidate-adjudication-v2"
HISTORICAL_PROMPT_VERSION = "public-benchmark-full-candidate-semantic-review-v1"
HISTORICAL_REVIEW_METHOD = (
    "independent_behavior_blind_candidate_semantic_review_v1"
)
HISTORICAL_PROMPT_CONTRACT_SHA256 = (
    "c40fba7bb1dd0972c62d7cf9e2a86fb2f650ae78e777e9eb910b5a1f06f3fe54"
)
HISTORICAL_SOURCE_FINGERPRINTS = {
    "model_router_runtime_sha256": (
        "1ee2d718ea10e515bf2e72579c376db9beb4edba4ef85d3e0636153705c4e684"
    ),
    "route_identity_helper_sha256": (
        "40ec559e2caea4c634c5842813e94337f0afcca9c97c82e8fc73b0ed8cdf6c61"
    ),
    "runner_sha256": (
        "10a5c8dcd1109f8f411d208f508e3e3d476ea1f1056bea2ba425b45b5fa991ad"
    ),
}
HISTORICAL_JUDGMENT_FIELDS = (
    "answer_unique",
    "distractor_factually_false",
    "answer_alias_disjoint",
    "neutral_unrelated",
    "neutral_semantically_neutral",
)

CANDIDATE_CHECKPOINT_NAME = "candidate_review_checkpoint.jsonl"
CANDIDATE_JOURNAL_NAME = "candidate_review_checkpoint.journal.jsonl"

CANDIDATE_PARAMETERS: Dict[str, Any] = {
    "reviewer_provider_profile": "openai",
    "reviewer_model": "gpt-5.5",
    "reviewer_reasoning_effort": "low",
    "reviewer_max_output_tokens": 16384,
    "expected_response_model": "gpt-5.5-2026-04-24",
    "max_retries": 2,
    "batch_size": 8,
    "checkpoint_compact_every": 128,
    "max_workers": 2,
}


class CompletionError(RuntimeError):
    """Base class for fail-closed stability-gate errors."""


class CompletionBlockedError(CompletionError):
    """The audit cannot safely continue."""


class ExternalWriterRunning(CompletionError):
    """The explicitly named review is still being written elsewhere."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        super().__init__(f"external writer is still running: {self.manifest_path}")


class OrchestratorLockedError(CompletionError):
    """Another stability watcher/auditor owns the lock."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CompletionBlockedError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CompletionBlockedError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise CompletionBlockedError(f"Expected non-empty JSONL: {path}")
    return rows


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def append_journal(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(value)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[int]:
    """Acquire one non-blocking lock; retain the lock inode after release."""

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OrchestratorLockedError(
                f"another stability controller owns {path}"
            ) from exc
        metadata = canonical_json_bytes(
            {
                "tool_version": TOOL_VERSION,
                "pid": os.getpid(),
                "acquired_at": utc_now(),
            }
        ) + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, metadata)
        os.fsync(descriptor)
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, *, seen: Optional[set[Path]] = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    visited = set() if seen is None else seen
    if path in visited:
        raise CompletionBlockedError(f"config extends cycle: {path}")
    visited.add(path)
    value = read_json(path)
    parent = value.get("extends")
    if not parent:
        return value
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        project_relative = (PROJECT_ROOT / parent_path).resolve()
        parent_path = (
            project_relative
            if project_relative.is_file()
            else (path.parent / parent_path).resolve()
        )
    return _deep_merge(load_config(parent_path, seen=visited), value)


def _load_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if isinstance(existing, ModuleType):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CompletionBlockedError(f"cannot load validation module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_binding_path(binding: Mapping[str, Any], owner_path: Path) -> Path:
    value = binding.get("path")
    if not isinstance(value, str) or not value.strip():
        value = binding.get("filename")
    if not isinstance(value, str) or not value.strip():
        raise CompletionBlockedError("artifact binding has no path or filename")
    path = Path(value)
    if not path.is_absolute():
        path = Path(owner_path).resolve().parent / path
    return path.resolve()


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompletionBlockedError(f"{label} must be a non-empty string")
    return value.strip()


def _artifact_binding_from_payload(
    path: Path,
    payload: bytes,
    *,
    exists: bool,
    record_count: int,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "exists": exists,
        "sha256": sha256_bytes(payload) if exists else None,
        "byte_count": len(payload),
        "record_count": record_count,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _validate_bound_file(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    jsonl: bool,
    expected_schema: Optional[str] = None,
) -> Tuple[Path, Any, Dict[str, Any], bytes]:
    if not isinstance(binding, dict):
        raise CompletionBlockedError(f"{label} binding is missing")
    path = _resolve_binding_path(binding, owner_path)
    if not path.is_file():
        raise CompletionBlockedError(f"{label} is missing: {path}")
    payload = path.read_bytes()
    if binding.get("sha256") != sha256_bytes(payload):
        raise CompletionBlockedError(f"{label} SHA-256 mismatch")
    if binding.get("byte_count") != len(payload):
        raise CompletionBlockedError(f"{label} byte_count mismatch")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise CompletionBlockedError(f"{label} schema_version mismatch")
    if jsonl:
        rows = _parse_complete_jsonl_payload(payload, path=path, label=label)
        if binding.get("record_count") != len(rows):
            raise CompletionBlockedError(f"{label} record_count mismatch")
        value: Any = rows
    else:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(f"Malformed {label}: {path}") from exc
        if not isinstance(value, dict):
            raise CompletionBlockedError(f"Expected JSON object for {label}: {path}")
    return path, value, dict(binding), payload


def _parse_complete_jsonl_payload(
    payload: bytes, *, path: Path, label: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(
                f"Malformed {label} line {line_number}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise CompletionBlockedError(
                f"Expected JSON object in {label} line {line_number}: {path}"
            )
        rows.append(value)
    return rows


def _parse_interrupted_journal_payload(
    payload: bytes, *, path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse durable journal rows and tolerate only one unterminated tail.

    The historical runner intentionally ignored an incomplete final journal
    line on resume.  This audit mirrors only that narrow recovery rule: a
    malformed terminated line or malformed non-final line fails closed.
    """

    rows: List[Dict[str, Any]] = []
    lines = payload.splitlines(keepends=True)
    nonempty_line_count = sum(bool(line.strip()) for line in lines)
    truncated_tail_detected = False
    unterminated_parseable_tail_detected = False
    ignored_tail_byte_count = 0
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        is_last = index == len(lines) - 1
        terminated = raw_line.endswith((b"\n", b"\r"))
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_last and not terminated:
                truncated_tail_detected = True
                ignored_tail_byte_count = len(raw_line)
                break
            raise CompletionBlockedError(
                f"Malformed checkpoint journal line {index + 1}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise CompletionBlockedError(
                f"Expected JSON object in checkpoint journal line {index + 1}: {path}"
            )
        rows.append(value)
        if is_last and not terminated:
            unterminated_parseable_tail_detected = True
    return rows, {
        "raw_nonempty_line_count": nonempty_line_count,
        "parsed_record_count": len(rows),
        "truncated_tail_detected": truncated_tail_detected,
        "truncated_tail_ignored": truncated_tail_detected,
        "ignored_truncated_tail_byte_count": ignored_tail_byte_count,
        "unterminated_parseable_tail_detected": (
            unterminated_parseable_tail_detected
        ),
        "malformed_non_tail_record_absent": True,
    }


def _historical_v2_units(
    *,
    predecessor: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Reconstruct v2 input identity without importing the upgraded runner."""

    predecessor_artifacts = predecessor.get("artifacts")
    contract_artifacts = run_contract.get("input_artifacts")
    if not isinstance(predecessor_artifacts, dict) or not isinstance(
        contract_artifacts, dict
    ):
        raise CompletionBlockedError("historical v2 input artifact bindings are missing")
    expected_schemas = {
        "full_base_facts": "public-benchmark-full-provisional-base-fact-v1",
        "distractor_candidates": "public-benchmark-dual-distractor-candidate-v1",
        "neutral_reference_candidates": (
            "public-benchmark-neutral-reference-candidate-v1"
        ),
    }
    rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    normalized_bindings: Dict[str, Dict[str, Any]] = {}
    for label, schema in expected_schemas.items():
        source = predecessor_artifacts.get(label)
        contract_binding = contract_artifacts.get(label)
        if not isinstance(source, dict) or not isinstance(contract_binding, dict):
            raise CompletionBlockedError(f"historical v2 {label} binding is missing")
        path = Path(source.get("path") or "").resolve()
        if _resolve_binding_path(contract_binding, manifest_path) != path:
            raise CompletionBlockedError(f"historical v2 {label} path mismatch")
        source_binding = source.get("binding")
        rows = source.get("value")
        if not isinstance(source_binding, dict) or not isinstance(rows, list):
            raise CompletionBlockedError(f"historical v2 {label} evidence is invalid")
        for field in ("sha256", "byte_count", "record_count", "schema_version"):
            if contract_binding.get(field) != source_binding.get(field):
                raise CompletionBlockedError(
                    f"historical v2 {label} {field} mismatch"
                )
        if contract_binding.get("schema_version") != schema:
            raise CompletionBlockedError(
                f"historical v2 {label} schema_version mismatch"
            )
        if not path.is_file():
            raise CompletionBlockedError(f"historical v2 {label} is missing: {path}")
        payload = path.read_bytes()
        if (
            contract_binding.get("sha256") != sha256_bytes(payload)
            or contract_binding.get("byte_count") != len(payload)
            or contract_binding.get("record_count") != len(rows)
        ):
            raise CompletionBlockedError(
                f"historical v2 {label} file binding is stale"
            )
        rows_by_label[label] = [dict(row) for row in rows]
        normalized_bindings[label] = {
            **copy.deepcopy(contract_binding),
            "path": str(path),
        }

    facts: Dict[str, Dict[str, Any]] = {}
    for row in rows_by_label["full_base_facts"]:
        base_fact_id = _required_string(row.get("base_fact_id"), "base_fact_id")
        if base_fact_id in facts:
            raise CompletionBlockedError(
                f"duplicate historical v2 base_fact_id: {base_fact_id}"
            )
        facts[base_fact_id] = row

    by_fact: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    seen_candidate_ids: set[str] = set()
    for kind, label, id_field in (
        ("distractor", "distractor_candidates", "distractor_id"),
        ("neutral", "neutral_reference_candidates", "neutral_candidate_id"),
    ):
        for row in rows_by_label[label]:
            candidate_id = _required_string(row.get(id_field), f"{kind}.{id_field}")
            if candidate_id in seen_candidate_ids:
                raise CompletionBlockedError(
                    f"duplicate historical v2 candidate_id: {candidate_id}"
                )
            seen_candidate_ids.add(candidate_id)
            target_id = _required_string(row.get("base_fact_id"), f"{candidate_id}.base_fact_id")
            source_id = _required_string(
                row.get("source_base_fact_id"),
                f"{candidate_id}.source_base_fact_id",
            )
            if target_id not in facts or source_id not in facts:
                raise CompletionBlockedError(
                    f"historical v2 candidate references an unknown fact: {candidate_id}"
                )
            by_fact.setdefault(target_id, {"distractor": [], "neutral": []})[
                kind
            ].append(row)
    for grouped in by_fact.values():
        for kind in ("distractor", "neutral"):
            grouped[kind].sort(
                key=lambda row: (
                    int(row.get("slot")),
                    str(row.get("distractor_id") or row.get("neutral_candidate_id")),
                )
            )

    units: List[Dict[str, Any]] = []
    for target_id in sorted(by_fact):
        for kind in ("distractor", "neutral"):
            id_field = "distractor_id" if kind == "distractor" else "neutral_candidate_id"
            for candidate in by_fact[target_id][kind]:
                source_id = str(candidate["source_base_fact_id"])
                units.append(
                    {
                        "input_index": len(units),
                        "candidate_id": str(candidate[id_field]),
                        "candidate_kind": kind,
                        "candidate_row_sha256": sha256_value(candidate),
                        "target_base_fact_id": target_id,
                        "target_base_fact_row_sha256": sha256_value(facts[target_id]),
                        "source_base_fact_id": source_id,
                        "source_base_fact_row_sha256": sha256_value(facts[source_id]),
                    }
                )

    selected_ids = [unit["candidate_id"] for unit in units]
    by_kind = {
        kind: [unit["candidate_id"] for unit in units if unit["candidate_kind"] == kind]
        for kind in ("distractor", "neutral")
    }
    if (
        run_contract.get("selected_candidate_count") != len(units)
        or run_contract.get("full_candidate_count") != len(units)
        or run_contract.get("selection_is_full_candidate_set") is not True
        or run_contract.get("selected_candidate_ids_sha256")
        != sha256_value(selected_ids)
    ):
        raise CompletionBlockedError("historical v2 selected candidate scope is stale")
    selected_by_kind = run_contract.get("selected_candidate_ids_by_kind")
    if not isinstance(selected_by_kind, dict):
        raise CompletionBlockedError("historical v2 selected kind bindings are missing")
    for kind, ids in by_kind.items():
        value = selected_by_kind.get(kind)
        if not isinstance(value, dict) or value != {
            "record_count": len(ids),
            "ordered_ids_sha256": sha256_value(ids),
        }:
            raise CompletionBlockedError(
                f"historical v2 selected {kind} identity is stale"
            )
    return units, normalized_bindings


def _unit_field(unit: Any, field: str) -> Any:
    return unit[field] if isinstance(unit, Mapping) else getattr(unit, field)


def _validate_historical_v2_checkpoint_row(
    row: Mapping[str, Any],
    *,
    units_by_id: Mapping[str, Any],
    run_contract_sha256: str,
) -> Tuple[str, Dict[str, Any]]:
    candidate_id = _required_string(row.get("candidate_id"), "checkpoint.candidate_id")
    unit = units_by_id.get(candidate_id)
    if unit is None:
        raise CompletionBlockedError(
            f"historical v2 checkpoint candidate is outside selected scope: {candidate_id}"
        )
    if row.get("schema_version") != HISTORICAL_CHECKPOINT_SCHEMA:
        raise CompletionBlockedError(
            f"historical v2 checkpoint schema is unsupported: {candidate_id}"
        )
    if row.get("tool_version") != HISTORICAL_CANDIDATE_TOOL_VERSION:
        raise CompletionBlockedError(
            f"historical v2 checkpoint tool version is unsupported: {candidate_id}"
        )
    if row.get("run_contract_sha256") != run_contract_sha256:
        raise CompletionBlockedError(
            f"historical v2 checkpoint run contract is stale: {candidate_id}"
        )
    for field in (
        "candidate_kind",
        "input_index",
        "candidate_row_sha256",
        "target_base_fact_id",
        "target_base_fact_row_sha256",
        "source_base_fact_id",
        "source_base_fact_row_sha256",
    ):
        if row.get(field) != _unit_field(unit, field):
            raise CompletionBlockedError(
                f"historical v2 checkpoint {field} is stale: {candidate_id}"
            )
    if row.get("terminal_status") not in {"completed", "failed"}:
        raise CompletionBlockedError(
            f"historical v2 checkpoint terminal_status is invalid: {candidate_id}"
        )
    if row.get("behavior_blind") is not True or row.get("human_gold") is not False:
        raise CompletionBlockedError(
            f"historical v2 checkpoint evidence boundary is invalid: {candidate_id}"
        )
    for field in (
        "hf_model_execution_count",
        "hf_tokenizer_execution_count",
        "behavior_execution_count",
        "validation_behavior_exposure_count",
        "sealed_behavior_exposure_count",
    ):
        if row.get(field) != 0:
            raise CompletionBlockedError(
                f"historical v2 checkpoint execution boundary is invalid: {candidate_id}.{field}"
            )
    calls = row.get("model_calls")
    if not isinstance(calls, list) or not isinstance(row.get("retry_history"), list):
        raise CompletionBlockedError(
            f"historical v2 checkpoint trace/history is invalid: {candidate_id}"
        )
    for call in calls:
        if not isinstance(call, dict):
            raise CompletionBlockedError(
                f"historical v2 checkpoint model call is invalid: {candidate_id}"
            )
        if (
            call.get("prompt_version") != HISTORICAL_PROMPT_VERSION
            or call.get("provider_profile")
            != CANDIDATE_PARAMETERS["reviewer_provider_profile"]
            or call.get("requested_model")
            != CANDIDATE_PARAMETERS["reviewer_model"]
            or call.get("expected_response_model")
            != CANDIDATE_PARAMETERS["expected_response_model"]
            or call.get("credentials_or_endpoints_included") is not False
        ):
            raise CompletionBlockedError(
                f"historical v2 checkpoint model route is invalid: {candidate_id}"
            )
        if call.get("response_model") is not None and (
            call.get("response_model")
            != CANDIDATE_PARAMETERS["expected_response_model"]
            or call.get("response_model_identity_status") != "matched"
        ):
            raise CompletionBlockedError(
                f"historical v2 checkpoint response identity is invalid: {candidate_id}"
            )
    if row["terminal_status"] == "failed":
        if row.get("strict_adjudication") is not None or row.get("model_verdict") is not None:
            raise CompletionBlockedError(
                f"historical v2 failed checkpoint contains a verdict: {candidate_id}"
            )
        return candidate_id, dict(row)

    verdict = row.get("model_verdict")
    decision = row.get("strict_adjudication")
    if not isinstance(verdict, dict) or not isinstance(decision, dict):
        raise CompletionBlockedError(
            f"historical v2 completed checkpoint lacks adjudication: {candidate_id}"
        )
    expected_identity = {
        "schema_version": HISTORICAL_ADJUDICATION_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_kind": _unit_field(unit, "candidate_kind"),
        "candidate_row_sha256": _unit_field(unit, "candidate_row_sha256"),
        "target_base_fact_id": _unit_field(unit, "target_base_fact_id"),
        "target_base_fact_row_sha256": _unit_field(
            unit, "target_base_fact_row_sha256"
        ),
        "source_base_fact_id": _unit_field(unit, "source_base_fact_id"),
        "source_base_fact_row_sha256": _unit_field(
            unit, "source_base_fact_row_sha256"
        ),
        "reviewer_type": "independent_model_proxy",
        "reviewer_id": "openai:gpt-5.5-2026-04-24",
        "requested_model": CANDIDATE_PARAMETERS["reviewer_model"],
        "expected_response_model": CANDIDATE_PARAMETERS[
            "expected_response_model"
        ],
        "response_model": CANDIDATE_PARAMETERS["expected_response_model"],
        "response_model_identity_status": "matched",
        "review_method": HISTORICAL_REVIEW_METHOD,
        "behavior_blind": True,
        "target_behavior_consumed": False,
        "human_gold": False,
    }
    for field, expected in expected_identity.items():
        if decision.get(field) != expected:
            raise CompletionBlockedError(
                f"historical v2 adjudication {field} is invalid: {candidate_id}"
            )
    if verdict.get("candidate_id") != candidate_id or verdict.get(
        "candidate_kind"
    ) != _unit_field(unit, "candidate_kind"):
        raise CompletionBlockedError(
            f"historical v2 model verdict identity is stale: {candidate_id}"
        )
    for field in ("overall_decision", *HISTORICAL_JUDGMENT_FIELDS, "rationale", "confidence"):
        if decision.get(field) != verdict.get(field):
            raise CompletionBlockedError(
                f"historical v2 checkpoint verdict/adjudication mismatch: {candidate_id}.{field}"
            )
    if decision.get("overall_decision") not in {"accept", "reject", "defer"}:
        raise CompletionBlockedError(
            f"historical v2 decision is invalid: {candidate_id}"
        )
    if decision.get("confidence") not in {"low", "medium", "high"}:
        raise CompletionBlockedError(
            f"historical v2 confidence is invalid: {candidate_id}"
        )
    _required_string(decision.get("rationale"), f"{candidate_id}.rationale")
    relevant = (
        HISTORICAL_JUDGMENT_FIELDS[:3]
        if _unit_field(unit, "candidate_kind") == "distractor"
        else HISTORICAL_JUDGMENT_FIELDS[3:]
    )
    irrelevant = (
        HISTORICAL_JUDGMENT_FIELDS[3:]
        if _unit_field(unit, "candidate_kind") == "distractor"
        else HISTORICAL_JUDGMENT_FIELDS[:3]
    )
    if any(decision.get(field) is not None for field in irrelevant):
        raise CompletionBlockedError(
            f"historical v2 irrelevant judgment is non-null: {candidate_id}"
        )
    values = [decision.get(field) for field in relevant]
    if any(value is not None and not isinstance(value, bool) for value in values):
        raise CompletionBlockedError(
            f"historical v2 relevant judgment type is invalid: {candidate_id}"
        )
    overall = decision["overall_decision"]
    decision_valid = (
        (overall == "accept" and all(value is True for value in values))
        or (overall == "reject" and any(value is False for value in values))
        or (
            overall == "defer"
            and all(value is not False for value in values)
            and any(value is None for value in values)
        )
    )
    if not decision_valid:
        raise CompletionBlockedError(
            f"historical v2 overall decision contradicts judgments: {candidate_id}"
        )
    final_call = calls[-1] if calls and isinstance(calls[-1], dict) else None
    if not isinstance(final_call, dict):
        raise CompletionBlockedError(
            f"historical v2 completed checkpoint lacks model call: {candidate_id}"
        )
    for field, expected in {
        "terminal_status": "completed",
        "prompt_version": HISTORICAL_PROMPT_VERSION,
        "provider_profile": CANDIDATE_PARAMETERS["reviewer_provider_profile"],
        "requested_model": CANDIDATE_PARAMETERS["reviewer_model"],
        "expected_response_model": CANDIDATE_PARAMETERS[
            "expected_response_model"
        ],
        "response_model": CANDIDATE_PARAMETERS["expected_response_model"],
        "response_model_identity_status": "matched",
    }.items():
        if final_call.get(field) != expected:
            raise CompletionBlockedError(
                f"historical v2 final model call {field} is invalid: {candidate_id}"
            )
    return candidate_id, dict(row)


def _validate_historical_v2_run_contract(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    predecessor_path: Path,
    predecessor: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
        raise CompletionBlockedError("historical v2 manifest schema is unsupported")
    if manifest.get("tool_version") != HISTORICAL_CANDIDATE_TOOL_VERSION:
        raise CompletionBlockedError("historical v2 manifest tool version is unsupported")
    contract = manifest.get("run_contract")
    if not isinstance(contract, dict) or manifest.get(
        "run_contract_sha256"
    ) != sha256_value(contract):
        raise CompletionBlockedError("historical v2 run contract is stale")
    if contract.get("tool_version") != HISTORICAL_CANDIDATE_TOOL_VERSION:
        raise CompletionBlockedError("historical v2 contract tool version is stale")
    source = contract.get("candidate_manifest")
    if not isinstance(source, dict):
        raise CompletionBlockedError("historical v2 predecessor binding is missing")
    if _resolve_binding_path(source, manifest_path) != Path(predecessor_path).resolve():
        raise CompletionBlockedError("historical v2 review binds another predecessor")
    if source.get("sha256") != sha256_file(predecessor_path):
        raise CompletionBlockedError("historical v2 predecessor SHA-256 is stale")
    expected_reviewer = {
        "provider_profile": CANDIDATE_PARAMETERS["reviewer_provider_profile"],
        "model": CANDIDATE_PARAMETERS["reviewer_model"],
        "reasoning_effort": CANDIDATE_PARAMETERS["reviewer_reasoning_effort"],
        "max_output_tokens": CANDIDATE_PARAMETERS["reviewer_max_output_tokens"],
        "max_retries": CANDIDATE_PARAMETERS["max_retries"],
        "temperature": 0.0,
        "json_mode": True,
    }
    expected_protocol_identity = {
        "provider_profile": CANDIDATE_PARAMETERS["reviewer_provider_profile"],
        "requested_model": CANDIDATE_PARAMETERS["reviewer_model"],
        "expected_response_model": CANDIDATE_PARAMETERS[
            "expected_response_model"
        ],
    }
    required = (
        contract.get("batch_size") == CANDIDATE_PARAMETERS["batch_size"],
        contract.get("checkpoint_compact_every")
        == CANDIDATE_PARAMETERS["checkpoint_compact_every"],
        contract.get("max_workers") == CANDIDATE_PARAMETERS["max_workers"],
        contract.get("expected_response_model")
        == CANDIDATE_PARAMETERS["expected_response_model"],
        contract.get("source_fingerprints") == HISTORICAL_SOURCE_FINGERPRINTS,
        contract.get("prompt_contract_sha256")
        == HISTORICAL_PROMPT_CONTRACT_SHA256,
        contract.get("reviewer_spec") == expected_reviewer,
        contract.get("protocol_reviewer_identity") == expected_protocol_identity,
        contract.get("behavior_blind") is True,
        contract.get("human_gold") is False,
    )
    if not all(required):
        raise CompletionBlockedError("historical v2 frozen protocol contract changed")
    route_identity = contract.get("route_identity")
    if not isinstance(route_identity, dict):
        raise CompletionBlockedError("historical v2 route identity is missing")
    route_payload = {
        key: value
        for key, value in route_identity.items()
        if key != "route_identity_set_sha256"
    }
    records = route_identity.get("records")
    if (
        route_identity.get("schema_version")
        != "redacted-model-route-identity-set-v1"
        or route_identity.get("credentials_or_endpoints_included") is not False
        or not isinstance(records, list)
        or route_identity.get("record_count") != len(records)
        or len(records) != 1
        or route_identity.get("route_identity_set_sha256")
        != sha256_value(route_payload)
    ):
        raise CompletionBlockedError("historical v2 route identity set is stale")
    route = records[0]
    if not isinstance(route, dict):
        raise CompletionBlockedError("historical v2 route identity is stale")
    route_without_sha = {
        key: value for key, value in route.items() if key != "route_identity_sha256"
    }
    if (
        route.get("credentials_or_endpoints_included") is not False
        or route.get("provider_profile")
        != CANDIDATE_PARAMETERS["reviewer_provider_profile"]
        or route.get("requested_model") != CANDIDATE_PARAMETERS["reviewer_model"]
        or route.get("expected_response_model")
        != CANDIDATE_PARAMETERS["expected_response_model"]
        or route.get("route_identity_sha256") != sha256_value(route_without_sha)
    ):
        raise CompletionBlockedError("historical v2 route identity is stale")
    evidence_boundary = manifest.get("evidence_boundary")
    if not isinstance(evidence_boundary, dict):
        raise CompletionBlockedError(
            "historical v2 manifest evidence boundary is missing"
        )
    expected_boundary = {
        "reviewer_type": "independent_model_proxy",
        "human_gold": False,
        "behavior_blind": True,
        "target_behavior_consumed": False,
        "hf_model_execution_count": 0,
        "hf_tokenizer_execution_count": 0,
        "behavior_execution_count": 0,
        "validation_behavior_exposure_count": 0,
        "sealed_behavior_exposure_count": 0,
        "review_freeze_performed": False,
        "split_freeze_performed": False,
        "candidate_promotion_performed": False,
    }
    if any(evidence_boundary.get(field) != value for field, value in expected_boundary.items()):
        raise CompletionBlockedError(
            "historical v2 manifest evidence boundary is invalid"
        )
    units, input_bindings = _historical_v2_units(
        predecessor=predecessor,
        run_contract=contract,
        manifest_path=manifest_path,
    )
    return contract, units, input_bindings


class ExistingValidators:
    """Reuse current candidate/resolution validators without mutating artifacts."""

    def __init__(self) -> None:
        self.resolver = _load_module(
            "_candidate_stability_resolution", RESOLUTION_SCRIPT
        )
        self.candidate = self.resolver.candidate_review

    def load_postreview(self, path: Path) -> Dict[str, Any]:
        try:
            return self.resolver._load_postreview(Path(path).resolve())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(str(exc)) from exc

    def load_complete_candidate_review(
        self, *, predecessor_path: Path, manifest_path: Path
    ) -> Dict[str, Any]:
        manifest_path = Path(manifest_path).resolve()
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(str(exc)) from exc
        if manifest.get("tool_version") == HISTORICAL_CANDIDATE_TOOL_VERSION:
            return self._load_historical_v2_complete_candidate_review(
                predecessor_path=predecessor_path,
                manifest_path=manifest_path,
                manifest=manifest,
            )
        try:
            return self.resolver._load_candidate_review(
                predecessor_path=Path(predecessor_path).resolve(),
                review_manifest_path=manifest_path,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(str(exc)) from exc

    def load_current_review_units(
        self, *, predecessor_path: Path
    ) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
        try:
            _, manifest_kind, input_bindings, units = self.candidate.load_review_units(
                candidate_manifest_path=Path(predecessor_path).resolve()
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(str(exc)) from exc
        if manifest_kind != "postreview_rebuild":
            raise CompletionBlockedError("candidate review predecessor kind is invalid")
        return list(units), copy.deepcopy(input_bindings)

    def validate_current_checkpoint_row(
        self,
        row: Mapping[str, Any],
        *,
        units_by_id: Mapping[str, Any],
        run_contract_sha256: str,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            return self.candidate._validate_checkpoint_row(
                row,
                units_by_id=units_by_id,
                run_contract_sha256=run_contract_sha256,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(str(exc)) from exc

    def _load_historical_v2_complete_candidate_review(
        self,
        *,
        predecessor_path: Path,
        manifest_path: Path,
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        predecessor_path = Path(predecessor_path).resolve()
        predecessor = self.load_postreview(predecessor_path)
        _, units, _ = _validate_historical_v2_run_contract(
            manifest=manifest,
            manifest_path=manifest_path,
            predecessor_path=predecessor_path,
            predecessor=predecessor,
        )
        if manifest.get("status") != "completed":
            raise CompletionBlockedError(
                "historical v2 candidate review is not completed"
            )
        if not all(
            manifest.get(field) is True
            for field in (
                "selection_is_full_candidate_set",
                "candidate_review_complete_for_selected_scope",
                "candidate_review_complete_for_full_set",
            )
        ):
            raise CompletionBlockedError(
                "historical v2 candidate review lacks full coverage"
            )
        evidence_boundary = manifest.get("evidence_boundary")
        if not isinstance(evidence_boundary, dict):
            raise CompletionBlockedError(
                "historical v2 candidate review evidence boundary is missing"
            )
        if (
            evidence_boundary.get("reviewer_type")
            != "independent_model_proxy"
            or evidence_boundary.get("human_gold") is not False
            or evidence_boundary.get("behavior_blind") is not True
            or evidence_boundary.get("target_behavior_consumed") is not False
        ):
            raise CompletionBlockedError(
                "historical v2 candidate review evidence boundary is invalid"
            )
        for field in (
            "hf_model_execution_count",
            "hf_tokenizer_execution_count",
            "behavior_execution_count",
            "validation_behavior_exposure_count",
            "sealed_behavior_exposure_count",
        ):
            if evidence_boundary.get(field) != 0:
                raise CompletionBlockedError(
                    f"historical v2 candidate review execution boundary is invalid: {field}"
                )
        for field in (
            "review_freeze_performed",
            "split_freeze_performed",
            "candidate_promotion_performed",
        ):
            if evidence_boundary.get(field) is not False:
                raise CompletionBlockedError(
                    f"historical v2 candidate review freeze boundary is invalid: {field}"
                )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise CompletionBlockedError("historical v2 review artifacts are missing")
        checkpoint_path, checkpoints, checkpoint_binding, _ = _validate_bound_file(
            artifacts.get("checkpoint"),
            owner_path=manifest_path,
            label="historical v2 checkpoint",
            jsonl=True,
            expected_schema=HISTORICAL_CHECKPOINT_SCHEMA,
        )
        adjudications_path, adjudications, adjudications_binding, _ = (
            _validate_bound_file(
                artifacts.get("adjudications"),
                owner_path=manifest_path,
                label="historical v2 adjudications",
                jsonl=True,
                expected_schema=HISTORICAL_ADJUDICATION_SCHEMA,
            )
        )
        if len(checkpoints) != len(units) or len(adjudications) != len(units):
            raise CompletionBlockedError(
                "historical v2 review artifacts do not cover the full set"
            )
        units_by_id = {
            str(_unit_field(unit, "candidate_id")): unit for unit in units
        }
        checkpoint_index: Dict[str, Dict[str, Any]] = {}
        validated_rows: List[Dict[str, Any]] = []
        for row in checkpoints:
            candidate_id, validated = _validate_historical_v2_checkpoint_row(
                row,
                units_by_id=units_by_id,
                run_contract_sha256=str(manifest["run_contract_sha256"]),
            )
            if candidate_id in checkpoint_index:
                raise CompletionBlockedError(
                    f"duplicate historical v2 checkpoint candidate_id: {candidate_id}"
                )
            if validated.get("terminal_status") != "completed":
                raise CompletionBlockedError(
                    f"historical v2 review contains failed row: {candidate_id}"
                )
            checkpoint_index[candidate_id] = validated
            validated_rows.append(validated)
        if [row["input_index"] for row in validated_rows] != list(range(len(units))):
            raise CompletionBlockedError(
                "historical v2 checkpoint input_index sequence is not contiguous"
            )
        adjudication_index: Dict[str, Dict[str, Any]] = {}
        for row in adjudications:
            candidate_id = _required_string(
                row.get("candidate_id"), "historical v2 adjudication candidate_id"
            )
            if candidate_id in adjudication_index:
                raise CompletionBlockedError(
                    f"duplicate historical v2 adjudication candidate_id: {candidate_id}"
                )
            checkpoint = checkpoint_index.get(candidate_id)
            if checkpoint is None or checkpoint.get("strict_adjudication") != row:
                raise CompletionBlockedError(
                    f"historical v2 checkpoint/adjudication mismatch: {candidate_id}"
                )
            adjudication_index[candidate_id] = dict(row)
        unit_ids = [str(_unit_field(unit, "candidate_id")) for unit in units]
        if set(checkpoint_index) != set(unit_ids) or set(adjudication_index) != set(
            unit_ids
        ):
            raise CompletionBlockedError(
                "historical v2 candidate ID coverage is stale"
            )
        decision_counts = dict(
            sorted(Counter(str(row["overall_decision"]) for row in adjudications).items())
        )
        if (
            manifest.get("terminal_status_counts") != {"completed": len(units)}
            or manifest.get("overall_decision_counts") != decision_counts
        ):
            raise CompletionBlockedError("historical v2 terminal counts are stale")
        return {
            "manifest": dict(manifest),
            "manifest_path": manifest_path,
            "manifest_binding": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "byte_count": manifest_path.stat().st_size,
                "schema_version": CANDIDATE_MANIFEST_SCHEMA,
            },
            "checkpoint_path": checkpoint_path,
            "checkpoint_binding": {
                **checkpoint_binding,
                "path": str(checkpoint_path),
            },
            "checkpoint_rows": validated_rows,
            "adjudications_path": adjudications_path,
            "adjudications_binding": {
                **adjudications_binding,
                "path": str(adjudications_path),
            },
            "adjudications": [dict(row) for row in adjudications],
            "adjudication_index": adjudication_index,
            "units": units,
        }


def decide_candidate_action(
    *, status: str, full_count: int, completed_count: int, decision_counts: Mapping[str, int]
) -> str:
    if status == "running":
        return "running"
    if status == "completed_with_failures":
        if completed_count >= full_count:
            raise CompletionBlockedError(
                "candidate failure status has no missing technical terminal rows"
            )
        return "technical_recovery_required"
    if status != "completed":
        raise CompletionBlockedError(f"unsupported candidate status: {status}")
    if completed_count != full_count:
        raise CompletionBlockedError("candidate completed status has incomplete coverage")
    unknown = set(decision_counts) - {"accept", "reject", "defer"}
    if unknown or sum(int(value) for value in decision_counts.values()) != full_count:
        raise CompletionBlockedError("candidate decision counts are invalid")
    if int(decision_counts.get("reject", 0)) or int(decision_counts.get("defer", 0)):
        return "terminal_with_nonaccept"
    if decision_counts != {"accept": full_count}:
        raise CompletionBlockedError("candidate all-accept declaration is stale")
    return "terminal_all_accept"


class CompletionOrchestrator:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        validators: Optional[ExistingValidators] = None,
        now_fn: Callable[[], str] = utc_now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.repo_root = Path(args.repo_root).resolve()
        if self.repo_root != PROJECT_ROOT:
            raise CompletionBlockedError(
                f"repo-root must be this checkout: {PROJECT_ROOT}"
            )
        self.universe_root = Path(args.universe_root).resolve()
        self.start_round = int(args.start_round)
        self.postreview_manifest = Path(args.start_postreview_manifest).resolve()
        self.candidate_manifest = Path(args.start_candidate_review_manifest).resolve()
        self.config_path = Path(args.config).resolve()
        self.state_dir = (
            Path(args.state_dir).resolve()
            if args.state_dir is not None
            else self.universe_root / "candidate-stability-gate-v1"
        )
        self.wait_for_external_writer = bool(args.wait_for_external_writer)
        self.acknowledge_interrupted_review = bool(
            getattr(args, "acknowledge_interrupted_review", False)
        )
        self.external_poll_seconds = float(args.external_poll_seconds)
        if self.start_round < 1:
            raise CompletionBlockedError("start-round must be at least 1")
        if not 1.0 <= self.external_poll_seconds <= 60.0:
            raise CompletionBlockedError(
                "external-poll-seconds must be between 1 and 60"
            )
        if self.wait_for_external_writer and self.acknowledge_interrupted_review:
            raise CompletionBlockedError(
                "--wait-for-external-writer and --acknowledge-interrupted-review "
                "are mutually exclusive"
            )
        for label, path, directory in (
            ("universe-root", self.universe_root, True),
            ("start postreview manifest", self.postreview_manifest, False),
            ("start candidate review manifest", self.candidate_manifest, False),
            ("config", self.config_path, False),
        ):
            if directory and not path.is_dir():
                raise CompletionBlockedError(f"{label} is not a directory: {path}")
            if not directory and not path.is_file():
                raise CompletionBlockedError(f"{label} is not a file: {path}")
        if self.postreview_manifest.parent.parent != self.universe_root:
            raise CompletionBlockedError("start postreview is outside universe-root")
        if self.candidate_manifest.parent.parent != self.universe_root:
            raise CompletionBlockedError("start candidate review is outside universe-root")
        self.state_path = self.state_dir / "state.json"
        self.journal_path = self.state_dir / "journal.jsonl"
        self.lock_path = self.state_dir / "orchestrator.lock"
        self.validators = validators or ExistingValidators()
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.config = load_config(self.config_path)
        self.contract = self._build_contract()
        self.contract_sha256 = sha256_value(self.contract)
        self.state: Dict[str, Any] = {}

    def _build_contract(self) -> Dict[str, Any]:
        return {
            "tool_version": TOOL_VERSION,
            "mode": "audit_only_no_downstream_execution",
            "repo_root": str(self.repo_root),
            "universe_root": str(self.universe_root),
            "start_round": self.start_round,
            "postreview_manifest": {
                "path": str(self.postreview_manifest),
                "sha256": sha256_file(self.postreview_manifest),
            },
            # The terminal candidate manifest SHA is recorded only after the
            # external writer publishes it atomically.
            "candidate_review_manifest_path": str(self.candidate_manifest),
            "config": {
                "path": str(self.config_path),
                "file_sha256": sha256_file(self.config_path),
                "resolved_config_sha256": sha256_value(self.config),
            },
            "candidate_parameters": copy.deepcopy(CANDIDATE_PARAMETERS),
            "source_scripts": {
                CANDIDATE_REVIEW_SCRIPT.name: sha256_file(CANDIDATE_REVIEW_SCRIPT),
                RESOLUTION_SCRIPT.name: sha256_file(RESOLUTION_SCRIPT),
            },
            "safety_boundary": {
                "automatic_candidate_retry": False,
                "automatic_resolution_or_successor": False,
                "automatic_additional_review_round": False,
                "automatic_zh": False,
                "scope_owner_attestation_created_or_validated": False,
                "freeze_performed": False,
                "hf_model_or_tokenizer_executed": False,
                "behavior_executed": False,
                "validation_exposed": False,
                "sealed_exposed": False,
                "human_gold": False,
            },
        }

    def _initial_state(self) -> Dict[str, Any]:
        now = self.now_fn()
        return {
            "schema_version": STATE_SCHEMA,
            "tool_version": TOOL_VERSION,
            "contract": copy.deepcopy(self.contract),
            "contract_sha256": self.contract_sha256,
            "status": "initialized",
            "phase": "candidate_review",
            "created_at": now,
            "updated_at": now,
            "safety_boundary": copy.deepcopy(self.contract["safety_boundary"]),
        }

    def _persist_state(
        self, *, event: str, details: Optional[Mapping[str, Any]] = None
    ) -> None:
        self.state["updated_at"] = self.now_fn()
        atomic_write_json(self.state_path, self.state)
        append_journal(
            self.journal_path,
            {
                "schema_version": JOURNAL_SCHEMA,
                "tool_version": TOOL_VERSION,
                "at": self.state["updated_at"],
                "event": event,
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "state_sha256": sha256_file(self.state_path),
                "details": copy.deepcopy(dict(details or {})),
            },
        )

    def _load_or_initialize_state(self) -> None:
        if not self.state_path.is_file():
            self.state = self._initial_state()
            self._persist_state(
                event="state_initialized", details={"round": self.start_round}
            )
            return
        state = read_json(self.state_path)
        if state.get("schema_version") != STATE_SCHEMA:
            raise CompletionBlockedError("unsupported stability state schema")
        if state.get("contract_sha256") != self.contract_sha256 or state.get(
            "contract"
        ) != self.contract:
            raise CompletionBlockedError(
                "stability audit contract changed; refusing unsafe resume"
            )
        if state.get("safety_boundary") != self.contract["safety_boundary"]:
            raise CompletionBlockedError("stability safety boundary changed")
        self.state = state

    def _set_status(
        self,
        *,
        status: str,
        phase: str,
        event: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.state["status"] = status
        self.state["phase"] = phase
        self._persist_state(event=event, details=details)

    def _candidate_manifest_contract(
        self, *, predecessor: Mapping[str, Any]
    ) -> Dict[str, Any]:
        manifest = read_json(self.candidate_manifest)
        if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA:
            raise CompletionBlockedError("candidate manifest schema is unsupported")
        tool_version = manifest.get("tool_version")
        if tool_version not in SUPPORTED_CANDIDATE_TOOL_VERSIONS:
            raise CompletionBlockedError("candidate runner version is unsupported")
        if tool_version == HISTORICAL_CANDIDATE_TOOL_VERSION:
            _validate_historical_v2_run_contract(
                manifest=manifest,
                manifest_path=self.candidate_manifest,
                predecessor_path=self.postreview_manifest,
                predecessor=predecessor,
            )
            if manifest["run_contract"].get("config_sha256") != sha256_value(
                self.config
            ):
                raise CompletionBlockedError(
                    "historical v2 config binding differs from the supplied config"
                )
            return manifest
        contract = manifest.get("run_contract")
        if not isinstance(contract, dict) or manifest.get(
            "run_contract_sha256"
        ) != sha256_value(contract):
            raise CompletionBlockedError("candidate run contract is stale")
        source = contract.get("candidate_manifest")
        if not isinstance(source, dict):
            raise CompletionBlockedError("candidate predecessor binding is missing")
        if _resolve_binding_path(source, self.candidate_manifest) != self.postreview_manifest:
            raise CompletionBlockedError("candidate review binds another predecessor")
        if source.get("sha256") != sha256_file(self.postreview_manifest):
            raise CompletionBlockedError("candidate predecessor SHA-256 is stale")
        expected_count = len(predecessor["distractors"]) + len(predecessor["neutrals"])
        required = (
            contract.get("tool_version") == CURRENT_CANDIDATE_TOOL_VERSION,
            contract.get("selected_candidate_count") == expected_count,
            contract.get("full_candidate_count") == expected_count,
            contract.get("selection_is_full_candidate_set") is True,
            contract.get("batch_size") == CANDIDATE_PARAMETERS["batch_size"],
            contract.get("checkpoint_compact_every")
            == CANDIDATE_PARAMETERS["checkpoint_compact_every"],
            contract.get("max_workers") == CANDIDATE_PARAMETERS["max_workers"],
            contract.get("expected_response_model")
            == CANDIDATE_PARAMETERS["expected_response_model"],
            contract.get("config_sha256") == sha256_value(self.config),
            contract.get("source_fingerprints")
            == self.validators.candidate.source_fingerprints(),
        )
        if not all(required):
            raise CompletionBlockedError("candidate parameters or full-set coverage changed")
        expected_reviewer = {
            "provider_profile": CANDIDATE_PARAMETERS["reviewer_provider_profile"],
            "model": CANDIDATE_PARAMETERS["reviewer_model"],
            "reasoning_effort": CANDIDATE_PARAMETERS["reviewer_reasoning_effort"],
            "max_output_tokens": CANDIDATE_PARAMETERS[
                "reviewer_max_output_tokens"
            ],
            "max_retries": CANDIDATE_PARAMETERS["max_retries"],
            "temperature": 0.0,
            "json_mode": True,
        }
        if contract.get("reviewer_spec") != expected_reviewer:
            raise CompletionBlockedError("candidate reviewer parameters changed")
        return manifest

    def inspect_candidate(self) -> Dict[str, Any]:
        predecessor = self.validators.load_postreview(self.postreview_manifest)
        manifest = self._candidate_manifest_contract(predecessor=predecessor)
        status = str(manifest.get("status") or "")
        expected_count = len(predecessor["distractors"]) + len(predecessor["neutrals"])
        if status == "running":
            return {
                "action": "running",
                "manifest": manifest,
                "predecessor": predecessor,
            }
        terminal_counts = manifest.get("terminal_status_counts")
        if not isinstance(terminal_counts, dict) or sum(
            int(value) for value in terminal_counts.values()
        ) != expected_count:
            raise CompletionBlockedError("candidate terminal counts are incomplete")
        completed_count = int(terminal_counts.get("completed", 0))
        if status == "completed_with_failures":
            if int(terminal_counts.get("failed", 0)) < 1:
                raise CompletionBlockedError(
                    "candidate failure status has no failed rows"
                )
            return {
                "action": decide_candidate_action(
                    status=status,
                    full_count=expected_count,
                    completed_count=completed_count,
                    decision_counts={},
                ),
                "manifest": manifest,
            }
        review = self.validators.load_complete_candidate_review(
            predecessor_path=self.postreview_manifest,
            manifest_path=self.candidate_manifest,
        )
        decision_counts = dict(
            sorted(
                Counter(
                    str(row["overall_decision"])
                    for row in review["adjudications"]
                ).items()
            )
        )
        if terminal_counts != {"completed": expected_count}:
            raise CompletionBlockedError("candidate completed counts are stale")
        if manifest.get("overall_decision_counts") != decision_counts:
            raise CompletionBlockedError("candidate decision counts are stale")
        return {
            "action": decide_candidate_action(
                status=status,
                full_count=expected_count,
                completed_count=completed_count,
                decision_counts=decision_counts,
            ),
            "manifest": manifest,
            "review": review,
            "decision_counts": decision_counts,
        }

    def _load_interrupted_candidate_review(
        self,
        *,
        predecessor: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Read and validate a stopped writer's durable partial evidence."""

        if manifest.get("status") != "running":
            raise CompletionBlockedError(
                "interrupted-review acknowledgement requires status=running"
            )
        tool_version = str(manifest.get("tool_version") or "")
        run_contract = manifest.get("run_contract")
        if not isinstance(run_contract, dict):
            raise CompletionBlockedError("interrupted review run contract is missing")
        run_contract_sha256 = str(manifest.get("run_contract_sha256") or "")
        if run_contract_sha256 != sha256_value(run_contract):
            raise CompletionBlockedError("interrupted review run contract is stale")

        if tool_version == HISTORICAL_CANDIDATE_TOOL_VERSION:
            _, units, input_bindings = _validate_historical_v2_run_contract(
                manifest=manifest,
                manifest_path=self.candidate_manifest,
                predecessor_path=self.postreview_manifest,
                predecessor=predecessor,
            )
            checkpoint_schema = HISTORICAL_CHECKPOINT_SCHEMA

            def validate_row(
                row: Mapping[str, Any], *, units_by_id: Mapping[str, Any]
            ) -> Tuple[str, Dict[str, Any]]:
                return _validate_historical_v2_checkpoint_row(
                    row,
                    units_by_id=units_by_id,
                    run_contract_sha256=run_contract_sha256,
                )

        elif tool_version == CURRENT_CANDIDATE_TOOL_VERSION:
            units, input_bindings = self.validators.load_current_review_units(
                predecessor_path=self.postreview_manifest
            )
            if run_contract.get("input_artifacts") != input_bindings:
                raise CompletionBlockedError(
                    "current candidate input artifact bindings are stale"
                )
            unit_ids = [str(_unit_field(unit, "candidate_id")) for unit in units]
            if (
                run_contract.get("selected_candidate_count") != len(units)
                or run_contract.get("full_candidate_count") != len(units)
                or run_contract.get("selection_is_full_candidate_set") is not True
                or run_contract.get("selected_candidate_ids_sha256")
                != sha256_value(unit_ids)
            ):
                raise CompletionBlockedError(
                    "current candidate selected scope is stale"
                )
            checkpoint_schema = str(self.validators.candidate.CHECKPOINT_SCHEMA)

            def validate_row(
                row: Mapping[str, Any], *, units_by_id: Mapping[str, Any]
            ) -> Tuple[str, Dict[str, Any]]:
                return self.validators.validate_current_checkpoint_row(
                    row,
                    units_by_id=units_by_id,
                    run_contract_sha256=run_contract_sha256,
                )

        else:
            raise CompletionBlockedError(
                "interrupted review runner version is unsupported"
            )

        unit_ids = [str(_unit_field(unit, "candidate_id")) for unit in units]
        expected_indices = [int(_unit_field(unit, "input_index")) for unit in units]
        if len(set(unit_ids)) != len(unit_ids):
            raise CompletionBlockedError("interrupted review input IDs are duplicated")
        if expected_indices != list(range(len(units))):
            raise CompletionBlockedError(
                "interrupted review input_index sequence is not contiguous"
            )
        units_by_id = {
            str(_unit_field(unit, "candidate_id")): unit for unit in units
        }

        manifest_payload = self.candidate_manifest.read_bytes()
        try:
            observed_manifest = json.loads(manifest_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompletionBlockedError(
                "interrupted candidate manifest is malformed"
            ) from exc
        if observed_manifest != manifest:
            raise CompletionBlockedError(
                "interrupted candidate manifest changed during inspection"
            )

        checkpoint_path = self.candidate_manifest.parent / CANDIDATE_CHECKPOINT_NAME
        journal_path = self.candidate_manifest.parent / CANDIDATE_JOURNAL_NAME
        checkpoint_exists = checkpoint_path.is_file()
        journal_exists = journal_path.is_file()
        checkpoint_payload = checkpoint_path.read_bytes() if checkpoint_exists else b""
        journal_payload = journal_path.read_bytes() if journal_exists else b""
        if checkpoint_payload and not checkpoint_payload.endswith((b"\n", b"\r")):
            raise CompletionBlockedError(
                "interrupted compact checkpoint has an unterminated tail"
            )
        checkpoint_rows = _parse_complete_jsonl_payload(
            checkpoint_payload,
            path=checkpoint_path,
            label="interrupted compact checkpoint",
        )
        journal_rows, journal_integrity = _parse_interrupted_journal_payload(
            journal_payload,
            path=journal_path,
        )

        checkpoint_index: Dict[str, Dict[str, Any]] = {}
        checkpoint_ids: List[str] = []
        for row in checkpoint_rows:
            candidate_id, validated = validate_row(row, units_by_id=units_by_id)
            if candidate_id in checkpoint_index:
                raise CompletionBlockedError(
                    f"duplicate interrupted compact checkpoint candidate_id: {candidate_id}"
                )
            checkpoint_ids.append(candidate_id)
            checkpoint_index[candidate_id] = validated

        journal_seen: set[str] = set()
        journal_duplicate_ids: set[str] = set()
        journal_ids: List[str] = []
        merged = dict(checkpoint_index)
        for row in journal_rows:
            candidate_id, validated = validate_row(row, units_by_id=units_by_id)
            if candidate_id in journal_seen:
                journal_duplicate_ids.add(candidate_id)
            journal_seen.add(candidate_id)
            journal_ids.append(candidate_id)
            # This is the runner's durable resume rule: later journal rows
            # supersede the compact base (and earlier journal rows) by ID.
            merged[candidate_id] = validated

        ordered_rows = [
            merged[candidate_id] for candidate_id in unit_ids if candidate_id in merged
        ]
        merged_ids = [str(row["candidate_id"]) for row in ordered_rows]
        merged_indices = [int(row["input_index"]) for row in ordered_rows]
        if len(set(merged_ids)) != len(merged_ids) or len(set(merged_indices)) != len(
            merged_indices
        ):
            raise CompletionBlockedError(
                "interrupted merged checkpoint contains duplicate identity"
            )
        if merged_indices != list(range(len(ordered_rows))):
            raise CompletionBlockedError(
                "interrupted merged checkpoint input_index is not a contiguous prefix"
            )

        completed_rows = [
            row for row in ordered_rows if row.get("terminal_status") == "completed"
        ]
        failed_rows = [
            row for row in ordered_rows if row.get("terminal_status") == "failed"
        ]
        completed_ids = {str(row["candidate_id"]) for row in completed_rows}
        failed_ids = {str(row["candidate_id"]) for row in failed_rows}
        pending_ids = [candidate_id for candidate_id in unit_ids if candidate_id not in completed_ids]
        unrecorded_ids = [candidate_id for candidate_id in unit_ids if candidate_id not in merged]
        adjudications = [dict(row["strict_adjudication"]) for row in completed_rows]
        decision_counts = dict(
            sorted(Counter(str(row["overall_decision"]) for row in adjudications).items())
        )

        snapshots = (
            (self.candidate_manifest, manifest_payload, True),
            (checkpoint_path, checkpoint_payload, checkpoint_exists),
            (journal_path, journal_payload, journal_exists),
        )
        for path, payload, existed in snapshots:
            if path.is_file() != existed or (existed and path.read_bytes() != payload):
                raise CompletionBlockedError(
                    f"interrupted review evidence changed during inspection: {path}"
                )

        checkpoint_binding = _artifact_binding_from_payload(
            checkpoint_path,
            checkpoint_payload,
            exists=checkpoint_exists,
            record_count=len(checkpoint_rows),
            schema_version=checkpoint_schema,
        )
        journal_binding = _artifact_binding_from_payload(
            journal_path,
            journal_payload,
            exists=journal_exists,
            record_count=len(journal_rows),
            schema_version=checkpoint_schema,
        )
        return {
            "manifest": dict(manifest),
            "manifest_path": self.candidate_manifest,
            "manifest_binding": {
                "path": str(self.candidate_manifest),
                "sha256": sha256_bytes(manifest_payload),
                "byte_count": len(manifest_payload),
                "schema_version": CANDIDATE_MANIFEST_SCHEMA,
                "status": "running",
                "record_count": len(units),
            },
            "input_bindings": input_bindings,
            "checkpoint_path": checkpoint_path,
            "checkpoint_binding": checkpoint_binding,
            "journal_path": journal_path,
            "journal_binding": journal_binding,
            "checkpoint_rows": ordered_rows,
            "adjudications": adjudications,
            "adjudications_binding": {
                "derived_from": "validated_checkpoint_plus_journal",
                "schema_version": (
                    HISTORICAL_ADJUDICATION_SCHEMA
                    if tool_version == HISTORICAL_CANDIDATE_TOOL_VERSION
                    else str(self.validators.candidate.ADJUDICATION_SCHEMA)
                ),
                "record_count": len(adjudications),
                "sha256": sha256_value(adjudications),
            },
            "units": units,
            "decision_counts": decision_counts,
            "partial_counts": {
                "expected": len(units),
                "completed": len(completed_rows),
                "pending": len(pending_ids),
                "failed_terminal": len(failed_rows),
                "unrecorded": len(unrecorded_ids),
            },
            "completed_candidate_ids": [
                candidate_id for candidate_id in unit_ids if candidate_id in completed_ids
            ],
            "pending_candidate_ids": pending_ids,
            "failed_candidate_ids": [
                candidate_id for candidate_id in unit_ids if candidate_id in failed_ids
            ],
            "unrecorded_candidate_ids": unrecorded_ids,
            "integrity": {
                "input_indices_expected_contiguous": True,
                "merged_input_indices_contiguous_prefix": True,
                "checkpoint_candidate_ids_unique": True,
                "journal_candidate_ids_unique": not journal_duplicate_ids,
                "journal_duplicate_candidate_ids": sorted(journal_duplicate_ids),
                "checkpoint_journal_overlap_candidate_ids": sorted(
                    set(checkpoint_ids).intersection(journal_ids)
                ),
                "journal_last_write_wins_merge_applied": True,
                "merged_candidate_ids_unique": True,
                "merged_input_indices_unique": True,
                "checkpoint_unterminated_tail_absent": True,
                **journal_integrity,
                "snapshot_stable_during_audit": True,
            },
            "source_snapshots": snapshots,
        }

    @staticmethod
    def _checkpoint_call_signature(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        calls = row.get("model_calls")
        if not isinstance(calls, list):
            return None
        for call in reversed(calls):
            if not isinstance(call, dict) or call.get("terminal_status") != "completed":
                continue
            return {
                "prompt_sha256": call.get("prompt_sha256"),
                "batch_id": call.get("batch_id"),
                "ordered_candidate_ids_sha256": call.get(
                    "ordered_candidate_ids_sha256"
                ),
                "prompt_version": call.get("prompt_version"),
                "requested_model": call.get("requested_model"),
                "response_model": call.get("response_model"),
            }
        return None

    @staticmethod
    def _adjudication_index(
        review: Mapping[str, Any]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        result: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in review["adjudications"]:
            key = (str(row["candidate_id"]), str(row["candidate_row_sha256"]))
            if key in result:
                raise CompletionBlockedError(
                    f"duplicate candidate stability key: {key[0]}"
                )
            result[key] = dict(row)
        return result

    @staticmethod
    def _checkpoint_index(review: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        supplied_rows = review.get("checkpoint_rows")
        rows = (
            [dict(row) for row in supplied_rows]
            if isinstance(supplied_rows, list)
            else read_jsonl(Path(review["checkpoint_path"]))
        )
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id or candidate_id in result:
                raise CompletionBlockedError(
                    "candidate checkpoint IDs are missing or duplicated"
                )
            result[candidate_id] = row
        return result

    def _cross_round_stability_report(
        self,
        *,
        current_review: Mapping[str, Any],
        review_interrupted: bool = False,
    ) -> Dict[str, Any]:
        previous_round = self.start_round - 1
        previous_postreview = (
            self.universe_root
            / f"postreview-successor-r{previous_round:02d}"
            / "postreview_rebuild_manifest.json"
        )
        previous_manifest = (
            self.universe_root
            / f"candidate-review-r{previous_round:02d}"
            / "candidate_review_run_manifest.json"
        )
        if not previous_postreview.is_file() or not previous_manifest.is_file():
            raise CompletionBlockedError(
                "previous candidate round is unavailable for stability audit"
            )
        previous_review = self.validators.load_complete_candidate_review(
            predecessor_path=previous_postreview,
            manifest_path=previous_manifest,
        )
        previous_contract = previous_review.get("manifest", {}).get("run_contract")
        if not isinstance(previous_contract, dict) or previous_contract.get(
            "config_sha256"
        ) != sha256_value(self.config):
            raise CompletionBlockedError(
                "previous candidate review config binding differs from current audit"
            )
        previous_index = self._adjudication_index(previous_review)
        current_index = self._adjudication_index(current_review)
        previous_checkpoints = self._checkpoint_index(previous_review)
        current_checkpoints = self._checkpoint_index(current_review)
        overlap = sorted(set(previous_index).intersection(current_index))
        transitions: Counter[str] = Counter()
        drifts: List[Dict[str, Any]] = []
        same_prompt_batch_drift_count = 0
        for key in overlap:
            previous = previous_index[key]
            current = current_index[key]
            previous_decision = str(previous["overall_decision"])
            current_decision = str(current["overall_decision"])
            transitions[f"{previous_decision}->{current_decision}"] += 1
            if previous_decision == current_decision:
                continue
            candidate_id, candidate_row_sha256 = key
            previous_call = self._checkpoint_call_signature(
                previous_checkpoints[candidate_id]
            )
            current_call = self._checkpoint_call_signature(
                current_checkpoints[candidate_id]
            )
            same_prompt_and_batch = bool(
                previous_call is not None
                and current_call is not None
                and previous_call == current_call
            )
            same_prompt_batch_drift_count += int(same_prompt_and_batch)
            drifts.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_row_sha256": candidate_row_sha256,
                    "candidate_kind": current.get("candidate_kind"),
                    "target_base_fact_id": current.get("target_base_fact_id"),
                    "source_base_fact_id": current.get("source_base_fact_id"),
                    "previous_decision": previous_decision,
                    "current_decision": current_decision,
                    "target_base_fact_row_sha256_unchanged": (
                        previous.get("target_base_fact_row_sha256")
                        == current.get("target_base_fact_row_sha256")
                    ),
                    "source_base_fact_row_sha256_unchanged": (
                        previous.get("source_base_fact_row_sha256")
                        == current.get("source_base_fact_row_sha256")
                    ),
                    "same_prompt_and_batch_signature": same_prompt_and_batch,
                    "previous_call_signature": previous_call,
                    "current_call_signature": current_call,
                }
            )
        report: Dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": "protocol_review_required",
            "coverage_complete": not review_interrupted,
            "review_interrupted": review_interrupted,
            "comparison": {
                "previous_round": previous_round,
                "current_round": self.start_round,
                "identity_key": ["candidate_id", "candidate_row_sha256"],
                "previous_postreview_manifest": {
                    "path": str(previous_postreview.resolve()),
                    "sha256": sha256_file(previous_postreview),
                },
                "previous_candidate_review_manifest": {
                    "path": str(previous_manifest.resolve()),
                    "sha256": sha256_file(previous_manifest),
                },
                "current_postreview_manifest": {
                    "path": str(self.postreview_manifest),
                    "sha256": sha256_file(self.postreview_manifest),
                },
                "current_candidate_review_manifest": {
                    "path": str(self.candidate_manifest),
                    "sha256": sha256_file(self.candidate_manifest),
                },
                "previous_adjudications": copy.deepcopy(
                    previous_review["adjudications_binding"]
                ),
                "current_adjudications": copy.deepcopy(
                    current_review["adjudications_binding"]
                ),
                "previous_checkpoint": copy.deepcopy(
                    previous_review["checkpoint_binding"]
                ),
                "current_checkpoint": copy.deepcopy(
                    current_review["checkpoint_binding"]
                ),
            },
            "counts": {
                "previous_candidate_count": len(previous_index),
                "current_candidate_count": len(current_index),
                "exact_unchanged_candidate_overlap": len(overlap),
                "unchanged_candidate_decision_drift": len(drifts),
                "same_prompt_and_batch_decision_drift": same_prompt_batch_drift_count,
            },
            "decision_transition_counts": dict(sorted(transitions.items())),
            "decision_drifts": drifts,
            "gate": {
                "unchanged_decision_drift_is_zero": not drifts,
                "optional_stopping_risk_detected": bool(drifts),
                "automatic_resolution_successor_review_loop_authorized": False,
                "automatic_zh_start_authorized": False,
                "protocol_review_required": True,
            },
            "evidence_boundary": {
                "report_is_offline": True,
                "model_calls_made": 0,
                "source_outputs_modified": False,
                "human_gold": False,
                "hf_model_executed": False,
                "hf_tokenizer_executed": False,
                "behavior_executed": False,
                "validation_exposed": False,
                "sealed_exposed": False,
            },
        }
        if review_interrupted:
            partial_counts = current_review.get("partial_counts")
            integrity = current_review.get("integrity")
            if not isinstance(partial_counts, dict) or not isinstance(integrity, dict):
                raise CompletionBlockedError(
                    "interrupted review lacks partial coverage evidence"
                )
            completed_ids = list(current_review.get("completed_candidate_ids") or [])
            pending_ids = list(current_review.get("pending_candidate_ids") or [])
            failed_ids = list(current_review.get("failed_candidate_ids") or [])
            unrecorded_ids = list(current_review.get("unrecorded_candidate_ids") or [])
            if (
                partial_counts.get("expected")
                != partial_counts.get("completed") + partial_counts.get("pending")
                or partial_counts.get("pending") != len(pending_ids)
                or partial_counts.get("completed") != len(completed_ids)
            ):
                raise CompletionBlockedError(
                    "interrupted review completed/pending partition is stale"
                )
            report["counts"].update(
                {
                    "expected_candidate_count": partial_counts["expected"],
                    "completed_candidate_count": partial_counts["completed"],
                    "pending_candidate_count": partial_counts["pending"],
                    "failed_terminal_candidate_count": partial_counts[
                        "failed_terminal"
                    ],
                    "unrecorded_candidate_count": partial_counts["unrecorded"],
                }
            )
            report["completed"] = {
                "record_count": len(completed_ids),
                "ordered_candidate_ids_sha256": sha256_value(completed_ids),
            }
            report["pending"] = {
                "record_count": len(pending_ids),
                "ordered_candidate_ids_sha256": sha256_value(pending_ids),
            }
            report["failed_terminal"] = {
                "record_count": len(failed_ids),
                "ordered_candidate_ids_sha256": sha256_value(failed_ids),
            }
            report["unrecorded"] = {
                "record_count": len(unrecorded_ids),
                "ordered_candidate_ids_sha256": sha256_value(unrecorded_ids),
            }
            report["interruption_acknowledgement"] = {
                "required_cli_flag": "--acknowledge-interrupted-review",
                "operator_asserted_external_writer_stopped": True,
                "writer_liveness_inferred_by_tool": False,
                "terminal_manifest_synthesized": False,
            }
            report["interrupted_review_evidence"] = {
                "input_manifest": copy.deepcopy(current_review["manifest_binding"]),
                "input_artifacts": copy.deepcopy(current_review["input_bindings"]),
                "checkpoint": copy.deepcopy(current_review["checkpoint_binding"]),
                "journal": copy.deepcopy(current_review["journal_binding"]),
                "merged_checkpoint": {
                    "record_count": len(current_review["checkpoint_rows"]),
                    "ordered_rows_sha256": sha256_value(
                        current_review["checkpoint_rows"]
                    ),
                    "merge_rule": (
                        "compact_checkpoint_base_then_journal_last_write_wins_by_candidate_id"
                    ),
                },
                "derived_completed_adjudications": copy.deepcopy(
                    current_review["adjudications_binding"]
                ),
            }
            report["integrity_checks"] = copy.deepcopy(integrity)
            report["comparison"]["current_journal"] = copy.deepcopy(
                current_review["journal_binding"]
            )
            report["gate"].update(
                {
                    "coverage_complete": False,
                    "review_interrupted": True,
                    "partial_evidence_can_authorize_downstream": False,
                }
            )
            snapshots = current_review.get("source_snapshots")
            if not isinstance(snapshots, tuple):
                raise CompletionBlockedError(
                    "interrupted review source snapshot contract is missing"
                )
            for path, payload, existed in snapshots:
                if path.is_file() != existed or (
                    existed and path.read_bytes() != payload
                ):
                    raise CompletionBlockedError(
                        f"interrupted review evidence changed before report: {path}"
                    )

        report_name = (
            "candidate_cross_round_partial_stability"
            if review_interrupted
            else "candidate_cross_round_stability"
        )
        report_path = self.state_dir / (
            f"{report_name}_r{previous_round:02d}_r{self.start_round:02d}.json"
        )
        if report_path.is_file():
            if read_json(report_path) != report:
                raise CompletionBlockedError(
                    "existing stability report differs from current evidence"
                )
        else:
            atomic_write_json(report_path, report)
        return {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
            "report": report,
        }

    def _validate_protocol_review_state(self) -> Dict[str, Any]:
        binding = self.state.get("candidate_stability_report")
        if not isinstance(binding, dict):
            raise CompletionBlockedError("protocol-review state lacks stability report")
        path = Path(str(binding.get("path") or "")).resolve()
        if not path.is_file() or binding.get("sha256") != sha256_file(path):
            raise CompletionBlockedError("candidate stability report binding is stale")
        report = read_json(path)
        gate = report.get("gate")
        if (
            report.get("schema_version") != REPORT_SCHEMA
            or report.get("tool_version") != TOOL_VERSION
            or report.get("status") != "protocol_review_required"
            or not isinstance(gate, dict)
            or gate.get("automatic_resolution_successor_review_loop_authorized")
            is not False
            or gate.get("automatic_zh_start_authorized") is not False
            or gate.get("protocol_review_required") is not True
        ):
            raise CompletionBlockedError("candidate stability report gate is stale")
        comparison = report.get("comparison")
        if not isinstance(comparison, dict):
            raise CompletionBlockedError("candidate stability comparison is missing")
        review_interrupted = report.get("review_interrupted") is True
        if review_interrupted:
            if (
                report.get("coverage_complete") is not False
                or gate.get("coverage_complete") is not False
                or gate.get("review_interrupted") is not True
                or gate.get("partial_evidence_can_authorize_downstream") is not False
                or gate.get("protocol_review_required") is not True
            ):
                raise CompletionBlockedError(
                    "interrupted candidate stability gate is stale"
                )
            interrupted_evidence = report.get("interrupted_review_evidence")
            if not isinstance(interrupted_evidence, dict) or comparison.get(
                "current_adjudications"
            ) != interrupted_evidence.get("derived_completed_adjudications"):
                raise CompletionBlockedError(
                    "interrupted candidate derived adjudication binding is stale"
                )
        elif report.get("coverage_complete") is not True:
            raise CompletionBlockedError(
                "terminal candidate stability coverage declaration is stale"
            )
        required_file_bindings = [
            "previous_postreview_manifest",
            "previous_candidate_review_manifest",
            "current_postreview_manifest",
            "current_candidate_review_manifest",
            "previous_adjudications",
            "previous_checkpoint",
            "current_checkpoint",
        ]
        if not review_interrupted:
            required_file_bindings.append("current_adjudications")
        else:
            required_file_bindings.append("current_journal")
        for label in required_file_bindings:
            source = comparison.get(label)
            if not isinstance(source, dict):
                raise CompletionBlockedError(
                    f"candidate stability source binding is missing: {label}"
                )
            source_path = _resolve_binding_path(source, path)
            expected_exists = source.get("exists", True)
            if not isinstance(expected_exists, bool):
                raise CompletionBlockedError(
                    f"candidate stability source existence is invalid: {label}"
                )
            if source_path.is_file() is not expected_exists:
                raise CompletionBlockedError(
                    f"candidate stability source existence is stale: {label}"
                )
            if expected_exists and source.get("sha256") != sha256_file(source_path):
                raise CompletionBlockedError(
                    f"candidate stability source binding is stale: {label}"
                )
        result = {
            "status": "protocol_review_required",
            "state_path": str(self.state_path),
            "stability_report_path": str(path),
            "unchanged_candidate_decision_drift": report["counts"][
                "unchanged_candidate_decision_drift"
            ],
            "same_prompt_and_batch_decision_drift": report["counts"][
                "same_prompt_and_batch_decision_drift"
            ],
            "automatic_downstream_execution_performed": False,
            "hf_or_behavior_work_performed": False,
        }
        if review_interrupted:
            counts = report.get("counts")
            if not isinstance(counts, dict):
                raise CompletionBlockedError(
                    "interrupted candidate stability counts are missing"
                )
            result.update(
                {
                    "coverage_complete": False,
                    "review_interrupted": True,
                    "completed_candidate_count": counts.get(
                        "completed_candidate_count"
                    ),
                    "pending_candidate_count": counts.get(
                        "pending_candidate_count"
                    ),
                }
            )
        return result

    def _run_locked(self) -> Dict[str, Any]:
        if self.state.get("status") == "protocol_review_required":
            return self._validate_protocol_review_state()
        if self.state.get("status") == "blocked_fail_closed":
            return {
                "status": "blocked_fail_closed",
                "state_path": str(self.state_path),
                "reason": self.state.get("blocked_reason"),
                "automatic_downstream_execution_performed": False,
            }
        outcome = self.inspect_candidate()
        action = outcome["action"]
        if action == "running":
            if self.acknowledge_interrupted_review:
                partial_review = self._load_interrupted_candidate_review(
                    predecessor=outcome["predecessor"],
                    manifest=outcome["manifest"],
                )
                stability = self._cross_round_stability_report(
                    current_review=partial_review,
                    review_interrupted=True,
                )
                report = stability["report"]
                self.state["candidate_stability_report"] = {
                    "path": stability["path"],
                    "sha256": stability["sha256"],
                }
                self.state["audited_candidate_review"] = {
                    "round": self.start_round,
                    "postreview_manifest": {
                        "path": str(self.postreview_manifest),
                        "sha256": sha256_file(self.postreview_manifest),
                    },
                    "candidate_review_manifest": copy.deepcopy(
                        partial_review["manifest_binding"]
                    ),
                    "review_interrupted": True,
                    "coverage_complete": False,
                    "completed_candidate_count": report["counts"][
                        "completed_candidate_count"
                    ],
                    "pending_candidate_count": report["counts"][
                        "pending_candidate_count"
                    ],
                }
                self.state["protocol_review_reason"] = (
                    "operator_acknowledged_interrupted_candidate_review"
                )
                self._set_status(
                    status="protocol_review_required",
                    phase="candidate_stability_gate",
                    event="interrupted_candidate_stability_gate_stopped",
                    details={
                        "report_path": stability["path"],
                        "review_interrupted": True,
                        "coverage_complete": False,
                        "completed_candidate_count": report["counts"][
                            "completed_candidate_count"
                        ],
                        "pending_candidate_count": report["counts"][
                            "pending_candidate_count"
                        ],
                        "automatic_downstream_execution_performed": False,
                    },
                )
                return self._validate_protocol_review_state()
            raise ExternalWriterRunning(self.candidate_manifest)
        if action == "technical_recovery_required":
            self._set_status(
                status="candidate_terminal_recovery_required",
                phase="candidate_review",
                event="candidate_terminal_recovery_required",
                details={
                    "manifest_path": str(self.candidate_manifest),
                    "required_manual_command_mode": (
                        "same_parameters_resume_retry_failed"
                    ),
                    "automatic_retry_performed": False,
                },
            )
            return {
                "status": "candidate_terminal_recovery_required",
                "state_path": str(self.state_path),
                "candidate_review_manifest": str(self.candidate_manifest),
                "automatic_command_started": False,
                "hf_or_behavior_work_performed": False,
            }
        if action not in {"terminal_with_nonaccept", "terminal_all_accept"}:
            raise CompletionBlockedError(f"unsupported candidate action: {action}")
        stability = self._cross_round_stability_report(
            current_review=outcome["review"]
        )
        report = stability["report"]
        self.state["candidate_stability_report"] = {
            "path": stability["path"],
            "sha256": stability["sha256"],
        }
        self.state["audited_candidate_review"] = {
            "round": self.start_round,
            "postreview_manifest": {
                "path": str(self.postreview_manifest),
                "sha256": sha256_file(self.postreview_manifest),
            },
            "candidate_review_manifest": {
                "path": str(self.candidate_manifest),
                "sha256": sha256_file(self.candidate_manifest),
            },
            "decision_counts": copy.deepcopy(outcome["decision_counts"]),
        }
        self.state["protocol_review_reason"] = (
            "unchanged_candidate_decision_drift_detected"
            if report["counts"]["unchanged_candidate_decision_drift"]
            else "automatic_multi_round_optional_stopping_not_authorized"
        )
        self._set_status(
            status="protocol_review_required",
            phase="candidate_stability_gate",
            event="candidate_stability_gate_stopped",
            details={
                "report_path": stability["path"],
                "unchanged_candidate_decision_drift": report["counts"][
                    "unchanged_candidate_decision_drift"
                ],
                "automatic_downstream_execution_performed": False,
            },
        )
        return self._validate_protocol_review_state()

    def _wait_for_external_terminal(self) -> None:
        while True:
            manifest = read_json(self.candidate_manifest)
            status = str(manifest.get("status") or "")
            if status in {"completed", "completed_with_failures"}:
                return
            if status != "running":
                raise CompletionBlockedError(
                    f"external writer entered unsupported status: {status}"
                )
            self.sleep_fn(self.external_poll_seconds)

    def run(self) -> Dict[str, Any]:
        while True:
            external: Optional[ExternalWriterRunning] = None
            with exclusive_lock(self.lock_path):
                self._load_or_initialize_state()
                try:
                    return self._run_locked()
                except ExternalWriterRunning as exc:
                    external = exc
                    self._set_status(
                        status="waiting_for_external_writer",
                        phase="external_writer_barrier",
                        event="external_writer_observed",
                        details={"manifest_path": str(exc.manifest_path)},
                    )
                except CompletionBlockedError as exc:
                    self.state["blocked_reason"] = str(exc)
                    self._set_status(
                        status="blocked_fail_closed",
                        phase="blocked",
                        event="blocked_fail_closed",
                        details={"reason": str(exc)},
                    )
                    return {
                        "status": "blocked_fail_closed",
                        "state_path": str(self.state_path),
                        "reason": str(exc),
                        "automatic_downstream_execution_performed": False,
                    }
            if external is None:
                raise CompletionBlockedError("external writer wait state was lost")
            if not self.wait_for_external_writer:
                return {
                    "status": "waiting_for_external_writer",
                    "state_path": str(self.state_path),
                    "manifest_path": str(external.manifest_path),
                    "automatic_command_started": False,
                    "hf_or_behavior_work_performed": False,
                }
            # The pre-existing writer does not honor this tool's lock.  Poll
            # without the lock and acquire it only after a terminal manifest is
            # visible.  No liveness guess and no resume is ever attempted.
            self._wait_for_external_terminal()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--universe-root", type=Path, required=True)
    parser.add_argument("--start-round", type=int, required=True)
    parser.add_argument("--start-postreview-manifest", type=Path, required=True)
    parser.add_argument("--start-candidate-review-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--wait-for-external-writer", action="store_true")
    parser.add_argument(
        "--acknowledge-interrupted-review",
        action="store_true",
        help=(
            "operator confirms the status=running writer has stopped; emit only "
            "an incomplete checkpoint+journal stability diagnostic"
        ),
    )
    parser.add_argument("--external-poll-seconds", type=float, default=30.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = CompletionOrchestrator(args).run()
    except (CompletionError, OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "status": "blocked_fail_closed",
            "reason": f"{type(exc).__name__}: {exc}",
            "automatic_downstream_execution_performed": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") == "protocol_review_required":
        return 0
    if result.get("status") == "waiting_for_external_writer":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
