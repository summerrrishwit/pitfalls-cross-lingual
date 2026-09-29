#!/usr/bin/env python3
"""Freeze the reviewed full public-benchmark pool immediately before exact HF.

This is an offline, fail-closed finalizer.  It validates the complete reviewed
post-review rebuild, distractor/Neutral adjudications, zh translation review,
formal-universe-v2 lineage, and the scope-owner historical-exposure
attestation.  Only after every binding and record-level coverage check passes
does it atomically emit the exact reviewed bundle plus review, split, and
pre-exact-HF gate manifests.

The accepted semantic claim is deliberately narrow: completion of the frozen,
declared bounded candidate protocol.  The tool never claims exhaustive
semantic-near-duplicate recall, human-gold review, a formal probe relation,
exact-HF binding, target-model behavior, hidden states, or interventions.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import datetime as dt
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factual_pitfalls.route_identity import validate_route_identity_set  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "full_public_benchmark_8969_pre_exact_hf_zh_v1.json"

TOOL_VERSION = "full-public-benchmark-pre-exact-hf-freezer-v1"
STATUS = "completed_for_declared_bounded_protocol"
BUNDLE_SCHEMA_VERSION = "public-benchmark-full-exact-reviewed-base-fact-v1"
REVIEW_FREEZE_SCHEMA_VERSION = "public-benchmark-full-review-freeze-manifest-v1"
SPLIT_FREEZE_SCHEMA_VERSION = "public-benchmark-full-split-freeze-manifest-v1"
GATE_SCHEMA_VERSION = "public-benchmark-full-pre-exact-hf-gate-manifest-v1"

EXPECTED_ORIGINAL_UNIVERSE_COUNT = 8969
SPLITS = ("development", "validation", "sealed")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FreezeValidationError(ValueError):
    """An input contract is incomplete, inconsistent, or stale."""


def _load_sibling(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


postreview_tool = _load_sibling(
    "_full_freeze_postreview", "materialize_full_public_benchmark_postreview.py"
)
candidate_tool = _load_sibling(
    "_full_freeze_candidate_review", "run_full_public_benchmark_candidate_review.py"
)
candidate_resolution_tool = _load_sibling(
    "_full_freeze_candidate_resolution",
    "resolve_full_public_benchmark_candidate_review.py",
)
accepted_projection_tool = _load_sibling(
    "_full_freeze_accepted_projection",
    "materialize_full_public_benchmark_accepted_projection.py",
)
zh_accepted_projection_tool = _load_sibling(
    "_full_freeze_zh_accepted_projection",
    "materialize_full_public_benchmark_zh_accepted_projection.py",
)
zh_tool = _load_sibling(
    "_full_freeze_zh_review", "run_full_public_benchmark_zh_review.py"
)
formal_universe_tool = _load_sibling(
    "_full_freeze_formal_universe_v2",
    "materialize_full_public_benchmark_formal_universe_v2.py",
)
exposure_tool = _load_sibling(
    "_full_freeze_historical_exposure",
    "materialize_full_public_benchmark_historical_exposure_contract_v2.py",
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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeValidationError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezeValidationError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeValidationError(f"Invalid UTF-8 JSONL: {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FreezeValidationError(
                f"Invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise FreezeValidationError(
                f"Expected JSON object at {path}:{line_number}"
            )
        rows.append(row)
    if not rows and not allow_empty:
        raise FreezeValidationError(f"JSONL is empty: {path}")
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
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows),
    )


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreezeValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FreezeValidationError(f"{label} must be a non-negative integer")
    return value


def _required_sha256(value: Any, label: str) -> str:
    value = _required_string(value, label)
    if SHA256_RE.fullmatch(value) is None:
        raise FreezeValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _resolve_binding_path(binding: Mapping[str, Any], owner_path: Path, label: str) -> Path:
    raw = binding.get("path") or binding.get("filename")
    path = Path(_required_string(raw, f"{label}.path"))
    return path.resolve() if path.is_absolute() else (owner_path.parent / path).resolve()


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    source = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(Path(output_path).resolve() if output_path else source),
        "sha256": sha256_file(source),
        "byte_count": source.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _verify_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    jsonl: bool,
    expected_schema: Optional[str] = None,
    explicit_path: Optional[Path] = None,
    allow_empty: bool = False,
) -> Tuple[Path, Any, Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise FreezeValidationError(f"{label} binding must be an object")
    path = _resolve_binding_path(binding, Path(owner_path).resolve(), label)
    if explicit_path is not None and path != Path(explicit_path).resolve():
        raise FreezeValidationError(f"{label} binding does not match supplied path")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = _required_sha256(binding.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected_sha:
        raise FreezeValidationError(f"{label} SHA-256 binding is stale")
    byte_count = _required_nonnegative_int(binding.get("byte_count"), f"{label}.byte_count")
    if path.stat().st_size != byte_count:
        raise FreezeValidationError(f"{label} byte_count binding is stale")
    if expected_schema is not None and binding.get("schema_version") not in {
        expected_schema,
        None,
    }:
        raise FreezeValidationError(f"{label} schema_version is unsupported")
    value: Any = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if jsonl:
        count = _required_nonnegative_int(binding.get("record_count"), f"{label}.record_count")
        if count != len(value):
            raise FreezeValidationError(f"{label} record_count binding is stale")
        schemas = sorted(
            {
                str(row.get("schema_version"))
                for row in value
                if row.get("schema_version") is not None
            }
        )
        if expected_schema is not None and value and schemas != [expected_schema]:
            raise FreezeValidationError(f"{label} row schema_version is unsupported")
        if "schema_versions" in binding and binding.get("schema_versions") != schemas:
            raise FreezeValidationError(f"{label} schema_versions binding is stale")
    return path, value, dict(binding)


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        identifier = _required_string(row.get(field), f"{label}[{position}].{field}")
        if identifier in result:
            raise FreezeValidationError(f"Duplicate {label} {field}: {identifier}")
        result[identifier] = dict(row)
    return result


def _same_binding(
    actual: Any, expected: Mapping[str, Any], *, owner_path: Path, label: str
) -> None:
    if not isinstance(actual, dict):
        raise FreezeValidationError(f"{label} binding is missing")
    actual_path = _resolve_binding_path(actual, owner_path, label)
    expected_path = Path(str(expected["path"])).resolve()
    if actual_path != expected_path:
        raise FreezeValidationError(f"{label} path does not match exact artifact")
    if actual.get("sha256") != expected.get("sha256"):
        raise FreezeValidationError(f"{label}.sha256 does not match exact artifact")
    for field in ("byte_count", "record_count", "schema_version"):
        if field in actual and field in expected and actual.get(field) != expected.get(field):
            raise FreezeValidationError(f"{label}.{field} does not match exact artifact")


def _zero_execution_contract(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise FreezeValidationError(f"{label} is missing")
    for field in (
        "hf_model_execution_count",
        "hf_tokenizer_execution_count",
        "behavior_execution_count",
        "validation_behavior_exposure_count",
        "sealed_behavior_exposure_count",
    ):
        if value.get(field) != 0:
            raise FreezeValidationError(f"{label}.{field} must equal 0")
    for field in ("hidden_state_collection_count", "intervention_count"):
        if field in value and value.get(field) != 0:
            raise FreezeValidationError(f"{label}.{field} must equal 0")
    for field in (
        "hf_model_execution",
        "hf_tokenizer_execution",
        "hf_model_executed",
        "hf_tokenizer_executed",
        "behavior_executed",
        "validation_exposed",
        "sealed_exposed",
    ):
        if field in value and value.get(field) is not False:
            raise FreezeValidationError(f"{label}.{field} must equal false")


def _validate_config(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    config = zh_tool.load_config(path)
    contract = config.get("pre_exact_hf_contract")
    expected = {
        "fact_review_evidence": "independent_model_proxy_not_human_gold",
        "translation_review_evidence": "independent_model_proxy_not_human_gold",
        "semantic_closure_scope": (
            "frozen_deterministic_candidate_protocol_with_terminal_pair_adjudication"
        ),
        "semantic_near_duplicate_recall_guaranteed": False,
        "historical_exposure_authority": (
            "predeclared_pending_scope_owner_attestation"
        ),
        "review_and_split_freeze_required_before_hf_binding": True,
        "hf_model_execution_authorized": False,
        "hf_tokenizer_execution_authorized": False,
        "validation_behavior_exposure_authorized": False,
        "sealed_behavior_exposure_authorized": False,
    }
    if not isinstance(contract, dict) or any(
        contract.get(field) != value for field, value in expected.items()
    ):
        raise FreezeValidationError("pre_exact_hf_contract is unsupported or incomplete")
    return {
        "value": config,
        "binding": _binding(path, schema_version=str(config.get("config_version") or "")),
    }


def _configured_route_protocols(config: Mapping[str, Any]) -> Dict[str, str]:
    profiles = config.get("provider_profiles")
    if not isinstance(profiles, dict):
        raise FreezeValidationError("config.provider_profiles must be an object")
    protocols: Dict[str, str] = {}
    for provider_profile, profile in profiles.items():
        if not isinstance(provider_profile, str) or not provider_profile.strip():
            raise FreezeValidationError("config provider profile name is invalid")
        if not isinstance(profile, dict):
            raise FreezeValidationError(
                f"config provider profile is invalid: {provider_profile}"
            )
        protocol = profile.get("protocol")
        if not isinstance(protocol, str) or not protocol.strip():
            raise FreezeValidationError(
                f"config provider protocol is invalid: {provider_profile}"
            )
        protocols[provider_profile] = protocol
    return protocols


def _configured_zh_routes(
    config: Mapping[str, Any],
) -> Tuple[Tuple[str, str, str], Tuple[str, str, str]]:
    model_roles = config.get("model_roles")
    translation = model_roles.get("translation") if isinstance(model_roles, dict) else None
    if not isinstance(translation, dict):
        raise FreezeValidationError("config translation model roles are missing")
    routes: List[Tuple[str, str, str]] = []
    for role, key in (("generator", "primary"), ("reviewer", "reviewer")):
        model = translation.get(key)
        if not isinstance(model, dict):
            raise FreezeValidationError(f"config zh {role} model identity is missing")
        routes.append(
            (
                _required_string(
                    model.get("provider_profile"),
                    f"config zh {role} provider_profile",
                ),
                _required_string(model.get("model"), f"config zh {role} model"),
                _required_string(
                    model.get("expected_response_model"),
                    f"config zh {role} expected_response_model",
                ),
            )
        )
    return routes[0], routes[1]


def _validate_formal_lineage_and_exposure(
    *,
    universe_manifest_path: Path,
    contract_path: Path,
    attestation_path: Path,
    expected_original_count: int,
) -> Dict[str, Any]:
    universe_manifest_path = Path(universe_manifest_path).resolve()
    contract_path = Path(contract_path).resolve()
    attestation_path = Path(attestation_path).resolve()
    try:
        universe_validation = formal_universe_tool.validate_successor_universe(
            universe_manifest_path=universe_manifest_path,
            scope_owner_attestation_path=attestation_path,
            expected_record_count=expected_original_count,
        )
        exposure_validation = exposure_tool.validate_contract(
            contract_path=contract_path,
            scope_owner_attestation_path=attestation_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise FreezeValidationError(str(exc)) from exc

    if universe_validation.get("status") != (
        "valid_pending_successor_with_compatible_attestation_candidate"
    ):
        raise FreezeValidationError("formal-universe-v2 has no compatible attestation")
    if universe_validation.get("future_attestation_binding_candidate") is None:
        raise FreezeValidationError("formal-universe-v2 attestation binding is missing")
    if exposure_validation.get("scope_owner_attestation_valid") is not True:
        raise FreezeValidationError("historical exposure is not scope-owner confirmed")
    if exposure_validation.get("ready_for_successor_universe_binding") is not True:
        raise FreezeValidationError("historical exposure attestation is not binding-ready")

    manifest = read_json(universe_manifest_path)
    if manifest.get("schema_version") != formal_universe_tool.MANIFEST_SCHEMA_VERSION:
        raise FreezeValidationError("formal-universe-v2 schema_version is unsupported")
    if manifest.get("record_count") != expected_original_count:
        raise FreezeValidationError("formal-universe-v2 record_count is stale")
    authority = manifest.get("historical_exposure_authority")
    if not isinstance(authority, dict):
        raise FreezeValidationError("formal-universe-v2 historical authority is missing")
    contract_binding = authority.get("contract")
    contract_bound_path, _, verified_contract_binding = _verify_binding(
        contract_binding,
        owner_path=universe_manifest_path,
        label="historical exposure contract",
        jsonl=False,
        expected_schema=exposure_tool.CONTRACT_SCHEMA_VERSION,
        explicit_path=contract_path,
    )
    items_path, items, items_binding = _verify_binding(
        manifest.get("items"),
        owner_path=universe_manifest_path,
        label="formal-universe-v2 items",
        jsonl=True,
        expected_schema=formal_universe_tool.ITEM_SCHEMA_VERSION,
    )
    ids = [
        _required_string(row.get("source_base_fact_id"), "formal universe source_base_fact_id")
        for row in items
    ]
    if len(ids) != len(set(ids)) or len(ids) != expected_original_count:
        raise FreezeValidationError("formal-universe-v2 item ID coverage is invalid")
    if manifest.get("ordered_source_base_fact_ids_sha256") != sha256_value(ids):
        raise FreezeValidationError("formal-universe-v2 ordered ID digest is stale")

    lineage = manifest.get("lineage")
    if not isinstance(lineage, dict):
        raise FreezeValidationError("formal-universe-v2 predecessor lineage is missing")
    source_manifest_path, source_manifest, source_manifest_binding = _verify_binding(
        lineage.get("predecessor_manifest"),
        owner_path=universe_manifest_path,
        label="formal-universe-v2 predecessor manifest",
        jsonl=False,
        expected_schema=exposure_tool.SOURCE_UNIVERSE_SCHEMA_VERSION,
    )
    source_items_path, source_items, source_items_binding = _verify_binding(
        lineage.get("predecessor_items"),
        owner_path=universe_manifest_path,
        label="formal-universe-v2 predecessor items",
        jsonl=True,
        expected_schema=exposure_tool.SOURCE_ITEM_SCHEMA_VERSION,
    )
    source_universe_id = _required_string(
        lineage.get("predecessor_universe_id"),
        "formal-universe-v2 predecessor_universe_id",
    )
    if source_manifest.get("universe_id") != source_universe_id:
        raise FreezeValidationError("formal-universe-v2 predecessor universe_id is stale")
    _verify_binding(
        source_manifest.get("items"),
        owner_path=source_manifest_path,
        label="predecessor universe items",
        jsonl=True,
        expected_schema=exposure_tool.SOURCE_ITEM_SCHEMA_VERSION,
        explicit_path=source_items_path,
    )
    source_ids = [
        _required_string(
            row.get("source_base_fact_id"),
            "predecessor universe source_base_fact_id",
        )
        for row in source_items
    ]
    if source_ids != ids:
        raise FreezeValidationError(
            "formal-universe-v2 predecessor and successor scopes differ"
        )

    attestation_candidate = universe_validation["future_attestation_binding_candidate"]
    attestation_verified_path, attestation, attestation_binding = _verify_binding(
        attestation_candidate,
        owner_path=universe_manifest_path,
        label="scope-owner attestation",
        jsonl=False,
        expected_schema=exposure_tool.ATTESTATION_SCHEMA_VERSION,
        explicit_path=attestation_path,
    )
    if exposure_validation.get("contract_path") != str(contract_bound_path):
        raise FreezeValidationError("exposure validation targets a different contract")
    if exposure_validation.get("contract_sha256") != verified_contract_binding["sha256"]:
        raise FreezeValidationError("exposure validation contract SHA is stale")
    if exposure_validation.get("attestation_path") != str(attestation_verified_path):
        raise FreezeValidationError("exposure validation targets a different attestation")
    if exposure_validation.get("attestation_sha256") != attestation_binding["sha256"]:
        raise FreezeValidationError("exposure validation attestation SHA is stale")
    for field in ("contract_id", "attestation_challenge_sha256"):
        expected = authority.get(field)
        actual = (
            exposure_validation.get(field)
            if field == "contract_id"
            else attestation_candidate.get(field)
        )
        if actual != expected:
            raise FreezeValidationError(f"historical exposure {field} binding is stale")

    known_ids = exposure_validation.get(
        "known_historical_exposure_source_base_fact_ids"
    )
    if not isinstance(known_ids, list) or known_ids != sorted(set(known_ids)):
        raise FreezeValidationError("known historical exposure IDs are invalid")
    if not set(known_ids).issubset(set(ids)):
        raise FreezeValidationError("known historical exposure IDs are outside universe")
    if exposure_validation.get("known_historical_exposure_count") != len(known_ids):
        raise FreezeValidationError("known historical exposure count is stale")
    if attestation.get("known_historical_exposure_source_base_fact_ids") != known_ids:
        raise FreezeValidationError("attestation exposure registry is stale")

    cohort_item_by_source = {
        str(row["source_base_fact_id"]): str(row["cohort_item_id"]) for row in items
    }
    return {
        "manifest": manifest,
        "manifest_path": universe_manifest_path,
        "manifest_binding": _binding(
            universe_manifest_path,
            schema_version=formal_universe_tool.MANIFEST_SCHEMA_VERSION,
        ),
        "items": items,
        "items_path": items_path,
        "items_binding": items_binding,
        "source_base_fact_ids": ids,
        "cohort_item_by_source": cohort_item_by_source,
        "source_universe_id": source_universe_id,
        "source_universe_manifest": source_manifest,
        "source_universe_manifest_path": source_manifest_path,
        "source_universe_manifest_binding": source_manifest_binding,
        "source_universe_items": source_items,
        "source_universe_items_path": source_items_path,
        "source_universe_items_binding": source_items_binding,
        "contract_path": contract_bound_path,
        "contract_binding": verified_contract_binding,
        "attestation_path": attestation_verified_path,
        "attestation": attestation,
        "attestation_binding": {
            **attestation_binding,
            "contract_id": exposure_validation["contract_id"],
            "attestation_challenge_sha256": authority[
                "attestation_challenge_sha256"
            ],
            "known_historical_exposure_count": len(known_ids),
            "ordered_known_historical_exposure_source_base_fact_ids_sha256": (
                sha256_value(known_ids)
            ),
        },
        "known_exposure_ids": known_ids,
        "attested_by": exposure_validation.get("attested_by"),
        "attested_at": exposure_validation.get("attested_at"),
        "attested_through": exposure_validation.get("attested_through"),
    }


def _validate_semantic_runtime_identity(
    *,
    run_manifest_binding: Any,
    owner_path: Path,
    semantic_contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if semantic_contract.get(
        "authority_mode", "direct_semantic_review_run"
    ) != "direct_semantic_review_run":
        raise FreezeValidationError("postreview semantic authority mode is not direct")
    run_manifest_path, run_manifest, verified_binding = _verify_binding(
        run_manifest_binding,
        owner_path=owner_path,
        label="postreview semantic run manifest",
        jsonl=False,
        expected_schema=postreview_tool.semantic_runner.RUN_MANIFEST_SCHEMA,
    )
    semantic_runner = postreview_tool.semantic_runner
    if run_manifest.get("tool_version") != semantic_runner.TOOL_VERSION:
        raise FreezeValidationError("semantic run manifest tool_version is unsupported")
    if run_manifest.get("status") != "completed":
        raise FreezeValidationError("semantic run manifest is incomplete")
    run_contract = run_manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise FreezeValidationError("semantic run contract is missing")
    run_contract_sha256 = sha256_value(run_contract)
    if run_manifest.get("run_contract_sha256") != run_contract_sha256:
        raise FreezeValidationError("semantic run contract SHA is stale")
    if run_contract.get("tool_version") != semantic_runner.TOOL_VERSION:
        raise FreezeValidationError("semantic run contract tool_version is unsupported")
    if run_contract.get("config_sha256") != sha256_value(config):
        raise FreezeValidationError("semantic run config SHA is stale")
    if run_contract.get("source_fingerprints") != semantic_runner.source_fingerprints():
        raise FreezeValidationError("semantic run source fingerprints are stale")
    try:
        protocol_identity = semantic_runner.protocol_reviewer_identity(config)
    except ValueError as exc:
        raise FreezeValidationError(str(exc)) from exc
    if run_contract.get("protocol_reviewer_identity") != protocol_identity:
        raise FreezeValidationError("semantic run protocol reviewer identity is stale")
    reviewer_spec = run_contract.get("reviewer_spec")
    if not isinstance(reviewer_spec, dict) or any(
        (
            reviewer_spec.get("provider_profile")
            != protocol_identity["provider_profile"],
            reviewer_spec.get("model") != protocol_identity["requested_model"],
            run_contract.get("expected_response_model")
            != protocol_identity["expected_response_model"],
        )
    ):
        raise FreezeValidationError("semantic run model identity is unsupported")
    route_identity = run_contract.get("route_identity")
    if not isinstance(route_identity, dict):
        raise FreezeValidationError("semantic run route identity is missing")
    try:
        validate_route_identity_set(
            route_identity,
            expected_routes=[
                (
                    protocol_identity["provider_profile"],
                    protocol_identity["requested_model"],
                    protocol_identity["expected_response_model"],
                )
            ],
            expected_protocols=_configured_route_protocols(config),
        )
    except ValueError as exc:
        raise FreezeValidationError(f"semantic run route identity is invalid: {exc}") from exc
    if semantic_contract.get("semantic_run_contract_sha256") != run_contract_sha256:
        raise FreezeValidationError("postreview semantic run contract binding is stale")
    if semantic_contract.get("route_identity_set_sha256") != route_identity.get(
        "route_identity_set_sha256"
    ):
        raise FreezeValidationError("postreview semantic route identity binding is stale")
    if semantic_contract.get("source_run_contract_sha256", {
        "direct": run_contract_sha256
    }) != {"direct": run_contract_sha256}:
        raise FreezeValidationError("postreview direct semantic contract map is stale")
    if semantic_contract.get("source_route_identity_set_sha256", {
        "direct": route_identity["route_identity_set_sha256"]
    }) != {"direct": route_identity["route_identity_set_sha256"]}:
        raise FreezeValidationError("postreview direct semantic route map is stale")
    return {
        "authority_mode": "direct_semantic_review_run",
        "manifest": run_manifest,
        "manifest_path": run_manifest_path,
        "manifest_binding": verified_binding,
        "run_contract_sha256": run_contract_sha256,
        "route_identity": route_identity,
        "snapshot_bindings": [verified_binding],
    }


def _validate_semantic_composite_identity(
    *,
    composite_manifest_binding: Any,
    owner_path: Path,
    semantic_inputs: Mapping[str, Any],
    semantic_contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Replay both source authorities without collapsing their identities."""

    composite_tool = postreview_tool._load_semantic_composite_builder()
    composite_path, _, verified_binding = _verify_binding(
        composite_manifest_binding,
        owner_path=owner_path,
        label="postreview semantic composite manifest",
        jsonl=False,
        expected_schema=composite_tool.MANIFEST_SCHEMA,
    )
    try:
        authority = composite_tool.load_composite_authority(composite_path)
    except (ValueError, FileNotFoundError) as exc:
        raise FreezeValidationError(
            f"semantic composite authority is invalid: {exc}"
        ) from exc
    parent = authority["parent"]
    supplement = authority["supplement"]
    parent_contract = parent.manifest.get("run_contract")
    supplement_contract = supplement["manifest"].get("run_contract")
    if not isinstance(parent_contract, dict) or not isinstance(
        supplement_contract, dict
    ):
        raise FreezeValidationError("semantic composite source contracts are missing")
    config_sha = sha256_value(config)
    if parent_contract.get("config_sha256") != config_sha or supplement_contract.get(
        "config_sha256"
    ) != config_sha:
        raise FreezeValidationError("semantic composite source config SHA is stale")

    parent_identity = parent_contract.get("protocol_reviewer_identity")
    if not isinstance(parent_identity, dict):
        raise FreezeValidationError("semantic composite parent identity is missing")
    try:
        configured_reviewer_identity = (
            postreview_tool.semantic_runner.protocol_reviewer_identity(config)
        )
    except ValueError as exc:
        raise FreezeValidationError(str(exc)) from exc
    if parent_identity != configured_reviewer_identity:
        raise FreezeValidationError(
            "semantic composite parent reviewer differs from config"
        )
    supplement_identity = supplement_contract.get("protocol_reviewer_identity")
    reviewer_spec = supplement_contract.get("reviewer_model_spec")
    if (
        supplement_identity != configured_reviewer_identity
        or not isinstance(reviewer_spec, dict)
        or reviewer_spec.get("provider_profile")
        != configured_reviewer_identity["provider_profile"]
        or reviewer_spec.get("model")
        != configured_reviewer_identity["requested_model"]
        or reviewer_spec.get("expected_response_model")
        != configured_reviewer_identity["expected_response_model"]
    ):
        raise FreezeValidationError(
            "semantic composite supplement reviewer differs from config"
        )
    try:
        validate_route_identity_set(
            parent_contract.get("route_identity"),
            expected_routes=[
                (
                    parent_identity["provider_profile"],
                    parent_identity["requested_model"],
                    parent_identity["expected_response_model"],
                )
            ],
            expected_protocols=_configured_route_protocols(config),
        )
        generator_spec = supplement_contract["generator_model_spec"]
        validate_route_identity_set(
            supplement_contract.get("route_identity"),
            expected_routes=[
                (
                    generator_spec["provider_profile"],
                    generator_spec["model"],
                    generator_spec["expected_response_model"],
                ),
                (
                    reviewer_spec["provider_profile"],
                    reviewer_spec["model"],
                    reviewer_spec["expected_response_model"],
                ),
            ],
            expected_protocols=_configured_route_protocols(config),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FreezeValidationError(
            f"semantic composite route identity is invalid: {exc}"
        ) from exc

    expected_contracts = authority["source_run_contract_sha256"]
    expected_routes = authority["source_route_identity_set_sha256"]
    if semantic_contract.get("authority_mode") != (
        "composite_semantic_review_authority"
    ):
        raise FreezeValidationError("postreview semantic authority mode is not composite")
    if semantic_contract.get("semantic_run_contract_sha256") is not None or (
        semantic_contract.get("route_identity_set_sha256") is not None
    ):
        raise FreezeValidationError("composite semantic authority claims one direct run")
    if semantic_contract.get("source_run_contract_sha256") != expected_contracts:
        raise FreezeValidationError("semantic composite run contract map is stale")
    if semantic_contract.get("source_route_identity_set_sha256") != expected_routes:
        raise FreezeValidationError("semantic composite route identity map is stale")

    candidate_binding = semantic_inputs.get("candidate_manifest")
    composite_candidate = authority["manifest"].get("candidate_manifest")
    if not isinstance(candidate_binding, dict) or not isinstance(
        composite_candidate, dict
    ) or _resolve_binding_path(
        candidate_binding, owner_path, "postreview semantic candidate manifest"
    ) != _resolve_binding_path(
        composite_candidate,
        authority["manifest_path"],
        "semantic composite candidate manifest",
    ) or candidate_binding.get("sha256") != composite_candidate.get("sha256"):
        raise FreezeValidationError("postreview semantic candidate binding is stale")
    adjudication_binding = semantic_inputs.get("adjudications")
    if not isinstance(adjudication_binding, dict) or _resolve_binding_path(
        adjudication_binding, owner_path, "postreview semantic adjudications"
    ) != authority["adjudications_path"] or adjudication_binding.get(
        "sha256"
    ) != authority["adjudications_binding"]["sha256"]:
        raise FreezeValidationError("postreview composite adjudication binding is stale")
    return {
        "authority_mode": "composite_semantic_review_authority",
        "manifest": authority["manifest"],
        "manifest_path": authority["manifest_path"],
        "manifest_binding": verified_binding,
        "source_run_contract_sha256": expected_contracts,
        "source_route_identity_set_sha256": expected_routes,
        "snapshot_bindings": authority["snapshot_bindings"],
    }


def _validate_postreview(
    path: Path, *, config: Mapping[str, Any]
) -> Dict[str, Any]:
    path = Path(path).resolve()
    manifest = read_json(path)
    manifest_schema = manifest.get("schema_version")
    supported_schemas = {
        postreview_tool.MANIFEST_SCHEMA_VERSION,
        candidate_resolution_tool.SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA,
    }
    if manifest_schema not in supported_schemas:
        raise FreezeValidationError("postreview rebuild schema_version is unsupported")
    if manifest.get("status") != (
        "postreview_rebuilt_bounded_semantic_recall_not_frozen"
    ):
        raise FreezeValidationError("postreview rebuild is not complete")

    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict):
        raise FreezeValidationError("postreview safety_contract is missing")
    required_false = (
        "canonical_freeze_emitted",
        "review_freeze_emitted",
        "split_freeze_emitted",
        "distractor_candidates_verified",
        "neutral_candidates_verified_unrelated",
        "translation_review_complete",
        "historical_exposure_complete_for_full_pool",
        "hf_checkpoint_bound",
        "hf_tokenizer_bound",
        "hf_model_executed",
        "hf_tokenizer_executed",
        "behavior_executed",
        "validation_exposed",
        "sealed_exposed",
        "perturbation_authorized",
        "path_not_token_authorized",
        "human_gold",
    )
    if any(safety.get(field) is not False for field in required_false):
        raise FreezeValidationError("postreview safety flags are invalid")
    if safety.get("output_is_provisional") is not True or safety.get(
        "split_recomputed"
    ) is not True:
        raise FreezeValidationError("postreview provisional/split contract is invalid")

    fact_contract = manifest.get("fact_review_contract")
    if not isinstance(fact_contract, dict) or not all(
        (
            fact_contract.get("mode")
            in {"legacy_fact_review_apply", "fact_resolution_successor_cohort"},
            fact_contract.get("scope_exactly_matches_input_universe") is True,
            fact_contract.get("selection_is_complete_retained_cohort") is True,
            fact_contract.get("revise_defer_missing_count") == 0,
            fact_contract.get("proxy_review_only") is True,
            fact_contract.get("human_gold") is False,
        )
    ):
        raise FreezeValidationError("postreview fact review is incomplete")
    semantic = manifest.get("semantic_review_contract")
    if not isinstance(semantic, dict) or not all(
        (
            semantic.get("candidate_adjudication_complete_for_bounded_set") is True,
            semantic.get("runtime_identity_evidence_bound") is True,
            semantic.get("candidate_generation_is_bounded") is True,
            semantic.get("all_record_pairs_enumerated") is False,
            semantic.get("semantic_near_duplicate_recall_guaranteed") is False,
            semantic.get("semantic_closure_complete_for_full_pool") is False,
            semantic.get("reviewer_evidence_is_human_gold") is False,
        )
    ):
        raise FreezeValidationError("declared bounded semantic protocol is invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise FreezeValidationError("postreview inputs are missing")
    semantic_inputs = inputs.get("semantic_review")
    if not isinstance(semantic_inputs, dict):
        raise FreezeValidationError("postreview semantic review inputs are missing")
    authority_mode = semantic.get("authority_mode", "direct_semantic_review_run")
    input_authority_mode = semantic_inputs.get(
        "authority_mode", "direct_semantic_review_run"
    )
    if input_authority_mode != authority_mode:
        raise FreezeValidationError("postreview semantic authority mode is inconsistent")
    direct_binding = semantic_inputs.get("run_manifest")
    composite_binding = semantic_inputs.get("composite_manifest")
    if authority_mode == "direct_semantic_review_run":
        if not isinstance(direct_binding, dict) or composite_binding is not None:
            raise FreezeValidationError(
                "postreview semantic run manifest violates direct/composite XOR contract"
            )
        semantic_runtime = _validate_semantic_runtime_identity(
            run_manifest_binding=direct_binding,
            owner_path=path,
            semantic_contract=semantic,
            config=config,
        )
    else:
        if direct_binding is not None or not isinstance(composite_binding, dict):
            raise FreezeValidationError(
                "postreview composite semantic authority violates XOR contract"
            )
        semantic_runtime = _validate_semantic_composite_identity(
            composite_manifest_binding=composite_binding,
            owner_path=path,
            semantic_inputs=semantic_inputs,
            semantic_contract=semantic,
            config=config,
        )
    integrity_ref = manifest.get("integrity")
    if not isinstance(integrity_ref, dict) or integrity_ref.get(
        "all_checks_passed"
    ) is not True:
        raise FreezeValidationError("postreview integrity status is incomplete")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise FreezeValidationError("postreview outputs are missing")
    specs = {
        "full_base_facts": (True, postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION, False),
        "cohort_exclusions": (True, postreview_tool.EXCLUSION_SCHEMA_VERSION, True),
        "leakage_components": (True, postreview_tool.pre_hf.COMPONENT_SCHEMA_VERSION, False),
        "split_manifest": (False, postreview_tool.pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION, False),
        "distractor_candidates": (True, postreview_tool.pre_hf.DISTRACTOR_SCHEMA_VERSION, False),
        "neutral_reference_candidates": (True, postreview_tool.pre_hf.NEUTRAL_SCHEMA_VERSION, False),
        "candidate_shortfalls": (True, postreview_tool.pre_hf.SHORTFALL_SCHEMA_VERSION, True),
        "integrity_audit": (False, postreview_tool.INTEGRITY_SCHEMA_VERSION, False),
        "summary": (False, postreview_tool.SUMMARY_SCHEMA_VERSION, False),
    }
    artifacts: Dict[str, Dict[str, Any]] = {}
    for label, (jsonl, schema, allow_empty) in specs.items():
        artifact_path, value, binding = _verify_binding(
            outputs.get(label),
            owner_path=path,
            label=f"postreview {label}",
            jsonl=jsonl,
            expected_schema=schema,
            allow_empty=allow_empty,
        )
        binding["path"] = str(artifact_path)
        artifacts[label] = {"path": artifact_path, "value": value, "binding": binding}

    rows = artifacts["full_base_facts"]["value"]
    exclusions = artifacts["cohort_exclusions"]["value"]
    components = artifacts["leakage_components"]["value"]
    distractors = artifacts["distractor_candidates"]["value"]
    neutrals = artifacts["neutral_reference_candidates"]["value"]
    shortfalls = artifacts["candidate_shortfalls"]["value"]
    split_manifest = artifacts["split_manifest"]["value"]
    integrity = artifacts["integrity_audit"]["value"]
    summary = artifacts["summary"]["value"]
    if shortfalls:
        raise FreezeValidationError("candidate shortfalls must be empty before freeze")
    if integrity.get("all_checks_passed") is not True:
        raise FreezeValidationError("postreview integrity audit did not pass")
    limitations = integrity.get("limitations")
    if not isinstance(limitations, dict) or not all(
        (
            limitations.get("semantic_candidate_generation_is_bounded") is True,
            limitations.get("all_record_pairs_enumerated") is False,
            limitations.get("semantic_near_duplicate_recall_guaranteed") is False,
            limitations.get("semantic_closure_complete_for_full_pool") is False,
        )
    ):
        raise FreezeValidationError("postreview semantic limitations were weakened")
    _zero_execution_contract(summary.get("execution"), "postreview summary.execution")

    row_index = _unique_index(rows, "base_fact_id", "postreview facts")
    exclusion_index = _unique_index(exclusions, "base_fact_id", "postreview exclusions")
    if set(row_index).intersection(exclusion_index):
        raise FreezeValidationError("retained and excluded postreview IDs overlap")
    for base_fact_id, row in row_index.items():
        if row.get("human_gold") is not False:
            raise FreezeValidationError(f"postreview fact became human gold: {base_fact_id}")
        if row.get("canonical_status") != "pending_review":
            raise FreezeValidationError(f"postreview fact is not pending freeze: {base_fact_id}")
        if row.get("split_status") != "provisional_not_frozen":
            raise FreezeValidationError(f"postreview split is not provisional: {base_fact_id}")
        if row.get("split_assignment") not in SPLITS:
            raise FreezeValidationError(f"postreview split is invalid: {base_fact_id}")
        if row.get("hf_model_execution_status") != "not_run" or row.get(
            "hf_tokenizer_execution_status"
        ) != "not_run":
            raise FreezeValidationError(f"postreview fact contains HF execution: {base_fact_id}")
        if row.get("probe_relation_id") is not None:
            raise FreezeValidationError(f"provisional relation was promoted: {base_fact_id}")
        if row.get("relation_normalized") is not None:
            raise FreezeValidationError(f"provisional relation was normalized as formal: {base_fact_id}")
        _required_string(row.get("relation_partition_id"), f"{base_fact_id}.relation_partition_id")
        _required_string(row.get("leakage_component_id"), f"{base_fact_id}.leakage_component_id")
        fact_review = row.get("fact_review")
        completion = fact_review.get("review_completion") if isinstance(fact_review, dict) else None
        if not isinstance(fact_review, dict) or not all(
            (
                fact_review.get("proxy_evidence_only") is True,
                fact_review.get("human_gold") is False,
                isinstance(completion, dict),
                isinstance(completion, dict)
                and all(
                    completion.get(field) is True
                    for field in (
                        "codex_proxy_scope_review_complete",
                        "member_review_complete",
                        "alias_review_complete",
                        "distractor_review_complete",
                    )
                ),
            )
        ):
            raise FreezeValidationError(f"postreview fact evidence is incomplete: {base_fact_id}")
    for base_fact_id, row in exclusion_index.items():
        if row.get("disposition") != "cohort_exclude" or row.get("human_gold") is not False:
            raise FreezeValidationError(f"invalid postreview exclusion: {base_fact_id}")

    candidate_repair_contract = manifest.get("candidate_repair_contract")
    if candidate_repair_contract is None:
        candidate_stage_exclusion_count = 0
    else:
        if manifest_schema != candidate_resolution_tool.SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA:
            raise FreezeValidationError(
                "candidate-repaired postreview must use successor manifest schema v3"
            )
        try:
            candidate_resolution_tool.validate_candidate_evidence_reuse_policy(
                candidate_repair_contract,
                label="Postreview successor",
            )
        except ValueError as exc:
            raise FreezeValidationError(str(exc)) from exc
        if not isinstance(candidate_repair_contract, dict) or not all(
            (
                candidate_repair_contract.get("resolution_applied") is True,
                candidate_repair_contract.get("fixed_point_reached") is True,
                candidate_repair_contract.get("candidate_shortfall_count") == 0,
                candidate_repair_contract.get("candidate_policy_relaxed") is False,
                candidate_repair_contract.get("predecessor_zh_reviews_valid") is False,
                candidate_repair_contract.get("full_zh_translation_and_review_required")
                is True,
                candidate_repair_contract.get("human_gold") is False,
            )
        ):
            raise FreezeValidationError("postreview candidate repair contract is invalid")
        candidate_stage_exclusion_count = candidate_repair_contract.get(
            "candidate_stage_cohort_exclusion_count"
        )
        if (
            isinstance(candidate_stage_exclusion_count, bool)
            or not isinstance(candidate_stage_exclusion_count, int)
            or candidate_stage_exclusion_count < 0
        ):
            raise FreezeValidationError(
                "postreview candidate-stage exclusion count is invalid"
            )

    component_index = _unique_index(
        components, "leakage_component_id", "postreview components"
    )
    component_by_fact: Dict[str, str] = {}
    for component_id, component in component_index.items():
        members = component.get("base_fact_ids")
        if not isinstance(members, list) or component.get("member_count") != len(members):
            raise FreezeValidationError(f"invalid component membership: {component_id}")
        assignment = component.get("split_assignment")
        if assignment not in SPLITS:
            raise FreezeValidationError(f"invalid component split: {component_id}")
        for base_fact_id in members:
            if base_fact_id in component_by_fact or base_fact_id not in row_index:
                raise FreezeValidationError(f"invalid component fact coverage: {base_fact_id}")
            if row_index[base_fact_id].get("leakage_component_id") != component_id:
                raise FreezeValidationError(f"stale component assignment: {base_fact_id}")
            if row_index[base_fact_id].get("split_assignment") != assignment:
                raise FreezeValidationError(f"component split is not atomic: {base_fact_id}")
            component_by_fact[base_fact_id] = component_id
    if set(component_by_fact) != set(row_index):
        raise FreezeValidationError("components do not exactly cover retained facts")

    source_reject_exclusion_count = sum(
        row.get("source_review_outcome") == "reject" for row in exclusions
    )
    if not isinstance(split_manifest, dict) or not all(
        (
            split_manifest.get("split_status") == "provisional_not_frozen",
            split_manifest.get("formal_split_freeze_performed") is False,
            split_manifest.get("base_fact_count") == len(rows),
            split_manifest.get("fact_review_reject_excluded_count")
            == source_reject_exclusion_count,
            split_manifest.get("fact_resolution_cohort_excluded_count")
            == len(exclusions) - candidate_stage_exclusion_count,
            (
                candidate_repair_contract is None
                or split_manifest.get("candidate_repair_cohort_excluded_count")
                == candidate_stage_exclusion_count
            ),
            (
                candidate_repair_contract is None
                or split_manifest.get("total_cohort_excluded_count")
                == len(exclusions)
            ),
            split_manifest.get("leakage_component_count") == len(components),
            split_manifest.get("bounded_semantic_candidate_adjudication_complete") is True,
            split_manifest.get("semantic_paraphrase_closure_complete_for_full_pool") is False,
            split_manifest.get("semantic_near_duplicate_recall_guaranteed") is False,
        )
    ):
        raise FreezeValidationError("postreview split manifest contract is invalid")

    counts = manifest.get("counts")
    expected_counts = {
        "input_base_facts": fact_contract.get("source_universe_base_fact_count"),
        "retained_base_facts": len(rows),
        "rejected_base_facts": len(exclusions),
        "revised_base_facts": fact_contract.get(
            "revised_facts_retained_after_independent_rereview"
        ),
        "leakage_components": len(components),
        "distractor_candidates": len(distractors),
        "neutral_reference_candidates": len(neutrals),
        "candidate_shortfalls": 0,
    }
    if not isinstance(counts, dict) or any(
        counts.get(field) != value for field, value in expected_counts.items()
    ):
        raise FreezeValidationError("postreview counts are stale")
    if summary.get("counts") != counts:
        raise FreezeValidationError("postreview summary counts differ from manifest")

    candidate_resolution: Optional[Dict[str, Any]] = None
    repair_inputs = inputs.get("candidate_repair")
    if candidate_repair_contract is None:
        if repair_inputs is not None:
            raise FreezeValidationError(
                "postreview candidate repair input lacks a repair contract"
            )
    else:
        if not isinstance(repair_inputs, dict):
            raise FreezeValidationError("postreview candidate repair inputs are missing")
        resolution_manifest_path, _, resolution_binding = _verify_binding(
            repair_inputs.get("candidate_resolution_manifest"),
            owner_path=path,
            label="postreview candidate resolution manifest",
            jsonl=False,
        )
        if resolution_binding.get(
            "schema_version"
        ) not in candidate_resolution_tool.SUPPORTED_RESOLUTION_MANIFEST_SCHEMAS:
            raise FreezeValidationError(
                "postreview candidate resolution schema is unsupported"
            )
        predecessor_manifest_path, _, predecessor_binding = _verify_binding(
            repair_inputs.get("predecessor_postreview_manifest"),
            owner_path=path,
            label="postreview candidate repair predecessor",
            jsonl=False,
        )
        if predecessor_binding.get("schema_version") not in supported_schemas:
            raise FreezeValidationError(
                "postreview candidate repair predecessor schema is unsupported"
            )
        try:
            candidate_resolution = candidate_resolution_tool.load_resolution_context(
                resolution_manifest_path=resolution_manifest_path,
                predecessor_postreview_manifest_path=predecessor_manifest_path,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise FreezeValidationError(str(exc)) from exc
        if resolution_binding.get("sha256") != candidate_resolution[
            "manifest_binding"
        ].get("sha256"):
            raise FreezeValidationError(
                "postreview candidate resolution manifest binding is stale"
            )
        if predecessor_binding.get("sha256") != candidate_resolution[
            "predecessor_binding"
        ].get("sha256"):
            raise FreezeValidationError(
                "postreview candidate repair predecessor binding is stale"
            )
        root_binding = repair_inputs.get("root_postreview_manifest")
        _same_binding(
            root_binding,
            candidate_resolution["root_binding"],
            owner_path=path,
            label="postreview candidate repair root",
        )
        if candidate_repair_contract.get("resolution_id") != candidate_resolution[
            "resolution_id"
        ]:
            raise FreezeValidationError("postreview candidate resolution ID is stale")
        if candidate_stage_exclusion_count != len(
            candidate_resolution["candidate_excluded_ids"]
        ):
            raise FreezeValidationError(
                "postreview candidate-stage exclusion count is stale"
            )
    input_binding = inputs.get("full_base_facts")
    input_path, input_rows, verified_input_binding = _verify_binding(
        input_binding,
        owner_path=path,
        label="postreview input full_base_facts",
        jsonl=True,
        expected_schema=postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
    )
    input_ids = [str(row.get("base_fact_id") or "") for row in input_rows]
    if any(not value for value in input_ids) or len(input_ids) != len(set(input_ids)):
        raise FreezeValidationError("postreview input universe IDs are invalid")
    fact_mode = str(fact_contract["mode"])
    resolution: Optional[Dict[str, Any]] = None
    revised_ids: List[str] = []
    if fact_mode == "legacy_fact_review_apply":
        if inputs.get("fact_resolution") is not None:
            raise FreezeValidationError(
                "legacy postreview unexpectedly binds fact resolution"
            )
        source_ids = input_ids
        expected_retained_ids = [
            base_fact_id for base_fact_id in source_ids if base_fact_id in row_index
        ]
        expected_excluded_ids = [
            base_fact_id for base_fact_id in source_ids if base_fact_id in exclusion_index
        ]
    else:
        fact_bindings = inputs.get("fact_review")
        resolution_bindings = inputs.get("fact_resolution")
        if (
            not isinstance(fact_bindings, dict)
            or not isinstance(resolution_bindings, dict)
            or fact_bindings != resolution_bindings
        ):
            raise FreezeValidationError(
                "postreview fact-resolution bindings are missing or inconsistent"
            )
        resolution_manifest_path, _, _ = _verify_binding(
            resolution_bindings.get("fact_resolution_manifest"),
            owner_path=path,
            label="postreview fact-resolution manifest",
            jsonl=False,
            expected_schema=(
                postreview_tool.semantic_runner.materializer.RESOLUTION_MANIFEST_SCHEMA
            ),
        )
        try:
            resolution = (
                postreview_tool.semantic_runner.materializer.load_fact_resolution_context(
                    fact_resolution_manifest_path=resolution_manifest_path,
                    resolved_full_base_facts_path=input_path,
                )
            )
        except (ValueError, FileNotFoundError) as exc:
            raise FreezeValidationError(str(exc)) from exc
        source_ids = list(resolution["source_ids"])
        expected_retained_ids = list(resolution["retained_ids"])
        expected_excluded_ids = list(resolution["excluded_ids"])
        revised_ids = list(resolution["revised_ids"])
        if input_ids != expected_retained_ids:
            raise FreezeValidationError(
                "postreview resolution input differs from the complete retained cohort"
            )

    # A legacy retained row must come directly from an accepted review.  A
    # fact-resolution successor may instead retain a revised row whose source
    # proxy outcome was revise/reject/defer, but every resolution field must
    # exactly replay the validated ledger and revision-lineage artifacts.
    for base_fact_id, row in row_index.items():
        fact_review = row["fact_review"]
        if fact_mode == "legacy_fact_review_apply":
            if fact_review.get("review_outcome") != "accept":
                raise FreezeValidationError(
                    f"legacy postreview fact is not accepted: {base_fact_id}"
                )
            continue
        if resolution is None:
            raise FreezeValidationError("postreview fact resolution was not loaded")
        ledger = resolution["ledger_index"].get(base_fact_id)
        if not isinstance(ledger, dict):
            raise FreezeValidationError(
                f"postreview fact lacks a resolution ledger row: {base_fact_id}"
            )
        disposition = ledger.get("final_disposition")
        source_outcome = ledger.get("source_review_outcome")
        if disposition == "retain_accepted":
            if source_outcome != "accept":
                raise FreezeValidationError(
                    f"accepted resolution disposition has non-accept source: {base_fact_id}"
                )
            expected_revision_applied = False
            expected_lineage = None
        elif disposition == "retain_revised":
            if source_outcome not in {"revise", "reject", "defer"}:
                raise FreezeValidationError(
                    f"revised resolution disposition has invalid source: {base_fact_id}"
                )
            expected_revision_applied = True
            expected_lineage = resolution["lineage_index"].get(base_fact_id)
            if not isinstance(expected_lineage, dict):
                raise FreezeValidationError(
                    f"revised postreview fact lacks revision lineage: {base_fact_id}"
                )
        else:
            raise FreezeValidationError(
                f"retained postreview fact has non-retained resolution state: {base_fact_id}"
            )
        expected_resolution_evidence = {
            "review_outcome": source_outcome,
            "resolution_successor_universe_id": resolution["successor_id"],
            "final_disposition": disposition,
            "fact_resolution_ledger_row_sha256": sha256_value(ledger),
            "fact_revision_lineage": expected_lineage,
            "revision_applied": expected_revision_applied,
        }
        for field, expected in expected_resolution_evidence.items():
            if fact_review.get(field) != expected:
                raise FreezeValidationError(
                    f"postreview fact resolution evidence is stale: "
                    f"{base_fact_id}.{field}"
                )
    fact_excluded_ids = list(expected_excluded_ids)
    candidate_excluded_ids = (
        list(candidate_resolution["candidate_excluded_ids"])
        if candidate_resolution is not None
        else []
    )
    if set(candidate_excluded_ids).intersection(fact_excluded_ids):
        raise FreezeValidationError(
            "candidate-stage exclusions overlap fact-stage exclusions"
        )
    fact_retained_ids = list(expected_retained_ids)
    expected_retained_ids = [
        base_fact_id
        for base_fact_id in fact_retained_ids
        if base_fact_id not in set(candidate_excluded_ids)
    ]
    expected_excluded_ids = [*fact_excluded_ids, *candidate_excluded_ids]
    retained_ids = [str(row["base_fact_id"]) for row in rows]
    excluded_ids = [str(row["base_fact_id"]) for row in exclusions]
    if retained_ids != expected_retained_ids or excluded_ids != expected_excluded_ids:
        raise FreezeValidationError(
            "postreview retained/excluded order differs from fact-review lineage"
        )
    if set(source_ids) != set(row_index) | set(exclusion_index):
        raise FreezeValidationError("postreview dispositions do not cover input universe")
    if not all(
        (
            fact_contract.get("source_universe_base_fact_count") == len(source_ids),
            fact_contract.get("accepted_facts_retained") == len(rows),
            fact_contract.get("revised_facts_retained_after_independent_rereview")
            == len(revised_ids),
            fact_contract.get("facts_cohort_excluded") == len(exclusions),
            (
                candidate_resolution is None
                or fact_contract.get("fact_stage_retained_base_fact_count")
                == len(fact_retained_ids)
            ),
            (
                candidate_resolution is None
                or fact_contract.get("fact_stage_cohort_excluded")
                == len(exclusions) - len(candidate_excluded_ids)
            ),
            (
                candidate_resolution is None
                or fact_contract.get("candidate_stage_cohort_excluded")
                == len(candidate_excluded_ids)
            ),
        )
    ):
        raise FreezeValidationError("postreview fact-review lineage counts are stale")
    digests = manifest.get("lineage_digests")
    if not isinstance(digests, dict) or not all(
        (
            digests.get("ordered_input_base_fact_ids_sha256") == sha256_value(source_ids),
            digests.get("ordered_retained_base_fact_ids_sha256")
            == sha256_value(retained_ids),
            digests.get("ordered_rejected_base_fact_ids_sha256")
            == sha256_value(excluded_ids),
            digests.get("ordered_revised_base_fact_ids_sha256")
            == sha256_value(revised_ids),
            (
                candidate_resolution is None
                or digests.get("ordered_fact_stage_retained_base_fact_ids_sha256")
                == sha256_value(fact_retained_ids)
            ),
            (
                candidate_resolution is None
                or digests.get("ordered_fact_stage_excluded_base_fact_ids_sha256")
                == sha256_value(fact_excluded_ids)
            ),
            (
                candidate_resolution is None
                or digests.get("ordered_candidate_stage_excluded_base_fact_ids_sha256")
                == sha256_value(candidate_excluded_ids)
            ),
            digests.get("ordered_component_ids_sha256")
            == sha256_value([row["leakage_component_id"] for row in components]),
            digests.get("ordered_distractor_ids_sha256")
            == sha256_value([row["distractor_id"] for row in distractors]),
            digests.get("ordered_neutral_candidate_ids_sha256")
            == sha256_value([row["neutral_candidate_id"] for row in neutrals]),
        )
    ):
        raise FreezeValidationError("postreview lineage digests are stale")

    _deterministically_replay_postreview(
        manifest_path=path,
        manifest=manifest,
        input_rows=input_rows,
        artifacts=artifacts,
        split_seed=_required_string(split_manifest.get("split_seed"), "split_seed"),
    )

    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(path, schema_version=str(manifest_schema)),
        "artifacts": artifacts,
        "rows": rows,
        "row_index": row_index,
        "exclusions": exclusions,
        "exclusion_index": exclusion_index,
        "components": components,
        "component_by_fact": component_by_fact,
        "distractors": distractors,
        "neutrals": neutrals,
        "split_manifest": split_manifest,
        "input_path": input_path,
        "input_rows": input_rows,
        "input_ids": source_ids,
        "resolved_input_ids": input_ids,
        "input_binding": verified_input_binding,
        "candidate_resolution": candidate_resolution,
        "semantic_runtime": semantic_runtime,
    }


def _deterministically_replay_postreview(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
    split_seed: str,
) -> None:
    """Replay the complete upstream materializer and compare every emitted byte."""

    inputs = manifest.get("inputs")
    fact = inputs.get("fact_review") if isinstance(inputs, dict) else None
    semantic = inputs.get("semantic_review") if isinstance(inputs, dict) else None
    if not isinstance(fact, dict) or not isinstance(semantic, dict):
        raise FreezeValidationError("postreview replay inputs are missing")
    candidate_repair = inputs.get("candidate_repair")
    if candidate_repair is not None:
        if not isinstance(candidate_repair, dict):
            raise FreezeValidationError(
                "postreview candidate repair replay input is invalid"
            )
        resolution_path, _, _ = _verify_binding(
            candidate_repair.get("candidate_resolution_manifest"),
            owner_path=manifest_path,
            label="postreview replay candidate resolution",
            jsonl=False,
        )
        predecessor_path, _, _ = _verify_binding(
            candidate_repair.get("predecessor_postreview_manifest"),
            owner_path=manifest_path,
            label="postreview replay candidate predecessor",
            jsonl=False,
        )
        with tempfile.TemporaryDirectory(
            prefix="full-postreview-candidate-repair-replay-"
        ) as directory:
            replay_dir = Path(directory) / "replayed"
            try:
                candidate_resolution_tool.materialize_successor(
                    predecessor_postreview_manifest_path=predecessor_path,
                    candidate_resolution_manifest_path=resolution_path,
                    output_dir=replay_dir,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise FreezeValidationError(
                    f"postreview candidate-repair deterministic replay failed: {exc}"
                ) from exc
            for label, artifact in artifacts.items():
                replay_path = replay_dir / Path(str(artifact["path"])).name
                if not replay_path.is_file() or sha256_file(replay_path) != artifact[
                    "binding"
                ].get("sha256"):
                    raise FreezeValidationError(
                        "postreview candidate-repair deterministic replay mismatch: "
                        f"{label}"
                    )
            replay_manifest = replay_dir / "postreview_rebuild_manifest.json"
            if sha256_file(replay_manifest) != sha256_file(manifest_path):
                raise FreezeValidationError(
                    "postreview candidate-repair manifest replay mismatch"
                )
        return
    fact_contract = manifest.get("fact_review_contract")
    if not isinstance(fact_contract, dict):
        raise FreezeValidationError("postreview fact-review contract is missing")
    fact_mode = fact_contract.get("mode")
    source_record_count = fact_contract.get("source_universe_base_fact_count")
    if (
        isinstance(source_record_count, bool)
        or not isinstance(source_record_count, int)
        or source_record_count < 1
    ):
        raise FreezeValidationError(
            "postreview source-universe record count is invalid"
        )

    apply_path: Optional[Path] = None
    resolution_manifest_path: Optional[Path] = None
    if fact_mode == "legacy_fact_review_apply":
        if inputs.get("fact_resolution") is not None:
            raise FreezeValidationError(
                "legacy postreview replay unexpectedly binds fact resolution"
            )
        apply_path, _, _ = _verify_binding(
            fact.get("apply_manifest"),
            owner_path=manifest_path,
            label="postreview fact-review apply manifest",
            jsonl=False,
            expected_schema=postreview_tool.review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
        )
        if source_record_count != len(input_rows):
            raise FreezeValidationError(
                "legacy postreview source count differs from full-base input"
            )
    elif fact_mode == "fact_resolution_successor_cohort":
        fact_resolution = inputs.get("fact_resolution")
        if not isinstance(fact_resolution, dict) or fact_resolution != fact:
            raise FreezeValidationError(
                "postreview fact-resolution replay binding is missing or inconsistent"
            )
        resolution_manifest_path, _, _ = _verify_binding(
            fact_resolution.get("fact_resolution_manifest"),
            owner_path=manifest_path,
            label="postreview fact-resolution manifest",
            jsonl=False,
            expected_schema=(
                postreview_tool.semantic_runner.materializer.RESOLUTION_MANIFEST_SCHEMA
            ),
        )
        full_base_path = _resolve_binding_path(
            inputs.get("full_base_facts"),
            manifest_path,
            "postreview full_base_facts",
        )
        _verify_binding(
            fact_resolution.get("revised_full_base_facts"),
            owner_path=manifest_path,
            label="postreview fact-resolution revised full-base facts",
            jsonl=True,
            expected_schema=postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
            explicit_path=full_base_path,
        )
        source_binding = fact_resolution.get("source_full_base_facts")
        if not isinstance(source_binding, dict) or source_binding.get(
            "record_count"
        ) != source_record_count:
            raise FreezeValidationError(
                "postreview fact-resolution source count binding is stale"
            )
    else:
        raise FreezeValidationError(
            f"unsupported postreview fact-review mode: {fact_mode!r}"
        )
    authority_mode = semantic.get("authority_mode", "direct_semantic_review_run")
    semantic_run_manifest_path: Optional[Path] = None
    semantic_composite_manifest_path: Optional[Path] = None
    if authority_mode == "direct_semantic_review_run":
        if semantic.get("composite_manifest") is not None:
            raise FreezeValidationError(
                "postreview replay direct semantic authority violates XOR contract"
            )
        semantic_run_manifest_path, _, _ = _verify_binding(
            semantic.get("run_manifest"),
            owner_path=manifest_path,
            label="postreview semantic run manifest",
            jsonl=False,
            expected_schema=postreview_tool.semantic_runner.RUN_MANIFEST_SCHEMA,
        )
    elif authority_mode == "composite_semantic_review_authority":
        if semantic.get("run_manifest") is not None:
            raise FreezeValidationError(
                "postreview replay composite semantic authority violates XOR contract"
            )
        composite_tool = postreview_tool._load_semantic_composite_builder()
        semantic_composite_manifest_path, _, _ = _verify_binding(
            semantic.get("composite_manifest"),
            owner_path=manifest_path,
            label="postreview semantic composite manifest",
            jsonl=False,
            expected_schema=composite_tool.MANIFEST_SCHEMA,
        )
    else:
        raise FreezeValidationError("postreview replay semantic authority mode is invalid")
    semantic_manifest_path, _, _ = _verify_binding(
        semantic.get("candidate_manifest"),
        owner_path=manifest_path,
        label="postreview semantic candidate manifest",
        jsonl=False,
        expected_schema=postreview_tool.semantic_runner.materializer.MANIFEST_SCHEMA,
    )
    semantic_adjudications_path, _, _ = _verify_binding(
        semantic.get("adjudications"),
        owner_path=manifest_path,
        label="postreview semantic adjudications",
        jsonl=True,
        expected_schema=postreview_tool.semantic_runner.ADJUDICATION_SCHEMA,
        allow_empty=True,
    )
    with tempfile.TemporaryDirectory(prefix="full-postreview-freeze-replay-") as directory:
        replay_dir = Path(directory) / "replayed"
        try:
            postreview_tool.materialize(
                full_base_facts_path=Path(
                    _resolve_binding_path(
                        manifest["inputs"]["full_base_facts"],
                        manifest_path,
                        "postreview full_base_facts",
                    )
                ),
                fact_review_apply_manifest_path=apply_path,
                fact_resolution_manifest_path=resolution_manifest_path,
                semantic_candidate_manifest_path=semantic_manifest_path,
                semantic_adjudications_path=semantic_adjudications_path,
                semantic_run_manifest_path=semantic_run_manifest_path,
                semantic_composite_manifest_path=semantic_composite_manifest_path,
                output_dir=replay_dir,
                expected_input_count=source_record_count,
                split_seed=split_seed,
            )
        except (ValueError, FileNotFoundError) as exc:
            raise FreezeValidationError(f"postreview deterministic replay failed: {exc}") from exc
        for label, artifact in artifacts.items():
            replay_path = replay_dir / Path(str(artifact["path"])).name
            if not replay_path.is_file() or sha256_file(replay_path) != artifact["binding"].get(
                "sha256"
            ):
                raise FreezeValidationError(
                    f"postreview deterministic replay mismatch: {label}"
                )
        replay_manifest = replay_dir / "postreview_rebuild_manifest.json"
        if sha256_file(replay_manifest) != sha256_file(manifest_path):
            raise FreezeValidationError("postreview manifest deterministic replay mismatch")


def _validate_candidate_adjudication(
    row: Mapping[str, Any],
    unit: Any,
    protocol_identity: Mapping[str, str],
    *,
    require_accept: bool = True,
) -> Dict[str, Any]:
    expected_fields = {
        "schema_version",
        "candidate_id",
        "candidate_kind",
        "candidate_row_sha256",
        "target_base_fact_id",
        "target_base_fact_row_sha256",
        "source_base_fact_id",
        "source_base_fact_row_sha256",
        "overall_decision",
        *candidate_tool.FIELD_NAMES,
        "rationale",
        "confidence",
        "reviewer_type",
        "reviewer_id",
        "requested_model",
        "expected_response_model",
        "response_model",
        "response_model_identity_status",
        "review_method",
        "reviewed_at",
        "behavior_blind",
        "target_behavior_consumed",
        "human_gold",
    }
    candidate_id = str(unit.candidate_id)
    if set(row) != expected_fields:
        raise FreezeValidationError(
            f"candidate adjudication fields are invalid: {candidate_id}"
        )
    expected = {
        "schema_version": candidate_tool.ADJUDICATION_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_kind": unit.candidate_kind,
        "candidate_row_sha256": unit.candidate_row_sha256,
        "target_base_fact_id": unit.target_base_fact_id,
        "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
        "source_base_fact_id": unit.source_base_fact_id,
        "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise FreezeValidationError(
                f"candidate adjudication {field} is stale: {candidate_id}"
            )
    if row.get("reviewer_type") != "independent_model_proxy":
        raise FreezeValidationError(f"candidate reviewer type is invalid: {candidate_id}")
    if row.get("review_method") != candidate_tool.REVIEW_METHOD:
        raise FreezeValidationError(f"candidate review method is invalid: {candidate_id}")
    expected_identity = {
        "reviewer_id": (
            f"{protocol_identity['provider_profile']}:"
            f"{protocol_identity['expected_response_model']}"
        ),
        "requested_model": protocol_identity["requested_model"],
        "expected_response_model": protocol_identity["expected_response_model"],
        "response_model": protocol_identity["expected_response_model"],
        "response_model_identity_status": "matched",
    }
    for field, value in expected_identity.items():
        if row.get(field) != value:
            raise FreezeValidationError(
                f"candidate adjudication model identity is invalid: {candidate_id}.{field}"
            )
    _required_string(row.get("reviewed_at"), f"{candidate_id}.reviewed_at")
    _required_string(row.get("rationale"), f"{candidate_id}.rationale")
    if row.get("behavior_blind") is not True or row.get(
        "target_behavior_consumed"
    ) is not False:
        raise FreezeValidationError(f"candidate review is not behavior-blind: {candidate_id}")
    if row.get("human_gold") is not False:
        raise FreezeValidationError(f"candidate review claims human gold: {candidate_id}")
    verdict = {
        "candidate_id": candidate_id,
        "candidate_kind": unit.candidate_kind,
        "overall_decision": row.get("overall_decision"),
        **{field: row.get(field) for field in candidate_tool.FIELD_NAMES},
        "rationale": row.get("rationale"),
        "confidence": row.get("confidence"),
    }
    try:
        candidate_tool.validate_batch_response(
            {
                "schema_version": candidate_tool.BATCH_RESPONSE_SCHEMA,
                "records": [verdict],
            },
            [unit],
        )
    except ValueError as exc:
        raise FreezeValidationError(str(exc)) from exc
    if require_accept and row.get("overall_decision") != "accept":
        raise FreezeValidationError(
            f"candidate is not accepted; repair/regenerate and re-review: {candidate_id}"
        )
    return dict(row)


def _validate_candidate_review(
    *,
    manifest_path: Path,
    adjudications_path: Path,
    postreview: Mapping[str, Any],
    config: Mapping[str, Any],
    require_all_accept: bool = True,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    adjudications_path = Path(adjudications_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != candidate_tool.RUN_MANIFEST_SCHEMA:
        raise FreezeValidationError("candidate review manifest schema_version is unsupported")
    if manifest.get("tool_version") != candidate_tool.TOOL_VERSION:
        raise FreezeValidationError("candidate review manifest tool_version is unsupported")
    if manifest.get("status") != "completed":
        raise FreezeValidationError("candidate review contains failed terminal rows")
    if manifest.get("candidate_review_complete_for_selected_scope") is not True or manifest.get(
        "candidate_review_complete_for_full_set"
    ) is not True:
        raise FreezeValidationError("candidate review does not cover the full candidate set")
    if manifest.get("selection_is_full_candidate_set") is not True:
        raise FreezeValidationError("candidate review is a limited selection")
    if manifest.get("formal_review_complete") is not False:
        raise FreezeValidationError("candidate runner improperly claimed formal completion")

    try:
        _, manifest_kind, input_bindings, units = candidate_tool.load_review_units(
            candidate_manifest_path=Path(postreview["manifest_path"]),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise FreezeValidationError(str(exc)) from exc
    if manifest_kind != "postreview_rebuild":
        raise FreezeValidationError("candidate review must bind the postreview rebuild")
    unit_ids = [str(unit.candidate_id) for unit in units]
    if not units or len(unit_ids) != len(set(unit_ids)):
        raise FreezeValidationError("candidate review units are empty or duplicated")
    expected_candidate_count = len(postreview["distractors"]) + len(postreview["neutrals"])
    if len(units) != expected_candidate_count:
        raise FreezeValidationError("candidate review unit count is stale")

    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise FreezeValidationError("candidate review run_contract is missing")
    if manifest.get("run_contract_sha256") != sha256_value(run_contract):
        raise FreezeValidationError("candidate review run_contract SHA is stale")
    if run_contract.get("tool_version") != candidate_tool.TOOL_VERSION:
        raise FreezeValidationError("candidate review run_contract tool_version is unsupported")
    if run_contract.get("config_sha256") != sha256_value(config):
        raise FreezeValidationError("candidate review config SHA is stale")
    if run_contract.get("source_fingerprints") != candidate_tool.source_fingerprints():
        raise FreezeValidationError("candidate review source fingerprints are stale")
    try:
        protocol_identity = candidate_tool.protocol_reviewer_identity(config)
    except ValueError as exc:
        raise FreezeValidationError(str(exc)) from exc
    route_identity = run_contract.get("route_identity")
    if not isinstance(route_identity, dict):
        raise FreezeValidationError("candidate review route identity is missing")
    try:
        validate_route_identity_set(
            route_identity,
            expected_routes=[
                (
                    protocol_identity["provider_profile"],
                    protocol_identity["requested_model"],
                    protocol_identity["expected_response_model"],
                )
            ],
            expected_protocols=_configured_route_protocols(config),
        )
    except ValueError as exc:
        raise FreezeValidationError(f"candidate review route identity is invalid: {exc}") from exc
    if run_contract.get("protocol_reviewer_identity") != protocol_identity:
        raise FreezeValidationError("candidate review protocol reviewer identity is stale")
    reviewer_spec = run_contract.get("reviewer_spec")
    if not isinstance(reviewer_spec, dict) or any(
        (
            reviewer_spec.get("provider_profile")
            != protocol_identity["provider_profile"],
            reviewer_spec.get("model") != protocol_identity["requested_model"],
            run_contract.get("expected_response_model")
            != protocol_identity["expected_response_model"],
        )
    ):
        raise FreezeValidationError(
            "candidate review model route or response identity is unsupported"
        )
    expected_semantics_contract = candidate_tool.review_semantics_contract(
        reviewer_spec=reviewer_spec,
        expected_response_model=protocol_identity["expected_response_model"],
        protocol_identity=protocol_identity,
        route_identity=route_identity,
    )
    expected_semantics_sha256 = sha256_value(expected_semantics_contract)
    if run_contract.get(
        "review_semantics_contract"
    ) != expected_semantics_contract or run_contract.get(
        "review_semantics_contract_sha256"
    ) != expected_semantics_sha256:
        raise FreezeValidationError("candidate review semantics contract is stale")
    candidate_manifest_binding = run_contract.get("candidate_manifest")
    if not isinstance(candidate_manifest_binding, dict):
        raise FreezeValidationError("candidate run does not bind the source manifest")
    _same_binding(
        candidate_manifest_binding,
        postreview["manifest_binding"],
        owner_path=manifest_path,
        label="candidate review source manifest",
    )
    if candidate_manifest_binding.get("manifest_kind") != "postreview_rebuild":
        raise FreezeValidationError("candidate review source kind is invalid")
    run_inputs = run_contract.get("input_artifacts")
    if not isinstance(run_inputs, dict):
        raise FreezeValidationError("candidate review input_artifacts are missing")
    for label, expected in input_bindings.items():
        _same_binding(
            run_inputs.get(label), expected, owner_path=manifest_path, label=f"candidate {label}"
        )
    if not all(
        (
            run_contract.get("selected_candidate_count") == len(units),
            run_contract.get("full_candidate_count") == len(units),
            run_contract.get("selection_is_full_candidate_set") is True,
            run_contract.get("selected_candidate_ids_sha256") == sha256_value(unit_ids),
            manifest.get("selected_candidate_count") == len(units),
            manifest.get("full_candidate_count") == len(units),
            manifest.get("selected_candidate_ids_sha256") == sha256_value(unit_ids),
        )
    ):
        raise FreezeValidationError("candidate review selected ID/count contract is stale")
    expected_by_kind: Dict[str, Dict[str, Any]] = {}
    for kind in sorted(candidate_tool.CANDIDATE_KINDS):
        ids = [unit.candidate_id for unit in units if unit.candidate_kind == kind]
        expected_by_kind[kind] = {
            "record_count": len(ids),
            "ordered_ids_sha256": sha256_value(ids),
        }
    if run_contract.get("selected_candidate_ids_by_kind") != expected_by_kind or manifest.get(
        "selected_candidate_ids_by_kind"
    ) != expected_by_kind:
        raise FreezeValidationError("candidate review kind coverage is stale")

    try:
        evidence = candidate_tool.load_review_evidence_manifest(manifest_path)
    except (ValueError, FileNotFoundError) as exc:
        raise FreezeValidationError(
            f"candidate review evidence lineage is invalid: {exc}"
        ) from exc
    if evidence["candidate_manifest_path"] != Path(postreview["manifest_path"]).resolve():
        raise FreezeValidationError(
            "candidate review evidence binds a different postreview rebuild"
        )

    boundary = manifest.get("evidence_boundary")
    _zero_execution_contract(boundary, "candidate evidence_boundary")
    if not isinstance(boundary, dict) or not all(
        (
            boundary.get("human_gold") is False,
            boundary.get("behavior_blind") is True,
            boundary.get("target_behavior_consumed") is False,
            boundary.get("review_freeze_performed") is False,
            boundary.get("split_freeze_performed") is False,
            boundary.get("candidate_promotion_performed") is False,
        )
    ):
        raise FreezeValidationError("candidate review evidence boundary is invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FreezeValidationError("candidate review artifacts are missing")
    _, adjudications, adjudication_binding = _verify_binding(
        artifacts.get("adjudications"),
        owner_path=manifest_path,
        label="candidate adjudications",
        jsonl=True,
        expected_schema=candidate_tool.ADJUDICATION_SCHEMA,
        explicit_path=adjudications_path,
    )
    _, checkpoints, checkpoint_binding = _verify_binding(
        artifacts.get("checkpoint"),
        owner_path=manifest_path,
        label="candidate checkpoint",
        jsonl=True,
        expected_schema=candidate_tool.CHECKPOINT_SCHEMA,
    )
    if len(adjudications) != len(units) or len(checkpoints) != len(units):
        raise FreezeValidationError("candidate review artifacts do not fully cover candidates")
    adjudication_index = _unique_index(
        adjudications, "candidate_id", "candidate adjudications"
    )
    checkpoint_index = _unique_index(checkpoints, "candidate_id", "candidate checkpoints")
    if set(adjudication_index) != set(unit_ids) or set(checkpoint_index) != set(unit_ids):
        raise FreezeValidationError("candidate review artifact ID coverage is incomplete")
    units_by_id = {unit.candidate_id: unit for unit in units}
    validated: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        candidate_id = unit.candidate_id
        try:
            _, checkpoint = candidate_tool._validate_checkpoint_row(
                checkpoint_index[candidate_id],
                units_by_id=units_by_id,
                run_contract_sha256=str(manifest["run_contract_sha256"]),
                review_semantics_contract_sha256=expected_semantics_sha256,
            )
        except ValueError as exc:
            raise FreezeValidationError(str(exc)) from exc
        if checkpoint.get("terminal_status") != "completed":
            raise FreezeValidationError(f"candidate checkpoint failed: {candidate_id}")
        if checkpoint.get("strict_adjudication") != adjudication_index[candidate_id]:
            raise FreezeValidationError(f"candidate checkpoint/adjudication mismatch: {candidate_id}")
        validated[candidate_id] = _validate_candidate_adjudication(
            adjudication_index[candidate_id],
            unit,
            protocol_identity,
            require_accept=require_all_accept,
        )
    expected_status_counts = {"completed": len(units)}
    if manifest.get("terminal_status_counts") != expected_status_counts:
        raise FreezeValidationError("candidate terminal_status_counts are stale")
    expected_decision_counts = dict(
        sorted(Counter(row["overall_decision"] for row in validated.values()).items())
    )
    if manifest.get("overall_decision_counts") != expected_decision_counts:
        raise FreezeValidationError("candidate decision counts are stale")
    if require_all_accept and expected_decision_counts != {"accept": len(units)}:
        raise FreezeValidationError("candidate decision counts include non-accept outcomes")
    expected_kind_counts = dict(
        sorted(Counter(unit.candidate_kind for unit in units).items())
    )
    expected_accepted_kind_counts = {
        kind: sum(
            1
            for unit in units
            if unit.candidate_kind == kind
            and validated[unit.candidate_id]["overall_decision"] == "accept"
        )
        for kind in sorted(candidate_tool.CANDIDATE_KINDS)
    }
    if manifest.get("completed_kind_counts") != expected_kind_counts:
        raise FreezeValidationError("candidate completed kind counts are stale")
    if manifest.get("accepted_kind_counts") != expected_accepted_kind_counts:
        raise FreezeValidationError("candidate accepted kind counts are stale")

    by_fact: MutableMapping[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"distractor": [], "neutral": []}
    )
    for unit in units:
        if validated[unit.candidate_id]["overall_decision"] == "accept":
            by_fact[unit.target_base_fact_id][unit.candidate_kind].append(
                validated[unit.candidate_id]
            )
    if require_all_accept:
        for base_fact_id in postreview["row_index"]:
            if len(by_fact[base_fact_id]["distractor"]) != 2 or len(
                by_fact[base_fact_id]["neutral"]
            ) != 2:
                raise FreezeValidationError(
                    "candidate review does not provide exactly two accepted "
                    f"candidates per kind: {base_fact_id}"
                )
            for kind in ("distractor", "neutral"):
                source_ids = [
                    str(row["source_base_fact_id"])
                    for row in by_fact[base_fact_id][kind]
                ]
                if len(source_ids) != len(set(source_ids)):
                    raise FreezeValidationError(
                        f"accepted {kind} candidates repeat one source fact: {base_fact_id}"
                    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_binding": _binding(
            manifest_path, schema_version=candidate_tool.RUN_MANIFEST_SCHEMA
        ),
        "adjudications": adjudications,
        "adjudication_index": validated,
        "adjudications_binding": adjudication_binding,
        "checkpoint_binding": checkpoint_binding,
        "review_semantics_contract_sha256": expected_semantics_sha256,
        "evidence_origin_counts": dict(manifest["evidence_origin_counts"]),
        "fresh_model_review_candidate_count": manifest[
            "fresh_model_review_candidate_count"
        ],
        "carried_forward_candidate_count": manifest[
            "carried_forward_candidate_count"
        ],
        "full_evidence_partition_complete": manifest[
            "full_evidence_partition_complete"
        ],
        "evidence_bundle": evidence,
        "units": units,
        "by_fact": dict(by_fact),
        "overall_decision_counts": expected_decision_counts,
        "accepted_candidate_count": sum(expected_accepted_kind_counts.values()),
        "all_candidates_terminal": True,
        "all_candidates_accepted": expected_decision_counts == {"accept": len(units)},
    }


def _candidate_evidence_lineage(
    evidence_bundle: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    lineage: List[Dict[str, Any]] = []
    current: Optional[Mapping[str, Any]] = evidence_bundle
    while current is not None:
        manifest_path = Path(current["manifest_path"]).resolve()
        manifest = current["manifest"]
        lineage.append(
            {
                "review_manifest": _binding(
                    manifest_path,
                    schema_version=candidate_tool.RUN_MANIFEST_SCHEMA,
                ),
                "checkpoint": copy.deepcopy(current["checkpoint_binding"]),
                "adjudications": copy.deepcopy(current["adjudication_binding"]),
                "run_contract_sha256": current["run_contract_sha256"],
                "review_semantics_contract_sha256": current[
                    "review_semantics_contract_sha256"
                ],
                "evidence_origin_counts": copy.deepcopy(
                    manifest["evidence_origin_counts"]
                ),
                "fresh_model_review_candidate_count": manifest[
                    "fresh_model_review_candidate_count"
                ],
                "carried_forward_candidate_count": manifest[
                    "carried_forward_candidate_count"
                ],
                "full_evidence_partition_complete": manifest[
                    "full_evidence_partition_complete"
                ],
            }
        )
        source = current.get("source_bundle")
        current = source if isinstance(source, Mapping) else None
    return lineage


def _validate_candidate_projection(
    *,
    manifest_path: Path,
    postreview: Mapping[str, Any],
    candidate_review: Mapping[str, Any],
    adjudications_path: Path,
) -> Dict[str, Any]:
    """Validate and bind an accepted-only 1+1 projection of full review support."""

    manifest_path = Path(manifest_path).resolve()
    try:
        projection = accepted_projection_tool.load_projection_context(
            manifest_path,
            expected_postreview_manifest_path=Path(postreview["manifest_path"]),
            expected_review_manifest_path=Path(candidate_review["manifest_path"]),
            expected_adjudications_path=Path(adjudications_path),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise FreezeValidationError(f"accepted candidate projection is invalid: {exc}") from exc

    manifest = projection["manifest"]
    selection = manifest.get("selection_contract")
    source_review = manifest.get("source_review_contract")
    safety = manifest.get("safety_contract")
    if not isinstance(selection, dict) or not all(
        (
            selection.get("cardinality_per_target")
            == {"distractor": 1, "neutral": 1},
            selection.get("accepted_candidates_only") is True,
            selection.get("source_review_full_set_complete") is True,
            selection.get("source_review_nonaccept_allowed") is True,
            selection.get("selected_projection_all_accept") is True,
            selection.get("donor_must_be_in_projection_target_cohort") is True,
            selection.get("candidate_rows_rewritten") is False,
            selection.get("split_recomputed") is False,
            selection.get("components_recomputed") is False,
            selection.get("human_gold") is False,
        )
    ):
        raise FreezeValidationError("accepted candidate projection selection contract is invalid")
    if not isinstance(source_review, dict) or not all(
        (
            source_review.get("review_status") == "completed",
            source_review.get("selection_is_full_candidate_set") is True,
            source_review.get("candidate_review_complete_for_full_set") is True,
            source_review.get("source_review_nonaccept_allowed") is True,
            source_review.get("selected_projection_all_accept") is True,
            source_review.get("behavior_blind") is True,
            source_review.get("human_gold") is False,
        )
    ):
        raise FreezeValidationError("accepted candidate projection source review is invalid")
    if not isinstance(safety, dict) or not all(
        (
            safety.get("output_is_provisional") is True,
            safety.get("canonical_freeze_emitted") is False,
            safety.get("review_freeze_emitted") is False,
            safety.get("split_freeze_emitted") is False,
            safety.get("candidate_rows_formally_promoted") is False,
            safety.get("hf_checkpoint_bound") is False,
            safety.get("hf_tokenizer_bound") is False,
            safety.get("hf_model_executed") is False,
            safety.get("hf_tokenizer_executed") is False,
            safety.get("behavior_executed") is False,
            safety.get("validation_exposed") is False,
            safety.get("sealed_exposed") is False,
            safety.get("human_gold") is False,
        )
    ):
        raise FreezeValidationError("accepted candidate projection safety contract is invalid")

    rows = projection["full_base_facts"]
    target_ids = [str(value) for value in projection["projected_target_ids"]]
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise FreezeValidationError("accepted candidate projection target IDs are invalid")
    if [str(row.get("base_fact_id") or "") for row in rows] != target_ids:
        raise FreezeValidationError("accepted candidate projection fact order is stale")
    if any(
        postreview["row_index"].get(base_fact_id) != row
        for base_fact_id, row in zip(target_ids, rows)
    ):
        raise FreezeValidationError(
            "accepted candidate projection facts differ from full postreview support"
        )
    quarantine = projection["quarantine"]
    quarantine_ids = [str(row.get("base_fact_id") or "") for row in quarantine]
    if (
        any(not value for value in quarantine_ids)
        or len(quarantine_ids) != len(set(quarantine_ids))
        or set(target_ids).intersection(quarantine_ids)
        or set(target_ids) | set(quarantine_ids) != set(postreview["row_index"])
    ):
        raise FreezeValidationError(
            "accepted candidate projection does not partition retained postreview targets"
        )

    distractors = projection["distractor_candidates"]
    neutrals = projection["neutral_reference_candidates"]
    candidate_rows_by_id: Dict[str, Mapping[str, Any]] = {}
    for kind, values, id_field in (
        ("distractor", distractors, "distractor_id"),
        ("neutral", neutrals, "neutral_candidate_id"),
    ):
        grouped = _candidate_rows_by_fact(values, id_field=id_field)
        if set(grouped) != set(target_ids):
            raise FreezeValidationError(
                f"accepted candidate projection {kind} target coverage is incomplete"
            )
        for base_fact_id in target_ids:
            if len(grouped[base_fact_id]) != 1:
                raise FreezeValidationError(
                    "accepted candidate projection must provide exactly one "
                    f"{kind} per target: {base_fact_id}"
                )
            candidate_id = str(grouped[base_fact_id][0][id_field])
            if candidate_id in candidate_rows_by_id:
                raise FreezeValidationError(
                    f"accepted candidate projection repeats candidate ID: {candidate_id}"
                )
            candidate_rows_by_id[candidate_id] = grouped[base_fact_id][0]

    selected_ids = [str(value) for value in projection["selected_candidate_ids"]]
    if len(selected_ids) != 2 * len(target_ids) or set(selected_ids) != set(
        candidate_rows_by_id
    ):
        raise FreezeValidationError("accepted candidate projection selected IDs are stale")
    selected_adjudications = projection["selected_adjudications"]
    if len(selected_adjudications) != len(selected_ids):
        raise FreezeValidationError(
            "accepted candidate projection adjudication coverage is incomplete"
        )
    selected_review_by_id: Dict[str, Dict[str, Any]] = {}
    unit_by_id = {unit.candidate_id: unit for unit in candidate_review["units"]}
    by_fact: MutableMapping[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"distractor": [], "neutral": []}
    )
    for candidate_id, adjudication in zip(selected_ids, selected_adjudications):
        if str(adjudication.get("candidate_id") or "") != candidate_id:
            raise FreezeValidationError(
                "accepted candidate projection adjudication order is stale"
            )
        validated = candidate_review["adjudication_index"].get(candidate_id)
        if validated != adjudication or adjudication.get("overall_decision") != "accept":
            raise FreezeValidationError(
                f"projected candidate lacks accepted full-review evidence: {candidate_id}"
            )
        unit = unit_by_id.get(candidate_id)
        candidate_row = candidate_rows_by_id[candidate_id]
        if unit is None or any(
            (
                unit.target_base_fact_id != candidate_row.get("base_fact_id"),
                unit.source_base_fact_id != candidate_row.get("source_base_fact_id"),
                unit.candidate_row_sha256 != sha256_value(candidate_row),
            )
        ):
            raise FreezeValidationError(
                f"projected candidate row/full-review binding is stale: {candidate_id}"
            )
        selected_review_by_id[candidate_id] = dict(adjudication)
        by_fact[unit.target_base_fact_id][unit.candidate_kind].append(
            dict(adjudication)
        )
    for base_fact_id in target_ids:
        if any(len(by_fact[base_fact_id][kind]) != 1 for kind in ("distractor", "neutral")):
            raise FreezeValidationError(
                f"accepted projection review evidence is not 1+1: {base_fact_id}"
            )

    artifacts: Dict[str, Dict[str, Any]] = {}
    for label, value in (
        ("full_base_facts", rows),
        ("distractor_candidates", distractors),
        ("neutral_reference_candidates", neutrals),
        ("leakage_components", projection["leakage_components"]),
        ("split_manifest", projection["split_manifest"]),
        ("quarantine", quarantine),
    ):
        bound = dict(projection["output_bindings"][label])
        bound["path"] = str(projection["output_paths"][label])
        artifacts[label] = {
            "path": projection["output_paths"][label],
            "value": value,
            "binding": bound,
        }

    return {
        **projection,
        "manifest_binding": _binding(
            manifest_path,
            schema_version=accepted_projection_tool.MANIFEST_SCHEMA_VERSION,
        ),
        "rows": rows,
        "row_index": _unique_index(rows, "base_fact_id", "projected facts"),
        "distractors": distractors,
        "neutrals": neutrals,
        "components": projection["leakage_components"],
        "target_ids": target_ids,
        "quarantine_ids": quarantine_ids,
        "selected_review_by_id": selected_review_by_id,
        "by_fact": dict(by_fact),
        "artifacts": artifacts,
        "candidate_cardinality_per_target": {"distractor": 1, "neutral": 1},
    }


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(row.get("split_assignment") or "") for row in rows)
    if set(counts) - set(SPLITS) or "" in counts:
        raise FreezeValidationError("invalid split assignment in retained cohort")
    return {split: counts.get(split, 0) for split in SPLITS}


def _candidate_rows_by_fact(
    rows: Sequence[Mapping[str, Any]], *, id_field: str
) -> Dict[str, List[Dict[str, Any]]]:
    result: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["base_fact_id"])].append(dict(row))
    for values in result.values():
        values.sort(key=lambda row: (int(row["slot"]), str(row[id_field])))
    return dict(result)


def _zh_item(
    *,
    base_fact_id: str,
    cohort_item_id: str,
    fact: Mapping[str, Any],
    distractors: Sequence[Mapping[str, Any]],
    neutrals: Sequence[Mapping[str, Any]],
    expected_candidates_per_kind: int = 2,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "base_fact_id": base_fact_id,
        "cohort_item_id": cohort_item_id,
        "split_assignment": fact["split_assignment"],
        "source": {
            "prompt_en": fact.get("prompt_en"),
            "canonical_fact_en": fact.get("canonical_fact_en"),
            "answer_en": fact.get("answer_en"),
            "answer_aliases_en": list(fact.get("answer_aliases_en") or []),
        },
        "distractors": [
            {
                "distractor_id": row["distractor_id"],
                "slot": row.get("slot"),
                "text_en": row.get("distractor_text_en"),
            }
            for row in distractors
        ],
        "neutral_candidates": [
            {
                "neutral_candidate_id": row["neutral_candidate_id"],
                "slot": row.get("slot"),
                "text_en": row.get("neutral_context_candidate_en"),
            }
            for row in neutrals
        ],
    }
    for field, value in item["source"].items():
        if field == "answer_aliases_en":
            if not isinstance(value, list) or not value or any(
                not isinstance(alias, str) or not alias.strip() for alias in value
            ):
                raise FreezeValidationError(f"invalid zh source aliases: {base_fact_id}")
        elif not isinstance(value, str) or not value.strip():
            raise FreezeValidationError(f"invalid zh source {field}: {base_fact_id}")
    if (
        len(item["distractors"]) != expected_candidates_per_kind
        or len(item["neutral_candidates"]) != expected_candidates_per_kind
    ):
        raise FreezeValidationError(
            "zh item does not have the declared candidate cardinality "
            f"{expected_candidates_per_kind}+{expected_candidates_per_kind}: {base_fact_id}"
        )
    item["input_record_sha256"] = sha256_value(item)
    return item


def _validate_zh_stage_base(
    *,
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    schema_version: str,
    stage: str,
    model: Mapping[str, Any],
    reviewer: bool,
) -> None:
    base_fact_id = str(item["base_fact_id"])
    expected = {
        "schema_version": schema_version,
        "base_fact_id": base_fact_id,
        "input_record_sha256": item["input_record_sha256"],
        "target_language": "zh",
        "human_gold": False,
        "stage": stage,
        "terminal_status": "completed",
        "request_model": model.get("model"),
        "provider_profile": model.get("provider_profile"),
        "expected_response_model": model.get("expected_response_model"),
        "response_model": model.get("expected_response_model"),
        "response_model_identity_status": "matched",
        "credentials_or_endpoints_included": False,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise FreezeValidationError(f"zh {stage} {field} is stale: {base_fact_id}")
    if reviewer and row.get("reviewer_type") != "independent_model_proxy":
        raise FreezeValidationError(f"zh {stage} reviewer type is invalid: {base_fact_id}")
    if not isinstance(row.get("parsed_response"), dict):
        raise FreezeValidationError(f"zh {stage} parsed response is missing: {base_fact_id}")


def _validate_zh_review(
    *,
    manifest_path: Path,
    postreview: Mapping[str, Any],
    universe: Mapping[str, Any],
    config: Mapping[str, Any],
    candidate_review: Optional[Mapping[str, Any]] = None,
    candidate_projection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != zh_tool.MANIFEST_SCHEMA:
        raise FreezeValidationError("zh review manifest schema_version is unsupported")
    if manifest.get("tool_version") != zh_tool.TOOL_VERSION:
        raise FreezeValidationError("zh review tool_version is unsupported")
    if manifest.get("source_fingerprints") != zh_tool.source_fingerprints():
        raise FreezeValidationError("zh review source fingerprints are stale")
    expected_status = (
        "completed_accepted_candidate_projection_zh_proxy_review"
        if candidate_projection is not None
        else "completed_full_zh_proxy_review"
    )
    if manifest.get("status") != expected_status:
        raise FreezeValidationError("zh review is incomplete or contains quarantine")
    if manifest.get("formal_universe_id") != universe["source_universe_id"]:
        raise FreezeValidationError(
            "zh review binds a different predecessor formal universe"
        )

    models = manifest.get("models")
    if (
        not isinstance(models, dict)
        or not isinstance(models.get("generator"), dict)
        or not isinstance(models.get("reviewer"), dict)
    ):
        raise FreezeValidationError("zh model identities are missing")
    generator_model = models["generator"]
    reviewer_model = models["reviewer"]
    generator_route = (
        _required_string(
            generator_model.get("provider_profile"),
            "zh generator provider_profile",
        ),
        _required_string(generator_model.get("model"), "zh generator model"),
        _required_string(
            generator_model.get("expected_response_model"),
            "zh generator expected_response_model",
        ),
    )
    reviewer_route = (
        _required_string(
            reviewer_model.get("provider_profile"),
            "zh reviewer provider_profile",
        ),
        _required_string(reviewer_model.get("model"), "zh reviewer model"),
        _required_string(
            reviewer_model.get("expected_response_model"),
            "zh reviewer expected_response_model",
        ),
    )
    configured_generator_route, configured_reviewer_route = _configured_zh_routes(config)
    if (
        generator_route != configured_generator_route
        or reviewer_route != configured_reviewer_route
    ):
        raise FreezeValidationError("zh model identities differ from the frozen config")
    route_identity = manifest.get("route_identity")
    if not isinstance(route_identity, dict):
        raise FreezeValidationError("zh review route identity is missing")
    try:
        route_records = validate_route_identity_set(
            route_identity,
            expected_routes=[generator_route, reviewer_route],
            expected_protocols=_configured_route_protocols(config),
        )
    except ValueError as exc:
        raise FreezeValidationError(f"zh review route identity is invalid: {exc}") from exc
    expected_review_independence = {
        "model_identity_distinct": generator_route[1:] != reviewer_route[1:],
        "provider_profile_labels_distinct": generator_route[0] != reviewer_route[0],
        "provider_endpoint_distinct": len(
            {
                str(record["normalized_request_route_sha256"])
                for record in route_records
            }
        )
        == len(route_records),
        "provider_infrastructure_independence_claimed": False,
        "claim_scope": "distinct_model_identity_only",
    }
    if (
        expected_review_independence["model_identity_distinct"] is not True
        or expected_review_independence["provider_profile_labels_distinct"] is not True
    ):
        raise FreezeValidationError("zh generation and review identities are not independent")
    if manifest.get("review_independence") != expected_review_independence:
        raise FreezeValidationError("zh review independence declaration is stale")

    input_bindings = manifest.get("input_bindings")
    if not isinstance(input_bindings, dict):
        raise FreezeValidationError("zh input_bindings are missing")
    candidate_scope = candidate_projection or postreview
    expected_inputs = {
        "formal_universe_manifest": universe["source_universe_manifest_binding"],
        "formal_universe_items": universe["source_universe_items_binding"],
        "full_base_facts": candidate_scope["artifacts"]["full_base_facts"]["binding"],
        "distractor_candidates": candidate_scope["artifacts"]["distractor_candidates"]["binding"],
        "neutral_candidates": candidate_scope["artifacts"]["neutral_reference_candidates"]["binding"],
        "split_manifest": candidate_scope["artifacts"]["split_manifest"]["binding"],
    }
    if candidate_projection is not None:
        if candidate_review is None:
            raise FreezeValidationError("candidate review is required for projection zh validation")
        expected_inputs.update(
            {
                "accepted_candidate_projection_manifest": candidate_projection[
                    "manifest_binding"
                ],
                "projection_source_postreview_manifest": postreview[
                    "manifest_binding"
                ],
                "projection_source_candidate_review_manifest": candidate_review[
                    "manifest_binding"
                ],
                "projection_source_candidate_review_adjudications": candidate_review[
                    "adjudications_binding"
                ],
            }
        )
    for label, expected in expected_inputs.items():
        _same_binding(
            input_bindings.get(label),
            expected,
            owner_path=manifest_path,
            label=f"zh {label}",
        )

    candidate_binding = manifest.get("candidate_binding")
    if not isinstance(candidate_binding, dict):
        raise FreezeValidationError("zh candidate_binding is missing")
    for label in ("distractor_candidates", "neutral_candidates"):
        _same_binding(
            candidate_binding.get(label),
            expected_inputs[label],
            owner_path=manifest_path,
            label=f"zh candidate_binding.{label}",
        )
    expected_combined = sha256_value(
        {
            label: expected_inputs[label]["sha256"]
            for label in ("distractor_candidates", "neutral_candidates")
        }
    )
    if candidate_binding.get("combined_sha256") != expected_combined:
        raise FreezeValidationError("zh combined candidate binding is stale")

    retained_order = (
        list(candidate_projection["target_ids"])
        if candidate_projection is not None
        else [
            base_fact_id
            for base_fact_id in universe["source_base_fact_ids"]
            if base_fact_id in postreview["row_index"]
        ]
    )
    excluded_ids = sorted(set(universe["source_base_fact_ids"]) - set(retained_order))
    if len(retained_order) != len(candidate_scope["rows"]):
        raise FreezeValidationError("retained facts are outside formal-universe-v2")
    if candidate_projection is None:
        is_original_full = not excluded_ids
        complete_retained_extension = manifest.get(
            "selection_is_complete_retained_cohort"
        ) is True and manifest.get("excluded_base_fact_ids_sha256") == sha256_value(
            excluded_ids
        )
        if not (manifest.get("selection_is_full_universe") is True and is_original_full) and not (
            complete_retained_extension and not is_original_full
        ):
            raise FreezeValidationError(
                "zh selection is not the complete retained postreview cohort"
            )
    else:
        expected_projection_summary = {
            "projection_id": candidate_projection["manifest"].get("projection_id"),
            "status": candidate_projection["manifest"].get("status"),
            "selection_contract": candidate_projection["manifest"].get(
                "selection_contract"
            ),
            "source_review_contract": candidate_projection["manifest"].get(
                "source_review_contract"
            ),
            "counts": candidate_projection["manifest"].get("counts"),
            "lineage_digests": candidate_projection["manifest"].get(
                "lineage_digests"
            ),
            "projected_record_count": len(retained_order),
            "quarantined_record_count": len(candidate_projection["quarantine"]),
            "selected_candidate_count": len(
                candidate_projection["selected_candidate_ids"]
            ),
            "selected_projection_all_accept": True,
            "human_gold": False,
        }
        if manifest.get("accepted_candidate_projection") != expected_projection_summary:
            raise FreezeValidationError("zh accepted candidate projection contract is stale")
        if not all(
            (
                manifest.get("selection_is_complete_accepted_projection") is True,
                manifest.get("accepted_projection_record_count") == len(retained_order),
                manifest.get("accepted_projection_quarantined_source_record_count")
                == len(candidate_projection["quarantine"]),
                manifest.get("selection_is_full_universe") is False,
            )
        ):
            raise FreezeValidationError(
                "zh selection is not the complete accepted candidate projection"
            )
    expected_split_counts = _split_counts(
        [candidate_scope["row_index"][base_fact_id] for base_fact_id in retained_order]
    )
    if not all(
        (
            manifest.get("selected_record_count") == len(retained_order),
            manifest.get("universe_record_count") == len(universe["source_base_fact_ids"]),
            manifest.get("selected_base_fact_ids_sha256") == sha256_value(retained_order),
            manifest.get("selected_split_counts") == expected_split_counts,
        )
    ):
        raise FreezeValidationError("zh selected ID/count/split contract is stale")

    language_scope = manifest.get("language_scope")
    if not isinstance(language_scope, dict) or not all(
        (
            language_scope.get("processed_language_codes") == ["zh"],
            language_scope.get("other_registered_target_languages_status")
            == "unchanged_pending_translation",
            language_scope.get("does_not_mark_unprocessed_languages_complete") is True,
        )
    ):
        raise FreezeValidationError("zh language scope is invalid")
    boundary = manifest.get("evidence_boundary")
    _zero_execution_contract(boundary, "zh evidence_boundary")
    if not isinstance(boundary, dict) or not all(
        (
            boundary.get("human_gold") is False,
            boundary.get("reviewer_type") == "independent_model_proxy",
            boundary.get("independence_scope") == "distinct_model_identity_only",
            boundary.get("provider_infrastructure_independence_claimed") is False,
            boundary.get("translation_only") is True,
            boundary.get("factual_truth_review_performed") is False,
            boundary.get("distractor_factual_falsehood_review_performed") is False,
            boundary.get("neutral_unrelatedness_review_performed") is False,
            boundary.get("formal_relation_freeze_performed") is False,
            boundary.get("formal_split_freeze_performed") is False,
        )
    ):
        raise FreezeValidationError("zh evidence boundary is invalid")
    completion = manifest.get("completion_claims")
    projection_completion_valid = (
        completion.get("accepted_candidate_projection_zh_translation_review_complete")
        is True
        and completion.get("full_pool_zh_translation_review_complete") is False
        if isinstance(completion, dict) and candidate_projection is not None
        else isinstance(completion, dict)
        and completion.get("full_pool_zh_translation_review_complete") is True
        and completion.get(
            "accepted_candidate_projection_zh_translation_review_complete", False
        )
        is False
    )
    if not isinstance(completion, dict) or not all(
        (
            completion.get("selected_scope_processing_complete") is True,
            projection_completion_valid,
            completion.get("zh_proxy_review_not_human_gold") is True,
            completion.get("other_15_target_languages_complete") is False,
            completion.get("hf_or_behavior_work_performed") is False,
        )
    ):
        raise FreezeValidationError("zh completion claims are incomplete")

    output_bindings = manifest.get("output_artifacts")
    if not isinstance(output_bindings, dict):
        raise FreezeValidationError("zh output artifacts are missing")
    output_specs = {
        "translation_initial": (zh_tool.GENERATION_SCHEMA, False),
        "translation_initial_reviews": (zh_tool.REVIEW_SCHEMA, False),
        "translation_repairs": (zh_tool.GENERATION_SCHEMA, True),
        "translation_repair_reviews": (zh_tool.REVIEW_SCHEMA, True),
        "zh_translation_review_records": (zh_tool.FINAL_SCHEMA, False),
        "translation_quarantine": (zh_tool.QUARANTINE_SCHEMA, True),
    }
    artifacts: Dict[str, Dict[str, Any]] = {}
    for label, (schema, allow_empty) in output_specs.items():
        artifact_path, rows, binding = _verify_binding(
            output_bindings.get(label),
            owner_path=manifest_path,
            label=f"zh {label}",
            jsonl=True,
            expected_schema=schema,
            allow_empty=allow_empty,
        )
        artifacts[label] = {"path": artifact_path, "rows": rows, "binding": binding}
    if artifacts["translation_quarantine"]["rows"]:
        raise FreezeValidationError("zh translation quarantine must be empty")

    distractors_by_fact = _candidate_rows_by_fact(
        candidate_scope["distractors"], id_field="distractor_id"
    )
    neutrals_by_fact = _candidate_rows_by_fact(
        candidate_scope["neutrals"], id_field="neutral_candidate_id"
    )
    items: Dict[str, Dict[str, Any]] = {}
    for base_fact_id in retained_order:
        items[base_fact_id] = _zh_item(
            base_fact_id=base_fact_id,
            cohort_item_id=universe["cohort_item_by_source"][base_fact_id],
            fact=candidate_scope["row_index"][base_fact_id],
            distractors=distractors_by_fact.get(base_fact_id, []),
            neutrals=neutrals_by_fact.get(base_fact_id, []),
            expected_candidates_per_kind=(1 if candidate_projection is not None else 2),
        )

    initial = _unique_index(
        artifacts["translation_initial"]["rows"], "base_fact_id", "zh initial"
    )
    initial_reviews = _unique_index(
        artifacts["translation_initial_reviews"]["rows"],
        "base_fact_id",
        "zh initial reviews",
    )
    repairs = _unique_index(
        artifacts["translation_repairs"]["rows"], "base_fact_id", "zh repairs"
    )
    repair_reviews = _unique_index(
        artifacts["translation_repair_reviews"]["rows"],
        "base_fact_id",
        "zh repair reviews",
    )
    final_rows = artifacts["zh_translation_review_records"]["rows"]
    final_index = _unique_index(final_rows, "base_fact_id", "zh final records")
    if set(initial) != set(items) or set(initial_reviews) != set(items) or set(final_index) != set(items):
        raise FreezeValidationError("zh artifacts do not exactly cover retained facts")

    accepted_initial: set[str] = set()
    rejected_initial: set[str] = set()
    validated_final: Dict[str, Dict[str, Any]] = {}
    for base_fact_id in retained_order:
        item = items[base_fact_id]
        generation = initial[base_fact_id]
        review = initial_reviews[base_fact_id]
        _validate_zh_stage_base(
            row=generation,
            item=item,
            schema_version=zh_tool.GENERATION_SCHEMA,
            stage="initial_generation",
            model=generator_model,
            reviewer=False,
        )
        _validate_zh_stage_base(
            row=review,
            item=item,
            schema_version=zh_tool.REVIEW_SCHEMA,
            stage="initial_review",
            model=reviewer_model,
            reviewer=True,
        )
        try:
            zh_tool.validate_translation(generation["parsed_response"], item)
            zh_tool.validate_review(review["parsed_response"], item)
        except ValueError as exc:
            raise FreezeValidationError(str(exc)) from exc
        if review.get("translation_record_sha256") != sha256_value(generation):
            raise FreezeValidationError(f"zh initial review lineage is stale: {base_fact_id}")
        if review["parsed_response"].get("decision") == "accept":
            accepted_initial.add(base_fact_id)
        else:
            rejected_initial.add(base_fact_id)

    if set(repairs) != rejected_initial or set(repair_reviews) != rejected_initial:
        raise FreezeValidationError("zh repair coverage does not equal rejected initial reviews")
    for base_fact_id in sorted(rejected_initial):
        item = items[base_fact_id]
        repair = repairs[base_fact_id]
        review = repair_reviews[base_fact_id]
        _validate_zh_stage_base(
            row=repair,
            item=item,
            schema_version=zh_tool.GENERATION_SCHEMA,
            stage="repair_1",
            model=generator_model,
            reviewer=False,
        )
        _validate_zh_stage_base(
            row=review,
            item=item,
            schema_version=zh_tool.REVIEW_SCHEMA,
            stage="repair_1_review",
            model=reviewer_model,
            reviewer=True,
        )
        try:
            zh_tool.validate_translation(repair["parsed_response"], item)
            zh_tool.validate_review(review["parsed_response"], item)
        except ValueError as exc:
            raise FreezeValidationError(str(exc)) from exc
        if repair.get("repaired_translation_record_sha256") != sha256_value(
            initial[base_fact_id]
        ) or repair.get("repaired_review_record_sha256") != sha256_value(
            initial_reviews[base_fact_id]
        ):
            raise FreezeValidationError(f"zh repair lineage is stale: {base_fact_id}")
        if review.get("translation_record_sha256") != sha256_value(repair):
            raise FreezeValidationError(f"zh repair review lineage is stale: {base_fact_id}")
        if review["parsed_response"].get("decision") != "accept":
            raise FreezeValidationError(f"zh repaired translation is rejected: {base_fact_id}")

    if [str(row["base_fact_id"]) for row in final_rows] != retained_order:
        raise FreezeValidationError("zh final record order differs from formal lineage")
    for base_fact_id in retained_order:
        item = items[base_fact_id]
        row = final_index[base_fact_id]
        origin = "initial" if base_fact_id in accepted_initial else "repair_1"
        generation = initial[base_fact_id] if origin == "initial" else repairs[base_fact_id]
        review = initial_reviews[base_fact_id] if origin == "initial" else repair_reviews[base_fact_id]
        expected_scalar = {
            "schema_version": zh_tool.FINAL_SCHEMA,
            "base_fact_id": base_fact_id,
            "cohort_item_id": item["cohort_item_id"],
            "input_record_sha256": item["input_record_sha256"],
            "split_assignment": item["split_assignment"],
            "target_language": "zh",
            "human_gold": False,
            "reviewer_type": "independent_model_proxy",
            "terminal_status": "completed",
            "translation_origin": origin,
            "source_fields": item["source"],
            "source_distractors": item["distractors"],
            "source_neutral_candidates": item["neutral_candidates"],
            "translation": generation["parsed_response"],
            "translation_equivalence_review": review["parsed_response"],
            "quarantine_reasons": [],
        }
        for field, value in expected_scalar.items():
            if row.get(field) != value:
                raise FreezeValidationError(f"zh final {field} is stale: {base_fact_id}")
        if row.get("maximum_translation_repairs_performed") != (0 if origin == "initial" else 1):
            raise FreezeValidationError(f"zh final repair count is stale: {base_fact_id}")
        claims = row.get("formal_claims")
        if claims != {
            "translation_proxy_accepted": True,
            "human_gold": False,
            "factual_truth_reviewed": False,
            "distractor_falsehood_reviewed": False,
            "neutral_unrelatedness_reviewed": False,
        }:
            raise FreezeValidationError(f"zh final claims are invalid: {base_fact_id}")
        lineage = row.get("lineage")
        if not isinstance(lineage, dict) or not all(
            (
                lineage.get("initial_generation_record_sha256")
                == sha256_value(initial[base_fact_id]),
                lineage.get("initial_review_record_sha256")
                == sha256_value(initial_reviews[base_fact_id]),
                lineage.get("repair_record_sha256")
                == (sha256_value(repairs[base_fact_id]) if origin == "repair_1" else None),
                lineage.get("repair_review_record_sha256")
                == (
                    sha256_value(repair_reviews[base_fact_id])
                    if origin == "repair_1"
                    else None
                ),
            )
        ):
            raise FreezeValidationError(f"zh final lineage is stale: {base_fact_id}")
        validated_final[base_fact_id] = dict(row)

    counts = manifest.get("counts")
    if not isinstance(counts, dict) or not all(
        (
            counts.get("universe_records") == len(universe["source_base_fact_ids"]),
            counts.get("selected_records") == len(items),
            counts.get("proxy_accepted_records") == len(items),
            counts.get("quarantined_records") == 0,
            counts.get("initial_review_accepted_records") == len(accepted_initial),
            counts.get("repair_attempted_records") == len(repairs),
            counts.get("repair_review_accepted_records") == len(repair_reviews),
        )
    ):
        raise FreezeValidationError("zh completion counts are stale")

    contract_fields = (
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
    contract_payload = {field: manifest.get(field) for field in contract_fields}
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
            contract_payload[field] = manifest[field]
    if manifest.get("run_fingerprint") != sha256_value(contract_payload):
        raise FreezeValidationError("zh run_fingerprint is stale")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_binding": _binding(manifest_path, schema_version=zh_tool.MANIFEST_SCHEMA),
        "artifacts": artifacts,
        "items": items,
        "retained_order": retained_order,
        "final_rows": final_rows,
        "final_index": validated_final,
    }


def _validate_zh_accepted_projection(
    *,
    manifest_path: Path,
    parent_zh_review_manifest_path: Path,
    candidate_projection: Optional[Mapping[str, Any]],
    universe: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate an offline accepted-only projection of a terminal zh run."""

    if candidate_projection is None:
        raise FreezeValidationError(
            "zh accepted projection requires an accepted candidate projection"
        )
    try:
        projection = zh_accepted_projection_tool.load_projection_context(
            Path(manifest_path),
            expected_accepted_candidate_projection_manifest_path=Path(
                candidate_projection["manifest_path"]
            ),
            expected_zh_review_manifest_path=Path(parent_zh_review_manifest_path),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise FreezeValidationError(f"zh accepted projection is invalid: {exc}") from exc

    parent = projection["parent"]
    parent_manifest = parent["zh_manifest"]
    parent_manifest_path = Path(parent["zh_manifest_path"])
    if parent_manifest.get("formal_universe_id") != universe["source_universe_id"]:
        raise FreezeValidationError(
            "parent zh review binds a different predecessor formal universe"
        )
    if parent_manifest.get("universe_record_count") != len(
        universe["source_base_fact_ids"]
    ):
        raise FreezeValidationError("parent zh universe count is stale")

    parent_inputs = parent_manifest.get("input_bindings")
    if not isinstance(parent_inputs, dict):
        raise FreezeValidationError("parent zh input bindings are missing")
    for label, expected in (
        ("formal_universe_manifest", universe["source_universe_manifest_binding"]),
        ("formal_universe_items", universe["source_universe_items_binding"]),
        ("accepted_candidate_projection_manifest", candidate_projection["manifest_binding"]),
    ):
        _same_binding(
            parent_inputs.get(label),
            expected,
            owner_path=parent_manifest_path,
            label=f"parent zh {label}",
        )

    models = parent_manifest.get("models")
    if (
        not isinstance(models, dict)
        or not isinstance(models.get("generator"), dict)
        or not isinstance(models.get("reviewer"), dict)
    ):
        raise FreezeValidationError("parent zh model identities are missing")
    generator_model = models["generator"]
    reviewer_model = models["reviewer"]
    generator_route = (
        _required_string(
            generator_model.get("provider_profile"),
            "parent zh generator provider_profile",
        ),
        _required_string(generator_model.get("model"), "parent zh generator model"),
        _required_string(
            generator_model.get("expected_response_model"),
            "parent zh generator expected_response_model",
        ),
    )
    reviewer_route = (
        _required_string(
            reviewer_model.get("provider_profile"),
            "parent zh reviewer provider_profile",
        ),
        _required_string(reviewer_model.get("model"), "parent zh reviewer model"),
        _required_string(
            reviewer_model.get("expected_response_model"),
            "parent zh reviewer expected_response_model",
        ),
    )
    configured_generator_route, configured_reviewer_route = _configured_zh_routes(config)
    if (
        generator_route != configured_generator_route
        or reviewer_route != configured_reviewer_route
    ):
        raise FreezeValidationError(
            "parent zh model identities differ from the frozen config"
        )
    route_identity = parent_manifest.get("route_identity")
    if not isinstance(route_identity, dict):
        raise FreezeValidationError("parent zh route identity is missing")
    try:
        route_records = validate_route_identity_set(
            route_identity,
            expected_routes=[generator_route, reviewer_route],
            expected_protocols=_configured_route_protocols(config),
        )
    except ValueError as exc:
        raise FreezeValidationError(
            f"parent zh route identity is invalid: {exc}"
        ) from exc
    expected_review_independence = {
        "model_identity_distinct": generator_route[1:] != reviewer_route[1:],
        "provider_profile_labels_distinct": generator_route[0] != reviewer_route[0],
        "provider_endpoint_distinct": len(
            {
                str(record["normalized_request_route_sha256"])
                for record in route_records
            }
        )
        == len(route_records),
        "provider_infrastructure_independence_claimed": False,
        "claim_scope": "distinct_model_identity_only",
    }
    if (
        expected_review_independence["model_identity_distinct"] is not True
        or expected_review_independence["provider_profile_labels_distinct"] is not True
        or parent_manifest.get("review_independence")
        != expected_review_independence
    ):
        raise FreezeValidationError("parent zh review independence is invalid")

    language_scope = parent_manifest.get("language_scope")
    if not isinstance(language_scope, dict) or not all(
        (
            language_scope.get("processed_language_codes") == ["zh"],
            language_scope.get("other_registered_target_languages_status")
            == "unchanged_pending_translation",
            language_scope.get("does_not_mark_unprocessed_languages_complete") is True,
        )
    ):
        raise FreezeValidationError("parent zh language scope is invalid")
    boundary = parent_manifest.get("evidence_boundary")
    _zero_execution_contract(boundary, "parent zh evidence_boundary")
    if not isinstance(boundary, dict) or not all(
        (
            boundary.get("human_gold") is False,
            boundary.get("reviewer_type") == "independent_model_proxy",
            boundary.get("independence_scope") == "distinct_model_identity_only",
            boundary.get("provider_infrastructure_independence_claimed") is False,
            boundary.get("translation_only") is True,
            boundary.get("factual_truth_review_performed") is False,
            boundary.get("distractor_factual_falsehood_review_performed") is False,
            boundary.get("neutral_unrelatedness_review_performed") is False,
            boundary.get("formal_relation_freeze_performed") is False,
            boundary.get("formal_split_freeze_performed") is False,
        )
    ):
        raise FreezeValidationError("parent zh evidence boundary is invalid")

    projected_ids = [str(value) for value in projection["projected_target_ids"]]
    if not projected_ids or not set(projected_ids).issubset(
        set(candidate_projection["target_ids"])
    ):
        raise FreezeValidationError("zh accepted projection target IDs are invalid")
    final_rows = projection["filtered_rows"]
    if [str(row.get("base_fact_id") or "") for row in final_rows] != projected_ids:
        raise FreezeValidationError("projected zh final order is stale")
    final_index = _unique_index(
        final_rows, "base_fact_id", "projected zh final records"
    )
    if any(
        row.get("terminal_status") != "completed"
        or row.get("formal_claims", {}).get("translation_proxy_accepted") is not True
        or row.get("human_gold") is not False
        for row in final_rows
    ):
        raise FreezeValidationError("projected zh records are not all proxy accepted")

    secondary_manifest = projection["manifest"]
    selection = secondary_manifest.get("selection_contract")
    safety = secondary_manifest.get("safety_contract")
    if not isinstance(selection, dict) or not all(
        (
            selection.get("quarantine_ids_seed_the_projection") is True,
            selection.get("only_parent_zh_quarantine_targets_are_excluded") is True,
            selection.get("parent_candidate_projection_is_support_cohort") is True,
            selection.get("parent_support_cohort_record_count")
            == len(candidate_projection["target_ids"]),
            selection.get(
                "selected_candidate_donor_must_remain_in_projected_target_cohort"
            )
            is False,
            selection.get(
                "selected_candidate_donor_must_remain_in_parent_support_cohort"
            )
            is True,
            selection.get("support_only_donors_are_not_behavior_or_vector_targets")
            is True,
            selection.get("candidate_reselection_performed") is False,
            selection.get("model_calls_performed") is False,
            selection.get("component_recomputed") is False,
            selection.get("split_recomputed") is False,
            selection.get("human_gold") is False,
        )
    ):
        raise FreezeValidationError("zh accepted projection selection contract is invalid")
    if not isinstance(safety, dict) or not all(
        (
            safety.get("output_is_pre_freeze_projection") is True,
            safety.get("review_freeze_emitted") is False,
            safety.get("split_freeze_emitted") is False,
            safety.get("hf_checkpoint_bound") is False,
            safety.get("hf_tokenizer_bound") is False,
            safety.get("hf_model_executed") is False,
            safety.get("hf_tokenizer_executed") is False,
            safety.get("behavior_executed") is False,
            safety.get("validation_exposed") is False,
            safety.get("sealed_exposed") is False,
            safety.get("path_not_token_authorized") is False,
            safety.get("support_only_donors_are_execution_targets") is False,
            safety.get("human_gold") is False,
        )
    ):
        raise FreezeValidationError("zh accepted projection safety contract is invalid")

    support_only_donor_ids = [
        str(value) for value in projection["support_only_donor_ids"]
    ]
    if set(projection["excluded_ids"]) != set(projection["seed_ids"]):
        raise FreezeValidationError(
            "zh accepted projection excludes targets beyond parent zh quarantine"
        )
    if (
        set(support_only_donor_ids).intersection(projected_ids)
        or not set(support_only_donor_ids).issubset(
            set(candidate_projection["target_ids"])
        )
    ):
        raise FreezeValidationError("zh support-only donor partition is invalid")

    return {
        "manifest": projection["filtered_review_manifest"],
        "manifest_path": projection["filtered_review_manifest_path"],
        "manifest_binding": projection["filtered_review_manifest_binding"],
        "parent_manifest": parent_manifest,
        "parent_manifest_path": parent_manifest_path,
        "parent_manifest_binding": parent["zh_manifest_binding"],
        "artifacts": {
            "zh_translation_review_records": {
                "path": projection["filtered_rows_path"],
                "rows": final_rows,
                "binding": projection["filtered_rows_binding"],
            },
            "translation_quarantine": {
                "path": projection["empty_quarantine_path"],
                "rows": [],
                "binding": projection["empty_quarantine_binding"],
            },
        },
        "items": {
            base_fact_id: parent["items_by_id"][base_fact_id]
            for base_fact_id in projected_ids
        },
        "retained_order": projected_ids,
        "final_rows": final_rows,
        "final_index": final_index,
        "secondary_projection": projection,
    }


def _validate_history_against_split(
    *,
    universe: Mapping[str, Any],
    postreview: Mapping[str, Any],
    frozen_target_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    known = set(universe["known_exposure_ids"])
    retained = set(postreview["row_index"])
    if frozen_target_ids is not None:
        retained = {str(value) for value in frozen_target_ids}
    if not retained.issubset(set(postreview["row_index"])):
        raise FreezeValidationError("frozen projection contains unknown postreview targets")
    excluded = (
        set(postreview["exclusion_index"])
        if frozen_target_ids is None
        else set(universe["source_base_fact_ids"]) - retained
    )
    retained_known = sorted(known & retained)
    excluded_known = sorted(known & excluded)
    sensitive = sorted(
        base_fact_id
        for base_fact_id in retained_known
        if postreview["row_index"][base_fact_id]["split_assignment"]
        in {"validation", "sealed"}
    )
    if sensitive:
        raise FreezeValidationError(
            "historically exposed facts remain in Validation/Sealed: "
            + ", ".join(sensitive[:5])
        )
    retained_by_split = Counter(
        str(postreview["row_index"][base_fact_id]["split_assignment"])
        for base_fact_id in retained_known
    )
    return {
        "known_historical_exposure_count": len(known),
        "retained_known_historical_exposure_count": len(retained_known),
        "excluded_known_historical_exposure_count": len(excluded_known),
        "known_exposure_outside_postreview_dispositions_count": len(
            known - retained - excluded
        ),
        "retained_known_exposure_counts_by_split": {
            split: retained_by_split.get(split, 0) for split in SPLITS
        },
        "validation_historical_exposure_count": retained_by_split.get(
            "validation", 0
        ),
        "sealed_historical_exposure_count": retained_by_split.get("sealed", 0),
        "known_historical_exposure_ids_sha256": sha256_value(sorted(known)),
        "retained_known_historical_exposure_ids_sha256": sha256_value(
            retained_known
        ),
        "excluded_known_historical_exposure_ids_sha256": sha256_value(
            excluded_known
        ),
    }


def _exact_bundle_rows(
    *,
    postreview: Mapping[str, Any],
    candidate_review: Mapping[str, Any],
    zh_review: Mapping[str, Any],
    universe: Mapping[str, Any],
    candidate_projection: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    candidate_scope = candidate_projection or postreview
    distractor_index = _unique_index(
        candidate_scope["distractors"], "distractor_id", "distractor candidates"
    )
    neutral_index = _unique_index(
        candidate_scope["neutrals"], "neutral_candidate_id", "neutral candidates"
    )
    unit_by_id = {unit.candidate_id: unit for unit in candidate_review["units"]}
    known_exposure_ids = set(universe["known_exposure_ids"])
    rows: List[Dict[str, Any]] = []
    for base_fact_id in zh_review["retained_order"]:
        source = candidate_scope["row_index"][base_fact_id]
        candidate_evidence = (
            candidate_projection["by_fact"][base_fact_id]
            if candidate_projection is not None
            else candidate_review["by_fact"][base_fact_id]
        )
        distractor_reviews = sorted(
            candidate_evidence["distractor"],
            key=lambda row: int(distractor_index[str(row["candidate_id"])]["slot"]),
        )
        neutral_reviews = sorted(
            candidate_evidence["neutral"],
            key=lambda row: int(neutral_index[str(row["candidate_id"])]["slot"]),
        )
        row = copy.deepcopy(dict(source))
        row["source_schema_version"] = row.get("schema_version")
        row["schema_version"] = BUNDLE_SCHEMA_VERSION
        row["canonical_status"] = "frozen"
        row["evidence_tier"] = "independent_adjudicated"
        row["reviewer_type"] = "independent_model_proxy"
        row["review_only"] = False
        row["human_gold"] = False
        row["split_status"] = "frozen"
        row["split_group_id"] = row["leakage_component_id"]
        row["probe_relation_id"] = None
        row["relation_formalization_status"] = (
            "provisional_partition_not_formal_probe_relation_id"
        )
        row["semantic_protocol_status"] = STATUS
        row["semantic_near_duplicate_recall_guaranteed"] = False
        row["distractors_en"] = [
            {
                **copy.deepcopy(distractor_index[str(review["candidate_id"])]),
                "verified": True,
                "review_status": "accepted_independent_model_proxy",
                "review_evidence_tier": "independent_model_proxy",
                "human_gold": False,
                "review_adjudication": copy.deepcopy(review),
            }
            for review in distractor_reviews
        ]
        row["neutral_candidates_en"] = [
            {
                **copy.deepcopy(neutral_index[str(review["candidate_id"])]),
                "verified_unrelated": True,
                "review_status": "accepted_independent_model_proxy",
                "review_evidence_tier": "independent_model_proxy",
                "human_gold": False,
                "review_adjudication": copy.deepcopy(review),
            }
            for review in neutral_reviews
        ]
        row["zh_translation_review"] = copy.deepcopy(
            zh_review["final_index"][base_fact_id]
        )
        row["historical_exposure"] = {
            "status": (
                "known_historically_exposed_development_only"
                if base_fact_id in known_exposure_ids
                else "not_listed_as_historically_exposed_in_scope_owner_attestation"
            ),
            "historically_exposed": base_fact_id in known_exposure_ids,
            "scope_owner_attestation_sha256": universe["attestation_binding"][
                "sha256"
            ],
        }
        row["freeze_evidence"] = {
            "postreview_record_sha256": sha256_value(source),
            "candidate_adjudication_ids": [
                str(review["candidate_id"])
                for review in [*distractor_reviews, *neutral_reviews]
            ],
            "candidate_adjudication_row_hashes_sha256": sha256_value(
                [
                    sha256_value(review)
                    for review in [*distractor_reviews, *neutral_reviews]
                ]
            ),
            "zh_translation_review_record_sha256": sha256_value(
                zh_review["final_index"][base_fact_id]
            ),
            "formal_cohort_item_id": universe["cohort_item_by_source"][
                base_fact_id
            ],
            "formal_universe_id": universe["manifest"]["universe_id"],
            "proxy_evidence_only": True,
            "human_gold": False,
        }
        if candidate_projection is not None:
            row["freeze_evidence"]["accepted_candidate_projection_id"] = (
                candidate_projection["manifest"]["projection_id"]
            )
            row["freeze_evidence"]["accepted_candidate_projection_manifest_sha256"] = (
                candidate_projection["manifest_binding"]["sha256"]
            )
        if zh_review.get("secondary_projection") is not None:
            secondary = zh_review["secondary_projection"]
            row["freeze_evidence"]["zh_accepted_projection_id"] = secondary[
                "manifest"
            ]["projection_id"]
            row["freeze_evidence"]["zh_accepted_projection_manifest_sha256"] = (
                secondary["manifest_binding"]["sha256"]
            )
        row["translation_status"] = "zh_proxy_reviewed_accepted"
        row["candidate_cardinality_per_target"] = {
            "distractor": len(distractor_reviews),
            "neutral": len(neutral_reviews),
        }
        row["distractor_status"] = (
            "one_proxy_reviewed_accepted"
            if candidate_projection is not None
            else "two_proxy_reviewed_accepted"
        )
        row["neutral_status"] = (
            "one_proxy_reviewed_accepted"
            if candidate_projection is not None
            else "two_proxy_reviewed_accepted"
        )
        row["exact_hf_checkpoint_status"] = "pending_selection_and_binding"
        row["exact_hf_tokenizer_status"] = "pending_selection_and_binding"
        row["hf_model_execution_status"] = "not_run"
        row["hf_tokenizer_execution_status"] = "not_run"
        row["behavior_execution_status"] = "not_run"
        row["path_not_token_experiment_ready"] = False
        if any(
            unit_by_id[review["candidate_id"]].target_base_fact_id != base_fact_id
            for review in [*distractor_reviews, *neutral_reviews]
        ):
            raise FreezeValidationError(f"candidate target binding is stale: {base_fact_id}")
        rows.append(row)
    return rows


def _collect_bound_snapshots(
    value: Any, *, owner_path: Path, output: MutableMapping[str, str]
) -> None:
    if isinstance(value, dict):
        if "sha256" in value and ("path" in value or "filename" in value):
            try:
                path = _resolve_binding_path(value, owner_path, "snapshot")
            except FreezeValidationError:
                path = None
            if path is not None and path.is_file():
                expected = value.get("sha256")
                if expected != sha256_file(path):
                    raise FreezeValidationError(f"snapshot binding is stale: {path}")
                output[str(path)] = str(expected)
        for child in value.values():
            _collect_bound_snapshots(child, owner_path=owner_path, output=output)
    elif isinstance(value, list):
        for child in value:
            _collect_bound_snapshots(child, owner_path=owner_path, output=output)


def _assert_snapshots_current(snapshots: Mapping[str, str]) -> None:
    for raw_path, expected_sha in snapshots.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FreezeValidationError(f"input changed during freeze: {path}")


def _publish_directory_no_replace(staged_dir: Path, output_dir: Path) -> None:
    """Atomically publish a directory without replacing a concurrent target."""

    staged_dir = Path(staged_dir).resolve()
    output_dir = Path(output_dir).resolve()
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("renamex_np is unavailable; refusing unsafe publish")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staged_dir), os.fsencode(output_dir), 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("renameat2 is unavailable; refusing unsafe publish")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100,
            os.fsencode(staged_dir),
            -100,
            os.fsencode(output_dir),
            0x00000001,
        )
    elif os.name == "nt":
        os.rename(staged_dir, output_dir)
        return
    else:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), str(output_dir))
    raise OSError(error_number, os.strerror(error_number), str(output_dir))


def freeze(
    *,
    postreview_rebuild_manifest_path: Path,
    candidate_review_manifest_path: Path,
    candidate_adjudications_path: Path,
    accepted_candidate_projection_manifest_path: Optional[Path] = None,
    zh_accepted_projection_manifest_path: Optional[Path] = None,
    zh_review_manifest_path: Path,
    formal_universe_v2_manifest_path: Path,
    historical_exposure_contract_path: Path,
    scope_owner_attestation_path: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    expected_original_universe_count: int = EXPECTED_ORIGINAL_UNIVERSE_COUNT,
    now_fn: Any = utc_now,
) -> Dict[str, Any]:
    """Validate all pre-HF evidence and atomically publish immutable freeze files."""

    expected_original_universe_count = _required_nonnegative_int(
        expected_original_universe_count, "expected_original_universe_count"
    )
    if expected_original_universe_count < 1:
        raise FreezeValidationError("expected_original_universe_count must be positive")
    output_dir = Path(output_dir).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")

    config = _validate_config(Path(config_path))
    universe = _validate_formal_lineage_and_exposure(
        universe_manifest_path=Path(formal_universe_v2_manifest_path),
        contract_path=Path(historical_exposure_contract_path),
        attestation_path=Path(scope_owner_attestation_path),
        expected_original_count=expected_original_universe_count,
    )
    postreview = _validate_postreview(
        Path(postreview_rebuild_manifest_path), config=config["value"]
    )
    if postreview["input_ids"] != universe["source_base_fact_ids"]:
        raise FreezeValidationError(
            "postreview input IDs/order differ from formal-universe-v2 lineage"
        )
    if len(postreview["input_ids"]) != expected_original_universe_count:
        raise FreezeValidationError("postreview original universe count is stale")
    candidate_review = _validate_candidate_review(
        manifest_path=Path(candidate_review_manifest_path),
        adjudications_path=Path(candidate_adjudications_path),
        postreview=postreview,
        config=config["value"],
        require_all_accept=accepted_candidate_projection_manifest_path is None,
    )
    candidate_projection = (
        _validate_candidate_projection(
            manifest_path=Path(accepted_candidate_projection_manifest_path),
            postreview=postreview,
            candidate_review=candidate_review,
            adjudications_path=Path(candidate_adjudications_path),
        )
        if accepted_candidate_projection_manifest_path is not None
        else None
    )
    if (
        zh_accepted_projection_manifest_path is not None
        and candidate_projection is None
    ):
        raise FreezeValidationError(
            "zh accepted projection requires --accepted-candidate-projection-manifest"
        )
    candidate_evidence_lineage = _candidate_evidence_lineage(
        candidate_review["evidence_bundle"]
    )
    zh_review = (
        _validate_zh_accepted_projection(
            manifest_path=Path(zh_accepted_projection_manifest_path),
            parent_zh_review_manifest_path=Path(zh_review_manifest_path),
            candidate_projection=candidate_projection,
            universe=universe,
            config=config["value"],
        )
        if zh_accepted_projection_manifest_path is not None
        else _validate_zh_review(
            manifest_path=Path(zh_review_manifest_path),
            postreview=postreview,
            universe=universe,
            config=config["value"],
            candidate_review=candidate_review,
            candidate_projection=candidate_projection,
        )
    )
    zh_accepted_projection = zh_review.get("secondary_projection")
    exposure_audit = _validate_history_against_split(
        universe=universe,
        postreview=postreview,
        frozen_target_ids=zh_review["retained_order"],
    )
    if exposure_audit["known_exposure_outside_postreview_dispositions_count"] != 0:
        raise FreezeValidationError("historical exposure IDs lack postreview disposition")

    exact_rows = _exact_bundle_rows(
        postreview=postreview,
        candidate_review=candidate_review,
        zh_review=zh_review,
        universe=universe,
        candidate_projection=candidate_projection,
    )
    retained_ids = [str(row["base_fact_id"]) for row in exact_rows]
    postreview_excluded_ids = [
        str(row["base_fact_id"]) for row in postreview["exclusions"]
    ]
    projection_excluded_ids = (
        list(candidate_projection["quarantine_ids"])
        if candidate_projection is not None
        else []
    )
    zh_projection_excluded_ids = (
        list(zh_accepted_projection["excluded_ids"])
        if zh_accepted_projection is not None
        else []
    )
    excluded_set = (
        set(postreview_excluded_ids)
        | set(projection_excluded_ids)
        | set(zh_projection_excluded_ids)
    )
    excluded_ids = [
        base_fact_id
        for base_fact_id in universe["source_base_fact_ids"]
        if base_fact_id in excluded_set
    ]
    if set(retained_ids).intersection(excluded_set) or (
        set(retained_ids) | excluded_set
    ) != set(universe["source_base_fact_ids"]):
        raise FreezeValidationError(
            "final zh projection does not partition the formal universe"
        )
    candidate_excluded_ids = (
        list(postreview["candidate_resolution"]["candidate_excluded_ids"])
        if postreview.get("candidate_resolution") is not None
        else []
    )
    fact_stage_excluded_count = len(postreview_excluded_ids) - len(
        candidate_excluded_ids
    )
    split_counts = _split_counts(exact_rows)
    component_ids = sorted(
        {postreview["component_by_fact"][base_fact_id] for base_fact_id in retained_ids}
    )
    candidate_cardinality = (
        dict(candidate_projection["candidate_cardinality_per_target"])
        if candidate_projection is not None
        else {"distractor": 2, "neutral": 2}
    )
    retained_id_set = set(retained_ids)
    selected_distractor_count = sum(
        str(row["base_fact_id"]) in retained_id_set
        for row in (
            candidate_projection["distractors"]
            if candidate_projection is not None
            else postreview["distractors"]
        )
    )
    selected_neutral_count = sum(
        str(row["base_fact_id"]) in retained_id_set
        for row in (
            candidate_projection["neutrals"]
            if candidate_projection is not None
            else postreview["neutrals"]
        )
    )
    created_at = _required_string(now_fn(), "created_at")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.", suffix=".tmp", dir=str(output_dir.parent)
        )
    )
    published = False
    try:
        bundle_stage = staged_dir / "exact_reviewed_bundle.jsonl"
        review_stage = staged_dir / "review_freeze_manifest.json"
        split_stage = staged_dir / "split_freeze_manifest.json"
        gate_stage = staged_dir / "pre_exact_hf_gate_manifest.json"
        write_jsonl(bundle_stage, exact_rows)
        bundle_binding = _binding(
            bundle_stage,
            schema_version=BUNDLE_SCHEMA_VERSION,
            record_count=len(exact_rows),
            output_path=output_dir / bundle_stage.name,
        )
        bundle_binding.update(
            {
                "ordered_base_fact_ids_sha256": sha256_value(retained_ids),
                "ordered_row_hashes_sha256": sha256_value(
                    [sha256_value(row) for row in exact_rows]
                ),
            }
        )

        review_manifest: Dict[str, Any] = {
            "schema_version": REVIEW_FREEZE_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": STATUS,
            "created_at": created_at,
            "source_universe": {
                "formal_universe_id": universe["manifest"]["universe_id"],
                "original_record_count": len(universe["source_base_fact_ids"]),
                "ordered_source_base_fact_ids_sha256": sha256_value(
                    universe["source_base_fact_ids"]
                ),
                "formal_universe_v2_manifest": universe["manifest_binding"],
                "formal_universe_v2_items": universe["items_binding"],
            },
            "postreview_rebuild": postreview["manifest_binding"],
            "accepted_candidate_projection": (
                {
                    "manifest": candidate_projection["manifest_binding"],
                    "projection_id": candidate_projection["manifest"]["projection_id"],
                    "source_postreview_manifest": postreview["manifest_binding"],
                    "source_candidate_review_manifest": candidate_review[
                        "manifest_binding"
                    ],
                    "source_candidate_review_adjudications": candidate_review[
                        "adjudications_binding"
                    ],
                    "projected_target_count": len(candidate_projection["target_ids"]),
                    "quarantined_target_count": len(
                        candidate_projection["quarantine_ids"]
                    ),
                    "selected_candidate_count": len(
                        candidate_projection["selected_candidate_ids"]
                    ),
                    "cardinality_per_target": candidate_cardinality,
                    "source_review_full_set_complete": True,
                    "source_review_all_accept": candidate_review[
                        "all_candidates_accepted"
                    ],
                    "selected_projection_all_accept": True,
                    "human_gold": False,
                }
                if candidate_projection is not None
                else None
            ),
            "zh_accepted_projection": (
                {
                    "manifest": zh_accepted_projection["manifest_binding"],
                    "projection_id": zh_accepted_projection["manifest"][
                        "projection_id"
                    ],
                    "parent_zh_review_manifest": zh_review[
                        "parent_manifest_binding"
                    ],
                    "projected_target_count": len(
                        zh_accepted_projection["projected_target_ids"]
                    ),
                    "projected_target_ids_sha256": sha256_value(
                        zh_accepted_projection["projected_target_ids"]
                    ),
                    "parent_zh_quarantine_seed_count": len(
                        zh_accepted_projection["seed_ids"]
                    ),
                    "target_only_exclusion_count": len(
                        zh_accepted_projection["excluded_ids"]
                    ),
                    "support_only_donor_count": len(
                        zh_accepted_projection["support_only_donor_ids"]
                    ),
                    "support_only_donor_ids_sha256": sha256_value(
                        zh_accepted_projection["support_only_donor_ids"]
                    ),
                    "donor_must_be_in_final_target_cohort": False,
                    "donor_must_be_in_parent_support_cohort": True,
                    "support_only_donors_are_execution_targets": False,
                    "candidate_reselection_performed": False,
                    "component_recomputed": False,
                    "split_recomputed": False,
                    "model_calls_performed": False,
                    "human_gold": False,
                }
                if zh_accepted_projection is not None
                else None
            ),
            "candidate_resolution": (
                {
                    "manifest": postreview["candidate_resolution"][
                        "manifest_binding"
                    ],
                    "resolution_id": postreview["candidate_resolution"][
                        "resolution_id"
                    ],
                    "pair_exclusion_count": len(
                        postreview["candidate_resolution"]["pair_rows"]
                    ),
                    "cohort_exclusion_count": len(candidate_excluded_ids),
                    "predecessor_reviews_reused": False,
                }
                if postreview.get("candidate_resolution") is not None
                else None
            ),
            "candidate_review": {
                "manifest": candidate_review["manifest_binding"],
                "checkpoint": candidate_review["checkpoint_binding"],
                "adjudications": candidate_review["adjudications_binding"],
                "candidate_count": len(candidate_review["units"]),
                "accepted_candidate_count": candidate_review[
                    "accepted_candidate_count"
                ],
                "overall_decision_counts": candidate_review[
                    "overall_decision_counts"
                ],
                "all_candidates_terminal": candidate_review[
                    "all_candidates_terminal"
                ],
                "all_candidates_terminal_and_accepted": candidate_review[
                    "all_candidates_accepted"
                ],
                "selected_projection_candidates_all_accepted": (
                    True if candidate_projection is not None else None
                ),
                "review_semantics_contract_sha256": candidate_review[
                    "review_semantics_contract_sha256"
                ],
                "evidence_origin_counts": candidate_review[
                    "evidence_origin_counts"
                ],
                "fresh_model_review_candidate_count": candidate_review[
                    "fresh_model_review_candidate_count"
                ],
                "carried_forward_candidate_count": candidate_review[
                    "carried_forward_candidate_count"
                ],
                "full_evidence_partition_complete": candidate_review[
                    "full_evidence_partition_complete"
                ],
                "complete_evidence_lineage": candidate_evidence_lineage,
            },
            "zh_translation_review": {
                "manifest": zh_review["manifest_binding"],
                "parent_manifest": zh_review.get("parent_manifest_binding"),
                "zh_accepted_projection": (
                    zh_accepted_projection["manifest_binding"]
                    if zh_accepted_projection is not None
                    else None
                ),
                "records": zh_review["artifacts"][
                    "zh_translation_review_records"
                ]["binding"],
                "record_count": len(zh_review["final_rows"]),
                "all_records_terminal_and_proxy_accepted": True,
                "selection_mode": (
                    "accepted_candidate_projection_1_plus_1"
                    if candidate_projection is not None
                    else "complete_retained_postreview_exactly_two"
                ),
                "candidate_cardinality_per_target": candidate_cardinality,
                "target_language": "zh",
                "other_registered_languages_status": "pending_translation_not_required_for_zh_first_gate",
            },
            "exact_reviewed_bundle": bundle_binding,
            "counts": {
                "original_universe": len(universe["source_base_fact_ids"]),
                "retained_frozen": len(exact_rows),
                "cohort_excluded": len(excluded_ids),
                "fact_stage_cohort_excluded": fact_stage_excluded_count,
                "candidate_stage_cohort_excluded": len(candidate_excluded_ids),
                "accepted_projection_cohort_excluded": len(projection_excluded_ids),
                "zh_accepted_projection_cohort_excluded": len(
                    zh_projection_excluded_ids
                ),
                "accepted_distractors": selected_distractor_count,
                "accepted_neutral_candidates": selected_neutral_count,
                "candidate_cardinality_per_target": candidate_cardinality,
                "full_review_candidate_count": len(candidate_review["units"]),
                "full_review_accepted_candidate_count": candidate_review[
                    "accepted_candidate_count"
                ],
                "fresh_candidate_model_reviews": candidate_review[
                    "fresh_model_review_candidate_count"
                ],
                "carried_forward_candidate_reviews": candidate_review[
                    "carried_forward_candidate_count"
                ],
                "candidate_shortfalls": 0,
                "zh_translation_failures": 0,
            },
            "review_contract": {
                "review_complete": True,
                "canonical_freeze_authorized": True,
                "fact_review_complete": True,
                "full_candidate_review_terminal": True,
                "full_candidate_review_all_accept": candidate_review[
                    "all_candidates_accepted"
                ],
                "selected_candidate_projection_all_accept": (
                    True if candidate_projection is not None else None
                ),
                "zh_accepted_projection_applied": (
                    True if zh_accepted_projection is not None else None
                ),
                "distractor_falsehood_and_uniqueness_review_complete": True,
                "neutral_unrelatedness_review_complete": True,
                "zh_translation_equivalence_review_complete": True,
                "proxy_evidence_only": True,
                "human_gold": False,
            },
            "semantic_protocol": {
                "status": STATUS,
                "candidate_protocol_frozen": True,
                "bounded_candidate_adjudication_complete": True,
                "all_record_pairs_enumerated": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "exhaustive_semantic_closure_claimed": False,
            },
            "relation_contract": {
                "relation_partition_status": "provisional",
                "formal_probe_relation_id_assigned": False,
                "relation_partition_promoted_to_probe_relation_id": False,
            },
            "evidence_boundary": {
                "human_gold": False,
                "hf_checkpoint_bound": False,
                "hf_tokenizer_bound": False,
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
            },
        }
        write_json(review_stage, review_manifest)
        review_binding = _binding(
            review_stage,
            schema_version=REVIEW_FREEZE_SCHEMA_VERSION,
            output_path=output_dir / review_stage.name,
        )

        assignments = [
            {
                "base_fact_id": row["base_fact_id"],
                "split_group_id": row["split_group_id"],
                "split_assignment": row["split_assignment"],
            }
            for row in exact_rows
        ]
        split_manifest: Dict[str, Any] = {
            "schema_version": SPLIT_FREEZE_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": STATUS,
            "created_at": created_at,
            "exact_reviewed_bundle": bundle_binding,
            "review_freeze_manifest": review_binding,
            "source_postreview_split_manifest": postreview["artifacts"][
                "split_manifest"
            ]["binding"],
            "source_leakage_components": postreview["artifacts"][
                "leakage_components"
            ]["binding"],
            "accepted_candidate_projection": (
                candidate_projection["manifest_binding"]
                if candidate_projection is not None
                else None
            ),
            "zh_accepted_projection": (
                zh_accepted_projection["manifest_binding"]
                if zh_accepted_projection is not None
                else None
            ),
            "projection_contract": {
                "enabled": candidate_projection is not None,
                "zh_accepted_projection_enabled": zh_accepted_projection is not None,
                "freezes_only_projected_target_cohort": candidate_projection
                is not None,
                "full_postreview_support_preserved_by_binding": True,
                "structural_snapshots_reused_without_recompute": candidate_projection
                is not None,
                "zh_quarantine_exclusion_is_target_only": (
                    True if zh_accepted_projection is not None else None
                ),
                "support_only_donors_are_execution_targets": (
                    False if zh_accepted_projection is not None else None
                ),
                "candidate_cardinality_per_target": candidate_cardinality,
                "human_gold": False,
            },
            "split_status": "frozen",
            "split_policy_version": postreview["split_manifest"][
                "split_policy_version"
            ],
            "split_seed": postreview["split_manifest"].get("split_seed"),
            "base_fact_count": len(exact_rows),
            "original_universe_count": len(universe["source_base_fact_ids"]),
            "fact_stage_cohort_excluded_count": fact_stage_excluded_count,
            "candidate_stage_cohort_excluded_count": len(candidate_excluded_ids),
            "accepted_projection_cohort_excluded_count": len(
                projection_excluded_ids
            ),
            "zh_accepted_projection_cohort_excluded_count": len(
                zh_projection_excluded_ids
            ),
            "total_cohort_excluded_count": len(excluded_ids),
            "leakage_component_count": len(component_ids),
            "source_leakage_component_snapshot_count": len(
                postreview["components"]
            ),
            "split_counts": split_counts,
            "ordered_split_assignments_sha256": sha256_value(assignments),
            "ordered_leakage_component_ids_sha256": sha256_value(component_ids),
            "cross_component_split_violation_count": 0,
            "candidate_shortfall_count": 0,
            "candidate_cardinality_per_target": candidate_cardinality,
            "historical_exposure": {
                "status": "confirmed_by_bound_scope_owner_attestation",
                "historical_exposure_check_complete": True,
                "contract": universe["contract_binding"],
                "scope_owner_attestation": universe["attestation_binding"],
                "attested_by": universe["attested_by"],
                "attested_at": universe["attested_at"],
                "attested_through": universe["attested_through"],
                **exposure_audit,
            },
            "semantic_protocol": {
                "status": STATUS,
                "candidate_protocol_frozen": True,
                "bounded_candidate_adjudication_complete": True,
                "semantic_near_duplicate_recall_guaranteed": False,
                "exhaustive_semantic_closure_claimed": False,
            },
            "freeze_contract": {
                "review_freeze_verified": True,
                "component_atomicity_verified": True,
                "validation_historical_exposure_count": 0,
                "sealed_historical_exposure_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
                "split_freeze_authorized": True,
                "human_gold": False,
            },
        }
        write_json(split_stage, split_manifest)
        split_binding = _binding(
            split_stage,
            schema_version=SPLIT_FREEZE_SCHEMA_VERSION,
            output_path=output_dir / split_stage.name,
        )

        gate_manifest: Dict[str, Any] = {
            "schema_version": GATE_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": STATUS,
            "created_at": created_at,
            "pre_exact_hf_gate_passed": True,
            "blockers": [],
            "exact_reviewed_bundle": bundle_binding,
            "review_freeze_manifest": review_binding,
            "split_freeze_manifest": split_binding,
            "accepted_candidate_projection": (
                candidate_projection["manifest_binding"]
                if candidate_projection is not None
                else None
            ),
            "zh_accepted_projection": (
                zh_accepted_projection["manifest_binding"]
                if zh_accepted_projection is not None
                else None
            ),
            "config": config["binding"],
            "counts": {
                "original_universe": len(universe["source_base_fact_ids"]),
                "retained_frozen": len(exact_rows),
                "cohort_excluded": len(excluded_ids),
                "fact_stage_cohort_excluded": fact_stage_excluded_count,
                "candidate_stage_cohort_excluded": len(candidate_excluded_ids),
                "accepted_projection_cohort_excluded": len(projection_excluded_ids),
                "zh_accepted_projection_cohort_excluded": len(
                    zh_projection_excluded_ids
                ),
                "leakage_components": len(component_ids),
                "distractor_candidates": selected_distractor_count,
                "neutral_candidates": selected_neutral_count,
                "candidate_cardinality_per_target": candidate_cardinality,
                "full_review_candidate_count": len(candidate_review["units"]),
                "full_review_accepted_candidate_count": candidate_review[
                    "accepted_candidate_count"
                ],
                "zh_translation_records": len(zh_review["final_rows"]),
                "fresh_candidate_model_reviews": candidate_review[
                    "fresh_model_review_candidate_count"
                ],
                "carried_forward_candidate_reviews": candidate_review[
                    "carried_forward_candidate_count"
                ],
                "candidate_shortfalls": 0,
                "translation_failures": 0,
                "validation_historical_exposure": 0,
                "sealed_historical_exposure": 0,
            },
            "completed_checks": {
                "formal_universe_v2_lineage_verified": True,
                "fact_review_terminal": True,
                "declared_bounded_semantic_protocol_complete": True,
                "candidate_review_terminal": candidate_review[
                    "all_candidates_terminal"
                ],
                "candidate_review_terminal_and_all_accepted": candidate_review[
                    "all_candidates_accepted"
                ],
                "accepted_projection_candidates_all_accepted": (
                    True if candidate_projection is not None else None
                ),
                "zh_accepted_projection_replayed": (
                    True if zh_accepted_projection is not None else None
                ),
                "candidate_review_evidence_partition_complete": True,
                "candidate_review_provenance_chain_verified": True,
                "zh_translation_review_terminal_and_all_accepted": True,
                "scope_owner_historical_exposure_attestation_verified": True,
                "split_component_atomicity_verified": True,
                "all_sha_record_count_and_id_coverage_verified": True,
            },
            "claim_boundaries": {
                "human_gold": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "exhaustive_semantic_closure_claimed": False,
                "relation_partition_is_formal_probe_relation_id": False,
                "exact_hf_checkpoint_bound": False,
                "exact_hf_tokenizer_bound": False,
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "hidden_state_collection_count": 0,
                "intervention_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
            },
            "authorization": {
                "ready_for_exact_hf_checkpoint_and_tokenizer_binding": True,
                "hf_model_execution_authorized": False,
                "hf_tokenizer_execution_authorized": False,
                "behavior_execution_authorized": False,
                "validation_behavior_exposure_authorized": False,
                "sealed_behavior_exposure_authorized": False,
                "path_not_token_authorized": False,
            },
            "next_required_step": (
                "select_and_immutably_bind_one_exact_HF_checkpoint_and_tokenizer; "
                "rerun the downstream authorization gate before any model execution"
            ),
        }
        write_json(gate_stage, gate_manifest)

        snapshots: Dict[str, str] = {
            str(Path(config_path).resolve()): config["binding"]["sha256"],
            str(universe["manifest_path"]): universe["manifest_binding"]["sha256"],
            str(universe["items_path"]): universe["items_binding"]["sha256"],
            str(universe["contract_path"]): universe["contract_binding"]["sha256"],
            str(universe["attestation_path"]): universe["attestation_binding"]["sha256"],
            str(postreview["manifest_path"]): postreview["manifest_binding"]["sha256"],
            str(candidate_review["manifest_path"]): candidate_review["manifest_binding"]["sha256"],
            str(zh_review["manifest_path"]): zh_review["manifest_binding"]["sha256"],
        }
        snapshot_manifests = [
            (universe["manifest"], universe["manifest_path"]),
            (read_json(universe["contract_path"]), universe["contract_path"]),
            (postreview["manifest"], postreview["manifest_path"]),
            (candidate_review["manifest"], candidate_review["manifest_path"]),
            (zh_review["manifest"], zh_review["manifest_path"]),
        ]
        if candidate_projection is not None:
            snapshots[str(candidate_projection["manifest_path"])] = (
                candidate_projection["manifest_binding"]["sha256"]
            )
            snapshot_manifests.append(
                (
                    candidate_projection["manifest"],
                    candidate_projection["manifest_path"],
                )
            )
        if zh_accepted_projection is not None:
            snapshots[str(zh_accepted_projection["manifest_path"])] = (
                zh_accepted_projection["manifest_binding"]["sha256"]
            )
            snapshots[str(zh_review["parent_manifest_path"])] = (
                zh_review["parent_manifest_binding"]["sha256"]
            )
            snapshot_manifests.extend(
                [
                    (
                        zh_accepted_projection["manifest"],
                        zh_accepted_projection["manifest_path"],
                    ),
                    (zh_review["parent_manifest"], zh_review["parent_manifest_path"]),
                ]
            )
        if postreview.get("candidate_resolution") is not None:
            snapshots[str(postreview["candidate_resolution"]["manifest_path"])] = (
                postreview["candidate_resolution"]["manifest_binding"]["sha256"]
            )
            snapshot_manifests.append(
                (
                    postreview["candidate_resolution"]["manifest"],
                    postreview["candidate_resolution"]["manifest_path"],
                )
            )
        for manifest_value, owner in snapshot_manifests:
            _collect_bound_snapshots(manifest_value, owner_path=owner, output=snapshots)
        for binding in postreview["semantic_runtime"].get(
            "snapshot_bindings", []
        ):
            _collect_bound_snapshots(
                binding,
                owner_path=postreview["semantic_runtime"]["manifest_path"],
                output=snapshots,
            )
        _assert_snapshots_current(snapshots)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
        _publish_directory_no_replace(staged_dir, output_dir)
        published = True
    finally:
        if not published:
            shutil.rmtree(staged_dir, ignore_errors=True)

    return {
        "status": STATUS,
        "pre_exact_hf_gate_passed": True,
        "output_dir": str(output_dir),
        "exact_reviewed_bundle_path": str(output_dir / "exact_reviewed_bundle.jsonl"),
        "review_freeze_manifest_path": str(output_dir / "review_freeze_manifest.json"),
        "split_freeze_manifest_path": str(output_dir / "split_freeze_manifest.json"),
        "pre_exact_hf_gate_manifest_path": str(
            output_dir / "pre_exact_hf_gate_manifest.json"
        ),
        "original_universe_count": len(universe["source_base_fact_ids"]),
        "retained_frozen_count": len(exact_rows),
        "cohort_excluded_count": len(excluded_ids),
        "accepted_candidate_projection_used": candidate_projection is not None,
        "zh_accepted_projection_used": zh_accepted_projection is not None,
        "zh_target_only_excluded_count": len(zh_projection_excluded_ids),
        "support_only_donor_count": (
            len(zh_accepted_projection["support_only_donor_ids"])
            if zh_accepted_projection is not None
            else 0
        ),
        "candidate_cardinality_per_target": candidate_cardinality,
        "semantic_near_duplicate_recall_guaranteed": False,
        "human_gold": False,
        "exact_hf_checkpoint_bound": False,
        "exact_hf_tokenizer_bound": False,
        "behavior_execution_count": 0,
        "validation_behavior_exposure_count": 0,
        "sealed_behavior_exposure_count": 0,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--postreview-rebuild-manifest", type=Path, required=True)
    result.add_argument("--candidate-review-manifest", type=Path, required=True)
    result.add_argument("--candidate-adjudications", type=Path, required=True)
    result.add_argument("--accepted-candidate-projection-manifest", type=Path)
    result.add_argument("--zh-accepted-projection-manifest", type=Path)
    result.add_argument("--zh-review-manifest", type=Path, required=True)
    result.add_argument("--formal-universe-v2-manifest", type=Path, required=True)
    result.add_argument("--historical-exposure-contract", type=Path, required=True)
    result.add_argument("--scope-owner-attestation", type=Path, required=True)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--expected-original-universe-count", type=int, default=8969)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = freeze(
            postreview_rebuild_manifest_path=args.postreview_rebuild_manifest,
            candidate_review_manifest_path=args.candidate_review_manifest,
            candidate_adjudications_path=args.candidate_adjudications,
            accepted_candidate_projection_manifest_path=(
                args.accepted_candidate_projection_manifest
            ),
            zh_accepted_projection_manifest_path=(
                args.zh_accepted_projection_manifest
            ),
            zh_review_manifest_path=args.zh_review_manifest,
            formal_universe_v2_manifest_path=args.formal_universe_v2_manifest,
            historical_exposure_contract_path=args.historical_exposure_contract,
            scope_owner_attestation_path=args.scope_owner_attestation,
            output_dir=args.output_dir,
            config_path=args.config,
            expected_original_universe_count=args.expected_original_universe_count,
        )
    except (FreezeValidationError, FileNotFoundError, FileExistsError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked_fail_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "pre_exact_hf_gate_passed": False,
                    "review_freeze_emitted": False,
                    "split_freeze_emitted": False,
                    "exact_hf_checkpoint_bound": False,
                    "exact_hf_tokenizer_bound": False,
                    "behavior_execution_count": 0,
                    "validation_behavior_exposure_count": 0,
                    "sealed_behavior_exposure_count": 0,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
