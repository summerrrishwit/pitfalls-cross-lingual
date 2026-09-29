#!/usr/bin/env python3
"""Run SHA-bound EN->zh translation and independent proxy review for the full pool.

This runner is deliberately translation-only.  It does not call an HF model,
run behavior evaluation, expose a holdout to the target model, freeze a split,
or turn proxy review into human-gold evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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


TOOL_VERSION = "full-public-benchmark-zh-review-runner-v5"
MANIFEST_SCHEMA = "public-benchmark-full-zh-review-run-manifest-v3"
ACCEPTED_PROJECTION_MANIFEST_SCHEMA = (
    "public-benchmark-full-accepted-candidate-projection-manifest-v1"
)
ACCEPTED_PROJECTION_STATUS = (
    "accepted_only_min1_per_kind_materialized_provisional_offline"
)
POSTREVIEW_MANIFEST_SCHEMA = "public-benchmark-full-postreview-rebuild-manifest-v2"
POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA = (
    "public-benchmark-full-postreview-rebuild-manifest-v3"
)
SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS = frozenset(
    {POSTREVIEW_MANIFEST_SCHEMA, POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA}
)
POSTREVIEW_STATUS = "postreview_rebuilt_bounded_semantic_recall_not_frozen"
GENERATION_SCHEMA = "public-benchmark-zh-translation-generation-v1"
REVIEW_SCHEMA = "public-benchmark-zh-translation-independent-review-v1"
FINAL_SCHEMA = "public-benchmark-zh-translation-reviewed-record-v1"
QUARANTINE_SCHEMA = "public-benchmark-zh-translation-quarantine-v1"
PROMPT_VERSION = "public-benchmark-full-en-zh-translation-v1"
REVIEW_PROMPT_VERSION = "public-benchmark-full-en-zh-equivalence-review-v1"
REPAIR_PROMPT_VERSION = "public-benchmark-full-en-zh-translation-repair-v1"
REPAIR_REVIEW_PROMPT_VERSION = (
    "public-benchmark-full-en-zh-equivalence-repair-review-v1"
)

DEFAULT_GENERATOR_ROUTE = "qwen3.8-max-0902/bailian/bailian"
DEFAULT_GENERATOR_RESPONSE_MODEL = "qwen3.8-max"
DEFAULT_REVIEWER_MODEL = "gpt-5.5"
DEFAULT_REVIEWER_RESPONSE_MODEL = "gpt-5.5-2026-04-24"
REGISTERED_TARGET_LANGUAGES = (
    "am",
    "ar",
    "bn",
    "de",
    "es",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "sw",
    "uk",
    "yo",
    "zh",
    "zu",
)
OTHER_TARGET_LANGUAGES = tuple(
    language for language in REGISTERED_TARGET_LANGUAGES if language != "zh"
)
MODEL_SPEC_OUTPUT_FIELDS = (
    "provider_profile",
    "model",
    "temperature",
    "max_output_tokens",
    "reasoning_effort",
    "disable_thinking",
    "json_mode",
    "timeout_seconds",
    "max_retries",
)
SCALAR_REVIEW_FIELDS = ("prompt", "canonical_fact", "answer")
LIST_REVIEW_FIELDS = ("answer_aliases", "distractors", "neutral_candidates")


def _load_semantic_materializer() -> Any:
    path = PROJECT_ROOT / "scripts" / "materialize_full_public_benchmark_semantic_closure.py"
    spec = importlib.util.spec_from_file_location(
        "_full_zh_resolution_semantic_materializer", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load successor-cohort validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_candidate_resolution_tool() -> Any:
    path = (
        PROJECT_ROOT
        / "scripts"
        / "resolve_full_public_benchmark_candidate_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_full_zh_candidate_resolution", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load candidate-resolution validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_accepted_projection_tool() -> Any:
    path = (
        PROJECT_ROOT
        / "scripts"
        / "materialize_full_public_benchmark_accepted_projection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_full_zh_accepted_candidate_projection", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load accepted-projection validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints() -> Dict[str, str]:
    """Bind resume to the runner, router, and redacted route-binding code."""

    return {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "model_router_runtime_sha256": sha256_file(Path(runtime.__file__).resolve()),
        "route_identity_helper_sha256": sha256_file(
            PROJECT_ROOT / "factual_pitfalls" / "route_identity.py"
        ),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path, *, missing_ok: bool = False) -> List[Dict[str, Any]]:
    if missing_ok and not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_jsonl_journal(path: Path) -> List[Dict[str, Any]]:
    """Read durable journal entries, ignoring only an incomplete final line."""
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload.endswith(b"\n"):
        complete = payload
    else:
        final_newline = payload.rfind(b"\n")
        complete = payload[: final_newline + 1] if final_newline >= 0 else b""
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(complete.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid journal JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected journal JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(
        canonical_json(dict(row)).encode("utf-8") + b"\n" for row in rows
    )
    _atomic_write(path, payload)


def append_jsonl_journal(path: Path, row: Mapping[str, Any]) -> None:
    """Append one complete record durably; this function has exactly one caller thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json(dict(row)).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _safe_model_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    output = {
        field: spec[field]
        for field in MODEL_SPEC_OUTPUT_FIELDS
        if field in spec and spec[field] is not None
    }
    for field in ("provider_profile", "model"):
        if not isinstance(output.get(field), str) or not output[field].strip():
            raise ValueError(f"model {field} must be a non-empty string")
    return output


def _safe_attempts(value: Any) -> List[Dict[str, Any]]:
    allowed = {
        "attempt",
        "status",
        "latency_ms",
        "error_type",
        "http_status",
        "provider_type",
        "provider_code",
        "provider_param",
    }
    if not isinstance(value, list):
        return []
    return [
        {key: entry[key] for key in sorted(set(entry).intersection(allowed))}
        for entry in value
        if isinstance(entry, dict)
    ]


def _safe_usage(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: number
        for key, number in value.items()
        if key in {"input_tokens", "output_tokens", "total_tokens"}
        and isinstance(number, int)
    }


def load_config(path: Path) -> Dict[str, Any]:
    config = read_json(path)
    parent = config.get("extends")
    if not parent:
        return config
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = (PROJECT_ROOT / parent_path).resolve()
    return _deep_merge(load_config(parent_path), config)


def artifact_binding(
    path: Path,
    *,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = path.resolve()
    result: Dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
    }
    if rows is not None:
        result["record_count"] = len(rows)
        schemas = sorted(
            {
                str(row.get("schema_version"))
                for row in rows
                if row.get("schema_version") is not None
            }
        )
        result["schema_versions"] = schemas
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _require_unique(rows: Sequence[Mapping[str, Any]], key: str, label: str) -> None:
    values = [str(row.get(key) or "") for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{label} contains a missing {key}")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate {key}: {duplicates[0]}")


def _verify_manifest_file_binding(
    manifest_binding: Mapping[str, Any], actual: Mapping[str, Any], label: str
) -> None:
    expected_sha = str(manifest_binding.get("sha256") or "")
    if not expected_sha or expected_sha != actual["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch")
    expected_bytes = manifest_binding.get("byte_count")
    if expected_bytes is not None and int(expected_bytes) != int(actual["byte_count"]):
        raise ValueError(f"{label} byte_count mismatch")
    expected_count = manifest_binding.get("record_count")
    if expected_count is not None and int(expected_count) != int(actual["record_count"]):
        raise ValueError(f"{label} record_count mismatch")


def _resolved_manifest_path(
    binding: Mapping[str, Any], *, owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = owner_path.parent / path
    return path.resolve()


def _verify_supplied_output_binding(
    binding: Any,
    *,
    owner_path: Path,
    supplied_path: Path,
    label: str,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    schema_version: Optional[str] = None,
) -> None:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    bound_path = _resolved_manifest_path(binding, owner_path=owner_path, label=label)
    supplied_path = Path(supplied_path).resolve()
    if bound_path != supplied_path:
        raise ValueError(f"{label} path differs from supplied input")
    actual = artifact_binding(
        supplied_path, rows=rows, schema_version=schema_version
    )
    _verify_manifest_file_binding(binding, actual, label)
    if schema_version is not None and binding.get("schema_version") != schema_version:
        raise ValueError(f"{label} schema_version mismatch")


def _validate_accepted_projection(
    *,
    projection_path: Path,
    projection: Mapping[str, Any],
    base_rows: Sequence[Mapping[str, Any]],
    distractor_rows: Sequence[Mapping[str, Any]],
    neutral_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Dict[str, Any]:
    """Validate that projected candidates have immutable completed accept evidence."""

    if projection.get("schema_version") != ACCEPTED_PROJECTION_MANIFEST_SCHEMA:
        raise ValueError("accepted candidate projection schema_version is unsupported")
    if projection.get("status") != ACCEPTED_PROJECTION_STATUS:
        raise ValueError("accepted candidate projection is not complete")

    selection = projection.get("selection_contract")
    if not isinstance(selection, dict) or not all(
        (
            selection.get("cardinality_per_target")
            == {"distractor": 1, "neutral": 1},
            selection.get("accepted_candidates_only") is True,
            selection.get("source_review_full_set_complete") is True,
            selection.get("selected_projection_all_accept") is True,
            isinstance(selection.get("source_review_all_accept"), bool),
            selection.get("donor_must_be_in_projection_target_cohort") is True,
            selection.get("fixed_point_mode")
            == "monotone_simultaneous_target_removal_until_stable",
            selection.get("candidate_rows_rewritten") is False,
            selection.get("human_gold") is False,
        )
    ):
        raise ValueError("accepted candidate projection selection contract is invalid")

    inputs = projection.get("inputs")
    outputs = projection.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("accepted candidate projection bindings are missing")

    candidate_manifest_binding = inputs.get("candidate_review_manifest")
    adjudications_binding = inputs.get("candidate_review_adjudications")
    postreview_binding = inputs.get("postreview_manifest")
    for label, binding in (
        ("candidate review manifest", candidate_manifest_binding),
        ("candidate review adjudications", adjudications_binding),
        ("postreview manifest", postreview_binding),
    ):
        if not isinstance(binding, dict):
            raise ValueError(f"accepted candidate projection {label} binding is missing")

    candidate_manifest_path = _resolved_manifest_path(
        candidate_manifest_binding,
        owner_path=projection_path,
        label="accepted candidate projection candidate review manifest",
    )
    candidate_manifest = read_json(candidate_manifest_path)
    _verify_supplied_output_binding(
        candidate_manifest_binding,
        owner_path=projection_path,
        supplied_path=candidate_manifest_path,
        label="accepted candidate projection candidate review manifest",
        schema_version="public-benchmark-full-candidate-review-run-manifest-v1",
    )
    if not all(
        (
            candidate_manifest.get("status") == "completed",
            candidate_manifest.get("selection_is_full_candidate_set") is True,
            candidate_manifest.get("candidate_review_complete_for_selected_scope")
            is True,
            candidate_manifest.get("candidate_review_complete_for_full_set") is True,
            candidate_manifest.get("full_evidence_partition_complete") is True,
        )
    ):
        raise ValueError(
            "accepted candidate projection source candidate review is not completed "
            "for its full candidate set"
        )
    selected_count = candidate_manifest.get("selected_candidate_count")
    full_count = candidate_manifest.get("full_candidate_count")
    if (
        type(selected_count) is not int
        or type(full_count) is not int
        or selected_count != full_count
    ):
        raise ValueError("accepted candidate projection source review counts are invalid")

    adjudications_path = _resolved_manifest_path(
        adjudications_binding,
        owner_path=projection_path,
        label="accepted candidate projection candidate review adjudications",
    )
    adjudications = read_jsonl(adjudications_path)
    adjudication_schema = str(adjudications_binding.get("schema_version") or "")
    if adjudication_schema != "public-benchmark-full-candidate-adjudication-v3":
        raise ValueError("accepted candidate projection adjudication schema is unsupported")
    _verify_supplied_output_binding(
        adjudications_binding,
        owner_path=projection_path,
        supplied_path=adjudications_path,
        label="accepted candidate projection candidate review adjudications",
        rows=adjudications,
        schema_version=adjudication_schema,
    )
    artifacts = candidate_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source candidate review artifacts are missing")
    _verify_supplied_output_binding(
        artifacts.get("adjudications"),
        owner_path=candidate_manifest_path,
        supplied_path=adjudications_path,
        label="source candidate review adjudications",
        rows=adjudications,
        schema_version=adjudication_schema,
    )
    if len(adjudications) != selected_count:
        raise ValueError("source candidate review adjudication count mismatch")
    _require_unique(adjudications, "candidate_id", "candidate review adjudications")
    adjudications_by_id = {
        str(row["candidate_id"]): dict(row) for row in adjudications
    }

    postreview_path = _resolved_manifest_path(
        postreview_binding,
        owner_path=projection_path,
        label="accepted candidate projection postreview manifest",
    )
    postreview = read_json(postreview_path)
    postreview_schema = str(postreview_binding.get("schema_version") or "")
    if postreview_schema not in SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS:
        raise ValueError("accepted candidate projection postreview schema is unsupported")
    _verify_supplied_output_binding(
        postreview_binding,
        owner_path=projection_path,
        supplied_path=postreview_path,
        label="accepted candidate projection postreview manifest",
        schema_version=postreview_schema,
    )
    if postreview.get("status") != POSTREVIEW_STATUS:
        raise ValueError("accepted candidate projection postreview manifest is incomplete")
    run_contract = candidate_manifest.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("source candidate review run contract is missing")
    source_candidate_manifest = run_contract.get("candidate_manifest")
    if not isinstance(source_candidate_manifest, dict):
        raise ValueError("source candidate review postreview binding is missing")
    _verify_supplied_output_binding(
        source_candidate_manifest,
        owner_path=candidate_manifest_path,
        supplied_path=postreview_path,
        label="source candidate review postreview manifest",
        schema_version=postreview_schema,
    )

    for role, rows in (
        ("full_base_facts", base_rows),
        ("distractor_candidates", distractor_rows),
        ("neutral_reference_candidates", neutral_rows),
    ):
        path_role = "neutral_candidates" if role == "neutral_reference_candidates" else role
        _verify_supplied_output_binding(
            outputs.get(role),
            owner_path=projection_path,
            supplied_path=paths[path_role],
            label=f"accepted candidate projection {role}",
            rows=rows,
        )
    _verify_supplied_output_binding(
        outputs.get("split_manifest"),
        owner_path=projection_path,
        supplied_path=paths["split_manifest"],
        label="accepted candidate projection split_manifest",
        schema_version=str(split_manifest.get("schema_version") or ""),
    )

    base_ids = [str(row["base_fact_id"]) for row in base_rows]
    base_id_set = set(base_ids)
    projected_candidate_ids: List[str] = []
    for kind, rows, id_field in (
        ("distractor", distractor_rows, "distractor_id"),
        ("neutral", neutral_rows, "neutral_candidate_id"),
    ):
        counts = Counter(str(row.get("base_fact_id") or "") for row in rows)
        if set(counts) != base_id_set or any(count != 1 for count in counts.values()):
            raise ValueError(
                f"accepted candidate projection must contain exactly one {kind} per target"
            )
        for row in rows:
            candidate_id = str(row.get(id_field) or "")
            evidence = adjudications_by_id.get(candidate_id)
            if not candidate_id or evidence is None:
                raise ValueError(
                    f"accepted candidate projection lacks review evidence: {candidate_id}"
                )
            if not all(
                (
                    evidence.get("overall_decision") == "accept",
                    evidence.get("candidate_kind") == kind,
                    evidence.get("target_base_fact_id") == row.get("base_fact_id"),
                    evidence.get("candidate_row_sha256") == sha256_value(row),
                    evidence.get("human_gold") is False,
                    evidence.get("target_behavior_consumed") is False,
                )
            ):
                raise ValueError(
                    f"accepted candidate projection has invalid accept evidence: {candidate_id}"
                )
            projected_candidate_ids.append(candidate_id)

    counts = projection.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("accepted candidate projection counts are missing")
    expected_counts = {
        "projected_base_facts": len(base_rows),
        "selected_distractor_candidates": len(distractor_rows),
        "selected_neutral_reference_candidates": len(neutral_rows),
        "selected_candidate_total": len(projected_candidate_ids),
    }
    for field, expected in expected_counts.items():
        if field in counts and counts.get(field) != expected:
            raise ValueError(f"accepted candidate projection {field} mismatch")

    projection_binding = artifact_binding(
        projection_path,
        schema_version=ACCEPTED_PROJECTION_MANIFEST_SCHEMA,
    )
    return {
        "projection_id": projection.get("projection_id"),
        "status": projection.get("status"),
        "selection_is_complete_accepted_projection": True,
        "projected_record_count": len(base_rows),
        "source_candidate_count": full_count,
        "projected_candidate_count": len(projected_candidate_ids),
        "ordered_projected_base_fact_ids_sha256": sha256_value(base_ids),
        "projected_candidate_ids_sha256": sha256_value(sorted(projected_candidate_ids)),
        "source_review_all_accept": selection["source_review_all_accept"],
        "bindings": {
            "accepted_candidate_projection_manifest": projection_binding,
            "projection_source_postreview_manifest": artifact_binding(
                postreview_path, schema_version=postreview_schema
            ),
            "projection_source_candidate_review_manifest": artifact_binding(
                candidate_manifest_path,
                schema_version="public-benchmark-full-candidate-review-run-manifest-v1",
            ),
            "projection_source_candidate_review_adjudications": artifact_binding(
                adjudications_path,
                rows=adjudications,
                schema_version=adjudication_schema,
            ),
        },
    }


def _load_accepted_projection_context(
    *,
    projection_manifest_path: Path,
    supplied_full_base_facts_path: Optional[Path],
    supplied_distractor_candidates_path: Optional[Path],
    supplied_neutral_candidates_path: Optional[Path],
    supplied_split_manifest_path: Optional[Path],
) -> Dict[str, Any]:
    """Replay and validate the projection with its authoritative producer."""

    projection_tool = _load_accepted_projection_tool()
    context = projection_tool.load_projection_context(
        projection_manifest_path=Path(projection_manifest_path).resolve()
    )
    manifest = context["manifest"]
    if (
        manifest.get("schema_version") != ACCEPTED_PROJECTION_MANIFEST_SCHEMA
        or manifest.get("status") != ACCEPTED_PROJECTION_STATUS
    ):
        raise ValueError("accepted candidate projection contract is unsupported")
    supplied = {
        "full_base_facts": supplied_full_base_facts_path,
        "distractor_candidates": supplied_distractor_candidates_path,
        "neutral_reference_candidates": supplied_neutral_candidates_path,
        "split_manifest": supplied_split_manifest_path,
    }
    for role, supplied_path in supplied.items():
        if supplied_path is not None and Path(supplied_path).resolve() != context[
            "output_paths"
        ][role]:
            raise ValueError(
                f"accepted candidate projection {role} path differs from supplied input"
            )
    _validate_accepted_projection(
        projection_path=context["manifest_path"],
        projection=manifest,
        base_rows=context["full_base_facts"],
        distractor_rows=context["distractor_candidates"],
        neutral_rows=context["neutral_reference_candidates"],
        split_manifest=context["split_manifest"],
        paths={
            "full_base_facts": context["output_paths"]["full_base_facts"],
            "distractor_candidates": context["output_paths"][
                "distractor_candidates"
            ],
            "neutral_candidates": context["output_paths"][
                "neutral_reference_candidates"
            ],
            "split_manifest": context["output_paths"]["split_manifest"],
        },
    )
    return context


def _validate_successor_chain(
    *,
    formal_manifest: Mapping[str, Any],
    universe_ids: Sequence[str],
    base_rows: Sequence[Mapping[str, Any]],
    distractor_rows: Sequence[Mapping[str, Any]],
    neutral_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    fact_resolution_manifest_path: Path,
    postreview_rebuild_manifest_path: Path,
) -> Dict[str, Any]:
    semantic_materializer = _load_semantic_materializer()
    resolution = semantic_materializer.load_fact_resolution_context(
        fact_resolution_manifest_path=fact_resolution_manifest_path
    )
    if resolution["source_ids"] != list(universe_ids):
        raise ValueError(
            "Fact-resolution source universe differs from formal universe v2"
        )

    postreview_path = Path(postreview_rebuild_manifest_path).resolve()
    postreview = read_json(postreview_path)
    postreview_schema = postreview.get("schema_version")
    if postreview_schema not in SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS:
        raise ValueError("postreview rebuild manifest schema_version is unsupported")
    if postreview.get("status") != POSTREVIEW_STATUS:
        raise ValueError("postreview rebuild manifest is not complete")
    fact_contract = postreview.get("fact_review_contract")
    if not isinstance(fact_contract, dict) or not all(
        (
            fact_contract.get("mode") == "fact_resolution_successor_cohort",
            fact_contract.get("selection_is_complete_retained_cohort") is True,
            fact_contract.get("source_universe_base_fact_count")
            == len(universe_ids),
        )
    ):
        raise ValueError("postreview successor-cohort contract is invalid")
    post_inputs = postreview.get("inputs")
    if not isinstance(post_inputs, dict) or post_inputs.get(
        "fact_resolution"
    ) != resolution["bindings"]:
        raise ValueError("postreview fact-resolution binding is stale")
    candidate_resolution = None
    candidate_excluded_ids: List[str] = []
    candidate_repair = post_inputs.get("candidate_repair")
    if candidate_repair is not None:
        if postreview_schema != POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA:
            raise ValueError(
                "candidate-repaired postreview must use successor manifest schema v3"
            )
        if not isinstance(candidate_repair, dict):
            raise ValueError("postreview candidate-repair bindings are invalid")
        for label in (
            "predecessor_postreview_manifest",
            "root_postreview_manifest",
        ):
            binding = candidate_repair.get(label)
            if not isinstance(binding, dict) or binding.get(
                "schema_version"
            ) not in SUPPORTED_POSTREVIEW_MANIFEST_SCHEMAS:
                raise ValueError(f"postreview {label} schema binding is unsupported")
        candidate_tool = _load_candidate_resolution_tool()
        candidate_resolution_path = _resolved_manifest_path(
            candidate_repair.get("candidate_resolution_manifest"),
            owner_path=postreview_path,
            label="candidate resolution manifest",
        )
        predecessor_path = _resolved_manifest_path(
            candidate_repair.get("predecessor_postreview_manifest"),
            owner_path=postreview_path,
            label="candidate repair predecessor postreview",
        )
        candidate_resolution = candidate_tool.load_resolution_context(
            resolution_manifest_path=candidate_resolution_path,
            predecessor_postreview_manifest_path=predecessor_path,
        )
        _verify_supplied_output_binding(
            candidate_repair.get("candidate_resolution_manifest"),
            owner_path=postreview_path,
            supplied_path=candidate_resolution["manifest_path"],
            label="postreview candidate-resolution manifest",
            schema_version=str(
                candidate_repair["candidate_resolution_manifest"].get(
                    "schema_version"
                )
            ),
        )
        _verify_supplied_output_binding(
            candidate_repair.get("predecessor_postreview_manifest"),
            owner_path=postreview_path,
            supplied_path=candidate_resolution["predecessor_path"],
            label="postreview candidate-repair predecessor",
            schema_version=str(
                candidate_repair["predecessor_postreview_manifest"].get(
                    "schema_version"
                )
            ),
        )
        _verify_supplied_output_binding(
            candidate_repair.get("root_postreview_manifest"),
            owner_path=postreview_path,
            supplied_path=candidate_resolution["root_path"],
            label="postreview candidate-repair root",
            schema_version=str(
                candidate_repair["root_postreview_manifest"].get("schema_version")
            ),
        )
        _verify_supplied_output_binding(
            candidate_repair.get("candidate_pair_exclusions"),
            owner_path=postreview_path,
            supplied_path=candidate_resolution["pair_path"],
            label="postreview candidate pair exclusions",
            rows=candidate_resolution["pair_rows"],
            schema_version=candidate_tool.PAIR_EXCLUSION_SCHEMA,
        )
        _verify_supplied_output_binding(
            candidate_repair.get("candidate_cohort_exclusions"),
            owner_path=postreview_path,
            supplied_path=candidate_resolution["cohort_exclusion_path"],
            label="postreview candidate cohort exclusions",
            rows=candidate_resolution["cohort_exclusion_rows"],
            schema_version=candidate_tool.COHORT_EXCLUSION_SCHEMA,
        )
        candidate_excluded_ids = list(
            candidate_resolution["candidate_excluded_ids"]
        )
        repair_contract = postreview.get("candidate_repair_contract")
        if not isinstance(repair_contract, dict) or not all(
            (
                repair_contract.get("resolution_id")
                == candidate_resolution["resolution_id"],
                repair_contract.get("resolution_applied") is True,
                repair_contract.get("pair_exclusion_count")
                == len(candidate_resolution["pair_rows"]),
                repair_contract.get("candidate_stage_cohort_exclusion_count")
                == len(candidate_excluded_ids),
                repair_contract.get("fixed_point_reached") is True,
                repair_contract.get("candidate_shortfall_count") == 0,
                repair_contract.get("candidate_policy_relaxed") is False,
                repair_contract.get(
                    "predecessor_candidate_reviews_blanket_valid"
                )
                is False,
                repair_contract.get(
                    "exact_v3_accepted_projection_evidence_carry_forward_eligible"
                )
                is True,
                repair_contract.get(
                    "candidate_id_or_row_sha_alone_sufficient_for_review_reuse"
                )
                is False,
                repair_contract.get("full_candidate_evidence_coverage_required")
                is True,
                repair_contract.get("full_candidate_model_rereview_required")
                is False,
                repair_contract.get(
                    "changed_or_new_candidate_fresh_model_review_required"
                )
                is True,
                repair_contract.get("predecessor_zh_reviews_valid") is False,
                repair_contract.get("full_zh_translation_and_review_required")
                is True,
                repair_contract.get("human_gold") is False,
            )
        ):
            raise ValueError("postreview candidate-repair contract is invalid")
    outputs = postreview.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("postreview output bindings are missing")
    for role, rows in (
        ("full_base_facts", base_rows),
        ("distractor_candidates", distractor_rows),
        ("neutral_reference_candidates", neutral_rows),
    ):
        path_role = (
            "neutral_candidates"
            if role == "neutral_reference_candidates"
            else role
        )
        _verify_supplied_output_binding(
            outputs.get(role),
            owner_path=postreview_path,
            supplied_path=paths[path_role],
            label=f"postreview {role}",
            rows=rows,
        )
    _verify_supplied_output_binding(
        outputs.get("split_manifest"),
        owner_path=postreview_path,
        supplied_path=paths["split_manifest"],
        label="postreview split_manifest",
        schema_version=str(split_manifest.get("schema_version") or ""),
    )

    base_ids = [str(row["base_fact_id"]) for row in base_rows]
    expected_retained_ids = [
        base_fact_id
        for base_fact_id in resolution["retained_ids"]
        if base_fact_id not in set(candidate_excluded_ids)
    ]
    expected_excluded_ids = [
        *resolution["excluded_ids"],
        *candidate_excluded_ids,
    ]
    if set(candidate_excluded_ids).intersection(resolution["excluded_ids"]):
        raise ValueError("candidate-stage exclusions overlap fact-stage exclusions")
    if base_ids != expected_retained_ids:
        raise ValueError("postreview base facts differ from resolved retained cohort")
    lineage = postreview.get("lineage_digests")
    expected_lineage = {
        "ordered_input_base_fact_ids_sha256": sha256_value(resolution["source_ids"]),
        "ordered_retained_base_fact_ids_sha256": sha256_value(base_ids),
        "ordered_rejected_base_fact_ids_sha256": sha256_value(
            expected_excluded_ids
        ),
        "ordered_revised_base_fact_ids_sha256": sha256_value(
            resolution["revised_ids"]
        ),
    }
    if candidate_resolution is not None:
        expected_lineage.update(
            {
                "ordered_fact_stage_retained_base_fact_ids_sha256": sha256_value(
                    resolution["retained_ids"]
                ),
                "ordered_fact_stage_excluded_base_fact_ids_sha256": sha256_value(
                    resolution["excluded_ids"]
                ),
                "ordered_candidate_stage_excluded_base_fact_ids_sha256": sha256_value(
                    candidate_excluded_ids
                ),
            }
        )
    if not isinstance(lineage, dict) or any(
        lineage.get(field) != expected
        for field, expected in expected_lineage.items()
    ):
        raise ValueError("postreview successor lineage digests are stale")
    invalidation = postreview.get("review_invalidation")
    if not isinstance(invalidation, dict) or not all(
        (
            invalidation.get("pre_resolution_translation_reviews_valid") is False,
            invalidation.get("pre_postreview_split_translation_reviews_valid") is False,
            invalidation.get("new_translation_and_translation_review_required") is True,
            (
                candidate_resolution is None
                or invalidation.get(
                    "predecessor_candidate_reviews_blanket_valid"
                )
                is False
            ),
            (
                candidate_resolution is None
                or invalidation.get("predecessor_translation_reviews_valid") is False
            ),
            (
                candidate_resolution is None
                or invalidation.get(
                    "candidate_id_or_row_sha_alone_sufficient_for_review_reuse"
                )
                is False
            ),
            (
                candidate_resolution is None
                or invalidation.get(
                    "exact_v3_accepted_projection_evidence_carry_forward_eligible"
                )
                is True
            ),
            (
                candidate_resolution is None
                or invalidation.get("full_candidate_evidence_coverage_required")
                is True
            ),
            (
                candidate_resolution is None
                or invalidation.get("full_candidate_model_rereview_required")
                is False
            ),
            (
                candidate_resolution is None
                or invalidation.get(
                    "changed_or_new_candidate_fresh_model_review_required"
                )
                is True
            ),
            (
                candidate_resolution is None
                or invalidation.get("new_distractor_review_required") is True
            ),
            (
                candidate_resolution is None
                or invalidation.get("new_neutral_review_required") is True
            ),
        )
    ):
        raise ValueError("postreview translation invalidation contract is missing")
    return {
        "successor_universe_id": resolution["successor_id"],
        "source_formal_universe_id": formal_manifest.get("universe_id"),
        "selection_is_complete_retained_cohort": True,
        "source_record_count": len(resolution["source_ids"]),
        "fact_resolution_retained_record_count": len(resolution["retained_ids"]),
        "retained_record_count": len(expected_retained_ids),
        "revised_record_count": len(resolution["revised_ids"]),
        "fact_resolution_excluded_record_count": len(resolution["excluded_ids"]),
        "candidate_resolution_excluded_record_count": len(candidate_excluded_ids),
        "excluded_record_count": len(expected_excluded_ids),
        "ordered_retained_base_fact_ids_sha256": sha256_value(
            expected_retained_ids
        ),
        "ordered_excluded_base_fact_ids_sha256": sha256_value(
            expected_excluded_ids
        ),
        # The ordered digest above preserves the resolution ledger lineage.
        # The freeze contract deliberately uses a canonical set digest so that
        # the same excluded cohort can be compared across independently
        # ordered artifacts.
        "excluded_base_fact_ids_sha256": sha256_value(
            sorted(expected_excluded_ids)
        ),
        "candidate_resolution_id": (
            candidate_resolution["resolution_id"]
            if candidate_resolution is not None
            else None
        ),
        "bindings": {
            "fact_resolution_manifest": artifact_binding(
                Path(fact_resolution_manifest_path)
            ),
            "postreview_rebuild_manifest": artifact_binding(postreview_path),
            **(
                {
                    "candidate_resolution_manifest": artifact_binding(
                        candidate_resolution["manifest_path"],
                        schema_version=candidate_tool.RESOLUTION_MANIFEST_SCHEMA,
                    )
                }
                if candidate_resolution is not None
                else {}
            ),
        },
    }


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(row.get("split_assignment") or "") for row in rows)
    allowed = {"development", "validation", "sealed"}
    if set(counts) - allowed or "" in counts:
        raise ValueError("full_base_facts contains an invalid split_assignment")
    return {key: counts.get(key, 0) for key in sorted(allowed)}


def load_bound_inputs(
    *,
    formal_universe_manifest_path: Path,
    formal_universe_items_path: Path,
    full_base_facts_path: Optional[Path],
    distractor_candidates_path: Optional[Path],
    neutral_candidates_path: Optional[Path],
    split_manifest_path: Optional[Path],
    fact_resolution_manifest_path: Optional[Path] = None,
    postreview_rebuild_manifest_path: Optional[Path] = None,
    accepted_candidate_projection_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    projection_context = None
    if accepted_candidate_projection_manifest_path is not None:
        if (
            fact_resolution_manifest_path is not None
            or postreview_rebuild_manifest_path is not None
        ):
            raise ValueError(
                "accepted candidate projection cannot be combined with explicit "
                "fact-resolution or postreview manifests"
            )
        projection_context = _load_accepted_projection_context(
            projection_manifest_path=accepted_candidate_projection_manifest_path,
            supplied_full_base_facts_path=full_base_facts_path,
            supplied_distractor_candidates_path=distractor_candidates_path,
            supplied_neutral_candidates_path=neutral_candidates_path,
            supplied_split_manifest_path=split_manifest_path,
        )
        full_base_facts_path = projection_context["output_paths"]["full_base_facts"]
        distractor_candidates_path = projection_context["output_paths"][
            "distractor_candidates"
        ]
        neutral_candidates_path = projection_context["output_paths"][
            "neutral_reference_candidates"
        ]
        split_manifest_path = projection_context["output_paths"]["split_manifest"]
    missing_paths = [
        label
        for label, path in (
            ("full_base_facts", full_base_facts_path),
            ("distractor_candidates", distractor_candidates_path),
            ("neutral_candidates", neutral_candidates_path),
            ("split_manifest", split_manifest_path),
        )
        if path is None
    ]
    if missing_paths:
        raise ValueError(
            "missing required input paths without an accepted candidate projection: "
            + ", ".join(missing_paths)
        )
    paths = {
        "formal_universe_manifest": formal_universe_manifest_path.resolve(),
        "formal_universe_items": formal_universe_items_path.resolve(),
        "full_base_facts": Path(full_base_facts_path).resolve(),
        "distractor_candidates": Path(distractor_candidates_path).resolve(),
        "neutral_candidates": Path(neutral_candidates_path).resolve(),
        "split_manifest": Path(split_manifest_path).resolve(),
    }
    formal_manifest = read_json(paths["formal_universe_manifest"])
    split_manifest = read_json(paths["split_manifest"])
    universe_rows = read_jsonl(paths["formal_universe_items"])
    base_rows = read_jsonl(paths["full_base_facts"])
    distractor_rows = read_jsonl(paths["distractor_candidates"])
    neutral_rows = read_jsonl(paths["neutral_candidates"])

    if formal_manifest.get("schema_version") != "public-benchmark-formal-cohort-universe-v2":
        raise ValueError("formal universe manifest must use v2")
    if formal_manifest.get("selection_policy", {}).get("mode") != "full-pool":
        raise ValueError("formal universe must declare full-pool selection")
    if formal_manifest.get("universe_status") not in {
        "declared_immutable_not_reviewed",
        "reviewed",
        "reviewed_frozen",
    }:
        raise ValueError("unsupported formal universe status")

    _require_unique(universe_rows, "source_base_fact_id", "formal universe items")
    _require_unique(base_rows, "base_fact_id", "full_base_facts")
    _require_unique(distractor_rows, "distractor_id", "distractor candidates")
    _require_unique(neutral_rows, "neutral_candidate_id", "neutral candidates")

    bindings = {
        "formal_universe_manifest": artifact_binding(
            paths["formal_universe_manifest"],
            schema_version=str(formal_manifest["schema_version"]),
        ),
        "formal_universe_items": artifact_binding(
            paths["formal_universe_items"], rows=universe_rows
        ),
        "full_base_facts": artifact_binding(paths["full_base_facts"], rows=base_rows),
        "distractor_candidates": artifact_binding(
            paths["distractor_candidates"], rows=distractor_rows
        ),
        "neutral_candidates": artifact_binding(
            paths["neutral_candidates"], rows=neutral_rows
        ),
        "split_manifest": artifact_binding(
            paths["split_manifest"],
            schema_version=str(split_manifest.get("schema_version") or ""),
        ),
    }
    _verify_manifest_file_binding(
        formal_manifest.get("items", {}),
        bindings["formal_universe_items"],
        "formal universe items",
    )

    universe_ids = [str(row["source_base_fact_id"]) for row in universe_rows]
    base_by_id = {str(row["base_fact_id"]): dict(row) for row in base_rows}
    if int(formal_manifest.get("record_count", -1)) != len(universe_ids):
        raise ValueError("formal universe manifest record_count mismatch")
    has_resolution = fact_resolution_manifest_path is not None
    has_postreview = postreview_rebuild_manifest_path is not None
    if has_resolution != has_postreview:
        raise ValueError(
            "fact-resolution and postreview manifests must be supplied together"
        )
    successor = None
    if has_resolution:
        successor = _validate_successor_chain(
            formal_manifest=formal_manifest,
            universe_ids=universe_ids,
            base_rows=base_rows,
            distractor_rows=distractor_rows,
            neutral_rows=neutral_rows,
            split_manifest=split_manifest,
            paths=paths,
            fact_resolution_manifest_path=Path(fact_resolution_manifest_path),
            postreview_rebuild_manifest_path=Path(postreview_rebuild_manifest_path),
        )
        bindings.update(successor["bindings"])
    elif projection_context is not None:
        projected_ids = projection_context["projected_target_ids"]
        projection_inputs = projection_context["manifest"]["inputs"]
        if [str(row["base_fact_id"]) for row in base_rows] != projected_ids:
            raise ValueError(
                "accepted candidate projection target order differs from full_base_facts"
            )
        if not set(projected_ids).issubset(set(universe_ids)):
            raise ValueError("accepted candidate projection is outside formal universe")
        bindings.update(
            {
                "accepted_candidate_projection_manifest": artifact_binding(
                    projection_context["manifest_path"],
                    schema_version=ACCEPTED_PROJECTION_MANIFEST_SCHEMA,
                ),
                "projection_source_postreview_manifest": artifact_binding(
                    projection_context["postreview_manifest_path"],
                    schema_version=str(
                        projection_inputs["postreview_manifest"]["schema_version"]
                    ),
                ),
                "projection_source_candidate_review_manifest": artifact_binding(
                    projection_context["candidate_review_manifest_path"],
                    schema_version=str(
                        projection_inputs["candidate_review_manifest"][
                            "schema_version"
                        ]
                    ),
                ),
                "projection_source_candidate_review_adjudications": artifact_binding(
                    projection_context["candidate_review_adjudications_path"],
                    rows=read_jsonl(
                        projection_context["candidate_review_adjudications_path"]
                    ),
                    schema_version=str(
                        projection_inputs["candidate_review_adjudications"][
                            "schema_version"
                        ]
                    ),
                ),
            }
        )
    elif set(universe_ids) != set(base_by_id):
        raise ValueError(
            "formal universe IDs differ from full_base_facts IDs without a "
            "validated successor-cohort chain"
        )
    if successor is not None or projection_context is not None:
        if not set(base_by_id).issubset(set(universe_ids)):
            raise ValueError("successor cohort is not a subset of formal universe")
    actual_split_counts = _split_counts(base_rows)
    if projection_context is None:
        if split_manifest.get("base_fact_count") != len(base_rows):
            raise ValueError("split manifest base_fact_count mismatch")
        declared_split_counts = split_manifest.get("actual_base_fact_counts")
        if declared_split_counts != actual_split_counts:
            raise ValueError("split manifest counts differ from full_base_facts")

    base_ids = set(base_by_id)
    for label, rows in (
        ("distractor candidates", distractor_rows),
        ("neutral candidates", neutral_rows),
    ):
        unknown = sorted(
            str(row.get("base_fact_id") or "")
            for row in rows
            if str(row.get("base_fact_id") or "") not in base_ids
        )
        if unknown:
            raise ValueError(f"{label} contain unknown base_fact_id: {unknown[0]}")
        for row in rows:
            base = base_by_id[str(row["base_fact_id"])]
            if row.get("split_assignment") != base.get("split_assignment"):
                raise ValueError(f"{label} split differs from full_base_facts")
            if successor is not None or projection_context is not None:
                donor_id = row.get("source_base_fact_id")
                if not isinstance(donor_id, str) or donor_id not in base_ids:
                    raise ValueError(
                        f"{label} contain excluded or unknown source_base_fact_id: {donor_id}"
                    )
                if base_by_id[donor_id].get("split_assignment") != base.get(
                    "split_assignment"
                ):
                    raise ValueError(f"{label} donor split differs from target fact")

    distractors_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    neutrals_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in distractor_rows:
        distractors_by_id[str(row["base_fact_id"])].append(dict(row))
    for row in neutral_rows:
        neutrals_by_id[str(row["base_fact_id"])].append(dict(row))

    items: List[Dict[str, Any]] = []
    active_universe_rows = [
        row for row in universe_rows if str(row["source_base_fact_id"]) in base_by_id
    ]
    if [str(row["source_base_fact_id"]) for row in active_universe_rows] != [
        str(row["base_fact_id"]) for row in base_rows
    ]:
        raise ValueError("selected full_base_facts does not preserve universe order")
    for universe_row in active_universe_rows:
        base_fact_id = str(universe_row["source_base_fact_id"])
        base = base_by_id[base_fact_id]
        distractors = sorted(
            distractors_by_id.get(base_fact_id, []),
            key=lambda row: (int(row.get("slot") or 0), str(row["distractor_id"])),
        )
        neutrals = sorted(
            neutrals_by_id.get(base_fact_id, []),
            key=lambda row: (
                int(row.get("slot") or 0),
                str(row["neutral_candidate_id"]),
            ),
        )
        if not neutrals:
            raise ValueError(f"no Neutral candidates for {base_fact_id}")
        item = {
            "base_fact_id": base_fact_id,
            "cohort_item_id": universe_row.get("cohort_item_id"),
            "split_assignment": base["split_assignment"],
            "source": {
                "prompt_en": base.get("prompt_en"),
                "canonical_fact_en": base.get("canonical_fact_en"),
                "answer_en": base.get("answer_en"),
                "answer_aliases_en": list(base.get("answer_aliases_en") or []),
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
        for name, value in item["source"].items():
            if name == "answer_aliases_en":
                if not isinstance(value, list) or not all(
                    isinstance(alias, str) and alias.strip() for alias in value
                ):
                    raise ValueError(f"invalid {name}: {base_fact_id}")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid {name}: {base_fact_id}")
        if any(
            not isinstance(row["text_en"], str) or not row["text_en"].strip()
            for row in [*item["distractors"], *item["neutral_candidates"]]
        ):
            raise ValueError(f"empty candidate text: {base_fact_id}")
        item["input_record_sha256"] = sha256_value(item)
        items.append(item)

    candidate_binding = {
        "distractor_candidates": bindings["distractor_candidates"],
        "neutral_candidates": bindings["neutral_candidates"],
    }
    candidate_binding["combined_sha256"] = sha256_value(
        {
            key: value["sha256"]
            for key, value in sorted(candidate_binding.items())
        }
    )
    return {
        "formal_manifest": formal_manifest,
        "split_manifest": split_manifest,
        "bindings": bindings,
        "candidate_binding": candidate_binding,
        "items": items,
        "split_counts": actual_split_counts,
        "successor_cohort": successor,
        "accepted_candidate_projection": (
            {
                "projection_id": projection_context["manifest"].get("projection_id"),
                "status": projection_context["manifest"].get("status"),
                "selection_contract": projection_context["manifest"].get(
                    "selection_contract"
                ),
                "source_review_contract": projection_context["manifest"].get(
                    "source_review_contract"
                ),
                "counts": projection_context["manifest"].get("counts"),
                "lineage_digests": projection_context["manifest"].get(
                    "lineage_digests"
                ),
                "projected_record_count": len(
                    projection_context["projected_target_ids"]
                ),
                "quarantined_record_count": len(projection_context["quarantine"]),
                "selected_candidate_count": len(
                    projection_context["selected_candidate_ids"]
                ),
                "selected_projection_all_accept": True,
                "human_gold": False,
            }
            if projection_context is not None
            else None
        ),
    }


def translation_batch_prompt(
    items: Sequence[Mapping[str, Any]],
    *,
    repairs_by_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    records: List[Dict[str, Any]] = []
    for item in items:
        record: Dict[str, Any] = {
            "base_fact_id": item["base_fact_id"],
            "source": item["source"],
            "distractors": item["distractors"],
            "neutral_candidates": item["neutral_candidates"],
        }
        if repairs_by_id is not None:
            repair = repairs_by_id[str(item["base_fact_id"])]
            record["rejected_translation"] = repair["translation"]
            record["independent_review"] = repair["review"]
        records.append(record)
    payload: Dict[str, Any] = {"records": records}
    instruction = (
        "Translate every English field in this record into natural Simplified Chinese. "
        "Preserve factual meaning, answer identity, incomplete prompt form, alias identity, "
        "distractor identity, and Neutral-context meaning. Translation is not permission to "
        "correct, strengthen, or fact-check the source. Do not put the answer into prompt_zh. "
        "Return one JSON object with a records array. Return exactly one record for every "
        "input base_fact_id. Each record must contain base_fact_id and translation. translation "
        "must contain prompt_zh, canonical_fact_zh, answer_zh, answer_aliases_zh (same length "
        "and order as the English aliases), distractors (objects with exactly distractor_id "
        "and text_zh), and neutral_candidates (objects with exactly neutral_candidate_id and "
        "text_zh)."
    )
    if repairs_by_id is not None:
        instruction = (
            "Repair the rejected Simplified-Chinese translation exactly once. Correct only "
            "the field-level equivalence or naturalness issues identified by the independent "
            "review; do not fact-check or alter the English source. "
        ) + instruction
    return instruction + " Input: " + canonical_json(payload)


def validate_translation_batch(
    value: Dict[str, Any], items: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    if set(value) != {"records"}:
        raise ValueError("invalid_translation_batch_fields")
    records = value.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("invalid_translation_batch_records")
    expected_by_id = {str(item["base_fact_id"]): item for item in items}
    actual_ids = [str(row.get("base_fact_id") or "") for row in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_by_id):
        raise ValueError("translation_batch_base_fact_id_mismatch")
    result: Dict[str, Dict[str, Any]] = {}
    for row in records:
        if set(row) != {"base_fact_id", "translation"}:
            raise ValueError("invalid_translation_batch_record_fields")
        base_fact_id = str(row["base_fact_id"])
        translation = row.get("translation")
        if not isinstance(translation, dict):
            raise ValueError("invalid_translation_batch_record")
        validate_translation(translation, expected_by_id[base_fact_id])
        result[base_fact_id] = translation
    return result


def validate_translation(value: Dict[str, Any], item: Mapping[str, Any]) -> None:
    required_keys = {
        "prompt_zh",
        "canonical_fact_zh",
        "answer_zh",
        "answer_aliases_zh",
        "distractors",
        "neutral_candidates",
    }
    for key in required_keys:
        if key not in value:
            raise ValueError(f"missing_{key}")
    if set(value) != required_keys:
        raise ValueError("invalid_translation_fields")
    for key in ("prompt_zh", "canonical_fact_zh", "answer_zh"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"invalid_{key}")
    aliases = value["answer_aliases_zh"]
    if not isinstance(aliases, list) or len(aliases) != len(
        item["source"]["answer_aliases_en"]
    ):
        raise ValueError("answer_alias_translation_count_mismatch")
    if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
        raise ValueError("invalid_answer_aliases_zh")
    _validate_translated_candidates(
        value["distractors"], item["distractors"], "distractor_id", "distractors"
    )
    _validate_translated_candidates(
        value["neutral_candidates"],
        item["neutral_candidates"],
        "neutral_candidate_id",
        "neutral_candidates",
    )


def _validate_translated_candidates(
    actual: Any,
    expected: Sequence[Mapping[str, Any]],
    id_key: str,
    label: str,
) -> None:
    if not isinstance(actual, list) or any(not isinstance(row, dict) for row in actual):
        raise ValueError(f"invalid_{label}")
    if any(set(row) != {id_key, "text_zh"} for row in actual):
        raise ValueError(f"invalid_{label}_fields")
    actual_ids = [str(row.get(id_key) or "") for row in actual]
    expected_ids = [str(row[id_key]) for row in expected]
    if actual_ids != expected_ids:
        raise ValueError(f"{label}_id_or_order_mismatch")
    if any(not isinstance(row.get("text_zh"), str) or not row["text_zh"].strip() for row in actual):
        raise ValueError(f"invalid_{label}_translation")


def review_batch_prompt(
    items: Sequence[Mapping[str, Any]],
    translations_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = {
        "records": [
            {
                "base_fact_id": item["base_fact_id"],
                "source": item["source"],
                "distractors": item["distractors"],
                "neutral_candidates": item["neutral_candidates"],
                "translation": translations_by_id[str(item["base_fact_id"])],
            }
            for item in items
        ]
    }
    return (
        "Independently review this English-to-Simplified-Chinese translation field by field. "
        "Judge translation equivalence and natural Chinese only; do not certify factual truth, "
        "distractor falsehood, or Neutral unrelatedness. Return one JSON object with a records "
        "array and exactly one record for every input base_fact_id. Each record must contain "
        "base_fact_id and review. review contains decision (accept|reject), issues (array of "
        "strings), and checks. checks must contain prompt, "
        "canonical_fact, and answer objects; each has translation_equivalent, natural_zh "
        "(booleans), and issues (array). checks.answer_aliases must contain one object per alias "
        "with alias_index, translation_equivalent, natural_zh, issues. checks.distractors must "
        "contain one object per distractor with distractor_id and those same three review fields. "
        "checks.neutral_candidates must contain one object per Neutral candidate with "
        "neutral_candidate_id and those same three review fields. Accept if and only if every "
        "translation_equivalent and natural_zh value is true. Input: "
        + canonical_json(payload)
    )


def validate_review_batch(
    value: Dict[str, Any], items: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    if set(value) != {"records"}:
        raise ValueError("invalid_review_batch_fields")
    records = value.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("invalid_review_batch_records")
    expected_by_id = {str(item["base_fact_id"]): item for item in items}
    actual_ids = [str(row.get("base_fact_id") or "") for row in records]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_by_id):
        raise ValueError("review_batch_base_fact_id_mismatch")
    result: Dict[str, Dict[str, Any]] = {}
    for row in records:
        if set(row) != {"base_fact_id", "review"}:
            raise ValueError("invalid_review_batch_record_fields")
        base_fact_id = str(row["base_fact_id"])
        review = row.get("review")
        if not isinstance(review, dict):
            raise ValueError("invalid_review_batch_record")
        validate_review(review, expected_by_id[base_fact_id])
        result[base_fact_id] = review
    return result


def _validate_check(value: Any, label: str) -> bool:
    if not isinstance(value, dict):
        raise ValueError(f"invalid_{label}_check")
    if set(value) != {"translation_equivalent", "natural_zh", "issues"}:
        raise ValueError(f"invalid_{label}_check_fields")
    if not isinstance(value.get("translation_equivalent"), bool):
        raise ValueError(f"invalid_{label}_translation_equivalent")
    if not isinstance(value.get("natural_zh"), bool):
        raise ValueError(f"invalid_{label}_natural_zh")
    issues = value.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues):
        raise ValueError(f"invalid_{label}_issues")
    return value["translation_equivalent"] and value["natural_zh"]


def _validate_list_checks(
    actual: Any,
    expected_ids: Sequence[Any],
    id_key: str,
    label: str,
) -> List[bool]:
    if not isinstance(actual, list) or any(not isinstance(row, dict) for row in actual):
        raise ValueError(f"invalid_{label}_checks")
    if any(
        set(row) != {id_key, "translation_equivalent", "natural_zh", "issues"}
        for row in actual
    ):
        raise ValueError(f"invalid_{label}_check_fields")
    actual_ids = [row.get(id_key) for row in actual]
    if actual_ids != list(expected_ids):
        raise ValueError(f"{label}_review_id_or_order_mismatch")
    return [
        _validate_check(
            {key: value for key, value in row.items() if key != id_key},
            f"{label}_{identifier}",
        )
        for row, identifier in zip(actual, expected_ids)
    ]


def validate_review(value: Dict[str, Any], item: Mapping[str, Any]) -> None:
    if set(value) != {"decision", "issues", "checks"}:
        raise ValueError("invalid_review_fields")
    if value.get("decision") not in {"accept", "reject"}:
        raise ValueError("invalid_review_decision")
    if not isinstance(value.get("issues"), list) or any(
        not isinstance(issue, str) for issue in value["issues"]
    ):
        raise ValueError("invalid_review_issues")
    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != {
        *SCALAR_REVIEW_FIELDS,
        *LIST_REVIEW_FIELDS,
    }:
        raise ValueError("invalid_field_review_keys")
    verdicts = [_validate_check(checks[field], field) for field in SCALAR_REVIEW_FIELDS]
    verdicts.extend(
        _validate_list_checks(
            checks["answer_aliases"],
            list(range(len(item["source"]["answer_aliases_en"]))),
            "alias_index",
            "answer_aliases",
        )
    )
    verdicts.extend(
        _validate_list_checks(
            checks["distractors"],
            [row["distractor_id"] for row in item["distractors"]],
            "distractor_id",
            "distractors",
        )
    )
    verdicts.extend(
        _validate_list_checks(
            checks["neutral_candidates"],
            [row["neutral_candidate_id"] for row in item["neutral_candidates"]],
            "neutral_candidate_id",
            "neutral_candidates",
        )
    )
    all_pass = all(verdicts)
    if (value["decision"] == "accept") != all_pass:
        raise ValueError("review_decision_inconsistent_with_field_checks")


def _batch_result_metadata(
    *,
    result: runtime.ModelResult,
    spec: Mapping[str, Any],
    expected_response_model: Optional[str],
    prompt: str,
    prompt_version: str,
    item_ids: Sequence[str],
) -> Dict[str, Any]:
    terminal_status = result.terminal_status
    identity_status = "not_checked"
    identity_error_type = None
    if expected_response_model:
        if result.response_model is None:
            identity_status = "not_observed"
        else:
            identity_status = (
                "matched"
                if result.response_model == expected_response_model
                else "mismatch"
            )
        if terminal_status == "completed" and identity_status != "matched":
            terminal_status = "failed"
            identity_error_type = (
                "ResponseModelMissing"
                if identity_status == "not_observed"
                else "ResponseModelMismatch"
            )
    return {
        "terminal_status": terminal_status,
        "request_model": spec["model"],
        "provider_profile": spec["provider_profile"],
        "expected_response_model": expected_response_model,
        "response_model": result.response_model,
        "response_model_identity_status": identity_status,
        "response_model_identity_error_type": identity_error_type,
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "batch_id": "zh_batch_" + sha256_value([prompt_version, *item_ids])[:20],
        "batch_item_count": len(item_ids),
        "batch_base_fact_ids_sha256": sha256_value(list(item_ids)),
        "batch_parsed_response_sha256": (
            sha256_value(result.parsed_response)
            if result.parsed_response is not None
            else None
        ),
        "batch_raw_response_sha256": (
            hashlib.sha256(result.raw_response.encode("utf-8")).hexdigest()
            if result.raw_response is not None
            else None
        ),
        "usage": _safe_usage(result.usage),
        "usage_scope": "shared_batch_not_per_item",
        "latency_ms": result.latency_ms,
        "attempt_count": result.attempt_count,
        "attempts": _safe_attempts(result.attempts),
        "created_at": utc_now(),
        "credentials_or_endpoints_included": False,
    }


class BatchRequestFailure(RuntimeError):
    def __init__(self, metadata: Mapping[str, Any]):
        super().__init__("batch request did not produce a valid terminal response")
        self.metadata = dict(metadata)


def _should_split_batch_failure(error: Exception) -> bool:
    if not isinstance(error, BatchRequestFailure):
        return True
    metadata = error.metadata
    if metadata.get("response_model_identity_error_type") in {
        "ResponseModelMissing",
        "ResponseModelMismatch",
    }:
        return False
    attempts = metadata.get("attempts")
    attempts = attempts if isinstance(attempts, list) else []
    if any(
        entry.get("error_type")
        in {"AuthenticationError", "PermissionDeniedError", "StaticProxyIdentityError"}
        or entry.get("http_status") in {401, 403}
        for entry in attempts
        if isinstance(entry, dict)
    ):
        return False
    return True


def _stage_error_record(
    item: Mapping[str, Any],
    stage: str,
    error: Exception,
    fallback_history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result = {
        "schema_version": GENERATION_SCHEMA if "review" not in stage else REVIEW_SCHEMA,
        "base_fact_id": item["base_fact_id"],
        "input_record_sha256": item["input_record_sha256"],
        "target_language": "zh",
        "human_gold": False,
        "stage": stage,
        "terminal_status": "failed",
        "parsed_response": None,
        "attempt_count": 0,
        "attempts": [{"attempt": 0, "status": "failed", **runtime.safe_error(error)}],
        "created_at": utc_now(),
        "batch_fallback_history": list(fallback_history),
    }
    if isinstance(error, BatchRequestFailure):
        result.update(error.metadata)
        result["terminal_status"] = "failed"
        result["parsed_response"] = None
        result["batch_fallback_history"] = list(fallback_history)
    return result


def _validate_stage_checkpoint_record(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    stage: str,
    schema_version: str,
    prompt_version: str,
    model_spec: Mapping[str, Any],
    expected_response_model: str,
    parsed_validator: Callable[[Dict[str, Any], Mapping[str, Any]], None],
    lineage: Optional[Mapping[str, Any]] = None,
) -> None:
    base_fact_id = str(item["base_fact_id"])
    if row.get("schema_version") != schema_version:
        raise ValueError(f"{stage} checkpoint schema mismatch: {base_fact_id}")
    if row.get("base_fact_id") != base_fact_id:
        raise ValueError(f"{stage} checkpoint base_fact_id mismatch: {base_fact_id}")
    if row.get("stage") != stage:
        raise ValueError(f"{stage} checkpoint stage mismatch: {base_fact_id}")
    if row.get("target_language") != "zh" or row.get("human_gold") is not False:
        raise ValueError(f"{stage} checkpoint safety contract mismatch: {base_fact_id}")
    terminal_status = row.get("terminal_status")
    if terminal_status not in {"completed", "failed"}:
        raise ValueError(f"{stage} checkpoint terminal status mismatch: {base_fact_id}")
    if not isinstance(row.get("batch_fallback_history"), list):
        raise ValueError(f"{stage} checkpoint fallback history is invalid: {base_fact_id}")
    if "retry_history" in row and not isinstance(row.get("retry_history"), list):
        raise ValueError(f"{stage} checkpoint retry history is invalid: {base_fact_id}")
    if terminal_status == "failed":
        if row.get("parsed_response") is not None:
            raise ValueError(f"{stage} failed checkpoint has parsed output: {base_fact_id}")
        return
    if row.get("request_model") != model_spec["model"]:
        raise ValueError(f"{stage} checkpoint request model mismatch: {base_fact_id}")
    if row.get("provider_profile") != model_spec["provider_profile"]:
        raise ValueError(f"{stage} checkpoint provider mismatch: {base_fact_id}")
    if row.get("expected_response_model") != expected_response_model:
        raise ValueError(f"{stage} checkpoint expected model mismatch: {base_fact_id}")
    if (
        row.get("response_model") != expected_response_model
        or row.get("response_model_identity_status") != "matched"
    ):
        raise ValueError(f"{stage} checkpoint response model mismatch: {base_fact_id}")
    if row.get("prompt_version") != prompt_version:
        raise ValueError(f"{stage} checkpoint prompt version mismatch: {base_fact_id}")
    if row.get("credentials_or_endpoints_included") is not False:
        raise ValueError(f"{stage} checkpoint redaction contract mismatch: {base_fact_id}")
    parsed = row.get("parsed_response")
    if not isinstance(parsed, dict):
        raise ValueError(f"{stage} completed checkpoint has no parsed output: {base_fact_id}")
    parsed_validator(parsed, item)
    for field, expected in dict(lineage or {}).items():
        if row.get(field) != expected:
            raise ValueError(f"{stage} checkpoint {field} mismatch: {base_fact_id}")


def run_checkpoint_stage(
    *,
    items: Sequence[Dict[str, Any]],
    output_path: Path,
    stage: str,
    run_fingerprint: str,
    batch_worker: Callable[[Sequence[Dict[str, Any]]], List[Dict[str, Any]]],
    max_workers: int,
    batch_size: int,
    checkpoint_every: int,
    resume: bool,
    retry_failed: bool = False,
    record_validator: Optional[
        Callable[[Mapping[str, Any], Mapping[str, Any]], None]
    ] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("run_fingerprint must be a non-empty string")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    journal_path = output_path.with_suffix(output_path.suffix + ".journal")
    if (output_path.exists() or journal_path.exists()) and not resume:
        raise FileExistsError(f"checkpoint exists; pass --resume: {output_path}")
    compacted_rows = read_jsonl(output_path, missing_ok=True) if resume else []
    journal_rows = read_jsonl_journal(journal_path) if resume else []
    _require_unique(compacted_rows, "base_fact_id", f"{stage} checkpoint")
    _require_unique(journal_rows, "base_fact_id", f"{stage} journal")
    allowed = {item["base_fact_id"]: item for item in items}
    for row in [*compacted_rows, *journal_rows]:
        item = allowed.get(str(row["base_fact_id"]))
        if item is None:
            raise ValueError(f"{stage} checkpoint contains an out-of-scope base_fact_id")
        if row.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"{stage} checkpoint run fingerprint mismatch")
        if row.get("input_record_sha256") != item["input_record_sha256"]:
            raise ValueError(f"{stage} checkpoint input fingerprint mismatch")
        if record_validator is not None:
            record_validator(row, item)
    existing = {str(row["base_fact_id"]): row for row in compacted_rows}
    existing.update({str(row["base_fact_id"]): row for row in journal_rows})
    pending = [
        item
        for item in items
        if item["base_fact_id"] not in existing
        or (
            retry_failed
            and existing[item["base_fact_id"]].get("terminal_status") == "failed"
        )
    ]

    def failure_summary(
        batch: Sequence[Dict[str, Any]], error: Exception
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "batch_item_count": len(batch),
            "batch_base_fact_ids_sha256": sha256_value(
                [item["base_fact_id"] for item in batch]
            ),
            **runtime.safe_error(error),
        }
        if isinstance(error, BatchRequestFailure):
            result.update(
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
        return result

    def process_batch(
        batch: Sequence[Dict[str, Any]],
        history: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        try:
            rows = batch_worker(batch)
            if len(rows) != len(batch):
                raise ValueError("batch worker returned the wrong record count")
            by_id = {str(row.get("base_fact_id") or ""): row for row in rows}
            expected_ids = [str(item["base_fact_id"]) for item in batch]
            if len(by_id) != len(rows) or set(by_id) != set(expected_ids):
                raise ValueError("batch worker returned mismatched base_fact_id values")
            ordered = [by_id[base_fact_id] for base_fact_id in expected_ids]
            for row in ordered:
                row["run_fingerprint"] = run_fingerprint
                row["batch_fallback_history"] = list(history)
                if record_validator is not None:
                    record_validator(row, allowed[str(row["base_fact_id"])])
            return ordered
        except Exception as error:
            next_history = [*history, failure_summary(batch, error)]
            if len(batch) > 1 and _should_split_batch_failure(error):
                midpoint = len(batch) // 2
                return [
                    *process_batch(batch[:midpoint], next_history),
                    *process_batch(batch[midpoint:], next_history),
                ]
            return [
                _stage_error_record(item, stage, error, next_history)
                for item in batch
            ]

    def compact() -> None:
        write_jsonl(
            output_path,
            [existing[item["base_fact_id"]] for item in items if item["base_fact_id"] in existing],
        )
        if journal_path.exists():
            journal_path.unlink()

    # A crash can occur after replacing the compact file but before deleting
    # the old journal.  Merge by key, compact once, and start from a clean
    # journal before any new append.
    if resume and journal_path.exists():
        compact()

    appended_since_compaction = 0

    def record(row: Dict[str, Any]) -> None:
        nonlocal appended_since_compaction
        row = dict(row)
        existing_run_fingerprint = row.get("run_fingerprint")
        if (
            existing_run_fingerprint is not None
            and existing_run_fingerprint != run_fingerprint
        ):
            raise ValueError(f"{stage} produced a mismatched run fingerprint")
        row["run_fingerprint"] = run_fingerprint
        base_fact_id = str(row["base_fact_id"])
        previous = existing.get(base_fact_id)
        if (
            retry_failed
            and previous is not None
            and previous.get("terminal_status") == "failed"
        ):
            retry_history = copy.deepcopy(previous.get("retry_history", []))
            retry_history.append(
                {
                    "terminal_status": "failed",
                    "failed_record_sha256": sha256_value(previous),
                    "response_model_identity_status": previous.get(
                        "response_model_identity_status"
                    ),
                    "attempt_count": previous.get("attempt_count"),
                    "attempts": copy.deepcopy(previous.get("attempts", [])),
                    "batch_fallback_history": copy.deepcopy(
                        previous.get("batch_fallback_history", [])
                    ),
                    "created_at": previous.get("created_at"),
                }
            )
            row = {**row, "retry_history": retry_history}
        if record_validator is not None:
            record_validator(row, allowed[str(row["base_fact_id"])])
        append_jsonl_journal(journal_path, row)
        existing[base_fact_id] = row
        appended_since_compaction += 1
        if appended_since_compaction >= checkpoint_every:
            compact()
            appended_since_compaction = 0

    if not pending:
        compact()
        return [existing[item["base_fact_id"]] for item in items if item["base_fact_id"] in existing]

    wave_size = batch_size * max_workers
    for wave_start in range(0, len(pending), wave_size):
        wave = pending[wave_start : wave_start + wave_size]
        batches = [
            wave[index : index + batch_size]
            for index in range(0, len(wave), batch_size)
        ]
        if max_workers == 1:
            for row in process_batch(batches[0]):
                record(row)
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
                futures = {executor.submit(process_batch, part): part for part in batches}
                for future in as_completed(futures):
                    for row in future.result():
                        record(row)
    compact()
    return [existing[item["base_fact_id"]] for item in items]


def _model_specs(config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    roles = config.get("model_roles", {}).get("translation", {})
    generator = copy.deepcopy(roles.get("primary") or {})
    reviewer = copy.deepcopy(roles.get("reviewer") or {})
    generator.update(
        {
            "model": args.generator_model,
            "provider_profile": args.generator_provider_profile,
            "max_output_tokens": args.generator_max_output_tokens,
        }
    )
    reviewer.update(
        {
            "model": args.reviewer_model,
            "provider_profile": args.reviewer_provider_profile,
            "max_output_tokens": args.reviewer_max_output_tokens,
        }
    )
    generator["json_mode"] = True
    reviewer["json_mode"] = True
    generator = _safe_model_spec(generator)
    reviewer = _safe_model_spec(reviewer)
    if generator["model"] != DEFAULT_GENERATOR_ROUTE:
        raise ValueError(
            "generator must use the verified qwen3.8-max-0902/bailian/bailian route"
        )
    if reviewer["model"] != DEFAULT_REVIEWER_MODEL:
        raise ValueError("reviewer must be gpt-5.5")
    declared_expected = {
        "generator": str(
            (roles.get("primary") or {}).get(
                "expected_response_model", DEFAULT_GENERATOR_RESPONSE_MODEL
            )
        ).strip(),
        "reviewer": str(
            (roles.get("reviewer") or {}).get(
                "expected_response_model", DEFAULT_REVIEWER_RESPONSE_MODEL
            )
        ).strip(),
    }
    for role, expected in (
        ("generator", args.generator_expected_response_model),
        ("reviewer", args.reviewer_expected_response_model),
    ):
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"{role} expected response model must be non-empty")
        if not declared_expected[role] or expected.strip() != declared_expected[role]:
            raise ValueError(
                f"{role} expected response model differs from the frozen config identity"
            )
    if generator["provider_profile"] == reviewer["provider_profile"]:
        raise ValueError("generation and review must use distinct provider profile labels")
    if (
        generator["model"],
        args.generator_expected_response_model,
    ) == (
        reviewer["model"],
        args.reviewer_expected_response_model,
    ):
        raise ValueError("generation and review model identities must be distinct")
    profiles = config.get("provider_profiles", {})
    for role, spec in (("generator", generator), ("reviewer", reviewer)):
        if spec["provider_profile"] not in profiles:
            raise ValueError(f"missing provider profile for {role}")
    return {"generator": generator, "reviewer": reviewer}


def _review_accepted(row: Optional[Mapping[str, Any]]) -> bool:
    return bool(
        row
        and row.get("terminal_status") == "completed"
        and row.get("parsed_response", {}).get("decision") == "accept"
    )


def _by_id(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["base_fact_id"]): dict(row) for row in rows}


def _binding_for_output(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return artifact_binding(path, rows=rows)


def execute(
    args: argparse.Namespace,
    *,
    router_factory: Callable[..., runtime.ModelRouter] = runtime.ModelRouter,
) -> Dict[str, Any]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    projection_manifest_path = getattr(
        args, "accepted_candidate_projection_manifest", None
    )
    if projection_manifest_path is not None and args.limit is not None:
        raise ValueError(
            "limit is not allowed with an accepted candidate projection; "
            "projection completion must cover the complete projected cohort"
        )
    if args.retry_failed and not args.resume:
        raise ValueError("retry_failed requires resume")
    config = load_config(args.config.resolve())
    specs = _model_specs(config, args)
    max_workers = (
        int(args.max_workers)
        if args.max_workers is not None
        else int(config.get("execution", {}).get("max_workers", 1))
    )
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if args.generator_max_output_tokens < 1 or args.reviewer_max_output_tokens < 1:
        raise ValueError("model max output token limits must be positive")

    loaded = load_bound_inputs(
        formal_universe_manifest_path=args.formal_universe_manifest,
        formal_universe_items_path=args.formal_universe_items,
        full_base_facts_path=args.full_base_facts,
        distractor_candidates_path=args.distractor_candidates,
        neutral_candidates_path=args.neutral_candidates,
        split_manifest_path=args.split_manifest,
        fact_resolution_manifest_path=getattr(
            args, "fact_resolution_manifest", None
        ),
        postreview_rebuild_manifest_path=getattr(
            args, "postreview_rebuild_manifest", None
        ),
        accepted_candidate_projection_manifest_path=projection_manifest_path,
    )
    all_items = loaded["items"]
    selected = all_items[: args.limit] if args.limit is not None else all_items
    selected_ids = [item["base_fact_id"] for item in selected]
    selected_split_counts = _split_counts(selected)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    checkpoint_paths = (
        output_dir / "translation_initial.jsonl",
        output_dir / "translation_initial_reviews.jsonl",
        output_dir / "translation_repairs.jsonl",
        output_dir / "translation_repair_reviews.jsonl",
    )
    managed_output_paths = (
        *checkpoint_paths,
        *(path.with_suffix(path.suffix + ".journal") for path in checkpoint_paths),
        output_dir / "zh_translation_review_records.jsonl",
        output_dir / "translation_quarantine.jsonl",
        output_dir / "redacted_events.jsonl",
    )
    if args.resume and any(path.exists() for path in managed_output_paths) and not manifest_path.is_file():
        raise ValueError("resume requires an existing run manifest when managed outputs exist")
    router = router_factory(
        config, args.env_file.resolve(), output_dir / "redacted_events.jsonl"
    )
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
    route_records = route_identity["records"]
    provider_endpoint_distinct = (
        len(
            {
                str(record["normalized_request_route_sha256"])
                for record in route_records
            }
        )
        == len(route_records)
    )

    successor = loaded["successor_cohort"]
    accepted_projection = loaded["accepted_candidate_projection"]
    original_universe_count = (
        int(successor["source_record_count"])
        if successor is not None
        else int(loaded["formal_manifest"]["record_count"])
    )
    selection_is_complete_retained_cohort = bool(
        successor is not None and len(selected) == len(all_items)
    )
    selection_is_full_universe = bool(
        accepted_projection is None
        and len(selected) == len(all_items)
        and (
            successor is None
            or int(successor["excluded_record_count"]) == 0
        )
    )
    contract = {
        "schema_version": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "input_bindings": loaded["bindings"],
        "source_fingerprints": source_fingerprints(),
        "route_identity": route_identity,
        "review_independence": {
            "model_identity_distinct": True,
            "provider_profile_labels_distinct": True,
            "provider_endpoint_distinct": provider_endpoint_distinct,
            "provider_infrastructure_independence_claimed": False,
            "claim_scope": "distinct_model_identity_only",
        },
        "formal_universe_id": loaded["formal_manifest"].get("universe_id"),
        "formal_universe_status": loaded["formal_manifest"].get("universe_status"),
        "successor_cohort": loaded["successor_cohort"],
        "accepted_candidate_projection": accepted_projection,
        "selected_base_fact_ids_sha256": sha256_value(selected_ids),
        "selected_record_count": len(selected),
        "universe_record_count": original_universe_count,
        "selection_is_full_universe": selection_is_full_universe,
        "selected_split_counts": selected_split_counts,
        "models": {
            "generator": {
                **specs["generator"],
                "expected_response_model": args.generator_expected_response_model,
                "role": "translation_generation_and_single_repair",
            },
            "reviewer": {
                **specs["reviewer"],
                "expected_response_model": args.reviewer_expected_response_model,
                "role": "independent_translation_equivalence_review",
            },
        },
        "candidate_binding": loaded["candidate_binding"],
        "invalidation_contract": {
            "provisional_candidate_binding": True,
            "fact_resolution_or_postreview_change_invalidates_run": True,
            "accepted_candidate_projection_or_source_review_change_invalidates_run": True,
            "full_base_facts_change_invalidates_run": True,
            "distractor_or_neutral_candidate_change_invalidates_run": True,
            "invalidated_by_recluster_or_candidate_regeneration": True,
            "invalidated_by_split_change": True,
            "resume_requires_identical_run_fingerprint": True,
        },
        "language_scope": {
            "processed_language_codes": ["zh"],
            "other_registered_target_language_codes": list(OTHER_TARGET_LANGUAGES),
            "other_registered_target_languages_status": "unchanged_pending_translation",
            "does_not_mark_unprocessed_languages_complete": True,
        },
        "evidence_boundary": {
            "human_gold": False,
            "reviewer_type": "independent_model_proxy",
            "independence_scope": "distinct_model_identity_only",
            "provider_infrastructure_independence_claimed": False,
            "translation_only": True,
            "factual_truth_review_performed": False,
            "distractor_factual_falsehood_review_performed": False,
            "neutral_unrelatedness_review_performed": False,
            "formal_relation_freeze_performed": False,
            "formal_split_freeze_performed": False,
            "hf_model_execution_count": 0,
            "hf_tokenizer_execution_count": 0,
            "behavior_execution_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
        },
        "execution_policy": {
            "max_workers": max_workers,
            "batch_size": args.batch_size,
            "one_model_request_per_batch": True,
            "recursive_binary_split_on_batch_failure": True,
            "checkpoint_strategy": "single_writer_append_journal_with_atomic_compaction",
            "journal_fsync_after_each_item": True,
            "atomic_compaction_every_items": args.checkpoint_every,
            "atomic_compaction_at_stage_completion": True,
            "resume_merges_compact_file_and_complete_journal_lines": True,
            "incomplete_final_journal_line_ignored": True,
            "resume_supported": True,
            "retry_failed_supported": True,
            "failed_items_quarantined": True,
            "maximum_translation_repairs_per_item": 1,
            "response_model_identity_fail_closed": True,
            "checkpoint_payload_revalidated_on_resume": True,
            "upstream_stage_lineage_revalidated_on_resume": True,
            "credential_and_endpoint_fields_excluded_from_model_specs": True,
        },
    }
    if successor is not None:
        contract.update(
            {
                "selection_is_complete_retained_cohort": (
                    selection_is_complete_retained_cohort
                ),
                "excluded_base_fact_ids_sha256": successor[
                    "excluded_base_fact_ids_sha256"
                ],
            }
        )
    if accepted_projection is not None:
        contract.update(
            {
                "selection_is_complete_accepted_projection": bool(
                    len(selected) == len(all_items)
                ),
                "accepted_projection_record_count": len(all_items),
                "accepted_projection_quarantined_source_record_count": int(
                    accepted_projection["quarantined_record_count"]
                ),
            }
        )
    run_fingerprint = sha256_value(contract)
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"run manifest exists; pass --resume: {manifest_path}")
        previous_manifest = read_json(manifest_path)
        if previous_manifest.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "resume run fingerprint mismatch; an input, candidate set, split, selection, or model changed"
            )
    manifest = {
        **contract,
        "run_fingerprint": run_fingerprint,
        "status": "running",
        "invocation_resumed": bool(args.resume),
        "retry_failed_enabled": bool(args.retry_failed),
        "started_or_resumed_at": utc_now(),
    }
    write_json(manifest_path, manifest)

    if hasattr(router, "_event"):
        event_lock = threading.Lock()
        original_event = router._event

        def locked_event(*event_args: Any, **event_kwargs: Any) -> Any:
            with event_lock:
                return original_event(*event_args, **event_kwargs)

        router._event = locked_event

    def validate_initial_generation_record(
        row: Mapping[str, Any], item: Mapping[str, Any]
    ) -> None:
        _validate_stage_checkpoint_record(
            row,
            item,
            stage="initial_generation",
            schema_version=GENERATION_SCHEMA,
            prompt_version=PROMPT_VERSION,
            model_spec=specs["generator"],
            expected_response_model=args.generator_expected_response_model,
            parsed_validator=validate_translation,
        )

    def generate(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        item_ids = [item["base_fact_id"] for item in batch]
        prompt = translation_batch_prompt(batch)
        result = router.request_json(
            "full_zh_translation_generation",
            "batch_" + sha256_value(item_ids)[:20],
            {
                **specs["generator"],
                "expected_response_model": args.generator_expected_response_model,
            },
            prompt,
            lambda value: validate_translation_batch(value, batch),
        )
        metadata = _batch_result_metadata(
            result=result,
            spec=specs["generator"],
            expected_response_model=args.generator_expected_response_model,
            prompt=prompt,
            prompt_version=PROMPT_VERSION,
            item_ids=item_ids,
        )
        if metadata["terminal_status"] != "completed":
            raise BatchRequestFailure(metadata)
        parsed_by_id = validate_translation_batch(result.parsed_response or {}, batch)
        return [
            {
                "schema_version": GENERATION_SCHEMA,
                "base_fact_id": item["base_fact_id"],
                "input_record_sha256": item["input_record_sha256"],
                "target_language": "zh",
                "human_gold": False,
                "stage": "initial_generation",
                **metadata,
                "parsed_response": parsed_by_id[item["base_fact_id"]],
            }
            for item in batch
        ]

    initial_generations = run_checkpoint_stage(
        items=selected,
        output_path=output_dir / "translation_initial.jsonl",
        stage="initial_generation",
        run_fingerprint=run_fingerprint,
        batch_worker=generate,
        max_workers=max_workers,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        retry_failed=args.retry_failed,
        record_validator=validate_initial_generation_record,
    )
    initial_by_id = _by_id(initial_generations)
    review_items = [
        item
        for item in selected
        if initial_by_id[item["base_fact_id"]].get("terminal_status") == "completed"
    ]

    def review_initial(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        item_ids = [item["base_fact_id"] for item in batch]
        translations_by_id = {
            item["base_fact_id"]: initial_by_id[item["base_fact_id"]]["parsed_response"]
            for item in batch
        }
        prompt = review_batch_prompt(batch, translations_by_id)
        result = router.request_json(
            "full_zh_translation_review",
            "batch_" + sha256_value(item_ids)[:20],
            {
                **specs["reviewer"],
                "expected_response_model": args.reviewer_expected_response_model,
            },
            prompt,
            lambda value: validate_review_batch(value, batch),
        )
        metadata = _batch_result_metadata(
            result=result,
            spec=specs["reviewer"],
            expected_response_model=args.reviewer_expected_response_model,
            prompt=prompt,
            prompt_version=REVIEW_PROMPT_VERSION,
            item_ids=item_ids,
        )
        if metadata["terminal_status"] != "completed":
            raise BatchRequestFailure(metadata)
        parsed_by_id = validate_review_batch(result.parsed_response or {}, batch)
        return [
            {
                "schema_version": REVIEW_SCHEMA,
                "base_fact_id": item["base_fact_id"],
                "input_record_sha256": item["input_record_sha256"],
                "translation_record_sha256": sha256_value(
                    initial_by_id[item["base_fact_id"]]
                ),
                "target_language": "zh",
                "reviewer_type": "independent_model_proxy",
                "human_gold": False,
                "stage": "initial_review",
                **metadata,
                "parsed_response": parsed_by_id[item["base_fact_id"]],
            }
            for item in batch
        ]

    def validate_initial_review_record(
        row: Mapping[str, Any], item: Mapping[str, Any]
    ) -> None:
        base_fact_id = str(item["base_fact_id"])
        _validate_stage_checkpoint_record(
            row,
            item,
            stage="initial_review",
            schema_version=REVIEW_SCHEMA,
            prompt_version=REVIEW_PROMPT_VERSION,
            model_spec=specs["reviewer"],
            expected_response_model=args.reviewer_expected_response_model,
            parsed_validator=validate_review,
            lineage={
                "translation_record_sha256": sha256_value(
                    initial_by_id[base_fact_id]
                )
            },
        )

    initial_reviews = run_checkpoint_stage(
        items=review_items,
        output_path=output_dir / "translation_initial_reviews.jsonl",
        stage="initial_review",
        run_fingerprint=run_fingerprint,
        batch_worker=review_initial,
        max_workers=max_workers,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        retry_failed=args.retry_failed,
        record_validator=validate_initial_review_record,
    )
    initial_review_by_id = _by_id(initial_reviews)
    repair_items = [
        item
        for item in review_items
        if initial_review_by_id.get(item["base_fact_id"], {}).get("terminal_status")
        == "completed"
        and not _review_accepted(initial_review_by_id[item["base_fact_id"]])
    ]

    def repair(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        item_ids = [item["base_fact_id"] for item in batch]
        repairs_by_id = {
            item["base_fact_id"]: {
                "translation": initial_by_id[item["base_fact_id"]]["parsed_response"],
                "review": initial_review_by_id[item["base_fact_id"]]["parsed_response"],
            }
            for item in batch
        }
        prompt = translation_batch_prompt(batch, repairs_by_id=repairs_by_id)
        result = router.request_json(
            "full_zh_translation_repair",
            "batch_" + sha256_value(item_ids)[:20],
            {
                **specs["generator"],
                "expected_response_model": args.generator_expected_response_model,
            },
            prompt,
            lambda value: validate_translation_batch(value, batch),
        )
        metadata = _batch_result_metadata(
            result=result,
            spec=specs["generator"],
            expected_response_model=args.generator_expected_response_model,
            prompt=prompt,
            prompt_version=REPAIR_PROMPT_VERSION,
            item_ids=item_ids,
        )
        if metadata["terminal_status"] != "completed":
            raise BatchRequestFailure(metadata)
        parsed_by_id = validate_translation_batch(result.parsed_response or {}, batch)
        return [
            {
                "schema_version": GENERATION_SCHEMA,
                "base_fact_id": item["base_fact_id"],
                "input_record_sha256": item["input_record_sha256"],
                "repaired_translation_record_sha256": sha256_value(
                    initial_by_id[item["base_fact_id"]]
                ),
                "repaired_review_record_sha256": sha256_value(
                    initial_review_by_id[item["base_fact_id"]]
                ),
                "target_language": "zh",
                "human_gold": False,
                "stage": "repair_1",
                "repair_number": 1,
                **metadata,
                "parsed_response": parsed_by_id[item["base_fact_id"]],
            }
            for item in batch
        ]

    def validate_repair_record(
        row: Mapping[str, Any], item: Mapping[str, Any]
    ) -> None:
        base_fact_id = str(item["base_fact_id"])
        _validate_stage_checkpoint_record(
            row,
            item,
            stage="repair_1",
            schema_version=GENERATION_SCHEMA,
            prompt_version=REPAIR_PROMPT_VERSION,
            model_spec=specs["generator"],
            expected_response_model=args.generator_expected_response_model,
            parsed_validator=validate_translation,
            lineage={
                "repaired_translation_record_sha256": sha256_value(
                    initial_by_id[base_fact_id]
                ),
                "repaired_review_record_sha256": sha256_value(
                    initial_review_by_id[base_fact_id]
                ),
            },
        )

    repairs = run_checkpoint_stage(
        items=repair_items,
        output_path=output_dir / "translation_repairs.jsonl",
        stage="repair_1",
        run_fingerprint=run_fingerprint,
        batch_worker=repair,
        max_workers=max_workers,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        retry_failed=args.retry_failed,
        record_validator=validate_repair_record,
    )
    repair_by_id = _by_id(repairs)
    repair_review_items = [
        item
        for item in repair_items
        if repair_by_id[item["base_fact_id"]].get("terminal_status") == "completed"
    ]

    def review_repair(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        item_ids = [item["base_fact_id"] for item in batch]
        translations_by_id = {
            item["base_fact_id"]: repair_by_id[item["base_fact_id"]]["parsed_response"]
            for item in batch
        }
        prompt = review_batch_prompt(batch, translations_by_id)
        result = router.request_json(
            "full_zh_translation_repair_review",
            "batch_" + sha256_value(item_ids)[:20],
            {
                **specs["reviewer"],
                "expected_response_model": args.reviewer_expected_response_model,
            },
            prompt,
            lambda value: validate_review_batch(value, batch),
        )
        metadata = _batch_result_metadata(
            result=result,
            spec=specs["reviewer"],
            expected_response_model=args.reviewer_expected_response_model,
            prompt=prompt,
            prompt_version=REPAIR_REVIEW_PROMPT_VERSION,
            item_ids=item_ids,
        )
        if metadata["terminal_status"] != "completed":
            raise BatchRequestFailure(metadata)
        parsed_by_id = validate_review_batch(result.parsed_response or {}, batch)
        return [
            {
                "schema_version": REVIEW_SCHEMA,
                "base_fact_id": item["base_fact_id"],
                "input_record_sha256": item["input_record_sha256"],
                "translation_record_sha256": sha256_value(
                    repair_by_id[item["base_fact_id"]]
                ),
                "target_language": "zh",
                "reviewer_type": "independent_model_proxy",
                "human_gold": False,
                "stage": "repair_1_review",
                "repair_number": 1,
                **metadata,
                "parsed_response": parsed_by_id[item["base_fact_id"]],
            }
            for item in batch
        ]

    def validate_repair_review_record(
        row: Mapping[str, Any], item: Mapping[str, Any]
    ) -> None:
        base_fact_id = str(item["base_fact_id"])
        _validate_stage_checkpoint_record(
            row,
            item,
            stage="repair_1_review",
            schema_version=REVIEW_SCHEMA,
            prompt_version=REPAIR_REVIEW_PROMPT_VERSION,
            model_spec=specs["reviewer"],
            expected_response_model=args.reviewer_expected_response_model,
            parsed_validator=validate_review,
            lineage={
                "translation_record_sha256": sha256_value(
                    repair_by_id[base_fact_id]
                )
            },
        )

    repair_reviews = run_checkpoint_stage(
        items=repair_review_items,
        output_path=output_dir / "translation_repair_reviews.jsonl",
        stage="repair_1_review",
        run_fingerprint=run_fingerprint,
        batch_worker=review_repair,
        max_workers=max_workers,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        retry_failed=args.retry_failed,
        record_validator=validate_repair_review_record,
    )
    repair_review_by_id = _by_id(repair_reviews)

    final_rows: List[Dict[str, Any]] = []
    quarantine_rows: List[Dict[str, Any]] = []
    for item in selected:
        base_fact_id = item["base_fact_id"]
        generation = initial_by_id[base_fact_id]
        initial_review = initial_review_by_id.get(base_fact_id)
        repaired = repair_by_id.get(base_fact_id)
        repaired_review = repair_review_by_id.get(base_fact_id)
        if _review_accepted(initial_review):
            final_status = "completed"
            origin = "initial"
            translation = generation.get("parsed_response")
            review = initial_review.get("parsed_response")
            reasons: List[str] = []
        elif _review_accepted(repaired_review):
            final_status = "completed"
            origin = "repair_1"
            translation = repaired.get("parsed_response") if repaired else None
            review = repaired_review.get("parsed_response")
            reasons = []
        else:
            final_status = "quarantined"
            origin = (
                "repair_1_rejected"
                if repaired is not None
                else "initial_rejected_or_failed"
            )
            translation = (
                repaired.get("parsed_response")
                if repaired and repaired.get("terminal_status") == "completed"
                else generation.get("parsed_response")
            )
            review = (
                repaired_review.get("parsed_response")
                if repaired_review
                else initial_review.get("parsed_response") if initial_review else None
            )
            reasons = []
            if generation.get("terminal_status") != "completed":
                reasons.append("initial_generation_failed")
            elif not initial_review or initial_review.get("terminal_status") != "completed":
                reasons.append("initial_review_failed")
            elif repaired is None:
                reasons.append("initial_review_rejected_without_repair_result")
            elif repaired.get("terminal_status") != "completed":
                reasons.append("repair_failed")
            elif not repaired_review or repaired_review.get("terminal_status") != "completed":
                reasons.append("repair_review_failed")
            else:
                reasons.append("repair_review_rejected")
        final_row = {
            "schema_version": FINAL_SCHEMA,
            "base_fact_id": base_fact_id,
            "cohort_item_id": item["cohort_item_id"],
            "input_record_sha256": item["input_record_sha256"],
            "split_assignment": item["split_assignment"],
            "target_language": "zh",
            "human_gold": False,
            "reviewer_type": "independent_model_proxy",
            "terminal_status": final_status,
            "translation_origin": origin,
            "source_fields": item["source"],
            "source_distractors": item["distractors"],
            "source_neutral_candidates": item["neutral_candidates"],
            "translation": translation,
            "translation_equivalence_review": review,
            "quarantine_reasons": reasons,
            "maximum_translation_repairs_performed": 1 if repaired is not None else 0,
            "formal_claims": {
                "translation_proxy_accepted": final_status == "completed",
                "human_gold": False,
                "factual_truth_reviewed": False,
                "distractor_falsehood_reviewed": False,
                "neutral_unrelatedness_reviewed": False,
            },
            "lineage": {
                "initial_generation_record_sha256": sha256_value(generation),
                "initial_review_record_sha256": (
                    sha256_value(initial_review) if initial_review else None
                ),
                "repair_record_sha256": sha256_value(repaired) if repaired else None,
                "repair_review_record_sha256": (
                    sha256_value(repaired_review) if repaired_review else None
                ),
            },
        }
        final_rows.append(final_row)
        if final_status == "quarantined":
            quarantine_rows.append(
                {
                    "schema_version": QUARANTINE_SCHEMA,
                    "base_fact_id": base_fact_id,
                    "input_record_sha256": item["input_record_sha256"],
                    "target_language": "zh",
                    "human_gold": False,
                    "reasons": reasons,
                    "final_record_sha256": sha256_value(final_row),
                }
            )

    final_path = output_dir / "zh_translation_review_records.jsonl"
    quarantine_path = output_dir / "translation_quarantine.jsonl"
    write_jsonl(final_path, final_rows)
    write_jsonl(quarantine_path, quarantine_rows)
    completed_count = sum(row["terminal_status"] == "completed" for row in final_rows)
    is_complete_selected_scope = len(selected) == len(all_items)
    is_full = bool(is_complete_selected_scope and accepted_projection is None)
    is_complete_accepted_projection = bool(
        is_complete_selected_scope and accepted_projection is not None
    )
    manifest.update(
        {
            "status": (
                "completed_full_zh_proxy_review"
                if is_full and not quarantine_rows
                else (
                    "completed_accepted_candidate_projection_zh_proxy_review"
                    if is_complete_accepted_projection and not quarantine_rows
                    else "completed_with_quarantine_or_limited_scope"
                )
            ),
            "completed_at": utc_now(),
            "counts": {
                "universe_records": original_universe_count,
                "selected_records": len(selected),
                "proxy_accepted_records": completed_count,
                "quarantined_records": len(quarantine_rows),
                "initial_review_accepted_records": sum(
                    _review_accepted(row) for row in initial_reviews
                ),
                "repair_attempted_records": len(repairs),
                "repair_review_accepted_records": sum(
                    _review_accepted(row) for row in repair_reviews
                ),
            },
            "completion_claims": {
                "selected_scope_processing_complete": True,
                "full_pool_zh_translation_review_complete": bool(
                    is_full and completed_count == len(all_items)
                ),
                "accepted_candidate_projection_zh_translation_review_complete": bool(
                    is_complete_accepted_projection
                    and completed_count == len(all_items)
                ),
                "zh_proxy_review_not_human_gold": True,
                "other_15_target_languages_complete": False,
                "hf_or_behavior_work_performed": False,
            },
            "output_artifacts": {
                "translation_initial": _binding_for_output(
                    output_dir / "translation_initial.jsonl", initial_generations
                ),
                "translation_initial_reviews": _binding_for_output(
                    output_dir / "translation_initial_reviews.jsonl", initial_reviews
                ),
                "translation_repairs": _binding_for_output(
                    output_dir / "translation_repairs.jsonl", repairs
                ),
                "translation_repair_reviews": _binding_for_output(
                    output_dir / "translation_repair_reviews.jsonl", repair_reviews
                ),
                "zh_translation_review_records": _binding_for_output(
                    final_path, final_rows
                ),
                "translation_quarantine": _binding_for_output(
                    quarantine_path, quarantine_rows
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
    parser.add_argument("--formal-universe-manifest", type=Path, required=True)
    parser.add_argument("--formal-universe-items", type=Path, required=True)
    parser.add_argument("--full-base-facts", type=Path)
    parser.add_argument("--distractor-candidates", type=Path)
    parser.add_argument("--neutral-candidates", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--fact-resolution-manifest", type=Path)
    parser.add_argument("--postreview-rebuild-manifest", type=Path)
    parser.add_argument("--accepted-candidate-projection-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_ROUTE)
    parser.add_argument("--generator-provider-profile", default="aliyun")
    parser.add_argument("--generator-max-output-tokens", type=int, default=16384)
    parser.add_argument(
        "--generator-expected-response-model",
        default=DEFAULT_GENERATOR_RESPONSE_MODEL,
    )
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--reviewer-provider-profile", default="openai")
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=16384)
    parser.add_argument(
        "--reviewer-expected-response-model", default=DEFAULT_REVIEWER_RESPONSE_MODEL
    )
    return parser


def _manifest_exit_code(manifest: Mapping[str, Any]) -> int:
    claims = manifest.get("completion_claims")
    counts = manifest.get("counts")
    status = manifest.get("status")
    scope_complete = bool(
        status == "completed_full_zh_proxy_review"
        and isinstance(claims, dict)
        and claims.get("full_pool_zh_translation_review_complete") is True
    ) or bool(
        status == "completed_accepted_candidate_projection_zh_proxy_review"
        and isinstance(claims, dict)
        and claims.get(
            "accepted_candidate_projection_zh_translation_review_complete"
        )
        is True
    )
    complete = bool(
        scope_complete
        and isinstance(counts, dict)
        and counts.get("quarantined_records") == 0
    )
    return 0 if complete else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = execute(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return _manifest_exit_code(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
