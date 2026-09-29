#!/usr/bin/env python3
"""Materialize a non-frozen duplicate closure and provisional split.

The command is intentionally offline and fail closed.  ``candidates`` first
checks the complete public-benchmark pool against an authoritative upstream
manifest and an explicit expected record count, overlays reviewed (or still
provisional) cohort rows, replays the repository's lexical candidate generator,
and retains every candidate edge in a graph component reachable from a cohort
row.  ``resolve`` requires a SHA-bound decision for every retained edge before
it writes any resolved artifact.

Neither command emits a review freeze, a split freeze, a perturbation input, or
any claim of human-gold review.  No model, API, tokenizer, or network call is
made.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


TOOL_VERSION = "public-benchmark-duplicate-closure-v3"
CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-candidates-v2"
)
ADJUDICATION_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-adjudication-v1"
)
COMPONENT_SCHEMA_VERSION = "public-benchmark-leakage-component-v2"
EXCLUSION_SCHEMA_VERSION = "public-benchmark-cohort-exclusion-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-provisional-split-v2"
)
RESOLUTION_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-duplicate-closure-resolution-v2"
)

INPUT_BUNDLE_SCHEMA_VERSION = "factual-perturbation-input-bundle-v1"
AUDIT_SUMMARY_SCHEMA_VERSION = "public-benchmark-lexical-candidate-audit-v2"
AUDIT_PAIR_SCHEMA_VERSION = "public-benchmark-lexical-candidate-pair-v2"
EXACT_DUPLICATE_MATCH_TYPES = frozenset(
    {"question_exact", "canonical_fact_exact"}
)
DECISIONS = frozenset(
    {"same_fact", "same_leakage_component", "distinct", "exclude_cohort"}
)
SPLITS = ("development", "validation", "sealed")
SPLIT_RATIOS = {
    "development": 0.60,
    "validation": 0.20,
    "sealed": 0.20,
}
DEFAULT_SPLIT_SEED = "public-benchmark-duplicate-closure-split-v1"
SOURCE_UNIVERSE_MANIFEST_SCHEMA_VERSION = (
    "public-benchmark-formal-cohort-universe-v2"
)
SOURCE_BEHAVIOR_BINDING_LOCATOR = "source_artifacts.behavior_bundle"
RECONSTRUCTED_AUDIT_LOGICAL_PATH = (
    "manifest://reconstruction/reconstructed_full_pool.jsonl"
)
AUDIT_PROVENANCE_NORMALIZATION_VERSION = (
    "internal-reconstruction-logical-path-v1"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_public_benchmark_near_duplicates.py"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"Input JSONL is empty: {path}")
    return rows


def serialize_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_write(path, payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, serialize_jsonl(rows))


def file_binding(
    path: Path,
    *,
    record_count: Optional[int] = None,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(path).resolve()
    result: Dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if record_count is not None:
        result["record_count"] = record_count
    if schema_version is not None:
        result["schema_version"] = schema_version
    return result


def _authoritative_full_pool_binding(
    source_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        source_manifest.get("schema_version")
        != SOURCE_UNIVERSE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported authoritative source-universe manifest")
    source_artifacts = source_manifest.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ValueError("Authoritative source manifest lacks source_artifacts")
    binding = source_artifacts.get("behavior_bundle")
    if not isinstance(binding, dict):
        raise ValueError("Authoritative source manifest lacks behavior_bundle binding")
    return binding


def _validate_authoritative_full_pool(
    *,
    source_manifest: Mapping[str, Any],
    full_pool_path: Path,
    full_rows: Sequence[Mapping[str, Any]],
    expected_record_count: int,
) -> Mapping[str, Any]:
    if (
        isinstance(expected_record_count, bool)
        or not isinstance(expected_record_count, int)
        or expected_record_count < 1
    ):
        raise ValueError("expected_full_pool_record_count must be a positive integer")
    binding = _authoritative_full_pool_binding(source_manifest)
    required = {
        "path",
        "sha256",
        "byte_count",
        "record_count",
        "schema_version",
        "record_ids_sha256",
        "record_row_hashes_sha256",
    }
    missing = sorted(required - set(binding))
    if missing:
        raise ValueError(
            "Authoritative behavior_bundle binding lacks fields: "
            + ", ".join(missing)
        )

    bound_path = binding.get("path")
    if not isinstance(bound_path, str) or Path(bound_path).resolve() != full_pool_path:
        raise ValueError("Authoritative behavior_bundle path does not match full pool")
    if binding.get("schema_version") != INPUT_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Authoritative behavior_bundle schema is unsupported")
    if binding.get("record_count") != expected_record_count:
        raise ValueError(
            "Authoritative behavior_bundle record count does not match expected count"
        )
    if len(full_rows) != expected_record_count:
        raise ValueError(
            f"Full pool record count is {len(full_rows)}, expected {expected_record_count}"
        )
    if binding.get("sha256") != sha256_file(full_pool_path):
        raise ValueError("Authoritative behavior_bundle SHA-256 is stale")
    if binding.get("byte_count") != full_pool_path.stat().st_size:
        raise ValueError("Authoritative behavior_bundle byte count is stale")

    identifiers = [str(row["base_fact_id"]) for row in full_rows]
    if binding.get("record_ids_sha256") != sha256_value(identifiers):
        raise ValueError("Authoritative behavior_bundle record ID order is stale")
    if binding.get("record_row_hashes_sha256") != sha256_value(
        [sha256_value(row) for row in full_rows]
    ):
        raise ValueError("Authoritative behavior_bundle row hashes are stale")
    return binding


def _normalize_audit_outputs(
    *,
    audit_summary: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    reconstructed_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Remove the output directory from semantic audit rows and hashes."""

    expected_path = str(Path(reconstructed_path).resolve())
    normalized_rows: List[Dict[str, Any]] = copy.deepcopy(list(candidate_rows))
    for index, pair in enumerate(normalized_rows, start=1):
        for side in ("left", "right"):
            endpoint = pair.get(side)
            provenance = (
                endpoint.get("provenance") if isinstance(endpoint, dict) else None
            )
            if not isinstance(provenance, dict):
                raise ValueError(
                    f"Audit candidate row {index} has invalid {side} provenance"
                )
            if provenance.get("input_role") != "current_bundle":
                continue
            if provenance.get("input_path") != expected_path:
                raise ValueError(
                    f"Audit candidate row {index} has unexpected reconstructed path"
                )
            provenance["input_path"] = RECONSTRUCTED_AUDIT_LOGICAL_PATH

    normalized_pair_sha256 = sha256_bytes(serialize_jsonl(normalized_rows))
    normalized_summary: Dict[str, Any] = copy.deepcopy(dict(audit_summary))
    inputs = normalized_summary.get("input")
    behavior = inputs.get("behavior_bundle") if isinstance(inputs, dict) else None
    if not isinstance(behavior, dict) or behavior.get("path") != expected_path:
        raise ValueError("Lexical audit summary has unexpected reconstructed path")
    behavior["path"] = RECONSTRUCTED_AUDIT_LOGICAL_PATH
    artifacts = normalized_summary.get("artifacts")
    candidate_binding = (
        artifacts.get("candidate_pairs") if isinstance(artifacts, dict) else None
    )
    if not isinstance(candidate_binding, dict):
        raise ValueError("Lexical audit summary lacks candidate-pair binding")
    candidate_binding["sha256"] = normalized_pair_sha256
    return normalized_summary, normalized_rows, normalized_pair_sha256


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_duplicate_closure_lexical_audit", AUDIT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load lexical audit: {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} row {index} lacks {field}")
        if value in output:
            raise ValueError(f"Duplicate {field} in {label}: {value}")
        output[value] = dict(row)
    return output


def _validate_bundle_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> Dict[str, Dict[str, Any]]:
    index = _unique_index(rows, "base_fact_id", label)
    for base_fact_id, row in index.items():
        if row.get("schema_version") != INPUT_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported {label} schema for {base_fact_id}: "
                f"{row.get('schema_version')!r}"
            )
    return index


def _endpoint_key(pair: Mapping[str, Any], side: str) -> str:
    endpoint = pair.get(side)
    provenance = endpoint.get("provenance") if isinstance(endpoint, dict) else None
    if not isinstance(provenance, dict):
        raise ValueError(f"Candidate pair has invalid {side} endpoint")
    role = provenance.get("input_role")
    if role == "current_bundle":
        identifier = provenance.get("base_fact_id")
    elif role == "comparison_canonical":
        identifier = provenance.get("audit_record_id")
    else:
        raise ValueError(f"Candidate pair has unsupported input_role: {role!r}")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"Candidate pair has invalid {side} identity")
    return f"{role}:{identifier}"


def _closure_from_pairs(
    pairs: Sequence[Mapping[str, Any]], cohort_ids: Sequence[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Mapping[str, Any]]]:
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    node_metadata: Dict[str, Mapping[str, Any]] = {}
    normalized_pairs: List[Tuple[str, str, Dict[str, Any]]] = []
    seen_pair_ids: Set[str] = set()
    for index, raw_pair in enumerate(pairs, start=1):
        pair = dict(raw_pair)
        if pair.get("schema_version") != AUDIT_PAIR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported candidate pair schema at row {index}")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Candidate pair row {index} lacks pair_id")
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate candidate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        left = _endpoint_key(pair, "left")
        right = _endpoint_key(pair, "right")
        adjacency[left].add(right)
        adjacency[right].add(left)
        node_metadata[left] = pair["left"]["provenance"]
        node_metadata[right] = pair["right"]["provenance"]
        normalized_pairs.append((left, right, pair))

    seeds = [f"current_bundle:{base_fact_id}" for base_fact_id in cohort_ids]
    present_seeds = sorted(set(seeds) & set(adjacency))
    reachable: Set[str] = set(present_seeds)
    queue: deque[str] = deque(present_seeds)
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adjacency[node]):
            if neighbour not in reachable:
                reachable.add(neighbour)
                queue.append(neighbour)

    closure_pairs = [
        pair
        for left, right, pair in normalized_pairs
        if left in reachable and right in reachable
    ]
    closure_pairs.sort(key=lambda row: str(row["pair_id"]))

    cohort_set = set(cohort_ids)
    current_ids = {
        str(node_metadata[node]["base_fact_id"])
        for node in reachable
        if node_metadata[node].get("input_role") == "current_bundle"
    }
    canonical_nodes = {
        node
        for node in reachable
        if node_metadata[node].get("input_role") == "comparison_canonical"
    }

    component_count = len(cohort_ids) - len(present_seeds)
    unvisited = set(reachable)
    component_sizes: Counter[int] = Counter()
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        component = {start}
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbour in adjacency[node]:
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    component.add(neighbour)
                    pending.append(neighbour)
        component_count += 1
        component_sizes[len(component)] += 1
    if len(cohort_ids) > len(present_seeds):
        component_sizes[1] += len(cohort_ids) - len(present_seeds)

    stats = {
        "cohort_seed_count": len(cohort_ids),
        "cohort_seed_with_candidate_count": len(present_seeds),
        "cohort_seed_without_candidate_count": len(cohort_ids) - len(present_seeds),
        "cohort_seed_touched_count": len(current_ids & cohort_set),
        "closure_pair_count": len(closure_pairs),
        "closure_cross_split_pair_count": sum(
            row.get("cross_split") is True for row in closure_pairs
        ),
        "closure_node_count": len(reachable),
        "closure_current_pool_node_count": len(current_ids),
        "closure_out_of_cohort_current_pool_node_count": len(
            current_ids - cohort_set
        ),
        "closure_comparison_canonical_node_count": len(canonical_nodes),
        "closure_component_count_including_isolated_seeds": component_count,
        "closure_component_size_counts": {
            str(size): count for size, count in sorted(component_sizes.items())
        },
        "ordered_closure_node_ids_sha256": sha256_value(sorted(reachable)),
        "ordered_external_current_base_fact_ids_sha256": sha256_value(
            sorted(current_ids - cohort_set)
        ),
    }
    return closure_pairs, stats, node_metadata


def _prepare_new_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    return output_dir


def materialize_candidates(
    *,
    full_pool_path: Path,
    source_manifest_path: Path,
    expected_full_pool_record_count: int,
    cohort_bundle_path: Path,
    comparison_canonical_path: Path,
    output_dir: Path,
    question_threshold: float = 0.80,
    fact_threshold: float = 0.80,
    max_bucket_neighbors: int = 24,
    max_examples: int = 20,
) -> Dict[str, Any]:
    """Create an overlaid full pool and cohort-seeded conservative closure."""

    full_pool_path = Path(full_pool_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    cohort_bundle_path = Path(cohort_bundle_path).resolve()
    comparison_canonical_path = Path(comparison_canonical_path).resolve()
    full_rows = read_jsonl(full_pool_path)
    source_manifest = read_json(source_manifest_path)
    cohort_rows = read_jsonl(cohort_bundle_path)
    full_index = _validate_bundle_rows(full_rows, "full pool")
    authoritative_binding = _validate_authoritative_full_pool(
        source_manifest=source_manifest,
        full_pool_path=full_pool_path,
        full_rows=full_rows,
        expected_record_count=expected_full_pool_record_count,
    )
    cohort_index = _validate_bundle_rows(cohort_rows, "cohort bundle")
    cohort_ids = list(cohort_index)
    missing = sorted(set(cohort_ids) - set(full_index))
    if missing:
        raise ValueError(
            "Cohort base_fact_id is absent from the full pool: " + ", ".join(missing[:5])
        )
    if not comparison_canonical_path.is_file():
        raise FileNotFoundError(comparison_canonical_path)

    reconstructed_rows = [
        cohort_index.get(str(row["base_fact_id"]), dict(row)) for row in full_rows
    ]
    output_dir = _prepare_new_output_dir(output_dir)
    reconstructed_path = output_dir / "reconstructed_full_pool.jsonl"
    closure_pairs_path = output_dir / "closure_candidate_pairs.jsonl"
    manifest_path = output_dir / "candidate_manifest.json"

    try:
        write_jsonl(reconstructed_path, reconstructed_rows)
        audit_tool = _load_audit_module()
        with tempfile.TemporaryDirectory(prefix="duplicate-closure-audit-") as directory:
            audit_dir = Path(directory)
            audit_summary = audit_tool.audit(
                reconstructed_path,
                audit_dir,
                comparison_canonical_path=comparison_canonical_path,
                question_threshold=question_threshold,
                fact_threshold=fact_threshold,
                max_bucket_neighbors=max_bucket_neighbors,
                max_examples=max_examples,
            )
            full_candidate_path = audit_dir / "lexical_candidate_pairs.jsonl"
            raw_candidate_rows = read_jsonl(full_candidate_path, allow_empty=True)

        audit_summary, full_candidate_rows, full_candidate_sha256 = (
            _normalize_audit_outputs(
                audit_summary=audit_summary,
                candidate_rows=raw_candidate_rows,
                reconstructed_path=reconstructed_path,
            )
        )

        closure_pairs, closure_stats, _ = _closure_from_pairs(
            full_candidate_rows, cohort_ids
        )
        write_jsonl(closure_pairs_path, closure_pairs)

        full_binding = file_binding(
            full_pool_path,
            record_count=len(full_rows),
            schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
        )
        cohort_binding = file_binding(
            cohort_bundle_path,
            record_count=len(cohort_rows),
            schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
        )
        canonical_binding = file_binding(
            comparison_canonical_path,
            record_count=int(audit_summary["input"]["comparison_canonical"]["record_count"]),
        )
        manifest: Dict[str, Any] = {
            "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "conservative_candidate_closure_complete",
            "inputs": {
                "authoritative_source_manifest": file_binding(
                    source_manifest_path,
                    schema_version=SOURCE_UNIVERSE_MANIFEST_SCHEMA_VERSION,
                ),
                "full_pool": full_binding,
                "cohort_bundle": cohort_binding,
                "comparison_canonical": canonical_binding,
            },
            "full_pool_authority": {
                "binding_locator": SOURCE_BEHAVIOR_BINDING_LOCATOR,
                "binding_payload_sha256": sha256_value(authoritative_binding),
                "expected_record_count": expected_full_pool_record_count,
            },
            "cohort": {
                "base_fact_count": len(cohort_ids),
                "ordered_base_fact_ids_sha256": sha256_value(cohort_ids),
                "ordered_row_hashes_sha256": sha256_value(
                    [sha256_value(row) for row in cohort_rows]
                ),
            },
            "reconstruction": {
                **file_binding(
                    reconstructed_path,
                    record_count=len(reconstructed_rows),
                    schema_version=INPUT_BUNDLE_SCHEMA_VERSION,
                ),
                "policy": "replace_full_pool_row_by_matching_cohort_base_fact_id_v1",
                "overlaid_row_count": len(cohort_rows),
                "full_pool_order_preserved": True,
            },
            "lexical_audit": {
                "summary_schema_version": AUDIT_SUMMARY_SCHEMA_VERSION,
                "pair_schema_version": AUDIT_PAIR_SCHEMA_VERSION,
                "policy": audit_summary["policy"],
                "full_candidate_pair_count": len(full_candidate_rows),
                "full_candidate_pairs_sha256": full_candidate_sha256,
                "audit_summary_payload_sha256": sha256_value(audit_summary),
                "provenance_normalization_version": (
                    AUDIT_PROVENANCE_NORMALIZATION_VERSION
                ),
                "semantic_review_complete": False,
            },
            "closure": {
                **closure_stats,
                "ordered_pair_ids_sha256": sha256_value(
                    [str(row["pair_id"]) for row in closure_pairs]
                ),
                "candidate_pairs": file_binding(
                    closure_pairs_path,
                    record_count=len(closure_pairs),
                    schema_version=AUDIT_PAIR_SCHEMA_VERSION,
                ),
                "selection_policy": (
                    "all_edges_in_undirected_candidate_components_reachable_"
                    "from_cohort_base_fact_ids_v1"
                ),
            },
            "limitations": {
                "bounded_lexical_candidate_generation_only": True,
                "semantic_equivalence_decisions_performed": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "revisions_must_be_present_in_cohort_overlay_before_generation": True,
            },
            "safety_contract": {
                "network_or_model_used": False,
                "review_freeze_emitted": False,
                "split_freeze_emitted": False,
                "perturbation_authorized": False,
                "human_gold": False,
            },
        }
        write_json(manifest_path, manifest)
        return {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "closure_pair_count": len(closure_pairs),
            "closure_current_pool_node_count": closure_stats[
                "closure_current_pool_node_count"
            ],
            "split_freeze_emitted": False,
            "perturbation_authorized": False,
        }
    except Exception:
        shutil.rmtree(output_dir)
        raise


def _resolve_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    jsonl: bool,
    allow_empty: bool = False,
) -> Tuple[Path, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is invalid")
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is invalid")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte count is stale")
    value: Any = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if jsonl and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record count is stale")
    return path, value


def _validate_candidate_manifest(
    manifest_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported duplicate-closure candidate manifest")
    if manifest.get("status") != "conservative_candidate_closure_complete":
        raise ValueError("Duplicate-closure candidate manifest is incomplete")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or safety.get("perturbation_authorized") is not False:
        raise ValueError("Candidate manifest safety contract is invalid")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Candidate manifest inputs are invalid")
    _, source_manifest = _resolve_binding(
        inputs.get("authoritative_source_manifest"),
        label="authoritative source manifest",
        jsonl=False,
    )
    full_path, full_rows = _resolve_binding(
        inputs.get("full_pool"), label="full pool", jsonl=True
    )
    cohort_path, cohort_rows = _resolve_binding(
        inputs.get("cohort_bundle"), label="cohort bundle", jsonl=True
    )
    canonical_path, _ = _resolve_binding(
        inputs.get("comparison_canonical"),
        label="comparison canonical",
        jsonl=True,
    )
    reconstructed_path, reconstructed_rows = _resolve_binding(
        manifest.get("reconstruction"),
        label="reconstructed full pool",
        jsonl=True,
    )
    closure = manifest.get("closure")
    if not isinstance(closure, dict):
        raise ValueError("Candidate manifest closure is invalid")
    candidate_binding = closure.get("candidate_pairs")
    if (
        not isinstance(candidate_binding, dict)
        or candidate_binding.get("schema_version") != AUDIT_PAIR_SCHEMA_VERSION
    ):
        raise ValueError("Closure candidate pair schema binding is invalid")
    _, closure_pairs = _resolve_binding(
        candidate_binding,
        label="closure candidate pairs",
        jsonl=True,
        allow_empty=True,
    )

    full_index = _validate_bundle_rows(full_rows, "full pool")
    cohort_index = _validate_bundle_rows(cohort_rows, "cohort bundle")
    cohort_ids = list(cohort_index)
    authority = manifest.get("full_pool_authority")
    if not isinstance(authority, dict):
        raise ValueError("Candidate manifest full-pool authority is invalid")
    if authority.get("binding_locator") != SOURCE_BEHAVIOR_BINDING_LOCATOR:
        raise ValueError("Candidate manifest full-pool binding locator is unsupported")
    expected_record_count = authority.get("expected_record_count")
    authoritative_binding = _validate_authoritative_full_pool(
        source_manifest=source_manifest,
        full_pool_path=full_path,
        full_rows=full_rows,
        expected_record_count=expected_record_count,
    )
    if authority.get("binding_payload_sha256") != sha256_value(
        authoritative_binding
    ):
        raise ValueError("Candidate manifest authoritative binding hash is stale")
    if manifest.get("cohort", {}).get("ordered_base_fact_ids_sha256") != sha256_value(
        cohort_ids
    ):
        raise ValueError("Candidate manifest cohort identity order is stale")
    if manifest.get("cohort", {}).get("ordered_row_hashes_sha256") != sha256_value(
        [sha256_value(row) for row in cohort_rows]
    ):
        raise ValueError("Candidate manifest cohort row hashes are stale")
    missing = sorted(set(cohort_ids) - set(full_index))
    if missing:
        raise ValueError("Candidate manifest cohort is not a full-pool subset")
    expected_reconstructed = [
        cohort_index.get(str(row["base_fact_id"]), dict(row)) for row in full_rows
    ]
    if serialize_jsonl(expected_reconstructed) != reconstructed_path.read_bytes():
        raise ValueError("Reconstructed full pool does not equal the bound cohort overlay")

    lexical = manifest.get("lexical_audit")
    if not isinstance(lexical, dict):
        raise ValueError("Candidate manifest lexical audit contract is invalid")
    if lexical.get("summary_schema_version") != AUDIT_SUMMARY_SCHEMA_VERSION:
        raise ValueError("Candidate manifest lexical summary schema is unsupported")
    if lexical.get("pair_schema_version") != AUDIT_PAIR_SCHEMA_VERSION:
        raise ValueError("Candidate manifest lexical pair schema is unsupported")
    if (
        lexical.get("provenance_normalization_version")
        != AUDIT_PROVENANCE_NORMALIZATION_VERSION
    ):
        raise ValueError("Candidate manifest audit provenance normalization is unsupported")
    policy = lexical.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Candidate manifest lexical audit policy is invalid")
    if policy.get("network_or_model_used") is not False:
        raise ValueError("Candidate manifest unexpectedly permits a model or network")

    audit_tool = _load_audit_module()
    with tempfile.TemporaryDirectory(prefix="duplicate-closure-replay-") as directory:
        audit_dir = Path(directory)
        replay_summary = audit_tool.audit(
            reconstructed_path,
            audit_dir,
            comparison_canonical_path=canonical_path,
            question_threshold=policy.get("question_near_threshold"),
            fact_threshold=policy.get("canonical_fact_near_threshold"),
            max_bucket_neighbors=policy.get("max_bucket_neighbors"),
            max_examples=policy.get("max_examples"),
        )
        replay_path = audit_dir / "lexical_candidate_pairs.jsonl"
        raw_replay_rows = read_jsonl(replay_path, allow_empty=True)
    replay_summary, replay_rows, replay_sha256 = _normalize_audit_outputs(
        audit_summary=replay_summary,
        candidate_rows=raw_replay_rows,
        reconstructed_path=reconstructed_path,
    )
    if replay_summary.get("schema_version") != lexical.get("summary_schema_version"):
        raise ValueError("Lexical audit summary schema changed during replay")
    if replay_summary.get("policy") != policy:
        raise ValueError("Lexical audit policy changed during replay")
    if len(replay_rows) != lexical.get("full_candidate_pair_count"):
        raise ValueError("Full lexical candidate count changed during replay")
    if replay_sha256 != lexical.get("full_candidate_pairs_sha256"):
        raise ValueError("Full lexical candidate bytes changed during replay")
    if sha256_value(replay_summary) != lexical.get("audit_summary_payload_sha256"):
        raise ValueError("Lexical audit summary changed during replay")

    replay_closure, replay_stats, _ = _closure_from_pairs(replay_rows, cohort_ids)
    if serialize_jsonl(replay_closure) != serialize_jsonl(closure_pairs):
        raise ValueError("Closure candidate pairs differ from deterministic replay")
    for key, expected in replay_stats.items():
        if closure.get(key) != expected:
            raise ValueError(f"Closure statistic is stale: {key}")
    if closure.get("ordered_pair_ids_sha256") != sha256_value(
        [str(row["pair_id"]) for row in closure_pairs]
    ):
        raise ValueError("Closure candidate pair order is stale")
    return manifest, cohort_rows, closure_pairs


class DisjointSet:
    def __init__(self, identifiers: Iterable[str]) -> None:
        self.parent = {identifier: identifier for identifier in identifiers}

    def add(self, identifier: str) -> None:
        self.parent.setdefault(identifier, identifier)

    def find(self, identifier: str) -> str:
        self.add(identifier)
        parent = self.parent[identifier]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[identifier] = parent
        return parent

    def union(self, first: str, second: str) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        smaller, larger = sorted((left, right))
        self.parent[larger] = smaller


def _validate_adjudications(
    path: Path,
    candidate_pairs: Sequence[Mapping[str, Any]],
    cohort_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    rows = read_jsonl(path, allow_empty=True)
    pair_index = _unique_index(candidate_pairs, "pair_id", "closure candidates")
    decision_index = _unique_index(rows, "pair_id", "closure adjudications")
    missing = sorted(set(pair_index) - set(decision_index))
    extra = sorted(set(decision_index) - set(pair_index))
    if missing or extra:
        raise ValueError(
            "Closure adjudications must exactly cover candidate pairs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    excluded_by_pair: Dict[str, List[str]] = defaultdict(list)
    validated: List[Dict[str, Any]] = []
    for pair_id in sorted(pair_index):
        pair = pair_index[pair_id]
        row = decision_index[pair_id]
        if row.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
            raise ValueError(f"Invalid adjudication schema: {pair_id}")
        if row.get("candidate_row_sha256") != sha256_value(pair):
            raise ValueError(f"Stale candidate row hash: {pair_id}")
        for side in ("left", "right"):
            provenance = pair[side]["provenance"]
            if row.get(f"{side}_audit_record_id") != provenance.get(
                "audit_record_id"
            ):
                raise ValueError(f"Wrong {side} endpoint identity: {pair_id}")
            if row.get(f"{side}_input_record_sha256") != provenance.get(
                "input_record_sha256"
            ):
                raise ValueError(f"Stale {side} endpoint hash: {pair_id}")
        decision = row.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"Unresolved duplicate-closure decision: {pair_id}")
        if decision == "distinct" and EXACT_DUPLICATE_MATCH_TYPES.intersection(
            pair.get("match_types", [])
        ):
            raise ValueError(f"Exact duplicate candidate cannot be distinct: {pair_id}")
        if row.get("reviewer_type") != "codex_proxy":
            raise ValueError(f"Only codex_proxy adjudications are supported: {pair_id}")
        if row.get("human_gold") is not False:
            raise ValueError(f"codex_proxy adjudication must set human_gold=false: {pair_id}")
        for field in ("reviewer_id", "review_method", "reviewed_at", "rationale"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Adjudication {pair_id} lacks non-empty {field}")
        excluded = row.get("excluded_cohort_base_fact_ids")
        if not isinstance(excluded, list) or any(
            not isinstance(value, str) or not value for value in excluded
        ):
            raise ValueError(f"Adjudication {pair_id} has invalid exclusion list")
        if len(excluded) != len(set(excluded)):
            raise ValueError(f"Adjudication {pair_id} repeats an excluded ID")
        pair_cohort_ids = {
            str(pair[side]["provenance"].get("base_fact_id"))
            for side in ("left", "right")
            if pair[side]["provenance"].get("input_role") == "current_bundle"
            and pair[side]["provenance"].get("base_fact_id") in cohort_ids
        }
        if decision == "exclude_cohort":
            if not excluded or not set(excluded) <= pair_cohort_ids:
                raise ValueError(
                    f"exclude_cohort must name a cohort endpoint for {pair_id}"
                )
            for base_fact_id in excluded:
                excluded_by_pair[base_fact_id].append(pair_id)
        elif excluded:
            raise ValueError(
                f"Only exclude_cohort may name excluded cohort IDs: {pair_id}"
            )
        validated.append(row)
    return validated, {key: sorted(value) for key, value in excluded_by_pair.items()}


def _relation_id(row: Mapping[str, Any]) -> Tuple[str, str]:
    for field in ("probe_relation_id", "probe_relation_candidate"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    raise ValueError(
        f"Cohort row {row.get('base_fact_id')} lacks probe_relation_id/candidate"
    )


def _input_split_groups(
    cohort_by_id: Mapping[str, Mapping[str, Any]],
    cohort_ids: Sequence[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for base_fact_id in cohort_ids:
        value = cohort_by_id[base_fact_id].get("split_group_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Cohort row {base_fact_id} lacks split_group_id")
        groups[value.strip()].append(base_fact_id)
    ordered_groups = [
        {
            "split_group_id": split_group_id,
            "ordered_cohort_base_fact_ids": members,
        }
        for split_group_id, members in sorted(groups.items())
    ]
    return dict(groups), {
        "source_field": "split_group_id",
        "source_semantics": "upstream_provisional_normalized_answer_group",
        "union_policy": "unconditional_union_before_provisional_resplit_v1",
        "group_count": len(groups),
        "multi_member_group_count": sum(len(members) > 1 for members in groups.values()),
        "ordered_group_members_sha256": sha256_value(ordered_groups),
    }


def _integer_targets(total: int) -> Dict[str, int]:
    raw = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
    targets = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(targets.values())
    order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - targets[split]), SPLITS.index(split)),
    )
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def _assign_splits(
    components: List[Dict[str, Any]],
    cohort_by_id: Mapping[str, Mapping[str, Any]],
    seed: str,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    relation_totals: Counter[str] = Counter()
    for component in components:
        counts: Counter[str] = Counter()
        relation_fields: Dict[str, str] = {}
        for base_fact_id in component["retained_cohort_base_fact_ids"]:
            relation, field = _relation_id(cohort_by_id[base_fact_id])
            counts[relation] += 1
            relation_fields[relation] = field
        component["relation_counts"] = dict(sorted(counts.items()))
        component["relation_source_fields"] = dict(sorted(relation_fields.items()))
        relation_totals.update(counts)

    targets = {
        relation: _integer_targets(total)
        for relation, total in sorted(relation_totals.items())
    }
    current = {
        relation: {split: 0 for split in SPLITS} for relation in targets
    }
    ordered = sorted(
        components,
        key=lambda component: (
            -len(component["retained_cohort_base_fact_ids"]),
            -len(component["relation_counts"]),
            hashlib.sha256(
                f"{seed}\u241f{component['component_id']}".encode("utf-8")
            ).hexdigest(),
            component["component_id"],
        ),
    )
    assignment: Dict[str, str] = {}
    for component in ordered:
        weights = component["relation_counts"]
        choices = []
        for split in SPLITS:
            overflow = sum(
                max(0, current[relation][split] + count - targets[relation][split])
                for relation, count in weights.items()
            )
            filled_deficit = sum(
                min(count, max(0, targets[relation][split] - current[relation][split]))
                for relation, count in weights.items()
            )
            projected_abs_deviation = 0
            for relation in targets:
                for candidate_split in SPLITS:
                    projected = current[relation][candidate_split]
                    if candidate_split == split:
                        projected += int(weights.get(relation, 0))
                    projected_abs_deviation += abs(
                        targets[relation][candidate_split] - projected
                    )
            tie_break = hashlib.sha256(
                f"{seed}\u241f{component['component_id']}\u241f{split}".encode(
                    "utf-8"
                )
            ).hexdigest()
            choices.append(
                (
                    overflow,
                    -filled_deficit,
                    projected_abs_deviation,
                    tie_break,
                    split,
                )
            )
        selected = min(choices)[-1]
        assignment[component["component_id"]] = selected
        component["provisional_split_assignment"] = selected
        for relation, count in weights.items():
            current[relation][selected] += int(count)

    total_abs_deviation = sum(
        abs(current[relation][split] - targets[relation][split])
        for relation in targets
        for split in SPLITS
    )
    return assignment, {
        "relation_totals": dict(sorted(relation_totals.items())),
        "target_relation_split_counts": targets,
        "actual_relation_split_counts": current,
        "exact_relation_targets_achieved": total_abs_deviation == 0,
        "total_absolute_relation_count_deviation": total_abs_deviation,
    }


def resolve_candidates(
    *,
    candidate_manifest_path: Path,
    adjudications_path: Path,
    output_dir: Path,
    split_seed: str = DEFAULT_SPLIT_SEED,
) -> Dict[str, Any]:
    """Resolve a complete candidate closure into non-frozen components/splits."""

    if not isinstance(split_seed, str) or not split_seed:
        raise ValueError("split_seed must be a non-empty string")
    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    adjudications_path = Path(adjudications_path).resolve()
    manifest, cohort_rows, candidate_pairs = _validate_candidate_manifest(
        candidate_manifest_path
    )
    cohort_by_id = _validate_bundle_rows(cohort_rows, "cohort bundle")
    cohort_ids = list(cohort_by_id)
    adjudications, explicit_exclusions = _validate_adjudications(
        adjudications_path, candidate_pairs, set(cohort_ids)
    )
    pair_by_id = _unique_index(candidate_pairs, "pair_id", "closure candidates")
    decision_by_id = _unique_index(adjudications, "pair_id", "closure adjudications")

    all_nodes = {f"current_bundle:{base_fact_id}" for base_fact_id in cohort_ids}
    node_metadata: Dict[str, Mapping[str, Any]] = {}
    pair_nodes: Dict[str, Tuple[str, str]] = {}
    for pair in candidate_pairs:
        left = _endpoint_key(pair, "left")
        right = _endpoint_key(pair, "right")
        pair_nodes[str(pair["pair_id"])] = (left, right)
        all_nodes.update((left, right))
        node_metadata[left] = pair["left"]["provenance"]
        node_metadata[right] = pair["right"]["provenance"]

    same_fact = DisjointSet(all_nodes)
    leakage = DisjointSet(all_nodes)
    input_split_groups, input_split_group_contract = _input_split_groups(
        cohort_by_id, cohort_ids
    )
    for members in input_split_groups.values():
        anchor = f"current_bundle:{members[0]}"
        for base_fact_id in members[1:]:
            leakage.union(anchor, f"current_bundle:{base_fact_id}")
    for pair_id in sorted(pair_by_id):
        decision = decision_by_id[pair_id]["decision"]
        left, right = pair_nodes[pair_id]
        if decision == "same_fact":
            same_fact.union(left, right)
            leakage.union(left, right)
        elif decision == "same_leakage_component":
            leakage.union(left, right)

    cohort_order = {base_fact_id: index for index, base_fact_id in enumerate(cohort_ids)}
    explicitly_excluded = set(explicit_exclusions)
    fact_groups: Dict[str, List[str]] = defaultdict(list)
    for base_fact_id in cohort_ids:
        if base_fact_id not in explicitly_excluded:
            fact_groups[same_fact.find(f"current_bundle:{base_fact_id}")].append(
                base_fact_id
            )
    duplicate_of: Dict[str, str] = {}
    duplicate_reason_pairs: Dict[str, List[str]] = defaultdict(list)
    for members in fact_groups.values():
        ordered_members = sorted(members, key=lambda value: cohort_order[value])
        if len(ordered_members) < 2:
            continue
        representative = ordered_members[0]
        group_nodes = {
            f"current_bundle:{base_fact_id}" for base_fact_id in ordered_members
        }
        evidence_pairs = sorted(
            pair_id
            for pair_id, (left, right) in pair_nodes.items()
            if decision_by_id[pair_id]["decision"] == "same_fact"
            and same_fact.find(left) == same_fact.find(next(iter(group_nodes)))
            and same_fact.find(right) == same_fact.find(next(iter(group_nodes)))
        )
        for base_fact_id in ordered_members[1:]:
            duplicate_of[base_fact_id] = representative
            duplicate_reason_pairs[base_fact_id] = evidence_pairs

    removed_ids = explicitly_excluded | set(duplicate_of)
    retained_ids = [value for value in cohort_ids if value not in removed_ids]

    leakage_members: Dict[str, Set[str]] = defaultdict(set)
    for node in all_nodes:
        leakage_members[leakage.find(node)].add(node)
    component_rows: List[Dict[str, Any]] = []
    for root, members in leakage_members.items():
        retained = sorted(
            (
                node.split(":", 1)[1]
                for node in members
                if node.startswith("current_bundle:")
                and node.split(":", 1)[1] in set(retained_ids)
            ),
            key=lambda value: cohort_order[value],
        )
        if not retained:
            continue
        ordered_members = sorted(members)
        all_cohort = sorted(
            (
                node.split(":", 1)[1]
                for node in members
                if node.startswith("current_bundle:")
                and node.split(":", 1)[1] in cohort_by_id
            ),
            key=lambda value: cohort_order[value],
        )
        external = sorted(
            node.split(":", 1)[1]
            for node in members
            if node.startswith("current_bundle:")
            and node.split(":", 1)[1] not in cohort_by_id
        )
        canonical = sorted(
            node.split(":", 1)[1]
            for node in members
            if node.startswith("comparison_canonical:")
        )
        component_input_split_group_ids = sorted(
            {
                str(cohort_by_id[base_fact_id]["split_group_id"]).strip()
                for base_fact_id in all_cohort
            }
        )
        supporting_input_split_group_ids = [
            split_group_id
            for split_group_id in component_input_split_group_ids
            if len(input_split_groups[split_group_id]) > 1
        ]
        component_pair_ids = sorted(
            pair_id
            for pair_id, (left, right) in pair_nodes.items()
            if left in members
            and right in members
            and decision_by_id[pair_id]["decision"]
            in {"same_fact", "same_leakage_component"}
        )
        component_id = "leakage_component_" + sha256_value(ordered_members)[:24]
        component_rows.append(
            {
                "schema_version": COMPONENT_SCHEMA_VERSION,
                "component_id": component_id,
                "member_node_ids": ordered_members,
                "member_node_count": len(ordered_members),
                "retained_cohort_base_fact_ids": retained,
                "all_cohort_base_fact_ids": all_cohort,
                "external_current_pool_base_fact_ids": external,
                "comparison_canonical_audit_record_ids": canonical,
                "supporting_union_pair_ids": component_pair_ids,
                "input_split_group_ids": component_input_split_group_ids,
                "supporting_input_split_group_ids": (
                    supporting_input_split_group_ids
                ),
            }
        )
    component_rows.sort(key=lambda row: str(row["component_id"]))
    _, split_stats = _assign_splits(
        component_rows, cohort_by_id, split_seed
    )

    exclusions: List[Dict[str, Any]] = []
    for base_fact_id in cohort_ids:
        if base_fact_id in explicitly_excluded:
            exclusions.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "base_fact_id": base_fact_id,
                    "cohort_index": cohort_order[base_fact_id],
                    "disposition": "excluded_by_adjudication",
                    "representative_base_fact_id": None,
                    "reason_pair_ids": explicit_exclusions[base_fact_id],
                }
            )
        elif base_fact_id in duplicate_of:
            exclusions.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "base_fact_id": base_fact_id,
                    "cohort_index": cohort_order[base_fact_id],
                    "disposition": "deduplicated_same_fact",
                    "representative_base_fact_id": duplicate_of[base_fact_id],
                    "reason_pair_ids": duplicate_reason_pairs[base_fact_id],
                }
            )

    # All validation and deterministic computation above happens before the
    # output directory is created.  Missing/stale decisions therefore leave no
    # misleading partially resolved artifacts.
    output_dir = _prepare_new_output_dir(output_dir)
    components_path = output_dir / "leakage_components.jsonl"
    exclusions_path = output_dir / "dedup_exclusions.jsonl"
    split_path = output_dir / "provisional_split_manifest.json"
    resolution_path = output_dir / "resolution_manifest.json"
    try:
        write_jsonl(components_path, component_rows)
        write_jsonl(exclusions_path, exclusions)
        adjudication_binding = file_binding(
            adjudications_path,
            record_count=len(adjudications),
            schema_version=ADJUDICATION_SCHEMA_VERSION,
        )
        component_binding = file_binding(
            components_path,
            record_count=len(component_rows),
            schema_version=COMPONENT_SCHEMA_VERSION,
        )
        exclusion_binding = file_binding(
            exclusions_path,
            record_count=len(exclusions),
            schema_version=EXCLUSION_SCHEMA_VERSION,
        )
        candidate_manifest_binding = file_binding(candidate_manifest_path)
        split_manifest: Dict[str, Any] = {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "provisional_recomputed_not_frozen",
            "split_status": "provisional_not_frozen",
            "split_policy_version": (
                "relation-stratified-component-greedy-sha256-v2"
            ),
            "split_seed": split_seed,
            "ratios": SPLIT_RATIOS,
            "leakage_grouping_contract": {
                "grouping_sources": [
                    "input_cohort_split_group_id",
                    "adjudicated_same_fact",
                    "adjudicated_same_leakage_component",
                ],
                "input_split_group": input_split_group_contract,
                "same_fact_implies_same_leakage_component": True,
            },
            "candidate_manifest": candidate_manifest_binding,
            "adjudications": adjudication_binding,
            "components": component_binding,
            "dedup_exclusions": exclusion_binding,
            "cohort_counts": {
                "input": len(cohort_ids),
                "retained": len(retained_ids),
                "deduplicated": len(duplicate_of),
                "explicitly_excluded": len(explicitly_excluded),
                "replacement_required_to_restore_input_size": len(cohort_ids)
                - len(retained_ids),
            },
            **split_stats,
            "relation_balance_status": (
                "exact_targets_achieved"
                if split_stats["exact_relation_targets_achieved"]
                else "best_effort_greedy_imbalance_recorded"
            ),
            "component_assignments": [
                {
                    "component_id": row["component_id"],
                    "split_assignment": row["provisional_split_assignment"],
                    "retained_cohort_base_fact_ids": row[
                        "retained_cohort_base_fact_ids"
                    ],
                    "relation_counts": row["relation_counts"],
                }
                for row in component_rows
            ],
            "historical_exposure_status": "not_supplied_pending",
            "semantic_paraphrase_closure_status": "not_performed",
            "unresolved_candidate_count": 0,
            "safety_contract": {
                "component_atomicity_enforced": True,
                "input_split_group_atomicity_enforced": True,
                "copied_provisional_split_assignments": False,
                "network_or_model_used": False,
                "review_freeze_emitted": False,
                "split_freeze_emitted": False,
                "perturbation_authorized": False,
                "human_gold": False,
            },
        }
        write_json(split_path, split_manifest)
        resolution_manifest: Dict[str, Any] = {
            "schema_version": RESOLUTION_MANIFEST_SCHEMA_VERSION,
            "tool_version": TOOL_VERSION,
            "status": "bounded_lexical_closure_resolved_non_frozen",
            "candidate_manifest": candidate_manifest_binding,
            "adjudications": adjudication_binding,
            "outputs": {
                "components": component_binding,
                "dedup_exclusions": exclusion_binding,
                "provisional_split_manifest": file_binding(
                    split_path,
                    schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
                ),
            },
            "counts": split_manifest["cohort_counts"],
            "leakage_grouping_contract": split_manifest[
                "leakage_grouping_contract"
            ],
            "unresolved_candidate_count": 0,
            "semantic_paraphrase_closure_complete": False,
            "historical_exposure_complete": False,
            "formal_bundle_emitted": False,
            "review_freeze_emitted": False,
            "split_freeze_emitted": False,
            "perturbation_authorized": False,
        }
        write_json(resolution_path, resolution_manifest)
        return {
            "manifest_path": str(resolution_path),
            "manifest_sha256": sha256_file(resolution_path),
            "retained_cohort_count": len(retained_ids),
            "dedup_exclusion_count": len(exclusions),
            "component_count": len(component_rows),
            "exact_relation_targets_achieved": split_stats[
                "exact_relation_targets_achieved"
            ],
            "split_freeze_emitted": False,
            "perturbation_authorized": False,
        }
    except Exception:
        shutil.rmtree(output_dir)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    candidates = commands.add_parser(
        "candidates", help="Generate the cohort-seeded conservative closure"
    )
    candidates.add_argument("--full-pool", type=Path, required=True)
    candidates.add_argument("--source-manifest", type=Path, required=True)
    candidates.add_argument(
        "--expected-full-pool-record-count", type=int, required=True
    )
    candidates.add_argument("--cohort-bundle", type=Path, required=True)
    candidates.add_argument("--comparison-canonical", type=Path, required=True)
    candidates.add_argument("--output-dir", type=Path, required=True)
    candidates.add_argument("--question-threshold", type=float, default=0.80)
    candidates.add_argument("--fact-threshold", type=float, default=0.80)
    candidates.add_argument("--max-bucket-neighbors", type=int, default=24)
    candidates.add_argument("--max-examples", type=int, default=20)

    resolve = commands.add_parser(
        "resolve", help="Resolve every closure edge and recompute a provisional split"
    )
    resolve.add_argument("--candidate-manifest", type=Path, required=True)
    resolve.add_argument("--adjudications", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)
    resolve.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "candidates":
        result = materialize_candidates(
            full_pool_path=args.full_pool,
            source_manifest_path=args.source_manifest,
            expected_full_pool_record_count=args.expected_full_pool_record_count,
            cohort_bundle_path=args.cohort_bundle,
            comparison_canonical_path=args.comparison_canonical,
            output_dir=args.output_dir,
            question_threshold=args.question_threshold,
            fact_threshold=args.fact_threshold,
            max_bucket_neighbors=args.max_bucket_neighbors,
            max_examples=args.max_examples,
        )
    else:
        result = resolve_candidates(
            candidate_manifest_path=args.candidate_manifest,
            adjudications_path=args.adjudications,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
