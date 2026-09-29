#!/usr/bin/env python3
"""Build and replay one exact semantic authority from parent plus supplement.

The parent supplies its completed adjudications and immutable 1,826-row order;
the supplement may replace exactly the parent's failed set.  Neither source is
rewritten.  Every composite row has lineage to a source run contract,
checkpoint row, and adjudication row.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOOL_VERSION = "full-public-benchmark-semantic-review-composite-builder-v1"
MANIFEST_SCHEMA = (
    "public-benchmark-full-semantic-review-composite-authority-manifest-v1"
)
LINEAGE_SCHEMA = (
    "public-benchmark-full-semantic-review-composite-adjudication-lineage-v1"
)
STATUS = "completed_exact_full_candidate_set"
EXPECTED_TOTAL_COUNT = 1826
EXPECTED_PARENT_COMPLETED_COUNT = 1798
EXPECTED_SUPPLEMENT_COUNT = 28


def _load_supplement_runner() -> Any:
    path = Path(__file__).resolve().with_name(
        "run_full_public_benchmark_semantic_review_supplement.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_semantic_review_supplement_for_composite", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load semantic supplement runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


supplement_runner = _load_supplement_runner()
parent_runner = supplement_runner.parent_runner


def canonical_json_bytes(value: Any) -> bytes:
    return parent_runner.canonical_json_bytes(value)


def sha256_value(value: Any) -> str:
    return parent_runner.sha256_value(value)


def sha256_file(path: Path) -> str:
    return parent_runner.sha256_file(path)


def read_json(path: Path) -> Dict[str, Any]:
    return parent_runner.read_json(Path(path))


def read_jsonl(path: Path, *, allow_empty: bool = False) -> List[Dict[str, Any]]:
    return parent_runner.read_jsonl(Path(path), allow_empty=allow_empty)


def _atomic_write(path: Path, payload: bytes) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=str(resolved.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, resolved)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
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
    return supplement_runner._required_string(value, label)


def _binding(
    path: Path,
    *,
    schema_version: Optional[str] = None,
    record_count: Optional[int] = None,
    relative_to: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    result: Dict[str, Any] = {
        ("filename" if relative_to is not None else "path"): (
            resolved.relative_to(Path(relative_to).resolve()).as_posix()
            if relative_to is not None
            else str(resolved)
        ),
        "sha256": sha256_file(resolved),
        "byte_count": resolved.stat().st_size,
    }
    if schema_version is not None:
        result["schema_version"] = schema_version
    if record_count is not None:
        result["record_count"] = record_count
    return result


def _resolve_binding_path(
    binding: Mapping[str, Any], owner_path: Path, label: str
) -> Path:
    raw = binding.get("path") or binding.get("filename")
    candidate = Path(_required_string(raw, f"{label}.path"))
    if not candidate.is_absolute():
        candidate = Path(owner_path).resolve().parent / candidate
    return candidate.resolve()


def _verify_binding(
    binding: Any,
    *,
    owner_path: Path,
    label: str,
    expected_schema: Optional[str] = None,
    jsonl: bool = False,
    allow_empty: bool = False,
) -> Tuple[Path, Any, Dict[str, Any]]:
    if not isinstance(binding, dict):
        raise ValueError(f"{label} binding is missing")
    path = _resolve_binding_path(binding, owner_path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA-256 is stale")
    if "byte_count" in binding and binding.get("byte_count") != path.stat().st_size:
        raise ValueError(f"{label} byte_count is stale")
    if expected_schema is not None and binding.get("schema_version") != expected_schema:
        raise ValueError(f"{label} schema_version is unsupported")
    value = read_jsonl(path, allow_empty=allow_empty) if jsonl else read_json(path)
    if jsonl and binding.get("record_count") != len(value):
        raise ValueError(f"{label} record_count is stale")
    return path, value, dict(binding)


def _unique_index(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = _required_string(row.get(field), f"{label}.{field}")
        if key in output:
            raise ValueError(f"Duplicate {field} in {label}: {key}")
        output[key] = dict(row)
    return output


def _runtime_source_bindings() -> Dict[str, Dict[str, Any]]:
    return {
        "composite_builder": _binding(Path(__file__).resolve()),
        "supplement_runner": _binding(Path(supplement_runner.__file__).resolve()),
        "parent_runner": _binding(Path(parent_runner.__file__).resolve()),
    }


def _source_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for path in paths:
        resolved = Path(path).resolve()
        output[str(resolved)] = sha256_file(resolved)
    return output


def _assert_hashes_current(snapshots: Mapping[str, str]) -> None:
    for raw_path, digest in snapshots.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Source authority changed during composition: {path}")


def _authority_summary(authority: Mapping[str, Any], *, role: str) -> Dict[str, Any]:
    if role == "parent":
        parent = authority
        return {
            "run_manifest": parent.inputs["parent_run_manifest"],
            "run_contract_sha256": parent.manifest["run_contract_sha256"],
            "checkpoint": parent.inputs["parent_checkpoint"],
            "adjudications": parent.inputs["parent_adjudications"],
            "completed_pair_ids_sha256": sha256_value(parent.completed_ids),
            "failed_pair_ids": list(parent.failed_ids),
            "failed_pair_ids_sha256": sha256_value(parent.failed_ids),
            "route_identity": copy.deepcopy(
                parent.manifest["run_contract"]["route_identity"]
            ),
        }
    supplement = authority
    return {
        "run_manifest": supplement["manifest_binding"],
        "run_contract_sha256": supplement["run_contract_sha256"],
        "checkpoint": supplement["checkpoint_binding"],
        "adjudications": supplement["adjudications_binding"],
        "selected_pair_ids": list(supplement["completed_ids"]),
        "selected_pair_ids_sha256": sha256_value(supplement["completed_ids"]),
        "route_identity": copy.deepcopy(supplement["route_identity"]),
    }


def _expected_composition(
    parent: Any, supplement: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parent_decisions = {
        str(row["pair_id"]): dict(row) for row in parent.adjudications
    }
    supplement_decisions = {
        str(row["pair_id"]): dict(row) for row in supplement["adjudications"]
    }
    if set(parent_decisions).intersection(supplement_decisions):
        raise ValueError("Parent and supplement semantic adjudications overlap")
    if list(supplement["completed_ids"]) != list(parent.failed_ids):
        raise ValueError("Supplement IDs do not exactly equal parent failed IDs")
    ordered_ids = [str(unit.pair_id) for unit in parent.units]
    if set(ordered_ids) != set(parent_decisions) | set(supplement_decisions):
        raise ValueError("Composite semantic union has missing or unknown IDs")
    parent_checkpoints = _unique_index(
        parent.checkpoints, "pair_id", "parent semantic checkpoint"
    )
    supplement_checkpoints = _unique_index(
        supplement["checkpoints"], "pair_id", "supplement semantic checkpoint"
    )
    decisions: List[Dict[str, Any]] = []
    lineage: List[Dict[str, Any]] = []
    for input_index, pair_id in enumerate(ordered_ids):
        if pair_id in parent_decisions:
            source = "parent_completed"
            decision = parent_decisions[pair_id]
            checkpoint = parent_checkpoints[pair_id]
            source_manifest = parent.inputs["parent_run_manifest"]
            source_contract_sha = parent.manifest["run_contract_sha256"]
        else:
            source = "supplement_recovery"
            decision = supplement_decisions[pair_id]
            checkpoint = supplement_checkpoints[pair_id]
            source_manifest = supplement["manifest_binding"]
            source_contract_sha = supplement["run_contract_sha256"]
        decisions.append(dict(decision))
        lineage.append(
            {
                "schema_version": LINEAGE_SCHEMA,
                "input_index": input_index,
                "pair_id": pair_id,
                "source_authority": source,
                "source_run_manifest_sha256": source_manifest["sha256"],
                "source_run_contract_sha256": source_contract_sha,
                "source_checkpoint_record_sha256": sha256_value(checkpoint),
                "source_adjudication_record_sha256": sha256_value(decision),
                "behavior_blind": True,
                "human_gold": False,
            }
        )
    return decisions, lineage


def build_composite_authority(
    *,
    parent_run_manifest_path: Path,
    supplement_run_manifest_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    parent = supplement_runner.inspect_parent_authority(parent_run_manifest_path)
    supplement = supplement_runner.load_supplement_authority(
        supplement_run_manifest_path, parent=parent
    )
    decisions, lineage = _expected_composition(parent, supplement)
    if len(decisions) != len(parent.units) or len(lineage) != len(parent.units):
        raise ValueError("Composite semantic authority does not cover its parent")

    source_paths = {
        Path(binding["path"])
        for binding in parent.inputs.values()
    } | {
        Path(supplement["manifest_path"]),
        Path(supplement["checkpoint_path"]),
        Path(supplement["adjudications_path"]),
    }
    snapshots = _source_hashes(sorted(source_paths, key=str))

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    decisions_path = output_dir / "composite_semantic_adjudications.jsonl"
    lineage_path = output_dir / "composite_semantic_adjudication_lineage.jsonl"
    manifest_path = output_dir / "semantic_review_composite_authority_manifest.json"
    try:
        write_jsonl(decisions_path, decisions)
        write_jsonl(lineage_path, lineage)
        decisions_binding = _binding(
            decisions_path,
            schema_version=parent_runner.ADJUDICATION_SCHEMA,
            record_count=len(decisions),
            relative_to=output_dir,
        )
        lineage_binding = _binding(
            lineage_path,
            schema_version=LINEAGE_SCHEMA,
            record_count=len(lineage),
            relative_to=output_dir,
        )
        parent_summary = _authority_summary(parent, role="parent")
        supplement_summary = _authority_summary(supplement, role="supplement")
        runtime_sources = _runtime_source_bindings()
        authority_payload = {
            "parent_run_manifest_sha256": parent_summary["run_manifest"]["sha256"],
            "parent_run_contract_sha256": parent_summary["run_contract_sha256"],
            "supplement_run_manifest_sha256": supplement_summary["run_manifest"]["sha256"],
            "supplement_run_contract_sha256": supplement_summary["run_contract_sha256"],
            "composite_adjudications_sha256": decisions_binding["sha256"],
            "lineage_sha256": lineage_binding["sha256"],
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "tool_version": TOOL_VERSION,
            "status": STATUS,
            "authority_id": "semantic_review_composite_"
            + sha256_value(authority_payload)[:20],
            "formal_universe_id": parent.manifest["run_contract"].get(
                "formal_universe_id"
            ),
            "candidate_manifest": parent.inputs["candidate_manifest"],
            "candidate_pairs": parent.inputs["candidate_pairs"],
            "adjudication_template": parent.inputs["adjudication_template"],
            "runtime_sources": runtime_sources,
            "parent_authority": parent_summary,
            "supplement_authority": supplement_summary,
            "counts": {
                "parent_completed": len(parent.completed_ids),
                "parent_failed": len(parent.failed_ids),
                "supplement_completed": len(supplement["completed_ids"]),
                "composite_adjudications": len(parent.units),
            },
            "composition": {
                "ordered_pair_ids_sha256": sha256_value(
                    [str(unit.pair_id) for unit in parent.units]
                ),
                "parent_and_supplement_overlap_count": 0,
                "missing_pair_count": 0,
                "unknown_pair_count": 0,
                "supplement_exactly_equals_parent_failed_set": True,
                "parent_completed_rows_copied_without_change": True,
                "source_authority_artifacts_modified": False,
            },
            "artifacts": {
                "adjudications": decisions_binding,
                "adjudication_lineage": lineage_binding,
            },
            "evidence_contract": {
                "authority_mode": "composite_semantic_review_authority",
                "parent_and_supplement_run_contracts_preserved": True,
                "parent_and_supplement_route_identity_sets_preserved": True,
                "single_global_route_identity_claimed": False,
                "infrastructure_independence_claimed": False,
                "behavior_blind": True,
                "human_gold": False,
            },
            "human_gold": False,
            "safety_contract": {
                "source_run_artifacts_modified": False,
                "external_api_or_model_used": False,
                "behavior_execution_performed": False,
                "hf_model_execution_performed": False,
                "hf_tokenizer_execution_performed": False,
                "validation_exposed": False,
                "sealed_exposed": False,
                "allow_partial_apply_authorized": False,
                "human_gold": False,
            },
        }
        write_json(manifest_path, manifest)
        _assert_hashes_current(snapshots)
        loaded = load_composite_authority(manifest_path)
        return {
            "status": loaded["manifest"]["status"],
            "authority_id": loaded["manifest"]["authority_id"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "adjudications_path": str(decisions_path),
            "adjudications_sha256": sha256_file(decisions_path),
            "adjudication_count": len(decisions),
        }
    except BaseException:
        shutil.rmtree(output_dir)
        raise


def load_composite_authority(path: Path) -> Dict[str, Any]:
    """Fully replay a composite from both immutable source authorities."""

    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Semantic composite manifest schema is unsupported")
    if manifest.get("tool_version") != TOOL_VERSION or manifest.get("status") != STATUS:
        raise ValueError("Semantic composite authority is incomplete")
    if manifest.get("runtime_sources") != _runtime_source_bindings():
        raise ValueError("Semantic composite runtime source bindings are stale")
    if manifest.get("human_gold") is not False:
        raise ValueError("Semantic composite must keep human_gold=false")
    parent_summary = manifest.get("parent_authority")
    supplement_summary = manifest.get("supplement_authority")
    if not isinstance(parent_summary, dict) or not isinstance(supplement_summary, dict):
        raise ValueError("Semantic composite source authorities are missing")
    parent_manifest_path, _, _ = _verify_binding(
        parent_summary.get("run_manifest"),
        owner_path=path,
        label="semantic composite parent run manifest",
        expected_schema=parent_runner.RUN_MANIFEST_SCHEMA,
    )
    parent = supplement_runner.inspect_parent_authority(parent_manifest_path)
    if parent_summary != _authority_summary(parent, role="parent"):
        raise ValueError("Semantic composite parent authority is stale")
    expected_formal_universe_id = parent.manifest["run_contract"].get(
        "formal_universe_id"
    )
    if manifest.get("formal_universe_id") != expected_formal_universe_id:
        raise ValueError("Semantic composite formal_universe_id is stale")
    supplement_manifest_path, _, _ = _verify_binding(
        supplement_summary.get("run_manifest"),
        owner_path=path,
        label="semantic composite supplement run manifest",
        expected_schema=supplement_runner.RUN_MANIFEST_SCHEMA,
    )
    supplement = supplement_runner.load_supplement_authority(
        supplement_manifest_path, parent=parent
    )
    if supplement_summary != _authority_summary(supplement, role="supplement"):
        raise ValueError("Semantic composite supplement authority is stale")
    for label in ("candidate_manifest", "candidate_pairs", "adjudication_template"):
        supplement_runner._same_bound_file(
            manifest.get(label),
            parent.inputs[label],
            actual_owner=path,
            expected_owner=parent.manifest_path,
            label=f"semantic composite {label}",
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Semantic composite artifacts are missing")
    adjudications_path, adjudications, adjudications_binding = _verify_binding(
        artifacts.get("adjudications"),
        owner_path=path,
        label="semantic composite adjudications",
        expected_schema=parent_runner.ADJUDICATION_SCHEMA,
        jsonl=True,
    )
    lineage_path, lineage, lineage_binding = _verify_binding(
        artifacts.get("adjudication_lineage"),
        owner_path=path,
        label="semantic composite adjudication lineage",
        expected_schema=LINEAGE_SCHEMA,
        jsonl=True,
    )
    if len(adjudications) != len(parent.units) or len(lineage) != len(parent.units):
        raise ValueError("Semantic composite does not exactly cover its parent")
    expected_adjudications, expected_lineage = _expected_composition(parent, supplement)
    if adjudications != expected_adjudications or lineage != expected_lineage:
        raise ValueError("Semantic composite does not replay from bound authorities")
    ordered_ids = [str(row["pair_id"]) for row in adjudications]
    _unique_index(adjudications, "pair_id", "semantic composite adjudications")
    _unique_index(lineage, "pair_id", "semantic composite lineage")
    for input_index, (decision, row) in enumerate(zip(adjudications, lineage)):
        if not all(
            (
                row.get("schema_version") == LINEAGE_SCHEMA,
                row.get("input_index") == input_index,
                row.get("pair_id") == decision.get("pair_id"),
                row.get("source_authority")
                in {"parent_completed", "supplement_recovery"},
                row.get("source_adjudication_record_sha256")
                == sha256_value(decision),
                row.get("behavior_blind") is True,
                row.get("human_gold") is False,
            )
        ):
            raise ValueError(f"Semantic composite lineage is invalid at {input_index}")
    if manifest.get("counts") != {
        "parent_completed": len(parent.completed_ids),
        "parent_failed": len(parent.failed_ids),
        "supplement_completed": len(supplement["completed_ids"]),
        "composite_adjudications": len(parent.units),
    }:
        raise ValueError("Semantic composite counts are stale")
    composition = manifest.get("composition")
    if not isinstance(composition, dict) or not all(
        (
            composition.get("ordered_pair_ids_sha256") == sha256_value(ordered_ids),
            composition.get("parent_and_supplement_overlap_count") == 0,
            composition.get("missing_pair_count") == 0,
            composition.get("unknown_pair_count") == 0,
            composition.get("supplement_exactly_equals_parent_failed_set") is True,
            composition.get("parent_completed_rows_copied_without_change") is True,
            composition.get("source_authority_artifacts_modified") is False,
        )
    ):
        raise ValueError("Semantic composite composition contract is invalid")
    evidence = manifest.get("evidence_contract")
    if not isinstance(evidence, dict) or not all(
        (
            evidence.get("authority_mode") == "composite_semantic_review_authority",
            evidence.get("parent_and_supplement_run_contracts_preserved") is True,
            evidence.get("parent_and_supplement_route_identity_sets_preserved") is True,
            evidence.get("single_global_route_identity_claimed") is False,
            evidence.get("infrastructure_independence_claimed") is False,
            evidence.get("behavior_blind") is True,
            evidence.get("human_gold") is False,
        )
    ):
        raise ValueError("Semantic composite evidence contract is invalid")
    safety = manifest.get("safety_contract")
    if not isinstance(safety, dict) or not all(
        (
            safety.get("source_run_artifacts_modified") is False,
            safety.get("external_api_or_model_used") is False,
            safety.get("behavior_execution_performed") is False,
            safety.get("hf_model_execution_performed") is False,
            safety.get("hf_tokenizer_execution_performed") is False,
            safety.get("validation_exposed") is False,
            safety.get("sealed_exposed") is False,
            safety.get("allow_partial_apply_authorized") is False,
            safety.get("human_gold") is False,
        )
    ):
        raise ValueError("Semantic composite safety contract is invalid")
    expected_authority_id = "semantic_review_composite_" + sha256_value(
        {
            "parent_run_manifest_sha256": parent_summary["run_manifest"]["sha256"],
            "parent_run_contract_sha256": parent_summary["run_contract_sha256"],
            "supplement_run_manifest_sha256": supplement_summary["run_manifest"]["sha256"],
            "supplement_run_contract_sha256": supplement_summary["run_contract_sha256"],
            "composite_adjudications_sha256": adjudications_binding["sha256"],
            "lineage_sha256": lineage_binding["sha256"],
        }
    )[:20]
    if manifest.get("authority_id") != expected_authority_id:
        raise ValueError("Semantic composite authority_id is stale")

    snapshot_bindings: List[Dict[str, Any]] = [
        _binding(path, schema_version=MANIFEST_SCHEMA),
        _binding(
            adjudications_path,
            schema_version=parent_runner.ADJUDICATION_SCHEMA,
            record_count=len(adjudications),
        ),
        _binding(
            lineage_path, schema_version=LINEAGE_SCHEMA, record_count=len(lineage)
        ),
        *[copy.deepcopy(binding) for binding in parent.inputs.values()],
        copy.deepcopy(supplement["manifest_binding"]),
        copy.deepcopy(supplement["checkpoint_binding"]),
        copy.deepcopy(supplement["adjudications_binding"]),
    ]
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_binding": _binding(path, schema_version=MANIFEST_SCHEMA),
        "adjudications": adjudications,
        "adjudications_path": adjudications_path,
        "adjudications_binding": adjudications_binding,
        "lineage": lineage,
        "lineage_path": lineage_path,
        "lineage_binding": lineage_binding,
        "parent": parent,
        "supplement": supplement,
        "snapshot_bindings": snapshot_bindings,
        "source_run_contract_sha256": {
            "parent": parent.manifest["run_contract_sha256"],
            "supplement": supplement["run_contract_sha256"],
        },
        "source_route_identity_set_sha256": {
            "parent": parent.manifest["run_contract"]["route_identity"][
                "route_identity_set_sha256"
            ],
            "supplement": supplement["route_identity"][
                "route_identity_set_sha256"
            ],
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parent-run-manifest", type=Path, required=True)
    result.add_argument("--supplement-run-manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    result = build_composite_authority(
        parent_run_manifest_path=args.parent_run_manifest,
        supplement_run_manifest_path=args.supplement_run_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
