#!/usr/bin/env python3
"""Compare independent Qwen3 MCQ attribution runs and select the canonical run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ARTIFACT_KEYS = ("metrics", "activation_index", "resid_pre")


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--superseded-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_run(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "mcq_decision_attribution_manifest.json"
    manifest = read_json(manifest_path)
    failures = []
    if manifest.get("schema_version") != "qwen3-mcq-decision-attribution-manifest-v2":
        failures.append("schema_not_v2")
    if manifest.get("status") != "complete" or manifest.get("render_count") != 960:
        failures.append("run_incomplete")
    if manifest.get("activation_shape") != [960, 36, 4096]:
        failures.append("activation_shape_changed")
    parity = manifest.get("forward_parity", {})
    if parity.get("exact_global_rank_mismatch_count") != 0:
        failures.append("final_rank_mismatch")
    if parity.get("choice_order_mismatch_render_count") != 0:
        failures.append("choice_order_mismatch")
    artifact_bindings = {}
    for key in ARTIFACT_KEYS:
        declared = manifest.get("artifacts", {}).get(key, {})
        path = run_dir / str(declared.get("path", ""))
        if not path.is_file():
            failures.append(f"missing_{key}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != declared.get("sha256"):
            failures.append(f"sha_mismatch_{key}")
        artifact_bindings[key] = {
            "path": str(path),
            "byte_count": path.stat().st_size,
            "sha256": actual_sha,
        }
    return {
        "directory": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "artifacts": artifact_bindings,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    canonical = verify_run(args.canonical_dir.resolve())
    repeat = verify_run(args.repeat_dir.resolve())
    failures = [
        *(f"canonical:{value}" for value in canonical["failures"]),
        *(f"repeat:{value}" for value in repeat["failures"]),
    ]
    comparisons = {}
    for key in ARTIFACT_KEYS:
        matches = canonical["artifacts"].get(key, {}).get("sha256") == repeat["artifacts"].get(
            key, {}
        ).get("sha256")
        comparisons[f"{key}_sha256_identical"] = matches
        if not matches:
            failures.append(f"repeat_sha_mismatch:{key}")
    inputs_identical = canonical["manifest"].get("inputs") == repeat["manifest"].get("inputs")
    comparisons["input_bindings_identical"] = inputs_identical
    if not inputs_identical:
        failures.append("repeat_input_bindings_changed")

    superseded = None
    if args.superseded_dir is not None:
        superseded_dir = args.superseded_dir.resolve()
        superseded_manifest_path = superseded_dir / "mcq_decision_attribution_manifest.json"
        superseded_manifest = read_json(superseded_manifest_path)
        superseded = {
            "directory": str(superseded_dir),
            "manifest_sha256": sha256_file(superseded_manifest_path),
            "schema_version": superseded_manifest.get("schema_version"),
            "forward_parity": superseded_manifest.get("forward_parity"),
            "authoritative": False,
            "reason": "single-prefix forward shape did not exactly replay behavior scoring",
        }

    output = {
        "schema_version": "qwen3-mcq-decision-attribution-stability-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_frozen_runtime_reproduction" if not failures else "failed",
        "failures": failures,
        "canonical": {
            "directory": canonical["directory"],
            "manifest_sha256": canonical["manifest_sha256"],
            "artifacts": canonical["artifacts"],
        },
        "repeat": {
            "directory": repeat["directory"],
            "manifest_sha256": repeat["manifest_sha256"],
            "artifacts": repeat["artifacts"],
        },
        "comparisons": comparisons,
        "superseded": superseded,
        "authorization_state": {
            "development_mcq_attribution_complete": not failures,
            "development_vector_intervention_authorized": False,
            "validation_authorized": False,
            "sealed_authorized": False,
        },
    }
    write_json_atomic(args.output.resolve(), output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
