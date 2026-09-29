#!/usr/bin/env python3
"""Rebind reviewed closure decisions to a regenerated candidate manifest.

Endpoint-identical decisions are reused; every new endpoint pair must be
listed explicitly in an override file.  The output is a simplified review
shard for ``materialize_public_benchmark_duplicate_adjudications.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    path.write_bytes(payload)


def endpoint_key(pair: Mapping[str, Any]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    values = []
    for side in ("left", "right"):
        provenance = pair[side]["provenance"]
        identifier = provenance.get("base_fact_id") or provenance.get("audit_record_id")
        values.append((str(provenance["input_role"]), str(identifier)))
    return tuple(sorted(values))  # type: ignore[return-value]


def review_key(row: Mapping[str, Any]) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    values = []
    for side in ("left", "right"):
        endpoint = row[side]
        values.append((str(endpoint["input_role"]), str(endpoint["id"])))
    return tuple(sorted(values))  # type: ignore[return-value]


def rebind_reviews(
    *,
    old_candidates_path: Path,
    old_adjudications_path: Path,
    new_candidates_path: Path,
    overrides_path: Path,
    output_path: Path,
) -> Dict[str, int]:
    old_candidates = {row["pair_id"]: row for row in read_jsonl(old_candidates_path)}
    old_by_key: Dict[Tuple[Tuple[str, str], Tuple[str, str]], Dict[str, Any]] = {}
    for decision in read_jsonl(old_adjudications_path):
        candidate = old_candidates.get(decision["pair_id"])
        if candidate is None:
            raise ValueError(f"Old decision has no candidate: {decision['pair_id']}")
        key = endpoint_key(candidate)
        if key in old_by_key:
            raise ValueError(f"Duplicate old endpoint pair: {key}")
        old_by_key[key] = decision
    overrides = {review_key(row): row for row in read_jsonl(overrides_path)}
    output = []
    reused = 0
    overridden = 0
    used_overrides = set()
    for candidate in read_jsonl(new_candidates_path):
        key = endpoint_key(candidate)
        if key in overrides:
            source = overrides[key]
            used_overrides.add(key)
            overridden += 1
        elif key in old_by_key:
            source = old_by_key[key]
            reused += 1
        else:
            raise ValueError(f"Unreviewed regenerated endpoint pair: {key}")
        output.append(
            {
                "pair_id": candidate["pair_id"],
                "decision": source["decision"],
                "rationale": source["rationale"],
                "reviewer_id": "codex-proxy-cohort-repair-closure-review-v1",
                "confidence": source["confidence"],
            }
        )
    unused = sorted(set(overrides) - used_overrides)
    if unused:
        raise ValueError(f"Unused closure review overrides: {unused}")
    write_jsonl(output_path, output)
    return {"record_count": len(output), "reused": reused, "overridden": overridden}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-candidates", type=Path, required=True)
    parser.add_argument("--old-adjudications", type=Path, required=True)
    parser.add_argument("--new-candidates", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = rebind_reviews(
        old_candidates_path=args.old_candidates,
        old_adjudications_path=args.old_adjudications,
        new_candidates_path=args.new_candidates,
        overrides_path=args.overrides,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
