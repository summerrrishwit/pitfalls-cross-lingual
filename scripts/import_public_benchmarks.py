#!/usr/bin/env python3
"""Import revision-pinned public benchmarks into the raw QA schema."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "configs" / "public_benchmark_sources_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "source" / "public-benchmarks-v1"
USER_AGENT = "pitfalls-cross-lingual-public-benchmark-import/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sources", nargs="+", default=None)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--records-per-source", type=int, default=None)
    size.add_argument("--all", action="store_true", help="Import every eligible source record.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--cache-dir", default=None, help="Cache Global-MMLU pages for resumable downloads.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_catalog(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("catalog_version") != "public-benchmark-sources-v1":
        raise ValueError("unsupported public benchmark catalog version")
    if not isinstance(catalog.get("sources"), dict) or not catalog["sources"]:
        raise ValueError("catalog.sources must be a non-empty object")
    return catalog


def fetch_bytes(
    session: requests.Session,
    url: str,
    timeout_seconds: float,
    expected_sha256: Optional[str] = None,
) -> Tuple[bytes, str]:
    response = get_with_retry(session, url, timeout_seconds=timeout_seconds)
    response.raise_for_status()
    payload = response.content
    digest = sha256_bytes(payload)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {digest}")
    return payload, digest


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout_seconds: float,
    params: Optional[Dict[str, Any]] = None,
    max_attempts: int = 8,
) -> requests.Response:
    response: Optional[requests.Response] = None
    for attempt in range(max_attempts):
        response = session.get(url, params=params, timeout=timeout_seconds)
        if response.status_code != 429 and response.status_code < 500:
            return response
        if attempt + 1 < max_attempts:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 30)
            delay = min(delay, 30)
            time.sleep(delay)
    assert response is not None
    return response


def fetch_global_mmlu(
    session: requests.Session,
    spec: Dict[str, Any],
    timeout_seconds: float,
    records_per_source: Optional[int],
    seed: int,
    source_name: str,
    cache_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    expected_total = spec.get("expected_record_count")
    if not isinstance(expected_total, int) or expected_total <= 0:
        raise ValueError("Global-MMLU requires a positive expected_record_count")
    if records_per_source is None:
        requests_to_make = [(offset, min(100, expected_total - offset)) for offset in range(0, expected_total, 100)]
        selected_indices: Optional[List[int]] = None
    else:
        derived_seed = int(
            hashlib.sha256(f"{seed}:{source_name}".encode("utf-8")).hexdigest()[:16], 16
        )
        selected_indices = sorted(
            random.Random(derived_seed).sample(range(expected_total), records_per_source)
        )
        requests_to_make = [(offset, 1) for offset in selected_indices]
    total: Optional[int] = None
    page_cache = cache_dir / "global_mmlu" / spec["revision"]
    page_cache.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    for request_number, (offset, length) in enumerate(requests_to_make, start=1):
        cache_path = page_cache / f"offset-{offset:06d}-length-{length:03d}.json"
        if cache_path.exists():
            page = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_hits += 1
        else:
            response = get_with_retry(
                session,
                spec["url"],
                timeout_seconds=timeout_seconds,
                params={
                    "dataset": "CohereLabs/Global-MMLU",
                    "config": spec["config"],
                    "split": spec["split"],
                    "offset": offset,
                    "length": length,
                    "revision": spec["revision"],
                },
            )
            response.raise_for_status()
            page = response.json()
            write_json_atomic(cache_path, page)
        page_rows = page.get("rows")
        if not isinstance(page_rows, list):
            raise ValueError("Global-MMLU rows response does not contain a rows list")
        page_total = page.get("num_rows_total")
        if not isinstance(page_total, int):
            raise ValueError("Global-MMLU rows response does not contain num_rows_total")
        if total is None:
            total = page_total
        elif total != page_total:
            raise ValueError("Global-MMLU row count changed during the pinned download")
        rows.extend(page_rows)
        if records_per_source is None and (request_number % 25 == 0 or request_number == len(requests_to_make)):
            print(
                f"Global-MMLU download: {len(rows)} / {expected_total} rows "
                f"({request_number}/{len(requests_to_make)} pages)",
                flush=True,
            )
    if total != expected_total:
        raise ValueError(f"Global-MMLU record count mismatch: expected {expected_total}, got {total}")
    expected_downloaded = expected_total if records_per_source is None else records_per_source
    if len(rows) != expected_downloaded:
        raise ValueError(
            f"Global-MMLU download incomplete: expected {expected_downloaded}, got {len(rows)}"
        )
    retrieval = {
        "retrieval_kind": "huggingface_datasets_server_rows",
        "retrieved_payload_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "page_count": len(requests_to_make),
        "cache_hit_count": cache_hits,
        "eligibility_scope": "all_upstream_rows" if records_per_source is None else "selected_rows_only",
    }
    if selected_indices is not None:
        retrieval["selected_row_indices"] = selected_indices
    return rows, retrieval


def parse_jsonl(payload: bytes) -> List[Dict[str, Any]]:
    records = []
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        records.append(record)
    return records


def fetch_direct_source(
    session: requests.Session, spec: Dict[str, Any], timeout_seconds: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload, digest = fetch_bytes(
        session, spec["url"], timeout_seconds, expected_sha256=spec.get("expected_sha256")
    )
    adapter = spec["adapter"]
    if adapter == "mkqa_jsonl_gz":
        records = parse_jsonl(gzip.decompress(payload))
    elif adapter == "msqa_jsonl":
        records = parse_jsonl(payload)
    elif adapter == "openbookqa_zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            records = parse_jsonl(archive.read(spec["archive_member"]))
    else:
        raise ValueError(f"unsupported direct-download adapter: {adapter}")
    return records, {"retrieval_kind": "direct_file", "download_sha256": digest}


def clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def normalized_record(
    *,
    question: Any,
    answer: Any,
    source: str,
    source_format: str,
    subject: Any,
    choices: Optional[Iterable[Any]],
    upstream_id: Any,
    revision: str,
    metadata: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    clean_question = clean_text(question)
    clean_answer = clean_text(answer)
    if not clean_question:
        return None, "missing_or_empty_question"
    if not clean_answer:
        return None, "missing_or_empty_answer"
    clean_choices = [clean_text(choice) for choice in (choices or [])]
    if any(choice is None for choice in clean_choices):
        return None, "invalid_choice"
    choice_values = [choice for choice in clean_choices if choice is not None]
    if source_format == "multiple_choice":
        if not choice_values:
            return None, "missing_or_empty_choices"
        if clean_answer not in choice_values:
            return None, "answer_not_in_choices"
    elif source_format == "open_qa":
        if choice_values:
            return None, "open_qa_choices_must_be_empty"
    else:
        return None, "unsupported_source_format"
    record = {
        "question": clean_question,
        "choices": choice_values,
        "answer": clean_answer,
        "source": source,
        "source_format": source_format,
        "subject": clean_text(subject),
        "upstream_id": str(upstream_id),
        "upstream_revision": revision,
        "upstream_metadata": metadata,
    }
    return record, None


def normalize_global_mmlu(
    raw_records: List[Dict[str, Any]], spec: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    normalized = []
    exclusions: Counter[str] = Counter()
    labels = ("A", "B", "C", "D")
    for position, wrapped in enumerate(raw_records):
        raw = wrapped.get("row", wrapped)
        answer_label = clean_text(raw.get("answer"))
        choices = [raw.get(f"option_{label.lower()}") for label in labels]
        answer = choices[labels.index(answer_label.upper())] if answer_label and answer_label.upper() in labels else None
        record, error = normalized_record(
            question=raw.get("question"),
            answer=answer,
            source="global_mmlu",
            source_format="multiple_choice",
            subject=raw.get("subject"),
            choices=choices,
            upstream_id=raw.get("sample_id", wrapped.get("row_idx", position)),
            revision=spec["revision"],
            metadata={
                "split": spec["split"],
                "language": spec["config"],
                "subject_category": raw.get("subject_category"),
                "answer_label": answer_label,
                "row_index": wrapped.get("row_idx", position),
            },
        )
        if error:
            exclusions[error] += 1
        else:
            normalized.append(record)
    return normalized, exclusions


def normalize_mkqa(
    raw_records: List[Dict[str, Any]], spec: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    normalized = []
    exclusions: Counter[str] = Counter()
    accepted_types = set(spec["accepted_answer_types"])
    for position, raw in enumerate(raw_records):
        answers = raw.get("answers", {}).get("en", [])
        answer = answers[0] if isinstance(answers, list) and answers else {}
        answer_type = answer.get("type")
        if answer_type not in accepted_types:
            exclusions[f"answer_type:{answer_type or 'missing'}"] += 1
            continue
        record, error = normalized_record(
            question=raw.get("queries", {}).get("en"),
            answer=answer.get("text"),
            source="mkqa",
            source_format="open_qa",
            subject=answer_type,
            choices=None,
            upstream_id=raw.get("example_id", position),
            revision=spec["revision"],
            metadata={
                "language": "en",
                "answer_type": answer_type,
                "answer_aliases": answer.get("aliases", []),
            },
        )
        if error:
            exclusions[error] += 1
        else:
            normalized.append(record)
    return normalized, exclusions


def normalize_msqa(
    raw_records: List[Dict[str, Any]], spec: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    normalized = []
    exclusions: Counter[str] = Counter()
    for position, raw in enumerate(raw_records):
        if raw.get("language") != spec["language"]:
            exclusions["non_target_language"] += 1
            continue
        record, error = normalized_record(
            question=raw.get("question"),
            answer=raw.get("answer"),
            source="msqa",
            source_format="open_qa",
            subject=raw.get("category"),
            choices=None,
            upstream_id=raw.get("id", position),
            revision=spec["revision"],
            metadata={
                "language": raw.get("language"),
                "culture_circle": raw.get("culture_circle"),
                "category": raw.get("category"),
                "source_url": raw.get("source_url"),
                "source_url_desc": raw.get("source_url_desc"),
            },
        )
        if error:
            exclusions[error] += 1
        else:
            normalized.append(record)
    return normalized, exclusions


def normalize_openbookqa(
    raw_records: List[Dict[str, Any]], spec: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    normalized = []
    exclusions: Counter[str] = Counter()
    for position, raw in enumerate(raw_records):
        question = raw.get("question", {})
        raw_choices = question.get("choices", []) if isinstance(question, dict) else []
        label_to_text = {
            str(choice.get("label")): choice.get("text")
            for choice in raw_choices
            if isinstance(choice, dict)
        }
        answer_label = str(raw.get("answerKey", ""))
        record, error = normalized_record(
            question=question.get("stem") if isinstance(question, dict) else None,
            answer=label_to_text.get(answer_label),
            source="openbookqa",
            source_format="multiple_choice",
            subject="openbookqa_test",
            choices=[choice.get("text") for choice in raw_choices if isinstance(choice, dict)],
            upstream_id=raw.get("id", position),
            revision=spec["revision"],
            metadata={"split": spec["split"], "answer_label": answer_label},
        )
        if error:
            exclusions[error] += 1
        else:
            normalized.append(record)
    return normalized, exclusions


NORMALIZERS = {
    "global_mmlu_hf_rows": normalize_global_mmlu,
    "mkqa_jsonl_gz": normalize_mkqa,
    "msqa_jsonl": normalize_msqa,
    "openbookqa_zip": normalize_openbookqa,
}


def deterministic_sample(
    records: List[Dict[str, Any]], count: int, seed: int, source_name: str
) -> List[Dict[str, Any]]:
    if count <= 0:
        raise ValueError("records-per-source must be a positive integer")
    if len(records) < count:
        raise ValueError(f"{source_name} has only {len(records)} eligible records; requested {count}")
    stable = sorted(records, key=lambda record: record["upstream_id"])
    derived_seed = int(hashlib.sha256(f"{seed}:{source_name}".encode("utf-8")).hexdigest()[:16], 16)
    indices = random.Random(derived_seed).sample(range(len(stable)), count)
    return [stable[index] for index in indices]


def write_json_atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return sha256_bytes(payload)


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog).resolve()
    output_dir = Path(args.output_dir).resolve()
    catalog = load_catalog(catalog_path)
    sources = catalog["sources"]
    selected_names = args.sources or [
        name for name, spec in sources.items() if spec.get("enabled_by_default")
    ]
    unknown = sorted(set(selected_names) - set(sources))
    if unknown:
        raise ValueError(f"unknown catalog sources: {unknown}")
    disabled = [name for name in selected_names if not sources[name].get("enabled_by_default")]
    if disabled:
        raise ValueError(f"disabled catalog sources require catalog review before import: {disabled}")

    records_per_source = (
        None if args.all else (args.records_per_source or catalog["pilot"]["records_per_source"])
    )
    seed = args.seed if args.seed is not None else catalog["pilot"]["seed"]
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / ".download-cache"
    output_paths = [output_dir / sources[name]["output_file"] for name in selected_names]
    manifest_path = output_dir / "provenance_manifest.json"
    existing = [path for path in [*output_paths, manifest_path] if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"refusing to overwrite existing imports without --force: {existing}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    pending_outputs: Dict[Path, List[Dict[str, Any]]] = {}
    provenance_sources: Dict[str, Any] = {}
    for source_name in selected_names:
        spec = sources[source_name]
        if spec["adapter"] == "global_mmlu_hf_rows":
            raw_records, retrieval = fetch_global_mmlu(
                session,
                spec,
                args.timeout_seconds,
                records_per_source,
                seed,
                source_name,
                cache_dir,
            )
        else:
            raw_records, retrieval = fetch_direct_source(session, spec, args.timeout_seconds)
        eligible, exclusions = NORMALIZERS[spec["adapter"]](raw_records, spec)
        if records_per_source is None:
            selected = eligible
        elif spec["adapter"] == "global_mmlu_hf_rows":
            if len(eligible) != records_per_source:
                raise ValueError(
                    f"Global-MMLU selected rows yielded {len(eligible)} eligible records; "
                    f"expected {records_per_source}"
                )
            selected = eligible
        else:
            selected = deterministic_sample(eligible, records_per_source, seed, source_name)
        output_path = output_dir / spec["output_file"]
        pending_outputs[output_path] = selected
        provenance_sources[source_name] = {
            "repository": spec["repository"],
            "revision": spec["revision"],
            "retrieval_url": spec["url"],
            "license": spec["license"],
            **retrieval,
            "upstream_record_count": spec.get("expected_record_count", len(raw_records)),
            "eligible_record_count": (
                None
                if spec["adapter"] == "global_mmlu_hf_rows" and records_per_source is not None
                else len(eligible)
            ),
            "exclusions": dict(sorted(exclusions.items())),
            "normalization_scope": (
                "selected_rows"
                if spec["adapter"] == "global_mmlu_hf_rows" and records_per_source is not None
                else "all_upstream_records"
            ),
            "normalized_records_sha256": sha256_bytes(canonical_json_bytes(eligible)),
            "selected_count": len(selected),
            "selected_upstream_ids": [record["upstream_id"] for record in selected],
            "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        }

    for path, records in pending_outputs.items():
        digest = write_json_atomic(path, records)
        source_name = next(name for name in selected_names if sources[name]["output_file"] == path.name)
        provenance_sources[source_name]["output_sha256"] = digest
    manifest = {
        "manifest_version": "public-benchmark-import-v1",
        "created_at": utc_now(),
        "catalog_path": str(catalog_path.relative_to(PROJECT_ROOT)),
        "catalog_sha256": sha256_bytes(catalog_path.read_bytes()),
        "seed": seed,
        "records_per_source": "all" if records_per_source is None else records_per_source,
        "total_selected_count": sum(len(records) for records in pending_outputs.values()),
        "sources": provenance_sources,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Wrote {manifest['total_selected_count']} records from {len(selected_names)} sources to {output_dir}")
    print(f"Provenance manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, requests.RequestException) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
