#!/usr/bin/env python3
"""Bounded structured Qwen probes; no credentials/endpoints in output artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CASES = [
    {"id": "forward_author", "source": "The author of the novel Sample Book is Mira.", "subject": "Sample Book", "relation": "written by", "answer": "Mira", "expected": "written_work_to_author"},
    {"id": "inverse_definition", "source": "A term meaning a three-sided polygon is triangle.", "subject": "three-sided polygon", "relation": "term for", "answer": "triangle", "expected": "description_to_term"},
    {"id": "definition", "source": "A triangle is a polygon with three sides.", "subject": "triangle", "relation": "definition", "answer": "a polygon with three sides", "expected": "term_to_definition"},
    {"id": "ambiguous_date", "source": "Sample Film has different release dates in different markets; no market or date is supplied.", "subject": "Sample Film", "relation": "release date", "answer": "unknown", "expected": "unresolved"},
]


def safe_error(error: Exception) -> dict[str, Any]:
    value: dict[str, Any] = {"error_type": type(error).__name__}
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        value["http_status"] = status
    if isinstance(error, RuntimeError) and str(error) in {"response_model_identity_mismatch", "gateway_zero_usage_error_response"}:
        value["error_code"] = str(error)
    return value


def safe_model_label(value: Any, *secrets: str) -> str | None:
    """Keep diagnostic model labels without logging credentials or URLs."""
    if value is None:
        return None
    label = str(value)
    if (any(secret and secret in label for secret in secrets)
            or re.search(r"https?://|sk-", label, re.IGNORECASE)
            or not re.fullmatch(r"[\w.:/ -]{0,128}", label)):
        return "[redacted_model_label]"
    return label


def gateway_error_hint(content: str, usage: dict[str, Any] | None, *secrets: str) -> str | None:
    """Recognize short zero-token error wrappers, never a proposal completion.

    Such responses are retried as transport failures. They are never accepted,
    and a structured proposal from the wrong model still fails the identity gate.
    """
    if not usage or usage.get("total_tokens") != 0 or not 0 < len(content) <= 256:
        return None
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        value = content
    if not (isinstance(value, str) or isinstance(value, dict) and "error" in value and "results" not in value):
        return None
    hint = content
    for secret in secrets:
        if secret:
            hint = hint.replace(secret, "[redacted]")
    hint = re.sub(r"https?://\S+|sk-[\w-]+", "[redacted]", hint, flags=re.IGNORECASE)
    return hint[:256]


def load_route(env_file: Path) -> tuple[str, str, str]:
    from dotenv import dotenv_values
    values = dotenv_values(env_file)
    required = ["AILYUN_API_KEY", "AILYUN_BASE_URL", "AILYUN_MODEL"]
    missing = [key for key in required if not str(values.get(key) or "").strip()]
    if missing:
        raise ValueError("Missing configured variables: " + ", ".join(missing))
    return tuple(str(values[key]).strip() for key in required)


def percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path("utils/.env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.waves < 1 or args.waves > 4 or any(c not in (1, 8, 16) for c in args.levels):
        parser.error("Use levels 1/8/16 and one to four bounded waves")
    from openai import OpenAI
    from qwen_stream_transport import stream_completion
    key, base, model = load_route(args.env_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    route_hash = hashlib.sha256((base.rstrip("/") + "\n" + model).encode()).hexdigest()
    report: dict[str, Any] = {
        "schema_version": "relation-expansion-api-probe-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_model": model, "route_sha256": route_hash,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transport_helper_sha256": hashlib.sha256(Path(__file__).with_name("qwen_stream_transport.py").read_bytes()).hexdigest(),
        "input_policy": "synthetic_static_cases_only", "synthetic_output_not_production_data": True,
        "human_gold": False, "retries": 0, "credentials_or_endpoints_included": False,
        "max_output_tokens": args.max_output_tokens,
        "levels": [],
    }
    expected_model: str | None = None
    for level in args.levels:
        count = 1 if level == 1 else level * args.waves
        lock = threading.Lock()
        active = 0
        peak = 0
        gate = threading.Barrier(min(level, count))

        def work(index: int) -> dict[str, Any]:
            nonlocal active, peak
            prompt = (
                "Return only a JSON object with results: [{id, relation_leaf_id}]. "
                "Classify the stated fictional facts using only supplied text. "
                "Allowed leaf IDs: written_work_to_author, description_to_term, term_to_definition, unresolved. "
                "Do not invert a relation or invent missing dates. No external knowledge. "
                f"Request identifier: c{level}-{index}. Cases:\n"
                + json.dumps([{k: v for k, v in case.items() if k != "expected"} for case in CASES])
            )
            if index < level:
                gate.wait(timeout=30)
            started = time.monotonic()
            with lock:
                active += 1
                peak = max(peak, active)
            result: dict[str, Any] = {"request_index": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            try:
                with OpenAI(api_key=key, base_url=base, timeout=args.timeout, max_retries=0) as client:
                    response = stream_completion(client, expected_model='Qwen-3.8-Max',
                        model=model, messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}, temperature=0,
                        max_tokens=args.max_output_tokens, extra_body={"enable_thinking": False},
                    )
                content = response.choices[0].message.content or ""
                result["response_model"] = safe_model_label(response.model, key, base)
                result["response_diagnostics"] = {
                    "finish_reason": response.streaming_diagnostics["finish_reason"],
                    **response.streaming_diagnostics,
                    "content_characters": len(content),
                    "reasoning_characters": len(getattr(response.choices[0].message, "reasoning_content", "") or ""),
                    "usage": response.usage.model_dump() if response.usage else {},
                }
                value = json.loads(content)
                rows = value.get("results")
                if not isinstance(rows, list) or len(rows) != len(CASES):
                    raise ValueError("invalid_results_count")
                by_id = {r["id"]: r["relation_leaf_id"] for r in rows}
                if len(by_id) != len(CASES) or set(by_id) != {c["id"] for c in CASES}:
                    raise ValueError("invalid_results_ids")
                result.update(status="completed",
                              semantic_case_matches=sum(by_id[c["id"]] == c["expected"] for c in CASES),
                              results=rows, usage=response.usage.model_dump() if response.usage else {})
            except Exception as error:
                result.update(status="failed", **safe_error(error))
            finally:
                result["latency_seconds"] = round(time.monotonic() - started, 3)
                with lock:
                    active -= 1
            return result

        start = time.monotonic()
        rows = []
        with ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(work, i) for i in range(count)]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(json.dumps({"concurrency": level, "finished": len(rows), "total": count, "status": row["status"], "http_status": row.get("http_status")}), flush=True)
        completed = [r for r in rows if r["status"] == "completed"]
        models = Counter(r.get("response_model") for r in completed)
        if expected_model is None and len(models) == 1:
            expected_model = next(iter(models))
        identities_ok = bool(completed) and bool(expected_model) and set(models) == {expected_model}
        wall = time.monotonic() - start
        summary = {
            "concurrency": level, "requests": count, "peak_in_flight": peak,
            "completed": len(completed), "failed": count - len(completed),
            "http_status_counts": dict(Counter(str(r["http_status"]) for r in rows if "http_status" in r)),
            "response_models": dict(models), "identity_stable": identities_ok,
            "wall_seconds": round(wall, 3),
            "p50_seconds": percentile([r["latency_seconds"] for r in completed], .5),
            "p95_seconds": percentile([r["latency_seconds"] for r in completed], .95),
            "requests_per_minute": round(len(completed) * 60 / wall, 3),
            "semantic_case_matches": sum(r["semantic_case_matches"] for r in completed),
            "semantic_cases_total": len(completed) * len(CASES),
            "bounded_probe_passed": len(completed) == count and identities_ok and all(r["semantic_case_matches"] == len(CASES) for r in completed),
        }
        (args.output_dir / f"requests_c{level}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["request_index"])))
        report["levels"].append(summary)
        report["observed_response_model"] = expected_model
        (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        if not summary["bounded_probe_passed"]:
            break
    return 0 if len(report["levels"]) == len(args.levels) and all(x["bounded_probe_passed"] for x in report["levels"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
