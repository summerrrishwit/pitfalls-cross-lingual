"""Offline inventory; reads source artifacts without changing them. Run from repo root."""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed/factual_triples"
OUT = Path(__file__).resolve().parent
inputs = {}


def read(path, lines=False):
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)


def norm(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).casefold()).strip()


def overlap(left, right, fields):
    a = {tuple(norm(x.get(k)) for k in fields) for x in left}
    b = {tuple(norm(x.get(k)) for k in fields) for x in right}
    a.discard(tuple("" for _ in fields)); b.discard(tuple("" for _ in fields))
    return {"left_unique": len(a), "right_unique": len(b), "intersection_unique": len(a & b)}


legacy = DATA / "triple-full-v1/canonical-v2"
old = read(legacy / "triples_normalized.jsonl", True)
decisions = read(legacy / "canonical_decisions.jsonl", True)
prompts = read(legacy / "factual_prompts.jsonl", True)
distractors = read(legacy / "distractor_candidates.jsonl", True)
consensus = [x for x in old if x["canonical_reason"] == "dual_model_extraction_and_answer_consensus"]
signatures = Counter(x["relation_signature_id"] for x in old)
result = {
    "method": "Exact NFKC/casefold/whitespace comparisons only; no semantic deduplication or new model review.",
    "legacy": {
        "canonical_records": len(old),
        "decision_reasons": dict(Counter(x["canonical_reason"] for x in decisions)),
        "normalization_status": dict(Counter(x["normalization_status"] for x in old)),
        "mapped_relations": dict(Counter(x["relation_normalized"] for x in old if x["normalization_status"] == "mapped")),
        "unique_relation_signatures": len(signatures),
        "singleton_relation_signatures": sum(n == 1 for n in signatures.values()),
        "consensus_rows": len(consensus),
        "consensus_with_any_field_conflict": sum(bool(x["field_conflicts"]) for x in consensus),
        "consensus_conflict_fields": dict(Counter(k for x in consensus for k in x["field_conflicts"])),
        "prompt_sources": dict(Counter(x["prompt_source"] for x in prompts)),
        "distractor_candidates": len(distractors),
        "verified_distractor_candidates": sum(x["distractor_verified"] is True for x in distractors),
    },
    "raw_ledgers": {},
}
raw_public = None
for run, models in [("triple-full-v1", ["sensenova-6.7-flash-lite", "qwen3.7-plus"]), ("public-benchmarks-full-v1", ["qwen3.7-plus"])]:
    for model in models:
        rows = read(DATA / run / "models" / model / "triple_extractions.jsonl", True)
        result["raw_ledgers"][run + "/" + model] = {
            "rows": len(rows), "unique_source_ids": len({x["source_id"] for x in rows}),
            "extraction_status": dict(Counter(x["extraction_status"] for x in rows)),
            "validation_errors": sum(bool(x["validation_errors"]) for x in rows),
        }
        if run == "public-benchmarks-full-v1":
            raw_public = [x for x in rows if x["terminal_status"] == "completed" and x["extraction_status"] == "extracted" and not x["validation_errors"]]
manifests = {}
for run in ["triple-full-v1", "public-benchmarks-full-v1"]:
    m = read(DATA / run / "raw_source_manifest.json")
    manifests[run] = [{"source_question": x["source_snapshot"]["record"].get("question"), "source_answer": x["source_snapshot"]["record"].get("answer")} for x in m["candidates"]]
result["overlap"] = {
    "source_manifests_question": overlap(*manifests.values(), ["source_question"]),
    "legacy_canonical_vs_public_valid_extracted_question": overlap(old, raw_public, ["source_question"]),
    "legacy_canonical_vs_public_valid_extracted_sra": overlap(old, raw_public, ["subject", "relation_raw", "answer"]),
    "legacy_canonical_vs_public_valid_extracted_fact": overlap(old, raw_public, ["canonical_fact"]),
}
public_root = DATA / "public-benchmarks-full-v1/provisional/qwen3.7-plus-v1"
multi = read(public_root / "full-4513-multilingual-review-r09-recovery-v3-c16-r8-20260929/multilingual_translation_review_records.jsonl", True)
def common(x):
    return all((x["languages"].get(lang, {}).get("translation_equivalence_review") or {}).get("decision") == "accept" for lang in ["ar", "de", "es", "ja", "sw"])
groups = {}
for x in multi:
    group = groups.setdefault(x["relation_partition_id"], {"all": 0, "common": 0, "common_splits": Counter()})
    group["all"] += 1
    if common(x):
        group["common"] += 1
        group["common_splits"][str(x["split_assignment"])] += 1
result["multilingual"] = {"records": len(multi), "common": sum(common(x) for x in multi), "groups": groups}
result["overlap"]["legacy_canonical_vs_multilingual_4513_fact"] = overlap(old, [{"canonical_fact": x["source_fields"]["canonical_fact_en"]} for x in multi], ["canonical_fact"])
result["inputs"] = inputs
(OUT / "audit_evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k:v for k,v in result.items() if k not in ["inputs", "multilingual"]}, ensure_ascii=False, indent=2))
print("multilingual:", len(multi), "common:", sum(common(x) for x in multi))
