#!/usr/bin/env python3
"""Read-only snapshot verification; historical absolute bindings map into this checkout."""
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "analysis_output/zh_normalize_translate_20261023_v1"
ORIGIN = Path("/Users/qiqing/.codex/worktrees/762b/pitfalls-cross-lingual")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


def mapped(path):
    # Never silently read another worktree when verifying the submitted snapshot.
    return REPO / Path(path).relative_to(ORIGIN)


def main():
    manifest = read(RUN / "manifest.json")
    imported = read(Path(__file__).parent / "worktree_import_manifest.json")
    for item in imported["files"]:
        check(sha(REPO / item["path"]) == item["sha256"], item["path"])
    for path, expected in manifest["bindings"].items():
        check(sha(mapped(path)) == expected, path)
    sources = rows(RUN / "input.jsonl")
    by_id = {row["record_id"]: row for row in sources}
    check(len(sources) == len(by_id) == 7250, "source denominator/uniqueness")
    normalized, translated, reviewed = {}, {}, {}
    batches = sorted((RUN / "batches").glob("batch_*.json"))
    check(len(batches) == 3625, "batch count")
    for path in batches:
        batch = read(path)
        offset = int(path.stem.split("_")[1]) * manifest["batch_size"]
        check(batch["input_sha256"] == digest(sources[offset:offset + 2]), str(path))
        check(batch["status"] in {"completed", "failed"}, "pending batch")
        for row in batch["normalized"]:
            rid = row["record_id"]
            check(rid not in normalized and row["input_sha256"] == digest(by_id[rid]), rid)
            normalized[rid] = row
        for row in batch["translated"]:
            rid = row["record_id"]
            check(rid not in translated and row["english_sha256"] == normalized[rid]["english_sha256"], rid)
            check(digest(row["english"]) == row["english_sha256"], rid)
            check(digest(row["translation"]) == row["translation_sha256"], rid)
            translated[rid] = row
    packets = sorted((RUN / "review_packets").glob("packet_*/final_manifest.json"))
    for path in packets:
        folder = path.parent
        final = read(path)
        packet = read(folder / "manifest.json")
        check(sha(folder / "final.jsonl") == final["final_sha256"], str(path))
        check(sha(folder / "dispatch.json") == final["dispatch_sha256"], str(path))
        check(sha(folder / "input.jsonl") == packet["input_file_sha256"], str(path))
        check(sha(folder / "instructions.md") == packet["instructions_sha256"], str(path))
        check(sha(RUN / "manifest.json") == packet["run_manifest_sha256"], str(path))
        check(sha(folder / "manifest.json") == read(folder / "dispatch.json")["packet_manifest_sha256"], str(path))
        for binding in final["real_agent_bindings"]:
            check(sha(mapped(binding["output_path"])) == binding["output_sha256"], str(path))
        for row in rows(folder / "final.jsonl"):
            rid = row["record_id"]
            check(rid not in reviewed and rid in translated, rid)
            check(row["english_sha256"] == translated[rid]["english_sha256"], rid)
            check(row["translation_sha256"] == translated[rid]["translation_sha256"], rid)
            check(row["formal_experiment_eligible"] is False and row["human_gold"] is False, rid)
            reviewed[rid] = row
    check(set(reviewed) == set(translated), "review coverage")
    exports = RUN / "exports"
    summary = read(exports / "summary.json")
    for name, expected in summary["files"].items():
        check(sha(exports / name) == expected, name)
    accepted = rows(exports / "accepted_chinese_candidates.jsonl")
    check({r["record_id"]: r for r in accepted} ==
          {rid: r for rid, r in reviewed.items() if r["status"] == "zh_translation_accepted"}, "accepted export")
    ready = rows(exports / "normalized_candidates.jsonl")
    check({r["record_id"]: r for r in ready} ==
          {rid: r for rid, r in normalized.items() if r["decision"] == "ready"}, "normalized export")
    coverage = rows(exports / "source_coverage.jsonl")
    check(len(coverage) == len({r["record_id"] for r in coverage}) == 7250, "coverage uniqueness")
    check({r["record_id"] for r in coverage} == set(by_id), "coverage membership")
    for row in coverage:
        check(row["source_sha256"] == digest(by_id[row["record_id"]]), "coverage source binding")
    check(dict(Counter(r["status"] for r in coverage)) == summary["source_status_counts"], "coverage counts")
    archives = sorted((RUN / "recovery_history").glob("*/manifest.json"))
    exposure = (RUN / "exposure.jsonl").read_bytes()
    for path in archives:
        recovery = read(path)
        for item in recovery["archived_files"].values():
            check(sha(mapped(item["archive"])) == item["sha256"], str(path))
        check(hashlib.sha256(exposure[:recovery["exposure_prefix_bytes"]]).hexdigest() ==
              recovery["exposure_prefix_sha256"], str(path))
    tails = sorted((RUN / "transport_retry_history").glob("epoch_*/*.json"))
    for path in tails:
        before = read(path)
        after = read(RUN / "batches" / path.name)
        attempts = before["attempts"]
        check(after["attempts"][:len(attempts)] == attempts, "tail attempt prefix")
    catalog = read(exports / "proposed_relation_catalog.json")
    print(json.dumps({"status": "pass", "imported_file_hashes": len(imported["files"]),
                      "source_count": len(sources), "batches": len(batches),
                      "normalized": len(normalized), "normalized_ready_export": len(ready),
                      "translated_and_reviewed": len(reviewed), "review_packets": len(packets),
                      "review_counts": dict(Counter(r["status"] for r in reviewed.values())),
                      "recovery_archives": len(archives), "tail_attempt_prefixes": len(tails),
                      "proposed_relation_ids": len(catalog["relations"]),
                      "conflicting_relation_ids": len(catalog["conflicting_ids"]),
                      "historical_bindings_rewritten": False, "formal_experiment_eligible": False}, indent=2))


if __name__ == "__main__":
    main()
