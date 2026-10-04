import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "manage_codex_review_shards.py"
SPEC = importlib.util.spec_from_file_location("manage_codex_review_shards", SCRIPT_PATH)
shards = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = shards
SPEC.loader.exec_module(shards)


def jsonl_payload(rows):
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_bytes(jsonl_payload(rows))


def binding(path, schema_version, record_count=None, ids=None):
    payload = path.read_bytes()
    result = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "schema_version": schema_version,
    }
    if record_count is not None:
        result["record_count"] = record_count
    if ids is not None:
        result["base_fact_ids_sha256"] = shards.sha256_value(sorted(ids))
    return result


class CodexReviewShardTests(unittest.TestCase):
    def make_parent_review(self, root, *, name="review", changed_indices=None):
        changed_indices = set(changed_indices or [])
        review_dir = root / name
        review_dir.mkdir()
        input_rows = []
        template_rows = []
        for index in range(shards.EXPECTED_RECORD_COUNT):
            base_fact_id = f"pbf_{index:03d}"
            review_input = {
                "schema_version": shards.REVIEW_INPUT_RECORD_SCHEMA,
                "base_fact_id": base_fact_id,
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "review_blinded_to_behavior_results": True,
                "payload": {"index": index},
                "variants": [
                    {
                        "distractor_id": f"d{index}_{rank}",
                        "context_candidates": [
                            {"candidate_id": f"ctx{index}_{rank}"}
                        ],
                    }
                    for rank in (1, 2)
                ],
            }
            if index in changed_indices:
                review_input["payload"]["repair_revision"] = 1
            input_rows.append(review_input)
            template_rows.append(
                {
                    "schema_version": shards.CODEX_DECISION_SCHEMA,
                    "base_fact_id": base_fact_id,
                    "review_input_record_sha256": shards.sha256_value(review_input),
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                    "terminal_status": "pending",
                    "decision": "defer",
                    "checks": {"fact_valid": None, "translation_valid": None},
                    "variants": [
                        {
                            "distractor_id": f"d{index}_{rank}",
                            "selected_candidate_id": None,
                            "decision": "defer",
                            "checks": {
                                "context_valid": None,
                                "neutral_valid": None,
                            },
                        }
                        for rank in (1, 2)
                    ],
                }
            )

        input_path = review_dir / "codex_adjudication_input.jsonl"
        template_path = review_dir / "codex_adjudication_decisions.template.jsonl"
        manifest_path = review_dir / "codex_adjudication_input_manifest.json"
        write_jsonl(input_path, input_rows)
        write_jsonl(template_path, template_rows)
        ids = [row["base_fact_id"] for row in input_rows]
        write_json(
            manifest_path,
            {
                "schema_version": shards.REVIEW_INPUT_MANIFEST_SCHEMA,
                "status": "codex_review_ready_behavior_blind",
                "base_fact_count": shards.EXPECTED_RECORD_COUNT,
                "behavior_authorized": False,
                "simulation_results_checked": True,
                "simulation_result_count": 0,
                "review_blinded_to_behavior_results": True,
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "review_input": binding(
                    input_path,
                    shards.REVIEW_INPUT_RECORD_SCHEMA,
                    record_count=len(input_rows),
                    ids=ids,
                ),
                "decision_template": binding(
                    template_path,
                    shards.CODEX_DECISION_SCHEMA,
                    record_count=len(template_rows),
                ),
            },
        )
        return manifest_path, input_path

    def completed_parent_decisions(self, manifest_path, output_path):
        loaded = shards._load_parent_review(manifest_path)
        review_by_id = {row["base_fact_id"]: row for row in loaded[4]}
        template_rows = loaded[-1]
        rows = []
        for row in template_rows:
            rows.append(
                self.complete_template_decision(
                    row, review_by_id[row["base_fact_id"]]
                )
            )
        write_jsonl(output_path, rows)
        return rows

    def complete_template_decision(self, template, review):
        row = json.loads(json.dumps(template))
        row["terminal_status"] = "completed"
        row["decision"] = "accept"
        row["checks"] = {key: True for key in row["checks"]}
        review_variants = {
            value["distractor_id"]: value for value in review["variants"]
        }
        for variant in row["variants"]:
            variant["terminal_status"] = "completed"
            variant["decision"] = "accept"
            variant["checks"] = {key: True for key in variant["checks"]}
            variant["selected_candidate_id"] = review_variants[
                variant["distractor_id"]
            ]["context_candidates"][0]["candidate_id"]
        return row

    def completed_decisions(self, shard_dir, *, duplicate_first=False):
        paths = []
        for shard_index in range(1, shards.SHARD_COUNT + 1):
            template_path = shard_dir / (
                f"codex_adjudication_decisions.shard-{shard_index:02d}-of-04.template.jsonl"
            )
            rows = shards.read_jsonl(template_path)
            for row in rows:
                row["terminal_status"] = "completed"
                row["decision"] = "accept"
            if duplicate_first and shard_index == 1:
                rows[1]["base_fact_id"] = rows[0]["base_fact_id"]
            path = shard_dir / f"completed-{shard_index:02d}.jsonl"
            write_jsonl(path, rows)
            paths.append(path)
        return paths

    def test_create_and_merge_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_manifest, _ = self.make_parent_review(root)
            shard_dir = root / "shards-v1"
            created = shards.create_shards(parent_manifest, shard_dir)

            self.assertEqual(created["shard_count"], 4)
            self.assertEqual(
                [entry["record_count"] for entry in created["shards"]],
                [40, 40, 40, 40],
            )
            decision_paths = self.completed_decisions(shard_dir)
            output_decisions = root / "codex_adjudication_decisions.jsonl"
            output_manifest = root / "codex_adjudication_merge_manifest.json"
            merged = shards.merge_decisions(
                shard_dir / "shard_manifest.json",
                decision_paths,
                output_decisions,
                output_manifest,
            )

            rows = shards.read_jsonl(output_decisions)
            self.assertEqual(len(rows), 160)
            self.assertEqual(
                [row["base_fact_id"] for row in rows],
                sorted(row["base_fact_id"] for row in rows),
            )
            self.assertEqual(
                merged["status"],
                "structurally_merged_pending_manage_static_g0a_semantic_validation",
            )
            self.assertFalse(merged["semantic_validation_performed"])

    def test_merge_rejects_parent_changed_after_create(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_manifest, input_path = self.make_parent_review(root)
            shard_dir = root / "shards-v1"
            shards.create_shards(parent_manifest, shard_dir)
            decision_paths = self.completed_decisions(shard_dir)
            with input_path.open("ab") as handle:
                handle.write(b"\n")

            with self.assertRaisesRegex(
                ValueError, "(SHA-256 mismatch|changed after shard creation)"
            ):
                shards.merge_decisions(
                    shard_dir / "shard_manifest.json",
                    decision_paths,
                    root / "merged.jsonl",
                    root / "merge.json",
                )

    def test_merge_rejects_duplicate_or_missing_shard_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_manifest, _ = self.make_parent_review(root)
            shard_dir = root / "shards-v1"
            shards.create_shards(parent_manifest, shard_dir)
            decision_paths = self.completed_decisions(
                shard_dir, duplicate_first=True
            )

            with self.assertRaisesRegex(ValueError, "duplicate decision shard"):
                shards.merge_decisions(
                    shard_dir / "shard_manifest.json",
                    decision_paths,
                    root / "merged.jsonl",
                    root / "merge.json",
                )

    def test_rebase_copies_only_identical_rows_and_resets_changed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_manifest, _ = self.make_parent_review(root, name="old-review")
            new_manifest, _ = self.make_parent_review(
                root, name="new-review", changed_indices={7}
            )
            old_decisions_path = root / "old-decisions.jsonl"
            old_decisions = self.completed_parent_decisions(
                old_manifest, old_decisions_path
            )
            output_decisions = root / "rebased-decisions.jsonl"
            output_manifest = root / "rebase-manifest.json"

            result = shards.rebase_decisions(
                old_manifest,
                old_decisions_path,
                new_manifest,
                output_decisions,
                output_manifest,
            )

            rebased = {
                row["base_fact_id"]: row for row in shards.read_jsonl(output_decisions)
            }
            old_by_id = {row["base_fact_id"]: row for row in old_decisions}
            new_template_by_id = {
                row["base_fact_id"]: row
                for row in shards._load_parent_review(new_manifest)[-1]
            }
            changed_id = "pbf_007"
            unchanged_id = "pbf_006"
            self.assertEqual(rebased[unchanged_id], old_by_id[unchanged_id])
            self.assertEqual(rebased[changed_id], new_template_by_id[changed_id])
            self.assertEqual(rebased[changed_id]["terminal_status"], "pending")
            self.assertEqual(rebased[changed_id]["decision"], "defer")
            self.assertNotEqual(
                rebased[changed_id]["review_input_record_sha256"],
                old_by_id[changed_id]["review_input_record_sha256"],
            )
            self.assertEqual(result["copied_count"], 159)
            self.assertEqual(result["pending_count"], 1)
            self.assertEqual(result["pending_base_fact_ids"], [changed_id])
            self.assertEqual(
                result["rebased_decisions"]["sha256"],
                hashlib.sha256(output_decisions.read_bytes()).hexdigest(),
            )

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                shards.rebase_decisions(
                    old_manifest,
                    old_decisions_path,
                    new_manifest,
                    output_decisions,
                    root / "second-rebase-manifest.json",
                )

    def test_rebase_rejects_old_decision_with_rewritten_input_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_manifest, _ = self.make_parent_review(root, name="old-review")
            new_manifest, _ = self.make_parent_review(
                root, name="new-review", changed_indices={7}
            )
            old_decisions_path = root / "old-decisions.jsonl"
            old_decisions = self.completed_parent_decisions(
                old_manifest, old_decisions_path
            )
            old_decisions[0]["review_input_record_sha256"] = "0" * 64
            write_jsonl(old_decisions_path, old_decisions)

            with self.assertRaisesRegex(ValueError, "review input hash mismatch"):
                shards.rebase_decisions(
                    old_manifest,
                    old_decisions_path,
                    new_manifest,
                    root / "rebased-decisions.jsonl",
                    root / "rebase-manifest.json",
                )
            self.assertFalse((root / "rebased-decisions.jsonl").exists())
            self.assertFalse((root / "rebase-manifest.json").exists())

    def test_complete_rebase_replaces_exact_pending_ids_and_preserves_bound_rebase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_manifest, _ = self.make_parent_review(root, name="old-review")
            new_manifest, _ = self.make_parent_review(
                root, name="new-review", changed_indices={7, 9}
            )
            old_decisions_path = root / "old-decisions.jsonl"
            old_decisions = self.completed_parent_decisions(
                old_manifest, old_decisions_path
            )
            rebased_path = root / "rebased-decisions.jsonl"
            rebase_manifest_path = root / "rebase-manifest.json"
            shards.rebase_decisions(
                old_manifest,
                old_decisions_path,
                new_manifest,
                rebased_path,
                rebase_manifest_path,
            )
            rebased_sha_before = hashlib.sha256(rebased_path.read_bytes()).hexdigest()

            new_loaded = shards._load_parent_review(new_manifest)
            new_review_by_id = {
                row["base_fact_id"]: row for row in new_loaded[4]
            }
            new_template_by_id = {
                row["base_fact_id"]: row for row in new_loaded[-1]
            }
            pending_ids = ["pbf_007", "pbf_009"]
            pending_rows = [
                self.complete_template_decision(
                    new_template_by_id[base_fact_id],
                    new_review_by_id[base_fact_id],
                )
                for base_fact_id in pending_ids
            ]
            pending_path = root / "new-pending-decisions.jsonl"
            write_jsonl(pending_path, pending_rows)
            completed_path = root / "completed-decisions.jsonl"
            completion_manifest_path = root / "completion-manifest.json"

            result = shards.complete_rebase(
                rebase_manifest_path,
                pending_path,
                completed_path,
                completion_manifest_path,
            )

            self.assertEqual(
                hashlib.sha256(rebased_path.read_bytes()).hexdigest(),
                rebased_sha_before,
            )
            completed_by_id = {
                row["base_fact_id"]: row for row in shards.read_jsonl(completed_path)
            }
            old_by_id = {row["base_fact_id"]: row for row in old_decisions}
            pending_by_id = {row["base_fact_id"]: row for row in pending_rows}
            self.assertEqual(completed_by_id["pbf_006"], old_by_id["pbf_006"])
            self.assertEqual(completed_by_id["pbf_007"], pending_by_id["pbf_007"])
            self.assertTrue(
                all(
                    row["terminal_status"] == "completed"
                    for row in completed_by_id.values()
                )
            )
            self.assertEqual(result["copied_decision_count"], 158)
            self.assertEqual(result["newly_completed_decision_count"], 2)
            self.assertEqual(result["newly_completed_base_fact_ids"], pending_ids)
            self.assertEqual(
                result["completed_decisions"]["sha256"],
                hashlib.sha256(completed_path.read_bytes()).hexdigest(),
            )

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                shards.complete_rebase(
                    rebase_manifest_path,
                    pending_path,
                    completed_path,
                    root / "second-completion-manifest.json",
                )

    def test_complete_rebase_rejects_incomplete_pending_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_manifest, _ = self.make_parent_review(root, name="old-review")
            new_manifest, _ = self.make_parent_review(
                root, name="new-review", changed_indices={7, 9}
            )
            old_decisions_path = root / "old-decisions.jsonl"
            self.completed_parent_decisions(old_manifest, old_decisions_path)
            rebased_path = root / "rebased-decisions.jsonl"
            rebase_manifest_path = root / "rebase-manifest.json"
            shards.rebase_decisions(
                old_manifest,
                old_decisions_path,
                new_manifest,
                rebased_path,
                rebase_manifest_path,
            )
            new_loaded = shards._load_parent_review(new_manifest)
            review_by_id = {row["base_fact_id"]: row for row in new_loaded[4]}
            template_by_id = {row["base_fact_id"]: row for row in new_loaded[-1]}
            pending_path = root / "incomplete-pending.jsonl"
            write_jsonl(
                pending_path,
                [
                    self.complete_template_decision(
                        template_by_id["pbf_007"], review_by_id["pbf_007"]
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                shards.complete_rebase(
                    rebase_manifest_path,
                    pending_path,
                    root / "completed-decisions.jsonl",
                    root / "completion-manifest.json",
                )
            self.assertFalse((root / "completed-decisions.jsonl").exists())

    def test_complete_rebase_rejects_pending_decision_with_wrong_review_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_manifest, _ = self.make_parent_review(root, name="old-review")
            new_manifest, _ = self.make_parent_review(
                root, name="new-review", changed_indices={7}
            )
            old_decisions_path = root / "old-decisions.jsonl"
            self.completed_parent_decisions(old_manifest, old_decisions_path)
            rebased_path = root / "rebased-decisions.jsonl"
            rebase_manifest_path = root / "rebase-manifest.json"
            shards.rebase_decisions(
                old_manifest,
                old_decisions_path,
                new_manifest,
                rebased_path,
                rebase_manifest_path,
            )
            new_loaded = shards._load_parent_review(new_manifest)
            review_by_id = {row["base_fact_id"]: row for row in new_loaded[4]}
            template_by_id = {row["base_fact_id"]: row for row in new_loaded[-1]}
            decision = self.complete_template_decision(
                template_by_id["pbf_007"], review_by_id["pbf_007"]
            )
            decision["review_input_record_sha256"] = "0" * 64
            pending_path = root / "wrong-hash-pending.jsonl"
            write_jsonl(pending_path, [decision])

            with self.assertRaisesRegex(ValueError, "review input hash mismatch"):
                shards.complete_rebase(
                    rebase_manifest_path,
                    pending_path,
                    root / "completed-decisions.jsonl",
                    root / "completion-manifest.json",
                )
            self.assertFalse((root / "completed-decisions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
