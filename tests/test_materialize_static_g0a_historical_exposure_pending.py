import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "materialize_static_g0a_historical_exposure_pending.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_static_g0a_historical_exposure_pending", SCRIPT_PATH
)
exposure = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = exposure
SPEC.loader.exec_module(exposure)


class HistoricalExposurePendingTests(unittest.TestCase):
    def test_repository_scan_covers_untracked_worktree_and_deleted_git_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            historical = root / "historical_output.jsonl"
            historical.write_text(
                json.dumps(
                    {
                        "source_id": "source_holdout",
                        "simulation_id": "sim-1",
                        "model": "fixture-model",
                        "raw_response": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(root), "add", historical.name], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "historical behavior blob",
                ],
                check=True,
            )
            historical.unlink()
            subprocess.run(["git", "-C", str(root), "add", "-u"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-q",
                    "-m",
                    "delete historical artifact",
                ],
                check=True,
            )
            untracked = root / "review.jsonl"
            untracked.write_text(
                json.dumps(
                    {
                        "source_id": "source_holdout",
                        "model": "fixture-reviewer",
                        "raw_response": "review text",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = exposure._materialize_repository_inventory(
                repo_root=root,
                identifier_to_base_fact_ids={"source_holdout": {"pbf_holdout"}},
            )
            self.assertEqual(result["worktree_file_count"], 1)
            self.assertGreaterEqual(result["reachable_git_blob_count"], 1)
            self.assertEqual(result["behavior_hit_base_fact_ids"], ["pbf_holdout"])
            hit_sources = [
                row["source_kind"]
                for row in result["results"]
                if row["behavior_hit_base_fact_ids"]
            ]
            self.assertEqual(hit_sources, ["reachable_git_blob"])
            untracked_rows = [
                row
                for row in result["results"]
                if row["source_kind"] == "current_worktree_file"
            ]
            self.assertEqual(len(untracked_rows), 1)
            self.assertEqual(untracked_rows[0]["behavior_hit_base_fact_ids"], [])

    def test_simulation_scan_uses_identifiers_but_not_response_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "simulation_results.jsonl"
            rows = [
                {
                    "source_id": "source-hit",
                    "candidate_id": "candidate-other",
                    "raw_response": "pbf_response_only",
                },
                {
                    "source_id": "source-other",
                    "raw_response": "source-response-only",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            scan = exposure._scan_simulation_file(
                path,
                identifier_to_base_fact_ids={
                    "source-hit": {"pbf_hit"},
                    "pbf_response_only": {"pbf_must_not_match"},
                    "source-response-only": {"pbf_must_not_match_either"},
                },
                repo_root=root,
            )
            self.assertEqual(scan["record_count"], 2)
            self.assertEqual(scan["matched_holdout_base_fact_count"], 1)
            self.assertEqual(
                scan["matches"],
                [{"base_fact_id": "pbf_hit", "matched_identifiers": ["source-hit"]}],
            )
            self.assertFalse(scan["behavior_outcome_fields_inspected"])

    def test_existing_output_fails_before_git_or_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                exposure.materialize_pending_exposure(
                    resolved_manifest_path=Path("unused"),
                    repo_root=Path(directory),
                    output_dir=output,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
