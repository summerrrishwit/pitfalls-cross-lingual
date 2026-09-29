import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_selected_context_lengths.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_selected_context_lengths", SCRIPT_PATH
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class SelectedContextLengthAuditTests(unittest.TestCase):
    def fixture(self, root):
        staging_manifest = root / "static_staging_manifest.json"
        staging_manifest.write_text("{}\n", encoding="utf-8")
        review_input = root / "codex_adjudication_input.jsonl"
        review_input.write_text("{}\n", encoding="utf-8")
        review_manifest_path = root / "codex_adjudication_input_manifest.json"
        review_manifest_path.write_text(
            json.dumps(
                {"staging_manifest": {"path": str(staging_manifest)}},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        review_rows = []
        review_by_id = {}
        decisions_by_id = {}
        split_boundaries = ((96, "development"), (128, "validation"), (160, "sealed"))
        for index in range(160):
            split = next(name for boundary, name in split_boundaries if index < boundary)
            base_fact_id = f"pbf_{index:03d}"
            variants = []
            decision_variants = []
            for rank in (1, 2):
                distractor_id = f"d_{index:03d}_{rank}"
                candidate_id = f"ctx_{index:03d}_{rank}"
                neutral_en = "equal neutral"
                targeted_en = "equal target"
                neutral_zh = "中性"
                targeted_zh = "目标"
                if index == 0 and rank == 1:
                    neutral_en = "one  two\tthree"
                    targeted_en = "one two"
                    neutral_zh = "甲 乙！"
                    targeted_zh = "丙丁"
                elif index == 0 and rank == 2:
                    neutral_en = "single"
                    targeted_en = "one two"
                    neutral_zh = "甲"
                    targeted_zh = "乙丙"
                candidate = {
                    "candidate_id": candidate_id,
                    "candidate_record_sha256": f"{index * 2 + rank:064x}",
                    "candidate_source": "fixture",
                    "neutral_context_en": neutral_en,
                    "targeted_context_en": targeted_en,
                    "neutral_context_zh": neutral_zh,
                    "targeted_context_zh": targeted_zh,
                }
                variants.append(
                    {
                        "distractor_id": distractor_id,
                        "context_candidates": [candidate],
                    }
                )
                decision_variants.append(
                    {
                        "distractor_id": distractor_id,
                        "selected_candidate_id": candidate_id,
                    }
                )
            review = {
                "base_fact_id": base_fact_id,
                "source": {"split_assignment": split},
                "variants": variants,
            }
            decision = {
                "base_fact_id": base_fact_id,
                "variants": decision_variants,
            }
            review_rows.append(review)
            review_by_id[base_fact_id] = review
            decisions_by_id[base_fact_id] = decision

        decisions_path = root / "codex_adjudication_decisions.jsonl"
        audit.g0a.write_jsonl(
            decisions_path,
            [decisions_by_id[base_fact_id] for base_fact_id in sorted(decisions_by_id)],
        )
        review_manifest = {
            "variant_count": 320,
            "review_input": audit.g0a._binding(review_input, record_count=160),
        }
        return {
            "review_manifest_path": review_manifest_path,
            "decisions_path": decisions_path,
            "review_manifest": review_manifest,
            "review_rows": review_rows,
            "review_by_id": review_by_id,
            "decisions_by_id": decisions_by_id,
        }

    def test_happy_path_counts_ratios_distributions_and_sha_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            output_dir = root / "audit"
            records_path = output_dir / audit.RECORDS_FILENAME
            summary_path = output_dir / audit.SUMMARY_FILENAME
            normalized_decisions = copy.deepcopy(fixture["decisions_by_id"])
            for decision in normalized_decisions.values():
                decision["revisions"] = {}
                decision["proposed_patch"] = {"set": {}}
            with mock.patch.object(
                audit.g0a,
                "_load_review_input",
                return_value=(
                    fixture["review_manifest"],
                    fixture["review_rows"],
                    fixture["review_by_id"],
                ),
            ) as load_review, mock.patch.object(
                audit.g0a,
                "_validate_codex_decisions",
                return_value=(
                    normalized_decisions,
                    [],
                    Counter({"accept": 160}),
                ),
            ) as validate_decisions:
                summary = audit.audit_selected_context_lengths(
                    review_input_manifest_path=fixture["review_manifest_path"],
                    codex_decisions_path=fixture["decisions_path"],
                    output_dir_path=output_dir,
                )

            load_review.assert_called_once()
            validate_decisions.assert_called_once_with(
                fixture["decisions_path"].resolve(), fixture["review_by_id"]
            )
            rows = read_jsonl(records_path)
            self.assertEqual(len(rows), 320)
            first, second = rows[:2]
            self.assertEqual(first["en_neutral_word_count"], 3)
            self.assertEqual(first["en_targeted_word_count"], 2)
            self.assertEqual(first["en_neutral_to_targeted_ratio"], 1.5)
            self.assertEqual(first["zh_neutral_nonspace_char_count"], 3)
            self.assertEqual(first["zh_targeted_nonspace_char_count"], 2)
            self.assertEqual(first["zh_neutral_to_targeted_ratio"], 1.5)
            self.assertEqual(second["en_neutral_to_targeted_ratio"], 0.5)
            self.assertEqual(second["zh_neutral_to_targeted_ratio"], 0.5)
            self.assertEqual(
                first["decision_record_sha256"],
                audit.g0a.sha256_value(
                    next(
                        row
                        for row in audit.g0a.read_jsonl(fixture["decisions_path"])
                        if row["base_fact_id"] == first["base_fact_id"]
                    )
                ),
            )
            self.assertNotEqual(
                first["decision_record_sha256"],
                audit.g0a.sha256_value(
                    normalized_decisions[first["base_fact_id"]]
                ),
            )

            self.assertTrue(summary["descriptive_only"])
            self.assertFalse(summary["hard_threshold_applied"])
            self.assertFalse(summary["gate_decision_produced"])
            self.assertEqual(summary["counts"]["base_fact_count"], 160)
            self.assertEqual(summary["counts"]["selected_variant_count"], 320)
            self.assertEqual(
                summary["counts"]["selected_variant_counts_by_split"],
                {"development": 192, "sealed": 64, "validation": 64},
            )
            en_ratio = summary["distribution"]["en_neutral_to_targeted_ratio"]
            self.assertEqual(en_ratio["count"], 320)
            self.assertEqual(en_ratio["below_one_count"], 1)
            self.assertEqual(en_ratio["equal_one_count"], 318)
            self.assertEqual(en_ratio["above_one_count"], 1)
            self.assertEqual(
                summary["distribution_by_split"]["development"][
                    "en_neutral_to_targeted_ratio"
                ]["count"],
                192,
            )

            persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted_summary, summary)
            self.assertEqual(
                summary["inputs"]["review_input_manifest"]["sha256"],
                audit.g0a.sha256_file(fixture["review_manifest_path"]),
            )
            self.assertEqual(
                summary["inputs"]["codex_decisions"]["sha256"],
                audit.g0a.sha256_file(fixture["decisions_path"]),
            )
            self.assertEqual(
                summary["outputs"]["selected_context_lengths"]["sha256"],
                audit.g0a.sha256_file(records_path),
            )

    def test_manager_blockers_fail_closed_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            output_dir = root / "audit"
            records_path = output_dir / audit.RECORDS_FILENAME
            summary_path = output_dir / audit.SUMMARY_FILENAME
            blocker = "codex_variant_not_accepted:pbf_000:d_000_1"
            with mock.patch.object(
                audit.g0a,
                "_load_review_input",
                return_value=(
                    fixture["review_manifest"],
                    fixture["review_rows"],
                    fixture["review_by_id"],
                ),
            ), mock.patch.object(
                audit.g0a,
                "_validate_codex_decisions",
                return_value=(
                    fixture["decisions_by_id"],
                    [blocker],
                    Counter({"accept": 160}),
                ),
            ):
                with self.assertRaises(audit.g0a.GateBlocked) as captured:
                    audit.audit_selected_context_lengths(
                        review_input_manifest_path=fixture["review_manifest_path"],
                        codex_decisions_path=fixture["decisions_path"],
                        output_dir_path=output_dir,
                    )

            self.assertEqual(captured.exception.blockers, [blocker])
            self.assertFalse(records_path.exists())
            self.assertFalse(summary_path.exists())

    def test_input_change_during_validation_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            output_dir = root / "audit"

            def mutate_decisions(*_args):
                with fixture["decisions_path"].open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                return (
                    fixture["decisions_by_id"],
                    [],
                    Counter({"accept": 160}),
                )

            with mock.patch.object(
                audit.g0a,
                "_load_review_input",
                return_value=(
                    fixture["review_manifest"],
                    fixture["review_rows"],
                    fixture["review_by_id"],
                ),
            ), mock.patch.object(
                audit.g0a,
                "_validate_codex_decisions",
                side_effect=mutate_decisions,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "input changed while audit was running"
                ):
                    audit.audit_selected_context_lengths(
                        review_input_manifest_path=fixture["review_manifest_path"],
                        codex_decisions_path=fixture["decisions_path"],
                        output_dir_path=output_dir,
                    )

            self.assertFalse(output_dir.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "audit"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit.audit_selected_context_lengths(
                    review_input_manifest_path=root / "missing-manifest.json",
                    codex_decisions_path=root / "missing-decisions.jsonl",
                    output_dir_path=output_dir,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel\n")

    def test_atomic_directory_publish_failure_leaves_no_committed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "audit"
            with mock.patch.object(audit.os, "rename", side_effect=OSError("boom")):
                with self.assertRaisesRegex(OSError, "boom"):
                    audit._publish_output_dir(
                        output_dir,
                        {"records.jsonl": b"{}\n", "summary.json": b"{}\n"},
                    )
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob(".audit.*")), [])


if __name__ == "__main__":
    unittest.main()
