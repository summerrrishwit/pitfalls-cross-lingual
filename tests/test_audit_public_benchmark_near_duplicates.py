import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_public_benchmark_near_duplicates.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_public_benchmark_near_duplicates", SCRIPT_PATH
)
audit_module = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = audit_module
SCRIPT_SPEC.loader.exec_module(audit_module)


def bundle_record(
    identifier,
    *,
    split="development",
    split_group=None,
    question="Who wrote the novel Dune?",
    fact="Frank Herbert wrote the novel Dune.",
    answer="Frank Herbert",
    aliases=None,
):
    return {
        "schema_version": "factual-perturbation-input-bundle-v1",
        "base_fact_id": f"pbf_{identifier}",
        "candidate_id": f"candidate_{identifier}",
        "source_id": f"source_{identifier}",
        "source_dataset": "fixture",
        "source_subset": "unit",
        "source_question_en": question,
        "canonical_fact_en": fact,
        "answer_en": answer,
        "answer_aliases_en": list(aliases or [answer]),
        "split_assignment": split,
        "split_group_id": split_group or f"group_{identifier}",
        "split_status": "provisional_not_frozen",
        "provenance": {
            "input_path": "/raw/triple_extractions.jsonl",
            "input_record_sha256": identifier.rjust(64, "0"),
            "source_model": "qwen3.7-plus",
        },
    }


def canonical_record(
    identifier,
    *,
    question="Who is the author of the Dune novel?",
    fact="Frank Herbert is the author of the novel Dune.",
    answer="Frank Herbert",
):
    return {
        "source_id": f"old_source_{identifier}",
        "candidate_id": f"old_candidate_{identifier}",
        "source_dataset": "old_fixture",
        "source_path": "/raw/old.json",
        "source_original_index": 7,
        "source_question": question,
        "canonical_fact": fact,
        "answer": answer,
        "source_answer": answer,
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records
        ),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def find_candidate_pair(pairs, first_id, second_id):
    target = tuple(sorted((first_id, second_id)))
    matches = [
        pair
        for pair in pairs
        if tuple(
            sorted(
                pair[side]["provenance"]["candidate_id"]
                for side in ("left", "right")
            )
        )
        == target
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one pair for {target}, got {len(matches)}")
    return matches[0]


def expected_review(*required_adjudications):
    return {
        "status": "pending_adjudication",
        "required_adjudications": list(required_adjudications),
        **{name: None for name in required_adjudications},
    }


class PublicBenchmarkNearDuplicateAuditTests(unittest.TestCase):
    def test_reports_alias_connected_and_declared_group_cross_split_leakage(self):
        rows = [
            bundle_record(
                "a",
                split="development",
                split_group="declared_bad",
                answer="United States",
                aliases=["United States", "America"],
                question="Which country contains the city of Boston?",
                fact="Boston is in the United States.",
            ),
            bundle_record(
                "b",
                split="validation",
                split_group="declared_bad",
                answer="USA",
                aliases=["USA", "America", "U.S.A."],
                question="In which country is Boston located?",
                fact="The city of Boston is located in the USA.",
            ),
            bundle_record(
                "c",
                split="sealed",
                answer="US",
                aliases=["US", "U.S.A."],
                question="What country is abbreviated US?",
                fact="US abbreviates the United States.",
            ),
            bundle_record(
                "d",
                split="development",
                answer="Paris",
                aliases=["Paris"],
                question="What is the capital of France?",
                fact="Paris is the capital of France.",
            ),
            bundle_record(
                "e",
                split="validation",
                answer=" PARIS ",
                aliases=["PARIS"],
                question="Name France's capital.",
                fact="The capital of France is Paris.",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "audit"
            write_jsonl(input_path, rows)
            summary = audit_module.audit(input_path, output_dir)

            checks = summary["split_integrity"]["checks"]
            self.assertEqual(checks["normalized_primary_answer"]["cross_split_group_count"], 1)
            self.assertEqual(
                checks["normalized_answer_alias_key"]["cross_split_group_count"], 3
            )
            self.assertEqual(
                checks["answer_alias_connected_component"]["cross_split_group_count"], 2
            )
            self.assertEqual(checks["declared_split_group_id"]["cross_split_group_count"], 1)
            self.assertTrue(summary["split_integrity"]["candidate_split_leakage_detected"])
            self.assertEqual(
                summary["split_integrity"]["recommended_action"],
                "adjudicate_detected_links_then_regroup_accepted_components_before_split_freeze",
            )
            self.assertEqual(
                summary["split_integrity"]["emitted_cross_split_candidate_pair_count"],
                3,
            )
            self.assertEqual(
                summary["split_integrity"][
                    "emitted_cross_split_alias_candidate_pair_count"
                ],
                3,
            )
            self.assertEqual(
                summary["split_integrity"][
                    "emitted_cross_split_text_candidate_pair_count"
                ],
                1,
            )
            self.assertTrue(
                summary["split_integrity"][
                    "emitted_cross_split_subcounts_are_nonexclusive"
                ]
            )

            pairs = read_jsonl(output_dir / "lexical_candidate_pairs.jsonl")
            alias_pairs = [item for item in pairs if "answer_alias_exact" in item["match_types"]]
            self.assertTrue(alias_pairs)
            self.assertTrue(any(item["cross_split"] for item in alias_pairs))
            alias_only_pair = find_candidate_pair(
                pairs, "candidate_b", "candidate_c"
            )
            self.assertEqual(
                alias_only_pair["review"],
                expected_review("alias_valid", "same_answer_entity"),
            )
            mixed_pair = find_candidate_pair(pairs, "candidate_d", "candidate_e")
            self.assertEqual(
                mixed_pair["review"],
                expected_review(
                    "alias_valid", "same_answer_entity", "semantic_duplicate"
                ),
            )
            all_provenance = [
                item[side]["provenance"] for item in alias_pairs for side in ("left", "right")
            ]
            self.assertTrue(
                all(item["input_path"] == str(input_path.resolve()) for item in all_provenance)
            )
            self.assertIn(
                audit_module.sha256_value(rows[0]),
                {item["input_record_sha256"] for item in all_provenance},
            )
            self.assertTrue(
                all(
                    item["upstream_provenance"]["source_model"] == "qwen3.7-plus"
                    for item in all_provenance
                )
            )

    def test_emits_exact_and_near_question_fact_candidates_against_old_canonical(self):
        current = [
            bundle_record("a"),
            bundle_record(
                "b",
                split="validation",
                question="Who wrote Dune novel",
                fact="Frank Herbert wrote novel Dune",
                answer="F. Herbert",
                aliases=["F. Herbert"],
            ),
            bundle_record(
                "c",
                question="WHO WROTE THE NOVEL DUNE!",
                fact="FRANK HERBERT WROTE THE NOVEL DUNE",
                answer="Herbert",
                aliases=["Herbert"],
            ),
        ]
        comparison = [canonical_record("one")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            comparison_path = root / "canonical.jsonl"
            output_dir = root / "audit"
            write_jsonl(input_path, current)
            write_jsonl(comparison_path, comparison)
            summary = audit_module.audit(
                input_path,
                output_dir,
                comparison_canonical_path=comparison_path,
                question_threshold=0.75,
                fact_threshold=0.75,
            )

            pairs = read_jsonl(output_dir / "lexical_candidate_pairs.jsonl")
            match_types = {kind for item in pairs for kind in item["match_types"]}
            self.assertIn("question_exact", match_types)
            self.assertIn("canonical_fact_exact", match_types)
            self.assertIn("question_near_lexical", match_types)
            self.assertIn("canonical_fact_near_lexical", match_types)
            self.assertTrue(
                any(item["audit_scope"] == "current_vs_comparison_canonical" for item in pairs)
            )
            self.assertEqual(
                find_candidate_pair(pairs, "candidate_a", "candidate_b")["review"],
                expected_review("semantic_duplicate"),
            )
            self.assertEqual(
                find_candidate_pair(pairs, "candidate_a", "candidate_c")["review"],
                expected_review("semantic_duplicate"),
            )
            self.assertEqual(
                find_candidate_pair(pairs, "candidate_a", "old_candidate_one")[
                    "review"
                ],
                expected_review("alias_valid", "same_answer_entity"),
            )
            for pair in pairs:
                for match_type, raw_key in (
                    ("question_near_lexical", "question_raw"),
                    ("canonical_fact_near_lexical", "canonical_fact_raw"),
                ):
                    if match_type not in pair["match_types"]:
                        continue
                    evidence = pair["match_evidence"][match_type]
                    self.assertEqual(
                        evidence["left_normalized_text"],
                        audit_module.normalize_lexical(pair["left"][raw_key]),
                    )
                    self.assertEqual(
                        evidence["right_normalized_text"],
                        audit_module.normalize_lexical(pair["right"][raw_key]),
                    )
            self.assertEqual(summary["semantic_review_status"], "not_performed")
            self.assertFalse(summary["semantic_review_complete"])
            self.assertEqual(
                summary["limitations"]["audit_kind"], "lexical_candidate_audit_only"
            )
            self.assertFalse(summary["policy"]["network_or_model_used"])
            self.assertEqual(
                summary["policy"]["pair_review_contract"],
                {
                    "status": "pending_adjudication",
                    "answer_alias_exact_requires": [
                        "alias_valid",
                        "same_answer_entity",
                    ],
                    "alias_valid_scope": (
                        "pair_level_at_least_one_shared_alias_is_valid_for_both_records"
                    ),
                    "question_or_fact_exact_or_near_requires": [
                        "semantic_duplicate"
                    ],
                    "mixed_evidence_requires_union": True,
                },
            )
            self.assertFalse(
                summary["limitations"]["semantic_adjudication_performed"]
            )
            self.assertEqual(
                summary["candidate_generation"]["emitted_representative_pair_count"],
                summary["candidate_generation"]["candidate_pair_count"],
            )
            self.assertFalse(
                summary["candidate_generation"]["exhaustive_record_pair_census"]
            )
            self.assertEqual(
                summary["schema_version"],
                "public-benchmark-lexical-candidate-audit-v2",
            )
            self.assertEqual(
                {pair["schema_version"] for pair in pairs},
                {"public-benchmark-lexical-candidate-pair-v2"},
            )
            self.assertEqual(
                summary["artifacts"]["candidate_pairs"]["sha256"],
                audit_module.sha256_file(output_dir / "lexical_candidate_pairs.jsonl"),
            )

    def test_near_only_cross_split_pair_triggers_candidate_leakage(self):
        rows = [
            bundle_record("a"),
            bundle_record(
                "b",
                split="validation",
                question="Who wrote Dune novel",
                fact="Frank Herbert wrote novel Dune",
                answer="F. Herbert",
                aliases=["F. Herbert"],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            output_dir = root / "audit"
            write_jsonl(input_path, rows)
            summary = audit_module.audit(
                input_path,
                output_dir,
                question_threshold=0.75,
                fact_threshold=0.75,
            )

            pairs = read_jsonl(output_dir / "lexical_candidate_pairs.jsonl")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(
                pairs[0]["match_types"],
                ["canonical_fact_near_lexical", "question_near_lexical"],
            )
            self.assertTrue(pairs[0]["cross_split"])
            split_integrity = summary["split_integrity"]
            self.assertEqual(
                split_integrity["cross_split_violation_group_count_sum"], 0
            )
            self.assertEqual(
                summary["candidate_generation"]["cross_split_candidate_pair_count"],
                1,
            )
            self.assertEqual(
                split_integrity["emitted_cross_split_candidate_pair_count"], 1
            )
            self.assertEqual(
                split_integrity["emitted_cross_split_text_candidate_pair_count"], 1
            )
            self.assertEqual(
                split_integrity["emitted_cross_split_alias_candidate_pair_count"], 0
            )
            self.assertTrue(split_integrity["candidate_split_leakage_detected"])
            self.assertEqual(
                split_integrity["recommended_action"],
                "adjudicate_detected_links_then_regroup_accepted_components_before_split_freeze",
            )

    def test_each_text_evidence_type_requires_semantic_adjudication(self):
        text_evidence_types = (
            "question_exact",
            "question_near_lexical",
            "canonical_fact_exact",
            "canonical_fact_near_lexical",
        )
        self.assertEqual(
            audit_module.TEXT_EVIDENCE_TYPES,
            frozenset(text_evidence_types),
        )
        for match_type in text_evidence_types:
            with self.subTest(match_type=match_type):
                self.assertEqual(
                    audit_module.build_review_contract([match_type]),
                    expected_review("semantic_duplicate"),
                )
        with self.assertRaisesRegex(ValueError, "No review contract"):
            audit_module.build_review_contract(["unsupported_match_type"])

    def test_invalid_split_recommendation_is_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            write_jsonl(input_path, [bundle_record("invalid", split="training")])
            summary = audit_module.audit(input_path, root / "audit")

            split_integrity = summary["split_integrity"]
            self.assertEqual(
                split_integrity["missing_or_invalid_split_record_count"], 1
            )
            self.assertTrue(split_integrity["candidate_split_leakage_detected"])
            self.assertEqual(
                split_integrity["recommended_action"],
                "repair_missing_or_invalid_split_assignments_before_split_freeze",
            )
            self.assertEqual(
                summary["recommended_next_step"],
                split_integrity["recommended_action"],
            )

    def test_invalid_split_and_detected_link_recommendation_combines_actions(self):
        rows = [
            bundle_record("a"),
            bundle_record(
                "b",
                split="validation",
                question="Who wrote Dune novel",
                fact="Frank Herbert wrote novel Dune",
                answer="F. Herbert",
            ),
            bundle_record(
                "invalid",
                split="training",
                question="Which planet is closest to the Sun?",
                fact="Mercury is the closest planet to the Sun.",
                answer="Mercury",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            write_jsonl(input_path, rows)
            summary = audit_module.audit(
                input_path,
                root / "audit",
                question_threshold=0.75,
                fact_threshold=0.75,
            )

            split_integrity = summary["split_integrity"]
            self.assertEqual(
                split_integrity["missing_or_invalid_split_record_count"], 1
            )
            self.assertEqual(
                split_integrity["emitted_cross_split_candidate_pair_count"], 1
            )
            self.assertEqual(
                split_integrity["recommended_action"],
                (
                    "repair_missing_or_invalid_split_assignments_and_adjudicate_"
                    "detected_links_before_split_freeze"
                ),
            )
            self.assertEqual(
                summary["recommended_next_step"],
                split_integrity["recommended_action"],
            )

    def test_declared_group_only_cross_split_still_triggers_leakage(self):
        rows = [
            bundle_record(
                "a",
                split="development",
                split_group="shared_group",
                question="Who wrote the novel Dune?",
                fact="Frank Herbert wrote the novel Dune.",
                answer="Frank Herbert",
            ),
            bundle_record(
                "b",
                split="validation",
                split_group="shared_group",
                question="Which planet is closest to the Sun?",
                fact="Mercury is the closest planet to the Sun.",
                answer="Mercury",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            write_jsonl(input_path, rows)
            summary = audit_module.audit(input_path, root / "audit")

            split_integrity = summary["split_integrity"]
            self.assertEqual(
                summary["candidate_generation"]["candidate_pair_count"], 0
            )
            self.assertEqual(
                split_integrity["emitted_cross_split_candidate_pair_count"], 0
            )
            self.assertEqual(
                split_integrity["checks"]["declared_split_group_id"][
                    "cross_split_group_count"
                ],
                1,
            )
            self.assertTrue(split_integrity["candidate_split_leakage_detected"])

    def test_fixed_input_produces_byte_identical_outputs(self):
        rows = [
            bundle_record("a"),
            bundle_record(
                "b",
                question="Who wrote Dune novel",
                fact="Frank Herbert wrote novel Dune",
                answer="F. Herbert",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            write_jsonl(input_path, rows)
            first = root / "first"
            second = root / "second"
            first_summary = audit_module.audit(
                input_path, first, question_threshold=0.70, fact_threshold=0.70
            )
            audit_module.audit(input_path, second, question_threshold=0.70, fact_threshold=0.70)
            for name in ("lexical_candidate_pairs.jsonl", "lexical_audit_summary.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertFalse(list(first.glob("*.tmp")))
            self.assertFalse(list(second.glob("*.tmp")))
            self.assertGreater(
                first_summary["candidate_generation"]["candidate_pair_count"], 0
            )
            self.assertEqual(
                first_summary["split_integrity"][
                    "emitted_cross_split_candidate_pair_count"
                ],
                0,
            )
            self.assertFalse(
                first_summary["split_integrity"]["candidate_split_leakage_detected"]
            )

    def test_bounded_blocking_does_not_enumerate_all_pairs(self):
        rows = [
            bundle_record(
                str(index),
                question=f"Which city number {index} hosts festival code {index}?",
                fact=f"City number {index} hosts festival code {index}.",
                answer=f"Answer {index}",
            )
            for index in range(240)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "bundle.jsonl"
            write_jsonl(input_path, rows)
            summary = audit_module.audit(
                input_path,
                root / "audit",
                max_bucket_neighbors=4,
            )
            stats = summary["candidate_generation"]["question_near"]
            self.assertFalse(stats["all_pairs_enumerated"])
            self.assertLess(
                stats["blocked_candidate_text_pair_count"], stats["possible_all_pairs"]
            )
            self.assertGreater(stats["truncated_bucket_count"], 0)

    def test_cli_and_schema_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_path = root / "invalid.jsonl"
            write_jsonl(invalid_path, [{"schema_version": "wrong", "base_fact_id": "pbf_x"}])
            with self.assertRaisesRegex(ValueError, "Unsupported bundle schema"):
                audit_module.audit(invalid_path, root / "invalid-output")

            input_path = root / "bundle.jsonl"
            output_dir = root / "cli-output"
            write_jsonl(input_path, [bundle_record("valid")])
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input-bundle",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertFalse(payload["semantic_review_complete"])
            self.assertTrue((output_dir / "lexical_audit_summary.json").is_file())
            self.assertTrue((output_dir / "lexical_candidate_pairs.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
