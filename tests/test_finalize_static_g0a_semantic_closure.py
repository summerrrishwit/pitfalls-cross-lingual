import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "finalize_static_g0a_semantic_closure.py"
SPEC = importlib.util.spec_from_file_location(
    "finalize_static_g0a_semantic_closure", SCRIPT_PATH
)
closure = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = closure
SPEC.loader.exec_module(closure)


def lexical_pair(pair_id, left, right):
    return {
        "schema_version": closure.duplicate_closure.AUDIT_PAIR_SCHEMA_VERSION,
        "pair_id": pair_id,
        "match_types": ["answer_alias_exact"],
        "left": {
            "provenance": {
                "input_role": "current_bundle",
                "base_fact_id": left,
                "audit_record_id": f"audit_{left}",
                "input_record_sha256": "1" * 64,
            }
        },
        "right": {
            "provenance": {
                "input_role": "current_bundle",
                "base_fact_id": right,
                "audit_record_id": f"audit_{right}",
                "input_record_sha256": "2" * 64,
            }
        },
    }


def review(pair_id, decision):
    return {
        "pair_id": pair_id,
        "decision": decision,
        "rationale": f"reason for {pair_id}",
        "reviewer_id": "codex-test",
        "confidence": "high",
    }


class FinalizeStaticG0ASemanticClosureTests(unittest.TestCase):
    def test_structural_correction_is_explicit_and_provenance_preserving(self):
        pair = lexical_pair("lex_external", "outside_a", "outside_b")
        source = review("lex_external", "exclude_cohort")
        source_binding = {
            "path": "/tmp/source-shard.jsonl",
            "sha256": "3" * 64,
            "byte_count": 1,
            "record_count": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            correction_path = Path(directory) / "corrections.json"
            correction_path.write_text(
                json.dumps(
                    {
                        "reason": "exclude_cohort cannot target external nodes",
                        "corrections": {
                            "lex_external": "The external facts are distinct."
                        },
                    }
                ),
                encoding="utf-8",
            )
            corrected, audit, stats = closure._apply_structural_corrections(
                candidate_pairs=[pair],
                review_rows=[source],
                review_provenance={
                    "lex_external": {
                        "source_shard": source_binding,
                        "source_review_record_sha256": closure.g0a.sha256_value(source),
                    }
                },
                correction_path=correction_path,
                cohort_ids={"inside"},
            )

        self.assertEqual(corrected[0]["decision"], "distinct")
        self.assertEqual(source["decision"], "exclude_cohort")
        self.assertEqual(audit[0]["source_decision"], "exclude_cohort")
        self.assertEqual(audit[0]["corrected_decision"], "distinct")
        self.assertEqual(audit[0]["source_review_shard"], source_binding)
        self.assertFalse(audit[0]["semantic_equivalence_silently_rewritten"])
        self.assertEqual(stats["structural_correction_count"], 1)

    def test_structural_corrections_must_exactly_cover_invalid_exclusions(self):
        pair = lexical_pair("lex_external", "outside_a", "outside_b")
        with tempfile.TemporaryDirectory() as directory:
            correction_path = Path(directory) / "corrections.json"
            correction_path.write_text(
                json.dumps({"reason": "none", "corrections": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must exactly cover"):
                closure._apply_structural_corrections(
                    candidate_pairs=[pair],
                    review_rows=[review("lex_external", "exclude_cohort")],
                    review_provenance={},
                    correction_path=correction_path,
                    cohort_ids={"inside"},
                )

    def test_preapplied_structural_correction_is_revalidated_not_reapplied(self):
        pair = lexical_pair("lex_external", "outside_a", "outside_b")
        corrected_source = review("lex_external", "distinct")
        corrected_source["rationale"] = "The external records are distinct."
        corrected_source["reviewer_id"] = "codex-test+root-structural-audit"
        with tempfile.TemporaryDirectory() as directory:
            correction_path = Path(directory) / "corrections.json"
            correction_path.write_text(
                json.dumps(
                    {
                        "reason": "exclude_cohort cannot target external nodes",
                        "corrections": {
                            "lex_external": "The external records are distinct."
                        },
                    }
                ),
                encoding="utf-8",
            )
            corrected, audit, _ = closure._apply_structural_corrections(
                candidate_pairs=[pair],
                review_rows=[corrected_source],
                review_provenance={
                    "lex_external": {
                        "source_shard": {
                            "path": "/tmp/corrected-shard.jsonl",
                            "sha256": "4" * 64,
                            "byte_count": 1,
                            "record_count": 1,
                        },
                        "source_review_record_sha256": closure.g0a.sha256_value(
                            corrected_source
                        ),
                    }
                },
                correction_path=correction_path,
                cohort_ids={"inside"},
            )

        self.assertEqual(corrected, [corrected_source])
        self.assertEqual(
            audit[0]["correction_application_status"],
            "already_applied_in_source_shard_and_revalidated",
        )

    def test_positive_full_pool_connectivity_projects_through_external_node(self):
        first = lexical_pair("lex_1", "pbf_a", "outside")
        second = lexical_pair("lex_2", "outside", "pbf_b")
        adjudications = [
            {"pair_id": "lex_1", "decision": "same_leakage_component"},
            {"pair_id": "lex_2", "decision": "same_fact"},
        ]
        projected, direct, components = closure._project_full_pool_connectivity(
            candidate_pairs=[first, second],
            adjudications=adjudications,
            cohort_ids={"pbf_a", "pbf_b"},
        )

        self.assertEqual(direct, {})
        self.assertEqual(
            projected[("pbf_a", "pbf_b")]["relationship"],
            "same_leakage_component",
        )
        self.assertEqual(
            projected[("pbf_a", "pbf_b")][
                "supporting_full_pool_positive_pair_ids"
            ],
            ["lex_1", "lex_2"],
        )
        self.assertEqual(len(components), 1)

    def test_overlap_disagreement_fails_closed(self):
        candidate = {
            "schema_version": closure.g0a.SEMANTIC_CANDIDATE_SCHEMA,
            "candidate_id": "semantic_pair_a",
            "left_base_fact_id": "pbf_a",
            "right_base_fact_id": "pbf_b",
        }
        decision = {
            "candidate_id": "semantic_pair_a",
            "decision": "same_leakage_component",
        }
        with self.assertRaisesRegex(ValueError, "reviews disagree"):
            closure._merge_current_semantic_evidence(
                semantic_candidates=[candidate],
                semantic_decision_by_id={"semantic_pair_a": decision},
                projected_pairs={},
                direct_current_reviews={("pbf_a", "pbf_b"): "same_fact"},
                staging_by_id={},
                resolved_by_id={},
            )

    def test_positive_component_crossing_existing_split_fails_closed(self):
        rows = {
            "pbf_a": {
                "split_assignment": "development",
                "split_group_id": "group_a",
            },
            "pbf_b": {
                "split_assignment": "validation",
                "split_group_id": "group_b",
            },
        }
        with self.assertRaisesRegex(ValueError, "crosses existing split assignments"):
            closure._validate_split_integrity(
                resolved_by_id=rows, positive_pairs=[("pbf_a", "pbf_b")]
            )

    def test_replacement_screen_enumerates_every_other_current_row(self):
        replacement = {
            "base_fact_id": "pbf_replacement",
            "source_question_en": "What is a quasar?",
            "canonical_fact_en": "A quasar is an active galactic nucleus.",
            "answer_en": "active galactic nucleus",
            "answer_aliases_en": ["active galactic nucleus"],
            "cohort_repair_lineage": {"disposition": "replacement"},
        }
        other_a = {
            "base_fact_id": "pbf_a",
            "source_question_en": "Who wrote Hamlet?",
            "canonical_fact_en": "William Shakespeare wrote Hamlet.",
            "answer_en": "William Shakespeare",
            "answer_aliases_en": ["William Shakespeare"],
        }
        other_b = {
            "base_fact_id": "pbf_b",
            "source_question_en": "When was Apollo 11 launched?",
            "canonical_fact_en": "Apollo 11 launched in 1969.",
            "answer_en": "1969",
            "answer_aliases_en": ["1969"],
        }
        policy = {
            "answer_normalization_version": "nfkc-casefold-whitespace-v1",
            "text_normalization_version": "nfkc-casefold-unicode-word-v1",
            "question_near_threshold": 0.8,
            "canonical_fact_near_threshold": 0.8,
        }
        rows, summary = closure._replacement_all_pairs_screen(
            resolved_rows=[replacement, other_a, other_b],
            fresh_pairs=[],
            lexical_policy=policy,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["screened_pair_count"], 2)
        self.assertEqual(summary["other_current_row_count_per_replacement"], 2)
        self.assertTrue(summary["all_replacement_to_other_current_pairs_screened"])
        self.assertFalse(summary["semantic_near_duplicate_recall_guaranteed"])


if __name__ == "__main__":
    unittest.main()
