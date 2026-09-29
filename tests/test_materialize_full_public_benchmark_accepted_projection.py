import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "materialize_full_public_benchmark_accepted_projection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "full_accepted_candidate_projection", SCRIPT_PATH
)
projection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(projection)

RUNNER_TEST_PATH = PROJECT_ROOT / "tests" / "test_run_full_public_benchmark_candidate_review.py"
RUNNER_TEST_SPEC = importlib.util.spec_from_file_location(
    "candidate_review_test_helpers_for_projection", RUNNER_TEST_PATH
)
runner_helpers = importlib.util.module_from_spec(RUNNER_TEST_SPEC)
assert RUNNER_TEST_SPEC.loader is not None
RUNNER_TEST_SPEC.loader.exec_module(runner_helpers)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_bytes(
        b"".join(
            projection.candidate_review.canonical_json_bytes(dict(row)) + b"\n"
            for row in rows
        )
    )


def jsonl_binding(path, schema, rows):
    return {
        "filename": path.name,
        "sha256": projection.candidate_review.sha256_file(path),
        "byte_count": path.stat().st_size,
        "record_count": len(rows),
        "schema_version": schema,
    }


def json_binding(path, schema):
    return {
        "filename": path.name,
        "sha256": projection.candidate_review.sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema,
    }


class AcceptedCandidateProjectionTests(unittest.TestCase):
    FACT_SPECS = (
        ("pbf_a", "author", "Author A", "Author A wrote Book A.", "Who wrote Book A?"),
        ("pbf_b", "author", "Author B", "Author B wrote Book B.", "Who wrote Book B?"),
        ("pbf_c", "author", "Author C", "Author C wrote Book C.", "Who wrote Book C?"),
        ("pbf_g", "author", "Author G", "Author G wrote Book G.", "Who wrote Book G?"),
        ("pbf_d", "location", "City D", "Museum D is in City D.", "Where is Museum D?"),
        ("pbf_e", "location", "City E", "Museum E is in City E.", "Where is Museum E?"),
    )

    def make_fixture(self, root, *, cascade=False, reject_ids=()):
        source = root / "source"
        source.mkdir()
        facts = [runner_helpers.full_fact(*spec) for spec in self.FACT_SPECS]
        facts_by_id = {row["base_fact_id"]: row for row in facts}
        distractor_sources = {
            "pbf_a": ["pbf_b", "pbf_c"],
            "pbf_b": ["pbf_a", "pbf_a" if cascade else "pbf_c"],
            "pbf_c": ["pbf_g", "pbf_g"],
            "pbf_g": ["pbf_c", "pbf_c"],
            "pbf_d": ["pbf_e", "pbf_e"],
            "pbf_e": ["pbf_d", "pbf_d"],
        }
        neutral_sources = {
            "pbf_a": ["pbf_d", "pbf_e"],
            "pbf_b": ["pbf_d", "pbf_e"],
            "pbf_c": ["pbf_d", "pbf_e"],
            "pbf_g": ["pbf_d", "pbf_e"],
            "pbf_d": ["pbf_c", "pbf_g"],
            "pbf_e": ["pbf_c", "pbf_g"],
        }
        distractors = []
        neutrals = []
        for target_id, sources in distractor_sources.items():
            target = facts_by_id[target_id]
            for slot, source_id in enumerate(sources, start=1):
                donor = facts_by_id[source_id]
                distractors.append(
                    {
                        "schema_version": projection.candidate_review.DISTRACTOR_SCHEMA,
                        "distractor_id": f"d_{target_id}_{slot}",
                        "base_fact_id": target_id,
                        "slot": slot,
                        "source_base_fact_id": source_id,
                        "distractor_text_en": donor["answer_en"],
                        "relation_partition_id": target["relation_partition_id"],
                        "answer_type_match_tier": "same_family_same_answer_type_bucket",
                        "split_assignment": "development",
                        "same_split": True,
                        "same_leakage_component": False,
                        "verified": False,
                        "review_status": "pending_factual_uniqueness_and_semantic_review",
                    }
                )
        for target_id, sources in neutral_sources.items():
            target = facts_by_id[target_id]
            for slot, source_id in enumerate(sources, start=1):
                donor = facts_by_id[source_id]
                neutrals.append(
                    {
                        "schema_version": projection.candidate_review.NEUTRAL_SCHEMA,
                        "neutral_candidate_id": f"n_{target_id}_{slot}",
                        "base_fact_id": target_id,
                        "slot": slot,
                        "source_base_fact_id": source_id,
                        "neutral_context_candidate_en": donor["canonical_fact_en"],
                        "source_relation_partition_id": donor[
                            "relation_partition_id"
                        ],
                        "target_relation_partition_id": target[
                            "relation_partition_id"
                        ],
                        "split_assignment": "development",
                        "same_split": True,
                        "same_leakage_component": False,
                        "verified_unrelated": False,
                        "review_status": "pending_unrelatedness_and_length_review",
                    }
                )
        components = [
            {
                "schema_version": projection.COMPONENT_SCHEMA_VERSION,
                "leakage_component_id": row["leakage_component_id"],
                "base_fact_ids": [row["base_fact_id"]],
                "member_count": 1,
                "split_assignment": row["split_assignment"],
            }
            for row in facts
        ]
        split_manifest = {
            "schema_version": projection.SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_status": "provisional_not_frozen",
            "formal_split_freeze_performed": False,
            "base_fact_count": len(facts),
            "leakage_component_count": len(components),
            "actual_base_fact_counts": {"development": len(facts)},
        }

        full_path = source / "full_base_facts.jsonl"
        distractor_path = source / "distractor_candidates.jsonl"
        neutral_path = source / "neutral_reference_candidates.jsonl"
        component_path = source / "leakage_components.jsonl"
        split_path = source / "split_manifest.json"
        write_jsonl(full_path, facts)
        write_jsonl(distractor_path, distractors)
        write_jsonl(neutral_path, neutrals)
        write_jsonl(component_path, components)
        write_json(split_path, split_manifest)
        outputs = {
            "full_base_facts": jsonl_binding(
                full_path, projection.candidate_review.FULL_FACT_SCHEMA, facts
            ),
            "distractor_candidates": jsonl_binding(
                distractor_path,
                projection.candidate_review.DISTRACTOR_SCHEMA,
                distractors,
            ),
            "neutral_reference_candidates": jsonl_binding(
                neutral_path,
                projection.candidate_review.NEUTRAL_SCHEMA,
                neutrals,
            ),
            "leakage_components": jsonl_binding(
                component_path, projection.COMPONENT_SCHEMA_VERSION, components
            ),
            "split_manifest": json_binding(
                split_path, projection.SPLIT_MANIFEST_SCHEMA_VERSION
            ),
        }
        false_flags = {
            field: False
            for field in (
                "canonical_freeze_emitted",
                "review_freeze_emitted",
                "split_freeze_emitted",
                "distractor_candidates_verified",
                "neutral_candidates_verified_unrelated",
                "hf_checkpoint_bound",
                "hf_tokenizer_bound",
                "hf_model_executed",
                "hf_tokenizer_executed",
                "behavior_executed",
                "validation_exposed",
                "sealed_exposed",
                "perturbation_authorized",
                "path_not_token_authorized",
                "human_gold",
            )
        }
        postreview_manifest = {
            "schema_version": projection.candidate_review.POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA,
            "status": projection.candidate_review.POSTREVIEW_STATUS,
            "outputs": outputs,
            "counts": {
                "retained_base_facts": len(facts),
                "distractor_candidates": len(distractors),
                "neutral_reference_candidates": len(neutrals),
            },
            "safety_contract": {
                "output_is_provisional": True,
                "split_recomputed": True,
                **false_flags,
            },
            "fact_review_contract": {
                "revise_defer_missing_count": 0,
                "proxy_review_only": True,
                "human_gold": False,
            },
            "semantic_review_contract": {
                "candidate_adjudication_complete_for_bounded_set": True,
                "reviewer_evidence_is_human_gold": False,
            },
            "candidate_repair_contract": {
                "predecessor_candidate_reviews_blanket_valid": False,
                "exact_v3_accepted_projection_evidence_carry_forward_eligible": True,
                "candidate_id_or_row_sha_alone_sufficient_for_review_reuse": False,
                "full_candidate_evidence_coverage_required": True,
                "full_candidate_model_rereview_required": False,
                "changed_or_new_candidate_fresh_model_review_required": True,
            },
            "integrity": {"all_checks_passed": True},
        }
        manifest_path = source / "postreview_rebuild_manifest.json"
        write_json(manifest_path, postreview_manifest)
        paths = {
            "manifest": manifest_path,
            "full": full_path,
            "distractors": distractor_path,
            "neutrals": neutral_path,
            "facts": facts,
            "distractor_rows": distractors,
            "neutral_rows": neutrals,
        }
        runner_case = runner_helpers.CandidateReviewRunnerTests()
        review_result = runner_case.run_fixture(
            paths,
            root / "review",
            runner_helpers.FakeRouter(reject_ids=reject_ids),
            batch_size=8,
            max_workers=1,
        )
        return {
            "postreview": manifest_path,
            "review_manifest": Path(review_result["manifest_path"]),
            "adjudications": Path(review_result["adjudications_path"]),
            "components": component_path,
            "split": split_path,
        }

    def materialize_fixture(self, fixture, output):
        return projection.materialize(
            postreview_manifest_path=fixture["postreview"],
            candidate_review_manifest_path=fixture["review_manifest"],
            candidate_adjudications_path=fixture["adjudications"],
            output_dir=output,
        )

    def test_nonaccept_source_rows_are_allowed_and_slot_order_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root, reject_ids={"d_pbf_a_1"})
            result = self.materialize_fixture(fixture, root / "projection-a")
            context = projection.load_projection_context(
                Path(result["manifest_path"]),
                expected_postreview_manifest_path=fixture["postreview"],
                expected_review_manifest_path=fixture["review_manifest"],
                expected_adjudications_path=fixture["adjudications"],
            )
            self.assertEqual(result["projected_base_fact_count"], 6)
            self.assertEqual(result["quarantined_base_fact_count"], 0)
            self.assertFalse(result["source_review_all_accept"])
            self.assertTrue(result["selected_projection_all_accept"])
            selected = {
                row["base_fact_id"]: row["distractor_id"]
                for row in context["distractor_candidates"]
            }
            self.assertEqual(selected["pbf_a"], "d_pbf_a_2")
            self.assertTrue(
                all(row["slot"] == 1 for row in context["neutral_reference_candidates"])
            )
            self.assertEqual(
                projection.candidate_review.sha256_file(
                    context["output_paths"]["leakage_components"]
                ),
                projection.candidate_review.sha256_file(fixture["components"]),
            )
            self.assertEqual(
                projection.candidate_review.sha256_file(
                    context["output_paths"]["split_manifest"]
                ),
                projection.candidate_review.sha256_file(fixture["split"]),
            )
            second = self.materialize_fixture(fixture, root / "projection-b")
            self.assertEqual(
                Path(result["manifest_path"]).read_bytes(),
                Path(second["manifest_path"]).read_bytes(),
            )

    def test_monotone_fixed_point_quarantines_donor_dependants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(
                root,
                cascade=True,
                reject_ids={"d_pbf_a_1", "d_pbf_a_2"},
            )
            result = self.materialize_fixture(fixture, root / "projection")
            context = projection.load_projection_context(Path(result["manifest_path"]))
            quarantine = {
                row["base_fact_id"]: row for row in context["quarantine"]
            }
            self.assertEqual(context["projected_target_ids"], [
                "pbf_c",
                "pbf_g",
                "pbf_d",
                "pbf_e",
            ])
            self.assertEqual(quarantine["pbf_a"]["fixed_point_round"], 1)
            self.assertEqual(quarantine["pbf_b"]["fixed_point_round"], 2)
            self.assertEqual(
                quarantine["pbf_b"]["reason_by_kind"]["distractor"],
                "all_accepted_candidate_donors_outside_projection_cohort",
            )
            self.assertTrue(
                all(row["factual_falsehood_asserted"] is False for row in quarantine.values())
            )
            retained = set(context["projected_target_ids"])
            for row in (
                context["distractor_candidates"]
                + context["neutral_reference_candidates"]
            ):
                self.assertIn(row["source_base_fact_id"], retained)

    def test_rejects_incomplete_full_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            review_manifest = json.loads(
                fixture["review_manifest"].read_text(encoding="utf-8")
            )
            review_manifest["status"] = "running"
            write_json(fixture["review_manifest"], review_manifest)
            with self.assertRaisesRegex(ValueError, "not completed"):
                self.materialize_fixture(fixture, root / "projection")
            self.assertFalse((root / "projection").exists())

    def test_loader_rejects_tampered_projection_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            result = self.materialize_fixture(fixture, root / "projection")
            manifest_path = Path(result["manifest_path"])
            candidate_path = manifest_path.parent / "distractor_candidates.jsonl"
            candidate_path.write_text(
                candidate_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                projection.load_projection_context(manifest_path)


if __name__ == "__main__":
    unittest.main()
