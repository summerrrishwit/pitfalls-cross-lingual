import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from factual_pitfalls import perturbation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_public_benchmark_provisional.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "prepare_public_benchmark_provisional", SCRIPT_PATH
)
provisional = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(provisional)


def record(
    candidate_id,
    *,
    model="qwen3.7-plus",
    terminal_status="completed",
    extraction_status="extracted",
    validation_errors=None,
    subject="Example Film",
    subject_type="film",
    relation_raw="release date",
    answer="1 January 2020",
    answer_type="date",
    canonical_fact="Example Film was released on 1 January 2020.",
    source_question="When was Example Film released?",
    source_choices=None,
):
    extracted = extraction_status == "extracted"
    return {
        "source_id": candidate_id,
        "source_dataset": "fixture",
        "source_subset": "unit",
        "source_path": "data/source/fixture.json",
        "source_original_index": int(candidate_id.rsplit("_", 1)[-1]),
        "source_question": source_question,
        "source_choices": list(source_choices or []),
        "source_answer": answer,
        "candidate_id": candidate_id,
        "source_snapshot": {
            "dataset": "fixture",
            "record": {
                "source_format": "multiple_choice" if source_choices else "open_qa",
                "upstream_metadata": {"answer_aliases": [answer]},
            },
        },
        "created_at": "2026-09-13T00:00:00Z",
        "extraction": {
            "model": model,
            "api_model": model,
            "prompt_version": "atomic-factual-triple-v6",
            "attempt_count": 1,
        },
        "terminal_status": terminal_status,
        "extraction_status": extraction_status,
        "exclusion_reason": None if extracted else "not_factual",
        "subject": subject if extracted else None,
        "subject_type": subject_type if extracted else None,
        "relation_raw": relation_raw if extracted else None,
        "answer": answer if extracted else None,
        "answer_type": answer_type if extracted else None,
        "canonical_fact": canonical_fact if extracted else None,
        "extraction_confidence": 0.95,
        "parsed_response": {},
        "validation_errors": list(validation_errors or []),
    }


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PublicBenchmarkProvisionalPreparationTests(unittest.TestCase):
    def fixture_records(self):
        return [
            record(
                "raw_fixture_000001",
                source_choices=["1 January 2020", "2 February 2020"],
            ),
            record(
                "raw_fixture_000002",
                subject=" example film ",
                relation_raw="Release Date",
                answer="1 JANUARY 2020",
                canonical_fact="Example Film was released on 1 JANUARY 2020.",
            ),
            record(
                "raw_fixture_000003",
                subject="France",
                subject_type="country",
                relation_raw="capital",
                answer="Paris",
                answer_type="city",
                canonical_fact="Paris is the capital of France.",
                source_question="What is the capital of France?",
            ),
            record("raw_fixture_000004", extraction_status="not_extractable"),
            record("raw_fixture_000005", validation_errors=["invalid_relation"]),
            record("raw_fixture_000006", terminal_status="validation_failed"),
        ]

    def run_fixture(self, directory, records=None, **overrides):
        root = Path(directory)
        input_path = root / "triple_extractions.jsonl"
        write_jsonl(input_path, records or self.fixture_records())
        output_dir = root / "output"
        summary = provisional.prepare(
            input_path,
            output_dir,
            provisional.DEFAULT_POLICY,
            review_sample_size=400,
            **overrides,
        )
        return input_path, output_dir, summary

    def test_filters_status_and_validation_errors_without_promoting_to_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, summary = self.run_fixture(directory)
            self.assertEqual(summary["selection"]["eligible_record_count"], 3)
            self.assertEqual(
                summary["selection"]["filter_reason_counts"],
                {
                    "extraction_status_not_extracted": 1,
                    "terminal_status_not_completed": 1,
                    "validation_errors_present": 1,
                },
            )
            rows = read_jsonl(output_dir / "provisional_triples.jsonl")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(item["canonical_status"] == "pending_review" for item in rows))
            self.assertTrue(
                all(item["evidence_tier"] == "provisional_single_model" for item in rows)
            )
            self.assertTrue(all(item["human_gold"] is False for item in rows))
            self.assertTrue(all(item["probe_relation_id"] is None for item in rows))

    def test_normalized_duplicate_records_share_one_base_fact_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, summary = self.run_fixture(directory)
            rows = read_jsonl(output_dir / "provisional_triples.jsonl")
            by_id = {item["candidate_id"]: item for item in rows}
            self.assertEqual(
                by_id["raw_fixture_000001"]["base_fact_id"],
                by_id["raw_fixture_000002"]["base_fact_id"],
            )
            self.assertEqual(summary["counts"]["base_fact_clusters"], 2)
            self.assertEqual(summary["counts"]["duplicate_base_fact_clusters"], 1)
            self.assertEqual(summary["counts"]["duplicate_record_excess"], 1)
            clusters = read_jsonl(output_dir / "base_fact_clusters.jsonl")
            duplicate = next(item for item in clusters if item["member_count"] == 2)
            self.assertEqual(duplicate["duplicate_status"], "duplicate_cluster")

    def test_unmapped_relation_does_not_gate_prompt_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_fixture(directory)
            rows = read_jsonl(output_dir / "provisional_triples.jsonl")
            unmatched = next(item for item in rows if item["relation_raw"] == "capital")
            self.assertIsNone(unmatched["probe_relation_candidate"])
            self.assertIsNone(unmatched["probe_relation_id"])
            self.assertEqual(unmatched["probe_relation_status"], "unassigned")
            self.assertTrue(unmatched["prompt_ready"])
            self.assertEqual(unmatched["prompt_quality_tier"], "open_answer_fallback")

            bundle = read_jsonl(output_dir / "behavior_input_bundle.jsonl")
            bundle_row = next(item for item in bundle if item["relation_raw"] == "capital")
            self.assertEqual(bundle_row["schema_version"], "factual-perturbation-input-bundle-v1")
            self.assertEqual(bundle_row["base_fact_id"], bundle_row["base_id"])
            self.assertTrue(bundle_row["prompt_ready"])
            self.assertFalse(bundle_row["admission"]["behavior_screening_ready"])
            self.assertEqual(bundle_row["experiment_status"]["hidden_state_collection"], "not_run")

            release_row = next(item for item in bundle if item["relation_raw"] == "release date")
            self.assertEqual(release_row["distractor_candidates"][0]["text_en"], "2 February 2020")

    def test_behavior_bundle_is_readable_by_perturbation_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_fixture(directory)
            bundle_path = output_dir / "behavior_input_bundle.jsonl"
            rows, metadata = perturbation._load_input_bundle(bundle_path)
            self.assertEqual(metadata["schema_version"], perturbation.INPUT_BUNDLE_SCHEMA_VERSION)
            self.assertEqual(len(rows), 2)
            self.assertTrue(
                all(
                    perturbation._input_canonical_status(row, metadata["schema_version"])
                    == "pending_review"
                    for row in rows
                )
            )
            release_row = next(item for item in rows if item["relation_raw"] == "release date")
            prepared = perturbation._prepared_distractors(release_row, max_distractors=3)
            self.assertEqual(prepared[0]["text_en"], "2 February 2020")

    def test_probe_candidates_are_versioned_and_keep_probe_id_null(self):
        records = [
            record("raw_fixture_000011"),
            record(
                "raw_fixture_000012",
                subject="1984",
                subject_type="book",
                relation_raw="release year",
                answer="1949",
                answer_type="year",
                canonical_fact="1984 was released in 1949.",
            ),
            record(
                "raw_fixture_000013",
                subject="Character A",
                subject_type="fictional character",
                relation_raw="portrayed by",
                answer="Actor A",
                answer_type="actor",
                canonical_fact="Character A was portrayed by Actor A.",
            ),
            record(
                "raw_fixture_000014",
                subject="Character B",
                subject_type="fictional character",
                relation_raw="voice actor",
                answer="Actor B",
                answer_type="actor",
                canonical_fact="The voice actor for Character B is Actor B.",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_fixture(directory, records=records)
            rows = {
                item["candidate_id"]: item
                for item in read_jsonl(output_dir / "provisional_triples.jsonl")
            }
            self.assertEqual(rows["raw_fixture_000011"]["probe_relation_candidate"], "work_to_release_date")
            self.assertEqual(rows["raw_fixture_000012"]["probe_relation_candidate"], "work_to_release_year")
            self.assertEqual(rows["raw_fixture_000013"]["probe_relation_candidate"], "character_to_actor")
            self.assertEqual(
                rows["raw_fixture_000014"]["probe_relation_candidate"],
                "character_to_voice_actor",
            )
            for item in rows.values():
                self.assertIsNone(item["probe_relation_id"])
                self.assertEqual(
                    item["probe_relation_status"], "pending_semantic_and_balance_review"
                )
                self.assertEqual(
                    item["probe_relation_candidate_policy"],
                    "public-benchmark-qwen-probe-relation-candidates-v2",
                )
            self.assertEqual(
                rows["raw_fixture_000011"]["probe_relation_candidate_direction"],
                "work -> release date",
            )

            artifact_names = (
                "base_fact_clusters.jsonl",
                "review_queue.jsonl",
                "review_sample.jsonl",
                "review_sample_overall_random.jsonl",
                "behavior_input_bundle.jsonl",
            )
            for artifact_name in artifact_names:
                artifact_rows = read_jsonl(output_dir / artifact_name)
                release = next(
                    item
                    for item in artifact_rows
                    if item["base_fact_id"]
                    == rows["raw_fixture_000011"]["base_fact_id"]
                )
                self.assertEqual(
                    release["probe_relation_candidate_direction"],
                    "work -> release date",
                )

    def test_review_queue_and_deterministic_stratified_sample_cover_base_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, summary = self.run_fixture(directory)
            queue = read_jsonl(output_dir / "review_queue.jsonl")
            sample = read_jsonl(output_dir / "review_sample.jsonl")
            self.assertEqual(len(queue), summary["counts"]["base_fact_clusters"])
            self.assertEqual(len(sample), len(queue))
            coverage = summary["review_sampling"]["coverage_oriented_sample"]
            self.assertEqual(coverage["actual_size"], len(sample))
            self.assertFalse(coverage["sampling_weights_provided"])
            self.assertFalse(coverage["population_estimation_supported"])
            self.assertTrue(all(item["review_decision"] is None for item in queue))
            self.assertTrue(all(item["canonical_status"] == "pending_review" for item in sample))

    def test_term_for_definition_is_explicitly_direction_unresolved_in_v2_policy(self):
        rows = [
            record(
                "raw_fixture_000051",
                subject="A prolonged lack of rain",
                subject_type="definition",
                relation_raw="term for definition",
                answer="drought",
                answer_type="term",
                canonical_fact="The term for a prolonged lack of rain is drought.",
                source_question="What is the term for a prolonged lack of rain?",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, summary = self.run_fixture(directory, records=rows)
            provisional_row = read_jsonl(output_dir / "provisional_triples.jsonl")[0]
            self.assertEqual(
                provisional_row["probe_relation_candidate"],
                "definition_term_direction_unresolved",
            )
            self.assertEqual(
                provisional_row["probe_relation_candidate_direction"], "unresolved"
            )
            self.assertEqual(
                provisional_row["probe_relation_status"],
                "pending_direction_semantic_and_balance_review",
            )
            self.assertEqual(
                summary["policy"]["schema_version"],
                "probe-relation-candidate-policy-v2",
            )
            self.assertEqual(
                summary["policy"]["policy_id"],
                "public-benchmark-qwen-probe-relation-candidates-v2",
            )

            cluster = read_jsonl(output_dir / "base_fact_clusters.jsonl")[0]
            queue_row = read_jsonl(output_dir / "review_queue.jsonl")[0]
            bundle_row = read_jsonl(output_dir / "behavior_input_bundle.jsonl")[0]
            for item in (cluster, queue_row, bundle_row):
                self.assertEqual(
                    item["probe_relation_candidate_direction"], "unresolved"
                )
                self.assertEqual(item["canonical_status"], "pending_review")
                self.assertIsNone(item["probe_relation_id"])
            self.assertIn(
                "probe_relation_candidate_direction_unresolved",
                queue_row["review_reasons"],
            )
            self.assertIn(
                "probe_relation_candidate_direction_unresolved",
                bundle_row["admission"]["blocking_reasons"],
            )

    def test_cluster_marks_conflicting_candidate_directions_for_review(self):
        item = record("raw_fixture_000052")
        policy = provisional.read_json(provisional.DEFAULT_POLICY)
        compiled = provisional.validate_and_compile_policy(policy)
        first = provisional.make_provisional_record(
            item,
            Path("/tmp/input.jsonl"),
            "0" * 64,
            policy["policy_id"],
            compiled,
        )
        second = copy.deepcopy(first)
        second["candidate_id"] = "raw_fixture_000053"
        second["provisional_id"] = "provisional_raw_fixture_000053"
        second["probe_relation_candidate_direction"] = "date -> work"
        clusters, _ = provisional.cluster_records([first, second])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(
            clusters[0]["probe_relation_candidate"], "work_to_release_date"
        )
        self.assertIsNone(clusters[0]["probe_relation_candidate_direction"])
        self.assertEqual(
            clusters[0]["probe_relation_status"], "candidate_direction_conflict"
        )
        self.assertEqual(clusters[0]["canonical_status"], "pending_review")

    def test_mcq_fallback_prompt_risk_flags_are_review_hints_only(self):
        cases = [
            (
                "raw_fixture_000061",
                "Choice reference",
                "Which of the following describes the example?",
                provisional.MCQ_FALLBACK_EXPLICIT_CHOICE_REFERENCE,
            ),
            (
                "raw_fixture_000062",
                "Blank",
                "The example is ____ in this sentence.",
                provisional.MCQ_FALLBACK_BLANK,
            ),
            (
                "raw_fixture_000063",
                "Negative",
                "Which answer is not correct?",
                provisional.MCQ_FALLBACK_NEGATIVE_OR_EXCEPTION,
            ),
            (
                "raw_fixture_000064",
                "Incomplete",
                "The relevant concept refers to:",
                provisional.MCQ_FALLBACK_INCOMPLETE,
            ),
        ]
        rows = [
            record(
                candidate_id,
                subject=subject,
                relation_raw="related term",
                answer=f"Target {index}",
                answer_type="term",
                canonical_fact=f"Target {index} appears before the end of this fact.",
                source_question=question,
                source_choices=[f"Target {index}", f"Distractor {index}"],
            )
            for index, (candidate_id, subject, question, _) in enumerate(cases, start=1)
        ]
        rows.append(
            record(
                "raw_fixture_000065",
                subject="Safe fallback",
                relation_raw="related term",
                answer="Safe target",
                answer_type="term",
                canonical_fact="Safe target appears before the end of this fact.",
                source_question="What concept describes the example?",
                source_choices=["Safe target", "Safe distractor"],
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_fixture(directory, records=rows)
            converted = {
                item["candidate_id"]: item
                for item in read_jsonl(output_dir / "provisional_triples.jsonl")
            }
            for candidate_id, _, _, expected_flag in cases:
                flags = converted[candidate_id]["prompt_risk_flags"]
                self.assertIn(provisional.MCQ_FALLBACK_STRUCTURE_RISK, flags)
                self.assertIn(expected_flag, flags)
                self.assertEqual(converted[candidate_id]["canonical_status"], "pending_review")
                self.assertTrue(converted[candidate_id]["prompt_ready"])
            self.assertEqual(converted["raw_fixture_000065"]["prompt_risk_flags"], [])

            queue = read_jsonl(output_dir / "review_queue.jsonl")
            flagged = [item for item in queue if item["prompt_risk_flags"]]
            self.assertEqual(len(flagged), 4)
            self.assertTrue(
                all(
                    item["prompt_risk_assessment"]
                    == "programmatic_review_hint_only"
                    for item in flagged
                )
            )
            self.assertTrue(all(item["review_decision"] is None for item in flagged))

    def test_review_sample_types_are_deterministic_and_state_statistical_scope(self):
        rows = [
            record(
                f"raw_fixture_{index:06d}",
                subject=f"Subject {index}",
                relation_raw="related term",
                answer=f"Target {index}",
                answer_type="term",
                canonical_fact=f"Target {index} appears before the end of this fact.",
                source_question=(
                    "Which of the following describes this item?"
                    if index % 2
                    else "What concept describes this item?"
                ),
                source_choices=[f"Target {index}", f"Distractor {index}"],
            )
            for index in range(71, 79)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "triple_extractions.jsonl"
            write_jsonl(input_path, rows)
            first = root / "first"
            second = root / "second"
            kwargs = {
                "review_sample_size": 5,
                "review_sample_seed": "coverage-test-seed",
                "random_review_sample_size": 4,
                "random_review_sample_seed": "random-test-seed",
                "risk_review_sample_size": 3,
                "risk_review_sample_seed": "risk-test-seed",
            }
            first_summary = provisional.prepare(
                input_path, first, provisional.DEFAULT_POLICY, **kwargs
            )
            provisional.prepare(input_path, second, provisional.DEFAULT_POLICY, **kwargs)

            sample_files = (
                "review_sample.jsonl",
                "review_sample_overall_random.jsonl",
                "review_sample_prompt_risk.jsonl",
            )
            for name in sample_files:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                sample_rows = read_jsonl(first / name)
                self.assertTrue(
                    all(item["canonical_status"] == "pending_review" for item in sample_rows)
                )
                self.assertTrue(all(item["human_gold"] is False for item in sample_rows))
                self.assertTrue(all(item["review_decision"] is None for item in sample_rows))
                self.assertTrue(
                    all("probe_relation_candidate_direction" in item for item in sample_rows)
                )
                self.assertTrue(
                    all(
                        item["sampling"]["sample_status"] == "selected_pending_review"
                        for item in sample_rows
                    )
                )

            random_rows = read_jsonl(first / "review_sample_overall_random.jsonl")
            risk_rows = read_jsonl(first / "review_sample_prompt_risk.jsonl")
            self.assertEqual(len(random_rows), 4)
            self.assertEqual(len(risk_rows), 3)
            self.assertTrue(
                all(item["prompt_risk_flags"] for item in risk_rows)
            )
            sampling = first_summary["review_sampling"]
            self.assertFalse(
                sampling["coverage_oriented_sample"]["population_estimation_supported"]
            )
            self.assertTrue(
                sampling["overall_random_sample"]["population_estimation_supported"]
            )
            self.assertFalse(
                sampling["prompt_risk_targeted_sample"][
                    "population_estimation_supported"
                ]
            )
            self.assertEqual(
                sampling["population_estimation_guidance"]["supported_sample"],
                "overall_random_sample",
            )

    def test_same_normalized_answer_group_never_crosses_provisional_split(self):
        rows = [
            record(
                "raw_fixture_000041",
                subject="France",
                subject_type="country",
                relation_raw="capital",
                answer="Paris",
                answer_type="city",
                canonical_fact="Paris is the capital of France.",
                source_question="What is the capital of France?",
            ),
            record(
                "raw_fixture_000042",
                subject="A fictional destination",
                subject_type="place",
                relation_raw="named after",
                answer="  PARIS  ",
                answer_type="city",
                canonical_fact="A fictional destination was named after PARIS.",
                source_question="What was the fictional destination named after?",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, summary = self.run_fixture(directory, records=rows)
            bundle = read_jsonl(output_dir / "behavior_input_bundle.jsonl")
            self.assertEqual(len(bundle), 2)
            self.assertNotEqual(bundle[0]["base_fact_id"], bundle[1]["base_fact_id"])
            self.assertEqual(bundle[0]["split_group_id"], bundle[1]["split_group_id"])
            self.assertEqual(bundle[0]["split_assignment"], bundle[1]["split_assignment"])
            self.assertTrue(all(item["split_status"] == "provisional_not_frozen" for item in bundle))
            self.assertTrue(all(item["sealed_evaluation_status"] == "not_run" for item in bundle))

            manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_group_count"], 1)
            self.assertEqual(manifest["base_fact_count"], 2)
            self.assertEqual(manifest["sealed_evaluation_status"], "not_run")
            self.assertEqual(summary["provisional_split"]["split_status"], "provisional_not_frozen")
            self.assertEqual(summary["provisional_split"]["sealed_evaluation_status"], "not_run")

    def test_optional_comparison_canonical_writes_exact_overlap_audit(self):
        fixtures = self.fixture_records()
        rows = [fixtures[0], fixtures[2]]
        comparison_first = copy.deepcopy(rows[0])
        comparison_second = copy.deepcopy(rows[1])
        comparison_second["source_id"] = "prior_other_source"
        comparison_second["candidate_id"] = "prior_other_candidate"
        comparison_second["subject"] = "Different subject"
        comparison_second["relation_raw"] = "different relation"
        comparison_second["canonical_fact"] = "A deliberately different canonical fact."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "triple_extractions.jsonl"
            comparison_path = root / "canonical_triples.jsonl"
            output_dir = root / "output"
            write_jsonl(input_path, rows)
            write_jsonl(comparison_path, [comparison_first, comparison_second])
            summary = provisional.prepare(
                input_path,
                output_dir,
                provisional.DEFAULT_POLICY,
                comparison_canonical_path=comparison_path,
            )

            audit = json.loads((output_dir / "overlap_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(
                audit["comparison"]["sha256"], provisional.sha256_file(comparison_path)
            )
            self.assertEqual(
                audit["exact_overlap_counts"],
                {
                    "normalized_canonical_fact": 1,
                    "normalized_question": 2,
                    "normalized_question_answer": 2,
                    "normalized_raw_triple": 1,
                    "normalized_subject_answer": 1,
                    "source_id": 1,
                },
            )
            self.assertEqual(
                audit["limitations"]["semantic_near_duplicate_clustering"], "not_performed"
            )
            source_examples = audit["metrics"]["source_id"]["matched_examples"]
            self.assertEqual(len(source_examples), 1)
            self.assertEqual(source_examples[0]["current_source_id"], rows[0]["source_id"])
            self.assertEqual(
                source_examples[0]["comparison_source_id"], comparison_first["source_id"]
            )
            self.assertTrue(source_examples[0]["current_base_fact_id"].startswith("pbf_"))
            self.assertEqual(
                summary["comparison_overlap_audit"]["artifact"], "overlap_audit.json"
            )
            self.assertIn("overlap_audit.json", summary["artifacts"])

    def test_same_input_and_policy_produce_byte_identical_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "triple_extractions.jsonl"
            write_jsonl(input_path, self.fixture_records())
            first = root / "first"
            second = root / "second"
            provisional.prepare(input_path, first, provisional.DEFAULT_POLICY)
            provisional.prepare(input_path, second, provisional.DEFAULT_POLICY)
            expected_files = {
                "provisional_triples.jsonl",
                "base_fact_clusters.jsonl",
                "relation_inventory.json",
                "review_queue.jsonl",
                "review_sample.jsonl",
                "review_sample_overall_random.jsonl",
                "review_sample_prompt_risk.jsonl",
                "behavior_input_bundle.jsonl",
                "split_manifest.json",
                "summary.json",
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected_files)
            self.assertEqual({path.name for path in second.iterdir()}, expected_files)
            for name in expected_files:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertFalse(list(first.glob("*.tmp")))
            self.assertFalse(list(second.glob("*.tmp")))

    def test_mixed_source_model_is_rejected(self):
        rows = self.fixture_records()[:2]
        rows[1]["extraction"]["model"] = "another-model"
        rows[1]["extraction"]["api_model"] = "another-model"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "triple_extractions.jsonl"
            write_jsonl(input_path, rows)
            with self.assertRaisesRegex(ValueError, "Expected exactly source model"):
                provisional.prepare(input_path, root / "output", provisional.DEFAULT_POLICY)

    def test_record_conversion_does_not_mutate_input(self):
        item = record("raw_fixture_000021")
        original = copy.deepcopy(item)
        policy = provisional.read_json(provisional.DEFAULT_POLICY)
        compiled = provisional.validate_and_compile_policy(policy)
        converted = provisional.make_provisional_record(
            item,
            Path("/tmp/input.jsonl"),
            "0" * 64,
            policy["policy_id"],
            compiled,
        )
        self.assertEqual(item, original)
        self.assertEqual(converted["canonical_status"], "pending_review")
        self.assertIsNone(converted["probe_relation_id"])

    def test_full_source_and_retry_provenance_is_preserved_in_provisional_record(self):
        item = record("raw_fixture_000031")
        item["extraction"]["raw_response"] = '{"extraction_status":"extracted"}'
        item["parsed_response"] = {"schema_version": "atomic-factual-triple-v6"}
        item["retry_history"] = [{"round": 1, "terminal_status": "validation_failed"}]
        item["local_adjudication"] = {
            "adjudication_version": "local-validation-adjudication-v1",
            "reviewer": "human",
        }
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self.run_fixture(directory, records=[item])
            converted = read_jsonl(output_dir / "provisional_triples.jsonl")[0]
            self.assertEqual(converted["source_snapshot"], item["source_snapshot"])
            self.assertEqual(converted["extraction"], item["extraction"])
            self.assertEqual(converted["parsed_response"], item["parsed_response"])
            self.assertEqual(converted["retry_history"], item["retry_history"])
            self.assertEqual(converted["local_adjudication"], item["local_adjudication"])


if __name__ == "__main__":
    unittest.main()
