import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "materialize_full_public_benchmark_pre_hf.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "materialize_full_public_benchmark_pre_hf", SCRIPT_PATH
)
pre_hf = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(pre_hf)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FullPublicBenchmarkPreHfTests(unittest.TestCase):
    def make_source(self, root, count=18):
        source_dir = Path(root) / "source"
        source_dir.mkdir(parents=True)
        provisional = []
        clusters = []
        behavior = []
        review = []
        for index in range(count):
            base_fact_id = f"pbf_fixture_{index:03d}"
            candidate_id = f"raw_fixture_{index:03d}"
            if index < count // 2:
                relation_raw = "release date"
                probe = "work_to_release_date"
                answer_type = "date"
                answer = f"{2000 + index}"
                subject = f"Film {index}"
            else:
                relation_raw = "capital"
                probe = None
                answer_type = "city"
                answer = f"City {index}"
                subject = f"Country {index}"
            question = f"What is the answer for {subject}?"
            split_group_id = "shared_answer_group" if index in (0, 1) else f"answer_group_{index}"
            common = {
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
            }
            provisional.append(
                {
                    "schema_version": "provisional-factual-triple-v1",
                    **common,
                    "candidate_id": candidate_id,
                    "base_fact_id": base_fact_id,
                }
            )
            clusters.append(
                {
                    "schema_version": "provisional-base-fact-cluster-v1",
                    **common,
                    "base_fact_id": base_fact_id,
                    "representative_candidate_id": candidate_id,
                    "members": [{"candidate_id": candidate_id}],
                    "split_group_id": split_group_id,
                }
            )
            behavior.append(
                {
                    "schema_version": "factual-perturbation-input-bundle-v1",
                    **common,
                    "base_fact_id": base_fact_id,
                    "candidate_id": candidate_id,
                    "source_pool_id": "fixture",
                    "source_dataset": "fixture_a" if index % 2 == 0 else "fixture_b",
                    "source_subset": "unit",
                    "source_id": candidate_id,
                    "source_format": "multiple_choice",
                    "source_question_en": question,
                    "source_choices_en": [answer, f"Wrong {index}"],
                    "subject_en": subject,
                    "answer_en": answer,
                    "answer_aliases_en": [answer],
                    "answer_type": answer_type,
                    "canonical_fact_en": f"{subject} has answer {answer}.",
                    "prompt_en": f"{question}\nAnswer:",
                    "prompt_quality_tier": "strict_factual_completion",
                    "prompt_risk_flags": [],
                    "relation_raw": relation_raw,
                    "relation_signature_id": f"rel_{index % 3}",
                    "relation_normalized": None,
                    "probe_relation_candidate": probe,
                    "probe_relation_id": None,
                    "split_group_id": split_group_id,
                }
            )
            review.append(
                {
                    "schema_version": "provisional-semantic-review-v1",
                    **common,
                    "base_fact_id": base_fact_id,
                }
            )
        artifacts = {
            "provisional_triples.jsonl": provisional,
            "base_fact_clusters.jsonl": clusters,
            "behavior_input_bundle.jsonl": behavior,
            "review_queue.jsonl": review,
        }
        for filename, rows in artifacts.items():
            write_jsonl(source_dir / filename, rows)
        summary = {
            "schema_version": "single-model-provisional-summary-v2",
            "counts": {"base_fact_clusters": count},
            "artifacts": {
                filename: {"sha256": pre_hf.sha256_file(source_dir / filename)}
                for filename in artifacts
            },
        }
        write_json(source_dir / "summary.json", summary)
        return source_dir, behavior

    def make_languages(self, root, behavior):
        target_path = Path(root) / "Chinese.json"
        write_json(
            target_path,
            [
                {
                    "oriquestion": behavior[0]["source_question_en"],
                    "answer": behavior[0]["answer_en"],
                    "transori": "目标问题？",
                    "transanswer": "目标答案",
                    "transchoices": ["目标答案", "错误答案"],
                },
                {
                    "oriquestion": "An unrelated question?",
                    "answer": "Other",
                    "transori": "无关问题？",
                    "transanswer": "其他",
                    "transchoices": ["其他"],
                },
            ],
        )
        registry_path = Path(root) / "registry.json"
        write_json(
            registry_path,
            {
                "schema_version": "public-benchmark-multilingual-registry-v1",
                "registry_id": "fixture-registry",
                "source_language": "en",
                "alignment_policy": {
                    "method": "normalized_original_question_plus_answer_alias_exact_match_v1",
                    "allow_fuzzy_or_semantic_match": False,
                },
                "languages": [
                    {
                        "language_code": "en",
                        "language_name": "English",
                        "role": "source_language",
                        "dataset_path": None,
                    },
                    {
                        "language_code": "zh",
                        "language_name": "Chinese",
                        "role": "external_translation_reference",
                        "dataset_path": str(target_path),
                    },
                ],
            },
        )
        return registry_path

    def make_frozen(self, root, conflicting=False):
        frozen_path = Path(root) / "frozen.jsonl"
        rows = [
            {
                "schema_version": "fixture-frozen-v1",
                "base_fact_id": "pbf_fixture_000",
                "split_assignment": "development",
                "leakage_component_id": "frozen_component_a",
            },
            {
                "schema_version": "fixture-frozen-v1",
                "base_fact_id": "pbf_fixture_001",
                "split_assignment": "sealed" if conflicting else "development",
                "leakage_component_id": "frozen_component_b",
            },
            {
                "schema_version": "fixture-frozen-v1",
                "base_fact_id": "pbf_fixture_002",
                "split_assignment": "validation",
                "leakage_component_id": "frozen_component_c",
            },
        ]
        write_jsonl(frozen_path, rows)
        return frozen_path

    def test_materializes_complete_offline_contract_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir, behavior = self.make_source(directory)
            registry = self.make_languages(directory, behavior)
            frozen = self.make_frozen(directory)
            output_dir = Path(directory) / "output"
            summary = pre_hf.materialize(
                source_dir=source_dir,
                output_dir=output_dir,
                language_registry_path=registry,
                frozen_split_bundle_path=frozen,
                split_seed="fixture-seed",
            )

            self.assertEqual(summary["counts"]["base_facts"], 18)
            self.assertEqual(summary["execution"]["behavior_output_count"], 0)
            self.assertFalse(summary["formal_hf_behavior_authorized"])
            full_rows = read_jsonl(output_dir / "full_base_facts.jsonl")
            self.assertEqual(len(full_rows), 18)
            by_id = {row["base_fact_id"]: row for row in full_rows}
            self.assertEqual(by_id["pbf_fixture_000"]["split_assignment"], "development")
            self.assertEqual(by_id["pbf_fixture_001"]["split_assignment"], "development")
            self.assertEqual(by_id["pbf_fixture_002"]["split_assignment"], "validation")
            self.assertEqual(
                by_id["pbf_fixture_000"]["leakage_component_id"],
                by_id["pbf_fixture_001"]["leakage_component_id"],
            )
            self.assertIsNone(by_id["pbf_fixture_010"]["relation_normalized"])
            self.assertEqual(
                by_id["pbf_fixture_010"]["relation_partition_id"],
                "location_or_habitat",
            )

            availability = read_jsonl(output_dir / "multilingual_availability.jsonl")
            first = {row["base_fact_id"]: row for row in availability}["pbf_fixture_000"]
            self.assertEqual(
                first["languages"]["zh"]["availability_status"],
                "aligned_external_translation_candidate_unreviewed",
            )
            language_rows = read_jsonl(output_dir / "multilingual_preperturbation_rows.jsonl")
            self.assertEqual(len(language_rows), 19)
            gate = json.loads((output_dir / "pre_hf_gate_manifest.json").read_text())
            self.assertTrue(gate["pre_hf_boundary_reached"])
            self.assertFalse(gate["execution_evidence"]["hf_model_execution"])
            self.assertFalse(gate["execution_evidence"]["hf_tokenizer_execution"])
            self.assertFalse(gate["authorization"]["hf_behavior_authorized"])

            second_output = Path(directory) / "output-two"
            pre_hf.materialize(
                source_dir=source_dir,
                output_dir=second_output,
                language_registry_path=registry,
                frozen_split_bundle_path=frozen,
                split_seed="fixture-seed",
            )
            for filename in (
                "full_base_facts.jsonl",
                "relation_assignments.jsonl",
                "leakage_components.jsonl",
                "split_manifest.json",
                "pre_hf_gate_manifest.json",
            ):
                self.assertEqual(
                    pre_hf.sha256_file(output_dir / filename),
                    pre_hf.sha256_file(second_output / filename),
                )

    def test_rejects_conflicting_frozen_splits_in_one_component(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir, behavior = self.make_source(directory)
            registry = self.make_languages(directory, behavior)
            frozen = self.make_frozen(directory, conflicting=True)
            with self.assertRaisesRegex(ValueError, "Frozen split conflict"):
                pre_hf.materialize(
                    source_dir=source_dir,
                    output_dir=Path(directory) / "output",
                    language_registry_path=registry,
                    frozen_split_bundle_path=frozen,
                    split_seed="fixture-seed",
                )


if __name__ == "__main__":
    unittest.main()
