import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_path_not_token_chinese.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("prepare_path_not_token_chinese", SCRIPT_PATH)
preparation = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(preparation)


class PathNotTokenChinesePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chinese_path = PROJECT_ROOT / "data/Chinese.json"
        cls.canonical_path = (
            PROJECT_ROOT
            / "data_processed/factual_triples/triple-full-v1/canonical-v2/factual_prompts.jsonl"
        )
        cls.review_path = PROJECT_ROOT / "configs/path_not_token_chinese_v1_review.json"
        cls.chinese_rows = preparation.read_json(cls.chinese_path)
        cls.bases, cls.variants, cls.exclusions = preparation.build_records(
            cls.chinese_rows,
            preparation.read_jsonl(cls.canonical_path),
            preparation.read_json(cls.review_path),
            {
                "qwen3.7-plus": preparation.read_jsonl(
                    PROJECT_ROOT
                    / "data_processed/factual_triples/triple-full-v1/models/qwen3.7-plus/triple_extractions.jsonl"
                ),
                "sensenova-6.7-flash-lite": preparation.read_jsonl(
                    PROJECT_ROOT
                    / "data_processed/factual_triples/triple-full-v1/models/sensenova-6.7-flash-lite/triple_extractions.jsonl"
                ),
            },
            preparation.read_jsonl(
                PROJECT_ROOT
                / "data_processed/factual_triples/triple-full-v1/canonical-v2/canonical_decisions.jsonl"
            ),
        )

    def test_current_frozen_funnel_counts(self):
        self.assertEqual(len(self.chinese_rows), 342)
        self.assertEqual(len(self.bases), 13)
        self.assertEqual(len(self.variants), 91)
        self.assertEqual(len(self.exclusions), 39)
        self.assertEqual(sum(row["perturbation_count"] for row in self.bases), 91)
        self.assertEqual(sum(row["perturbation_count"] for row in self.exclusions), 251)

    def test_every_variant_links_to_one_base_and_has_open_prompts(self):
        base_ids = {row["id"] for row in self.bases}
        self.assertEqual(len(base_ids), len(self.bases))
        self.assertEqual(len({row["id"] for row in self.variants}), len(self.variants))
        for row in self.variants:
            self.assertIn(row["base_id"], base_ids)
            self.assertTrue(row["perturbed_open_prompt_en"].endswith("\nAnswer:"))
            self.assertTrue(row["perturbed_open_prompt_target"].endswith("\n答案："))
            self.assertFalse(row["choices_in_prompt"])

    def test_preparation_does_not_claim_mechanism_readiness(self):
        for row in self.bases:
            self.assertIsNone(row["probe_relation_id"])
            self.assertFalse(row["admission"]["relation_conditioned_mechanism_eligible"])
            self.assertFalse(row["admission"]["path_not_token_experiment_ready"])
            self.assertEqual(row["answer_tokenization"]["status"], "pending_target_model_and_tokenizer")
            self.assertEqual(row["mechanism_analysis"]["status"], "not_run")

    def test_relation_inventory_preserves_direction_without_freezing_probe_ids(self):
        inventory = preparation.build_relation_inventory(self.bases)
        self.assertEqual(inventory["relation_signature_count"], 13)
        self.assertEqual(inventory["frozen_probe_relation_count"], 0)
        self.assertFalse(inventory["relation_conditioned_experiment_ready"])
        self.assertTrue(all(row["probe_relation_id"] is None for row in inventory["relations"]))
        self.assertEqual(
            sum(row["base_fact_count"] for row in inventory["relations"]),
            len(self.bases),
        )

    def test_outputs_are_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base_facts.jsonl"
            preparation.write_jsonl(path, self.bases)
            parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(parsed, self.bases)


if __name__ == "__main__":
    unittest.main()
