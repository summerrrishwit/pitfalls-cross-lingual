import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("bridge", Path(__file__).resolve().parents[1] / "scripts/analyze_qwen3_option_free.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MissingJudgmentTests(unittest.TestCase):
    def test_missing_target_not_negative(self):
        original = {"generation_correct": True}
        neutral = {"generation_correct": True}
        target = {"generation_correct": None,"generated_answer_group":None,"designated_distractor_id":"foil"}
        self.assertIsNone(module.defect(original, neutral, target, True))

    def test_failed_baseline_not_induced(self):
        target = {"generation_correct":False,"generated_answer_group":"foil","designated_distractor_id":"foil"}
        self.assertFalse(module.defect({"generation_correct":False},{"generation_correct":True},target,True))

    def test_non_target_error_is_not_directed(self):
        target = {"generation_correct":False,"generated_answer_group":"other","designated_distractor_id":"foil"}
        self.assertTrue(module.defect({"generation_correct":True},{"generation_correct":True},target,False))
        self.assertFalse(module.defect({"generation_correct":True},{"generation_correct":True},target,True))

    def test_unresolved_second_variant_is_not_a_fact_negative(self):
        self.assertIsNone(module.any_known([False,None]))
        self.assertTrue(module.any_known([True,None]))


class OptionalMCQTests(unittest.TestCase):
    def make_run(self, root, target_count=1, mcq=False):
        inputs, outputs = [], []
        for language in ("en", "zh"):
            for target, arm in [(None, "original"), *[(f"d{i}", arm) for i in range(target_count) for arm in ("neutral", "targeted")]]:
                source = {"option_free_id": f"f-{language}-{arm}-{target}", "base_fact_id": "f",
                          "split_assignment": "development",
                          "language": language, "variant": arm, "designated_distractor_id": target,
                          "user_prompt_sha256": "prompt", "development_fold_id": "fold",
                          "probe_relation_id": "relation", "leakage_component_id": "component",
                          "factual_completion_prompt": "question", "answer_groups": []}
                wrong = language == "zh" and arm == "targeted"
                inputs.append(source)
                outputs.append({**source, "generation": "wrong" if wrong else "gold",
                                "generation_correct": not wrong, "generated_answer_group": target if wrong else "gold",
                                "top_scored_answer_group": target if wrong else "gold",
                                "gold_minus_designated_distractor_margin": -1 if wrong else 1})
        (root / "inputs/g2a").mkdir(parents=True)
        (root / "results").mkdir()
        module.jsonl(root / "inputs/g2a/option_free_three_arm_inputs.jsonl", inputs)
        module.jsonl(root / "results/open_results.jsonl", outputs)
        if mcq:
            module.jsonl(root / "inputs/mcq_labels.jsonl", [
                {"base_fact_id": "f", "distractor_id": f"d{i}", "hf_directed_candidate": False, "hf_zh_specific_strict": False}
                for i in range(target_count)])
        return inputs, outputs

    def analyze(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(["--run-dir", str(root)]), 0)
        return json.loads((root / "results/bridge_summary.json").read_text())

    def test_single_candidate_without_mcq(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root)
            summary = self.analyze(root)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed_inputs"], 6)
            self.assertEqual(summary["fact_counts"]["open_induced_directed"], {"True": 1})
            self.assertEqual(summary["mcq_comparison_facts"], 0)
            self.assertEqual(summary["mcq_by_open_directed_fact_table"], [])
            self.assertIsNone(module.rows(root / "results/open_fact_labels.jsonl")[0]["mcq_directed"])

    def test_mcq_negative_is_not_an_admission_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root, mcq=True)
            summary = self.analyze(root)
            self.assertEqual(summary["mcq_by_open_directed_fact_table"], [
                {"mcq_directed": "False", "open_directed": "True", "facts": 1}])

    def test_two_variants_preserve_one_fact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root, target_count=2, mcq=True)
            summary = self.analyze(root)
            self.assertEqual(summary["expected_inputs"], 10)
            self.assertEqual(summary["variants"], 2)
            self.assertEqual(summary["facts_with_both_variants"], 1)
            self.assertEqual(summary["fact_counts"]["open_induced_directed"], {"True": 1})

    def test_missing_outputs_remain_in_fact_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root)
            module.jsonl(root / "results/open_results.jsonl", [])
            summary = self.analyze(root)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["assigned_facts"], 1)
            self.assertEqual(summary["fact_counts"]["open_induced_directed"], {"None": 1})

    def test_missing_second_candidate_is_unknown_not_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, outputs = self.make_run(root, target_count=2)
            keep = [r for r in outputs if r["designated_distractor_id"] != "d1"]
            for r in keep:
                r["generation_correct"] = True
                r["generated_answer_group"] = "gold"
            module.jsonl(root / "results/open_results.jsonl", keep)
            summary = self.analyze(root)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["fact_counts"]["open_induced_directed"], {"None": 1})

    def test_missing_planned_arm_fails_input_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, _ = self.make_run(Path(folder))
            with self.assertRaisesRegex(ValueError, "paired"):
                module.validate_inputs(inputs[:-1])

    def test_holdout_is_not_silently_reported_as_development(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs, _ = self.make_run(Path(folder))
            inputs[0]["split_assignment"] = "validation"
            with self.assertRaisesRegex(ValueError, "Development"):
                module.validate_inputs(inputs)


if __name__ == "__main__":
    unittest.main()
