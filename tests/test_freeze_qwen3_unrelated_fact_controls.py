import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "freeze_qwen3_unrelated_fact_controls.py"
SPEC = importlib.util.spec_from_file_location("freeze_qwen3_unrelated_fact_controls", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class Qwen3UnrelatedFactControlTests(unittest.TestCase):
    def test_natural_prompt_matches_frozen_baseline_instruction(self):
        self.assertEqual(
            module.natural_prompt("法国的首都是", "zh"),
            "请补全以下事实，只输出缺失的答案，不要解释：\n法国的首都是",
        )
        self.assertEqual(
            module.natural_prompt("The capital of France is", "en"),
            "Complete the following factual statement with only the missing answer:\nThe capital of France is",
        )

    def test_current_control_sources_materialize_without_cohort_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = module.materialize(
                REPO_ROOT / "data_processed" / "path_not_token" / "chinese-v1" / "base_facts.jsonl",
                REPO_ROOT / "data_processed" / "factual_perturbation" / "static-g0a-160-dual-v8-20260916" / "frozen-v1" / "frozen_static_bundle.jsonl",
                REPO_ROOT / "configs" / "qwen3_unrelated_fact_control_codex_review_v1.json",
                REPO_ROOT / "data_processed" / "factual_perturbation" / "static-g0a-160-dual-v8-20260916" / "exact-hf-qwen3-8b-g0b-v3" / "g0b_gate_manifest.json",
                Path(temporary),
            )
        self.assertEqual(manifest["counts"]["control_facts"], 13)
        self.assertEqual(manifest["counts"]["control_inputs"], 26)
        self.assertEqual(manifest["counts"]["validation_facts_used"], 0)
        self.assertFalse(manifest["authorization_state"]["model_execution_authorized"])
        self.assertFalse(manifest["authorization_state"]["vector_source_authorized"])


if __name__ == "__main__":
    unittest.main()
