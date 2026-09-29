import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_qwen3_exact_hf_g0b.py"
)
SPEC = importlib.util.spec_from_file_location("freeze_qwen3_exact_hf_g0b", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class ExactHfG0BTests(unittest.TestCase):
    def test_simulation_prompt_preserves_mcq_contract(self):
        prompt = module.simulation_prompt(
            "法国的首都是",
            "这是背景。",
            {"A": "巴黎", "B": "伦敦", "C": "罗马"},
            "zh",
        )
        self.assertIn("A. 巴黎", prompt)
        self.assertIn("choice 必须是 A/B/C 之一", prompt)
        self.assertTrue(prompt.endswith('例如：{"choice":"A"}。'))

    def test_choice_completion_is_canonical_and_deterministic(self):
        self.assertEqual(module.canonical_json({"choice": "B"}), '{"choice":"B"}')
        self.assertEqual(
            module.sha256_value(["revision", "stimulus", "exact-hf-render-v3"]),
            module.sha256_value(["revision", "stimulus", "exact-hf-render-v3"]),
        )

    def test_choice_decision_follows_shared_json_prefix(self):
        metadata = module.choice_decision_metadata(
            [10, 11],
            {
                "A": [20, 21, 30],
                "B": [20, 21, 31],
                "C": [20, 21, 32],
            },
        )
        self.assertEqual(metadata["choice_common_prefix_token_ids"], [20, 21])
        self.assertEqual(metadata["choice_decision_input_ids"], [10, 11, 20, 21])
        self.assertEqual(metadata["choice_decision_position"], 3)
        self.assertEqual(metadata["choice_token_ids"], {"A": 30, "B": 31, "C": 32})

    def test_choice_decision_rejects_ambiguous_first_specific_token(self):
        with self.assertRaisesRegex(ValueError, "not unique"):
            module.choice_decision_metadata(
                [10],
                {"A": [20, 30], "B": [20, 30], "C": [20, 31]},
            )


if __name__ == "__main__":
    unittest.main()
