import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_qwen3_exact_hf_natural_baseline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_qwen3_exact_hf_natural_baseline", SCRIPT_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


class NaturalBaselineTests(unittest.TestCase):
    def test_alias_correct_is_case_and_punctuation_insensitive(self):
        self.assertTrue(module.alias_correct("Paris.", ["PARIS"]))
        self.assertTrue(module.alias_correct("史蒂夫·汪达。", ["史蒂夫·汪达"]))
        self.assertTrue(
            module.alias_correct(
                "The actor who portrayed Sirius Black is Gary Oldman.",
                ["Gary Oldman"],
            )
        )
        self.assertFalse(module.alias_correct("London", ["Paris"]))

    def test_normalize_answer_handles_full_width_text(self):
        self.assertEqual(module.normalize_answer("Ａ． Paris!"), "aparis")


if __name__ == "__main__":
    unittest.main()
