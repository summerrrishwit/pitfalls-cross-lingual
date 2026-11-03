"""Safety-relevant checks; execute on the remote environment with the run."""
import importlib.util
from collections import UserDict
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "remote_open", Path(__file__).resolve().parents[1] / "scripts/run_qwen3_option_free_remote.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OpenScoringTests(unittest.TestCase):
    def setUp(self):
        self.groups = [
            {"option_id": "gold", "aliases_en": ["Paris"], "aliases_zh": ["巴黎"]},
            {"option_id": "foil", "aliases_en": ["London"], "aliases_zh": ["伦敦"]},
        ]

    def test_negated_gold_is_not_accepted_by_substring(self):
        result = module.generation_labels("Not Paris, it is London.", self.groups, "en")
        self.assertIsNone(result["generation_correct"])
        self.assertEqual(result["correctness_basis"], "pending_semantic_review")

    def test_exact_distractor_is_identified_separately(self):
        result = module.generation_labels("London.", self.groups, "en")
        self.assertIs(result["generation_correct"], False)
        self.assertEqual(result["generated_answer_group"], "foil")

    def test_cross_language_alias_establishes_answer_identity(self):
        self.assertTrue(module.generation_labels("Paris", self.groups, "zh")["generation_correct"])

    def test_deferred_review_remains_missing(self):
        result = module.generation_labels("Paris", self.groups, "en", {"semantic_correct": None})
        self.assertIsNone(result["generation_correct"])

    def test_answer_prefix_mismatch_rejected(self):
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": [9, 2, 3]}
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            module.suffix(Tokenizer(), "prompt", [1, 2], "answer")

    def test_transformers_mapping_tokenizer_return(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, **kwargs):
                return UserDict({"input_ids": [1, 2, 3]}) if tokenize else "rendered"
        self.assertEqual(module.render(Tokenizer(), "prompt"), ("rendered", [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
