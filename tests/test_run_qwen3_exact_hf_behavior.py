import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_qwen3_exact_hf_behavior.py"
)
SPEC = importlib.util.spec_from_file_location("run_qwen3_exact_hf_behavior", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def row(language, variant, choice, correct_choice="A", target_choice="B"):
    return {
        "base_fact_id": "f1",
        "source_id": "s1",
        "distractor_id": None if variant == "original" else "d1",
        "probe_relation_id": "r1",
        "answer_type": "entity",
        "language": language,
        "variant": variant,
        "choice": choice,
        "correct": choice == correct_choice,
        "distractor_hit": variant == "targeted" and choice == target_choice,
        "gold_minus_best_distractor_logprob_margin": 1.0 if choice == correct_choice else -1.0,
    }


class ExactHfBehaviorTests(unittest.TestCase):
    def test_directed_and_zh_specific_labels_are_nested(self):
        rows = [
            row("en", "original", "A"),
            row("en", "neutral", "A"),
            row("en", "targeted", "A"),
            row("zh", "original", "A"),
            row("zh", "neutral", "A"),
            row("zh", "targeted", "B"),
        ]
        label = module.build_labels(rows)[0]
        self.assertTrue(label["hf_directed_candidate"])
        self.assertTrue(label["hf_zh_specific_strict"])
        self.assertFalse(label["hf_resistant_control"])
        self.assertFalse(label["neutral_unstable"])
        self.assertFalse(label["off_target_failure"])

    def test_resistant_control_is_not_directed(self):
        rows = [row(language, variant, "A") for language in ("en", "zh") for variant in ("original", "neutral", "targeted")]
        label = module.build_labels(rows)[0]
        self.assertFalse(label["hf_directed_candidate"])
        self.assertFalse(label["hf_zh_specific_strict"])
        self.assertTrue(label["hf_resistant_control"])


if __name__ == "__main__":
    unittest.main()
