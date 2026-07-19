import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from eva import (
    build_detailed_results,
    build_results,
    extract_model_progress,
    get_model_checkpoint_path,
    merge_model_progresses,
)
from utils.answers import simulate
from utils.tools import model_dict


class ModelRegistryTests(unittest.TestCase):
    def test_new_openai_compatible_models_are_registered(self):
        expected_models = {
            "qwen3.5-plus",
            "qwen3-max",
            "deepseek-v3",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
        }
        self.assertTrue(expected_models.issubset(model_dict))
        for model in expected_models:
            self.assertEqual(model_dict[model], model)


class PerModelCheckpointTests(unittest.TestCase):
    def test_checkpoint_path_uses_language_and_model(self):
        path = get_model_checkpoint_path(
            "data/Chinese",
            "Chinese",
            "deepseek-v4-pro",
        )
        self.assertEqual(
            path,
            "data/Chinese/Chinese_results_deepseek-v4-pro_progress.json",
        )

    def test_extract_and_merge_keep_only_selected_models(self):
        combined = {
            "input_file": "data/Chinese.json",
            "lang": "Chinese",
            "records": {
                "0": {
                    "English": {"model-a": 1.0, "model-b": 0.0},
                    "Chinese": {"model-a": 0.0, "model-b": 1.0},
                }
            },
        }
        model_a = extract_model_progress(
            combined,
            "data/Chinese.json",
            "Chinese",
            "model-a",
        )
        model_b = extract_model_progress(
            combined,
            "data/Chinese.json",
            "Chinese",
            "model-b",
        )

        self.assertEqual(model_a["model"], "model-a")
        self.assertEqual(set(model_a["records"]["0"]["English"]), {"model-a"})
        self.assertEqual(set(model_b["records"]["0"]["English"]), {"model-b"})

        merged = merge_model_progresses(
            {"model-a": model_a},
            "data/Chinese.json",
            "Chinese",
        )
        self.assertEqual(set(merged["records"]["0"]["English"]), {"model-a"})


class SimulateDetailsTests(unittest.TestCase):
    def test_returns_raw_and_extracted_answers(self):
        responses = iter([
            "I choose option B.",
            '{"final_answer": "B"}',
        ])
        with patch("utils.answers.get_response", side_effect=lambda **_kwargs: next(responses)):
            with redirect_stdout(io.StringIO()):
                result = simulate(
                    language="English",
                    question="Choose B.",
                    model_list=["test-model"],
                    choices=["A", "B"],
                    ground_truth="B",
                    return_details=True,
                )

        model_result = result["models"]["test-model"]
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(model_result["status"], "success")
        self.assertEqual(model_result["raw_response"], "I choose option B.")
        self.assertEqual(model_result["extracted_answer"], "B")
        self.assertTrue(model_result["correct"])

    def test_marks_invalid_extraction_without_counting_it(self):
        responses = iter([
            "I choose option B.",
            "default failure",
        ])
        with patch("utils.answers.get_response", side_effect=lambda **_kwargs: next(responses)):
            with redirect_stdout(io.StringIO()):
                result = simulate(
                    language="English",
                    question="Choose B.",
                    model_list=["test-model"],
                    choices=["A", "B"],
                    ground_truth="B",
                    return_details=True,
                )

        model_result = result["models"]["test-model"]
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(model_result["status"], "invalid_extraction_json")
        self.assertEqual(model_result["extraction_response"], "default failure")


class DetailedOutputTests(unittest.TestCase):
    def setUp(self):
        self.entry = {
            "question": "English question",
            "choices": ["A", "B"],
            "answer": "B",
            "transquestion": "中文问题",
            "transchoices": ["甲", "乙"],
            "transanswer": "乙",
            "source": "ai2_arc",
            "category": "Science & Technology",
            "prequestion": "construction-only prefix",
            "sufquestion": "construction-only suffix",
            "rate_ori": 1,
            "rate_trans": 0.2,
            "value": 3.2,
        }
        self.progress = {
            "input_file": "data/Chinese.json",
            "lang": "Chinese",
            "records": {
                "7": {
                    "English": {
                        "new-model": {
                            "status": "success",
                            "raw_response": "B",
                            "extracted_answer": "B",
                            "correct": True,
                            "score": 1.0,
                            "extractor_model": "qwen3.7-plus",
                        },
                        "legacy-model": 1.0,
                    },
                    "Chinese": {},
                }
            },
        }

    def test_summary_accepts_new_and_legacy_checkpoint_records(self):
        summary = build_results(
            self.progress,
            [(7, self.entry)],
            ["new-model", "legacy-model"],
            "Chinese",
        )
        self.assertEqual(summary["English"]["new-model"], 1.0)
        self.assertEqual(summary["English"]["legacy-model"], 1.0)
        self.assertEqual(summary["Completed"]["English"]["new-model"], 1)

    def test_details_keep_only_evaluation_fields(self):
        details = build_detailed_results(
            self.progress,
            [(7, self.entry)],
            ["new-model", "legacy-model"],
            "Chinese",
        )
        item = details["items"][0]
        self.assertEqual(item["original_index"], 7)
        self.assertEqual(item["source"], "ai2_arc")
        self.assertEqual(item["English"]["ground_truth"], "B")
        self.assertEqual(item["Chinese"]["ground_truth"], "乙")
        self.assertNotIn("prequestion", item)
        self.assertNotIn("sufquestion", item)
        self.assertNotIn("rate_trans", item)
        self.assertNotIn("value", item)
        self.assertEqual(
            item["English"]["models"]["legacy-model"]["status"],
            "legacy_score_only",
        )


if __name__ == "__main__":
    unittest.main()
