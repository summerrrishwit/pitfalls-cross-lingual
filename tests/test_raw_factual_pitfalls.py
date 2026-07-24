import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factual_pitfalls import pipeline


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_raw_factual_pitfalls.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_raw_factual_pitfalls", SCRIPT_PATH)
build_script = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(build_script)


def source_record(question, source="mmlu", subject="geography"):
    return {
        "question": question,
        "choices": ["Paris", "Rome", "Berlin"],
        "answer": "Paris",
        "source": source,
        "subject": subject,
    }


def config(paths):
    return {
        "source_datasets": paths,
        "raw_sample_size": 4,
        "seed": 11,
        "target_languages": ["Chinese", "Japanese", "French"],
        "supported_models": ["qwen3.7-plus", "qwen3-max", "deepseek-v3"],
        "roles": {
            "factual_audit_model": "qwen3.7-plus",
            "perturbation_generator_model": "qwen3-max",
            "translation_model": "qwen3.7-plus",
            "screen_models": ["qwen3-max", "deepseek-v3"],
            "answer_extract_model": "qwen3.7-plus",
        },
        "weakness_thresholds": {"min_rate_ori": 0.8, "max_rate_trans": 0.5, "min_pitfall_score": 0.3},
        "runtime": {"temperature": 0.0, "timeout_seconds": 5, "max_retries": 0},
        "output_root": "derived",
    }


class RawSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first = self.root / "first.json"
        self.second = self.root / "second.json"
        self.first.write_text(json.dumps([source_record("Capital of France?"), source_record("Capital of Italy?", subject="history")]), encoding="utf-8")
        self.second.write_text(json.dumps([source_record("Capital of France?", source="sciq"), source_record("Capital of Germany?", source="sciq")]), encoding="utf-8")
        self.config = config({"mmlu": str(self.first), "sciq": str(self.second)})

    def tearDown(self):
        self.temp.cleanup()

    def test_raw_schema_requires_answer_in_choices(self):
        record = source_record("Broken")
        record["answer"] = "Not a choice"
        self.assertEqual(pipeline.raw_schema_error(record), "answer_not_in_choices")

    def test_cross_source_dedup_and_manifest_contains_no_downstream_fields(self):
        manifest = pipeline.build_raw_manifest(self.config, self.root, "test-run")
        selected_questions = [item["source_snapshot"]["record"]["question"] for item in manifest["candidates"]]
        self.assertEqual(selected_questions.count("Capital of France?"), 1)
        self.assertNotIn("translations", manifest)
        self.assertNotIn("rate_ori", json.dumps(manifest))
        self.assertEqual(manifest["source_datasets"]["allocation"]["actual_count"], 3)

    def test_same_seed_reproduces_raw_selection(self):
        first = pipeline.build_raw_manifest(self.config, self.root, "first")
        second = pipeline.build_raw_manifest(self.config, self.root, "second")
        first_ids = [candidate["candidate_id"] for candidate in first["candidates"]]
        second_ids = [candidate["candidate_id"] for candidate in second["candidates"]]
        self.assertEqual(first_ids, second_ids)


class StageContractTests(unittest.TestCase):
    def setUp(self):
        self.raw = source_record("What is the capital of France?")
        self.candidate = {
            "candidate_id": "raw_mmlu_000000",
            "stratum": "mmlu::geography",
            "source_snapshot": {"dataset": "mmlu", "source_path": "data/source/mmlu.json", "original_index": 0, "record": self.raw},
        }
        self.config = config({"mmlu": "data/source/mmlu.json"})

    def test_audit_prompt_excludes_downstream_fields(self):
        prompt = pipeline.raw_audit_prompt(self.candidate)
        self.assertIn("What is the capital of France?", prompt)
        self.assertNotIn("Chinese", prompt)
        self.assertNotIn("rate_ori", prompt)
        self.assertNotIn("巴黎", prompt)

    def test_accepting_audit_contract_requires_factual_fields(self):
        response = {
            "decision": "accept", "is_factual_recall": True, "factual_type": "geography", "answer_unique": True,
            "has_material_ambiguity": False, "option_dependent": False, "requires_multistep_reasoning": False,
            "reason": "Direct fact.", "subject_en": "France", "relation_en": "capital", "answer_en": "Paris", "prompt_en": "The capital of France is",
        }
        self.assertEqual(pipeline.validate_audit_response(response, "Paris"), ("accept", []))
        del response["prompt_en"]
        decision, errors = pipeline.validate_audit_response(response, "Paris")
        self.assertEqual(decision, "needs_review")
        self.assertIn("invalid_prompt_en", errors)

    def test_generation_and_translation_validation(self):
        generated, errors = pipeline.validate_generation_response({"distraction": "A small detail is worth noting.", "insertion": "prefix", "answer_preserved": True})
        self.assertTrue(generated)
        self.assertEqual(errors, [])
        translated, errors = pipeline.validate_translation_response({"question": "法国的首都是哪里？", "choices": ["巴黎", "罗马", "柏林"], "answer": "巴黎"}, 3)
        self.assertTrue(translated)
        self.assertEqual(errors, [])

    def test_screening_uses_ensemble_rates_and_thresholds(self):
        generation = {"generation_id": "g1", "candidate_id": "raw_mmlu_000000", "enhanced_question": "Detail. What is the capital of France?", "choices": self.raw["choices"], "answer": "Paris"}
        translation = {"translation_id": "g1_chinese", "target_language": "Chinese", "generation": generation, "translated": {"question": "法国的首都是哪里？", "choices": ["巴黎", "罗马", "柏林"], "answer": "巴黎"}}
        outcomes = iter([
            {"score": 1.0}, {"score": 1.0},  # English qwen/deepseek
            {"score": 0.0}, {"score": 1.0},  # Chinese qwen/deepseek
        ])
        def fake_screen(*_args, **_kwargs):
            result = next(outcomes)
            return {"status": "success", "correct": result["score"] == 1.0, **result}
        with patch("factual_pitfalls.pipeline.screen_question", side_effect=fake_screen):
            record = pipeline.screen_translation(translation, self.config, client=None)
        self.assertEqual(record["rate_ori"], 1.0)
        self.assertEqual(record["rate_trans"], 0.5)
        self.assertTrue(record["retained"])

    def test_permanent_failure_is_not_retryable(self):
        self.assertFalse(pipeline.transient_error(ValueError("bad configuration")))


class ResumeStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "factual_reviews.jsonl"
        pipeline.write_jsonl(
            self.output,
            [{"review_id": "raw_1", "terminal_status": "audit_failed", "created_at": "first", "audit": {"error": "Connection error"}}],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_resume_skips_failed_but_retry_failed_replaces_it_with_history(self):
        inputs = [{"candidate_id": "raw_1"}]
        calls = []

        def transform(item):
            calls.append(item["candidate_id"])
            return [{"review_id": item["candidate_id"], "terminal_status": "completed", "created_at": "second", "audit": {}}]

        build_script.run_stage(inputs, self.output, "review_id", lambda item: [item["candidate_id"]], transform, True, False, None)
        self.assertEqual(calls, [])

        records = build_script.run_stage(inputs, self.output, "review_id", lambda item: [item["candidate_id"]], transform, True, True, None)
        self.assertEqual(calls, ["raw_1"])
        self.assertEqual(records[0]["terminal_status"], "completed")
        self.assertEqual(records[0]["retry_history"][0]["terminal_status"], "audit_failed")
