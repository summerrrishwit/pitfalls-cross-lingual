import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

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
        "supported_models": ["sensenova-6.7-flash-lite", "qwen3.7-plus", "qwen3-max", "deepseek-v3"],
        "roles": {
            "factual_audit_model": "sensenova-6.7-flash-lite",
            "perturbation_generator_model": "qwen3-max",
            "translation_model": "qwen3.7-plus",
            "screen_models": ["qwen3-max", "deepseek-v3"],
            "answer_extract_model": "qwen3.7-plus",
        },
        "weakness_thresholds": {"min_rate_ori": 0.8, "max_rate_trans": 0.5, "min_pitfall_score": 0.3},
        "runtime": {"temperature": 0.0, "max_tokens": 4096, "timeout_seconds": 5, "max_retries": 0},
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

    def test_all_mode_selects_every_deduplicated_candidate(self):
        all_config = {**self.config, "raw_sample_size": "all"}
        manifest = pipeline.build_raw_manifest(all_config, self.root, "all")
        allocation = manifest["source_datasets"]["allocation"]
        self.assertEqual(allocation["selection_mode"], "full_population")
        self.assertEqual(allocation["requested_count"], "all")
        self.assertEqual(allocation["actual_count"], allocation["available_count"])


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
        self.assertIn("character-for-character", prompt)
        self.assertIn("contextual inference", prompt)
        self.assertNotIn("Chinese", prompt)
        self.assertNotIn("rate_ori", prompt)
        self.assertNotIn("巴黎", prompt)

    def test_accepting_audit_contract_requires_factual_fields(self):
        response = {
            "decision": "accept", "reason_code": "direct_fact", "is_factual_recall": True, "factual_type": "geography", "answer_unique": True,
            "has_material_ambiguity": False, "option_dependent": False, "requires_multistep_reasoning": False,
            "requires_contextual_inference": False, "is_time_sensitive": False, "question_premise_valid": True,
            "canonical_answer_well_formed": True,
            "reason": "Direct fact.", "subject_en": "France", "relation_en": "capital", "answer_en": "Paris", "prompt_en": "The capital of France is",
        }
        self.assertEqual(pipeline.validate_audit_response(response, "Paris"), ("accept", []))
        del response["prompt_en"]
        decision, errors = pipeline.validate_audit_response(response, "Paris")
        self.assertEqual(decision, "needs_review")
        self.assertIn("invalid_prompt_en", errors)

    def test_accept_cannot_require_contextual_inference(self):
        response = {
            "decision": "accept", "reason_code": "direct_fact", "is_factual_recall": True, "factual_type": "other",
            "answer_unique": True, "has_material_ambiguity": False, "option_dependent": False,
            "requires_multistep_reasoning": False, "requires_contextual_inference": True,
            "is_time_sensitive": False, "question_premise_valid": True,
            "canonical_answer_well_formed": True, "reason": "Clue solving.",
            "subject_en": "game", "relation_en": "identified as", "answer_en": "hockey game",
            "prompt_en": "The game is",
        }
        decision, errors = pipeline.validate_audit_response(response, "hockey game")
        self.assertEqual(decision, "needs_review")
        self.assertIn("contradictory_accept_requires_contextual_inference", errors)

    def test_malformed_question_must_be_sent_to_review(self):
        response = {
            "decision": "reject", "reason_code": "malformed_question", "is_factual_recall": True,
            "factual_type": "science", "answer_unique": True, "has_material_ambiguity": False,
            "option_dependent": False, "requires_multistep_reasoning": False,
            "requires_contextual_inference": False, "is_time_sensitive": False,
            "question_premise_valid": False, "canonical_answer_well_formed": True,
            "reason": "The premise is scientifically incorrect.",
        }
        decision, errors = pipeline.validate_audit_response(response, "photosynthesis")
        self.assertEqual(decision, "needs_review")
        self.assertIn("invalid_premise_decision", errors)
        self.assertIn("invalid_needs_review_reason_decision", errors)

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

    def test_sensenova_audit_uses_anthropic_messages_api(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"decision":"reject"}')]
        )
        call = pipeline.call_audit_model(
            client,
            pipeline.AUDIT_MODEL,
            "Review this item.",
            self.config["runtime"],
        )
        self.assertEqual(call["raw_response"], '{"decision":"reject"}')
        client.messages.create.assert_called_once_with(
            model="sensenova-6.7-flash-lite",
            max_tokens=4096,
            system="You are a careful dataset pipeline component. Follow the requested output format.",
            messages=[{"role": "user", "content": "Review this item."}],
            temperature=0.0,
        )

    @patch("factual_pitfalls.pipeline.Anthropic")
    def test_sensenova_client_uses_bearer_auth_token(self, anthropic_client):
        with patch.dict(
            "os.environ",
            {"SENSENOVA_API_KEY": "fill-me", "SENSENOVA_BASE_URL": "https://token.sensenova.cn"},
            clear=False,
        ):
            pipeline.make_audit_client(Path("."), 30)
        anthropic_client.assert_called_once_with(
            api_key="",
            auth_token="fill-me",
            base_url="https://token.sensenova.cn",
            timeout=30,
            max_retries=0,
            default_headers={
                "Authorization": "Bearer fill-me",
                "X-Api-Key": ANY,
            },
        )


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

    def test_parallel_stage_processes_each_pending_item_once(self):
        inputs = [{"candidate_id": f"raw_{index}"} for index in range(4)]

        def transform(item):
            return [{"review_id": item["candidate_id"], "terminal_status": "completed"}]

        records = build_script.run_stage(
            inputs,
            self.output,
            "review_id",
            lambda item: [item["candidate_id"]],
            transform,
            False,
            False,
            None,
            max_workers=2,
        )
        self.assertEqual({record["review_id"] for record in records}, {f"raw_{index}" for index in range(4)})

    def test_full_manifest_expansion_preserves_checkpoint_candidates(self):
        candidate = {"candidate_id": "raw_1", "source_snapshot": {"original_index": 1}}
        previous = {"candidates": [candidate]}
        expanded = {"candidates": [candidate, {"candidate_id": "raw_2", "source_snapshot": {"original_index": 2}}]}
        metadata = build_script.expansion_metadata(previous, expanded, self.output)
        self.assertEqual(metadata["previous_candidate_count"], 1)
        self.assertEqual(metadata["existing_factual_review_count"], 1)

    def test_prompt_calibration_can_reuse_pilot_candidate_ids(self):
        manifest = {"candidates": [{"candidate_id": "raw_1"}, {"candidate_id": "raw_2"}]}
        selected = build_script.audit_inputs(manifest, self.output.parent, self.output.name)
        self.assertEqual([item["candidate_id"] for item in selected], ["raw_1"])

    def test_resume_rejects_a_different_prompt_version(self):
        pipeline.write_jsonl(
            self.output,
            [{"review_id": "raw_1", "audit": {"prompt_version": "raw-english-factual-audit-v1"}}],
        )
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            build_script.validate_audit_resume_version(self.output, True)

    def test_resume_rejects_qwen_audit_records(self):
        pipeline.write_jsonl(
            self.output,
            [{
                "review_id": "raw_1",
                "audit": {
                    "prompt_version": pipeline.PROMPT_VERSION,
                    "model": "qwen3.7-plus",
                },
            }],
        )
        with self.assertRaisesRegex(ValueError, "existing audit models"):
            build_script.validate_audit_resume_version(self.output, True)
