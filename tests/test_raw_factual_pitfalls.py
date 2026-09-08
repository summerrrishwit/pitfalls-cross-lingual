import importlib.util
import json
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
        "sampling_strategy": "balanced_sources",
        "supported_models": ["sensenova-6.7-flash-lite", "qwen3.7-plus"],
        "roles": {
            "triple_label_models": ["sensenova-6.7-flash-lite", "qwen3.7-plus"],
            "relation_normalization_model": "qwen3.7-plus",
        },
        "runtime": {"temperature": 0.0, "max_tokens": 2048, "timeout_seconds": 5, "max_retries": 0},
        "output_root": "derived",
    }


def candidate():
    raw = source_record("What is the capital of France?")
    return {
        "candidate_id": "raw_mmlu_000000",
        "stratum": "mmlu::geography",
        "source_snapshot": {
            "dataset": "mmlu",
            "source_path": "data/source/mmlu.json",
            "original_index": 0,
            "record": raw,
        },
    }


def extracted_response(**overrides):
    result = {
        "schema_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
        "extraction_status": "extracted",
        "exclusion_reason": None,
        "subject": "France",
        "subject_type": "country",
        "relation_raw": "capital city of the country",
        "answer": "Paris",
        "answer_type": "city",
        "canonical_fact": "The capital of France is Paris.",
        "extraction_confidence": 0.99,
    }
    result.update(overrides)
    return result


class RawSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first = self.root / "first.json"
        self.second = self.root / "second.json"
        self.first.write_text(
            json.dumps([source_record("Capital of France?"), source_record("Capital of Italy?", subject="history")]),
            encoding="utf-8",
        )
        self.second.write_text(
            json.dumps([source_record("Capital of France?", source="sciq"), source_record("Capital of Germany?", source="sciq")]),
            encoding="utf-8",
        )
        self.config = config({"mmlu": str(self.first), "sciq": str(self.second)})

    def tearDown(self):
        self.temp.cleanup()

    def test_raw_schema_requires_answer_in_choices(self):
        record = source_record("Broken")
        record["answer"] = "Not a choice"
        self.assertEqual(pipeline.raw_schema_error(record), "answer_not_in_choices")

    def test_cross_source_dedup_and_manifest_contains_no_llm_fields(self):
        manifest = pipeline.build_raw_manifest(self.config, self.root, "test-run")
        selected_questions = [item["source_snapshot"]["record"]["question"] for item in manifest["candidates"]]
        self.assertEqual(selected_questions.count("Capital of France?"), 1)
        serialized = json.dumps(manifest)
        self.assertNotIn("relation_raw", serialized)
        self.assertNotIn("prompt_en", serialized)
        self.assertEqual(manifest["source_datasets"]["allocation"]["actual_count"], 3)

    def test_same_seed_reproduces_raw_selection(self):
        first = pipeline.build_raw_manifest(self.config, self.root, "first")
        second = pipeline.build_raw_manifest(self.config, self.root, "second")
        self.assertEqual(
            [item["candidate_id"] for item in first["candidates"]],
            [item["candidate_id"] for item in second["candidates"]],
        )

    def test_all_mode_selects_every_deduplicated_candidate(self):
        manifest = pipeline.build_raw_manifest({**self.config, "raw_sample_size": "all"}, self.root, "all")
        allocation = manifest["source_datasets"]["allocation"]
        self.assertEqual(allocation["selection_mode"], "full_population")
        self.assertEqual(allocation["actual_count"], allocation["available_count"])

    def test_balanced_source_strategy_does_not_let_large_source_dominate(self):
        large = self.root / "large.json"
        large.write_text(
            json.dumps([source_record(f"Large source question {index}?", source="large") for index in range(20)]),
            encoding="utf-8",
        )
        small = self.root / "small.json"
        small.write_text(
            json.dumps([source_record(f"Small source question {index}?", source="small") for index in range(4)]),
            encoding="utf-8",
        )
        balanced = config({"large": str(large), "small": str(small)})
        balanced["raw_sample_size"] = 8
        manifest = pipeline.build_raw_manifest(balanced, self.root, "balanced")
        self.assertEqual(manifest["source_datasets"]["allocation"]["dataset_allocations"], {"large": 4, "small": 4})
        self.assertEqual(
            [candidate["source_snapshot"]["dataset"] for candidate in manifest["candidates"][:4]],
            ["large", "small", "large", "small"],
        )

    def test_prior_manifest_candidate_ids_are_excluded(self):
        first = pipeline.build_raw_manifest(self.config, self.root, "first")
        excluded = {first["candidates"][0]["candidate_id"]}
        second = pipeline.build_raw_manifest(
            self.config, self.root, "second", excluded_candidate_ids=excluded
        )
        self.assertTrue(excluded.isdisjoint({item["candidate_id"] for item in second["candidates"]}))
        self.assertEqual(second["sampling_config"]["excluded_candidate_count"], 1)


class ExtractionContractTests(unittest.TestCase):
    def setUp(self):
        self.candidate = candidate()
        self.config = config({"mmlu": "data/source/mmlu.json"})

    def test_prompt_omits_choices_and_normalized_relation(self):
        prompt = pipeline.triple_extraction_prompt(self.candidate)
        self.assertIn("What is the capital of France?", prompt)
        self.assertIn("Canonical English answer: Paris", prompt)
        self.assertNotIn("Rome", prompt)
        self.assertNotIn("Berlin", prompt)
        self.assertIn("Do not create relation_normalized", prompt)
        self.assertIn("Do not generate prompt_en", prompt)
        self.assertIn("full-sentence", prompt)
        self.assertIn('"described by"', prompt)
        self.assertIn("subject must not equal", prompt)
        self.assertIn("single-definition", prompt)
        self.assertIn("option-dependent non-unique", prompt)

    def test_valid_extracted_response_allows_grounded_short_answer(self):
        self.assertEqual(pipeline.validate_triple_extraction_response(extracted_response(), "Paris"), [])
        self.assertEqual(
            pipeline.validate_triple_extraction_response(
                extracted_response(answer="George Santayana"),
                'George Santayana wrote "Only the dead have seen the end of war"',
            ),
            [],
        )
        errors = pipeline.validate_triple_extraction_response(extracted_response(answer="London"), "Paris")
        self.assertIn("answer_not_grounded_in_source", errors)

    def test_extracted_subject_cannot_repeat_answer(self):
        errors = pipeline.validate_triple_extraction_response(
            extracted_response(subject="Paris", answer="Paris"), "Paris"
        )
        self.assertIn("subject_matches_answer", errors)

    def test_sentence_like_source_answer_can_supply_grounded_object(self):
        answer = "A mass of sediment is deposited at the mouth of a river."
        self.assertTrue(pipeline.source_answer_is_unsuitable(answer))
        self.assertFalse(pipeline.source_answer_is_unsuitable("Voluntary reliance on an external power for security"))
        errors = pipeline.validate_triple_extraction_response(
            extracted_response(answer="sediment"), answer
        )
        self.assertEqual(errors, [])

    def test_grounding_accepts_reordered_content_words(self):
        self.assertTrue(
            pipeline.answer_is_grounded_in_source("deflated cuff", "The cuff is deflated.")
        )

    def test_grounding_adjustment_repairs_near_typo_and_downgrades_explanation(self):
        typo, typo_changes = pipeline.adjust_extraction_response_grounding(
            extracted_response(answer="sweet glands"), "sweat glands"
        )
        self.assertEqual(typo["answer"], "sweat glands")
        self.assertIn("answer_near_match_replaced_with_source_answer", typo_changes)
        explanation, explanation_changes = pipeline.adjust_extraction_response_grounding(
            extracted_response(answer="abbreviation for Christmas"),
            "It means the same because it's an abbreviation",
        )
        self.assertEqual(explanation["extraction_status"], "not_extractable")
        self.assertEqual(explanation["exclusion_reason"], "answer_not_suitable")
        self.assertIn("ungrounded_sentence_answer_downgraded_to_not_extractable", explanation_changes)

    def test_not_extractable_requires_null_triple_fields(self):
        response = {
            "schema_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
            "extraction_status": "not_extractable",
            "exclusion_reason": "scenario_classification",
            "subject": None,
            "subject_type": None,
            "relation_raw": None,
            "answer": None,
            "answer_type": None,
            "canonical_fact": None,
            "extraction_confidence": 0.98,
        }
        self.assertEqual(pipeline.validate_triple_extraction_response(response, "Paris"), [])
        response["subject"] = "France"
        self.assertIn(
            "not_extractable_has_subject",
            pipeline.validate_triple_extraction_response(response, "Paris"),
        )

    @patch("factual_pitfalls.pipeline.call_extraction_model")
    def test_terminal_record_copies_source_choices_programmatically(self, call):
        call.return_value = {
            "raw_response": json.dumps(extracted_response()),
            "attempt_count": 1,
            "error": None,
        }
        record = pipeline.extract_triple_candidate(self.candidate, self.config, client=None)
        self.assertEqual(record["terminal_status"], "completed")
        self.assertEqual(record["source_id"], "raw_mmlu_000000")
        self.assertEqual(record["source_choices"], ["Paris", "Rome", "Berlin"])
        self.assertEqual(record["source_answer"], "Paris")
        self.assertEqual(record["answer"], "Paris")
        self.assertNotIn("prompt_en", record)
        self.candidate["source_snapshot"]["record"]["choices"][0] = "changed"
        self.assertEqual(record["source_choices"][0], "Paris")

    @patch("factual_pitfalls.pipeline.call_extraction_model")
    def test_sentence_answer_is_sent_to_model_for_object_normalization(self, call):
        item = candidate()
        item["source_snapshot"]["record"]["answer"] = "A complete explanatory sentence ends here."
        item["source_snapshot"]["record"]["choices"][0] = "A complete explanatory sentence ends here."
        call.return_value = {"raw_response": json.dumps({
            "schema_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
            "extraction_status": "not_extractable",
            "exclusion_reason": "answer_not_suitable",
            **{field: None for field in pipeline.TRIPLE_FIELDS},
            "extraction_confidence": 0.99,
        }), "attempt_count": 1, "error": None}
        record = pipeline.extract_triple_candidate(item, self.config, client=None)
        call.assert_called_once()
        self.assertEqual(record["extraction_status"], "not_extractable")
        self.assertEqual(record["exclusion_reason"], "answer_not_suitable")
        self.assertEqual(record["extraction"]["attempt_count"], 1)

    def test_sensenova_extraction_uses_messages_api(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(extracted_response()))]
        )
        call = pipeline.call_extraction_model(
            client,
            pipeline.TRIPLE_EXTRACTION_MODEL,
            "Extract this item.",
            self.config["runtime"],
        )
        self.assertIsNone(call["error"])
        client.messages.create.assert_called_once_with(
            model="sensenova-6.7-flash-lite",
            max_tokens=2048,
            system="You are a careful atomic factual-triple extraction component. Follow the JSON contract exactly.",
            messages=[{"role": "user", "content": "Extract this item."}],
            temperature=0.0,
        )

    @patch.object(pipeline.SENSENOVA_REQUEST_START_LIMITER, "wait")
    def test_sensenova_extraction_applies_global_request_spacing(self, wait):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(extracted_response()))]
        )
        with patch.dict(
            "os.environ",
            {"SENSENOVA_MIN_REQUEST_INTERVAL_SECONDS": "1.1"},
            clear=False,
        ):
            call = pipeline.call_extraction_model(
                client,
                pipeline.TRIPLE_EXTRACTION_MODEL,
                "Extract this item.",
                self.config["runtime"],
            )
        self.assertIsNone(call["error"])
        wait.assert_called_once_with(1.1)

    @patch("factual_pitfalls.pipeline.Anthropic")
    def test_sensenova_client_uses_bearer_token(self, anthropic_client):
        with patch.dict(
            "os.environ",
            {"SENSENOVA_API_KEY": "fill-me", "SENSENOVA_BASE_URL": "https://token.sensenova.cn"},
            clear=False,
        ):
            pipeline.make_extraction_client(Path("."), 30, "sensenova-6.7-flash-lite")
        anthropic_client.assert_called_once_with(
            api_key="",
            auth_token="fill-me",
            base_url="https://token.sensenova.cn",
            timeout=30,
            max_retries=0,
            default_headers={"Authorization": "Bearer fill-me", "X-Api-Key": ANY},
        )

    def test_qwen_extraction_uses_openai_chat_api(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(extracted_response())))]
        )
        call = pipeline.call_extraction_model(
            client,
            "qwen3.7-plus",
            "Extract this item.",
            self.config["runtime"],
        )
        self.assertIsNone(call["error"])
        client.chat.completions.create.assert_called_once()

    @patch("factual_pitfalls.pipeline.OpenAI")
    def test_qwen_client_uses_required_openai_endpoint(self, openai_client):
        with patch.dict(
            "os.environ",
            {
                "QWEN_EXTRACTION_PROVIDER": "openai",
                "OPENAI_API_KEY": "fill-me",
                "OPENAI_BASE_URL": pipeline.QWEN_OPENAI_BASE_URL,
            },
            clear=False,
        ):
            pipeline.make_extraction_client(Path("."), 30, "qwen3.7-plus")
        openai_client.assert_called_once_with(
            api_key="fill-me",
            base_url="https://ctapi.csxdtx.com:16000/v1",
            timeout=30,
            max_retries=0,
        )

    @patch("factual_pitfalls.pipeline.OpenAI")
    def test_qwen_client_uses_bailian_by_default(self, openai_client):
        with patch.dict(
            "os.environ",
            {
                "BAILIAN_API_KEY": "fill-me",
                "BAILIAN_BASE_URL": pipeline.BAILIAN_OPENAI_BASE_URL,
            },
            clear=False,
        ):
            pipeline.make_extraction_client(Path("."), 30, "qwen3.7-plus")
        openai_client.assert_called_once_with(
            api_key="fill-me",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30,
            max_retries=0,
        )

    def test_qwen_bailian_uses_pinned_plus_snapshot_without_thinking(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(extracted_response())))]
        )
        with patch.dict(
            "os.environ",
            {"QWEN_EXTRACTION_PROVIDER": "bailian"},
            clear=False,
        ):
            call = pipeline.call_extraction_model(
                client,
                "qwen3.7-plus",
                "Extract this item.",
                self.config["runtime"],
            )
        self.assertEqual(call["api_model"], "qwen3.7-plus-2026-05-26")
        client.chat.completions.create.assert_called_once_with(
            model="qwen3.7-plus-2026-05-26",
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful atomic factual-triple extraction component. Follow the JSON contract exactly.",
                },
                {"role": "user", "content": "Extract this item."},
            ],
            temperature=0.0,
            max_tokens=2048,
            extra_body={"enable_thinking": False},
        )


class LocalAdjudicationTests(unittest.TestCase):
    def test_reviewed_override_preserves_raw_response_and_clears_validation_failure(self):
        record = {
            "source_id": "raw_1",
            "source_answer": "amino",
            "terminal_status": "validation_failed",
            "extraction_status": "extracted",
            "validation_errors": ["answer_not_grounded_in_source"],
            "extraction_confidence": 0.9,
            "extraction": {"raw_response": "original", "program_adjustments": []},
            **extracted_response(answer="amino acids"),
        }
        fields = {field: record[field] for field in pipeline.TRIPLE_FIELDS}
        fields["answer"] = "amino"
        policy = {
            "adjudication_version": "local-extraction-adjudication-v1",
            "model": pipeline.TRIPLE_EXTRACTION_MODEL,
            "prompt_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
            "reviewer": "codex",
            "decisions": [{
                "source_id": "raw_1",
                "action": "override_extracted",
                "fields": fields,
                "rationale": "Use the exact source answer.",
            }],
        }
        result = pipeline.apply_extraction_adjudications(
            [record], policy, pipeline.TRIPLE_EXTRACTION_MODEL
        )[0]
        self.assertEqual(result["terminal_status"], "completed")
        self.assertEqual(result["answer"], "amino")
        self.assertEqual(result["extraction"]["raw_response"], "original")
        self.assertEqual(result["validation_errors"], [])
        self.assertIn("prior", result["local_adjudication"])

    def test_reviewed_not_extractable_nulls_triple(self):
        record = {
            "source_id": "raw_1",
            "source_answer": "north pole",
            "terminal_status": "validation_failed",
            "extraction_status": "extracted",
            "validation_errors": ["subject_matches_answer"],
            "extraction_confidence": 0.9,
            "extraction": {"raw_response": "original", "program_adjustments": []},
            **extracted_response(subject="north pole", answer="north pole"),
        }
        policy = {
            "adjudication_version": "local-extraction-adjudication-v1",
            "model": pipeline.TRIPLE_EXTRACTION_MODEL,
            "prompt_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
            "decisions": [{
                "source_id": "raw_1",
                "action": "mark_not_extractable",
                "exclusion_reason": "non_unique_answer",
                "rationale": "The answer is not uniquely determined.",
            }],
        }
        result = pipeline.apply_extraction_adjudications(
            [record], policy, pipeline.TRIPLE_EXTRACTION_MODEL
        )[0]
        self.assertEqual(result["terminal_status"], "completed")
        self.assertEqual(result["extraction_status"], "not_extractable")
        self.assertTrue(all(result[field] is None for field in pipeline.TRIPLE_FIELDS))

    def test_reapplying_same_policy_is_idempotent(self):
        record = {
            "source_id": "raw_1",
            "source_answer": "north pole",
            "terminal_status": "completed",
            "local_adjudication": {
                "adjudication_version": "local-extraction-adjudication-v1"
            },
        }
        policy = {
            "adjudication_version": "local-extraction-adjudication-v1",
            "model": pipeline.TRIPLE_EXTRACTION_MODEL,
            "prompt_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
            "decisions": [{
                "source_id": "raw_1",
                "action": "mark_not_extractable",
                "exclusion_reason": "non_unique_answer",
            }],
        }
        result = pipeline.apply_extraction_adjudications(
            [record], policy, pipeline.TRIPLE_EXTRACTION_MODEL
        )[0]
        self.assertEqual(result, record)


class CodexCalibrationTests(unittest.TestCase):
    def test_reference_overrides_and_evaluation_gate(self):
        base = [{
            "candidate_id": "raw_1",
            "source_id": "raw_1",
            "source_dataset": "sciq",
            "source_original_index": 1,
            "source_question": "Definition?",
            "source_choices": ["term", "other"],
            "source_answer": "term",
            "extraction_status": "not_extractable",
        }]
        policy = {
            "reference_version": pipeline.CODEX_REFERENCE_VERSION,
            "overrides": [{
                "candidate_id": "raw_1",
                "extraction_status": "extracted",
                "codex_review": "single definition",
                "expected_triple": {**{field: "value" for field in pipeline.TRIPLE_FIELDS}, "answer": "term"},
            }],
        }
        reference = pipeline.build_codex_reference(base, policy)
        self.assertEqual(reference[0]["extraction_status"], "extracted")
        observed = [{
            "candidate_id": "raw_1",
            "terminal_status": "completed",
            "extraction_status": "extracted",
            **{field: "value" for field in pipeline.TRIPLE_FIELDS},
            "answer": "term",
        }]
        evaluation = pipeline.evaluate_model_against_codex_reference(
            reference, observed, "qwen3.7-plus", threshold=0.03
        )
        self.assertTrue(evaluation["gate_passed"])
        self.assertEqual(evaluation["label_error_rate"], 0.0)

    def test_missing_reference_record_blocks_gate(self):
        reference = [{"candidate_id": "raw_1", "extraction_status": "not_extractable"}]
        evaluation = pipeline.evaluate_model_against_codex_reference(
            reference, [], "qwen3.7-plus", threshold=0.03
        )
        self.assertFalse(evaluation["gate_passed"])
        self.assertEqual(evaluation["incomplete_count"], 1)

    def test_dual_model_summary_routes_disagreement_to_codex(self):
        first = [{"candidate_id": "raw_1", "terminal_status": "completed", "extraction_status": "extracted"}]
        second = [{"candidate_id": "raw_1", "terminal_status": "completed", "extraction_status": "not_extractable"}]
        summary = pipeline.summarize_dual_model_records({
            "sensenova-6.7-flash-lite": first,
            "qwen3.7-plus": second,
        })
        self.assertEqual(summary["agreement_counts"]["requires_codex_review"], 1)
        self.assertEqual(len(summary["disagreements"]), 1)


class RelationNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "terminal_status": "completed",
                "extraction_status": "extracted",
                "validation_errors": [],
                "source_id": "raw_1",
                "source_dataset": "mmlu",
                **{key: value for key, value in extracted_response().items() if key in pipeline.TRIPLE_FIELDS},
            },
            {
                "terminal_status": "completed",
                "extraction_status": "extracted",
                "validation_errors": [],
                "source_id": "raw_2",
                "source_dataset": "sciq",
                **{key: value for key, value in extracted_response(subject="Germany", answer="Berlin", canonical_fact="The capital of Germany is Berlin.").items() if key in pipeline.TRIPLE_FIELDS},
            },
        ]

    def test_inventory_groups_same_relation_signature(self):
        inventory = pipeline.build_relation_inventory(self.records, max_examples=1)
        self.assertEqual(inventory["extracted_record_count"], 2)
        self.assertEqual(inventory["signature_count"], 1)
        self.assertEqual(inventory["entries"][0]["count"], 2)
        self.assertEqual(len(inventory["entries"][0]["examples"]), 1)

    def test_mapping_is_type_checked_and_applied_deterministically(self):
        inventory = pipeline.build_relation_inventory(self.records)
        signature_id = inventory["entries"][0]["signature_id"]
        taxonomy = {
            "taxonomy_version": "relation-taxonomy-v1",
            "relations": [
                {
                    "id": "country_capital",
                    "definition": "capital city of a country",
                    "subject_type": "country",
                    "answer_type": "city",
                    "direction": "country -> city",
                    "examples": [["France", "Paris"]],
                }
            ],
        }
        mapping = {
            "mapping_version": "relation-mapping-v1",
            "inventory_version": inventory["inventory_version"],
            "taxonomy_version": taxonomy["taxonomy_version"],
            "mappings": [
                {
                    "signature_id": signature_id,
                    "normalization_status": "mapped",
                    "relation_normalized": "country_capital",
                    "subject_type_normalized": "country",
                    "answer_type_normalized": "city",
                    "normalization_reason": "Definition, types, and direction match.",
                    "normalization_confidence": 0.99,
                }
            ],
        }
        first = pipeline.apply_relation_mapping(self.records, inventory, taxonomy, mapping)
        second = pipeline.apply_relation_mapping(self.records, inventory, taxonomy, mapping)
        self.assertEqual([row["relation_normalized"] for row in first], ["country_capital"] * 2)
        self.assertEqual(
            [row["relation_normalized"] for row in first],
            [row["relation_normalized"] for row in second],
        )

        mapping["mappings"][0]["answer_type_normalized"] = "country"
        with self.assertRaisesRegex(ValueError, "normalized answer type mismatch"):
            pipeline.apply_relation_mapping(self.records, inventory, taxonomy, mapping)


class ResumeStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "triple_extractions.jsonl"
        pipeline.write_jsonl(
            self.output,
            [
                {
                    "candidate_id": "raw_1",
                    "terminal_status": "extraction_failed",
                    "created_at": "first",
                    "extraction": {"error": {"category": "transient_failure"}},
                }
            ],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_retry_failed_replaces_record_and_keeps_history(self):
        inputs = [{"candidate_id": "raw_1"}]
        calls = []

        def transform(item):
            calls.append(item["candidate_id"])
            return [{"candidate_id": item["candidate_id"], "terminal_status": "completed", "created_at": "second", "extraction": {}}]

        build_script.run_stage(inputs, self.output, "candidate_id", lambda item: [item["candidate_id"]], transform, True, False, None)
        self.assertEqual(calls, [])
        records = build_script.run_stage(inputs, self.output, "candidate_id", lambda item: [item["candidate_id"]], transform, True, True, None)
        self.assertEqual(calls, ["raw_1"])
        self.assertEqual(records[0]["retry_history"][0]["terminal_status"], "extraction_failed")

    def test_resume_rejects_old_audit_prompt_version(self):
        pipeline.write_jsonl(
            self.output,
            [
                {
                    "candidate_id": "raw_1",
                    "extraction": {
                        "prompt_version": "raw-english-factual-audit-v7",
                        "model": pipeline.TRIPLE_EXTRACTION_MODEL,
                    },
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            build_script.validate_extraction_resume(self.output, True)

    def test_resume_rejects_checkpoint_from_other_label_model(self):
        pipeline.write_jsonl(
            self.output,
            [{
                "candidate_id": "raw_1",
                "extraction": {
                    "prompt_version": pipeline.TRIPLE_EXTRACTION_PROMPT_VERSION,
                    "model": "sensenova-6.7-flash-lite",
                },
            }],
        )
        with self.assertRaisesRegex(ValueError, "qwen3.7-plus"):
            build_script.validate_extraction_resume(self.output, True, "qwen3.7-plus")


if __name__ == "__main__":
    unittest.main()
