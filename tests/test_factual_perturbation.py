import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "factual_pitfalls" / "perturbation.py"
MODULE_SPEC = importlib.util.spec_from_file_location("factual_perturbation_runtime", MODULE_PATH)
perturbation = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC and MODULE_SPEC.loader
sys.modules[MODULE_SPEC.name] = perturbation
MODULE_SPEC.loader.exec_module(perturbation)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_factual_perturbation.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("factual_perturbation_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


class PerturbationRuntimeTests(unittest.TestCase):
    def test_config_overlay_replaces_simulation_models_without_losing_roles(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_sensitivity_v2.json"
        )
        self.assertEqual(config["config_version"], "factual-perturbation-zh-sensitivity-v2")
        self.assertEqual(config["model_roles"]["translation"]["primary"]["model"], "qwen3.8-max")
        self.assertEqual(
            [row["model"] for row in config["model_roles"]["simulation"]["models"]],
            ["qwen3.6-27b", "bailian/deepseek-v3.2", "gpt-5-mini", "gemini-3.7-flash"],
        )

    def test_strength_overlay_uses_matched_candidate_neutral(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_strength_v3.json"
        )
        self.assertEqual(config["model_roles"]["perturbation_generation"]["max_output_tokens"], 2048)
        self.assertEqual(config["model_roles"]["simulation"]["neutral_context_mode"], "matched_per_candidate")

    def test_streamlined_overlay_uses_repairs_dual_l2_and_soft_auto_judge(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_streamlined_v5.json"
        )
        self.assertTrue(config["model_roles"]["translation"]["repair_rejected"])
        self.assertTrue(config["model_roles"]["distractor_generation"]["enabled"])
        self.assertEqual(
            config["model_roles"]["perturbation_generation"]["strength_levels"],
            ["l2_lexical", "l2_relational"],
        )
        self.assertEqual(
            config["model_roles"]["perturbation_generation"]["primary"]["max_output_tokens"],
            2048,
        )
        self.assertFalse(config["model_roles"]["perturbation_validation"]["hard_gate"])

    def test_ollama_overlay_uses_two_local_models_serially(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_ollama_v6.json"
        )
        self.assertEqual(
            [row["model"] for row in config["model_roles"]["simulation"]["models"]],
            ["gemma3:12b", "llama3.1:8b"],
        )
        self.assertEqual(config["provider_profiles"]["ollama_local"]["authentication"], "none")
        self.assertEqual(config["execution"]["max_workers"], 1)
        self.assertTrue(config["model_roles"]["simulation"]["batch_by_model"])

    def test_five_model_overlay_combines_api_and_local_simulation_models(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_five_model_v7.json"
        )
        models = config["model_roles"]["simulation"]["models"]
        self.assertEqual(
            [row["model"] for row in models],
            [
                "qwen3.6-27b",
                "bailian/deepseek-v3.2",
                "gemini-3.7-flash",
                "gemma3:12b",
                "llama3.1:8b",
            ],
        )
        self.assertEqual(
            [row["provider_profile"] for row in models],
            ["aliyun", "deepseek", "test_gateway", "ollama_local", "ollama_local"],
        )
        self.assertEqual(config["provider_profiles"]["ollama_local"]["authentication"], "none")
        self.assertEqual(config["execution"]["provider_max_concurrency"]["ollama_local"], 1)
        self.assertEqual(config["execution"]["timeout_seconds"], 180)
        self.assertEqual(
            config["model_roles"]["perturbation_generation"]["target_distractors_per_source"],
            1,
        )
        self.assertEqual(config["model_roles"]["simulation"]["endpoint"], "multi_option")
        self.assertEqual(config["model_roles"]["simulation"]["neutral_context_mode"], "matched_per_candidate")
        self.assertTrue(config["model_roles"]["simulation"]["batch_by_model"])
        self.assertEqual(models[3]["timeout_seconds"], 900)
        self.assertEqual(models[4]["timeout_seconds"], 900)
        self.assertEqual(models[0]["timeout_seconds"], 30)
        self.assertEqual(models[2]["max_retries"], 1)

        simulation_config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_five_model_simulation_v7.json"
        )
        self.assertEqual(simulation_config["execution"]["timeout_seconds"], 180)
        self.assertEqual(
            [row["model"] for row in simulation_config["model_roles"]["simulation"]["models"]],
            [row["model"] for row in models],
        )

    def test_router_allows_unauthenticated_loopback_profile(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"local": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "http://127.0.0.1:11434/v1",
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            profile, key, base = router._profile(
                {"provider_profile": "local", "model": "gemma3:12b"}
            )
        self.assertEqual(profile["authentication"], "none")
        self.assertEqual(key, "local-no-auth")
        self.assertEqual(base, "http://127.0.0.1:11434/v1")

    def test_router_rejects_unauthenticated_remote_profile(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"remote": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "https://example.com/v1",
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            with self.assertRaisesRegex(RuntimeError, "unauthenticated_provider_requires_loopback"):
                router._profile({"provider_profile": "remote", "model": "m"})

    def test_router_allows_model_specific_retry_limit(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 3},
            "provider_profiles": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            with patch.object(router, "_one_call", side_effect=RuntimeError("boom")), patch.object(
                perturbation.time, "sleep"
            ):
                result = router.request_json(
                    "stage",
                    "item",
                    {"provider_profile": "p", "model": "m", "max_retries": 1},
                    "prompt",
                    lambda value: None,
                )
        self.assertEqual(result.terminal_status, "failed")
        self.assertEqual(result.attempt_count, 2)

    def test_router_passes_json_mode_to_openai_compatible_provider(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"local": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "http://localhost:11434/v1",
                "json_mode": True,
            }},
        }
        captured = {}

        class FakeCompletions:
            @staticmethod
            def create(**request):
                captured.update(request)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"choice":"A"}'))],
                    model="m",
                    usage=None,
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            perturbation, "OpenAI", return_value=fake_client
        ) as client_factory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            text, response_model, _ = router._one_call(
                {"provider_profile": "local", "model": "m", "timeout_seconds": 7}, "prompt"
            )
        self.assertEqual(text, '{"choice":"A"}')
        self.assertEqual(response_model, "m")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(client_factory.call_args.kwargs["timeout"], 7.0)

    def test_translation_schema_requires_exact_distractor_coverage(self):
        value = {
            "prompt_zh": "法国的首都是",
            "answer_zh": "巴黎",
            "answer_aliases_zh": ["巴黎"],
            "distractors": [{"distractor_id": "d1", "text_zh": "柏林"}],
        }
        perturbation.validate_translation(value, ["d1"])
        with self.assertRaisesRegex(ValueError, "distractor_translation_id_mismatch"):
            perturbation.validate_translation(value, ["d1", "d2"])

    def test_generated_distractor_schema_requires_exact_unique_bilingual_rows(self):
        value = {"distractors": [
            {"text_en": "Berlin", "text_zh": "柏林"},
            {"text_en": "Rome", "text_zh": "罗马"},
        ]}
        perturbation.validate_generated_distractors(value, 2)
        value["distractors"][1] = dict(value["distractors"][0])
        with self.assertRaisesRegex(ValueError, "duplicate_generated_distractors"):
            perturbation.validate_generated_distractors(value, 2)

    def test_distractor_review_requires_prompt_fit_and_single_answer(self):
        checks = {key: True for key in perturbation.DISTRACTOR_CHECKS}
        perturbation.validate_decision(
            {"decision": "accept", "issues": [], "checks": checks},
            perturbation.DISTRACTOR_CHECKS,
        )
        checks.pop("prompt_completion_fit")
        with self.assertRaisesRegex(ValueError, "invalid_checks"):
            perturbation.validate_decision(
                {"decision": "accept", "issues": [], "checks": checks},
                perturbation.DISTRACTOR_CHECKS,
            )

    def test_soft_auto_judge_review_packet_includes_rejected_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "soft-review",
                "perturbation_auto_judge_hard_gate": False,
                "records": [{
                    "source_id": "s1", "source_dataset": "toy", "answer_type": "city",
                    "prompt_en": "The capital is", "answer_en": "Paris",
                    "canonical_fact": "The capital is Paris.", "relation_raw": "capital",
                    "relation_normalized": "entity_location",
                }],
            })
            perturbation.write_jsonl(run_dir / "translations.jsonl", [{
                "source_id": "s1", "terminal_status": "completed",
                "parsed_response": {"prompt_zh": "首都是", "answer_zh": "巴黎"},
            }])
            accepted = {
                "terminal_status": "completed",
                "parsed_response": {"decision": "accept", "checks": {"valid": True}},
            }
            perturbation.write_jsonl(run_dir / "verified_distractors.jsonl", [
                {"item_id": "d1", "source_id": "s1", "distractor_en": "Berlin", "distractor_zh": "柏林", **accepted},
                {"item_id": "d2", "source_id": "s1", "distractor_en": "Rome", "distractor_zh": "罗马", **accepted},
            ])
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
                "candidate": {"strength": "l2_lexical"}, "terminal_status": "completed",
                "parsed_response": {"decision": "reject", "checks": {"valid": False}},
            }])
            summary = perturbation.export_codex_review_packet(run_dir)
            packet = perturbation.read_json(run_dir / "codex_proxy_review_input_v5.json")
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(packet["records"][0]["candidates"][0]["candidate_id"], "c1")

    def test_choice_layout_is_deterministic_and_balanced_per_language(self):
        first = perturbation._choice_layout("s1", "d1", "en", "Paris", "Berlin")
        second = perturbation._choice_layout("s1", "d1", "en", "Paris", "Berlin")
        self.assertEqual(first, second)
        self.assertEqual(set(first[0].values()), {"Paris", "Berlin"})
        self.assertIn(first[1], {"A", "B"})

    def test_multi_choice_layout_is_deterministic_and_tracks_target(self):
        first = perturbation._multi_choice_layout(
            "s1", "d1", "Paris", [("d1", "Berlin"), ("d2", "Rome")], 3
        )
        second = perturbation._multi_choice_layout(
            "s1", "d1", "Paris", [("d1", "Berlin"), ("d2", "Rome")], 3
        )
        self.assertEqual(first, second)
        options, correct_choice, target_choice, option_ids = first
        self.assertEqual(set(options.values()), {"Paris", "Berlin", "Rome"})
        self.assertNotEqual(correct_choice, target_choice)
        self.assertEqual(option_ids[list(options).index(target_choice)], "d1")
        perturbation.validate_choice({"choice": "C"}, tuple(options))

    def test_strength_candidates_require_ordered_levels_and_matched_neutral(self):
        candidates = {
            "candidates": [
                {
                    "strength": level,
                    "english_context": f"target {level}",
                    "chinese_context": f"目标 {level}",
                    "neutral_english_context": f"neutral {level}",
                    "neutral_chinese_context": f"中性 {level}",
                }
                for level in ("l1", "l2")
            ]
        }
        perturbation.validate_perturbations(candidates, 2, ("l1", "l2"), True)
        candidates["candidates"].reverse()
        with self.assertRaisesRegex(ValueError, "invalid_strength_levels"):
            perturbation.validate_perturbations(candidates, 2, ("l1", "l2"), True)

    def test_stage_resume_does_not_repeat_completed_items(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            calls = []

            def worker(item):
                calls.append(item["item_id"])
                return {"item_id": item["item_id"], "terminal_status": "completed"}

            items = [{"item_id": "a"}, {"item_id": "b"}]
            perturbation.stage_run(items, output, "item_id", worker, 1)
            first_bytes = output.read_bytes()
            perturbation.stage_run(items, output, "item_id", worker, 1)
            self.assertEqual(calls, ["a", "b"])
            self.assertEqual(first_bytes, output.read_bytes())

    def test_stage_can_freeze_terminal_failures_for_proxy_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            failed = {
                "item_id": "a", "terminal_status": "failed",
                "attempt_count": 8, "attempts": [{"attempt": index} for index in range(8)],
            }
            perturbation.write_jsonl(output, [failed])
            calls = []

            def worker(item):
                calls.append(item["item_id"])
                return {"item_id": item["item_id"], "terminal_status": "completed"}

            rows = perturbation.stage_run(
                [{"item_id": "a"}], output, "item_id", worker, 1,
                retry_terminal_failures=False,
            )
            self.assertEqual(calls, [])
            self.assertEqual(rows, [failed])

    def test_serial_stage_resume_preserves_failed_attempt_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            perturbation.write_jsonl(output, [{
                "item_id": "a", "terminal_status": "failed", "attempt_count": 2,
                "attempts": [{"attempt": 1}, {"attempt": 2}],
            }])

            def worker(item):
                return {
                    "item_id": item["item_id"], "terminal_status": "completed",
                    "attempt_count": 1, "attempts": [{"attempt": 1, "status": "ok"}],
                }

            rows = perturbation.stage_run([{"item_id": "a"}], output, "item_id", worker, 1)
            self.assertEqual(rows[0]["attempt_count"], 3)
            self.assertEqual(len(rows[0]["attempts"]), 3)
            self.assertEqual(rows[0]["resume_run_count"], 1)

    def test_stratified_sample_is_deterministic_and_covers_datasets(self):
        rows = []
        for dataset in ("a", "b", "c", "d", "e"):
            for index in range(4):
                rows.append({
                    "source_id": f"{dataset}-{index}", "source_dataset": dataset,
                    "prompt_quality_tier": "strict" if index % 2 else "fallback",
                    "answer_type": f"type-{index}",
                })
        first = perturbation.stratified_sample(rows, 10, 7)
        second = perturbation.stratified_sample(rows, 10, 7)
        self.assertEqual(first, second)
        self.assertEqual({row["source_dataset"] for row in first}, {"a", "b", "c", "d", "e"})

    def test_target_distractor_cap_keeps_full_foil_pool_unchanged(self):
        accepted = [
            {"source_id": "s1", "item_id": "d1", "distractor_source": "original_wrong_option"},
            {"source_id": "s1", "item_id": "d2", "distractor_source": "original_wrong_option"},
            {"source_id": "s1", "item_id": "d3", "distractor_source": "generated_replacement"},
        ]
        targeted = perturbation._select_target_distractors(accepted, 1, 7)
        self.assertEqual(len(targeted), 1)
        self.assertEqual(len(accepted), 3)
        self.assertIn(targeted[0]["item_id"], {"d1", "d2"})
        self.assertGreaterEqual(
            len([row for row in accepted if row["item_id"] != targeted[0]["item_id"]]),
            2,
        )

    def test_redacted_event_contains_no_prompt_or_endpoint(self):
        config = {"execution": {"timeout_seconds": 1, "max_retries": 0}, "provider_profiles": {}}
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            router = perturbation.ModelRouter(config, Path(directory) / ".env", event_path)
            result = perturbation.ModelResult("failed", None, None, None, {}, 1, 1, [{"attempt": 1, "status": "failed"}])
            router._event("stage", "item", {"provider_profile": "p", "model": "m"}, result)
            event = json.loads(event_path.read_text())
            self.assertFalse(event["credentials_or_endpoints_included"])
            self.assertNotIn("prompt", event)
            self.assertNotIn("base_url", event)

    def test_simulation_inputs_share_baselines_across_candidates(self):
        item_by_id = {
            "s1": {"source_id": "s1", "prompt_en": "Capital of France:", "answer_en": "Paris"}
        }
        translation_by_id = {
            "s1": {"parsed_response": {"prompt_zh": "法国的首都是：", "answer_zh": "巴黎"}}
        }
        distractor_by_id = {
            "d1": {"item_id": "d1", "distractor_en": "Berlin", "distractor_zh": "柏林"}
        }
        perturbations = [
            {
                "candidate_id": f"c{index}", "source_id": "s1", "distractor_id": "d1",
                "candidate": {"english_context": f"context {index}", "chinese_context": f"语境 {index}"},
            }
            for index in (1, 2)
        ]
        rows = perturbation._build_simulation_inputs(
            perturbations, item_by_id, translation_by_id, distractor_by_id,
            [{"model": "m1", "provider_profile": "p1"}],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["simulation_id"] for row in rows}), 8)
        shared = [row for row in rows if row["variant"] in {"original", "neutral"}]
        targeted = [row for row in rows if row["variant"] == "targeted"]
        self.assertEqual(len(shared), 4)
        self.assertEqual(len(targeted), 4)
        self.assertTrue(all(row["candidate_id"] is None for row in shared))
        self.assertTrue(all(row["candidate_ids"] == ["c1", "c2"] for row in shared))
        self.assertEqual({row["candidate_id"] for row in targeted}, {"c1", "c2"})

    def test_matched_neutral_is_candidate_specific(self):
        item_by_id = {"s1": {"source_id": "s1", "prompt_en": "Capital:", "answer_en": "Paris"}}
        translation_by_id = {"s1": {"parsed_response": {"prompt_zh": "首都：", "answer_zh": "巴黎"}}}
        distractor_by_id = {"d1": {"item_id": "d1", "distractor_en": "Berlin", "distractor_zh": "柏林"}}
        perturbations = [
            {
                "candidate_id": f"c{index}", "source_id": "s1", "distractor_id": "d1",
                "candidate": {
                    "english_context": f"target {index}", "chinese_context": f"目标 {index}",
                    "neutral_english_context": f"neutral {index}",
                    "neutral_chinese_context": f"中性 {index}",
                },
            }
            for index in (1, 2)
        ]
        rows = perturbation._build_simulation_inputs(
            perturbations, item_by_id, translation_by_id, distractor_by_id,
            [{"model": "m1", "provider_profile": "p1"}],
            ("original",), "matched_per_candidate",
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(len([row for row in rows if row["variant"] == "original"]), 2)
        self.assertEqual(len([row for row in rows if row["variant"] == "neutral"]), 4)
        self.assertTrue(all(row["candidate_id"] for row in rows if row["variant"] == "neutral"))

    def test_multi_option_simulation_includes_target_and_foil(self):
        item_by_id = {"s1": {"source_id": "s1", "prompt_en": "Capital:", "answer_en": "Paris"}}
        translation_by_id = {"s1": {"parsed_response": {"prompt_zh": "首都：", "answer_zh": "巴黎"}}}
        distractor_by_id = {
            "d1": {"item_id": "d1", "source_id": "s1", "distractor_en": "Berlin", "distractor_zh": "柏林"},
            "d2": {"item_id": "d2", "source_id": "s1", "distractor_en": "Rome", "distractor_zh": "罗马"},
        }
        candidates = [{
            "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
            "candidate": {
                "english_context": "target", "chinese_context": "目标",
                "neutral_english_context": "neutral", "neutral_chinese_context": "中性",
            },
        }]
        rows = perturbation._build_simulation_inputs(
            candidates, item_by_id, translation_by_id, distractor_by_id,
            [{"model": "m", "provider_profile": "p"}],
            ("original",), "matched_per_candidate", "multi_option", 3,
        )
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["option_count"] == 3 for row in rows))
        self.assertTrue(all(row["target_choice"] != row["correct_choice"] for row in rows))
        self.assertTrue(all(set(row["options"].values()) in ({"Paris", "Berlin", "Rome"}, {"巴黎", "柏林", "罗马"}) for row in rows))

    def test_simulation_ids_are_stable_and_baselines_ignore_candidate(self):
        original_a = perturbation._simulation_id("s", "d", "m", "zh", "original", "c1")
        original_b = perturbation._simulation_id("s", "d", "m", "zh", "original", "c2")
        neutral = perturbation._simulation_id("s", "d", "m", "zh", "neutral", "c1")
        targeted_a = perturbation._simulation_id("s", "d", "m", "zh", "targeted", "c1")
        targeted_b = perturbation._simulation_id("s", "d", "m", "zh", "targeted", "c2")
        self.assertEqual(original_a, original_b)
        self.assertNotEqual(original_a, neutral)
        self.assertNotEqual(targeted_a, targeted_b)

    def test_detects_duplicate_baseline_inconsistency(self):
        common = {
            "source_id": "s", "distractor_id": "d", "model": "m", "language": "zh",
            "variant": "original", "terminal_status": "completed",
        }
        rows = [
            {**common, "simulation_id": "one", "candidate_id": "c1", "choice": "A", "correct": True},
            {**common, "simulation_id": "two", "candidate_id": "c2", "choice": "B", "correct": False},
        ]
        inconsistencies = perturbation.detect_baseline_inconsistencies(rows)
        self.assertEqual(len(inconsistencies), 1)
        self.assertEqual(inconsistencies[0]["candidate_ids"], ["c1", "c2"])

    def test_three_arm_analysis_does_not_count_non_target_foil_as_target_flip(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "multi",
                "simulation_models": ["m"],
                "simulation_variants": list(perturbation.SIMULATION_VARIANTS),
                "shared_simulation_variants": ["original"],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1"],
                },
            })
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
                "terminal_status": "completed",
            }])
            rows = []
            for language in perturbation.SIMULATION_LANGUAGES:
                for variant in perturbation.SIMULATION_VARIANTS:
                    is_nontarget_error = language == "zh" and variant == "targeted"
                    rows.append({
                        "simulation_id": f"{language}-{variant}",
                        "candidate_id": None if variant == "original" else "c1",
                        "source_id": "s1", "distractor_id": "d1", "model": "m",
                        "language": language, "variant": variant,
                        "terminal_status": "completed",
                        "choice": "C" if is_nontarget_error else "A",
                        "correct": not is_nontarget_error,
                        "distractor_hit": False,
                    })
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", rows)
            result = perturbation.analyze_three_arm(run_dir)
            candidate = result["candidate_results"][0]
            self.assertEqual(candidate["targeted_zh_wrong_models"], ["m"])
            self.assertEqual(candidate["targeted_zh_flip_models"], [])
            self.assertEqual(candidate["targeted_zh_nontarget_wrong_models"], ["m"])
            self.assertEqual(candidate["strict_targeted_zh_flip_models"], [])
            self.assertFalse(candidate["paths_not_taken_candidate"])

    def test_three_arm_analysis_preserves_model_specific_strict_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "model-specific",
                "simulation_models": ["signal-model", "other-model"],
                "simulation_variants": list(perturbation.SIMULATION_VARIANTS),
                "shared_simulation_variants": ["original"],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1"],
                },
            })
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
                "terminal_status": "completed",
            }])
            rows = []
            for model in ("signal-model", "other-model"):
                for language in perturbation.SIMULATION_LANGUAGES:
                    for variant in perturbation.SIMULATION_VARIANTS:
                        is_signal = (
                            model == "signal-model" and language == "zh" and variant == "targeted"
                        )
                        other_control_failure = (
                            model == "other-model" and language == "en" and variant == "targeted"
                        )
                        rows.append({
                            "simulation_id": f"{model}-{language}-{variant}",
                            "candidate_id": None if variant == "original" else "c1",
                            "source_id": "s1", "distractor_id": "d1", "model": model,
                            "language": language, "variant": variant,
                            "terminal_status": "completed",
                            "choice": "B" if is_signal or other_control_failure else "A",
                            "correct": not (is_signal or other_control_failure),
                            "distractor_hit": is_signal,
                        })
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", rows)
            result = perturbation.analyze_three_arm(run_dir)
            candidate = result["candidate_results"][0]
            self.assertEqual(candidate["strict_targeted_zh_flip_models"], ["signal-model"])
            self.assertTrue(candidate["paths_not_taken_candidate"])
            self.assertEqual(
                result["strict_signal_count_by_model"],
                {"signal-model": 1, "other-model": 0},
            )
            self.assertEqual(
                result["model_arm_metrics"]["signal-model"]["zh_targeted"],
                {
                    "call_count": 1,
                    "completed_count": 1,
                    "accuracy_completed": 0.0,
                    "distractor_hit_rate_completed": 1.0,
                },
            )

    def test_codex_proxy_allowlist_filters_model_accepted_perturbations(self):
        rows = [
            {
                "candidate_id": candidate_id,
                "terminal_status": "completed",
                "parsed_response": {
                    "decision": "reject" if candidate_id == "keep" else "accept",
                    "checks": {"valid": candidate_id != "keep"},
                },
            }
            for candidate_id in ("keep", "drop")
        ]
        manifest = {"codex_proxy_review": {"accepted_perturbation_ids": ["keep"]}}
        accepted = perturbation._accepted_perturbation_rows(rows, manifest)
        self.assertEqual([row["candidate_id"] for row in accepted], ["keep"])

    def test_multi_option_proxy_rerun_excludes_sources_without_two_accepted_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_run = project_root / "source"
            target_run = project_root / "target"
            source_run.mkdir()
            target_run.mkdir()
            perturbation.write_json(source_run / "run_manifest.json", {
                "run_id": "source", "seed": 7, "config_sha256": "old",
            })
            source_perturbations = [
                {"candidate_id": "eligible", "source_id": "s1", "distractor_id": "d1"},
                {"candidate_id": "excluded", "source_id": "s2", "distractor_id": "d3"},
            ]
            perturbation.write_jsonl(source_run / "perturbations.jsonl", source_perturbations)
            accepted_review = {
                "terminal_status": "completed",
                "parsed_response": {"decision": "accept", "checks": {"valid": True}},
            }
            perturbation.write_jsonl(source_run / "verified_distractors.jsonl", [
                {"item_id": "d1", "source_id": "s1", **accepted_review},
                {"item_id": "d2", "source_id": "s1", **accepted_review},
                {"item_id": "d3", "source_id": "s2", **accepted_review},
            ])
            for name in ("translations.jsonl", "translation_reviews.jsonl", "perturbation_generations.jsonl"):
                perturbation.write_jsonl(source_run / name, [])
            review_path = source_run / "codex_proxy_review_v1.json"
            perturbation.write_json(review_path, {
                "run_id": "source",
                "review_version": "v1",
                "reviewer_type": "codex_proxy",
                "not_human_gold": True,
                "perturbation_reviews": [
                    {"item_id": "eligible", "decision": "accept"},
                    {"item_id": "excluded", "decision": "accept"},
                ],
            })
            config = {
                "config_version": "multi-v1",
                "model_roles": {"simulation": {
                    "endpoint": "multi_option", "max_options": 3,
                    "neutral_context_mode": "matched_per_candidate",
                    "models": [{"model": "m"}],
                }},
            }
            manifest = perturbation.prepare_proxy_rerun(
                config, project_root, source_run, target_run, "target", review_path,
            )
            self.assertEqual(
                manifest["codex_proxy_review"]["accepted_perturbation_ids"], ["eligible"]
            )
            self.assertEqual(manifest["endpoint_excluded_candidate_ids"], ["excluded"])
            self.assertEqual(
                manifest["endpoint_exclusions"][0]["reason"],
                "fewer_than_two_auto_accepted_distractors",
            )

    def test_frozen_proxy_rerun_keeps_generated_distractor_foils(self):
        accepted = {
            "terminal_status": "completed",
            "parsed_response": {"decision": "accept", "checks": {"valid": True}},
        }
        rows = [
            {"item_id": "source_d1", "source_id": "s1", **accepted},
            {
                "item_id": "generated_d2", "source_id": "s1",
                "distractor_source": "generated_replacement", **accepted,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified_distractors.jsonl"
            perturbation.write_jsonl(path, rows)
            frozen_rows = perturbation.read_jsonl(path)
        self.assertEqual(
            {row["item_id"] for row in frozen_rows},
            {"source_d1", "generated_d2"},
        )

    def test_simulation_rerun_reuses_frozen_candidates_without_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_run = project_root / "source"
            target_run = project_root / "target"
            source_run.mkdir()
            target_run.mkdir()
            source_manifest = {
                "run_id": "source",
                "config_sha256": "old",
                "records": [{"source_id": "s1"}],
                "simulation_endpoint": "multi_option",
                "simulation_max_options": 3,
                "neutral_context_mode": "matched_per_candidate",
                "shared_simulation_variants": ["original"],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1"],
                },
            }
            perturbation.write_json(source_run / "run_manifest.json", source_manifest)
            for name in (
                "translations.jsonl", "translation_reviews.jsonl",
                "verified_distractors.jsonl", "perturbation_generations.jsonl",
            ):
                perturbation.write_jsonl(source_run / name, [{"source_id": "s1"}])
            perturbation.write_jsonl(source_run / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "terminal_status": "completed",
            }])
            perturbation.write_jsonl(source_run / "simulation_results.jsonl", [{
                "simulation_id": "old", "model": "old-model",
            }])
            config = {
                "config_version": "local-v1",
                "model_roles": {"simulation": {
                    "endpoint": "multi_option",
                    "max_options": 3,
                    "neutral_context_mode": "matched_per_candidate",
                    "models": [
                        {"model": "gemma3:12b", "provider_profile": "ollama_local"},
                        {"model": "llama3.1:8b", "provider_profile": "ollama_local"},
                    ],
                }},
            }
            manifest = perturbation.prepare_simulation_rerun(
                config, project_root, source_run, target_run, "target"
            )
            self.assertEqual(manifest["resumed_from_run_id"], "source")
            self.assertEqual(manifest["simulation_models"], ["gemma3:12b", "llama3.1:8b"])
            self.assertEqual(manifest["simulation_rerun"]["frozen_candidate_ids"], ["c1"])
            self.assertFalse((target_run / "simulation_results.jsonl").exists())
            self.assertEqual(
                perturbation.read_jsonl(target_run / "perturbations.jsonl")[0]["candidate_id"],
                "c1",
            )
            updated_config = {
                **config,
                "config_version": "local-v2",
                "model_roles": {"simulation": {
                    **config["model_roles"]["simulation"],
                    "max_output_tokens": 64,
                }},
            }
            resumed = perturbation.prepare_simulation_rerun(
                updated_config, project_root, source_run, target_run, "target"
            )
            self.assertEqual(resumed["config_version"], "local-v2")
            self.assertEqual(
                resumed["config_amendments"][-1]["reason"],
                "update Simulation request settings; completed checkpoints retained",
            )

    def test_perturbation_review_requires_salience_and_no_prompt_duplication(self):
        checks = {key: True for key in perturbation.PERTURBATION_CHECKS}
        perturbation.validate_decision({"decision": "accept", "issues": [], "checks": checks}, perturbation.PERTURBATION_CHECKS)
        checks.pop("target_distractor_salience")
        with self.assertRaisesRegex(ValueError, "invalid_checks"):
            perturbation.validate_decision({"decision": "accept", "issues": [], "checks": checks}, perturbation.PERTURBATION_CHECKS)

    def test_summary_blocks_simulation_before_codex_proxy_review(self):
        summary = perturbation.summarize_run(
            {
                "run_id": "screen", "selected_count": 0, "minimum_translation_acceptance": 0.5,
                "currency_cost_gate": 1, "simulation_requires_codex_proxy_review": True,
                "simulation_models": ["m"], "simulation_languages": ["en", "zh"],
                "simulation_variants": ["original", "neutral", "targeted"],
            },
            [], [], [], [], [], [],
        )
        self.assertIn("codex_proxy_review_required_before_simulation", summary["gate_reasons"])


if __name__ == "__main__":
    unittest.main()
