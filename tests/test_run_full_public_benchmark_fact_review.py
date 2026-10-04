import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_full_public_benchmark_fact_review.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_full_public_benchmark_fact_review", RUNNER_PATH
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(runner)

REVIEW_TEST_PATH = PROJECT_ROOT / "tests" / "test_review_public_benchmark_bundle.py"
REVIEW_TEST_SPEC = importlib.util.spec_from_file_location(
    "review_bundle_fixture_helpers_for_fact_runner", REVIEW_TEST_PATH
)
review_test_helpers = importlib.util.module_from_spec(REVIEW_TEST_SPEC)
assert REVIEW_TEST_SPEC and REVIEW_TEST_SPEC.loader
REVIEW_TEST_SPEC.loader.exec_module(review_test_helpers)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeRouter:
    def __init__(
        self,
        *,
        fail_multi_generator=False,
        fail_reviewer_ids=None,
        response_models=None,
    ):
        self.fail_multi_generator = fail_multi_generator
        self.fail_reviewer_ids = set(fail_reviewer_ids or [])
        self.response_models = dict(response_models or {})
        self.calls = []

    @staticmethod
    def _evidence_rows(stage, payload):
        if stage == "fact_review_generator":
            return payload["items"]
        return [entry["evidence"] for entry in payload["items"]]

    @staticmethod
    def _verdict(evidence):
        return {
            "base_fact_id": evidence["base_fact_id"],
            "overall_decision": "accept",
            "fact_review": {
                "decision": "accept",
                "reason": "The fixture source supports the proposition.",
            },
            "relation_review": {
                "decision": "accept",
                "reason": "The relation matches the fixture question and answer.",
            },
            "member_reviews": [
                {
                    "candidate_id": member["candidate_id"],
                    "decision": "accept",
                    "reason": "The member matches the source snapshot.",
                }
                for member in evidence["members"]
            ],
            "alias_reviews": [
                {
                    "alias_id": target["alias_id"],
                    "decision": "accept",
                    "reason": "The alias denotes the same answer.",
                }
                for target in evidence["alias_targets"]
            ],
            "distractor_reviews": [
                {
                    "distractor_id": target["distractor_id"],
                    "decision": "accept",
                    "reason": "The distractor is not a valid answer.",
                }
                for target in evidence["distractor_targets"]
            ],
            "notes": "Fixture proxy review completed from source evidence only.",
        }

    @staticmethod
    def _failure():
        return SimpleNamespace(
            terminal_status="failed",
            parsed_response=None,
            raw_response="invalid batch response",
            response_model=None,
            usage={},
            latency_ms=3,
            attempt_count=2,
            attempts=[
                {
                    "attempt": 1,
                    "status": "failed",
                    "latency_ms": 1,
                    "error_type": "ValueError",
                    "unsafe_message": "https://secret.example.invalid/key",
                },
                {
                    "attempt": 2,
                    "status": "failed",
                    "latency_ms": 2,
                    "error_type": "ValueError",
                },
            ],
        )

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = runner._prompt_payload(prompt)
        evidence = self._evidence_rows(stage, payload)
        ids = [entry["base_fact_id"] for entry in evidence]
        self.calls.append((stage, ids, item_id, dict(model_spec)))
        if stage == "fact_review_generator" and self.fail_multi_generator and len(ids) > 1:
            return self._failure()
        if stage == "fact_review_reviewer" and set(ids).intersection(
            self.fail_reviewer_ids
        ):
            return self._failure()
        response = {
            "schema_version": runner.BATCH_RESPONSE_SCHEMA_VERSION,
            "items": [self._verdict(entry) for entry in evidence],
        }
        validator(response)
        raw = json.dumps(response, ensure_ascii=False, sort_keys=True)
        return SimpleNamespace(
            terminal_status="completed",
            parsed_response=response,
            raw_response=raw,
            response_model=self.response_models.get(
                stage, f"{model_spec['model']}-resolved"
            ),
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            latency_ms=5,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "ok", "latency_ms": 5}],
        )


class FullPublicBenchmarkFactReviewRunnerTests(unittest.TestCase):
    def make_export(self, root):
        helper = review_test_helpers.PublicBenchmarkReviewToolTests()
        source, fact_a, fact_b = helper.make_source(root)
        scope = runner.review_tool.create_scope(
            output_dir=root / "scope",
            artifact_paths=runner.review_tool.source_paths(source),
            base_fact_ids=[fact_a, fact_b],
            review_sample_path=None,
        )
        export = runner.review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "export",
        )
        return source, fact_a, fact_b, Path(scope["manifest_path"]), Path(
            export["manifest_path"]
        )

    @staticmethod
    def config():
        return {
            "config_version": "fact-review-test-v1",
            "provider_profiles": {"generator": {}, "reviewer": {}},
            "execution": {"max_workers": 1, "max_retries": 0},
        }

    @staticmethod
    def specs():
        return (
            {
                "provider_profile": "generator",
                "model": "proposal-model",
                "expected_response_model": "proposal-model-resolved",
                "max_output_tokens": 1024,
                "max_retries": 0,
            },
            {
                "provider_profile": "reviewer",
                "model": "review-model",
                "expected_response_model": "review-model-resolved",
                "max_output_tokens": 1024,
                "max_retries": 0,
            },
        )

    @staticmethod
    def now():
        return "2026-09-23T00:00:00+00:00"

    def test_behavior_blind_projection_excludes_behavior_and_split_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, export_manifest = self.make_export(root)
            _, _, _, units = runner.load_review_units(
                export_manifest_path=export_manifest, limit=1
            )
            projection_text = json.dumps(units[0].projection, sort_keys=True)
            self.assertNotIn("behavior_row", projection_text)
            self.assertNotIn("split_assignment", projection_text)
            self.assertNotIn("experiment_status", projection_text)
            self.assertNotIn("admission", projection_text)
            self.assertIn("source_snapshot", projection_text)
            self.assertIn("extracted_triple", projection_text)
            prompt_payload = runner._prompt_payload(runner.build_generator_prompt(units))
            self.assertEqual(
                prompt_payload["items"][0]["base_fact_id"], units[0].base_fact_id
            )

    def test_run_writes_strict_apply_compatible_decisions_and_redacted_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, scope_manifest, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            fake = FakeRouter()
            result = runner.run_review(
                export_manifest_path=export_manifest,
                review_items_path=root / "export" / "review_items.jsonl",
                decision_template_path=root / "export" / "review_decisions_template.jsonl",
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                batch_size=2,
                max_workers=1,
                router=fake,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["terminal_status_counts"], {"completed": 2})
            self.assertEqual([call[0] for call in fake.calls], [
                "fact_review_generator",
                "fact_review_reviewer",
            ])
            checkpoints = read_jsonl(result["checkpoint_path"])
            decisions = read_jsonl(result["decisions_path"])
            self.assertEqual(
                [row["input_index"] for row in checkpoints], [0, 1]
            )
            self.assertEqual(len(decisions), 2)
            for checkpoint, decision in zip(checkpoints, decisions):
                self.assertTrue(checkpoint["behavior_blind"])
                self.assertFalse(checkpoint["human_gold"])
                self.assertEqual(
                    set(decision), runner.review_tool.DECISION_ALLOWED_FIELDS
                )
                self.assertFalse(decision["human_gold"])
                self.assertEqual(
                    decision["review_provenance"]["reviewer_type"], "codex_proxy"
                )
                for call in checkpoint["model_calls"]:
                    self.assertEqual(len(call["prompt_sha256"]), 64)
                    self.assertEqual(len(call["request_sha256"]), 64)
                    self.assertEqual(len(call["response_sha256"]), 64)
                    self.assertEqual(len(call["raw_response_sha256"]), 64)
                    self.assertNotIn("raw_response", call)
                    serialized_call = json.dumps(call).lower()
                    self.assertNotIn("http://", serialized_call)
                    self.assertNotIn("https://", serialized_call)
                    self.assertNotIn("api_key", serialized_call)
                    self.assertNotIn("base_url", serialized_call)
                    self.assertNotIn("unsafe_message", json.dumps(call))

            applied = runner.review_tool.apply_decisions(
                scope_manifest_path=scope_manifest,
                export_manifest_path=export_manifest,
                decisions_path=Path(result["decisions_path"]),
                output_dir=root / "applied",
            )
            self.assertEqual(applied["outcome_counts"], {"accept": 2})

    def test_batch_failure_splits_to_singletons_and_keeps_deterministic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, fact_b, _, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            fake = FakeRouter(fail_multi_generator=True)
            result = runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                batch_size=2,
                max_workers=1,
                router=fake,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                [(call[0], len(call[1])) for call in fake.calls],
                [
                    ("fact_review_generator", 2),
                    ("fact_review_generator", 1),
                    ("fact_review_reviewer", 1),
                    ("fact_review_generator", 1),
                    ("fact_review_reviewer", 1),
                ],
            )
            checkpoints = read_jsonl(result["checkpoint_path"])
            self.assertEqual(
                [row["base_fact_id"] for row in checkpoints], [fact_a, fact_b]
            )
            self.assertTrue(
                all(len(row["model_calls"]) == 3 for row in checkpoints)
            )
            serialized = Path(result["checkpoint_path"]).read_text(encoding="utf-8")
            self.assertNotIn("secret.example.invalid", serialized)

    def test_resume_retries_only_failed_items_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, fact_b, _, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            first_router = FakeRouter(fail_reviewer_ids={fact_b})
            first = runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                batch_size=1,
                max_workers=1,
                router=first_router,
                now_fn=self.now,
            )
            self.assertEqual(first["status"], "completed_with_failures")
            self.assertEqual(
                first["terminal_status_counts"], {"completed": 1, "failed": 1}
            )

            second_router = FakeRouter()
            second = runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                batch_size=1,
                max_workers=1,
                retry_failed=True,
                router=second_router,
                now_fn=self.now,
            )
            self.assertEqual(second["status"], "completed")
            self.assertEqual(
                [call[1] for call in second_router.calls], [[fact_b], [fact_b]]
            )
            checkpoints = {
                row["base_fact_id"]: row for row in read_jsonl(second["checkpoint_path"])
            }
            self.assertEqual(checkpoints[fact_a]["retry_history"], [])
            self.assertEqual(len(checkpoints[fact_b]["retry_history"]), 1)
            self.assertEqual(
                checkpoints[fact_b]["retry_history"][0]["failure_stage"], "reviewer"
            )
            decisions = read_jsonl(second["decisions_path"])
            self.assertEqual(
                [row["base_fact_id"] for row in decisions], [fact_a, fact_b]
            )

    def test_interrupted_run_recovers_main_plus_truncated_append_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, fact_b, _, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            first_router = FakeRouter()
            original_process_batch = runner._process_batch

            def interrupt_second_batch(**kwargs):
                if kwargs["units"][0].base_fact_id == fact_b:
                    time.sleep(0.05)
                    raise RuntimeError("simulated interruption")
                return original_process_batch(**kwargs)

            with mock.patch.object(
                runner, "_process_batch", side_effect=interrupt_second_batch
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner.run_review(
                        export_manifest_path=export_manifest,
                        output_dir=root / "run",
                        config=self.config(),
                        env_path=root / ".env",
                        generator_spec=generator_spec,
                        reviewer_spec=reviewer_spec,
                        batch_size=1,
                        checkpoint_compact_every=128,
                        max_workers=1,
                        router=first_router,
                        now_fn=self.now,
                    )

            journal_path = root / "run" / "fact_review_checkpoint.journal.jsonl"
            self.assertTrue(journal_path.is_file())
            journal_rows, needs_repair = runner._read_journal(journal_path)
            self.assertFalse(needs_repair)
            self.assertEqual([row["base_fact_id"] for row in journal_rows], [fact_a])
            with journal_path.open("ab") as handle:
                handle.write(b'{"interrupted":')

            second_router = FakeRouter()
            resumed = runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                batch_size=1,
                checkpoint_compact_every=128,
                max_workers=1,
                router=second_router,
                now_fn=self.now,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(
                [call[1] for call in second_router.calls], [[fact_b], [fact_b]]
            )
            self.assertFalse(journal_path.exists())
            self.assertEqual(
                [row["base_fact_id"] for row in read_jsonl(resumed["checkpoint_path"])],
                [fact_a, fact_b],
            )

    def test_explicit_artifacts_must_match_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, export_manifest = self.make_export(root)
            wrong = root / "copied_items.jsonl"
            wrong.write_bytes((root / "export" / "review_items.jsonl").read_bytes())
            with self.assertRaisesRegex(ValueError, "differs from export-manifest binding"):
                runner.load_review_units(
                    export_manifest_path=export_manifest,
                    review_items_path=wrong,
                    limit=1,
                )

    def test_model_specs_can_come_from_config_and_cli_overrides(self):
        config = {
            "fact_review": {
                "generator": {
                    "provider_profile": "old-generator",
                    "model": "old-model",
                    "expected_response_model": "proposal-model-resolved",
                },
                "reviewer": {
                    "provider_profile": "reviewer",
                    "model": "review-model",
                    "expected_response_model": "review-model-resolved",
                },
            }
        }
        generator, reviewer = runner.resolve_model_specs(
            config,
            generator_provider_profile="generator",
            generator_model="proposal-model",
            reviewer_reasoning_effort="low",
            max_retries=2,
        )
        self.assertEqual(generator["provider_profile"], "generator")
        self.assertEqual(generator["model"], "proposal-model")
        self.assertEqual(reviewer["reasoning_effort"], "low")
        self.assertEqual(generator["max_retries"], 2)
        self.assertEqual(reviewer["max_retries"], 2)
        self.assertTrue(generator["json_mode"])

    def test_response_model_identity_mismatch_fails_closed_without_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            fake = FakeRouter(
                response_models={"fact_review_generator": "unexpected-model"}
            )
            result = runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                batch_size=2,
                max_workers=1,
                router=fake,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["terminal_status_counts"], {"failed": 2})
            self.assertEqual(len(fake.calls), 1)
            checkpoints = read_jsonl(result["checkpoint_path"])
            for row in checkpoints:
                call = row["model_calls"][0]
                self.assertEqual(
                    call["response_model_identity_status"], "mismatch"
                )
                self.assertEqual(
                    call["post_validation_error_type"],
                    "ResponseModelIdentityError",
                )

    def test_resume_rejects_expanding_selected_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                limit=1,
                batch_size=1,
                max_workers=1,
                router=FakeRouter(),
                now_fn=self.now,
            )
            with self.assertRaisesRegex(ValueError, "resume run contract mismatch"):
                runner.run_review(
                    export_manifest_path=export_manifest,
                    output_dir=root / "run",
                    config=self.config(),
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    limit=2,
                    batch_size=1,
                    max_workers=1,
                    router=FakeRouter(),
                    now_fn=self.now,
                )

    def test_resume_rejects_tampered_completed_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, export_manifest = self.make_export(root)
            generator_spec, reviewer_spec = self.specs()
            result = runner.run_review(
                export_manifest_path=export_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                limit=1,
                batch_size=1,
                max_workers=1,
                router=FakeRouter(),
                now_fn=self.now,
            )
            rows = read_jsonl(result["checkpoint_path"])
            rows[0]["strict_decision"]["human_gold"] = True
            runner.write_jsonl(Path(result["checkpoint_path"]), rows)
            with self.assertRaisesRegex(ValueError, "human_gold=false"):
                runner.run_review(
                    export_manifest_path=export_manifest,
                    output_dir=root / "run",
                    config=self.config(),
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    limit=1,
                    batch_size=1,
                    max_workers=1,
                    router=FakeRouter(),
                    now_fn=self.now,
                )


if __name__ == "__main__":
    unittest.main()
