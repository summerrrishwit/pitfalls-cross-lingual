import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    PROJECT_ROOT / "scripts" / "run_full_public_benchmark_semantic_review.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_full_public_benchmark_semantic_review", RUNNER_PATH
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER_SPEC.loader.exec_module(runner)

MATERIALIZER_TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_materialize_full_public_benchmark_semantic_closure.py"
)
MATERIALIZER_TEST_SPEC = importlib.util.spec_from_file_location(
    "semantic_closure_fixture_helpers", MATERIALIZER_TEST_PATH
)
fixture_helpers = importlib.util.module_from_spec(MATERIALIZER_TEST_SPEC)
assert MATERIALIZER_TEST_SPEC and MATERIALIZER_TEST_SPEC.loader
MATERIALIZER_TEST_SPEC.loader.exec_module(fixture_helpers)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeRouter:
    def __init__(
        self,
        config=None,
        env_path=None,
        event_log=None,
        *,
        fail_multi=False,
        fail_pair_ids=None,
        response_model=runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
    ):
        self.config = config
        self.env_path = env_path
        self.event_log = event_log
        self.fail_multi = fail_multi
        self.fail_pair_ids = set(fail_pair_ids or [])
        self.response_model = response_model
        self.calls = []

    @staticmethod
    def _failure():
        return SimpleNamespace(
            terminal_status="failed",
            parsed_response=None,
            raw_response="unsafe https://secret.example.invalid/key",
            response_model=None,
            usage={},
            latency_ms=2,
            attempt_count=1,
            attempts=[
                {
                    "attempt": 1,
                    "status": "failed",
                    "latency_ms": 2,
                    "error_type": "TimeoutError",
                    "unsafe_message": "https://secret.example.invalid/key",
                }
            ],
        )

    @staticmethod
    def _verdict(pair):
        alias_evidence = "answer_alias_exact" in pair["match_types"]
        return {
            "pair_id": pair["pair_id"],
            "relationship": "same_fact",
            "alias_valid": True if alias_evidence else None,
            "same_answer_entity": True,
            "same_subject_entity": True,
            "relation_semantics_same": True,
            "temporal_scope_compatible": True,
            "answer_compatible": True,
            "semantic_duplicate": True,
            "rationale": "Both fixture endpoints express the same proposition.",
            "confidence": "high",
        }

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = runner._prompt_payload(prompt)
        pairs = payload["pairs"]
        pair_ids = [pair["pair_id"] for pair in pairs]
        self.calls.append((stage, tuple(pair_ids), item_id, dict(model_spec), payload))
        if self.fail_multi and len(pair_ids) > 1:
            return self._failure()
        if set(pair_ids).intersection(self.fail_pair_ids):
            return self._failure()
        parsed = {
            "schema_version": runner.BATCH_RESPONSE_SCHEMA,
            "records": [self._verdict(pair) for pair in pairs],
        }
        validator(parsed)
        raw = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        return SimpleNamespace(
            terminal_status="completed",
            parsed_response=parsed,
            raw_response=raw,
            response_model=self.response_model,
            usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            latency_ms=3,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "ok", "latency_ms": 3}],
        )


class NoCallRouter:
    def request_json(self, *args, **kwargs):
        raise AssertionError("resume should not call the model for terminal rows")


class FullSemanticReviewRunnerTests(unittest.TestCase):
    @staticmethod
    def now():
        return "2026-09-23T00:00:00+00:00"

    @staticmethod
    def config(max_workers=2):
        return {
            "config_version": "semantic-review-fixture-v1",
            "provider_profiles": {
                "openai": {
                    "protocol": "openai",
                    "base_url": "https://semantic-review-fixture.invalid/v1",
                }
            },
            "execution": {"max_workers": max_workers, "max_retries": 0},
            "model_roles": {
                "full_pool_semantic_closure_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "reasoning_effort": "low",
                        "max_output_tokens": 1024,
                        "max_retries": 0,
                    }
                }
            },
        }

    @staticmethod
    def spec():
        return {
            "provider_profile": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "low",
            "max_output_tokens": 1024,
            "max_retries": 0,
        }

    def make_candidates(self, root):
        helper = fixture_helpers.FullSemanticClosureMaterializerTests()
        fixture = helper.make_inputs(root)
        candidate_dir = Path(root) / "candidates"
        helper.run_materializer(fixture, candidate_dir)
        return {
            **fixture,
            "candidate_dir": candidate_dir,
            "candidate_manifest": candidate_dir
            / "semantic_closure_candidate_manifest.json",
            "candidates": candidate_dir / "semantic_closure_candidate_pairs.jsonl",
            "template": candidate_dir
            / "semantic_closure_adjudication_template.jsonl",
        }

    def run_review(self, fixture, output, router, **overrides):
        values = {
            "candidate_manifest_path": fixture["candidate_manifest"],
            "candidate_pairs_path": fixture["candidates"],
            "adjudication_template_path": fixture["template"],
            "output_dir": output,
            "config": self.config(),
            "env_path": Path(output).parent / ".env",
            "reviewer_spec": self.spec(),
            "expected_response_model": runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
            "batch_size": 2,
            "max_workers": 1,
            "router": router,
            "now_fn": self.now,
        }
        values.update(overrides)
        return runner.run_review(**values)

    def test_behavior_blind_batch_review_writes_strict_hash_bound_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            _, _, _, units = runner.load_review_units(
                candidate_manifest_path=fixture["candidate_manifest"], limit=2
            )
            payload = runner._prompt_payload(runner.build_review_prompt(units))
            serialized = json.dumps(payload, sort_keys=True)
            for forbidden in runner.FORBIDDEN_PROMPT_KEYS:
                self.assertNotIn(f'"{forbidden}"', serialized)

            fake = FakeRouter()
            result = self.run_review(
                fixture, root / "run", fake, limit=2, batch_size=2
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["selected_pair_count"], 2)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(len(fake.calls[0][1]), 2)

            checkpoints = read_jsonl(result["checkpoint_path"])
            decisions = read_jsonl(result["adjudications_path"])
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(len(decisions), 2)
            candidates = {
                row["pair_id"]: row for row in read_jsonl(fixture["candidates"])
            }
            templates = {
                row["pair_id"]: row for row in read_jsonl(fixture["template"])
            }
            for checkpoint, decision in zip(checkpoints, decisions):
                pair_id = checkpoint["pair_id"]
                self.assertTrue(checkpoint["behavior_blind"])
                self.assertFalse(checkpoint["human_gold"])
                self.assertEqual(checkpoint["hf_model_execution_count"], 0)
                self.assertEqual(checkpoint["behavior_execution_count"], 0)
                self.assertEqual(
                    decision["candidate_row_sha256"],
                    runner.sha256_value(candidates[pair_id]),
                )
                self.assertEqual(
                    decision["adjudication_template_row_sha256"],
                    runner.sha256_value(templates[pair_id]),
                )
                self.assertEqual(decision["relationship"], "same_fact")
                self.assertTrue(decision["semantic_duplicate"])
                self.assertFalse(decision["human_gold"])
                self.assertEqual(
                    decision["reviewer_type"], "independent_model_proxy"
                )
                for call in checkpoint["model_calls"]:
                    self.assertNotIn("raw_response", call)
                    self.assertNotIn("https://secret", json.dumps(call))

            manifest = runner.read_json(Path(result["manifest_path"]))
            self.assertTrue(manifest["selection_is_full_candidate_set"])
            self.assertTrue(
                manifest["candidate_adjudication_complete_for_selected_scope"]
            )
            self.assertTrue(manifest["candidate_adjudication_complete_for_full_set"])
            self.assertFalse(manifest["semantic_closure_complete"])
            self.assertEqual(
                manifest["run_contract"]["reviewer_spec"]["model"], "gpt-5.5"
            )
            self.assertEqual(
                manifest["run_contract"]["expected_response_model"],
                "gpt-5.5-2026-04-24",
            )
            self.assertEqual(
                manifest["evidence_boundary"]["hf_model_execution_count"], 0
            )

    def test_failed_multi_item_batch_recursively_splits_and_checkpoints_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            fake = FakeRouter(fail_multi=True)
            result = self.run_review(
                fixture, root / "run", fake, limit=2, batch_size=2
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual([len(call[1]) for call in fake.calls], [2, 1, 1])
            rows = read_jsonl(result["checkpoint_path"])
            self.assertEqual([row["input_index"] for row in rows], [0, 1])
            self.assertTrue(all(len(row["model_calls"]) == 2 for row in rows))
            self.assertTrue(
                all(row["model_calls"][-1]["batch_item_count"] == 1 for row in rows)
            )

    def test_resume_retries_only_failed_pair_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            pairs = read_jsonl(fixture["candidates"])
            first_id, second_id = pairs[0]["pair_id"], pairs[1]["pair_id"]
            first_router = FakeRouter(fail_pair_ids={second_id})
            first = self.run_review(
                fixture,
                root / "run",
                first_router,
                limit=2,
                batch_size=1,
            )
            self.assertEqual(first["status"], "completed_with_failures")
            self.assertEqual(
                first["terminal_status_counts"], {"completed": 1, "failed": 1}
            )

            second_router = FakeRouter()
            second = self.run_review(
                fixture,
                root / "run",
                second_router,
                limit=2,
                batch_size=1,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(second["status"], "completed")
            self.assertEqual([call[1] for call in second_router.calls], [(second_id,)])
            checkpoints = {
                row["pair_id"]: row for row in read_jsonl(second["checkpoint_path"])
            }
            self.assertEqual(checkpoints[first_id]["retry_history"], [])
            self.assertEqual(len(checkpoints[second_id]["retry_history"]), 1)
            self.assertEqual(
                checkpoints[second_id]["retry_history"][0]["failure_stage"],
                "reviewer",
            )

    def test_expected_response_model_mismatch_is_terminal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            result = self.run_review(
                fixture,
                root / "run",
                FakeRouter(response_model="unexpected-model"),
                limit=1,
                batch_size=1,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            checkpoint = read_jsonl(result["checkpoint_path"])[0]
            self.assertEqual(checkpoint["terminal_status"], "failed")
            self.assertEqual(
                checkpoint["model_calls"][0]["response_model_identity_status"],
                "mismatch",
            )
            self.assertEqual(
                checkpoint["model_calls"][0]["post_validation_error_type"],
                "ResponseModelMismatch",
            )

    def test_missing_response_model_fails_whole_batch_without_recursive_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            fake = FakeRouter(response_model=None)
            result = self.run_review(
                fixture, root / "run", fake, limit=2, batch_size=2
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(len(fake.calls), 1)
            rows = read_jsonl(result["checkpoint_path"])
            self.assertTrue(all(row["terminal_status"] == "failed" for row in rows))
            self.assertTrue(
                all(
                    row["model_calls"][-1]["response_model_identity_status"]
                    == "not_observed"
                    for row in rows
                )
            )

    def test_route_fingerprint_is_redacted_and_env_drift_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            output = root / "run"
            env_path = root / ".env"
            config = self.config()
            config["provider_profiles"]["openai"]["base_url_env"] = (
                "OPENAI_BASE_URL"
            )
            env_path.write_text(
                "OPENAI_BASE_URL=https://semantic-route-one.invalid/v1\n",
                encoding="utf-8",
            )
            result = self.run_review(
                fixture,
                output,
                FakeRouter(),
                limit=1,
                config=config,
                env_path=env_path,
            )
            manifest = runner.read_json(Path(result["manifest_path"]))
            route = manifest["run_contract"]["route_identity"]
            serialized = json.dumps(route, sort_keys=True)
            self.assertNotIn("semantic-route-one", serialized)
            self.assertFalse(route["credentials_or_endpoints_included"])
            self.assertEqual(len(manifest["run_contract"]["source_fingerprints"]), 3)

            env_path.write_text(
                "OPENAI_BASE_URL=https://semantic-route-two.invalid/v1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Resume run contract mismatch"):
                self.run_review(
                    fixture,
                    output,
                    NoCallRouter(),
                    limit=1,
                    config=config,
                    env_path=env_path,
                    resume=True,
                )

    def test_checkpoint_writes_only_from_coordinator_with_concurrent_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            thread_names = []
            original = runner.write_jsonl

            def tracked_write(path, rows):
                thread_names.append(threading.current_thread().name)
                return original(path, rows)

            with mock.patch.object(runner, "write_jsonl", side_effect=tracked_write):
                result = self.run_review(
                    fixture,
                    root / "run",
                    FakeRouter(),
                    limit=2,
                    batch_size=1,
                    max_workers=2,
                )
            self.assertEqual(result["status"], "completed")
            self.assertTrue(thread_names)
            self.assertTrue(all(name == "MainThread" for name in thread_names))

    def test_rejects_stale_candidate_and_non_gpt_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            with fixture["candidates"].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "candidate_pairs SHA-256 mismatch"):
                runner.load_review_units(
                    candidate_manifest_path=fixture["candidate_manifest"]
                )
        with self.assertRaisesRegex(ValueError, "must use gpt-5.5"):
            runner.resolve_reviewer_spec(self.config(), model="other-model")

    def test_protocol_identity_cannot_be_omitted_or_self_consistently_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            with self.assertRaisesRegex(ValueError, "expected_response_model is required"):
                self.run_review(
                    fixture,
                    root / "missing-identity",
                    FakeRouter(),
                    limit=1,
                    expected_response_model=None,
                )
            with self.assertRaisesRegex(ValueError, "protocol identity"):
                self.run_review(
                    fixture,
                    root / "alternate-identity",
                    FakeRouter(response_model="alternate-response-model"),
                    limit=1,
                    expected_response_model="alternate-response-model",
                )

    def test_resume_rejects_tampered_completed_identity_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_candidates(root)
            output = root / "run"
            result = self.run_review(
                fixture, output, FakeRouter(), limit=1, batch_size=1
            )
            checkpoint = read_jsonl(result["checkpoint_path"])
            checkpoint[0]["model_calls"][-1]["response_model_identity_status"] = (
                "not_checked"
            )
            runner.write_jsonl(Path(result["checkpoint_path"]), checkpoint)
            with self.assertRaisesRegex(ValueError, "response model is not verified"):
                self.run_review(
                    fixture,
                    output,
                    NoCallRouter(),
                    limit=1,
                    batch_size=1,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
