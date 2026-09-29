import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "run_full_public_benchmark_fact_review_supplement.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_full_public_benchmark_fact_review_supplement", SCRIPT_PATH
)
supplement = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(supplement)

REVIEW_TEST_PATH = PROJECT_ROOT / "tests" / "test_review_public_benchmark_bundle.py"
REVIEW_TEST_SPEC = importlib.util.spec_from_file_location(
    "review_bundle_fixture_helpers_for_supplement", REVIEW_TEST_PATH
)
review_helpers = importlib.util.module_from_spec(REVIEW_TEST_SPEC)
assert REVIEW_TEST_SPEC and REVIEW_TEST_SPEC.loader
REVIEW_TEST_SPEC.loader.exec_module(review_helpers)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ParentFailureRouter:
    def request_json(self, stage, item_id, model_spec, prompt, validator):
        del stage, item_id, prompt, validator
        return SimpleNamespace(
            terminal_status="failed",
            parsed_response=None,
            raw_response="invalid response at https://secret.example.invalid/key",
            response_model=model_spec["expected_response_model"],
            usage={},
            latency_ms=2,
            attempt_count=1,
            attempts=[
                {
                    "attempt": 1,
                    "status": "failed",
                    "latency_ms": 2,
                    "error_type": "ValueError",
                    "unsafe_message": "https://secret.example.invalid/key",
                }
            ],
        )


class SupplementRouter:
    def __init__(
        self,
        *,
        invalid_accept=False,
        response_models=None,
        terminal_failure_error_types=None,
    ):
        self.invalid_accept = invalid_accept
        self.response_models = dict(response_models or {})
        self.terminal_failure_error_types = dict(terminal_failure_error_types or {})
        self.calls = []

    @staticmethod
    def verdict(evidence):
        return {
            "schema_version": supplement.RESPONSE_SCHEMA_VERSION,
            "base_fact_id": evidence["base_fact_id"],
            "overall_decision": "accept",
            "fact_review": {
                "decision": "accept",
                "reason": "The source snapshot supports the fact.",
            },
            "relation_review": {
                "decision": "accept",
                "reason": "The relation matches the source question.",
            },
            "member_reviews": [
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "accept",
                    "reason": "The member matches the source snapshot.",
                }
                for row in evidence["members"]
            ],
            "alias_reviews": [
                {
                    "alias_id": row["alias_id"],
                    "decision": "accept",
                    "reason": "The alias denotes the canonical answer.",
                }
                for row in evidence["alias_targets"]
            ],
            "distractor_reviews": [
                {
                    "distractor_id": row["distractor_id"],
                    "decision": "accept",
                    "reason": "The distractor is not valid under the same scope.",
                }
                for row in evidence["distractor_targets"]
            ],
            "notes": "Independent source-only review completed.",
        }

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = supplement._prompt_payload(prompt)
        evidence = payload["evidence"]
        strict = model_spec.get("strict_json_schema")
        self.calls.append(
            {
                "stage": stage,
                "item_id": item_id,
                "model_spec": dict(model_spec),
                "payload": payload,
            }
        )
        self.assert_strict_binding(strict, evidence["base_fact_id"])
        terminal_error_type = self.terminal_failure_error_types.get(stage)
        if terminal_error_type:
            return SimpleNamespace(
                terminal_status="failed",
                parsed_response=None,
                raw_response=None,
                response_model=self.response_models.get(stage),
                usage={},
                latency_ms=2,
                attempt_count=1,
                attempts=[
                    {
                        "attempt": 1,
                        "status": "failed",
                        "latency_ms": 2,
                        "error_type": terminal_error_type,
                    }
                ],
            )
        response = self.verdict(evidence)
        if self.invalid_accept and stage == "fact_review_supplement_generator":
            response["fact_review"]["decision"] = "defer"
        raw = json.dumps(response, ensure_ascii=False, sort_keys=True)
        try:
            validator(response)
        except supplement.SupplementValidationError:
            return SimpleNamespace(
                terminal_status="failed",
                parsed_response=None,
                raw_response=raw + " https://secret.example.invalid/key",
                response_model=self.response_models.get(
                    stage, model_spec["expected_response_model"]
                ),
                usage={},
                latency_ms=3,
                attempt_count=1,
                attempts=[
                    {
                        "attempt": 1,
                        "status": "failed",
                        "latency_ms": 3,
                        "error_type": "SupplementValidationError",
                        "unsafe_message": "https://secret.example.invalid/key",
                    }
                ],
            )
        return SimpleNamespace(
            terminal_status="completed",
            parsed_response=response,
            raw_response=raw,
            response_model=self.response_models.get(
                stage, model_spec["expected_response_model"]
            ),
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            latency_ms=4,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "ok", "latency_ms": 4}],
        )

    @staticmethod
    def assert_strict_binding(strict, base_fact_id):
        if not isinstance(strict, dict) or strict.get("strict") is not True:
            raise AssertionError("strict JSON Schema was not supplied")
        schema = strict.get("schema")
        if not isinstance(schema, dict) or schema.get("additionalProperties") is not False:
            raise AssertionError("strict JSON Schema is not closed")
        if schema["properties"]["base_fact_id"]["enum"] != [base_fact_id]:
            raise AssertionError("strict JSON Schema is not bound to the singleton ID")


class StaticGuardSensitiveRouter(SupplementRouter):
    """Reproduce the shared router's static-proxy-only guard failure."""

    def __init__(self):
        super().__init__()
        self.static_guard_triggered = False

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        if model_spec.get("require_response_model_identity") is True:
            self.static_guard_triggered = True
            return SimpleNamespace(
                terminal_status="failed",
                parsed_response=None,
                raw_response=None,
                response_model=model_spec["expected_response_model"],
                usage={},
                latency_ms=1,
                attempt_count=1,
                attempts=[
                    {
                        "attempt": 1,
                        "status": "failed",
                        "latency_ms": 1,
                        "error_type": "StaticProxyIdentityError",
                    }
                ],
            )
        return super().request_json(
            stage, item_id, model_spec, prompt, validator
        )


class FactReviewSupplementTests(unittest.TestCase):
    def run_supplement(self, router, **kwargs):
        authority = supplement._inspect_parent_authority(
            kwargs["parent_run_manifest_path"],
            expected_selected_count=3,
            expected_completed_count=0,
            expected_failed_count=3,
            expected_decisions_count=0,
        )
        with mock.patch.object(
            supplement, "inspect_parent_authority", return_value=authority
        ), mock.patch.object(
            supplement, "_create_runtime_router", return_value=router
        ):
            return supplement.run_recovery(**kwargs)

    @staticmethod
    def config():
        return {
            "config_version": "supplement-test-v1",
            "provider_profiles": {
                "generator": {"protocol": "openai_compatible"},
                "reviewer": {"protocol": "openai"},
            },
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
                "require_response_model_identity": True,
            },
            {
                "provider_profile": "reviewer",
                "model": "review-model",
                "expected_response_model": "review-model-resolved",
                "max_output_tokens": 1024,
                "max_retries": 0,
                "require_response_model_identity": True,
            },
        )

    @staticmethod
    def now():
        return "2026-09-24T00:00:00+00:00"

    def make_export(self, root, count=3):
        source = root / "source"
        ids = [f"pbf_fact_{index}" for index in range(count)]
        triples = []
        clusters = []
        queues = []
        bundles = []
        for index, base_fact_id in enumerate(ids):
            candidate_id = f"candidate_{index}"
            answer = f"Answer {index}"
            input_hash = str(index + 1) * 64
            triples.append(
                review_helpers.triple(
                    base_fact_id, candidate_id, input_hash, answer=answer
                )
            )
            clusters.append(
                review_helpers.cluster(
                    base_fact_id,
                    [candidate_id],
                    [input_hash],
                    answer=answer,
                )
            )
            queues.append(
                review_helpers.queue(base_fact_id, [candidate_id], answer=answer)
            )
            bundles.append(
                review_helpers.behavior(
                    base_fact_id,
                    candidate_id,
                    answer=answer,
                    distractors=[] if index == 0 else None,
                )
            )
        review_helpers.write_jsonl(source / "provisional_triples.jsonl", triples)
        review_helpers.write_jsonl(source / "base_fact_clusters.jsonl", clusters)
        review_helpers.write_jsonl(source / "review_queue.jsonl", queues)
        review_helpers.write_jsonl(source / "behavior_input_bundle.jsonl", bundles)
        scope = supplement.review_tool.create_scope(
            output_dir=root / "scope",
            artifact_paths=supplement.review_tool.source_paths(source),
            base_fact_ids=ids,
            review_sample_path=None,
        )
        export = supplement.review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "export",
        )
        return ids, Path(export["manifest_path"])

    def make_parent(self, root, count=3):
        ids, export_manifest = self.make_export(root, count=count)
        generator_spec, reviewer_spec = self.specs()
        result = supplement.parent_runner.run_review(
            export_manifest_path=export_manifest,
            output_dir=root / "parent-run",
            config=self.config(),
            env_path=root / ".env",
            generator_spec=generator_spec,
            reviewer_spec=reviewer_spec,
            batch_size=1,
            max_workers=1,
            router=ParentFailureRouter(),
            now_fn=self.now,
        )
        self.assertEqual(result["status"], "completed_with_failures")
        return ids, Path(result["manifest_path"])

    def test_parent_authority_derives_exact_failed_ids_without_manual_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids, parent_manifest = self.make_parent(root)
            authority = supplement._inspect_parent_authority(
                parent_manifest,
                expected_selected_count=3,
                expected_completed_count=0,
                expected_failed_count=3,
                expected_decisions_count=0,
            )
            self.assertEqual(
                [unit.base_fact_id for unit in authority.failed_units], ids
            )
            self.assertEqual(set(authority.inputs), {
                "parent_run_manifest",
                "parent_checkpoint",
                "parent_decisions",
                "export_manifest",
                "review_items",
                "decision_template",
            })
            self.assertEqual(
                authority.inputs["parent_checkpoint"]["record_count"], 3
            )
            self.assertEqual(
                authority.inputs["parent_decisions"]["record_count"], 0
            )
            zero_target_schema = supplement.strict_response_schema(
                authority.failed_units[0]
            )["properties"]["distractor_reviews"]
            self.assertEqual(zero_target_schema["minItems"], 0)
            self.assertEqual(zero_target_schema["maxItems"], 0)
            self.assertNotIn(
                "enum", zero_target_schema["items"]["properties"]["distractor_id"]
            )

    def test_public_parent_authority_rejects_small_three_failure_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root, count=3)
            with self.assertRaisesRegex(ValueError, "selected_count must be exactly 8969"):
                supplement.inspect_parent_authority(parent_manifest)

    def test_run_writes_independent_bound_strict_outputs_without_touching_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids, parent_manifest = self.make_parent(root)
            parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
            parent_paths = [
                parent_manifest,
                Path(parent["artifacts"]["checkpoint"]["path"]),
                Path(parent["artifacts"]["decisions"]["path"]),
            ]
            before = {
                str(path): supplement.parent_runner.sha256_file(path)
                for path in parent_paths
            }
            generator_spec, reviewer_spec = self.specs()
            router = SupplementRouter()
            result = self.run_supplement(
                router,
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "supplement-run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["selected_count"], 3)
            self.assertEqual(result["terminal_status_counts"], {"completed": 3})
            self.assertEqual(len(router.calls), 6)
            self.assertEqual(
                [call["stage"] for call in router.calls],
                [
                    "fact_review_supplement_generator",
                    "fact_review_supplement_reviewer",
                ]
                * 3,
            )
            for call in router.calls:
                self.assertEqual(call["item_id"], call["payload"]["evidence"]["base_fact_id"])
                self.assertNotIn("json_mode", call["model_spec"])
                self.assertNotIn(
                    "require_response_model_identity", call["model_spec"]
                )

            checkpoints = read_jsonl(result["checkpoint_path"])
            decisions = read_jsonl(result["decisions_path"])
            self.assertEqual([row["base_fact_id"] for row in checkpoints], ids)
            self.assertEqual(len(decisions), 3)
            for row, decision in zip(checkpoints, decisions):
                self.assertEqual(row["schema_version"], supplement.CHECKPOINT_SCHEMA_VERSION)
                self.assertEqual(row["tool_version"], supplement.TOOL_VERSION)
                self.assertIsNone(row["validation_error_code"])
                self.assertFalse(row["human_gold"])
                self.assertEqual(row["strict_decision"], decision)
                self.assertFalse(decision["human_gold"])
                self.assertEqual(
                    decision["review_provenance"]["review_method"],
                    supplement.REVIEW_METHOD,
                )
                for call in row["model_calls"]:
                    self.assertEqual(
                        call["structured_output_mode"],
                        "strict_json_schema_no_fallback",
                    )
                    self.assertIsNone(call["validation_error_code"])
                    self.assertNotIn("raw_response", call)

            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["selected_count"], 3)
            self.assertEqual(manifest["batch_size"], 1)
            self.assertFalse(manifest["human_gold"])
            self.assertEqual(manifest["failed_base_fact_ids"], ids)
            self.assertEqual(
                manifest["run_contract_sha256"],
                supplement.parent_runner.sha256_value(manifest["run_contract"]),
            )
            runtime_sources = manifest["run_contract"]["runtime_sources"]
            self.assertEqual(
                runtime_sources["supplement_runner"]["sha256"],
                supplement.parent_runner.sha256_file(SCRIPT_PATH),
            )
            self.assertEqual(
                set(runtime_sources),
                {"supplement_runner", "parent_runner", "review_contract"},
            )
            self.assertEqual(set(manifest["inputs"]), {
                "parent_run_manifest",
                "parent_checkpoint",
                "parent_decisions",
                "export_manifest",
                "review_items",
                "decision_template",
            })
            self.assertTrue(
                manifest["safety_contract"]["strict_json_schema_required"]
            )
            self.assertFalse(
                manifest["safety_contract"]["json_mode_fallback_allowed"]
            )
            self.assertNotIn(
                "require_response_model_identity",
                manifest["model_roles"]["generator"],
            )
            self.assertEqual(
                manifest["run_contract"]["response_identity_enforcement"],
                "runner_exact_post_response_without_static_proxy_guard_v2",
            )
            after = {
                str(path): supplement.parent_runner.sha256_file(path)
                for path in parent_paths
            }
            self.assertEqual(before, after)

    def test_cross_field_failure_records_finite_code_without_raw_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            result = self.run_supplement(
                SupplementRouter(invalid_accept=True),
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "supplement-run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["terminal_status_counts"], {"failed": 3})
            self.assertEqual(read_jsonl(result["decisions_path"]), [])
            rows = read_jsonl(result["checkpoint_path"])
            for row in rows:
                self.assertEqual(
                    row["validation_error_code"], "accept_fact_inconsistent"
                )
                self.assertEqual(
                    row["model_calls"][-1]["validation_error_code"],
                    "accept_fact_inconsistent",
                )
            serialized = Path(result["checkpoint_path"]).read_text(encoding="utf-8")
            self.assertNotIn("secret.example.invalid", serialized)
            self.assertNotIn('"raw_response":', serialized)

    def test_exact_match_succeeds_without_static_proxy_identity_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            router = StaticGuardSensitiveRouter()
            result = self.run_supplement(
                router,
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "supplement-run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed")
            self.assertFalse(router.static_guard_triggered)
            self.assertTrue(router.calls)
            self.assertTrue(
                all(
                    "require_response_model_identity" not in call["model_spec"]
                    for call in router.calls
                )
            )

    def test_completed_response_without_model_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            result = self.run_supplement(
                SupplementRouter(
                    response_models={"fact_review_supplement_generator": None}
                ),
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "supplement-run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            rows = read_jsonl(result["checkpoint_path"])
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(
                    row["validation_error_code"],
                    "response_model_identity_missing",
                )
                self.assertEqual(
                    row["model_calls"][0]["response_model_identity_status"],
                    "not_observed",
                )
                self.assertEqual(
                    row["model_calls"][0]["terminal_status"], "failed"
                )

    def test_static_proxy_identity_failure_has_distinct_finite_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            result = self.run_supplement(
                SupplementRouter(
                    terminal_failure_error_types={
                        "fact_review_supplement_generator": "StaticProxyIdentityError"
                    }
                ),
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "supplement-run",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            for row in read_jsonl(result["checkpoint_path"]):
                self.assertEqual(
                    row["validation_error_code"], "static_proxy_identity_failure"
                )

    def test_expected_and_actual_role_response_identities_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            reviewer_same_expected = {
                **reviewer_spec,
                "expected_response_model": generator_spec[
                    "expected_response_model"
                ],
            }
            with self.assertRaisesRegex(
                ValueError, "distinct expected response models"
            ):
                self.run_supplement(
                    SupplementRouter(),
                    parent_run_manifest_path=parent_manifest,
                    output_dir=root / "same-expected",
                    config=self.config(),
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_same_expected,
                    now_fn=self.now,
                )

            result = self.run_supplement(
                SupplementRouter(
                    response_models={
                        "fact_review_supplement_reviewer": generator_spec[
                            "expected_response_model"
                        ]
                    }
                ),
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "same-actual",
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            self.assertEqual(result["terminal_status_counts"], {"failed": 3})
            for row in read_jsonl(result["checkpoint_path"]):
                self.assertEqual(row["failure_stage"], "reviewer")
                self.assertEqual(
                    row["validation_error_code"],
                    "response_model_identity_mismatch",
                )

    def test_resume_rejects_runtime_source_contract_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            output_dir = root / "supplement-run"
            self.run_supplement(
                SupplementRouter(),
                parent_run_manifest_path=parent_manifest,
                output_dir=output_dir,
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            changed_sources = supplement._runtime_source_bindings()
            changed_sources["supplement_runner"] = {
                **changed_sources["supplement_runner"],
                "sha256": "0" * 64,
            }
            with mock.patch.object(
                supplement,
                "_runtime_source_bindings",
                return_value=changed_sources,
            ):
                with self.assertRaisesRegex(
                    ValueError, "resume supplement run contract mismatch"
                ):
                    self.run_supplement(
                        SupplementRouter(),
                        parent_run_manifest_path=parent_manifest,
                        output_dir=output_dir,
                        config=self.config(),
                        env_path=root / ".env",
                        generator_spec=generator_spec,
                        reviewer_spec=reviewer_spec,
                        now_fn=self.now,
                    )

    def test_resume_rejects_unsafe_checkpoint_trace_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            output_dir = root / "supplement-run"
            result = self.run_supplement(
                SupplementRouter(),
                parent_run_manifest_path=parent_manifest,
                output_dir=output_dir,
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            rows = read_jsonl(result["checkpoint_path"])
            rows[0]["model_calls"][0]["raw_response"] = "secret"
            supplement.parent_runner.write_jsonl(Path(result["checkpoint_path"]), rows)
            with self.assertRaisesRegex(ValueError, "contains unsafe fields"):
                self.run_supplement(
                    SupplementRouter(),
                    parent_run_manifest_path=parent_manifest,
                    output_dir=output_dir,
                    config=self.config(),
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    now_fn=self.now,
                )

    def test_resume_recursively_rejects_unsafe_retry_history_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            output_dir = root / "supplement-run"
            self.run_supplement(
                SupplementRouter(invalid_accept=True),
                parent_run_manifest_path=parent_manifest,
                output_dir=output_dir,
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=self.now,
            )
            recovered = self.run_supplement(
                SupplementRouter(),
                parent_run_manifest_path=parent_manifest,
                output_dir=output_dir,
                config=self.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                retry_failed=True,
                now_fn=self.now,
            )
            rows = read_jsonl(recovered["checkpoint_path"])
            self.assertTrue(rows[0]["retry_history"])
            rows[0]["retry_history"][0]["model_calls"][0]["prompt"] = "secret"
            supplement.parent_runner.write_jsonl(
                Path(recovered["checkpoint_path"]), rows
            )
            with self.assertRaisesRegex(ValueError, "contains unsafe fields"):
                self.run_supplement(
                    SupplementRouter(),
                    parent_run_manifest_path=parent_manifest,
                    output_dir=output_dir,
                    config=self.config(),
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    now_fn=self.now,
                )

    def test_batch_size_and_provider_protocol_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, parent_manifest = self.make_parent(root)
            generator_spec, reviewer_spec = self.specs()
            with self.assertRaisesRegex(ValueError, "batch_size is fixed at 1"):
                self.run_supplement(
                    SupplementRouter(),
                    parent_run_manifest_path=parent_manifest,
                    output_dir=root / "bad-batch",
                    config=self.config(),
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    batch_size=2,
                    now_fn=self.now,
                )
            config = self.config()
            config["provider_profiles"]["generator"] = {"protocol": "anthropic"}
            with self.assertRaisesRegex(ValueError, "does not support required strict JSON Schema"):
                self.run_supplement(
                    SupplementRouter(),
                    parent_run_manifest_path=parent_manifest,
                    output_dir=root / "bad-route",
                    config=config,
                    env_path=root / ".env",
                    generator_spec=generator_spec,
                    reviewer_spec=reviewer_spec,
                    now_fn=self.now,
                )
            self.assertNotIn("router", inspect.signature(supplement.run_recovery).parameters)


if __name__ == "__main__":
    unittest.main()
