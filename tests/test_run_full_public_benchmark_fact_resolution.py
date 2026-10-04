import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from factual_pitfalls import perturbation as perturbation_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "run_full_public_benchmark_fact_resolution_tests",
    PROJECT_ROOT / "scripts" / "run_full_public_benchmark_fact_resolution.py",
)
resolver_test_helpers = load_module(
    "fact_resolution_fixture_helpers_for_runner",
    PROJECT_ROOT / "tests" / "test_resolve_full_public_benchmark_fact_review.py",
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
        *,
        fail_multi_proposer=False,
        fail_reviewer_ids=None,
        response_models=None,
        force_invalid_defer_revision=False,
        route_base_urls=None,
    ):
        self.fail_multi_proposer = fail_multi_proposer
        self.fail_reviewer_ids = set(fail_reviewer_ids or [])
        self.response_models = dict(response_models or {})
        self.force_invalid_defer_revision = force_invalid_defer_revision
        self.route_base_urls = dict(route_base_urls or {})
        self.calls = []

    def _profile(self, model_spec):
        profile_name = model_spec["provider_profile"]
        return (
            {"protocol": "openai"},
            "test-key-never-persisted",
            self.route_base_urls.get(
                profile_name, "https://shared-gateway.example.invalid/v1"
            ),
        )

    @staticmethod
    def _evidence(stage, payload):
        if stage == "fact_resolution_proposer":
            return payload["items"]
        return [entry["evidence"] for entry in payload["items"]]

    def _verdict(self, evidence, *, reviewer):
        base_fact_id = evidence["base_fact_id"]
        revise = evidence["source_review_outcome"] == "revise" or (
            evidence["source_review_outcome"] == "reject"
            and "apply_revision" in evidence["allowed_dispositions"]
        )
        if self.force_invalid_defer_revision and evidence["source_review_outcome"] == "defer":
            revise = True
        if revise:
            answer = evidence["source_values"]["answer_en"]
            rejected_alias_ids = (
                evidence.get("source_reject_override_eligibility") or {}
            ).get("target_resolutions", {}).get("rejected_alias_ids", [])
            updates = (
                {"answer_aliases_en": [answer]}
                if evidence["source_review_outcome"] == "revise" or rejected_alias_ids
                else {}
            )
            row = {
                "base_fact_id": base_fact_id,
                "disposition": "apply_revision",
                "updates": updates,
                "revision_reason": "Remove an unsupported answer alias.",
                "cohort_exclusion_reason": None,
                "rationale": "The source supports the canonical answer but not the extra alias.",
            }
            if reviewer:
                row["field_reviews"] = {
                    field: {
                        "decision": "accept",
                        "reason": f"The final {field} is source-grounded and consistent.",
                    }
                    for field in runner.resolver.MUTABLE_FIELDS
                }
            return row
        row = {
            "base_fact_id": base_fact_id,
            "disposition": "cohort_exclude",
            "updates": None,
            "revision_reason": None,
            "cohort_exclusion_reason": (
                "The non-accept source review cannot be resolved under the current schema."
            ),
            "rationale": "Fail closed without asserting that the underlying fact is false.",
        }
        if reviewer:
            row["field_reviews"] = None
        return row

    @staticmethod
    def _failure():
        return SimpleNamespace(
            terminal_status="failed",
            parsed_response=None,
            raw_response="https://secret.example.invalid/key",
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
        evidence = self._evidence(stage, payload)
        ids = [entry["base_fact_id"] for entry in evidence]
        self.calls.append((stage, ids, item_id, dict(model_spec)))
        if (
            stage == "fact_resolution_proposer"
            and self.fail_multi_proposer
            and len(ids) > 1
        ):
            return self._failure()
        if stage == "fact_resolution_reviewer" and set(ids).intersection(
            self.fail_reviewer_ids
        ):
            return self._failure()
        reviewer = stage == "fact_resolution_reviewer"
        response = {
            "schema_version": (
                runner.REVIEW_RESPONSE_SCHEMA_VERSION
                if reviewer
                else runner.PROPOSAL_RESPONSE_SCHEMA_VERSION
            ),
            "items": [self._verdict(entry, reviewer=reviewer) for entry in evidence],
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


class StaticGuardSensitiveAliasRouter(FakeRouter):
    """Reproduce the real shared-guard failure for the Qwen alias route."""

    def __init__(self):
        super().__init__(
            response_models={
                "fact_resolution_proposer": runner.DEFAULT_PROPOSER_RESPONSE_MODEL,
                "fact_resolution_reviewer": "gpt-review-resolved",
            }
        )
        self.static_guard_triggered = False

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        if (
            stage == "fact_resolution_proposer"
            and model_spec.get("require_response_model_identity") is True
        ):
            self.static_guard_triggered = True
            try:
                perturbation_runtime._require_static_proxy_response_model_identity(
                    model_spec["model"],
                    model_spec["provider_profile"],
                    runner.DEFAULT_PROPOSER_RESPONSE_MODEL,
                )
            except perturbation_runtime.StaticProxyIdentityError as error:
                return SimpleNamespace(
                    terminal_status="failed",
                    parsed_response=None,
                    raw_response=None,
                    response_model=runner.DEFAULT_PROPOSER_RESPONSE_MODEL,
                    usage={},
                    latency_ms=1,
                    attempt_count=1,
                    attempts=[
                        {
                            "attempt": 1,
                            "status": "failed",
                            "latency_ms": 1,
                            "error_type": type(error).__name__,
                        }
                    ],
                )
            raise AssertionError("alias route unexpectedly passed the shared static guard")
        return super().request_json(stage, item_id, model_spec, prompt, validator)


class FullPublicBenchmarkFactResolutionRunnerTests(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "config_version": "fact-resolution-test-v1",
            "provider_profiles": {"proposer": {}, "reviewer": {}},
            "execution": {"max_workers": 1, "max_retries": 0},
        }

    @staticmethod
    def specs():
        return (
            {
                "provider_profile": "proposer",
                "model": "qwen-proposal",
                "expected_response_model": "qwen-proposal-resolved",
                "max_output_tokens": 1024,
                "max_retries": 0,
            },
            {
                "provider_profile": "reviewer",
                "model": "gpt-review",
                "expected_response_model": "gpt-review-resolved",
                "max_output_tokens": 1024,
                "max_retries": 0,
            },
        )

    @staticmethod
    def now():
        return "2026-09-23T02:00:00+00:00"

    @staticmethod
    def materialize(root):
        helper = resolver_test_helpers.FullFactResolutionTests()
        chain, result, manifest, templates = helper.materialize(root)
        return chain, Path(result["manifest_path"]), manifest, templates

    def run_fixture(self, root, *, router=None, **overrides):
        chain, template_manifest, _, _ = self.materialize(root)
        proposer_spec, reviewer_spec = self.specs()
        arguments = {
            "template_manifest_path": template_manifest,
            "output_dir": root / "run",
            "config": self.config(),
            "env_path": root / ".env",
            "proposer_spec": proposer_spec,
            "reviewer_spec": reviewer_spec,
            "batch_size": 3,
            "max_workers": 1,
            "router": router or FakeRouter(),
            "now_fn": self.now,
        }
        arguments.update(overrides)
        return chain, template_manifest, runner.run_resolution(**arguments)

    def test_loads_only_nonaccept_items_and_projection_is_behavior_blind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, _, _ = self.materialize(root)
            _, units, selection = runner.load_resolution_units(
                template_manifest_path=template_manifest
            )
            self.assertIsNone(selection)
            self.assertEqual(
                [unit.base_fact_id for unit in units],
                ["fact_revise", "fact_defer", "fact_reject"],
            )
            serialized = json.dumps(units[0].projection, sort_keys=True)
            self.assertNotIn("split_assignment", serialized)
            self.assertNotIn("behavior_row", serialized)
            self.assertNotIn("hf_model", serialized)
            self.assertIn("source_snapshot", serialized)
            self.assertEqual(
                units[1].item["allowed_dispositions"], ["cohort_exclude"]
            )

    def test_full_run_writes_strict_apply_compatible_resolutions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = FakeRouter()
            chain, template_manifest, result = self.run_fixture(root, router=fake)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["strict_resolution_apply_ready"])
            self.assertEqual(
                result["disposition_counts"],
                {"apply_revision": 2, "cohort_exclude": 1},
            )
            decisions = read_jsonl(result["resolutions_path"])
            self.assertEqual(
                [row["base_fact_id"] for row in decisions],
                ["fact_revise", "fact_defer", "fact_reject"],
            )
            self.assertFalse(any(row["human_gold"] for row in decisions))
            revised = decisions[0]["revision"]
            self.assertEqual(set(revised["updates"]), {"answer_aliases_en"})
            self.assertEqual(
                revised["rereview"]["reviewer_type"], "independent_model_proxy"
            )
            self.assertEqual(decisions[1]["disposition"], "cohort_exclude")
            self.assertFalse(
                decisions[1]["cohort_exclusion"]["factual_falsehood_asserted"]
            )
            self.assertEqual(decisions[2]["disposition"], "apply_revision")
            self.assertEqual(
                decisions[2]["revision"]["source_reject_override"][
                    "original_review_outcome"
                ],
                "reject",
            )

            applied = runner.resolver.apply_resolutions(
                template_manifest_path=template_manifest,
                resolutions_path=Path(result["resolutions_path"]),
                output_dir=root / "applied",
            )
            self.assertEqual(applied["revision_count"], 2)
            self.assertEqual(applied["cohort_exclusion_count"], 1)
            output = Path(result["checkpoint_path"]).read_text(encoding="utf-8")
            self.assertNotIn("secret.example.invalid", output)
            self.assertNotIn('"raw_response":', output)
            self.assertNotIn("base_url", output)
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            runtime_sources = manifest["run_contract"]["runtime_sources"]
            self.assertEqual(
                runtime_sources["resolution_runner"]["sha256"],
                runner.sha256_file(Path(runner.__file__)),
            )
            self.assertEqual(
                set(runtime_sources),
                {"resolution_runner", "resolution_contract", "model_router"},
            )
            self.assertEqual(
                manifest["run_contract"]["response_identity_enforcement"],
                runner.RESPONSE_IDENTITY_ENFORCEMENT,
            )
            route_fingerprints = manifest["run_contract"]["route_fingerprints"]
            self.assertEqual(set(route_fingerprints), {"proposer", "reviewer"})
            for fingerprint in route_fingerprints.values():
                self.assertEqual(fingerprint["protocol"], "openai")
                self.assertEqual(len(fingerprint["base_url_sha256"]), 64)
                self.assertEqual(len(fingerprint["request_route_sha256"]), 64)
                self.assertFalse(fingerprint["credentials_or_endpoints_included"])
            serialized_manifest = json.dumps(manifest, sort_keys=True)
            self.assertNotIn("example.invalid", serialized_manifest)
            self.assertNotIn("test-key-never-persisted", serialized_manifest)
            self.assertNotIn(
                "require_response_model_identity",
                manifest["model_roles"]["proposer"],
            )
            self.assertTrue(
                manifest["model_roles"][
                    "expected_response_model_identities_distinct"
                ]
            )
            self.assertFalse(
                manifest["model_roles"]["effective_endpoint_identities_distinct"]
            )
            self.assertFalse(
                manifest["model_roles"]["infrastructure_distinct"]
            )
            self.assertFalse(
                manifest["model_roles"]["infrastructure_independence_claimed"]
            )
            self.assertTrue(
                all(
                    "require_response_model_identity" not in call[3]
                    for call in fake.calls
                )
            )

    def test_batch_failure_bisects_and_preserves_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = FakeRouter(fail_multi_proposer=True)
            _, _, result = self.run_fixture(root, router=fake)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                [(stage, len(ids)) for stage, ids, _, _ in fake.calls],
                [
                    ("fact_resolution_proposer", 3),
                    ("fact_resolution_proposer", 1),
                    ("fact_resolution_reviewer", 1),
                    ("fact_resolution_proposer", 2),
                    ("fact_resolution_proposer", 1),
                    ("fact_resolution_reviewer", 1),
                    ("fact_resolution_proposer", 1),
                    ("fact_resolution_reviewer", 1),
                ],
            )
            self.assertEqual(
                [row["input_index"] for row in read_jsonl(result["checkpoint_path"])],
                [0, 1, 2],
            )

    def test_response_model_identity_mismatch_fails_closed_without_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = FakeRouter(
                response_models={"fact_resolution_proposer": "unexpected-model"}
            )
            _, _, result = self.run_fixture(root, router=fake)
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["terminal_status_counts"], {"failed": 3})
            self.assertEqual(len(fake.calls), 1)
            for row in read_jsonl(result["checkpoint_path"]):
                call = row["model_calls"][0]
                self.assertEqual(call["response_model_identity_status"], "mismatch")
                self.assertEqual(
                    call["post_validation_error_type"], "ResponseModelIdentityError"
                )

    def test_response_model_identity_missing_fails_closed_without_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = FakeRouter(
                response_models={"fact_resolution_proposer": None}
            )
            _, _, result = self.run_fixture(root, router=fake)
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["terminal_status_counts"], {"failed": 3})
            self.assertEqual(len(fake.calls), 1)
            for row in read_jsonl(result["checkpoint_path"]):
                call = row["model_calls"][0]
                self.assertEqual(
                    call["response_model_identity_status"], "not_observed"
                )
                self.assertEqual(
                    call["post_validation_error_type"], "ResponseModelIdentityError"
                )

    def test_qwen_alias_exact_match_bypasses_real_static_guard(self):
        with self.assertRaises(perturbation_runtime.StaticProxyIdentityError):
            perturbation_runtime._require_static_proxy_response_model_identity(
                runner.DEFAULT_PROPOSER_ROUTE,
                "aliyun",
                runner.DEFAULT_PROPOSER_RESPONSE_MODEL,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposer_spec, reviewer_spec = self.specs()
            proposer_spec.update(
                {
                    "provider_profile": "aliyun",
                    "model": runner.DEFAULT_PROPOSER_ROUTE,
                    "expected_response_model": runner.DEFAULT_PROPOSER_RESPONSE_MODEL,
                    "require_response_model_identity": True,
                }
            )
            config = self.config()
            config["provider_profiles"]["aliyun"] = {}
            router = StaticGuardSensitiveAliasRouter()
            _, template_manifest, _, _ = self.materialize(root)
            result = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "run",
                config=config,
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                batch_size=3,
                max_workers=1,
                router=router,
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed")
            self.assertFalse(router.static_guard_triggered)
            self.assertTrue(router.calls)
            self.assertTrue(
                all(
                    "require_response_model_identity" not in call[3]
                    for call in router.calls
                )
            )

    def test_defer_revision_attempt_is_rejected_and_cannot_bypass_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, template_manifest, _, _ = self.materialize(root)
            selected = root / "selected.json"
            selected.write_text(json.dumps(["fact_defer"]), encoding="utf-8")
            proposer_spec, reviewer_spec = self.specs()
            result = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                base_fact_ids_path=selected,
                batch_size=1,
                max_workers=1,
                router=FakeRouter(force_invalid_defer_revision=True),
                now_fn=self.now,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["terminal_status_counts"], {"failed": 1})
            self.assertEqual(read_jsonl(result["resolutions_path"]), [])
            self.assertFalse(result["strict_resolution_apply_ready"])

    def test_distractor_only_eligible_reject_can_retain_with_empty_fact_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = resolver_test_helpers.FullFactResolutionTests()
            chain = helper.build_chain(root, reject_target="distractor")
            materialized = runner.resolver.materialize_template(
                full_base_facts_path=chain["full"],
                review_apply_manifest_path=chain["apply"],
                review_export_manifest_path=chain["export"],
                output_dir=root / "resolution-template",
            )
            proposer_spec, reviewer_spec = self.specs()
            result = runner.run_resolution(
                template_manifest_path=Path(materialized["manifest_path"]),
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                batch_size=3,
                max_workers=1,
                router=FakeRouter(),
                now_fn=self.now,
            )
            decisions = {
                row["base_fact_id"]: row
                for row in read_jsonl(result["resolutions_path"])
            }
            self.assertEqual(decisions["fact_reject"]["disposition"], "apply_revision")
            self.assertEqual(decisions["fact_reject"]["revision"]["updates"], {})
            applied = runner.resolver.apply_resolutions(
                template_manifest_path=Path(materialized["manifest_path"]),
                resolutions_path=Path(result["resolutions_path"]),
                output_dir=root / "applied",
            )
            self.assertEqual(applied["revision_count"], 2)

    def test_eligible_reject_falls_back_to_exclusion_when_rereviewer_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, _, _ = self.materialize(root)
            _, units, _ = runner.load_resolution_units(
                template_manifest_path=template_manifest
            )
            original = next(unit for unit in units if unit.base_fact_id == "fact_reject")
            item = json.loads(json.dumps(original.item))
            item["source_review_evidence"]["review_provenance"]["reviewer_id"] = (
                "reviewer:gpt-review-resolved"
            )
            template = runner.resolver._decision_template(item)
            unit = original._replace(
                item=item,
                template=template,
                resolution_item_sha256=runner.sha256_value(item),
                decision_template_sha256=runner.sha256_value(template),
            )
            fake = FakeRouter()
            proposal = fake._verdict(unit.projection, reviewer=False)
            verdict = fake._verdict(unit.projection, reviewer=True)
            proposer_spec, reviewer_spec = self.specs()
            decision = runner._strict_resolution(
                unit,
                proposal,
                verdict,
                proposer_spec,
                reviewer_spec,
                "qwen-proposal-resolved",
                "gpt-review-resolved",
                self.now(),
            )
            self.assertEqual(decision["disposition"], "cohort_exclude")
            self.assertEqual(
                decision["cohort_exclusion"]["reason"],
                runner.SOURCE_REVIEWER_IDENTITY_FALLBACK_REASON,
            )

    def test_selected_id_artifact_is_bound_and_resume_rejects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, _, _ = self.materialize(root)
            selected = root / "selected.json"
            selected.write_text(json.dumps(["fact_revise"]), encoding="utf-8")
            proposer_spec, reviewer_spec = self.specs()
            result = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                base_fact_ids_path=selected,
                batch_size=1,
                max_workers=1,
                router=FakeRouter(),
                now_fn=self.now,
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["selection"]["selected_count"], 1)
            self.assertEqual(
                manifest["selection"]["selected_ids_artifact"]["sha256"],
                runner.sha256_file(selected),
            )
            self.assertFalse(result["strict_resolution_apply_ready"])
            selected.write_text(json.dumps(["fact_defer"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume run contract mismatch"):
                runner.run_resolution(
                    template_manifest_path=template_manifest,
                    output_dir=root / "run",
                    config=self.config(),
                    env_path=root / ".env",
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    base_fact_ids_path=selected,
                    batch_size=1,
                    max_workers=1,
                    router=FakeRouter(),
                    now_fn=self.now,
                )

    def test_resume_retries_failed_only_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, template_manifest, _, _ = self.materialize(root)
            proposer_spec, reviewer_spec = self.specs()
            first = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                batch_size=1,
                max_workers=1,
                router=FakeRouter(fail_reviewer_ids={"fact_defer"}),
                now_fn=self.now,
            )
            self.assertEqual(first["terminal_status_counts"], {"completed": 2, "failed": 1})
            fake = FakeRouter()
            second = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                batch_size=1,
                max_workers=1,
                retry_failed=True,
                router=fake,
                now_fn=self.now,
            )
            self.assertEqual(second["status"], "completed")
            self.assertEqual([call[1] for call in fake.calls], [["fact_defer"], ["fact_defer"]])
            rows = {row["base_fact_id"]: row for row in read_jsonl(second["checkpoint_path"])}
            self.assertEqual(len(rows["fact_defer"]["retry_history"]), 1)
            self.assertEqual(rows["fact_defer"]["retry_history"][0]["failure_stage"], "reviewer")

    def test_resume_rejects_runtime_source_contract_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, result = self.run_fixture(root)
            proposer_spec, reviewer_spec = self.specs()
            changed_sources = runner._runtime_source_bindings()
            changed_sources["resolution_runner"] = {
                **changed_sources["resolution_runner"],
                "sha256": "0" * 64,
            }
            with mock.patch.object(
                runner, "_runtime_source_bindings", return_value=changed_sources
            ):
                with self.assertRaisesRegex(
                    ValueError, "resume run contract mismatch"
                ):
                    runner.run_resolution(
                        template_manifest_path=template_manifest,
                        output_dir=Path(result["manifest_path"]).parent,
                        config=self.config(),
                        env_path=root / ".env",
                        proposer_spec=proposer_spec,
                        reviewer_spec=reviewer_spec,
                        batch_size=3,
                        max_workers=1,
                        router=FakeRouter(),
                        now_fn=self.now,
                    )

    def test_resume_rejects_effective_endpoint_env_route_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_router = FakeRouter(
                route_base_urls={
                    "proposer": "https://proxy-a.example.invalid/v1",
                    "reviewer": "https://review.example.invalid/v1",
                }
            )
            _, template_manifest, result = self.run_fixture(
                root, router=first_router
            )
            proposer_spec, reviewer_spec = self.specs()
            second_router = FakeRouter(
                route_base_urls={
                    "proposer": "https://proxy-b.example.invalid/v1",
                    "reviewer": "https://review.example.invalid/v1",
                }
            )
            with self.assertRaisesRegex(
                ValueError, "resume run contract mismatch"
            ):
                runner.run_resolution(
                    template_manifest_path=template_manifest,
                    output_dir=Path(result["manifest_path"]).parent,
                    config=self.config(),
                    env_path=root / ".env",
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    batch_size=3,
                    max_workers=1,
                    router=second_router,
                    now_fn=self.now,
                )

    def test_resume_rejects_forged_matched_response_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, result = self.run_fixture(root)
            rows = read_jsonl(result["checkpoint_path"])
            proposer_call = next(
                call
                for call in rows[0]["model_calls"]
                if call["stage"] == "fact_resolution_proposer"
                and call["terminal_status"] == "completed"
            )
            proposer_call["response_model"] = "forged-response-model"
            proposer_call["response_model_identity_status"] = "matched"
            runner.write_jsonl(Path(result["checkpoint_path"]), rows)
            proposer_spec, reviewer_spec = self.specs()
            with self.assertRaisesRegex(
                ValueError, "lacks a matched fact_resolution_proposer call"
            ):
                runner.run_resolution(
                    template_manifest_path=template_manifest,
                    output_dir=Path(result["manifest_path"]).parent,
                    config=self.config(),
                    env_path=root / ".env",
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_spec,
                    batch_size=3,
                    max_workers=1,
                    router=FakeRouter(),
                    now_fn=self.now,
                )

    def test_interrupted_run_recovers_durable_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, _, _ = self.materialize(root)
            proposer_spec, reviewer_spec = self.specs()
            original = runner._process_batch

            def interrupt_second(**kwargs):
                if kwargs["units"][0].base_fact_id == "fact_defer":
                    time.sleep(0.02)
                    raise RuntimeError("simulated interruption")
                return original(**kwargs)

            with mock.patch.object(runner, "_process_batch", side_effect=interrupt_second):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner.run_resolution(
                        template_manifest_path=template_manifest,
                        output_dir=root / "run",
                        config=self.config(),
                        env_path=root / ".env",
                        proposer_spec=proposer_spec,
                        reviewer_spec=reviewer_spec,
                        batch_size=1,
                        checkpoint_compact_every=128,
                        max_workers=1,
                        router=FakeRouter(),
                        now_fn=self.now,
                    )
            journal = root / "run" / "fact_resolution_checkpoint.journal.jsonl"
            rows, needs_repair = runner._read_journal(journal)
            self.assertFalse(needs_repair)
            self.assertEqual([row["base_fact_id"] for row in rows], ["fact_revise"])
            with journal.open("ab") as handle:
                handle.write(b'{"interrupted":')
            resumed = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "run",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                batch_size=1,
                checkpoint_compact_every=128,
                max_workers=1,
                router=FakeRouter(),
                now_fn=self.now,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertFalse(journal.exists())
            self.assertEqual(
                [row["base_fact_id"] for row in read_jsonl(resumed["checkpoint_path"])],
                ["fact_revise", "fact_defer", "fact_reject"],
            )

    def test_model_specs_use_current_zh_first_config_roles(self):
        config = {
            "model_roles": {
                "translation": {
                    "primary": {
                        "provider_profile": "aliyun",
                        "model": runner.DEFAULT_PROPOSER_ROUTE,
                    },
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": runner.DEFAULT_REVIEWER_MODEL,
                    },
                },
                "full_pool_fact_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": runner.DEFAULT_REVIEWER_MODEL,
                    }
                },
            },
            "execution": {"max_retries": 2},
        }
        proposer, reviewer = runner.resolve_model_specs(config)
        self.assertEqual(proposer["expected_response_model"], "qwen3.8-max")
        self.assertEqual(reviewer["expected_response_model"], "gpt-5.5-2026-04-24")
        self.assertNotIn("require_response_model_identity", proposer)
        self.assertNotIn("require_response_model_identity", reviewer)
        self.assertEqual(reviewer["max_retries"], 2)

    def test_expected_and_actual_role_response_identities_must_be_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, template_manifest, _, _ = self.materialize(root)
            proposer_spec, reviewer_spec = self.specs()
            reviewer_same_expected = {
                **reviewer_spec,
                "expected_response_model": proposer_spec["expected_response_model"],
            }
            with self.assertRaisesRegex(
                ValueError, "distinct expected response models"
            ):
                runner.run_resolution(
                    template_manifest_path=template_manifest,
                    output_dir=root / "same-expected",
                    config=self.config(),
                    env_path=root / ".env",
                    proposer_spec=proposer_spec,
                    reviewer_spec=reviewer_same_expected,
                    batch_size=3,
                    max_workers=1,
                    router=FakeRouter(),
                    now_fn=self.now,
                )

            result = runner.run_resolution(
                template_manifest_path=template_manifest,
                output_dir=root / "same-actual",
                config=self.config(),
                env_path=root / ".env",
                proposer_spec=proposer_spec,
                reviewer_spec=reviewer_spec,
                batch_size=3,
                max_workers=1,
                router=FakeRouter(
                    response_models={
                        "fact_resolution_reviewer": proposer_spec[
                            "expected_response_model"
                        ]
                    }
                ),
                now_fn=self.now,
            )
            self.assertEqual(result["terminal_status_counts"], {"failed": 3})
            for row in read_jsonl(result["checkpoint_path"]):
                self.assertEqual(row["failure_stage"], "reviewer")
                self.assertEqual(
                    row["model_calls"][-1]["response_model_identity_status"],
                    "mismatch",
                )


if __name__ == "__main__":
    unittest.main()
