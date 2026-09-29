import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


supplement = load_module(
    "semantic_review_supplement_under_test",
    PROJECT_ROOT
    / "scripts"
    / "run_full_public_benchmark_semantic_review_supplement.py",
)
composite = load_module(
    "semantic_review_composite_under_test",
    PROJECT_ROOT
    / "scripts"
    / "build_full_public_benchmark_semantic_review_composite.py",
)
review_fixtures = load_module(
    "semantic_review_fixture_helpers_for_supplement",
    PROJECT_ROOT / "tests" / "test_run_full_public_benchmark_semantic_review.py",
)
postreview = load_module(
    "semantic_postreview_under_supplement_test",
    PROJECT_ROOT / "scripts" / "materialize_full_public_benchmark_postreview.py",
)
freezer = load_module(
    "semantic_freezer_under_supplement_test",
    PROJECT_ROOT / "scripts" / "freeze_full_public_benchmark_pre_exact_hf.py",
)
candidate_resolver = load_module(
    "semantic_candidate_resolver_under_supplement_test",
    PROJECT_ROOT / "scripts" / "resolve_full_public_benchmark_candidate_review.py",
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SupplementRouter:
    def __init__(self, *, response_models=None, invalid=False):
        self.response_models = dict(response_models or {})
        self.invalid = invalid
        self.calls = []

    @staticmethod
    def verdict(pair):
        alias_required = "answer_alias_exact" in pair["match_types"]
        return {
            "schema_version": supplement.RESPONSE_SCHEMA,
            "pair_id": pair["pair_id"],
            "relationship": "same_fact",
            "alias_valid": True if alias_required else None,
            "same_answer_entity": True,
            "same_subject_entity": True,
            "relation_semantics_same": True,
            "temporal_scope_compatible": True,
            "answer_compatible": True,
            "semantic_duplicate": True,
            "rationale": "Both endpoints express the same fixture proposition.",
            "confidence": "high",
        }

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = supplement._prompt_payload(prompt)
        evidence = payload["evidence"]
        strict = model_spec.get("strict_json_schema")
        if not isinstance(strict, dict) or strict.get("strict") is not True:
            raise AssertionError("strict schema was not supplied")
        schema = strict["schema"]
        if schema.get("additionalProperties") is not False:
            raise AssertionError("strict schema is not closed")
        if schema["properties"]["pair_id"]["enum"] != [item_id]:
            raise AssertionError("strict schema is not singleton-bound")
        self.calls.append((stage, item_id, dict(model_spec), payload))
        response = self.verdict(evidence)
        if self.invalid and stage == "semantic_review_supplement_generator":
            response["semantic_duplicate"] = False
        raw = json.dumps(response, ensure_ascii=False, sort_keys=True)
        try:
            validator(response)
        except supplement.SupplementValidationError:
            return SimpleNamespace(
                terminal_status="failed",
                parsed_response=None,
                raw_response=raw + " https://secret.invalid/key",
                response_model=model_spec["expected_response_model"],
                usage={},
                latency_ms=1,
                attempt_count=1,
                attempts=[
                    {
                        "attempt": 1,
                        "status": "failed",
                        "latency_ms": 1,
                        "error_type": "SupplementValidationError",
                        "unsafe_message": "secret",
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
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            latency_ms=1,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "ok", "latency_ms": 1}],
        )


class SemanticSupplementTests(unittest.TestCase):
    @staticmethod
    def now():
        return "2026-09-24T00:00:00+00:00"

    @staticmethod
    def config():
        return {
            "config_version": "semantic-supplement-test-v1",
            "provider_profiles": {
                "aliyun": {
                    "protocol": "openai_compatible",
                    "base_url": "https://aliyun-fixture.invalid/v1",
                },
                "openai": {
                    "protocol": "openai",
                    "base_url": "https://openai-fixture.invalid/v1",
                },
            },
            "execution": {"max_workers": 1, "max_retries": 0},
            "model_roles": {
                "translation": {
                    "primary": {
                        "provider_profile": "aliyun",
                        "model": "qwen3.7-plus",
                        "expected_response_model": "qwen3.7-plus-resolved",
                        "max_output_tokens": 1024,
                        "max_retries": 0,
                    }
                },
                "full_pool_semantic_closure_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "expected_response_model": "gpt-5.5-2026-04-24",
                        "reasoning_effort": "low",
                        "max_output_tokens": 1024,
                        "max_retries": 0,
                    }
                },
            },
        }

    def make_parent(self, root):
        helper = review_fixtures.FullSemanticReviewRunnerTests()
        fixture = helper.make_candidates(root)
        pairs = read_jsonl(fixture["candidates"])
        failed_id = pairs[1]["pair_id"]
        result = supplement.parent_runner.run_review(
            candidate_manifest_path=fixture["candidate_manifest"],
            candidate_pairs_path=fixture["candidates"],
            adjudication_template_path=fixture["template"],
            output_dir=root / "parent-run",
            config=self.config(),
            env_path=root / ".env",
            reviewer_spec={
                "provider_profile": "openai",
                "model": "gpt-5.5",
                "reasoning_effort": "low",
                "max_output_tokens": 1024,
                "max_retries": 0,
            },
            expected_response_model="gpt-5.5-2026-04-24",
            batch_size=1,
            max_workers=1,
            router=review_fixtures.FakeRouter(fail_pair_ids={failed_id}),
            now_fn=self.now,
        )
        authority = supplement._inspect_parent_authority(
            Path(result["manifest_path"]),
            expected_selected_count=2,
            expected_completed_count=1,
            expected_failed_count=1,
            expected_adjudication_count=1,
        )
        return fixture, authority

    def make_fact_apply(self, root, fixture):
        review_tool = review_fixtures.fixture_helpers.review_tool
        base_fact_ids = [
            row["base_fact_id"] for row in read_jsonl(fixture["full_base"])
        ]
        scope = review_tool.create_scope(
            output_dir=root / "fact-scope",
            artifact_paths=review_tool.source_paths(fixture["source_dir"]),
            base_fact_ids=base_fact_ids,
        )
        export = review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "fact-export",
        )
        export_manifest = review_tool.read_json(Path(export["manifest_path"]))
        templates = review_tool.read_jsonl(
            Path(
                export_manifest["artifacts"]["review_decisions_template"]["path"]
            )
        )
        decisions = []
        for template in templates:
            decision = copy.deepcopy(template)
            decision["decision"] = "accept"
            decision["review_provenance"] = {
                "reviewer_type": "codex_proxy",
                "reviewer_id": "semantic-supplement-fixture",
                "review_method": "full_member_semantic_review_v1",
                "reviewed_at": "2026-09-24T00:00:00+00:00",
            }
            for field in ("member_reviews", "alias_reviews", "distractor_reviews"):
                for target in decision[field]:
                    target["decision"] = "accept"
            decision["notes"] = "Fixture review only."
            decisions.append(decision)
        decisions_path = root / "fact-decisions.jsonl"
        review_tool.write_jsonl(decisions_path, decisions)
        applied = review_tool.apply_decisions(
            scope_manifest_path=Path(scope["manifest_path"]),
            export_manifest_path=Path(export["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=root / "fact-apply",
        )
        return Path(applied["manifest_path"])

    def run_supplement(self, root, authority, router=None, **overrides):
        generator, reviewer = supplement.resolve_model_specs(self.config())
        values = {
            "parent_run_manifest_path": authority.manifest_path,
            "output_dir": root / "supplement-run",
            "config": self.config(),
            "env_path": root / ".env",
            "generator_spec": generator,
            "reviewer_spec": reviewer,
            "now_fn": self.now,
        }
        values.update(overrides)
        router = router or SupplementRouter()
        with mock.patch.object(
            supplement, "inspect_parent_authority", return_value=authority
        ), mock.patch.object(
            supplement, "_create_runtime_router", return_value=router
        ):
            return supplement.run_recovery(**values), router

    def test_prompts_state_cross_field_invariants(self):
        for instructions in (
            supplement.GENERATOR_INSTRUCTIONS,
            supplement.REVIEWER_INSTRUCTIONS,
        ):
            normalized = " ".join(instructions.split())
            self.assertIn(
                "alias_valid=true means the shared alias validly names the same answer entity",
                normalized,
            )
            self.assertIn(
                "relationship=same_fact if and only if semantic_duplicate=true",
                normalized,
            )
            self.assertIn(
                "same_answer_entity, same_subject_entity, relation_semantics_same",
                normalized,
            )

    def test_parent_failure_set_is_derived_and_fully_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, authority = self.make_parent(Path(directory))
            self.assertEqual(len(authority.units), 2)
            self.assertEqual(len(authority.completed_ids), 1)
            self.assertEqual(len(authority.failed_ids), 1)
            self.assertEqual(
                authority.failed_ids,
                [str(authority.failed_units[0].pair_id)],
            )
            manifest = json.loads(authority.manifest_path.read_text(encoding="utf-8"))
            manifest["run_contract"]["prompt_contract_sha256"] = "0" * 64
            manifest["run_contract_sha256"] = supplement.parent_runner.sha256_value(
                manifest["run_contract"]
            )
            tampered = Path(directory) / "tampered-parent.json"
            supplement.parent_runner.write_json(tampered, manifest)
            with self.assertRaisesRegex(ValueError, "prompt contract"):
                supplement._inspect_parent_authority(
                    tampered,
                    expected_selected_count=2,
                    expected_completed_count=1,
                    expected_failed_count=1,
                    expected_adjudication_count=1,
                )

    def test_parent_checkpoint_shape_and_synchronized_identity_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            manifest = supplement.parent_runner.read_json(authority.manifest_path)
            checkpoint_path = Path(authority.inputs["parent_checkpoint"]["path"])
            checkpoint_rows = read_jsonl(checkpoint_path)
            checkpoint_rows[0]["unexpected_field"] = "unsafe"
            supplement.parent_runner.write_jsonl(checkpoint_path, checkpoint_rows)
            manifest["artifacts"]["checkpoint"].update(
                {
                    "sha256": supplement.parent_runner.sha256_file(checkpoint_path),
                    "byte_count": checkpoint_path.stat().st_size,
                }
            )
            supplement.parent_runner.write_json(authority.manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "checkpoint fields"):
                supplement._inspect_parent_authority(
                    authority.manifest_path,
                    expected_selected_count=2,
                    expected_completed_count=1,
                    expected_failed_count=1,
                    expected_adjudication_count=1,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            manifest = supplement.parent_runner.read_json(authority.manifest_path)
            checkpoint_path = Path(authority.inputs["parent_checkpoint"]["path"])
            adjudications_path = Path(authority.inputs["parent_adjudications"]["path"])
            checkpoint_rows = read_jsonl(checkpoint_path)
            adjudications = read_jsonl(adjudications_path)
            completed = next(
                row for row in checkpoint_rows if row["terminal_status"] == "completed"
            )
            completed["strict_adjudication"]["reviewer_id"] = "openai:tampered"
            adjudications[0]["reviewer_id"] = "openai:tampered"
            supplement.parent_runner.write_jsonl(checkpoint_path, checkpoint_rows)
            supplement.parent_runner.write_jsonl(adjudications_path, adjudications)
            for label, path in (
                ("checkpoint", checkpoint_path),
                ("adjudications", adjudications_path),
            ):
                manifest["artifacts"][label].update(
                    {
                        "sha256": supplement.parent_runner.sha256_file(path),
                        "byte_count": path.stat().st_size,
                    }
                )
            supplement.parent_runner.write_json(authority.manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "reviewer_id"):
                supplement._inspect_parent_authority(
                    authority.manifest_path,
                    expected_selected_count=2,
                    expected_completed_count=1,
                    expected_failed_count=1,
                    expected_adjudication_count=1,
                )

    def test_singleton_dual_model_run_is_strict_and_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            result, router = self.run_supplement(root, authority)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["selected_pair_count"], 1)
            self.assertEqual(
                [row[0] for row in router.calls],
                [
                    "semantic_review_supplement_generator",
                    "semantic_review_supplement_reviewer",
                ],
            )
            rows = read_jsonl(result["checkpoint_path"])
            self.assertEqual([row["pair_id"] for row in rows], authority.failed_ids)
            self.assertTrue(all(row["behavior_blind"] for row in rows))
            self.assertTrue(all(not row["human_gold"] for row in rows))
            manifest = supplement.parent_runner.read_json(Path(result["manifest_path"]))
            self.assertFalse(manifest["model_roles"]["infrastructure_distinct"])
            self.assertFalse(
                manifest["model_roles"]["infrastructure_independence_claimed"]
            )
            loaded = supplement.load_supplement_authority(
                Path(result["manifest_path"]), parent=authority
            )
            self.assertEqual(loaded["completed_ids"], authority.failed_ids)

    def test_finite_schema_failure_and_identity_collision_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            result, _ = self.run_supplement(
                root, authority, router=SupplementRouter(invalid=True)
            )
            row = read_jsonl(result["checkpoint_path"])[0]
            self.assertEqual(
                row["validation_error_code"],
                "relationship_semantic_duplicate_inconsistent",
            )
            self.assertNotIn(
                "secret.invalid", Path(result["checkpoint_path"]).read_text()
            )

            generator, reviewer = supplement.resolve_model_specs(self.config())
            with self.assertRaisesRegex(ValueError, "distinct resolved identities"):
                self.run_supplement(
                    root / "collision",
                    authority,
                    generator_spec={
                        **generator,
                        "expected_response_model": reviewer[
                            "expected_response_model"
                        ],
                    },
                    reviewer_spec=reviewer,
                    output_dir=root / "collision" / "run",
                )

            with self.assertRaisesRegex(ValueError, "protocol config"):
                supplement.resolve_model_specs(
                    self.config(), reviewer_model="gpt-5.5-unbound"
                )

    def test_parent_source_and_route_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            result, _ = self.run_supplement(root, authority)
            checkpoint = authority.inputs["parent_checkpoint"]
            original = Path(checkpoint["path"]).read_bytes()
            Path(checkpoint["path"]).write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 is stale"):
                supplement.load_supplement_authority(
                    Path(result["manifest_path"]), parent=authority
                )

    def test_checkpoint_trace_and_manifest_contract_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            result, _ = self.run_supplement(root, authority)
            generator, reviewer = supplement.resolve_model_specs(self.config())
            row = read_jsonl(result["checkpoint_path"])[0]
            row["model_calls"][0]["response_sha256"] = None
            row["model_calls"][0]["raw_response_sha256"] = None
            row["model_calls"][0]["attempt_count"] = 0
            row["model_calls"][0]["attempts"] = []
            with self.assertRaisesRegex(ValueError, "Completed semantic supplement call"):
                supplement._validate_checkpoint_row(
                    row,
                    units_by_id={
                        str(unit.pair_id): unit for unit in authority.failed_units
                    },
                    parent_rows=authority.failed_checkpoint_rows,
                    run_contract_sha256=supplement.parent_runner.read_json(
                        Path(result["manifest_path"])
                    )["run_contract_sha256"],
                    expected_specs={
                        "semantic_review_supplement_generator": generator,
                        "semantic_review_supplement_reviewer": reviewer,
                    },
                )

            manifest_path = Path(result["manifest_path"])
            original_manifest = supplement.parent_runner.read_json(manifest_path)
            tampered = copy.deepcopy(original_manifest)
            tampered["model_roles"]["reviewer"]["model"] = "gpt-5.5-unbound"
            supplement.parent_runner.write_json(manifest_path, tampered)
            with self.assertRaisesRegex(ValueError, "model role contract"):
                supplement.load_supplement_authority(manifest_path, parent=authority)
            tampered = copy.deepcopy(original_manifest)
            tampered["evidence_boundary"]["human_gold"] = True
            supplement.parent_runner.write_json(manifest_path, tampered)
            with self.assertRaisesRegex(ValueError, "evidence boundary"):
                supplement.load_supplement_authority(manifest_path, parent=authority)

    def test_failed_reviewer_trace_requires_successful_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, authority = self.make_parent(root)
            result, _ = self.run_supplement(
                root,
                authority,
                router=SupplementRouter(
                    response_models={
                        "semantic_review_supplement_reviewer": "unexpected-model"
                    }
                ),
            )
            row = read_jsonl(result["checkpoint_path"])[0]
            self.assertEqual(row["failure_stage"], "reviewer")
            identity_tampered = copy.deepcopy(row)
            identity_tampered["model_calls"][-1]["response_model"] = None
            generator, reviewer = supplement.resolve_model_specs(self.config())
            manifest = supplement.parent_runner.read_json(Path(result["manifest_path"]))
            validation_args = {
                "units_by_id": {
                    str(unit.pair_id): unit for unit in authority.failed_units
                },
                "parent_rows": authority.failed_checkpoint_rows,
                "run_contract_sha256": manifest["run_contract_sha256"],
                "expected_specs": {
                    "semantic_review_supplement_generator": generator,
                    "semantic_review_supplement_reviewer": reviewer,
                },
            }
            with self.assertRaisesRegex(ValueError, "identity is inconsistent"):
                supplement._validate_checkpoint_row(
                    identity_tampered, **validation_args
                )
            row["model_calls"][0]["terminal_status"] = "failed"
            row["model_calls"][0]["validation_error_code"] = (
                "provider_or_transport_failure"
            )
            row["model_calls"][0]["attempts"][-1]["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "lineage"):
                supplement._validate_checkpoint_row(row, **validation_args)

    def test_composite_union_order_lineage_replay_and_postreview_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, authority = self.make_parent(root)
            result, _ = self.run_supplement(root, authority)
            patches = (
                mock.patch.object(
                    composite.supplement_runner,
                    "inspect_parent_authority",
                    return_value=authority,
                ),
                mock.patch.object(composite, "EXPECTED_TOTAL_COUNT", 2),
                mock.patch.object(composite, "EXPECTED_PARENT_COMPLETED_COUNT", 1),
                mock.patch.object(composite, "EXPECTED_SUPPLEMENT_COUNT", 1),
            )
            with patches[0], patches[1], patches[2]:
                built = composite.build_composite_authority(
                    parent_run_manifest_path=authority.manifest_path,
                    supplement_run_manifest_path=Path(result["manifest_path"]),
                    output_dir=root / "composite",
                )
                loaded = composite.load_composite_authority(
                    Path(built["manifest_path"])
                )
                self.assertEqual(
                    [row["pair_id"] for row in loaded["adjudications"]],
                    [str(unit.pair_id) for unit in authority.units],
                )
                self.assertEqual(
                    [row["source_authority"] for row in loaded["lineage"]],
                    ["parent_completed", "supplement_recovery"],
                )
                fake_tool = SimpleNamespace(
                    load_composite_authority=lambda _path: loaded,
                    supplement_runner=composite.supplement_runner,
                )
                with mock.patch.object(
                    postreview,
                    "_load_semantic_composite_builder",
                    return_value=fake_tool,
                ):
                    evidence = postreview._load_semantic_review_evidence(
                        candidate_manifest_path=fixture["candidate_manifest"],
                        adjudications_path=None,
                        run_manifest_path=None,
                        composite_manifest_path=Path(built["manifest_path"]),
                        full_base_facts_path=fixture["full_base"],
                    )
                self.assertEqual(
                    evidence["authority_mode"],
                    "composite_semantic_review_authority",
                )
                self.assertEqual(len(evidence["decisions"]), 2)

                semantic_inputs = {
                    "authority_mode": "composite_semantic_review_authority",
                    "run_manifest": None,
                    "composite_manifest": loaded["manifest_binding"],
                    "candidate_manifest": authority.inputs["candidate_manifest"],
                    "adjudications": supplement._binding(
                        loaded["adjudications_path"],
                        schema_version=supplement.parent_runner.ADJUDICATION_SCHEMA,
                        record_count=2,
                    ),
                }
                semantic_contract = {
                    "authority_mode": "composite_semantic_review_authority",
                    "semantic_run_contract_sha256": None,
                    "route_identity_set_sha256": None,
                    "source_run_contract_sha256": loaded[
                        "source_run_contract_sha256"
                    ],
                    "source_route_identity_set_sha256": loaded[
                        "source_route_identity_set_sha256"
                    ],
                }
                with mock.patch.object(
                    freezer.postreview_tool,
                    "_load_semantic_composite_builder",
                    return_value=composite,
                ):
                    runtime = freezer._validate_semantic_composite_identity(
                        composite_manifest_binding=loaded["manifest_binding"],
                        owner_path=root / "postreview_rebuild_manifest.json",
                        semantic_inputs=semantic_inputs,
                        semantic_contract=semantic_contract,
                        config=self.config(),
                    )
                self.assertEqual(
                    runtime["authority_mode"],
                    "composite_semantic_review_authority",
                )
                self.assertGreaterEqual(len(runtime["snapshot_bindings"]), 9)

                original_manifest = composite.read_json(Path(built["manifest_path"]))
                tampered_manifest = copy.deepcopy(original_manifest)
                tampered_manifest["safety_contract"][
                    "source_run_artifacts_modified"
                ] = True
                composite.write_json(Path(built["manifest_path"]), tampered_manifest)
                with self.assertRaisesRegex(ValueError, "safety contract"):
                    composite.load_composite_authority(Path(built["manifest_path"]))
                tampered_manifest = copy.deepcopy(original_manifest)
                tampered_manifest["formal_universe_id"] = "tampered-universe"
                composite.write_json(Path(built["manifest_path"]), tampered_manifest)
                with self.assertRaisesRegex(ValueError, "formal_universe_id"):
                    composite.load_composite_authority(Path(built["manifest_path"]))
                composite.write_json(Path(built["manifest_path"]), original_manifest)

                fact_apply = self.make_fact_apply(root, fixture)
                with mock.patch.object(
                    postreview,
                    "_load_semantic_composite_builder",
                    return_value=composite,
                ):
                    rebuilt = postreview.materialize(
                        full_base_facts_path=fixture["full_base"],
                        fact_review_apply_manifest_path=fact_apply,
                        semantic_candidate_manifest_path=fixture[
                            "candidate_manifest"
                        ],
                        semantic_adjudications_path=None,
                        semantic_run_manifest_path=None,
                        semantic_composite_manifest_path=Path(
                            built["manifest_path"]
                        ),
                        output_dir=root / "postreview-composite",
                        expected_input_count=4,
                        split_seed="fixture-composite-postreview-seed",
                    )
                with mock.patch.object(
                    candidate_resolver.postreview,
                    "_load_semantic_composite_builder",
                    return_value=composite,
                ):
                    root_context = candidate_resolver._load_root_context(
                        Path(rebuilt["manifest_path"])
                    )
                    self.assertEqual(
                        root_context["root"]["manifest"][
                            "semantic_review_contract"
                        ]["authority_mode"],
                        "composite_semantic_review_authority",
                    )
                    self.assertEqual(len(root_context["semantic_decisions"]), 2)
                    self.assertGreater(len(root_context["root"]["shortfalls"]), 0)

                lineage = read_jsonl(loaded["lineage_path"])
                lineage[0]["source_authority"] = "supplement_recovery"
                composite.write_jsonl(loaded["lineage_path"], lineage)
                manifest = composite.read_json(Path(built["manifest_path"]))
                manifest["artifacts"]["adjudication_lineage"] = composite._binding(
                    loaded["lineage_path"],
                    schema_version=composite.LINEAGE_SCHEMA,
                    record_count=2,
                    relative_to=Path(built["manifest_path"]).parent,
                )
                composite.write_json(Path(built["manifest_path"]), manifest)
                with self.assertRaisesRegex(ValueError, "replay"):
                    composite.load_composite_authority(Path(built["manifest_path"]))

    def test_postreview_python_api_enforces_direct_xor_composite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "full_base_facts_path": root / "missing-full.jsonl",
                "fact_review_apply_manifest_path": root / "missing-fact.json",
                "fact_resolution_manifest_path": None,
                "semantic_candidate_manifest_path": root / "missing-candidates.json",
                "output_dir": root / "out",
            }
            with self.assertRaisesRegex(ValueError, "Supply both semantic"):
                postreview.materialize(
                    **common,
                    semantic_adjudications_path=None,
                    semantic_run_manifest_path=None,
                    semantic_composite_manifest_path=None,
                )
            with self.assertRaisesRegex(ValueError, "XOR"):
                postreview.materialize(
                    **common,
                    semantic_adjudications_path=root / "direct.jsonl",
                    semantic_run_manifest_path=root / "direct.json",
                    semantic_composite_manifest_path=root / "composite.json",
                )


if __name__ == "__main__":
    unittest.main()
