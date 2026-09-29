import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


composite = load_module(
    "full_fact_review_composite_test",
    "build_full_public_benchmark_fact_review_composite.py",
)
review_tool = load_module(
    "full_fact_review_apply_test", "review_public_benchmark_bundle.py"
)

REVIEW_TEST_PATH = PROJECT_ROOT / "tests" / "test_review_public_benchmark_bundle.py"
REVIEW_TEST_SPEC = importlib.util.spec_from_file_location(
    "review_fixture_helpers_for_composite", REVIEW_TEST_PATH
)
review_helpers = importlib.util.module_from_spec(REVIEW_TEST_SPEC)
assert REVIEW_TEST_SPEC and REVIEW_TEST_SPEC.loader
REVIEW_TEST_SPEC.loader.exec_module(review_helpers)

SUPPLEMENT_TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_run_full_public_benchmark_fact_review_supplement.py"
)
SUPPLEMENT_TEST_SPEC = importlib.util.spec_from_file_location(
    "supplement_fixture_helpers_for_composite", SUPPLEMENT_TEST_PATH
)
supplement_helpers = importlib.util.module_from_spec(SUPPLEMENT_TEST_SPEC)
assert SUPPLEMENT_TEST_SPEC and SUPPLEMENT_TEST_SPEC.loader
SUPPLEMENT_TEST_SPEC.loader.exec_module(supplement_helpers)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(composite.canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )


def binding(path, *, schema=None, rows=None):
    result = {
        "path": str(path.resolve()),
        "sha256": composite.sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema is not None:
        result["schema_version"] = schema
    if rows is not None:
        result["record_count"] = len(rows)
    return result


class FullFactReviewCompositeTests(unittest.TestCase):
    def make_chain(self, root):
        helper = review_helpers.PublicBenchmarkReviewToolTests()
        source, fact_a, fact_b, scope_result, export_result = helper.create_chain(root)
        scope_path = Path(scope_result["manifest_path"])
        export_path = Path(export_result["manifest_path"])
        export = review_tool.read_json(export_path)
        items_path = Path(export["artifacts"]["review_items"]["path"])
        templates_path = Path(
            export["artifacts"]["review_decisions_template"]["path"]
        )
        templates = {
            row["base_fact_id"]: row for row in review_tool.read_jsonl(templates_path)
        }
        items = {
            row["base_fact_id"]: row for row in review_tool.read_jsonl(items_path)
        }
        parent_decision = helper.valid_accept(templates[fact_a])
        supplement_decision = helper.valid_accept(templates[fact_b])

        parent_dir = root / "parent"
        parent_dir.mkdir()
        parent_decisions_path = parent_dir / "review_decisions_codex_proxy.jsonl"
        write_jsonl(parent_decisions_path, [parent_decision])
        parent_contract = {
            "tool_version": composite.PARENT_TOOL_VERSION,
            "export_manifest_sha256": composite.sha256_file(export_path),
            "review_items_sha256": composite.sha256_file(items_path),
            "decision_template_sha256": composite.sha256_file(templates_path),
            "selected_base_fact_ids_sha256": composite.sha256_value(
                [fact_a, fact_b]
            ),
            "selected_count": 2,
            "full_export_count": 2,
            "selection_is_full_export": True,
            "behavior_blind": True,
            "human_gold": False,
        }
        parent_contract_sha = composite.sha256_value(parent_contract)
        parent_checkpoints = [
            {
                "schema_version": composite.PARENT_CHECKPOINT_SCHEMA,
                "tool_version": composite.PARENT_TOOL_VERSION,
                "base_fact_id": fact_a,
                "input_index": 0,
                "run_contract_sha256": parent_contract_sha,
                "review_item_sha256": composite.sha256_value(items[fact_a]),
                "decision_template_sha256": composite.sha256_value(
                    templates[fact_a]
                ),
                "terminal_status": "completed",
                "strict_decision": parent_decision,
                "model_calls": [
                    {
                        "stage": "fact_review_generator",
                        "terminal_status": "completed",
                        "response_model_identity_status": "matched",
                    },
                    {
                        "stage": "fact_review_reviewer",
                        "terminal_status": "completed",
                        "response_model_identity_status": "matched",
                    },
                ],
                "behavior_blind": True,
                "human_gold": False,
            },
            {
                "schema_version": composite.PARENT_CHECKPOINT_SCHEMA,
                "tool_version": composite.PARENT_TOOL_VERSION,
                "base_fact_id": fact_b,
                "input_index": 1,
                "run_contract_sha256": parent_contract_sha,
                "review_item_sha256": composite.sha256_value(items[fact_b]),
                "decision_template_sha256": composite.sha256_value(
                    templates[fact_b]
                ),
                "terminal_status": "failed",
                "strict_decision": None,
                "model_calls": [],
                "behavior_blind": True,
                "human_gold": False,
            },
        ]
        parent_checkpoint_path = parent_dir / "fact_review_checkpoint.jsonl"
        write_jsonl(parent_checkpoint_path, parent_checkpoints)
        parent_manifest = {
            "schema_version": composite.PARENT_RUN_MANIFEST_SCHEMA,
            "tool_version": composite.PARENT_TOOL_VERSION,
            "status": "completed_with_failures",
            "scope_id": export["scope_id"],
            "export_manifest": binding(export_path),
            "review_items": binding(items_path, rows=read_jsonl(items_path)),
            "decision_template": binding(
                templates_path,
                rows=read_jsonl(templates_path),
            ),
            "run_contract": parent_contract,
            "run_contract_sha256": parent_contract_sha,
            "selected_count": 2,
            "full_export_count": 2,
            "terminal_status_counts": {"completed": 1, "failed": 1},
            "decision_counts": {"accept": 1},
            "artifacts": {
                "checkpoint": binding(
                    parent_checkpoint_path,
                    schema=composite.PARENT_CHECKPOINT_SCHEMA,
                    rows=parent_checkpoints,
                ),
                "decisions": binding(
                    parent_decisions_path,
                    schema=composite.DECISION_SCHEMA,
                    rows=[parent_decision],
                ),
            },
            "human_gold": False,
        }
        parent_manifest_path = parent_dir / "fact_review_run_manifest.json"
        write_json(parent_manifest_path, parent_manifest)

        supplement_dir = root / "supplement"
        supplement_dir.mkdir()
        supplement_contract = {
            "tool_version": composite.SUPPLEMENT_TOOL_VERSION,
            "runtime_sources": composite._load_supplement_runner()._runtime_source_bindings(),
            "parent_run_manifest_sha256": composite.sha256_file(
                parent_manifest_path
            ),
            "parent_checkpoint_sha256": composite.sha256_file(
                parent_checkpoint_path
            ),
            "parent_decisions_sha256": composite.sha256_file(
                parent_decisions_path
            ),
            "parent_run_contract_sha256": parent_contract_sha,
            "parent_selected_count": 2,
            "parent_completed_count": 1,
            "parent_failed_count": 1,
            "parent_decisions_count": 1,
            "export_manifest_sha256": composite.sha256_file(export_path),
            "review_items_sha256": composite.sha256_file(items_path),
            "decision_template_sha256": composite.sha256_file(templates_path),
            "generator_model_spec": {
                "provider_profile": "test-profile-generator",
                "model": "supplement-generator-requested-model",
                "expected_response_model": "supplement-generator-test-model"
            },
            "reviewer_model_spec": {
                "provider_profile": "test-profile-reviewer",
                "model": "supplement-reviewer-requested-model",
                "expected_response_model": "supplement-reviewer-test-model"
            },
            "review_method": composite.SUPPLEMENT_REVIEW_METHOD,
            "response_identity_enforcement": (
                composite.SUPPLEMENT_RESPONSE_IDENTITY_ENFORCEMENT
            ),
            "failed_base_fact_ids": [fact_b],
            "failed_base_fact_ids_sha256": composite.sha256_value([fact_b]),
            "selected_count": 1,
            "batch_size": 1,
            "behavior_blind": True,
            "human_gold": False,
        }
        supplement_contract_sha = composite.sha256_value(supplement_contract)

        def supplement_call(stage, model):
            is_generator = stage == "fact_review_supplement_generator"
            return {
                "stage": stage,
                "prompt_version": (
                    supplement_helpers.supplement.GENERATOR_PROMPT_VERSION
                    if is_generator
                    else supplement_helpers.supplement.REVIEWER_PROMPT_VERSION
                ),
                "base_fact_id": fact_b,
                "provider_profile": (
                    "test-profile-generator"
                    if is_generator
                    else "test-profile-reviewer"
                ),
                "requested_model": (
                    "supplement-generator-requested-model"
                    if is_generator
                    else "supplement-reviewer-requested-model"
                ),
                "expected_response_model": model,
                "response_model": model,
                "response_model_identity_status": "matched",
                "terminal_status": "completed",
                "validation_error_code": None,
                "structured_output_mode": "strict_json_schema_no_fallback",
                "strict_json_schema_sha256": "1" * 64,
                "prompt_sha256": "2" * 64,
                "request_sha256": "3" * 64,
                "response_sha256": "4" * 64,
                "raw_response_sha256": "4" * 64,
                "usage": {},
                "latency_ms": 1,
                "attempt_count": 0,
                "attempts": [],
                "credentials_or_endpoints_included": False,
            }

        supplement_checkpoints = [
            {
                "schema_version": composite.SUPPLEMENT_CHECKPOINT_SCHEMA,
                "tool_version": composite.SUPPLEMENT_TOOL_VERSION,
                "base_fact_id": fact_b,
                "input_index": 1,
                "parent_failed_checkpoint_record_sha256": composite.sha256_value(
                    parent_checkpoints[1]
                ),
                "run_contract_sha256": supplement_contract_sha,
                "review_item_sha256": composite.sha256_value(items[fact_b]),
                "decision_template_sha256": composite.sha256_value(
                    templates[fact_b]
                ),
                "terminal_status": "completed",
                "strict_decision": supplement_decision,
                "validation_error_code": None,
                "model_calls": [
                    supplement_call(
                        "fact_review_supplement_generator",
                        "supplement-generator-test-model",
                    ),
                    supplement_call(
                        "fact_review_supplement_reviewer",
                        "supplement-reviewer-test-model",
                    ),
                ],
                "retry_history": [],
                "behavior_blind": True,
                "human_gold": False,
            }
        ]
        supplement_checkpoint_path = (
            supplement_dir / "supplement_fact_review_checkpoint.jsonl"
        )
        supplement_decisions_path = (
            supplement_dir / "supplement_review_decisions_codex_proxy.jsonl"
        )
        write_jsonl(supplement_checkpoint_path, supplement_checkpoints)
        write_jsonl(supplement_decisions_path, [supplement_decision])
        supplement_manifest = {
            "schema_version": composite.SUPPLEMENT_RUN_MANIFEST_SCHEMA,
            "tool_version": composite.SUPPLEMENT_TOOL_VERSION,
            "status": "completed",
            "scope_id": export["scope_id"],
            "inputs": {
                "parent_run_manifest": binding(
                    parent_manifest_path,
                    schema=composite.PARENT_RUN_MANIFEST_SCHEMA,
                ),
                "parent_checkpoint": parent_manifest["artifacts"]["checkpoint"],
                "parent_decisions": parent_manifest["artifacts"]["decisions"],
                "export_manifest": parent_manifest["export_manifest"],
                "review_items": parent_manifest["review_items"],
                "decision_template": parent_manifest["decision_template"],
            },
            "parent_run_contract_sha256": parent_contract_sha,
            "failed_base_fact_ids": [fact_b],
            "failed_base_fact_ids_sha256": composite.sha256_value([fact_b]),
            "run_contract": supplement_contract,
            "run_contract_sha256": supplement_contract_sha,
            "selected_count": 1,
            "batch_size": 1,
            "terminal_status_counts": {"completed": 1},
            "decision_counts": {"accept": 1},
            "artifacts": {
                "checkpoint": binding(
                    supplement_checkpoint_path,
                    schema=composite.SUPPLEMENT_CHECKPOINT_SCHEMA,
                    rows=supplement_checkpoints,
                ),
                "decisions": binding(
                    supplement_decisions_path,
                    schema=composite.DECISION_SCHEMA,
                    rows=[supplement_decision],
                ),
            },
            "human_gold": False,
        }
        supplement_manifest_path = (
            supplement_dir / "supplement_fact_review_run_manifest.json"
        )
        write_json(supplement_manifest_path, supplement_manifest)
        return {
            "source": source,
            "fact_a": fact_a,
            "fact_b": fact_b,
            "scope": scope_path,
            "export": export_path,
            "parent_manifest": parent_manifest_path,
            "parent_decisions": parent_decisions_path,
            "supplement_manifest": supplement_manifest_path,
            "supplement_decisions": supplement_decisions_path,
            "parent_decision": parent_decision,
            "supplement_decision": supplement_decision,
        }

    def count_contract(self):
        return mock.patch.multiple(
            composite,
            EXPECTED_TOTAL_COUNT=2,
            EXPECTED_PARENT_COMPLETED_COUNT=1,
            EXPECTED_SUPPLEMENT_COUNT=1,
        )

    def test_builds_exact_ordered_union_and_full_apply_accepts_authority(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract(), mock.patch.object(
            review_tool, "FULL_PUBLIC_BENCHMARK_COUNT", 2
        ), mock.patch.object(
            review_tool, "_COMPOSITE_AUTHORITY_TOOL", composite
        ):
            root = Path(directory)
            fixture = self.make_chain(root)
            parent_sha_before = composite.sha256_file(fixture["parent_decisions"])
            result = composite.build_composite_authority(
                parent_run_manifest_path=fixture["parent_manifest"],
                supplement_run_manifest_path=fixture["supplement_manifest"],
                output_dir=root / "composite",
            )
            self.assertEqual(composite.sha256_file(fixture["parent_decisions"]), parent_sha_before)
            decisions = read_jsonl(result["decisions_path"])
            self.assertEqual(
                decisions,
                [fixture["parent_decision"], fixture["supplement_decision"]],
            )
            authority = composite.load_composite_authority(
                Path(result["manifest_path"])
            )
            self.assertEqual(len(authority["lineage"]), 2)
            self.assertEqual(
                [row["source_authority"] for row in authority["lineage"]],
                ["parent_completed", "supplement_recovery"],
            )

            applied = review_tool.apply_decisions(
                scope_manifest_path=fixture["scope"],
                export_manifest_path=fixture["export"],
                decisions_path=None,
                composite_authority_manifest_path=Path(result["manifest_path"]),
                output_dir=root / "apply",
                allow_partial=False,
            )
            self.assertEqual(applied["outcome_counts"], {"accept": 2})
            self.assertEqual(
                applied["decision_authority_mode"],
                "composite_fact_review_authority",
            )
            apply_manifest = review_tool.read_json(Path(applied["manifest_path"]))
            self.assertEqual(
                apply_manifest["decision_authority"]["manifest"]["sha256"],
                composite.sha256_file(Path(result["manifest_path"])),
            )

    def test_rejects_supplement_that_does_not_equal_parent_failed_set(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract():
            root = Path(directory)
            fixture = self.make_chain(root)
            manifest = composite.read_json(fixture["supplement_manifest"])
            wrong_ids = [fixture["fact_a"]]
            manifest["failed_base_fact_ids"] = wrong_ids
            manifest["failed_base_fact_ids_sha256"] = composite.sha256_value(wrong_ids)
            write_json(fixture["supplement_manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "parent failures"):
                composite.build_composite_authority(
                    parent_run_manifest_path=fixture["parent_manifest"],
                    supplement_run_manifest_path=fixture["supplement_manifest"],
                    output_dir=root / "composite",
                )

    def test_rejects_legacy_v1_supplement_runner_manifest(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract():
            root = Path(directory)
            fixture = self.make_chain(root)
            manifest = composite.read_json(fixture["supplement_manifest"])
            manifest["tool_version"] = (
                "full-public-benchmark-fact-review-supplement-runner-v1"
            )
            manifest["run_contract"]["tool_version"] = manifest["tool_version"]
            manifest["run_contract"]["review_method"] = (
                "behavior_blind_dual_model_singleton_fact_review_recovery_v1"
            )
            manifest["run_contract"].pop("response_identity_enforcement", None)
            manifest["run_contract_sha256"] = composite.sha256_value(
                manifest["run_contract"]
            )
            write_json(fixture["supplement_manifest"], manifest)
            with self.assertRaisesRegex(
                ValueError, "Supplement fact-review runner version is unsupported"
            ):
                composite.build_composite_authority(
                    parent_run_manifest_path=fixture["parent_manifest"],
                    supplement_run_manifest_path=fixture["supplement_manifest"],
                    output_dir=root / "composite",
                )

    def test_rejects_parent_decision_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract():
            root = Path(directory)
            fixture = self.make_chain(root)
            decisions = read_jsonl(fixture["parent_decisions"])
            decisions[0]["notes"] = "tampered"
            write_jsonl(fixture["parent_decisions"], decisions)
            with self.assertRaisesRegex(ValueError, "parent decisions SHA-256 is stale"):
                composite.build_composite_authority(
                    parent_run_manifest_path=fixture["parent_manifest"],
                    supplement_run_manifest_path=fixture["supplement_manifest"],
                    output_dir=root / "composite",
                )

    def test_full_direct_and_composite_apply_forbid_allow_partial(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract(), mock.patch.object(
            review_tool, "FULL_PUBLIC_BENCHMARK_COUNT", 2
        ), mock.patch.object(
            review_tool, "_COMPOSITE_AUTHORITY_TOOL", composite
        ):
            root = Path(directory)
            fixture = self.make_chain(root)
            with self.assertRaisesRegex(ValueError, "forbids allow_partial"):
                review_tool.apply_decisions(
                    scope_manifest_path=fixture["scope"],
                    export_manifest_path=fixture["export"],
                    decisions_path=fixture["parent_decisions"],
                    output_dir=root / "direct-apply",
                    allow_partial=True,
                )
            result = composite.build_composite_authority(
                parent_run_manifest_path=fixture["parent_manifest"],
                supplement_run_manifest_path=fixture["supplement_manifest"],
                output_dir=root / "composite",
            )
            with self.assertRaisesRegex(ValueError, "forbids allow_partial"):
                review_tool.apply_decisions(
                    scope_manifest_path=fixture["scope"],
                    export_manifest_path=fixture["export"],
                    decisions_path=None,
                    composite_authority_manifest_path=Path(result["manifest_path"]),
                    output_dir=root / "composite-apply",
                    allow_partial=True,
                )

    def test_rejects_supplement_decision_with_stale_review_item_binding(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract():
            root = Path(directory)
            fixture = self.make_chain(root)
            decisions = read_jsonl(fixture["supplement_decisions"])
            decisions[0]["review_item_sha256"] = "0" * 64
            write_jsonl(fixture["supplement_decisions"], decisions)
            manifest = composite.read_json(fixture["supplement_manifest"])
            checkpoint_path = Path(manifest["artifacts"]["checkpoint"]["path"])
            checkpoint = read_jsonl(checkpoint_path)
            checkpoint[0]["strict_decision"] = decisions[0]
            write_jsonl(checkpoint_path, checkpoint)
            manifest["artifacts"]["checkpoint"] = binding(
                checkpoint_path,
                schema=composite.SUPPLEMENT_CHECKPOINT_SCHEMA,
                rows=checkpoint,
            )
            manifest["artifacts"]["decisions"] = binding(
                fixture["supplement_decisions"],
                schema=composite.DECISION_SCHEMA,
                rows=decisions,
            )
            write_json(fixture["supplement_manifest"], manifest)
            with self.assertRaises(ValueError):
                composite.build_composite_authority(
                    parent_run_manifest_path=fixture["parent_manifest"],
                    supplement_run_manifest_path=fixture["supplement_manifest"],
                    output_dir=root / "composite",
                )

    def test_rejects_supplement_completed_call_without_exact_response_identity(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract():
            root = Path(directory)
            fixture = self.make_chain(root)
            manifest = composite.read_json(fixture["supplement_manifest"])
            checkpoint_path = Path(manifest["artifacts"]["checkpoint"]["path"])
            checkpoint = read_jsonl(checkpoint_path)
            checkpoint[0]["model_calls"][0]["response_model"] = "not_observed"
            write_jsonl(checkpoint_path, checkpoint)
            manifest["artifacts"]["checkpoint"] = binding(
                checkpoint_path,
                schema=composite.SUPPLEMENT_CHECKPOINT_SCHEMA,
                rows=checkpoint,
            )
            write_json(fixture["supplement_manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "exact response identity"):
                composite.build_composite_authority(
                    parent_run_manifest_path=fixture["parent_manifest"],
                    supplement_run_manifest_path=fixture["supplement_manifest"],
                    output_dir=root / "composite",
                )

    def test_composite_apply_rejects_decisions_replaced_after_loader_validation(self):
        with tempfile.TemporaryDirectory() as directory, self.count_contract(), mock.patch.object(
            review_tool, "FULL_PUBLIC_BENCHMARK_COUNT", 2
        ):
            root = Path(directory)
            fixture = self.make_chain(root)
            result = composite.build_composite_authority(
                parent_run_manifest_path=fixture["parent_manifest"],
                supplement_run_manifest_path=fixture["supplement_manifest"],
                output_dir=root / "composite",
            )

            class SwappingLoader:
                @staticmethod
                def load_composite_authority(path):
                    authority = composite.load_composite_authority(path)
                    swapped = [dict(row) for row in authority["decisions"]]
                    swapped[0]["notes"] = "replaced after authority validation"
                    write_jsonl(Path(authority["decisions_path"]), swapped)
                    return authority

            with mock.patch.object(
                review_tool, "_COMPOSITE_AUTHORITY_TOOL", SwappingLoader
            ), self.assertRaisesRegex(
                ValueError, "changed after authority validation"
            ):
                review_tool.apply_decisions(
                    scope_manifest_path=fixture["scope"],
                    export_manifest_path=fixture["export"],
                    decisions_path=None,
                    composite_authority_manifest_path=Path(result["manifest_path"]),
                    output_dir=root / "apply",
                    allow_partial=False,
                )

    def test_accepts_actual_supplement_runner_manifest_contract(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            composite,
            EXPECTED_TOTAL_COUNT=3,
            EXPECTED_PARENT_COMPLETED_COUNT=0,
            EXPECTED_SUPPLEMENT_COUNT=3,
        ), mock.patch.multiple(
            supplement_helpers.supplement,
            EXPECTED_PARENT_SELECTED_COUNT=3,
            EXPECTED_PARENT_COMPLETED_COUNT=0,
            EXPECTED_PARENT_DECISIONS_COUNT=0,
            EXPECTED_FAILED_COUNT=3,
        ):
            root = Path(directory)
            helper = supplement_helpers.FactReviewSupplementTests()
            _, parent_manifest = helper.make_parent(root)
            generator_spec, reviewer_spec = helper.specs()
            recovered = helper.run_supplement(
                supplement_helpers.SupplementRouter(),
                parent_run_manifest_path=parent_manifest,
                output_dir=root / "supplement-run",
                config=helper.config(),
                env_path=root / ".env",
                generator_spec=generator_spec,
                reviewer_spec=reviewer_spec,
                now_fn=helper.now,
            )
            result = composite.build_composite_authority(
                parent_run_manifest_path=parent_manifest,
                supplement_run_manifest_path=Path(recovered["manifest_path"]),
                output_dir=root / "composite",
            )
            authority = composite.load_composite_authority(
                Path(result["manifest_path"])
            )
            self.assertEqual(len(authority["decisions"]), 3)
            self.assertTrue(
                all(
                    row["source_authority"] == "supplement_recovery"
                    for row in authority["lineage"]
                )
            )


if __name__ == "__main__":
    unittest.main()
