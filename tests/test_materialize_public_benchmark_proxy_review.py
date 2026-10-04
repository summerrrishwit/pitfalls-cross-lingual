import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = PROJECT_ROOT / "scripts" / "materialize_public_benchmark_proxy_review.py"
MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_public_benchmark_proxy_review", MATERIALIZER_PATH
)
materializer = importlib.util.module_from_spec(MATERIALIZER_SPEC)
assert MATERIALIZER_SPEC and MATERIALIZER_SPEC.loader
MATERIALIZER_SPEC.loader.exec_module(materializer)

REVIEW_TEST_PATH = PROJECT_ROOT / "tests" / "test_review_public_benchmark_bundle.py"
REVIEW_TEST_SPEC = importlib.util.spec_from_file_location(
    "review_public_benchmark_test_helpers", REVIEW_TEST_PATH
)
review_test_helpers = importlib.util.module_from_spec(REVIEW_TEST_SPEC)
assert REVIEW_TEST_SPEC and REVIEW_TEST_SPEC.loader
REVIEW_TEST_SPEC.loader.exec_module(review_test_helpers)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class ProxyReviewMaterializerTests(unittest.TestCase):
    def make_export(self, root):
        helper = review_test_helpers.PublicBenchmarkReviewToolTests()
        source, fact_a, fact_b = helper.make_source(root)
        bundle_path = source / "behavior_input_bundle.jsonl"
        bundle = read_jsonl(bundle_path)
        bundle[0]["probe_relation_candidate"] = "family_a"
        bundle[1]["probe_relation_candidate"] = "family_b"
        review_test_helpers.write_jsonl(bundle_path, bundle)
        scope = materializer.review_tool.create_scope(
            output_dir=root / "scope",
            artifact_paths=materializer.review_tool.source_paths(source),
            base_fact_ids=[fact_a, fact_b],
            review_sample_path=None,
        )
        export = materializer.review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "export",
        )
        return (
            fact_a,
            fact_b,
            Path(export["manifest_path"]),
            root / "export" / "review_items.jsonl",
        )

    def plan(self, export_manifest, *, overrides_a=None, overrides_b=None):
        return {
            "schema_version": materializer.PLAN_SCHEMA_VERSION,
            "export_manifest_sha256": materializer.sha256_file(export_manifest),
            "family_reviews": {
                "family_a": {
                    "reviewed_all_items": True,
                    "default_decision": "accept",
                    "default_reason": "Audited fixture A.",
                    "reviewer_type": "codex_proxy",
                    "reviewer_id": "codex-fixture-a",
                    "review_method": "full_item_review_v1",
                    "reviewed_at": "2026-09-14T00:00:00Z",
                    "human_gold": False,
                    "item_overrides": overrides_a or {},
                },
                "family_b": {
                    "reviewed_all_items": True,
                    "default_decision": "accept",
                    "default_reason": "Audited fixture B.",
                    "reviewer_type": "codex_proxy",
                    "reviewer_id": "codex-fixture-b",
                    "review_method": "full_item_review_v1",
                    "reviewed_at": "2026-09-14T00:00:00Z",
                    "human_gold": False,
                    "item_overrides": overrides_b or {},
                },
            },
        }

    def write_plan(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_materialize_supports_audited_alias_and_distractor_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fact_a, _, export_manifest, items_path = self.make_export(root)
            item_a = next(row for row in read_jsonl(items_path) if row["base_fact_id"] == fact_a)
            alias_id = item_a["review_targets"]["aliases"][1]["alias_id"]
            distractor_id = item_a["review_targets"]["distractors"][0]["distractor_id"]
            plan = self.plan(
                export_manifest,
                overrides_a={
                    fact_a: {
                        "decision": "accept",
                        "reason": "Fact accepted after target-level review.",
                        "alias_decisions": {alias_id: "reject"},
                        "distractor_decisions": {distractor_id: "reject"},
                    }
                },
            )
            plan_path = root / "plan.json"
            self.write_plan(plan_path, plan)
            result = materializer.materialize(
                export_manifest_path=export_manifest,
                review_plan_path=plan_path,
                output_dir=root / "decisions",
            )
            decisions = {
                row["base_fact_id"]: row for row in read_jsonl(result["decisions_path"])
            }
            alias_reviews = {
                row["alias_id"]: row["decision"] for row in decisions[fact_a]["alias_reviews"]
            }
            distractor_reviews = {
                row["distractor_id"]: row["decision"]
                for row in decisions[fact_a]["distractor_reviews"]
            }
            self.assertEqual(alias_reviews[alias_id], "reject")
            self.assertEqual(distractor_reviews[distractor_id], "reject")
            self.assertEqual(result["decision_counts"], {"accept": 2})

    def test_rejects_override_for_an_item_in_another_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_b, export_manifest, _ = self.make_export(root)
            plan = self.plan(
                export_manifest,
                overrides_a={
                    fact_b: {"decision": "reject", "reason": "Wrong family."}
                },
            )
            plan_path = root / "plan.json"
            self.write_plan(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "target another family"):
                materializer.materialize(
                    export_manifest_path=export_manifest,
                    review_plan_path=plan_path,
                    output_dir=root / "decisions",
                )

    def test_rejects_unknown_target_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fact_a, _, export_manifest, _ = self.make_export(root)
            plan = self.plan(
                export_manifest,
                overrides_a={
                    fact_a: {
                        "decision": "accept",
                        "reason": "Invalid target fixture.",
                        "alias_decisions": {"alias_missing": "reject"},
                    }
                },
            )
            plan_path = root / "plan.json"
            self.write_plan(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "Unknown alias_decisions targets"):
                materializer.materialize(
                    export_manifest_path=export_manifest,
                    review_plan_path=plan_path,
                    output_dir=root / "decisions",
                )


if __name__ == "__main__":
    unittest.main()
