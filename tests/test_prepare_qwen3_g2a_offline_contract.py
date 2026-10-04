import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_qwen3_g2a_offline_contract.py"
SPEC = importlib.util.spec_from_file_location("prepare_qwen3_g2a_offline_contract", SCRIPT_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)

AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_qwen3_g2a_offline_contract.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_qwen3_g2a_offline_contract", AUDIT_SCRIPT_PATH)
audit_module = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT_SPEC.loader.exec_module(audit_module)


class Qwen3G2AOfflineContractTests(unittest.TestCase):
    def test_option_free_prompt_preserves_natural_prompt_suffix(self):
        fact = {
            "prompt_en": "The capital of France is",
            "prompt_zh": "法国的首都是",
        }
        original = {"language": "en", "context": ""}
        targeted = {"language": "zh", "context": "伦敦是一座著名城市。"}
        self.assertEqual(
            module.option_free_prompt(fact, original),
            module.natural_prompt(fact, "en"),
        )
        self.assertEqual(
            module.option_free_prompt(fact, targeted),
            "背景信息：\n伦敦是一座著名城市。\n\n" + module.natural_prompt(fact, "zh"),
        )

    def test_answer_groups_reject_cross_option_alias_overlap(self):
        fact = {
            "base_fact_id": "fact-1",
            "answer_en": "Paris",
            "answer_zh": "巴黎",
            "answer_aliases_en": [],
            "answer_aliases_zh": [],
            "static_mcq": {
                "options": [
                    {"option_id": "gold", "choice": "A", "kind": "gold", "text_en": "Paris", "text_zh": "巴黎"},
                    {"option_id": "d1", "choice": "B", "kind": "distractor", "text_en": "London", "text_zh": "伦敦"},
                    {"option_id": "d2", "choice": "C", "kind": "distractor", "text_en": "Rome", "text_zh": "罗马"},
                ],
                "variants": [
                    {"distractor_id": "d1", "answer_aliases_en": ["PARIS"], "answer_aliases_zh": []},
                    {"distractor_id": "d2", "answer_aliases_en": [], "answer_aliases_zh": []},
                ],
            },
        }
        with self.assertRaisesRegex(ValueError, "answer aliases overlap"):
            module.answer_groups(fact)

    def test_donors_are_deterministic_same_relation_and_different_component(self):
        target = {
            "base_fact_id": "target",
            "probe_relation_id": "relation-a",
            "leakage_component_id": "component-target",
        }
        sources = [target]
        for index in range(7):
            sources.append({
                "base_fact_id": f"donor-{index}",
                "probe_relation_id": "relation-a",
                "leakage_component_id": f"component-{index}",
            })
        sources.extend([
            {
                "base_fact_id": "same-component",
                "probe_relation_id": "relation-a",
                "leakage_component_id": "component-target",
            },
            {
                "base_fact_id": "wrong-relation",
                "probe_relation_id": "relation-b",
                "leakage_component_id": "component-x",
            },
        ])
        first = module.deterministic_donors(target, sources, "dev-fold-01", "zh")
        second = module.deterministic_donors(target, list(reversed(sources)), "dev-fold-01", "zh")
        self.assertEqual(
            [row["base_fact_id"] for row in first],
            [row["base_fact_id"] for row in second],
        )
        self.assertEqual(len(first), 5)
        self.assertTrue(all(row["probe_relation_id"] == "relation-a" for row in first))
        self.assertTrue(all(row["leakage_component_id"] != "component-target" for row in first))

    def test_current_materialized_contract_passes_full_offline_audit(self):
        manifest = (
            REPO_ROOT
            / "data_processed"
            / "factual_perturbation"
            / "static-g0a-160-dual-v8-20260916"
            / "qwen3-8b-pnt-development-v1"
            / "g2a-option-free-preparation-v1"
            / "g2a_offline_contract_manifest.json"
        )
        if not manifest.exists():
            self.skipTest("materialized G2A offline contract is not present")
        report = audit_module.audit_contract(manifest)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["option_free"]["row_count"], 960)
        self.assertEqual(report["vector_sources"]["row_count"], 1920)
        self.assertEqual(report["split_counts"], {"development": 96, "sealed": 32, "validation": 32})
        self.assertEqual(
            report["checks"]["natural_semantic_deferrals_have_missing_value_policy"],
            "passed",
        )
        self.assertEqual(
            report["checks"]["behavior_blind_unrelated_fact_control_frozen"],
            "passed",
        )


if __name__ == "__main__":
    unittest.main()
