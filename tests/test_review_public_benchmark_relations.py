import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "review_public_benchmark_relations.py"
)
SPEC = importlib.util.spec_from_file_location("review_public_benchmark_relations", SCRIPT_PATH)
relations = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(relations)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class PublicBenchmarkRelationReviewTests(unittest.TestCase):
    def fixture(self, root):
        source_dir = root / "source"
        output_dir = root / "export"
        policy_path = root / "policy.json"
        relation_specs = [
            ("song_to_performer", "song -> performer", "singer", "song", "person"),
            ("character_to_actor", "character -> actor", "played by", "character", "actor"),
            ("term_to_definition", "term -> definition", "definition", "term", "definition"),
            ("work_to_release_date", "work -> release date", "release date", "film", "date"),
        ]
        policy = {
            "schema_version": relations.POLICY_SCHEMA_VERSION,
            "policy_id": "fixture-review-v1",
            "source_candidate_policy_id": "fixture-candidates-v1",
            "status": "pre_perturbation_offline_only",
            "human_gold": False,
            "broad_relation_normalized_required": False,
            "selection_seed": "fixture-seed",
            "selection_contract": {
                "prompt_ready_required": True,
                "prompt_risk_flags_must_be_empty": True,
                "one_base_fact_per_split_group": True,
                "model_outputs_must_not_be_used": True,
                "selection_precedes_fact_review": True,
                "shortfall_policy": "fail_closed",
            },
            "relations": [
                {
                    "probe_relation_id": relation_id,
                    "direction": direction,
                    "prompt_template_en": "Relation for {subject} is",
                    "review_frame_split_counts": {split: 1 for split in relations.SPLITS},
                    "target_accept_split_counts": {split: 1 for split in relations.SPLITS},
                }
                for relation_id, direction, _, _, _ in relation_specs
            ],
            "unresolved_direction_policy": {
                "candidate_id": "definition_term_direction_unresolved",
                "action": "separate_item_level_review_required",
                "eligible_for_balanced_frame": False,
            },
        }
        write_json(policy_path, policy)
        bundle_rows = []
        provisional_rows = []
        inventory_entries = []
        for relation_index, (relation_id, direction, raw, subject_type, answer_type) in enumerate(
            relation_specs
        ):
            signature_id = f"sig_{relation_index}"
            member_ids = []
            for split_index, split in enumerate(relations.SPLITS):
                base_id = f"base_{relation_index}_{split_index}"
                candidate_id = f"candidate_{relation_index}_{split_index}"
                member_ids.append(base_id)
                bundle_rows.append(
                    {
                        "schema_version": "factual-perturbation-input-bundle-v1",
                        "base_fact_id": base_id,
                        "candidate_id": candidate_id,
                        "relation_signature_id": signature_id,
                        "probe_relation_candidate": relation_id,
                        "probe_relation_candidate_direction": direction,
                        "prompt_ready": True,
                        "prompt_risk_flags": [],
                        "subject_en": f"subject {relation_index} {split_index}",
                        "answer_en": f"answer {relation_index} {split_index}",
                        "answer_aliases_en": [f"answer {relation_index} {split_index}"],
                        "canonical_fact_en": "fact",
                        "source_question_en": "question",
                        "source_dataset": "fixture",
                        "source_format": "open_qa",
                        "prompt_en": "prompt",
                        "split_group_id": f"group_{relation_index}_{split_index}",
                        "split_assignment": split,
                        "split_status": "provisional_not_frozen",
                        "canonical_status": "pending_review",
                        "relation_raw": raw,
                    }
                )
                provisional_rows.append(
                    {
                        "schema_version": "provisional-factual-triple-v1",
                        "candidate_id": candidate_id,
                        "base_fact_id": base_id,
                        "relation_signature_id": signature_id,
                        "probe_relation_candidate": relation_id,
                        "probe_relation_candidate_direction": direction,
                        "probe_relation_status": relations.DIRECTIONAL_STATUS,
                        "relation_raw": raw,
                        "subject_type": subject_type,
                        "answer_type": answer_type,
                        "source_dataset": "fixture",
                        "source_question": "question",
                        "subject": f"subject {relation_index} {split_index}",
                        "answer": f"answer {relation_index} {split_index}",
                        "canonical_fact": "fact",
                        "prompt_quality_tier": "strict_factual_completion",
                    }
                )
            inventory_entries.append(
                {
                    "signature_id": signature_id,
                    "probe_relation_candidate": relation_id,
                    "probe_relation_candidate_direction": direction,
                    "probe_relation_id": None,
                    "probe_relation_status": relations.DIRECTIONAL_STATUS,
                }
            )
        unresolved = {
            "schema_version": "provisional-factual-triple-v1",
            "candidate_id": "candidate_unresolved",
            "base_fact_id": "base_unresolved",
            "relation_signature_id": "sig_unresolved",
            "probe_relation_candidate": "definition_term_direction_unresolved",
            "probe_relation_candidate_direction": "unresolved",
            "probe_relation_status": relations.UNRESOLVED_STATUS,
            "relation_raw": "term for definition",
            "subject_type": "definition",
            "answer_type": "term",
            "source_dataset": "fixture",
            "source_question": "question",
            "subject": "description",
            "answer": "term",
            "canonical_fact": "fact",
            "prompt_quality_tier": "strict_factual_completion",
        }
        provisional_rows.append(unresolved)
        inventory_entries.append(
            {
                "signature_id": "sig_unresolved",
                "probe_relation_candidate": "definition_term_direction_unresolved",
                "probe_relation_candidate_direction": "unresolved",
                "probe_relation_id": None,
                "probe_relation_status": relations.UNRESOLVED_STATUS,
            }
        )
        write_jsonl(source_dir / "behavior_input_bundle.jsonl", bundle_rows)
        write_jsonl(source_dir / "provisional_triples.jsonl", provisional_rows)
        write_json(
            source_dir / "relation_inventory.json",
            {
                "schema_version": "provisional-relation-inventory-v1",
                "probe_relation_candidate_policy": "fixture-candidates-v1",
                "signature_count": len(inventory_entries),
                "entries": inventory_entries,
            },
        )
        result = relations.export_review(
            source_dir=source_dir, policy_path=policy_path, output_dir=output_dir
        )
        return source_dir, output_dir, result, relation_specs

    def review_plan(self, manifest_path, relation_specs):
        reviewer = {
            "reviewed_all_signature_items": True,
            "default_decision": "approve_mapping",
            "default_confidence": "high",
            "default_reason": "Reviewed fixture signature and examples.",
            "signature_overrides": {},
            "reviewer_type": "codex_proxy",
            "reviewer_id": "fixture-reviewer",
            "review_method": "manual fixture inspection",
            "reviewed_at": "2026-09-14T00:00:00+08:00",
            "human_gold": False,
        }
        return {
            "schema_version": relations.REVIEW_PLAN_SCHEMA_VERSION,
            "export_manifest_sha256": relations.sha256_file(manifest_path),
            "directional_family_reviews": {
                relation_id: dict(reviewer)
                for relation_id, _, _, _, _ in relation_specs
            },
            "direction_unresolved_review": {
                **reviewer,
                "default_decision": "defer",
                "default_confidence": "low",
                "default_reason": "Direction requires item-level review.",
            },
        }

    def test_export_separates_direction_and_predeclares_balanced_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, output_dir, result, _ = self.fixture(Path(temporary))
            self.assertEqual(result["directional_signatures"], 4)
            self.assertEqual(result["direction_unresolved_signatures"], 1)
            self.assertEqual(result["predeclared_probe_review_frame"], 12)
            self.assertEqual(result["initial_probe_review_batch"], 12)
            frame = relations.read_jsonl(output_dir / "predeclared_probe_review_frame.jsonl")
            self.assertTrue(all(row["perturbation_ready"] is False for row in frame))
            self.assertTrue(all(row["probe_relation_id"] is None for row in frame))

    def test_materialize_and_apply_keep_item_review_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, output_dir, result, relation_specs = self.fixture(root)
            manifest_path = Path(result["manifest_path"])
            plan_path = root / "review_plan.json"
            write_json(plan_path, self.review_plan(manifest_path, relation_specs))
            decision_dir = root / "decisions"
            relations.materialize_decisions(
                manifest_path=manifest_path,
                review_plan_path=plan_path,
                output_dir=decision_dir,
            )
            applied = relations.apply_decisions(
                manifest_path=manifest_path,
                directional_decisions_path=decision_dir
                / "directional_signature_decisions.jsonl",
                unresolved_decisions_path=decision_dir
                / "direction_unresolved_signature_decisions.jsonl",
                output_dir=root / "applied",
            )
            self.assertEqual(applied["signature_approved_review_frame"], 12)
            self.assertEqual(applied["signature_approved_item_review_batch"], 12)
            summary = relations.read_json(Path(applied["summary_path"]))
            self.assertIn(
                "item_level_fact_review",
                summary["remaining_blockers_before_perturbation"],
            )
            approved = relations.read_jsonl(
                root / "applied" / "signature_approved_probe_review_frame.jsonl"
            )
            self.assertTrue(all(row["perturbation_ready"] is False for row in approved))
            self.assertTrue(all(row["probe_relation_id"] for row in approved))

    def test_export_manifest_detects_tampered_review_item(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, output_dir, result, relation_specs = self.fixture(root)
            manifest_path = Path(result["manifest_path"])
            item_path = output_dir / "directional_signature_review_items.jsonl"
            with item_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            plan_path = root / "review_plan.json"
            write_json(plan_path, self.review_plan(manifest_path, relation_specs))
            with self.assertRaisesRegex(ValueError, "Artifact SHA mismatch"):
                relations.materialize_decisions(
                    manifest_path=manifest_path,
                    review_plan_path=plan_path,
                    output_dir=root / "decisions",
                )


if __name__ == "__main__":
    unittest.main()
