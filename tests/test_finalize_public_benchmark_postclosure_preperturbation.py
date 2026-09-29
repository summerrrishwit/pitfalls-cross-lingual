import importlib.util
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "finalize_public_benchmark_postclosure_preperturbation.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "finalize_public_benchmark_postclosure_preperturbation", SCRIPT_PATH
)
finalizer = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = finalizer
SCRIPT_SPEC.loader.exec_module(finalizer)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def row(label, old_split):
    answer = f"Actor {label.upper()}"
    return {
        "schema_version": finalizer.INPUT_BUNDLE_SCHEMA_VERSION,
        "base_fact_id": f"pbf_{label}",
        "subject_en": f"Character {label.upper()}",
        "answer_en": answer,
        "answer_aliases_en": [answer, f"Performer {label.upper()}"],
        "probe_relation_id": "character_to_actor",
        "bundle_status": "preperturbation_provisional_not_frozen",
        "canonical_freeze_status": "provisional_not_frozen",
        "split_assignment": old_split,
        "split_group_id": f"old_group_{label}",
        "split_policy_version": "normalized-answer-group-sha256-v1",
        "split_status": "provisional_not_frozen",
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "distractor_candidates": [
            {
                "distractor_id": f"obsolete_{label}",
                "donor_base_fact_id": "pbf_g",
                "split_assignment": old_split,
            }
        ],
        "subject_target": None,
        "answer_target": None,
        "prompt_target": None,
        "target_language": None,
        "subject_zh": None,
        "answer_zh": None,
        "answer_aliases_zh": None,
        "canonical_fact_zh": None,
        "prompt_zh": None,
        "distractor_candidates_zh": None,
        "translation_status": "pending_generation_and_review",
        "perturbation_ready": False,
        "perturbation_status": "not_run",
        "path_not_token_experiment_ready": False,
        "sealed_evaluation_status": "not_run",
        "experiment_status": {
            "ollama_behavior_screening": "not_run",
            "hf_exact_checkpoint_reproduction": "not_run",
        },
        "admission": {
            "prompt_ready": True,
            "behavior_screening_ready": False,
            "hf_bridge_ready": False,
            "relation_conditioned_mechanism_eligible": False,
            "path_not_token_experiment_ready": False,
            "blocking_reasons": [
                "canonical_freeze_pending",
                "split_freeze_pending",
                "distractor_review_pending",
            ],
        },
    }


class PostclosurePreperturbationFinalizerTests(unittest.TestCase):
    def make_fixture(
        self,
        root,
        *,
        insufficient_donors=False,
        split_group_atomicity_violation=False,
    ):
        bundle_path = root / "preperturbation.jsonl"
        rows = [
            row("a", "development"),
            row("b", "development"),
            row("c", "development"),
            row("d", "validation"),
            row("e", "validation"),
            row("f", "validation"),
            row("g", "sealed"),
            row("h", "sealed"),
        ]
        if split_group_atomicity_violation:
            rows[1]["split_group_id"] = rows[0]["split_group_id"]
        finalizer.write_jsonl(bundle_path, rows)
        bundle_index = {item["base_fact_id"]: item for item in rows}
        input_split_groups, input_split_group_contract = (
            finalizer._expected_input_split_group_contract(bundle_index)
        )
        leakage_grouping_contract = {
            "grouping_sources": finalizer.LEAKAGE_GROUPING_SOURCES,
            "input_split_group": input_split_group_contract,
            "same_fact_implies_same_leakage_component": True,
        }

        selection_path = root / "selection_manifest.json"
        selection = {
            "schema_version": finalizer.SELECTION_MANIFEST_SCHEMA_VERSION,
            "selected_base_fact_count": len(rows),
            "artifacts": {
                "preperturbation_behavior_input_bundle": finalizer._file_binding(
                    bundle_path,
                    schema_version=finalizer.INPUT_BUNDLE_SCHEMA_VERSION,
                    record_count=len(rows),
                )
            },
            "safety_contract": {
                "network_or_model_used": False,
                "model_inference_performed": False,
                "translation_performed": False,
                "perturbation_performed": False,
                "canonical_freeze_performed": False,
                "split_freeze_performed": False,
                "path_not_token_experiment_performed": False,
            },
        }
        finalizer.write_json(selection_path, selection)

        candidate_path = root / "candidate_manifest.json"
        candidate = {
            "schema_version": finalizer.CANDIDATE_MANIFEST_SCHEMA_VERSION,
            "status": "conservative_candidate_closure_complete",
            "inputs": {
                "cohort_bundle": finalizer._file_binding(
                    bundle_path,
                    schema_version=finalizer.INPUT_BUNDLE_SCHEMA_VERSION,
                    record_count=len(rows),
                )
            },
            "cohort": {
                "base_fact_count": len(rows),
                "ordered_base_fact_ids_sha256": finalizer.sha256_value(
                    [item["base_fact_id"] for item in rows]
                ),
                "ordered_row_hashes_sha256": finalizer.sha256_value(
                    [finalizer.sha256_value(item) for item in rows]
                ),
            },
            "safety_contract": {
                "network_or_model_used": False,
                "review_freeze_emitted": False,
                "split_freeze_emitted": False,
                "perturbation_authorized": False,
                "human_gold": False,
            },
        }
        finalizer.write_json(candidate_path, candidate)

        adjudications_path = root / "adjudications.jsonl"
        adjudications = [
            {
                "schema_version": finalizer.ADJUDICATION_SCHEMA_VERSION,
                "pair_id": "pair_g_a",
                "decision": "same_fact",
                "excluded_cohort_base_fact_ids": [],
                "reviewer_type": "codex_proxy",
                "human_gold": False,
            },
            {
                "schema_version": finalizer.ADJUDICATION_SCHEMA_VERSION,
                "pair_id": "pair_h_f",
                "decision": "exclude_cohort",
                "excluded_cohort_base_fact_ids": ["pbf_h"],
                "reviewer_type": "codex_proxy",
                "human_gold": False,
            },
        ]
        finalizer.write_jsonl(adjudications_path, adjudications)

        components_path = root / "leakage_components.jsonl"
        if insufficient_donors:
            final_splits = {
                "a": "validation",
                "b": "validation",
                "c": "sealed",
                "d": "development",
                "e": "development",
                "f": "development",
            }
        else:
            final_splits = {
                "a": "validation",
                "b": "validation",
                "c": "validation",
                "d": "development",
                "e": "development",
                "f": "development",
            }
        components = []
        for label, split in final_splits.items():
            base_fact_id = f"pbf_{label}"
            input_split_group_id = bundle_index[base_fact_id]["split_group_id"]
            components.append(
                {
                    "schema_version": finalizer.COMPONENT_SCHEMA_VERSION,
                    "component_id": f"component_{label}",
                    "member_node_ids": [f"current_bundle:pbf_{label}"],
                    "member_node_count": 1,
                    "retained_cohort_base_fact_ids": [f"pbf_{label}"],
                    "all_cohort_base_fact_ids": [f"pbf_{label}"],
                    "input_split_group_ids": [input_split_group_id],
                    "supporting_input_split_group_ids": (
                        [input_split_group_id]
                        if len(input_split_groups[input_split_group_id]) > 1
                        else []
                    ),
                    "external_current_pool_base_fact_ids": [],
                    "comparison_canonical_audit_record_ids": [],
                    "supporting_union_pair_ids": [],
                    "relation_counts": {"character_to_actor": 1},
                    "relation_source_fields": {
                        "character_to_actor": "probe_relation_id"
                    },
                    "provisional_split_assignment": split,
                }
            )
        finalizer.write_jsonl(components_path, components)

        exclusions_path = root / "dedup_exclusions.jsonl"
        exclusions = [
            {
                "schema_version": finalizer.EXCLUSION_SCHEMA_VERSION,
                "base_fact_id": "pbf_g",
                "cohort_index": 6,
                "disposition": "deduplicated_same_fact",
                "representative_base_fact_id": "pbf_a",
                "reason_pair_ids": ["pair_g_a"],
            },
            {
                "schema_version": finalizer.EXCLUSION_SCHEMA_VERSION,
                "base_fact_id": "pbf_h",
                "cohort_index": 7,
                "disposition": "excluded_by_adjudication",
                "representative_base_fact_id": None,
                "reason_pair_ids": ["pair_h_f"],
            },
        ]
        finalizer.write_jsonl(exclusions_path, exclusions)

        actual = {
            "character_to_actor": {split: 0 for split in finalizer.SPLITS}
        }
        for split in final_splits.values():
            actual["character_to_actor"][split] += 1
        counts = {
            "input": 8,
            "retained": 6,
            "deduplicated": 1,
            "explicitly_excluded": 1,
            "replacement_required_to_restore_input_size": 2,
        }
        candidate_binding = finalizer._file_binding(candidate_path)
        adjudication_binding = finalizer._file_binding(
            adjudications_path,
            schema_version=finalizer.ADJUDICATION_SCHEMA_VERSION,
            record_count=len(adjudications),
        )
        component_binding = finalizer._file_binding(
            components_path,
            schema_version=finalizer.COMPONENT_SCHEMA_VERSION,
            record_count=len(components),
        )
        exclusion_binding = finalizer._file_binding(
            exclusions_path,
            schema_version=finalizer.EXCLUSION_SCHEMA_VERSION,
            record_count=len(exclusions),
        )
        split_path = root / "provisional_split_manifest.json"
        split = {
            "schema_version": finalizer.SPLIT_MANIFEST_SCHEMA_VERSION,
            "status": "provisional_recomputed_not_frozen",
            "split_status": "provisional_not_frozen",
            "split_policy_version": (
                "relation-stratified-component-greedy-sha256-v2"
            ),
            "split_seed": "fixture-semantic-split-v1",
            "leakage_grouping_contract": leakage_grouping_contract,
            "candidate_manifest": candidate_binding,
            "adjudications": adjudication_binding,
            "components": component_binding,
            "dedup_exclusions": exclusion_binding,
            "cohort_counts": counts,
            "relation_totals": {"character_to_actor": 6},
            "actual_relation_split_counts": actual,
            "unresolved_candidate_count": 0,
            "component_assignments": [
                {
                    "component_id": item["component_id"],
                    "split_assignment": item["provisional_split_assignment"],
                    "retained_cohort_base_fact_ids": item[
                        "retained_cohort_base_fact_ids"
                    ],
                    "relation_counts": item["relation_counts"],
                }
                for item in components
            ],
            "safety_contract": {
                "component_atomicity_enforced": True,
                "input_split_group_atomicity_enforced": True,
                "copied_provisional_split_assignments": False,
                "network_or_model_used": False,
                "review_freeze_emitted": False,
                "split_freeze_emitted": False,
                "perturbation_authorized": False,
                "human_gold": False,
            },
        }
        finalizer.write_json(split_path, split)

        resolution_path = root / "resolution_manifest.json"
        resolution = {
            "schema_version": finalizer.RESOLUTION_MANIFEST_SCHEMA_VERSION,
            "status": "bounded_lexical_closure_resolved_non_frozen",
            "leakage_grouping_contract": leakage_grouping_contract,
            "candidate_manifest": candidate_binding,
            "adjudications": adjudication_binding,
            "outputs": {
                "components": component_binding,
                "dedup_exclusions": exclusion_binding,
                "provisional_split_manifest": finalizer._file_binding(
                    split_path,
                    schema_version=finalizer.SPLIT_MANIFEST_SCHEMA_VERSION,
                ),
            },
            "counts": counts,
            "unresolved_candidate_count": 0,
            "semantic_paraphrase_closure_complete": False,
            "historical_exposure_complete": False,
            "formal_bundle_emitted": False,
            "review_freeze_emitted": False,
            "split_freeze_emitted": False,
            "perturbation_authorized": False,
        }
        finalizer.write_json(resolution_path, resolution)
        return {
            "bundle_path": bundle_path,
            "selection_path": selection_path,
            "candidate_path": candidate_path,
            "adjudications_path": adjudications_path,
            "components_path": components_path,
            "exclusions_path": exclusions_path,
            "split_path": split_path,
            "resolution_path": resolution_path,
        }

    def run_finalizer(self, fixture, output):
        return finalizer.finalize_postclosure_preperturbation(
            preperturbation_bundle_path=fixture["bundle_path"],
            selection_manifest_path=fixture["selection_path"],
            closure_resolution_manifest_path=fixture["resolution_path"],
            output_dir=output,
        )

    def rewrite_split_manifest(self, fixture, mutator):
        split = json.loads(fixture["split_path"].read_text(encoding="utf-8"))
        mutator(split)
        finalizer.write_json(fixture["split_path"], split)
        resolution = json.loads(
            fixture["resolution_path"].read_text(encoding="utf-8")
        )
        resolution["outputs"]["provisional_split_manifest"] = (
            finalizer._file_binding(
                fixture["split_path"],
                schema_version=finalizer.SPLIT_MANIFEST_SCHEMA_VERSION,
            )
        )
        finalizer.write_json(fixture["resolution_path"], resolution)

    def add_semantic_repair_fixture(self, root, fixture):
        current_rows = read_jsonl(fixture["bundle_path"])
        source_rows = [item for item in current_rows if item["base_fact_id"] != "pbf_h"]
        source_rows.append(row("x", "sealed"))
        source_bundle_path = root / "semantic_source_bundle.jsonl"
        finalizer.write_jsonl(source_bundle_path, source_rows)

        candidates = [
            {
                "schema_version": finalizer.SEMANTIC_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": "sem_active_a_d",
                "behavior_blind": True,
                "left_base_fact_id": "pbf_a",
                "right_base_fact_id": "pbf_d",
                "left_split_assignment": "development",
                "right_split_assignment": "validation",
            },
            {
                "schema_version": finalizer.SEMANTIC_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": "sem_retired_a_x",
                "behavior_blind": True,
                "left_base_fact_id": "pbf_a",
                "right_base_fact_id": "pbf_x",
                "left_split_assignment": "development",
                "right_split_assignment": "sealed",
            },
        ]
        candidate_path = root / "semantic_candidates.jsonl"
        finalizer.write_jsonl(candidate_path, candidates)
        decisions = [
            {
                "schema_version": finalizer.SEMANTIC_DECISION_SCHEMA_VERSION,
                "candidate_id": candidate["candidate_id"],
                "candidate_record_sha256": finalizer.sha256_value(candidate),
                "decision": decision,
                "rationale": "fixture semantic decision",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "behavior_blind": True,
                "terminal_status": "completed",
            }
            for candidate, decision in zip(
                candidates, ["same_leakage_component", "same_fact"]
            )
        ]
        decision_path = root / "semantic_decisions.jsonl"
        finalizer.write_jsonl(decision_path, decisions)
        semantic_manifest_path = root / "semantic_manifest.json"
        source_binding = finalizer._file_binding(
            source_bundle_path,
            schema_version=finalizer.INPUT_BUNDLE_SCHEMA_VERSION,
            record_count=len(source_rows),
        )
        source_binding["base_fact_ids_sha256"] = finalizer.sha256_value(
            sorted(item["base_fact_id"] for item in source_rows)
        )
        finalizer.write_json(
            semantic_manifest_path,
            {
                "schema_version": finalizer.SEMANTIC_CLOSURE_MANIFEST_SCHEMA_VERSION,
                "status": "blocked_cross_split_semantic_edges",
                "behavior_blind": True,
                "human_gold": False,
                "candidate_generation_complete": True,
                "candidate_adjudication_complete": True,
                "cohort_scope_complete": True,
                "out_of_cohort_bridge_search_complete": True,
                "out_of_cohort_bridge_adjudication_complete": True,
                "source_revision_binding_complete": True,
                "unresolved_candidate_count": 0,
                "semantic_paraphrase_closure_scope": (
                    "declared_bounded_candidate_protocol_only"
                ),
                "input_bundle": source_binding,
                "candidates": finalizer._file_binding(
                    candidate_path,
                    schema_version=finalizer.SEMANTIC_CANDIDATE_SCHEMA_VERSION,
                    record_count=len(candidates),
                ),
                "adjudications": finalizer._file_binding(
                    decision_path,
                    schema_version=finalizer.SEMANTIC_DECISION_SCHEMA_VERSION,
                    record_count=len(decisions),
                ),
                "decision_counts": {
                    "same_fact": 1,
                    "same_leakage_component": 1,
                },
                "cross_split_positive_count": 2,
            },
        )

        repair_manifest_path = root / "cohort_repair_manifest.json"
        finalizer.write_json(
            repair_manifest_path,
            {
                "schema_version": finalizer.COHORT_REPAIR_MANIFEST_SCHEMA_VERSION,
                "status": "cohort_repaired_not_frozen",
                "artifacts": {
                    "preperturbation_behavior_input_bundle": finalizer._file_binding(
                        fixture["bundle_path"],
                        schema_version=finalizer.INPUT_BUNDLE_SCHEMA_VERSION,
                        record_count=len(current_rows),
                    )
                },
                "repair": {
                    "removed_base_fact_id": "pbf_x",
                    "replacement_base_fact_id": "pbf_h",
                },
            },
        )
        replacement = next(
            item for item in current_rows if item["base_fact_id"] == "pbf_h"
        )
        comparison_ids = [
            item["base_fact_id"]
            for item in current_rows
            if item["base_fact_id"] != "pbf_h"
        ]
        replacement_review_path = root / "replacement_semantic_review.json"
        finalizer.write_json(
            replacement_review_path,
            {
                "schema_version": (
                    finalizer.REPLACEMENT_SEMANTIC_REVIEW_SCHEMA_VERSION
                ),
                "base_fact_id": "pbf_h",
                "behavior_blind": True,
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "terminal_status": "completed",
                "decision": (
                    "no_same_fact_or_leakage_component_candidate_identified"
                ),
                "comparison_base_fact_count": len(comparison_ids),
                "comparison_base_fact_ids_sha256": finalizer.sha256_value(
                    comparison_ids
                ),
                "replacement_record_sha256": finalizer.sha256_value(replacement),
                "replacement_semantic_identity_sha256": finalizer.sha256_value(
                    finalizer._semantic_identity(replacement)
                ),
                "review_method": "fixture-review-v1",
                "review_scope": "fixture-current-cohort",
                "reviewed_at": "2026-09-16T00:00:00Z",
                "reviewer_id": "codex-proxy-fixture",
                "rationale": "fixture replacement has no matching semantic edge",
            },
        )
        return {
            "semantic_manifest_path": semantic_manifest_path,
            "repair_manifest_path": repair_manifest_path,
            "replacement_review_path": replacement_review_path,
        }

    def test_removes_closed_duplicates_rewrites_splits_and_regenerates_donors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            output = root / "finalized"
            result = self.run_finalizer(fixture, output)

            rows = read_jsonl(
                output / "postclosure_preperturbation_behavior_input_bundle.jsonl"
            )
            by_id = {item["base_fact_id"]: item for item in rows}
            self.assertEqual(set(by_id), {f"pbf_{label}" for label in "abcdef"})
            self.assertNotIn("pbf_g", by_id)
            self.assertNotIn("pbf_h", by_id)
            self.assertEqual(result["retained_base_fact_count"], 6)
            self.assertEqual(result["removed_base_fact_count"], 2)
            self.assertFalse(result["perturbation_authorized"])

            expected_split = {
                **{f"pbf_{label}": "validation" for label in "abc"},
                **{f"pbf_{label}": "development" for label in "def"},
            }
            for base_fact_id, item in by_id.items():
                self.assertEqual(item["split_assignment"], expected_split[base_fact_id])
                self.assertEqual(
                    item["split_group_id"],
                    f"component_{base_fact_id.removeprefix('pbf_')}",
                )
                self.assertEqual(
                    item["bundle_status"],
                    "postclosure_preperturbation_provisional_not_frozen",
                )
                self.assertEqual(item["split_status"], "provisional_not_frozen")
                self.assertEqual(
                    item["split_policy_version"],
                    finalizer.POSTCLOSURE_SPLIT_POLICY_VERSION,
                )
                self.assertEqual(
                    item["postclosure_lineage"][
                        "preclosure_split_policy_version"
                    ],
                    "normalized-answer-group-sha256-v1",
                )
                self.assertEqual(
                    item["postclosure_lineage"][
                        "postclosure_split_policy_version"
                    ],
                    finalizer.POSTCLOSURE_SPLIT_POLICY_VERSION,
                )
                self.assertFalse(item["perturbation_ready"])
                self.assertEqual(item["perturbation_status"], "not_run")
                self.assertFalse(item["path_not_token_experiment_ready"])
                self.assertIn(
                    "duplicate_replacement_pending",
                    item["admission"]["blocking_reasons"],
                )
                self.assertEqual(len(item["distractor_candidates"]), 2)
                for candidate in item["distractor_candidates"]:
                    donor_id = candidate["donor_base_fact_id"]
                    self.assertIn(donor_id, by_id)
                    self.assertNotIn(donor_id, {"pbf_g", "pbf_h"})
                    self.assertEqual(
                        by_id[donor_id]["probe_relation_id"],
                        item["probe_relation_id"],
                    )
                    self.assertEqual(
                        by_id[donor_id]["split_assignment"],
                        item["split_assignment"],
                    )
                    self.assertFalse(candidate["verified"])
                    self.assertFalse(candidate["human_gold"])
                    self.assertEqual(
                        candidate["status"], "codex_proxy_pending_review"
                    )

            distractors = read_jsonl(output / "distractor_candidates.jsonl")
            self.assertEqual(len(distractors), 12)
            self.assertTrue(
                all(not item["distractor_id"].startswith("obsolete_") for item in distractors)
            )
            assignments = {
                item["base_fact_id"]: item
                for item in read_jsonl(output / "postclosure_assignments.jsonl")
            }
            self.assertEqual(assignments["pbf_g"]["disposition"], "deduplicated_same_fact")
            self.assertEqual(assignments["pbf_g"]["representative_base_fact_id"], "pbf_a")
            self.assertIsNone(assignments["pbf_g"]["postclosure_split_assignment"])
            self.assertEqual(assignments["pbf_h"]["disposition"], "excluded_by_adjudication")
            self.assertIsNone(assignments["pbf_h"]["representative_base_fact_id"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["schema_version"],
                "public-benchmark-postclosure-preperturbation-summary-v2",
            )
            self.assertEqual(summary["replacement_required_to_restore_input_size"], 2)
            self.assertEqual(
                summary["next_required_stage"],
                "replacement_selection_and_closure_rerun",
            )
            self.assertTrue(summary["all_donors_retained"])
            self.assertTrue(summary["all_donors_same_relation_and_final_split"])
            self.assertTrue(summary["input_split_group_atomicity_preserved"])
            self.assertEqual(
                summary["split_policy_version"],
                finalizer.POSTCLOSURE_SPLIT_POLICY_VERSION,
            )
            self.assertTrue(
                summary["all_retained_rows_use_validated_split_policy"]
            )

            manifest = json.loads(
                (output / "finalization_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema_version"],
                "public-benchmark-postclosure-preperturbation-finalization-v3",
            )
            self.assertEqual(
                manifest["tool_version"],
                "public-benchmark-postclosure-preperturbation-finalizer-v3",
            )
            self.assertTrue(
                manifest["consistency_checks"][
                    "input_split_group_atomicity_preserved"
                ]
            )
            self.assertEqual(
                manifest["split_policy_version"],
                finalizer.POSTCLOSURE_SPLIT_POLICY_VERSION,
            )
            self.assertTrue(
                manifest["consistency_checks"][
                    "all_retained_rows_use_validated_split_policy"
                ]
            )

    def test_semantic_positive_edges_are_union_closed_and_cross_split_edges_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            semantic = self.add_semantic_repair_fixture(root, fixture)
            output = root / "semantic-finalized"
            with mock.patch.object(finalizer, "DISTRACTORS_PER_FACT", 0):
                result = finalizer.finalize_postclosure_preperturbation(
                    preperturbation_bundle_path=fixture["bundle_path"],
                    selection_manifest_path=fixture["selection_path"],
                    closure_resolution_manifest_path=fixture["resolution_path"],
                    output_dir=output,
                    semantic_closure_manifest_path=semantic[
                        "semantic_manifest_path"
                    ],
                    cohort_repair_manifest_path=semantic["repair_manifest_path"],
                    replacement_semantic_review_path=semantic[
                        "replacement_review_path"
                    ],
                )

            rows = {
                item["base_fact_id"]: item
                for item in read_jsonl(
                    output
                    / "postclosure_preperturbation_behavior_input_bundle.jsonl"
                )
            }
            self.assertEqual(
                rows["pbf_a"]["leakage_component_id"],
                rows["pbf_d"]["leakage_component_id"],
            )
            self.assertEqual(
                rows["pbf_a"]["split_assignment"],
                rows["pbf_d"]["split_assignment"],
            )
            self.assertEqual(result["semantic_aware_component_count"], 5)
            resolutions = {
                item["edge_id"]: item
                for item in read_jsonl(output / "semantic_edge_resolutions.jsonl")
            }
            self.assertEqual(
                resolutions["sem_active_a_d"]["resolution_status"],
                "resolved_same_component_and_split",
            )
            self.assertEqual(
                resolutions["sem_retired_a_x"]["resolution_status"],
                "retired_by_cohort_repair",
            )
            split_manifest = json.loads(
                (output / "semantic_aware_split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(split_manifest["exact_relation_targets_achieved"])
            self.assertEqual(
                split_manifest[
                    "source_cross_split_positive_edges_resolved_or_retired"
                ],
                2,
            )

    def test_stale_selection_bundle_binding_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            selection = json.loads(
                fixture["selection_path"].read_text(encoding="utf-8")
            )
            selection["artifacts"]["preperturbation_behavior_input_bundle"][
                "sha256"
            ] = "0" * 64
            finalizer.write_json(fixture["selection_path"], selection)
            output = root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "SHA-256 is stale"):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_stale_component_fails_through_resolution_binding_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            components = read_jsonl(fixture["components_path"])
            components[0]["provisional_split_assignment"] = "sealed"
            finalizer.write_jsonl(fixture["components_path"], components)
            output = root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "closure components SHA-256 is stale"):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_fewer_than_two_final_split_donors_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root, insufficient_donors=True)
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "Fewer than 2 retained same-relation/final-split"
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_input_split_group_split_across_components_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(
                root, split_group_atomicity_violation=True
            )
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "Input split_group_id is not atomic after closure"
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_missing_input_split_group_safety_flag_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            split = json.loads(fixture["split_path"].read_text(encoding="utf-8"))
            split["safety_contract"].pop("input_split_group_atomicity_enforced")
            finalizer.write_json(fixture["split_path"], split)
            resolution = json.loads(
                fixture["resolution_path"].read_text(encoding="utf-8")
            )
            resolution["outputs"]["provisional_split_manifest"] = (
                finalizer._file_binding(
                    fixture["split_path"],
                    schema_version=finalizer.SPLIT_MANIFEST_SCHEMA_VERSION,
                )
            )
            finalizer.write_json(fixture["resolution_path"], resolution)
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "Closure split safety contract is invalid"
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_stale_input_split_group_contract_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            split = json.loads(fixture["split_path"].read_text(encoding="utf-8"))
            split["leakage_grouping_contract"]["input_split_group"][
                "ordered_group_members_sha256"
            ] = "0" * 64
            finalizer.write_json(fixture["split_path"], split)
            resolution = json.loads(
                fixture["resolution_path"].read_text(encoding="utf-8")
            )
            resolution["leakage_grouping_contract"] = copy.deepcopy(
                split["leakage_grouping_contract"]
            )
            resolution["outputs"]["provisional_split_manifest"] = (
                finalizer._file_binding(
                    fixture["split_path"],
                    schema_version=finalizer.SPLIT_MANIFEST_SCHEMA_VERSION,
                )
            )
            finalizer.write_json(fixture["resolution_path"], resolution)
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "Closure input split-group contract is stale"
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_stale_split_policy_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            self.rewrite_split_manifest(
                fixture,
                lambda split: split.__setitem__(
                    "split_policy_version",
                    "relation-stratified-component-greedy-sha256-v1",
                ),
            )
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError, "Closure split policy is unsupported"
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_missing_split_policy_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            self.rewrite_split_manifest(
                fixture,
                lambda split: split.pop("split_policy_version"),
            )
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError,
                "Closure split_policy_version must be a non-empty string",
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())

    def test_non_string_split_policy_fails_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            self.rewrite_split_manifest(
                fixture,
                lambda split: split.__setitem__("split_policy_version", 2),
            )
            output = root / "must-not-exist"
            with self.assertRaisesRegex(
                ValueError,
                "Closure split_policy_version must be a non-empty string",
            ):
                self.run_finalizer(fixture, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
