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
    PROJECT_ROOT / "scripts" / "resolve_full_public_benchmark_candidate_review.py"
)
SPEC = importlib.util.spec_from_file_location(
    "full_candidate_resolution", SCRIPT_PATH
)
resolver = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = resolver
SPEC.loader.exec_module(resolver)

FREEZER_PATH = (
    PROJECT_ROOT / "scripts" / "freeze_full_public_benchmark_pre_exact_hf.py"
)
FREEZER_SPEC = importlib.util.spec_from_file_location(
    "candidate_resolution_freezer", FREEZER_PATH
)
freezer = importlib.util.module_from_spec(FREEZER_SPEC)
assert FREEZER_SPEC and FREEZER_SPEC.loader
sys.modules[FREEZER_SPEC.name] = freezer
FREEZER_SPEC.loader.exec_module(freezer)


def fact(base_fact_id, relation, number):
    answer = f"Answer {base_fact_id}"
    return {
        "schema_version": resolver.pre_hf.FULL_FACT_SCHEMA_VERSION,
        "base_fact_id": base_fact_id,
        "candidate_id": f"source_{base_fact_id}",
        "source_pool_id": "fixture",
        "source_dataset": "fixture",
        "source_subset": "unit",
        "source_id": f"source_{base_fact_id}",
        "source_format": "multiple_choice",
        "source_question_en": f"Question {base_fact_id}?",
        "source_choices_en": [answer],
        "subject_en": f"Subject {base_fact_id}",
        "answer_en": answer,
        "answer_aliases_en": [answer],
        "answer_type": "entity",
        "canonical_fact_en": f"Subject {base_fact_id} has {answer}.",
        "prompt_en": f"Subject {base_fact_id} has",
        "prompt_quality_tier": "strict_factual_completion",
        "prompt_risk_flags": [],
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "relation_raw": relation,
        "relation_signature_id": f"signature_{relation}",
        "relation_signature_ids": [f"signature_{relation}"],
        "relation_normalized": None,
        "probe_relation_candidate": None,
        "probe_relation_id": None,
        "relation_partition_id": relation,
        "relation_partition_status": "broad_rule_candidate_partition",
        "answer_type_bucket": "entity",
        "leakage_component_id": f"source_component_{base_fact_id}",
        "split_assignment": "development",
        "split_policy_version": "fixture-source-split-v1",
        "split_status": "provisional_not_frozen",
        "translation_status": "pending_translation_or_rebind_to_final_candidates",
        "distractor_status": "pending",
        "hf_model_execution_status": "not_run",
        "hf_tokenizer_execution_status": "not_run",
        "fact_review": {
            "review_outcome": "accept",
            "proxy_evidence_only": True,
            "human_gold": False,
        },
        "fixture_number": number,
    }


def root_context(group_sizes):
    rows = []
    for relation, size in group_sizes.items():
        rows.extend(
            fact(f"{relation}_{number}", relation, number)
            for number in range(size)
        )
    components = []
    frozen_split = {}
    frozen_component = {}
    for row in rows:
        base_fact_id = row["base_fact_id"]
        component_id = row["leakage_component_id"]
        components.append(
            {
                "schema_version": resolver.pre_hf.COMPONENT_SCHEMA_VERSION,
                "leakage_component_id": component_id,
                "member_count": 1,
                "base_fact_ids": [base_fact_id],
                "source_split_group_ids": [f"group_{base_fact_id}"],
                "frozen_semantic_component_ids": [],
                "relation_partition_counts": {row["relation_partition_id"]: 1},
                "source_dataset_counts": {"fixture": 1},
                "answer_type_bucket_counts": {"entity": 1},
                "pinned_split_assignment": "development",
                "pinned_fact_count": 1,
                "split_assignment": "development",
            }
        )
        frozen_split[base_fact_id] = "development"
        frozen_component[base_fact_id] = f"frozen_{base_fact_id}"
    by_id = {row["base_fact_id"]: row for row in rows}
    return {
        "fact_retained_ids": sorted(by_id),
        "source_index": by_id,
        "source_rows": rows,
        "fact_base_index": by_id,
        "semantic_candidates": [],
        "semantic_decisions": {},
        "semantic_components": components,
        "semantic_split_manifest": {"split_seed": "fixture-repair-seed"},
        "frozen": {
            "binding": None,
            "excluded_source_fact_count": 0,
            "split_by_fact": frozen_split,
            "component_by_fact": frozen_component,
        },
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class CandidateResolutionTests(unittest.TestCase):
    def test_semantic_fact_resolution_context_adapts_excluded_ids(self):
        source = {"excluded_ids": ["fact_b", "fact_a"], "successor_id": "u1"}
        adapted = resolver._semantic_fact_resolution_context(source)
        self.assertEqual(adapted["rejected_ids"], ["fact_b", "fact_a"])
        self.assertNotIn("rejected_ids", source)

        with self.assertRaisesRegex(ValueError, "aliases conflict"):
            resolver._semantic_fact_resolution_context(
                {"excluded_ids": ["fact_a"], "rejected_ids": ["fact_b"]}
            )

    def test_target_ineligibility_requires_unanimous_complete_distractor_evidence(self):
        def row(candidate_id, target_id, source_id, *, decision, answer_unique, kind="distractor"):
            return {
                "candidate_id": candidate_id,
                "candidate_kind": kind,
                "target_base_fact_id": target_id,
                "source_base_fact_id": source_id,
                "candidate_row_sha256": candidate_id.ljust(64, "0")[:64],
                "overall_decision": decision,
                "answer_unique": answer_unique,
            }

        adjudications = [
            row("a1", "target_trigger", "source_1", decision="reject", answer_unique=False),
            row("a2", "target_trigger", "source_2", decision="reject", answer_unique=False),
            row("b1", "target_mixed", "source_3", decision="reject", answer_unique=False),
            row("b2", "target_mixed", "source_4", decision="accept", answer_unique=True),
            row("c1", "target_single", "source_5", decision="reject", answer_unique=False),
            row("d1", "target_same_source", "source_6", decision="reject", answer_unique=False),
            row("d2", "target_same_source", "source_6", decision="reject", answer_unique=False),
            row(
                "n1",
                "target_neutral",
                "source_7",
                decision="reject",
                answer_unique=False,
                kind="neutral",
            ),
            row(
                "n2",
                "target_neutral",
                "source_8",
                decision="reject",
                answer_unique=False,
                kind="neutral",
            ),
        ]

        evidence = resolver._target_ineligibility_evidence(adjudications)

        self.assertEqual(list(evidence), ["target_trigger"])
        target = evidence["target_trigger"]
        self.assertEqual(target["published_distractor_judgment_count"], 2)
        self.assertEqual(target["distinct_distractor_candidate_count"], 2)
        self.assertEqual(target["distinct_distractor_source_count"], 2)
        self.assertEqual(target["candidate_ids"], ["a1", "a2"])
        self.assertEqual(target["source_base_fact_ids"], ["source_1", "source_2"])
        self.assertEqual(len(target["adjudication_row_sha256s"]), 2)
        self.assertTrue(target["all_published_distractor_judgments_non_accept"])
        self.assertTrue(target["all_published_distractor_answer_unique_false"])
        self.assertTrue(target["proxy_review_only"])
        self.assertFalse(target["human_gold"])
        self.assertEqual(
            resolver._target_ineligibility_evidence(list(reversed(adjudications))),
            evidence,
        )

    def make_materialization_fixture(self, workspace, group_sizes):
        root_path = workspace / "root-postreview.json"
        root_manifest = {
            "schema_version": resolver.postreview.MANIFEST_SCHEMA_VERSION,
            "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
            "inputs": {
                "full_base_facts": {"path": "unused-full.jsonl"},
                "fact_review": {"apply_manifest": {"path": "unused-apply.json"}},
                "fact_resolution": None,
                "semantic_review": {
                    "candidate_manifest": {"path": "unused-semantic.json"},
                    "adjudications": {"path": "unused-semantic-review.jsonl"},
                },
                "frozen_split_constraint": None,
            },
            "fact_review_contract": {
                "mode": "legacy_fact_review_apply",
                "scope_exactly_matches_input_universe": True,
                "selection_is_complete_retained_cohort": True,
                "source_universe_base_fact_count": sum(group_sizes.values()),
                "accepted_facts_retained": sum(group_sizes.values()),
                "revised_facts_retained_after_independent_rereview": 0,
                "facts_cohort_excluded": 0,
                "revise_defer_missing_count": 0,
                "proxy_review_only": True,
                "human_gold": False,
            },
            "semantic_review_contract": {
                "candidate_adjudication_complete_for_bounded_set": True,
                "candidate_generation_is_bounded": True,
                "all_record_pairs_enumerated": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "semantic_closure_complete_for_full_pool": False,
                "reviewer_evidence_is_human_gold": False,
            },
            "policy_reuse": {},
            "review_invalidation": {},
        }
        write_json(root_path, root_manifest)
        root = root_context(group_sizes)
        root.update(
            {
                "root": {
                    "path": root_path.resolve(),
                    "manifest": root_manifest,
                    "binding": resolver._binding(
                        root_path,
                        schema_version=resolver.postreview.MANIFEST_SCHEMA_VERSION,
                    ),
                },
                "source_path": workspace / "unused-full.jsonl",
                "source_binding": {"path": "unused-full.jsonl"},
                "fact_exclusions": [],
                "fact_excluded_ids": [],
                "original_ids": sorted(root["fact_retained_ids"]),
                "revised_ids": [],
                "fact_mode": "legacy_fact_review_apply",
            }
        )
        relation_by_fact = {
            base_fact_id: {
                "relation_partition_id": row["relation_partition_id"],
                "answer_type_bucket": row["answer_type_bucket"],
            }
            for base_fact_id, row in root["source_index"].items()
        }
        component_by_fact = {
            row["base_fact_id"]: row["leakage_component_id"]
            for row in root["source_rows"]
        }
        split_by_fact = {
            row["base_fact_id"]: "development" for row in root["source_rows"]
        }
        distractors, neutrals, selection = resolver.pre_hf.select_static_candidates(
            root["source_rows"],
            relation_by_fact,
            component_by_fact,
            split_by_fact,
        )
        distractor_ids = selection.pop("distractor_ids")
        neutral_ids = selection.pop("neutral_ids")
        shortfalls = resolver.postreview._candidate_shortfalls(
            base_fact_ids=list(root["source_index"]),
            rows_by_id=root["source_index"],
            distractor_ids=distractor_ids,
            neutral_ids=neutral_ids,
        )
        predecessor = {
            "path": root_path.resolve(),
            "manifest": root_manifest,
            "binding": resolver._binding(
                root_path,
                schema_version=resolver.postreview.MANIFEST_SCHEMA_VERSION,
            ),
            "rows": sorted(root["source_rows"], key=lambda row: row["base_fact_id"]),
            "row_index": root["source_index"],
            "exclusions": [],
            "exclusion_index": {},
            "components": root["semantic_components"],
            "shortfalls": shortfalls,
            "distractors": distractors,
            "neutrals": neutrals,
        }
        return root_path, root, predecessor

    def make_review_context(self, workspace, name, adjudications):
        review_path = workspace / f"{name}.json"
        adjudications_path = workspace / f"{name}-adjudications.jsonl"
        checkpoint_path = workspace / f"{name}-checkpoint.jsonl"
        write_json(
            review_path,
            {
                "schema_version": resolver.candidate_review.RUN_MANIFEST_SCHEMA,
                "status": "completed",
            },
        )
        resolver.pre_hf.write_jsonl(adjudications_path, adjudications)
        resolver.pre_hf.write_jsonl(checkpoint_path, [])
        return review_path, {
            "manifest_binding": resolver._binding(
                review_path,
                schema_version=resolver.candidate_review.RUN_MANIFEST_SCHEMA,
            ),
            "adjudications_binding": resolver._binding(
                adjudications_path,
                schema_version=resolver.candidate_review.ADJUDICATION_SCHEMA,
                record_count=len(adjudications),
            ),
            "adjudications_path": adjudications_path.resolve(),
            "checkpoint_path": checkpoint_path.resolve(),
            "checkpoint_binding": resolver._binding(
                checkpoint_path,
                schema_version=resolver.candidate_review.CHECKPOINT_SCHEMA,
                record_count=0,
            ),
            "adjudications": adjudications,
        }

    @staticmethod
    def candidate_adjudication(
        candidate,
        *,
        overall_decision="reject",
        answer_unique=False,
    ):
        return {
            "candidate_id": candidate["distractor_id"],
            "candidate_kind": "distractor",
            "target_base_fact_id": candidate["base_fact_id"],
            "source_base_fact_id": candidate["source_base_fact_id"],
            "candidate_row_sha256": resolver.pre_hf.sha256_value(candidate),
            "overall_decision": overall_decision,
            "answer_unique": answer_unique,
        }

    def test_pair_ban_selects_replacement_without_relaxing_policy(self):
        root = root_context({"relation_a": 4, "relation_b": 3, "relation_c": 3})
        banned = {("distractor", "relation_a_0", "relation_a_1")}
        state = resolver._resolve_fixed_point(
            root=root,
            excluded_pairs=banned,
            prior_candidate_excluded_ids=[],
        )
        self.assertEqual(state["candidate_excluded_ids"], [])
        selected = {
            ("distractor", row["base_fact_id"], row["source_base_fact_id"])
            for row in state["distractors"]
        }
        self.assertNotIn(next(iter(banned)), selected)
        self.assertEqual(len(state["distractors"]), 20)
        self.assertEqual(len(state["neutrals"]), 20)
        self.assertTrue(
            all(row["same_split"] is True for row in state["distractors"])
        )
        self.assertTrue(
            all(
                row["same_leakage_component"] is False
                for row in [*state["distractors"], *state["neutrals"]]
            )
        )

    def test_structural_shortfall_cascades_to_fixed_point_cohort_exclusion(self):
        root = root_context({"relation_a": 3, "relation_b": 3, "relation_small": 2})
        state = resolver._resolve_fixed_point(
            root=root,
            excluded_pairs=set(),
            prior_candidate_excluded_ids=[],
        )
        self.assertEqual(
            state["candidate_excluded_ids"],
            ["relation_small_0", "relation_small_1"],
        )
        self.assertEqual(len(state["rounds"]), 1)
        self.assertEqual(len(state["retained_ids"]), 6)
        self.assertEqual(len(state["distractors"]), 12)
        self.assertEqual(len(state["neutrals"]), 12)
        self.assertEqual(state["selection"]["dual_distractor_fact_count"], 6)
        self.assertEqual(state["selection"]["dual_neutral_fact_count"], 6)

    def test_target_ineligibility_seeds_structural_fixed_point_cascade(self):
        root = root_context({"relation_a": 3, "relation_b": 3, "relation_c": 3})
        state = resolver._resolve_fixed_point(
            root=root,
            excluded_pairs=set(),
            prior_candidate_excluded_ids=[],
            target_ineligible_ids=["relation_a_0"],
        )

        self.assertEqual(
            state["candidate_excluded_ids"],
            ["relation_a_0", "relation_a_1", "relation_a_2"],
        )
        self.assertEqual(len(state["rounds"]), 1)
        self.assertEqual(
            set(state["latest_shortfall_by_id"]),
            {"relation_a_1", "relation_a_2"},
        )
        self.assertNotIn("relation_a_0", state["latest_shortfall_by_id"])
        self.assertEqual(len(state["retained_ids"]), 6)

    def test_rejected_pair_can_force_target_group_exclusion_without_source_falsehood(self):
        root = root_context({"relation_a": 4, "relation_b": 3, "relation_c": 3})
        state = resolver._resolve_fixed_point(
            root=root,
            excluded_pairs={
                ("distractor", "relation_b_0", "relation_b_1")
            },
            prior_candidate_excluded_ids=[],
        )
        self.assertEqual(
            state["candidate_excluded_ids"],
            ["relation_b_0", "relation_b_1", "relation_b_2"],
        )
        self.assertEqual(len(state["retained_ids"]), 7)
        self.assertEqual(len(state["distractors"]), 14)
        self.assertEqual(len(state["neutrals"]), 14)
        retained_sources = {
            str(row["source_base_fact_id"])
            for row in [*state["distractors"], *state["neutrals"]]
        }
        self.assertTrue(retained_sources.isdisjoint(state["candidate_excluded_ids"]))

    def test_target_ineligibility_materializes_and_replays_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            original_load_postreview = resolver._load_postreview
            root_path, root, predecessor = self.make_materialization_fixture(
                workspace,
                {"relation_a": 3, "relation_b": 4, "relation_c": 3},
            )
            target_id = "relation_a_0"
            published = [
                row
                for row in predecessor["distractors"]
                if row["base_fact_id"] == target_id
            ]
            self.assertEqual(len(published), 2)
            self.assertEqual(
                len({row["source_base_fact_id"] for row in published}), 2
            )
            adjudications = []
            for row in predecessor["distractors"]:
                is_target = row["base_fact_id"] == target_id
                adjudications.append(
                    self.candidate_adjudication(
                        row,
                        overall_decision="reject" if is_target else "accept",
                        answer_unique=False if is_target else True,
                    )
                )
            for row in predecessor["neutrals"]:
                adjudications.append(
                    {
                        "candidate_id": row["neutral_candidate_id"],
                        "candidate_kind": "neutral",
                        "target_base_fact_id": row["base_fact_id"],
                        "source_base_fact_id": row["source_base_fact_id"],
                        "candidate_row_sha256": resolver.pre_hf.sha256_value(row),
                        "overall_decision": "accept",
                        "answer_unique": None,
                    }
                )
            review_path, review_context = self.make_review_context(
                workspace, "target-review", adjudications
            )
            resolution_dir = workspace / "target-resolution"

            with mock.patch.object(
                resolver, "_load_postreview", return_value=predecessor
            ), mock.patch.object(
                resolver, "_prior_resolution", return_value=None
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ), mock.patch.object(
                resolver, "_load_candidate_review", return_value=review_context
            ):
                result = resolver.materialize_resolution(
                    predecessor_postreview_manifest_path=root_path,
                    candidate_review_manifest_path=review_path,
                    output_dir=resolution_dir,
                )
                context = resolver.load_resolution_context(
                    resolution_manifest_path=Path(result["manifest_path"]),
                    predecessor_postreview_manifest_path=root_path,
                )
                successor = resolver.materialize_successor(
                    predecessor_postreview_manifest_path=root_path,
                    candidate_resolution_manifest_path=Path(result["manifest_path"]),
                    output_dir=workspace / "target-successor",
                )

            self.assertEqual(context["new_target_ineligible_ids"], [target_id])
            self.assertEqual(
                context["candidate_excluded_ids"],
                ["relation_a_0", "relation_a_1", "relation_a_2"],
            )
            self.assertEqual(context["plan"]["fixed_point_round_count"], 1)
            self.assertEqual(
                context["plan"]["new_target_ineligibility_exclusion_count"], 1
            )
            target_row = context["cohort_exclusion_rows"][0]
            self.assertEqual(
                target_row["exclusion_stage"], resolver.TARGET_INELIGIBILITY_STAGE
            )
            self.assertEqual(
                target_row["exclusion_reason"], resolver.TARGET_INELIGIBILITY_REASON
            )
            evidence = target_row["target_ineligibility_evidence"]
            self.assertEqual(evidence["published_distractor_judgment_count"], 2)
            self.assertEqual(
                evidence["candidate_ids"],
                sorted(row["distractor_id"] for row in published),
            )
            self.assertEqual(len(evidence["adjudication_row_sha256s"]), 2)
            self.assertFalse(target_row["factual_falsehood_asserted"])
            self.assertFalse(target_row["human_gold"])
            self.assertEqual(
                [row["exclusion_stage"] for row in context["cohort_exclusion_rows"]],
                [
                    resolver.TARGET_INELIGIBILITY_STAGE,
                    "candidate_availability_resolution",
                    "candidate_availability_resolution",
                ],
            )
            self.assertEqual(result["candidate_target_ineligibility_exclusion_count"], 1)
            self.assertEqual(successor["retained_base_fact_count"], 7)
            self.assertEqual(
                successor["candidate_target_ineligibility_exclusion_count"], 1
            )
            successor_rows = resolver.pre_hf.read_jsonl(
                Path(successor["manifest_path"]).parent / "full_base_facts.jsonl"
            )
            self.assertNotIn(
                target_id, {str(row["base_fact_id"]) for row in successor_rows}
            )
            successor_manifest = resolver.pre_hf.read_json(
                Path(successor["manifest_path"])
            )
            self.assertFalse(successor_manifest["safety_contract"]["human_gold"])
            for field in (
                "hf_model_executed",
                "hf_tokenizer_executed",
                "behavior_executed",
                "validation_exposed",
                "sealed_exposed",
            ):
                self.assertFalse(successor_manifest["safety_contract"][field])

            successor_path = Path(successor["manifest_path"])
            successor_predecessor = original_load_postreview(successor_path)
            pair_target = "relation_b_0"
            pair_candidate = next(
                row
                for row in successor_predecessor["distractors"]
                if row["base_fact_id"] == pair_target
            )
            round_two_adjudications = []
            for row in successor_predecessor["distractors"]:
                rejected = row["distractor_id"] == pair_candidate["distractor_id"]
                round_two_adjudications.append(
                    self.candidate_adjudication(
                        row,
                        overall_decision="reject" if rejected else "accept",
                        answer_unique=True,
                    )
                )
            for row in successor_predecessor["neutrals"]:
                round_two_adjudications.append(
                    {
                        "candidate_id": row["neutral_candidate_id"],
                        "candidate_kind": "neutral",
                        "target_base_fact_id": row["base_fact_id"],
                        "source_base_fact_id": row["source_base_fact_id"],
                        "candidate_row_sha256": resolver.pre_hf.sha256_value(row),
                        "overall_decision": "accept",
                        "answer_unique": None,
                    }
                )
            review_two_path, review_two_context = self.make_review_context(
                workspace, "pair-review", round_two_adjudications
            )

            def load_round_postreview(path):
                resolved = Path(path).resolve()
                if resolved == root_path.resolve():
                    return predecessor
                if resolved == successor_path.resolve():
                    return successor_predecessor
                return original_load_postreview(path)

            review_contexts = {
                review_path.resolve(): review_context,
                review_two_path.resolve(): review_two_context,
            }

            def load_round_review(*, predecessor_path, review_manifest_path, **_):
                del predecessor_path
                return review_contexts[Path(review_manifest_path).resolve()]

            with mock.patch.object(
                resolver, "_load_postreview", side_effect=load_round_postreview
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ), mock.patch.object(
                resolver, "_load_candidate_review", side_effect=load_round_review
            ):
                resolution_two = resolver.materialize_resolution(
                    predecessor_postreview_manifest_path=successor_path,
                    candidate_review_manifest_path=review_two_path,
                    output_dir=workspace / "pair-resolution",
                )
                context_two = resolver.load_resolution_context(
                    resolution_manifest_path=Path(resolution_two["manifest_path"]),
                    predecessor_postreview_manifest_path=successor_path,
                )
                successor_two = resolver.materialize_successor(
                    predecessor_postreview_manifest_path=successor_path,
                    candidate_resolution_manifest_path=Path(
                        resolution_two["manifest_path"]
                    ),
                    output_dir=workspace / "pair-successor",
                )

            self.assertEqual(
                context_two["previous_candidate_excluded_ids"],
                ["relation_a_0", "relation_a_1", "relation_a_2"],
            )
            self.assertEqual(context_two["new_target_ineligible_ids"], [])
            self.assertEqual(context_two["all_target_ineligible_ids"], [target_id])
            self.assertEqual(
                context_two["candidate_excluded_ids"],
                ["relation_a_0", "relation_a_1", "relation_a_2"],
            )
            self.assertEqual(
                context_two["cohort_exclusion_rows"][0],
                context["cohort_exclusion_rows"][0],
            )
            self.assertEqual(
                successor_two["candidate_target_ineligibility_exclusion_count"], 1
            )

            pristine = resolver.pre_hf.read_json(Path(result["manifest_path"]))
            tampered = copy.deepcopy(pristine)
            tampered["target_ineligibility_contract"][
                "new_target_ineligibility_base_fact_ids"
            ] = []
            resolver.pre_hf.write_json(Path(result["manifest_path"]), tampered)
            with mock.patch.object(
                resolver, "_load_postreview", return_value=predecessor
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ), mock.patch.object(
                resolver, "_load_candidate_review", return_value=review_context
            ), self.assertRaisesRegex(ValueError, "target-ineligibility contract is stale"):
                resolver.load_resolution_context(
                    resolution_manifest_path=Path(result["manifest_path"]),
                    predecessor_postreview_manifest_path=root_path,
                )
            resolver.pre_hf.write_json(Path(result["manifest_path"]), pristine)

    def test_structural_resolution_materializes_replayable_successor(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root_path = workspace / "root-postreview.json"
            root_manifest = {
                "schema_version": resolver.postreview.MANIFEST_SCHEMA_VERSION,
                "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
                "inputs": {
                    "full_base_facts": {"path": "unused-full.jsonl"},
                    "fact_review": {"apply_manifest": {"path": "unused-apply.json"}},
                    "fact_resolution": None,
                    "semantic_review": {
                        "candidate_manifest": {"path": "unused-semantic.json"},
                        "adjudications": {"path": "unused-semantic-review.jsonl"},
                    },
                    "frozen_split_constraint": None,
                },
                "fact_review_contract": {
                    "mode": "legacy_fact_review_apply",
                    "scope_exactly_matches_input_universe": True,
                    "selection_is_complete_retained_cohort": True,
                    "source_universe_base_fact_count": 8,
                    "accepted_facts_retained": 8,
                    "revised_facts_retained_after_independent_rereview": 0,
                    "facts_cohort_excluded": 0,
                    "revise_defer_missing_count": 0,
                    "proxy_review_only": True,
                    "human_gold": False,
                },
                "semantic_review_contract": {
                    "candidate_adjudication_complete_for_bounded_set": True,
                    "candidate_generation_is_bounded": True,
                    "all_record_pairs_enumerated": False,
                    "semantic_near_duplicate_recall_guaranteed": False,
                    "semantic_closure_complete_for_full_pool": False,
                    "reviewer_evidence_is_human_gold": False,
                },
                "policy_reuse": {},
                "review_invalidation": {},
            }
            write_json(root_path, root_manifest)
            root = root_context(
                {"relation_a": 3, "relation_b": 3, "relation_small": 2}
            )
            root.update(
                {
                    "root": {
                        "path": root_path,
                        "manifest": root_manifest,
                        "binding": resolver._binding(
                            root_path,
                            schema_version=resolver.postreview.MANIFEST_SCHEMA_VERSION,
                        ),
                    },
                    "source_path": workspace / "unused-full.jsonl",
                    "source_binding": {"path": "unused-full.jsonl"},
                    "fact_exclusions": [],
                    "fact_excluded_ids": [],
                    "original_ids": sorted(root["fact_retained_ids"]),
                    "revised_ids": [],
                    "fact_mode": "legacy_fact_review_apply",
                }
            )
            initial = resolver._resolve_fixed_point(
                root=root,
                excluded_pairs=set(),
                prior_candidate_excluded_ids=[],
            )
            # The predecessor has the pre-repair candidate set, including the
            # two structurally short relation members.
            relation_by_fact = {
                base_fact_id: {
                    "relation_partition_id": row["relation_partition_id"],
                    "answer_type_bucket": row["answer_type_bucket"],
                }
                for base_fact_id, row in root["source_index"].items()
            }
            component_by_fact = {
                row["base_fact_id"]: row["leakage_component_id"]
                for row in root["source_rows"]
            }
            split_by_fact = {
                row["base_fact_id"]: "development" for row in root["source_rows"]
            }
            old_distractors, old_neutrals, old_selection = (
                resolver.pre_hf.select_static_candidates(
                    root["source_rows"],
                    relation_by_fact,
                    component_by_fact,
                    split_by_fact,
                )
            )
            old_distractor_ids = old_selection.pop("distractor_ids")
            old_neutral_ids = old_selection.pop("neutral_ids")
            shortfalls = resolver.postreview._candidate_shortfalls(
                base_fact_ids=list(root["source_index"]),
                rows_by_id=root["source_index"],
                distractor_ids=old_distractor_ids,
                neutral_ids=old_neutral_ids,
            )
            predecessor = {
                "path": root_path,
                "manifest": root_manifest,
                "binding": resolver._binding(
                    root_path,
                    schema_version=resolver.postreview.MANIFEST_SCHEMA_VERSION,
                ),
                "rows": sorted(
                    root["source_rows"], key=lambda row: row["base_fact_id"]
                ),
                "row_index": root["source_index"],
                "exclusions": [],
                "exclusion_index": {},
                "components": root["semantic_components"],
                "shortfalls": shortfalls,
                "distractors": old_distractors,
                "neutrals": old_neutrals,
            }
            resolution_dir = workspace / "resolution"
            with mock.patch.object(
                resolver, "_load_postreview", return_value=predecessor
            ), mock.patch.object(
                resolver, "_prior_resolution", return_value=None
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ):
                result = resolver.materialize_resolution(
                    predecessor_postreview_manifest_path=root_path,
                    output_dir=resolution_dir,
                )
            self.assertEqual(result["candidate_cohort_exclusion_count"], 2)

            successor_dir = workspace / "successor"
            with mock.patch.object(
                resolver, "_load_postreview", return_value=predecessor
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ):
                successor = resolver.materialize_successor(
                    predecessor_postreview_manifest_path=root_path,
                    candidate_resolution_manifest_path=Path(result["manifest_path"]),
                    output_dir=successor_dir,
                )
            self.assertEqual(successor["retained_base_fact_count"], 6)
            self.assertEqual(successor["candidate_stage_cohort_excluded_count"], 2)
            self.assertEqual(successor["candidate_shortfall_count"], 0)
            successor_manifest = resolver.pre_hf.read_json(
                Path(successor["manifest_path"])
            )
            self.assertEqual(
                successor_manifest["schema_version"],
                resolver.SUCCESSOR_POSTREVIEW_MANIFEST_SCHEMA,
            )
            self.assertFalse(
                successor_manifest["candidate_repair_contract"][
                    "predecessor_candidate_reviews_blanket_valid"
                ]
            )
            self.assertTrue(
                successor_manifest["candidate_repair_contract"][
                    "exact_v3_accepted_projection_evidence_carry_forward_eligible"
                ]
            )
            self.assertTrue(
                successor_manifest["candidate_repair_contract"][
                    "full_candidate_evidence_coverage_required"
                ]
            )
            self.assertFalse(
                successor_manifest["candidate_repair_contract"][
                    "full_candidate_model_rereview_required"
                ]
            )
            self.assertTrue(
                successor_manifest["candidate_repair_contract"][
                    "changed_or_new_candidate_fresh_model_review_required"
                ]
            )
            self.assertNotIn(
                "full_candidate_rereview_required",
                successor_manifest["candidate_repair_contract"],
            )
            self.assertFalse(
                successor_manifest["candidate_repair_contract"][
                    "predecessor_zh_reviews_valid"
                ]
            )
            _, kind, _, units = resolver.candidate_review.load_review_units(
                candidate_manifest_path=Path(successor["manifest_path"])
            )
            self.assertEqual(kind, "postreview_rebuild")
            self.assertEqual(len(units), 24)
            self.assertEqual(initial["candidate_excluded_ids"], [
                "relation_small_0",
                "relation_small_1",
            ])

            replay_artifacts = {
                name: {
                    "path": successor_dir / artifact["filename"],
                    "binding": artifact,
                }
                for name, artifact in successor_manifest["outputs"].items()
            }
            freezer_predecessor = {
                **predecessor,
                "path": Path(predecessor["path"]).resolve(),
            }
            freezer_root = {
                **root,
                "root": {
                    **root["root"],
                    "path": Path(root["root"]["path"]).resolve(),
                },
            }
            with mock.patch.object(
                freezer.candidate_resolution_tool,
                "_load_postreview",
                return_value=freezer_predecessor,
            ), mock.patch.object(
                freezer.candidate_resolution_tool,
                "_load_root_context",
                return_value=freezer_root,
            ):
                freezer._deterministically_replay_postreview(
                    manifest_path=Path(successor["manifest_path"]),
                    manifest=successor_manifest,
                    input_rows=root["source_rows"],
                    artifacts=replay_artifacts,
                    split_seed="fixture-repair-seed",
                )

            resolution_path = Path(result["manifest_path"])
            current_resolution = resolver.pre_hf.read_json(resolution_path)
            legacy_resolution = copy.deepcopy(current_resolution)
            legacy_resolution["tool_version"] = resolver.LEGACY_TOOL_VERSION
            legacy_resolution.pop("target_ineligibility_contract")
            legacy_resolution["counts"].pop(
                "candidate_target_ineligibility_exclusion_count"
            )
            legacy_resolution["counts"].pop(
                "new_candidate_target_ineligibility_exclusion_count"
            )
            for field in (
                "new_target_ineligibility_exclusion_count",
                "ordered_new_target_ineligibility_base_fact_ids_sha256",
                "ordered_target_ineligibility_evidence_sha256",
            ):
                legacy_resolution["resolution_plan"].pop(field)
            legacy_resolution["resolution_id"] = (
                "candidate_resolution_"
                + resolver.pre_hf.sha256_value(
                    {
                        "root_postreview_sha256": root["root"]["binding"]["sha256"],
                        "predecessor_postreview_sha256": predecessor["binding"][
                            "sha256"
                        ],
                        "plan": legacy_resolution["resolution_plan"],
                    }
                )[:24]
            )
            resolver.pre_hf.write_json(resolution_path, legacy_resolution)
            legacy_successor_dir = workspace / "legacy-successor"
            with mock.patch.object(
                resolver, "_load_postreview", return_value=predecessor
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ):
                resolver.materialize_successor(
                    predecessor_postreview_manifest_path=root_path,
                    candidate_resolution_manifest_path=resolution_path,
                    output_dir=legacy_successor_dir,
                )
            legacy_manifest = resolver.pre_hf.read_json(
                legacy_successor_dir / "postreview_rebuild_manifest.json"
            )
            legacy_summary = resolver.pre_hf.read_json(
                legacy_successor_dir / "summary.json"
            )
            legacy_split = resolver.pre_hf.read_json(
                legacy_successor_dir / "split_manifest.json"
            )
            legacy_integrity = resolver.pre_hf.read_json(
                legacy_successor_dir / "integrity_audit.json"
            )
            self.assertEqual(
                legacy_manifest["tool_version"], resolver.LEGACY_TOOL_VERSION
            )
            self.assertEqual(
                legacy_summary["tool_version"], resolver.LEGACY_TOOL_VERSION
            )
            self.assertEqual(
                legacy_split["tool_version"], resolver.LEGACY_TOOL_VERSION
            )
            self.assertNotIn(
                "candidate_target_ineligibility_exclusion_count",
                legacy_manifest["candidate_repair_contract"],
            )
            self.assertNotIn(
                "ordered_candidate_target_ineligible_base_fact_ids_sha256",
                legacy_manifest["lineage_digests"],
            )
            self.assertNotIn(
                "candidate_target_ineligibility_exclusion_count",
                legacy_integrity["candidate_repair"],
            )
            resolver.pre_hf.write_json(resolution_path, current_resolution)

    def test_cumulative_resolution_preserves_prior_exclusions_and_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root_path, root, root_predecessor = self.make_materialization_fixture(
                workspace,
                {
                    "relation_a": 4,
                    "relation_b": 3,
                    "relation_c": 3,
                    "relation_small": 2,
                },
            )
            original_load_postreview = resolver._load_postreview

            def load_postreview(path):
                if Path(path).resolve() == root_path.resolve():
                    return root_predecessor
                return original_load_postreview(path)

            resolution_1_dir = workspace / "resolution-1"
            with mock.patch.object(
                resolver, "_load_postreview", side_effect=load_postreview
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ):
                resolution_1 = resolver.materialize_resolution(
                    predecessor_postreview_manifest_path=root_path,
                    output_dir=resolution_1_dir,
                )
                successor_1 = resolver.materialize_successor(
                    predecessor_postreview_manifest_path=root_path,
                    candidate_resolution_manifest_path=Path(
                        resolution_1["manifest_path"]
                    ),
                    output_dir=workspace / "successor-1",
                )

            successor_1_path = Path(successor_1["manifest_path"])
            successor_1_predecessor = original_load_postreview(successor_1_path)
            review_path = workspace / "candidate-review.json"
            adjudications_path = workspace / "candidate-adjudications.jsonl"
            checkpoint_path = workspace / "candidate-checkpoint.jsonl"
            write_json(
                review_path,
                {
                    "schema_version": resolver.candidate_review.RUN_MANIFEST_SCHEMA,
                    "status": "completed",
                },
            )
            adjudications = [
                {
                    "candidate_id": "d_relation_b_0_relation_b_1",
                    "candidate_kind": "distractor",
                    "target_base_fact_id": "relation_b_0",
                    "source_base_fact_id": "relation_b_1",
                    "candidate_row_sha256": "1" * 64,
                    "overall_decision": "reject",
                },
                {
                    "candidate_id": "d_relation_a_0_relation_a_1",
                    "candidate_kind": "distractor",
                    "target_base_fact_id": "relation_a_0",
                    "source_base_fact_id": "relation_a_1",
                    "candidate_row_sha256": "2" * 64,
                    "overall_decision": "defer",
                },
            ]
            resolver.pre_hf.write_jsonl(adjudications_path, adjudications)
            resolver.pre_hf.write_jsonl(checkpoint_path, [])
            review_context = {
                "manifest_binding": resolver._binding(
                    review_path,
                    schema_version=resolver.candidate_review.RUN_MANIFEST_SCHEMA,
                ),
                "adjudications_binding": resolver._binding(
                    adjudications_path,
                    schema_version=resolver.candidate_review.ADJUDICATION_SCHEMA,
                    record_count=len(adjudications),
                ),
                "adjudications_path": adjudications_path.resolve(),
                "checkpoint_path": checkpoint_path.resolve(),
                "checkpoint_binding": resolver._binding(
                    checkpoint_path,
                    schema_version=resolver.candidate_review.CHECKPOINT_SCHEMA,
                    record_count=0,
                ),
                "adjudications": adjudications,
            }

            def load_round_postreview(path):
                resolved = Path(path).resolve()
                if resolved == root_path.resolve():
                    return root_predecessor
                if resolved == successor_1_path.resolve():
                    return successor_1_predecessor
                return original_load_postreview(path)

            resolution_2_dir = workspace / "resolution-2"
            with mock.patch.object(
                resolver, "_load_postreview", side_effect=load_round_postreview
            ), mock.patch.object(
                resolver, "_load_root_context", return_value=root
            ), mock.patch.object(
                resolver, "_load_candidate_review", return_value=review_context
            ):
                resolution_2 = resolver.materialize_resolution(
                    predecessor_postreview_manifest_path=successor_1_path,
                    candidate_review_manifest_path=review_path,
                    output_dir=resolution_2_dir,
                )
                context_2 = resolver.load_resolution_context(
                    resolution_manifest_path=Path(resolution_2["manifest_path"]),
                    predecessor_postreview_manifest_path=successor_1_path,
                )
                successor_2 = resolver.materialize_successor(
                    predecessor_postreview_manifest_path=successor_1_path,
                    candidate_resolution_manifest_path=Path(
                        resolution_2["manifest_path"]
                    ),
                    output_dir=workspace / "successor-2",
                )

            self.assertEqual(
                context_2["previous_candidate_excluded_ids"],
                ["relation_small_0", "relation_small_1"],
            )
            self.assertEqual(
                context_2["candidate_excluded_ids"],
                [
                    "relation_b_0",
                    "relation_b_1",
                    "relation_b_2",
                    "relation_small_0",
                    "relation_small_1",
                ],
            )
            self.assertEqual(len(context_2["pair_rows"]), 2)
            self.assertEqual(successor_2["candidate_stage_cohort_excluded_count"], 5)
            resolution_2_manifest = resolver.pre_hf.read_json(
                Path(resolution_2["manifest_path"])
            )
            self.assertEqual(
                resolution_2_manifest["schema_version"],
                resolver.RESOLUTION_MANIFEST_SCHEMA,
            )
            self.assertEqual(
                resolution_2_manifest["inputs"]["candidate_review_checkpoint"][
                    "sha256"
                ],
                review_context["checkpoint_binding"]["sha256"],
            )
            self.assertEqual(
                {
                    field: resolution_2_manifest["review_invalidation"][field]
                    for field in resolver.CANDIDATE_EVIDENCE_REUSE_POLICY
                },
                resolver.CANDIDATE_EVIDENCE_REUSE_POLICY,
            )
            self.assertEqual(
                resolution_2_manifest["inputs"][
                    "previous_candidate_resolution_manifest"
                ]["sha256"],
                resolution_1["manifest_sha256"],
            )
            successor_2_manifest = resolver.pre_hf.read_json(
                Path(successor_2["manifest_path"])
            )
            self.assertEqual(
                Path(
                    successor_2_manifest["inputs"]["candidate_repair"][
                        "predecessor_postreview_manifest"
                    ]["path"]
                ).resolve(),
                successor_1_path.resolve(),
            )

            pristine = copy.deepcopy(resolution_2_manifest)
            for field, message in (
                ("candidate_review_manifest", "review manifest binding is stale"),
                (
                    "previous_candidate_resolution_manifest",
                    "previous candidate resolution manifest SHA-256 mismatch",
                ),
            ):
                tampered = copy.deepcopy(pristine)
                tampered["inputs"][field]["sha256"] = "0" * 64
                write_json(Path(resolution_2["manifest_path"]), tampered)
                with mock.patch.object(
                    resolver, "_load_postreview", side_effect=load_round_postreview
                ), mock.patch.object(
                    resolver, "_load_root_context", return_value=root
                ), mock.patch.object(
                    resolver,
                    "_load_candidate_review",
                    return_value=review_context,
                ), self.assertRaisesRegex(ValueError, message):
                    resolver.load_resolution_context(
                        resolution_manifest_path=Path(
                            resolution_2["manifest_path"]
                        ),
                        predecessor_postreview_manifest_path=successor_1_path,
                    )
                write_json(Path(resolution_2["manifest_path"]), pristine)


if __name__ == "__main__":
    unittest.main()
