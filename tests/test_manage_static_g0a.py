import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "manage_static_g0a.py"
SPEC = importlib.util.spec_from_file_location("manage_static_g0a", SCRIPT_PATH)
g0a = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = g0a
SPEC.loader.exec_module(g0a)


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class StaticG0ATests(unittest.TestCase):
    def test_manager_is_bound_to_semantic_aware_v3_split_policy(self):
        self.assertEqual(
            g0a.EXPECTED_SPLIT_POLICY,
            "relation-stratified-lexical-semantic-component-greedy-sha256-v1",
        )

    def source_row(self, index, split):
        base_fact_id = f"pbf_{index}"
        answer = f"Answer {index}"
        return {
            "schema_version": g0a.FINAL_BUNDLE_SCHEMA,
            "base_fact_id": base_fact_id,
            "base_id": base_fact_id,
            "source_id": f"source_{index}",
            "candidate_id": f"candidate_{index}",
            "source_dataset": "fixture",
            "answer_type": "entity",
            "probe_relation_id": "fixture_relation",
            "prompt_en": f"Subject {index} was linked to",
            "answer_en": answer,
            "answer_aliases_en": [answer],
            "canonical_fact": f"Subject {index} was linked to {answer}.",
            "canonical_fact_en": f"Subject {index} was linked to {answer}.",
            "canonical_status": "pending_review",
            "canonical_freeze_status": "provisional_not_frozen",
            "bundle_status": "postclosure_preperturbation_provisional_not_frozen",
            "evidence_tier": "provisional_single_model",
            "human_gold": False,
            "split_group_id": f"component_{index}",
            "leakage_component_id": f"component_{index}",
            "split_assignment": split,
            "split_policy_version": g0a.EXPECTED_SPLIT_POLICY,
            "split_status": "provisional_not_frozen",
            "prompt_ready": True,
            "prompt_quality_tier": "strict_factual_completion",
            "distractor_candidates": [
                {
                    "schema_version": "public-benchmark-preperturbation-distractor-candidate-v1",
                    "distractor_id": f"d{index}_{rank}",
                    "target_base_fact_id": base_fact_id,
                    "donor_base_fact_id": f"donor_{index}_{rank}",
                    "text_en": f"Wrong {index} {rank}",
                    "answer_en": f"Wrong {index} {rank}",
                    "answer_aliases_en": [f"Wrong {index} {rank}"],
                    "split_assignment": split,
                    "status": "codex_proxy_pending_review",
                    "verification_status": "codex_proxy_pending_review",
                    "verified": False,
                    "human_gold": False,
                }
                for rank in (1, 2)
            ],
            "admission": {"prompt_ready": True, "semantic_review_complete": False},
        }

    def make_source(self, root):
        rows = [
            self.source_row(1, "development"),
            self.source_row(2, "validation"),
            self.source_row(3, "sealed"),
        ]
        # Exercise answer revision propagation through a real donor edge.
        rows[1]["distractor_candidates"][0].update(
            {
                "donor_base_fact_id": "pbf_1",
                "text_en": "Answer 1",
                "answer_en": "Answer 1",
                "answer_aliases_en": ["Answer 1"],
            }
        )
        bundle = root / "source.jsonl"
        g0a.write_jsonl(bundle, rows)
        closure = root / "old_closure.json"
        g0a.write_json(
            closure,
            {
                "schema_version": "public-benchmark-duplicate-closure-resolution-v2",
                "semantic_paraphrase_closure_complete": False,
                "historical_exposure_complete": False,
            },
        )
        manifest = root / "source_manifest.json"
        g0a.write_json(
            manifest,
            {
                "schema_version": g0a.EXPECTED_SOURCE_MANIFEST_SCHEMA,
                "status": g0a.EXPECTED_SOURCE_STATUS,
                "split_policy_version": g0a.EXPECTED_SPLIT_POLICY,
                "safety_contract": {
                    "network_or_model_used": False,
                    "model_inference_performed": False,
                    "canonical_freeze_performed": False,
                    "split_freeze_performed": False,
                    "distractors_are_verified": False,
                },
                "artifacts": {
                    "postclosure_preperturbation_behavior_input_bundle": g0a._binding(
                        bundle,
                        schema_version=g0a.FINAL_BUNDLE_SCHEMA,
                        record_count=len(rows),
                    )
                },
                "inputs": {"closure_resolution_manifest": g0a._binding(closure)},
            },
        )
        return rows, bundle, manifest

    def make_source_decisions(
        self,
        root,
        rows,
        bundle,
        *,
        patches=None,
        distractor_verdicts=None,
        name="source-review",
    ):
        patches = patches or {}
        distractor_verdicts = distractor_verdicts or {}
        bundle_sha = g0a.sha256_file(bundle)
        audit_rows = []
        for row in rows:
            patch = copy.deepcopy(patches.get(row["base_fact_id"]))
            audit_rows.append(
                {
                    "schema_version": g0a.CODEX_FACT_AUDIT_SCHEMA,
                    "input_bundle_sha256": bundle_sha,
                    "base_fact_id": row["base_fact_id"],
                    "source_id": row["source_id"],
                    "split_assignment": row["split_assignment"],
                    "probe_relation_id": row["probe_relation_id"],
                    "decision": "revise" if patch else "accept",
                    "issues": [],
                    "distractors": [
                        {
                            "distractor_id": item["distractor_id"],
                            "rank": rank,
                            "text_en": item["text_en"],
                            "verdict": distractor_verdicts.get(
                                (row["base_fact_id"], item["distractor_id"]),
                                "accept",
                            ),
                            "reason": "fixture source distractor adjudication",
                        }
                        for rank, item in enumerate(
                            row["distractor_candidates"], start=1
                        )
                    ],
                    "blind_to_behavior": True,
                    "reviewer_type": "codex_proxy",
                    "review_status": "codex_adjudicated",
                    "human_gold": False,
                    "proposed_patch": patch,
                    "audit_revision": "fixture-v2",
                }
            )
        audit_path = root / f"{name}.jsonl"
        output_dir = root / f"{name}-adapted"
        g0a.write_jsonl(audit_path, audit_rows)
        g0a.adapt_source_audit(
            input_bundle_path=bundle,
            audit_path=audit_path,
            output_dir=output_dir,
            expected_record_count=len(rows),
        )
        return (
            output_dir / "source_decisions.jsonl",
            output_dir / "source_decisions_manifest.json",
        )

    def stage(self, root):
        rows, bundle, source_manifest = self.make_source(root)
        decisions, decisions_manifest = self.make_source_decisions(root, rows, bundle)
        stage_dir = root / "stage"
        result = g0a.stage_static_g0a(
            input_bundle_path=bundle,
            source_finalization_manifest_path=source_manifest,
            source_decisions_path=decisions,
            source_decisions_manifest_path=decisions_manifest,
            output_dir=stage_dir,
            expected_record_count=3,
        )
        return rows, stage_dir, result

    def make_upstream(self, stage_dir, *, automatic_review_rejects=False):
        staging_manifest = g0a.read_json(stage_dir / "static_staging_manifest.json")
        staging_rows = jsonl(stage_dir / "static_staging_records.jsonl")
        run_manifest = {
            "schema_version": g0a.runtime.SCHEMA_VERSION,
            "run_id": "fixture-static",
            "static_generation_authorized": True,
            "behavior_authorized": False,
            "simulation_authorized": False,
            "simulation_requires_codex_proxy_review": True,
            "simulation_models": [],
            "codex_proxy_review": None,
            "automatic_review_authority": {
                "translation": "advisory",
                "source_audit_accepted_distractor": "advisory",
                "generated_replacement_distractor": "hard_gate",
                "perturbation_context": "advisory",
                "final_authority": "codex_proxy",
            },
            "g0a_static_staging": g0a._binding(
                stage_dir / "static_staging_manifest.json",
                schema_version=g0a.STAGING_MANIFEST_SCHEMA,
            ),
            "input_bundle_sha256": staging_manifest["source_bundle"]["sha256"],
            "prompt_input_sha256": staging_manifest["source_bundle"]["sha256"],
            "input_bundle_record_count": staging_manifest["source_bundle"]["record_count"],
            "selected_count": len(staging_rows),
            "selected_source_ids_sha256": g0a.sha256_value(
                sorted(row["source_id"] for row in staging_rows)
            ),
            "records": [g0a._runtime_record(row) for row in staging_rows],
        }
        run_manifest["source_audit_replacement_generation_request_count"] = sum(
            row["replacement_generation_required_count"]
            for row in run_manifest["records"]
        )
        run_manifest["distractor_replenishment_enabled"] = True
        run_manifest["records_sha256"] = g0a.sha256_value(run_manifest["records"])
        g0a.write_json(stage_dir / "run_manifest.json", run_manifest)
        translations = []
        reviews = []
        distractors = []
        perturbations = []
        for record in staging_rows:
            source_id = record["source_id"]
            translations.append(
                {
                    "source_id": source_id,
                    "terminal_status": "completed",
                    "parsed_response": {
                        "prompt_zh": f"主题 {record['base_fact_id']} 对应",
                        "answer_zh": f"答案 {record['base_fact_id']}",
                        "answer_aliases_zh": [f"答案 {record['base_fact_id']}"],
                        "distractors": [
                            {
                                "distractor_id": item["distractor_id"],
                                "text_zh": f"错误 {item['distractor_id']}",
                            }
                            for item in record["distractors"]
                        ],
                    },
                }
            )
            reviews.append(
                {
                    "source_id": source_id,
                    "terminal_status": "completed",
                    "parsed_response": {
                        "decision": (
                            "reject" if automatic_review_rejects else "accept"
                        ),
                        "checks": {"all": not automatic_review_rejects},
                    },
                }
            )
            for item in record["distractors"]:
                distractor_id = item["distractor_id"]
                distractors.append(
                    {
                        "item_id": distractor_id,
                        "source_id": source_id,
                        "distractor_en": item["text_en"],
                        "distractor_zh": f"错误 {distractor_id}",
                        "terminal_status": "completed",
                        "parsed_response": {
                            "decision": (
                                "reject" if automatic_review_rejects else "accept"
                            ),
                            "checks": {"all": not automatic_review_rejects},
                        },
                    }
                )
                perturbations.append(
                    {
                        "candidate_id": f"ctx_{distractor_id}",
                        "source_id": source_id,
                        "distractor_id": distractor_id,
                        "terminal_status": "completed",
                        "candidate": {
                            "english_context": f"Subject assertion {distractor_id}.",
                            "chinese_context": f"目标断言 {distractor_id}。",
                            "neutral_english_context": f"Neutral context {distractor_id}.",
                            "neutral_chinese_context": f"中性上下文 {distractor_id}。",
                        },
                        "parsed_response": {
                            "decision": (
                                "reject" if automatic_review_rejects else "accept"
                            ),
                            "checks": {"all": not automatic_review_rejects},
                        },
                    }
                )
        g0a.write_jsonl(stage_dir / "translations.jsonl", translations)
        g0a.write_jsonl(stage_dir / "translation_reviews.jsonl", reviews)
        g0a.write_jsonl(stage_dir / "verified_distractors.jsonl", distractors)
        g0a.write_jsonl(stage_dir / "perturbations.jsonl", perturbations)
        g0a.write_jsonl(stage_dir / "simulation_results.jsonl", [])
        g0a.write_json(
            stage_dir / "g0a_upstream_stop_manifest.json",
            {
                "schema_version": "static-g0a-upstream-stop-manifest-v1",
                "status": "static_upstream_complete_simulation_blocked",
                "run_manifest": g0a._binding(
                    stage_dir / "run_manifest.json",
                    schema_version=g0a.runtime.SCHEMA_VERSION,
                ),
                "simulation_result_count": 0,
                "simulation_results": g0a._binding(
                    stage_dir / "simulation_results.jsonl", record_count=0
                ),
                "behavior_authorized": False,
                "simulation_authorized": False,
            },
        )
        review_dir = stage_dir / "review"
        g0a.export_codex_review(
            staging_manifest_path=stage_dir / "static_staging_manifest.json",
            upstream_run_dir=stage_dir,
            output_dir=review_dir,
        )
        return review_dir

    def context_repair_row(self, stage_dir, review_dir):
        staging_by_id = {
            row["base_fact_id"]: row
            for row in jsonl(stage_dir / "static_staging_records.jsonl")
        }
        distractor_by_id = {
            row["item_id"]: row
            for row in jsonl(stage_dir / "verified_distractors.jsonl")
        }
        context_by_id = {
            row["candidate_id"]: row
            for row in jsonl(stage_dir / "perturbations.jsonl")
        }
        review = jsonl(review_dir / "codex_adjudication_input.jsonl")[0]
        variant = review["variants"][0]
        superseded = variant["context_candidates"][0]
        staging = staging_by_id[review["base_fact_id"]]
        distractor = distractor_by_id[variant["distractor_id"]]
        context = context_by_id[superseded["candidate_id"]]
        return {
            "schema_version": g0a.CONTEXT_REPAIR_RECORD_SCHEMA,
            "candidate_id": f"{superseded['candidate_id']}_codex_repair_v1",
            "base_fact_id": review["base_fact_id"],
            "source_id": review["source_id"],
            "staging_record_sha256": g0a.sha256_value(staging),
            "distractor_id": variant["distractor_id"],
            "distractor_record_sha256": g0a.sha256_value(distractor),
            "superseded_candidate_id": superseded["candidate_id"],
            "superseded_candidate_record_sha256": g0a.sha256_value(context),
            "targeted_context_en": superseded["targeted_context_en"],
            "targeted_context_zh": superseded["targeted_context_zh"],
            "neutral_context_en": "A repaired, unrelated, truthful neutral fact.",
            "neutral_context_zh": "一条经过修复、无关且真实的中性事实。",
            "reason": "replace an ambiguous neutral context without behavior evidence",
            "terminal_status": "completed",
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "review_blinded_to_behavior_results": True,
        }

    def completed_decisions(self, review_dir, *, revise_first=True):
        reviews = jsonl(review_dir / "codex_adjudication_input.jsonl")
        rows = []
        for index, review in enumerate(reviews):
            revise = revise_first and index == 0
            rows.append(
                {
                    "schema_version": g0a.CODEX_DECISION_SCHEMA,
                    "base_fact_id": review["base_fact_id"],
                    "review_input_record_sha256": g0a.sha256_value(review),
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                    "terminal_status": "completed",
                    "decision": "revise" if revise else "accept",
                    "reason": "controlled prompt repair" if revise else "all checks passed",
                    "checks": {field: True for field in g0a.STATIC_REVIEW_CHECKS},
                    "revisions": {},
                    "proposed_patch": {"set": (
                        {
                            "prompt_en": "Revised subject was linked to",
                            "canonical_fact_en": "Revised subject was linked to Corrected Answer 1.",
                            "answer_en": "Corrected Answer 1",
                            "answer_zh": "修订答案一",
                            "answer_aliases_en": ["Corrected Answer 1", "A1"],
                            "answer_aliases_zh": ["修订答案一"],
                        }
                        if revise
                        else {}
                    )},
                    "variants": [
                        {
                            "distractor_id": variant["distractor_id"],
                            "selected_candidate_id": variant["context_candidates"][0]["candidate_id"],
                            "decision": "accept",
                            "reason": "all checks passed",
                            "checks": {field: True for field in g0a.VARIANT_REVIEW_CHECKS},
                        }
                        for variant in review["variants"]
                    ],
                }
            )
        path = review_dir / "codex_decisions.jsonl"
        g0a.write_jsonl(path, rows)
        return path

    def evidence(self, stage_dir, review_dir, decisions, root):
        staging_manifest = g0a.read_json(stage_dir / "static_staging_manifest.json")
        staging_rows = {
            row["base_fact_id"]: row for row in jsonl(stage_dir / "static_staging_records.jsonl")
        }
        resolved_dir = root / "resolved"
        resolved_result = g0a.resolve_static_g0a_draft(
            staging_manifest_path=stage_dir / "static_staging_manifest.json",
            review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
            codex_decisions_path=decisions,
            output_dir=resolved_dir,
        )
        resolved_manifest_path = Path(
            resolved_result["resolved_static_draft_manifest"]
        )
        resolved_manifest = g0a.read_json(resolved_manifest_path)
        resolved_rows = {
            row["base_fact_id"]: row
            for row in jsonl(resolved_dir / "resolved_static_draft.jsonl")
        }
        ids = sorted(staging_rows)
        semantic_candidates = [
            {
                "schema_version": g0a.SEMANTIC_CANDIDATE_SCHEMA,
                "candidate_id": "semantic_pair_1",
                "left_base_fact_id": ids[0],
                "right_base_fact_id": ids[1],
                "left_source_record_sha256": staging_rows[ids[0]]["source_record_sha256"],
                "right_source_record_sha256": staging_rows[ids[1]]["source_record_sha256"],
                "left_effective_source_record_sha256": staging_rows[ids[0]][
                    "effective_source_record_sha256"
                ],
                "right_effective_source_record_sha256": staging_rows[ids[1]][
                    "effective_source_record_sha256"
                ],
                "left_input_record_sha256": g0a.sha256_value(resolved_rows[ids[0]]),
                "right_input_record_sha256": g0a.sha256_value(resolved_rows[ids[1]]),
            }
        ]
        semantic_decisions = [
            {
                "schema_version": g0a.SEMANTIC_DECISION_SCHEMA,
                "candidate_id": "semantic_pair_1",
                "candidate_record_sha256": g0a.sha256_value(semantic_candidates[0]),
                "terminal_status": "completed",
                "decision": "distinct",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
            }
        ]
        semantic_candidates_path = root / "semantic_candidates.jsonl"
        semantic_decisions_path = root / "semantic_decisions.jsonl"
        g0a.write_jsonl(semantic_candidates_path, semantic_candidates)
        g0a.write_jsonl(semantic_decisions_path, semantic_decisions)
        semantic_manifest_path = root / "semantic_manifest.json"
        g0a.write_json(
            semantic_manifest_path,
            {
                "schema_version": g0a.SEMANTIC_CLOSURE_MANIFEST_SCHEMA,
                "status": "complete",
                "input_bundle": staging_manifest["source_bundle"],
                "effective_staging_bundle": staging_manifest[
                    "effective_staging_bundle"
                ],
                "resolved_static_draft": resolved_manifest["resolved_rows"],
                "review_blinded_to_behavior_results": True,
                "behavior_result_count": 0,
                "semantic_paraphrase_closure_complete": True,
                "candidate_generation_complete": True,
                "cohort_scope_complete": True,
                "out_of_cohort_bridge_search_complete": True,
                "unresolved_candidate_count": 0,
                "policy_version": "fixture-semantic-policy-v1",
                "candidates": g0a._binding(
                    semantic_candidates_path,
                    schema_version=g0a.SEMANTIC_CANDIDATE_SCHEMA,
                    record_count=1,
                ),
                "adjudications": g0a._binding(
                    semantic_decisions_path,
                    schema_version=g0a.SEMANTIC_DECISION_SCHEMA,
                    record_count=1,
                ),
            },
        )
        checks = []
        for base_fact_id in ids[1:]:
            checks.append(
                {
                    "schema_version": g0a.EXPOSURE_CHECK_SCHEMA,
                    "base_fact_id": base_fact_id,
                    "source_record_sha256": staging_rows[base_fact_id]["source_record_sha256"],
                    "effective_source_record_sha256": staging_rows[base_fact_id][
                        "effective_source_record_sha256"
                    ],
                    "input_record_sha256": g0a.sha256_value(
                        resolved_rows[base_fact_id]
                    ),
                    "split_assignment": staging_rows[base_fact_id]["split_assignment"],
                    "terminal_status": "completed",
                    "historically_exposed": False,
                    "checked_sources": ["fixture-run-registry"],
                    "query_fingerprint_sha256": g0a.sha256_value([base_fact_id, "query"]),
                    "checked_at": "2026-09-16T00:00:00+00:00",
                }
            )
        checks_path = root / "exposure_checks.jsonl"
        registry_path = root / "exposure_registry.jsonl"
        g0a.write_jsonl(checks_path, checks)
        g0a.write_jsonl(registry_path, [])
        exposure_manifest_path = root / "exposure_manifest.json"
        g0a.write_json(
            exposure_manifest_path,
            {
                "schema_version": g0a.EXPOSURE_MANIFEST_SCHEMA,
                "status": "complete",
                "input_bundle": staging_manifest["source_bundle"],
                "effective_staging_bundle": staging_manifest[
                    "effective_staging_bundle"
                ],
                "resolved_static_draft": resolved_manifest["resolved_rows"],
                "review_blinded_to_behavior_results": True,
                "behavior_result_count": 0,
                "historical_exposure_complete": True,
                "zero_registry_supported_by_per_fact_checks": True,
                "policy_version": "fixture-exposure-policy-v1",
                "checks": g0a._binding(
                    checks_path,
                    schema_version=g0a.EXPOSURE_CHECK_SCHEMA,
                    record_count=len(checks),
                ),
                "registry": g0a._binding(
                    registry_path,
                    schema_version=g0a.EXPOSURE_REGISTRY_SCHEMA,
                    record_count=0,
                ),
            },
        )
        return resolved_manifest_path, semantic_manifest_path, exposure_manifest_path

    def test_stage_preserves_review_only_boundary_and_two_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, result = self.stage(root)
            self.assertFalse(result["behavior_authorized"])
            manifest = g0a.read_json(stage_dir / "static_staging_manifest.json")
            self.assertEqual(manifest["counts"]["base_facts"], 3)
            self.assertEqual(manifest["counts"]["distractors"], 6)
            self.assertEqual(manifest["counts"]["required_unique_behavior_inputs"], 30)
            self.assertFalse(manifest["formal_prepare_authorized"])
            self.assertEqual(
                manifest["effective_staging_bundle"]["sha256"],
                manifest["staging_records"]["sha256"],
            )
            self.assertEqual(manifest["source_decisions_manifest"]["record_count"], 3)
            self.assertIn("source_semantic_paraphrase_closure_incomplete", manifest["known_blockers"])
            self.assertTrue(all(len(row["distractors"]) == 2 for row in jsonl(stage_dir / "static_staging_records.jsonl")))

    def test_stage_requires_manifest_bound_full_source_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            decisions, _ = self.make_source_decisions(root, rows, bundle)
            with self.assertRaises(g0a.GateBlocked) as captured:
                g0a.stage_static_g0a(
                    input_bundle_path=bundle,
                    source_finalization_manifest_path=source_manifest,
                    source_decisions_path=decisions,
                    output_dir=root / "stage",
                    expected_record_count=3,
                )
            self.assertIn(
                "source_decisions_manifest_missing", captured.exception.blockers
            )

    def test_stage_refuses_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            decisions, decisions_manifest = self.make_source_decisions(
                root, rows, bundle
            )
            stage_dir = root / "stage"
            stage_dir.mkdir()
            stale = stage_dir / "translations.jsonl"
            g0a.write_jsonl(stale, [{"source_id": "stale"}])
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse non-empty"):
                g0a.stage_static_g0a(
                    input_bundle_path=bundle,
                    source_finalization_manifest_path=source_manifest,
                    source_decisions_path=decisions,
                    source_decisions_manifest_path=decisions_manifest,
                    output_dir=stage_dir,
                    expected_record_count=3,
                )
            self.assertEqual(jsonl(stale), [{"source_id": "stale"}])

    def test_stage_emits_runtime_compatible_manifest_with_empty_simulation_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            source_decisions, source_decisions_manifest = self.make_source_decisions(
                root, rows, bundle, name="runtime-source-review"
            )
            config_path = root / "config.json"
            g0a.write_json(
                config_path,
                {
                    "config_version": "fixture-v1",
                    "provider_profiles": {},
                    "execution": {"max_workers": 1},
                    "pilot": {"seed": 7},
                    "preflight": {},
                    "model_roles": {
                        "translation": {"repair_rejected": False},
                        "distractor_generation": {"enabled": False},
                        "perturbation_generation": {
                            "manipulation_family": g0a.MANIPULATION_FAMILY,
                            "target_distractors_per_source": 2,
                            "matched_neutral": True,
                            "candidates_per_distractor": 1,
                        },
                        "perturbation_validation": {
                            "hard_gate": True,
                            "required_checks": list(g0a.runtime.EXPLICIT_FALSE_ASSERTION_CHECKS),
                        },
                        "simulation": {
                            "models": [{"model": "must-be-removed"}],
                            "endpoint": "multi_option",
                            "max_options": 3,
                            "neutral_context_mode": "matched_per_candidate",
                        },
                    },
                },
            )
            stage_dir = root / "stage"
            g0a.stage_static_g0a(
                input_bundle_path=bundle,
                source_finalization_manifest_path=source_manifest,
                source_decisions_path=source_decisions,
                source_decisions_manifest_path=source_decisions_manifest,
                output_dir=stage_dir,
                config_path=config_path,
                run_id="fixture-static",
                expected_record_count=3,
            )
            static_config = g0a.read_json(stage_dir / "static_upstream_config.json")
            run_manifest = g0a.read_json(stage_dir / "run_manifest.json")
            self.assertEqual(static_config["model_roles"]["simulation"]["models"], [])
            self.assertEqual(run_manifest["simulation_models"], [])
            self.assertTrue(run_manifest["simulation_requires_codex_proxy_review"])
            self.assertFalse(run_manifest["behavior_authorized"])
            self.assertEqual(run_manifest["selected_count"], len(rows))
            run_manifest["records"][0]["answer_en"] = "stale checkpoint input"
            run_manifest["records_sha256"] = g0a.sha256_value(run_manifest["records"])
            g0a.write_json(stage_dir / "run_manifest.json", run_manifest)
            with self.assertRaisesRegex(
                ValueError, "records differ from effective staging projection"
            ):
                g0a._validate_runtime_manifest_staging_contract(
                    stage_dir / "run_manifest.json"
                )

    def test_source_decisions_apply_subject_answer_and_donor_updates_before_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            source_decisions, source_decisions_manifest = self.make_source_decisions(
                root,
                rows,
                bundle,
                name="revised-source-review",
                patches={
                    "pbf_1": {
                        "set": {
                            "subject_en": "Revised subject",
                            "prompt_en": "Revised subject was linked to",
                            "canonical_fact_en": "Revised subject was linked to Corrected Answer 1.",
                            "answer_en": "Corrected Answer 1",
                            "answer_aliases_en": ["Corrected Answer 1", "A1"],
                        },
                        "regenerate_dependents": True,
                    }
                },
            )
            config_path = root / "config.json"
            g0a.write_json(
                config_path,
                {
                    "config_version": "fixture-v1",
                    "provider_profiles": {},
                    "execution": {"max_workers": 1},
                    "pilot": {"seed": 7},
                    "preflight": {},
                    "model_roles": {
                        "translation": {"repair_rejected": False},
                        "distractor_generation": {"enabled": False},
                        "perturbation_generation": {
                            "manipulation_family": g0a.MANIPULATION_FAMILY,
                            "target_distractors_per_source": 2,
                            "matched_neutral": True,
                            "candidates_per_distractor": 1,
                        },
                        "perturbation_validation": {
                            "required_checks": list(g0a.runtime.EXPLICIT_FALSE_ASSERTION_CHECKS)
                        },
                        "simulation": {
                            "models": [],
                            "endpoint": "multi_option",
                            "max_options": 3,
                            "neutral_context_mode": "matched_per_candidate",
                        },
                    },
                },
            )
            stage_dir = root / "stage"
            g0a.stage_static_g0a(
                input_bundle_path=bundle,
                source_finalization_manifest_path=source_manifest,
                source_decisions_path=source_decisions,
                source_decisions_manifest_path=source_decisions_manifest,
                output_dir=stage_dir,
                config_path=config_path,
                run_id="fixture-static",
                expected_record_count=3,
            )
            staging = {
                row["base_fact_id"]: row for row in jsonl(stage_dir / "static_staging_records.jsonl")
            }
            runtime_rows = {
                row["base_fact_id"]: row
                for row in g0a.read_json(stage_dir / "run_manifest.json")["records"]
            }
            self.assertEqual(staging["pbf_1"]["subject_en"], "Revised subject")
            self.assertEqual(runtime_rows["pbf_1"]["answer_en"], "Corrected Answer 1")
            self.assertEqual(
                runtime_rows["pbf_1"]["canonical_fact"],
                "Revised subject was linked to Corrected Answer 1.",
            )
            donor_variant = next(
                row
                for row in runtime_rows["pbf_2"]["distractors"]
                if row["donor_base_fact_id"] == "pbf_1"
            )
            self.assertEqual(donor_variant["text_en"], "Corrected Answer 1")
            self.assertEqual(donor_variant["answer_aliases_en"], ["Corrected Answer 1", "A1"])
            manifest = g0a.read_json(stage_dir / "static_staging_manifest.json")
            self.assertEqual(
                manifest["source_decisions"]["decision_counts"],
                {"accept": 2, "revise": 1},
            )
            self.assertEqual(manifest["source_decisions"]["donor_reference_update_count"], 1)

            review_dir = self.make_upstream(stage_dir)
            decisions = self.completed_decisions(review_dir, revise_first=False)
            resolved, semantic, exposure = self.evidence(
                stage_dir, review_dir, decisions, root
            )
            frozen_dir = root / "frozen-source-revision"
            finalized = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions,
                resolved_static_draft_manifest_path=resolved,
                semantic_closure_manifest_path=semantic,
                historical_exposure_manifest_path=exposure,
                output_dir=frozen_dir,
            )
            self.assertEqual(finalized["status"], "g0a_static_stimulus_frozen")
            frozen = {
                row["base_fact_id"]: row
                for row in jsonl(frozen_dir / "frozen_static_bundle.jsonl")
            }
            self.assertEqual(frozen["pbf_1"]["subject_en"], "Revised subject")
            self.assertEqual(frozen["pbf_1"]["answer_en"], "Corrected Answer 1")
            propagated = next(
                row
                for row in frozen["pbf_2"]["static_mcq"]["variants"]
                if row["donor_base_fact_id"] == "pbf_1"
            )
            self.assertEqual(propagated["text_en"], "Corrected Answer 1")

    def test_adapt_source_audit_binds_bundle_rows_and_preserves_audit_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, _ = self.make_source(root)
            bundle_sha = g0a.sha256_file(bundle)
            audit_rows = []
            for index, row in enumerate(rows):
                revise = index == 0
                audit_rows.append(
                    {
                        "schema_version": g0a.CODEX_FACT_AUDIT_SCHEMA,
                        "input_bundle_sha256": bundle_sha,
                        "base_fact_id": row["base_fact_id"],
                        "source_id": row["source_id"],
                        "split_assignment": row["split_assignment"],
                        "probe_relation_id": row["probe_relation_id"],
                        "decision": "revise" if revise else "accept",
                        "issues": [],
                        "distractors": [
                            {
                                "distractor_id": item["distractor_id"],
                                "rank": rank,
                                "text_en": item["text_en"],
                                "verdict": "accept",
                                "reason": "fixture",
                            }
                            for rank, item in enumerate(row["distractor_candidates"], start=1)
                        ],
                        "blind_to_behavior": True,
                        "reviewer_type": "codex_proxy",
                        "review_status": "codex_adjudicated",
                        "human_gold": False,
                        "proposed_patch": (
                            {
                                "set": {
                                    "subject_en": "Revised subject",
                                    "prompt_en": "Revised subject was linked to",
                                    "canonical_fact_en": "Revised subject was linked to Answer 1.",
                                },
                                "regenerate_dependents": True,
                            }
                            if revise
                            else None
                        ),
                        "audit_revision": "fixture-v2",
                    }
                )
            audit_path = root / "audit.jsonl"
            g0a.write_jsonl(audit_path, audit_rows)
            output = root / "adapted"
            result = g0a.adapt_source_audit(
                input_bundle_path=bundle,
                audit_path=audit_path,
                output_dir=output,
                expected_record_count=3,
            )
            self.assertEqual(result["decision_counts"], {"accept": 2, "revise": 1})
            decisions = jsonl(output / "source_decisions.jsonl")
            revised = next(row for row in decisions if row["base_fact_id"] == "pbf_1")
            self.assertEqual(revised["proposed_patch"]["set"]["subject_en"], "Revised subject")
            self.assertEqual(revised["source_record_sha256"], g0a.sha256_value(rows[0]))
            self.assertEqual(revised["source_audit"]["audit_record_sha256"], g0a.sha256_value(audit_rows[0]))
            manifest = g0a.read_json(output / "source_decisions_manifest.json")
            self.assertEqual(manifest["input_bundle"]["sha256"], bundle_sha)
            self.assertTrue(manifest["blind_to_behavior"])

    def test_stage_rejects_source_decision_with_stale_bundle_hash_even_if_manifest_is_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            decisions_path, decisions_manifest_path = self.make_source_decisions(
                root, rows, bundle
            )
            decisions = jsonl(decisions_path)
            decisions[0]["input_bundle_sha256"] = "0" * 64
            g0a.write_jsonl(decisions_path, decisions)
            decisions_manifest = g0a.read_json(decisions_manifest_path)
            decisions_manifest["source_decisions"] = g0a._binding(
                decisions_path,
                schema_version=g0a.SOURCE_DECISION_SCHEMA,
                record_count=len(decisions),
                id_values=[row["base_fact_id"] for row in decisions],
                id_digest_field="base_fact_ids_sha256",
            )
            g0a.write_json(decisions_manifest_path, decisions_manifest)
            with self.assertRaisesRegex(ValueError, "stale input bundle hash"):
                g0a.stage_static_g0a(
                    input_bundle_path=bundle,
                    source_finalization_manifest_path=source_manifest,
                    source_decisions_path=decisions_path,
                    source_decisions_manifest_path=decisions_manifest_path,
                    output_dir=root / "stage",
                    expected_record_count=3,
                )

    def test_adapter_accepts_and_applies_distractor_only_revision_with_strict_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            bundle_sha = g0a.sha256_file(bundle)
            update = {
                "target_base_fact_id": "pbf_2",
                "distractor_id": "d2_1",
                "donor_base_fact_id": "pbf_1",
                "old_text_en": "Answer 1",
                "new_text_en": "Corrected Answer 1",
                "propagation_reason": "donor answer revision",
            }
            audit_rows = []
            for row in rows:
                base_fact_id = row["base_fact_id"]
                if base_fact_id == "pbf_1":
                    disposition = "revise"
                    proposed_patch = {
                        "set": {
                            "answer_en": "Corrected Answer 1",
                            "answer_aliases_en": ["Corrected Answer 1", "A1"],
                        },
                        "distractor_updates": [copy.deepcopy(update)],
                        "regenerate_dependents": True,
                    }
                elif base_fact_id == "pbf_2":
                    disposition = "revise"
                    proposed_patch = {
                        "distractor_updates": [copy.deepcopy(update)],
                        "regenerate_dependents": True,
                    }
                else:
                    disposition = "accept"
                    proposed_patch = None
                audit_rows.append(
                    {
                        "schema_version": g0a.CODEX_FACT_AUDIT_SCHEMA,
                        "input_bundle_sha256": bundle_sha,
                        "base_fact_id": base_fact_id,
                        "source_id": row["source_id"],
                        "split_assignment": row["split_assignment"],
                        "probe_relation_id": row["probe_relation_id"],
                        "decision": disposition,
                        "issues": [],
                        "distractors": [
                            {
                                "distractor_id": item["distractor_id"],
                                "rank": rank,
                                "text_en": item["text_en"],
                                "verdict": (
                                    "revise"
                                    if base_fact_id == "pbf_2" and item["distractor_id"] == "d2_1"
                                    else "accept"
                                ),
                                "reason": "fixture",
                            }
                            for rank, item in enumerate(row["distractor_candidates"], start=1)
                        ],
                        "blind_to_behavior": True,
                        "reviewer_type": "codex_proxy",
                        "review_status": "codex_adjudicated",
                        "human_gold": False,
                        "proposed_patch": proposed_patch,
                        "audit_revision": "fixture-donor-propagation-v2",
                    }
                )
            audit_path = root / "audit-with-distractor-update.jsonl"
            g0a.write_jsonl(audit_path, audit_rows)
            adapted_dir = root / "adapted"
            result = g0a.adapt_source_audit(
                input_bundle_path=bundle,
                audit_path=audit_path,
                output_dir=adapted_dir,
                expected_record_count=3,
            )
            self.assertEqual(result["decision_counts"], {"accept": 1, "revise": 2})
            self.assertEqual(result["distractor_update_declaration_count"], 2)
            self.assertEqual(result["unique_declared_distractor_update_count"], 1)

            stage_dir = root / "stage"
            g0a.stage_static_g0a(
                input_bundle_path=bundle,
                source_finalization_manifest_path=source_manifest,
                source_decisions_path=adapted_dir / "source_decisions.jsonl",
                source_decisions_manifest_path=(
                    adapted_dir / "source_decisions_manifest.json"
                ),
                output_dir=stage_dir,
                expected_record_count=3,
            )
            staged = {
                row["base_fact_id"]: row
                for row in jsonl(stage_dir / "static_staging_records.jsonl")
            }
            revised_distractor = next(
                item for item in staged["pbf_2"]["distractors"] if item["distractor_id"] == "d2_1"
            )
            self.assertEqual(revised_distractor["text_en"], "Corrected Answer 1")
            stage_manifest = g0a.read_json(stage_dir / "static_staging_manifest.json")
            provenance = stage_manifest["source_decisions"]
            self.assertEqual(provenance["distractor_update_declaration_count"], 2)
            self.assertEqual(provenance["unique_declared_distractor_update_count"], 1)
            self.assertEqual(provenance["applied_declared_distractor_update_count"], 1)

            invalid_rows = copy.deepcopy(audit_rows)
            invalid_rows[1]["proposed_patch"]["distractor_updates"][0][
                "new_text_en"
            ] = "Unbound value"
            invalid_path = root / "invalid-audit.jsonl"
            g0a.write_jsonl(invalid_path, invalid_rows)
            with self.assertRaisesRegex(ValueError, "new_text_en disagrees with revised donor"):
                g0a.adapt_source_audit(
                    input_bundle_path=bundle,
                    audit_path=invalid_path,
                    output_dir=root / "invalid-adapted",
                    expected_record_count=3,
                )

    def test_source_distractor_verdicts_control_effective_staging_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            update = {
                "target_base_fact_id": "pbf_2",
                "distractor_id": "d2_1",
                "donor_base_fact_id": "pbf_1",
                "old_text_en": "Answer 1",
                "new_text_en": "Corrected Answer 1",
                "propagation_reason": "donor answer revision",
            }
            decisions, decisions_manifest = self.make_source_decisions(
                root,
                rows,
                bundle,
                name="mixed-distractor-verdicts",
                patches={
                    "pbf_1": {
                        "set": {
                            "answer_en": "Corrected Answer 1",
                            "answer_aliases_en": ["Corrected Answer 1", "A1"],
                        },
                        "distractor_updates": [copy.deepcopy(update)],
                        "regenerate_dependents": True,
                    },
                    "pbf_2": {
                        "set": {},
                        "distractor_updates": [copy.deepcopy(update)],
                        "regenerate_dependents": True,
                    },
                },
                distractor_verdicts={
                    ("pbf_2", "d2_1"): "revise",
                    ("pbf_2", "d2_2"): "reject",
                },
            )
            config_path = root / "config.json"
            g0a.write_json(
                config_path,
                {
                    "config_version": "fixture-source-verdict-v1",
                    "provider_profiles": {},
                    "execution": {"max_workers": 1},
                    "pilot": {"seed": 7},
                    "preflight": {},
                    "model_roles": {
                        "translation": {"repair_rejected": False},
                        "distractor_generation": {"enabled": True},
                        "perturbation_generation": {
                            "manipulation_family": g0a.MANIPULATION_FAMILY,
                            "target_distractors_per_source": 2,
                            "matched_neutral": True,
                            "candidates_per_distractor": 1,
                        },
                        "perturbation_validation": {
                            "required_checks": list(
                                g0a.runtime.EXPLICIT_FALSE_ASSERTION_CHECKS
                            )
                        },
                        "simulation": {
                            "models": [],
                            "endpoint": "multi_option",
                            "max_options": 3,
                            "neutral_context_mode": "matched_per_candidate",
                        },
                    },
                },
            )
            stage_dir = root / "stage-v2"
            g0a.stage_static_g0a(
                input_bundle_path=bundle,
                source_finalization_manifest_path=source_manifest,
                source_decisions_path=decisions,
                source_decisions_manifest_path=decisions_manifest,
                output_dir=stage_dir,
                config_path=config_path,
                run_id="fixture-source-verdict",
                expected_record_count=3,
            )

            effective = {
                row["base_fact_id"]: row
                for row in jsonl(stage_dir / "effective_source_bundle.jsonl")
            }
            effective_distractors = {
                row["distractor_id"]: row
                for row in effective["pbf_2"]["distractor_candidates"]
            }
            revised = effective_distractors["d2_1"]
            rejected = effective_distractors["d2_2"]
            self.assertEqual(revised["text_en"], "Corrected Answer 1")
            self.assertEqual(
                revised["g0a_source_distractor_review"]["verdict"], "revise"
            )
            self.assertTrue(
                revised["g0a_source_distractor_review"][
                    "translation_validation_eligible"
                ]
            )
            self.assertEqual(
                rejected["g0a_source_distractor_review"]["verdict"], "reject"
            )
            self.assertTrue(
                rejected["g0a_source_distractor_review"][
                    "replacement_generation_required"
                ]
            )

            run_manifest = g0a.read_json(stage_dir / "run_manifest.json")
            runtime_row = next(
                row for row in run_manifest["records"] if row["base_fact_id"] == "pbf_2"
            )
            self.assertEqual(
                [row["distractor_id"] for row in runtime_row["distractors"]],
                ["d2_1"],
            )
            self.assertEqual(runtime_row["distractors"][0]["text_en"], "Corrected Answer 1")
            self.assertEqual(
                runtime_row["replacement_generation_requests"],
                [
                    {
                        "replaces_source_distractor_id": "d2_2",
                        "rank": 2,
                        "source_audit_verdict": "reject",
                        "source_audit_record_sha256": rejected[
                            "g0a_source_distractor_review"
                        ]["audit_record_sha256"],
                        "action": "generate_then_validate_replacement",
                    }
                ],
            )
            runtime_payload = json.dumps(runtime_row, ensure_ascii=False)
            self.assertNotIn("Wrong 2 2", runtime_payload)
            self.assertNotIn("Wrong 2 2", g0a.runtime._translation_prompt(runtime_row))
            self.assertNotIn(
                "Wrong 2 2",
                g0a.runtime._translation_review_prompt(
                    runtime_row,
                    {
                        "prompt_zh": "主题 2 对应",
                        "answer_zh": "答案 2",
                        "answer_aliases_zh": ["答案 2"],
                        "distractors": [
                            {"distractor_id": "d2_1", "text_zh": "修订干扰项"}
                        ],
                    },
                ),
            )
            self.assertEqual(
                run_manifest[
                    "source_audit_replacement_generation_request_count"
                ],
                1,
            )
            staging_manifest = g0a.read_json(
                stage_dir / "static_staging_manifest.json"
            )
            self.assertEqual(
                staging_manifest["counts"][
                    "translation_validation_eligible_distractors"
                ],
                5,
            )
            self.assertEqual(
                staging_manifest["counts"]["replacement_generation_requests"],
                1,
            )
            g0a._validate_runtime_manifest_staging_contract(
                stage_dir / "run_manifest.json"
            )

    def test_stage_rejects_revised_distractor_without_bound_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, bundle, source_manifest = self.make_source(root)
            decisions, decisions_manifest = self.make_source_decisions(
                root,
                rows,
                bundle,
                distractor_verdicts={("pbf_2", "d2_1"): "revise"},
                name="unbound-distractor-revision",
            )
            with self.assertRaisesRegex(
                ValueError,
                "revised source distractor lacks a bound adjudicated update",
            ):
                g0a.stage_static_g0a(
                    input_bundle_path=bundle,
                    source_finalization_manifest_path=source_manifest,
                    source_decisions_path=decisions,
                    source_decisions_manifest_path=decisions_manifest,
                    output_dir=root / "stage-v2",
                    expected_record_count=3,
                )

    def test_export_rejects_behavior_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            self.make_upstream(stage_dir)
            g0a.write_jsonl(stage_dir / "simulation_results.jsonl", [{"simulation_id": "leak"}])
            with self.assertRaises(g0a.GateBlocked) as captured:
                g0a.export_codex_review(
                    staging_manifest_path=stage_dir / "static_staging_manifest.json",
                    upstream_run_dir=stage_dir,
                    output_dir=root / "review-again",
                )
            self.assertIn("behavior_results_present_review_not_blind", captured.exception.blockers)

    def test_export_preserves_advisory_rejects_for_final_codex_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(
                stage_dir, automatic_review_rejects=True
            )
            reviews = jsonl(review_dir / "codex_adjudication_input.jsonl")
            self.assertEqual(len(reviews), 3)
            self.assertTrue(
                all(
                    row["translation_review"]["decision"] == "reject"
                    and row["translation_review_terminal_status"] == "completed"
                    for row in reviews
                )
            )
            self.assertTrue(
                all(len(row["variants"]) == 2 for row in reviews)
            )
            self.assertTrue(
                all(
                    variant["automatic_review"]["decision"] == "reject"
                    and variant["automatic_review_terminal_status"] == "completed"
                    for row in reviews
                    for variant in row["variants"]
                )
            )

    def test_export_ignores_accepted_backup_without_a_context_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            shutil.rmtree(review_dir)
            distractors_path = stage_dir / "verified_distractors.jsonl"
            distractors = jsonl(distractors_path)
            extra = copy.deepcopy(distractors[0])
            extra["item_id"] = "accepted_unused_backup"
            extra["distractor_en"] = "Unused backup"
            extra["distractor_zh"] = "未使用的备用项"
            distractors.append(extra)
            g0a.write_jsonl(distractors_path, distractors)

            result = g0a.export_codex_review(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                upstream_run_dir=stage_dir,
                output_dir=review_dir,
            )

            self.assertEqual(result["variant_count"], 6)
            reviews = jsonl(review_dir / "codex_adjudication_input.jsonl")
            self.assertNotIn(
                "accepted_unused_backup",
                {
                    variant["distractor_id"]
                    for review in reviews
                    for variant in review["variants"]
                },
            )

    def test_context_repair_overlay_adds_bound_candidate_without_rewriting_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            original_review_dir = self.make_upstream(stage_dir)
            repair = self.context_repair_row(stage_dir, original_review_dir)
            repair_path = root / "context_repairs.jsonl"
            g0a.write_jsonl(repair_path, [repair])
            upstream_sha_before = g0a.sha256_file(stage_dir / "perturbations.jsonl")

            review_dir = root / "review-with-repair"
            result = g0a.export_codex_review(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                upstream_run_dir=stage_dir,
                context_repairs_path=repair_path,
                output_dir=review_dir,
            )

            self.assertEqual(result["context_repair_count"], 1)
            self.assertEqual(
                g0a.sha256_file(stage_dir / "perturbations.jsonl"),
                upstream_sha_before,
            )
            manifest = g0a.read_json(
                review_dir / "codex_adjudication_input_manifest.json"
            )
            self.assertEqual(manifest["context_repair_count"], 1)
            self.assertEqual(
                manifest["context_repair_overlay"]["sha256"],
                g0a.sha256_file(repair_path),
            )
            reviews = jsonl(review_dir / "codex_adjudication_input.jsonl")
            repaired_review = next(
                row for row in reviews if row["base_fact_id"] == repair["base_fact_id"]
            )
            repaired_variant = next(
                row
                for row in repaired_review["variants"]
                if row["distractor_id"] == repair["distractor_id"]
            )
            candidates = {
                row["candidate_id"]: row
                for row in repaired_variant["context_candidates"]
            }
            self.assertIn(repair["superseded_candidate_id"], candidates)
            self.assertEqual(
                candidates[repair["candidate_id"]]["candidate_record_sha256"],
                g0a.sha256_value(repair),
            )
            self.assertIsNone(
                candidates[repair["candidate_id"]]["automatic_review"]
            )
            g0a._load_review_input(
                review_dir / "codex_adjudication_input_manifest.json",
                stage_dir / "static_staging_manifest.json",
            )

    def test_context_repair_overlay_rejects_stale_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            original_review_dir = self.make_upstream(stage_dir)
            repair = self.context_repair_row(stage_dir, original_review_dir)
            repair["superseded_candidate_record_sha256"] = "0" * 64
            repair_path = root / "stale_context_repair.jsonl"
            g0a.write_jsonl(repair_path, [repair])

            with self.assertRaisesRegex(
                ValueError, "stale superseded candidate hash"
            ):
                g0a.export_codex_review(
                    staging_manifest_path=stage_dir / "static_staging_manifest.json",
                    upstream_run_dir=stage_dir,
                    context_repairs_path=repair_path,
                    output_dir=root / "review-with-stale-repair",
                )

    def test_review_loader_rejects_forged_repair_candidate_after_rebinding_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            original_review_dir = self.make_upstream(stage_dir)
            repair = self.context_repair_row(stage_dir, original_review_dir)
            repair_path = root / "context_repairs.jsonl"
            g0a.write_jsonl(repair_path, [repair])
            review_dir = root / "review-with-repair"
            g0a.export_codex_review(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                upstream_run_dir=stage_dir,
                context_repairs_path=repair_path,
                output_dir=review_dir,
            )

            review_path = review_dir / "codex_adjudication_input.jsonl"
            reviews = jsonl(review_path)
            repaired_review = next(
                row for row in reviews if row["base_fact_id"] == repair["base_fact_id"]
            )
            repaired_variant = next(
                row
                for row in repaired_review["variants"]
                if row["distractor_id"] == repair["distractor_id"]
            )
            repaired_candidate = next(
                row
                for row in repaired_variant["context_candidates"]
                if row["candidate_id"] == repair["candidate_id"]
            )
            repaired_candidate["neutral_context_en"] = "Forged review text."
            g0a.write_jsonl(review_path, reviews)
            manifest_path = review_dir / "codex_adjudication_input_manifest.json"
            manifest = g0a.read_json(manifest_path)
            manifest["review_input"] = g0a._binding(
                review_path,
                schema_version=g0a.REVIEW_INPUT_RECORD_SCHEMA,
                record_count=len(reviews),
                id_values=[row["base_fact_id"] for row in reviews],
                id_digest_field="base_fact_ids_sha256",
            )
            g0a.write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError, "differs from bound provenance"
            ):
                g0a._load_review_input(
                    manifest_path,
                    stage_dir / "static_staging_manifest.json",
                )

    def test_finalize_without_closure_or_exposure_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            decisions = self.completed_decisions(review_dir)
            resolved_dir = root / "resolved"
            resolved = g0a.resolve_static_g0a_draft(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions,
                output_dir=resolved_dir,
            )["resolved_static_draft_manifest"]
            output = root / "blocked"
            result = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions,
                resolved_static_draft_manifest_path=Path(resolved),
                output_dir=output,
            )
            self.assertEqual(result["status"], "blocked_fail_closed")
            self.assertIn("semantic_paraphrase_closure_evidence_missing", result["blockers"])
            self.assertIn("historical_exposure_evidence_missing", result["blockers"])
            self.assertFalse((output / "frozen_static_bundle.jsonl").exists())

    def test_reject_or_defer_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            decisions_path = self.completed_decisions(review_dir)
            decisions = jsonl(decisions_path)
            decisions[1]["decision"] = "reject"
            g0a.write_jsonl(decisions_path, decisions)
            result = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions_path,
                output_dir=root / "blocked",
            )
            self.assertTrue(any(value.startswith("codex_decision_reject:") for value in result["blockers"]))
            self.assertFalse(result["frozen_artifacts_emitted"])

    def test_same_leakage_component_cross_split_blocks_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            decisions = self.completed_decisions(review_dir, revise_first=False)
            resolved, semantic, exposure = self.evidence(
                stage_dir, review_dir, decisions, root
            )
            semantic_manifest = g0a.read_json(semantic)
            semantic_decisions_path = root / "semantic_decisions.jsonl"
            semantic_decisions = jsonl(semantic_decisions_path)
            semantic_decisions[0]["decision"] = "same_leakage_component"
            g0a.write_jsonl(semantic_decisions_path, semantic_decisions)
            semantic_manifest["adjudications"] = g0a._binding(
                semantic_decisions_path,
                schema_version=g0a.SEMANTIC_DECISION_SCHEMA,
                record_count=1,
            )
            g0a.write_json(semantic, semantic_manifest)
            result = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions,
                resolved_static_draft_manifest_path=resolved,
                semantic_closure_manifest_path=semantic,
                historical_exposure_manifest_path=exposure,
                output_dir=root / "blocked-cross-split",
            )
            self.assertEqual(result["status"], "blocked_fail_closed")
            self.assertTrue(
                any(
                    "same_leakage_component pair crosses split groups" in blocker
                    for blocker in result["blockers"]
                )
            )
            self.assertFalse(result["frozen_artifacts_emitted"])

    def test_final_revision_after_resolved_draft_invalidates_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            decisions_path = self.completed_decisions(review_dir)
            resolved, semantic, exposure = self.evidence(
                stage_dir, review_dir, decisions_path, root
            )
            decisions = jsonl(decisions_path)
            decisions[0]["proposed_patch"]["set"]["prompt_en"] = (
                "Changed after closure"
            )
            decisions[0]["revisions"] = copy.deepcopy(
                decisions[0]["proposed_patch"]["set"]
            )
            g0a.write_jsonl(decisions_path, decisions)
            result = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions_path,
                resolved_static_draft_manifest_path=resolved,
                semantic_closure_manifest_path=semantic,
                historical_exposure_manifest_path=exposure,
                output_dir=root / "blocked-stale-resolved",
            )
            self.assertEqual(result["status"], "blocked_fail_closed")
            self.assertTrue(
                any(
                    blocker.startswith("resolved_static_draft_invalid:")
                    for blocker in result["blockers"]
                )
            )

    def test_semantic_evidence_must_bind_effective_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            decisions = self.completed_decisions(review_dir, revise_first=False)
            resolved, semantic, exposure = self.evidence(
                stage_dir, review_dir, decisions, root
            )
            candidates_path = root / "semantic_candidates.jsonl"
            candidates = jsonl(candidates_path)
            candidates[0]["left_effective_source_record_sha256"] = candidates[0][
                "left_source_record_sha256"
            ]
            g0a.write_jsonl(candidates_path, candidates)
            semantic_manifest = g0a.read_json(semantic)
            semantic_manifest["candidates"] = g0a._binding(
                candidates_path,
                schema_version=g0a.SEMANTIC_CANDIDATE_SCHEMA,
                record_count=len(candidates),
            )
            g0a.write_json(semantic, semantic_manifest)
            result = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions,
                resolved_static_draft_manifest_path=resolved,
                semantic_closure_manifest_path=semantic,
                historical_exposure_manifest_path=exposure,
                output_dir=root / "blocked-stale-effective",
            )
            self.assertEqual(result["status"], "blocked_fail_closed")
            self.assertTrue(
                any(
                    "stale left effective source hash" in blocker
                    for blocker in result["blockers"]
                )
            )

    def test_finalize_applies_revision_and_emits_formal_compatible_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, stage_dir, _ = self.stage(root)
            review_dir = self.make_upstream(stage_dir)
            decisions = self.completed_decisions(review_dir)
            resolved, semantic, exposure = self.evidence(
                stage_dir, review_dir, decisions, root
            )
            output = root / "frozen"
            result = g0a.finalize_static_g0a(
                staging_manifest_path=stage_dir / "static_staging_manifest.json",
                review_input_manifest_path=review_dir / "codex_adjudication_input_manifest.json",
                codex_decisions_path=decisions,
                resolved_static_draft_manifest_path=resolved,
                semantic_closure_manifest_path=semantic,
                historical_exposure_manifest_path=exposure,
                output_dir=output,
            )
            self.assertEqual(result["status"], "g0a_static_stimulus_frozen")
            rows = jsonl(output / "frozen_static_bundle.jsonl")
            revised = next(row for row in rows if row["base_fact_id"] == "pbf_1")
            self.assertEqual(revised["prompt_en"], "Revised subject was linked to")
            self.assertEqual(revised["answer_en"], "Corrected Answer 1")
            self.assertEqual(revised["canonical_fact"], revised["canonical_fact_en"])
            self.assertEqual(revised["answer_aliases_en"], ["Corrected Answer 1", "A1"])
            self.assertEqual(len(revised["static_mcq"]["variants"]), 2)
            self.assertEqual(len(revised["static_mcq"]["behavior_inputs"]), 10)
            target = next(row for row in rows if row["base_fact_id"] == "pbf_2")
            propagated = next(
                row
                for row in target["static_mcq"]["variants"]
                if row["donor_base_fact_id"] == "pbf_1"
            )
            self.assertEqual(propagated["text_en"], "Corrected Answer 1")
            self.assertEqual(propagated["answer_aliases_en"], ["Corrected Answer 1", "A1"])
            self.assertTrue(propagated["donor_answer_revision_propagated"])
            static_manifest = g0a.read_json(output / "static_freeze_manifest.json")
            self.assertEqual(static_manifest["codex_decision_counts"]["revise"], 1)
            self.assertFalse(static_manifest["behavior_authorized"])
            validated = g0a.formal_admission.validate_external_formal_admission(
                input_bundle_path=output / "frozen_static_bundle.jsonl",
                input_schema_version=g0a.FINAL_BUNDLE_SCHEMA,
                records=rows,
                review_freeze_manifest_path=output / "review_freeze_manifest.json",
                split_freeze_manifest_path=output / "split_freeze_manifest.json",
            )
            self.assertTrue(validated["authorized"])


if __name__ == "__main__":
    unittest.main()
