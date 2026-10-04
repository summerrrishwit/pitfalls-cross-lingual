import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_full_public_benchmark_pre_exact_hf.py"
)
SPEC = importlib.util.spec_from_file_location("full_pre_exact_hf_freezer", SCRIPT_PATH)
freezer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(freezer)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_bytes(
        b"".join(freezer.canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )


def binding(path, *, schema=None, rows=None, relative=False):
    result = {
        "path" if not relative else "filename": str(path.resolve())
        if not relative
        else path.name,
        "sha256": freezer.sha256_file(path),
        "byte_count": path.stat().st_size,
    }
    if schema is not None:
        result["schema_version"] = schema
    if rows is not None:
        result["record_count"] = len(rows)
    return result


def zh_binding(path, rows):
    result = binding(path, rows=rows)
    result["schema_versions"] = sorted(
        {str(row["schema_version"]) for row in rows if row.get("schema_version")}
    )
    return result


def refresh_route_identity_hashes(identity_set):
    for record in identity_set["records"]:
        record["route_identity_sha256"] = freezer.sha256_value(
            {
                key: value
                for key, value in record.items()
                if key != "route_identity_sha256"
            }
        )
    identity_set["route_identity_set_sha256"] = freezer.sha256_value(
        {
            key: value
            for key, value in identity_set.items()
            if key != "route_identity_set_sha256"
        }
    )


def refresh_zh_run_fingerprint(manifest):
    contract_fields = (
        "schema_version",
        "tool_version",
        "input_bindings",
        "source_fingerprints",
        "route_identity",
        "review_independence",
        "formal_universe_id",
        "formal_universe_status",
        "selected_base_fact_ids_sha256",
        "selected_record_count",
        "universe_record_count",
        "selection_is_full_universe",
        "selected_split_counts",
        "models",
        "candidate_binding",
        "invalidation_contract",
        "language_scope",
        "evidence_boundary",
        "execution_policy",
    )
    contract = {field: manifest.get(field) for field in contract_fields}
    for field in (
        "successor_cohort",
        "selection_is_complete_retained_cohort",
        "excluded_base_fact_ids_sha256",
    ):
        if field in manifest:
            contract[field] = manifest[field]
    manifest["run_fingerprint"] = freezer.sha256_value(contract)


class AcceptAllZhRouter:
    reject_ids = frozenset()

    def __init__(self, config, env_path, event_log):
        self.config = config
        self.env_path = env_path
        self.event_log = event_log

    @staticmethod
    def _check(ok=True):
        return {
            "translation_equivalent": ok,
            "natural_zh": ok,
            "issues": [] if ok else ["prompt mismatch"],
        }

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = json.loads(prompt.split(" Input: ", 1)[1])
        if stage in {
            "full_zh_translation_generation",
            "full_zh_translation_repair",
        }:
            records = []
            for source in payload["records"]:
                records.append(
                    {
                        "base_fact_id": source["base_fact_id"],
                        "translation": {
                            "prompt_zh": "项目是：",
                            "canonical_fact_zh": "项目对应答案。",
                            "answer_zh": "答案",
                            "answer_aliases_zh": [
                                f"别名{index}"
                                for index, _ in enumerate(
                                    source["source"]["answer_aliases_en"]
                                )
                            ],
                            "distractors": [
                                {
                                    "distractor_id": row["distractor_id"],
                                    "text_zh": "干扰答案",
                                }
                                for row in source["distractors"]
                            ],
                            "neutral_candidates": [
                                {
                                    "neutral_candidate_id": row[
                                        "neutral_candidate_id"
                                    ],
                                    "text_zh": "中性事实。",
                                }
                                for row in source["neutral_candidates"]
                            ],
                        },
                    }
                )
            response_model = freezer.zh_tool.DEFAULT_GENERATOR_RESPONSE_MODEL
        else:
            records = []
            for source in payload["records"]:
                accepted = source["base_fact_id"] not in self.reject_ids
                records.append(
                    {
                        "base_fact_id": source["base_fact_id"],
                        "review": {
                            "decision": "accept" if accepted else "reject",
                            "issues": [] if accepted else ["prompt mismatch"],
                            "checks": {
                                "prompt": self._check(accepted),
                                "canonical_fact": self._check(),
                                "answer": self._check(),
                                "answer_aliases": [
                                    {"alias_index": index, **self._check()}
                                    for index, _ in enumerate(
                                        source["source"]["answer_aliases_en"]
                                    )
                                ],
                                "distractors": [
                                    {
                                        "distractor_id": row["distractor_id"],
                                        **self._check(),
                                    }
                                    for row in source["distractors"]
                                ],
                                "neutral_candidates": [
                                    {
                                        "neutral_candidate_id": row[
                                            "neutral_candidate_id"
                                        ],
                                        **self._check(),
                                    }
                                    for row in source["neutral_candidates"]
                                ],
                            },
                        },
                    }
                )
            response_model = freezer.zh_tool.DEFAULT_REVIEWER_RESPONSE_MODEL
        parsed = {"records": records}
        validator(parsed)
        return freezer.zh_tool.runtime.ModelResult(
            "completed",
            parsed,
            json.dumps(parsed, ensure_ascii=False),
            response_model,
            {"total_tokens": 1},
            1,
            1,
            [{"attempt": 1, "status": "ok", "latency_ms": 1}],
        )


class RejectSelectedZhRouter(AcceptAllZhRouter):
    reject_ids = frozenset()


class FullPreExactHFFreezerTests(unittest.TestCase):
    def make_fixture(self, root):
        config_path = root / "config.json"
        config = {
            "config_version": "fixture-full-pre-exact-hf-v1",
            "provider_profiles": {
                "aliyun": {
                    "protocol": "openai_compatible",
                    "base_url": "https://aliyun-fixture.invalid/v1",
                },
                "openai": {
                    "protocol": "openai_compatible",
                    "base_url": "https://openai-fixture.invalid/v1",
                },
            },
            "model_roles": {
                "translation": {
                    "primary": {
                        "provider_profile": "aliyun",
                        "model": "qwen3.8-max-0902/bailian/bailian",
                        "expected_response_model": "qwen3.8-max",
                    },
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "expected_response_model": "gpt-5.5-2026-04-24",
                    },
                },
                "full_pool_candidate_semantic_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "expected_response_model": "gpt-5.5-2026-04-24",
                    }
                },
                "full_pool_semantic_closure_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "expected_response_model": "gpt-5.5-2026-04-24",
                    }
                },
            },
            "pre_exact_hf_contract": {
                "fact_review_evidence": "independent_model_proxy_not_human_gold",
                "translation_review_evidence": "independent_model_proxy_not_human_gold",
                "semantic_closure_scope": "frozen_deterministic_candidate_protocol_with_terminal_pair_adjudication",
                "semantic_near_duplicate_recall_guaranteed": False,
                "historical_exposure_authority": "predeclared_pending_scope_owner_attestation",
                "review_and_split_freeze_required_before_hf_binding": True,
                "hf_model_execution_authorized": False,
                "hf_tokenizer_execution_authorized": False,
                "validation_behavior_exposure_authorized": False,
                "sealed_behavior_exposure_authorized": False,
            },
        }
        write_json(config_path, config)

        retained = []
        for relation, prefix in (("author", "a"), ("location", "b")):
            for number in range(3):
                base_fact_id = f"pbf_{prefix}{number}"
                answer = f"Answer {prefix.upper()}{number}"
                retained.append(
                    {
                        "schema_version": freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                        "base_fact_id": base_fact_id,
                        "candidate_id": f"source_{base_fact_id}",
                        "source_id": f"source_{base_fact_id}",
                        "source_question_en": f"Who is item {prefix.upper()}{number}?",
                        "subject_en": f"item {prefix.upper()}{number}",
                        "relation_raw": relation,
                        "canonical_fact_en": f"Item {prefix.upper()}{number} is {answer}.",
                        "prompt_en": f"Item {prefix.upper()}{number} is",
                        "answer_en": answer,
                        "answer_aliases_en": [answer, f"{answer} alias"],
                        "answer_type": "person" if relation == "author" else "location",
                        "answer_type_bucket": "entity",
                        "relation_partition_id": relation,
                        "relation_partition_status": "broad_rule_candidate_partition",
                        "relation_normalized": None,
                        "probe_relation_id": None,
                        "probe_relation_candidate": None,
                        "split_assignment": "development",
                        "split_policy_version": "fixture-component-split-v1",
                        "split_status": "provisional_not_frozen",
                        "leakage_component_id": f"component_{base_fact_id}",
                        "canonical_status": "pending_review",
                        "human_gold": False,
                        "hf_model_execution_status": "not_run",
                        "hf_tokenizer_execution_status": "not_run",
                        "fact_review": {
                            "review_outcome": "accept",
                            "staging_status": "accepted_for_reviewed_rebuild",
                            "review_provenance": {"reviewer_type": "independent_model_proxy"},
                            "review_item_sha256": "1" * 64,
                            "review_completion": {
                                "codex_proxy_scope_review_complete": True,
                                "member_review_complete": True,
                                "alias_review_complete": True,
                                "distractor_review_complete": True,
                            },
                            "member_reviews": [],
                            "alias_reviews": [],
                            "source_choice_distractor_reviews": [],
                            "review_notes": "fixture",
                            "structured_review_notes": None,
                            "proxy_evidence_only": True,
                            "human_gold": False,
                        },
                    }
                )
        excluded = {
            **copy.deepcopy(retained[0]),
            "base_fact_id": "pbf_excluded",
            "candidate_id": "source_pbf_excluded",
            "source_id": "source_pbf_excluded",
        }
        input_rows = [*retained, excluded]

        distractors = []
        neutrals = []
        groups = {
            "author": [row for row in retained if row["relation_partition_id"] == "author"],
            "location": [row for row in retained if row["relation_partition_id"] == "location"],
        }
        for target in retained:
            relation = target["relation_partition_id"]
            other_relation = "location" if relation == "author" else "author"
            same_sources = [row for row in groups[relation] if row != target]
            neutral_sources = groups[other_relation][:2]
            for slot, source in enumerate(same_sources, 1):
                distractors.append(
                    {
                        "schema_version": freezer.candidate_tool.DISTRACTOR_SCHEMA,
                        "distractor_id": f"d_{target['base_fact_id']}_{slot}",
                        "base_fact_id": target["base_fact_id"],
                        "slot": slot,
                        "source_base_fact_id": source["base_fact_id"],
                        "distractor_text_en": source["answer_en"],
                        "relation_partition_id": relation,
                        "answer_type_match_tier": "same_family_same_answer_type_bucket",
                        "split_assignment": "development",
                        "same_split": True,
                        "same_leakage_component": False,
                        "verified": False,
                        "review_status": "pending_factual_uniqueness_and_semantic_review",
                    }
                )
            for slot, source in enumerate(neutral_sources, 1):
                neutrals.append(
                    {
                        "schema_version": freezer.candidate_tool.NEUTRAL_SCHEMA,
                        "neutral_candidate_id": f"n_{target['base_fact_id']}_{slot}",
                        "base_fact_id": target["base_fact_id"],
                        "slot": slot,
                        "source_base_fact_id": source["base_fact_id"],
                        "neutral_context_candidate_en": source["canonical_fact_en"],
                        "target_relation_partition_id": relation,
                        "source_relation_partition_id": other_relation,
                        "split_assignment": "development",
                        "same_split": True,
                        "same_leakage_component": False,
                        "verified_unrelated": False,
                        "review_status": "pending_unrelatedness_and_length_review",
                    }
                )

        components = [
            {
                "schema_version": freezer.postreview_tool.pre_hf.COMPONENT_SCHEMA_VERSION,
                "leakage_component_id": row["leakage_component_id"],
                "member_count": 1,
                "base_fact_ids": [row["base_fact_id"]],
                "split_assignment": "development",
            }
            for row in retained
        ]
        exclusions = [
            {
                "schema_version": freezer.postreview_tool.EXCLUSION_SCHEMA_VERSION,
                "base_fact_id": "pbf_excluded",
                "disposition": "cohort_exclude",
                "source_review_outcome": "reject",
                "human_gold": False,
            }
        ]
        split_manifest = {
            "schema_version": freezer.postreview_tool.pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_policy_version": "fixture-component-split-v1",
            "split_seed": "fixture-seed",
            "split_status": "provisional_not_frozen",
            "formal_split_freeze_performed": False,
            "base_fact_count": len(retained),
            "actual_base_fact_counts": {"development": len(retained)},
            "fact_review_reject_excluded_count": len(exclusions),
            "fact_resolution_cohort_excluded_count": len(exclusions),
            "leakage_component_count": len(components),
            "bounded_semantic_candidate_adjudication_complete": True,
            "semantic_paraphrase_closure_complete_for_full_pool": False,
            "semantic_near_duplicate_recall_guaranteed": False,
        }
        integrity = {
            "schema_version": freezer.postreview_tool.INTEGRITY_SCHEMA_VERSION,
            "all_checks_passed": True,
            "limitations": {
                "semantic_candidate_generation_is_bounded": True,
                "all_record_pairs_enumerated": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "semantic_closure_complete_for_full_pool": False,
            },
        }
        counts = {
            "input_base_facts": len(input_rows),
            "retained_base_facts": len(retained),
            "rejected_base_facts": len(exclusions),
            "revised_base_facts": 0,
            "leakage_components": len(components),
            "semantic_candidate_pairs": 1,
            "semantic_positive_edges": 0,
            "distractor_candidates": len(distractors),
            "neutral_reference_candidates": len(neutrals),
            "candidate_shortfalls": 0,
        }
        summary = {
            "schema_version": freezer.postreview_tool.SUMMARY_SCHEMA_VERSION,
            "counts": counts,
            "execution": {
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
                "hidden_state_collection_count": 0,
                "intervention_count": 0,
            },
        }
        files = {
            "input": ("input_full_base_facts.jsonl", input_rows, freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION),
            "full_base_facts": ("full_base_facts.jsonl", retained, freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION),
            "cohort_exclusions": ("cohort_exclusions.jsonl", exclusions, freezer.postreview_tool.EXCLUSION_SCHEMA_VERSION),
            "leakage_components": ("leakage_components.jsonl", components, freezer.postreview_tool.pre_hf.COMPONENT_SCHEMA_VERSION),
            "distractor_candidates": ("distractor_candidates.jsonl", distractors, freezer.candidate_tool.DISTRACTOR_SCHEMA),
            "neutral_reference_candidates": ("neutral_reference_candidates.jsonl", neutrals, freezer.candidate_tool.NEUTRAL_SCHEMA),
            "candidate_shortfalls": ("candidate_shortfalls.jsonl", [], freezer.postreview_tool.pre_hf.SHORTFALL_SCHEMA_VERSION),
        }
        paths = {}
        for label, (name, rows, _schema) in files.items():
            paths[label] = root / name
            write_jsonl(paths[label], rows)
        paths["split_manifest"] = root / "split_manifest.json"
        paths["integrity_audit"] = root / "integrity_audit.json"
        paths["summary"] = root / "summary.json"
        write_json(paths["split_manifest"], split_manifest)
        write_json(paths["integrity_audit"], integrity)
        write_json(paths["summary"], summary)
        outputs = {
            label: binding(paths[label], schema=schema, rows=rows, relative=True)
            for label, (_name, rows, schema) in files.items()
            if label != "input"
        }
        outputs.update(
            {
                "split_manifest": binding(
                    paths["split_manifest"],
                    schema=freezer.postreview_tool.pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
                    relative=True,
                ),
                "integrity_audit": binding(
                    paths["integrity_audit"],
                    schema=freezer.postreview_tool.INTEGRITY_SCHEMA_VERSION,
                    relative=True,
                ),
                "summary": binding(
                    paths["summary"],
                    schema=freezer.postreview_tool.SUMMARY_SCHEMA_VERSION,
                    relative=True,
                ),
            }
        )
        false_flags = {
            field: False
            for field in (
                "canonical_freeze_emitted",
                "review_freeze_emitted",
                "split_freeze_emitted",
                "distractor_candidates_verified",
                "neutral_candidates_verified_unrelated",
                "translation_review_complete",
                "historical_exposure_complete_for_full_pool",
                "hf_checkpoint_bound",
                "hf_tokenizer_bound",
                "hf_model_executed",
                "hf_tokenizer_executed",
                "behavior_executed",
                "validation_exposed",
                "sealed_exposed",
                "perturbation_authorized",
                "path_not_token_authorized",
                "human_gold",
            )
        }
        fact_apply_path = root / "fact_review_apply_manifest.json"
        semantic_manifest_path = root / "semantic_candidate_manifest.json"
        semantic_adjudications_path = root / "semantic_adjudications.jsonl"
        write_json(
            fact_apply_path,
            {"schema_version": freezer.postreview_tool.review_tool.APPLY_MANIFEST_SCHEMA_VERSION},
        )
        write_json(
            semantic_manifest_path,
            {
                "schema_version": freezer.postreview_tool.semantic_runner.materializer.MANIFEST_SCHEMA
            },
        )
        write_jsonl(semantic_adjudications_path, [])
        semantic_runner = freezer.postreview_tool.semantic_runner
        semantic_protocol_identity = semantic_runner.protocol_reviewer_identity(config)
        semantic_reviewer_spec = {
            "provider_profile": semantic_protocol_identity["provider_profile"],
            "model": semantic_protocol_identity["requested_model"],
        }
        semantic_route_identity = semantic_runner.build_route_identity_set(
            config=config,
            env_path=root / ".env",
            routes=[
                (
                    semantic_reviewer_spec,
                    semantic_protocol_identity["expected_response_model"],
                )
            ],
            resolved_env={},
        )
        semantic_run_contract = {
            "tool_version": semantic_runner.TOOL_VERSION,
            "config_sha256": freezer.sha256_value(config),
            "source_fingerprints": semantic_runner.source_fingerprints(),
            "route_identity": semantic_route_identity,
            "reviewer_spec": semantic_reviewer_spec,
            "expected_response_model": semantic_protocol_identity[
                "expected_response_model"
            ],
            "protocol_reviewer_identity": semantic_protocol_identity,
        }
        semantic_run_contract_sha256 = freezer.sha256_value(semantic_run_contract)
        semantic_run_manifest = {
            "schema_version": semantic_runner.RUN_MANIFEST_SCHEMA,
            "tool_version": semantic_runner.TOOL_VERSION,
            "status": "completed",
            "run_contract": semantic_run_contract,
            "run_contract_sha256": semantic_run_contract_sha256,
        }
        semantic_run_manifest_path = root / "semantic_review_run_manifest.json"
        write_json(semantic_run_manifest_path, semantic_run_manifest)
        post_manifest = {
            "schema_version": freezer.postreview_tool.MANIFEST_SCHEMA_VERSION,
            "status": "postreview_rebuilt_bounded_semantic_recall_not_frozen",
            "inputs": {
                "full_base_facts": binding(
                    paths["input"],
                    schema=freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                    rows=input_rows,
                ),
                "fact_review": {
                    "apply_manifest": binding(
                        fact_apply_path,
                        schema=freezer.postreview_tool.review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
                    )
                },
                "semantic_review": {
                    "run_manifest": binding(
                        semantic_run_manifest_path,
                        schema=semantic_runner.RUN_MANIFEST_SCHEMA,
                    ),
                    "candidate_manifest": binding(
                        semantic_manifest_path,
                        schema=freezer.postreview_tool.semantic_runner.materializer.MANIFEST_SCHEMA,
                    ),
                    "adjudications": binding(
                        semantic_adjudications_path,
                        schema=freezer.postreview_tool.semantic_runner.ADJUDICATION_SCHEMA,
                        rows=[],
                    ),
                },
            },
            "outputs": outputs,
            "counts": counts,
            "fact_review_contract": {
                "mode": "legacy_fact_review_apply",
                "scope_exactly_matches_input_universe": True,
                "selection_is_complete_retained_cohort": True,
                "source_universe_base_fact_count": len(input_rows),
                "accepted_facts_retained": len(retained),
                "revised_facts_retained_after_independent_rereview": 0,
                "facts_cohort_excluded": len(exclusions),
                "revise_defer_missing_count": 0,
                "proxy_review_only": True,
                "human_gold": False,
            },
            "semantic_review_contract": {
                "candidate_adjudication_complete_for_bounded_set": True,
                "runtime_identity_evidence_bound": True,
                "semantic_run_contract_sha256": semantic_run_contract_sha256,
                "route_identity_set_sha256": semantic_route_identity[
                    "route_identity_set_sha256"
                ],
                "candidate_generation_is_bounded": True,
                "all_record_pairs_enumerated": False,
                "semantic_near_duplicate_recall_guaranteed": False,
                "semantic_closure_complete_for_full_pool": False,
                "reviewer_evidence_is_human_gold": False,
            },
            "integrity": {
                "all_checks_passed": True,
                "integrity_audit_sha256": outputs["integrity_audit"]["sha256"],
            },
            "lineage_digests": {
                "ordered_input_base_fact_ids_sha256": freezer.sha256_value(
                    [row["base_fact_id"] for row in input_rows]
                ),
                "ordered_retained_base_fact_ids_sha256": freezer.sha256_value(
                    [row["base_fact_id"] for row in retained]
                ),
                "ordered_rejected_base_fact_ids_sha256": freezer.sha256_value(
                    [row["base_fact_id"] for row in exclusions]
                ),
                "ordered_revised_base_fact_ids_sha256": freezer.sha256_value([]),
                "ordered_component_ids_sha256": freezer.sha256_value(
                    [row["leakage_component_id"] for row in components]
                ),
                "ordered_distractor_ids_sha256": freezer.sha256_value(
                    [row["distractor_id"] for row in distractors]
                ),
                "ordered_neutral_candidate_ids_sha256": freezer.sha256_value(
                    [row["neutral_candidate_id"] for row in neutrals]
                ),
            },
            "safety_contract": {
                "output_is_provisional": True,
                "split_recomputed": True,
                **false_flags,
            },
        }
        post_path = root / "postreview_rebuild_manifest.json"
        write_json(post_path, post_manifest)

        _, _, candidate_inputs, units = freezer.candidate_tool.load_review_units(
            candidate_manifest_path=post_path
        )
        candidate_adjudications = []
        reviewer_spec = {"provider_profile": "openai", "model": "gpt-5.5"}
        for unit in units:
            distractor = unit.candidate_kind == "distractor"
            candidate_adjudications.append(
                {
                    "schema_version": freezer.candidate_tool.ADJUDICATION_SCHEMA,
                    "candidate_id": unit.candidate_id,
                    "candidate_kind": unit.candidate_kind,
                    "candidate_row_sha256": unit.candidate_row_sha256,
                    "target_base_fact_id": unit.target_base_fact_id,
                    "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
                    "source_base_fact_id": unit.source_base_fact_id,
                    "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
                    "overall_decision": "accept",
                    "distractor_factually_false": True if distractor else None,
                    "answer_unique": True if distractor else None,
                    "answer_alias_disjoint": True if distractor else None,
                    "neutral_semantically_neutral": None if distractor else True,
                    "neutral_unrelated": None if distractor else True,
                    "overlapping_target_aliases": [] if distractor else None,
                    "answer_alias_disjoint_unresolved_evidence": None,
                    "rationale": "fixture independently accepts all required fields",
                    "confidence": "high",
                    "reviewer_type": "independent_model_proxy",
                    "reviewer_id": "openai:gpt-5.5-2026-04-24",
                    "requested_model": "gpt-5.5",
                    "expected_response_model": "gpt-5.5-2026-04-24",
                    "response_model": "gpt-5.5-2026-04-24",
                    "response_model_identity_status": "matched",
                    "review_method": freezer.candidate_tool.REVIEW_METHOD,
                    "reviewed_at": "2026-09-23T00:00:00+00:00",
                    "behavior_blind": True,
                    "target_behavior_consumed": False,
                    "human_gold": False,
                }
            )
        candidate_adjudications_path = root / "candidate_semantic_adjudications.jsonl"
        write_jsonl(candidate_adjudications_path, candidate_adjudications)
        unit_ids = [unit.candidate_id for unit in units]
        ids_by_kind = {
            kind: [unit.candidate_id for unit in units if unit.candidate_kind == kind]
            for kind in sorted(freezer.candidate_tool.CANDIDATE_KINDS)
        }
        candidate_route_identity = freezer.candidate_tool.build_route_identity_set(
            config=config,
            env_path=root / ".env",
            routes=[
                (
                    reviewer_spec,
                    "gpt-5.5-2026-04-24",
                )
            ],
            resolved_env={},
        )
        candidate_protocol_identity = {
            "provider_profile": "openai",
            "requested_model": "gpt-5.5",
            "expected_response_model": "gpt-5.5-2026-04-24",
        }
        review_semantics_contract = freezer.candidate_tool.review_semantics_contract(
            reviewer_spec=reviewer_spec,
            expected_response_model="gpt-5.5-2026-04-24",
            protocol_identity=candidate_protocol_identity,
            route_identity=candidate_route_identity,
        )
        review_semantics_contract_sha256 = freezer.sha256_value(
            review_semantics_contract
        )
        planned_evidence_partition = {
            "fresh_model_review_candidate_count": len(units),
            "carried_forward_candidate_count": 0,
            "fresh_model_review_candidate_ids_sha256": freezer.sha256_value(
                unit_ids
            ),
            "carried_forward_candidate_ids_sha256": freezer.sha256_value([]),
            "full_evidence_partition_complete": True,
        }
        run_contract = {
            "tool_version": freezer.candidate_tool.TOOL_VERSION,
            "candidate_manifest": {
                "path": str(post_path),
                "sha256": freezer.sha256_file(post_path),
                "schema_version": post_manifest["schema_version"],
                "manifest_kind": "postreview_rebuild",
            },
            "input_artifacts": candidate_inputs,
            "selected_candidate_ids_sha256": freezer.sha256_value(unit_ids),
            "selected_candidate_count": len(units),
            "selected_candidate_ids_by_kind": {
                kind: {
                    "record_count": len(ids),
                    "ordered_ids_sha256": freezer.sha256_value(ids),
                }
                for kind, ids in ids_by_kind.items()
            },
            "full_candidate_count": len(units),
            "selection_is_full_candidate_set": True,
            "config_sha256": freezer.sha256_value(config),
            "source_fingerprints": freezer.candidate_tool.source_fingerprints(),
            "route_identity": candidate_route_identity,
            "reviewer_spec": reviewer_spec,
            "expected_response_model": "gpt-5.5-2026-04-24",
            "protocol_reviewer_identity": candidate_protocol_identity,
            "prompt_contract_sha256": freezer.sha256_value(
                freezer.candidate_tool.prompt_contract_payload()
            ),
            "review_semantics_contract": review_semantics_contract,
            "review_semantics_contract_sha256": review_semantics_contract_sha256,
            "evidence_reuse_contract": {
                "schema_version": freezer.candidate_tool.EVIDENCE_REUSE_CONTRACT_SCHEMA,
                "mode": "fresh_full_v3_baseline",
                "source_review_manifest": None,
                "source_run_contract_sha256": None,
                "direct_predecessor_lineage": None,
                "identity_schema": freezer.candidate_tool.EVIDENCE_IDENTITY_SCHEMA,
                "only_prior_accept": True,
                "exact_nonaccept_policy": "fail_closed",
                "changed_or_new_policy": "fresh_model_review",
                "operator_rereview_override_allowed": False,
                "planned_evidence_partition": planned_evidence_partition,
            },
            "batch_size": 8,
            "checkpoint_compact_every": 128,
            "max_workers": 2,
            "answer_alias_contract": {
                "answer_alias_disjoint_scope": "target_fact.answer_aliases_en_only",
                "candidate_source_fact_role": "provenance_only",
                "candidate_source_aliases_in_scope": False,
                "contradictory_alias_evidence_fails_closed": True,
            },
            "behavior_blind": True,
            "human_gold": False,
        }
        run_sha = freezer.sha256_value(run_contract)
        checkpoints = []
        for index, (unit, decision) in enumerate(zip(units, candidate_adjudications)):
            model_verdict = {
                "candidate_id": unit.candidate_id,
                "candidate_kind": unit.candidate_kind,
                "overall_decision": decision["overall_decision"],
                **{
                    field: decision[field]
                    for field in freezer.candidate_tool.FIELD_NAMES
                },
                "rationale": decision["rationale"],
                "confidence": decision["confidence"],
            }
            checkpoints.append(
                {
                    "schema_version": freezer.candidate_tool.CHECKPOINT_SCHEMA,
                    "tool_version": freezer.candidate_tool.TOOL_VERSION,
                    "candidate_id": unit.candidate_id,
                    "candidate_kind": unit.candidate_kind,
                    "input_index": unit.input_index,
                    "candidate_row_sha256": unit.candidate_row_sha256,
                    "target_base_fact_id": unit.target_base_fact_id,
                    "target_base_fact_row_sha256": unit.target_base_fact_row_sha256,
                    "source_base_fact_id": unit.source_base_fact_id,
                    "source_base_fact_row_sha256": unit.source_base_fact_row_sha256,
                    "run_contract_sha256": run_sha,
                    "review_semantics_contract_sha256": review_semantics_contract_sha256,
                    "projection_sha256": freezer.candidate_tool.unit_projection_sha256(
                        unit
                    ),
                    "evidence_identity_sha256": freezer.candidate_tool.unit_evidence_identity_sha256(
                        unit, review_semantics_contract_sha256
                    ),
                    "current_unit_binding_sha256": freezer.candidate_tool.current_unit_binding_sha256(
                        unit
                    ),
                    "evidence_origin": freezer.candidate_tool.FRESH_EVIDENCE_ORIGIN,
                    "model_calls_scope": "current_run",
                    "carry_forward_provenance": None,
                    "terminal_status": "completed",
                    "failure_stage": None,
                    "model_verdict": model_verdict,
                    "strict_adjudication": decision,
                    "behavior_blind": True,
                    "human_gold": False,
                    "hf_model_execution_count": 0,
                    "hf_tokenizer_execution_count": 0,
                    "behavior_execution_count": 0,
                    "validation_behavior_exposure_count": 0,
                    "sealed_behavior_exposure_count": 0,
                    "model_calls": [
                        {
                            "prompt_version": freezer.candidate_tool.PROMPT_VERSION,
                            "provider_profile": "openai",
                            "requested_model": "gpt-5.5",
                            "expected_response_model": "gpt-5.5-2026-04-24",
                            "response_model": "gpt-5.5-2026-04-24",
                            "response_model_identity_status": "matched",
                        }
                    ],
                    "retry_history": [],
                    "completed_at": "2026-09-23T00:00:00+00:00",
                }
            )
        checkpoint_path = root / "candidate_review_checkpoint.jsonl"
        write_jsonl(checkpoint_path, checkpoints)
        kind_counts = {kind: len(ids) for kind, ids in ids_by_kind.items()}
        candidate_manifest = {
            "schema_version": freezer.candidate_tool.RUN_MANIFEST_SCHEMA,
            "tool_version": freezer.candidate_tool.TOOL_VERSION,
            "status": "completed",
            "run_contract": run_contract,
            "run_contract_sha256": run_sha,
            "invocation_resumed": False,
            "started_or_resumed_at": "2026-09-23T00:00:00+00:00",
            "completed_at": "2026-09-23T00:00:00+00:00",
            "selected_candidate_count": len(units),
            "full_candidate_count": len(units),
            "selection_is_full_candidate_set": True,
            "selected_candidate_ids_sha256": freezer.sha256_value(unit_ids),
            "selected_candidate_ids_by_kind": run_contract[
                "selected_candidate_ids_by_kind"
            ],
            "terminal_status_counts": {"completed": len(units)},
            "completed_kind_counts": kind_counts,
            "accepted_kind_counts": kind_counts,
            "overall_decision_counts": {"accept": len(units)},
            "evidence_origin_counts": {
                freezer.candidate_tool.FRESH_EVIDENCE_ORIGIN: len(units)
            },
            **planned_evidence_partition,
            "candidate_review_complete_for_selected_scope": True,
            "candidate_review_complete_for_full_set": True,
            "formal_review_complete": False,
            "batch_size": 8,
            "checkpoint_compact_every": 128,
            "max_workers": 2,
            "resume_enabled": False,
            "retry_failed_enabled": False,
            "evidence_boundary": {
                "reviewer_type": "independent_model_proxy",
                "human_gold": False,
                "behavior_blind": True,
                "target_behavior_consumed": False,
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
                "review_freeze_performed": False,
                "split_freeze_performed": False,
                "candidate_promotion_performed": False,
            },
            "artifacts": {
                "checkpoint": binding(
                    checkpoint_path,
                    schema=freezer.candidate_tool.CHECKPOINT_SCHEMA,
                    rows=checkpoints,
                ),
                "adjudications": binding(
                    candidate_adjudications_path,
                    schema=freezer.candidate_tool.ADJUDICATION_SCHEMA,
                    rows=candidate_adjudications,
                ),
            },
            "execution_policy": {
                "one_model_request_per_batch": True,
                "real_batch_size": 8,
                "max_workers": 2,
                "recursive_binary_split_on_batch_failure": True,
                "response_model_identity_fail_closed": True,
                "contradictory_alias_evidence_fail_closed": True,
                "answer_alias_disjoint_uses_target_aliases_only": True,
                "single_writer_append_journal_with_atomic_compaction": True,
                "journal_fsync_after_each_commit": True,
                "incomplete_final_journal_line_ignored_on_resume": True,
                "terminal_journal_removed": True,
                "resume_supported": True,
                "retry_failed_supported": True,
            },
            "limitations": {
                "proxy_review_only": True,
                "human_gold": False,
                "candidate_rows_promoted": False,
                "review_freeze_emitted": False,
                "split_freeze_emitted": False,
                "pre_hf_compatibility_input_is_not_final_postreview_input": False,
            },
        }
        candidate_manifest_path = root / "candidate_review_run_manifest.json"
        write_json(candidate_manifest_path, candidate_manifest)

        contract_path = root / "historical_contract.json"
        attestation_path = root / "scope_owner_attestation.json"
        write_json(contract_path, {"schema_version": freezer.exposure_tool.CONTRACT_SCHEMA_VERSION})
        write_json(
            attestation_path,
            {
                "schema_version": freezer.exposure_tool.ATTESTATION_SCHEMA_VERSION,
                "known_historical_exposure_source_base_fact_ids": [],
            },
        )
        source_universe_id = "formal_cohort_source_fixture"
        source_universe_items = [
            {
                "schema_version": freezer.exposure_tool.SOURCE_ITEM_SCHEMA_VERSION,
                "universe_id": source_universe_id,
                "selection_index": index,
                "cohort_item_id": f"item_{row['base_fact_id']}",
                "source_base_fact_id": row["base_fact_id"],
            }
            for index, row in enumerate(input_rows)
        ]
        source_universe_items_path = root / "source_formal_cohort_universe_items.jsonl"
        write_jsonl(source_universe_items_path, source_universe_items)
        source_universe_manifest_path = root / "source_formal_cohort_universe_manifest.json"
        write_json(
            source_universe_manifest_path,
            {
                "schema_version": freezer.exposure_tool.SOURCE_UNIVERSE_SCHEMA_VERSION,
                "universe_id": source_universe_id,
                "universe_status": "declared_immutable_not_reviewed",
                "record_count": len(source_universe_items),
                "selection_policy": {"mode": "full-pool"},
                "items": binding(
                    source_universe_items_path,
                    schema=freezer.exposure_tool.SOURCE_ITEM_SCHEMA_VERSION,
                    rows=source_universe_items,
                ),
            },
        )
        source_universe_manifest_binding = binding(
            source_universe_manifest_path,
            schema=freezer.exposure_tool.SOURCE_UNIVERSE_SCHEMA_VERSION,
        )
        source_universe_items_binding = binding(
            source_universe_items_path,
            schema=freezer.exposure_tool.SOURCE_ITEM_SCHEMA_VERSION,
            rows=source_universe_items,
        )

        universe_items = [
            {
                "schema_version": freezer.formal_universe_tool.ITEM_SCHEMA_VERSION,
                "universe_id": "formal_cohort_fixture",
                "selection_index": index,
                "cohort_item_id": f"item_{row['base_fact_id']}",
                "source_base_fact_id": row["base_fact_id"],
            }
            for index, row in enumerate(input_rows)
        ]
        universe_items_path = root / "formal_cohort_universe_items.jsonl"
        write_jsonl(universe_items_path, universe_items)
        universe_manifest = {
            "schema_version": freezer.formal_universe_tool.MANIFEST_SCHEMA_VERSION,
            "universe_id": "formal_cohort_fixture",
            "universe_status": freezer.formal_universe_tool.UNIVERSE_STATUS,
            "record_count": len(universe_items),
            "lineage": {
                "predecessor_universe_id": source_universe_id,
                "predecessor_manifest": source_universe_manifest_binding,
                "predecessor_items": source_universe_items_binding,
            },
            "ordered_source_base_fact_ids_sha256": freezer.sha256_value(
                [row["source_base_fact_id"] for row in universe_items]
            ),
            "items": binding(
                universe_items_path,
                schema=freezer.formal_universe_tool.ITEM_SCHEMA_VERSION,
                rows=universe_items,
            ),
            "historical_exposure_authority": {
                "contract_id": "historical_exposure_contract_fixture",
                "attestation_challenge_sha256": "2" * 64,
                "contract": binding(
                    contract_path,
                    schema=freezer.exposure_tool.CONTRACT_SCHEMA_VERSION,
                ),
            },
        }
        universe_manifest_path = root / "formal_cohort_universe_manifest.json"
        write_json(universe_manifest_path, universe_manifest)

        distractors_by_fact = freezer._candidate_rows_by_fact(
            distractors, id_field="distractor_id"
        )
        neutrals_by_fact = freezer._candidate_rows_by_fact(
            neutrals, id_field="neutral_candidate_id"
        )
        retained_ids = [row["base_fact_id"] for row in retained]
        zh_items = {
            base_fact_id: freezer._zh_item(
                base_fact_id=base_fact_id,
                cohort_item_id=f"item_{base_fact_id}",
                fact=next(row for row in retained if row["base_fact_id"] == base_fact_id),
                distractors=distractors_by_fact[base_fact_id],
                neutrals=neutrals_by_fact[base_fact_id],
            )
            for base_fact_id in retained_ids
        }
        generator_model = {
            "provider_profile": "aliyun",
            "model": "qwen3.8-max-0902/bailian/bailian",
            "expected_response_model": "qwen3.8-max",
            "role": "translation_generation_and_single_repair",
        }
        zh_reviewer_model = {
            "provider_profile": "openai",
            "model": "gpt-5.5",
            "expected_response_model": "gpt-5.5-2026-04-24",
            "role": "independent_translation_equivalence_review",
        }
        zh_route_identity = freezer.zh_tool.build_route_identity_set(
            config=config,
            env_path=root / ".env",
            routes=[
                (generator_model, generator_model["expected_response_model"]),
                (zh_reviewer_model, zh_reviewer_model["expected_response_model"]),
            ],
            resolved_env={},
        )
        initial_rows = []
        review_rows = []
        final_rows = []
        for base_fact_id in retained_ids:
            item = zh_items[base_fact_id]
            translation = {
                "prompt_zh": "项目是",
                "canonical_fact_zh": "项目对应答案。",
                "answer_zh": "答案",
                "answer_aliases_zh": ["答案", "答案别名"],
                "distractors": [
                    {"distractor_id": row["distractor_id"], "text_zh": "干扰答案"}
                    for row in item["distractors"]
                ],
                "neutral_candidates": [
                    {"neutral_candidate_id": row["neutral_candidate_id"], "text_zh": "中性事实。"}
                    for row in item["neutral_candidates"]
                ],
            }
            initial = {
                "schema_version": freezer.zh_tool.GENERATION_SCHEMA,
                "base_fact_id": base_fact_id,
                "input_record_sha256": item["input_record_sha256"],
                "target_language": "zh",
                "human_gold": False,
                "stage": "initial_generation",
                "terminal_status": "completed",
                "request_model": generator_model["model"],
                "provider_profile": generator_model["provider_profile"],
                "expected_response_model": generator_model["expected_response_model"],
                "response_model": generator_model["expected_response_model"],
                "response_model_identity_status": "matched",
                "credentials_or_endpoints_included": False,
                "parsed_response": translation,
            }
            checks = {
                "prompt": {"translation_equivalent": True, "natural_zh": True, "issues": []},
                "canonical_fact": {"translation_equivalent": True, "natural_zh": True, "issues": []},
                "answer": {"translation_equivalent": True, "natural_zh": True, "issues": []},
                "answer_aliases": [
                    {"alias_index": index, "translation_equivalent": True, "natural_zh": True, "issues": []}
                    for index in range(2)
                ],
                "distractors": [
                    {"distractor_id": row["distractor_id"], "translation_equivalent": True, "natural_zh": True, "issues": []}
                    for row in item["distractors"]
                ],
                "neutral_candidates": [
                    {"neutral_candidate_id": row["neutral_candidate_id"], "translation_equivalent": True, "natural_zh": True, "issues": []}
                    for row in item["neutral_candidates"]
                ],
            }
            review = {
                "schema_version": freezer.zh_tool.REVIEW_SCHEMA,
                "base_fact_id": base_fact_id,
                "input_record_sha256": item["input_record_sha256"],
                "translation_record_sha256": freezer.sha256_value(initial),
                "target_language": "zh",
                "human_gold": False,
                "reviewer_type": "independent_model_proxy",
                "stage": "initial_review",
                "terminal_status": "completed",
                "request_model": zh_reviewer_model["model"],
                "provider_profile": zh_reviewer_model["provider_profile"],
                "expected_response_model": zh_reviewer_model["expected_response_model"],
                "response_model": zh_reviewer_model["expected_response_model"],
                "response_model_identity_status": "matched",
                "credentials_or_endpoints_included": False,
                "parsed_response": {"decision": "accept", "issues": [], "checks": checks},
            }
            initial_rows.append(initial)
            review_rows.append(review)
            final_rows.append(
                {
                    "schema_version": freezer.zh_tool.FINAL_SCHEMA,
                    "base_fact_id": base_fact_id,
                    "cohort_item_id": item["cohort_item_id"],
                    "input_record_sha256": item["input_record_sha256"],
                    "split_assignment": item["split_assignment"],
                    "target_language": "zh",
                    "human_gold": False,
                    "reviewer_type": "independent_model_proxy",
                    "terminal_status": "completed",
                    "translation_origin": "initial",
                    "source_fields": item["source"],
                    "source_distractors": item["distractors"],
                    "source_neutral_candidates": item["neutral_candidates"],
                    "translation": translation,
                    "translation_equivalence_review": review["parsed_response"],
                    "quarantine_reasons": [],
                    "maximum_translation_repairs_performed": 0,
                    "formal_claims": {
                        "translation_proxy_accepted": True,
                        "human_gold": False,
                        "factual_truth_reviewed": False,
                        "distractor_falsehood_reviewed": False,
                        "neutral_unrelatedness_reviewed": False,
                    },
                    "lineage": {
                        "initial_generation_record_sha256": freezer.sha256_value(initial),
                        "initial_review_record_sha256": freezer.sha256_value(review),
                        "repair_record_sha256": None,
                        "repair_review_record_sha256": None,
                    },
                }
            )
        zh_files = {
            "translation_initial": (root / "translation_initial.jsonl", initial_rows),
            "translation_initial_reviews": (root / "translation_initial_reviews.jsonl", review_rows),
            "translation_repairs": (root / "translation_repairs.jsonl", []),
            "translation_repair_reviews": (root / "translation_repair_reviews.jsonl", []),
            "zh_translation_review_records": (root / "zh_translation_review_records.jsonl", final_rows),
            "translation_quarantine": (root / "translation_quarantine.jsonl", []),
        }
        for _label, (path, rows) in zh_files.items():
            write_jsonl(path, rows)
        zh_inputs = {
            "formal_universe_manifest": source_universe_manifest_binding,
            "formal_universe_items": source_universe_items_binding,
            "full_base_facts": zh_binding(paths["full_base_facts"], retained),
            "distractor_candidates": zh_binding(paths["distractor_candidates"], distractors),
            "neutral_candidates": zh_binding(paths["neutral_reference_candidates"], neutrals),
            "split_manifest": binding(
                paths["split_manifest"],
                schema=freezer.postreview_tool.pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
            ),
        }
        zh_candidate_binding = {
            "distractor_candidates": zh_inputs["distractor_candidates"],
            "neutral_candidates": zh_inputs["neutral_candidates"],
        }
        zh_candidate_binding["combined_sha256"] = freezer.sha256_value(
            {
                key: value["sha256"]
                for key, value in sorted(zh_candidate_binding.items())
            }
        )
        zh_contract = {
            "schema_version": freezer.zh_tool.MANIFEST_SCHEMA,
            "tool_version": freezer.zh_tool.TOOL_VERSION,
            "input_bindings": zh_inputs,
            "source_fingerprints": freezer.zh_tool.source_fingerprints(),
            "route_identity": zh_route_identity,
            "review_independence": {
                "model_identity_distinct": True,
                "provider_profile_labels_distinct": True,
                "provider_endpoint_distinct": True,
                "provider_infrastructure_independence_claimed": False,
                "claim_scope": "distinct_model_identity_only",
            },
            "formal_universe_id": source_universe_id,
            "formal_universe_status": "declared_immutable_not_reviewed",
            "selected_base_fact_ids_sha256": freezer.sha256_value(retained_ids),
            "selected_record_count": len(retained_ids),
            "universe_record_count": len(universe_items),
            "selection_is_full_universe": False,
            "selected_split_counts": {"development": len(retained_ids), "validation": 0, "sealed": 0},
            "models": {"generator": generator_model, "reviewer": zh_reviewer_model},
            "candidate_binding": zh_candidate_binding,
            "invalidation_contract": {
                "provisional_candidate_binding": True,
                "invalidated_by_recluster_or_candidate_regeneration": True,
                "invalidated_by_split_change": True,
                "resume_requires_identical_run_fingerprint": True,
            },
            "language_scope": {
                "processed_language_codes": ["zh"],
                "other_registered_target_language_codes": list(freezer.zh_tool.OTHER_TARGET_LANGUAGES),
                "other_registered_target_languages_status": "unchanged_pending_translation",
                "does_not_mark_unprocessed_languages_complete": True,
            },
            "evidence_boundary": {
                "human_gold": False,
                "reviewer_type": "independent_model_proxy",
                "independence_scope": "distinct_model_identity_only",
                "provider_infrastructure_independence_claimed": False,
                "translation_only": True,
                "factual_truth_review_performed": False,
                "distractor_factual_falsehood_review_performed": False,
                "neutral_unrelatedness_review_performed": False,
                "formal_relation_freeze_performed": False,
                "formal_split_freeze_performed": False,
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
            },
            "execution_policy": {"fixture": True},
            "selection_is_complete_retained_cohort": True,
            "excluded_base_fact_ids_sha256": freezer.sha256_value(["pbf_excluded"]),
        }
        zh_manifest = {
            **zh_contract,
            "run_fingerprint": freezer.sha256_value(zh_contract),
            "status": "completed_full_zh_proxy_review",
            "counts": {
                "universe_records": len(universe_items),
                "selected_records": len(retained_ids),
                "proxy_accepted_records": len(retained_ids),
                "quarantined_records": 0,
                "initial_review_accepted_records": len(retained_ids),
                "repair_attempted_records": 0,
                "repair_review_accepted_records": 0,
            },
            "completion_claims": {
                "selected_scope_processing_complete": True,
                "full_pool_zh_translation_review_complete": True,
                "zh_proxy_review_not_human_gold": True,
                "other_15_target_languages_complete": False,
                "hf_or_behavior_work_performed": False,
            },
            "output_artifacts": {
                label: zh_binding(path, rows) for label, (path, rows) in zh_files.items()
            },
        }
        zh_manifest_path = root / "zh_run_manifest.json"
        write_json(zh_manifest_path, zh_manifest)

        attestation_binding = binding(
            attestation_path, schema=freezer.exposure_tool.ATTESTATION_SCHEMA_VERSION
        )
        attestation_binding.update(
            {
                "contract_id": "historical_exposure_contract_fixture",
                "attestation_challenge_sha256": "2" * 64,
                "known_historical_exposure_count": 0,
                "ordered_known_historical_exposure_source_base_fact_ids_sha256": freezer.sha256_value([]),
            }
        )
        formal_validation = {
            "status": "valid_pending_successor_with_compatible_attestation_candidate",
            "future_attestation_binding_candidate": attestation_binding,
        }
        exposure_validation = {
            "scope_owner_attestation_valid": True,
            "ready_for_successor_universe_binding": True,
            "contract_id": "historical_exposure_contract_fixture",
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": freezer.sha256_file(contract_path),
            "attestation_path": str(attestation_path.resolve()),
            "attestation_sha256": freezer.sha256_file(attestation_path),
            "attested_by": "fixture owner",
            "attested_at": "2026-09-23T00:00:00+00:00",
            "attested_through": "2026-09-23T00:00:00+00:00",
            "known_historical_exposure_count": 0,
            "known_historical_exposure_source_base_fact_ids": [],
        }
        return {
            "config": config_path,
            "postreview": post_path,
            "semantic_run_manifest": semantic_run_manifest_path,
            "semantic_run_manifest_value": semantic_run_manifest,
            "candidate_manifest": candidate_manifest_path,
            "candidate_adjudications": candidate_adjudications_path,
            "zh_manifest": zh_manifest_path,
            "universe_manifest": universe_manifest_path,
            "contract": contract_path,
            "attestation": attestation_path,
            "formal_validation": formal_validation,
            "exposure_validation": exposure_validation,
            "shortfalls": paths["candidate_shortfalls"],
            "candidate_manifest_value": candidate_manifest,
            "candidate_adjudication_rows": candidate_adjudications,
            "candidate_checkpoint_rows": checkpoints,
            "candidate_units": units,
            "zh_manifest_value": zh_manifest,
            "retained": retained,
            "universe_items": universe_items,
        }

    def run_freeze(self, fixture, output):
        with mock.patch.object(
            freezer.formal_universe_tool,
            "validate_successor_universe",
            return_value=fixture["formal_validation"],
        ), mock.patch.object(
            freezer.exposure_tool,
            "validate_contract",
            return_value=fixture["exposure_validation"],
        ), mock.patch.object(
            freezer,
            "_deterministically_replay_postreview",
            return_value=None,
        ):
            return freezer.freeze(
                postreview_rebuild_manifest_path=fixture["postreview"],
                candidate_review_manifest_path=fixture["candidate_manifest"],
                candidate_adjudications_path=fixture["candidate_adjudications"],
                accepted_candidate_projection_manifest_path=fixture.get(
                    "candidate_projection"
                ),
                zh_accepted_projection_manifest_path=fixture.get(
                    "zh_accepted_projection"
                ),
                zh_review_manifest_path=fixture["zh_manifest"],
                formal_universe_v2_manifest_path=fixture["universe_manifest"],
                historical_exposure_contract_path=fixture["contract"],
                scope_owner_attestation_path=fixture["attestation"],
                output_dir=output,
                config_path=fixture["config"],
                expected_original_universe_count=len(fixture["universe_items"]),
                now_fn=lambda: "2026-09-23T01:00:00+00:00",
            )

    def rewrite_candidate_contract(self, fixture):
        manifest = fixture["candidate_manifest_value"]
        run_contract = manifest["run_contract"]
        if all(
            isinstance(run_contract.get(field), dict)
            for field in (
                "reviewer_spec",
                "protocol_reviewer_identity",
                "route_identity",
            )
        ) and isinstance(run_contract.get("expected_response_model"), str):
            semantics = freezer.candidate_tool.review_semantics_contract(
                reviewer_spec=run_contract["reviewer_spec"],
                expected_response_model=run_contract["expected_response_model"],
                protocol_identity=run_contract["protocol_reviewer_identity"],
                route_identity=run_contract["route_identity"],
            )
            semantics_sha = freezer.sha256_value(semantics)
            run_contract["review_semantics_contract"] = semantics
            run_contract["review_semantics_contract_sha256"] = semantics_sha
        run_contract_sha256 = freezer.sha256_value(manifest["run_contract"])
        manifest["run_contract_sha256"] = run_contract_sha256
        checkpoints = fixture["candidate_checkpoint_rows"]
        for checkpoint, unit in zip(checkpoints, fixture["candidate_units"]):
            checkpoint["run_contract_sha256"] = run_contract_sha256
            if "semantics_sha" in locals():
                checkpoint["review_semantics_contract_sha256"] = semantics_sha
                checkpoint[
                    "evidence_identity_sha256"
                ] = freezer.candidate_tool.unit_evidence_identity_sha256(
                    unit, semantics_sha
                )
        checkpoint_path = Path(manifest["artifacts"]["checkpoint"]["path"])
        write_jsonl(checkpoint_path, checkpoints)
        manifest["artifacts"]["checkpoint"] = binding(
            checkpoint_path,
            schema=freezer.candidate_tool.CHECKPOINT_SCHEMA,
            rows=checkpoints,
        )
        write_json(fixture["candidate_manifest"], manifest)

    def rewrite_semantic_runtime(self, fixture):
        run_manifest = fixture["semantic_run_manifest_value"]
        run_contract = run_manifest["run_contract"]
        run_manifest["run_contract_sha256"] = freezer.sha256_value(run_contract)
        write_json(fixture["semantic_run_manifest"], run_manifest)
        postreview = freezer.read_json(fixture["postreview"])
        postreview["inputs"]["semantic_review"]["run_manifest"] = binding(
            fixture["semantic_run_manifest"],
            schema=freezer.postreview_tool.semantic_runner.RUN_MANIFEST_SCHEMA,
        )
        postreview["semantic_review_contract"][
            "semantic_run_contract_sha256"
        ] = run_manifest["run_contract_sha256"]
        postreview["semantic_review_contract"][
            "route_identity_set_sha256"
        ] = run_contract["route_identity"]["route_identity_set_sha256"]
        write_json(fixture["postreview"], postreview)

    def rewrite_zh_contract(self, fixture):
        manifest = fixture["zh_manifest_value"]
        refresh_zh_run_fingerprint(manifest)
        write_json(fixture["zh_manifest"], manifest)

    def test_complete_contract_emits_four_frozen_artifacts_without_hf_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            output = root / "freeze"
            result = self.run_freeze(fixture, output)
            self.assertEqual(result["status"], freezer.STATUS)
            self.assertEqual(result["retained_frozen_count"], 6)
            self.assertEqual(result["cohort_excluded_count"], 1)
            self.assertFalse(result["semantic_near_duplicate_recall_guaranteed"])
            rows = freezer.read_jsonl(output / "exact_reviewed_bundle.jsonl")
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["canonical_status"] == "frozen" for row in rows))
            self.assertTrue(all(row["split_status"] == "frozen" for row in rows))
            self.assertTrue(all(row["human_gold"] is False for row in rows))
            self.assertTrue(all(row["probe_relation_id"] is None for row in rows))
            self.assertTrue(all(len(row["distractors_en"]) == 2 for row in rows))
            self.assertTrue(all(len(row["neutral_candidates_en"]) == 2 for row in rows))
            self.assertTrue(
                all(
                    candidate["verified"] is True
                    and candidate["human_gold"] is False
                    for row in rows
                    for candidate in row["distractors_en"]
                )
            )
            self.assertTrue(
                all(
                    candidate["verified_unrelated"] is True
                    and candidate["human_gold"] is False
                    for row in rows
                    for candidate in row["neutral_candidates_en"]
                )
            )
            gate = freezer.read_json(output / "pre_exact_hf_gate_manifest.json")
            self.assertTrue(gate["pre_exact_hf_gate_passed"])
            self.assertFalse(gate["claim_boundaries"]["semantic_near_duplicate_recall_guaranteed"])
            self.assertFalse(gate["authorization"]["behavior_execution_authorized"])
            self.assertTrue(
                gate["authorization"]["ready_for_exact_hf_checkpoint_and_tokenizer_binding"]
            )
            review_freeze = freezer.read_json(output / "review_freeze_manifest.json")
            candidate_evidence = review_freeze["candidate_review"]
            self.assertEqual(
                candidate_evidence["evidence_origin_counts"],
                {
                    freezer.candidate_tool.FRESH_EVIDENCE_ORIGIN: len(
                        fixture["candidate_units"]
                    )
                },
            )
            self.assertEqual(
                candidate_evidence["fresh_model_review_candidate_count"],
                len(fixture["candidate_units"]),
            )
            self.assertEqual(
                candidate_evidence["carried_forward_candidate_count"], 0
            )
            self.assertTrue(candidate_evidence["full_evidence_partition_complete"])
            self.assertEqual(len(candidate_evidence["complete_evidence_lineage"]), 1)
            self.assertEqual(
                candidate_evidence["complete_evidence_lineage"][0]["checkpoint"][
                    "sha256"
                ],
                candidate_evidence["checkpoint"]["sha256"],
            )

    def test_candidate_evidence_partition_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            manifest = fixture["candidate_manifest_value"]
            manifest["evidence_origin_counts"] = {
                freezer.candidate_tool.FRESH_EVIDENCE_ORIGIN: len(
                    fixture["candidate_units"]
                )
                - 1
            }
            write_json(fixture["candidate_manifest"], manifest)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError,
                "evidence lineage is invalid.*origin counts",
            ):
                self.run_freeze(fixture, root / "freeze")

    def test_actual_zh_successor_manifest_with_exclusion_passes_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            retained = fixture["retained"]
            retained_ids = [row["base_fact_id"] for row in retained]
            excluded_ids = ["pbf_excluded"]
            distractors = freezer.read_jsonl(root / "distractor_candidates.jsonl")
            neutrals = freezer.read_jsonl(
                root / "neutral_reference_candidates.jsonl"
            )
            distractors_by_fact = freezer._candidate_rows_by_fact(
                distractors, id_field="distractor_id"
            )
            neutrals_by_fact = freezer._candidate_rows_by_fact(
                neutrals, id_field="neutral_candidate_id"
            )
            zh_items = [
                freezer._zh_item(
                    base_fact_id=base_fact_id,
                    cohort_item_id=f"item_{base_fact_id}",
                    fact=next(
                        row for row in retained if row["base_fact_id"] == base_fact_id
                    ),
                    distractors=distractors_by_fact[base_fact_id],
                    neutrals=neutrals_by_fact[base_fact_id],
                )
                for base_fact_id in retained_ids
            ]
            prior_zh = fixture["zh_manifest_value"]
            successor = {
                "successor_universe_id": "resolved_fixture",
                "source_formal_universe_id": "formal_cohort_source_fixture",
                "selection_is_complete_retained_cohort": True,
                "source_record_count": len(fixture["universe_items"]),
                "retained_record_count": len(retained_ids),
                "revised_record_count": 0,
                "excluded_record_count": len(excluded_ids),
                "ordered_retained_base_fact_ids_sha256": freezer.sha256_value(
                    retained_ids
                ),
                "ordered_excluded_base_fact_ids_sha256": freezer.sha256_value(
                    excluded_ids
                ),
                "excluded_base_fact_ids_sha256": freezer.sha256_value(
                    sorted(excluded_ids)
                ),
                "bindings": {
                    "fact_resolution_manifest": {"sha256": "1" * 64},
                    "postreview_rebuild_manifest": {"sha256": "2" * 64},
                },
            }
            loaded = {
                "formal_manifest": {
                    "universe_id": "formal_cohort_source_fixture",
                    "universe_status": "declared_immutable_not_reviewed",
                },
                "bindings": copy.deepcopy(prior_zh["input_bindings"]),
                "candidate_binding": copy.deepcopy(prior_zh["candidate_binding"]),
                "items": zh_items,
                "split_counts": {"development": len(retained_ids), "validation": 0, "sealed": 0},
                "successor_cohort": successor,
                "accepted_candidate_projection": None,
            }
            zh_config = root / "zh-config.json"
            write_json(
                zh_config,
                {
                    "provider_profiles": {
                        "aliyun": {
                            "protocol": "openai_compatible",
                            "base_url": "https://aliyun-fixture.invalid/v1",
                        },
                        "openai": {
                            "protocol": "openai_compatible",
                            "base_url": "https://openai-fixture.invalid/v1",
                        },
                    },
                    "execution": {"max_workers": 2, "max_retries": 0},
                    "model_roles": {"translation": {}},
                },
            )
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            zh_output = root / "actual-zh-output"
            args = freezer.zh_tool.build_parser().parse_args(
                [
                    "--config",
                    str(zh_config),
                    "--env-file",
                    str(env_path),
                    "--formal-universe-manifest",
                    str(root / "source_formal_cohort_universe_manifest.json"),
                    "--formal-universe-items",
                    str(root / "source_formal_cohort_universe_items.jsonl"),
                    "--full-base-facts",
                    str(root / "full_base_facts.jsonl"),
                    "--distractor-candidates",
                    str(root / "distractor_candidates.jsonl"),
                    "--neutral-candidates",
                    str(root / "neutral_reference_candidates.jsonl"),
                    "--split-manifest",
                    str(root / "split_manifest.json"),
                    "--fact-resolution-manifest",
                    str(root / "fact_resolution_manifest.json"),
                    "--postreview-rebuild-manifest",
                    str(fixture["postreview"]),
                    "--output-dir",
                    str(zh_output),
                    "--max-workers",
                    "2",
                    "--batch-size",
                    "3",
                ]
            )
            with mock.patch.object(
                freezer.zh_tool, "load_bound_inputs", return_value=loaded
            ):
                zh_manifest = freezer.zh_tool.execute(
                    args, router_factory=AcceptAllZhRouter
                )
            self.assertFalse(zh_manifest["selection_is_full_universe"])
            self.assertTrue(
                zh_manifest["selection_is_complete_retained_cohort"]
            )
            self.assertEqual(
                zh_manifest["excluded_base_fact_ids_sha256"],
                freezer.sha256_value(excluded_ids),
            )
            self.assertEqual(
                zh_manifest["counts"]["universe_records"],
                len(fixture["universe_items"]),
            )

            fixture["zh_manifest"] = zh_output / "run_manifest.json"
            result = self.run_freeze(fixture, root / "freeze-with-actual-zh")
            self.assertEqual(result["status"], freezer.STATUS)
            self.assertEqual(result["cohort_excluded_count"], 1)

    def test_resolution_successor_validates_source_universe_and_exclusion_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            post = freezer.read_json(fixture["postreview"])

            source_path = root / "input_full_base_facts.jsonl"
            revised_path = root / "full_base_facts.jsonl"
            source_rows = freezer.read_jsonl(source_path)
            retained_ids = [row["base_fact_id"] for row in fixture["retained"]]
            excluded_ids = ["pbf_excluded"]
            revised_id = retained_ids[0]
            successor_id = "resolved_fixture_successor"
            lineage_index = {
                revised_id: {
                    "schema_version": "fixture-fact-revision-lineage-v1",
                    "base_fact_id": revised_id,
                    "changed_fields": ["canonical_fact_en"],
                }
            }
            ledger_index = {}
            resolved_rows = copy.deepcopy(fixture["retained"])
            for row in resolved_rows:
                base_fact_id = row["base_fact_id"]
                is_revised = base_fact_id == revised_id
                ledger = {
                    "base_fact_id": base_fact_id,
                    "source_review_outcome": "revise" if is_revised else "accept",
                    "final_disposition": (
                        "retain_revised" if is_revised else "retain_accepted"
                    ),
                }
                ledger_index[base_fact_id] = ledger
                row["fact_review"].update(
                    {
                        "review_outcome": ledger["source_review_outcome"],
                        "resolution_successor_universe_id": successor_id,
                        "final_disposition": ledger["final_disposition"],
                        "fact_resolution_ledger_row_sha256": freezer.sha256_value(
                            ledger
                        ),
                        "fact_revision_lineage": lineage_index.get(base_fact_id),
                        "revision_applied": is_revised,
                    }
                )
            write_jsonl(revised_path, resolved_rows)

            resolution_manifest_path = root / "fact_resolution_manifest.json"
            write_json(
                resolution_manifest_path,
                {
                    "schema_version": (
                        freezer.postreview_tool.semantic_runner.materializer.RESOLUTION_MANIFEST_SCHEMA
                    )
                },
            )
            resolution_bindings = {
                "fact_resolution_manifest": binding(
                    resolution_manifest_path,
                    schema=(
                        freezer.postreview_tool.semantic_runner.materializer.RESOLUTION_MANIFEST_SCHEMA
                    ),
                ),
                "source_full_base_facts": binding(
                    source_path,
                    schema=freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                    rows=source_rows,
                ),
                "revised_full_base_facts": binding(
                    revised_path,
                    schema=freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                    rows=resolved_rows,
                ),
            }
            post["inputs"]["full_base_facts"] = copy.deepcopy(
                resolution_bindings["revised_full_base_facts"]
            )
            post["inputs"]["fact_review"] = copy.deepcopy(resolution_bindings)
            post["inputs"]["fact_resolution"] = copy.deepcopy(
                resolution_bindings
            )
            post["fact_review_contract"]["mode"] = (
                "fact_resolution_successor_cohort"
            )
            post["fact_review_contract"][
                "revised_facts_retained_after_independent_rereview"
            ] = 1
            post["counts"]["revised_base_facts"] = 1
            post["lineage_digests"][
                "ordered_revised_base_fact_ids_sha256"
            ] = freezer.sha256_value([revised_id])
            post["outputs"]["full_base_facts"] = binding(
                revised_path,
                schema=freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                rows=resolved_rows,
                relative=True,
            )

            summary_path = root / "summary.json"
            summary = freezer.read_json(summary_path)
            summary["counts"]["revised_base_facts"] = 1
            write_json(summary_path, summary)
            post["outputs"]["summary"] = binding(
                summary_path,
                schema=freezer.postreview_tool.SUMMARY_SCHEMA_VERSION,
                relative=True,
            )

            exclusions_path = root / "cohort_exclusions.jsonl"
            exclusions = freezer.read_jsonl(exclusions_path)
            exclusions[0]["source_review_outcome"] = "defer"
            write_jsonl(exclusions_path, exclusions)
            post["outputs"]["cohort_exclusions"] = binding(
                exclusions_path,
                schema=freezer.postreview_tool.EXCLUSION_SCHEMA_VERSION,
                rows=exclusions,
                relative=True,
            )

            split_path = root / "split_manifest.json"
            split_manifest = freezer.read_json(split_path)
            split_manifest["fact_review_reject_excluded_count"] = 0
            split_manifest["fact_resolution_cohort_excluded_count"] = 1
            write_json(split_path, split_manifest)
            post["outputs"]["split_manifest"] = binding(
                split_path,
                schema=freezer.postreview_tool.pre_hf.SPLIT_MANIFEST_SCHEMA_VERSION,
                relative=True,
            )
            write_json(fixture["postreview"], post)

            resolution_context = {
                "source_ids": [row["base_fact_id"] for row in source_rows],
                "retained_ids": retained_ids,
                "excluded_ids": excluded_ids,
                "revised_ids": [revised_id],
                "successor_id": successor_id,
                "ledger_index": ledger_index,
                "lineage_index": lineage_index,
            }
            with mock.patch.object(
                freezer.postreview_tool.semantic_runner.materializer,
                "load_fact_resolution_context",
                return_value=resolution_context,
            ) as load_resolution, mock.patch.object(
                freezer, "_deterministically_replay_postreview"
            ) as replay:
                validated = freezer._validate_postreview(
                    fixture["postreview"],
                    config=freezer.read_json(fixture["config"]),
                )

            self.assertEqual(validated["input_ids"], resolution_context["source_ids"])
            self.assertEqual(validated["resolved_input_ids"], retained_ids)
            self.assertEqual(len(validated["input_rows"]), len(fixture["retained"]))
            load_resolution.assert_called_once_with(
                fact_resolution_manifest_path=resolution_manifest_path.resolve(),
                resolved_full_base_facts_path=revised_path.resolve(),
            )
            replay.assert_called_once()

            resolved_rows[0]["fact_review"]["revision_applied"] = False
            write_jsonl(revised_path, resolved_rows)
            revised_binding = binding(
                revised_path,
                schema=freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                rows=resolved_rows,
            )
            post["inputs"]["full_base_facts"] = copy.deepcopy(revised_binding)
            post["inputs"]["fact_review"]["revised_full_base_facts"] = copy.deepcopy(
                revised_binding
            )
            post["inputs"]["fact_resolution"][
                "revised_full_base_facts"
            ] = copy.deepcopy(revised_binding)
            post["outputs"]["full_base_facts"] = binding(
                revised_path,
                schema=freezer.postreview_tool.pre_hf.FULL_FACT_SCHEMA_VERSION,
                rows=resolved_rows,
                relative=True,
            )
            write_json(fixture["postreview"], post)
            with mock.patch.object(
                freezer.postreview_tool.semantic_runner.materializer,
                "load_fact_resolution_context",
                return_value=resolution_context,
            ), mock.patch.object(
                freezer, "_deterministically_replay_postreview"
            ), self.assertRaisesRegex(
                freezer.FreezeValidationError,
                "fact resolution evidence is stale",
            ):
                freezer._validate_postreview(
                    fixture["postreview"],
                    config=freezer.read_json(fixture["config"]),
                )

    def test_stale_postreview_sha_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            with (root / "full_base_facts.jsonl").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(freezer.FreezeValidationError, "SHA-256"):
                self.run_freeze(fixture, root / "freeze")
            self.assertFalse((root / "freeze").exists())

    def test_postreview_requires_bound_semantic_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            postreview = freezer.read_json(fixture["postreview"])
            postreview["inputs"]["semantic_review"].pop("run_manifest")
            write_json(fixture["postreview"], postreview)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError, "semantic run manifest"
            ):
                self.run_freeze(fixture, root / "freeze")

    def test_semantic_runtime_identity_tamper_fails_closed(self):
        for target in ("source", "route", "model"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.make_fixture(root)
                run_manifest = fixture["semantic_run_manifest_value"]
                run_contract = run_manifest["run_contract"]
                if target == "source":
                    run_contract["source_fingerprints"]["runner_sha256"] = "0" * 64
                    expected_error = "source fingerprints"
                elif target == "route":
                    run_contract["route_identity"]["records"][0][
                        "requested_model"
                    ] = "alternate-reviewer"
                    refresh_route_identity_hashes(run_contract["route_identity"])
                    expected_error = "frozen model contract"
                else:
                    run_contract["reviewer_spec"]["model"] = "alternate-reviewer"
                    expected_error = "model identity"
                self.rewrite_semantic_runtime(fixture)
                with self.assertRaisesRegex(
                    freezer.FreezeValidationError, expected_error
                ):
                    self.run_freeze(fixture, root / "freeze")

    def test_candidate_shortfall_fails_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            write_jsonl(
                fixture["shortfalls"],
                [
                    {
                        "schema_version": freezer.postreview_tool.pre_hf.SHORTFALL_SCHEMA_VERSION,
                        "base_fact_id": "pbf_a0",
                    }
                ],
            )
            post = json.loads(fixture["postreview"].read_text(encoding="utf-8"))
            post["outputs"]["candidate_shortfalls"] = binding(
                fixture["shortfalls"],
                schema=freezer.postreview_tool.pre_hf.SHORTFALL_SCHEMA_VERSION,
                rows=freezer.read_jsonl(fixture["shortfalls"]),
                relative=True,
            )
            post["counts"]["candidate_shortfalls"] = 1
            write_json(fixture["postreview"], post)
            with self.assertRaisesRegex(freezer.FreezeValidationError, "shortfalls"):
                self.run_freeze(fixture, root / "freeze")
            self.assertFalse((root / "freeze").exists())

    def test_nonaccepted_candidate_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture["candidate_adjudication_rows"]
            rows[0]["overall_decision"] = "reject"
            rows[0]["distractor_factually_false"] = False
            write_jsonl(fixture["candidate_adjudications"], rows)
            checkpoints = fixture["candidate_checkpoint_rows"]
            checkpoints[0]["strict_adjudication"] = rows[0]
            checkpoints[0]["model_verdict"]["overall_decision"] = "reject"
            checkpoints[0]["model_verdict"]["distractor_factually_false"] = False
            checkpoint_path = root / "candidate_review_checkpoint.jsonl"
            write_jsonl(checkpoint_path, checkpoints)
            manifest = fixture["candidate_manifest_value"]
            manifest["overall_decision_counts"] = {
                "accept": len(rows) - 1,
                "reject": 1,
            }
            manifest["accepted_kind_counts"] = dict(
                manifest["accepted_kind_counts"]
            )
            manifest["accepted_kind_counts"]["distractor"] -= 1
            manifest["artifacts"]["adjudications"] = binding(
                fixture["candidate_adjudications"],
                schema=freezer.candidate_tool.ADJUDICATION_SCHEMA,
                rows=rows,
            )
            manifest["artifacts"]["checkpoint"] = binding(
                checkpoint_path,
                schema=freezer.candidate_tool.CHECKPOINT_SCHEMA,
                rows=checkpoints,
            )
            write_json(fixture["candidate_manifest"], manifest)
            with self.assertRaisesRegex(freezer.FreezeValidationError, "not accepted"):
                self.run_freeze(fixture, root / "freeze")

    def test_accepted_only_projection_freezes_one_plus_one_from_complete_mixed_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)

            rows = fixture["candidate_adjudication_rows"]
            rejected = rows[0]
            rejected["overall_decision"] = "reject"
            rejected["distractor_factually_false"] = False
            write_jsonl(fixture["candidate_adjudications"], rows)
            checkpoints = fixture["candidate_checkpoint_rows"]
            checkpoints[0]["strict_adjudication"] = copy.deepcopy(rejected)
            checkpoints[0]["model_verdict"]["overall_decision"] = "reject"
            checkpoints[0]["model_verdict"]["distractor_factually_false"] = False
            checkpoint_path = root / "candidate_review_checkpoint.jsonl"
            write_jsonl(checkpoint_path, checkpoints)
            manifest = fixture["candidate_manifest_value"]
            manifest["overall_decision_counts"] = {
                "accept": len(rows) - 1,
                "reject": 1,
            }
            manifest["accepted_kind_counts"] = dict(
                manifest["accepted_kind_counts"]
            )
            manifest["accepted_kind_counts"]["distractor"] -= 1
            manifest["artifacts"]["adjudications"] = binding(
                fixture["candidate_adjudications"],
                schema=freezer.candidate_tool.ADJUDICATION_SCHEMA,
                rows=rows,
            )
            manifest["artifacts"]["checkpoint"] = binding(
                checkpoint_path,
                schema=freezer.candidate_tool.CHECKPOINT_SCHEMA,
                rows=checkpoints,
            )
            write_json(fixture["candidate_manifest"], manifest)

            projection_dir = root / "accepted-projection"
            projection_result = freezer.accepted_projection_tool.materialize(
                postreview_manifest_path=fixture["postreview"],
                candidate_review_manifest_path=fixture["candidate_manifest"],
                candidate_adjudications_path=fixture["candidate_adjudications"],
                output_dir=projection_dir,
            )
            projection_path = Path(projection_result["manifest_path"])
            projection = freezer.accepted_projection_tool.load_projection_context(
                projection_path
            )
            self.assertFalse(projection_result["source_review_all_accept"])
            self.assertEqual(len(projection["distractor_candidates"]), 6)
            self.assertEqual(len(projection["neutral_reference_candidates"]), 6)
            self.assertNotIn(
                rejected["candidate_id"], projection["selected_candidate_ids"]
            )

            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            zh_output = root / "projection-zh"
            args = freezer.zh_tool.build_parser().parse_args(
                [
                    "--config",
                    str(fixture["config"]),
                    "--env-file",
                    str(env_path),
                    "--formal-universe-manifest",
                    str(root / "source_formal_cohort_universe_manifest.json"),
                    "--formal-universe-items",
                    str(root / "source_formal_cohort_universe_items.jsonl"),
                    "--accepted-candidate-projection-manifest",
                    str(projection_path),
                    "--output-dir",
                    str(zh_output),
                    "--max-workers",
                    "2",
                    "--batch-size",
                    "3",
                ]
            )
            zh_manifest = freezer.zh_tool.execute(
                args, router_factory=AcceptAllZhRouter
            )
            self.assertEqual(
                zh_manifest["status"],
                "completed_accepted_candidate_projection_zh_proxy_review",
            )
            self.assertTrue(
                zh_manifest["selection_is_complete_accepted_projection"]
            )
            self.assertFalse(
                zh_manifest["completion_claims"][
                    "full_pool_zh_translation_review_complete"
                ]
            )

            fixture["candidate_projection"] = projection_path
            fixture["zh_manifest"] = zh_output / "run_manifest.json"
            result = self.run_freeze(fixture, root / "projection-freeze")
            self.assertTrue(result["pre_exact_hf_gate_passed"])
            self.assertTrue(result["accepted_candidate_projection_used"])
            self.assertEqual(
                result["candidate_cardinality_per_target"],
                {"distractor": 1, "neutral": 1},
            )
            frozen = freezer.read_jsonl(
                root / "projection-freeze" / "exact_reviewed_bundle.jsonl"
            )
            self.assertEqual(len(frozen), 6)
            self.assertTrue(all(len(row["distractors_en"]) == 1 for row in frozen))
            self.assertTrue(
                all(len(row["neutral_candidates_en"]) == 1 for row in frozen)
            )
            self.assertTrue(all(row["human_gold"] is False for row in frozen))
            review_freeze = freezer.read_json(
                root / "projection-freeze" / "review_freeze_manifest.json"
            )
            self.assertFalse(
                review_freeze["candidate_review"][
                    "all_candidates_terminal_and_accepted"
                ]
            )
            self.assertTrue(
                review_freeze["accepted_candidate_projection"][
                    "selected_projection_all_accept"
                ]
            )
            self.assertEqual(
                review_freeze["counts"]["candidate_cardinality_per_target"],
                {"distractor": 1, "neutral": 1},
            )
            gate = freezer.read_json(
                root / "projection-freeze" / "pre_exact_hf_gate_manifest.json"
            )
            self.assertEqual(gate["claim_boundaries"]["hf_model_execution_count"], 0)
            self.assertEqual(
                gate["claim_boundaries"]["validation_behavior_exposure_count"], 0
            )

    def test_zh_target_projection_keeps_excluded_fact_as_support_only_donor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            projection_result = freezer.accepted_projection_tool.materialize(
                postreview_manifest_path=fixture["postreview"],
                candidate_review_manifest_path=fixture["candidate_manifest"],
                candidate_adjudications_path=fixture["candidate_adjudications"],
                output_dir=root / "accepted-projection",
            )
            projection_path = Path(projection_result["manifest_path"])
            parent_projection = freezer.accepted_projection_tool.load_projection_context(
                projection_path
            )
            selected_rows = [
                *parent_projection["distractor_candidates"],
                *parent_projection["neutral_reference_candidates"],
            ]
            support_row = next(
                row
                for row in selected_rows
                if row["source_base_fact_id"] != row["base_fact_id"]
            )
            rejected_target = str(support_row["source_base_fact_id"])
            surviving_dependent = str(support_row["base_fact_id"])

            RejectSelectedZhRouter.reject_ids = frozenset({rejected_target})
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            zh_output = root / "projection-zh-with-quarantine"
            args = freezer.zh_tool.build_parser().parse_args(
                [
                    "--config",
                    str(fixture["config"]),
                    "--env-file",
                    str(env_path),
                    "--formal-universe-manifest",
                    str(root / "source_formal_cohort_universe_manifest.json"),
                    "--formal-universe-items",
                    str(root / "source_formal_cohort_universe_items.jsonl"),
                    "--accepted-candidate-projection-manifest",
                    str(projection_path),
                    "--output-dir",
                    str(zh_output),
                    "--max-workers",
                    "2",
                    "--batch-size",
                    "3",
                ]
            )
            zh_manifest = freezer.zh_tool.execute(
                args, router_factory=RejectSelectedZhRouter
            )
            self.assertEqual(
                zh_manifest["status"], "completed_with_quarantine_or_limited_scope"
            )
            self.assertEqual(zh_manifest["counts"]["quarantined_records"], 1)

            secondary_result = freezer.zh_accepted_projection_tool.materialize(
                accepted_candidate_projection_manifest_path=projection_path,
                zh_review_manifest_path=zh_output / "run_manifest.json",
                output_dir=root / "zh-accepted-projection",
            )
            secondary_path = Path(secondary_result["manifest_path"])
            secondary = freezer.zh_accepted_projection_tool.load_projection_context(
                secondary_path
            )
            self.assertEqual(set(secondary["excluded_ids"]), {rejected_target})
            self.assertEqual(len(secondary["projected_target_ids"]), 5)
            self.assertIn(surviving_dependent, secondary["projected_target_ids"])
            self.assertIn(rejected_target, secondary["support_only_donor_ids"])

            fixture["candidate_projection"] = projection_path
            fixture["zh_manifest"] = zh_output / "run_manifest.json"
            fixture["zh_accepted_projection"] = secondary_path
            result = self.run_freeze(fixture, root / "projection-freeze")
            self.assertTrue(result["pre_exact_hf_gate_passed"])
            self.assertTrue(result["zh_accepted_projection_used"])
            self.assertEqual(result["retained_frozen_count"], 5)
            self.assertEqual(result["zh_target_only_excluded_count"], 1)

            frozen = freezer.read_jsonl(
                root / "projection-freeze" / "exact_reviewed_bundle.jsonl"
            )
            frozen_ids = {str(row["base_fact_id"]) for row in frozen}
            self.assertNotIn(rejected_target, frozen_ids)
            self.assertIn(surviving_dependent, frozen_ids)
            dependent = next(
                row for row in frozen if row["base_fact_id"] == surviving_dependent
            )
            self.assertTrue(
                any(
                    candidate["source_base_fact_id"] == rejected_target
                    for candidate in [
                        *dependent["distractors_en"],
                        *dependent["neutral_candidates_en"],
                    ]
                )
            )

    def test_zh_target_projection_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            projection_result = freezer.accepted_projection_tool.materialize(
                postreview_manifest_path=fixture["postreview"],
                candidate_review_manifest_path=fixture["candidate_manifest"],
                candidate_adjudications_path=fixture["candidate_adjudications"],
                output_dir=root / "accepted-projection",
            )
            projection_path = Path(projection_result["manifest_path"])
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            zh_output = root / "projection-zh"
            args = freezer.zh_tool.build_parser().parse_args(
                [
                    "--config",
                    str(fixture["config"]),
                    "--env-file",
                    str(env_path),
                    "--formal-universe-manifest",
                    str(root / "source_formal_cohort_universe_manifest.json"),
                    "--formal-universe-items",
                    str(root / "source_formal_cohort_universe_items.jsonl"),
                    "--accepted-candidate-projection-manifest",
                    str(projection_path),
                    "--output-dir",
                    str(zh_output),
                ]
            )
            freezer.zh_tool.execute(args, router_factory=AcceptAllZhRouter)
            secondary_result = freezer.zh_accepted_projection_tool.materialize(
                accepted_candidate_projection_manifest_path=projection_path,
                zh_review_manifest_path=zh_output / "run_manifest.json",
                output_dir=root / "zh-accepted-projection",
            )
            secondary_path = Path(secondary_result["manifest_path"])
            secondary_manifest = freezer.read_json(secondary_path)
            secondary_manifest["selection_contract"][
                "support_only_donors_are_not_behavior_or_vector_targets"
            ] = False
            write_json(secondary_path, secondary_manifest)

            fixture["candidate_projection"] = projection_path
            fixture["zh_manifest"] = zh_output / "run_manifest.json"
            fixture["zh_accepted_projection"] = secondary_path
            with self.assertRaisesRegex(
                freezer.FreezeValidationError,
                "zh accepted projection is invalid",
            ):
                self.run_freeze(fixture, root / "projection-freeze")
            self.assertFalse((root / "projection-freeze").exists())

    def test_zh_target_projection_rejects_stage_identity_metadata_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            projection_result = freezer.accepted_projection_tool.materialize(
                postreview_manifest_path=fixture["postreview"],
                candidate_review_manifest_path=fixture["candidate_manifest"],
                candidate_adjudications_path=fixture["candidate_adjudications"],
                output_dir=root / "accepted-projection",
            )
            projection_path = Path(projection_result["manifest_path"])
            env_path = root / ".env"
            env_path.write_text("", encoding="utf-8")
            zh_output = root / "projection-zh"
            args = freezer.zh_tool.build_parser().parse_args(
                [
                    "--config",
                    str(fixture["config"]),
                    "--env-file",
                    str(env_path),
                    "--formal-universe-manifest",
                    str(root / "source_formal_cohort_universe_manifest.json"),
                    "--formal-universe-items",
                    str(root / "source_formal_cohort_universe_items.jsonl"),
                    "--accepted-candidate-projection-manifest",
                    str(projection_path),
                    "--output-dir",
                    str(zh_output),
                ]
            )
            freezer.zh_tool.execute(args, router_factory=AcceptAllZhRouter)

            initial_path = zh_output / "translation_initial.jsonl"
            initial_rows = freezer.read_jsonl(initial_path)
            initial_rows[0]["request_model"] = "tampered-model"
            write_jsonl(initial_path, initial_rows)
            parent_manifest_path = zh_output / "run_manifest.json"
            parent_manifest = freezer.read_json(parent_manifest_path)
            parent_manifest["output_artifacts"]["translation_initial"] = zh_binding(
                initial_path, initial_rows
            )
            write_json(parent_manifest_path, parent_manifest)

            output = root / "zh-accepted-projection"
            with self.assertRaisesRegex(ValueError, "request model mismatch"):
                freezer.zh_accepted_projection_tool.materialize(
                    accepted_candidate_projection_manifest_path=projection_path,
                    zh_review_manifest_path=parent_manifest_path,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_candidate_model_identity_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            manifest = fixture["candidate_manifest_value"]
            manifest["run_contract"]["reviewer_spec"]["model"] = "alternate-reviewer"
            manifest["run_contract"]["expected_response_model"] = "alternate-response"
            manifest["run_contract"]["protocol_reviewer_identity"] = {
                "provider_profile": "openai",
                "requested_model": "alternate-reviewer",
                "expected_response_model": "alternate-response",
            }
            manifest["run_contract_sha256"] = freezer.sha256_value(
                manifest["run_contract"]
            )
            write_json(fixture["candidate_manifest"], manifest)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError, "protocol reviewer identity"
            ):
                self.run_freeze(fixture, root / "freeze-alternate-identity")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            rows = fixture["candidate_adjudication_rows"]
            rows[0]["response_model_identity_status"] = "not_checked"
            write_jsonl(fixture["candidate_adjudications"], rows)
            checkpoints = fixture["candidate_checkpoint_rows"]
            checkpoints[0]["strict_adjudication"] = rows[0]
            checkpoint_path = root / "candidate_review_checkpoint.jsonl"
            write_jsonl(checkpoint_path, checkpoints)
            manifest = fixture["candidate_manifest_value"]
            manifest["artifacts"]["adjudications"] = binding(
                fixture["candidate_adjudications"],
                schema=freezer.candidate_tool.ADJUDICATION_SCHEMA,
                rows=rows,
            )
            manifest["artifacts"]["checkpoint"] = binding(
                checkpoint_path,
                schema=freezer.candidate_tool.CHECKPOINT_SCHEMA,
                rows=checkpoints,
            )
            write_json(fixture["candidate_manifest"], manifest)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError, "model identity is invalid"
            ):
                self.run_freeze(fixture, root / "freeze-unverified-identity")

    def test_candidate_identity_fields_are_required(self):
        cases = (
            ("manifest_tool_version", "manifest tool_version"),
            ("run_contract_tool_version", "run_contract tool_version"),
            ("source_fingerprints", "source fingerprints"),
            ("route_identity", "route identity"),
        )
        for field, expected_error in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.make_fixture(root)
                manifest = fixture["candidate_manifest_value"]
                if field == "manifest_tool_version":
                    manifest.pop("tool_version")
                    write_json(fixture["candidate_manifest"], manifest)
                else:
                    manifest["run_contract"].pop(field.replace("run_contract_", ""))
                    self.rewrite_candidate_contract(fixture)
                with self.assertRaisesRegex(
                    freezer.FreezeValidationError, expected_error
                ):
                    self.run_freeze(fixture, root / "freeze")

    def test_candidate_source_fingerprint_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            manifest = fixture["candidate_manifest_value"]
            manifest["run_contract"]["source_fingerprints"]["runner_sha256"] = "0" * 64
            self.rewrite_candidate_contract(fixture)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError, "source fingerprints"
            ):
                self.run_freeze(fixture, root / "freeze")

    def test_candidate_route_hash_tamper_fails_closed(self):
        for target in ("record", "set"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.make_fixture(root)
                manifest = fixture["candidate_manifest_value"]
                route_identity = manifest["run_contract"]["route_identity"]
                if target == "record":
                    route_identity["records"][0]["route_identity_sha256"] = "0" * 64
                    route_identity["route_identity_set_sha256"] = freezer.sha256_value(
                        {
                            key: value
                            for key, value in route_identity.items()
                            if key != "route_identity_set_sha256"
                        }
                    )
                    expected_error = "record SHA is stale"
                else:
                    route_identity["route_identity_set_sha256"] = "0" * 64
                    expected_error = "set SHA is stale"
                self.rewrite_candidate_contract(fixture)
                with self.assertRaisesRegex(
                    freezer.FreezeValidationError, expected_error
                ):
                    self.run_freeze(fixture, root / "freeze")

    def test_candidate_duplicate_route_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            manifest = fixture["candidate_manifest_value"]
            route_identity = manifest["run_contract"]["route_identity"]
            route_identity["records"].append(copy.deepcopy(route_identity["records"][0]))
            route_identity["record_count"] = len(route_identity["records"])
            refresh_route_identity_hashes(route_identity)
            self.rewrite_candidate_contract(fixture)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError, "route identity"
            ):
                self.run_freeze(fixture, root / "freeze")

    def test_candidate_route_identity_and_protocol_mismatch_fail_closed(self):
        for target in ("requested_model", "protocol"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.make_fixture(root)
                manifest = fixture["candidate_manifest_value"]
                record = manifest["run_contract"]["route_identity"]["records"][0]
                if target == "requested_model":
                    record["requested_model"] = "alternate-reviewer"
                    expected_error = "frozen model contract"
                else:
                    record["protocol"] = "anthropic"
                    expected_error = "protocol differs from config"
                refresh_route_identity_hashes(
                    manifest["run_contract"]["route_identity"]
                )
                self.rewrite_candidate_contract(fixture)
                with self.assertRaisesRegex(
                    freezer.FreezeValidationError, expected_error
                ):
                    self.run_freeze(fixture, root / "freeze")

    def test_zh_review_independence_tamper_fails_closed(self):
        cases = (
            ("review_independence", "review independence"),
            ("evidence_boundary", "evidence boundary"),
        )
        for section, expected_error in cases:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.make_fixture(root)
                manifest = fixture["zh_manifest_value"]
                manifest[section][
                    "provider_infrastructure_independence_claimed"
                ] = True
                self.rewrite_zh_contract(fixture)
                with self.assertRaisesRegex(
                    freezer.FreezeValidationError, expected_error
                ):
                    self.run_freeze(fixture, root / "freeze")

    def test_zh_source_and_route_tamper_fail_closed(self):
        for target in ("source", "route"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.make_fixture(root)
                manifest = fixture["zh_manifest_value"]
                if target == "source":
                    manifest["source_fingerprints"]["runner_sha256"] = "0" * 64
                    expected_error = "source fingerprints"
                else:
                    manifest["route_identity"]["records"][0][
                        "requested_model"
                    ] = "alternate-generator"
                    refresh_route_identity_hashes(manifest["route_identity"])
                    expected_error = "frozen model contract"
                self.rewrite_zh_contract(fixture)
                with self.assertRaisesRegex(
                    freezer.FreezeValidationError, expected_error
                ):
                    self.run_freeze(fixture, root / "freeze")

    def test_zh_model_and_route_cannot_jointly_drift_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            manifest = fixture["zh_manifest_value"]
            manifest["models"]["generator"]["model"] = "alternate-generator"
            config = freezer.read_json(fixture["config"])
            manifest["route_identity"] = freezer.zh_tool.build_route_identity_set(
                config=config,
                env_path=root / ".env",
                routes=[
                    (
                        manifest["models"]["generator"],
                        manifest["models"]["generator"]["expected_response_model"],
                    ),
                    (
                        manifest["models"]["reviewer"],
                        manifest["models"]["reviewer"]["expected_response_model"],
                    ),
                ],
                resolved_env={},
            )
            self.rewrite_zh_contract(fixture)
            with self.assertRaisesRegex(
                freezer.FreezeValidationError, "model identities differ"
            ):
                self.run_freeze(fixture, root / "freeze")

    def test_zh_shared_gateway_is_valid_when_endpoint_distinct_is_false(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            config = json.loads(fixture["config"].read_text(encoding="utf-8"))
            shared_url = "https://shared-gateway-fixture.invalid/v1"
            config["provider_profiles"]["aliyun"]["base_url"] = shared_url
            config["provider_profiles"]["openai"]["base_url"] = shared_url
            write_json(fixture["config"], config)

            semantic_manifest = fixture["semantic_run_manifest_value"]
            semantic_contract = semantic_manifest["run_contract"]
            semantic_contract["config_sha256"] = freezer.sha256_value(config)
            semantic_contract[
                "route_identity"
            ] = freezer.postreview_tool.semantic_runner.build_route_identity_set(
                config=config,
                env_path=root / ".env",
                routes=[
                    (
                        semantic_contract["reviewer_spec"],
                        semantic_contract["expected_response_model"],
                    )
                ],
                resolved_env={},
            )
            self.rewrite_semantic_runtime(fixture)

            candidate_manifest = fixture["candidate_manifest_value"]
            candidate_contract = candidate_manifest["run_contract"]
            candidate_contract["config_sha256"] = freezer.sha256_value(config)
            candidate_contract["candidate_manifest"]["sha256"] = freezer.sha256_file(
                fixture["postreview"]
            )
            candidate_contract[
                "route_identity"
            ] = freezer.candidate_tool.build_route_identity_set(
                config=config,
                env_path=root / ".env",
                routes=[
                    (
                        candidate_contract["reviewer_spec"],
                        candidate_contract["expected_response_model"],
                    )
                ],
                resolved_env={},
            )
            self.rewrite_candidate_contract(fixture)

            zh_manifest = fixture["zh_manifest_value"]
            zh_manifest["route_identity"] = freezer.zh_tool.build_route_identity_set(
                config=config,
                env_path=root / ".env",
                routes=[
                    (
                        zh_manifest["models"]["generator"],
                        zh_manifest["models"]["generator"][
                            "expected_response_model"
                        ],
                    ),
                    (
                        zh_manifest["models"]["reviewer"],
                        zh_manifest["models"]["reviewer"][
                            "expected_response_model"
                        ],
                    ),
                ],
                resolved_env={},
            )
            zh_manifest["review_independence"]["provider_endpoint_distinct"] = False
            self.rewrite_zh_contract(fixture)

            result = self.run_freeze(fixture, root / "freeze")
            self.assertEqual(result["status"], freezer.STATUS)

    def test_incomplete_zh_review_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            manifest = fixture["zh_manifest_value"]
            manifest["status"] = "completed_with_quarantine_or_limited_scope"
            write_json(fixture["zh_manifest"], manifest)
            with self.assertRaisesRegex(freezer.FreezeValidationError, "zh review is incomplete"):
                self.run_freeze(fixture, root / "freeze")

    def test_unconfirmed_historical_exposure_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            fixture["exposure_validation"]["scope_owner_attestation_valid"] = False
            with self.assertRaisesRegex(freezer.FreezeValidationError, "not scope-owner confirmed"):
                self.run_freeze(fixture, root / "freeze")

    def test_known_validation_or_sealed_exposure_fails_closed(self):
        universe = {"known_exposure_ids": ["pbf_sensitive"]}
        postreview = {
            "row_index": {
                "pbf_sensitive": {"split_assignment": "validation"},
            },
            "exclusion_index": {},
        }
        with self.assertRaisesRegex(
            freezer.FreezeValidationError, "Validation/Sealed"
        ):
            freezer._validate_history_against_split(
                universe=universe, postreview=postreview
            )


if __name__ == "__main__":
    unittest.main()
