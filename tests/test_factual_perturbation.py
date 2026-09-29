import ast
import hashlib
import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "factual_pitfalls" / "perturbation.py"
MODULE_SPEC = importlib.util.spec_from_file_location("factual_perturbation_runtime", MODULE_PATH)
perturbation = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC and MODULE_SPEC.loader
sys.modules[MODULE_SPEC.name] = perturbation
MODULE_SPEC.loader.exec_module(perturbation)

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_factual_perturbation.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("factual_perturbation_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC and RUNNER_SPEC.loader
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)


def write_formal_freeze_manifests(
    bundle_path,
    records,
    *,
    review_overrides=None,
    split_overrides=None,
):
    identity = perturbation.formal_admission.bundle_identity(
        bundle_path,
        perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
        records,
    )
    binding = {
        field: identity[field]
        for field in (
            "sha256",
            "schema_version",
            "record_count",
            "base_fact_ids_sha256",
        )
    }
    split_policy_versions = {
        row["split_policy_version"]
        for row in records
        if isinstance(row.get("split_policy_version"), str)
        and row["split_policy_version"].strip()
    }
    split_policy_version = next(iter(split_policy_versions), "answer-group-v1")
    row_by_id = {
        perturbation.formal_admission.explicit_base_fact_id(row, "fixture row"): row
        for row in records
    }
    frozen_ids = sorted(
        base_fact_id
        for base_fact_id, row in row_by_id.items()
        if row.get("canonical_status") == "frozen"
    )
    rejected_ids = sorted(
        base_fact_id
        for base_fact_id, row in row_by_id.items()
        if row.get("canonical_status") == "rejected"
    )
    terminal_ids = sorted([*frozen_ids, *rejected_ids])

    def artifact_binding(path, rows, schema_version, id_field, digest_field):
        perturbation.write_jsonl(path, rows)
        identifiers = sorted(str(row[id_field]).strip() for row in rows)
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema_version": schema_version,
            "record_count": len(rows),
            digest_field: perturbation.formal_admission.sha256_value(identifiers),
        }

    scope_rows = [
        {
            "schema_version": perturbation.formal_admission.REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
            "base_fact_id": base_fact_id,
            "input_record_sha256": perturbation.formal_admission.sha256_value(
                row_by_id[base_fact_id]
            ),
        }
        for base_fact_id in terminal_ids
    ]
    decision_rows = [
        {
            "schema_version": perturbation.formal_admission.REVIEW_DECISION_SCHEMA_VERSION,
            "base_fact_id": base_fact_id,
            "input_record_sha256": perturbation.formal_admission.sha256_value(
                row_by_id[base_fact_id]
            ),
            "decision": "accept" if base_fact_id in frozen_ids else "reject",
            "reviewer_type": "codex_proxy",
            "human_gold": False,
            "evidence_tier": row_by_id[base_fact_id]["evidence_tier"],
            "member_review_complete": True,
            "alias_review_complete": True,
            "distractor_review_complete": True,
        }
        for base_fact_id in terminal_ids
    ]
    scope_binding = artifact_binding(
        bundle_path.with_name("formal_review_scope.jsonl"),
        scope_rows,
        perturbation.formal_admission.REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
        "base_fact_id",
        "base_fact_ids_sha256",
    )
    decision_binding = artifact_binding(
        bundle_path.with_name("formal_review_decisions.jsonl"),
        decision_rows,
        perturbation.formal_admission.REVIEW_DECISION_SCHEMA_VERSION,
        "base_fact_id",
        "base_fact_ids_sha256",
    )
    review = {
        "schema_version": (
            perturbation.formal_admission.REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION
        ),
        "input_bundle": binding,
        "source_universe": {
            "record_count": len(row_by_id),
            "base_fact_ids_sha256": perturbation.formal_admission.sha256_value(
                sorted(row_by_id)
            ),
        },
        "review_scope": scope_binding,
        "review_decisions": decision_binding,
        "decision_counts": {
            "accept": len(frozen_ids),
            "reject": len(rejected_ids),
            "defer": 0,
            "revise": 0,
            "missing": 0,
        },
        "review_complete": True,
        "canonical_freeze_authorized": True,
    }
    review.update(review_overrides or {})
    review_path = bundle_path.with_name("review_freeze_manifest.json")
    perturbation.write_json(review_path, review)

    risk_pairs = perturbation.formal_admission._mechanical_cross_group_risk_pairs(records)
    candidate_rows = []
    adjudication_rows = []
    for left, right in sorted(risk_pairs):
        candidate_id = f"near_{perturbation.formal_admission.sha256_value([left, right])[:16]}"
        candidate_row = {
            "schema_version": (
                perturbation.formal_admission.NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION
            ),
            "candidate_id": candidate_id,
            "left_base_fact_id": left,
            "right_base_fact_id": right,
            "left_candidate_id": row_by_id[left]["candidate_id"],
            "right_candidate_id": row_by_id[right]["candidate_id"],
            "left_input_record_sha256": perturbation.formal_admission.sha256_value(
                row_by_id[left]
            ),
            "right_input_record_sha256": perturbation.formal_admission.sha256_value(
                row_by_id[right]
            ),
        }
        candidate_rows.append(candidate_row)
        adjudication_rows.append({
            "schema_version": (
                perturbation.formal_admission.NEAR_DUPLICATE_DECISION_SCHEMA_VERSION
            ),
            "candidate_id": candidate_id,
            "candidate_record_sha256": (
                perturbation.formal_admission.sha256_value(candidate_row)
            ),
            "left_candidate_id": candidate_row["left_candidate_id"],
            "right_candidate_id": candidate_row["right_candidate_id"],
            "left_input_record_sha256": candidate_row["left_input_record_sha256"],
            "right_input_record_sha256": candidate_row["right_input_record_sha256"],
            "decision": "distinct",
            "reviewer_type": "codex_proxy",
            "human_gold": False,
        })
    candidate_binding = artifact_binding(
        bundle_path.with_name("near_duplicate_candidates.jsonl"),
        candidate_rows,
        perturbation.formal_admission.NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION,
        "candidate_id",
        "candidate_ids_sha256",
    )
    adjudication_binding = artifact_binding(
        bundle_path.with_name("near_duplicate_adjudications.jsonl"),
        adjudication_rows,
        perturbation.formal_admission.NEAR_DUPLICATE_DECISION_SCHEMA_VERSION,
        "candidate_id",
        "candidate_ids_sha256",
    )
    registry_binding = artifact_binding(
        bundle_path.with_name("historical_exposure_registry.jsonl"),
        [],
        perturbation.formal_admission.HISTORICAL_EXPOSURE_REGISTRY_SCHEMA_VERSION,
        "exposure_id",
        "exposure_ids_sha256",
    )
    exposure_rows = [
        {
            "schema_version": (
                perturbation.formal_admission.HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION
            ),
            "base_fact_id": base_fact_id,
            "input_record_sha256": perturbation.formal_admission.sha256_value(
                row_by_id[base_fact_id]
            ),
            "split_assignment": row_by_id[base_fact_id]["split_assignment"],
            "historically_exposed": False,
        }
        for base_fact_id in frozen_ids
        if row_by_id[base_fact_id].get("split_assignment") in {"validation", "sealed"}
    ]
    exposure_binding = artifact_binding(
        bundle_path.with_name("historical_exposure_checks.jsonl"),
        exposure_rows,
        perturbation.formal_admission.HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION,
        "base_fact_id",
        "base_fact_ids_sha256",
    )
    assignments, _ = perturbation.formal_admission._split_assignments(records)
    split = {
        "schema_version": (
            perturbation.formal_admission.SPLIT_FREEZE_MANIFEST_SCHEMA_VERSION
        ),
        "input_bundle": binding,
        "review_freeze_manifest": {
            "path": str(review_path.resolve()),
            "sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "schema_version": (
                perturbation.formal_admission.REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION
            ),
        },
        "split_policy_version": split_policy_version,
        "split_assignments_sha256": perturbation.formal_admission.sha256_value(
            assignments
        ),
        "split_status": "frozen",
        "near_duplicate_policy_version": "synthetic-near-duplicate-policy-v1",
        "near_duplicate_review_complete": True,
        "unresolved_near_duplicate_count": 0,
        "near_duplicate_candidates": candidate_binding,
        "near_duplicate_adjudications": adjudication_binding,
        "historical_exposure_policy_version": "synthetic-exposure-policy-v1",
        "historical_exposure_check_complete": True,
        "historical_exposure_registry": registry_binding,
        "historical_exposure_checks": exposure_binding,
        "unresolved_cross_split_count": 0,
        "validation_exposure_count": 0,
        "sealed_exposure_count": 0,
    }
    split.update(split_overrides or {})
    split_path = bundle_path.with_name("split_freeze_manifest.json")
    perturbation.write_json(split_path, split)
    return review_path, split_path


def explicit_frozen_row():
    return {
        "source_id": "frozen-contract",
        "candidate_id": "frozen-contract",
        "base_fact_id": "public:frozen-contract",
        "source_dataset": "global_mmlu",
        "bundle_status": "reviewed_frozen",
        "admission": {"semantic_review_complete": True, "prompt_ready": True},
        "evidence_tier": "independent_adjudicated",
        "human_gold": False,
        "canonical_status": "frozen",
        "split_group_id": "answer-group::paris",
        "split_assignment": "development",
        "split_status": "frozen",
        "split_policy_version": "answer-group-v1",
        "prompt_ready": True,
        "prompt_quality_tier": "strict_factual_completion",
        "prompt_en": "The capital of France is",
        "answer_en": "Paris",
        "canonical_fact_en": "The capital of France is Paris.",
        "distractors": [],
    }


def write_static_proxy_freeze_fixture(root):
    rows = []
    split_assignments = ["development"] * 96 + ["validation"] * 32 + ["sealed"] * 32
    for index, split_assignment in enumerate(split_assignments):
        base_fact_id = f"static-fact-{index:03d}"
        source_id = f"static-source-{index:03d}"
        answer_en = f"Gold {index}"
        answer_zh = f"正确答案{index}"
        variants = [
            {
                "distractor_id": f"{base_fact_id}-d1",
                "selected_candidate_id": f"{base_fact_id}-d1-p1",
                "text_en": f"Wrong {index}A",
                "text_zh": f"错误答案{index}甲",
                "targeted_context_en": f"The source explicitly but falsely says Wrong {index}A.",
                "targeted_context_zh": f"资料明确但错误地称答案是错误答案{index}甲。",
                "neutral_context_en": f"An unrelated record contains neutral detail {index}A.",
                "neutral_context_zh": f"一条无关记录包含中性细节{index}甲。",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "verified": True,
                "verification_status": "codex_adjudicated",
            },
            {
                "distractor_id": f"{base_fact_id}-d2",
                "selected_candidate_id": f"{base_fact_id}-d2-p1",
                "text_en": f"Wrong {index}B",
                "text_zh": f"错误答案{index}乙",
                "targeted_context_en": f"The source explicitly but falsely says Wrong {index}B.",
                "targeted_context_zh": f"资料明确但错误地称答案是错误答案{index}乙。",
                "neutral_context_en": f"An unrelated record contains neutral detail {index}B.",
                "neutral_context_zh": f"一条无关记录包含中性细节{index}乙。",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
                "verified": True,
                "verification_status": "codex_adjudicated",
            },
        ]
        options = [
            {
                "option_id": variants[0]["distractor_id"],
                "kind": "distractor",
                "text_en": variants[0]["text_en"],
                "text_zh": variants[0]["text_zh"],
                "choice": "A",
            },
            {
                "option_id": "gold",
                "kind": "gold",
                "text_en": answer_en,
                "text_zh": answer_zh,
                "choice": "B",
            },
            {
                "option_id": variants[1]["distractor_id"],
                "kind": "distractor",
                "text_en": variants[1]["text_en"],
                "text_zh": variants[1]["text_zh"],
                "choice": "C",
            },
        ]
        prompt_en = f"Static question {index} is"
        prompt_zh = f"静态问题{index}是"
        stimuli = []
        for language, prompt in (("en", prompt_en), ("zh", prompt_zh)):
            stimuli.append({
                "stimulus_id": f"{base_fact_id}:{language}:original",
                "language": language,
                "arm": "original",
                "distractor_id": None,
                "prompt": prompt,
                "context": "",
                "options": options,
            })
        for variant in variants:
            for language, prompt in (("en", prompt_en), ("zh", prompt_zh)):
                for arm in ("neutral", "targeted"):
                    stimuli.append({
                        "stimulus_id": (
                            f"{base_fact_id}:{variant['distractor_id']}:{language}:{arm}"
                        ),
                        "language": language,
                        "arm": arm,
                        "distractor_id": variant["distractor_id"],
                        "prompt": prompt,
                        "context": variant[f"{arm}_context_{language}"],
                        "options": options,
                    })
        rows.append({
            "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
            "source_id": source_id,
            "candidate_id": f"candidate-{index:03d}",
            "base_fact_id": base_fact_id,
            "base_id": base_fact_id,
            "source_dataset": "synthetic",
            "bundle_status": "reviewed_frozen",
            "admission": {"semantic_review_complete": True, "prompt_ready": True},
            "evidence_tier": "independent_adjudicated",
            "human_gold": False,
            "canonical_status": "frozen",
            "split_group_id": f"static-group-{index:03d}",
            "split_assignment": split_assignment,
            "split_status": "frozen",
            "split_policy_version": "relation-stratified-component-greedy-sha256-v2",
            "prompt_ready": True,
            "prompt_quality_tier": "strict_factual_completion",
            "prompt_en": prompt_en,
            "prompt_zh": prompt_zh,
            "answer_en": answer_en,
            "answer_zh": answer_zh,
            "answer_aliases_en": [answer_en],
            "answer_aliases_zh": [answer_zh],
            "canonical_fact": f"{prompt_en} {answer_en}.",
            "canonical_fact_en": f"{prompt_en} {answer_en}.",
            "canonical_fact_zh": f"{prompt_zh}{answer_zh}。",
            "distractor_candidates": [
                {
                    "distractor_id": variant["distractor_id"],
                    "text_en": variant["text_en"],
                    "text_zh": variant["text_zh"],
                    "verified": True,
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                    "verification_status": "codex_adjudicated",
                }
                for variant in variants
            ],
            "static_mcq": {
                "schema_version": perturbation.STATIC_G0A_MCQ_SCHEMA_VERSION,
                "manipulation_family": (
                    perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                ),
                "option_order_policy": "base-fact-option-sha256-v1",
                "options": options,
                "variants": variants,
                "behavior_inputs": stimuli,
            },
            "static_stimulus_status": "frozen",
            "static_generation_complete": True,
            "behavior_authorized": False,
            "simulation_authorized": False,
            "codex_adjudication": {
                "decision": "accept",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
            },
        })
    bundle_path = root / "frozen_static_bundle.jsonl"
    perturbation.write_jsonl(bundle_path, rows)
    review_path, split_path = write_formal_freeze_manifests(bundle_path, rows)
    identity = perturbation.formal_admission.bundle_identity(
        bundle_path, perturbation.INPUT_BUNDLE_SCHEMA_VERSION, rows
    )

    def static_artifact(path):
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }

    static_manifest = {
        "schema_version": perturbation.STATIC_G0A_FREEZE_MANIFEST_SCHEMA_VERSION,
        "status": "g0a_static_stimulus_frozen",
        "artifacts": {
            bundle_path.name: static_artifact(bundle_path),
            review_path.name: static_artifact(review_path),
            split_path.name: static_artifact(split_path),
        },
        "frozen_bundle": {
            **{
                field: identity[field]
                for field in (
                    "sha256",
                    "schema_version",
                    "record_count",
                    "base_fact_ids_sha256",
                )
            },
            "path": bundle_path.name,
        },
        "base_fact_count": 160,
        "variant_count": 320,
        "unique_behavior_input_count": 1600,
        "reviewer_type": "codex_proxy",
        "human_gold": False,
        "static_frozen": True,
        "behavior_authorized": False,
        "simulation_authorized": False,
    }
    static_manifest_path = root / "static_freeze_manifest.json"
    perturbation.write_json(static_manifest_path, static_manifest)
    return rows, static_manifest_path


def static_proxy_route_runtime_identity_fixture(config):
    records = []
    for spec in config["model_roles"]["simulation"]["models"]:
        route = {
            "model": spec["model"],
            "provider_profile": spec["provider_profile"],
            "protocol": config["provider_profiles"][spec["provider_profile"]][
                "protocol"
            ],
            "base_url_sha256": hashlib.sha256(
                f"base:{spec['provider_profile']}".encode("utf-8")
            ).hexdigest(),
            "request_route_sha256": hashlib.sha256(
                f"route:{spec['provider_profile']}".encode("utf-8")
            ).hexdigest(),
            "credentials_or_endpoints_included": False,
        }
        route["route_identity_sha256"] = perturbation.sha256_value(route)
        if spec["provider_profile"] == "ollama_local":
            digest = f"sha256:{hashlib.sha256(spec['model'].encode('utf-8')).hexdigest()}"
            version = "fixture-ollama-1.0"
            runtime = {
                "kind": "ollama",
                "attestation_status": "observed",
                "server_version": version,
                "server_version_sha256": hashlib.sha256(
                    version.encode("utf-8")
                ).hexdigest(),
                "installed_model_digest": digest,
                "installed_model_digest_sha256": hashlib.sha256(
                    digest.encode("utf-8")
                ).hexdigest(),
                "template": {
                    "status": "unknown",
                    "sha256": None,
                    "source": "ollama_api_show",
                    "limitation": "fixture_template_not_observed",
                },
            }
        else:
            runtime = {
                "kind": "provider_managed",
                "attestation_status": "not_available",
                "server_version": None,
                "limitation": "provider_does_not_expose_stable_runtime_attestation",
            }
        record = {**route, "runtime": runtime}
        record["identity_sha256"] = perturbation.sha256_value(record)
        records.append(record)
    return {
        "schema_version": (
            perturbation.STATIC_PROXY_ROUTE_RUNTIME_IDENTITY_SCHEMA_VERSION
        ),
        "record_count": len(records),
        "records": records,
        "identity_sha256_by_model": {
            record["model"]: record["identity_sha256"] for record in records
        },
        "route_runtime_identity_set_sha256": perturbation.sha256_value(records),
        "credentials_or_endpoints_included": False,
    }


class PerturbationRuntimeTests(unittest.TestCase):
    @staticmethod
    def _manifest_config(inputs=None):
        return {
            "config_version": "test-input-bundle-v1",
            "inputs": inputs or {},
            "pilot": {
                "seed": 7,
                "stratify_by": ["source_dataset", "prompt_quality_tier", "answer_type"],
                "max_distractors_per_triple": 2,
            },
            "model_roles": {
                "simulation": {"models": []},
                "translation": {},
                "perturbation_validation": {},
            },
            "preflight": {},
        }

    @staticmethod
    def _write_candidate_freeze_fixture(run_dir, rows=()):
        rows = [dict(row) for row in rows]
        frozen_path = run_dir / "frozen_candidates.jsonl"
        perturbation.write_jsonl(frozen_path, rows)
        candidate_ids_sha256 = perturbation.sha256_value(
            [row["candidate_id"] for row in rows]
        )
        manifest = {
            "schema_version": perturbation.CANDIDATE_FREEZE_MANIFEST_SCHEMA_VERSION,
            "candidate_count": len(rows),
            "candidate_ids_sha256": candidate_ids_sha256,
            "frozen_candidates": {
                "path": frozen_path.name,
                "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                "record_count": len(rows),
                "schema_version": perturbation.CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
                "candidate_ids_sha256": candidate_ids_sha256,
            },
            "holdout_eligible": True,
            "paths_not_taken_enabled": False,
        }
        perturbation.write_json(run_dir / "candidate_freeze_manifest.json", manifest)
        return manifest

    @staticmethod
    def _candidate_freeze_row():
        return {
            "schema_version": perturbation.CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
            "freeze_version": perturbation.CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
            "candidate_id": "candidate-1",
            "source_id": "source-1",
            "distractor_id": "distractor-1",
            "candidate": {
                "english_context": "The misleading English context.",
                "chinese_context": "误导性的中文上下文。",
            },
            "badcase_type": "induced",
            "accuracy": {
                "en_original": 1.0,
                "en_perturbed": 1.0,
                "zh_original": 1.0,
                "zh_perturbed": 0.0,
            },
        }

    @staticmethod
    def _write_explicit_proxy_fixture(project_root):
        source_run = project_root / "source"
        source_run.mkdir()
        source_records = [
            {
                "source_id": "s-dev",
                "base_fact_id": "fact-dev",
                "split_assignment": "development",
            },
            {
                "source_id": "s-val",
                "base_fact_id": "fact-val",
                "split_assignment": "validation",
            },
        ]
        perturbation.write_json(source_run / "run_manifest.json", {
            "run_id": "source",
            "seed": 7,
            "config_sha256": "old",
            "selected_count": 2,
            "records": source_records,
            "manipulation_family": (
                perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            ),
            "designated_distractors_per_base_fact": 2,
            "simulation_endpoint": "multi_option",
            "simulation_max_options": 3,
            "neutral_context_mode": "matched_per_candidate",
            "shared_simulation_variants": ["original"],
        })
        source_perturbations = []
        accepted_distractors = []
        review_rows = []
        checks = {
            key: True for key in perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS
        }
        for source_id, prefix in (("s-dev", "dev"), ("s-val", "val")):
            for index in (1, 2):
                distractor_id = f"{prefix}-d{index}"
                candidate_id = f"{prefix}-c{index}"
                source_perturbations.append({
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "distractor_id": distractor_id,
                    "terminal_status": "completed",
                })
                accepted_distractors.append({
                    "item_id": distractor_id,
                    "source_id": source_id,
                    "terminal_status": "completed",
                    "parsed_response": {
                        "decision": "accept",
                        "checks": {"valid": True},
                    },
                })
                review_rows.append({
                    "item_id": candidate_id,
                    "decision": "accept",
                    "checks": checks,
                })
        perturbation.write_jsonl(
            source_run / "perturbations.jsonl", source_perturbations
        )
        perturbation.write_jsonl(
            source_run / "verified_distractors.jsonl", accepted_distractors
        )
        for name in (
            "translations.jsonl",
            "translation_reviews.jsonl",
            "perturbation_generations.jsonl",
        ):
            perturbation.write_jsonl(
                source_run / name,
                [{"source_id": row["source_id"]} for row in source_records],
            )
        review_path = source_run / "codex_proxy_review_v1.json"
        perturbation.write_json(review_path, {
            "run_id": "source",
            "review_version": "v1",
            "reviewer_type": "codex_proxy",
            "not_human_gold": True,
            "adjudicator_type": "codex",
            "review_status": "codex_adjudicated",
            "human_gold": False,
            "review_blinded_to_simulation_results": True,
            "review_contract_version": (
                perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            ),
            "manipulation_family": (
                perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
            ),
            "perturbation_reviews": review_rows,
        })
        config = {
            "config_version": "explicit-proxy-v1",
            "inputs": {"allowed_splits": ["development"]},
            "model_roles": {
                "perturbation_generation": {
                    "manipulation_family": (
                        perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                    ),
                    "target_distractors_per_source": 2,
                },
                "simulation": {
                    "endpoint": "multi_option",
                    "max_options": 3,
                    "neutral_context_mode": "matched_per_candidate",
                    "models": [{"model": "m"}],
                },
            },
        }
        return source_run, review_path, config

    def test_explicit_input_bundle_routes_only_frozen_rows_and_preserves_ids_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            run_dir = project_root / "runs" / "bundle-test"
            bundle_path = project_root / "inputs" / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "bundle_id": "reviewed-public-v1",
                "records": [
                    {
                        "source_id": "frozen-1",
                        "base_fact_id": "public:frozen-1",
                        "source_pool_id": "public-benchmarks-full-v1",
                        "source_dataset": "global_mmlu",
                        "source_format": "multiple_choice",
                        "bundle_status": "reviewed_frozen",
                        "admission": {
                            "semantic_review_complete": True,
                            "prompt_ready": True,
                        },
                        "evidence_tier": "independent_adjudicated",
                        "human_gold": False,
                        "canonical_status": "frozen",
                        "split_group_id": "answer-group::paris",
                        "split_assignment": "development",
                        "split_status": "frozen",
                        "split_policy_version": "answer-group-v1",
                        "sealed_evaluation_status": "not_run",
                        "prompt_ready": True,
                        "prompt_quality_tier": "strict_factual_completion",
                        "prompt_en": "The capital of France is",
                        "answer_en": "Paris",
                        "answer_aliases_en": ["Paris", "City of Paris"],
                        "answer_type": "city",
                        "canonical_fact_en": "The capital of France is Paris.",
                        "relation_raw": "capital city",
                        "relation_normalized": "entity_location",
                        "probe_relation_id": "country_capital",
                        "distractor_candidates": [{
                            "distractor_id": "frozen-1::berlin",
                            "answer_en": "Berlin",
                            "source": "reviewed_source_wrong_option",
                        }],
                    },
                    {
                        "source_id": "rejected-1",
                        "base_fact_id": "public:rejected-1",
                        "source_dataset": "mkqa",
                        "canonical_status": "rejected",
                        "evidence_tier": "independent_adjudicated",
                        "human_gold": False,
                    },
                ],
            })
            records = perturbation.read_json(bundle_path)["records"]
            review_manifest, split_manifest = write_formal_freeze_manifests(
                bundle_path, records
            )

            manifest = perturbation.prepare_manifest(
                self._manifest_config({
                    "input_bundle": "inputs/behavior_bundle.json",
                    "review_freeze_manifest": str(review_manifest),
                    "split_freeze_manifest": str(split_manifest),
                }),
                project_root,
                run_dir,
                10,
                "bundle-test",
            )

            self.assertEqual(manifest["input_bundle_schema_version"], perturbation.INPUT_BUNDLE_SCHEMA_VERSION)
            self.assertEqual(manifest["input_bundle_id"], "reviewed-public-v1")
            self.assertEqual(
                manifest["input_bundle_sha256"],
                hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["input_admission_counts"]["frozen"], 1)
            self.assertEqual(manifest["input_admission_counts"]["pending_review"], 0)
            self.assertEqual(manifest["selected_count"], 1)
            self.assertEqual(
                manifest["review_freeze_manifest"]["sha256"],
                hashlib.sha256(review_manifest.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["split_freeze_manifest"]["sha256"],
                hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
            )
            selected = manifest["records"][0]
            self.assertEqual(selected["base_fact_id"], "public:frozen-1")
            self.assertEqual(selected["base_id"], "public:frozen-1")
            self.assertEqual(selected["canonical_status"], "frozen")
            self.assertEqual(selected["split_group_id"], "answer-group::paris")
            self.assertEqual(selected["split_assignment"], "development")
            self.assertEqual(selected["split_status"], "frozen")
            self.assertEqual(selected["split_policy_version"], "answer-group-v1")
            self.assertEqual(selected["sealed_evaluation_status"], "not_run")
            self.assertEqual(selected["canonical_fact"], "The capital of France is Paris.")
            self.assertEqual(len(selected["input_record_sha256"]), 64)
            self.assertEqual(selected["answer_aliases_en"], ["City of Paris", "Paris"])
            self.assertEqual(selected["distractors"][0]["distractor_id"], "frozen-1::berlin")
            self.assertEqual(selected["distractors"][0]["text_en"], "Berlin")
            queue = perturbation.read_jsonl(run_dir / "input_review_queue.jsonl")
            self.assertEqual(queue, [])
            exclusions = perturbation.read_jsonl(run_dir / "input_exclusions.jsonl")
            self.assertEqual([row["source_id"] for row in exclusions], ["rejected-1"])

    def test_review_freeze_rejects_nonterminal_source_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            records = [
                explicit_frozen_row(),
                {
                    "source_id": "pending-contract",
                    "base_fact_id": "public:pending-contract",
                    "canonical_status": "pending_review",
                },
            ]
            perturbation.write_json(
                bundle_path,
                {
                    "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    "records": records,
                },
            )
            review_manifest, split_manifest = write_formal_freeze_manifests(
                bundle_path, records
            )

            with self.assertRaisesRegex(ValueError, "non-terminal canonical_status"):
                perturbation.prepare_manifest(
                    self._manifest_config(
                        {
                            "review_freeze_manifest": str(review_manifest),
                            "split_freeze_manifest": str(split_manifest),
                        }
                    ),
                    root,
                    root / "runs" / "nonterminal-universe",
                    10,
                    "nonterminal-universe",
                    input_bundle_path=bundle_path,
                )

    def test_explicit_input_bundle_rejects_renamed_provisional_formal_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [{
                    "source_id": "renamed-provisional",
                    "base_fact_id": "public:renamed-provisional",
                    "source_dataset": "global_mmlu",
                    "bundle_status": "provisional",
                    "admission": {
                        "semantic_review_complete": False,
                        "prompt_ready": True,
                    },
                    "evidence_tier": "qwen_single_model_pending",
                    "human_gold": False,
                    "canonical_status": "frozen",
                    "split_group_id": "answer-group::pending",
                    "split_assignment": "development",
                    "split_status": "proposed",
                    "split_policy_version": "answer-group-v1",
                    "sealed_evaluation_status": "not_run",
                    "prompt_ready": True,
                    "prompt_quality_tier": "strict_factual_completion",
                    "prompt_en": "The capital of France is",
                    "answer_en": "Paris",
                    "answer_type": "city",
                    "canonical_fact_en": "The capital of France is Paris.",
                    "distractors": [],
                }],
            })
            records = perturbation.read_json(bundle_path)["records"]
            review_manifest, split_manifest = write_formal_freeze_manifests(
                bundle_path, records
            )

            with self.assertRaisesRegex(ValueError, "canonical formal tier"):
                perturbation.prepare_manifest(
                    self._manifest_config({
                        "review_freeze_manifest": str(review_manifest),
                        "split_freeze_manifest": str(split_manifest),
                    }),
                    project_root,
                    project_root / "runs" / "provisional-test",
                    10,
                    "provisional-test",
                    input_bundle_path=bundle_path,
                )

    def test_explicit_frozen_row_without_external_manifests_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [explicit_frozen_row()],
            })

            manifest = perturbation.prepare_manifest(
                self._manifest_config(),
                project_root,
                project_root / "runs" / "missing-external-manifests",
                10,
                "missing-external-manifests",
                input_bundle_path=bundle_path,
            )

            self.assertEqual(manifest["selected_count"], 0)
            self.assertFalse(manifest["external_formal_admission"]["authorized"])
            queue = perturbation.read_jsonl(
                project_root
                / "runs"
                / "missing-external-manifests"
                / "input_review_queue.jsonl"
            )
            self.assertEqual(
                set(queue[0]["input_admission_blockers"]),
                {
                    "review_freeze_manifest_missing",
                    "split_freeze_manifest_missing",
                },
            )

    def test_configured_external_manifest_contract_failures_are_fatal(self):
        cases = (
            (
                "self_asserted_v1_manifest",
                {"schema_version": "factual-review-freeze-manifest-v1"},
                {},
                "unsupported review freeze manifest schema_version",
            ),
            (
                "stale_bundle_sha",
                {"input_bundle": {
                    "sha256": "0" * 64,
                    "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    "record_count": 1,
                    "base_fact_ids_sha256": perturbation.formal_admission.sha256_value(
                        ["public:frozen-contract"]
                    ),
                }},
                {},
                "input_bundle.sha256",
            ),
            (
                "review_incomplete",
                {"review_complete": False},
                {},
                "review_complete",
            ),
            (
                "canonical_freeze_not_authorized",
                {"canonical_freeze_authorized": False},
                {},
                "canonical_freeze_authorized",
            ),
            (
                "near_duplicate_incomplete",
                {},
                {"near_duplicate_review_complete": False},
                "near_duplicate_review_complete",
            ),
            (
                "unresolved_near_duplicate",
                {},
                {"unresolved_near_duplicate_count": 1},
                "unresolved_near_duplicate_count",
            ),
            (
                "unresolved_cross_split",
                {},
                {"unresolved_cross_split_count": 1},
                "unresolved_cross_split_count",
            ),
            (
                "historical_exposure_incomplete",
                {},
                {"historical_exposure_check_complete": False},
                "historical_exposure_check_complete",
            ),
            (
                "sealed_exposure_present",
                {},
                {"sealed_exposure_count": 1},
                "sealed_exposure_count",
            ),
            (
                "validation_exposure_present",
                {},
                {"validation_exposure_count": 1},
                "validation_exposure_count",
            ),
            (
                "split_assignment_digest_stale",
                {},
                {"split_assignments_sha256": "0" * 64},
                "split_assignments_sha256",
            ),
        )
        for name, review_overrides, split_overrides, expected_error in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                project_root = Path(directory)
                bundle_path = project_root / "behavior_bundle.json"
                records = [explicit_frozen_row()]
                perturbation.write_json(bundle_path, {
                    "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    "records": records,
                })
                review_manifest, split_manifest = write_formal_freeze_manifests(
                    bundle_path,
                    records,
                    review_overrides=review_overrides,
                    split_overrides=split_overrides,
                )
                config = self._manifest_config({
                    "review_freeze_manifest": str(review_manifest),
                    "split_freeze_manifest": str(split_manifest),
                })

                with self.assertRaisesRegex(ValueError, expected_error):
                    perturbation.prepare_manifest(
                        config,
                        project_root,
                        project_root / "runs" / name,
                        10,
                        name,
                        input_bundle_path=bundle_path,
                    )

    def test_precomputed_bundle_sha_must_match_real_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            records = [explicit_frozen_row()]
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": records,
            })

            with self.assertRaisesRegex(ValueError, "does not match input bundle bytes"):
                perturbation.formal_admission.bundle_identity(
                    bundle_path,
                    perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records,
                    "0" * 64,
                )

    def test_bundle_atomic_replacement_between_parse_and_admission_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            original_payload = {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [explicit_frozen_row()],
            }
            replacement_payload = {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [{**explicit_frozen_row(), "prompt_en": "Changed prompt"}],
            }
            perturbation.write_json(bundle_path, original_payload)
            original_read_bytes = Path.read_bytes
            state = {"swapped": False}

            def read_then_replace(path):
                raw = original_read_bytes(path)
                if path.resolve() == bundle_path.resolve() and not state["swapped"]:
                    perturbation.write_json(bundle_path, replacement_payload)
                    state["swapped"] = True
                return raw

            with patch.object(Path, "read_bytes", read_then_replace):
                with self.assertRaisesRegex(ValueError, "does not match input bundle bytes"):
                    perturbation.prepare_manifest(
                        self._manifest_config(),
                        root,
                        root / "runs" / "atomic-swap",
                        10,
                        "atomic-swap",
                        input_bundle_path=bundle_path,
                    )

    def test_manifest_hash_and_validation_use_one_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            records = [explicit_frozen_row()]
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": records,
            })
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)
            validated_review_bytes = review_path.read_bytes()
            validated_review_sha = hashlib.sha256(validated_review_bytes).hexdigest()
            original_read_bytes = Path.read_bytes
            state = {"swapped": False}

            def snapshot_then_replace(path):
                raw = original_read_bytes(path)
                if path.resolve() == review_path.resolve() and not state["swapped"]:
                    review_path.write_text(
                        json.dumps({"schema_version": "tampered-after-snapshot"}),
                        encoding="utf-8",
                    )
                    state["swapped"] = True
                return raw

            with patch.object(Path, "read_bytes", snapshot_then_replace):
                result = perturbation.formal_admission.validate_external_formal_admission(
                    input_bundle_path=bundle_path,
                    input_schema_version=perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records=records,
                    input_bundle_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                    review_freeze_manifest_path=review_path,
                    split_freeze_manifest_path=split_path,
                )

            self.assertTrue(result["authorized"])
            self.assertEqual(
                result["review_freeze_manifest"]["sha256"], validated_review_sha
            )
            self.assertNotEqual(
                result["review_freeze_manifest"]["sha256"],
                hashlib.sha256(review_path.read_bytes()).hexdigest(),
            )

    def test_codex_proxy_review_evidence_cannot_claim_human_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            records = [explicit_frozen_row()]
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": records,
            })
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)
            review = perturbation.read_json(review_path)
            decisions_path = Path(review["review_decisions"]["path"])
            decisions = perturbation.read_jsonl(decisions_path)
            decisions[0]["human_gold"] = True
            perturbation.write_jsonl(decisions_path, decisions)
            review["review_decisions"]["sha256"] = hashlib.sha256(
                decisions_path.read_bytes()
            ).hexdigest()
            perturbation.write_json(review_path, review)

            with self.assertRaisesRegex(ValueError, "codex_proxy must have human_gold=false"):
                perturbation.prepare_manifest(
                    self._manifest_config({
                        "review_freeze_manifest": str(review_path),
                        "split_freeze_manifest": str(split_path),
                    }),
                    root,
                    root / "runs" / "bad-codex-provenance",
                    10,
                    "bad-codex-provenance",
                    input_bundle_path=bundle_path,
                )

    def test_reviewer_type_must_be_an_exact_supported_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            records = [explicit_frozen_row()]
            perturbation.write_json(
                bundle_path,
                {
                    "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    "records": records,
                },
            )
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)
            review = perturbation.read_json(review_path)
            decisions_path = Path(review["review_decisions"]["path"])
            decisions = perturbation.read_jsonl(decisions_path)
            decisions[0]["reviewer_type"] = "codex_proxy "
            decisions[0]["human_gold"] = True
            perturbation.write_jsonl(decisions_path, decisions)
            review["review_decisions"]["sha256"] = hashlib.sha256(
                decisions_path.read_bytes()
            ).hexdigest()
            perturbation.write_json(review_path, review)

            with self.assertRaisesRegex(ValueError, "unsupported reviewer_type"):
                perturbation.prepare_manifest(
                    self._manifest_config(
                        {
                            "review_freeze_manifest": str(review_path),
                            "split_freeze_manifest": str(split_path),
                        }
                    ),
                    root,
                    root / "runs" / "bad-reviewer-type",
                    10,
                    "bad-reviewer-type",
                    input_bundle_path=bundle_path,
                )

    def test_split_freeze_rejects_omitted_mechanical_near_duplicate_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = explicit_frozen_row()
            second = {
                **explicit_frozen_row(),
                "source_id": "frozen-contract-2",
                "base_fact_id": "public:frozen-contract-2",
                "split_group_id": "answer-group::other-paris",
                "split_assignment": "validation",
            }
            records = [first, second]
            bundle_path = root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": records,
            })
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)
            split = perturbation.read_json(split_path)
            for field, digest_field in (
                ("near_duplicate_candidates", "candidate_ids_sha256"),
                ("near_duplicate_adjudications", "candidate_ids_sha256"),
            ):
                artifact_path = Path(split[field]["path"])
                perturbation.write_jsonl(artifact_path, [])
                split[field]["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                split[field]["record_count"] = 0
                split[field][digest_field] = perturbation.formal_admission.sha256_value([])
            perturbation.write_json(split_path, split)

            with self.assertRaisesRegex(ValueError, "omit mechanically detectable"):
                perturbation.prepare_manifest(
                    self._manifest_config({
                        "review_freeze_manifest": str(review_path),
                        "split_freeze_manifest": str(split_path),
                        "allowed_splits": ["development", "validation"],
                    }),
                    root,
                    root / "runs" / "missing-near-duplicate",
                    10,
                    "missing-near-duplicate",
                    input_bundle_path=bundle_path,
                )

    def test_exact_duplicate_cannot_be_adjudicated_distinct_across_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = explicit_frozen_row()
            second = {
                **explicit_frozen_row(),
                "source_id": "frozen-contract-copy",
                "base_fact_id": "public:frozen-contract-copy",
                "split_group_id": "answer-group::paris-copy",
                "split_assignment": "validation",
            }
            records = [first, second]
            bundle_path = root / "behavior_bundle.json"
            perturbation.write_json(
                bundle_path,
                {
                    "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    "records": records,
                },
            )
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)

            with self.assertRaisesRegex(
                ValueError, "exact duplicate candidate cannot be adjudicated distinct"
            ):
                perturbation.prepare_manifest(
                    self._manifest_config(
                        {
                            "review_freeze_manifest": str(review_path),
                            "split_freeze_manifest": str(split_path),
                        }
                    ),
                    root,
                    root / "runs" / "exact-duplicate-distinct",
                    10,
                    "exact-duplicate-distinct",
                    input_bundle_path=bundle_path,
                )

    def test_split_freeze_rejects_historically_exposed_sealed_fact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                **explicit_frozen_row(),
                "split_group_id": "sealed-group",
                "split_assignment": "sealed",
            }
            records = [row]
            bundle_path = root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": records,
            })
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)
            split = perturbation.read_json(split_path)
            checks_path = Path(split["historical_exposure_checks"]["path"])
            checks = perturbation.read_jsonl(checks_path)
            checks[0]["historically_exposed"] = True
            perturbation.write_jsonl(checks_path, checks)
            split["historical_exposure_checks"]["sha256"] = hashlib.sha256(
                checks_path.read_bytes()
            ).hexdigest()
            perturbation.write_json(split_path, split)

            with self.assertRaisesRegex(ValueError, "historical exposure is unresolved or present"):
                perturbation.prepare_manifest(
                    self._manifest_config({
                        "review_freeze_manifest": str(review_path),
                        "split_freeze_manifest": str(split_path),
                        "allowed_splits": ["sealed"],
                    }),
                    root,
                    root / "runs" / "exposed-sealed",
                    10,
                    "exposed-sealed",
                    input_bundle_path=bundle_path,
                )

    def test_resume_requires_exact_composite_admission_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [explicit_frozen_row()]
            bundle_path = root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": records,
            })
            review_path, split_path = write_formal_freeze_manifests(bundle_path, records)
            config = self._manifest_config({
                "review_freeze_manifest": str(review_path),
                "split_freeze_manifest": str(split_path),
            })
            run_dir = root / "runs" / "resume-fingerprint"
            perturbation.prepare_manifest(
                config, root, run_dir, 10, "resume-fingerprint", input_bundle_path=bundle_path
            )

            review = perturbation.read_json(review_path)
            decisions_path = Path(review["review_decisions"]["path"])
            decisions = perturbation.read_jsonl(decisions_path)
            decisions[0]["notes"] = "same decision, changed evidence bytes"
            perturbation.write_jsonl(decisions_path, decisions)
            review["review_decisions"]["sha256"] = hashlib.sha256(
                decisions_path.read_bytes()
            ).hexdigest()
            perturbation.write_json(review_path, review)
            split = perturbation.read_json(split_path)
            split["review_freeze_manifest"]["sha256"] = hashlib.sha256(
                review_path.read_bytes()
            ).hexdigest()
            perturbation.write_json(split_path, split)

            with self.assertRaisesRegex(ValueError, "formal admission fingerprint"):
                perturbation.prepare_manifest(
                    config,
                    root,
                    run_dir,
                    10,
                    "resume-fingerprint",
                    input_bundle_path=bundle_path,
                )

    def test_resume_rejects_changed_runtime_before_manifest_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "behavior_bundle.json"
            records = [explicit_frozen_row()]
            perturbation.write_json(
                bundle_path,
                {
                    "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                    "records": records,
                },
            )
            run_dir = root / "runs" / "runtime-fingerprint"
            perturbation.prepare_manifest(
                self._manifest_config(),
                root,
                run_dir,
                10,
                "runtime-fingerprint",
                input_bundle_path=bundle_path,
            )
            previous = perturbation.read_json(run_dir / "run_manifest.json")
            previous["runtime_sha256"] = "0" * 64
            perturbation.write_json(run_dir / "run_manifest.json", previous)

            with self.assertRaisesRegex(ValueError, "runtime fingerprint"):
                perturbation.prepare_manifest(
                    self._manifest_config(),
                    root,
                    run_dir,
                    10,
                    "runtime-fingerprint",
                    input_bundle_path=bundle_path,
                )

    def test_holdout_rejects_missing_or_changed_runtime_before_router_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {"config_version": "holdout-runtime-test"}
            self._write_candidate_freeze_fixture(run_dir)

            cases = (
                (None, "missing runtime fingerprint"),
                ("0" * 64, "runtime fingerprint does not match current code"),
            )
            for runtime_sha256, message in cases:
                with self.subTest(runtime_sha256=runtime_sha256):
                    manifest = {"config_sha256": perturbation.sha256_value(config)}
                    if runtime_sha256 is not None:
                        manifest["runtime_sha256"] = runtime_sha256
                    perturbation.write_json(run_dir / "run_manifest.json", manifest)

                    with patch.object(perturbation, "ModelRouter") as router:
                        with self.assertRaisesRegex(ValueError, message):
                            perturbation.run_holdout(
                                config,
                                run_dir,
                                run_dir / ".env",
                            )
                    router.assert_not_called()

    def test_candidate_freeze_manifest_binds_frozen_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(
                run_dir / "run_manifest.json",
                {"simulation_models": []},
            )
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [])
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", [])

            frozen = perturbation.freeze_candidates(
                run_dir,
                {"model_roles": {"simulation": {"models": []}}},
            )

            self.assertEqual(frozen, [])
            manifest = perturbation.read_json(
                run_dir / "candidate_freeze_manifest.json"
            )
            binding = manifest["frozen_candidates"]
            frozen_path = run_dir / "frozen_candidates.jsonl"
            self.assertEqual(
                manifest["schema_version"],
                perturbation.CANDIDATE_FREEZE_MANIFEST_SCHEMA_VERSION,
            )
            self.assertEqual(binding["path"], frozen_path.name)
            self.assertEqual(
                binding["sha256"], hashlib.sha256(frozen_path.read_bytes()).hexdigest()
            )
            self.assertEqual(binding["record_count"], 0)
            self.assertEqual(
                binding["schema_version"],
                perturbation.CANDIDATE_FREEZE_RECORD_SCHEMA_VERSION,
            )
            self.assertEqual(
                binding["candidate_ids_sha256"], perturbation.sha256_value([])
            )

    def test_holdout_rejects_stale_or_incomplete_candidate_freeze_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {"config_version": "holdout-freeze-binding-test"}
            row = self._candidate_freeze_row()
            base_manifest = self._write_candidate_freeze_fixture(run_dir, [row])
            perturbation.write_json(
                run_dir / "run_manifest.json",
                {
                    "config_sha256": perturbation.sha256_value(config),
                    "runtime_sha256": perturbation._runtime_sha256(),
                },
            )
            cases = (
                (
                    "old_manifest_schema",
                    lambda value: value.update({
                        "schema_version": "factual-perturbation-candidate-freeze-v1"
                    }),
                    "manifest schema",
                ),
                (
                    "missing_binding",
                    lambda value: value.pop("frozen_candidates"),
                    "missing frozen_candidates binding",
                ),
                (
                    "wrong_artifact_schema",
                    lambda value: value["frozen_candidates"].update({
                        "schema_version": "wrong"
                    }),
                    "artifact schema",
                ),
                (
                    "stale_sha",
                    lambda value: value["frozen_candidates"].update({
                        "sha256": "0" * 64
                    }),
                    "SHA-256 mismatch",
                ),
                (
                    "wrong_count",
                    lambda value: value["frozen_candidates"].update({
                        "record_count": 2
                    }),
                    "record count mismatch",
                ),
                (
                    "wrong_digest",
                    lambda value: value["frozen_candidates"].update({
                        "candidate_ids_sha256": "0" * 64
                    }),
                    "candidate ID digest mismatch",
                ),
            )
            for name, mutate, message in cases:
                with self.subTest(case=name):
                    manifest = json.loads(json.dumps(base_manifest))
                    mutate(manifest)
                    perturbation.write_json(
                        run_dir / "candidate_freeze_manifest.json", manifest
                    )
                    with patch.object(perturbation, "ModelRouter") as router:
                        with self.assertRaisesRegex(ValueError, message):
                            perturbation.run_holdout(
                                config,
                                run_dir,
                                run_dir / ".env",
                            )
                    router.assert_not_called()

    def test_holdout_rejects_invalid_frozen_candidate_row_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {"config_version": "holdout-row-contract-test"}
            perturbation.write_json(
                run_dir / "run_manifest.json",
                {
                    "config_sha256": perturbation.sha256_value(config),
                    "runtime_sha256": perturbation._runtime_sha256(),
                },
            )
            cases = (
                (
                    "missing_freeze_version",
                    lambda row: row.pop("freeze_version"),
                    "invalid freeze_version",
                ),
                (
                    "old_freeze_version",
                    lambda row: row.update({
                        "freeze_version": "factual-perturbation-candidate-freeze-v1"
                    }),
                    "invalid freeze_version",
                ),
                (
                    "missing_source_id",
                    lambda row: row.pop("source_id"),
                    "invalid source_id",
                ),
                (
                    "padded_distractor_id",
                    lambda row: row.update({"distractor_id": " distractor-1"}),
                    "invalid distractor_id",
                ),
                (
                    "candidate_not_object",
                    lambda row: row.update({"candidate": []}),
                    "invalid candidate object",
                ),
                (
                    "missing_chinese_context",
                    lambda row: row["candidate"].pop("chinese_context"),
                    "invalid candidate.chinese_context",
                ),
                (
                    "invalid_badcase_type",
                    lambda row: row.update({"badcase_type": "unknown"}),
                    "invalid badcase_type",
                ),
                (
                    "accuracy_not_object",
                    lambda row: row.update({"accuracy": []}),
                    "invalid accuracy object",
                ),
                (
                    "accuracy_wrong_keys",
                    lambda row: row["accuracy"].pop("zh_perturbed"),
                    "invalid accuracy keys",
                ),
                (
                    "accuracy_boolean_value",
                    lambda row: row["accuracy"].update({"zh_perturbed": False}),
                    "invalid accuracy value",
                ),
            )
            for name, mutate, message in cases:
                with self.subTest(case=name):
                    row = self._candidate_freeze_row()
                    mutate(row)
                    self._write_candidate_freeze_fixture(run_dir, [row])
                    with patch.object(perturbation, "ModelRouter") as router:
                        with self.assertRaisesRegex(ValueError, message):
                            perturbation.run_holdout(
                                config,
                                run_dir,
                                run_dir / ".env",
                            )
                    router.assert_not_called()

    def test_explicit_input_bundle_excludes_sealed_split_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [{
                    "source_id": "reviewed-sealed",
                    "base_fact_id": "public:reviewed-sealed",
                    "source_dataset": "global_mmlu",
                    "bundle_status": "reviewed_frozen",
                    "admission": {
                        "semantic_review_complete": True,
                        "prompt_ready": True,
                    },
                    "evidence_tier": "independent_adjudicated",
                    "human_gold": False,
                    "canonical_status": "frozen",
                    "split_group_id": "answer-group::sealed",
                    "split_assignment": "sealed",
                    "split_status": "frozen",
                    "split_policy_version": "answer-group-v1",
                    "sealed_evaluation_status": "not_run",
                    "prompt_ready": True,
                    "prompt_quality_tier": "strict_factual_completion",
                    "prompt_en": "The capital of France is",
                    "answer_en": "Paris",
                    "answer_type": "city",
                    "canonical_fact_en": "The capital of France is Paris.",
                    "distractors": [],
                }],
            })
            records = perturbation.read_json(bundle_path)["records"]
            review_manifest, split_manifest = write_formal_freeze_manifests(
                bundle_path, records
            )

            manifest = perturbation.prepare_manifest(
                self._manifest_config({
                    "review_freeze_manifest": str(review_manifest),
                    "split_freeze_manifest": str(split_manifest),
                }),
                project_root,
                project_root / "runs" / "sealed-test",
                10,
                "sealed-test",
                input_bundle_path=bundle_path,
            )

            self.assertEqual(manifest["selected_count"], 0)
            self.assertEqual(
                manifest["input_admission_policy"]["allowed_splits_for_this_prepare"],
                ["development"],
            )
            self.assertEqual(manifest["input_admission_counts"]["split_not_allowed"], 1)
            exclusions = perturbation.read_jsonl(
                project_root / "runs" / "sealed-test" / "input_exclusions.jsonl"
            )
            self.assertEqual(len(exclusions), 1)
            self.assertEqual(
                exclusions[0]["input_admission_status"],
                "split_not_allowed_for_prepare",
            )

    def test_explicit_formal_row_requires_complete_split_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [{
                    "source_id": "reviewed-missing-split-identity",
                    "base_fact_id": "public:reviewed-missing-split-identity",
                    "source_dataset": "global_mmlu",
                    "bundle_status": "reviewed_frozen",
                    "admission": {
                        "semantic_review_complete": True,
                        "prompt_ready": True,
                    },
                    "evidence_tier": "independent_adjudicated",
                    "human_gold": False,
                    "canonical_status": "frozen",
                    "split_assignment": "development",
                    "split_status": "frozen",
                    "split_policy_version": "answer-group-v1",
                    "prompt_ready": True,
                    "prompt_quality_tier": "strict_factual_completion",
                    "prompt_en": "The capital of France is",
                    "answer_en": "Paris",
                    "canonical_fact_en": "The capital of France is Paris.",
                    "distractors": [],
                }],
            })
            manifest = perturbation.prepare_manifest(
                self._manifest_config(),
                project_root,
                project_root / "runs" / "missing-split-identity",
                10,
                "missing-split-identity",
                input_bundle_path=bundle_path,
            )

            self.assertEqual(manifest["selected_count"], 0)
            queue = perturbation.read_jsonl(
                project_root / "runs" / "missing-split-identity" / "input_review_queue.jsonl"
            )
            self.assertIn("split_group_id_missing", queue[0]["input_admission_blockers"])

    def test_explicit_bundle_rejects_inconsistent_split_group(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            base = {
                "base_fact_id": "placeholder-overridden-below",
                "source_dataset": "global_mmlu",
                "bundle_status": "reviewed_frozen",
                "admission": {
                    "semantic_review_complete": True,
                    "prompt_ready": True,
                },
                "evidence_tier": "independent_adjudicated",
                "human_gold": False,
                "canonical_status": "frozen",
                "split_group_id": "answer-group::paris",
                "split_status": "frozen",
                "split_policy_version": "answer-group-v1",
                "prompt_ready": True,
                "prompt_quality_tier": "strict_factual_completion",
                "prompt_en": "The capital of France is",
                "answer_en": "Paris",
                "canonical_fact_en": "The capital of France is Paris.",
                "distractors": [],
            }
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [
                    {
                        **base,
                        "source_id": "split-dev",
                        "base_fact_id": "public:split-dev",
                        "split_assignment": "development",
                    },
                    {
                        **base,
                        "source_id": "split-val",
                        "base_fact_id": "public:split-val",
                        "split_assignment": "validation",
                    },
                ],
            })

            with self.assertRaisesRegex(ValueError, "inconsistent assignment"):
                perturbation.prepare_manifest(
                    self._manifest_config(),
                    project_root,
                    project_root / "runs" / "split-conflict",
                    10,
                    "split-conflict",
                    input_bundle_path=bundle_path,
                )

    def test_explicit_bundle_requires_one_split_policy_version(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            base = {
                "source_dataset": "global_mmlu",
                "bundle_status": "reviewed_frozen",
                "admission": {
                    "semantic_review_complete": True,
                    "prompt_ready": True,
                },
                "evidence_tier": "independent_adjudicated",
                "human_gold": False,
                "canonical_status": "frozen",
                "split_status": "frozen",
                "split_assignment": "development",
                "prompt_ready": True,
                "prompt_quality_tier": "strict_factual_completion",
                "prompt_en": "The capital of France is",
                "answer_en": "Paris",
                "canonical_fact_en": "The capital of France is Paris.",
                "distractors": [],
            }
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [
                    {
                        **base,
                        "source_id": "policy-one",
                        "base_fact_id": "public:policy-one",
                        "split_group_id": "group-one",
                        "split_policy_version": "answer-group-v1",
                    },
                    {
                        **base,
                        "source_id": "policy-two",
                        "base_fact_id": "public:policy-two",
                        "split_group_id": "group-two",
                        "split_policy_version": "answer-group-v2",
                    },
                ],
            })

            with self.assertRaisesRegex(ValueError, "exactly one split_policy_version"):
                perturbation.prepare_manifest(
                    self._manifest_config(),
                    project_root,
                    project_root / "runs" / "split-policy-conflict",
                    10,
                    "split-policy-conflict",
                    input_bundle_path=bundle_path,
                )

    def test_explicit_input_bundle_requires_consistent_prompt_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [{
                    "source_id": "conflicting-prompt-readiness",
                    "base_fact_id": "public:conflicting-prompt-readiness",
                    "source_dataset": "global_mmlu",
                    "bundle_status": "reviewed_frozen",
                    "admission": {
                        "semantic_review_complete": True,
                        "prompt_ready": False,
                    },
                    "evidence_tier": "independent_adjudicated",
                    "human_gold": False,
                    "canonical_status": "frozen",
                    "split_group_id": "answer-group::paris",
                    "split_assignment": "development",
                    "split_status": "frozen",
                    "split_policy_version": "answer-group-v1",
                    "sealed_evaluation_status": "not_run",
                    "prompt_ready": True,
                    "factual_prompt_status": "generated",
                    "prompt_quality_tier": "strict_factual_completion",
                    "prompt_en": "The capital of France is",
                    "answer_en": "Paris",
                    "answer_type": "city",
                    "canonical_fact_en": "The capital of France is Paris.",
                    "distractors": [],
                }],
            })
            records = perturbation.read_json(bundle_path)["records"]
            review_manifest, split_manifest = write_formal_freeze_manifests(
                bundle_path, records
            )

            manifest = perturbation.prepare_manifest(
                self._manifest_config({
                    "review_freeze_manifest": str(review_manifest),
                    "split_freeze_manifest": str(split_manifest),
                }),
                project_root,
                project_root / "runs" / "prompt-readiness-test",
                10,
                "prompt-readiness-test",
                input_bundle_path=bundle_path,
            )

            self.assertEqual(manifest["selected_count"], 0)
            self.assertEqual(manifest["input_admission_counts"]["frozen_prompt_not_ready"], 1)
            exclusions = perturbation.read_jsonl(
                project_root
                / "runs"
                / "prompt-readiness-test"
                / "input_exclusions.jsonl"
            )
            self.assertEqual(
                exclusions[0]["input_admission_status"],
                "frozen_but_prompt_not_ready",
            )

    def test_legacy_config_derives_sibling_prompt_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            data_dir = (
                project_root
                / perturbation.formal_admission.KNOWN_LEGACY_CANONICAL_RELATIVE_PATH.parent
            )
            canonical_path = data_dir / "canonical_triples.jsonl"
            perturbation.write_jsonl(canonical_path, [])
            prompt_path = data_dir / "factual_triples_with_distractors.jsonl"
            perturbation.write_jsonl(prompt_path, [
                {
                    "source_id": "legacy-1",
                    "source_dataset": "legacy",
                    "canonical_decision": "keep",
                    "factual_prompt_status": "generated",
                    "prompt_quality_tier": "strict_factual_completion",
                    "prompt_en": "The capital of France is",
                    "prompt_expected_answer": "Paris",
                    "source_answer": "Paris",
                    "answer_type": "city",
                    "canonical_fact": "The capital of France is Paris.",
                    "relation_raw": "capital city",
                    "distractor_candidates": [{
                        "distractor_answer": "Berlin",
                        "distractor_source": "original_wrong_option",
                        "source_choice_index": 2,
                    }],
                },
                {
                    "source_id": "legacy-explicit-fields-only",
                    "source_dataset": "legacy",
                    "canonical_decision": "keep",
                    "prompt_ready": True,
                    "canonical_fact_en": "This field belongs to the explicit schema.",
                },
            ])
            config = self._manifest_config({
                "canonical_triples": str(
                    perturbation.formal_admission.KNOWN_LEGACY_CANONICAL_RELATIVE_PATH
                ),
            })

            with (
                patch.object(
                    perturbation.formal_admission,
                    "KNOWN_LEGACY_CANONICAL_SHA256",
                    hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
                ),
                patch.object(
                    perturbation.formal_admission,
                    "KNOWN_LEGACY_INPUT_SHA256",
                    hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                ),
                patch.object(
                    perturbation.formal_admission,
                    "KNOWN_LEGACY_INPUT_RECORD_COUNT",
                    2,
                ),
            ):
                manifest = perturbation.prepare_manifest(
                    config,
                    project_root,
                    project_root / "runs" / "legacy-test",
                    10,
                    "legacy-test",
                )

            self.assertEqual(
                manifest["input_bundle_schema_version"],
                perturbation.LEGACY_INPUT_BUNDLE_SCHEMA_VERSION,
            )
            self.assertEqual(
                manifest["prompt_input"],
                str(
                    perturbation.formal_admission.KNOWN_LEGACY_INPUT_RELATIVE_PATH
                ),
            )
            self.assertEqual(manifest["selected_count"], 1)
            self.assertEqual(manifest["input_admission_counts"]["frozen_prompt_not_ready"], 1)
            self.assertEqual(
                manifest["records"][0]["distractors"][0]["distractor_id"],
                "legacy-1_source_wrong_2",
            )
            exclusions = perturbation.read_jsonl(
                project_root / "runs" / "legacy-test" / "input_exclusions.jsonl"
            )
            self.assertEqual(
                [row["source_id"] for row in exclusions],
                ["legacy-explicit-fields-only"],
            )

    def test_arbitrary_legacy_sibling_cannot_claim_compatibility_exemption(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            data_dir = project_root / "arbitrary-canonical-v2"
            canonical_path = data_dir / "canonical_triples.jsonl"
            prompt_path = data_dir / "factual_triples_with_distractors.jsonl"
            perturbation.write_jsonl(canonical_path, [])
            perturbation.write_jsonl(prompt_path, [{
                "source_id": "legacy-bypass",
                "source_dataset": "fixture",
                "canonical_decision": "keep",
                "factual_prompt_status": "generated",
                "prompt_quality_tier": "strict_factual_completion",
                "prompt_en": "The capital of France is",
                "prompt_expected_answer": "Paris",
                "canonical_fact": "The capital of France is Paris.",
                "distractor_candidates": [],
            }])

            with self.assertRaisesRegex(ValueError, "pinned historical artifact"):
                perturbation.prepare_manifest(
                    self._manifest_config({"canonical_triples": str(canonical_path)}),
                    project_root,
                    project_root / "runs" / "legacy-bypass",
                    10,
                    "legacy-bypass",
                )

    def test_explicit_input_bundle_requires_canonical_status(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "behavior_bundle.json"
            perturbation.write_json(bundle_path, {
                "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                "records": [{
                    "source_id": "missing-status",
                    "base_fact_id": "public:missing-status",
                }],
            })
            with self.assertRaisesRegex(ValueError, "invalid canonical_status"):
                perturbation.prepare_manifest(
                    self._manifest_config(),
                    project_root,
                    project_root / "runs" / "invalid",
                    10,
                    "invalid",
                    input_bundle_path=bundle_path,
                )

    def test_explicit_schema_less_jsonl_cannot_fall_back_to_legacy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            bundle_path = project_root / "schema-less.jsonl"
            perturbation.write_jsonl(bundle_path, [{
                "source_id": "would-bypass-as-legacy",
                "source_dataset": "fixture",
                "canonical_decision": "keep",
                "factual_prompt_status": "generated",
                "prompt_quality_tier": "strict_factual_completion",
                "prompt_en": "The capital of France is",
                "prompt_expected_answer": "Paris",
                "canonical_fact": "The capital of France is Paris.",
                "distractor_candidates": [],
            }])

            for config, override in (
                (self._manifest_config(), bundle_path),
                (self._manifest_config({"input_bundle": "schema-less.jsonl"}), None),
            ):
                with self.subTest(override=override is not None):
                    with self.assertRaisesRegex(
                        ValueError,
                        "explicit input bundle JSONL rows must declare schema_version",
                    ):
                        perturbation.prepare_manifest(
                            config,
                            project_root,
                            project_root / "runs" / f"reject-{override is not None}",
                            10,
                            "schema-less-explicit",
                            input_bundle_path=override,
                        )

    def test_explicit_bundle_requires_unique_consistent_base_fact_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "missing": ([{"source_id": "source-1"}], "missing non-empty base_fact_id"),
                "conflicting_alias": (
                    [{
                        "source_id": "source-1",
                        "base_fact_id": "fact-1",
                        "base_id": "fact-other",
                    }],
                    "conflicting base_fact_id and base_id",
                ),
                "duplicate": (
                    [
                        {"source_id": "source-1", "base_fact_id": "fact-1"},
                        {"source_id": "source-2", "base_fact_id": "fact-1"},
                    ],
                    "duplicate base_fact_id",
                ),
            }
            for name, (records, message) in cases.items():
                with self.subTest(case=name):
                    bundle_path = root / f"{name}.json"
                    perturbation.write_json(bundle_path, {
                        "schema_version": perturbation.INPUT_BUNDLE_SCHEMA_VERSION,
                        "records": records,
                    })
                    with self.assertRaisesRegex(ValueError, message):
                        perturbation._load_input_bundle(bundle_path)

    def test_config_overlay_replaces_simulation_models_without_losing_roles(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_sensitivity_v2.json"
        )
        self.assertEqual(config["config_version"], "factual-perturbation-zh-sensitivity-v2")
        self.assertEqual(config["model_roles"]["translation"]["primary"]["model"], "qwen3.8-max")
        self.assertEqual(
            [row["model"] for row in config["model_roles"]["simulation"]["models"]],
            ["qwen3.6-27b", "bailian/deepseek-v3.2", "gpt-5-mini", "gemini-3.7-flash"],
        )

    def test_strength_overlay_uses_matched_candidate_neutral(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_strength_v3.json"
        )
        self.assertEqual(config["model_roles"]["perturbation_generation"]["max_output_tokens"], 2048)
        self.assertEqual(config["model_roles"]["simulation"]["neutral_context_mode"], "matched_per_candidate")

    def test_streamlined_overlay_uses_repairs_dual_l2_and_soft_auto_judge(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_streamlined_v5.json"
        )
        self.assertTrue(config["model_roles"]["translation"]["repair_rejected"])
        self.assertTrue(config["model_roles"]["distractor_generation"]["enabled"])
        self.assertEqual(
            config["model_roles"]["perturbation_generation"]["strength_levels"],
            ["l2_lexical", "l2_relational"],
        )
        self.assertEqual(
            config["model_roles"]["perturbation_generation"]["primary"]["max_output_tokens"],
            2048,
        )
        self.assertFalse(config["model_roles"]["perturbation_validation"]["hard_gate"])

    def test_ollama_overlay_uses_two_local_models_serially(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_ollama_v6.json"
        )
        self.assertEqual(
            [row["model"] for row in config["model_roles"]["simulation"]["models"]],
            ["gemma3:12b", "llama3.1:8b"],
        )
        self.assertEqual(config["provider_profiles"]["ollama_local"]["authentication"], "none")
        self.assertEqual(config["execution"]["max_workers"], 1)
        self.assertTrue(config["model_roles"]["simulation"]["batch_by_model"])

    def test_five_model_overlay_combines_api_and_local_simulation_models(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_five_model_v7.json"
        )
        models = config["model_roles"]["simulation"]["models"]
        self.assertEqual(
            [row["model"] for row in models],
            [
                "qwen3.6-27b",
                "bailian/deepseek-v3.2",
                "gemini-3.7-flash",
                "gemma3:12b",
                "llama3.1:8b",
            ],
        )
        self.assertEqual(
            [row["provider_profile"] for row in models],
            ["aliyun", "deepseek", "test_gateway", "ollama_local", "ollama_local"],
        )
        self.assertEqual(config["provider_profiles"]["ollama_local"]["authentication"], "none")
        self.assertEqual(config["execution"]["provider_max_concurrency"]["ollama_local"], 1)
        self.assertEqual(config["execution"]["timeout_seconds"], 180)
        self.assertEqual(
            config["model_roles"]["perturbation_generation"]["target_distractors_per_source"],
            1,
        )
        self.assertEqual(config["model_roles"]["simulation"]["endpoint"], "multi_option")
        self.assertEqual(config["model_roles"]["simulation"]["neutral_context_mode"], "matched_per_candidate")
        self.assertTrue(config["model_roles"]["simulation"]["batch_by_model"])
        self.assertEqual(models[3]["timeout_seconds"], 900)
        self.assertEqual(models[4]["timeout_seconds"], 900)
        self.assertEqual(models[0]["timeout_seconds"], 30)
        self.assertEqual(models[2]["max_retries"], 1)

        simulation_config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_five_model_simulation_v7.json"
        )
        self.assertEqual(simulation_config["execution"]["timeout_seconds"], 180)
        self.assertEqual(
            [row["model"] for row in simulation_config["model_roles"]["simulation"]["models"]],
            [row["model"] for row in models],
        )

    def test_dual_distractor_four_model_v10_is_current_static_proxy_config(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_dual_distractor_proxy_v10.json"
        )
        config = runner.load_config(config_path)
        self.assertEqual(
            [
                (row["model"], row["provider_profile"])
                for row in config["model_roles"]["simulation"]["models"]
            ],
            list(perturbation.STATIC_PROXY_MODEL_ROSTER),
        )
        self.assertNotIn(
            "bailian/deepseek-v3.2",
            [row["model"] for row in config["model_roles"]["simulation"]["models"]],
        )
        model_by_name = {
            row["model"]: row
            for row in config["model_roles"]["simulation"]["models"]
        }
        self.assertEqual(
            model_by_name["gemini-3.7-flash"]["max_output_tokens"], 512
        )
        self.assertEqual(config["static_proxy"], {
            "roster_version": perturbation.STATIC_PROXY_ROSTER_VERSION,
            "expected_model_count": 4,
            "expected_calls_per_model": 960,
            "expected_simulation_call_count": 3840,
            "model_aggregation": "none",
        })
        tampered_config = json.loads(json.dumps(config))
        tampered_config["model_roles"]["simulation"]["models"].insert(1, {
            "model": "bailian/deepseek-v3.2",
            "provider_profile": "deepseek",
        })
        with self.assertRaisesRegex(ValueError, "frozen four-model roster"):
            perturbation._validate_static_proxy_config(tampered_config)
        self.assertTrue(
            set(perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS)
            <= set(config["model_roles"]["perturbation_validation"]["required_checks"])
        )
        manager_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "manage_static_g0a.py"
        )
        spec = importlib.util.spec_from_file_location("static_g0a_config_test", manager_path)
        manager = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(manager)
        static_config = manager._prepare_static_runtime_config(
            manager.load_config(config_path)
        )
        self.assertEqual(static_config["model_roles"]["simulation"]["models"], [])

    def test_static_single_qwen3_prepares_all_development_stimuli_without_calls(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_qwen3_8b_static_dev96_v1.json"
        )
        self.assertEqual(
            [
                (row["model"], row["provider_profile"])
                for row in perturbation._validate_static_single_qwen3_config(config)
            ],
            list(perturbation.STATIC_SINGLE_QWEN3_MODEL_ROSTER),
        )
        self.assertEqual(config["execution"]["max_workers"], 8)
        self.assertEqual(
            config["execution"]["provider_max_concurrency"]["qwen3_vllm"],
            8,
        )
        self.assertEqual(
            config["model_roles"]["simulation"]["models"][0]["max_retries"],
            1,
        )

        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(
                project_root
            )
            run_dir = project_root / "run"
            run_dir.mkdir()
            with patch.object(
                perturbation.ModelRouter, "request_json", autospec=True
            ) as request:
                manifest = perturbation.prepare_static_single_qwen3_run(
                    config,
                    project_root,
                    static_manifest_path,
                    run_dir,
                    "static-qwen3-test",
                )
            request.assert_not_called()

            self.assertEqual(
                manifest["execution_mode"],
                perturbation.STATIC_SINGLE_QWEN3_EXECUTION_MODE,
            )
            self.assertEqual(manifest["selected_count"], 96)
            self.assertEqual(manifest["expected_model_count"], 1)
            self.assertEqual(manifest["expected_calls_per_model"], 960)
            self.assertEqual(manifest["expected_simulation_call_count"], 960)
            self.assertFalse(manifest["exact_hf_evidence"])
            self.assertFalse(manifest["pnt_authorized"])
            self.assertFalse(manifest["pnt_eligible"])
            self.assertTrue(manifest["diagnostic_only"])
            self.assertEqual(
                manifest["behavior_evidence_class"], "single_model_api_behavior"
            )
            self.assertEqual(
                manifest["execution_limits"],
                {
                    "max_workers": 8,
                    "provider_max_concurrency": {"qwen3_vllm": 8},
                    "max_retries_by_model": {"qwen3-8b": 1},
                },
            )
            self.assertEqual(
                manifest["single_model_behavior_scope"],
                {
                    "purpose": "development_only_behavioral_defect_validation",
                    "prior_proxy_behavior_results_consumed": False,
                    "prior_proxy_labels_consumed": False,
                    "validation_or_sealed_consumed": False,
                    "exact_hf_checkpoint_identity_frozen": False,
                },
            )
            inputs = perturbation._load_static_proxy_behavior_inputs(
                manifest, run_dir
            )
            self.assertEqual(len(inputs), 960)
            self.assertEqual({row["split_assignment"] for row in inputs}, {"development"})
            self.assertEqual({row["model_spec"]["model"] for row in inputs}, {"qwen3-8b"})
            self.assertEqual(
                manifest["proxy_split_filter"]["validation_behavior_input_count"],
                0,
            )
            self.assertEqual(
                manifest["proxy_split_filter"]["sealed_behavior_input_count"],
                0,
            )

        tampered = json.loads(json.dumps(config))
        tampered["execution"]["max_workers"] = 7
        with self.assertRaisesRegex(ValueError, "max_workers=8"):
            perturbation._validate_static_single_qwen3_config(tampered)

    def test_static_single_qwen3_runs_and_analyzes_without_pnt_authorization(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_qwen3_8b_static_dev96_v1.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(
                project_root
            )
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_single_qwen3_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-qwen3-test",
            )

            def fake_request(
                _router, _stage, _item_id, spec, _prompt, validator
            ):
                parsed = {"choice": "B"}
                validator(parsed)
                return perturbation.ModelResult(
                    "completed",
                    parsed,
                    '{"choice":"B"}',
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            def one_write_stage(items, output_path, _key, worker, _workers):
                results = [worker(item) for item in items]
                perturbation.write_jsonl(output_path, results)
                return results

            identity_snapshot = static_proxy_route_runtime_identity_fixture(config)
            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_request,
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ), patch.object(
                perturbation, "stage_run", side_effect=one_write_stage
            ):
                probe = perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )
                summary = perturbation.run_preholdout(
                    config, project_root, run_dir, project_root / ".env"
                )

            self.assertTrue(probe["probe_passed"])
            self.assertEqual(probe["requested_model_count"], 1)
            self.assertEqual(summary["expected_simulation_call_count"], 960)
            self.assertEqual(summary["completed_simulation_call_count"], 960)
            self.assertEqual(
                summary["completed_call_count_by_model"], {"qwen3-8b": 960}
            )
            self.assertEqual(summary["validation_behavior_result_count"], 0)
            self.assertEqual(summary["sealed_behavior_result_count"], 0)
            self.assertFalse(summary["gate_passed"])
            self.assertFalse(summary["exact_hf_evidence"])
            self.assertFalse(summary["pnt_authorized"])

            analysis = perturbation.analyze_three_arm(run_dir)
            self.assertEqual(analysis["candidate_count"], 192)
            self.assertEqual(analysis["paths_not_taken_candidate_count"], 0)
            self.assertTrue(analysis["proxy_behavior_only"])
            self.assertFalse(analysis["exact_hf_evidence"])
            self.assertFalse(analysis["pnt_authorized"])

    def test_runner_parser_exposes_static_single_qwen3_commands(self):
        prepare_args = runner.parser().parse_args([
            "prepare-static-single-qwen3-run",
            "--run-id",
            "static-qwen3-test",
            "--static-freeze-manifest",
            "static_freeze_manifest.json",
        ])
        self.assertEqual(prepare_args.command, "prepare-static-single-qwen3-run")
        probe_args = runner.parser().parse_args([
            "probe-static-single-qwen3-run",
            "--run-id",
            "static-qwen3-test",
        ])
        self.assertEqual(probe_args.command, "probe-static-single-qwen3-run")

    def test_active_model_role_probes_exclude_retired_deepseek_v32(self):
        probe_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "probe_model_roles.py"
        )
        tree = ast.parse(probe_path.read_text(encoding="utf-8"))
        assignments = {
            target.id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for assignment_name in ("CASES", "MINIMAL_CASES"):
            string_values = {
                node.value
                for node in ast.walk(assignments[assignment_name])
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            self.assertNotIn("bailian/deepseek-v3.2", string_values)

    def test_static_proxy_v10_rejects_v8_config_and_legacy_run_artifacts(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        v8_config = runner.load_config(
            config_root / "factual_perturbation_zh_dual_distractor_proxy_v8.json"
        )
        with self.assertRaisesRegex(ValueError, "current roster contract"):
            perturbation._validate_static_proxy_config(v8_config)

        v10_config = runner.load_config(
            config_root / "factual_perturbation_zh_dual_distractor_proxy_v10.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_run_dir = root / "static-g0a-dev96-five-model-v8-20260916"
            legacy_run_dir.mkdir()
            legacy_manifest = {
                "run_id": "static-g0a-dev96-five-model-v8-20260916",
                "config_version": "factual-perturbation-zh-dual-distractor-proxy-v8",
                "config_sha256": perturbation.sha256_value(v8_config),
            }
            perturbation.write_json(legacy_run_dir / "run_manifest.json", legacy_manifest)
            legacy_probe = [{"model": "bailian/deepseek-v3.2", "terminal_status": "failed"}]
            perturbation.write_jsonl(
                legacy_run_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME,
                legacy_probe,
            )

            with self.assertRaisesRegex(
                ValueError,
                "target_run_already_prepared:static-g0a-dev96-five-model-v8-20260916",
            ):
                perturbation.prepare_static_proxy_run(
                    v10_config,
                    root,
                    root / "unused-static-freeze-manifest.json",
                    legacy_run_dir,
                    "static-g0a-dev96-five-model-v8-20260916",
                )
            self.assertEqual(
                perturbation.read_json(legacy_run_dir / "run_manifest.json"),
                legacy_manifest,
            )
            self.assertEqual(
                perturbation.read_jsonl(
                    legacy_run_dir
                    / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME
                ),
                legacy_probe,
            )
            with patch.object(
                perturbation.ModelRouter, "request_json", autospec=True
            ) as request, self.assertRaisesRegex(
                ValueError, "config fingerprint does not match current config"
            ):
                perturbation.run_static_proxy_representative_probe(
                    v10_config, root, legacy_run_dir, root / ".env"
                )
            request.assert_not_called()

            artifact_only_dir = root / "reused-run-id"
            artifact_only_dir.mkdir()
            perturbation.write_jsonl(
                artifact_only_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME,
                legacy_probe,
            )
            with self.assertRaisesRegex(
                ValueError, "target_run_directory_not_empty:new-four-model-run"
            ):
                perturbation.prepare_static_proxy_run(
                    v10_config,
                    root,
                    root / "unused-static-freeze-manifest.json",
                    artifact_only_dir,
                    "new-four-model-run",
                )

    def test_static_proxy_v9_rejects_legacy_probe_checkpoint_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            run_dir = root / "static-g0a-dev96-four-model-v9"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                root,
                static_manifest_path,
                run_dir,
                "static-g0a-dev96-four-model-v9",
            )
            perturbation.write_jsonl(
                run_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME,
                [{
                    "probe_id": "legacy-v8-probe-id",
                    "model": "bailian/deepseek-v3.2",
                    "terminal_status": "failed",
                }],
            )
            identity_snapshot = static_proxy_route_runtime_identity_fixture(config)
            with patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ), patch.object(
                perturbation.ModelRouter, "request_json", autospec=True
            ) as request, self.assertRaisesRegex(
                ValueError, "checkpoint contains unknown or duplicate rows"
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, root, run_dir, root / ".env"
                )
            request.assert_not_called()

    def test_static_g0a_bridge_materializes_exact_development_four_model_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            rows, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            run_dir = project_root / "run"
            run_dir.mkdir()
            manifest = perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )
            self.assertEqual(manifest["selected_count"], 96)
            self.assertEqual(
                manifest["proxy_split_filter"]["source_split_counts"],
                {"development": 96, "validation": 32, "sealed": 32},
            )
            self.assertEqual(
                manifest["proxy_split_filter"]["validation_behavior_input_count"], 0
            )
            self.assertEqual(
                manifest["proxy_split_filter"]["sealed_behavior_input_count"], 0
            )
            self.assertFalse(manifest["pnt_eligible"])
            self.assertEqual(manifest["proxy_roster_version"], "static-proxy-four-model-v9")
            self.assertEqual(manifest["expected_model_count"], 4)
            self.assertEqual(manifest["expected_calls_per_model"], 960)
            self.assertEqual(manifest["expected_simulation_call_count"], 3840)
            self.assertEqual(manifest["model_aggregation"], "none")
            inputs = perturbation.read_jsonl(run_dir / "proxy_behavior_inputs.jsonl")
            self.assertEqual(len(inputs), 3840)
            self.assertEqual(
                {
                    model: sum(
                        row["model_spec"]["model"] == model for row in inputs
                    )
                    for model, _ in perturbation.STATIC_PROXY_MODEL_ROSTER
                },
                {model: 960 for model, _ in perturbation.STATIC_PROXY_MODEL_ROSTER},
            )
            self.assertEqual({row["split_assignment"] for row in inputs}, {"development"})
            first_development = next(
                row for row in rows if row["split_assignment"] == "development"
            )
            frozen_options = first_development["static_mcq"]["options"]
            frozen_option_ids = [row["option_id"] for row in frozen_options]
            source_inputs = [
                row for row in inputs
                if row["base_fact_id"] == first_development["base_fact_id"]
            ]
            self.assertEqual(len(source_inputs), 40)
            self.assertTrue(
                all(row["option_ids"] == frozen_option_ids for row in source_inputs)
            )
            self.assertTrue(
                all(list(row["options"]) == ["A", "B", "C"] for row in source_inputs)
            )
            self.assertEqual(
                manifest["prebuilt_proxy_behavior_inputs"]["sha256"],
                hashlib.sha256((run_dir / "proxy_behavior_inputs.jsonl").read_bytes()).hexdigest(),
            )

    def test_static_proxy_representative_probe_uses_one_frozen_stimulus_for_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            config["execution"]["max_workers"] = 1
            run_dir = project_root / "run"
            run_dir.mkdir()
            run_manifest = perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )
            routed = []

            def fake_request(_router, stage, item_id, spec, prompt, validator):
                self.assertEqual(stage, "proxy_representative_probe")
                self.assertEqual(spec["max_retries"], 0)
                choice = "A" if spec["model"] == "llama3.1:8b" else "B"
                parsed = {"choice": choice}
                validator(parsed)
                routed.append((item_id, spec["model"], prompt))
                return perturbation.ModelResult(
                    "completed",
                    parsed,
                    json.dumps(parsed, separators=(",", ":")),
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            identity_snapshot = static_proxy_route_runtime_identity_fixture(config)
            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_request,
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ):
                probe_manifest = perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )

            self.assertEqual(probe_manifest["status"], "passed")
            self.assertTrue(probe_manifest["probe_passed"])
            self.assertTrue(probe_manifest["full_run_authorized"])
            self.assertEqual(probe_manifest["requested_model_count"], 4)
            self.assertEqual(probe_manifest["completed_model_count"], 4)
            self.assertEqual(len(routed), 4)
            self.assertEqual(
                [model for _, model, _ in routed],
                [model for model, _ in perturbation.STATIC_PROXY_MODEL_ROSTER],
            )
            self.assertEqual(len({prompt for _, _, prompt in routed}), 1)
            self.assertEqual(
                probe_manifest["representative_selection"]["language"], "zh"
            )
            self.assertEqual(
                probe_manifest["representative_selection"]["variant"], "targeted"
            )
            self.assertEqual(
                probe_manifest["run_manifest_sha256"],
                hashlib.sha256((run_dir / "run_manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                probe_manifest["g0a_static_freeze_manifest_sha256"],
                run_manifest["g0a_static_freeze_manifest"]["sha256"],
            )
            self.assertEqual(
                probe_manifest["proxy_behavior_inputs_sha256"],
                run_manifest["prebuilt_proxy_behavior_inputs"]["sha256"],
            )
            self.assertEqual(
                perturbation.read_jsonl(run_dir / "simulation_results.jsonl"), []
            )
            probe_results = perturbation.read_jsonl(
                run_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME
            )
            self.assertEqual(len(probe_results), 4)
            self.assertEqual(sum(row["correct"] is False for row in probe_results), 1)
            self.assertEqual(
                {row["static_stimulus_id"] for row in probe_results},
                {probe_manifest["representative_selection"]["static_stimulus_id"]},
            )
            inputs = perturbation._load_static_proxy_behavior_inputs(
                run_manifest, run_dir
            )
            perturbation._validate_static_proxy_representative_probe(
                config, run_manifest, inputs, run_dir
            )

    def test_static_proxy_representative_probe_failure_blocks_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            config["execution"]["max_workers"] = 1
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )

            def fake_request(_router, _stage, _item_id, spec, _prompt, _validator):
                if spec["model"] == "gemma3:12b":
                    return perturbation.ModelResult(
                        "failed", None, None, None, {}, 1, 1, []
                    )
                return perturbation.ModelResult(
                    "completed",
                    {"choice": "B"},
                    '{"choice":"B"}',
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            identity_snapshot = static_proxy_route_runtime_identity_fixture(config)
            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_request,
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ), self.assertRaisesRegex(
                RuntimeError, "static_proxy_representative_probe_blocked:gemma3:12b"
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )

            probe_manifest = perturbation.read_json(
                run_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME
            )
            self.assertEqual(probe_manifest["status"], "blocked")
            self.assertFalse(probe_manifest["probe_passed"])
            self.assertFalse(probe_manifest["full_run_authorized"])
            self.assertEqual(probe_manifest["failed_models"], ["gemma3:12b"])
            with self.assertRaisesRegex(
                ValueError, "representative probe is blocked or invalid"
            ):
                perturbation.run_preholdout(
                    config, project_root, run_dir, project_root / ".env"
                )
            self.assertEqual(
                perturbation.read_jsonl(run_dir / "simulation_results.jsonl"), []
            )

    def test_static_proxy_full_run_requires_representative_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            config["execution"]["max_workers"] = 1
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )
            with self.assertRaisesRegex(
                ValueError, "representative probe is required before full run"
            ):
                perturbation.run_preholdout(
                    config, project_root, run_dir, project_root / ".env"
                )

    def test_static_proxy_full_run_rejects_runtime_drift_before_behavior_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            config["execution"]["max_workers"] = 1
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )
            frozen_identity = static_proxy_route_runtime_identity_fixture(config)

            def fake_probe_request(
                _router, _stage, _item_id, spec, _prompt, validator
            ):
                parsed = {"choice": "B"}
                validator(parsed)
                return perturbation.ModelResult(
                    "completed",
                    parsed,
                    '{"choice":"B"}',
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_probe_request,
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=frozen_identity,
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )

            changed_identity = json.loads(json.dumps(frozen_identity))
            changed_record = changed_identity["records"][-1]
            changed_record["runtime"]["server_version"] = "fixture-ollama-2.0"
            changed_record["runtime"]["server_version_sha256"] = hashlib.sha256(
                b"fixture-ollama-2.0"
            ).hexdigest()
            identity_fields = dict(changed_record)
            identity_fields.pop("identity_sha256")
            changed_record["identity_sha256"] = perturbation.sha256_value(
                identity_fields
            )
            changed_identity["identity_sha256_by_model"][changed_record["model"]] = (
                changed_record["identity_sha256"]
            )
            changed_identity["route_runtime_identity_set_sha256"] = (
                perturbation.sha256_value(changed_identity["records"])
            )

            with patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=changed_identity,
            ), patch.object(
                perturbation.ModelRouter, "request_json", autospec=True
            ) as request, self.assertRaisesRegex(
                perturbation.StaticProxyIdentityError,
                "route_runtime_identity_drift_before_full_run",
            ):
                perturbation.run_preholdout(
                    config, project_root, run_dir, project_root / ".env"
                )
            request.assert_not_called()

    def test_static_proxy_representative_probe_rejects_changed_prepared_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )
            input_path = run_dir / "proxy_behavior_inputs.jsonl"
            inputs = perturbation.read_jsonl(input_path)
            inputs[0]["prompt"] = "tampered prompt"
            perturbation.write_jsonl(input_path, inputs)
            with self.assertRaisesRegex(
                ValueError, "behavior input binding is stale"
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )

    def test_static_proxy_representative_probe_self_rejects_tampered_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            config["execution"]["max_workers"] = 1
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )

            def fake_request(_router, _stage, _item_id, spec, _prompt, validator):
                parsed = {"choice": "B"}
                validator(parsed)
                return perturbation.ModelResult(
                    "completed",
                    parsed,
                    '{"choice":"B"}',
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            identity_snapshot = static_proxy_route_runtime_identity_fixture(config)
            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_request,
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )

            results_path = (
                run_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_RESULTS_FILENAME
            )
            results = perturbation.read_jsonl(results_path)
            results[0]["model"] = "tampered-model"
            perturbation.write_jsonl(results_path, results)
            with patch.object(
                perturbation.ModelRouter, "request_json", autospec=True
            ) as request, patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ), self.assertRaisesRegex(
                RuntimeError,
                "static_proxy_representative_probe_blocked:artifact_validation_failed",
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )
            request.assert_not_called()
            probe_manifest = perturbation.read_json(
                run_dir
                / perturbation.STATIC_PROXY_REPRESENTATIVE_PROBE_MANIFEST_FILENAME
            )
            self.assertEqual(probe_manifest["status"], "blocked")
            self.assertFalse(probe_manifest["full_run_authorized"])

    def test_runner_parser_exposes_static_proxy_representative_probe(self):
        args = runner.parser().parse_args(
            ["probe-static-proxy-run", "--run-id", "static-proxy-test"]
        )
        self.assertEqual(args.command, "probe-static-proxy-run")
        self.assertEqual(args.run_id, "static-proxy-test")

    def test_static_g0a_bridge_fails_closed_when_frozen_bundle_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )
            bundle_path = project_root / "frozen_static_bundle.jsonl"
            bundle_path.write_bytes(bundle_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "static G0A artifact SHA-256 mismatch"):
                perturbation.audit_run(run_dir)

    def test_static_g0a_proxy_runner_consumes_prebuilt_inputs_and_stays_non_pnt(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            _, static_manifest_path = write_static_proxy_freeze_fixture(project_root)
            config = runner.load_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
            )
            run_dir = project_root / "run"
            run_dir.mkdir()
            perturbation.prepare_static_proxy_run(
                config,
                project_root,
                static_manifest_path,
                run_dir,
                "static-proxy-test",
            )

            def fake_probe_request(
                _router, stage, _item_id, spec, _prompt, validator
            ):
                self.assertEqual(stage, "proxy_representative_probe")
                parsed = {"choice": "B"}
                validator(parsed)
                return perturbation.ModelResult(
                    "completed",
                    parsed,
                    '{"choice":"B"}',
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            identity_snapshot = static_proxy_route_runtime_identity_fixture(config)
            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_probe_request,
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ):
                perturbation.run_static_proxy_representative_probe(
                    config, project_root, run_dir, project_root / ".env"
                )
            routed_prompts = []

            def fake_request(_router, stage, _item_id, spec, prompt, validator):
                self.assertEqual(stage, "proxy_behavior")
                parsed = {"choice": "B"}
                validator(parsed)
                routed_prompts.append(prompt)
                return perturbation.ModelResult(
                    "completed", parsed, '{"choice":"B"}', spec["model"], {}, 1, 1, []
                )

            def one_write_stage(items, output_path, _key, worker, _workers):
                results = [worker(item) for item in items]
                perturbation.write_jsonl(output_path, results)
                return results

            with patch.object(
                perturbation.ModelRouter, "request_json", autospec=True, side_effect=fake_request
            ), patch.object(
                perturbation,
                "_static_proxy_route_runtime_identity_snapshot",
                return_value=identity_snapshot,
            ), patch.object(perturbation, "stage_run", side_effect=one_write_stage):
                summary = perturbation.run_preholdout(
                    config, project_root, run_dir, project_root / ".env"
                )
            self.assertEqual(summary["expected_simulation_call_count"], 3840)
            self.assertEqual(summary["completed_simulation_call_count"], 3840)
            self.assertTrue(summary["proxy_screen_complete"])
            self.assertFalse(summary["gate_passed"])
            self.assertFalse(summary["pnt_eligible"])
            self.assertEqual(summary["validation_behavior_result_count"], 0)
            self.assertEqual(summary["sealed_behavior_result_count"], 0)
            self.assertEqual(
                summary["completed_call_count_by_model"],
                {model: 960 for model, _ in perturbation.STATIC_PROXY_MODEL_ROSTER},
            )
            self.assertEqual(len(routed_prompts), 3840)
            self.assertIn("A. Wrong 0A\nB. Gold 0\nC. Wrong 0B", routed_prompts[0])

    def test_static_proxy_route_runtime_snapshot_hashes_routes_without_leaking_them(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
        )
        base_urls = {
            "aliyun": "https://dashscope.example/compatible-mode/v1",
            "test_gateway": "https://gateway.example/v1",
            "ollama_local": "http://127.0.0.1:11434/v1",
        }
        fixture = static_proxy_route_runtime_identity_fixture(config)
        ollama_runtime = {
            row["model"]: row["runtime"]
            for row in fixture["records"]
            if row["provider_profile"] == "ollama_local"
        }
        with tempfile.TemporaryDirectory() as directory:
            router_instance = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )

            def fake_profile(spec):
                profile = config["provider_profiles"][spec["provider_profile"]]
                return profile, "secret-api-key", base_urls[spec["provider_profile"]]

            with patch.object(
                router_instance, "_profile", side_effect=fake_profile
            ), patch.object(
                perturbation,
                "_ollama_runtime_identity",
                side_effect=lambda _base, model, _timeout: ollama_runtime[model],
            ):
                snapshot = perturbation._static_proxy_route_runtime_identity_snapshot(
                    router_instance,
                    config["model_roles"]["simulation"]["models"],
                )

        serialized = perturbation.canonical_json(snapshot)
        self.assertNotIn("dashscope.example", serialized)
        self.assertNotIn("127.0.0.1", serialized)
        self.assertNotIn("secret-api-key", serialized)
        self.assertEqual(snapshot["record_count"], 4)
        self.assertTrue(
            all(
                len(row["base_url_sha256"]) == 64
                and len(row["request_route_sha256"]) == 64
                for row in snapshot["records"]
            )
        )
        self.assertEqual(
            [
                row["runtime"]["kind"]
                for row in snapshot["records"]
                if row["provider_profile"] == "ollama_local"
            ],
            ["ollama", "ollama"],
        )

    def test_ollama_runtime_identity_records_digest_version_and_unknown_template(self):
        class FakeResponse:
            def __init__(self, body):
                self.body = body

            def raise_for_status(self):
                return None

            def json(self):
                return self.body

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, url):
                if url.endswith("/api/version"):
                    return FakeResponse({"version": "0.12.3"})
                if url.endswith("/api/tags"):
                    return FakeResponse({
                        "models": [{
                            "name": "gemma3:12b",
                            "model": "gemma3:12b",
                            "digest": "sha256:fixture-digest",
                        }]
                    })
                self.fail(f"unexpected GET {url}")

            def post(self, url, json):
                self.assertEqual(json["model"], "gemma3:12b")
                self.assertTrue(url.endswith("/api/show"))
                return FakeResponse({})

        fake_client = FakeClient()
        fake_client.fail = self.fail
        fake_client.assertEqual = self.assertEqual
        fake_client.assertTrue = self.assertTrue
        with patch.object(perturbation.httpx, "Client", return_value=fake_client):
            identity = perturbation._ollama_runtime_identity(
                "http://127.0.0.1:11434/v1", "gemma3:12b", 5
            )
        self.assertEqual(identity["server_version"], "0.12.3")
        self.assertEqual(
            identity["installed_model_digest"], "sha256:fixture-digest"
        )
        self.assertEqual(identity["template"]["status"], "unknown")
        self.assertEqual(
            identity["template"]["limitation"],
            "ollama_api_show_template_missing",
        )

    def test_static_proxy_response_model_drift_fails_without_retry(self):
        self.assertTrue(
            perturbation._static_proxy_response_model_identity(
                "bailian/deepseek-v4-flash-0731",
                "deepseek",
                "deepseek-v4-flash-0731",
            )["compatible"]
        )
        self.assertTrue(
            perturbation._static_proxy_response_model_identity(
                "gemini-3.7-flash", "test_gateway", "models/gemini-3.7-flash"
            )["compatible"]
        )
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 3},
            "provider_profiles": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            router_instance = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            with patch.object(
                router_instance,
                "_one_call",
                return_value=('{"choice":"A"}', "different-model", {}),
            ) as one_call:
                result = router_instance.request_json(
                    "proxy_behavior",
                    "item",
                    {
                        "provider_profile": "aliyun",
                        "model": "qwen3.6-27b",
                        "max_retries": 3,
                        "require_response_model_identity": True,
                    },
                    "prompt",
                    lambda value: perturbation.validate_choice(value, ("A", "B", "C")),
                )
        self.assertEqual(result.terminal_status, "failed")
        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.response_model, "different-model")
        self.assertEqual(one_call.call_count, 1)
        self.assertEqual(result.attempts[0]["error_type"], "StaticProxyIdentityError")

    def test_static_proxy_route_guard_blocks_changed_route_before_model_call(self):
        config = runner.load_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "factual_perturbation_zh_dual_distractor_proxy_v9.json"
        )
        base_urls = {
            "aliyun": "https://route-a.example/v1",
            "test_gateway": "https://gateway.example/v1",
            "ollama_local": "http://127.0.0.1:11434/v1",
        }
        fixture = static_proxy_route_runtime_identity_fixture(config)
        ollama_runtime = {
            row["model"]: row["runtime"]
            for row in fixture["records"]
            if row["provider_profile"] == "ollama_local"
        }
        with tempfile.TemporaryDirectory() as directory:
            router_instance = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )

            def fake_profile(spec):
                profile = config["provider_profiles"][spec["provider_profile"]]
                return profile, "secret", base_urls[spec["provider_profile"]]

            with patch.object(
                router_instance, "_profile", side_effect=fake_profile
            ), patch.object(
                perturbation,
                "_ollama_runtime_identity",
                side_effect=lambda _base, model, _timeout: ollama_runtime[model],
            ):
                snapshot = perturbation._static_proxy_route_runtime_identity_snapshot(
                    router_instance,
                    config["model_roles"]["simulation"]["models"],
                )
                router_instance.static_proxy_runtime_identity_guard = (
                    perturbation._StaticProxyRuntimeIdentityGuard(
                        router_instance, snapshot, perturbation.sha256_value(config)
                    )
                )
                base_urls["aliyun"] = "https://route-b.example/v1"
                with patch.object(router_instance, "_one_call") as one_call:
                    result = router_instance.request_json(
                        "proxy_behavior",
                        "item",
                        {
                            "provider_profile": "aliyun",
                            "model": "qwen3.6-27b",
                            "require_response_model_identity": True,
                        },
                        "prompt",
                        lambda value: None,
                    )
        self.assertEqual(result.terminal_status, "failed")
        self.assertEqual(result.attempt_count, 1)
        one_call.assert_not_called()

    def test_router_allows_unauthenticated_loopback_profile(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"local": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "http://127.0.0.1:11434/v1",
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            profile, key, base = router._profile(
                {"provider_profile": "local", "model": "gemma3:12b"}
            )
        self.assertEqual(profile["authentication"], "none")
        self.assertEqual(key, "local-no-auth")
        self.assertEqual(base, "http://127.0.0.1:11434/v1")

    def test_router_rejects_unauthenticated_remote_profile(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"remote": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "https://example.com/v1",
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            with self.assertRaisesRegex(RuntimeError, "unauthenticated_provider_requires_loopback"):
                router._profile({"provider_profile": "remote", "model": "m"})

    def test_router_allows_model_specific_retry_limit(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 3},
            "provider_profiles": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            with patch.object(router, "_one_call", side_effect=RuntimeError("boom")), patch.object(
                perturbation.time, "sleep"
            ):
                result = router.request_json(
                    "stage",
                    "item",
                    {"provider_profile": "p", "model": "m", "max_retries": 1},
                    "prompt",
                    lambda value: None,
                )
        self.assertEqual(result.terminal_status, "failed")
        self.assertEqual(result.attempt_count, 2)

    def test_router_checks_exact_response_identity_before_parse_or_retry(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 3},
            "provider_profiles": {},
        }
        cases = (
            (None, "ResponseModelMissing"),
            ("wrong-model", "ResponseModelMismatch"),
        )
        for response_model, error_type in cases:
            with self.subTest(response_model=response_model):
                with tempfile.TemporaryDirectory() as directory:
                    router = perturbation.ModelRouter(
                        config,
                        Path(directory) / ".env",
                        Path(directory) / "events.jsonl",
                    )
                    with patch.object(
                        router,
                        "_one_call",
                        return_value=("not-json", response_model, {}),
                    ) as one_call, patch.object(perturbation.time, "sleep"):
                        result = router.request_json(
                            "stage",
                            "item",
                            {
                                "provider_profile": "p",
                                "model": "requested-model",
                                "expected_response_model": "expected-model",
                            },
                            "prompt",
                            lambda value: None,
                        )
                self.assertEqual(result.terminal_status, "failed")
                self.assertEqual(result.attempt_count, 1)
                self.assertEqual(result.attempts[0]["error_type"], error_type)
                one_call.assert_called_once()

    def test_router_passes_json_mode_to_openai_compatible_provider(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"local": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "http://localhost:11434/v1",
                "json_mode": True,
            }},
        }
        captured = {}

        class FakeCompletions:
            @staticmethod
            def create(**request):
                captured.update(request)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"choice":"A"}'))],
                    model="m",
                    usage=None,
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            perturbation, "OpenAI", return_value=fake_client
        ) as client_factory:
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            text, response_model, _ = router._one_call(
                {"provider_profile": "local", "model": "m", "timeout_seconds": 7}, "prompt"
            )
        self.assertEqual(text, '{"choice":"A"}')
        self.assertEqual(response_model, "m")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(client_factory.call_args.kwargs["timeout"], 7.0)

    def test_router_passes_strict_mcq_schema_and_disables_qwen_thinking(self):
        config = {
            "execution": {"timeout_seconds": 1, "max_retries": 0},
            "provider_profiles": {"local": {
                "protocol": "openai_compatible",
                "authentication": "none",
                "base_url": "http://localhost:11434/v1",
                "json_mode": True,
            }},
        }
        captured = {}

        class FakeCompletions:
            @staticmethod
            def create(**request):
                captured.update(request)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"choice":"A"}'))],
                    model="qwen3-8b",
                    usage=None,
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            perturbation, "OpenAI", return_value=fake_client
        ):
            router = perturbation.ModelRouter(
                config, Path(directory) / ".env", Path(directory) / "events.jsonl"
            )
            text, response_model, _ = router._one_call(
                {
                    "provider_profile": "local",
                    "model": "qwen3-8b",
                    "disable_thinking": True,
                    "strict_mcq_choice_schema": True,
                },
                "prompt",
            )

        self.assertEqual(text, '{"choice":"A"}')
        self.assertEqual(response_model, "qwen3-8b")
        self.assertEqual(
            captured["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(
            captured["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "mcq_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": ["A", "B", "C"],
                            }
                        },
                        "required": ["choice"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def test_translation_schema_requires_exact_distractor_coverage(self):
        value = {
            "prompt_zh": "法国的首都是",
            "answer_zh": "巴黎",
            "answer_aliases_zh": ["巴黎"],
            "distractors": [{"distractor_id": "d1", "text_zh": "柏林"}],
        }
        perturbation.validate_translation(value, ["d1"])
        with self.assertRaisesRegex(ValueError, "distractor_translation_id_mismatch"):
            perturbation.validate_translation(value, ["d1", "d2"])

    def test_translation_repair_prompt_preserves_completion_boundary(self):
        prompt = perturbation._translation_repair_prompt(
            {
                "prompt_en": "The term for this fallacy is",
                "answer_en": "appeal to authority",
                "answer_aliases_en": ["appeal to authority"],
                "distractors": [
                    {"distractor_id": "d1", "text_en": "straw man"}
                ],
                "replacement_generation_requests": [
                    {"replaces_source_distractor_id": "source-rejected-d2"}
                ],
            },
            {
                "prompt_zh": "该谬误术语是",
                "answer_zh": "诉诸权威",
                "answer_aliases_zh": ["诉诸权威"],
                "distractors": [{"distractor_id": "d1", "text_zh": "稻草人"}],
            },
            {"decision": "reject", "issues": ["completion boundary"]},
        )
        self.assertIn(
            "prompt_zh and answer_zh must form one natural, complete factual statement",
            prompt,
        )
        self.assertIn("never move such content into answer_zh", prompt)
        self.assertIn("incomplete completion stem", prompt)
        self.assertIn("retain that term in English in prompt_zh", prompt)
        self.assertNotIn("source-rejected-d2", prompt)
        self.assertEqual(
            perturbation.TRANSLATION_REPAIR_PROMPT_VERSION,
            "factual-translation-repair-zh-v2",
        )

    def test_static_g0a_advisory_reviews_do_not_block_two_context_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = {
                "config_version": "static-advisory-fixture-v1",
                "g0a_static_only": True,
                "provider_profiles": {},
                "execution": {
                    "max_workers": 1,
                    "timeout_seconds": 1,
                    "max_retries": 0,
                },
                "model_roles": {
                    "translation": {
                        "primary": {"model": "translator", "provider_profile": "p"},
                        "reviewer": {"model": "reviewer", "provider_profile": "p"},
                        "repair_rejected": False,
                    },
                    "distractor_validation": {
                        "primary_judge": {"model": "judge", "provider_profile": "p"}
                    },
                    "distractor_generation": {
                        "enabled": True,
                        "primary": {"model": "generator", "provider_profile": "p"},
                        "target_accepted_per_source": 2,
                        "backup_candidates": 1,
                    },
                    "perturbation_generation": {
                        "primary": {"model": "context", "provider_profile": "p"},
                        "manipulation_family": (
                            perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                        ),
                        "target_distractors_per_source": 2,
                        "candidates_per_distractor": 1,
                        "strength_levels": [],
                        "matched_neutral": True,
                    },
                    "perturbation_validation": {
                        "primary_judge": {
                            "model": "context-reviewer",
                            "provider_profile": "p",
                        },
                        "hard_gate": False,
                        "required_checks": list(
                            perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS
                        ),
                    },
                    "simulation": {"models": []},
                },
            }
            item = {
                "base_fact_id": "fact-1",
                "base_id": "fact-1",
                "source_id": "source-1",
                "source_dataset": "fixture",
                "answer_type": "entity",
                "prompt_en": "The capital is",
                "answer_en": "Paris",
                "answer_aliases_en": ["Paris"],
                "canonical_fact": "The capital is Paris.",
                "distractors": [
                    {
                        "distractor_id": "source-accepted-d1",
                        "text_en": "Berlin",
                        "source_audit_verdict": "accept",
                        "source_audit_record_sha256": "a" * 64,
                    }
                ],
                "replacement_generation_requests": [
                    {
                        "replaces_source_distractor_id": "source-rejected-d2",
                        "source_audit_verdict": "reject",
                    }
                ],
                "replacement_generation_required_count": 1,
                "g0a_static_only": True,
            }
            manifest = {
                "run_id": "static-advisory-fixture",
                "config_sha256": perturbation.sha256_value(config),
                "runtime_sha256": perturbation._runtime_sha256(),
                "paths_not_taken_enabled": False,
                "holdout_enabled": False,
                "static_generation_authorized": True,
                "records": [item],
                "selected_count": 1,
                "seed": 7,
                "manipulation_family": (
                    perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                ),
                "simulation_requires_codex_proxy_review": True,
                "codex_proxy_review": None,
                "simulation_models": [],
                "simulation_languages": ["en", "zh"],
                "simulation_variants": ["original", "neutral", "targeted"],
                "shared_simulation_variants": ["original"],
                "minimum_translation_acceptance": 1.0,
                "currency_cost_gate": 1,
            }
            perturbation.write_json(run_dir / "run_manifest.json", manifest)
            routed = []

            def fake_request(_router, stage, item_id, spec, prompt, validator):
                routed.append((stage, item_id, prompt))
                if stage == "translation":
                    parsed = {
                        "prompt_zh": "首都是",
                        "answer_zh": "巴黎",
                        "answer_aliases_zh": ["巴黎"],
                        "distractors": [
                            {"distractor_id": "source-accepted-d1", "text_zh": "柏林"}
                        ],
                    }
                elif stage == "translation_review":
                    parsed = {
                        "decision": "reject",
                        "issues": ["advisory completion concern"],
                        "checks": {
                            key: key != "naturalness"
                            for key in perturbation.TRANSLATION_CHECKS
                        },
                    }
                elif stage == "distractor_validation":
                    parsed = {
                        "decision": "reject",
                        "issues": ["advisory plausibility concern"],
                        "checks": {
                            key: key != "plausible"
                            for key in perturbation.DISTRACTOR_CHECKS
                        },
                    }
                elif stage == "distractor_generation":
                    parsed = {
                        "distractors": [
                            {"text_en": "Rome", "text_zh": "罗马"},
                            {"text_en": "Madrid", "text_zh": "马德里"},
                        ]
                    }
                elif stage == "distractor_repair_validation":
                    accepted = item_id.endswith("_1")
                    parsed = {
                        "decision": "accept" if accepted else "reject",
                        "issues": [] if accepted else ["backup rejected"],
                        "checks": {
                            key: accepted for key in perturbation.DISTRACTOR_CHECKS
                        },
                    }
                elif stage == "perturbation_generation":
                    parsed = {
                        "candidates": [
                            {
                                "english_context": f"Target {item_id}",
                                "chinese_context": f"目标 {item_id}",
                                "neutral_english_context": f"Neutral {item_id}",
                                "neutral_chinese_context": f"中性 {item_id}",
                            }
                        ]
                    }
                elif stage == "perturbation_validation":
                    parsed = {
                        "decision": "reject",
                        "issues": ["advisory context concern"],
                        "checks": {
                            key: key != "naturalness"
                            for key in perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS
                        },
                    }
                else:
                    self.fail(f"unexpected model stage: {stage}")
                validator(parsed)
                return perturbation.ModelResult(
                    "completed",
                    parsed,
                    json.dumps(parsed, ensure_ascii=False),
                    spec["model"],
                    {},
                    1,
                    1,
                    [],
                )

            with patch.object(
                perturbation.ModelRouter,
                "request_json",
                autospec=True,
                side_effect=fake_request,
            ):
                perturbation.run_preholdout(
                    config, run_dir, run_dir, run_dir / ".env"
                )

            generation = perturbation.read_jsonl(
                run_dir / "distractor_generations.jsonl"
            )
            self.assertEqual(generation[0]["requested_count"], 2)
            context_targets = {
                row["item_id"]
                for row in perturbation.read_jsonl(
                    run_dir / "perturbation_generations.jsonl"
                )
            }
            self.assertEqual(
                context_targets,
                {"source-accepted-d1", "source-1_generated_wrong_1"},
            )
            self.assertEqual(
                perturbation.read_jsonl(
                    run_dir / "verified_original_distractors.jsonl"
                )[0]["parsed_response"]["decision"],
                "reject",
            )
            self.assertEqual(
                perturbation.read_jsonl(run_dir / "translation_reviews.jsonl")[
                    0
                ]["parsed_response"]["decision"],
                "reject",
            )
            self.assertFalse(
                any("source-rejected-d2" in prompt for _, _, prompt in routed)
            )

    def test_generated_distractor_schema_requires_exact_unique_bilingual_rows(self):
        value = {"distractors": [
            {"text_en": "Berlin", "text_zh": "柏林"},
            {"text_en": "Rome", "text_zh": "罗马"},
        ]}
        perturbation.validate_generated_distractors(value, 2)
        value["distractors"][1] = dict(value["distractors"][0])
        with self.assertRaisesRegex(ValueError, "duplicate_generated_distractors"):
            perturbation.validate_generated_distractors(value, 2)

    def test_distractor_review_requires_prompt_fit_and_single_answer(self):
        checks = {key: True for key in perturbation.DISTRACTOR_CHECKS}
        perturbation.validate_decision(
            {"decision": "accept", "issues": [], "checks": checks},
            perturbation.DISTRACTOR_CHECKS,
        )
        checks.pop("prompt_completion_fit")
        with self.assertRaisesRegex(ValueError, "invalid_checks"):
            perturbation.validate_decision(
                {"decision": "accept", "issues": [], "checks": checks},
                perturbation.DISTRACTOR_CHECKS,
            )

    def test_soft_auto_judge_review_packet_includes_rejected_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "soft-review",
                "perturbation_auto_judge_hard_gate": False,
                "records": [{
                    "source_id": "s1", "source_dataset": "toy", "answer_type": "city",
                    "prompt_en": "The capital is", "answer_en": "Paris",
                    "canonical_fact": "The capital is Paris.", "relation_raw": "capital",
                    "relation_normalized": "entity_location",
                }],
            })
            perturbation.write_jsonl(run_dir / "translations.jsonl", [{
                "source_id": "s1", "terminal_status": "completed",
                "parsed_response": {"prompt_zh": "首都是", "answer_zh": "巴黎"},
            }])
            accepted = {
                "terminal_status": "completed",
                "parsed_response": {"decision": "accept", "checks": {"valid": True}},
            }
            perturbation.write_jsonl(run_dir / "verified_distractors.jsonl", [
                {"item_id": "d1", "source_id": "s1", "distractor_en": "Berlin", "distractor_zh": "柏林", **accepted},
                {"item_id": "d2", "source_id": "s1", "distractor_en": "Rome", "distractor_zh": "罗马", **accepted},
            ])
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
                "candidate": {"strength": "l2_lexical"}, "terminal_status": "completed",
                "parsed_response": {"decision": "reject", "checks": {"valid": False}},
            }])
            summary = perturbation.export_codex_review_packet(run_dir)
            packet = perturbation.read_json(run_dir / "codex_proxy_review_input_v5.json")
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(packet["records"][0]["candidates"][0]["candidate_id"], "c1")

    def test_choice_layout_is_deterministic_and_balanced_per_language(self):
        first = perturbation._choice_layout("s1", "d1", "en", "Paris", "Berlin")
        second = perturbation._choice_layout("s1", "d1", "en", "Paris", "Berlin")
        self.assertEqual(first, second)
        self.assertEqual(set(first[0].values()), {"Paris", "Berlin"})
        self.assertIn(first[1], {"A", "B"})

    def test_multi_choice_layout_is_deterministic_and_tracks_target(self):
        first = perturbation._multi_choice_layout(
            "s1", "d1", "Paris", [("d1", "Berlin"), ("d2", "Rome")], 3
        )
        second = perturbation._multi_choice_layout(
            "s1", "d1", "Paris", [("d1", "Berlin"), ("d2", "Rome")], 3
        )
        self.assertEqual(first, second)
        options, correct_choice, target_choice, option_ids = first
        self.assertEqual(set(options.values()), {"Paris", "Berlin", "Rome"})
        self.assertNotEqual(correct_choice, target_choice)
        self.assertEqual(option_ids[list(options).index(target_choice)], "d1")
        perturbation.validate_choice({"choice": "C"}, tuple(options))

    def test_strength_candidates_require_ordered_levels_and_matched_neutral(self):
        candidates = {
            "candidates": [
                {
                    "strength": level,
                    "english_context": f"target {level}",
                    "chinese_context": f"目标 {level}",
                    "neutral_english_context": f"neutral {level}",
                    "neutral_chinese_context": f"中性 {level}",
                }
                for level in ("l1", "l2")
            ]
        }
        perturbation.validate_perturbations(candidates, 2, ("l1", "l2"), True)
        candidates["candidates"].reverse()
        with self.assertRaisesRegex(ValueError, "invalid_strength_levels"):
            perturbation.validate_perturbations(candidates, 2, ("l1", "l2"), True)

    def test_single_matched_pair_does_not_require_synthetic_strength_label(self):
        perturbation.validate_perturbations(
            {
                "candidates": [
                    {
                        "english_context": "Targeted assertion.",
                        "chinese_context": "目标断言。",
                        "neutral_english_context": "Matched neutral assertion.",
                        "neutral_chinese_context": "匹配的中性断言。",
                    }
                ]
            },
            1,
            (),
            True,
        )

    def test_stage_resume_does_not_repeat_completed_items(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            calls = []

            def worker(item):
                calls.append(item["item_id"])
                return {"item_id": item["item_id"], "terminal_status": "completed"}

            items = [{"item_id": "a"}, {"item_id": "b"}]
            perturbation.stage_run(items, output, "item_id", worker, 1)
            first_bytes = output.read_bytes()
            perturbation.stage_run(items, output, "item_id", worker, 1)
            self.assertEqual(calls, ["a", "b"])
            self.assertEqual(first_bytes, output.read_bytes())

    def test_stage_can_freeze_terminal_failures_for_proxy_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            failed = {
                "item_id": "a", "terminal_status": "failed",
                "attempt_count": 8, "attempts": [{"attempt": index} for index in range(8)],
            }
            perturbation.write_jsonl(output, [failed])
            calls = []

            def worker(item):
                calls.append(item["item_id"])
                return {"item_id": item["item_id"], "terminal_status": "completed"}

            rows = perturbation.stage_run(
                [{"item_id": "a"}], output, "item_id", worker, 1,
                retry_terminal_failures=False,
            )
            self.assertEqual(calls, [])
            self.assertEqual(rows, [failed])

    def test_serial_stage_resume_preserves_failed_attempt_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            perturbation.write_jsonl(output, [{
                "item_id": "a", "terminal_status": "failed", "attempt_count": 2,
                "attempts": [{"attempt": 1}, {"attempt": 2}],
            }])

            def worker(item):
                return {
                    "item_id": item["item_id"], "terminal_status": "completed",
                    "attempt_count": 1, "attempts": [{"attempt": 1, "status": "ok"}],
                }

            rows = perturbation.stage_run([{"item_id": "a"}], output, "item_id", worker, 1)
            self.assertEqual(rows[0]["attempt_count"], 3)
            self.assertEqual(len(rows[0]["attempts"]), 3)
            self.assertEqual(rows[0]["resume_run_count"], 1)

    def test_stratified_sample_is_deterministic_and_covers_datasets(self):
        rows = []
        for dataset in ("a", "b", "c", "d", "e"):
            for index in range(4):
                rows.append({
                    "source_id": f"{dataset}-{index}", "source_dataset": dataset,
                    "prompt_quality_tier": "strict" if index % 2 else "fallback",
                    "answer_type": f"type-{index}",
                })
        first = perturbation.stratified_sample(rows, 10, 7)
        second = perturbation.stratified_sample(rows, 10, 7)
        self.assertEqual(first, second)
        self.assertEqual({row["source_dataset"] for row in first}, {"a", "b", "c", "d", "e"})

    def test_target_distractor_cap_keeps_full_foil_pool_unchanged(self):
        accepted = [
            {"source_id": "s1", "item_id": "d1", "distractor_source": "original_wrong_option"},
            {"source_id": "s1", "item_id": "d2", "distractor_source": "original_wrong_option"},
            {"source_id": "s1", "item_id": "d3", "distractor_source": "generated_replacement"},
        ]
        targeted = perturbation._select_target_distractors(accepted, 1, 7)
        self.assertEqual(len(targeted), 1)
        self.assertEqual(len(accepted), 3)
        self.assertIn(targeted[0]["item_id"], {"d1", "d2"})
        self.assertGreaterEqual(
            len([row for row in accepted if row["item_id"] != targeted[0]["item_id"]]),
            2,
        )

    def test_redacted_event_contains_no_prompt_or_endpoint(self):
        config = {"execution": {"timeout_seconds": 1, "max_retries": 0}, "provider_profiles": {}}
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            router = perturbation.ModelRouter(config, Path(directory) / ".env", event_path)
            result = perturbation.ModelResult("failed", None, None, None, {}, 1, 1, [{"attempt": 1, "status": "failed"}])
            router._event("stage", "item", {"provider_profile": "p", "model": "m"}, result)
            event = json.loads(event_path.read_text())
            self.assertFalse(event["credentials_or_endpoints_included"])
            self.assertNotIn("prompt", event)
            self.assertNotIn("base_url", event)

    def test_simulation_inputs_share_baselines_across_candidates(self):
        item_by_id = {
            "s1": {"source_id": "s1", "prompt_en": "Capital of France:", "answer_en": "Paris"}
        }
        translation_by_id = {
            "s1": {"parsed_response": {"prompt_zh": "法国的首都是：", "answer_zh": "巴黎"}}
        }
        distractor_by_id = {
            "d1": {"item_id": "d1", "distractor_en": "Berlin", "distractor_zh": "柏林"}
        }
        perturbations = [
            {
                "candidate_id": f"c{index}", "source_id": "s1", "distractor_id": "d1",
                "candidate": {"english_context": f"context {index}", "chinese_context": f"语境 {index}"},
            }
            for index in (1, 2)
        ]
        rows = perturbation._build_simulation_inputs(
            perturbations, item_by_id, translation_by_id, distractor_by_id,
            [{"model": "m1", "provider_profile": "p1"}],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["simulation_id"] for row in rows}), 8)
        shared = [row for row in rows if row["variant"] in {"original", "neutral"}]
        targeted = [row for row in rows if row["variant"] == "targeted"]
        self.assertEqual(len(shared), 4)
        self.assertEqual(len(targeted), 4)
        self.assertTrue(all(row["candidate_id"] is None for row in shared))
        self.assertTrue(all(row["candidate_ids"] == ["c1", "c2"] for row in shared))
        self.assertEqual({row["candidate_id"] for row in targeted}, {"c1", "c2"})

    def test_matched_neutral_is_candidate_specific(self):
        item_by_id = {"s1": {"source_id": "s1", "prompt_en": "Capital:", "answer_en": "Paris"}}
        translation_by_id = {"s1": {"parsed_response": {"prompt_zh": "首都：", "answer_zh": "巴黎"}}}
        distractor_by_id = {"d1": {"item_id": "d1", "distractor_en": "Berlin", "distractor_zh": "柏林"}}
        perturbations = [
            {
                "candidate_id": f"c{index}", "source_id": "s1", "distractor_id": "d1",
                "candidate": {
                    "english_context": f"target {index}", "chinese_context": f"目标 {index}",
                    "neutral_english_context": f"neutral {index}",
                    "neutral_chinese_context": f"中性 {index}",
                },
            }
            for index in (1, 2)
        ]
        rows = perturbation._build_simulation_inputs(
            perturbations, item_by_id, translation_by_id, distractor_by_id,
            [{"model": "m1", "provider_profile": "p1"}],
            ("original",), "matched_per_candidate",
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual(len([row for row in rows if row["variant"] == "original"]), 2)
        self.assertEqual(len([row for row in rows if row["variant"] == "neutral"]), 4)
        self.assertTrue(all(row["candidate_id"] for row in rows if row["variant"] == "neutral"))

    def test_multi_option_simulation_includes_target_and_foil(self):
        item_by_id = {"s1": {"source_id": "s1", "prompt_en": "Capital:", "answer_en": "Paris"}}
        translation_by_id = {"s1": {"parsed_response": {"prompt_zh": "首都：", "answer_zh": "巴黎"}}}
        distractor_by_id = {
            "d1": {"item_id": "d1", "source_id": "s1", "distractor_en": "Berlin", "distractor_zh": "柏林"},
            "d2": {"item_id": "d2", "source_id": "s1", "distractor_en": "Rome", "distractor_zh": "罗马"},
        }
        candidates = [{
            "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
            "candidate": {
                "english_context": "target", "chinese_context": "目标",
                "neutral_english_context": "neutral", "neutral_chinese_context": "中性",
            },
        }]
        rows = perturbation._build_simulation_inputs(
            candidates, item_by_id, translation_by_id, distractor_by_id,
            [{"model": "m", "provider_profile": "p"}],
            ("original",), "matched_per_candidate", "multi_option", 3,
        )
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["option_count"] == 3 for row in rows))
        self.assertTrue(all(row["target_choice"] != row["correct_choice"] for row in rows))
        self.assertTrue(all(set(row["options"].values()) in ({"Paris", "Berlin", "Rome"}, {"巴黎", "柏林", "罗马"}) for row in rows))

    def test_simulation_ids_are_stable_and_baselines_ignore_candidate(self):
        original_a = perturbation._simulation_id("s", "d", "m", "zh", "original", "c1")
        original_b = perturbation._simulation_id("s", "d", "m", "zh", "original", "c2")
        neutral = perturbation._simulation_id("s", "d", "m", "zh", "neutral", "c1")
        targeted_a = perturbation._simulation_id("s", "d", "m", "zh", "targeted", "c1")
        targeted_b = perturbation._simulation_id("s", "d", "m", "zh", "targeted", "c2")
        self.assertEqual(original_a, original_b)
        self.assertNotEqual(original_a, neutral)
        self.assertNotEqual(targeted_a, targeted_b)

    def test_detects_duplicate_baseline_inconsistency(self):
        common = {
            "source_id": "s", "distractor_id": "d", "model": "m", "language": "zh",
            "variant": "original", "terminal_status": "completed",
        }
        rows = [
            {**common, "simulation_id": "one", "candidate_id": "c1", "choice": "A", "correct": True},
            {**common, "simulation_id": "two", "candidate_id": "c2", "choice": "B", "correct": False},
        ]
        inconsistencies = perturbation.detect_baseline_inconsistencies(rows)
        self.assertEqual(len(inconsistencies), 1)
        self.assertEqual(inconsistencies[0]["candidate_ids"], ["c1", "c2"])

    def test_three_arm_analysis_does_not_count_non_target_foil_as_target_flip(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "multi",
                "simulation_models": ["m"],
                "simulation_variants": list(perturbation.SIMULATION_VARIANTS),
                "shared_simulation_variants": ["original"],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1"],
                },
            })
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
                "terminal_status": "completed",
            }])
            rows = []
            for language in perturbation.SIMULATION_LANGUAGES:
                for variant in perturbation.SIMULATION_VARIANTS:
                    is_nontarget_error = language == "zh" and variant == "targeted"
                    rows.append({
                        "simulation_id": f"{language}-{variant}",
                        "candidate_id": None if variant == "original" else "c1",
                        "source_id": "s1", "distractor_id": "d1", "model": "m",
                        "language": language, "variant": variant,
                        "terminal_status": "completed",
                        "choice": "C" if is_nontarget_error else "A",
                        "correct": not is_nontarget_error,
                        "distractor_hit": False,
                    })
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", rows)
            result = perturbation.analyze_three_arm(run_dir)
            candidate = result["candidate_results"][0]
            self.assertEqual(candidate["targeted_zh_wrong_models"], ["m"])
            self.assertEqual(candidate["targeted_zh_flip_models"], [])
            self.assertEqual(candidate["targeted_zh_nontarget_wrong_models"], ["m"])
            self.assertEqual(candidate["strict_targeted_zh_flip_models"], [])
            self.assertFalse(candidate["paths_not_taken_candidate"])

    def test_three_arm_analysis_preserves_model_specific_strict_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "model-specific",
                "simulation_models": ["signal-model", "other-model"],
                "simulation_variants": list(perturbation.SIMULATION_VARIANTS),
                "shared_simulation_variants": ["original"],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1"],
                },
            })
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "distractor_id": "d1",
                "terminal_status": "completed",
            }])
            rows = []
            for model in ("signal-model", "other-model"):
                for language in perturbation.SIMULATION_LANGUAGES:
                    for variant in perturbation.SIMULATION_VARIANTS:
                        is_signal = (
                            model == "signal-model" and language == "zh" and variant == "targeted"
                        )
                        other_control_failure = (
                            model == "other-model" and language == "en" and variant == "targeted"
                        )
                        rows.append({
                            "simulation_id": f"{model}-{language}-{variant}",
                            "candidate_id": None if variant == "original" else "c1",
                            "source_id": "s1", "distractor_id": "d1", "model": model,
                            "language": language, "variant": variant,
                            "terminal_status": "completed",
                            "choice": "B" if is_signal or other_control_failure else "A",
                            "correct": not (is_signal or other_control_failure),
                            "distractor_hit": is_signal,
                        })
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", rows)
            result = perturbation.analyze_three_arm(run_dir)
            candidate = result["candidate_results"][0]
            self.assertEqual(candidate["strict_targeted_zh_flip_models"], ["signal-model"])
            self.assertTrue(candidate["paths_not_taken_candidate"])
            self.assertEqual(
                result["strict_signal_count_by_model"],
                {"signal-model": 1, "other-model": 0},
            )
            self.assertEqual(
                result["model_arm_metrics"]["signal-model"]["zh_targeted"],
                {
                    "call_count": 1,
                    "completed_count": 1,
                    "accuracy_completed": 0.0,
                    "distractor_hit_rate_completed": 1.0,
                },
            )

    def test_codex_proxy_allowlist_filters_model_accepted_perturbations(self):
        rows = [
            {
                "candidate_id": candidate_id,
                "terminal_status": "completed",
                "parsed_response": {
                    "decision": "reject" if candidate_id == "keep" else "accept",
                    "checks": {"valid": candidate_id != "keep"},
                },
            }
            for candidate_id in ("keep", "drop")
        ]
        manifest = {"codex_proxy_review": {"accepted_perturbation_ids": ["keep"]}}
        accepted = perturbation._accepted_perturbation_rows(rows, manifest)
        self.assertEqual([row["candidate_id"] for row in accepted], ["keep"])

    def test_explicit_proxy_rerun_filters_full_pool_to_development(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_run, review_path, config = self._write_explicit_proxy_fixture(
                project_root
            )
            target_run = project_root / "target"
            target_run.mkdir()
            manifest = perturbation.prepare_proxy_rerun(
                config,
                project_root,
                source_run,
                target_run,
                "target",
                review_path,
            )
            self.assertEqual(manifest["selected_count"], 1)
            self.assertEqual(
                [row["source_id"] for row in manifest["records"]], ["s-dev"]
            )
            self.assertEqual(
                manifest["codex_proxy_review"]["accepted_perturbation_ids"],
                ["dev-c1", "dev-c2"],
            )
            self.assertEqual(
                manifest["codex_proxy_review"]["accepted_after_split_filter"], 2
            )
            self.assertEqual(manifest["proxy_split_filter"]["allowed_splits"], [
                "development"
            ])
            self.assertEqual(manifest["proxy_split_filter"]["excluded_source_count"], 1)
            for name in (
                "translations.jsonl",
                "translation_reviews.jsonl",
                "verified_distractors.jsonl",
                "perturbation_generations.jsonl",
                "perturbations.jsonl",
            ):
                self.assertEqual(
                    {row["source_id"] for row in perturbation.read_jsonl(target_run / name)},
                    {"s-dev"},
                )

    def test_explicit_proxy_rerun_rejects_split_or_design_drift(self):
        cases = (
            (
                "validation split",
                lambda config: config["inputs"].update({
                    "allowed_splits": ["development", "validation"]
                }),
                "explicit_false_assertion_proxy_screen_requires_development_only",
            ),
            (
                "binary endpoint",
                lambda config: config["model_roles"]["simulation"].update({
                    "endpoint": "binary"
                }),
                "explicit_false_assertion_proxy_design_mismatch:simulation_endpoint",
            ),
            (
                "shared neutral",
                lambda config: config["model_roles"]["simulation"].update({
                    "neutral_context_mode": "generic_shared"
                }),
                "explicit_false_assertion_proxy_design_mismatch:neutral_context_mode",
            ),
            (
                "one distractor",
                lambda config: config["model_roles"][
                    "perturbation_generation"
                ].update({"target_distractors_per_source": 1}),
                (
                    "explicit_false_assertion_proxy_design_mismatch:"
                    "designated_distractors_per_base_fact"
                ),
            ),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                project_root = Path(directory)
                source_run, review_path, config = self._write_explicit_proxy_fixture(
                    project_root
                )
                mutate(config)
                target_run = project_root / "target"
                target_run.mkdir()
                with self.assertRaisesRegex(ValueError, expected_error):
                    perturbation.prepare_proxy_rerun(
                        config,
                        project_root,
                        source_run,
                        target_run,
                        "target",
                        review_path,
                    )

    def test_multi_option_proxy_rerun_excludes_sources_without_two_accepted_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_run = project_root / "source"
            target_run = project_root / "target"
            source_run.mkdir()
            target_run.mkdir()
            perturbation.write_json(source_run / "run_manifest.json", {
                "run_id": "source", "seed": 7, "config_sha256": "old",
                "records": [
                    {"source_id": "s1", "split_assignment": "development"},
                    {"source_id": "s2", "split_assignment": "development"},
                ],
            })
            source_perturbations = [
                {"candidate_id": "eligible", "source_id": "s1", "distractor_id": "d1"},
                {"candidate_id": "excluded", "source_id": "s2", "distractor_id": "d3"},
            ]
            perturbation.write_jsonl(source_run / "perturbations.jsonl", source_perturbations)
            accepted_review = {
                "terminal_status": "completed",
                "parsed_response": {"decision": "accept", "checks": {"valid": True}},
            }
            perturbation.write_jsonl(source_run / "verified_distractors.jsonl", [
                {"item_id": "d1", "source_id": "s1", **accepted_review},
                {"item_id": "d2", "source_id": "s1", **accepted_review},
                {"item_id": "d3", "source_id": "s2", **accepted_review},
            ])
            for name in ("translations.jsonl", "translation_reviews.jsonl", "perturbation_generations.jsonl"):
                perturbation.write_jsonl(source_run / name, [])
            review_path = source_run / "codex_proxy_review_v1.json"
            perturbation.write_json(review_path, {
                "run_id": "source",
                "review_version": "v1",
                "reviewer_type": "codex_proxy",
                "not_human_gold": True,
                "perturbation_reviews": [
                    {"item_id": "eligible", "decision": "accept"},
                    {"item_id": "excluded", "decision": "accept"},
                ],
            })
            config = {
                "config_version": "multi-v1",
                "inputs": {"allowed_splits": ["development"]},
                "model_roles": {"simulation": {
                    "endpoint": "multi_option", "max_options": 3,
                    "neutral_context_mode": "matched_per_candidate",
                    "models": [{"model": "m"}],
                }},
            }
            manifest = perturbation.prepare_proxy_rerun(
                config, project_root, source_run, target_run, "target", review_path,
            )
            self.assertEqual(
                manifest["codex_proxy_review"]["accepted_perturbation_ids"], ["eligible"]
            )
            self.assertEqual(manifest["endpoint_excluded_candidate_ids"], ["excluded"])
            self.assertEqual(
                manifest["endpoint_exclusions"][0]["reason"],
                "fewer_than_two_auto_accepted_distractors",
            )

    def test_frozen_proxy_rerun_keeps_generated_distractor_foils(self):
        accepted = {
            "terminal_status": "completed",
            "parsed_response": {"decision": "accept", "checks": {"valid": True}},
        }
        rows = [
            {"item_id": "source_d1", "source_id": "s1", **accepted},
            {
                "item_id": "generated_d2", "source_id": "s1",
                "distractor_source": "generated_replacement", **accepted,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified_distractors.jsonl"
            perturbation.write_jsonl(path, rows)
            frozen_rows = perturbation.read_jsonl(path)
        self.assertEqual(
            {row["item_id"] for row in frozen_rows},
            {"source_d1", "generated_d2"},
        )

    def test_simulation_rerun_reuses_frozen_candidates_without_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_run = project_root / "source"
            target_run = project_root / "target"
            source_run.mkdir()
            target_run.mkdir()
            source_manifest = {
                "run_id": "source",
                "config_sha256": "old",
                "records": [{"source_id": "s1"}],
                "simulation_endpoint": "multi_option",
                "simulation_max_options": 3,
                "neutral_context_mode": "matched_per_candidate",
                "shared_simulation_variants": ["original"],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1"],
                },
            }
            perturbation.write_json(source_run / "run_manifest.json", source_manifest)
            for name in (
                "translations.jsonl", "translation_reviews.jsonl",
                "verified_distractors.jsonl", "perturbation_generations.jsonl",
            ):
                perturbation.write_jsonl(source_run / name, [{"source_id": "s1"}])
            perturbation.write_jsonl(source_run / "perturbations.jsonl", [{
                "candidate_id": "c1", "source_id": "s1", "terminal_status": "completed",
            }])
            perturbation.write_jsonl(source_run / "simulation_results.jsonl", [{
                "simulation_id": "old", "model": "old-model",
            }])
            config = {
                "config_version": "local-v1",
                "model_roles": {"simulation": {
                    "endpoint": "multi_option",
                    "max_options": 3,
                    "neutral_context_mode": "matched_per_candidate",
                    "models": [
                        {"model": "gemma3:12b", "provider_profile": "ollama_local"},
                        {"model": "llama3.1:8b", "provider_profile": "ollama_local"},
                    ],
                }},
            }
            manifest = perturbation.prepare_simulation_rerun(
                config, project_root, source_run, target_run, "target"
            )
            self.assertEqual(manifest["resumed_from_run_id"], "source")
            self.assertEqual(manifest["simulation_models"], ["gemma3:12b", "llama3.1:8b"])
            self.assertEqual(manifest["simulation_rerun"]["frozen_candidate_ids"], ["c1"])
            self.assertFalse((target_run / "simulation_results.jsonl").exists())
            self.assertEqual(
                perturbation.read_jsonl(target_run / "perturbations.jsonl")[0]["candidate_id"],
                "c1",
            )
            updated_config = {
                **config,
                "config_version": "local-v2",
                "model_roles": {"simulation": {
                    **config["model_roles"]["simulation"],
                    "max_output_tokens": 64,
                }},
            }
            resumed = perturbation.prepare_simulation_rerun(
                updated_config, project_root, source_run, target_run, "target"
            )
            self.assertEqual(resumed["config_version"], "local-v2")
            self.assertEqual(
                resumed["config_amendments"][-1]["reason"],
                "update Simulation request settings; completed checkpoints retained",
            )

    def test_perturbation_review_requires_salience_and_no_prompt_duplication(self):
        checks = {key: True for key in perturbation.PERTURBATION_CHECKS}
        perturbation.validate_decision({"decision": "accept", "issues": [], "checks": checks}, perturbation.PERTURBATION_CHECKS)
        checks.pop("target_distractor_salience")
        with self.assertRaisesRegex(ValueError, "invalid_checks"):
            perturbation.validate_decision({"decision": "accept", "issues": [], "checks": checks}, perturbation.PERTURBATION_CHECKS)

    def test_summary_blocks_simulation_before_codex_proxy_review(self):
        summary = perturbation.summarize_run(
            {
                "run_id": "screen", "selected_count": 0, "minimum_translation_acceptance": 0.5,
                "currency_cost_gate": 1, "simulation_requires_codex_proxy_review": True,
                "simulation_models": ["m"], "simulation_languages": ["en", "zh"],
                "simulation_variants": ["original", "neutral", "targeted"],
            },
            [], [], [], [], [], [],
        )
        self.assertIn("codex_proxy_review_required_before_simulation", summary["gate_reasons"])

    def test_explicit_false_assertion_uses_separate_review_contract(self):
        item = {
            "canonical_fact": "The capital of France is Paris.",
            "prompt_en": "The capital of France is",
            "answer_en": "Paris",
        }
        translation = {
            "prompt_zh": "法国的首都是",
            "answer_zh": "巴黎",
        }
        distractor = {
            "distractor_en": "Berlin",
            "distractor_zh": "柏林",
        }
        prompt = perturbation._perturbation_prompt(
            item,
            translation,
            distractor,
            1,
            (),
            True,
            perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY,
        )
        review_prompt = perturbation._perturbation_review_prompt(
            item,
            translation,
            distractor,
            {
                "english_context": "A false claim.",
                "chinese_context": "一项错误断言。",
                "neutral_english_context": "A neutral claim.",
                "neutral_chinese_context": "一项中性陈述。",
            },
            True,
            perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY,
        )
        self.assertIn("one clear declarative false assertion", prompt)
        self.assertIn("designated_distractor_asserted", review_prompt)
        self.assertNotIn("no_explicit_falsehood", review_prompt)
        checks = {
            key: True for key in perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS
        }
        perturbation.validate_decision(
            {"decision": "accept", "issues": [], "checks": checks},
            perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS,
        )
        checks.pop("single_false_claim")
        with self.assertRaisesRegex(ValueError, "invalid_checks"):
            perturbation.validate_decision(
                {"decision": "accept", "issues": [], "checks": checks},
                perturbation.EXPLICIT_FALSE_ASSERTION_CHECKS,
            )

    def test_dual_distractor_design_shares_one_original_and_builds_ten_inputs(self):
        item_by_id = {
            "s1": {
                "source_id": "s1",
                "base_fact_id": "fact-1",
                "prompt_en": "Capital:",
                "answer_en": "Paris",
            }
        }
        translation_by_id = {
            "s1": {"parsed_response": {"prompt_zh": "首都：", "answer_zh": "巴黎"}}
        }
        distractor_by_id = {
            "d1": {
                "item_id": "d1", "source_id": "s1",
                "distractor_en": "Berlin", "distractor_zh": "柏林",
            },
            "d2": {
                "item_id": "d2", "source_id": "s1",
                "distractor_en": "Rome", "distractor_zh": "罗马",
            },
        }
        candidates = [
            {
                "candidate_id": f"c{index}",
                "source_id": "s1",
                "base_fact_id": "fact-1",
                "distractor_id": distractor_id,
                "candidate": {
                    "english_context": f"target {index}",
                    "chinese_context": f"目标 {index}",
                    "neutral_english_context": f"neutral {index}",
                    "neutral_chinese_context": f"中性 {index}",
                },
            }
            for index, distractor_id in ((1, "d1"), (2, "d2"))
        ]
        rows = perturbation._build_simulation_inputs(
            candidates,
            item_by_id,
            translation_by_id,
            distractor_by_id,
            [{"model": "m", "provider_profile": "p"}],
            ("original",),
            "matched_per_candidate",
            "multi_option",
            3,
        )
        self.assertEqual(len(rows), 10)
        originals = [row for row in rows if row["variant"] == "original"]
        self.assertEqual(len(originals), 2)
        self.assertTrue(all(row["distractor_id"] is None for row in originals))
        self.assertTrue(all(row["base_fact_id"] == "fact-1" for row in rows))
        self.assertTrue(all(row["option_count"] == 3 for row in rows))
        for language in perturbation.SIMULATION_LANGUAGES:
            layouts = {
                tuple(row["option_ids"])
                for row in rows if row["language"] == language
            }
            self.assertEqual(len(layouts), 1)
        for candidate in candidates:
            attached = perturbation._candidate_simulation_rows(
                candidate,
                rows,
                perturbation.SIMULATION_VARIANTS,
                ("original",),
            )
            self.assertEqual(len(attached), 6)

    def test_proxy_labels_are_model_variant_specific_and_base_fact_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "proxy-labels",
                "simulation_models": ["m"],
                "simulation_variants": list(perturbation.SIMULATION_VARIANTS),
                "shared_simulation_variants": ["original"],
                "records": [{"source_id": "s1", "base_fact_id": "fact-1"}],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1", "c2"],
                },
            })
            perturbations = [
                {
                    "candidate_id": f"c{index}",
                    "source_id": "s1",
                    "base_fact_id": "fact-1",
                    "distractor_id": distractor_id,
                    "terminal_status": "completed",
                }
                for index, distractor_id in ((1, "d1"), (2, "d2"))
            ]
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", perturbations)
            rows = []
            for language in perturbation.SIMULATION_LANGUAGES:
                rows.append({
                    "simulation_id": f"original-{language}",
                    "candidate_id": None,
                    "candidate_ids": ["c1", "c2"],
                    "base_fact_id": "fact-1",
                    "source_id": "s1",
                    "distractor_id": None,
                    "model": "m",
                    "language": language,
                    "variant": "original",
                    "simulation_scope": "shared_base_fact_baseline",
                    "terminal_status": "completed",
                    "choice": "A",
                    "correct": True,
                    "distractor_hit": False,
                })
                for candidate_id, distractor_id in (("c1", "d1"), ("c2", "d2")):
                    for variant in ("neutral", "targeted"):
                        en_failure = (
                            candidate_id == "c2"
                            and language == "en"
                            and variant == "targeted"
                        )
                        zh_target = language == "zh" and variant == "targeted"
                        rows.append({
                            "simulation_id": f"{candidate_id}-{language}-{variant}",
                            "candidate_id": candidate_id,
                            "candidate_ids": [candidate_id],
                            "base_fact_id": "fact-1",
                            "source_id": "s1",
                            "distractor_id": distractor_id,
                            "model": "m",
                            "language": language,
                            "variant": variant,
                            "simulation_scope": "candidate",
                            "terminal_status": "completed",
                            "choice": "B" if en_failure or zh_target else "A",
                            "correct": not (en_failure or zh_target),
                            "distractor_hit": zh_target or en_failure,
                        })
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", rows)
            result = perturbation.analyze_three_arm(run_dir)
            by_id = {row["candidate_id"]: row for row in result["candidate_results"]}
            self.assertEqual(by_id["c1"]["proxy_directed_candidate_models"], ["m"])
            self.assertEqual(by_id["c1"]["proxy_zh_specific_strict_models"], ["m"])
            self.assertEqual(by_id["c2"]["proxy_directed_candidate_models"], ["m"])
            self.assertEqual(by_id["c2"]["proxy_zh_specific_strict_models"], [])
            self.assertEqual(result["analyzed_base_fact_count"], 1)
            self.assertEqual(result["target_variant_count"], 2)
            self.assertEqual(
                result["proxy_directed_candidate_variant_count_by_model"], {"m": 2}
            )
            self.assertEqual(
                result["proxy_directed_candidate_base_fact_count_by_model"], {"m": 1}
            )
            self.assertEqual(
                result["proxy_zh_specific_strict_variant_count_by_model"], {"m": 1}
            )
            self.assertEqual(result["proxy_funnel_by_model"]["m"], {
                "N_assigned_facts": 1,
                "N_assigned_variants": 2,
                "N_10_input_terminal_facts": 1,
                "N_evaluable_variants": 2,
                "N_proxy_directed_candidate_variants": 2,
                "N_proxy_zh_specific_strict_variants": 1,
                "N_neutral_unstable": 0,
                "N_off_target_failure": 0,
                "N_shared_language_susceptibility": 1,
                "N_incomplete": 0,
                "base_fact_level": {
                    "N_proxy_directed_candidate_any": 1,
                    "N_proxy_directed_candidate_both": 1,
                    "N_proxy_zh_specific_strict_any": 1,
                    "N_proxy_zh_specific_strict_both": 0,
                    "N_neutral_unstable_any": 0,
                    "N_off_target_failure_any": 0,
                    "N_shared_language_susceptibility_any": 1,
                    "N_incomplete_facts": 0,
                },
            })
            labels = result["base_fact_results"][0]["model_labels"]["m"]
            self.assertTrue(labels["proxy_directed_both"])
            self.assertFalse(labels["proxy_zh_specific_strict_both"])

    def test_proxy_funnel_reports_failure_categories_and_missing_assigned_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            perturbation.write_json(run_dir / "run_manifest.json", {
                "run_id": "proxy-funnel",
                "manipulation_family": (
                    perturbation.EXPLICIT_FALSE_ASSERTION_MANIPULATION_FAMILY
                ),
                "designated_distractors_per_base_fact": 2,
                "simulation_models": ["m"],
                "simulation_variants": list(perturbation.SIMULATION_VARIANTS),
                "shared_simulation_variants": ["original"],
                "records": [
                    {"source_id": "s1", "base_fact_id": "fact-1"},
                    {"source_id": "s2", "base_fact_id": "fact-2"},
                ],
                "codex_proxy_review": {
                    "reviewer_type": "codex_proxy",
                    "not_human_gold": True,
                    "accepted_perturbation_ids": ["c1", "c2", "c3"],
                },
            })
            perturbation.write_jsonl(run_dir / "perturbations.jsonl", [
                {
                    "candidate_id": "c1",
                    "source_id": "s1",
                    "base_fact_id": "fact-1",
                    "distractor_id": "d1",
                    "terminal_status": "completed",
                },
                {
                    "candidate_id": "c2",
                    "source_id": "s1",
                    "base_fact_id": "fact-1",
                    "distractor_id": "d2",
                    "terminal_status": "completed",
                },
                {
                    "candidate_id": "c3",
                    "source_id": "s2",
                    "base_fact_id": "fact-2",
                    "distractor_id": "d3",
                    "terminal_status": "completed",
                },
            ])
            rows = []
            for language in perturbation.SIMULATION_LANGUAGES:
                rows.append({
                    "simulation_id": f"original-{language}",
                    "candidate_id": None,
                    "candidate_ids": ["c1", "c2"],
                    "base_fact_id": "fact-1",
                    "source_id": "s1",
                    "distractor_id": None,
                    "model": "m",
                    "language": language,
                    "variant": "original",
                    "simulation_scope": "shared_base_fact_baseline",
                    "terminal_status": "completed",
                    "choice": "A",
                    "correct": True,
                    "distractor_hit": False,
                })
                for candidate_id, distractor_id in (("c1", "d1"), ("c2", "d2")):
                    for variant in ("neutral", "targeted"):
                        neutral_failure = (
                            candidate_id == "c1"
                            and language == "zh"
                            and variant == "neutral"
                        )
                        shared_language_hit = (
                            candidate_id == "c2"
                            and language == "en"
                            and variant == "targeted"
                        )
                        off_target_failure = (
                            candidate_id == "c1"
                            and language == "zh"
                            and variant == "targeted"
                        )
                        zh_directed = (
                            candidate_id == "c2"
                            and language == "zh"
                            and variant == "targeted"
                        )
                        wrong = (
                            neutral_failure
                            or shared_language_hit
                            or off_target_failure
                            or zh_directed
                        )
                        rows.append({
                            "simulation_id": (
                                f"{candidate_id}-{language}-{variant}"
                            ),
                            "candidate_id": candidate_id,
                            "candidate_ids": [candidate_id],
                            "base_fact_id": "fact-1",
                            "source_id": "s1",
                            "distractor_id": distractor_id,
                            "model": "m",
                            "language": language,
                            "variant": variant,
                            "simulation_scope": "candidate",
                            "terminal_status": "completed",
                            "choice": "B" if wrong else "A",
                            "correct": not wrong,
                            "distractor_hit": shared_language_hit or zh_directed,
                        })
            rows.extend([
                {
                    "simulation_id": "c3-zh-original",
                    "candidate_id": None,
                    "candidate_ids": ["c3"],
                    "base_fact_id": "fact-2",
                    "source_id": "s2",
                    "distractor_id": None,
                    "model": "m",
                    "language": "zh",
                    "variant": "original",
                    "simulation_scope": "shared_base_fact_baseline",
                    "terminal_status": "completed",
                    "choice": "A",
                    "correct": True,
                    "distractor_hit": False,
                },
                {
                    "simulation_id": "c3-zh-neutral",
                    "candidate_id": "c3",
                    "candidate_ids": ["c3"],
                    "base_fact_id": "fact-2",
                    "source_id": "s2",
                    "distractor_id": "d3",
                    "model": "m",
                    "language": "zh",
                    "variant": "neutral",
                    "simulation_scope": "candidate",
                    "terminal_status": "completed",
                    "choice": "A",
                    "correct": True,
                    "distractor_hit": False,
                },
                {
                    "simulation_id": "c3-zh-targeted",
                    "candidate_id": "c3",
                    "candidate_ids": ["c3"],
                    "base_fact_id": "fact-2",
                    "source_id": "s2",
                    "distractor_id": "d3",
                    "model": "m",
                    "language": "zh",
                    "variant": "targeted",
                    "simulation_scope": "candidate",
                    "terminal_status": "completed",
                    "choice": "B",
                    "correct": False,
                    "distractor_hit": True,
                },
            ])
            perturbation.write_jsonl(run_dir / "simulation_results.jsonl", rows)
            result = perturbation.analyze_three_arm(run_dir)

            funnel = result["proxy_funnel_by_model"]["m"]
            self.assertEqual(funnel["N_assigned_facts"], 2)
            self.assertEqual(funnel["N_assigned_variants"], 4)
            self.assertEqual(funnel["N_10_input_terminal_facts"], 1)
            self.assertEqual(funnel["N_evaluable_variants"], 2)
            self.assertEqual(funnel["N_proxy_directed_candidate_variants"], 1)
            self.assertEqual(funnel["N_neutral_unstable"], 1)
            self.assertEqual(funnel["N_off_target_failure"], 1)
            self.assertEqual(funnel["N_shared_language_susceptibility"], 1)
            self.assertEqual(funnel["N_incomplete"], 2)
            self.assertEqual(
                funnel["base_fact_level"]["N_incomplete_facts"], 1
            )
            by_fact = {
                row["base_fact_id"]: row for row in result["base_fact_results"]
            }
            self.assertTrue(by_fact["fact-1"]["model_labels"]["m"][
                "ten_input_terminal"
            ])
            self.assertEqual(by_fact["fact-2"]["candidate_count"], 1)
            self.assertFalse(
                by_fact["fact-2"]["model_labels"]["m"][
                    "proxy_directed_candidate"
                ]
            )
            self.assertEqual(
                by_fact["fact-2"]["model_labels"]["m"][
                    "incomplete_variant_count"
                ],
                2,
            )


if __name__ == "__main__":
    unittest.main()
