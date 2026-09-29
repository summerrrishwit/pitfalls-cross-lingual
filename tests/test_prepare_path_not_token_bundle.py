import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_path_not_token_bundle.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("prepare_path_not_token_bundle", SCRIPT_PATH)
preparation = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(preparation)


def bundle_record(
    base_fact_id="bf_001",
    source_id="source_001",
    canonical_status="frozen",
    **overrides,
):
    record = {
        "schema_version": "factual-perturbation-input-bundle-v1",
        "base_fact_id": base_fact_id,
        "candidate_id": source_id,
        "source_id": source_id,
        "source_pool_id": "public-benchmarks-full-v1",
        "source_dataset": "global_mmlu",
        "source_format": "multiple_choice",
        "canonical_status": canonical_status,
        "bundle_status": "reviewed_frozen",
        "evidence_tier": "independent_adjudicated",
        "human_gold": False,
        "subject_en": "France",
        "relation_raw": "capital city",
        "relation_normalized": "entity_location",
        "relation_signature_id": "rel_sig_001",
        "probe_relation_id": "country_to_capital",
        "probe_relation_candidate": "country_to_capital",
        "probe_relation_status": "frozen",
        "split_policy_version": "test-normalized-answer-split-v1",
        "split_status": "frozen",
        "split_group_id": "split_group_paris",
        "split_assignment": "development",
        "sealed_evaluation_status": "not_run",
        "canonical_fact_en": "The capital of France is Paris.",
        "prompt_en": "The capital of France is",
        "prompt_version": "reviewed-factual-prompt-v1",
        "prompt_template_id": "subject_relation_completion",
        "prompt_quality_tier": "strict_factual_completion",
        "prompt_ready": True,
        "answer_en": "Paris",
        "answer_aliases_en": ["City of Paris", "paris"],
        "admission": {
            "semantic_review_complete": True,
            "prompt_ready": True,
        },
        "provenance": {"source_model": "qwen3.7-plus", "record_id": "source_001"},
    }
    record.update(overrides)
    return record


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_formal_freeze_manifests(
    input_path,
    records,
    *,
    review_overrides=None,
    split_overrides=None,
):
    identity = preparation.formal_admission.bundle_identity(
        input_path,
        preparation.INPUT_BUNDLE_SCHEMA_VERSION,
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
    split_policy_version = next(iter(split_policy_versions), "test-normalized-answer-split-v1")
    row_by_id = {
        preparation.formal_admission.explicit_base_fact_id(row, "fixture row"): row
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
        preparation.write_jsonl(path, rows)
        identifiers = sorted(str(row[id_field]).strip() for row in rows)
        return {
            "path": str(path.resolve()),
            "sha256": preparation.sha256_file(path),
            "schema_version": schema_version,
            "record_count": len(rows),
            digest_field: preparation.formal_admission.sha256_value(identifiers),
        }

    scope_rows = [
        {
            "schema_version": preparation.formal_admission.REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
            "base_fact_id": base_fact_id,
            "input_record_sha256": preparation.formal_admission.sha256_value(
                row_by_id[base_fact_id]
            ),
        }
        for base_fact_id in terminal_ids
    ]
    decision_rows = [
        {
            "schema_version": preparation.formal_admission.REVIEW_DECISION_SCHEMA_VERSION,
            "base_fact_id": base_fact_id,
            "input_record_sha256": preparation.formal_admission.sha256_value(
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
        input_path.with_name("formal_review_scope.jsonl"),
        scope_rows,
        preparation.formal_admission.REVIEW_SCOPE_RECORD_SCHEMA_VERSION,
        "base_fact_id",
        "base_fact_ids_sha256",
    )
    decision_binding = artifact_binding(
        input_path.with_name("formal_review_decisions.jsonl"),
        decision_rows,
        preparation.formal_admission.REVIEW_DECISION_SCHEMA_VERSION,
        "base_fact_id",
        "base_fact_ids_sha256",
    )
    review = {
        "schema_version": (
            preparation.formal_admission.REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION
        ),
        "input_bundle": binding,
        "source_universe": {
            "record_count": len(row_by_id),
            "base_fact_ids_sha256": preparation.formal_admission.sha256_value(
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
    review_path = input_path.with_name("review_freeze_manifest.json")
    preparation.write_json(review_path, review)

    risk_pairs = preparation.formal_admission._mechanical_cross_group_risk_pairs(records)
    candidate_rows = []
    adjudication_rows = []
    for left, right in sorted(risk_pairs):
        candidate_id = f"near_{preparation.formal_admission.sha256_value([left, right])[:16]}"
        candidate_row = {
            "schema_version": (
                preparation.formal_admission.NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION
            ),
            "candidate_id": candidate_id,
            "left_base_fact_id": left,
            "right_base_fact_id": right,
            "left_candidate_id": row_by_id[left]["candidate_id"],
            "right_candidate_id": row_by_id[right]["candidate_id"],
            "left_input_record_sha256": preparation.formal_admission.sha256_value(
                row_by_id[left]
            ),
            "right_input_record_sha256": preparation.formal_admission.sha256_value(
                row_by_id[right]
            ),
        }
        candidate_rows.append(candidate_row)
        adjudication_rows.append({
            "schema_version": (
                preparation.formal_admission.NEAR_DUPLICATE_DECISION_SCHEMA_VERSION
            ),
            "candidate_id": candidate_id,
            "candidate_record_sha256": (
                preparation.formal_admission.sha256_value(candidate_row)
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
        input_path.with_name("near_duplicate_candidates.jsonl"),
        candidate_rows,
        preparation.formal_admission.NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION,
        "candidate_id",
        "candidate_ids_sha256",
    )
    adjudication_binding = artifact_binding(
        input_path.with_name("near_duplicate_adjudications.jsonl"),
        adjudication_rows,
        preparation.formal_admission.NEAR_DUPLICATE_DECISION_SCHEMA_VERSION,
        "candidate_id",
        "candidate_ids_sha256",
    )
    registry_binding = artifact_binding(
        input_path.with_name("historical_exposure_registry.jsonl"),
        [],
        preparation.formal_admission.HISTORICAL_EXPOSURE_REGISTRY_SCHEMA_VERSION,
        "exposure_id",
        "exposure_ids_sha256",
    )
    exposure_rows = [
        {
            "schema_version": (
                preparation.formal_admission.HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION
            ),
            "base_fact_id": base_fact_id,
            "input_record_sha256": preparation.formal_admission.sha256_value(
                row_by_id[base_fact_id]
            ),
            "split_assignment": row_by_id[base_fact_id]["split_assignment"],
            "historically_exposed": False,
        }
        for base_fact_id in frozen_ids
        if row_by_id[base_fact_id].get("split_assignment") in {"validation", "sealed"}
    ]
    exposure_binding = artifact_binding(
        input_path.with_name("historical_exposure_checks.jsonl"),
        exposure_rows,
        preparation.formal_admission.HISTORICAL_EXPOSURE_CHECK_SCHEMA_VERSION,
        "base_fact_id",
        "base_fact_ids_sha256",
    )
    assignments, _ = preparation.formal_admission._split_assignments(records)
    split = {
        "schema_version": (
            preparation.formal_admission.SPLIT_FREEZE_MANIFEST_SCHEMA_VERSION
        ),
        "input_bundle": binding,
        "review_freeze_manifest": {
            "path": str(review_path.resolve()),
            "sha256": preparation.sha256_file(review_path),
            "schema_version": (
                preparation.formal_admission.REVIEW_FREEZE_MANIFEST_SCHEMA_VERSION
            ),
        },
        "split_policy_version": split_policy_version,
        "split_assignments_sha256": preparation.formal_admission.sha256_value(
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
    split_path = input_path.with_name("split_freeze_manifest.json")
    preparation.write_json(split_path, split)
    return review_path, split_path


def build_with_formal_freeze(records, *, allow_provisional=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "behavior_input_bundle.jsonl"
        write_jsonl(input_path, records)
        review_manifest, split_manifest = write_formal_freeze_manifests(
            input_path, records
        )
        return preparation.build_records(
            records,
            input_path,
            preparation.sha256_file(input_path),
            allow_provisional=allow_provisional,
            review_freeze_manifest_path=review_manifest,
            split_freeze_manifest_path=split_manifest,
        )


def build_without_formal_freeze(records, *, allow_provisional=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "behavior_input_bundle.jsonl"
        write_jsonl(input_path, records)
        return preparation.build_records(
            records,
            input_path,
            preparation.sha256_file(input_path),
            allow_provisional=allow_provisional,
        )


class PathNotTokenBundlePreparationTests(unittest.TestCase):
    def test_frozen_fact_is_prepared_but_never_claimed_experiment_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            records = [bundle_record()]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )

            summary = preparation.prepare(
                input_path,
                output_dir,
                review_freeze_manifest_path=review_manifest,
                split_freeze_manifest_path=split_manifest,
            )
            rows = preparation.read_jsonl(output_dir / "base_facts.jsonl")

            self.assertEqual(summary["counts"]["frozen_base_facts"], 1)
            self.assertEqual(summary["counts"]["review_only_base_facts"], 0)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["base_fact_id"], "bf_001")
            self.assertEqual(row["base_id"], "bf_001")
            self.assertEqual(row["canonical_status"], "frozen")
            self.assertEqual(row["evidence_tier"], "independent_adjudicated")
            self.assertEqual(row["prompt_format"], "factual_completion")
            self.assertEqual(row["answer_aliases_en"], ["Paris", "City of Paris"])
            self.assertEqual(row["split_policy_version"], "test-normalized-answer-split-v1")
            self.assertEqual(row["split_status"], "frozen")
            self.assertEqual(row["split_assignment"], "development")
            self.assertFalse(row["review_only"])
            self.assertFalse(row["path_not_token_experiment_ready"])
            self.assertTrue(row["admission"]["external_freeze_manifests_verified"])
            self.assertFalse(row["admission"]["path_not_token_experiment_ready"])
            self.assertEqual(
                row["answer_tokenization"]["status"],
                "pending_target_model_and_tokenizer",
            )
            self.assertEqual(row["mechanism_analysis"]["status"], "not_run")
            self.assertEqual(
                row["provenance"]["review_freeze_manifest"]["sha256"],
                preparation.sha256_file(review_manifest),
            )
            self.assertEqual(
                summary["split_freeze_manifest"]["sha256"],
                preparation.sha256_file(split_manifest),
            )

    def test_open_answer_fallback_uses_open_question_prompt_format(self):
        source = bundle_record(prompt_quality_tier="open_answer_fallback")
        record = preparation._path_record(
            source,
            "bf_001",
            Path("/tmp/behavior_input_bundle.jsonl"),
            "0" * 64,
        )
        self.assertEqual(record["prompt_format"], "open_question_with_answer_marker")

    def test_relation_specific_open_completion_uses_factual_completion_format(self):
        source = bundle_record(
            prompt_en="The capital of France is",
            prompt_template_id=(
                "probe_relation_open_completion_en_country_to_capital_v1"
            ),
            prompt_quality_tier="relation_specific_open_completion",
        )
        record = preparation._path_record(
            source,
            "bf_001",
            Path("/tmp/behavior_input_bundle.jsonl"),
            "0" * 64,
        )

        self.assertEqual(record["prompt_format"], "factual_completion")
        self.assertEqual(
            record["prompt_quality_tier"], "relation_specific_open_completion"
        )

    def test_unknown_prompt_quality_tier_is_rejected(self):
        source = bundle_record(prompt_quality_tier="unknown_open_completion")

        with self.assertRaisesRegex(
            ValueError,
            "unsupported prompt_quality_tier: unknown_open_completion",
        ):
            preparation._path_record(
                source,
                "bf_001",
                Path("/tmp/behavior_input_bundle.jsonl"),
                "0" * 64,
            )

    def test_default_excludes_provisional_and_rejected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            records = [
                bundle_record("bf_frozen", "source_frozen"),
                bundle_record(
                    "bf_rejected", "source_rejected", canonical_status="rejected"
                ),
            ]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )

            summary = preparation.prepare(
                input_path,
                output_dir,
                review_freeze_manifest_path=review_manifest,
                split_freeze_manifest_path=split_manifest,
            )
            frozen = preparation.read_jsonl(output_dir / "base_facts.jsonl")
            exclusions = preparation.read_jsonl(output_dir / "excluded_records.jsonl")

            self.assertEqual([row["base_fact_id"] for row in frozen], ["bf_frozen"])
            self.assertEqual(summary["counts"]["excluded"], 1)
            self.assertEqual(
                {row["base_fact_id"]: row["exclusion_reason"] for row in exclusions},
                {
                    "bf_rejected": "canonical_rejected",
                },
            )

    def test_review_freeze_rejects_nonterminal_source_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            records = [
                bundle_record("bf_frozen", "source_frozen"),
                bundle_record(
                    "bf_pending", "source_pending", canonical_status="pending_review"
                ),
            ]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )

            with self.assertRaisesRegex(ValueError, "non-terminal canonical_status"):
                preparation.prepare(
                    input_path,
                    output_dir,
                    review_freeze_manifest_path=review_manifest,
                    split_freeze_manifest_path=split_manifest,
                )

    def test_input_human_gold_cannot_override_bound_review_evidence(self):
        for input_human_gold in (True, None):
            with self.subTest(input_human_gold=input_human_gold), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                input_path = root / "behavior_input_bundle.jsonl"
                record = bundle_record()
                if input_human_gold is None:
                    del record["human_gold"]
                else:
                    record["human_gold"] = input_human_gold
                records = [record]
                write_jsonl(input_path, records)
                review_manifest, split_manifest = write_formal_freeze_manifests(
                    input_path, records
                )

                with self.assertRaisesRegex(ValueError, "input row human_gold"):
                    preparation.formal_admission.validate_external_formal_admission(
                        input_bundle_path=input_path,
                        input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                        records=records,
                        input_bundle_sha256=preparation.sha256_file(input_path),
                        review_freeze_manifest_path=review_manifest,
                        split_freeze_manifest_path=split_manifest,
                    )

    def test_input_evidence_tier_must_match_bound_review_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            records = [bundle_record()]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )
            review = json.loads(review_manifest.read_text(encoding="utf-8"))
            decisions_path = Path(review["review_decisions"]["path"])
            decisions = preparation.read_jsonl(decisions_path)
            decisions[0]["evidence_tier"] = "reviewed_canonical"
            preparation.write_jsonl(decisions_path, decisions)
            review["review_decisions"]["sha256"] = preparation.sha256_file(
                decisions_path
            )
            preparation.write_json(review_manifest, review)

            with self.assertRaisesRegex(
                ValueError, "evidence_tier does not match verified review evidence"
            ):
                preparation.formal_admission.validate_external_formal_admission(
                    input_bundle_path=input_path,
                    input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records=records,
                    input_bundle_sha256=preparation.sha256_file(input_path),
                    review_freeze_manifest_path=review_manifest,
                    split_freeze_manifest_path=split_manifest,
                )

    def test_formal_evidence_tier_requires_exact_canonical_value(self):
        for evidence_tier in (
            " independent_adjudicated ",
            "self_asserted_reviewed",
        ):
            with self.subTest(evidence_tier=evidence_tier), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                input_path = root / "behavior_input_bundle.jsonl"
                records = [bundle_record(evidence_tier=evidence_tier)]
                write_jsonl(input_path, records)
                review_manifest, split_manifest = write_formal_freeze_manifests(
                    input_path, records
                )

                with self.assertRaisesRegex(ValueError, "canonical formal tier"):
                    preparation.formal_admission.validate_external_formal_admission(
                        input_bundle_path=input_path,
                        input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                        records=records,
                        input_bundle_sha256=preparation.sha256_file(input_path),
                        review_freeze_manifest_path=review_manifest,
                        split_freeze_manifest_path=split_manifest,
                    )

    def test_provisional_human_gold_requires_strict_false_boolean(self):
        cases = (
            ("false", "human_gold must be boolean"),
            (None, "human_gold must be boolean"),
            (True, "provisional input must have human_gold=false"),
        )
        for human_gold, message in cases:
            with self.subTest(human_gold=human_gold), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                input_path = root / "behavior_input_bundle.jsonl"
                output_dir = root / "path-output"
                record = bundle_record(
                    canonical_status="pending_review",
                    bundle_status="draft_pending_review",
                    evidence_tier="provisional_single_model",
                    human_gold=human_gold,
                    split_status="provisional_not_frozen",
                )
                write_jsonl(input_path, [record])

                with self.assertRaisesRegex(ValueError, message):
                    preparation.prepare(
                        input_path,
                        output_dir,
                        allow_provisional=True,
                    )

    def test_legacy_review_decision_without_evidence_tier_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            records = [bundle_record()]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )
            review = json.loads(review_manifest.read_text(encoding="utf-8"))
            decisions_path = Path(review["review_decisions"]["path"])
            decisions = preparation.read_jsonl(decisions_path)
            decisions[0].pop("evidence_tier")
            decisions[0]["schema_version"] = "factual-review-freeze-decision-v1"
            preparation.write_jsonl(decisions_path, decisions)
            review["review_decisions"].update(
                {
                    "sha256": preparation.sha256_file(decisions_path),
                    "schema_version": "factual-review-freeze-decision-v1",
                }
            )
            preparation.write_json(review_manifest, review)

            with self.assertRaisesRegex(ValueError, "schema_version is unsupported"):
                preparation.formal_admission.validate_external_formal_admission(
                    input_bundle_path=input_path,
                    input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records=records,
                    input_bundle_sha256=preparation.sha256_file(input_path),
                    review_freeze_manifest_path=review_manifest,
                    split_freeze_manifest_path=split_manifest,
                )

    def test_near_duplicate_decision_binds_candidate_and_both_endpoints(self):
        binding_fields = {
            "candidate_record_sha256": "0" * 64,
            "left_candidate_id": "stale-left-candidate",
            "right_candidate_id": "stale-right-candidate",
            "left_input_record_sha256": "1" * 64,
            "right_input_record_sha256": "2" * 64,
        }
        for field, stale_value in binding_fields.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                input_path = root / "behavior_input_bundle.jsonl"
                records = [
                    bundle_record(),
                    bundle_record(
                        "bf_002",
                        "source_002",
                        canonical_fact_en="Paris is a major city in France.",
                        prompt_en="A major city in France is",
                        split_group_id="split_group_other_paris",
                        split_assignment="validation",
                    ),
                ]
                write_jsonl(input_path, records)
                review_manifest, split_manifest = write_formal_freeze_manifests(
                    input_path, records
                )
                split = json.loads(split_manifest.read_text(encoding="utf-8"))
                decisions_path = Path(split["near_duplicate_adjudications"]["path"])
                decisions = preparation.read_jsonl(decisions_path)
                self.assertEqual(len(decisions), 1)
                decisions[0][field] = stale_value
                preparation.write_jsonl(decisions_path, decisions)
                split["near_duplicate_adjudications"]["sha256"] = (
                    preparation.sha256_file(decisions_path)
                )
                preparation.write_json(split_manifest, split)

                with self.assertRaisesRegex(ValueError, field):
                    preparation.formal_admission.validate_external_formal_admission(
                        input_bundle_path=input_path,
                        input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                        records=records,
                        input_bundle_sha256=preparation.sha256_file(input_path),
                        review_freeze_manifest_path=review_manifest,
                        split_freeze_manifest_path=split_manifest,
                    )

    def test_legacy_unbound_near_duplicate_decision_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            records = [
                bundle_record(),
                bundle_record(
                    "bf_002",
                    "source_002",
                    canonical_fact_en="Paris is a major city in France.",
                    prompt_en="A major city in France is",
                    split_group_id="split_group_other_paris",
                    split_assignment="validation",
                ),
            ]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )
            split = json.loads(split_manifest.read_text(encoding="utf-8"))
            decisions_path = Path(split["near_duplicate_adjudications"]["path"])
            decisions = preparation.read_jsonl(decisions_path)
            decisions[0] = {
                "schema_version": "factual-near-duplicate-decision-v1",
                "candidate_id": decisions[0]["candidate_id"],
                "decision": decisions[0]["decision"],
                "reviewer_type": decisions[0]["reviewer_type"],
                "human_gold": decisions[0]["human_gold"],
            }
            preparation.write_jsonl(decisions_path, decisions)
            split["near_duplicate_adjudications"].update(
                {
                    "sha256": preparation.sha256_file(decisions_path),
                    "schema_version": "factual-near-duplicate-decision-v1",
                }
            )
            preparation.write_json(split_manifest, split)

            with self.assertRaisesRegex(ValueError, "schema_version is unsupported"):
                preparation.formal_admission.validate_external_formal_admission(
                    input_bundle_path=input_path,
                    input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records=records,
                    input_bundle_sha256=preparation.sha256_file(input_path),
                    review_freeze_manifest_path=review_manifest,
                    split_freeze_manifest_path=split_manifest,
                )

    def test_legacy_near_duplicate_candidate_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            records = [
                bundle_record(),
                bundle_record(
                    "bf_002",
                    "source_002",
                    canonical_fact_en="Paris is a major city in France.",
                    prompt_en="A major city in France is",
                    split_group_id="split_group_other_paris",
                    split_assignment="validation",
                ),
            ]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )
            split = json.loads(split_manifest.read_text(encoding="utf-8"))
            candidates_path = Path(split["near_duplicate_candidates"]["path"])
            candidates = preparation.read_jsonl(candidates_path)
            candidates[0]["schema_version"] = "factual-near-duplicate-candidate-v1"
            for field in ("left_candidate_id", "right_candidate_id"):
                candidates[0].pop(field)
            preparation.write_jsonl(candidates_path, candidates)
            split["near_duplicate_candidates"].update(
                {
                    "sha256": preparation.sha256_file(candidates_path),
                    "schema_version": "factual-near-duplicate-candidate-v1",
                }
            )
            preparation.write_json(split_manifest, split)

            with self.assertRaisesRegex(ValueError, "schema_version is unsupported"):
                preparation.formal_admission.validate_external_formal_admission(
                    input_bundle_path=input_path,
                    input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records=records,
                    input_bundle_sha256=preparation.sha256_file(input_path),
                    review_freeze_manifest_path=review_manifest,
                    split_freeze_manifest_path=split_manifest,
                )

    def test_near_duplicate_candidate_endpoint_id_is_bound_to_input_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            records = [
                bundle_record(),
                bundle_record(
                    "bf_002",
                    "source_002",
                    canonical_fact_en="Paris is a major city in France.",
                    prompt_en="A major city in France is",
                    split_group_id="split_group_other_paris",
                    split_assignment="validation",
                ),
            ]
            write_jsonl(input_path, records)
            review_manifest, split_manifest = write_formal_freeze_manifests(
                input_path, records
            )
            split = json.loads(split_manifest.read_text(encoding="utf-8"))
            candidates_path = Path(split["near_duplicate_candidates"]["path"])
            candidates = preparation.read_jsonl(candidates_path)
            candidates[0]["left_candidate_id"] = "forged-left-candidate"
            preparation.write_jsonl(candidates_path, candidates)
            split["near_duplicate_candidates"]["sha256"] = preparation.sha256_file(
                candidates_path
            )

            decisions_path = Path(split["near_duplicate_adjudications"]["path"])
            decisions = preparation.read_jsonl(decisions_path)
            decisions[0]["left_candidate_id"] = candidates[0]["left_candidate_id"]
            decisions[0]["candidate_record_sha256"] = (
                preparation.formal_admission.sha256_value(candidates[0])
            )
            preparation.write_jsonl(decisions_path, decisions)
            split["near_duplicate_adjudications"]["sha256"] = (
                preparation.sha256_file(decisions_path)
            )
            preparation.write_json(split_manifest, split)

            with self.assertRaisesRegex(ValueError, "stale left candidate_id"):
                preparation.formal_admission.validate_external_formal_admission(
                    input_bundle_path=input_path,
                    input_schema_version=preparation.INPUT_BUNDLE_SCHEMA_VERSION,
                    records=records,
                    input_bundle_sha256=preparation.sha256_file(input_path),
                    review_freeze_manifest_path=review_manifest,
                    split_freeze_manifest_path=split_manifest,
                )

    def test_allow_provisional_writes_only_review_artifact_without_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            pending = bundle_record(
                "bf_pending",
                "source_pending",
                canonical_status="pending_review",
                evidence_tier="provisional_single_model",
                bundle_status="draft_pending_review",
                admission={"semantic_review_complete": False, "prompt_ready": True},
                split_status="provisional_not_frozen",
                probe_relation_id=None,
                probe_relation_status="pending_semantic_and_balance_review",
            )
            write_jsonl(input_path, [pending])

            summary = preparation.prepare(input_path, output_dir, allow_provisional=True)
            review_rows = preparation.read_jsonl(
                output_dir / "review_only_base_facts.jsonl"
            )

            self.assertEqual((output_dir / "base_facts.jsonl").read_text(), "")
            self.assertEqual(summary["counts"]["review_only_base_facts"], 1)
            self.assertEqual(len(review_rows), 1)
            row = review_rows[0]
            self.assertTrue(row["review_only"])
            self.assertEqual(row["canonical_status"], "pending_review")
            self.assertEqual(row["evidence_tier"], "provisional_single_model")
            self.assertFalse(row["admission"]["canonical_frozen"])
            self.assertFalse(row["admission"]["path_not_token_experiment_ready"])
            self.assertIn("canonical_not_frozen", row["admission"]["blocking_reasons"])
            self.assertIn("split_not_frozen", row["admission"]["blocking_reasons"])
            self.assertEqual(row["mechanism_analysis"]["status"], "not_run")

    def test_preserves_relation_direction_and_bilingual_preparation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            pending = bundle_record(
                canonical_status="pending_review",
                evidence_tier="provisional_single_model",
                bundle_status="draft_pending_review",
                admission={"semantic_review_complete": False, "prompt_ready": True},
                split_status="provisional_not_frozen",
                normalization_status="not_required_for_first_probe_cohort",
                probe_relation_candidate_direction="country -> capital",
                probe_relation_candidate_policy="fixture-probe-policy-v1",
                target_language="zh",
                subject_target="法国",
                relation_target="首都",
                answer_target="巴黎",
                answer_aliases_target=["巴黎市"],
                prompt_target="法国的首都是",
                translation_prompt_en_to_target="Paris in Chinese is",
                translation_status="reviewed",
            )
            write_jsonl(input_path, [pending])

            preparation.prepare(input_path, output_dir, allow_provisional=True)
            row = preparation.read_jsonl(output_dir / "review_only_base_facts.jsonl")[0]

            self.assertEqual(row["normalization_status"], "not_required_for_first_probe_cohort")
            self.assertEqual(row["probe_relation_candidate_direction"], "country -> capital")
            self.assertEqual(row["probe_relation_candidate_policy"], "fixture-probe-policy-v1")
            self.assertEqual(row["answer_target"], "巴黎")
            self.assertEqual(row["translation_prompt_en_to_target"], "Paris in Chinese is")
            self.assertEqual(row["language_alignment"]["known_id"], "bf_001")
            self.assertTrue(row["admission"]["target_translation_ready"])
            self.assertNotIn(
                "target_translation_not_frozen", row["admission"]["blocking_reasons"]
            )

    def test_frozen_row_without_external_manifests_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            write_jsonl(input_path, [bundle_record()])

            summary = preparation.prepare(input_path, output_dir)
            exclusions = preparation.read_jsonl(output_dir / "excluded_records.jsonl")

            self.assertEqual((output_dir / "base_facts.jsonl").read_text(), "")
            self.assertEqual(summary["counts"]["frozen_base_facts"], 0)
            self.assertFalse(summary["external_formal_admission"]["authorized"])
            self.assertEqual(
                set(exclusions[0]["review_blockers"]),
                {
                    "review_freeze_manifest_missing",
                    "split_freeze_manifest_missing",
                },
            )

    def test_input_snapshot_replacement_before_admission_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            output_dir = root / "path-output"
            original = [bundle_record()]
            replacement = [bundle_record(prompt_en="Changed prompt")]
            write_jsonl(input_path, original)
            original_read_bytes = Path.read_bytes
            state = {"swapped": False}

            def read_then_replace(path):
                raw = original_read_bytes(path)
                if path.resolve() == input_path.resolve() and not state["swapped"]:
                    write_jsonl(input_path, replacement)
                    state["swapped"] = True
                return raw

            with patch.object(Path, "read_bytes", read_then_replace):
                with self.assertRaisesRegex(ValueError, "does not match input bundle bytes"):
                    preparation.prepare(input_path, output_dir)

    def test_prevalidated_admission_mapping_cannot_bypass_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "behavior_input_bundle.jsonl"
            records = [bundle_record()]
            write_jsonl(input_path, records)
            forged = {
                "authorized": True,
                "blockers": [],
                "review_freeze_manifest": {
                    "path": None,
                    "sha256": None,
                    "schema_version": None,
                },
                "split_freeze_manifest": {
                    "path": None,
                    "sha256": None,
                    "schema_version": None,
                },
            }

            with self.assertRaisesRegex(ValueError, "does not match revalidated evidence"):
                preparation.build_records(
                    records,
                    input_path,
                    preparation.sha256_file(input_path),
                    _validated_external_formal_admission=forged,
                )

    def test_changing_only_canonical_status_invalidates_formal_freeze(self):
        source = bundle_record(
            canonical_status="frozen",
            bundle_status="draft_pending_review",
            evidence_tier="provisional_single_model",
            admission={"semantic_review_complete": False, "prompt_ready": True},
            split_status="provisional_not_frozen",
            probe_relation_id=None,
            probe_relation_status="pending_semantic_and_balance_review",
        )

        with self.assertRaisesRegex(ValueError, "canonical formal tier"):
            build_with_formal_freeze([source])

    def test_formal_record_requires_complete_split_identity(self):
        source = bundle_record(split_group_id=None)
        frozen, review_only, exclusions = build_without_formal_freeze([source])

        self.assertEqual(frozen, [])
        self.assertEqual(review_only, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("split_group_id_missing", exclusions[0]["review_blockers"])

    def test_bundle_rejects_inconsistent_split_group(self):
        with self.assertRaisesRegex(ValueError, "inconsistent assignment"):
            build_without_formal_freeze(
                [
                    bundle_record("bf_001", "source_001"),
                    bundle_record(
                        "bf_002",
                        "source_002",
                        split_assignment="validation",
                    ),
                ]
            )

    def test_formal_rows_require_one_split_policy_version(self):
        with self.assertRaisesRegex(ValueError, "exactly one split_policy_version"):
            build_without_formal_freeze(
                [
                    bundle_record("bf_001", "source_001", split_group_id="group_1"),
                    bundle_record(
                        "bf_002",
                        "source_002",
                        split_group_id="group_2",
                        split_policy_version="another-split-policy-v1",
                    ),
                ]
            )

    def test_source_prompt_readiness_is_required_and_preserved(self):
        source = bundle_record(
            admission={"semantic_review_complete": True, "prompt_ready": False}
        )
        frozen, review_only, exclusions = build_with_formal_freeze([source])

        self.assertEqual(frozen, [])
        self.assertEqual(review_only, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("prompt_not_ready", exclusions[0]["review_blockers"])

        record = preparation._path_record(
            source,
            "bf_001",
            Path("/tmp/behavior_input_bundle.jsonl"),
            "0" * 64,
        )
        self.assertFalse(record["admission"]["prompt_ready"])
        self.assertIn("prompt_not_ready", record["admission"]["blocking_reasons"])

    def test_top_level_prompt_readiness_cannot_conflict_with_admission(self):
        source = bundle_record(prompt_ready=False)
        frozen, review_only, exclusions = build_with_formal_freeze([source])

        self.assertEqual(frozen, [])
        self.assertEqual(review_only, [])
        self.assertEqual(len(exclusions), 1)
        self.assertIn("prompt_not_ready", exclusions[0]["review_blockers"])

    def test_compatibility_aliases_are_read_but_output_uses_final_names(self):
        source = bundle_record()
        source["base_id"] = source.pop("base_fact_id")
        source["evidence_level"] = source.pop("evidence_tier")
        record = preparation._path_record(
            source,
            "bf_001",
            Path("/tmp/behavior_input_bundle.jsonl"),
            "0" * 64,
        )
        self.assertEqual(record["base_fact_id"], "bf_001")
        self.assertEqual(record["evidence_tier"], "independent_adjudicated")
        self.assertNotIn("evidence_level", record)

    def test_duplicate_base_fact_id_is_rejected(self):
        rows = [bundle_record(), bundle_record(source_id="source_002")]
        with self.assertRaisesRegex(ValueError, "duplicate base_fact_id"):
            build_without_formal_freeze(rows)

    def test_included_record_requires_contract_fields(self):
        source = bundle_record()
        del source["provenance"]
        with self.assertRaisesRegex(ValueError, "provenance must be an object"):
            build_with_formal_freeze([source])

    def test_input_schema_version_must_match_behavior_bundle_contract(self):
        source = bundle_record(schema_version="behavior-input-bundle-draft-v1")
        with self.assertRaisesRegex(ValueError, "unsupported input bundle schema"):
            build_without_formal_freeze([source])


if __name__ == "__main__":
    unittest.main()
