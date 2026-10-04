import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_COMPARISON_DATASET_ID = "unit-test-canonical-v2"


def load_script(module_name, filename):
    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalizer = load_script(
    "finalize_public_benchmark_bundle", "finalize_public_benchmark_bundle.py"
)
review_tool = load_script(
    "review_public_benchmark_bundle_for_finalizer_tests",
    "review_public_benchmark_bundle.py",
)
audit_tool = load_script(
    "audit_public_benchmark_near_duplicates_for_finalizer_tests",
    "audit_public_benchmark_near_duplicates.py",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for item in records
        ),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(root, labels=("a", "b", "c", "d")):
    source = root / "source"
    triples = []
    clusters = []
    queues = []
    behaviors = []
    base_fact_ids = []
    for index, label in enumerate(labels, start=1):
        base_fact_id = f"pbf_fact_{label}"
        candidate_id = f"candidate_{label}"
        input_hash = str(index) * 64
        answer = f"Answer {label.upper()}"
        base_fact_ids.append(base_fact_id)
        triples.append(
            {
                "schema_version": "provisional-factual-triple-v1",
                "base_fact_id": base_fact_id,
                "candidate_id": candidate_id,
                "subject": f"Subject {label.upper()}",
                "relation_raw": "example relation",
                "answer": answer,
                "answer_aliases_en": [answer],
                "canonical_fact": f"Subject {label.upper()} has {answer}.",
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "provenance": {"input_record_sha256": input_hash},
            }
        )
        clusters.append(
            {
                "schema_version": "provisional-base-fact-cluster-v1",
                "base_fact_id": base_fact_id,
                "representative_candidate_id": candidate_id,
                "member_count": 1,
                "members": [
                    {
                        "candidate_id": candidate_id,
                        "input_record_sha256": input_hash,
                        "source_id": f"source_{label}",
                        "source_dataset": "fixture",
                    }
                ],
                "subject": f"Subject {label.upper()}",
                "relation_raw": "example relation",
                "answer": answer,
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
            }
        )
        queues.append(
            {
                "schema_version": "provisional-semantic-review-v1",
                "base_fact_id": base_fact_id,
                "member_candidate_ids": [candidate_id],
                "subject": f"Subject {label.upper()}",
                "relation_raw": "example relation",
                "answer": answer,
                "canonical_fact": f"Subject {label.upper()} has {answer}.",
                "canonical_status": "pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "review_decision": None,
            }
        )
        behaviors.append(
            {
                "schema_version": "factual-perturbation-input-bundle-v1",
                "base_fact_id": base_fact_id,
                "base_id": base_fact_id,
                "candidate_id": candidate_id,
                "source_id": f"source_{label}",
                "source_dataset": "fixture",
                "source_subset": "unit",
                "source_question_en": f"What is answer {label.upper()}?",
                "subject_en": f"Subject {label.upper()}",
                "relation_raw": "example relation",
                "answer_en": answer,
                "answer_type": "entity",
                "answer_aliases_en": [answer, f"{answer} alias"],
                "canonical_fact": f"Subject {label.upper()} has {answer}.",
                "canonical_fact_en": f"Subject {label.upper()} has {answer}.",
                "distractor_candidates": [
                    {
                        "distractor_id": f"dist_{label}",
                        "text_en": f"Wrong {label.upper()}",
                        "answer_en": f"Wrong {label.upper()}",
                        "source_choice_index": 1,
                        "verification_status": "pending_review",
                        "verified": False,
                    }
                ],
                "split_assignment": "development",
                "split_group_id": f"group_{label}",
                "split_status": "provisional_not_frozen",
                "canonical_status": "pending_review",
                "bundle_status": "draft_pending_review",
                "evidence_tier": "provisional_single_model",
                "human_gold": False,
                "provenance": {"input_record_sha256": input_hash},
            }
        )
    write_jsonl(source / "provisional_triples.jsonl", triples)
    write_jsonl(source / "base_fact_clusters.jsonl", clusters)
    write_jsonl(source / "review_queue.jsonl", queues)
    write_jsonl(source / "behavior_input_bundle.jsonl", behaviors)
    comparison_rows = [
        {
                "source_id": "historical_source_a",
                "candidate_id": "historical_candidate_a",
                "source_dataset": "historical_fixture",
                "source_path": "/fixture/historical.jsonl",
                "source_original_index": 0,
                "source_question": behaviors[0]["source_question_en"],
                "canonical_fact": behaviors[0]["canonical_fact_en"],
                "answer": behaviors[0]["answer_en"],
                "source_answer": behaviors[0]["answer_en"],
                "canonical_policy_version": "fixture-canonical-v2",
                "canonical_decision": "keep",
        }
    ]
    comparison_path = source / "comparison_canonical.jsonl"
    write_jsonl(
        comparison_path,
        comparison_rows,
    )
    finalizer.TRUSTED_COMPARISON_CONTRACTS[FIXTURE_COMPARISON_DATASET_ID] = {
        "dataset_role": finalizer.COMPARISON_CANONICAL_DATASET_ROLE,
        "format_id": "canonical-triples-v2",
        "sha256": sha256_file(comparison_path),
        "byte_count": comparison_path.stat().st_size,
        "record_count": len(comparison_rows),
        "policy_versions": ["fixture-canonical-v2"],
        "ordered_record_ids_sha256": finalizer.sha256_value(
            [
                {
                    "source_id": row["source_id"],
                    "candidate_id": row["candidate_id"],
                }
                for row in comparison_rows
            ]
        ),
        "ordered_row_hashes_sha256": finalizer.sha256_value(
            [finalizer.sha256_value(row) for row in comparison_rows]
        ),
    }
    return source, base_fact_ids, {
        "provisional_triples": triples,
        "base_fact_clusters": clusters,
        "review_queue": queues,
        "behavior_bundle": behaviors,
    }


def declare_full_pool(source, output_dir, **kwargs):
    return finalizer.materialize_cohort_universe(
        source_dir=source,
        output_dir=output_dir,
        mode="full-pool",
        universe_label="synthetic-full-pool",
        selection_policy_id="fixture-policy-v1",
        comparison_canonical_path=source / "comparison_canonical.jsonl",
        comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
        **kwargs,
    )


def rewrite_bound_universe_items(manifest_path, items):
    manifest = read_json(manifest_path)
    items_path = Path(manifest["items"]["path"])
    write_jsonl(items_path, items)
    manifest["items"]["sha256"] = sha256_file(items_path)
    manifest["items"]["byte_count"] = items_path.stat().st_size
    manifest["items"]["record_count"] = len(items)
    manifest["record_count"] = len(items)
    manifest["ordered_cohort_item_ids_sha256"] = finalizer.sha256_value(
        [item["cohort_item_id"] for item in items]
    )
    manifest["ordered_source_base_fact_ids_sha256"] = finalizer.sha256_value(
        [item["source_base_fact_id"] for item in items]
    )
    manifest["ordered_item_row_hashes_sha256"] = finalizer.sha256_value(
        [finalizer.sha256_value(item) for item in items]
    )
    write_json(manifest_path, manifest)


def valid_accept(template):
    decision = copy.deepcopy(template)
    decision["decision"] = "accept"
    decision["review_provenance"] = {
        "reviewer_type": "codex_proxy",
        "reviewer_id": "codex-fixture",
        "review_method": "full_member_semantic_review_v1",
        "reviewed_at": "2026-09-13T00:00:00Z",
    }
    for field in ("member_reviews", "alias_reviews", "distractor_reviews"):
        for target in decision[field]:
            target["decision"] = "accept"
    decision["notes"] = "Synthetic fixture review."
    return decision


def nonaccept_decision(template, outcome):
    decision = copy.deepcopy(template)
    decision["decision"] = outcome
    decision["review_provenance"] = {
        "reviewer_type": "codex_proxy",
        "reviewer_id": "codex-fixture",
        "review_method": "triage_v1",
        "reviewed_at": "2026-09-13T00:00:00Z",
    }
    for member in decision["member_reviews"]:
        member["decision"] = "accept"
    return decision


class FinalizePublicBenchmarkBundleTests(unittest.TestCase):
    def test_full_pool_binds_all_sources_rows_members_and_never_authorizes_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base_fact_ids, rows = make_source(root, labels=("a", "b"))
            output_dir = root / "universe"
            result = declare_full_pool(source, output_dir)
            manifest_path = Path(result["manifest_path"])
            items_path = Path(result["items_path"])
            manifest = read_json(manifest_path)
            items = read_jsonl(items_path)

            self.assertEqual(
                [item["source_base_fact_id"] for item in items], base_fact_ids
            )
            self.assertEqual(
                [item["selection_index"] for item in items], [0, 1]
            )
            self.assertFalse(manifest["canonical_freeze_authorized"])
            self.assertFalse(manifest["split_freeze_authorized"])
            self.assertFalse(manifest["review_complete"])
            self.assertTrue(
                manifest["selection_policy"][
                    "review_sample_inference_role_not_inherited"
                ]
            )
            comparison_path = source / "comparison_canonical.jsonl"
            self.assertEqual(
                manifest["comparison_canonical"]["sha256"],
                sha256_file(comparison_path),
            )
            self.assertEqual(manifest["comparison_canonical"]["record_count"], 1)
            self.assertEqual(
                manifest["comparison_canonical"]["dataset_role"],
                finalizer.COMPARISON_CANONICAL_DATASET_ROLE,
            )
            self.assertEqual(
                manifest["comparison_canonical"]["dataset_id"],
                FIXTURE_COMPARISON_DATASET_ID,
            )
            self.assertEqual(
                manifest["comparison_canonical"]["format_id"],
                "canonical-triples-v2",
            )
            self.assertEqual(
                manifest["comparison_canonical"]["policy_versions"],
                ["fixture-canonical-v2"],
            )
            comparison_rows = read_jsonl(comparison_path)
            self.assertEqual(
                manifest["comparison_canonical"]["ordered_record_ids_sha256"],
                finalizer.sha256_value(
                    [
                        {
                            "source_id": row["source_id"],
                            "candidate_id": row["candidate_id"],
                        }
                        for row in comparison_rows
                    ]
                ),
            )
            self.assertEqual(
                manifest["comparison_canonical"]["ordered_row_hashes_sha256"],
                finalizer.sha256_value(
                    [finalizer.sha256_value(row) for row in comparison_rows]
                ),
            )
            self.assertEqual(
                manifest["near_duplicate_audit_contract"],
                finalizer.build_near_audit_contract(
                    question_threshold=0.80,
                    fact_threshold=0.80,
                    max_bucket_neighbors=24,
                    max_examples=20,
                ),
            )
            self.assertEqual(
                manifest["historical_exposure_contract_status"],
                "not_predeclared",
            )
            self.assertEqual(set(manifest["source_artifacts"]), set(finalizer.SOURCE_ARTIFACTS))
            for role, binding in manifest["source_artifacts"].items():
                self.assertEqual(binding["sha256"], sha256_file(Path(binding["path"])))
                self.assertEqual(binding["record_count"], len(rows[role]))

            indexes = {
                role: {
                    row[finalizer.SOURCE_ARTIFACTS[role][2]]: row
                    for row in role_rows
                }
                for role, role_rows in rows.items()
            }
            for item in items:
                base_fact_id = item["source_base_fact_id"]
                self.assertEqual(
                    item["source_row_bindings"],
                    {
                        "behavior_row_sha256": finalizer.sha256_value(
                            indexes["behavior_bundle"][base_fact_id]
                        ),
                        "review_queue_row_sha256": finalizer.sha256_value(
                            indexes["review_queue"][base_fact_id]
                        ),
                        "base_fact_cluster_row_sha256": finalizer.sha256_value(
                            indexes["base_fact_clusters"][base_fact_id]
                        ),
                    },
                )
                member = item["member_bindings"][0]
                self.assertEqual(
                    member["triple_record_sha256"],
                    finalizer.sha256_value(
                        indexes["provisional_triples"][member["candidate_id"]]
                    ),
                )

            original_manifest = manifest_path.read_bytes()
            original_items = items_path.read_bytes()
            with self.assertRaises(FileExistsError):
                declare_full_pool(source, output_dir)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertEqual(items_path.read_bytes(), original_items)

            manifest_path.unlink()
            items_path.unlink()
            repeated = declare_full_pool(source, output_dir)
            self.assertEqual(Path(repeated["manifest_path"]).read_bytes(), original_manifest)
            self.assertEqual(Path(repeated["items_path"]).read_bytes(), original_items)

    def test_loader_rebuilds_items_and_universe_id_after_resigned_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_source(root, labels=("a", "b"))
            result = declare_full_pool(source, root / "universe")
            manifest_path = Path(result["manifest_path"])
            items_path = Path(result["items_path"])
            original_manifest = manifest_path.read_bytes()
            original_items = items_path.read_bytes()

            cases = (
                (
                    "source row binding",
                    lambda items: items[0]["source_row_bindings"].__setitem__(
                        "behavior_row_sha256", "0" * 64
                    ),
                ),
                (
                    "member binding",
                    lambda items: items[0]["member_bindings"][0].__setitem__(
                        "triple_record_sha256", "0" * 64
                    ),
                ),
            )
            for label, mutate in cases:
                with self.subTest(label=label):
                    manifest_path.write_bytes(original_manifest)
                    items_path.write_bytes(original_items)
                    tampered_items = read_jsonl(items_path)
                    mutate(tampered_items)
                    rewrite_bound_universe_items(manifest_path, tampered_items)
                    with self.assertRaisesRegex(
                        ValueError, "does not match its source artifacts"
                    ):
                        finalizer.load_cohort_universe(manifest_path)

            manifest_path.write_bytes(original_manifest)
            items_path.write_bytes(original_items)
            tampered_manifest = read_json(manifest_path)
            tampered_manifest["universe_label"] = "tampered-label"
            write_json(manifest_path, tampered_manifest)
            with self.assertRaisesRegex(
                ValueError, "universe_id does not match its declaration"
            ):
                finalizer.load_cohort_universe(manifest_path)

            manifest_path.write_bytes(original_manifest)
            items_path.write_bytes(original_items)
            completed_manifest = read_json(manifest_path)
            completed_manifest["review_complete"] = True
            write_json(manifest_path, completed_manifest)
            with self.assertRaisesRegex(ValueError, "cannot claim review completion"):
                finalizer.load_cohort_universe(manifest_path)

            manifest_path.write_bytes(original_manifest)
            items_path.write_bytes(original_items)
            comparison_path = source / "comparison_canonical.jsonl"
            original_comparison = comparison_path.read_bytes()
            comparison_path.write_bytes(original_comparison + b"\n")
            with self.assertRaisesRegex(
                ValueError, "trusted dataset contract|binding is stale"
            ):
                finalizer.load_cohort_universe(manifest_path)
            comparison_path.write_bytes(original_comparison)

            loaded_manifest, loaded_items, _ = finalizer.load_cohort_universe(
                manifest_path
            )
            self.assertEqual(loaded_manifest["universe_id"], result["universe_id"])
            self.assertEqual(len(loaded_items), 2)

    def test_comparison_canonical_content_contract_and_source_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, rows = make_source(root, labels=("a",))
            comparison_path = source / "comparison_canonical.jsonl"
            original_rows = read_jsonl(comparison_path)

            missing_identity = copy.deepcopy(original_rows)
            missing_identity[0].pop("canonical_policy_version")
            write_jsonl(comparison_path, missing_identity)
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                declare_full_pool(source, root / "missing-field-universe")

            invalid_decision = copy.deepcopy(original_rows)
            invalid_decision[0]["canonical_decision"] = "reject"
            write_jsonl(comparison_path, invalid_decision)
            with self.assertRaisesRegex(ValueError, "canonical_decision must be keep"):
                declare_full_pool(source, root / "invalid-decision-universe")

            mixed_policy = copy.deepcopy(original_rows)
            second = copy.deepcopy(mixed_policy[0])
            second["source_id"] = "historical_source_b"
            second["candidate_id"] = "historical_candidate_b"
            second["canonical_policy_version"] = "other-canonical-v2"
            mixed_policy.append(second)
            write_jsonl(comparison_path, mixed_policy)
            with self.assertRaisesRegex(ValueError, "exactly one non-empty"):
                declare_full_pool(source, root / "mixed-policy-universe")

            behavior = rows["behavior_bundle"][0]
            self_derived_path = root / "self-derived-comparison.jsonl"
            write_jsonl(
                self_derived_path,
                [
                    {
                        "source_id": behavior["source_id"],
                        "candidate_id": behavior["candidate_id"],
                        "source_dataset": behavior["source_dataset"],
                        "source_question": behavior["source_question_en"],
                        "canonical_fact": behavior["canonical_fact_en"],
                        "answer": behavior["answer_en"],
                        "source_answer": behavior["answer_en"],
                        "canonical_policy_version": "self-derived-fake-v2",
                        "canonical_decision": "keep",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "trusted dataset contract"):
                finalizer.materialize_cohort_universe(
                    source_dir=source,
                    output_dir=root / "self-derived-universe",
                    mode="full-pool",
                    universe_label="self-derived",
                    selection_policy_id="fixture-policy-v1",
                    comparison_canonical_path=self_derived_path,
                    comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
                )

            write_jsonl(comparison_path, original_rows)
            with self.assertRaisesRegex(ValueError, "reuse source artifact path"):
                finalizer.materialize_cohort_universe(
                    source_dir=source,
                    output_dir=root / "same-path-universe",
                    mode="full-pool",
                    universe_label="same-path",
                    selection_policy_id="fixture-policy-v1",
                    comparison_canonical_path=source / "behavior_input_bundle.jsonl",
                    comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
                )

            copied_source = root / "copied-source.jsonl"
            copied_source.write_bytes(
                (source / "behavior_input_bundle.jsonl").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "reuse source artifact SHA"):
                finalizer.materialize_cohort_universe(
                    source_dir=source,
                    output_dir=root / "same-sha-universe",
                    mode="full-pool",
                    universe_label="same-sha",
                    selection_policy_id="fixture-policy-v1",
                    comparison_canonical_path=copied_source,
                    comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
                )

    def test_review_scope_preserves_sample_order_and_rejects_stale_scope_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base_fact_ids, rows = make_source(root, labels=("a", "b"))
            sample_path = source / "review_sample.jsonl"
            sample = [
                {
                    **rows["review_queue"][1],
                    "schema_version": "provisional-semantic-review-sample-v1",
                    "sampling": {"sample_index": 0},
                },
                {
                    **rows["review_queue"][0],
                    "schema_version": "provisional-semantic-review-sample-v1",
                    "sampling": {"sample_index": 1},
                },
            ]
            write_jsonl(sample_path, sample)
            scope_result = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=review_tool.source_paths(source),
                review_sample_path=sample_path,
            )
            scope_path = Path(scope_result["manifest_path"])
            result = finalizer.materialize_cohort_universe(
                source_dir=source,
                output_dir=root / "selected-universe",
                mode="review-scope",
                universe_label="synthetic-review-scope",
                selection_policy_id="review-sample-fixture-v1",
                comparison_canonical_path=source / "comparison_canonical.jsonl",
                comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
                review_scope_manifest_path=scope_path,
            )
            manifest = read_json(Path(result["manifest_path"]))
            items = read_jsonl(Path(result["items_path"]))
            loaded_manifest, loaded_items, _ = finalizer.load_cohort_universe(
                Path(result["manifest_path"])
            )
            self.assertEqual(loaded_manifest, manifest)
            self.assertEqual(loaded_items, items)
            self.assertEqual(
                [item["source_base_fact_id"] for item in items],
                [base_fact_ids[1], base_fact_ids[0]],
            )
            self.assertEqual(manifest["selection_source"]["sha256"], sha256_file(scope_path))
            self.assertEqual(
                manifest["selection_source"]["ordered_base_fact_ids_sha256"],
                finalizer.sha256_value([base_fact_ids[1], base_fact_ids[0]]),
            )

            original_sample = sample_path.read_bytes()
            write_jsonl(sample_path, list(reversed(sample)))
            with self.assertRaisesRegex(ValueError, "stale review sample|sample order"):
                finalizer.materialize_cohort_universe(
                    source_dir=source,
                    output_dir=root / "stale-sample-universe",
                    mode="review-scope",
                    universe_label="stale-sample",
                    selection_policy_id="fixture-policy-v1",
                    comparison_canonical_path=source / "comparison_canonical.jsonl",
                    comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
                    review_scope_manifest_path=scope_path,
                )
            sample_path.write_bytes(original_sample)

            behavior_path = source / "behavior_input_bundle.jsonl"
            original_behavior = behavior_path.read_bytes()
            changed = copy.deepcopy(rows["behavior_bundle"])
            changed[0]["source_question_en"] = "Changed after scope freeze"
            write_jsonl(behavior_path, changed)
            with self.assertRaisesRegex(ValueError, "stale source binding"):
                finalizer.materialize_cohort_universe(
                    source_dir=source,
                    output_dir=root / "stale-source-universe",
                    mode="review-scope",
                    universe_label="stale-source",
                    selection_policy_id="fixture-policy-v1",
                    comparison_canonical_path=source / "comparison_canonical.jsonl",
                    comparison_dataset_id=FIXTURE_COMPARISON_DATASET_ID,
                    review_scope_manifest_path=scope_path,
                )
            behavior_path.write_bytes(original_behavior)

    def test_bool_float_counts_indexes_and_non_object_list_members_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_source(root, labels=("a",))
            universe = declare_full_pool(source, root / "universe")
            manifest_path = Path(universe["manifest_path"])
            items_path = Path(universe["items_path"])
            original_manifest = manifest_path.read_bytes()
            original_items = items_path.read_bytes()

            for invalid_count in (True, 1.0):
                with self.subTest(record_count=invalid_count):
                    manifest_path.write_bytes(original_manifest)
                    invalid_manifest = read_json(manifest_path)
                    invalid_manifest["record_count"] = invalid_count
                    write_json(manifest_path, invalid_manifest)
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        finalizer.load_cohort_universe(manifest_path)

            manifest_path.write_bytes(original_manifest)
            items_path.write_bytes(original_items)
            invalid_items = read_jsonl(items_path)
            invalid_items[0]["selection_index"] = 0.0
            rewrite_bound_universe_items(manifest_path, invalid_items)
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                finalizer.load_cohort_universe(manifest_path)

        for invalid_member in (True, 1.0, "not-an-object"):
            with self.subTest(cluster_member=invalid_member):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source, _, rows = make_source(root, labels=("a",))
                    clusters = copy.deepcopy(rows["base_fact_clusters"])
                    clusters[0]["members"] = [invalid_member]
                    write_jsonl(source / "base_fact_clusters.jsonl", clusters)
                    with self.assertRaisesRegex(ValueError, "must be an object"):
                        declare_full_pool(source, root / "invalid-universe")

    def test_preflight_without_evidence_returns_two_and_emits_no_freeze_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_source(root, labels=("a", "b"))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    finalizer.parser().parse_args(
                        [
                            "declare-universe",
                            "--source-dir",
                            str(source),
                            "--output-dir",
                            str(root / "missing-comparison"),
                            "--mode",
                            "full-pool",
                            "--universe-label",
                            "fixture",
                            "--selection-policy-id",
                            "fixture-v1",
                        ]
                    )
            universe = declare_full_pool(source, root / "universe")
            report_path = root / "reports" / "preflight.json"
            report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=report_path,
            )
            self.assertFalse(report["ready_for_future_finalizer"])
            self.assertFalse(report["reviewed_bundle_finalizer_implemented"])
            self.assertFalse(report["freeze_outputs_emitted"])
            self.assertFalse(report["formal_bundle_emitted"])
            self.assertIn(
                "reviewed_bundle_finalizer_not_implemented",
                report["blocker_codes"],
            )
            self.assertIn("review_apply_manifest_missing", report["blocker_codes"])
            self.assertIn("near_duplicate_audit_missing", report["blocker_codes"])
            self.assertIn("historical_exposure_registry_missing", report["blocker_codes"])
            self.assertIn(
                "historical_exposure_authority_missing", report["blocker_codes"]
            )

            cli_path = root / "reports" / "cli-preflight.json"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = finalizer.main(
                    [
                        "preflight",
                        "--universe-manifest",
                        universe["manifest_path"],
                        "--output",
                        str(cli_path),
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertTrue(cli_path.is_file())
            for name in finalizer.FREEZE_OUTPUT_NAMES:
                self.assertFalse((root / "reports" / name).exists())

            existing_path = root / "reports" / "existing.json"
            existing_path.write_bytes(b"do-not-overwrite\n")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                finalizer.build_preflight_report(
                    universe_manifest_path=Path(universe["manifest_path"]),
                    output_path=existing_path,
                )
            self.assertEqual(existing_path.read_bytes(), b"do-not-overwrite\n")

            casefold_freeze = root / "reports" / "REVIEW_FREEZE_MANIFEST.JSON"
            with self.assertRaisesRegex(ValueError, "formal freeze artifact name"):
                finalizer.build_preflight_report(
                    universe_manifest_path=Path(universe["manifest_path"]),
                    output_path=casefold_freeze,
                )
            self.assertFalse(casefold_freeze.exists())

    def test_partial_review_apply_counts_outcomes_and_stale_apply_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base_fact_ids, rows = make_source(root)
            universe = declare_full_pool(source, root / "universe")
            scope = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=review_tool.source_paths(source),
                base_fact_ids=base_fact_ids,
            )
            export = review_tool.export_scope(
                scope_manifest_path=Path(scope["manifest_path"]),
                output_dir=root / "export",
            )
            export_manifest = read_json(Path(export["manifest_path"]))
            templates = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    Path(
                        export_manifest["artifacts"]["review_decisions_template"][
                            "path"
                        ]
                    )
                )
            }
            decisions = [
                valid_accept(templates[base_fact_ids[0]]),
                nonaccept_decision(templates[base_fact_ids[1]], "defer"),
                nonaccept_decision(templates[base_fact_ids[2]], "revise"),
            ]
            decisions_path = root / "partial-decisions.jsonl"
            write_jsonl(decisions_path, decisions)
            apply_result = review_tool.apply_decisions(
                scope_manifest_path=Path(scope["manifest_path"]),
                export_manifest_path=Path(export["manifest_path"]),
                decisions_path=decisions_path,
                output_dir=root / "apply",
                allow_partial=True,
            )

            report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "partial.json",
                review_apply_manifest_paths=[Path(apply_result["manifest_path"])],
            )
            self.assertEqual(
                report["review_counts"],
                {
                    "universe": 4,
                    "accept": 1,
                    "reject": 0,
                    "defer": 1,
                    "revise": 1,
                    "missing": 1,
                    "incomplete_target_checks": 0,
                },
            )
            self.assertNotIn("review_apply_manifest_invalid", report["blocker_codes"])
            self.assertIn("review_decision_missing", report["blocker_codes"])
            self.assertIn("review_decision_deferred", report["blocker_codes"])
            self.assertIn("revision_lineage_missing", report["blocker_codes"])
            self.assertIn("revision_rereview_missing", report["blocker_codes"])

            universe_manifest = read_json(Path(universe["manifest_path"]))
            revised_id = base_fact_ids[2]
            revised_item = next(
                item
                for item in read_jsonl(Path(universe["items_path"]))
                if item["source_base_fact_id"] == revised_id
            )
            original_behavior = next(
                row
                for row in rows["behavior_bundle"]
                if row["base_fact_id"] == revised_id
            )
            source_review = next(
                row
                for row in read_jsonl(Path(apply_result["staging_path"]))
                if row["base_fact_id"] == revised_id
            )
            revision_payload = {
                field: copy.deepcopy(original_behavior[field])
                for field in finalizer.REVISION_PAYLOAD_FIELDS
            }
            revision_payload["answer_en"] = "Rejected revised answer"
            revision_payload["answer_aliases_en"] = ["Rejected revised answer"]
            revision_payload["canonical_fact"] = (
                "Subject C has Rejected revised answer."
            )
            revision_payload["canonical_fact_en"] = (
                "Subject C has Rejected revised answer."
            )
            revised_record = copy.deepcopy(original_behavior)
            for field in finalizer.REVISION_MUTABLE_FIELDS:
                revised_record[field] = copy.deepcopy(revision_payload[field])
            after_hash = finalizer.sha256_value(revised_record)
            lineage_path = root / "revision-lineage.jsonl"
            rereview_path = root / "revision-rereview.jsonl"
            write_jsonl(
                lineage_path,
                [
                    {
                        "schema_version": finalizer.REVISION_LINEAGE_SCHEMA_VERSION,
                        "source_base_fact_id": revised_id,
                        "universe_id": universe_manifest["universe_id"],
                        "cohort_item_id": revised_item["cohort_item_id"],
                        "revision_status": "semantic_patch_staged_for_rereview",
                        "revised_record_role": (
                            "preflight_reconstruction_not_final_behavior_row"
                        ),
                        "derived_fields_recomputed": False,
                        "source_review_staging_row_sha256": (
                            finalizer.sha256_value(source_review)
                        ),
                        "editor_type": "codex_proxy",
                        "editor_id": "codex-test-editor",
                        "edit_method": "explicit_semantic_payload_patch",
                        "edited_at": "2026-09-14T00:00:00Z",
                        "before_record_sha256": revised_item[
                            "source_row_bindings"
                        ]["behavior_row_sha256"],
                        "after_record_sha256": after_hash,
                        "revision_payload": revision_payload,
                        "changed_fields": sorted(
                            field
                            for field in finalizer.REVISION_MUTABLE_FIELDS
                            if original_behavior.get(field)
                            != revised_record.get(field)
                        ),
                        "revised_record": revised_record,
                    }
                ],
            )
            write_jsonl(
                rereview_path,
                [
                    {
                        "schema_version": finalizer.REREVIEW_DECISION_SCHEMA_VERSION,
                        "source_base_fact_id": revised_id,
                        "universe_id": universe_manifest["universe_id"],
                        "cohort_item_id": revised_item["cohort_item_id"],
                        "revised_record_sha256": after_hash,
                        "decision": "reject",
                        "reviewer_type": "human",
                        "reviewer_id": "human-test-reviewer",
                        "review_method": "full_semantic_rereview",
                        "reviewed_at": "2026-09-14T00:01:00Z",
                        "human_gold": False,
                    }
                ],
            )
            rejected_revision = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "rejected-revision.json",
                review_apply_manifest_paths=[Path(apply_result["manifest_path"])],
                revision_lineage_path=lineage_path,
                revision_rereview_path=rereview_path,
            )
            self.assertNotIn(
                "revision_evidence_invalid", rejected_revision["blocker_codes"]
            )
            self.assertIn(
                "revision_rereview_rejected", rejected_revision["blocker_codes"]
            )
            rejection_blocker = next(
                blocker
                for blocker in rejected_revision["blockers"]
                if blocker["code"] == "revision_rereview_rejected"
            )
            self.assertEqual(rejection_blocker["examples"], [revised_id])
            self.assertEqual(
                rejected_revision["evidence"]["revision_rereview"][
                    "rejected_count"
                ],
                1,
            )

            staging_path = Path(apply_result["staging_path"])
            apply_manifest_path = Path(apply_result["manifest_path"])
            original_staging = staging_path.read_bytes()
            original_apply_manifest = apply_manifest_path.read_bytes()
            staging = read_jsonl(staging_path)
            staging[0]["notes"] = "Tampered after apply manifest creation"
            write_jsonl(staging_path, staging)
            stale = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "stale-apply.json",
                review_apply_manifest_paths=[Path(apply_result["manifest_path"])],
            )
            self.assertIn("review_apply_manifest_invalid", stale["blocker_codes"])
            invalid = next(
                blocker
                for blocker in stale["blockers"]
                if blocker["code"] == "review_apply_manifest_invalid"
            )
            self.assertIn("sha256 is stale", invalid["detail"])

            staging_path.write_bytes(original_staging)
            apply_manifest_path.write_bytes(original_apply_manifest)
            forged_staging = read_jsonl(staging_path)
            forged_staging[0]["notes"] = "Forged but re-signed staging"
            write_jsonl(staging_path, forged_staging)
            forged_manifest = read_json(apply_manifest_path)
            forged_manifest["artifacts"]["review_staging"]["sha256"] = sha256_file(
                staging_path
            )
            write_json(apply_manifest_path, forged_manifest)
            forged = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "forged-apply.json",
                review_apply_manifest_paths=[apply_manifest_path],
            )
            forged_blocker = next(
                blocker
                for blocker in forged["blockers"]
                if blocker["code"] == "review_apply_manifest_invalid"
            )
            self.assertIn("deterministic apply replay", forged_blocker["detail"])

            staging_path.write_bytes(original_staging)
            apply_manifest_path.write_bytes(original_apply_manifest)
            invalid_type_manifest = read_json(apply_manifest_path)
            invalid_type_manifest["allow_partial"] = 1
            write_json(apply_manifest_path, invalid_type_manifest)
            invalid_type = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "invalid-allow-partial.json",
                review_apply_manifest_paths=[apply_manifest_path],
            )
            type_blocker = next(
                blocker
                for blocker in invalid_type["blockers"]
                if blocker["code"] == "review_apply_manifest_invalid"
            )
            self.assertIn("allow_partial must be boolean", type_blocker["detail"])

    def test_review_apply_replays_empty_partial_decisions_as_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base_fact_ids, rows = make_source(root, labels=("a",))
            universe = declare_full_pool(source, root / "universe")
            scope = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=review_tool.source_paths(source),
                base_fact_ids=base_fact_ids,
            )
            export = review_tool.export_scope(
                scope_manifest_path=Path(scope["manifest_path"]),
                output_dir=root / "export",
            )
            decisions_path = root / "empty-decisions.jsonl"
            write_jsonl(decisions_path, [])
            apply_result = review_tool.apply_decisions(
                scope_manifest_path=Path(scope["manifest_path"]),
                export_manifest_path=Path(export["manifest_path"]),
                decisions_path=decisions_path,
                output_dir=root / "apply",
                allow_partial=True,
            )
            report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "report.json",
                review_apply_manifest_paths=[Path(apply_result["manifest_path"])],
            )
            self.assertNotIn(
                "review_apply_manifest_invalid", report["blocker_codes"]
            )
            self.assertIn("review_decision_missing", report["blocker_codes"])
            self.assertEqual(report["review_counts"]["missing"], 1)
            self.assertTrue(
                report["evidence"]["review_apply_manifests"][0][
                    "deterministic_replay"
                ]["performed"]
            )

    def test_composite_review_apply_replays_through_bound_authority_manifest(self):
        from tests import test_build_full_public_benchmark_fact_review_composite as composite_helpers

        composite = composite_helpers.composite
        composite_review_tool = composite_helpers.review_tool
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            composite,
            EXPECTED_TOTAL_COUNT=2,
            EXPECTED_PARENT_COMPLETED_COUNT=1,
            EXPECTED_SUPPLEMENT_COUNT=1,
        ), mock.patch.object(
            composite_review_tool, "FULL_PUBLIC_BENCHMARK_COUNT", 2
        ), mock.patch.object(
            composite_review_tool, "_COMPOSITE_AUTHORITY_TOOL", composite
        ), mock.patch.object(
            finalizer,
            "_load_sibling_module",
            return_value=composite_review_tool,
        ):
            root = Path(directory)
            fixture = composite_helpers.FullFactReviewCompositeTests().make_chain(root)
            authority = composite.build_composite_authority(
                parent_run_manifest_path=fixture["parent_manifest"],
                supplement_run_manifest_path=fixture["supplement_manifest"],
                output_dir=root / "composite",
            )
            applied = composite_review_tool.apply_decisions(
                scope_manifest_path=fixture["scope"],
                export_manifest_path=fixture["export"],
                decisions_path=None,
                composite_authority_manifest_path=Path(authority["manifest_path"]),
                output_dir=root / "apply",
                allow_partial=False,
            )
            first_behavior = read_jsonl(
                fixture["source"] / "behavior_input_bundle.jsonl"
            )[0]
            comparison_rows = [
                {
                    "source_id": "historical_source_composite",
                    "candidate_id": "historical_candidate_composite",
                    "source_dataset": "historical_fixture",
                    "source_path": "/fixture/historical.jsonl",
                    "source_original_index": 0,
                    "source_question": first_behavior["canonical_fact_en"],
                    "canonical_fact": first_behavior["canonical_fact_en"],
                    "answer": first_behavior["answer_en"],
                    "source_answer": first_behavior["answer_en"],
                    "canonical_policy_version": "fixture-canonical-v2",
                    "canonical_decision": "keep",
                }
            ]
            comparison_path = fixture["source"] / "comparison_canonical.jsonl"
            write_jsonl(comparison_path, comparison_rows)
            finalizer.TRUSTED_COMPARISON_CONTRACTS[
                FIXTURE_COMPARISON_DATASET_ID
            ] = {
                "dataset_role": finalizer.COMPARISON_CANONICAL_DATASET_ROLE,
                "format_id": "canonical-triples-v2",
                "sha256": sha256_file(comparison_path),
                "byte_count": comparison_path.stat().st_size,
                "record_count": 1,
                "policy_versions": ["fixture-canonical-v2"],
                "ordered_record_ids_sha256": finalizer.sha256_value(
                    [
                        {
                            "source_id": comparison_rows[0]["source_id"],
                            "candidate_id": comparison_rows[0]["candidate_id"],
                        }
                    ]
                ),
                "ordered_row_hashes_sha256": finalizer.sha256_value(
                    [finalizer.sha256_value(comparison_rows[0])]
                ),
            }
            universe = declare_full_pool(fixture["source"], root / "universe")
            report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "report.json",
                review_apply_manifest_paths=[Path(applied["manifest_path"])],
            )
            self.assertNotIn(
                "review_apply_manifest_invalid", report["blocker_codes"]
            )
            evidence = report["evidence"]["review_apply_manifests"][0]
            self.assertEqual(
                evidence["decision_authority"]["sha256"],
                composite.sha256_file(Path(authority["manifest_path"])),
            )

    def test_historical_checks_require_exact_universe_coverage_and_source_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_source(root, labels=("a", "b"))
            universe = declare_full_pool(source, root / "universe")
            universe_manifest = Path(universe["manifest_path"])
            items = read_jsonl(Path(universe["items_path"]))
            historical_dir = root / "historical"
            exposure_source = historical_dir / "prior-exposure.jsonl"
            write_jsonl(exposure_source, [{"prompt": "previously evaluated prompt"}])
            exposure_binding = {
                "exposure_source_id": "prior_eval_fixture",
                "path": str(exposure_source),
                "sha256": sha256_file(exposure_source),
                "byte_count": exposure_source.stat().st_size,
            }
            registry_path = historical_dir / "registry.json"
            registry = {
                "schema_version": finalizer.HISTORICAL_REGISTRY_SCHEMA_VERSION,
                "universe_id": items[0]["universe_id"],
                "inventory_status": "complete_attested",
                "source_count": 1,
                "sources_sha256": finalizer.sha256_value([exposure_binding]),
                "sources": [exposure_binding],
            }
            write_json(registry_path, registry)
            registry_sha = sha256_file(registry_path)
            checks_path = root / "historical" / "checks.jsonl"

            write_jsonl(checks_path, [])
            empty = finalizer.build_preflight_report(
                universe_manifest_path=universe_manifest,
                output_path=root / "reports" / "historical-empty.json",
                historical_exposure_registry_path=registry_path,
                historical_exposure_checks_path=checks_path,
            )
            self.assertIn(
                "historical_exposure_checks_invalid", empty["blocker_codes"]
            )
            empty_blocker = next(
                blocker
                for blocker in empty["blockers"]
                if blocker["code"] == "historical_exposure_checks_invalid"
            )
            self.assertIn("Input JSONL is empty", empty_blocker["detail"])

            checks = [
                {
                    "schema_version": finalizer.HISTORICAL_CHECK_SCHEMA_VERSION,
                    "universe_id": item["universe_id"],
                    "cohort_item_id": item["cohort_item_id"],
                    "source_base_fact_id": item["source_base_fact_id"],
                    "source_row_bindings": item["source_row_bindings"],
                    "registry_sha256": registry_sha,
                    "source_results": [
                        {
                            "exposure_source_id": "prior_eval_fixture",
                            "source_sha256": exposure_binding["sha256"],
                            "status": "resolved",
                            "historically_exposed": False,
                        }
                    ],
                    "status": "resolved",
                    "historically_exposed": False,
                }
                for item in items
            ]
            write_jsonl(checks_path, checks)
            valid = finalizer.build_preflight_report(
                universe_manifest_path=universe_manifest,
                output_path=root / "reports" / "historical-valid.json",
                historical_exposure_registry_path=registry_path,
                historical_exposure_checks_path=checks_path,
            )
            self.assertNotIn(
                "historical_exposure_checks_invalid", valid["blocker_codes"]
            )
            self.assertIn(
                "historical_exposure_authority_missing", valid["blocker_codes"]
            )
            self.assertEqual(
                valid["evidence"]["historical_exposure_checks"]["record_count"],
                len(items),
            )
            self.assertFalse(
                valid["evidence"]["historical_exposure_registry"][
                    "inventory_completeness_locally_provable"
                ]
            )
            self.assertEqual(
                valid["evidence"]["historical_exposure_registry"][
                    "verification_level"
                ],
                "runtime_self_attested_not_predeclared",
            )
            self.assertFalse(
                valid["evidence"]["historical_exposure_registry"][
                    "can_authorize_finalization"
                ]
            )
            self.assertIn(
                "external_authority",
                valid["trust_boundaries"][
                    "historical_exposure_inventory_completeness"
                ],
            )

            original_registry = registry_path.read_bytes()
            bad_digest_registry = read_json(registry_path)
            bad_digest_registry["sources_sha256"] = "0" * 64
            write_json(registry_path, bad_digest_registry)
            with self.assertRaisesRegex(ValueError, "sources_sha256 mismatch"):
                finalizer._validate_historical_registry(
                    registry_path, items[0]["universe_id"]
                )
            registry_path.write_bytes(original_registry)

            bool_count_registry = read_json(registry_path)
            bool_count_registry["source_count"] = True
            write_json(registry_path, bool_count_registry)
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                finalizer._validate_historical_registry(
                    registry_path, items[0]["universe_id"]
                )
            registry_path.write_bytes(original_registry)

            stale = copy.deepcopy(checks)
            stale[0]["source_row_bindings"]["behavior_row_sha256"] = "0" * 64
            write_jsonl(checks_path, stale)
            invalid = finalizer.build_preflight_report(
                universe_manifest_path=universe_manifest,
                output_path=root / "reports" / "historical-stale.json",
                historical_exposure_registry_path=registry_path,
                historical_exposure_checks_path=checks_path,
            )
            self.assertIn(
                "historical_exposure_checks_invalid", invalid["blocker_codes"]
            )

            stale_registry_check = copy.deepcopy(checks)
            stale_registry_check[0]["registry_sha256"] = "0" * 64
            write_jsonl(checks_path, stale_registry_check)
            stale_registry_report = finalizer.build_preflight_report(
                universe_manifest_path=universe_manifest,
                output_path=root / "reports" / "historical-stale-registry.json",
                historical_exposure_registry_path=registry_path,
                historical_exposure_checks_path=checks_path,
            )
            self.assertIn(
                "historical_exposure_checks_invalid",
                stale_registry_report["blocker_codes"],
            )

            exposed = copy.deepcopy(checks)
            exposed[0]["source_results"][0]["historically_exposed"] = True
            exposed[0]["historically_exposed"] = True
            write_jsonl(checks_path, exposed)
            exposed_report = finalizer.build_preflight_report(
                universe_manifest_path=universe_manifest,
                output_path=root / "reports" / "historical-exposed.json",
                historical_exposure_registry_path=registry_path,
                historical_exposure_checks_path=checks_path,
            )
            exposed_blocker = next(
                blocker
                for blocker in exposed_report["blockers"]
                if blocker["code"] == "historical_exposure_detected"
            )
            self.assertEqual(exposed_blocker["count"], 1)
            self.assertEqual(exposed_blocker["examples"], [items[0]["cohort_item_id"]])

            empty_registry = copy.deepcopy(registry)
            empty_registry["source_count"] = 0
            empty_registry["sources"] = []
            empty_registry["sources_sha256"] = finalizer.sha256_value([])
            write_json(registry_path, empty_registry)
            empty_registry_report = finalizer.build_preflight_report(
                universe_manifest_path=universe_manifest,
                output_path=root / "reports" / "historical-empty-registry.json",
                historical_exposure_registry_path=registry_path,
                historical_exposure_checks_path=checks_path,
            )
            self.assertIn(
                "historical_exposure_registry_invalid",
                empty_registry_report["blocker_codes"],
            )

    def test_completion_attestations_are_presence_only_and_cannot_make_ready_true(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = make_source(root, labels=("a",))
            universe = declare_full_pool(source, root / "universe")
            universe_manifest = read_json(Path(universe["manifest_path"]))
            attestation_path = root / "semantic-closure.json"
            write_json(
                attestation_path,
                {
                    "schema_version": finalizer.SEMANTIC_CLOSURE_SCHEMA_VERSION,
                    "universe_id": universe_manifest["universe_id"],
                    "status": "complete",
                    "unresolved_count": 0,
                },
            )
            report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "presence-only.json",
                semantic_closure_attestation_path=attestation_path,
            )
            evidence = report["evidence"]["semantic_closure_attestation"]
            self.assertEqual(
                evidence["verification_level"], "presence_only_self_attested"
            )
            self.assertFalse(evidence["can_authorize_finalization"])
            self.assertFalse(report["ready_for_future_finalizer"])
            self.assertIn(
                "reviewed_bundle_finalizer_not_implemented",
                report["blocker_codes"],
            )

    def test_revision_rereview_human_gold_must_be_a_strict_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base_fact_ids, rows = make_source(root, labels=("a",))
            universe = declare_full_pool(source, root / "universe")
            manifest = read_json(Path(universe["manifest_path"]))
            item = read_jsonl(Path(universe["items_path"]))[0]
            lineage_path = root / "revision-lineage.jsonl"
            rereview_path = root / "revision-rereview.jsonl"
            original_behavior = rows["behavior_bundle"][0]
            revision_payload = {
                field: copy.deepcopy(original_behavior[field])
                for field in finalizer.REVISION_PAYLOAD_FIELDS
            }
            revision_payload["answer_en"] = "Revised answer"
            revision_payload["answer_aliases_en"] = [
                "Revised answer",
                "Revised answer alias",
            ]
            revision_payload["canonical_fact"] = "Subject A has Revised answer."
            revision_payload["canonical_fact_en"] = "Subject A has Revised answer."
            revised_record = copy.deepcopy(original_behavior)
            for field in finalizer.REVISION_MUTABLE_FIELDS:
                revised_record[field] = copy.deepcopy(revision_payload[field])
            after_hash = finalizer.sha256_value(revised_record)
            source_review = {
                "base_fact_id": base_fact_ids[0],
                "review_outcome": "revise",
                "reviewer_type": "codex_proxy",
                "human_gold": False,
            }
            lineage = {
                "schema_version": finalizer.REVISION_LINEAGE_SCHEMA_VERSION,
                "source_base_fact_id": base_fact_ids[0],
                "universe_id": manifest["universe_id"],
                "cohort_item_id": item["cohort_item_id"],
                "revision_status": "semantic_patch_staged_for_rereview",
                "revised_record_role": "preflight_reconstruction_not_final_behavior_row",
                "derived_fields_recomputed": False,
                "source_review_staging_row_sha256": finalizer.sha256_value(
                    source_review
                ),
                "editor_type": "codex_proxy",
                "editor_id": "codex-test-editor",
                "edit_method": "explicit_semantic_payload_patch",
                "edited_at": "2026-09-14T00:00:00Z",
                "before_record_sha256": item["source_row_bindings"][
                    "behavior_row_sha256"
                ],
                "after_record_sha256": after_hash,
                "revision_payload": revision_payload,
                "changed_fields": sorted(
                    [
                        "answer_aliases_en",
                        "answer_en",
                        "canonical_fact",
                        "canonical_fact_en",
                    ]
                ),
                "revised_record": revised_record,
            }
            write_jsonl(
                lineage_path,
                [lineage],
            )
            rereview = {
                "schema_version": finalizer.REREVIEW_DECISION_SCHEMA_VERSION,
                "source_base_fact_id": base_fact_ids[0],
                "universe_id": manifest["universe_id"],
                "cohort_item_id": item["cohort_item_id"],
                "revised_record_sha256": after_hash,
                "decision": "accept",
                "reviewer_type": "human",
                "reviewer_id": "human-test-reviewer",
                "review_method": "full_semantic_rereview",
                "reviewed_at": "2026-09-14T00:01:00Z",
                "human_gold": None,
            }
            write_jsonl(rereview_path, [rereview])
            with self.assertRaisesRegex(ValueError, "human_gold must be boolean"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            rereview["human_gold"] = False
            write_jsonl(rereview_path, [rereview])
            _, _, rejected = finalizer._validate_revision_evidence(
                lineage_path,
                rereview_path,
                universe_id=manifest["universe_id"],
                revised_source_ids=base_fact_ids,
                item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )
            self.assertEqual(rejected, [])

            rejected_rereview = copy.deepcopy(rereview)
            rejected_rereview["decision"] = "reject"
            write_jsonl(rereview_path, [rejected_rereview])
            _, _, rejected = finalizer._validate_revision_evidence(
                lineage_path,
                rereview_path,
                universe_id=manifest["universe_id"],
                revised_source_ids=base_fact_ids,
                item_by_source_id={base_fact_ids[0]: item},
                source_behavior_by_id={base_fact_ids[0]: original_behavior},
                source_review_by_id={base_fact_ids[0]: source_review},
            )
            self.assertEqual(rejected, base_fact_ids)
            write_jsonl(rereview_path, [rereview])

            wrong_source_review_hash = copy.deepcopy(lineage)
            wrong_source_review_hash["source_review_staging_row_sha256"] = "0" * 64
            write_jsonl(lineage_path, [wrong_source_review_hash])
            with self.assertRaisesRegex(ValueError, "source review staging hash"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            invalid_editor_id = copy.deepcopy(lineage)
            invalid_editor_id["editor_id"] = []
            write_jsonl(lineage_path, [invalid_editor_id])
            with self.assertRaisesRegex(ValueError, "editor_id must be a non-empty string"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            wrong_before = copy.deepcopy(lineage)
            wrong_before["before_record_sha256"] = "0" * 64
            write_jsonl(lineage_path, [wrong_before])
            with self.assertRaisesRegex(ValueError, "not bound to the universe item"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            wrong_revised = copy.deepcopy(lineage)
            wrong_revised["revised_record"]["answer_en"] = "Tampered revision"
            write_jsonl(lineage_path, [wrong_revised])
            with self.assertRaisesRegex(ValueError, "not the source plus semantic payload"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            wrong_changed_fields = copy.deepcopy(lineage)
            wrong_changed_fields["changed_fields"] = ["answer_en"]
            write_jsonl(lineage_path, [wrong_changed_fields])
            with self.assertRaisesRegex(ValueError, "do not match revised_record"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            malformed_aliases = copy.deepcopy(lineage)
            malformed_aliases["revision_payload"]["answer_aliases_en"] = {
                "malformed": True
            }
            write_jsonl(lineage_path, [malformed_aliases])
            with self.assertRaisesRegex(ValueError, "must be a non-empty list"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            unicode_duplicate_aliases = copy.deepcopy(lineage)
            unicode_duplicate_aliases["revision_payload"]["answer_aliases_en"] = [
                "Ｒｅｖｉｓｅｄ　ａｎｓｗｅｒ",
                "Revised answer",
            ]
            write_jsonl(lineage_path, [unicode_duplicate_aliases])
            with self.assertRaisesRegex(ValueError, "aliases are not unique"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            immutable_change = copy.deepcopy(lineage)
            immutable_change["revised_record"]["source_id"] = "tampered-source"
            write_jsonl(lineage_path, [immutable_change])
            with self.assertRaisesRegex(ValueError, "not the source plus semantic payload"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            extra_lineage_field = copy.deepcopy(lineage)
            extra_lineage_field["canonical_freeze_authorized"] = True
            write_jsonl(lineage_path, [extra_lineage_field])
            with self.assertRaisesRegex(ValueError, "lineage fields are invalid"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            missing_lineage_field = copy.deepcopy(lineage)
            missing_lineage_field.pop("editor_id")
            write_jsonl(lineage_path, [missing_lineage_field])
            with self.assertRaisesRegex(ValueError, "lineage fields are invalid"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            write_jsonl(lineage_path, [lineage])
            extra_rereview_field = copy.deepcopy(rereview)
            extra_rereview_field["review_complete"] = True
            write_jsonl(rereview_path, [extra_rereview_field])
            with self.assertRaisesRegex(ValueError, "rereview fields are invalid"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            self_declared_evidence = copy.deepcopy(rereview)
            self_declared_evidence["evidence_tier"] = "provisional_single_model"
            write_jsonl(rereview_path, [self_declared_evidence])
            with self.assertRaisesRegex(ValueError, "rereview fields are invalid"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            missing_rereview_field = copy.deepcopy(rereview)
            missing_rereview_field.pop("reviewer_id")
            write_jsonl(rereview_path, [missing_rereview_field])
            with self.assertRaisesRegex(ValueError, "rereview fields are invalid"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            invalid_enum = copy.deepcopy(rereview)
            invalid_enum["human_gold"] = False
            invalid_enum["decision"] = []
            write_jsonl(rereview_path, [invalid_enum])
            with self.assertRaisesRegex(ValueError, "not terminal"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            invalid_reviewer_type = copy.deepcopy(rereview)
            invalid_reviewer_type["reviewer_type"] = []
            write_jsonl(rereview_path, [invalid_reviewer_type])
            with self.assertRaisesRegex(ValueError, "reviewer_type is invalid"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

            invalid_reviewer_id = copy.deepcopy(rereview)
            invalid_reviewer_id["reviewer_id"] = 123
            write_jsonl(rereview_path, [invalid_reviewer_id])
            with self.assertRaisesRegex(ValueError, "reviewer_id must be a non-empty string"):
                finalizer._validate_revision_evidence(
                    lineage_path,
                    rereview_path,
                    universe_id=manifest["universe_id"],
                    revised_source_ids=base_fact_ids,
                    item_by_source_id={base_fact_ids[0]: item},
                    source_behavior_by_id={base_fact_ids[0]: original_behavior},
                    source_review_by_id={base_fact_ids[0]: source_review},
                )

    def test_near_audit_binds_candidate_bytes_endpoint_sources_and_adjudications(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base_fact_ids, rows = make_source(root, labels=("a", "b"))
            universe = declare_full_pool(
                source, root / "universe", near_max_examples=7
            )
            comparison_path = source / "comparison_canonical.jsonl"
            near_dir = root / "near"
            summary = audit_tool.audit(
                source / "behavior_input_bundle.jsonl",
                near_dir,
                comparison_canonical_path=comparison_path,
                max_examples=7,
            )
            summary["policy"]["max_examples"] = 7
            summary_path = near_dir / "lexical_audit_summary.json"
            write_json(summary_path, summary)
            candidates_path = near_dir / "lexical_candidate_pairs.jsonl"
            candidates = read_jsonl(candidates_path)
            self.assertTrue(candidates)

            changed_policy_dir = root / "near-changed-policy"
            changed_policy_summary = audit_tool.audit(
                source / "behavior_input_bundle.jsonl",
                changed_policy_dir,
                comparison_canonical_path=comparison_path,
                question_threshold=0.90,
                max_examples=7,
            )
            changed_policy_path = changed_policy_dir / "lexical_audit_summary.json"
            write_json(changed_policy_path, changed_policy_summary)
            changed_policy_report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-changed-policy.json",
                near_audit_summary_path=changed_policy_path,
            )
            changed_policy_blocker = next(
                blocker
                for blocker in changed_policy_report["blockers"]
                if blocker["code"] == "near_duplicate_audit_invalid"
            )
            self.assertIn("predeclared universe contract", changed_policy_blocker["detail"])

            missing = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-missing.json",
                near_audit_summary_path=summary_path,
            )
            self.assertIn(
                "near_duplicate_adjudications_missing", missing["blocker_codes"]
            )
            self.assertEqual(
                missing["evidence"]["near_duplicate_audit"]["candidate_pair_count"],
                len(candidates),
            )

            adjudications_path = root / "near" / "adjudications.jsonl"
            write_jsonl(
                adjudications_path,
                [
                    {
                        "schema_version": finalizer.NEAR_ADJUDICATION_SCHEMA_VERSION,
                        "pair_id": pair["pair_id"],
                        "candidate_row_sha256": finalizer.sha256_value(pair),
                        "left_audit_record_id": pair["left"]["provenance"][
                            "audit_record_id"
                        ],
                        "left_input_record_sha256": pair["left"]["provenance"][
                            "input_record_sha256"
                        ],
                        "right_audit_record_id": pair["right"]["provenance"][
                            "audit_record_id"
                        ],
                        "right_input_record_sha256": pair["right"]["provenance"][
                            "input_record_sha256"
                        ],
                        "decision": (
                            "same_fact"
                            if finalizer.EXACT_DUPLICATE_MATCH_TYPES.intersection(
                                pair["match_types"]
                            )
                            else "same_leakage_component"
                        ),
                        "reviewer_type": "codex_proxy",
                        "human_gold": False,
                    }
                    for pair in candidates
                ],
            )
            valid = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-valid.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            self.assertNotIn(
                "near_duplicate_adjudications_missing", valid["blocker_codes"]
            )
            self.assertNotIn(
                "near_duplicate_adjudications_invalid", valid["blocker_codes"]
            )

            exact_index = next(
                index
                for index, pair in enumerate(candidates)
                if finalizer.EXACT_DUPLICATE_MATCH_TYPES.intersection(
                    pair["match_types"]
                )
            )
            invalid_adjudications = read_jsonl(adjudications_path)
            invalid_adjudications[exact_index]["decision"] = "distinct"
            write_jsonl(adjudications_path, invalid_adjudications)
            invalid_exact = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-exact-distinct.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            exact_blocker = next(
                blocker
                for blocker in invalid_exact["blockers"]
                if blocker["code"] == "near_duplicate_adjudications_invalid"
            )
            self.assertIn("cannot be adjudicated distinct", exact_blocker["detail"])

            invalid_adjudications[exact_index]["decision"] = []
            write_jsonl(adjudications_path, invalid_adjudications)
            invalid_enum = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-invalid-enum.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            enum_blocker = next(
                blocker
                for blocker in invalid_enum["blockers"]
                if blocker["code"] == "near_duplicate_adjudications_invalid"
            )
            self.assertIn("unresolved", enum_blocker["detail"])

            original_candidates = candidates_path.read_bytes()
            original_summary = summary_path.read_bytes()
            tampered_candidates = copy.deepcopy(candidates)
            tampered_candidates[0]["cross_split"] = not tampered_candidates[0][
                "cross_split"
            ]
            write_jsonl(candidates_path, tampered_candidates)
            resigned_summary = read_json(summary_path)
            resigned_summary["artifacts"]["candidate_pairs"]["sha256"] = sha256_file(
                candidates_path
            )
            write_json(summary_path, resigned_summary)
            stale_candidates = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-stale-candidates.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            self.assertIn("near_duplicate_audit_invalid", stale_candidates["blocker_codes"])
            candidates_path.write_bytes(original_candidates)
            summary_path.write_bytes(original_summary)

            missing_parameter = read_json(summary_path)
            del missing_parameter["policy"]["max_examples"]
            write_json(summary_path, missing_parameter)
            missing_parameter_report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-missing-max-examples.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            self.assertIn(
                "near_duplicate_audit_invalid",
                missing_parameter_report["blocker_codes"],
            )
            summary_path.write_bytes(original_summary)

            bool_policy = read_json(summary_path)
            bool_policy["policy"]["max_bucket_neighbors"] = True
            write_json(summary_path, bool_policy)
            bool_policy_report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-bool-policy.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            self.assertIn(
                "near_duplicate_audit_invalid", bool_policy_report["blocker_codes"]
            )
            summary_path.write_bytes(original_summary)

            missing_comparison = read_json(summary_path)
            missing_comparison["input"]["comparison_canonical"] = None
            write_json(summary_path, missing_comparison)
            missing_comparison_report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-missing-comparison.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            self.assertIn(
                "near_duplicate_audit_invalid",
                missing_comparison_report["blocker_codes"],
            )

            summary_path.write_bytes(original_summary)
            alternate_comparison = root / "alternate-comparison.jsonl"
            alternate_comparison.write_bytes(comparison_path.read_bytes())
            unbound_comparison = read_json(summary_path)
            unbound_comparison["input"]["comparison_canonical"]["path"] = str(
                alternate_comparison
            )
            write_json(summary_path, unbound_comparison)
            unbound_report = finalizer.build_preflight_report(
                universe_manifest_path=Path(universe["manifest_path"]),
                output_path=root / "reports" / "near-unbound-comparison.json",
                near_audit_summary_path=summary_path,
                near_adjudications_path=adjudications_path,
            )
            unbound_blocker = next(
                blocker
                for blocker in unbound_report["blockers"]
                if blocker["code"] == "near_duplicate_audit_invalid"
            )
            self.assertIn("differs from universe", unbound_blocker["detail"])


if __name__ == "__main__":
    unittest.main()
