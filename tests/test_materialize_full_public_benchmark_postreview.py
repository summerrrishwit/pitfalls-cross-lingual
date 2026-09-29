import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "materialize_full_public_benchmark_postreview.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "materialize_full_public_benchmark_postreview", SCRIPT_PATH
)
postreview = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = postreview
SCRIPT_SPEC.loader.exec_module(postreview)

REVIEW_TEST_PATH = PROJECT_ROOT / "tests" / "test_review_public_benchmark_bundle.py"
REVIEW_TEST_SPEC = importlib.util.spec_from_file_location(
    "postreview_review_fixture_helpers", REVIEW_TEST_PATH
)
review_fixtures = importlib.util.module_from_spec(REVIEW_TEST_SPEC)
assert REVIEW_TEST_SPEC and REVIEW_TEST_SPEC.loader
REVIEW_TEST_SPEC.loader.exec_module(review_fixtures)

FREEZER_PATH = PROJECT_ROOT / "scripts" / "freeze_full_public_benchmark_pre_exact_hf.py"
FREEZER_SPEC = importlib.util.spec_from_file_location(
    "postreview_pre_exact_hf_freezer", FREEZER_PATH
)
freezer = importlib.util.module_from_spec(FREEZER_SPEC)
assert FREEZER_SPEC and FREEZER_SPEC.loader
sys.modules[FREEZER_SPEC.name] = freezer
FREEZER_SPEC.loader.exec_module(freezer)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SemanticReviewRouter:
    def request_json(self, _stage, _item_id, _model_spec, prompt, validator):
        payload = postreview.semantic_runner._prompt_payload(prompt)
        records = []
        for pair in payload["pairs"]:
            same = (
                pair["left"]["canonical_fact_en"]
                == pair["right"]["canonical_fact_en"]
            )
            alias_required = "answer_alias_exact" in pair["match_types"]
            records.append(
                {
                    "pair_id": pair["pair_id"],
                    "relationship": "same_fact" if same else "distinct",
                    "alias_valid": same if alias_required else None,
                    "same_answer_entity": same,
                    "same_subject_entity": same,
                    "relation_semantics_same": same,
                    "temporal_scope_compatible": same,
                    "answer_compatible": same,
                    "semantic_duplicate": same,
                    "rationale": (
                        "Fixture endpoints are identical."
                        if same
                        else "Distinct fixture facts."
                    ),
                    "confidence": "high",
                }
            )
        response = {
            "schema_version": postreview.semantic_runner.BATCH_RESPONSE_SCHEMA,
            "records": records,
        }
        validator(response)
        raw = json.dumps(response, ensure_ascii=False, sort_keys=True)
        return SimpleNamespace(
            terminal_status="completed",
            parsed_response=response,
            raw_response=raw,
            response_model=postreview.semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
            usage={},
            latency_ms=1,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "ok", "latency_ms": 1}],
        )


def full_row(base_fact_id, component_id, split, family="family_a"):
    answer = "Answer A"
    return {
        "schema_version": postreview.pre_hf.FULL_FACT_SCHEMA_VERSION,
        "base_fact_id": base_fact_id,
        "candidate_id": f"candidate_{base_fact_id[-1]}1",
        "source_pool_id": "fixture",
        "source_dataset": None,
        "source_subset": None,
        "source_id": None,
        "source_format": None,
        "source_question_en": None,
        "source_choices_en": [],
        "subject_en": "Subject A",
        "answer_en": answer,
        "answer_aliases_en": [answer, f"{answer} alias"],
        "answer_type": None,
        "canonical_fact_en": f"Subject A has {answer}.",
        "prompt_en": None,
        "prompt_quality_tier": None,
        "prompt_risk_flags": [],
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "relation_raw": "example relation",
        "relation_signature_id": "fixture_relation",
        "relation_signature_ids": ["fixture_relation"],
        "relation_normalized": None,
        "probe_relation_candidate": None,
        "probe_relation_id": None,
        "relation_partition_id": family,
        "relation_partition_status": "broad_rule_candidate_partition",
        "answer_type_bucket": "other",
        "leakage_component_id": component_id,
        "split_assignment": split,
        "split_policy_version": "fixture-current-split-v1",
        "split_status": "provisional_not_frozen",
        "translation_status": "source_language_only_or_pending",
        "distractor_status": "candidate_generation_only",
        "hf_model_execution_status": "not_run",
        "hf_tokenizer_execution_status": "not_run",
    }


class FullPublicBenchmarkPostreviewTests(unittest.TestCase):
    def test_unresolved_fact_outcomes_fail_closed_and_reject_is_resolved(self):
        rows = [
            {
                "base_fact_id": "a",
                "subject_en": "A",
                "relation_raw": "r",
                "answer_en": "x",
                "canonical_fact_en": "A r x",
            },
            {
                "base_fact_id": "b",
                "subject_en": "B",
                "relation_raw": "r",
                "answer_en": "y",
                "canonical_fact_en": "B r y",
            },
        ]
        complete = {
            "codex_proxy_scope_review_complete": True,
            "member_review_complete": True,
            "alias_review_complete": True,
            "distractor_review_complete": True,
        }
        staging = {
            "a": {**rows[0], "review_outcome": "accept", "review_completion": complete},
            "b": {**rows[1], "review_outcome": "reject", "review_completion": {}},
        }
        result = postreview.classify_fact_review_staging(
            full_rows=rows, staging_index=staging
        )
        self.assertEqual(result["retained_ids"], ["a"])
        self.assertEqual(result["rejected_ids"], ["b"])

        staging["b"] = {
            **rows[1],
            "review_outcome": "revise",
            "review_completion": {},
        }
        with self.assertRaisesRegex(ValueError, "fail closed"):
            postreview.classify_fact_review_staging(
                full_rows=rows, staging_index=staging
            )

    def test_incomplete_semantic_adjudications_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly cover bounded candidates"):
            postreview._validate_semantic_adjudications(
                candidates=[{"pair_id": "pair_a"}],
                units=[SimpleNamespace(pair_id="pair_a")],
                adjudications=[],
            )

    def test_rejects_semantic_candidates_that_still_use_excluded_bridge(self):
        rows = [
            full_row("pbf_a", "old_a", "development"),
            full_row("pbf_b", "old_b", "sealed"),
            full_row("pbf_c", "old_c", "validation", family="family_b"),
        ]
        components = [
            {
                "schema_version": postreview.pre_hf.COMPONENT_SCHEMA_VERSION,
                "leakage_component_id": row["leakage_component_id"],
                "member_count": 1,
                "base_fact_ids": [row["base_fact_id"]],
                "source_split_group_ids": [f"group_{row['base_fact_id']}"],
                "frozen_semantic_component_ids": [],
                "split_assignment": row["split_assignment"],
            }
            for row in rows
        ]

        def endpoint(base_fact_id):
            return {
                "provenance": {
                    "input_role": "current_full_base",
                    "base_fact_id": base_fact_id,
                    "endpoint_id": f"endpoint_{base_fact_id}",
                }
            }

        candidates = [
            {
                "pair_id": "pair_ad",
                "left": endpoint("pbf_a"),
                "right": endpoint("pbf_b"),
            },
            {
                "pair_id": "pair_dc",
                "left": endpoint("pbf_b"),
                "right": endpoint("pbf_c"),
            },
        ]
        decisions = {
            "pair_ad": {"relationship": "same_leakage_component"},
            "pair_dc": {"relationship": "same_leakage_component"},
        }
        with self.assertRaisesRegex(
            ValueError, "excluded fact remains an endpoint"
        ):
            postreview.build_reviewed_components(
                full_rows=rows,
                retained_ids=["pbf_a", "pbf_c"],
                rejected_ids=["pbf_b"],
                current_components=components,
                semantic_candidates=candidates,
                semantic_decisions=decisions,
                frozen={"split_by_fact": {}, "component_by_fact": {}},
            )

    def _make_source(self, root):
        source = root / "source"
        fact_a = "pbf_fact_a"
        fact_b = "pbf_fact_b"
        triples = [
            review_fixtures.triple(fact_a, "candidate_a1", "1" * 64),
            review_fixtures.triple(fact_b, "candidate_b1", "2" * 64),
        ]
        clusters = [
            review_fixtures.cluster(fact_a, ["candidate_a1"], ["1" * 64]),
            review_fixtures.cluster(fact_b, ["candidate_b1"], ["2" * 64]),
        ]
        queues = [
            review_fixtures.queue(fact_a, ["candidate_a1"]),
            review_fixtures.queue(fact_b, ["candidate_b1"]),
        ]
        bundles = [
            review_fixtures.behavior(fact_a, "candidate_a1"),
            review_fixtures.behavior(fact_b, "candidate_b1"),
        ]
        review_fixtures.write_jsonl(source / "provisional_triples.jsonl", triples)
        review_fixtures.write_jsonl(source / "base_fact_clusters.jsonl", clusters)
        review_fixtures.write_jsonl(source / "review_queue.jsonl", queues)
        review_fixtures.write_jsonl(source / "behavior_input_bundle.jsonl", bundles)
        return source, [fact_a, fact_b]

    def _make_fact_apply(self, root, source, ids):
        scope = postreview.review_tool.create_scope(
            output_dir=root / "scope",
            artifact_paths=postreview.review_tool.source_paths(source),
            base_fact_ids=ids,
        )
        export = postreview.review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "export",
        )
        export_manifest = postreview.review_tool.read_json(Path(export["manifest_path"]))
        templates = postreview.review_tool.read_jsonl(
            Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
        )
        helper = review_fixtures.PublicBenchmarkReviewToolTests()
        decisions = [helper.valid_accept(template) for template in templates]
        decisions_path = root / "fact-decisions.jsonl"
        postreview.review_tool.write_jsonl(decisions_path, decisions)
        applied = postreview.review_tool.apply_decisions(
            scope_manifest_path=Path(scope["manifest_path"]),
            export_manifest_path=Path(export["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=root / "fact-apply",
        )
        return Path(applied["manifest_path"])

    def _make_semantic_inputs(self, root, source, ids, *, empty_candidates=False):
        semantic = postreview.semantic_runner.materializer
        rows_by_role = {
            role: semantic.read_jsonl(source / filename)
            for role, (filename, _, _) in semantic.SOURCE_ARTIFACTS.items()
        }
        source_bindings = {}
        for role, (filename, schema_version, identifier_field) in semantic.SOURCE_ARTIFACTS.items():
            path = source / filename
            rows = rows_by_role[role]
            source_bindings[role] = {
                "path": str(path.resolve()),
                "sha256": semantic.sha256_file(path),
                "byte_count": path.stat().st_size,
                "record_count": len(rows),
                "schema_version": schema_version,
                "record_ids_sha256": semantic.sha256_value(
                    [row[identifier_field] for row in rows]
                ),
                "record_row_hashes_sha256": semantic.sha256_value(
                    [semantic.sha256_value(row) for row in rows]
                ),
            }

        behavior = {row["base_fact_id"]: row for row in rows_by_role["behavior_bundle"]}
        clusters = {row["base_fact_id"]: row for row in rows_by_role["base_fact_clusters"]}
        reviews = {row["base_fact_id"]: row for row in rows_by_role["review_queue"]}
        triples = {row["candidate_id"]: row for row in rows_by_role["provisional_triples"]}
        universe_dir = root / "universe"
        items = []
        for index, base_fact_id in enumerate(ids):
            source_row = behavior[base_fact_id]
            cluster = clusters[base_fact_id]
            items.append(
                {
                    "schema_version": semantic.UNIVERSE_ITEM_SCHEMA,
                    "universe_id": "fixture-universe",
                    "cohort_item_id": f"item_{index}",
                    "selection_index": index,
                    "source_base_fact_id": base_fact_id,
                    "source_id": source_row.get("source_id"),
                    "member_count": len(cluster["members"]),
                    "member_bindings": [
                        {
                            "candidate_id": member["candidate_id"],
                            "source_input_record_sha256": member["input_record_sha256"],
                            "triple_record_sha256": semantic.sha256_value(
                                triples[member["candidate_id"]]
                            ),
                        }
                        for member in cluster["members"]
                    ],
                    "source_row_bindings": {
                        "behavior_row_sha256": semantic.sha256_value(source_row),
                        "review_queue_row_sha256": semantic.sha256_value(
                            reviews[base_fact_id]
                        ),
                        "base_fact_cluster_row_sha256": semantic.sha256_value(cluster),
                    },
                }
            )
        items_path = universe_dir / "formal_cohort_universe_items.jsonl"
        write_jsonl(items_path, items)

        canonical_path = root / "canonical-v2.jsonl"
        canonical_rows = [
            {
                "source_id": "canonical_other",
                "candidate_id": "canonical_other_candidate",
                "source_dataset": "fixture",
                "source_question": "What is unrelated?",
                "canonical_fact": "An unrelated fixture is unrelated.",
                "answer": "Other",
                "source_answer": "Other",
                "canonical_policy_version": "codex-adjudicated-canonical-v2",
                "canonical_decision": "keep",
            }
        ]
        write_jsonl(canonical_path, canonical_rows)
        comparison_binding = {
            "path": str(canonical_path.resolve()),
            "sha256": semantic.sha256_file(canonical_path),
            "byte_count": canonical_path.stat().st_size,
            "record_count": 1,
            "dataset_id": "canonical-v2",
            "dataset_role": "historical_canonical_v2_comparison",
            "format_id": "canonical-triples-v2",
            "policy_versions": ["codex-adjudicated-canonical-v2"],
            "ordered_record_ids_sha256": semantic.sha256_value(
                [
                    {
                        "source_id": canonical_rows[0]["source_id"],
                        "candidate_id": canonical_rows[0]["candidate_id"],
                    }
                ]
            ),
            "ordered_row_hashes_sha256": semantic.sha256_value(
                [semantic.sha256_value(canonical_rows[0])]
            ),
        }
        policy = {
            "text_normalization_version": "nfkc-casefold-unicode-word-v1",
            "answer_normalization_version": "nfkc-casefold-whitespace-v1",
            "blocking_version": "token-minhash-16x2-lexical-window-v1",
            "question_near_threshold": 0.8,
            "canonical_fact_near_threshold": 0.8,
            "max_bucket_neighbors": 24,
            "max_examples": 20,
            "minhash_components": 16,
            "minhash_band_rows": 2,
            "deterministic": True,
            "network_or_model_used": False,
            "pair_review_contract": {
                "status": "pending_adjudication",
                "answer_alias_exact_requires": ["alias_valid", "same_answer_entity"],
                "alias_valid_scope": (
                    "pair_level_at_least_one_shared_alias_is_valid_for_both_records"
                ),
                "question_or_fact_exact_or_near_requires": ["semantic_duplicate"],
                "mixed_evidence_requires_union": True,
            },
        }
        universe_manifest_path = universe_dir / "formal_cohort_universe_manifest.json"
        universe_manifest = {
            "schema_version": semantic.UNIVERSE_MANIFEST_SCHEMA,
            "tool_version": "fixture",
            "universe_id": "fixture-universe",
            "universe_label": "fixture",
            "universe_status": "declared_immutable_not_reviewed",
            "record_count": len(items),
            "selection_source": {
                "mode": "full_pool",
                "record_count": len(ids),
                "ordered_base_fact_ids_sha256": semantic.sha256_value(ids),
            },
            "source_artifacts": source_bindings,
            "comparison_canonical": comparison_binding,
            "near_duplicate_audit_contract": {
                "summary_schema_version": semantic.AUDIT_SUMMARY_SCHEMA,
                "pair_schema_version": semantic.AUDIT_PAIR_SCHEMA,
                "policy": policy,
            },
            "ordered_cohort_item_ids_sha256": semantic.sha256_value(
                [item["cohort_item_id"] for item in items]
            ),
            "ordered_source_base_fact_ids_sha256": semantic.sha256_value(ids),
            "ordered_item_row_hashes_sha256": semantic.sha256_value(
                [semantic.sha256_value(item) for item in items]
            ),
            "items": {
                "path": str(items_path.resolve()),
                "sha256": semantic.sha256_file(items_path),
                "byte_count": items_path.stat().st_size,
                "record_count": len(items),
                "schema_version": semantic.UNIVERSE_ITEM_SCHEMA,
            },
            "review_complete": False,
            "canonical_freeze_authorized": False,
            "split_freeze_authorized": False,
        }
        write_json(universe_manifest_path, universe_manifest)

        current_dir = root / "current"
        full_rows = [
            full_row(ids[0], "old_component_a", "development"),
            full_row(ids[1], "old_component_b", "validation"),
        ]
        full_path = current_dir / "full_base_facts.jsonl"
        write_jsonl(full_path, full_rows)
        components = [
            {
                "schema_version": semantic.COMPONENT_SCHEMA,
                "leakage_component_id": row["leakage_component_id"],
                "member_count": 1,
                "base_fact_ids": [row["base_fact_id"]],
                "source_split_group_ids": [row["leakage_component_id"]],
                "frozen_semantic_component_ids": [],
                "split_assignment": row["split_assignment"],
            }
            for row in full_rows
        ]
        components_path = current_dir / "leakage_components.jsonl"
        write_jsonl(components_path, components)
        split_path = current_dir / "split_manifest.json"
        write_json(
            split_path,
            {
                "schema_version": semantic.SPLIT_MANIFEST_SCHEMA,
                "split_status": "provisional_not_frozen",
                "formal_split_freeze_performed": False,
                "semantic_paraphrase_closure_complete_for_full_pool": False,
                "base_fact_count": 2,
                "leakage_component_count": 2,
                "split_policy_version": "fixture-current-split-v1",
                "actual_base_fact_counts": {
                    "development": 1,
                    "validation": 1,
                    "sealed": 0,
                },
                "frozen_split_constraint": None,
            },
        )
        candidate_dir = root / "semantic-candidates"
        semantic.materialize(
            formal_universe_manifest_path=universe_manifest_path,
            full_base_facts_path=full_path,
            source_dir=source,
            comparison_canonical_path=canonical_path,
            output_dir=candidate_dir,
            split_manifest_path=split_path,
            leakage_components_path=components_path,
            expected_record_count=2,
        )
        manifest_path = candidate_dir / "semantic_closure_candidate_manifest.json"
        if empty_candidates:
            candidate_manifest = postreview.pre_hf.read_json(manifest_path)
            for artifact_name in ("candidate_pairs", "adjudication_template"):
                artifact = candidate_manifest["artifacts"][artifact_name]
                artifact_path = candidate_dir / artifact["filename"]
                write_jsonl(artifact_path, [])
                artifact.update(
                    {
                        "sha256": postreview.pre_hf.sha256_file(artifact_path),
                        "byte_count": artifact_path.stat().st_size,
                        "record_count": 0,
                    }
                )
            candidate_manifest["candidate_summary"].update(
                {
                    "candidate_pair_count": 0,
                    "ordered_pair_ids_sha256": postreview.semantic_runner.sha256_value([]),
                    "scope_counts": {},
                    "match_type_counts": {},
                    "current_pair_count": 0,
                    "current_cross_split_candidate_pair_count": 0,
                    "current_same_component_candidate_pair_count": 0,
                    "adjudication_template_count": 0,
                }
            )
            write_json(manifest_path, candidate_manifest)
        _, _, _, units = postreview.semantic_runner.load_review_units(
            candidate_manifest_path=manifest_path
        )
        if empty_candidates:
            self.assertEqual(units, [])
        else:
            self.assertGreater(len(units), 0)
        config = {
            "execution": {"max_workers": 1},
            "provider_profiles": {
                "openai": {
                    "protocol": "openai",
                    "base_url": "https://semantic-fixture.invalid/v1",
                }
            },
            "model_roles": {
                "full_pool_semantic_closure_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "expected_response_model": (
                            postreview.semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL
                        ),
                        "max_output_tokens": 1024,
                        "max_retries": 0,
                    }
                }
            },
        }
        review_dir = root / "semantic-review"
        review_result = postreview.semantic_runner.run_review(
            candidate_manifest_path=manifest_path,
            output_dir=review_dir,
            config=config,
            env_path=root / ".env",
            reviewer_spec=postreview.semantic_runner.resolve_reviewer_spec(config),
            expected_response_model=(
                postreview.semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL
            ),
            batch_size=8,
            max_workers=1,
            router=SemanticReviewRouter(),
            now_fn=lambda: "2026-09-23T00:00:00+00:00",
        )
        return (
            full_path,
            manifest_path,
            Path(review_result["adjudications_path"]),
            Path(review_result["manifest_path"]),
        )

    def test_empty_semantic_candidate_run_manifest_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ids = self._make_source(root)
            fact_apply = self._make_fact_apply(root, source, ids)
            full_path, semantic_manifest, adjudications, semantic_run_manifest = (
                self._make_semantic_inputs(
                    root, source, ids, empty_candidates=True
                )
            )
            run_manifest = postreview.pre_hf.read_json(semantic_run_manifest)
            self.assertEqual(run_manifest["terminal_status_counts"], {})

            result = postreview.materialize(
                full_base_facts_path=full_path,
                fact_review_apply_manifest_path=fact_apply,
                semantic_candidate_manifest_path=semantic_manifest,
                semantic_adjudications_path=adjudications,
                semantic_run_manifest_path=semantic_run_manifest,
                output_dir=root / "postreview-empty-semantic-candidates",
                expected_input_count=2,
                split_seed="fixture-postreview-seed",
            )
            self.assertEqual(result["retained_base_fact_count"], 2)

    def test_semantic_adjudication_unverified_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ids = self._make_source(root)
            fact_apply = self._make_fact_apply(root, source, ids)
            full_path, semantic_manifest, adjudications_path, semantic_run_manifest = (
                self._make_semantic_inputs(root, source, ids)
            )
            adjudications = postreview.pre_hf.read_jsonl(adjudications_path)
            adjudications[0]["response_model_identity_status"] = "not_checked"
            write_jsonl(adjudications_path, adjudications)
            run_manifest = postreview.pre_hf.read_json(semantic_run_manifest)
            run_manifest["artifacts"]["adjudications"].update(
                {
                    "sha256": postreview.pre_hf.sha256_file(adjudications_path),
                    "byte_count": adjudications_path.stat().st_size,
                }
            )
            write_json(semantic_run_manifest, run_manifest)
            with self.assertRaisesRegex(ValueError, "model identity is invalid"):
                postreview.materialize(
                    full_base_facts_path=full_path,
                    fact_review_apply_manifest_path=fact_apply,
                    semantic_candidate_manifest_path=semantic_manifest,
                    semantic_adjudications_path=adjudications_path,
                    semantic_run_manifest_path=semantic_run_manifest,
                    output_dir=root / "postreview-invalid-identity",
                    expected_input_count=2,
                    split_seed="fixture-postreview-seed",
                )

    def test_semantic_run_source_fingerprint_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ids = self._make_source(root)
            fact_apply = self._make_fact_apply(root, source, ids)
            full_path, semantic_manifest, adjudications, semantic_run_manifest = (
                self._make_semantic_inputs(root, source, ids)
            )
            run_manifest = postreview.pre_hf.read_json(semantic_run_manifest)
            run_manifest["run_contract"]["source_fingerprints"][
                "runner_sha256"
            ] = "0" * 64
            run_manifest["run_contract_sha256"] = (
                postreview.semantic_runner.sha256_value(run_manifest["run_contract"])
            )
            write_json(semantic_run_manifest, run_manifest)

            with self.assertRaisesRegex(ValueError, "source fingerprints are stale"):
                postreview.materialize(
                    full_base_facts_path=full_path,
                    fact_review_apply_manifest_path=fact_apply,
                    semantic_candidate_manifest_path=semantic_manifest,
                    semantic_adjudications_path=adjudications,
                    semantic_run_manifest_path=semantic_run_manifest,
                    output_dir=root / "postreview-stale-source-fingerprint",
                    expected_input_count=2,
                    split_seed="fixture-postreview-seed",
                )

    def test_semantic_run_route_identity_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ids = self._make_source(root)
            fact_apply = self._make_fact_apply(root, source, ids)
            full_path, semantic_manifest, adjudications, semantic_run_manifest = (
                self._make_semantic_inputs(root, source, ids)
            )
            run_manifest = postreview.pre_hf.read_json(semantic_run_manifest)
            identity_set = run_manifest["run_contract"]["route_identity"]
            record = identity_set["records"][0]
            record["requested_model"] = "gpt-5.5-tampered"
            record["route_identity_sha256"] = (
                postreview.semantic_runner.sha256_value(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "route_identity_sha256"
                    }
                )
            )
            identity_set["route_identity_set_sha256"] = (
                postreview.semantic_runner.sha256_value(
                    {
                        key: value
                        for key, value in identity_set.items()
                        if key != "route_identity_set_sha256"
                    }
                )
            )
            run_manifest["run_contract_sha256"] = (
                postreview.semantic_runner.sha256_value(run_manifest["run_contract"])
            )
            write_json(semantic_run_manifest, run_manifest)

            with self.assertRaisesRegex(
                ValueError, "identities differ from the frozen model contract"
            ):
                postreview.materialize(
                    full_base_facts_path=full_path,
                    fact_review_apply_manifest_path=fact_apply,
                    semantic_candidate_manifest_path=semantic_manifest,
                    semantic_adjudications_path=adjudications,
                    semantic_run_manifest_path=semantic_run_manifest,
                    output_dir=root / "postreview-tampered-route-identity",
                    expected_input_count=2,
                    split_seed="fixture-postreview-seed",
                )

    def test_end_to_end_rebuild_is_hash_bound_deterministic_and_stays_pre_hf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, ids = self._make_source(root)
            fact_apply = self._make_fact_apply(root, source, ids)
            (
                full_path,
                semantic_manifest,
                adjudications,
                semantic_run_manifest,
            ) = self._make_semantic_inputs(root, source, ids)

            first = postreview.materialize(
                full_base_facts_path=full_path,
                fact_review_apply_manifest_path=fact_apply,
                semantic_candidate_manifest_path=semantic_manifest,
                semantic_adjudications_path=adjudications,
                semantic_run_manifest_path=semantic_run_manifest,
                output_dir=root / "postreview-one",
                expected_input_count=2,
                split_seed="fixture-postreview-seed",
            )
            second = postreview.materialize(
                full_base_facts_path=full_path,
                fact_review_apply_manifest_path=fact_apply,
                semantic_candidate_manifest_path=semantic_manifest,
                semantic_adjudications_path=adjudications,
                semantic_run_manifest_path=semantic_run_manifest,
                output_dir=root / "postreview-two",
                expected_input_count=2,
                split_seed="fixture-postreview-seed",
            )
            self.assertEqual(first["retained_base_fact_count"], 2)
            self.assertEqual(first["leakage_component_count"], 1)
            rows = read_jsonl(root / "postreview-one" / "full_base_facts.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["leakage_component_id"], rows[1]["leakage_component_id"])
            self.assertEqual(rows[0]["split_assignment"], rows[1]["split_assignment"])
            self.assertTrue(all(row["human_gold"] is False for row in rows))

            manifest = json.loads(
                (root / "postreview-one" / "postreview_rebuild_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(
                manifest["semantic_review_contract"][
                    "semantic_near_duplicate_recall_guaranteed"
                ]
            )
            self.assertFalse(
                manifest["semantic_review_contract"][
                    "semantic_closure_complete_for_full_pool"
                ]
            )
            semantic_run = postreview.pre_hf.read_json(semantic_run_manifest)
            semantic_inputs = manifest["inputs"]["semantic_review"]
            self.assertEqual(
                semantic_inputs["run_manifest"]["sha256"],
                postreview.pre_hf.sha256_file(semantic_run_manifest),
            )
            self.assertTrue(
                manifest["semantic_review_contract"][
                    "runtime_identity_evidence_bound"
                ]
            )
            self.assertEqual(
                manifest["semantic_review_contract"][
                    "semantic_run_contract_sha256"
                ],
                semantic_run["run_contract_sha256"],
            )
            self.assertEqual(
                manifest["semantic_review_contract"][
                    "route_identity_set_sha256"
                ],
                semantic_run["run_contract"]["route_identity"][
                    "route_identity_set_sha256"
                ],
            )
            self.assertFalse(manifest["safety_contract"]["hf_checkpoint_bound"])
            self.assertFalse(manifest["safety_contract"]["hf_tokenizer_bound"])
            self.assertFalse(manifest["safety_contract"]["behavior_executed"])
            self.assertFalse(manifest["safety_contract"]["validation_exposed"])
            self.assertFalse(manifest["safety_contract"]["sealed_exposed"])
            for name, binding in manifest["outputs"].items():
                first_path = root / "postreview-one" / binding["filename"]
                second_path = root / "postreview-two" / binding["filename"]
                self.assertEqual(
                    postreview.pre_hf.sha256_file(first_path),
                    postreview.pre_hf.sha256_file(second_path),
                    name,
                )

            replay_artifacts = {
                name: {
                    "path": root / "postreview-one" / artifact_binding["filename"],
                    "binding": artifact_binding,
                }
                for name, artifact_binding in manifest["outputs"].items()
            }
            freezer._deterministically_replay_postreview(
                manifest_path=(
                    root / "postreview-one" / "postreview_rebuild_manifest.json"
                ),
                manifest=manifest,
                input_rows=read_jsonl(full_path),
                artifacts=replay_artifacts,
                split_seed="fixture-postreview-seed",
            )


if __name__ == "__main__":
    unittest.main()
