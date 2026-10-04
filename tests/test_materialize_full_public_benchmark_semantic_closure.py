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
    PROJECT_ROOT / "scripts" / "materialize_full_public_benchmark_semantic_closure.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "materialize_full_public_benchmark_semantic_closure", SCRIPT_PATH
)
semantic = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = semantic
SCRIPT_SPEC.loader.exec_module(semantic)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


review_tool = load_module(
    "semantic_resolution_review_tool",
    PROJECT_ROOT / "scripts" / "review_public_benchmark_bundle.py",
)
resolver = load_module(
    "semantic_resolution_resolver",
    PROJECT_ROOT / "scripts" / "resolve_full_public_benchmark_fact_review.py",
)
postreview = load_module(
    "semantic_resolution_postreview",
    PROJECT_ROOT / "scripts" / "materialize_full_public_benchmark_postreview.py",
)
zh_review = load_module(
    "semantic_resolution_zh_review",
    PROJECT_ROOT / "scripts" / "run_full_public_benchmark_zh_review.py",
)
freezer = load_module(
    "semantic_resolution_pre_exact_hf_freezer",
    PROJECT_ROOT / "scripts" / "freeze_full_public_benchmark_pre_exact_hf.py",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(semantic.serialize_jsonl(rows))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
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
                        "Fixture same fact." if same else "Fixture distinct facts."
                    ),
                    "confidence": "high",
                }
            )
        response = {
            "schema_version": postreview.semantic_runner.BATCH_RESPONSE_SCHEMA,
            "records": records,
        }
        validator(response)
        return SimpleNamespace(
            terminal_status="completed",
            parsed_response=response,
            raw_response=json.dumps(response, ensure_ascii=False, sort_keys=True),
            response_model=postreview.semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
            usage={},
            latency_ms=1,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "ok", "latency_ms": 1}],
        )


class FullSemanticClosureMaterializerTests(unittest.TestCase):
    def _source_behavior_row(self, label, *, question, fact, answer, split):
        return {
            "schema_version": "factual-perturbation-input-bundle-v1",
            "base_fact_id": f"pbf_{label}",
            "base_id": f"pbf_{label}",
            "candidate_id": f"candidate_{label}",
            "source_id": f"source_{label}",
            "source_dataset": "fixture",
            "source_subset": "unit",
            "source_question_en": question,
            "subject_en": question,
            "relation_raw": "fixture relation",
            "canonical_fact_en": fact,
            "prompt_en": question,
            "answer_en": answer,
            "answer_aliases_en": [answer],
            "distractor_candidates": [],
            "canonical_status": "pending_review",
            "bundle_status": "draft_pending_review",
            "evidence_tier": "provisional_single_model",
            "human_gold": False,
            "split_assignment": split,
            "split_group_id": f"old_group_{label}",
            "provenance": {"source_model": "fixture"},
        }

    def make_inputs(self, root):
        root = Path(root)
        source_dir = root / "source"
        source_dir.mkdir()
        behavior = [
            self._source_behavior_row(
                "a",
                question="Who wrote the novel Dune?",
                fact="Frank Herbert wrote the novel Dune.",
                answer="Frank Herbert",
                split="development",
            ),
            self._source_behavior_row(
                "b",
                question="WHO WROTE THE NOVEL DUNE!",
                fact="FRANK HERBERT WROTE THE NOVEL DUNE",
                answer="Frank Herbert",
                # Deliberately stale relative to the current full-base split.
                split="validation",
            ),
            self._source_behavior_row(
                "c",
                question="What is the capital of France?",
                fact="Paris is the capital of France.",
                answer="Paris",
                split="validation",
            ),
            self._source_behavior_row(
                "d",
                question="What planet is known as the Red Planet?",
                fact="Mars is known as the Red Planet.",
                answer="Mars",
                split="sealed",
            ),
        ]
        clusters = []
        provisional = []
        review = []
        for row in behavior:
            base_fact_id = row["base_fact_id"]
            candidate_id = row["candidate_id"]
            input_hash = semantic.sha256_value(
                {"fixture_source_input": candidate_id}
            )
            provisional.append(
                {
                    "schema_version": "provisional-factual-triple-v1",
                    "base_fact_id": base_fact_id,
                    "candidate_id": candidate_id,
                    "subject": row["source_question_en"],
                    "relation_raw": "fixture relation",
                    "answer": row["answer_en"],
                    "answer_aliases_en": list(row["answer_aliases_en"]),
                    "canonical_fact": row["canonical_fact_en"],
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
                        }
                    ],
                    "subject": row["source_question_en"],
                    "relation_raw": "fixture relation",
                    "answer": row["answer_en"],
                    "canonical_status": "pending_review",
                    "evidence_tier": "provisional_single_model",
                    "human_gold": False,
                }
            )
            review.append(
                {
                    "schema_version": "provisional-semantic-review-v1",
                    "base_fact_id": base_fact_id,
                    "member_candidate_ids": [candidate_id],
                    "subject": row["source_question_en"],
                    "relation_raw": "fixture relation",
                    "answer": row["answer_en"],
                    "canonical_fact": row["canonical_fact_en"],
                    "canonical_status": "pending_review",
                    "evidence_tier": "provisional_single_model",
                    "human_gold": False,
                    "review_decision": None,
                }
            )

        rows_by_role = {
            "behavior_bundle": behavior,
            "review_queue": review,
            "base_fact_clusters": clusters,
            "provisional_triples": provisional,
        }
        source_bindings = {}
        for role, (filename, schema_version, identifier_field) in semantic.SOURCE_ARTIFACTS.items():
            path = source_dir / filename
            rows = rows_by_role[role]
            write_jsonl(path, rows)
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

        canonical_path = root / "canonical-v2.jsonl"
        canonical_rows = [
            {
                "source_id": "canonical_source_dune",
                "candidate_id": "canonical_candidate_dune",
                "source_dataset": "canonical_fixture",
                "source_question": "Who wrote the novel Dune?",
                "canonical_fact": "Frank Herbert wrote the novel Dune.",
                "answer": "Frank Herbert",
                "source_answer": "Frank Herbert",
                "canonical_policy_version": "codex-adjudicated-canonical-v2",
                "canonical_decision": "keep",
            }
        ]
        write_jsonl(canonical_path, canonical_rows)
        comparison_binding = {
            "path": str(canonical_path.resolve()),
            "sha256": semantic.sha256_file(canonical_path),
            "byte_count": canonical_path.stat().st_size,
            "record_count": len(canonical_rows),
            "dataset_id": "canonical-v2",
            "dataset_role": "historical_canonical_v2_comparison",
            "format_id": "canonical-triples-v2",
            "policy_versions": ["codex-adjudicated-canonical-v2"],
            "ordered_record_ids_sha256": semantic.sha256_value(
                [
                    {
                        "source_id": row["source_id"],
                        "candidate_id": row["candidate_id"],
                    }
                    for row in canonical_rows
                ]
            ),
            "ordered_row_hashes_sha256": semantic.sha256_value(
                [semantic.sha256_value(row) for row in canonical_rows]
            ),
        }

        universe_dir = root / "universe"
        universe_dir.mkdir()
        universe_id = "formal_cohort_fixture"
        items = []
        cluster_by_id = {row["base_fact_id"]: row for row in clusters}
        review_by_id = {row["base_fact_id"]: row for row in review}
        provisional_by_id = {row["candidate_id"]: row for row in provisional}
        for index, row in enumerate(behavior):
            cluster = cluster_by_id[row["base_fact_id"]]
            member = cluster["members"][0]
            items.append(
                {
                    "schema_version": semantic.UNIVERSE_ITEM_SCHEMA,
                    "universe_id": universe_id,
                    "cohort_item_id": f"cohort_item_{index}",
                    "selection_index": index,
                    "source_base_fact_id": row["base_fact_id"],
                    "source_id": row["source_id"],
                    "member_count": 1,
                    "member_bindings": [
                        {
                            "candidate_id": row["candidate_id"],
                            "source_input_record_sha256": member[
                                "input_record_sha256"
                            ],
                            "triple_record_sha256": semantic.sha256_value(
                                provisional_by_id[row["candidate_id"]]
                            ),
                        }
                    ],
                    "source_row_bindings": {
                        "behavior_row_sha256": semantic.sha256_value(row),
                        "review_queue_row_sha256": semantic.sha256_value(
                            review_by_id[row["base_fact_id"]]
                        ),
                        "base_fact_cluster_row_sha256": semantic.sha256_value(
                            cluster
                        ),
                    },
                }
            )
        items_path = universe_dir / "formal_cohort_universe_items.jsonl"
        write_jsonl(items_path, items)
        ids = [row["base_fact_id"] for row in behavior]
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
                "answer_alias_exact_requires": [
                    "alias_valid",
                    "same_answer_entity",
                ],
                "alias_valid_scope": (
                    "pair_level_at_least_one_shared_alias_is_valid_for_both_records"
                ),
                "question_or_fact_exact_or_near_requires": [
                    "semantic_duplicate"
                ],
                "mixed_evidence_requires_union": True,
            },
        }
        universe_manifest_path = universe_dir / "formal_cohort_universe_manifest.json"
        universe_manifest = {
            "schema_version": semantic.UNIVERSE_MANIFEST_SCHEMA,
            "tool_version": "fixture",
            "universe_id": universe_id,
            "universe_label": "fixture",
            "universe_status": "declared_immutable_not_reviewed",
            "record_count": len(items),
            "selection_source": {
                "mode": "full_pool",
                "record_count": len(ids),
                "ordered_base_fact_ids_sha256": semantic.sha256_value(ids),
            },
            "selection_policy": {"mode": "full-pool"},
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
        current_dir.mkdir()
        current_splits = {
            "pbf_a": "development",
            "pbf_b": "development",
            "pbf_c": "validation",
            "pbf_d": "sealed",
        }
        current_components = {
            "pbf_a": "component_ab",
            "pbf_b": "component_ab",
            "pbf_c": "component_c",
            "pbf_d": "component_d",
        }
        full_rows = []
        for row in behavior:
            full_rows.append(
                {
                    **{
                        key: row.get(key)
                        for key in (
                            "base_fact_id",
                            "candidate_id",
                            "source_id",
                            "source_dataset",
                            "source_subset",
                            "source_question_en",
                            "prompt_en",
                            "canonical_fact_en",
                            "answer_en",
                            "answer_aliases_en",
                        )
                    },
                    "schema_version": semantic.FULL_BASE_SCHEMA,
                    "subject_en": row["source_question_en"],
                    "relation_raw": "fixture relation",
                    "relation_partition_id": (
                        "fixture_relation_b"
                        if row["base_fact_id"] == "pbf_b"
                        else "fixture_relation"
                    ),
                    "answer_type_bucket": "other",
                    "canonical_status": "pending_review",
                    "human_gold": False,
                    "leakage_component_id": current_components[row["base_fact_id"]],
                    "split_assignment": current_splits[row["base_fact_id"]],
                    "split_policy_version": "fixture-current-split-v1",
                    "split_status": "provisional_not_frozen",
                    "hf_model_execution_status": "not_run",
                    "hf_tokenizer_execution_status": "not_run",
                }
            )
        full_path = current_dir / "full_base_facts.jsonl"
        write_jsonl(full_path, full_rows)
        components = [
            {
                "schema_version": semantic.COMPONENT_SCHEMA,
                "leakage_component_id": "component_ab",
                "member_count": 2,
                "base_fact_ids": ["pbf_a", "pbf_b"],
                "split_assignment": "development",
            },
            {
                "schema_version": semantic.COMPONENT_SCHEMA,
                "leakage_component_id": "component_c",
                "member_count": 1,
                "base_fact_ids": ["pbf_c"],
                "split_assignment": "validation",
            },
            {
                "schema_version": semantic.COMPONENT_SCHEMA,
                "leakage_component_id": "component_d",
                "member_count": 1,
                "base_fact_ids": ["pbf_d"],
                "split_assignment": "sealed",
            },
        ]
        components_path = current_dir / "leakage_components.jsonl"
        write_jsonl(components_path, components)
        split_manifest_path = current_dir / "split_manifest.json"
        write_json(
            split_manifest_path,
            {
                "schema_version": semantic.SPLIT_MANIFEST_SCHEMA,
                "split_status": "provisional_not_frozen",
                "formal_split_freeze_performed": False,
                "semantic_paraphrase_closure_complete_for_full_pool": False,
                "base_fact_count": 4,
                "leakage_component_count": 3,
                "split_policy_version": "fixture-current-split-v1",
                "actual_base_fact_counts": {
                    "development": 2,
                    "validation": 1,
                    "sealed": 1,
                },
            },
        )
        return {
            "source_dir": source_dir,
            "universe_manifest": universe_manifest_path,
            "canonical": canonical_path,
            "full_base": full_path,
            "split_manifest": split_manifest_path,
            "components": components_path,
            "full_rows": full_rows,
        }

    def run_materializer(self, fixture, output):
        return semantic.materialize(
            formal_universe_manifest_path=fixture["universe_manifest"],
            full_base_facts_path=fixture["full_base"],
            source_dir=fixture["source_dir"],
            comparison_canonical_path=fixture["canonical"],
            output_dir=output,
            split_manifest_path=fixture["split_manifest"],
            leakage_components_path=fixture["components"],
            expected_record_count=4,
        )

    def make_resolution(self, root, fixture):
        ids = [row["base_fact_id"] for row in fixture["full_rows"]]
        scope = review_tool.create_scope(
            output_dir=root / "review-scope",
            artifact_paths=review_tool.source_paths(fixture["source_dir"]),
            base_fact_ids=ids,
        )
        export = review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "review-export",
        )
        export_manifest = review_tool.read_json(Path(export["manifest_path"]))
        templates = review_tool.read_jsonl(
            Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
        )
        outcomes = {
            "pbf_a": "accept",
            "pbf_b": "revise",
            "pbf_c": "defer",
            "pbf_d": "reject",
        }
        decisions = []
        for template in templates:
            decision = copy.deepcopy(template)
            outcome = outcomes[decision["base_fact_id"]]
            decision["decision"] = outcome
            decision["review_provenance"] = {
                "reviewer_type": "codex_proxy",
                "reviewer_id": "fixture-source-reviewer",
                "review_method": resolver.STRICT_FACT_REVIEW_METHOD,
                "reviewed_at": "2026-09-23T00:00:00Z",
            }
            for field in ("member_reviews", "alias_reviews", "distractor_reviews"):
                for target in decision[field]:
                    target["decision"] = "accept"
            if outcome == "reject" and decision["member_reviews"]:
                decision["member_reviews"][0]["decision"] = "reject"
            decision["notes"] = json.dumps(
                {
                    "fact_review": {
                        "decision": "reject" if outcome == "reject" else "accept",
                        "reason": "Fixture fact review.",
                    },
                    "relation_review": {
                        "decision": "accept",
                        "reason": "Fixture relation review.",
                    },
                    "notes": f"Fixture {outcome}.",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            decisions.append(decision)
        decisions_path = root / "fact-review-decisions.jsonl"
        write_jsonl(decisions_path, decisions)
        applied = review_tool.apply_decisions(
            scope_manifest_path=Path(scope["manifest_path"]),
            export_manifest_path=Path(export["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=root / "fact-review-apply",
            allow_partial=False,
        )
        template_result = resolver.materialize_template(
            full_base_facts_path=fixture["full_base"],
            review_apply_manifest_path=Path(applied["manifest_path"]),
            review_export_manifest_path=Path(export["manifest_path"]),
            output_dir=root / "fact-resolution-template",
        )
        resolution_manifest = resolver.read_json(
            Path(template_result["manifest_path"])
        )
        resolution_templates = resolver.read_jsonl(
            Path(template_result["manifest_path"]).parent
            / resolution_manifest["artifacts"]["resolution_decisions_template"][
                "filename"
            ]
        )
        full_index = {
            row["base_fact_id"]: row for row in resolver.read_jsonl(fixture["full_base"])
        }
        resolutions = []
        for template in resolution_templates:
            base_fact_id = template["base_fact_id"]
            decision = copy.deepcopy(template)
            if base_fact_id == "pbf_b":
                updates = {
                    "answer_en": "Brian Herbert",
                    "answer_aliases_en": ["Brian Herbert"],
                    "canonical_fact_en": "Brian Herbert wrote the novel Dune.",
                }
                revised = copy.deepcopy(full_index[base_fact_id])
                revised.update(updates)
                decision["disposition"] = "apply_revision"
                decision["revision"] = {
                    "updates": updates,
                    "revision_reason": "Fixture correction before semantic closure.",
                    "source_reject_override": None,
                    "editor_provenance": {
                        "editor_type": "codex_proxy",
                        "editor_id": "fixture-editor",
                        "edit_method": "field_level_revision_from_review_evidence",
                        "edited_at": "2026-09-23T01:00:00Z",
                        "human_gold": False,
                    },
                    "rereview": {
                        "decision": "accept",
                        "revised_full_fact_row_sha256": resolver.sha256_value(revised),
                        "field_reviews": {
                            field: {
                                "decision": "accept",
                                "reason": f"Verified {field}.",
                            }
                            for field in resolver.MUTABLE_FIELDS
                        },
                        "rationale": "Fixture independent field review.",
                        "reviewer_type": "codex_proxy",
                        "reviewer_id": "fixture-independent-rereviewer",
                        "review_method": "independent_field_level_rereview",
                        "reviewed_at": "2026-09-23T01:10:00Z",
                        "human_gold": False,
                    },
                }
            else:
                decision["disposition"] = "cohort_exclude"
                decision["cohort_exclusion"] = {
                    "reason": "Fixture exclusion from the successor cohort.",
                    "scope": resolver.EXCLUSION_SCOPE,
                    "factual_falsehood_asserted": False,
                    "resolution_provenance": {
                        "reviewer_type": "codex_proxy",
                        "reviewer_id": "fixture-resolution-reviewer",
                        "review_method": "explicit_cohort_eligibility_resolution",
                        "reviewed_at": "2026-09-23T01:20:00Z",
                        "human_gold": False,
                    },
                }
            resolutions.append(decision)
        resolutions_path = root / "fact-resolutions.jsonl"
        write_jsonl(resolutions_path, resolutions)
        applied_resolution = resolver.apply_resolutions(
            template_manifest_path=Path(template_result["manifest_path"]),
            resolutions_path=resolutions_path,
            output_dir=root / "fact-resolution",
        )
        output_manifest_path = Path(applied_resolution["manifest_path"])
        output_manifest = resolver.read_json(output_manifest_path)
        revised_path = (
            output_manifest_path.parent
            / output_manifest["artifacts"]["revised_full_base_facts"]["filename"]
        )
        return output_manifest_path, revised_path

    def semantic_adjudications(self, candidate_manifest_path, output_path):
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
        review_dir = output_path.parent / f"{output_path.stem}-run"
        result = postreview.semantic_runner.run_review(
            candidate_manifest_path=candidate_manifest_path,
            output_dir=review_dir,
            config=config,
            env_path=output_path.parent / ".env",
            reviewer_spec=postreview.semantic_runner.resolve_reviewer_spec(config),
            expected_response_model=(
                postreview.semantic_runner.DEFAULT_EXPECTED_RESPONSE_MODEL
            ),
            batch_size=8,
            max_workers=1,
            router=SemanticReviewRouter(),
            now_fn=lambda: "2026-09-23T02:00:00+00:00",
        )
        return Path(result["adjudications_path"]), Path(result["manifest_path"])

    def test_materializes_rebound_candidates_and_field_level_template(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_inputs(directory)
            output = Path(directory) / "closure"
            result = self.run_materializer(fixture, output)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "semantic_closure_adjudication_template.jsonl",
                    "semantic_closure_candidate_manifest.json",
                    "semantic_closure_candidate_pairs.jsonl",
                ],
            )
            pairs = read_jsonl(output / "semantic_closure_candidate_pairs.jsonl")
            templates = read_jsonl(
                output / "semantic_closure_adjudication_template.jsonl"
            )
            manifest = json.loads(
                (output / "semantic_closure_candidate_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertGreaterEqual(len(pairs), 2)
            self.assertEqual(result["candidate_pair_count"], len(pairs))
            self.assertEqual(len(templates), len(pairs))

            current_pair = next(
                pair
                for pair in pairs
                if {
                    pair["left"]["provenance"].get("base_fact_id"),
                    pair["right"]["provenance"].get("base_fact_id"),
                }
                == {"pbf_a", "pbf_b"}
            )
            # The original behavior rows put a and b in different splits.  The
            # semantic-closure candidate must use the current full-base state.
            self.assertFalse(current_pair["cross_split"])
            self.assertTrue(current_pair["same_leakage_component"])
            self.assertTrue(current_pair["split_comparison_applicable"])
            self.assertTrue(current_pair["pair_id"].startswith("semantic_pair_"))
            for side in ("left", "right"):
                endpoint = current_pair[side]["provenance"]
                self.assertEqual(endpoint["input_role"], "current_full_base")
                self.assertEqual(
                    endpoint["input_path"], str(fixture["full_base"].resolve())
                )
                expected = next(
                    row
                    for row in fixture["full_rows"]
                    if row["base_fact_id"] == endpoint["base_fact_id"]
                )
                self.assertEqual(
                    endpoint["input_record_sha256"], semantic.sha256_value(expected)
                )

            canonical_pair = next(
                pair
                for pair in pairs
                if {
                    pair["left"]["provenance"]["input_role"],
                    pair["right"]["provenance"]["input_role"],
                }
                == {"current_full_base", "comparison_canonical"}
            )
            canonical_endpoint = next(
                canonical_pair[side]
                for side in ("left", "right")
                if canonical_pair[side]["provenance"]["input_role"]
                == "comparison_canonical"
            )
            self.assertEqual(
                canonical_endpoint["provenance"]["input_path"],
                str(fixture["canonical"].resolve()),
            )
            self.assertIsNone(canonical_pair["cross_split"])

            template_by_id = {row["pair_id"]: row for row in templates}
            template = template_by_id[current_pair["pair_id"]]
            self.assertEqual(
                template["candidate_row_sha256"],
                semantic.sha256_value(current_pair),
            )
            self.assertTrue(template["field_adjudications"]["alias_valid"]["required"])
            self.assertIsNone(template["field_adjudications"]["alias_valid"]["value"])
            self.assertIsNone(template["relationship_decision"])

            self.assertEqual(manifest["base_fact_count"], 4)
            self.assertEqual(
                manifest["inputs"]["current_full_base_facts"]["sha256"],
                semantic.sha256_file(fixture["full_base"]),
            )
            self.assertEqual(
                manifest["inputs"]["current_split_manifest"]["sha256"],
                semantic.sha256_file(fixture["split_manifest"]),
            )
            self.assertEqual(
                manifest["inputs"]["current_leakage_components"]["sha256"],
                semantic.sha256_file(fixture["components"]),
            )
            generation = manifest["lexical_alias_candidate_generation"]
            self.assertTrue(generation["all_emitted_candidates_retained"])
            self.assertFalse(generation["audit_pair_ids_reused"])
            self.assertFalse(generation["audit_cross_split_flags_reused"])
            self.assertTrue(manifest["limitations"]["candidate_generation_is_bounded"])
            self.assertTrue(manifest["limitations"]["semantic_decisions_pending"])
            self.assertFalse(
                manifest["limitations"]["semantic_near_duplicate_recall_guaranteed"]
            )
            self.assertFalse(manifest["safety_contract"]["split_freeze_emitted"])
            self.assertFalse(manifest["safety_contract"]["hf_checkpoint_bound"])

    def test_outputs_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_inputs(directory)
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            self.run_materializer(fixture, first)
            self.run_materializer(fixture, second)
            for filename in (
                "semantic_closure_candidate_pairs.jsonl",
                "semantic_closure_adjudication_template.jsonl",
                "semantic_closure_candidate_manifest.json",
            ):
                self.assertEqual(
                    (first / filename).read_bytes(), (second / filename).read_bytes()
                )

    def test_rejects_stale_current_component_assignment_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_inputs(directory)
            components = read_jsonl(fixture["components"])
            components[0]["split_assignment"] = "validation"
            write_jsonl(fixture["components"], components)
            output = Path(directory) / "closure"
            with self.assertRaisesRegex(ValueError, "Current component split is stale"):
                self.run_materializer(fixture, output)
            self.assertFalse(output.exists())

    def test_rejects_stale_source_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.make_inputs(directory)
            with fixture["source_dir"].joinpath("review_queue.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 is stale"):
                self.run_materializer(fixture, Path(directory) / "closure")

    def test_resolution_successor_drives_new_candidates_and_postreview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_inputs(root)
            resolution_manifest, revised_path = self.make_resolution(root, fixture)
            frozen_path = root / "frozen-split.jsonl"
            write_jsonl(
                frozen_path,
                [
                    {
                        "schema_version": "fixture-frozen-split-v1",
                        "base_fact_id": base_fact_id,
                        "split_assignment": "development",
                        "leakage_component_id": "fixture_frozen_" + base_fact_id,
                    }
                    for base_fact_id in ("pbf_a", "pbf_b")
                ],
            )
            source_split = json.loads(
                fixture["split_manifest"].read_text(encoding="utf-8")
            )
            source_split["frozen_split_constraint"] = {
                "path": str(frozen_path.resolve()),
                "sha256": semantic.sha256_file(frozen_path),
                "byte_count": frozen_path.stat().st_size,
                "record_count": 2,
                "schema_version": "fixture-frozen-split-v1",
            }
            write_json(fixture["split_manifest"], source_split)
            output = root / "resolved-closure"
            result = semantic.materialize(
                formal_universe_manifest_path=fixture["universe_manifest"],
                full_base_facts_path=revised_path,
                source_dir=fixture["source_dir"],
                comparison_canonical_path=fixture["canonical"],
                output_dir=output,
                split_manifest_path=fixture["split_manifest"],
                leakage_components_path=fixture["components"],
                fact_resolution_manifest_path=resolution_manifest,
                expected_record_count=4,
            )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            successor = manifest["resolved_cohort"]
            self.assertTrue(successor["selection_is_complete_retained_cohort"])
            self.assertEqual(successor["source_record_count"], 4)
            self.assertEqual(successor["retained_record_count"], 2)
            self.assertEqual(successor["revised_record_count"], 1)
            self.assertEqual(successor["excluded_record_count"], 2)
            self.assertEqual(manifest["base_fact_count"], 2)
            self.assertEqual(manifest["universe_id"], successor["successor_universe_id"])
            self.assertFalse(
                manifest["upstream_review_invalidation"][
                    "pre_resolution_translation_reviews_valid"
                ]
            )
            self.assertFalse(
                manifest["upstream_review_invalidation"][
                    "pre_resolution_distractor_reviews_valid"
                ]
            )
            pairs = read_jsonl(
                output / "semantic_closure_candidate_pairs.jsonl"
            )
            current_ids = {
                pair[side]["provenance"]["base_fact_id"]
                for pair in pairs
                for side in ("left", "right")
                if pair[side]["provenance"]["input_role"] == "current_full_base"
            }
            self.assertTrue(current_ids.issubset({"pbf_a", "pbf_b"}))
            self.assertFalse(current_ids.intersection({"pbf_c", "pbf_d"}))

            adjudications_path, semantic_run_manifest = self.semantic_adjudications(
                manifest_path, root / "resolved-semantic-adjudications.jsonl"
            )
            rebuilt = postreview.materialize(
                full_base_facts_path=revised_path,
                fact_review_apply_manifest_path=None,
                fact_resolution_manifest_path=resolution_manifest,
                semantic_candidate_manifest_path=manifest_path,
                semantic_adjudications_path=adjudications_path,
                semantic_run_manifest_path=semantic_run_manifest,
                output_dir=root / "postreview-resolved",
                expected_input_count=4,
                split_seed="fixture-resolution-successor-seed",
            )
            self.assertEqual(rebuilt["retained_base_fact_count"], 2)
            self.assertEqual(rebuilt["rejected_base_fact_count"], 2)
            post_manifest = json.loads(
                (root / "postreview-resolved" / "postreview_rebuild_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(post_manifest["counts"]["input_base_facts"], 4)
            self.assertEqual(post_manifest["counts"]["revised_base_facts"], 1)
            self.assertTrue(
                post_manifest["fact_review_contract"][
                    "selection_is_complete_retained_cohort"
                ]
            )
            self.assertFalse(
                post_manifest["semantic_review_contract"][
                    "excluded_facts_are_component_bridges"
                ]
            )
            self.assertFalse(
                post_manifest["review_invalidation"][
                    "pre_resolution_semantic_adjudications_valid"
                ]
            )
            reviewed = read_jsonl(root / "postreview-resolved" / "full_base_facts.jsonl")
            reviewed_by_id = {row["base_fact_id"]: row for row in reviewed}
            self.assertEqual(reviewed_by_id["pbf_b"]["answer_en"], "Brian Herbert")
            self.assertTrue(
                reviewed_by_id["pbf_b"]["fact_review"]["revision_applied"]
            )
            zh_inputs = zh_review.load_bound_inputs(
                formal_universe_manifest_path=fixture["universe_manifest"],
                formal_universe_items_path=(
                    fixture["universe_manifest"].parent
                    / "formal_cohort_universe_items.jsonl"
                ),
                full_base_facts_path=(
                    root / "postreview-resolved" / "full_base_facts.jsonl"
                ),
                distractor_candidates_path=(
                    root / "postreview-resolved" / "distractor_candidates.jsonl"
                ),
                neutral_candidates_path=(
                    root
                    / "postreview-resolved"
                    / "neutral_reference_candidates.jsonl"
                ),
                split_manifest_path=(
                    root / "postreview-resolved" / "split_manifest.json"
                ),
                fact_resolution_manifest_path=resolution_manifest,
                postreview_rebuild_manifest_path=(
                    root
                    / "postreview-resolved"
                    / "postreview_rebuild_manifest.json"
                ),
            )
            self.assertEqual(len(zh_inputs["items"]), 2)
            self.assertTrue(
                zh_inputs["successor_cohort"][
                    "selection_is_complete_retained_cohort"
                ]
            )
            self.assertEqual(
                zh_inputs["successor_cohort"]["excluded_record_count"], 2
            )

    def test_resolution_successor_postreview_replay_uses_resolution_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_inputs(root)
            resolution_manifest, revised_path = self.make_resolution(root, fixture)
            closure_dir = root / "resolved-closure-for-replay"
            semantic.materialize(
                formal_universe_manifest_path=fixture["universe_manifest"],
                full_base_facts_path=revised_path,
                source_dir=fixture["source_dir"],
                comparison_canonical_path=fixture["canonical"],
                output_dir=closure_dir,
                split_manifest_path=fixture["split_manifest"],
                leakage_components_path=fixture["components"],
                fact_resolution_manifest_path=resolution_manifest,
                expected_record_count=4,
            )
            candidate_manifest_path = (
                closure_dir / "semantic_closure_candidate_manifest.json"
            )
            adjudications_path, semantic_run_manifest = self.semantic_adjudications(
                candidate_manifest_path, root / "resolved-semantic-for-replay.jsonl"
            )
            postreview_dir = root / "postreview-resolution-for-replay"
            postreview.materialize(
                full_base_facts_path=revised_path,
                fact_review_apply_manifest_path=None,
                fact_resolution_manifest_path=resolution_manifest,
                semantic_candidate_manifest_path=candidate_manifest_path,
                semantic_adjudications_path=adjudications_path,
                semantic_run_manifest_path=semantic_run_manifest,
                output_dir=postreview_dir,
                expected_input_count=4,
                split_seed="fixture-resolution-replay-seed",
            )

            postreview_manifest_path = (
                postreview_dir / "postreview_rebuild_manifest.json"
            )
            postreview_manifest = json.loads(
                postreview_manifest_path.read_text(encoding="utf-8")
            )
            artifacts = {}
            for label, artifact_binding in postreview_manifest["outputs"].items():
                artifact_path = postreview_dir / artifact_binding["filename"]
                artifacts[label] = {
                    "path": artifact_path,
                    "binding": artifact_binding,
                }

            # Exercise the real replay.  In particular, do not mock the replay
            # helper or the postreview materializer: the regression is the
            # resolution-mode CLI contract itself.
            freezer._deterministically_replay_postreview(
                manifest_path=postreview_manifest_path,
                manifest=postreview_manifest,
                input_rows=read_jsonl(revised_path),
                artifacts=artifacts,
                split_seed="fixture-resolution-replay-seed",
            )


if __name__ == "__main__":
    unittest.main()
