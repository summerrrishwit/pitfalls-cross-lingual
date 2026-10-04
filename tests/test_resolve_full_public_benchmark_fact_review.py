import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


review_tool = load_module(
    "review_public_benchmark_bundle_resolution_tests",
    PROJECT_ROOT / "scripts" / "review_public_benchmark_bundle.py",
)
resolver = load_module(
    "resolve_full_public_benchmark_fact_review_tests",
    PROJECT_ROOT / "scripts" / "resolve_full_public_benchmark_fact_review.py",
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def triple(base_fact_id, candidate_id, input_hash, answer):
    return {
        "schema_version": "provisional-factual-triple-v1",
        "base_fact_id": base_fact_id,
        "candidate_id": candidate_id,
        "subject": f"Subject {base_fact_id}",
        "relation_raw": "has answer",
        "answer": answer,
        "answer_aliases_en": [answer, f"{answer} alias"],
        "canonical_fact": f"Subject {base_fact_id} has {answer}.",
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "provenance": {"input_record_sha256": input_hash},
    }


def cluster(base_fact_id, candidate_id, input_hash, answer):
    return {
        "schema_version": "provisional-base-fact-cluster-v1",
        "base_fact_id": base_fact_id,
        "representative_candidate_id": candidate_id,
        "member_count": 1,
        "members": [
            {
                "candidate_id": candidate_id,
                "input_record_sha256": input_hash,
                "source_id": candidate_id,
                "source_dataset": "fixture",
            }
        ],
        "subject": f"Subject {base_fact_id}",
        "relation_raw": "has answer",
        "answer": answer,
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
    }


def queue(base_fact_id, candidate_id, answer):
    return {
        "schema_version": "provisional-semantic-review-v1",
        "base_fact_id": base_fact_id,
        "member_candidate_ids": [candidate_id],
        "subject": f"Subject {base_fact_id}",
        "relation_raw": "has answer",
        "answer": answer,
        "canonical_fact": f"Subject {base_fact_id} has {answer}.",
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "review_decision": None,
    }


def behavior(base_fact_id, candidate_id, answer):
    return {
        "schema_version": "factual-perturbation-input-bundle-v1",
        "base_fact_id": base_fact_id,
        "base_id": base_fact_id,
        "candidate_id": candidate_id,
        "subject_en": f"Subject {base_fact_id}",
        "relation_raw": "has answer",
        "answer_en": answer,
        "answer_aliases_en": [answer, f"{answer} alias"],
        "canonical_fact_en": f"Subject {base_fact_id} has {answer}.",
        "distractor_candidates": [
            {
                "distractor_id": f"dist_{base_fact_id}",
                "text_en": "Wrong",
                "answer_en": "Wrong",
                "source_choice_index": 1,
                "verification_status": "pending_review",
                "verified": False,
            }
        ],
        "canonical_status": "pending_review",
        "bundle_status": "draft_pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
    }


def full_fact(behavior_row):
    return {
        "schema_version": resolver.FULL_BASE_SCHEMA,
        "base_fact_id": behavior_row["base_fact_id"],
        "candidate_id": behavior_row["candidate_id"],
        "subject_en": behavior_row["subject_en"],
        "relation_raw": behavior_row["relation_raw"],
        "answer_en": behavior_row["answer_en"],
        "answer_aliases_en": copy.deepcopy(behavior_row["answer_aliases_en"]),
        "canonical_fact_en": behavior_row["canonical_fact_en"],
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "split_assignment": "development",
        "split_status": "provisional_not_frozen",
        "leakage_component_id": f"component_{behavior_row['base_fact_id']}",
        "relation_partition_id": "fixture_relation",
        "hf_model_execution_status": "not_run",
        "hf_tokenizer_execution_status": "not_run",
    }


class FullFactResolutionTests(unittest.TestCase):
    def build_chain(
        self,
        root,
        *,
        reject_fact_decision="accept",
        reject_relation_decision="accept",
        reject_canonical_alias=False,
        reject_target="alias",
    ):
        source = root / "source"
        ids = ["fact_accept", "fact_revise", "fact_defer", "fact_reject"]
        answers = ["A", "B", "C", "D"]
        triples = []
        clusters = []
        queues = []
        behaviors = []
        for position, (base_fact_id, answer) in enumerate(zip(ids, answers), start=1):
            candidate_id = f"candidate_{position}"
            input_hash = str(position) * 64
            triples.append(triple(base_fact_id, candidate_id, input_hash, answer))
            clusters.append(cluster(base_fact_id, candidate_id, input_hash, answer))
            queues.append(queue(base_fact_id, candidate_id, answer))
            behaviors.append(behavior(base_fact_id, candidate_id, answer))
        write_jsonl(source / "provisional_triples.jsonl", triples)
        write_jsonl(source / "base_fact_clusters.jsonl", clusters)
        write_jsonl(source / "review_queue.jsonl", queues)
        write_jsonl(source / "behavior_input_bundle.jsonl", behaviors)
        full_path = root / "full_base_facts.jsonl"
        write_jsonl(full_path, [full_fact(row) for row in behaviors])

        scope = review_tool.create_scope(
            output_dir=root / "scope",
            artifact_paths=review_tool.source_paths(source),
            base_fact_ids=ids,
        )
        export = review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / "export",
        )
        export_manifest = json.loads(
            Path(export["manifest_path"]).read_text(encoding="utf-8")
        )
        templates = {
            row["base_fact_id"]: row
            for row in read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )
        }
        outcomes = {
            "fact_accept": "accept",
            "fact_revise": "revise",
            "fact_defer": "defer",
            "fact_reject": "reject",
        }
        decisions = []
        for base_fact_id in ids:
            decision = copy.deepcopy(templates[base_fact_id])
            outcome = outcomes[base_fact_id]
            decision["decision"] = outcome
            decision["review_provenance"] = {
                "reviewer_type": "codex_proxy",
                "reviewer_id": "source-reviewer",
                "review_method": resolver.STRICT_FACT_REVIEW_METHOD,
                "reviewed_at": "2026-09-23T00:00:00Z",
            }
            for member in decision["member_reviews"]:
                member["decision"] = "accept"
            if outcome == "accept":
                for alias in decision["alias_reviews"]:
                    alias["decision"] = (
                        "accept" if alias["text_en"] == "A" else "reject"
                    )
                for distractor in decision["distractor_reviews"]:
                    distractor["decision"] = "accept"
            elif outcome == "reject":
                for alias in decision["alias_reviews"]:
                    if reject_target == "alias":
                        alias["decision"] = (
                            "reject"
                            if (alias["text_en"] == "D") == reject_canonical_alias
                            else "accept"
                        )
                    else:
                        alias["decision"] = "accept"
                for distractor in decision["distractor_reviews"]:
                    distractor["decision"] = (
                        "reject" if reject_target == "distractor" else "accept"
                    )
            decision["notes"] = json.dumps(
                {
                    "fact_review": {
                        "decision": (
                            reject_fact_decision if outcome == "reject" else "accept"
                        ),
                        "reason": "Fixture core fact review.",
                    },
                    "relation_review": {
                        "decision": (
                            reject_relation_decision if outcome == "reject" else "accept"
                        ),
                        "reason": "Fixture core relation review.",
                    },
                    "notes": f"Fixture {outcome}.",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            decisions.append(decision)
        decisions_path = root / "decisions.jsonl"
        write_jsonl(decisions_path, decisions)
        apply = review_tool.apply_decisions(
            scope_manifest_path=Path(scope["manifest_path"]),
            export_manifest_path=Path(export["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=root / "apply",
            allow_partial=False,
        )
        return {
            "full": full_path,
            "apply": Path(apply["manifest_path"]),
            "export": Path(export["manifest_path"]),
        }

    def materialize(self, root):
        chain = self.build_chain(root)
        result = resolver.materialize_template(
            full_base_facts_path=chain["full"],
            review_apply_manifest_path=chain["apply"],
            review_export_manifest_path=chain["export"],
            output_dir=root / "resolution-template",
        )
        manifest_path = Path(result["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        templates = read_jsonl(
            manifest_path.parent
            / manifest["artifacts"]["resolution_decisions_template"]["filename"]
        )
        return chain, result, manifest, templates

    def exclusion(self, template, reason="Outside current cohort."):
        decision = copy.deepcopy(template)
        decision["disposition"] = "cohort_exclude"
        decision["cohort_exclusion"] = {
            "reason": reason,
            "scope": resolver.EXCLUSION_SCOPE,
            "factual_falsehood_asserted": False,
            "resolution_provenance": {
                "reviewer_type": "codex_proxy",
                "reviewer_id": "resolution-reviewer",
                "review_method": "explicit_cohort_eligibility_resolution",
                "reviewed_at": "2026-09-23T01:00:00Z",
                "human_gold": False,
            },
        }
        return decision

    def revision(self, template, source_row):
        decision = copy.deepcopy(template)
        decision["disposition"] = "apply_revision"
        updates = {
            "answer_en": "B revised",
            "answer_aliases_en": ["B revised", "Revised B"],
            "canonical_fact_en": "Subject fact_revise has B revised.",
        }
        revised = copy.deepcopy(source_row)
        revised.update(updates)
        decision["revision"] = {
            "updates": updates,
            "revision_reason": "Correct the answer and canonical statement.",
            "source_reject_override": None,
            "editor_provenance": {
                "editor_type": "codex_proxy",
                "editor_id": "revision-editor",
                "edit_method": "field_level_revision_from_review_evidence",
                "edited_at": "2026-09-23T01:10:00Z",
                "human_gold": False,
            },
            "rereview": {
                "decision": "accept",
                "revised_full_fact_row_sha256": resolver.sha256_value(revised),
                "field_reviews": {
                    field: {"decision": "accept", "reason": f"Verified {field}."}
                    for field in resolver.MUTABLE_FIELDS
                },
                "rationale": "All revised semantic fields are mutually consistent.",
                "reviewer_type": "codex_proxy",
                "reviewer_id": "independent-rereviewer",
                "review_method": "independent_field_level_rereview",
                "reviewed_at": "2026-09-23T01:20:00Z",
                "human_gold": False,
            },
        }
        return decision

    def reject_override_revision(self, template, item, source_row):
        decision = copy.deepcopy(template)
        decision["disposition"] = "apply_revision"
        eligibility = item["source_reject_override_eligibility"]
        updates = (
            {"answer_aliases_en": [source_row["answer_en"]]}
            if eligibility["target_resolutions"]["rejected_alias_ids"]
            else {}
        )
        revised = copy.deepcopy(source_row)
        revised.update(updates)
        decision["revision"] = {
            "updates": updates,
            "revision_reason": "Resolve only the rejected editable review targets.",
            "source_reject_override": {
                "original_review_outcome": "reject",
                "eligibility_evidence_sha256": resolver.sha256_value(eligibility),
                "override_reason": (
                    "Original reject was target-level only; core fact and relation "
                    "were accepted by the source reviewer."
                ),
                "target_resolutions": copy.deepcopy(
                    eligibility["target_resolutions"]
                ),
            },
            "editor_provenance": {
                "editor_type": "codex_proxy",
                "editor_id": "reject-override-editor",
                "edit_method": "drop_rejected_alias",
                "edited_at": "2026-09-23T01:30:00Z",
                "human_gold": False,
            },
            "rereview": {
                "decision": "accept",
                "revised_full_fact_row_sha256": resolver.sha256_value(revised),
                "field_reviews": {
                    field: {"decision": "accept", "reason": f"Verified {field}."}
                    for field in resolver.MUTABLE_FIELDS
                },
                "rationale": "Core semantics remain valid after dropping the bad alias.",
                "reviewer_type": "codex_proxy",
                "reviewer_id": "reject-override-rereviewer",
                "review_method": "independent_reject_override_rereview",
                "reviewed_at": "2026-09-23T01:40:00Z",
                "human_gold": False,
            },
        }
        return decision

    def complete_resolutions(self, chain, templates):
        template_by_id = {row["base_fact_id"]: row for row in templates}
        source_rows = {row["base_fact_id"]: row for row in read_jsonl(chain["full"])}
        return [
            self.revision(template_by_id["fact_revise"], source_rows["fact_revise"]),
            self.exclusion(template_by_id["fact_defer"], "Deferred evidence is unresolved."),
            self.exclusion(template_by_id["fact_reject"], "Not admitted to this cohort."),
        ]

    def test_materialize_template_covers_every_nonaccept_without_pool_size_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, result, manifest, templates = self.materialize(root)
            self.assertEqual(result["resolution_item_count"], 3)
            self.assertEqual(
                manifest["resolution_item_outcome_counts"],
                {"defer": 1, "reject": 1, "revise": 1},
            )
            self.assertEqual(
                [row["base_fact_id"] for row in templates],
                ["fact_revise", "fact_defer", "fact_reject"],
            )
            self.assertEqual(
                templates[0]["disposition"], None
            )
            self.assertFalse(templates[0]["human_gold"])

    def test_apply_revision_and_explicit_exclusions_then_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, result, _, templates = self.materialize(root)
            resolutions = self.complete_resolutions(chain, templates)
            resolutions_path = root / "resolutions.jsonl"
            write_jsonl(resolutions_path, resolutions)
            applied = resolver.apply_resolutions(
                template_manifest_path=Path(result["manifest_path"]),
                resolutions_path=resolutions_path,
                output_dir=root / "resolved",
            )
            manifest = json.loads(
                Path(applied["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["counts"]["source_record_count"], 4)
            self.assertEqual(manifest["counts"]["retained_record_count"], 2)
            self.assertEqual(manifest["counts"]["revision_count"], 1)
            self.assertEqual(manifest["counts"]["cohort_exclusion_count"], 2)
            self.assertEqual(manifest["counts"]["accepted_alias_filter_change_count"], 1)
            invalidation = manifest["semantic_closure_invalidation"]
            self.assertTrue(invalidation["prior_semantic_candidate_manifest_invalidated"])
            self.assertTrue(invalidation["semantic_candidates_require_rematerialization"])
            self.assertTrue(invalidation["split_assignments_require_recomputation"])
            output_rows = read_jsonl(
                Path(applied["manifest_path"]).parent / "revised_full_base_facts.jsonl"
            )
            self.assertEqual(
                [row["base_fact_id"] for row in output_rows],
                ["fact_accept", "fact_revise"],
            )
            self.assertEqual(output_rows[0]["answer_aliases_en"], ["A"])
            self.assertEqual(output_rows[1]["answer_en"], "B revised")
            exclusions = read_jsonl(
                Path(applied["manifest_path"]).parent / "cohort_exclusions.jsonl"
            )
            self.assertTrue(all(row["factual_falsehood_asserted"] is False for row in exclusions))
            validated = resolver.validate_resolution_output(Path(applied["manifest_path"]))
            self.assertTrue(validated["validated"])

    def test_eligible_target_only_reject_can_be_overridden_without_erasing_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, result, manifest, templates = self.materialize(root)
            template_by_id = {row["base_fact_id"]: row for row in templates}
            items = read_jsonl(
                Path(result["manifest_path"]).parent
                / manifest["artifacts"]["resolution_items"]["filename"]
            )
            item_by_id = {row["base_fact_id"]: row for row in items}
            eligibility = item_by_id["fact_reject"][
                "source_reject_override_eligibility"
            ]
            self.assertTrue(eligibility["eligible"])
            self.assertEqual(eligibility["fact_review"]["decision"], "accept")
            self.assertEqual(eligibility["relation_review"]["decision"], "accept")
            self.assertTrue(eligibility["canonical_answer_alias_accepted"])
            self.assertIn(
                "apply_revision", item_by_id["fact_reject"]["allowed_dispositions"]
            )

            source_rows = {
                row["base_fact_id"]: row for row in read_jsonl(chain["full"])
            }
            resolutions = self.complete_resolutions(chain, templates)
            resolutions[2] = self.reject_override_revision(
                template_by_id["fact_reject"],
                item_by_id["fact_reject"],
                source_rows["fact_reject"],
            )
            resolutions_path = root / "reject-override-resolutions.jsonl"
            write_jsonl(resolutions_path, resolutions)
            applied = resolver.apply_resolutions(
                template_manifest_path=Path(result["manifest_path"]),
                resolutions_path=resolutions_path,
                output_dir=root / "reject-override-output",
            )
            output_manifest = json.loads(
                Path(applied["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(output_manifest["counts"]["source_reject_override_count"], 1)
            ledger = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    Path(applied["manifest_path"]).parent
                    / "fact_resolution_ledger.jsonl"
                )
            }
            overridden = ledger["fact_reject"]
            self.assertEqual(overridden["source_review_outcome"], "reject")
            self.assertTrue(overridden["source_reject_override_applied"])
            self.assertIn("target-level", overridden["source_reject_override_reason"])
            lineage = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    Path(applied["manifest_path"]).parent
                    / "fact_revision_lineage.jsonl"
                )
            }["fact_reject"]
            self.assertEqual(
                lineage["source_reject_override"]["original_review_outcome"],
                "reject",
            )
            self.assertFalse(lineage["human_gold"])

    def test_core_fact_or_relation_reject_cannot_be_overridden(self):
        for field, kwargs in (
            ("fact", {"reject_fact_decision": "reject"}),
            ("relation", {"reject_relation_decision": "reject"}),
        ):
            with self.subTest(core_field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                chain = self.build_chain(root, **kwargs)
                result = resolver.materialize_template(
                    full_base_facts_path=chain["full"],
                    review_apply_manifest_path=chain["apply"],
                    review_export_manifest_path=chain["export"],
                    output_dir=root / "resolution-template",
                )
                manifest = json.loads(
                    Path(result["manifest_path"]).read_text(encoding="utf-8")
                )
                items = {
                    row["base_fact_id"]: row
                    for row in read_jsonl(
                        Path(result["manifest_path"]).parent
                        / manifest["artifacts"]["resolution_items"]["filename"]
                    )
                }
                reject_item = items["fact_reject"]
                self.assertFalse(
                    reject_item["source_reject_override_eligibility"]["eligible"]
                )
                self.assertEqual(
                    reject_item["allowed_dispositions"], ["cohort_exclude"]
                )

    def test_rejected_canonical_answer_alias_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = self.build_chain(root, reject_canonical_alias=True)
            result = resolver.materialize_template(
                full_base_facts_path=chain["full"],
                review_apply_manifest_path=chain["apply"],
                review_export_manifest_path=chain["export"],
                output_dir=root / "resolution-template",
            )
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            items = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    Path(result["manifest_path"]).parent
                    / manifest["artifacts"]["resolution_items"]["filename"]
                )
            }
            eligibility = items["fact_reject"][
                "source_reject_override_eligibility"
            ]
            self.assertFalse(eligibility["eligible"])
            self.assertIn(
                "canonical_answer_alias_not_accept", eligibility["blocking_reasons"]
            )
            self.assertEqual(
                items["fact_reject"]["allowed_dispositions"], ["cohort_exclude"]
            )

    def test_distractor_only_reject_override_may_use_empty_fact_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = self.build_chain(root, reject_target="distractor")
            result = resolver.materialize_template(
                full_base_facts_path=chain["full"],
                review_apply_manifest_path=chain["apply"],
                review_export_manifest_path=chain["export"],
                output_dir=root / "resolution-template",
            )
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            items = read_jsonl(
                Path(result["manifest_path"]).parent
                / manifest["artifacts"]["resolution_items"]["filename"]
            )
            item_by_id = {row["base_fact_id"]: row for row in items}
            templates = read_jsonl(
                Path(result["manifest_path"]).parent
                / manifest["artifacts"]["resolution_decisions_template"]["filename"]
            )
            template_by_id = {row["base_fact_id"]: row for row in templates}
            source_rows = {
                row["base_fact_id"]: row for row in read_jsonl(chain["full"])
            }
            reject_resolution = self.reject_override_revision(
                template_by_id["fact_reject"],
                item_by_id["fact_reject"],
                source_rows["fact_reject"],
            )
            self.assertEqual(reject_resolution["revision"]["updates"], {})
            resolutions = self.complete_resolutions(chain, templates)
            resolutions[2] = reject_resolution
            resolutions_path = root / "distractor-only-resolutions.jsonl"
            write_jsonl(resolutions_path, resolutions)
            applied = resolver.apply_resolutions(
                template_manifest_path=Path(result["manifest_path"]),
                resolutions_path=resolutions_path,
                output_dir=root / "distractor-only-output",
            )
            lineages = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    Path(applied["manifest_path"]).parent
                    / "fact_revision_lineage.jsonl"
                )
            }
            target_only = lineages["fact_reject"]
            self.assertEqual(target_only["changed_fields"], [])
            self.assertEqual(
                target_only["before_full_fact_row_sha256"],
                target_only["after_full_fact_row_sha256"],
            )
            output_manifest = json.loads(
                Path(applied["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(
                output_manifest["semantic_closure_invalidation"][
                    "prior_semantic_candidate_manifest_invalidated"
                ]
            )
            same_sha_invalidation = resolver._semantic_invalidation(
                source_sha="a" * 64,
                output_sha="a" * 64,
                revision_count=1,
                exclusion_count=0,
                alias_filter_change_count=0,
                prior_manifest_bound=True,
            )
            self.assertFalse(
                same_sha_invalidation["source_and_resolved_sha256_differ"]
            )
            self.assertTrue(
                same_sha_invalidation[
                    "prior_semantic_candidate_manifest_invalidated"
                ]
            )

    def test_missing_resolution_fails_closed_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, result, _, templates = self.materialize(root)
            incomplete = self.complete_resolutions(chain, templates)[:-1]
            path = root / "incomplete.jsonl"
            write_jsonl(path, incomplete)
            output = root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "must exactly cover"):
                resolver.apply_resolutions(
                    template_manifest_path=Path(result["manifest_path"]),
                    resolutions_path=path,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_defer_cannot_apply_revision_and_exclusion_cannot_assert_falsehood(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, result, _, templates = self.materialize(root)
            template_by_id = {row["base_fact_id"]: row for row in templates}
            source = {row["base_fact_id"]: row for row in read_jsonl(chain["full"])}

            invalid_revision = self.revision(
                template_by_id["fact_defer"], source["fact_defer"]
            )
            invalid_revision["revision"]["updates"] = {
                "answer_en": "C revised",
                "answer_aliases_en": ["C revised"],
                "canonical_fact_en": "Subject fact_defer has C revised.",
            }
            path = root / "invalid-revision.jsonl"
            valid = self.complete_resolutions(chain, templates)
            valid[1] = invalid_revision
            write_jsonl(path, valid)
            with self.assertRaisesRegex(ValueError, "disposition is invalid"):
                resolver.apply_resolutions(
                    template_manifest_path=Path(result["manifest_path"]),
                    resolutions_path=path,
                    output_dir=root / "invalid-output",
                )

            false_exclusion = self.complete_resolutions(chain, templates)
            false_exclusion[2]["cohort_exclusion"]["factual_falsehood_asserted"] = True
            path = root / "false-exclusion.jsonl"
            write_jsonl(path, false_exclusion)
            with self.assertRaisesRegex(ValueError, "must not assert factual falsehood"):
                resolver.apply_resolutions(
                    template_manifest_path=Path(result["manifest_path"]),
                    resolutions_path=path,
                    output_dir=root / "false-output",
                )

    def test_revision_requires_independent_complete_sha_bound_rereview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, result, _, templates = self.materialize(root)
            cases = []
            stale = self.complete_resolutions(chain, templates)
            stale[0]["revision"]["rereview"]["revised_full_fact_row_sha256"] = "0" * 64
            cases.append((stale, "row SHA is stale"))
            same_reviewer = self.complete_resolutions(chain, templates)
            same_reviewer[0]["revision"]["rereview"]["reviewer_id"] = "revision-editor"
            cases.append((same_reviewer, "must be independent"))
            incomplete = self.complete_resolutions(chain, templates)
            del incomplete[0]["revision"]["rereview"]["field_reviews"]["relation_raw"]
            cases.append((incomplete, "field_reviews are incomplete"))
            for position, (rows, message) in enumerate(cases):
                path = root / f"bad-rereview-{position}.jsonl"
                write_jsonl(path, rows)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        resolver.apply_resolutions(
                            template_manifest_path=Path(result["manifest_path"]),
                            resolutions_path=path,
                            output_dir=root / f"bad-output-{position}",
                        )

    def test_validate_detects_tampered_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain, result, _, templates = self.materialize(root)
            resolutions_path = root / "resolutions.jsonl"
            write_jsonl(resolutions_path, self.complete_resolutions(chain, templates))
            applied = resolver.apply_resolutions(
                template_manifest_path=Path(result["manifest_path"]),
                resolutions_path=resolutions_path,
                output_dir=root / "resolved",
            )
            output_path = root / "resolved" / "revised_full_base_facts.jsonl"
            output_path.write_bytes(output_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 binding is stale"):
                resolver.validate_resolution_output(Path(applied["manifest_path"]))


if __name__ == "__main__":
    unittest.main()
