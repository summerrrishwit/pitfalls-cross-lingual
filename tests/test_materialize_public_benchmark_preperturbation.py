import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


review_tool = load_script("review_public_benchmark_bundle")
consolidator = load_script("consolidate_public_benchmark_review_outcomes")
materializer = load_script("materialize_public_benchmark_preperturbation")


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


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PreperturbationMaterializerTests(unittest.TestCase):
    relation_id = "song_to_performer"
    direction = "song -> performer"
    prompt_template = "The performer of {subject} is"
    splits = ("development", "validation", "sealed")

    def make_source_and_frame(self, root, *, date_leak_rows=()):
        source_dir = root / "source"
        policy_path = root / "relation_policy.json"
        policy = {
            "schema_version": "probe-relation-review-policy-v1",
            "policy_id": "fixture-song-relation-policy-v1",
            "status": "pre_perturbation_offline_only",
            "human_gold": False,
            "broad_relation_normalized_required": False,
            "selection_contract": {
                "prompt_ready_required": True,
                "prompt_risk_flags_must_be_empty": True,
                "one_base_fact_per_split_group": True,
                "model_outputs_must_not_be_used": True,
                "selection_precedes_fact_review": True,
                "shortfall_policy": "fail_closed",
            },
            "relations": [
                {
                    "probe_relation_id": self.relation_id,
                    "direction": self.direction,
                    "prompt_template_en": self.prompt_template,
                    "review_frame_split_counts": {
                        "development": 6,
                        "validation": 6,
                        "sealed": 6,
                    },
                    "target_accept_split_counts": {
                        "development": 3,
                        "validation": 3,
                        "sealed": 3,
                    },
                }
            ],
        }
        write_json(policy_path, policy)

        triples = []
        clusters = []
        queues = []
        behaviors = []
        frames = []
        rank_counts = {"development": 5, "validation": 3, "sealed": 3}
        date_leak_years = {"dev_1": "1994", "dev_5": "2005"}
        for split in self.splits:
            for rank in range(1, rank_counts[split] + 1):
                token = f"{split[:3]}_{rank}"
                base_fact_id = f"pbf_{token}"
                candidate_id = f"candidate_{token}"
                if token in set(date_leak_rows):
                    year = date_leak_years[token]
                    subject = f"Film {token} ({year} film)"
                    answer = f"{year}-06-24"
                    answer_type = "date"
                    answer_aliases = [answer, f"24 June {year}"]
                else:
                    subject = "The Lion King" if token == "dev_1" else f"Song {token}"
                    answer = f"Performer {token}"
                    answer_type = "person_or_group"
                    answer_aliases = [answer, f"{answer} alias"]
                input_hash = f"{len(triples) + 1:064x}"
                canonical_fact = f"{subject} is performed by {answer}."
                behaviors.append(
                    {
                        "schema_version": "factual-perturbation-input-bundle-v1",
                        "base_fact_id": base_fact_id,
                        "base_id": base_fact_id,
                        "candidate_id": candidate_id,
                        "relation_signature_id": "rel_signature_song_performer",
                        "subject_en": subject,
                        "relation_raw": "performed by",
                        "answer_en": answer,
                        "answer_type": answer_type,
                        "answer_aliases_en": answer_aliases,
                        "canonical_fact": canonical_fact,
                        "canonical_fact_en": canonical_fact,
                        "source_question_en": f"Who performs {subject}?",
                        "source_dataset": "fixture",
                        "source_format": "multiple_choice",
                        "split_group_id": f"split_group_{token}",
                        "split_assignment": split,
                        "split_status": "provisional_not_frozen",
                        "probe_relation_candidate": self.relation_id,
                        "probe_relation_candidate_direction": self.direction,
                        "distractor_candidates": [
                            {
                                "distractor_id": f"legacy_dist_{token}",
                                "text_en": f"Legacy wrong {token}",
                                "answer_en": f"Legacy wrong {token}",
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
                )
                triples.append(
                    {
                        "schema_version": "provisional-factual-triple-v1",
                        "base_fact_id": base_fact_id,
                        "candidate_id": candidate_id,
                        "subject": subject,
                        "relation_raw": "performed by",
                        "answer": answer,
                        "answer_aliases_en": answer_aliases,
                        "canonical_fact": canonical_fact,
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
                                "source_id": candidate_id,
                                "source_dataset": "fixture",
                            }
                        ],
                        "subject": subject,
                        "relation_raw": "performed by",
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
                        "subject": subject,
                        "relation_raw": "performed by",
                        "answer": answer,
                        "canonical_fact": canonical_fact,
                        "canonical_status": "pending_review",
                        "evidence_tier": "provisional_single_model",
                        "human_gold": False,
                        "review_decision": None,
                    }
                )

        paths = {
            "provisional_triples": source_dir / "provisional_triples.jsonl",
            "base_fact_clusters": source_dir / "base_fact_clusters.jsonl",
            "review_queue": source_dir / "review_queue.jsonl",
            "behavior_bundle": source_dir / "behavior_input_bundle.jsonl",
        }
        write_jsonl(paths["provisional_triples"], triples)
        write_jsonl(paths["base_fact_clusters"], clusters)
        write_jsonl(paths["review_queue"], queues)
        write_jsonl(paths["behavior_bundle"], behaviors)

        for split in self.splits:
            split_rows = [row for row in behaviors if row["split_assignment"] == split]
            for rank, row in enumerate(split_rows, start=1):
                copied_fields = {
                    field: copy.deepcopy(row[field])
                    for field in (
                        "candidate_id",
                        "relation_signature_id",
                        "subject_en",
                        "answer_en",
                        "answer_aliases_en",
                        "canonical_fact_en",
                        "source_question_en",
                        "source_dataset",
                        "source_format",
                        "split_group_id",
                        "split_assignment",
                        "split_status",
                        "probe_relation_candidate",
                        "probe_relation_candidate_direction",
                    )
                }
                frames.append(
                    {
                        "schema_version": "probe-relation-fact-review-frame-v1",
                        "base_fact_id": row["base_fact_id"],
                        **copied_fields,
                        "source_record_sha256": materializer.sha256_value(row),
                        "probe_relation_id": self.relation_id,
                        "signature_review_status": "approved",
                        "selection_before_fact_review": True,
                        "canonical_status": "pending_review",
                        "perturbation_ready": False,
                        "frame_rank_within_relation_split": rank,
                        "target_accept_count_for_split": 3,
                        "initial_target_slot": rank <= 3,
                        "proposed_probe_prompt_en": self.prompt_template.format(
                            subject=row["subject_en"]
                        ),
                    }
                )
        frame_path = root / "frame.jsonl"
        write_jsonl(frame_path, frames)
        return paths, frame_path, policy_path, behaviors, frames

    def decision(self, template, outcome):
        value = copy.deepcopy(template)
        value["decision"] = outcome
        value["review_provenance"] = {
            "reviewer_type": "codex_proxy",
            "reviewer_id": "fixture-initial-reviewer",
            "review_method": "fixture_full_semantic_review_v1",
            "reviewed_at": "2026-09-14T09:00:00+08:00",
        }
        for member in value["member_reviews"]:
            member["decision"] = "reject" if outcome == "reject" else "accept"
        if outcome == "accept":
            for field in ("alias_reviews", "distractor_reviews"):
                for item in value[field]:
                    item["decision"] = "accept"
        value["notes"] = f"Fixture outcome: {outcome}."
        return value

    def make_apply(
        self,
        root,
        paths,
        ids,
        outcomes,
        *,
        omitted=(),
        name="review",
    ):
        scope = review_tool.create_scope(
            output_dir=root / f"{name}_scope",
            artifact_paths=paths,
            base_fact_ids=ids,
        )
        export = review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=root / f"{name}_export",
        )
        export_manifest = json.loads(Path(export["manifest_path"]).read_text(encoding="utf-8"))
        templates = {
            row["base_fact_id"]: row
            for row in read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )
        }
        decisions = [
            self.decision(templates[base_fact_id], outcomes.get(base_fact_id, "accept"))
            for base_fact_id in ids
            if base_fact_id not in set(omitted)
        ]
        decisions_path = root / f"{name}_decisions.jsonl"
        write_jsonl(decisions_path, decisions)
        return review_tool.apply_decisions(
            scope_manifest_path=Path(scope["manifest_path"]),
            export_manifest_path=Path(export["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=root / f"{name}_apply",
            allow_partial=bool(omitted),
        )

    def revision(self, staging_by_id, base_fact_id, trigger, updates, *, decision="accept"):
        return {
            "base_fact_id": base_fact_id,
            "trigger": trigger,
            "source_review_staging_row_sha256": materializer.sha256_value(
                staging_by_id[base_fact_id]
            ),
            "updates": updates,
            "editor_type": "codex_proxy",
            "editor_id": "fixture-revision-editor",
            "edit_method": "explicit_field_revision_v1",
            "edited_at": "2026-09-14T10:00:00+08:00",
            "rereview": {
                "decision": decision,
                "reviewer_type": "codex_proxy",
                "reviewer_id": "fixture-independent-rereviewer",
                "review_method": "independent_revision_review_v1",
                "reviewed_at": "2026-09-14T11:00:00+08:00",
                "human_gold": False,
            },
        }

    def make_plan(self, root, source_path, frame_path, policy_path, manifest_paths, revisions):
        plan = {
            "schema_version": "public-benchmark-preperturbation-revision-plan-v1",
            "source_behavior_bundle": {
                "path": str(source_path.resolve()),
                "sha256": materializer.sha256_file(source_path),
                "record_count": len(read_jsonl(source_path)),
            },
            "frame": {
                "path": str(frame_path.resolve()),
                "sha256": materializer.sha256_file(frame_path),
                "record_count": len(read_jsonl(frame_path)),
            },
            "relation_policy": {
                "path": str(policy_path.resolve()),
                "sha256": materializer.sha256_file(policy_path),
            },
            "review_apply_manifests": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": materializer.sha256_file(Path(path)),
                }
                for path in manifest_paths
            ],
            "revisions": revisions,
        }
        plan_path = root / "revision_plan.json"
        write_json(plan_path, plan)
        return plan_path, plan

    def build_fixture(
        self, root, *, outcome_overrides=None, omitted=(), date_leak_rows=()
    ):
        paths, frame_path, policy_path, behaviors, frames = self.make_source_and_frame(
            root, date_leak_rows=date_leak_rows
        )
        ids = [row["base_fact_id"] for row in frames]
        outcomes = {"pbf_dev_2": "reject", "pbf_dev_3": "revise"}
        outcomes.update(outcome_overrides or {})
        apply = self.make_apply(root, paths, ids, outcomes, omitted=omitted)
        manifest_path = Path(apply["manifest_path"])
        staging_by_id = {
            row["base_fact_id"]: row for row in read_jsonl(Path(apply["staging_path"]))
        }
        dev_1_date_leak = "dev_1" in set(date_leak_rows)
        revisions = [
            self.revision(
                staging_by_id,
                "pbf_dev_1",
                "semantic_closure_disambiguation",
                {
                    "subject_en": (
                        "Film dev_1 (United States wide theatrical release)"
                        if dev_1_date_leak
                        else "The Lion King (1994 film)"
                    ),
                    "canonical_fact": (
                        "Film dev_1 had its United States wide theatrical release "
                        "on 1994-06-24."
                        if dev_1_date_leak
                        else "The Lion King (1994 film) is performed by Performer dev_1."
                    ),
                    "canonical_fact_en": (
                        "Film dev_1 had its United States wide theatrical release "
                        "on 1994-06-24."
                        if dev_1_date_leak
                        else "The Lion King (1994 film) is performed by Performer dev_1."
                    ),
                },
            ),
            self.revision(
                staging_by_id,
                "pbf_dev_3",
                "review_requested_revision",
                {
                    "subject_en": "Song dev_3 (studio recording)",
                    "canonical_fact": (
                        "Song dev_3 (studio recording) is performed by Performer dev_3."
                    ),
                    "canonical_fact_en": (
                        "Song dev_3 (studio recording) is performed by Performer dev_3."
                    ),
                },
            ),
        ]
        source_path = paths["behavior_bundle"]
        plan_path, plan = self.make_plan(
            root,
            source_path,
            frame_path,
            policy_path,
            [manifest_path],
            revisions,
        )
        return {
            "paths": paths,
            "source_path": source_path,
            "frame_path": frame_path,
            "policy_path": policy_path,
            "behaviors": behaviors,
            "frames": frames,
            "ids": ids,
            "apply": apply,
            "manifest_paths": [manifest_path],
            "staging_by_id": staging_by_id,
            "plan_path": plan_path,
            "plan": plan,
        }

    def run_materializer(self, root, fixture, *, output_name="output", quota=3):
        return materializer.materialize_preperturbation_bundle(
            source_bundle_path=fixture["source_path"],
            frame_path=fixture["frame_path"],
            review_apply_manifest_paths=fixture["manifest_paths"],
            revision_plan_path=fixture["plan_path"],
            relation_policy_path=fixture["policy_path"],
            output_dir=root / output_name,
            development_quota=quota,
            validation_quota=quota,
            sealed_quota=quota,
            expected_frame_count=len(read_jsonl(fixture["frame_path"])),
        )

    def consolidate_fixture(self, root, fixture):
        retained_id = "pbf_dev_3"
        resolution_plan = {
            "schema_version": consolidator.PLAN_SCHEMA_VERSION,
            "review_apply_manifests": [
                {
                    "path": str(path.resolve()),
                    "sha256": consolidator.sha256_file(path),
                }
                for path in fixture["manifest_paths"]
            ],
            "resolutions": [
                {
                    "base_fact_id": retained_id,
                    "source_review_outcome": "revise",
                    "source_review_staging_row_sha256": consolidator.sha256_value(
                        fixture["staging_by_id"][retained_id]
                    ),
                    "disposition": "retain_revise",
                    "reason": "Retain the fixture revision for independent rereview.",
                    "reviewer_provenance": {
                        "reviewer_type": "codex_proxy",
                        "reviewer_id": "fixture-independent-terminal-reviewer",
                        "review_method": "independent_terminal_resolution_v1",
                        "reviewed_at": "2026-09-14T12:00:00+08:00",
                    },
                }
            ],
        }
        resolution_plan_path = root / "resolution_plan.json"
        write_json(resolution_plan_path, resolution_plan)
        consolidation = consolidator.consolidate_review_outcomes(
            review_apply_manifest_paths=fixture["manifest_paths"],
            resolution_plan_path=resolution_plan_path,
            output_dir=root / "consolidated",
        )
        consolidated_apply_path = Path(consolidation["review_apply_manifest_path"])
        apply_manifest = json.loads(
            consolidated_apply_path.read_text(encoding="utf-8")
        )
        consolidated_staging = {
            row["base_fact_id"]: row
            for row in read_jsonl(
                Path(apply_manifest["artifacts"]["review_staging"]["path"])
            )
        }
        old_revisions = {
            entry["base_fact_id"]: entry for entry in fixture["plan"]["revisions"]
        }
        revisions = [
            self.revision(
                consolidated_staging,
                base_fact_id,
                entry["trigger"],
                copy.deepcopy(entry["updates"]),
                decision=entry["rereview"]["decision"],
            )
            for base_fact_id, entry in old_revisions.items()
        ]
        plan_path, plan = self.make_plan(
            root,
            fixture["source_path"],
            fixture["frame_path"],
            fixture["policy_path"],
            [consolidated_apply_path],
            revisions,
        )
        consolidation_manifest_path = Path(consolidation["manifest_path"])
        plan["review_outcome_consolidation"] = {
            "path": str(consolidation_manifest_path.resolve()),
            "sha256": materializer.sha256_file(consolidation_manifest_path),
        }
        write_json(plan_path, plan)
        fixture.update(
            manifest_paths=[consolidated_apply_path],
            staging_by_id=consolidated_staging,
            plan_path=plan_path,
            plan=plan,
            consolidation=consolidation,
        )
        return fixture

    def rewrite_plan(self, fixture):
        write_json(fixture["plan_path"], fixture["plan"])

    def test_real_review_chain_selects_accept_reject_and_revised_rows_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            result = self.run_materializer(root, fixture)
            output = Path(result["output_dir"])

            selected = json.loads((output / "selected_base_fact_ids.json").read_text())
            self.assertEqual(
                selected["selected_base_fact_ids_by_relation_split"][
                    "song_to_performer/development"
                ],
                ["pbf_dev_1", "pbf_dev_3", "pbf_dev_4"],
            )
            self.assertNotIn("pbf_dev_2", selected["selected_base_fact_ids"])
            self.assertEqual(selected["selected_base_fact_count"], 9)

            rows = {
                row["base_fact_id"]: row
                for row in read_jsonl(output / "preperturbation_behavior_input_bundle.jsonl")
            }
            source_rows = {
                row["base_fact_id"]: row for row in read_jsonl(fixture["source_path"])
            }
            self.assertEqual(
                rows["pbf_dev_1"]["prompt_en"],
                "The Lion King (1994 film) is performed by",
            )
            self.assertEqual(
                rows["pbf_dev_3"]["prompt_en"],
                "Song dev_3 (studio recording) is performed by",
            )
            self.assertEqual(
                rows["pbf_dev_1"]["prompt_template_id"],
                "canonical_fact_answer_suffix_en_provisional_v1",
            )
            self.assertEqual(
                rows["pbf_dev_1"]["prompt_quality_tier"],
                "strict_factual_completion",
            )
            self.assertEqual(
                rows["pbf_dev_1"]["prompt_derivation_method"],
                "canonical_fact_en_terminal_answer_or_accepted_alias_suffix_split",
            )
            for row in rows.values():
                self.assertEqual(row["canonical_status"], "pending_review")
                self.assertEqual(row["split_status"], "provisional_not_frozen")
                self.assertFalse(row["human_gold"])
                self.assertFalse(row["perturbation_ready"])
                self.assertFalse(row["path_not_token_experiment_ready"])
                self.assertEqual(
                    row["preperturbation_lineage"]["source_record_sha256"],
                    materializer.sha256_value(source_rows[row["base_fact_id"]]),
                )
                self.assertEqual(row["translation_status"], "pending_generation_and_review")
                for field in (
                    "subject_zh",
                    "answer_zh",
                    "answer_aliases_zh",
                    "canonical_fact_zh",
                    "prompt_zh",
                    "distractor_candidates_zh",
                ):
                    self.assertIsNone(row[field])
                self.assertEqual(len(row["distractor_candidates"]), 2)
                for distractor in row["distractor_candidates"]:
                    self.assertEqual(
                        distractor["verification_status"], "codex_proxy_pending_review"
                    )
                    self.assertEqual(distractor["status"], "codex_proxy_pending_review")
                    self.assertFalse(distractor["verified"])

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["semantic_closure_disambiguation_count"], 1)
            self.assertEqual(summary["review_requested_revision_count"], 1)
            self.assertEqual(summary["revision_count"], 2)
            self.assertTrue(summary["safety_contract"]["output_is_provisional_preperturbation_only"])
            for key, value in summary["safety_contract"].items():
                if key != "output_is_provisional_preperturbation_only":
                    self.assertFalse(value, key)
            lineage = read_jsonl(output / "revision_lineage.jsonl")
            self.assertEqual({row["trigger"] for row in lineage}, {
                "semantic_closure_disambiguation",
                "review_requested_revision",
            })
            self.assertTrue(all(row["derived_prompt_recomputed"] for row in lineage))
            self.assertTrue(all(not row["canonical_or_split_freeze_performed"] for row in lineage))
            manifest = json.loads((output / "selection_manifest.json").read_text())
            self.assertEqual(
                manifest["inputs"]["revision_plan"]["sha256"],
                materializer.sha256_file(fixture["plan_path"]),
            )

    def test_accept_revision_requires_semantic_trigger_and_independent_rereview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            base_plan = copy.deepcopy(fixture["plan"])
            cases = (
                (
                    lambda revision: revision.update(trigger="review_requested_revision"),
                    "Accept outcome revisions require semantic_closure_disambiguation",
                ),
                (
                    lambda revision: revision["rereview"].update(
                        reviewer_id=revision["editor_id"]
                    ),
                    "editor and rereviewer must be independent",
                ),
            )
            for index, (mutate, error) in enumerate(cases):
                with self.subTest(error=error):
                    changed = copy.deepcopy(base_plan)
                    mutate(changed["revisions"][0])
                    fixture["plan"] = changed
                    self.rewrite_plan(fixture)
                    with self.assertRaisesRegex(ValueError, error):
                        self.run_materializer(root, fixture, output_name=f"invalid-{index}")

    def test_revision_cannot_change_immutable_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            fixture["plan"]["revisions"][0]["updates"]["split_assignment"] = "sealed"
            self.rewrite_plan(fixture)
            with self.assertRaisesRegex(ValueError, "immutable fields"):
                self.run_materializer(root, fixture)

    def test_missing_defer_and_unresolved_revise_before_cutoff_fail_closed(self):
        cases = ((None, True, "missing review"), ("defer", False, "defer review"), ("revise", False, "unresolved revise"))
        for index, (outcome, omitted, error) in enumerate(cases):
            with self.subTest(outcome=outcome or "missing"):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    overrides = {} if omitted else {"pbf_dev_1": outcome}
                    fixture = self.build_fixture(
                        root,
                        outcome_overrides=overrides,
                        omitted=("pbf_dev_1",) if omitted else (),
                    )
                    fixture["plan"]["revisions"] = [
                        entry
                        for entry in fixture["plan"]["revisions"]
                        if entry["base_fact_id"] != "pbf_dev_1"
                    ]
                    self.rewrite_plan(fixture)
                    with self.assertRaisesRegex(ValueError, error):
                        self.run_materializer(root, fixture, output_name=f"blocked-{index}")

    def test_quota_shortfall_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            fixture["plan"]["revisions"] = []
            # Reject all development rows; no eligible row can fill even quota one.
            paths, frame_path, policy_path, _, frames = self.make_source_and_frame(root / "short")
            ids = [row["base_fact_id"] for row in frames]
            apply = self.make_apply(
                root / "short",
                paths,
                ids,
                {base_fact_id: "reject" for base_fact_id in ids if base_fact_id.startswith("pbf_dev")},
            )
            short_fixture = {
                "source_path": paths["behavior_bundle"],
                "frame_path": frame_path,
                "policy_path": policy_path,
                "manifest_paths": [Path(apply["manifest_path"])],
            }
            plan_path, plan = self.make_plan(
                root / "short",
                short_fixture["source_path"],
                frame_path,
                policy_path,
                short_fixture["manifest_paths"],
                [],
            )
            short_fixture.update(plan_path=plan_path, plan=plan)
            with self.assertRaisesRegex(ValueError, "Insufficient eligible rows"):
                self.run_materializer(root / "short", short_fixture, quota=1)

    def test_revision_plan_hash_and_path_bindings_are_strict(self):
        mutations = (
            (lambda plan: plan["source_behavior_bundle"].update(sha256="0" * 64), "source_behavior_bundle.sha256 is stale"),
            (lambda plan: plan["frame"].update(sha256="0" * 64), "frame.sha256 is stale"),
            (lambda plan: plan["review_apply_manifests"][0].update(sha256="0" * 64), r"review_apply_manifests\[1\].sha256 is stale"),
            (lambda plan: plan["frame"].update(path="/tmp/not-the-bound-frame.jsonl"), "frame.path does not match"),
        )
        for index, (mutate, error) in enumerate(mutations):
            with self.subTest(error=error):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    fixture = self.build_fixture(root)
                    mutate(fixture["plan"])
                    self.rewrite_plan(fixture)
                    with self.assertRaisesRegex(ValueError, error):
                        self.run_materializer(root, fixture, output_name=f"tampered-{index}")

    def test_duplicate_frame_rank_and_stale_source_hash_fail(self):
        mutations = (
            (lambda rows: rows[1].update(frame_rank_within_relation_split=1), "Duplicate frame rank"),
            (lambda rows: rows[0].update(source_record_sha256="0" * 64), "source_record_sha256 is stale"),
        )
        for index, (mutate, error) in enumerate(mutations):
            with self.subTest(error=error):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    fixture = self.build_fixture(root)
                    rows = read_jsonl(fixture["frame_path"])
                    mutate(rows)
                    write_jsonl(fixture["frame_path"], rows)
                    fixture["plan"]["frame"]["sha256"] = materializer.sha256_file(
                        fixture["frame_path"]
                    )
                    self.rewrite_plan(fixture)
                    with self.assertRaisesRegex(ValueError, error):
                        self.run_materializer(root, fixture, output_name=f"bad-frame-{index}")

    def test_duplicate_base_fact_id_across_apply_manifests_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            duplicate = self.make_apply(
                root,
                fixture["paths"],
                ["pbf_dev_1"],
                {"pbf_dev_1": "accept"},
                name="duplicate",
            )
            fixture["manifest_paths"].append(Path(duplicate["manifest_path"]))
            fixture["plan"]["review_apply_manifests"].append(
                {
                    "path": str(Path(duplicate["manifest_path"]).resolve()),
                    "sha256": materializer.sha256_file(Path(duplicate["manifest_path"])),
                }
            )
            self.rewrite_plan(fixture)
            with self.assertRaisesRegex(ValueError, "Duplicate base_fact_id across"):
                self.run_materializer(root, fixture)

    def test_two_alias_disjoint_distractor_donors_are_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root)
            # Quota two yields only one possible donor per relation/split.
            with self.assertRaisesRegex(ValueError, "Fewer than 2 same-relation/split"):
                self.run_materializer(root, fixture, quota=2)

    def test_consolidation_evidence_is_validated_and_bound_to_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.consolidate_fixture(root, self.build_fixture(root))
            result = self.run_materializer(root, fixture)
            output = Path(result["output_dir"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["review_outcome_consolidation_used"])
            self.assertEqual(summary["cohort_resolution_record_count"], 1)
            self.assertEqual(
                summary["cohort_resolution_disposition_counts"],
                {"retain_revise": 1},
            )
            manifest = json.loads(
                (output / "selection_manifest.json").read_text(encoding="utf-8")
            )
            consolidation_sha = materializer.sha256_file(
                Path(fixture["consolidation"]["manifest_path"])
            )
            self.assertEqual(
                manifest["inputs"]["review_outcome_consolidation"]["sha256"],
                consolidation_sha,
            )
            rows = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    output / "preperturbation_behavior_input_bundle.jsonl"
                )
            }
            retained_lineage = rows["pbf_dev_3"]["preperturbation_lineage"]
            self.assertEqual(
                retained_lineage["review_outcome_consolidation_manifest_sha256"],
                consolidation_sha,
            )
            self.assertEqual(
                retained_lineage["cohort_eligibility_disposition"],
                "retain_revise",
            )
            self.assertFalse(
                retained_lineage["cohort_resolution_asserts_factual_rejection"]
            )

            fixture["plan"]["review_outcome_consolidation"]["sha256"] = "0" * 64
            self.rewrite_plan(fixture)
            with self.assertRaisesRegex(
                ValueError, "review_outcome_consolidation.sha256 is stale"
            ):
                self.run_materializer(root, fixture, output_name="tampered")

    def test_date_prompt_rejects_partial_answer_year_leakage(self):
        row = {
            "answer_en": "2019-07-19",
            "answer_aliases_en": ["2019-07-19", "19 July 2019"],
            "answer_type": "date",
        }
        with self.assertRaisesRegex(ValueError, "date answer year"):
            materializer._validate_prompt_no_answer(
                "The release date of The Lion King (2019 film) is",
                row,
                base_fact_id="pbf_date_fixture",
            )
        materializer._validate_prompt_no_answer(
            "The release date of The Lion King remake is",
            row,
            base_fact_id="pbf_date_fixture",
        )

    def test_frame_allows_unselected_leak_and_selected_revision_is_checked_finally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(
                root, date_leak_rows=("dev_1", "dev_5")
            )
            result = self.run_materializer(root, fixture)
            output = Path(result["output_dir"])
            selected = json.loads(
                (output / "selected_base_fact_ids.json").read_text(encoding="utf-8")
            )
            self.assertIn("pbf_dev_1", selected["selected_base_fact_ids"])
            self.assertNotIn("pbf_dev_5", selected["selected_base_fact_ids"])
            rows = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    output / "preperturbation_behavior_input_bundle.jsonl"
                )
            }
            self.assertNotIn("1994", rows["pbf_dev_1"]["prompt_en"])

    def test_selected_unrevised_date_year_leak_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.build_fixture(root, date_leak_rows=("dev_1",))
            fixture["plan"]["revisions"] = [
                entry
                for entry in fixture["plan"]["revisions"]
                if entry["base_fact_id"] != "pbf_dev_1"
            ]
            self.rewrite_plan(fixture)
            with self.assertRaisesRegex(ValueError, "date answer year"):
                self.run_materializer(root, fixture)

    def test_canonical_suffix_prompt_preserves_release_qualifiers_and_falls_back(self):
        relation_spec = {
            "prompt_template_en": "The release date of {subject} is"
        }
        cases = (
            (
                {
                    "subject_en": (
                        "The first Guardians of the Galaxy film "
                        "(United Kingdom theatrical release)"
                    ),
                    "answer_en": "2014-07-31",
                    "answer_aliases_en": ["2014-07-31", "31 July 2014"],
                    "answer_type": "date",
                    "canonical_fact_en": (
                        "The first Guardians of the Galaxy film was theatrically "
                        "released in the United Kingdom on 2014-07-31."
                    ),
                },
                (
                    "The first Guardians of the Galaxy film was theatrically "
                    "released in the United Kingdom on"
                ),
            ),
            (
                {
                    "subject_en": (
                        "A Star Is Born (United States theatrical release)"
                    ),
                    "answer_en": "2018-10-05",
                    "answer_aliases_en": ["2018-10-05", "5 October 2018"],
                    "answer_type": "date",
                    "canonical_fact_en": (
                        "A Star Is Born was theatrically released in the United "
                        "States on 5 October 2018."
                    ),
                },
                (
                    "A Star Is Born was theatrically released in the United States on"
                ),
            ),
        )
        for index, (row, expected_prompt) in enumerate(cases):
            with self.subTest(index=index):
                prompt, metadata = materializer._derive_open_completion_prompt(
                    relation_spec,
                    row,
                    relation_id="work_to_release_date",
                    base_fact_id=f"pbf_release_{index}",
                )
                self.assertEqual(prompt, expected_prompt)
                self.assertEqual(
                    metadata["prompt_template_id"],
                    "canonical_fact_answer_suffix_en_provisional_v1",
                )
                self.assertEqual(
                    metadata["prompt_quality_tier"], "strict_factual_completion"
                )
                self.assertEqual(
                    metadata["prompt_derivation_method"],
                    "canonical_fact_en_terminal_answer_or_accepted_alias_suffix_split",
                )
                materializer._validate_prompt_no_answer(
                    prompt, row, base_fact_id=f"pbf_release_{index}"
                )

        fallback_row = {
            "subject_en": "Example film",
            "answer_en": "2020-01-02",
            "answer_aliases_en": ["2020-01-02", "2 January 2020"],
            "answer_type": "date",
            "canonical_fact_en": "Example film has an unresolved release record.",
        }
        fallback_prompt, fallback_metadata = (
            materializer._derive_open_completion_prompt(
                relation_spec,
                fallback_row,
                relation_id="work_to_release_date",
                base_fact_id="pbf_release_fallback",
            )
        )
        self.assertEqual(fallback_prompt, "The release date of Example film is")
        self.assertEqual(
            fallback_metadata["prompt_template_id"],
            "probe_relation_open_completion_en_work_to_release_date_v1",
        )
        self.assertEqual(
            fallback_metadata["prompt_quality_tier"],
            "relation_specific_open_completion",
        )
        self.assertEqual(
            fallback_metadata["prompt_derivation_method"],
            "relation_template_fallback",
        )


if __name__ == "__main__":
    unittest.main()
