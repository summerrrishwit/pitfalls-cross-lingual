import copy
import importlib.util
import json
import shutil
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
    "review_public_benchmark_bundle_for_consolidation_test",
    PROJECT_ROOT / "scripts" / "review_public_benchmark_bundle.py",
)
finalizer = load_module(
    "finalize_public_benchmark_bundle_for_consolidation_test",
    PROJECT_ROOT / "scripts" / "finalize_public_benchmark_bundle.py",
)
consolidator = load_module(
    "consolidate_public_benchmark_review_outcomes",
    PROJECT_ROOT / "scripts" / "consolidate_public_benchmark_review_outcomes.py",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def triple(base_fact_id, candidate_id, answer):
    return {
        "schema_version": "provisional-factual-triple-v1",
        "base_fact_id": base_fact_id,
        "candidate_id": candidate_id,
        "subject": f"Subject {base_fact_id}",
        "relation_raw": "example relation",
        "answer": answer,
        "answer_aliases_en": [answer],
        "canonical_fact": f"Subject {base_fact_id} has {answer}.",
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "provenance": {"input_record_sha256": candidate_id[-1] * 64},
    }


def cluster(base_fact_id, candidate_id, answer):
    return {
        "schema_version": "provisional-base-fact-cluster-v1",
        "base_fact_id": base_fact_id,
        "representative_candidate_id": candidate_id,
        "member_count": 1,
        "members": [
            {
                "candidate_id": candidate_id,
                "input_record_sha256": candidate_id[-1] * 64,
                "source_id": candidate_id,
                "source_dataset": "fixture",
            }
        ],
        "subject": f"Subject {base_fact_id}",
        "relation_raw": "example relation",
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
        "relation_raw": "example relation",
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
        "relation_raw": "example relation",
        "answer_en": answer,
        "answer_aliases_en": [answer, f"{answer} alias"],
        "canonical_fact_en": f"Subject {base_fact_id} has {answer}.",
        "distractor_candidates": [
            {
                "distractor_id": f"dist_{base_fact_id}",
                "text_en": f"Wrong {answer}",
                "answer_en": f"Wrong {answer}",
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


class ConsolidatePublicBenchmarkReviewOutcomesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.ids = {
            "accept": "pbf_accept",
            "reject": "pbf_reject",
            "revise": "pbf_revise",
            "defer": "pbf_defer",
        }
        triples = []
        clusters = []
        queues = []
        bundles = []
        for index, base_fact_id in enumerate(self.ids.values(), start=1):
            candidate_id = f"candidate_{index}"
            answer = f"Answer {index}"
            triples.append(triple(base_fact_id, candidate_id, answer))
            clusters.append(cluster(base_fact_id, candidate_id, answer))
            queues.append(queue(base_fact_id, candidate_id, answer))
            bundles.append(behavior(base_fact_id, candidate_id, answer))
        write_jsonl(self.source / "provisional_triples.jsonl", triples)
        write_jsonl(self.source / "base_fact_clusters.jsonl", clusters)
        write_jsonl(self.source / "review_queue.jsonl", queues)
        write_jsonl(self.source / "behavior_input_bundle.jsonl", bundles)
        self.artifact_paths = review_tool.source_paths(self.source)
        self.chain_a = self.make_chain(
            "a", [(self.ids["accept"], "accept"), (self.ids["revise"], "revise")]
        )
        self.chain_b = self.make_chain(
            "b", [(self.ids["reject"], "reject"), (self.ids["defer"], "defer")]
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_decision(self, template, outcome, reviewer_id):
        decision = copy.deepcopy(template)
        decision["decision"] = outcome
        decision["review_provenance"] = {
            "reviewer_type": "codex_proxy",
            "reviewer_id": reviewer_id,
            "review_method": "fixture_semantic_review_v1",
            "reviewed_at": "2026-09-14T08:00:00+08:00",
        }
        for field in ("member_reviews", "alias_reviews", "distractor_reviews"):
            for target in decision[field]:
                target["decision"] = "accept"
        decision["notes"] = f"Original {outcome} review note."
        return decision

    def make_chain(self, name, id_outcomes):
        scope = review_tool.create_scope(
            output_dir=self.root / name / "scope",
            artifact_paths=self.artifact_paths,
            base_fact_ids=[base_fact_id for base_fact_id, _ in id_outcomes],
        )
        export = review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=self.root / name / "export",
        )
        export_manifest = json.loads(
            Path(export["manifest_path"]).read_text(encoding="utf-8")
        )
        template_path = Path(
            export_manifest["artifacts"]["review_decisions_template"]["path"]
        )
        templates = {
            row["base_fact_id"]: row for row in read_jsonl(template_path)
        }
        decisions = [
            self.make_decision(
                templates[base_fact_id],
                outcome,
                f"original-reviewer-{name}",
            )
            for base_fact_id, outcome in id_outcomes
        ]
        decisions_path = self.root / name / "decisions.jsonl"
        write_jsonl(decisions_path, decisions)
        applied = review_tool.apply_decisions(
            scope_manifest_path=Path(scope["manifest_path"]),
            export_manifest_path=Path(export["manifest_path"]),
            decisions_path=decisions_path,
            output_dir=self.root / name / "apply",
            allow_partial=False,
        )
        return {
            "scope": scope,
            "export": export,
            "decisions_path": decisions_path,
            "apply": applied,
        }

    def resolution(self, base_fact_id, source_outcome, disposition):
        rows = {}
        for chain in (self.chain_a, self.chain_b):
            for row in read_jsonl(Path(chain["apply"]["staging_path"])):
                rows[row["base_fact_id"]] = row
        return {
            "base_fact_id": base_fact_id,
            "source_review_outcome": source_outcome,
            "source_review_staging_row_sha256": consolidator.sha256_value(
                rows[base_fact_id]
            ),
            "disposition": disposition,
            "reason": f"Fixture resolution for {base_fact_id}.",
            "reviewer_provenance": {
                "reviewer_type": "codex_proxy",
                "reviewer_id": f"independent-terminal-reviewer-{base_fact_id}",
                "review_method": "independent_terminal_cohort_resolution_v1",
                "reviewed_at": "2026-09-14T09:00:00+08:00",
            },
        }

    def write_plan(self, resolutions, *, paths=None, name="resolution-plan.json"):
        manifest_paths = list(
            paths
            or [
                Path(self.chain_a["apply"]["manifest_path"]),
                Path(self.chain_b["apply"]["manifest_path"]),
            ]
        )
        plan = {
            "schema_version": consolidator.PLAN_SCHEMA_VERSION,
            "review_apply_manifests": [
                {
                    "path": str(path),
                    "sha256": consolidator.sha256_file(path),
                }
                for path in manifest_paths
            ],
            "resolutions": resolutions,
        }
        path = self.root / name
        write_json(path, plan)
        return path, manifest_paths

    def valid_resolutions(self):
        return [
            self.resolution(self.ids["revise"], "revise", "retain_revise"),
            self.resolution(self.ids["defer"], "defer", "cohort_exclude"),
        ]

    def test_consolidates_and_replays_without_changing_terminal_decisions(self):
        plan_path, manifest_paths = self.write_plan(self.valid_resolutions())
        output = self.root / "consolidated"
        result = consolidator.consolidate_review_outcomes(
            review_apply_manifest_paths=manifest_paths,
            resolution_plan_path=plan_path,
            output_dir=output,
        )
        self.assertEqual(
            result["outcome_counts"], {"accept": 1, "reject": 2, "revise": 1}
        )
        apply_manifest = json.loads(
            Path(result["review_apply_manifest_path"]).read_text(encoding="utf-8")
        )
        staging = {
            row["base_fact_id"]: row
            for row in read_jsonl(
                Path(apply_manifest["artifacts"]["review_staging"]["path"])
            )
        }
        self.assertEqual(staging[self.ids["accept"]]["review_outcome"], "accept")
        self.assertEqual(staging[self.ids["reject"]]["review_outcome"], "reject")
        self.assertEqual(staging[self.ids["revise"]]["review_outcome"], "revise")
        excluded = staging[self.ids["defer"]]
        self.assertEqual(excluded["review_outcome"], "reject")
        self.assertIn(consolidator.COHORT_EXCLUSION_NOTE, excluded["notes"])
        self.assertEqual(
            excluded["review_provenance"]["reviewer_id"],
            f"independent-terminal-reviewer-{self.ids['defer']}",
        )
        self.assertIn(
            consolidator.RETAIN_REVISE_NOTE,
            staging[self.ids["revise"]]["notes"],
        )
        self.assertEqual(
            staging[self.ids["accept"]]["review_provenance"]["reviewer_id"],
            "original-reviewer-a",
        )
        self.assertEqual(
            staging[self.ids["reject"]]["review_provenance"]["reviewer_id"],
            "original-reviewer-b",
        )
        new_decisions = {
            row["base_fact_id"]: row
            for row in read_jsonl(output / "decisions" / "review_decisions.jsonl")
        }
        original_decisions = {}
        for chain in (self.chain_a, self.chain_b):
            original_decisions.update(
                {
                    row["base_fact_id"]: row
                    for row in read_jsonl(chain["decisions_path"])
                }
            )
        for base_fact_id in (self.ids["accept"], self.ids["reject"]):
            expected = copy.deepcopy(original_decisions[base_fact_id])
            actual = copy.deepcopy(new_decisions[base_fact_id])
            for row in (expected, actual):
                row.pop("scope_id")
                row.pop("review_item_sha256")
            self.assertEqual(actual, expected)
        scope = json.loads(
            (output / "scope" / "review_scope_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        replayed, provenance = finalizer.validate_review_apply_manifest(
            Path(result["review_apply_manifest_path"]),
            universe={"source_artifacts": scope["source_artifacts"]},
            universe_source_ids=list(self.ids.values()),
        )
        self.assertEqual(set(replayed), set(self.ids.values()))
        self.assertTrue(provenance["deterministic_replay"]["performed"])
        manifest = json.loads(
            Path(result["manifest_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["counts"]["resolution_disposition_counts"],
            {"cohort_exclude": 1, "retain_revise": 1},
        )
        resolution_rows = {
            row["base_fact_id"]: row
            for row in read_jsonl(
                Path(
                    manifest["artifacts"]["cohort_eligibility_resolutions"][
                        "path"
                    ]
                )
            )
        }
        deferred_resolution = resolution_rows[self.ids["defer"]]
        self.assertEqual(
            deferred_resolution["original_review_outcome"], "defer"
        )
        self.assertEqual(
            deferred_resolution["cohort_eligibility_disposition"],
            "cohort_exclude",
        )
        self.assertEqual(
            deferred_resolution["cohort_exclusion_reason"],
            f"Fixture resolution for {self.ids['defer']}.",
        )
        self.assertFalse(deferred_resolution["factual_rejection_asserted"])
        self.assertEqual(
            deferred_resolution["original_review_bindings"]["decision"][
                "record_sha256"
            ],
            consolidator.sha256_value(
                {
                    row["base_fact_id"]: row
                    for row in read_jsonl(self.chain_b["decisions_path"])
                }[self.ids["defer"]]
            ),
        )
        self.assertTrue(
            manifest["cohort_eligibility_contract"][
                "selector_must_consult_cohort_eligibility_resolutions"
            ]
        )
        self.assertFalse(
            manifest["cohort_eligibility_contract"][
                "cohort_exclude_asserts_factual_falsehood"
            ]
        )
        self.assertFalse(manifest["safety_contract"]["model_or_api_used"])
        self.assertFalse(manifest["safety_contract"]["perturbation_performed"])
        self.assertFalse(
            manifest["safety_contract"][
                "factual_rejections_created_from_cohort_exclusions"
            ]
        )

    def test_plan_must_cover_every_and_only_revise_or_defer(self):
        cases = []
        cases.append((self.valid_resolutions()[:1], "exactly cover"))
        extra = self.valid_resolutions()
        extra.append(
            self.resolution(self.ids["accept"], "accept", "cohort_exclude")
        )
        cases.append((extra, "targets terminal input outcome"))
        for index, (resolutions, message) in enumerate(cases):
            with self.subTest(case=index):
                plan_path, manifest_paths = self.write_plan(
                    resolutions, name=f"invalid-plan-{index}.json"
                )
                with self.assertRaisesRegex(ValueError, message):
                    consolidator.consolidate_review_outcomes(
                        review_apply_manifest_paths=manifest_paths,
                        resolution_plan_path=plan_path,
                        output_dir=self.root / f"invalid-output-{index}",
                    )

    def test_plan_binds_manifest_path_and_sha(self):
        plan_path, manifest_paths = self.write_plan(self.valid_resolutions())
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["review_apply_manifests"][0]["sha256"] = "0" * 64
        write_json(plan_path, plan)
        with self.assertRaisesRegex(ValueError, "sha256 is stale"):
            consolidator.consolidate_review_outcomes(
                review_apply_manifest_paths=manifest_paths,
                resolution_plan_path=plan_path,
                output_dir=self.root / "stale-output",
            )

    def test_rejects_duplicate_base_fact_ids_across_distinct_manifest_paths(self):
        original = Path(self.chain_a["apply"]["manifest_path"])
        duplicate = self.root / "copied-review-apply-manifest.json"
        shutil.copyfile(original, duplicate)
        paths = [original, duplicate, Path(self.chain_b["apply"]["manifest_path"])]
        plan_path, manifest_paths = self.write_plan(
            self.valid_resolutions(), paths=paths, name="duplicate-plan.json"
        )
        with self.assertRaisesRegex(ValueError, "Duplicate base_fact_id"):
            consolidator.consolidate_review_outcomes(
                review_apply_manifest_paths=manifest_paths,
                resolution_plan_path=plan_path,
                output_dir=self.root / "duplicate-output",
            )

    def test_resolution_requires_independent_reviewer_and_defer_cannot_remain(self):
        same_reviewer = self.valid_resolutions()
        same_reviewer[0]["reviewer_provenance"]["reviewer_id"] = (
            "original-reviewer-a"
        )
        plan_path, manifest_paths = self.write_plan(
            same_reviewer, name="same-reviewer-plan.json"
        )
        with self.assertRaisesRegex(ValueError, "must differ from original reviewer"):
            consolidator.consolidate_review_outcomes(
                review_apply_manifest_paths=manifest_paths,
                resolution_plan_path=plan_path,
                output_dir=self.root / "same-reviewer-output",
            )

        invalid_defer = self.valid_resolutions()
        invalid_defer[1]["disposition"] = "retain_revise"
        plan_path, manifest_paths = self.write_plan(
            invalid_defer, name="invalid-defer-plan.json"
        )
        with self.assertRaisesRegex(ValueError, "Deferred review cannot remain revise"):
            consolidator.consolidate_review_outcomes(
                review_apply_manifest_paths=manifest_paths,
                resolution_plan_path=plan_path,
                output_dir=self.root / "invalid-defer-output",
            )

    def test_tampered_source_apply_chain_fails_deterministic_validation(self):
        staging_path = Path(self.chain_a["apply"]["staging_path"])
        rows = read_jsonl(staging_path)
        rows[0]["notes"] = "tampered"
        write_jsonl(staging_path, rows)
        plan_path, manifest_paths = self.write_plan(self.valid_resolutions())
        with self.assertRaisesRegex(ValueError, "sha256 is stale"):
            consolidator.consolidate_review_outcomes(
                review_apply_manifest_paths=manifest_paths,
                resolution_plan_path=plan_path,
                output_dir=self.root / "tampered-output",
            )


if __name__ == "__main__":
    unittest.main()
