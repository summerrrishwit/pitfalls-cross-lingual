import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "review_public_benchmark_bundle.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "review_public_benchmark_bundle", SCRIPT_PATH
)
review_tool = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(review_tool)


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


def triple(base_fact_id, candidate_id, input_hash, *, answer="Answer A"):
    return {
        "schema_version": "provisional-factual-triple-v1",
        "base_fact_id": base_fact_id,
        "candidate_id": candidate_id,
        "subject": "Subject A",
        "relation_raw": "example relation",
        "answer": answer,
        "answer_aliases_en": [answer],
        "canonical_fact": f"Subject A has {answer}.",
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "provenance": {"input_record_sha256": input_hash},
    }


def cluster(base_fact_id, candidate_ids, input_hashes, *, answer="Answer A"):
    return {
        "schema_version": "provisional-base-fact-cluster-v1",
        "base_fact_id": base_fact_id,
        "representative_candidate_id": candidate_ids[0],
        "member_count": len(candidate_ids),
        "members": [
            {
                "candidate_id": candidate_id,
                "input_record_sha256": input_hash,
                "source_id": candidate_id,
                "source_dataset": "fixture",
            }
            for candidate_id, input_hash in zip(candidate_ids, input_hashes)
        ],
        "subject": "Subject A",
        "relation_raw": "example relation",
        "answer": answer,
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
    }


def queue(base_fact_id, candidate_ids, *, answer="Answer A"):
    return {
        "schema_version": "provisional-semantic-review-v1",
        "base_fact_id": base_fact_id,
        "member_candidate_ids": candidate_ids,
        "subject": "Subject A",
        "relation_raw": "example relation",
        "answer": answer,
        "canonical_fact": f"Subject A has {answer}.",
        "canonical_status": "pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
        "review_decision": None,
    }


def behavior(
    base_fact_id,
    candidate_id,
    *,
    answer="Answer A",
    aliases=None,
    distractors=None,
):
    return {
        "schema_version": "factual-perturbation-input-bundle-v1",
        "base_fact_id": base_fact_id,
        "base_id": base_fact_id,
        "candidate_id": candidate_id,
        "subject_en": "Subject A",
        "relation_raw": "example relation",
        "answer_en": answer,
        "answer_aliases_en": list(aliases or [answer, f"{answer} alias"]),
        "canonical_fact_en": f"Subject A has {answer}.",
        "distractor_candidates": list(
            distractors
            if distractors is not None
            else [
                {
                    "distractor_id": f"dist_{base_fact_id}",
                    "text_en": "Wrong answer",
                    "answer_en": "Wrong answer",
                    "source_choice_index": 1,
                    "verification_status": "pending_review",
                    "verified": False,
                }
            ]
        ),
        "canonical_status": "pending_review",
        "bundle_status": "draft_pending_review",
        "evidence_tier": "provisional_single_model",
        "human_gold": False,
    }


class PublicBenchmarkReviewToolTests(unittest.TestCase):
    def test_explicit_ids_jsonl_reads_base_fact_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universe.jsonl"
            write_jsonl(
                path,
                [
                    {"base_fact_id": "pbf_fact_a", "other": 1},
                    {"source_base_fact_id": "pbf_fact_b", "other": 2},
                ],
            )

            self.assertEqual(
                review_tool._read_explicit_ids(path),
                ["pbf_fact_a", "pbf_fact_b"],
            )

    def test_explicit_ids_jsonl_rejects_missing_base_fact_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universe.jsonl"
            write_jsonl(path, [{"id": "pbf_fact_a"}])

            with self.assertRaisesRegex(ValueError, "lacks base_fact_id"):
                review_tool._read_explicit_ids(path)

    def test_review_contract_versions_are_v2(self):
        versions = (
            review_tool.TOOL_VERSION,
            review_tool.SCOPE_SCHEMA_VERSION,
            review_tool.SCOPE_SUMMARY_SCHEMA_VERSION,
            review_tool.EXPORT_ITEM_SCHEMA_VERSION,
            review_tool.DECISION_SCHEMA_VERSION,
            review_tool.EXPORT_MANIFEST_SCHEMA_VERSION,
            review_tool.EXPORT_SUMMARY_SCHEMA_VERSION,
            review_tool.STAGING_SCHEMA_VERSION,
            review_tool.APPLY_MANIFEST_SCHEMA_VERSION,
            review_tool.APPLY_SUMMARY_SCHEMA_VERSION,
        )
        for version in versions:
            self.assertTrue(version.endswith("-v2"), version)

    def test_jsonl_write_is_atomic_on_interrupted_record_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "artifact.jsonl"
            review_tool.write_jsonl(target, [{"state": "before"}])
            original = target.read_bytes()

            def interrupted_records():
                yield {"state": "partial"}
                raise RuntimeError("fixture interruption")

            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                review_tool.write_jsonl(target, interrupted_records())
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(list(root.glob(".artifact.jsonl.*.tmp")))

    def make_source(self, root):
        source = root / "source"
        fact_a = "pbf_fact_a"
        fact_b = "pbf_fact_b"
        triples = [
            triple(fact_a, "candidate_a1", "1" * 64),
            triple(fact_a, "candidate_a2", "2" * 64),
            triple(fact_b, "candidate_b1", "3" * 64, answer="Answer B"),
        ]
        clusters = [
            cluster(fact_a, ["candidate_a1", "candidate_a2"], ["1" * 64, "2" * 64]),
            cluster(fact_b, ["candidate_b1"], ["3" * 64], answer="Answer B"),
        ]
        queues = [
            queue(fact_a, ["candidate_a1", "candidate_a2"]),
            queue(fact_b, ["candidate_b1"], answer="Answer B"),
        ]
        bundles = [
            behavior(fact_a, "candidate_a1"),
            behavior(fact_b, "candidate_b1", answer="Answer B", distractors=[]),
        ]
        write_jsonl(source / "provisional_triples.jsonl", triples)
        write_jsonl(source / "base_fact_clusters.jsonl", clusters)
        write_jsonl(source / "review_queue.jsonl", queues)
        write_jsonl(source / "behavior_input_bundle.jsonl", bundles)
        sample = [
            {
                **queues[1],
                "schema_version": "provisional-semantic-review-sample-v1",
                "sampling": {"sample_index": 0},
            },
            {
                **queues[0],
                "schema_version": "provisional-semantic-review-sample-v1",
                "sampling": {"sample_index": 1},
            },
        ]
        write_jsonl(source / "review_sample.jsonl", sample)
        return source, fact_a, fact_b

    def artifact_paths(self, source):
        return review_tool.source_paths(source)

    def create_chain(self, root, *, ids=None, use_sample=False):
        source, fact_a, fact_b = self.make_source(root)
        scope_dir = root / "scope"
        scope = review_tool.create_scope(
            output_dir=scope_dir,
            artifact_paths=self.artifact_paths(source),
            base_fact_ids=None if use_sample else list(ids or [fact_a, fact_b]),
            review_sample_path=source / "review_sample.jsonl" if use_sample else None,
        )
        export_dir = root / "export"
        export = review_tool.export_scope(
            scope_manifest_path=Path(scope["manifest_path"]),
            output_dir=export_dir,
        )
        return source, fact_a, fact_b, scope, export

    def valid_accept(self, template):
        decision = copy.deepcopy(template)
        decision["decision"] = "accept"
        decision["review_provenance"] = {
            "reviewer_type": "codex_proxy",
            "reviewer_id": "codex-task-fixture",
            "review_method": "full_member_semantic_review_v1",
            "reviewed_at": "2026-09-13T00:00:00Z",
        }
        for field in ("member_reviews", "alias_reviews", "distractor_reviews"):
            for target in decision[field]:
                target["decision"] = "accept"
        decision["notes"] = "Fixture review only."
        return decision

    def nonaccept_decision(self, template, outcome):
        decision = copy.deepcopy(template)
        decision["decision"] = outcome
        decision["review_provenance"] = {
            "reviewer_type": "codex_proxy",
            "reviewer_id": "codex-task-fixture",
            "review_method": "triage_v1",
            "reviewed_at": "2026-09-13T00:00:00Z",
        }
        for member in decision["member_reviews"]:
            member["decision"] = "reject" if outcome == "reject" else "accept"
        return decision

    def test_scope_from_explicit_ids_binds_all_four_artifacts_and_hashes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fact_a, fact_b = self.make_source(root)
            result = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=self.artifact_paths(source),
                base_fact_ids=[fact_b, fact_a],
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["base_fact_ids"], [fact_b, fact_a])
            self.assertEqual(set(manifest["source_artifacts"]), set(review_tool.SOURCE_ARTIFACTS))
            for binding in manifest["source_artifacts"].values():
                self.assertEqual(binding["sha256"], review_tool.sha256_file(Path(binding["path"])))
            self.assertEqual(
                result["manifest_sha256"], review_tool.sha256_file(Path(result["manifest_path"]))
            )
            self.assertEqual(
                result["summary_sha256"], review_tool.sha256_file(Path(result["summary_path"]))
            )
            self.assertFalse(list((root / "scope").glob("*.tmp")))

    def test_scope_from_review_sample_preserves_sample_order_and_binds_sample_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fact_a, fact_b = self.make_source(root)
            result = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=self.artifact_paths(source),
                review_sample_path=source / "review_sample.jsonl",
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["base_fact_ids"], [fact_b, fact_a])
            self.assertEqual(manifest["selection_source"]["mode"], "review_sample")
            self.assertEqual(
                manifest["selection_source"]["sha256"],
                review_tool.sha256_file(source / "review_sample.jsonl"),
            )
            sample_rows = read_jsonl(source / "review_sample.jsonl")
            queue_rows = {
                row["base_fact_id"]: row for row in read_jsonl(source / "review_queue.jsonl")
            }
            row_bindings = manifest["selection_source"]["row_bindings"]
            self.assertEqual(
                [binding["base_fact_id"] for binding in row_bindings],
                [fact_b, fact_a],
            )
            for index, (sample_row, binding) in enumerate(zip(sample_rows, row_bindings)):
                self.assertEqual(binding["sample_index"], index)
                self.assertEqual(
                    binding["sample_row_sha256"], review_tool.sha256_value(sample_row)
                )
                self.assertEqual(
                    binding["review_queue_row_sha256"],
                    review_tool.sha256_value(queue_rows[binding["base_fact_id"]]),
                )
            self.assertEqual(
                manifest["selection_source"]["row_bindings_sha256"],
                review_tool.sha256_value(row_bindings),
            )

    def test_scope_rejects_review_sample_rows_mismatched_to_review_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fact_a, fact_b = self.make_source(root)
            sample_path = source / "review_sample.jsonl"
            sample = read_jsonl(sample_path)
            sample[0]["base_fact_id"] = fact_a
            sample[1]["base_fact_id"] = fact_b
            write_jsonl(sample_path, sample)
            with self.assertRaisesRegex(ValueError, "does not match review_queue fields"):
                review_tool.create_scope(
                    output_dir=root / "scope",
                    artifact_paths=self.artifact_paths(source),
                    review_sample_path=sample_path,
                )

    def test_export_rejects_stale_review_sample_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _ = self.make_source(root)
            scope = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=self.artifact_paths(source),
                review_sample_path=source / "review_sample.jsonl",
            )
            sample_path = source / "review_sample.jsonl"
            sample_path.write_bytes(sample_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Stale SHA-256 binding for review_sample"):
                review_tool.export_scope(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    output_dir=root / "stale-sample-export",
                )

    def test_scope_id_binds_normalized_source_paths_not_only_content_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fact_a, _ = self.make_source(root)
            scope = review_tool.create_scope(
                output_dir=root / "scope",
                artifact_paths=self.artifact_paths(source),
                base_fact_ids=[fact_a],
            )
            manifest_path = Path(scope["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            copied_bundle = root / "identical-behavior-copy.jsonl"
            shutil.copyfile(source / "behavior_input_bundle.jsonl", copied_bundle)
            manifest["source_artifacts"]["behavior_bundle"]["path"] = str(
                copied_bundle.resolve()
            )
            review_tool.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "scope_id does not match"):
                review_tool.export_scope(
                    scope_manifest_path=manifest_path,
                    output_dir=root / "rewritten-path-export",
                )

    def test_export_rejects_stale_source_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _, scope, _ = self.create_chain(root)
            bundle_path = source / "behavior_input_bundle.jsonl"
            bundle_path.write_bytes(bundle_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Stale SHA-256 binding for behavior_bundle"):
                review_tool.export_scope(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    output_dir=root / "stale-export",
                )

    def test_export_contains_every_duplicate_member_full_triple_and_row_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, _, _, export = self.create_chain(root)
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            items_path = Path(export_manifest["artifacts"]["review_items"]["path"])
            items = {row["base_fact_id"]: row for row in read_jsonl(items_path)}
            duplicate = items[fact_a]
            self.assertEqual(duplicate["member_count"], 2)
            self.assertEqual(
                {member["candidate_id"] for member in duplicate["members"]},
                {"candidate_a1", "candidate_a2"},
            )
            for member in duplicate["members"]:
                self.assertEqual(
                    member["triple_record_sha256"],
                    review_tool.sha256_value(member["triple"]),
                )
            self.assertEqual(
                set(duplicate["source_row_bindings"]),
                {
                    "behavior_row_sha256",
                    "review_queue_row_sha256",
                    "cluster_row_sha256",
                },
            )

    def test_accept_requires_all_duplicate_members_aliases_and_distractors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            templates = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )
            valid = self.valid_accept(templates[0])
            cases = []
            missing_member = copy.deepcopy(valid)
            missing_member["member_reviews"] = missing_member["member_reviews"][:1]
            cases.append((missing_member, "missing member_reviews"))
            missing_alias = copy.deepcopy(valid)
            missing_alias["alias_reviews"] = missing_alias["alias_reviews"][:1]
            cases.append((missing_alias, "missing alias_reviews"))
            missing_distractor = copy.deepcopy(valid)
            missing_distractor["distractor_reviews"] = []
            cases.append((missing_distractor, "missing distractor_reviews"))
            for index, (decision, message) in enumerate(cases):
                decisions_path = root / f"decisions-{index}.jsonl"
                write_jsonl(decisions_path, [decision])
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        review_tool.apply_decisions(
                            scope_manifest_path=Path(scope["manifest_path"]),
                            export_manifest_path=Path(export["manifest_path"]),
                            decisions_path=decisions_path,
                            output_dir=root / f"apply-{index}",
                        )

    def test_nonaccept_requires_every_member_disposition_and_tracks_partial_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]

            missing_members = self.nonaccept_decision(template, "reject")
            missing_members["member_reviews"] = []
            missing_path = root / "reject-missing-members.jsonl"
            write_jsonl(missing_path, [missing_members])
            with self.assertRaisesRegex(ValueError, "missing member_reviews"):
                review_tool.apply_decisions(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    export_manifest_path=Path(export["manifest_path"]),
                    decisions_path=missing_path,
                    output_dir=root / "reject-missing-members",
                )

            explicit = self.nonaccept_decision(template, "reject")
            explicit["member_reviews"][0]["decision"] = "accept"
            explicit_path = root / "reject-explicit-members.jsonl"
            write_jsonl(explicit_path, [explicit])
            result = review_tool.apply_decisions(
                scope_manifest_path=Path(scope["manifest_path"]),
                export_manifest_path=Path(export["manifest_path"]),
                decisions_path=explicit_path,
                output_dir=root / "reject-explicit-members",
            )
            staging = read_jsonl(Path(result["staging_path"]))[0]
            self.assertEqual(staging["base_fact_id"], fact_a)
            self.assertEqual(staging["review_outcome"], "reject")
            self.assertEqual(len(staging["member_reviews"]), 2)
            self.assertTrue(staging["review_completion"]["member_review_complete"])
            self.assertFalse(staging["review_completion"]["alias_review_complete"])
            self.assertFalse(staging["review_completion"]["distractor_review_complete"])
            self.assertFalse(
                staging["review_completion"]["codex_proxy_scope_review_complete"]
            )

    def test_supplied_distractor_metadata_must_exactly_match_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]
            decisions = [
                self.valid_accept(template),
                self.nonaccept_decision(template, "defer"),
            ]
            for index, decision in enumerate(decisions):
                decision["distractor_reviews"][0]["source_choice_index"] = 999
                decisions_path = root / f"wrong-distractor-metadata-{index}.jsonl"
                write_jsonl(decisions_path, [decision])
                with self.subTest(decision=decision["decision"]):
                    with self.assertRaisesRegex(
                        ValueError, "source_choice_index does not match"
                    ):
                        review_tool.apply_decisions(
                            scope_manifest_path=Path(scope["manifest_path"]),
                            export_manifest_path=Path(export["manifest_path"]),
                            decisions_path=decisions_path,
                            output_dir=root / f"wrong-distractor-metadata-{index}",
                        )

    def test_valid_accept_is_staged_without_any_automatic_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]
            decisions_path = root / "decisions.jsonl"
            write_jsonl(decisions_path, [self.valid_accept(template)])
            result = review_tool.apply_decisions(
                scope_manifest_path=Path(scope["manifest_path"]),
                export_manifest_path=Path(export["manifest_path"]),
                decisions_path=decisions_path,
                output_dir=root / "apply",
            )
            staging = read_jsonl(Path(result["staging_path"]))
            self.assertEqual(len(staging), 1)
            self.assertEqual(staging[0]["base_fact_id"], fact_a)
            self.assertEqual(staging[0]["review_outcome"], "accept")
            self.assertEqual(staging[0]["canonical_status"], "pending_review")
            self.assertEqual(staging[0]["bundle_status"], "draft_pending_review")
            self.assertEqual(staging[0]["evidence_tier"], "provisional_single_model")
            self.assertFalse(staging[0]["human_gold"])
            self.assertFalse(staging[0]["automatic_promotion_performed"])
            self.assertFalse(staging[0]["canonical_freeze_performed"])
            self.assertEqual(
                staging[0]["review_evidence_bindings"]["scope_manifest_sha256"],
                review_tool.sha256_file(Path(scope["manifest_path"])),
            )
            self.assertEqual(
                staging[0]["review_evidence_bindings"]["export_manifest_sha256"],
                review_tool.sha256_file(Path(export["manifest_path"])),
            )
            self.assertEqual(
                staging[0]["review_evidence_bindings"]["decision_record_sha256"],
                review_tool.sha256_value(self.valid_accept(template)),
            )
            self.assertNotIn(
                '"canonical_status":"frozen"',
                Path(result["staging_path"]).read_text(),
            )
            self.assertEqual(result["outcome_counts"], {"accept": 1})

    def test_partial_apply_keeps_missing_as_missing_not_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, fact_b, scope, export = self.create_chain(root)
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            templates = {
                row["base_fact_id"]: row
                for row in read_jsonl(
                    Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
                )
            }
            decisions_path = root / "partial.jsonl"
            write_jsonl(decisions_path, [self.nonaccept_decision(templates[fact_a], "defer")])
            with self.assertRaisesRegex(ValueError, "Missing decisions"):
                review_tool.apply_decisions(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    export_manifest_path=Path(export["manifest_path"]),
                    decisions_path=decisions_path,
                    output_dir=root / "not-partial",
                )
            result = review_tool.apply_decisions(
                scope_manifest_path=Path(scope["manifest_path"]),
                export_manifest_path=Path(export["manifest_path"]),
                decisions_path=decisions_path,
                output_dir=root / "partial-apply",
                allow_partial=True,
            )
            staging = {row["base_fact_id"]: row for row in read_jsonl(Path(result["staging_path"]))}
            self.assertEqual(staging[fact_a]["review_outcome"], "defer")
            self.assertIsNone(staging[fact_b]["review_outcome"])
            self.assertEqual(staging[fact_b]["staging_status"], "decision_missing")
            self.assertEqual(result["outcome_counts"], {"defer": 1, "missing": 1})
            self.assertNotIn("reject", result["outcome_counts"])

    def test_duplicate_unknown_and_unknown_outcome_decisions_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]
            valid = self.nonaccept_decision(template, "reject")
            cases = [
                ([valid, valid], "Duplicate base_fact_id"),
                ([{**valid, "base_fact_id": "pbf_unknown"}], "Unknown base_fact_id"),
                ([{**valid, "decision": "approve"}], "Unknown decision"),
            ]
            for index, (decisions, message) in enumerate(cases):
                decisions_path = root / f"invalid-{index}.jsonl"
                write_jsonl(decisions_path, decisions)
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        review_tool.apply_decisions(
                            scope_manifest_path=Path(scope["manifest_path"]),
                            export_manifest_path=Path(export["manifest_path"]),
                            decisions_path=decisions_path,
                            output_dir=root / f"invalid-apply-{index}",
                        )

    def test_revise_only_requests_revision_and_cannot_apply_rewritten_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, fact_a, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]
            revision = self.nonaccept_decision(template, "revise")
            revision["notes"] = "The relation wording needs a separate revision workflow."
            decisions_path = root / "revision.jsonl"
            write_jsonl(decisions_path, [revision])
            result = review_tool.apply_decisions(
                scope_manifest_path=Path(scope["manifest_path"]),
                export_manifest_path=Path(export["manifest_path"]),
                decisions_path=decisions_path,
                output_dir=root / "revision-apply",
            )
            staging = read_jsonl(Path(result["staging_path"]))[0]
            self.assertEqual(staging["base_fact_id"], fact_a)
            self.assertEqual(staging["review_outcome"], "revise")
            self.assertEqual(staging["staging_status"], "revision_requested_review_staging")
            self.assertEqual(staging["canonical_status"], "pending_review")
            self.assertEqual(staging["answer_en"], "Answer A")
            self.assertIn("revision_not_applied", staging["promotion_blockers"])
            self.assertIn("recluster_required_before_revision", staging["promotion_blockers"])
            self.assertFalse(staging["automatic_promotion_performed"])

            rewritten = copy.deepcopy(revision)
            rewritten["revised_answer_en"] = "Changed answer"
            rewritten_path = root / "rewritten-revision.jsonl"
            write_jsonl(rewritten_path, [rewritten])
            with self.assertRaisesRegex(ValueError, "revisions cannot be applied in v2"):
                review_tool.apply_decisions(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    export_manifest_path=Path(export["manifest_path"]),
                    decisions_path=rewritten_path,
                    output_dir=root / "rewritten-revision-apply",
                )

    def test_stale_review_item_hash_and_non_codex_or_human_gold_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]
            valid = self.valid_accept(template)
            stale = {**valid, "review_item_sha256": "0" * 64}
            human = {**valid, "human_gold": True}
            non_codex = copy.deepcopy(valid)
            non_codex["review_provenance"]["reviewer_type"] = "human"
            cases = [
                (stale, "stale review_item_sha256"),
                (human, "human_gold=false"),
                (non_codex, "reviewer_type must be codex_proxy"),
            ]
            for index, (decision, message) in enumerate(cases):
                decisions_path = root / f"invalid-provenance-{index}.jsonl"
                write_jsonl(decisions_path, [decision])
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        review_tool.apply_decisions(
                            scope_manifest_path=Path(scope["manifest_path"]),
                            export_manifest_path=Path(export["manifest_path"]),
                            decisions_path=decisions_path,
                            output_dir=root / f"invalid-provenance-apply-{index}",
                        )

    def test_apply_rederives_and_validates_bound_decision_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for case_name in ("empty", "content", "schema", "record_count"):
                with self.subTest(case=case_name):
                    case_root = root / case_name
                    _, _, _, scope, export = self.create_chain(
                        case_root, ids=["pbf_fact_a"]
                    )
                    export_path = Path(export["manifest_path"])
                    export_manifest = json.loads(export_path.read_text(encoding="utf-8"))
                    template_binding = export_manifest["artifacts"][
                        "review_decisions_template"
                    ]
                    template_path = Path(template_binding["path"])
                    template = read_jsonl(template_path)[0]
                    decision_path = case_root / "decision.jsonl"
                    write_jsonl(decision_path, [self.valid_accept(template)])

                    if case_name == "empty":
                        template_path.write_bytes(b"")
                        template_binding["sha256"] = review_tool.sha256_file(template_path)
                        expected_error = "Input JSONL is empty"
                    elif case_name == "content":
                        tampered = copy.deepcopy(template)
                        tampered["notes"] = "tampered template"
                        write_jsonl(template_path, [tampered])
                        template_binding["sha256"] = review_tool.sha256_file(template_path)
                        expected_error = "does not match its review item"
                    elif case_name == "schema":
                        template_binding["schema_version"] = "tampered-schema"
                        expected_error = "decision template schema binding is invalid"
                    else:
                        template_binding["record_count"] = 2
                        expected_error = "decision template record_count is stale"
                    review_tool.write_json(export_path, export_manifest)

                    with self.assertRaisesRegex(ValueError, expected_error):
                        review_tool.apply_decisions(
                            scope_manifest_path=Path(scope["manifest_path"]),
                            export_manifest_path=export_path,
                            decisions_path=decision_path,
                            output_dir=case_root / "apply",
                        )

            coverage_root = root / "coverage"
            _, _, _, scope, export = self.create_chain(coverage_root)
            export_path = Path(export["manifest_path"])
            export_manifest = json.loads(export_path.read_text(encoding="utf-8"))
            template_binding = export_manifest["artifacts"]["review_decisions_template"]
            template_path = Path(template_binding["path"])
            templates = read_jsonl(template_path)
            write_jsonl(template_path, templates[:1])
            template_binding["sha256"] = review_tool.sha256_file(template_path)
            template_binding["record_count"] = 1
            review_tool.write_json(export_path, export_manifest)
            decision_path = coverage_root / "decision.jsonl"
            write_jsonl(decision_path, [])
            with self.assertRaisesRegex(ValueError, "does not exactly cover"):
                review_tool.apply_decisions(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    export_manifest_path=export_path,
                    decisions_path=decision_path,
                    output_dir=coverage_root / "apply",
                    allow_partial=True,
                )

    def test_decision_cannot_inject_promotion_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, scope, export = self.create_chain(root, ids=["pbf_fact_a"])
            export_manifest = json.loads(
                Path(export["manifest_path"]).read_text(encoding="utf-8")
            )
            template = read_jsonl(
                Path(export_manifest["artifacts"]["review_decisions_template"]["path"])
            )[0]
            decision = self.valid_accept(template)
            decision["canonical_status"] = "frozen"
            decisions_path = root / "promotion.jsonl"
            write_jsonl(decisions_path, [decision])
            with self.assertRaisesRegex(ValueError, "forbidden promotion fields"):
                review_tool.apply_decisions(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    export_manifest_path=Path(export["manifest_path"]),
                    decisions_path=decisions_path,
                    output_dir=root / "promotion-apply",
                )

            missing_with_promotion = copy.deepcopy(template)
            missing_with_promotion["canonical_status"] = "frozen"
            missing_path = root / "missing-promotion.jsonl"
            write_jsonl(missing_path, [missing_with_promotion])
            with self.assertRaisesRegex(ValueError, "does not match the bound decision template"):
                review_tool.apply_decisions(
                    scope_manifest_path=Path(scope["manifest_path"]),
                    export_manifest_path=Path(export["manifest_path"]),
                    decisions_path=missing_path,
                    output_dir=root / "missing-promotion-apply",
                    allow_partial=True,
                )


if __name__ == "__main__":
    unittest.main()
