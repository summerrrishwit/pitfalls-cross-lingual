import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "materialize_public_benchmark_duplicate_adjudications.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "materialize_public_benchmark_duplicate_adjudications", SCRIPT_PATH
)
materializer = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = materializer
SCRIPT_SPEC.loader.exec_module(materializer)
closure = materializer.closure_tool


def bundle_row(
    label,
    *,
    question,
    fact,
    answer,
    aliases,
    relation="character_to_actor",
    split="development",
):
    return {
        "schema_version": closure.INPUT_BUNDLE_SCHEMA_VERSION,
        "base_fact_id": f"pbf_{label}",
        "candidate_id": f"candidate_{label}",
        "source_id": f"source_{label}",
        "source_dataset": "fixture",
        "source_subset": "unit",
        "source_question_en": question,
        "canonical_fact_en": fact,
        "canonical_fact": fact,
        "answer_en": answer,
        "answer_aliases_en": aliases,
        "probe_relation_id": relation,
        "probe_relation_candidate": relation,
        "split_assignment": split,
        "split_group_id": f"old_group_{label}",
        "split_status": "provisional_not_frozen",
        "provenance": {
            "input_path": "/immutable/source.jsonl",
            "input_record_sha256": label.rjust(64, "0"),
            "source_model": "fixture",
        },
    }


def canonical_row(label, *, question, fact, answer):
    return {
        "source_id": f"canonical_source_{label}",
        "candidate_id": f"canonical_candidate_{label}",
        "source_dataset": "canonical_fixture",
        "source_path": "/immutable/canonical.jsonl",
        "source_original_index": 0,
        "source_question": question,
        "canonical_fact": fact,
        "answer": answer,
        "source_answer": answer,
    }


def write_jsonl(path, rows):
    path.write_bytes(closure.serialize_jsonl(rows))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_source_manifest(path, full_path, rows):
    closure.write_json(
        path,
        {
            "schema_version": closure.SOURCE_UNIVERSE_MANIFEST_SCHEMA_VERSION,
            "source_artifacts": {
                "behavior_bundle": {
                    "path": str(full_path.resolve()),
                    "sha256": closure.sha256_file(full_path),
                    "byte_count": full_path.stat().st_size,
                    "record_count": len(rows),
                    "schema_version": closure.INPUT_BUNDLE_SCHEMA_VERSION,
                    "record_ids_sha256": closure.sha256_value(
                        [row["base_fact_id"] for row in rows]
                    ),
                    "record_row_hashes_sha256": closure.sha256_value(
                        [closure.sha256_value(row) for row in rows]
                    ),
                }
            },
        },
    )


class DuplicateAdjudicationMaterializerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate_dir = self._materialize_candidate_fixture()
        self.candidate_manifest = self.candidate_dir / "candidate_manifest.json"
        self.pairs = read_jsonl(
            self.candidate_dir / "closure_candidate_pairs.jsonl"
        )
        self.assertGreater(len(self.pairs), 1)

    def tearDown(self):
        self.temporary.cleanup()

    def _materialize_candidate_fixture(self, *, root=None, invalid_relation=False):
        root = self.root if root is None else Path(root)
        root.mkdir(parents=True, exist_ok=True)
        original_a = bundle_row(
            "a",
            question="Original question that must be replaced",
            fact="Original fact that must be replaced.",
            answer="Original answer",
            aliases=["Original answer"],
        )
        revised_a = bundle_row(
            "a",
            question="Who wrote the novel Dune?",
            fact="Frank Herbert wrote the novel Dune.",
            answer="Frank Herbert",
            aliases=["Frank Herbert", "F. Herbert"],
        )
        row_b = bundle_row(
            "b",
            question="WHO WROTE THE NOVEL DUNE!",
            fact="FRANK HERBERT WROTE THE NOVEL DUNE",
            answer="Frank Herbert",
            aliases=["Frank Herbert", "F. Herbert"],
        )
        row_c = bundle_row(
            "c",
            question="Who wrote the unrelated fixture?",
            fact="Another person wrote the unrelated fixture.",
            answer="Another person",
            aliases=["Another person", "F. Herbert", "Canonical Bridge"],
            split="validation",
        )
        row_d = bundle_row(
            "d",
            question="What does fixture isolation mean?",
            fact="Fixture isolation means an independent test row.",
            answer="an independent test row",
            aliases=["an independent test row"],
            relation="term_to_definition",
            split="sealed",
        )
        if invalid_relation:
            row_d.pop("probe_relation_id")
            row_d.pop("probe_relation_candidate")
        canonical = canonical_row(
            "one",
            question="Which canonical row is connected only through an alias?",
            fact="The canonical bridge belongs to the comparison fixture.",
            answer="Canonical Bridge",
        )
        full_path = root / "full.jsonl"
        cohort_path = root / "cohort.jsonl"
        canonical_path = root / "canonical.jsonl"
        source_manifest_path = root / "source-manifest.json"
        full_rows = [original_a, row_b, row_c, row_d]
        write_jsonl(full_path, full_rows)
        write_jsonl(cohort_path, [revised_a, row_b, row_d])
        write_jsonl(canonical_path, [canonical])
        write_source_manifest(source_manifest_path, full_path, full_rows)
        output = root / "candidates"
        closure.materialize_candidates(
            full_pool_path=full_path,
            source_manifest_path=source_manifest_path,
            expected_full_pool_record_count=4,
            cohort_bundle_path=cohort_path,
            comparison_canonical_path=canonical_path,
            output_dir=output,
            max_bucket_neighbors=24,
            max_examples=7,
        )
        return output

    def _reviews(self):
        rows = []
        for index, pair in enumerate(self.pairs):
            decision = (
                "same_fact"
                if closure.EXACT_DUPLICATE_MATCH_TYPES.intersection(
                    pair["match_types"]
                )
                else "distinct"
            )
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "decision": decision,
                    "rationale": f"Synthetic semantic review {index}.",
                    "reviewer_id": "codex-proxy-fixture-v1",
                    "confidence": "high" if index % 2 == 0 else "medium",
                }
            )
        return rows

    def _write_shards(self, rows=None):
        rows = list(self._reviews() if rows is None else rows)
        midpoint = max(1, len(rows) // 2)
        first = self.root / "review-part-1.jsonl"
        second = self.root / "review-part-2.jsonl"
        materializer.write_jsonl(first, rows[:midpoint])
        materializer.write_jsonl(second, rows[midpoint:])
        return [first, second]

    def _materialize(self, *, rows=None, output_name="adjudications"):
        output = self.root / output_name
        result = materializer.materialize(
            candidate_manifest_path=self.candidate_manifest,
            review_shard_paths=self._write_shards(rows),
            output_dir=output,
            reviewed_at="2026-09-14T18:30:00+08:00",
            review_method="fixture-offline-semantic-review-v1",
        )
        return output, result

    def test_materializes_resolver_ready_rows_and_bound_reports(self):
        output, result = self._materialize()
        adjudications_path = output / materializer.OUTPUT_ADJUDICATIONS_NAME
        summary_path = output / materializer.OUTPUT_SUMMARY_NAME
        manifest_path = output / materializer.OUTPUT_MANIFEST_NAME
        rows = read_jsonl(adjudications_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [row["pair_id"] for row in rows],
            [pair["pair_id"] for pair in self.pairs],
        )
        self.assertEqual(result["adjudication_count"], len(self.pairs))
        self.assertEqual(summary["counts"]["review_shard_count"], 2)
        self.assertTrue(
            summary["validation"]["candidate_manifest_deterministic_replay_passed"]
        )
        self.assertEqual(
            manifest["outputs"]["adjudications"]["sha256"],
            materializer.sha256_file(adjudications_path),
        )
        self.assertEqual(
            manifest["outputs"]["summary"]["sha256"],
            materializer.sha256_file(summary_path),
        )
        self.assertEqual(
            manifest["outputs"]["adjudications"]["path"],
            str(adjudications_path.resolve()),
        )
        for pair, row in zip(self.pairs, rows):
            self.assertEqual(
                row["schema_version"], closure.ADJUDICATION_SCHEMA_VERSION
            )
            self.assertEqual(row["candidate_row_sha256"], closure.sha256_value(pair))
            self.assertEqual(row["reviewer_type"], "codex_proxy")
            self.assertFalse(row["human_gold"])
            self.assertEqual(
                row["review_method"], "fixture-offline-semantic-review-v1"
            )
            self.assertEqual(row["reviewed_at"], "2026-09-14T18:30:00+08:00")

        resolution_output = self.root / "resolved"
        resolution = closure.resolve_candidates(
            candidate_manifest_path=self.candidate_manifest,
            adjudications_path=adjudications_path,
            output_dir=resolution_output,
        )
        self.assertTrue(Path(resolution["manifest_path"]).is_file())

    def test_duplicate_pair_across_shards_is_rejected(self):
        rows = self._reviews()
        first = self.root / "duplicate-part-1.jsonl"
        second = self.root / "duplicate-part-2.jsonl"
        materializer.write_jsonl(first, rows)
        materializer.write_jsonl(second, [rows[0]])
        output = self.root / "duplicate-output"
        with self.assertRaisesRegex(ValueError, "Duplicate review decision"):
            materializer.materialize(
                candidate_manifest_path=self.candidate_manifest,
                review_shard_paths=[first, second],
                output_dir=output,
                reviewed_at="2026-09-14T18:30:00+08:00",
            )
        self.assertFalse(output.exists())

    def test_missing_and_extra_pair_coverage_are_rejected(self):
        rows = self._reviews()
        with self.subTest("missing"):
            output = self.root / "missing-output"
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=self._write_shards(rows[:-1]),
                    output_dir=output,
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
            self.assertFalse(output.exists())
        with self.subTest("extra"):
            extra_rows = list(rows)
            extra_rows.append(
                {
                    "pair_id": "lexpair_not_in_manifest",
                    "decision": "distinct",
                    "rationale": "Synthetic extra row.",
                    "reviewer_id": "codex-proxy-fixture-v1",
                    "confidence": "high",
                }
            )
            output = self.root / "extra-output"
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=self._write_shards(extra_rows),
                    output_dir=output,
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
            self.assertFalse(output.exists())

    def test_exact_match_cannot_be_distinct(self):
        rows = self._reviews()
        exact_pair_id = next(
            pair["pair_id"]
            for pair in self.pairs
            if closure.EXACT_DUPLICATE_MATCH_TYPES.intersection(pair["match_types"])
        )
        for row in rows:
            if row["pair_id"] == exact_pair_id:
                row["decision"] = "distinct"
        output = self.root / "invalid-exact-output"
        with self.assertRaisesRegex(ValueError, "Exact duplicate candidate"):
            materializer.materialize(
                candidate_manifest_path=self.candidate_manifest,
                review_shard_paths=self._write_shards(rows),
                output_dir=output,
                reviewed_at="2026-09-14T18:30:00+08:00",
            )
        self.assertFalse(output.exists())

    def test_exclude_cohort_infers_one_endpoint_and_rejects_ambiguity(self):
        cohort_ids = {"pbf_a", "pbf_b", "pbf_d"}
        single_endpoint_pair = next(
            pair
            for pair in self.pairs
            if len(
                {
                    pair[side]["provenance"].get("base_fact_id")
                    for side in ("left", "right")
                }
                & cohort_ids
            )
            == 1
        )
        rows = self._reviews()
        for row in rows:
            if row["pair_id"] == single_endpoint_pair["pair_id"]:
                row["decision"] = "exclude_cohort"
        output, _ = self._materialize(rows=rows, output_name="single-exclusion")
        decision = next(
            row
            for row in read_jsonl(
                output / materializer.OUTPUT_ADJUDICATIONS_NAME
            )
            if row["pair_id"] == single_endpoint_pair["pair_id"]
        )
        expected = sorted(
            {
                single_endpoint_pair[side]["provenance"]["base_fact_id"]
                for side in ("left", "right")
            }
            & cohort_ids
        )
        self.assertEqual(decision["excluded_cohort_base_fact_ids"], expected)

        ambiguous_pair = next(
            pair
            for pair in self.pairs
            if len(
                {
                    pair[side]["provenance"].get("base_fact_id")
                    for side in ("left", "right")
                }
                & cohort_ids
            )
            == 2
        )
        ambiguous_rows = self._reviews()
        for row in ambiguous_rows:
            if row["pair_id"] == ambiguous_pair["pair_id"]:
                row["decision"] = "exclude_cohort"
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            materializer.materialize(
                candidate_manifest_path=self.candidate_manifest,
                review_shard_paths=self._write_shards(ambiguous_rows),
                output_dir=self.root / "ambiguous-exclusion",
                reviewed_at="2026-09-14T18:30:00+08:00",
            )

    def test_stale_candidate_manifest_binding_fails_deterministic_replay(self):
        pair_path = self.candidate_dir / "closure_candidate_pairs.jsonl"
        pair_path.write_bytes(pair_path.read_bytes() + b"\n")
        output = self.root / "stale-output"
        with self.assertRaisesRegex(ValueError, "SHA-256 is stale"):
            materializer.materialize(
                candidate_manifest_path=self.candidate_manifest,
                review_shard_paths=self._write_shards(),
                output_dir=output,
                reviewed_at="2026-09-14T18:30:00+08:00",
            )
        self.assertFalse(output.exists())

    def test_transitive_candidate_binding_change_before_publish_is_rejected(self):
        shards = self._write_shards()
        output = self.root / "transitive-race-output"
        pair_path = self.candidate_dir / "closure_candidate_pairs.jsonl"
        original_write_json = materializer.write_json

        def write_and_mutate(path, value):
            original_write_json(path, value)
            if Path(path).name == materializer.OUTPUT_MANIFEST_NAME:
                pair_path.write_bytes(pair_path.read_bytes() + b"\n")

        with mock.patch.object(materializer, "write_json", side_effect=write_and_mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=shards,
                    output_dir=output,
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        self.assertFalse(output.exists())

    def test_complete_resolver_preflight_rejects_invalid_cohort_contract(self):
        invalid_root = self.root / "invalid-relation-fixture"
        candidate_dir = self._materialize_candidate_fixture(
            root=invalid_root, invalid_relation=True
        )
        pairs = read_jsonl(candidate_dir / "closure_candidate_pairs.jsonl")
        reviews = [
            {
                "pair_id": pair["pair_id"],
                "decision": (
                    "same_fact"
                    if closure.EXACT_DUPLICATE_MATCH_TYPES.intersection(
                        pair["match_types"]
                    )
                    else "distinct"
                ),
                "rationale": "Synthetic resolver-preflight review.",
                "reviewer_id": "codex-proxy-fixture-v1",
                "confidence": "high",
            }
            for pair in pairs
        ]
        shard = invalid_root / "review.jsonl"
        materializer.write_jsonl(shard, reviews)
        output = invalid_root / "adjudications"
        with self.assertRaisesRegex(ValueError, "lacks probe_relation_id/candidate"):
            materializer.materialize(
                candidate_manifest_path=candidate_dir / "candidate_manifest.json",
                review_shard_paths=[shard],
                output_dir=output,
                reviewed_at="2026-09-14T18:30:00+08:00",
            )
        self.assertFalse(output.exists())

    def test_failure_after_staging_leaves_no_partial_output(self):
        shards = self._write_shards()
        output = self.root / "atomic-output"
        with mock.patch.object(
            materializer, "write_json", side_effect=OSError("synthetic failure")
        ):
            with self.assertRaisesRegex(OSError, "synthetic failure"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=shards,
                    output_dir=output,
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".atomic-output.*.tmp")), [])

    def test_atomic_publish_does_not_replace_concurrent_empty_directory(self):
        shards = self._write_shards()
        output = self.root / "publish-race-output"
        original_write_json = materializer.write_json

        def write_and_create_target(path, value):
            original_write_json(path, value)
            if Path(path).name == materializer.OUTPUT_MANIFEST_NAME:
                output.mkdir()

        with mock.patch.object(
            materializer, "write_json", side_effect=write_and_create_target
        ):
            with self.assertRaises(FileExistsError):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=shards,
                    output_dir=output,
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        self.assertTrue(output.is_dir())
        self.assertEqual(list(output.iterdir()), [])
        self.assertEqual(list(self.root.glob(".publish-race-output.*.tmp")), [])

    def test_shard_shape_decision_confidence_and_timestamp_are_validated(self):
        with self.subTest("invalid decision"):
            rows = self._reviews()
            rows[0]["decision"] = "maybe"
            with self.assertRaisesRegex(ValueError, "Invalid duplicate-closure decision"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=self._write_shards(rows),
                    output_dir=self.root / "invalid-decision-output",
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        with self.subTest("unhashable decision"):
            rows = self._reviews()
            rows[0]["decision"] = []
            with self.assertRaisesRegex(ValueError, "Invalid duplicate-closure decision"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=self._write_shards(rows),
                    output_dir=self.root / "unhashable-decision-output",
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        with self.subTest("invalid confidence"):
            rows = self._reviews()
            rows[0]["confidence"] = "certain"
            with self.assertRaisesRegex(ValueError, "Invalid confidence"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=self._write_shards(rows),
                    output_dir=self.root / "invalid-confidence-output",
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        with self.subTest("unhashable confidence"):
            rows = self._reviews()
            rows[0]["confidence"] = {}
            with self.assertRaisesRegex(ValueError, "Invalid confidence"):
                materializer.materialize(
                    candidate_manifest_path=self.candidate_manifest,
                    review_shard_paths=self._write_shards(rows),
                    output_dir=self.root / "unhashable-confidence-output",
                    reviewed_at="2026-09-14T18:30:00+08:00",
                )
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            materializer.materialize(
                candidate_manifest_path=self.candidate_manifest,
                review_shard_paths=self._write_shards(),
                output_dir=self.root / "invalid-time-output",
                reviewed_at="2026-09-14T18:30:00",
            )


if __name__ == "__main__":
    unittest.main()
