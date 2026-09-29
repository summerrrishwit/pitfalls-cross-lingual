import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "materialize_public_benchmark_duplicate_closure.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "materialize_public_benchmark_duplicate_closure", SCRIPT_PATH
)
closure = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
sys.modules[SCRIPT_SPEC.name] = closure
SCRIPT_SPEC.loader.exec_module(closure)


def bundle_row(
    label,
    *,
    question,
    fact,
    answer,
    aliases,
    relation="character_to_actor",
    split="development",
    split_group=None,
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
        "split_group_id": split_group or f"old_group_{label}",
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


class DuplicateClosureTests(unittest.TestCase):
    def make_inputs(self, root):
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
        # Cohort order is the deterministic same-fact representative order.
        write_jsonl(cohort_path, [revised_a, row_b, row_d])
        write_jsonl(canonical_path, [canonical])
        write_source_manifest(source_manifest_path, full_path, full_rows)
        return full_path, source_manifest_path, cohort_path, canonical_path, revised_a

    def materialize(self, root):
        (
            full_path,
            source_manifest_path,
            cohort_path,
            canonical_path,
            revised_a,
        ) = self.make_inputs(root)
        output = root / "candidates"
        result = closure.materialize_candidates(
            full_pool_path=full_path,
            source_manifest_path=source_manifest_path,
            expected_full_pool_record_count=4,
            cohort_bundle_path=cohort_path,
            comparison_canonical_path=canonical_path,
            output_dir=output,
            max_bucket_neighbors=24,
            max_examples=7,
        )
        return output, result, revised_a

    def decisions_for(self, candidate_dir):
        pairs = read_jsonl(candidate_dir / "closure_candidate_pairs.jsonl")
        decisions = []
        for pair in pairs:
            decision = (
                "same_fact"
                if closure.EXACT_DUPLICATE_MATCH_TYPES.intersection(
                    pair["match_types"]
                )
                else "distinct"
            )
            decisions.append(
                {
                    "schema_version": closure.ADJUDICATION_SCHEMA_VERSION,
                    "pair_id": pair["pair_id"],
                    "candidate_row_sha256": closure.sha256_value(pair),
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
                    "decision": decision,
                    "excluded_cohort_base_fact_ids": [],
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                    "reviewer_id": "codex-fixture",
                    "review_method": "fixture-semantic-review-v1",
                    "reviewed_at": "2026-09-14T00:00:00Z",
                    "rationale": "Deterministic unit-test decision.",
                }
            )
        return pairs, decisions

    def test_candidates_overlay_full_pool_and_keep_transitive_canonical_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, result, revised_a = self.materialize(root)
            manifest = json.loads(
                (output / "candidate_manifest.json").read_text(encoding="utf-8")
            )
            reconstructed = read_jsonl(output / "reconstructed_full_pool.jsonl")
            pairs = read_jsonl(output / "closure_candidate_pairs.jsonl")

            self.assertEqual(reconstructed[0], revised_a)
            self.assertEqual(len(reconstructed), 4)
            self.assertEqual(
                manifest["full_pool_authority"]["expected_record_count"], 4
            )
            self.assertEqual(
                manifest["full_pool_authority"]["binding_locator"],
                closure.SOURCE_BEHAVIOR_BINDING_LOCATOR,
            )
            self.assertEqual(manifest["cohort"]["base_fact_count"], 3)
            self.assertGreaterEqual(manifest["closure"]["closure_pair_count"], 3)
            self.assertGreaterEqual(
                manifest["closure"][
                    "closure_out_of_cohort_current_pool_node_count"
                ],
                1,
            )
            self.assertGreaterEqual(
                manifest["closure"]["closure_comparison_canonical_node_count"],
                1,
            )
            self.assertEqual(result["closure_pair_count"], len(pairs))
            self.assertFalse(result["split_freeze_emitted"])
            self.assertFalse(result["perturbation_authorized"])
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "candidate_manifest.json",
                    "closure_candidate_pairs.jsonl",
                    "reconstructed_full_pool.jsonl",
                ],
            )

    def test_candidates_require_expected_count_and_authoritative_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                full_path,
                source_manifest_path,
                cohort_path,
                canonical_path,
                _,
            ) = self.make_inputs(root)

            wrong_count_output = root / "wrong-count"
            with self.assertRaisesRegex(ValueError, "record count"):
                closure.materialize_candidates(
                    full_pool_path=full_path,
                    source_manifest_path=source_manifest_path,
                    expected_full_pool_record_count=8969,
                    cohort_bundle_path=cohort_path,
                    comparison_canonical_path=canonical_path,
                    output_dir=wrong_count_output,
                )
            self.assertFalse(wrong_count_output.exists())

            manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            manifest["source_artifacts"]["behavior_bundle"]["sha256"] = "0" * 64
            closure.write_json(source_manifest_path, manifest)
            stale_binding_output = root / "stale-binding"
            with self.assertRaisesRegex(ValueError, "SHA-256 is stale"):
                closure.materialize_candidates(
                    full_pool_path=full_path,
                    source_manifest_path=source_manifest_path,
                    expected_full_pool_record_count=4,
                    cohort_bundle_path=cohort_path,
                    comparison_canonical_path=canonical_path,
                    output_dir=stale_binding_output,
                )
            self.assertFalse(stale_binding_output.exists())

    def test_candidate_pairs_are_independent_of_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                full_path,
                source_manifest_path,
                cohort_path,
                canonical_path,
                _,
            ) = self.make_inputs(root)
            outputs = [root / "candidate-a", root / "candidate-b"]
            for output in outputs:
                closure.materialize_candidates(
                    full_pool_path=full_path,
                    source_manifest_path=source_manifest_path,
                    expected_full_pool_record_count=4,
                    cohort_bundle_path=cohort_path,
                    comparison_canonical_path=canonical_path,
                    output_dir=output,
                )

            first_pairs = (outputs[0] / "closure_candidate_pairs.jsonl").read_bytes()
            second_pairs = (outputs[1] / "closure_candidate_pairs.jsonl").read_bytes()
            self.assertEqual(first_pairs, second_pairs)
            first_manifest = json.loads(
                (outputs[0] / "candidate_manifest.json").read_text(encoding="utf-8")
            )
            second_manifest = json.loads(
                (outputs[1] / "candidate_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                first_manifest["lexical_audit"]["full_candidate_pairs_sha256"],
                second_manifest["lexical_audit"]["full_candidate_pairs_sha256"],
            )
            self.assertEqual(
                first_manifest["lexical_audit"]["audit_summary_payload_sha256"],
                second_manifest["lexical_audit"]["audit_summary_payload_sha256"],
            )
            for pair in read_jsonl(outputs[0] / "closure_candidate_pairs.jsonl"):
                for side in ("left", "right"):
                    provenance = pair[side]["provenance"]
                    if provenance["input_role"] == "current_bundle":
                        self.assertEqual(
                            provenance["input_path"],
                            closure.RECONSTRUCTED_AUDIT_LOGICAL_PATH,
                        )

            _, first_decisions = self.decisions_for(outputs[0])
            decisions_path = root / "portable-adjudications.jsonl"
            write_jsonl(decisions_path, first_decisions)
            resolved = root / "resolved-from-second-output"
            closure.resolve_candidates(
                candidate_manifest_path=outputs[1] / "candidate_manifest.json",
                adjudications_path=decisions_path,
                output_dir=resolved,
            )
            self.assertTrue((resolved / "resolution_manifest.json").is_file())

    def test_resolve_deduplicates_by_cohort_order_and_recomputes_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir, _, _ = self.materialize(root)
            pairs, decisions = self.decisions_for(candidate_dir)
            self.assertTrue(pairs)
            decisions_path = root / "adjudications.jsonl"
            write_jsonl(decisions_path, decisions)
            output = root / "resolved"

            result = closure.resolve_candidates(
                candidate_manifest_path=candidate_dir / "candidate_manifest.json",
                adjudications_path=decisions_path,
                output_dir=output,
                split_seed="fixture-split-v1",
            )

            exclusions = read_jsonl(output / "dedup_exclusions.jsonl")
            split = json.loads(
                (output / "provisional_split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            components = read_jsonl(output / "leakage_components.jsonl")
            self.assertEqual(len(exclusions), 1)
            self.assertEqual(exclusions[0]["base_fact_id"], "pbf_b")
            self.assertEqual(exclusions[0]["representative_base_fact_id"], "pbf_a")
            self.assertEqual(exclusions[0]["disposition"], "deduplicated_same_fact")
            self.assertEqual(split["cohort_counts"]["input"], 3)
            self.assertEqual(split["cohort_counts"]["retained"], 2)
            self.assertEqual(split["cohort_counts"]["deduplicated"], 1)
            assigned = [
                item
                for component in components
                for item in component["retained_cohort_base_fact_ids"]
            ]
            self.assertEqual(sorted(assigned), ["pbf_a", "pbf_d"])
            self.assertTrue(split["exact_relation_targets_achieved"])
            self.assertEqual(split["unresolved_candidate_count"], 0)
            self.assertEqual(split["historical_exposure_status"], "not_supplied_pending")
            self.assertFalse(split["safety_contract"]["split_freeze_emitted"])
            self.assertFalse(split["safety_contract"]["perturbation_authorized"])
            self.assertEqual(result["retained_cohort_count"], 2)

    def test_input_split_group_stays_atomic_when_candidate_is_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                full_path,
                source_manifest_path,
                cohort_path,
                canonical_path,
                _,
            ) = self.make_inputs(root)
            cohort_rows = read_jsonl(cohort_path)
            cohort_by_id = {row["base_fact_id"]: row for row in cohort_rows}
            full_by_id = {
                row["base_fact_id"]: row for row in read_jsonl(full_path)
            }
            row_c = dict(full_by_id["pbf_c"])
            row_d = dict(cohort_by_id["pbf_d"])
            for row in (row_c, row_d):
                row["split_group_id"] = "old_group_shared_answer"
                row["answer_aliases_en"] = [
                    *row["answer_aliases_en"],
                    "Shared lexical-only alias",
                ]
            write_jsonl(
                cohort_path,
                [
                    cohort_by_id["pbf_a"],
                    cohort_by_id["pbf_b"],
                    row_c,
                    row_d,
                ],
            )

            candidate_dir = root / "same-input-group-candidates"
            closure.materialize_candidates(
                full_pool_path=full_path,
                source_manifest_path=source_manifest_path,
                expected_full_pool_record_count=4,
                cohort_bundle_path=cohort_path,
                comparison_canonical_path=canonical_path,
                output_dir=candidate_dir,
            )
            pairs, decisions = self.decisions_for(candidate_dir)
            pair_index = next(
                index
                for index, pair in enumerate(pairs)
                if {
                    pair[side]["provenance"].get("base_fact_id")
                    for side in ("left", "right")
                }
                == {"pbf_c", "pbf_d"}
            )
            self.assertEqual(decisions[pair_index]["decision"], "distinct")
            decisions_path = root / "same-input-group-decisions.jsonl"
            write_jsonl(decisions_path, decisions)
            resolved = root / "same-input-group-resolved"
            closure.resolve_candidates(
                candidate_manifest_path=candidate_dir / "candidate_manifest.json",
                adjudications_path=decisions_path,
                output_dir=resolved,
            )

            components = read_jsonl(resolved / "leakage_components.jsonl")
            shared_components = [
                component
                for component in components
                if {"pbf_c", "pbf_d"}
                <= set(component["retained_cohort_base_fact_ids"])
            ]
            self.assertEqual(len(shared_components), 1)
            self.assertEqual(
                shared_components[0]["supporting_input_split_group_ids"],
                ["old_group_shared_answer"],
            )
            split_manifest = json.loads(
                (resolved / "provisional_split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            contract = split_manifest["leakage_grouping_contract"]
            self.assertIn("input_cohort_split_group_id", contract["grouping_sources"])
            self.assertEqual(
                contract["input_split_group"]["union_policy"],
                "unconditional_union_before_provisional_resplit_v1",
            )
            self.assertTrue(
                split_manifest["safety_contract"][
                    "input_split_group_atomicity_enforced"
                ]
            )

    def test_missing_or_stale_adjudication_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir, _, _ = self.materialize(root)
            _, decisions = self.decisions_for(candidate_dir)
            self.assertGreater(len(decisions), 1)

            incomplete_path = root / "incomplete.jsonl"
            write_jsonl(incomplete_path, decisions[:-1])
            incomplete_output = root / "incomplete-output"
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                closure.resolve_candidates(
                    candidate_manifest_path=candidate_dir / "candidate_manifest.json",
                    adjudications_path=incomplete_path,
                    output_dir=incomplete_output,
                )
            self.assertFalse(incomplete_output.exists())

            stale = list(decisions)
            stale[0] = dict(stale[0])
            stale[0]["candidate_row_sha256"] = "0" * 64
            stale_path = root / "stale.jsonl"
            write_jsonl(stale_path, stale)
            stale_output = root / "stale-output"
            with self.assertRaisesRegex(ValueError, "Stale candidate row hash"):
                closure.resolve_candidates(
                    candidate_manifest_path=candidate_dir / "candidate_manifest.json",
                    adjudications_path=stale_path,
                    output_dir=stale_output,
                )
            self.assertFalse(stale_output.exists())

    def test_exclude_cohort_requires_an_explicit_cohort_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_dir, _, _ = self.materialize(root)
            pairs, decisions = self.decisions_for(candidate_dir)
            external_pair_index = next(
                index
                for index, pair in enumerate(pairs)
                if not {
                    pair[side]["provenance"].get("base_fact_id")
                    for side in ("left", "right")
                }
                & {"pbf_a", "pbf_b", "pbf_d"}
            )
            decisions[external_pair_index] = dict(decisions[external_pair_index])
            decisions[external_pair_index]["decision"] = "exclude_cohort"
            decisions[external_pair_index]["excluded_cohort_base_fact_ids"] = [
                "pbf_a"
            ]
            decisions_path = root / "invalid-exclusion.jsonl"
            write_jsonl(decisions_path, decisions)
            output = root / "invalid-exclusion-output"
            with self.assertRaisesRegex(ValueError, "must name a cohort endpoint"):
                closure.resolve_candidates(
                    candidate_manifest_path=candidate_dir / "candidate_manifest.json",
                    adjudications_path=decisions_path,
                    output_dir=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
