import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "materialize_static_g0a_semantic_closure.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_static_g0a_semantic_closure", SCRIPT_PATH
)
closure = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = closure
SPEC.loader.exec_module(closure)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class StaticG0ASemanticClosureMaterializerTests(unittest.TestCase):
    def row(self, base_fact_id, answer, split_group_id):
        return {
            "base_fact_id": base_fact_id,
            "candidate_id": f"source_{base_fact_id}",
            "probe_relation_id": "work_to_release_date",
            "subject_en": f"subject {base_fact_id}",
            "subject_zh": None,
            "prompt_en": f"prompt {base_fact_id}",
            "prompt_zh": f"问题 {base_fact_id}",
            "canonical_fact_en": f"fact {base_fact_id} {answer}",
            "canonical_fact_zh": None,
            "answer_en": answer,
            "answer_zh": answer,
            "answer_aliases_en": [answer],
            "answer_aliases_zh": [answer],
            "split_group_id": split_group_id,
            "split_assignment": "development",
        }

    def staging(self, base_fact_id, digit):
        return {
            "base_fact_id": base_fact_id,
            "source_record_sha256": digit * 64,
            "effective_source_record_sha256": str(int(digit) + 3) * 64,
        }

    def active_edge(self, left, right):
        return {
            "schema_version": closure.SOURCE_EDGE_SCHEMA,
            "edge_id": "sem_old_advisory",
            "left_base_fact_id": left,
            "right_base_fact_id": right,
            "decision": "same_leakage_component",
            "rationale": "historical advisory only",
            "rebind_disposition": closure.ACTIVE_EDGE_DISPOSITION,
            "resolution_status": closure.ACTIVE_EDGE_STATUS,
            "source_candidate_record_sha256": "7" * 64,
            "source_decision_record_sha256": "8" * 64,
            "reviewer_type": "codex_proxy",
            "human_gold": False,
        }

    def test_candidate_union_deduplicates_and_old_verdict_is_advisory_only(self):
        resolved = {
            "pbf_a": self.row("pbf_a", "2001", "group_a"),
            "pbf_b": self.row("pbf_b", "2002", "group_b"),
            "pbf_c": self.row("pbf_c", "2003", "group_c"),
        }
        staging = {
            "pbf_a": self.staging("pbf_a", "1"),
            "pbf_b": self.staging("pbf_b", "2"),
            "pbf_c": self.staging("pbf_c", "3"),
        }
        mechanical = {
            ("pbf_a", "pbf_b"): {"answer_alias_exact"},
            ("pbf_b", "pbf_c"): {"prompt_answer_exact"},
        }
        with mock.patch.object(
            closure.g0a.formal_admission,
            "_mechanical_cross_group_risk_reasons",
            return_value=mechanical,
        ):
            candidates, counts = closure._build_candidates(
                staging_by_id=staging,
                resolved_by_id=resolved,
                active_edges=[self.active_edge("pbf_a", "pbf_b")],
            )

        self.assertEqual(len(candidates), 2)
        by_pair = {
            (row["left_base_fact_id"], row["right_base_fact_id"]): row
            for row in candidates
        }
        overlap = by_pair[("pbf_a", "pbf_b")]
        self.assertEqual(
            overlap["generation_reasons"],
            ["v3_active_positive_edge", "resolved_mechanical_cross_group_risk"],
        )
        advisory = overlap["v3_advisory_evidence"][0]
        self.assertEqual(
            advisory["historical_advisory_relationship"],
            "same_leakage_component",
        )
        self.assertTrue(advisory["advisory_only"])
        self.assertFalse(advisory["reused_as_current_decision"])
        self.assertNotIn("decision", overlap)
        self.assertEqual(
            overlap["left_input_record_sha256"],
            closure.g0a.sha256_value(resolved["pbf_a"]),
        )
        self.assertEqual(
            counts,
            {
                "resolved_mechanical_cross_group_risk": 2,
                "v3_active_positive_edge": 1,
            },
        )

        templates = closure._build_templates(candidates)
        self.assertTrue(all(row["terminal_status"] == "pending" for row in templates))
        self.assertTrue(all(row["decision"] == "defer" for row in templates))
        self.assertTrue(all(row["reviewer_type"] == "codex_proxy" for row in templates))
        self.assertTrue(all(row["human_gold"] is False for row in templates))
        self.assertEqual(
            [row["candidate_record_sha256"] for row in templates],
            [closure.g0a.sha256_value(row) for row in candidates],
        )

    def test_materialized_manifest_is_explicitly_pending_and_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_paths = []
            schemas = (
                closure.g0a.STAGING_MANIFEST_SCHEMA,
                closure.g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
                closure.g0a.EXPECTED_SOURCE_MANIFEST_SCHEMA,
                closure.SOURCE_EDGE_SCHEMA,
                closure.SOURCE_SPLIT_SCHEMA,
            )
            for index, schema in enumerate(schemas):
                path = root / f"input-{index}.json"
                path.write_text(json.dumps({"schema_version": schema}) + "\n")
                input_paths.append(path)
            snapshots = [
                closure._snapshot(path, schema_version=schema)
                for path, schema in zip(input_paths, schemas)
            ]
            resolved = {
                "pbf_a": self.row("pbf_a", "2001", "group_a"),
                "pbf_b": self.row("pbf_b", "2002", "group_b"),
            }
            staging = {
                "pbf_a": self.staging("pbf_a", "1"),
                "pbf_b": self.staging("pbf_b", "2"),
            }
            source_binding = {
                "path": str(root / "source.jsonl"),
                "sha256": "a" * 64,
                "record_count": 2,
                "schema_version": closure.g0a.FINAL_BUNDLE_SCHEMA,
                "base_fact_ids_sha256": "b" * 64,
            }
            staging_binding = {
                "path": str(root / "staging.jsonl"),
                "sha256": "c" * 64,
                "record_count": 2,
                "schema_version": closure.g0a.STAGING_RECORD_SCHEMA,
                "base_fact_ids_sha256": "d" * 64,
            }
            resolved_binding = {
                "path": str(root / "resolved.jsonl"),
                "sha256": "e" * 64,
                "record_count": 2,
                "schema_version": closure.g0a.FINAL_BUNDLE_SCHEMA,
                "base_fact_ids_sha256": "f" * 64,
            }
            loaded = {
                "staging_manifest": {
                    "source_bundle": source_binding,
                    "effective_staging_bundle": staging_binding,
                },
                "staging_by_id": staging,
                "resolved_manifest": {"resolved_rows": resolved_binding},
                "resolved_by_id": resolved,
                "active_edges": [self.active_edge("pbf_a", "pbf_b")],
                "retired_edges": [],
                "split_manifest": {
                    "source_semantic_scope": "declared_bounded_candidate_protocol_only",
                    "semantic_paraphrase_recall_guaranteed": False,
                },
                "legacy_reference": {
                    "path": "/private/tmp/g0a_closure_exposure/semantic_closure_manifest.json",
                    "sha256": "9" * 64,
                    "schema_version": "static-g0a-semantic-closure-manifest-v1",
                },
                "required_snapshots": snapshots,
                "transitive_snapshots": [],
            }
            output_dir = root / "semantic-closure-v2-pending"
            with mock.patch.object(
                closure, "_load_and_validate_inputs", return_value=loaded
            ), mock.patch.object(
                closure.g0a.formal_admission,
                "_mechanical_cross_group_risk_reasons",
                return_value={},
            ), mock.patch.object(
                closure.g0a.formal_admission,
                "_mechanical_cross_group_risk_pairs",
                return_value=set(),
            ):
                result = closure.materialize_semantic_closure(
                    staging_manifest_path=input_paths[0],
                    resolved_manifest_path=input_paths[1],
                    v3_finalization_manifest_path=input_paths[2],
                    v3_semantic_edges_path=input_paths[3],
                    v3_split_manifest_path=input_paths[4],
                    output_dir=output_dir,
                )

            manifest = json.loads(
                (output_dir / closure.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(
                manifest["status"],
                "pending_incomplete_candidate_generation_and_adjudication",
            )
            for field in (
                "semantic_paraphrase_closure_complete",
                "candidate_generation_complete",
                "cohort_scope_complete",
                "out_of_cohort_bridge_search_complete",
            ):
                self.assertIs(manifest[field], False)
            self.assertEqual(manifest["unresolved_candidate_count"], 1)
            self.assertFalse(manifest["behavior_authorized"])
            self.assertFalse(manifest["simulation_authorized"])
            legacy = manifest["upstream_v3"]["legacy_v1_semantic_closure_reference"]
            self.assertFalse(legacy["artifact_read_by_this_tool"])
            self.assertFalse(legacy["candidate_rows_reused"])
            self.assertFalse(legacy["verdicts_reused_as_current_decisions"])
            self.assertEqual(
                manifest["coverage"]["legacy_distinct_candidate_rows_available_in_bound_v3_edges"],
                0,
            )
            candidates = read_jsonl(output_dir / closure.CANDIDATES_NAME)
            templates = read_jsonl(output_dir / closure.ADJUDICATIONS_NAME)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(templates[0]["decision"], "defer")
            self.assertEqual(templates[0]["terminal_status"], "pending")
            self.assertEqual(
                manifest["candidates"]["sha256"],
                closure.g0a.sha256_file(output_dir / closure.CANDIDATES_NAME),
            )
            self.assertEqual(
                manifest["candidates"]["path"],
                str((output_dir / closure.CANDIDATES_NAME).resolve()),
            )

    def test_full_pool_template_binds_fresh_pair_and_stays_unresolved(self):
        pair = {
            "schema_version": closure.lexical_closure.AUDIT_PAIR_SCHEMA_VERSION,
            "pair_id": "lexpair_fresh",
            "audit_scope": "within_current_bundle",
            "left": {
                "provenance": {
                    "audit_record_id": "audit_left",
                    "input_record_sha256": "1" * 64,
                }
            },
            "right": {
                "provenance": {
                    "audit_record_id": "audit_right",
                    "input_record_sha256": "2" * 64,
                }
            },
        }
        rows = closure._build_full_pool_templates([pair])
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            set(rows[0]),
            {"pair_id", "decision", "rationale", "reviewer_id", "confidence"},
        )
        self.assertEqual(rows[0]["decision"], "defer")
        self.assertEqual(rows[0]["reviewer_id"], "")
        self.assertEqual(rows[0]["confidence"], "")

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "existing"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                closure.materialize_semantic_closure(
                    staging_manifest_path=Path("unused-a"),
                    resolved_manifest_path=Path("unused-b"),
                    v3_finalization_manifest_path=Path("unused-c"),
                    v3_semantic_edges_path=Path("unused-d"),
                    v3_split_manifest_path=Path("unused-e"),
                    output_dir=output_dir,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
