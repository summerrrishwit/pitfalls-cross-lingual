import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_full_public_benchmark_candidate_review.py"
)
SPEC = importlib.util.spec_from_file_location("full_candidate_review_runner", SCRIPT_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_bytes(
        b"".join(runner.canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def distractor_verdict(unit, **updates):
    record = {
        "candidate_id": unit.candidate_id,
        "candidate_kind": "distractor",
        "overall_decision": "accept",
        "distractor_factually_false": True,
        "answer_unique": True,
        "answer_alias_disjoint": True,
        "neutral_semantically_neutral": None,
        "neutral_unrelated": None,
        "overlapping_target_aliases": [],
        "answer_alias_disjoint_unresolved_evidence": None,
        "rationale": "Fixture alias evidence is internally consistent.",
        "confidence": "high",
    }
    record.update(updates)
    return record


def binding(path, schema, rows):
    return {
        "filename": path.name,
        "sha256": runner.sha256_file(path),
        "byte_count": path.stat().st_size,
        "record_count": len(rows),
        "schema_version": schema,
    }


def file_binding(path, schema):
    return {
        "path": str(path.resolve()),
        "sha256": runner.sha256_file(path),
        "byte_count": path.stat().st_size,
        "schema_version": schema,
    }


def full_fact(base_fact_id, relation, answer, fact, question):
    return {
        "schema_version": runner.FULL_FACT_SCHEMA,
        "base_fact_id": base_fact_id,
        "candidate_id": f"source_{base_fact_id}",
        "source_question_en": question,
        "subject_en": question.split()[1],
        "relation_raw": relation,
        "canonical_fact_en": fact,
        "answer_en": answer,
        "answer_aliases_en": [answer, f"{answer} alias"],
        "answer_type": "person" if relation == "author" else "other",
        "relation_partition_id": relation,
        "split_assignment": "development",
        "split_status": "provisional_not_frozen",
        "leakage_component_id": f"component_{base_fact_id}",
        "human_gold": False,
        "hf_model_execution_status": "not_run",
        "hf_tokenizer_execution_status": "not_run",
    }


class FakeRouter:
    def __init__(
        self,
        *,
        response_model=runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
        fail_multi=False,
        fail_ids=(),
        alias_contradiction_ids=(),
        reject_ids=(),
        defer_ids=(),
    ):
        self.response_model = response_model
        self.fail_multi = fail_multi
        self.fail_ids = set(fail_ids)
        self.alias_contradiction_ids = set(alias_contradiction_ids)
        self.reject_ids = set(reject_ids)
        self.defer_ids = set(defer_ids)
        self.calls = []

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = runner._prompt_payload(prompt)
        candidates = payload["candidates"]
        ids = [row["candidate_id"] for row in candidates]
        self.calls.append(ids)
        if (self.fail_multi and len(ids) > 1) or any(
            candidate_id in self.fail_ids for candidate_id in ids
        ):
            return SimpleNamespace(
                terminal_status="failed",
                parsed_response=None,
                raw_response=None,
                response_model=self.response_model,
                usage={},
                latency_ms=1,
                attempt_count=1,
                attempts=[{"attempt": 1, "status": "failed", "error_type": "FixtureFailure"}],
            )
        records = []
        for candidate in candidates:
            distractor = candidate["candidate_kind"] == "distractor"
            alias_contradiction = candidate["candidate_id"] in self.alias_contradiction_ids
            reject = candidate["candidate_id"] in self.reject_ids
            defer = candidate["candidate_id"] in self.defer_ids
            records.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_kind": candidate["candidate_kind"],
                    "overall_decision": (
                        "defer" if defer else ("reject" if reject else "accept")
                    ),
                    "distractor_factually_false": True if distractor else None,
                    "answer_unique": True if distractor else None,
                    "answer_alias_disjoint": (
                        None
                        if defer and distractor
                        else (
                            False
                            if reject and distractor
                            else (True if distractor else None)
                        )
                    ),
                    "neutral_semantically_neutral": (
                        False
                        if reject and not distractor
                        else (None if distractor else True)
                    ),
                    "neutral_unrelated": (
                        None if distractor or defer else True
                    ),
                    "overlapping_target_aliases": (
                        [candidate["target_fact"]["answer_aliases_en"][0]]
                        if alias_contradiction or (reject and distractor)
                        else ([] if distractor else None)
                    ),
                    "answer_alias_disjoint_unresolved_evidence": (
                        "Fixture evidence is intentionally unresolved."
                        if defer and distractor
                        else None
                    ),
                    "rationale": "Fixture evidence supports every required field.",
                    "confidence": "high",
                }
            )
        response = {
            "schema_version": runner.BATCH_RESPONSE_SCHEMA,
            "records": records,
        }
        validator(response)
        raw = json.dumps(response, ensure_ascii=False, sort_keys=True)
        return SimpleNamespace(
            terminal_status="completed",
            parsed_response=response,
            raw_response=raw,
            response_model=self.response_model,
            usage={"input_tokens": 11, "output_tokens": 7, "secret": 99},
            latency_ms=2,
            attempt_count=1,
            attempts=[{"attempt": 1, "status": "completed"}],
        )


class NoCallRouter:
    def request_json(self, *args, **kwargs):
        raise AssertionError("No model request was expected")


class CandidateReviewRunnerTests(unittest.TestCase):
    def test_postreview_manifest_schema_matches_v2_materializer_contract(self):
        self.assertEqual(
            runner.POSTREVIEW_MANIFEST_SCHEMA,
            "public-benchmark-full-postreview-rebuild-manifest-v2",
        )

    def make_fixture(self, root, *, pre_hf=False):
        facts = [
            full_fact("pbf_a", "author", "Author A", "Author A wrote Book A.", "Who wrote Book A?"),
            full_fact("pbf_b", "author", "Author B", "Author B wrote Book B.", "Who wrote Book B?"),
            full_fact("pbf_c", "author", "Author C", "Author C wrote Book C.", "Who wrote Book C?"),
            full_fact("pbf_d", "location", "City D", "Museum D is in City D.", "Where is Museum D?"),
            full_fact("pbf_e", "date", "1901", "Event E occurred in 1901.", "When did Event E occur?"),
        ]
        distractors = [
            {
                "schema_version": runner.DISTRACTOR_SCHEMA,
                "distractor_id": "full_distractor_one",
                "base_fact_id": "pbf_a",
                "slot": 1,
                "source_base_fact_id": "pbf_b",
                "distractor_text_en": "Author B",
                "relation_partition_id": "author",
                "answer_type_match_tier": "same_family_same_answer_type_bucket",
                "split_assignment": "development",
                "same_split": True,
                "same_leakage_component": False,
                "verified": False,
                "review_status": "pending_factual_uniqueness_and_semantic_review",
            },
            {
                "schema_version": runner.DISTRACTOR_SCHEMA,
                "distractor_id": "full_distractor_two",
                "base_fact_id": "pbf_a",
                "slot": 2,
                "source_base_fact_id": "pbf_c",
                "distractor_text_en": "Author C",
                "relation_partition_id": "author",
                "answer_type_match_tier": "same_family_same_answer_type_bucket",
                "split_assignment": "development",
                "same_split": True,
                "same_leakage_component": False,
                "verified": False,
                "review_status": "pending_factual_uniqueness_and_semantic_review",
            },
        ]
        neutrals = [
            {
                "schema_version": runner.NEUTRAL_SCHEMA,
                "neutral_candidate_id": "full_neutral_one",
                "base_fact_id": "pbf_a",
                "slot": 1,
                "source_base_fact_id": "pbf_d",
                "neutral_context_candidate_en": "Museum D is in City D.",
                "source_relation_partition_id": "location",
                "target_relation_partition_id": "author",
                "split_assignment": "development",
                "same_split": True,
                "same_leakage_component": False,
                "verified_unrelated": False,
                "review_status": "pending_unrelatedness_and_length_review",
            },
            {
                "schema_version": runner.NEUTRAL_SCHEMA,
                "neutral_candidate_id": "full_neutral_two",
                "base_fact_id": "pbf_a",
                "slot": 2,
                "source_base_fact_id": "pbf_e",
                "neutral_context_candidate_en": "Event E occurred in 1901.",
                "source_relation_partition_id": "date",
                "target_relation_partition_id": "author",
                "split_assignment": "development",
                "same_split": True,
                "same_leakage_component": False,
                "verified_unrelated": False,
                "review_status": "pending_unrelatedness_and_length_review",
            },
        ]
        full_path = root / "full_base_facts.jsonl"
        distractor_path = root / "distractor_candidates.jsonl"
        neutral_path = root / "neutral_reference_candidates.jsonl"
        write_jsonl(full_path, facts)
        write_jsonl(distractor_path, distractors)
        write_jsonl(neutral_path, neutrals)
        artifacts = {
            "full_base_facts": binding(full_path, runner.FULL_FACT_SCHEMA, facts),
            "distractor_candidates": binding(
                distractor_path, runner.DISTRACTOR_SCHEMA, distractors
            ),
            "neutral_reference_candidates": binding(
                neutral_path, runner.NEUTRAL_SCHEMA, neutrals
            ),
        }
        if pre_hf:
            manifest = {
                "schema_version": runner.PRE_HF_SUMMARY_SCHEMA,
                "status": runner.PRE_HF_STATUS,
                "artifacts": artifacts,
                "counts": {
                    "base_facts": len(facts),
                    "distractor_candidates": len(distractors),
                    "neutral_reference_candidates": len(neutrals),
                },
                "execution": {
                    "behavior_output_count": 0,
                    "development_behavior_exposure_count": 0,
                    "hf_model_execution": False,
                    "hf_tokenizer_execution": False,
                    "hidden_state_collection_count": 0,
                    "intervention_count": 0,
                    "validation_behavior_exposure_count": 0,
                    "sealed_behavior_exposure_count": 0,
                },
            }
            manifest_path = root / "summary.json"
        else:
            false_flags = {
                field: False
                for field in (
                    "canonical_freeze_emitted",
                    "review_freeze_emitted",
                    "split_freeze_emitted",
                    "distractor_candidates_verified",
                    "neutral_candidates_verified_unrelated",
                    "hf_checkpoint_bound",
                    "hf_tokenizer_bound",
                    "hf_model_executed",
                    "hf_tokenizer_executed",
                    "behavior_executed",
                    "validation_exposed",
                    "sealed_exposed",
                    "perturbation_authorized",
                    "path_not_token_authorized",
                    "human_gold",
                )
            }
            manifest = {
                "schema_version": runner.POSTREVIEW_MANIFEST_SCHEMA,
                "status": runner.POSTREVIEW_STATUS,
                "outputs": artifacts,
                "counts": {
                    "retained_base_facts": len(facts),
                    "distractor_candidates": len(distractors),
                    "neutral_reference_candidates": len(neutrals),
                },
                "safety_contract": {
                    "output_is_provisional": True,
                    "split_recomputed": True,
                    **false_flags,
                },
                "fact_review_contract": {
                    "revise_defer_missing_count": 0,
                    "proxy_review_only": True,
                    "human_gold": False,
                },
                "semantic_review_contract": {
                    "candidate_adjudication_complete_for_bounded_set": True,
                    "reviewer_evidence_is_human_gold": False,
                },
                "integrity": {"all_checks_passed": True},
            }
            manifest_path = root / "postreview_rebuild_manifest.json"
        write_json(manifest_path, manifest)
        return {
            "manifest": manifest_path,
            "full": full_path,
            "distractors": distractor_path,
            "neutrals": neutral_path,
            "facts": facts,
            "distractor_rows": distractors,
            "neutral_rows": neutrals,
        }

    @staticmethod
    def config():
        return {
            "provider_profiles": {
                "openai": {
                    "protocol": "openai",
                    "base_url": "https://candidate-review-fixture.invalid/v1",
                }
            },
            "execution": {"max_workers": 2, "max_retries": 1},
        }

    @staticmethod
    def reviewer_spec():
        return {
            "provider_profile": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "low",
            "max_output_tokens": 1024,
            "max_retries": 1,
        }

    @staticmethod
    def now():
        return "2026-09-23T00:00:00+00:00"

    def run_fixture(self, paths, output, router, **overrides):
        options = {
            "candidate_manifest_path": paths["manifest"],
            "output_dir": output,
            "config": self.config(),
            "env_path": output.parent / ".env",
            "reviewer_spec": self.reviewer_spec(),
            "expected_response_model": runner.DEFAULT_EXPECTED_RESPONSE_MODEL,
            "batch_size": 2,
            "checkpoint_compact_every": 3,
            "max_workers": 2,
            "router": router,
            "now_fn": self.now,
        }
        options.update(overrides)
        return runner.run_review(**options)

    def bind_direct_predecessor(self, current_paths, source_paths, source_result):
        source_review_path = Path(source_result["manifest_path"])
        source_review = json.loads(source_review_path.read_text(encoding="utf-8"))
        resolution_path = current_paths["manifest"].parent / "candidate_resolution_manifest.json"
        resolution = {
            "schema_version": "public-benchmark-full-candidate-resolution-manifest-v2",
            "inputs": {
                "predecessor_postreview_manifest": file_binding(
                    source_paths["manifest"], runner.POSTREVIEW_MANIFEST_SCHEMA
                ),
                "candidate_review_manifest": file_binding(
                    source_review_path, runner.RUN_MANIFEST_SCHEMA
                ),
                "candidate_review_adjudications": dict(
                    source_review["artifacts"]["adjudications"]
                ),
                "candidate_review_checkpoint": dict(
                    source_review["artifacts"]["checkpoint"]
                ),
            },
        }
        write_json(resolution_path, resolution)
        current_manifest = json.loads(
            current_paths["manifest"].read_text(encoding="utf-8")
        )
        current_manifest["schema_version"] = (
            runner.POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA
        )
        current_manifest["inputs"] = {
            "candidate_repair": {
                "candidate_resolution_manifest": file_binding(
                    resolution_path,
                    "public-benchmark-full-candidate-resolution-manifest-v2",
                ),
                "predecessor_postreview_manifest": file_binding(
                    source_paths["manifest"], runner.POSTREVIEW_MANIFEST_SCHEMA
                ),
            }
        }
        current_manifest["candidate_repair_contract"] = {
            "predecessor_candidate_reviews_blanket_valid": False,
            "exact_v3_accepted_projection_evidence_carry_forward_eligible": True,
            "candidate_id_or_row_sha_alone_sufficient_for_review_reuse": False,
            "full_candidate_evidence_coverage_required": True,
            "full_candidate_model_rereview_required": False,
            "changed_or_new_candidate_fresh_model_review_required": True,
        }
        write_json(current_paths["manifest"], current_manifest)

    def rebind_fixture_artifacts(self, paths):
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        artifact_key = (
            "artifacts"
            if manifest["schema_version"] == runner.PRE_HF_SUMMARY_SCHEMA
            else "outputs"
        )
        artifacts = manifest[artifact_key]
        for label, path, schema in (
            ("full_base_facts", paths["full"], runner.FULL_FACT_SCHEMA),
            ("distractor_candidates", paths["distractors"], runner.DISTRACTOR_SCHEMA),
            ("neutral_reference_candidates", paths["neutrals"], runner.NEUTRAL_SCHEMA),
        ):
            artifacts[label] = binding(path, schema, read_jsonl(path))
        write_json(paths["manifest"], manifest)

    def test_postreview_manifest_lineage_and_behavior_blind_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            manifest, kind, bindings, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"]
            )
            self.assertEqual(kind, "postreview_rebuild")
            self.assertEqual(len(units), 4)
            self.assertEqual(bindings["distractor_candidates"]["record_count"], 2)
            prompt = runner.build_review_prompt(units)
            payload = runner._prompt_payload(prompt)
            self.assertEqual(
                [row["candidate_id"] for row in payload["candidates"]],
                [unit.candidate_id for unit in units],
            )
            serialized = json.dumps(payload, sort_keys=True)
            for forbidden in runner.FORBIDDEN_PROMPT_KEYS:
                self.assertNotIn(f'"{forbidden}"', serialized)
            distractor = payload["candidates"][0]
            self.assertEqual(len(distractor["candidate_evidence"]["sibling_distractors"]), 1)
            self.assertEqual(
                distractor["target_fact"]["projection_role"],
                "target_fact_and_exclusive_answer_alias_scope",
            )
            self.assertEqual(
                distractor["target_fact"]["answer_aliases_en"],
                ["Author A", "Author A alias"],
            )
            self.assertEqual(
                distractor["candidate_source_fact"]["projection_role"],
                "provenance_only_not_target_alias_evidence",
            )
            self.assertNotIn(
                "answer_aliases_en", distractor["candidate_source_fact"]
            )
            self.assertFalse(manifest["safety_contract"]["behavior_executed"])

    def test_pre_hf_compatibility_schema_is_explicitly_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory), pre_hf=True)
            _, kind, _, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"]
            )
            self.assertEqual(kind, "pre_hf_schema_compatibility")
            self.assertEqual(len(units), 4)

    def test_end_to_end_writes_field_level_proxy_adjudications_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            router_instance = FakeRouter()
            result = self.run_fixture(paths, root / "run", router_instance)
            self.assertEqual(result["status"], "completed")
            rows = read_jsonl(Path(result["adjudications_path"]))
            self.assertEqual(len(rows), 4)
            distractor = next(row for row in rows if row["candidate_kind"] == "distractor")
            neutral = next(row for row in rows if row["candidate_kind"] == "neutral")
            self.assertTrue(distractor["distractor_factually_false"])
            self.assertTrue(distractor["answer_unique"])
            self.assertTrue(distractor["answer_alias_disjoint"])
            self.assertEqual(distractor["overlapping_target_aliases"], [])
            self.assertIsNone(
                distractor["answer_alias_disjoint_unresolved_evidence"]
            )
            self.assertIsNone(distractor["neutral_unrelated"])
            self.assertTrue(neutral["neutral_semantically_neutral"])
            self.assertTrue(neutral["neutral_unrelated"])
            self.assertIsNone(neutral["distractor_factually_false"])
            self.assertIsNone(neutral["overlapping_target_aliases"])
            self.assertTrue(all(row["human_gold"] is False for row in rows))
            self.assertTrue(all(row["behavior_blind"] is True for row in rows))
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertTrue(manifest["candidate_review_complete_for_full_set"])
            self.assertFalse(manifest["formal_review_complete"])
            self.assertEqual(manifest["evidence_boundary"]["hf_model_execution_count"], 0)
            self.assertEqual(
                manifest["evidence_boundary"]["validation_behavior_exposure_count"], 0
            )
            self.assertEqual(
                manifest["run_contract"]["answer_alias_contract"][
                    "answer_alias_disjoint_scope"
                ],
                "target_fact.answer_aliases_en_only",
            )
            self.assertFalse(
                manifest["run_contract"]["answer_alias_contract"][
                    "candidate_source_aliases_in_scope"
                ]
            )
            self.assertTrue(
                manifest["execution_policy"][
                    "contradictory_alias_evidence_fail_closed"
                ]
            )
            self.assertEqual(manifest["accepted_kind_counts"], {"distractor": 2, "neutral": 2})
            self.assertFalse((root / "run" / "candidate_review_checkpoint.journal.jsonl").exists())

    def test_first_v3_review_is_all_fresh_with_complete_evidence_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            result = self.run_fixture(paths, root / "run", FakeRouter())
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            candidate_ids = [row["candidate_id"] for row in checkpoint]

            self.assertEqual(
                manifest["run_contract"]["evidence_reuse_contract"]["mode"],
                "fresh_full_v3_baseline",
            )
            self.assertEqual(
                manifest["evidence_origin_counts"],
                {runner.FRESH_EVIDENCE_ORIGIN: 4},
            )
            self.assertEqual(manifest["fresh_model_review_candidate_count"], 4)
            self.assertEqual(manifest["carried_forward_candidate_count"], 0)
            self.assertEqual(
                manifest["fresh_model_review_candidate_ids_sha256"],
                runner.sha256_value(candidate_ids),
            )
            self.assertEqual(
                manifest["carried_forward_candidate_ids_sha256"],
                runner.sha256_value([]),
            )
            self.assertTrue(manifest["full_evidence_partition_complete"])
            self.assertTrue(
                all(
                    row["evidence_origin"] == runner.FRESH_EVIDENCE_ORIGIN
                    and row["model_calls_scope"] == "current_run"
                    and row["model_calls"]
                    and row["carry_forward_provenance"] is None
                    for row in checkpoint
                )
            )
            runner.load_review_evidence_manifest(Path(result["manifest_path"]))

    def test_exact_accept_is_carried_without_a_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "r01"
            current_root = root / "r02"
            source_root.mkdir()
            current_root.mkdir()
            source_paths = self.make_fixture(source_root)
            source_result = self.run_fixture(
                source_paths,
                source_root / "review",
                FakeRouter(),
                now_fn=lambda: "2026-09-23T00:00:00+00:00",
            )
            source_checkpoint = {
                row["candidate_id"]: row
                for row in read_jsonl(Path(source_result["checkpoint_path"]))
            }

            current_paths = self.make_fixture(current_root)
            self.bind_direct_predecessor(
                current_paths, source_paths, source_result
            )
            current_result = self.run_fixture(
                current_paths,
                current_root / "review",
                NoCallRouter(),
                now_fn=lambda: "2026-09-24T00:00:00+00:00",
            )
            current_manifest = json.loads(
                Path(current_result["manifest_path"]).read_text(encoding="utf-8")
            )
            current_checkpoint = read_jsonl(Path(current_result["checkpoint_path"]))

            self.assertEqual(current_result["fresh_model_review_candidate_count"], 0)
            self.assertEqual(current_result["carried_forward_candidate_count"], 4)
            self.assertEqual(
                current_manifest["evidence_origin_counts"],
                {runner.CARRIED_EVIDENCE_ORIGIN: 4},
            )
            self.assertTrue(current_manifest["full_evidence_partition_complete"])
            self.assertTrue(
                all(
                    row["evidence_origin"] == runner.CARRIED_EVIDENCE_ORIGIN
                    and row["model_calls_scope"] == "none_carried_forward"
                    and row["model_calls"] == []
                    and row["retry_history"] == []
                    for row in current_checkpoint
                )
            )
            for row in current_checkpoint:
                prior = source_checkpoint[row["candidate_id"]]
                self.assertEqual(
                    row["strict_adjudication"]["reviewed_at"],
                    prior["strict_adjudication"]["reviewed_at"],
                )
                self.assertEqual(
                    row["strict_adjudication"]["reviewer_id"],
                    prior["strict_adjudication"]["reviewer_id"],
                )
                self.assertEqual(
                    row["carry_forward_provenance"]["carry_depth"], 1
                )
            runner.load_review_evidence_manifest(
                Path(current_result["manifest_path"])
            )

            next_root = root / "r03"
            next_root.mkdir()
            next_paths = self.make_fixture(next_root)
            self.bind_direct_predecessor(
                next_paths, current_paths, current_result
            )
            next_result = self.run_fixture(
                next_paths, next_root / "review", NoCallRouter()
            )
            next_checkpoint = read_jsonl(Path(next_result["checkpoint_path"]))
            self.assertTrue(
                all(
                    row["carry_forward_provenance"]["carry_depth"] == 2
                    for row in next_checkpoint
                )
            )
            runner.load_review_evidence_manifest(Path(next_result["manifest_path"]))

    def test_administrative_row_drift_with_exact_projection_still_carries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "r01"
            current_root = root / "r02"
            source_root.mkdir()
            current_root.mkdir()
            source_paths = self.make_fixture(source_root)
            source_result = self.run_fixture(
                source_paths, source_root / "review", FakeRouter()
            )
            _, _, _, source_units = runner.load_review_units(
                candidate_manifest_path=source_paths["manifest"]
            )

            current_paths = self.make_fixture(current_root)
            facts = read_jsonl(current_paths["full"])
            distractors = read_jsonl(current_paths["distractors"])
            neutrals = read_jsonl(current_paths["neutrals"])
            for row in facts:
                row["split_assignment"] = "validation"
                row["split_status"] = "provisional_recomputed"
                row["leakage_component_id"] = f"recomputed_{row['base_fact_id']}"
            for row in [*distractors, *neutrals]:
                row["split_assignment"] = "validation"
            write_jsonl(current_paths["full"], facts)
            write_jsonl(current_paths["distractors"], distractors)
            write_jsonl(current_paths["neutrals"], neutrals)
            self.rebind_fixture_artifacts(current_paths)
            _, _, _, current_units = runner.load_review_units(
                candidate_manifest_path=current_paths["manifest"]
            )
            for prior, current in zip(source_units, current_units):
                self.assertEqual(prior.candidate_id, current.candidate_id)
                self.assertEqual(prior.projection, current.projection)
                self.assertNotEqual(
                    prior.candidate_row_sha256, current.candidate_row_sha256
                )
                self.assertNotEqual(
                    prior.target_base_fact_row_sha256,
                    current.target_base_fact_row_sha256,
                )
                self.assertNotEqual(
                    prior.source_base_fact_row_sha256,
                    current.source_base_fact_row_sha256,
                )

            self.bind_direct_predecessor(
                current_paths, source_paths, source_result
            )
            result = self.run_fixture(
                current_paths, current_root / "review", NoCallRouter()
            )
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            self.assertEqual(result["fresh_model_review_candidate_count"], 0)
            self.assertEqual(result["carried_forward_candidate_count"], 4)
            self.assertTrue(
                all(
                    row["candidate_row_sha256"]
                    != row["carry_forward_provenance"]["source_candidate_row_sha256"]
                    and row["target_base_fact_row_sha256"]
                    != row["carry_forward_provenance"][
                        "source_target_base_fact_row_sha256"
                    ]
                    and row["source_base_fact_row_sha256"]
                    != row["carry_forward_provenance"][
                        "source_source_base_fact_row_sha256"
                    ]
                    and row["model_calls"] == []
                    for row in checkpoint
                )
            )

    def test_projection_change_is_fresh_and_mixed_partition_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "r01"
            current_root = root / "r02"
            source_root.mkdir()
            current_root.mkdir()
            source_paths = self.make_fixture(source_root)
            source_result = self.run_fixture(
                source_paths, source_root / "review", FakeRouter()
            )

            current_paths = self.make_fixture(current_root)
            facts = read_jsonl(current_paths["full"])
            changed_id = "full_neutral_one"
            for row in facts:
                if row["base_fact_id"] == "pbf_d":
                    row["source_question_en"] = "In which city is Museum D?"
            write_jsonl(current_paths["full"], facts)
            self.rebind_fixture_artifacts(current_paths)
            self.bind_direct_predecessor(
                current_paths, source_paths, source_result
            )

            review_router = FakeRouter()
            result = self.run_fixture(
                current_paths,
                current_root / "review",
                review_router,
                batch_size=4,
                max_workers=1,
            )
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            by_id = {row["candidate_id"]: row for row in checkpoint}
            carried_ids = [
                row["candidate_id"]
                for row in checkpoint
                if row["evidence_origin"] == runner.CARRIED_EVIDENCE_ORIGIN
            ]

            self.assertEqual(review_router.calls, [[changed_id]])
            self.assertEqual(result["fresh_model_review_candidate_count"], 1)
            self.assertEqual(result["carried_forward_candidate_count"], 3)
            self.assertEqual(
                manifest["evidence_origin_counts"],
                {
                    runner.CARRIED_EVIDENCE_ORIGIN: 3,
                    runner.FRESH_EVIDENCE_ORIGIN: 1,
                },
            )
            self.assertTrue(manifest["full_evidence_partition_complete"])
            self.assertEqual(
                manifest["fresh_model_review_candidate_ids_sha256"],
                runner.sha256_value([changed_id]),
            )
            self.assertEqual(
                manifest["carried_forward_candidate_ids_sha256"],
                runner.sha256_value(carried_ids),
            )
            self.assertEqual(
                by_id[changed_id]["evidence_origin"], runner.FRESH_EVIDENCE_ORIGIN
            )
            self.assertTrue(by_id[changed_id]["model_calls"])
            self.assertTrue(
                all(by_id[candidate_id]["model_calls"] == [] for candidate_id in carried_ids)
            )
            runner.load_review_evidence_manifest(manifest_path)

            manifest["carried_forward_candidate_count"] = 2
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "Evidence partition field is stale"):
                runner.load_review_evidence_manifest(manifest_path)

    def test_exact_predecessor_reject_or_defer_survival_fails_closed(self):
        for decision, router_kwargs in (
            ("reject", {"reject_ids": {"full_distractor_one"}}),
            ("defer", {"defer_ids": {"full_distractor_one"}}),
        ):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_root = root / "r01"
                current_root = root / "r02"
                source_root.mkdir()
                current_root.mkdir()
                source_paths = self.make_fixture(source_root)
                source_result = self.run_fixture(
                    source_paths,
                    source_root / "review",
                    FakeRouter(**router_kwargs),
                )
                self.assertEqual(
                    source_result["overall_decision_counts"][decision], 1
                )
                current_paths = self.make_fixture(current_root)
                self.bind_direct_predecessor(
                    current_paths, source_paths, source_result
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "Exact predecessor non-accept survived candidate resolution",
                ):
                    self.run_fixture(
                        current_paths, current_root / "review", NoCallRouter()
                    )

    def test_source_sha_tampering_and_resume_source_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "r01"
            current_root = root / "r02"
            source_root.mkdir()
            current_root.mkdir()
            source_paths = self.make_fixture(source_root)
            source_result = self.run_fixture(
                source_paths, source_root / "review", FakeRouter()
            )
            current_paths = self.make_fixture(current_root)
            self.bind_direct_predecessor(
                current_paths, source_paths, source_result
            )
            with Path(source_result["checkpoint_path"]).open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(
                ValueError, "candidate review checkpoint SHA-256 mismatch"
            ):
                self.run_fixture(
                    current_paths, current_root / "review", NoCallRouter()
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "r01"
            current_root = root / "r02"
            source_root.mkdir()
            current_root.mkdir()
            source_paths = self.make_fixture(source_root)
            source_result = self.run_fixture(
                source_paths, source_root / "review", FakeRouter()
            )
            current_paths = self.make_fixture(current_root)
            self.bind_direct_predecessor(
                current_paths, source_paths, source_result
            )
            current_result = self.run_fixture(
                current_paths, current_root / "review", NoCallRouter()
            )
            with Path(source_result["manifest_path"]).open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(
                ValueError, "direct predecessor candidate review SHA-256 mismatch"
            ):
                self.run_fixture(
                    current_paths,
                    current_root / "review",
                    NoCallRouter(),
                    resume=True,
                )
            self.assertTrue(Path(current_result["checkpoint_path"]).is_file())

    def test_carry_provenance_break_and_cycle_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "r01"
            current_root = root / "r02"
            source_root.mkdir()
            current_root.mkdir()
            source_paths = self.make_fixture(source_root)
            source_result = self.run_fixture(
                source_paths, source_root / "review", FakeRouter()
            )
            current_paths = self.make_fixture(current_root)
            self.bind_direct_predecessor(
                current_paths, source_paths, source_result
            )
            current_result = self.run_fixture(
                current_paths, current_root / "review", NoCallRouter()
            )
            manifest_path = Path(current_result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint_path = Path(current_result["checkpoint_path"])
            checkpoint = read_jsonl(checkpoint_path)
            checkpoint[0]["carry_forward_provenance"][
                "source_checkpoint_row_sha256"
            ] = "0" * 64
            write_jsonl(checkpoint_path, checkpoint)
            manifest["artifacts"]["checkpoint"] = {
                **file_binding(checkpoint_path, runner.CHECKPOINT_SCHEMA),
                "record_count": len(checkpoint),
            }
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "Carry-forward provenance is stale"):
                runner.load_review_evidence_manifest(manifest_path)

            source_manifest_path = Path(source_result["manifest_path"]).resolve()
            with self.assertRaisesRegex(ValueError, "lineage cycle detected"):
                runner.load_review_evidence_manifest(
                    source_manifest_path,
                    _lineage_stack=(source_manifest_path,),
                )

    def test_batch_failure_recursively_splits_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            router_instance = FakeRouter(fail_multi=True)
            result = self.run_fixture(
                paths,
                root / "run",
                router_instance,
                batch_size=4,
                max_workers=1,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual([len(ids) for ids in router_instance.calls], [4, 2, 1, 1, 2, 1, 1])
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            self.assertEqual(
                [row["input_index"] for row in checkpoint], [0, 1, 2, 3]
            )
            self.assertTrue(all(len(row["model_calls"]) == 3 for row in checkpoint))

    def test_response_model_mismatch_fails_closed_without_binary_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            router_instance = FakeRouter(response_model="wrong-model")
            result = self.run_fixture(
                paths,
                root / "run",
                router_instance,
                batch_size=4,
                max_workers=1,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(len(router_instance.calls), 1)
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            self.assertTrue(all(row["terminal_status"] == "failed" for row in checkpoint))
            self.assertTrue(
                all(
                    row["model_calls"][-1]["response_model_identity_status"] == "mismatch"
                    for row in checkpoint
                )
            )

    def test_missing_response_model_fails_closed_without_binary_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            router_instance = FakeRouter(response_model=None)
            result = self.run_fixture(
                paths,
                root / "run",
                router_instance,
                batch_size=4,
                max_workers=1,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(len(router_instance.calls), 1)
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            self.assertTrue(
                all(
                    row["model_calls"][-1]["response_model_identity_status"]
                    == "not_observed"
                    for row in checkpoint
                )
            )

    def test_route_fingerprint_is_redacted_and_env_drift_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            output = root / "run"
            env_path = root / ".env"
            config = self.config()
            config["provider_profiles"]["openai"]["base_url_env"] = (
                "OPENAI_BASE_URL"
            )
            env_path.write_text(
                "OPENAI_BASE_URL=https://candidate-route-one.invalid/v1\n",
                encoding="utf-8",
            )
            result = self.run_fixture(
                paths,
                output,
                FakeRouter(),
                limit=1,
                config=config,
                env_path=env_path,
            )
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            route = manifest["run_contract"]["route_identity"]
            self.assertNotIn("candidate-route-one", json.dumps(route, sort_keys=True))
            self.assertFalse(route["credentials_or_endpoints_included"])
            self.assertEqual(len(manifest["run_contract"]["source_fingerprints"]), 3)

            env_path.write_text(
                "OPENAI_BASE_URL=https://candidate-route-two.invalid/v1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Resume run contract mismatch"):
                self.run_fixture(
                    paths,
                    output,
                    NoCallRouter(),
                    limit=1,
                    config=config,
                    env_path=env_path,
                    resume=True,
                )

    def test_retry_failed_only_and_resume_merges_durable_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            output = root / "run"
            failed_id = "full_distractor_two"
            first_router = FakeRouter(fail_ids={failed_id})
            first = self.run_fixture(
                paths, output, first_router, batch_size=1, max_workers=1
            )
            self.assertEqual(first["status"], "completed_with_failures")

            skipped = self.run_fixture(
                paths,
                output,
                NoCallRouter(),
                batch_size=1,
                max_workers=1,
                resume=True,
            )
            self.assertEqual(skipped["status"], "completed_with_failures")

            retry_router = FakeRouter()
            retried = self.run_fixture(
                paths,
                output,
                retry_router,
                batch_size=1,
                max_workers=1,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(retried["status"], "completed")
            self.assertEqual(retry_router.calls, [[failed_id]])
            checkpoint = read_jsonl(Path(retried["checkpoint_path"]))
            retried_row = next(row for row in checkpoint if row["candidate_id"] == failed_id)
            self.assertEqual(len(retried_row["retry_history"]), 1)

            # Simulate a crash after a durable journal append but before compact:
            # remove one compact row, append it to the journal, then add a partial tail.
            journal = output / "candidate_review_checkpoint.journal.jsonl"
            last = checkpoint[-1]
            write_jsonl(Path(retried["checkpoint_path"]), checkpoint[:-1])
            journal.write_bytes(runner.canonical_json_bytes(last) + b"\n{\"partial\":")
            resumed = self.run_fixture(
                paths,
                output,
                NoCallRouter(),
                batch_size=1,
                max_workers=1,
                resume=True,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(len(read_jsonl(Path(resumed["checkpoint_path"]))), 4)
            self.assertFalse(journal.exists())

    def test_manifest_sha_record_count_and_safety_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            with paths["distractors"].open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.load_review_units(candidate_manifest_path=paths["manifest"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["outputs"]["neutral_reference_candidates"]["record_count"] += 1
            write_json(paths["manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "record_count mismatch"):
                runner.load_review_units(candidate_manifest_path=paths["manifest"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["safety_contract"]["validation_exposed"] = True
            write_json(paths["manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "validation_exposed"):
                runner.load_review_units(candidate_manifest_path=paths["manifest"])

    def test_response_validator_requires_exact_ids_and_field_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            _, _, _, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"], limit=2
            )
            response = {
                "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                "records": [
                    {
                        "candidate_id": units[0].candidate_id,
                        "candidate_kind": "distractor",
                        "overall_decision": "accept",
                        "distractor_factually_false": True,
                        "answer_unique": True,
                        "answer_alias_disjoint": True,
                        "neutral_semantically_neutral": None,
                        "neutral_unrelated": None,
                        "overlapping_target_aliases": [],
                        "answer_alias_disjoint_unresolved_evidence": None,
                        "rationale": "Complete first record only.",
                        "confidence": "high",
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                runner.validate_batch_response(response, units)
            response["records"].append(
                {
                    **response["records"][0],
                    "candidate_id": units[1].candidate_id,
                    "overall_decision": "accept",
                    "answer_unique": None,
                }
            )
            with self.assertRaisesRegex(ValueError, "every relevant field true"):
                runner.validate_batch_response(response, units)
            response["records"][1]["answer_unique"] = 1
            with self.assertRaisesRegex(ValueError, "boolean or null"):
                runner.validate_batch_response(response, units)

    def test_answer_alias_scope_is_target_only_when_source_alias_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            _, _, _, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"], limit=1
            )
            unit = units[0]
            self.assertTrue(
                runner._same_text(
                    unit.projection["candidate_evidence"]["distractor_text_en"],
                    unit.projection["candidate_source_fact"]["answer_en"],
                )
            )
            self.assertNotIn(
                "answer_aliases_en", unit.projection["candidate_source_fact"]
            )
            runner.validate_batch_response(
                {
                    "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                    "records": [distractor_verdict(unit)],
                },
                [unit],
            )

    def test_overlap_evidence_must_be_nonempty_target_alias_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            _, _, _, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"], limit=1
            )
            unit = units[0]
            valid_subset = distractor_verdict(
                unit,
                overall_decision="reject",
                answer_alias_disjoint=False,
                overlapping_target_aliases=["Author A alias"],
            )
            runner.validate_batch_response(
                {
                    "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                    "records": [valid_subset],
                },
                [unit],
            )

            empty_overlap = {**valid_subset, "overlapping_target_aliases": []}
            with self.assertRaisesRegex(ValueError, "requires overlapping target aliases"):
                runner.validate_batch_response(
                    {
                        "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                        "records": [empty_overlap],
                    },
                    [unit],
                )

            source_alias_only = {
                **valid_subset,
                "overlapping_target_aliases": ["Author B"],
            }
            with self.assertRaisesRegex(ValueError, "only from target aliases"):
                runner.validate_batch_response(
                    {
                        "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                        "records": [source_alias_only],
                    },
                    [unit],
                )

    def test_exact_target_alias_contradiction_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            _, _, _, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"], limit=1
            )
            original = units[0]
            projection = json.loads(json.dumps(original.projection))
            projection["candidate_evidence"]["distractor_text_en"] = " author a "
            unit = original._replace(projection=projection)

            with self.assertRaisesRegex(ValueError, "Exact target-alias match contradicts"):
                runner.validate_batch_response(
                    {
                        "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                        "records": [distractor_verdict(unit)],
                    },
                    [unit],
                )

            exact_overlap = distractor_verdict(
                unit,
                overall_decision="reject",
                answer_alias_disjoint=False,
                overlapping_target_aliases=["Author A"],
            )
            runner.validate_batch_response(
                {
                    "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                    "records": [exact_overlap],
                },
                [unit],
            )

    def test_unresolved_alias_judgment_requires_explicit_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(Path(directory))
            _, _, _, units = runner.load_review_units(
                candidate_manifest_path=paths["manifest"], limit=1
            )
            unit = units[0]
            unresolved = distractor_verdict(
                unit,
                overall_decision="defer",
                answer_alias_disjoint=None,
                answer_alias_disjoint_unresolved_evidence=(
                    "The supplied aliases do not resolve a possible acronym expansion."
                ),
            )
            runner.validate_batch_response(
                {
                    "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                    "records": [unresolved],
                },
                [unit],
            )
            unresolved["answer_alias_disjoint_unresolved_evidence"] = None
            with self.assertRaisesRegex(ValueError, "unresolved_evidence"):
                runner.validate_batch_response(
                    {
                        "schema_version": runner.BATCH_RESPONSE_SCHEMA,
                        "records": [unresolved],
                    },
                    [unit],
                )

    def test_alias_evidence_contradiction_retries_and_never_becomes_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            bad_id = "full_distractor_one"
            router_instance = FakeRouter(alias_contradiction_ids={bad_id})
            result = self.run_fixture(
                paths,
                root / "run",
                router_instance,
                batch_size=2,
                max_workers=1,
            )
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertIn([bad_id], router_instance.calls)
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            failed = next(row for row in checkpoint if row["candidate_id"] == bad_id)
            self.assertEqual(failed["terminal_status"], "failed")
            self.assertIsNone(failed["strict_adjudication"])
            self.assertNotIn("reject", result["overall_decision_counts"])

    def test_legacy_resume_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            output = root / "run"
            result = self.run_fixture(
                paths, output, FakeRouter(), limit=1, batch_size=1, max_workers=1
            )
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            legacy_contract = dict(manifest["run_contract"])
            legacy_contract["tool_version"] = (
                "full-public-benchmark-candidate-review-runner-v2"
            )
            legacy_contract["prompt_contract_sha256"] = runner.sha256_value(
                {
                    "prompt_version": (
                        "public-benchmark-full-candidate-semantic-review-v1"
                    ),
                    "projection_schema": (
                        "public-benchmark-full-candidate-review-projection-v1"
                    ),
                    "batch_response_schema": (
                        "public-benchmark-full-candidate-review-batch-response-v1"
                    ),
                    "review_method": (
                        "independent_behavior_blind_candidate_semantic_review_v1"
                    ),
                }
            )
            manifest["run_contract"] = legacy_contract
            manifest["run_contract_sha256"] = runner.sha256_value(legacy_contract)
            write_json(Path(result["manifest_path"]), manifest)
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            checkpoint[0]["run_contract_sha256"] = manifest["run_contract_sha256"]
            write_jsonl(Path(result["checkpoint_path"]), checkpoint)

            with self.assertRaisesRegex(ValueError, "Resume run contract mismatch"):
                self.run_fixture(
                    paths,
                    output,
                    NoCallRouter(),
                    limit=1,
                    batch_size=1,
                    max_workers=1,
                    resume=True,
                )

    def test_limit_has_deterministic_selected_id_contract_and_no_full_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            result = self.run_fixture(
                paths, root / "run", FakeRouter(), limit=3, batch_size=3, max_workers=1
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            ids = [row["candidate_id"] for row in checkpoint]
            self.assertEqual(manifest["selected_candidate_ids_sha256"], runner.sha256_value(ids))
            self.assertEqual(manifest["selected_candidate_count"], 3)
            self.assertFalse(manifest["selection_is_full_candidate_set"])
            self.assertFalse(manifest["candidate_review_complete_for_full_set"])
            with self.assertRaisesRegex(ValueError, "Resume run contract mismatch"):
                self.run_fixture(
                    paths,
                    root / "run",
                    NoCallRouter(),
                    limit=4,
                    batch_size=3,
                    max_workers=1,
                    resume=True,
                )

    def test_model_spec_redacts_secret_fields_and_expected_identity_is_required(self):
        config = {
            "provider_profiles": {"openai": {"type": "fixture"}},
            "model_roles": {
                "full_pool_candidate_semantic_review": {
                    "reviewer": {
                        "provider_profile": "openai",
                        "model": "gpt-5.5",
                        "api_key": "must-not-escape",
                        "base_url": "https://secret.invalid",
                    }
                }
            },
        }
        spec = runner.resolve_reviewer_spec(config)
        self.assertNotIn("api_key", spec)
        self.assertNotIn("base_url", spec)
        self.assertTrue(spec["json_mode"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            with self.assertRaisesRegex(ValueError, "expected_response_model"):
                self.run_fixture(
                    paths,
                    root / "run",
                    FakeRouter(),
                    expected_response_model="",
                )

    def test_protocol_identity_rejects_omission_and_alternate_route(self):
        with self.assertRaisesRegex(ValueError, "protocol identity"):
            runner.resolve_reviewer_spec(self.config(), model="alternate-reviewer")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            with self.assertRaisesRegex(ValueError, "expected_response_model is required"):
                self.run_fixture(
                    paths,
                    root / "missing-identity",
                    FakeRouter(),
                    expected_response_model=None,
                )
            alternate_spec = {
                **self.reviewer_spec(),
                "model": "alternate-reviewer",
            }
            with self.assertRaisesRegex(ValueError, "protocol identity"):
                self.run_fixture(
                    paths,
                    root / "alternate-identity",
                    FakeRouter(response_model="alternate-response-model"),
                    reviewer_spec=alternate_spec,
                    expected_response_model="alternate-response-model",
                )

    def test_resume_rejects_tampered_completed_identity_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            output = root / "run"
            result = self.run_fixture(
                paths, output, FakeRouter(), limit=1, batch_size=1, max_workers=1
            )
            checkpoint = read_jsonl(Path(result["checkpoint_path"]))
            checkpoint[0]["model_calls"][-1]["response_model"] = "alternate-response-model"
            write_jsonl(Path(result["checkpoint_path"]), checkpoint)
            with self.assertRaisesRegex(ValueError, "response model is not verified"):
                self.run_fixture(
                    paths,
                    output,
                    NoCallRouter(),
                    limit=1,
                    batch_size=1,
                    max_workers=1,
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
