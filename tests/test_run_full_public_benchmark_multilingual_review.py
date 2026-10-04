import copy
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_full_public_benchmark_multilingual_review.py"
SPEC = importlib.util.spec_from_file_location("multilingual_review_test", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def failure(code="insufficient_user_quota", status=403):
    return m.BatchRequestFailure({
        "attempt_count": 1,
        "attempts": [{"attempt": 1, "status": "failed", "http_status": status,
                      "provider_code": code, "error_type": "PermissionDeniedError"
                      if status == 403 else "APIError"}],
    })


def success(item):
    return {"base_fact_id": item["base_fact_id"],
            "input_record_sha256": item["input_record_sha256"],
            "terminal_status": "completed", "parsed_response": {"ok": True}}


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "checkpoint.jsonl"
        self.items = [{"base_fact_id": f"fact_{i}", "input_record_sha256": f"sha_{i}"}
                      for i in range(12)]

    def run_stage(self, worker, **overrides):
        args = dict(items=self.items, output_path=self.output, stage="initial_generation",
                    schema_version=m.GENERATION_SCHEMA, language_codes=list(m.DEFAULT_LANGUAGES),
                    run_fingerprint="new", batch_worker=worker, record_validator=lambda row, item: None,
                    max_workers=1, batch_size=2, checkpoint_every=64, resume=False,
                    retry_failed=False)
        args.update(overrides)
        return m.run_checkpoint_stage(**args)

    def test_quota_stops_without_marking_unattempted_rows(self):
        calls = []
        def worker(batch):
            calls.append(batch)
            raise failure()
        with self.assertRaises(m.PipelineStopped) as caught:
            self.run_stage(worker)
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.summary["reason"], "provider_quota_exhausted")
        self.assertEqual(caught.exception.summary["unattempted_pending_source_records"], 10)
        self.assertEqual(len(m.read_jsonl(self.output)), 2)
        self.assertFalse(self.output.with_suffix(".jsonl.journal").exists())

    def test_parallel_stop_drains_only_in_flight_wave(self):
        barrier = threading.Barrier(2)
        calls = []
        def worker(batch):
            calls.append(batch)
            barrier.wait(timeout=5)
            if batch[0]["base_fact_id"] == "fact_0":
                raise failure()
            return [success(item) for item in batch]
        with self.assertRaises(m.PipelineStopped):
            self.run_stage(worker, max_workers=2)
        self.assertEqual(len(calls), 2)
        rows = m.read_jsonl(self.output)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["terminal_status"] == "completed" for r in rows), 2)

    def test_three_failed_batches_stop_transient_outage(self):
        calls = []
        def worker(batch):
            calls.append(batch)
            raise failure("server_error", 503)
        with self.assertRaises(m.PipelineStopped) as caught:
            self.run_stage(worker, batch_size=1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(caught.exception.summary["reason"], "consecutive_failed_batches")

    def test_success_resets_consecutive_failure_counter(self):
        def worker(batch):
            if batch[0]["base_fact_id"] in {"fact_0", "fact_1", "fact_3", "fact_4"}:
                raise failure("server_error", 503)
            return [success(item) for item in batch]
        self.assertEqual(len(self.run_stage(worker, batch_size=1)), 12)

    def test_resume_retries_failed_only_and_preserves_successful_bytes(self):
        accepted = {**success(self.items[0]), "run_fingerprint": "old"}
        rejected = {**success(self.items[1]), "run_fingerprint": "old",
                    "parsed_response": {"decision": "reject"}}
        failed = {**success(self.items[2]), "run_fingerprint": "old",
                  "terminal_status": "failed", "parsed_response": None}
        m.write_jsonl(self.output, [accepted, rejected, failed])
        called = []
        def worker(batch):
            called.extend(r["base_fact_id"] for r in batch)
            return [success(item) for item in batch]
        result = self.run_stage(worker, resume=True, retry_failed=True,
                                inherited_run_fingerprints=["old"])
        self.assertNotIn("fact_0", called)
        self.assertNotIn("fact_1", called)
        self.assertIn("fact_2", called)
        self.assertEqual(result[:2], [accepted, rejected])
        self.assertEqual(result[2]["retry_history"][0]["failed_record_sha256"], m.sha256_value(failed))

    def test_unbound_checkpoint_is_rejected_before_any_request(self):
        m.write_jsonl(self.output, [{**success(self.items[0]), "run_fingerprint": "unknown"}])
        def never(batch):
            self.fail("no model call is allowed")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.run_stage(never, resume=True, inherited_run_fingerprints=["old"])

    def test_journal_is_compacted_after_unexpected_exception(self):
        def worker(batch):
            return [success(item) for item in batch]
        def validator(row, item):
            if item["base_fact_id"] == "fact_2":
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage(worker, record_validator=validator)
        self.assertEqual(len(m.read_jsonl(self.output)), 2)

    def test_quota_under_429_is_fatal_but_rate_limit_is_not(self):
        self.assertEqual(m.service_stop_reason(failure("insufficient_quota", 429)),
                         "provider_quota_exhausted")
        self.assertIsNone(m.service_stop_reason(failure("rate_limit_exceeded", 429)))

    def test_operator_stop_drains_current_wave_and_keeps_successes(self):
        stop_file = self.output.parent / "STOP_AFTER_WAVE"
        calls = []
        def worker(batch):
            calls.append(batch)
            stop_file.touch()
            return [success(item) for item in batch]
        with self.assertRaises(m.PipelineStopped) as caught:
            self.run_stage(worker, max_workers=2, stop_file=stop_file)
        self.assertEqual(len(calls), 2)
        self.assertEqual(caught.exception.summary["reason"], "operator_stop_requested")
        rows = m.read_jsonl(self.output)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["terminal_status"] == "completed" for row in rows))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.parent_path = root / "parent" / "run_manifest.json"
        self.parent_path.parent.mkdir()
        self.output = root / "recovery"
        self.output.mkdir()
        self.contract = {
            "tool_version": m.TOOL_VERSION, "input_bindings": {"source": "sha"},
            "source_fingerprints": {"runner_sha256": "new", "shared_zh_runner_sha256": "same"},
            "execution_policy": {"stop": True},
            "models": {"generator": "same", "reviewer": "same"},
            "target_languages": list(m.DEFAULT_LANGUAGES),
            "selected_targets": {"sha256": "selection", "path": "new"},
            "route_identity": {"hash": "same"},
        }
        self.parent = copy.deepcopy(self.contract)
        self.parent.update(status="completed_with_quarantine_or_limited_scope",
                           tool_version="full-public-benchmark-multilingual-review-runner-v1",
                           run_fingerprint="old", output_artifacts={})
        self.parent["source_fingerprints"]["runner_sha256"] = "old"
        for key in ("translation_initial", "translation_reviews"):
            path = self.parent_path.parent / (key + ".jsonl")
            m.write_jsonl(path, [{"run_fingerprint": "old", "terminal_status": "completed"}])
            self.parent["output_artifacts"][key] = m.artifact_binding(path)
        m.write_json(self.parent_path, self.parent)

    def test_import_is_byte_identical_and_parent_is_unchanged(self):
        before = self.parent_path.read_bytes()
        lineage = m.recovery_source(self.parent_path, self.contract, self.output, resume=False)
        for key in ("translation_initial", "translation_reviews"):
            name = key + ".jsonl"
            self.assertEqual((self.output / name).read_bytes(),
                             (self.parent_path.parent / name).read_bytes())
        self.assertEqual(before, self.parent_path.read_bytes())
        self.assertEqual(lineage["parent_run_fingerprint"], "old")

    def test_changed_model_route_input_or_shared_runtime_cannot_reuse(self):
        for key in ("models", "route_identity", "input_bindings", "source_fingerprints"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.contract)
                changed[key] = {"different": True}
                with self.assertRaises(ValueError):
                    m.recovery_source(self.parent_path, changed, self.output, resume=False)

    def test_tampered_checkpoint_cannot_be_imported(self):
        (self.parent_path.parent / "translation_initial.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            m.recovery_source(self.parent_path, self.contract, self.output, resume=False)

    def test_interrupted_v2_snapshot_inherits_verified_ancestor(self):
        ancestor = self.parent_path.parent / "ancestor.json"
        m.write_json(ancestor, {"run_fingerprint": "legacy"})
        self.parent["tool_version"] = "full-public-benchmark-multilingual-review-runner-v2"
        self.parent["status"] = "interrupted_checkpoint_snapshot"
        self.parent["recovery_lineage"] = {
            "parent_manifest": m.artifact_binding(ancestor), "parent_run_fingerprint": "legacy",
        }
        m.write_json(self.parent_path, self.parent)
        lineage = m.recovery_source(self.parent_path, self.contract, self.output, resume=False)
        self.assertEqual(lineage["inherited_run_fingerprints"], ["legacy", "old"])

    def test_ancestor_tampering_blocks_import(self):
        ancestor = self.parent_path.parent / "ancestor.json"
        m.write_json(ancestor, {"run_fingerprint": "legacy"})
        self.parent["recovery_lineage"] = {
            "parent_manifest": m.artifact_binding(ancestor), "parent_run_fingerprint": "legacy",
        }
        m.write_json(self.parent_path, self.parent)
        ancestor.write_text('{}\n')
        with self.assertRaisesRegex(ValueError, "ancestor manifest hash mismatch"):
            m.recovery_source(self.parent_path, self.contract, self.output, resume=False)


class ExecutionTests(unittest.TestCase):
    def test_review_stop_resume_preserves_generation_and_separates_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = {
                "base_fact_id": "fact", "cohort_item_id": "cohort", "split_assignment": "development",
                "relation_partition_id": "relation", "relation_partition_status": "broad_rule_candidate_partition",
                "candidate_base_fact_row_sha256": "sha", "projection_item_sha256": "sha",
                "source_record_sha256": "sha", "input_record_sha256": "sha",
                "source": {"prompt_en": "Question?", "canonical_fact_en": "A fact", "answer_en": "answer",
                           "answer_aliases_en": ["answer"]},
                "distractors": [{"distractor_id": "d", "text": "distractor"}],
                "neutral_candidates": [{"neutral_candidate_id": "n", "text": "neutral"}],
            }
            loaded = {
                "items": [item], "bindings": {}, "projection_manifest": {"projection_id": "p"},
                "source_review_manifest": {"projection_id": "p"}, "source_projection_record_count": 1,
                "source_projection_split_counts": {"development": 1}, "relation_partition_status_counts": {},
                "ordered_relation_usable_base_fact_ids_sha256": "ids",
            }
            argv = []
            for flag in ("config", "env-file", "candidate-projection-manifest", "candidate-base-facts",
                         "projection-manifest", "projection-items", "source-review-manifest", "source-records"):
                argv += ["--" + flag, str(root / flag)]
            argv += ["--output-dir", str(root / "run"), "--max-workers", "1"]
            args = m.build_parser().parse_args(argv)
            args.generation_workers = 4
            args.review_workers = 1
            calls = []
            configs_seen = []
            fail_review = [True]

            class Router:
                def __init__(self, *a):
                    configs_seen.append(copy.deepcopy(a[0]))

                def request_json(self, stage, item_id, spec, prompt, validator):
                    calls.append(stage)
                    if "generation" in stage:
                        translation = {"prompt": "Question?", "canonical_fact": "A fact", "answer": "answer",
                                       "answer_aliases": ["answer"], "distractors": item["distractors"],
                                       "neutral_candidates": item["neutral_candidates"]}
                        parsed = {"records": [{"base_fact_id": "fact", "translations": {
                            lang: copy.deepcopy(translation) for lang in m.DEFAULT_LANGUAGES}}]}
                    elif fail_review[0]:
                        return m.runtime.ModelResult("failed", None, None, None, {}, 1, 1,
                                                     failure().metadata["attempts"])
                    else:
                        reviews = {}
                        for lang in m.DEFAULT_LANGUAGES:
                            reviews[lang] = {"decision": "reject" if lang == "sw" else "accept",
                                             "translation_equivalent": lang != "sw", "natural": True,
                                             "answer_not_leaked": True, "candidate_identity_preserved": True,
                                             "issues": ["meaning"] if lang == "sw" else []}
                        parsed = {"records": [{"base_fact_id": "fact", "reviews": reviews}]}
                    validator(parsed)
                    return m.runtime.ModelResult("completed", parsed, None, spec["expected_response_model"],
                                                 {}, 1, 1, [{"attempt": 1, "status": "completed"}])

            routes = {"records": [{"normalized_base_url_sha256": "one", "normalized_request_route_sha256": "one"},
                                  {"normalized_base_url_sha256": "two", "normalized_request_route_sha256": "two"}]}
            specs = {role: {"model": role, "provider_profile": role} for role in ("generator", "reviewer")}
            with mock.patch.object(m, "load_config", return_value={}), \
                 mock.patch.object(m, "load_bound_inputs", return_value=loaded), \
                 mock.patch.object(m, "_model_specs", return_value=specs), \
                 mock.patch.object(m, "build_route_identity_set", return_value=routes), \
                 mock.patch.object(m, "attach_route_identity_guard"):
                with self.assertRaises(m.PipelineStopped):
                    m.execute(args, router_factory=Router)
                manifest = m.read_json(root / "run/run_manifest.json")
                self.assertEqual(manifest["status"], "stopped_external_service")
                self.assertFalse(manifest["completion_claims"]["selected_scope_processing_complete"])
                generation_bytes = (root / "run/translation_initial.jsonl").read_bytes()
                fail_review[0] = False
                args.resume = args.retry_failed = True
                result = m.execute(args, router_factory=Router)
                self.assertEqual(configs_seen[-1]["execution"]["provider_max_concurrency"],
                                 {"generator": 4, "reviewer": 1})
                self.assertEqual(result["execution_policy"]["generation_workers"], 4)
                self.assertEqual(result["execution_policy"]["review_workers"], 1)
                self.assertEqual(generation_bytes, (root / "run/translation_initial.jsonl").read_bytes())
                self.assertEqual(calls.count("full_multilingual_translation_generation"), 1)
                self.assertEqual(result["counts"]["by_language"]["sw"]["semantic_rejected"], 1)
                self.assertEqual(result["counts"]["by_language"]["sw"]["operational_failed"], 0)
                self.assertTrue(result["completion_claims"]["full_4513_five_language_proxy_review_complete"])
                self.assertFalse(result["completion_claims"]["all_4513_five_language_pairs_accepted"])
                m.execute(args, router_factory=Router)
                self.assertEqual(len(calls), 3)  # completed semantic rejects are never re-reviewed


if __name__ == "__main__":
    unittest.main()
