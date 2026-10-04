import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_full_public_benchmark_zh_review.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_full_public_benchmark_zh_review", SCRIPT_PATH
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_bytes(
        b"".join(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
            for row in rows
        )
    )


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeRouter:
    calls = []

    def __init__(self, config, env_path, event_log):
        self.config = config
        self.env_path = env_path
        self.event_log = event_log

    @staticmethod
    def translation(base_fact_id, repaired=False):
        suffix = "修复" if repaired else "初译"
        return {
            "prompt_zh": f"{base_fact_id} 提示{suffix}：",
            "canonical_fact_zh": f"{base_fact_id} 事实{suffix}",
            "answer_zh": f"答案{suffix}",
            "answer_aliases_zh": [f"别名{suffix}"],
            "distractors": [
                {"distractor_id": f"d_{base_fact_id}", "text_zh": f"干扰项{suffix}"}
            ],
            "neutral_candidates": [
                {
                    "neutral_candidate_id": f"n_{base_fact_id}",
                    "text_zh": f"中性上下文{suffix}",
                }
            ],
        }

    @staticmethod
    def review(base_fact_id, accepted):
        def check(ok=True):
            return {
                "translation_equivalent": ok,
                "natural_zh": ok,
                "issues": [] if ok else ["prompt mismatch"],
            }

        return {
            "decision": "accept" if accepted else "reject",
            "issues": [] if accepted else ["prompt mismatch"],
            "checks": {
                "prompt": check(accepted),
                "canonical_fact": check(),
                "answer": check(),
                "answer_aliases": [{"alias_index": 0, **check()}],
                "distractors": [
                    {"distractor_id": f"d_{base_fact_id}", **check()}
                ],
                "neutral_candidates": [
                    {"neutral_candidate_id": f"n_{base_fact_id}", **check()}
                ],
            },
        }

    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = json.loads(prompt.split(" Input: ", 1)[1])
        base_fact_ids = [record["base_fact_id"] for record in payload["records"]]
        self.__class__.calls.append(
            (stage, tuple(base_fact_ids), model_spec["model"], prompt)
        )
        if stage == "full_zh_translation_generation" and "pbf_3" in base_fact_ids:
            return module.runtime.ModelResult(
                "failed",
                None,
                None,
                None,
                {},
                1,
                1,
                [{"attempt": 1, "status": "failed", "error_type": "TimeoutError"}],
            )
        if stage == "full_zh_translation_generation":
            parsed = {
                "records": [
                    {
                        "base_fact_id": base_fact_id,
                        "translation": self.translation(base_fact_id),
                    }
                    for base_fact_id in base_fact_ids
                ]
            }
            response_model = module.DEFAULT_GENERATOR_RESPONSE_MODEL
        elif stage == "full_zh_translation_review":
            parsed = {
                "records": [
                    {
                        "base_fact_id": base_fact_id,
                        "review": self.review(
                            base_fact_id, accepted=base_fact_id == "pbf_1"
                        ),
                    }
                    for base_fact_id in base_fact_ids
                ]
            }
            response_model = module.DEFAULT_REVIEWER_RESPONSE_MODEL
        elif stage == "full_zh_translation_repair":
            parsed = {
                "records": [
                    {
                        "base_fact_id": base_fact_id,
                        "translation": self.translation(base_fact_id, repaired=True),
                    }
                    for base_fact_id in base_fact_ids
                ]
            }
            response_model = module.DEFAULT_GENERATOR_RESPONSE_MODEL
        elif stage == "full_zh_translation_repair_review":
            parsed = {
                "records": [
                    {
                        "base_fact_id": base_fact_id,
                        "review": self.review(base_fact_id, accepted=True),
                    }
                    for base_fact_id in base_fact_ids
                ]
            }
            response_model = module.DEFAULT_REVIEWER_RESPONSE_MODEL
        else:
            raise AssertionError(stage)
        validator(parsed)
        return module.runtime.ModelResult(
            "completed",
            parsed,
            json.dumps(parsed, ensure_ascii=False),
            response_model,
            {"total_tokens": 10},
            1,
            1,
            [{"attempt": 1, "status": "ok", "latency_ms": 1}],
        )


class NoCallRouter:
    def __init__(self, config, env_path, event_log):
        pass

    def request_json(self, *args, **kwargs):
        raise AssertionError("resume should not repeat a terminal checkpoint")


class IdentityMismatchRouter(FakeRouter):
    def request_json(self, stage, item_id, model_spec, prompt, validator):
        result = super().request_json(stage, item_id, model_spec, prompt, validator)
        if stage != "full_zh_translation_generation" or result.terminal_status != "completed":
            return result
        return module.runtime.ModelResult(
            result.terminal_status,
            result.parsed_response,
            result.raw_response,
            "unexpected-model",
            result.usage,
            result.latency_ms,
            result.attempt_count,
            result.attempts,
        )


class IdentityMissingRouter(FakeRouter):
    def request_json(self, stage, item_id, model_spec, prompt, validator):
        result = super().request_json(stage, item_id, model_spec, prompt, validator)
        if stage != "full_zh_translation_generation" or result.terminal_status != "completed":
            return result
        return module.runtime.ModelResult(
            result.terminal_status,
            result.parsed_response,
            result.raw_response,
            None,
            result.usage,
            result.latency_ms,
            result.attempt_count,
            result.attempts,
        )


class RecoveryRouter(FakeRouter):
    def request_json(self, stage, item_id, model_spec, prompt, validator):
        payload = json.loads(prompt.split(" Input: ", 1)[1])
        base_fact_ids = [record["base_fact_id"] for record in payload["records"]]
        if stage != "full_zh_translation_generation" or "pbf_3" not in base_fact_ids:
            return super().request_json(stage, item_id, model_spec, prompt, validator)
        self.__class__.calls.append(
            (stage, tuple(base_fact_ids), model_spec["model"], prompt)
        )
        parsed = {
            "records": [
                {
                    "base_fact_id": base_fact_id,
                    "translation": self.translation(base_fact_id),
                }
                for base_fact_id in base_fact_ids
            ]
        }
        validator(parsed)
        return module.runtime.ModelResult(
            "completed",
            parsed,
            json.dumps(parsed, ensure_ascii=False),
            module.DEFAULT_GENERATOR_RESPONSE_MODEL,
            {"total_tokens": 10},
            1,
            1,
            [{"attempt": 1, "status": "ok", "latency_ms": 1}],
        )


class FullZhReviewTests(unittest.TestCase):
    def setUp(self):
        FakeRouter.calls = []

    def test_postreview_manifest_schema_matches_v2_materializer_contract(self):
        self.assertEqual(
            module.POSTREVIEW_MANIFEST_SCHEMA,
            "public-benchmark-full-postreview-rebuild-manifest-v2",
        )

    def make_fixture(self, root):
        root = Path(root)
        paths = {
            "config": root / "config.json",
            "env": root / ".env",
            "universe_manifest": root / "formal_cohort_universe_manifest.json",
            "universe_items": root / "formal_cohort_universe_items.jsonl",
            "base": root / "full_base_facts.jsonl",
            "distractors": root / "distractor_candidates.jsonl",
            "neutrals": root / "neutral_reference_candidates.jsonl",
            "split": root / "split_manifest.json",
            "output": root / "output",
        }
        write_json(
            paths["config"],
            {
                "provider_profiles": {
                    "aliyun": {
                        "protocol": "openai_compatible",
                        "api_key_env": "ALIYUN_API_KEY",
                        "base_url_env": "ALIYUN_BASE_URL",
                        "base_url": "https://aliyun-fixture.invalid/v1",
                    },
                    "openai": {
                        "protocol": "openai",
                        "api_key_env": "OPENAI_API_KEY",
                        "base_url_env": "OPENAI_BASE_URL",
                        "base_url": "https://openai-fixture.invalid/v1",
                    },
                },
                "execution": {
                    "max_workers": 2,
                    "max_retries": 0,
                    "timeout_seconds": 1,
                    "provider_max_concurrency": {"aliyun": 2, "openai": 2},
                },
                "model_roles": {
                    "translation": {
                        "primary": {
                            "model": "obsolete-short-name",
                            "provider_profile": "aliyun",
                            "max_output_tokens": 512,
                        },
                        "reviewer": {
                            "model": "gpt-5.5",
                            "provider_profile": "openai",
                            "max_output_tokens": 1024,
                        },
                    }
                },
            },
        )
        paths["env"].write_text("", encoding="utf-8")
        universe_rows = []
        base_rows = []
        distractor_rows = []
        neutral_rows = []
        splits = ("development", "validation", "sealed")
        for index, split in enumerate(splits, 1):
            base_fact_id = f"pbf_{index}"
            universe_rows.append(
                {
                    "schema_version": "public-benchmark-formal-cohort-item-v1",
                    "cohort_item_id": f"cohort_{index}",
                    "selection_index": index - 1,
                    "source_base_fact_id": base_fact_id,
                    "source_id": f"raw_{index}",
                }
            )
            base_rows.append(
                {
                    "schema_version": "public-benchmark-full-provisional-base-fact-v1",
                    "base_fact_id": base_fact_id,
                    "prompt_en": f"Prompt {index}:",
                    "canonical_fact_en": f"Fact {index} is Answer {index}.",
                    "answer_en": f"Answer {index}",
                    "answer_aliases_en": [f"Alias {index}"],
                    "split_assignment": split,
                }
            )
            distractor_rows.append(
                {
                    "schema_version": "public-benchmark-dual-distractor-candidate-v1",
                    "base_fact_id": base_fact_id,
                    "distractor_id": f"d_{base_fact_id}",
                    "distractor_text_en": f"Distractor {index}",
                    "slot": 1,
                    "split_assignment": split,
                }
            )
            neutral_rows.append(
                {
                    "schema_version": "public-benchmark-neutral-reference-candidate-v1",
                    "base_fact_id": base_fact_id,
                    "neutral_candidate_id": f"n_{base_fact_id}",
                    "neutral_context_candidate_en": f"Neutral {index}",
                    "slot": 1,
                    "split_assignment": split,
                }
            )
        write_jsonl(paths["universe_items"], universe_rows)
        write_jsonl(paths["base"], base_rows)
        write_jsonl(paths["distractors"], distractor_rows)
        write_jsonl(paths["neutrals"], neutral_rows)
        write_json(
            paths["universe_manifest"],
            {
                "schema_version": "public-benchmark-formal-cohort-universe-v2",
                "universe_id": "formal_test_universe",
                "universe_status": "declared_immutable_not_reviewed",
                "record_count": 3,
                "selection_policy": {"mode": "full-pool"},
                "items": {
                    "sha256": file_sha(paths["universe_items"]),
                    "record_count": 3,
                },
            },
        )
        write_json(
            paths["split"],
            {
                "schema_version": "public-benchmark-full-provisional-split-manifest-v1",
                "base_fact_count": 3,
                "actual_base_fact_counts": {
                    "development": 1,
                    "validation": 1,
                    "sealed": 1,
                },
                "formal_split_freeze_performed": False,
            },
        )
        return paths

    def args(self, paths, *extra):
        values = [
            "--config",
            str(paths["config"]),
            "--env-file",
            str(paths["env"]),
            "--formal-universe-manifest",
            str(paths["universe_manifest"]),
            "--formal-universe-items",
            str(paths["universe_items"]),
            "--full-base-facts",
            str(paths["base"]),
            "--distractor-candidates",
            str(paths["distractors"]),
            "--neutral-candidates",
            str(paths["neutrals"]),
            "--split-manifest",
            str(paths["split"]),
            "--output-dir",
            str(paths["output"]),
            "--max-workers",
            "2",
            "--batch-size",
            "3",
            *extra,
        ]
        return module.build_parser().parse_args(values)

    def make_projection_context(self, paths):
        root = paths["config"].parent
        projection_dir = root / "accepted-projection"
        projection_dir.mkdir()
        projected_ids = ["pbf_1", "pbf_2"]
        base_rows = [
            row
            for row in module.read_jsonl(paths["base"])
            if row["base_fact_id"] in projected_ids
        ]
        donor_by_target = {base_fact_id: base_fact_id for base_fact_id in projected_ids}
        distractor_rows = [
            {
                **row,
                "source_base_fact_id": donor_by_target[row["base_fact_id"]],
            }
            for row in module.read_jsonl(paths["distractors"])
            if row["base_fact_id"] in projected_ids
        ]
        neutral_rows = [
            {
                **row,
                "source_base_fact_id": donor_by_target[row["base_fact_id"]],
            }
            for row in module.read_jsonl(paths["neutrals"])
            if row["base_fact_id"] in projected_ids
        ]
        projected_paths = {
            "full_base_facts": projection_dir / "full_base_facts.jsonl",
            "distractor_candidates": projection_dir / "distractor_candidates.jsonl",
            "neutral_reference_candidates": projection_dir
            / "neutral_reference_candidates.jsonl",
            "split_manifest": projection_dir / "split_manifest.json",
        }
        write_jsonl(projected_paths["full_base_facts"], base_rows)
        write_jsonl(projected_paths["distractor_candidates"], distractor_rows)
        write_jsonl(projected_paths["neutral_reference_candidates"], neutral_rows)
        projected_paths["split_manifest"].write_bytes(paths["split"].read_bytes())

        projection_manifest_path = projection_dir / "accepted_projection_manifest.json"
        postreview_path = projection_dir / "postreview_rebuild_manifest.json"
        review_manifest_path = projection_dir / "candidate_review_run_manifest.json"
        adjudications_path = projection_dir / "candidate_semantic_adjudications.jsonl"
        write_json(postreview_path, {"status": "fixture"})
        write_json(review_manifest_path, {"status": "completed"})
        write_jsonl(adjudications_path, [])
        projection_manifest = {
            "schema_version": module.ACCEPTED_PROJECTION_MANIFEST_SCHEMA,
            "status": module.ACCEPTED_PROJECTION_STATUS,
            "projection_id": "accepted_projection_fixture",
            "inputs": {
                "postreview_manifest": {"schema_version": "fixture-postreview-v1"},
                "candidate_review_manifest": {
                    "schema_version": "public-benchmark-full-candidate-review-run-manifest-v1"
                },
                "candidate_review_adjudications": {
                    "schema_version": "public-benchmark-full-candidate-adjudication-v3"
                },
            },
            "selection_contract": {
                "cardinality_per_target": {"distractor": 1, "neutral": 1},
                "source_review_all_accept": False,
                "selected_projection_all_accept": True,
                "human_gold": False,
            },
            "source_review_contract": {
                "review_status": "completed",
                "source_review_all_accept": False,
                "selected_projection_all_accept": True,
                "human_gold": False,
            },
            "counts": {
                "projected_base_facts": 2,
                "selected_candidate_total": 4,
                "quarantined_base_facts": 1,
            },
            "lineage_digests": {
                "ordered_projected_base_fact_ids_sha256": module.sha256_value(
                    projected_ids
                )
            },
        }
        write_json(projection_manifest_path, projection_manifest)
        context = {
            "manifest_path": projection_manifest_path,
            "manifest": projection_manifest,
            "postreview_manifest_path": postreview_path,
            "candidate_review_manifest_path": review_manifest_path,
            "candidate_review_adjudications_path": adjudications_path,
            "source_review_bundle": {"adjudication_by_id": {}},
            "output_paths": projected_paths,
            "full_base_facts": base_rows,
            "distractor_candidates": distractor_rows,
            "neutral_reference_candidates": neutral_rows,
            "split_manifest": module.read_json(projected_paths["split_manifest"]),
            "quarantine": [{"base_fact_id": "pbf_3"}],
            "projected_target_ids": projected_ids,
            "selected_candidate_ids": [
                "d_pbf_1",
                "d_pbf_2",
                "n_pbf_1",
                "n_pbf_2",
            ],
            "selected_adjudications": [],
        }
        tool = mock.Mock()
        tool.load_projection_context.return_value = context
        return projection_manifest_path, context, tool

    def test_end_to_end_records_field_reviews_repair_and_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            manifest = module.execute(self.args(paths), router_factory=FakeRouter)

            self.assertEqual(manifest["counts"]["selected_records"], 3)
            self.assertEqual(manifest["counts"]["proxy_accepted_records"], 2)
            self.assertEqual(manifest["counts"]["repair_attempted_records"], 1)
            self.assertEqual(manifest["counts"]["quarantined_records"], 1)
            self.assertTrue(
                manifest["invalidation_contract"]["provisional_candidate_binding"]
            )
            self.assertTrue(
                manifest["invalidation_contract"][
                    "invalidated_by_recluster_or_candidate_regeneration"
                ]
            )
            self.assertTrue(
                manifest["invalidation_contract"]["invalidated_by_split_change"]
            )
            self.assertEqual(
                manifest["models"]["generator"]["model"],
                "qwen3.8-max-0902/bailian/bailian",
            )
            self.assertEqual(manifest["models"]["reviewer"]["model"], "gpt-5.5")
            self.assertEqual(
                manifest["evidence_boundary"]["hf_model_execution_count"], 0
            )
            self.assertEqual(
                manifest["evidence_boundary"]["behavior_execution_count"], 0
            )
            self.assertFalse(
                manifest["completion_claims"]["other_15_target_languages_complete"]
            )
            self.assertEqual(
                manifest["execution_policy"]["checkpoint_strategy"],
                "single_writer_append_journal_with_atomic_compaction",
            )
            self.assertTrue(
                manifest["execution_policy"]["journal_fsync_after_each_item"]
            )
            self.assertEqual(
                len(
                    manifest["language_scope"][
                        "other_registered_target_language_codes"
                    ]
                ),
                15,
            )

            final_rows = module.read_jsonl(
                paths["output"] / "zh_translation_review_records.jsonl"
            )
            final_by_id = {row["base_fact_id"]: row for row in final_rows}
            self.assertEqual(final_by_id["pbf_1"]["translation_origin"], "initial")
            self.assertEqual(final_by_id["pbf_2"]["translation_origin"], "repair_1")
            self.assertEqual(final_by_id["pbf_3"]["terminal_status"], "quarantined")
            self.assertEqual(
                final_by_id["pbf_3"]["quarantine_reasons"],
                ["initial_generation_failed"],
            )
            checks = final_by_id["pbf_2"]["translation_equivalence_review"][
                "checks"
            ]
            self.assertEqual(
                set(checks),
                {
                    "prompt",
                    "canonical_fact",
                    "answer",
                    "answer_aliases",
                    "distractors",
                    "neutral_candidates",
                },
            )
            self.assertFalse(final_by_id["pbf_2"]["human_gold"])
            quarantine = module.read_jsonl(
                paths["output"] / "translation_quarantine.jsonl"
            )
            self.assertEqual([row["base_fact_id"] for row in quarantine], ["pbf_3"])
            generation_batches = [
                call[1]
                for call in FakeRouter.calls
                if call[0] == "full_zh_translation_generation"
            ]
            self.assertIn(("pbf_1", "pbf_2", "pbf_3"), generation_batches)
            self.assertTrue(final_by_id["pbf_1"]["lineage"])
            initial_rows = module.read_jsonl(
                paths["output"] / "translation_initial.jsonl"
            )
            initial_by_id = {row["base_fact_id"]: row for row in initial_rows}
            self.assertEqual(initial_by_id["pbf_1"]["batch_item_count"], 1)
            self.assertEqual(len(initial_by_id["pbf_1"]["batch_fallback_history"]), 1)
            self.assertGreaterEqual(
                len(initial_by_id["pbf_3"]["batch_fallback_history"]), 2
            )
            temporary_files = list(paths["output"].glob(".*.tmp"))
            self.assertEqual(temporary_files, [])
            self.assertEqual(list(paths["output"].glob("*.journal")), [])

    def test_resume_merges_journal_ignores_partial_tail_and_compacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "stage.jsonl"
            journal_path = output_path.with_suffix(output_path.suffix + ".journal")
            run_fingerprint = "fixture-run-fingerprint"
            items = [
                {
                    "base_fact_id": f"pbf_{index}",
                    "input_record_sha256": f"sha_{index}",
                }
                for index in range(1, 4)
            ]
            write_jsonl(
                output_path,
                [
                    {
                        **items[0],
                        "run_fingerprint": run_fingerprint,
                        "terminal_status": "completed",
                        "parsed_response": {"value": 1},
                    }
                ],
            )
            complete_journal_row = {
                **items[1],
                "run_fingerprint": run_fingerprint,
                "terminal_status": "completed",
                "parsed_response": {"value": 2},
            }
            journal_path.write_bytes(
                json.dumps(complete_journal_row, sort_keys=True).encode("utf-8")
                + b"\n"
                + b'{"base_fact_id":"pbf_3","terminal_status"'
            )
            seen_batches = []

            def batch_worker(batch):
                seen_batches.append([item["base_fact_id"] for item in batch])
                return [
                    {
                        **item,
                        "terminal_status": "completed",
                        "parsed_response": {"value": 3},
                    }
                    for item in batch
                ]

            rows = module.run_checkpoint_stage(
                items=items,
                output_path=output_path,
                stage="test_stage",
                run_fingerprint=run_fingerprint,
                batch_worker=batch_worker,
                max_workers=2,
                batch_size=2,
                checkpoint_every=128,
                resume=True,
            )
            self.assertEqual(seen_batches, [["pbf_3"]])
            self.assertEqual([row["base_fact_id"] for row in rows], ["pbf_1", "pbf_2", "pbf_3"])
            self.assertEqual(
                [row["base_fact_id"] for row in module.read_jsonl(output_path)],
                ["pbf_1", "pbf_2", "pbf_3"],
            )
            self.assertFalse(journal_path.exists())

    def test_stage_rows_bind_run_fingerprint_and_reject_mixed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            manifest = module.execute(
                self.args(paths, "--limit", "1"), router_factory=FakeRouter
            )
            stage_paths = (
                paths["output"] / "translation_initial.jsonl",
                paths["output"] / "translation_initial_reviews.jsonl",
                paths["output"] / "translation_repairs.jsonl",
                paths["output"] / "translation_repair_reviews.jsonl",
            )
            for path in stage_paths:
                for row in module.read_jsonl(path):
                    self.assertEqual(row["run_fingerprint"], manifest["run_fingerprint"])

            initial_path = stage_paths[0]
            initial_rows = module.read_jsonl(initial_path)
            initial_rows[0]["run_fingerprint"] = "0" * 64
            write_jsonl(initial_path, initial_rows)
            with self.assertRaisesRegex(ValueError, "checkpoint run fingerprint mismatch"):
                module.execute(
                    self.args(paths, "--limit", "1", "--resume"),
                    router_factory=NoCallRouter,
                )

    def test_resume_requires_manifest_when_managed_outputs_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            module.execute(
                self.args(paths, "--limit", "1"), router_factory=FakeRouter
            )
            (paths["output"] / "run_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "requires an existing run manifest"):
                module.execute(
                    self.args(paths, "--limit", "1", "--resume"),
                    router_factory=NoCallRouter,
                )

    def test_resume_skips_all_terminal_stage_records(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            module.execute(self.args(paths), router_factory=FakeRouter)
            calls_before = len(FakeRouter.calls)
            resumed = module.execute(
                self.args(paths, "--resume"), router_factory=NoCallRouter
            )
            self.assertEqual(len(FakeRouter.calls), calls_before)
            self.assertTrue(resumed["invocation_resumed"])
            self.assertEqual(resumed["counts"]["proxy_accepted_records"], 2)

    def test_retry_failed_reprocesses_only_failed_rows_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            first = module.execute(self.args(paths), router_factory=FakeRouter)
            self.assertEqual(first["counts"]["quarantined_records"], 1)
            initial_path = paths["output"] / "translation_initial.jsonl"
            before = {
                row["base_fact_id"]: row for row in module.read_jsonl(initial_path)
            }
            self.assertEqual(before["pbf_3"]["terminal_status"], "failed")

            FakeRouter.calls = []
            resumed = module.execute(
                self.args(paths, "--resume", "--retry-failed"),
                router_factory=RecoveryRouter,
            )
            after = {
                row["base_fact_id"]: row for row in module.read_jsonl(initial_path)
            }
            self.assertEqual(after["pbf_1"], before["pbf_1"])
            self.assertEqual(after["pbf_2"], before["pbf_2"])
            self.assertEqual(after["pbf_3"]["terminal_status"], "completed")
            self.assertEqual(len(after["pbf_3"]["retry_history"]), 1)
            self.assertEqual(
                after["pbf_3"]["retry_history"][0]["failed_record_sha256"],
                module.sha256_value(before["pbf_3"]),
            )
            generation_batches = [
                call[1]
                for call in FakeRouter.calls
                if call[0] == "full_zh_translation_generation"
            ]
            self.assertEqual(generation_batches, [("pbf_3",)])
            self.assertEqual(resumed["status"], "completed_full_zh_proxy_review")
            self.assertEqual(resumed["counts"]["proxy_accepted_records"], 3)
            self.assertEqual(resumed["counts"]["quarantined_records"], 0)
            self.assertTrue(resumed["retry_failed_enabled"])

    def test_retry_failed_requires_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            with self.assertRaisesRegex(ValueError, "retry_failed requires resume"):
                module.execute(
                    self.args(paths, "--retry-failed"),
                    router_factory=NoCallRouter,
                )

    def test_resume_rejects_candidate_or_split_binding_change(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            module.execute(self.args(paths), router_factory=FakeRouter)
            with paths["distractors"].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "run fingerprint mismatch"):
                module.execute(
                    self.args(paths, "--resume"), router_factory=NoCallRouter
                )

    def test_candidate_repair_successor_subset_and_bindings_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            universe_rows = module.read_jsonl(paths["universe_items"])
            universe_ids = [row["source_base_fact_id"] for row in universe_rows]
            candidate_excluded_ids = [universe_ids[-1]]
            base_rows = module.read_jsonl(paths["base"])[:-1]
            distractor_rows = module.read_jsonl(paths["distractors"])[:-1]
            neutral_rows = module.read_jsonl(paths["neutrals"])[:-1]
            write_jsonl(paths["base"], base_rows)
            write_jsonl(paths["distractors"], distractor_rows)
            write_jsonl(paths["neutrals"], neutral_rows)
            split_manifest = module.read_json(paths["split"])
            split_manifest["base_fact_count"] = len(base_rows)
            split_manifest["actual_base_fact_counts"] = {
                "development": 1,
                "validation": 1,
                "sealed": 0,
            }
            write_json(paths["split"], split_manifest)

            fact_resolution_path = Path(directory) / "fact-resolution.json"
            predecessor_path = Path(directory) / "predecessor-postreview.json"
            root_path = Path(directory) / "root-postreview.json"
            candidate_resolution_path = Path(directory) / "candidate-resolution.json"
            pair_path = Path(directory) / "candidate-pairs.jsonl"
            cohort_exclusion_path = Path(directory) / "candidate-exclusions.jsonl"
            postreview_path = Path(directory) / "successor-postreview.json"
            for path in (
                fact_resolution_path,
                predecessor_path,
                root_path,
                candidate_resolution_path,
            ):
                write_json(path, {"fixture": path.stem})
            pair_rows = []
            cohort_exclusion_rows = [
                {
                    "base_fact_id": candidate_excluded_ids[0],
                    "disposition": "cohort_exclude",
                }
            ]
            write_jsonl(pair_path, pair_rows)
            write_jsonl(cohort_exclusion_path, cohort_exclusion_rows)

            fact_resolution_bindings = {
                "fact_resolution_manifest": module.artifact_binding(
                    fact_resolution_path
                )
            }
            resolution_context = {
                "source_ids": universe_ids,
                "retained_ids": universe_ids,
                "excluded_ids": [],
                "revised_ids": [],
                "successor_id": "fact_successor_fixture",
                "bindings": fact_resolution_bindings,
            }
            candidate_context = {
                "manifest_path": candidate_resolution_path.resolve(),
                "manifest_binding": module.artifact_binding(
                    candidate_resolution_path,
                    schema_version=(
                        "public-benchmark-full-candidate-resolution-manifest-v2"
                    ),
                ),
                "predecessor_path": predecessor_path.resolve(),
                "root_path": root_path.resolve(),
                "pair_path": pair_path.resolve(),
                "pair_rows": pair_rows,
                "cohort_exclusion_path": cohort_exclusion_path.resolve(),
                "cohort_exclusion_rows": cohort_exclusion_rows,
                "candidate_excluded_ids": candidate_excluded_ids,
                "resolution_id": "candidate_resolution_fixture",
            }
            candidate_tool = mock.Mock()
            candidate_tool.RESOLUTION_MANIFEST_SCHEMA = (
                "public-benchmark-full-candidate-resolution-manifest-v2"
            )
            candidate_tool.PAIR_EXCLUSION_SCHEMA = (
                "public-benchmark-full-candidate-pair-exclusion-v1"
            )
            candidate_tool.COHORT_EXCLUSION_SCHEMA = (
                "public-benchmark-full-candidate-cohort-exclusion-v1"
            )
            candidate_tool.load_resolution_context.return_value = candidate_context
            semantic_materializer = mock.Mock()
            semantic_materializer.load_fact_resolution_context.return_value = (
                resolution_context
            )

            candidate_repair = {
                "root_postreview_manifest": module.artifact_binding(
                    root_path,
                    schema_version=module.POSTREVIEW_MANIFEST_SCHEMA,
                ),
                "predecessor_postreview_manifest": module.artifact_binding(
                    predecessor_path,
                    schema_version=module.POSTREVIEW_MANIFEST_SCHEMA,
                ),
                "candidate_resolution_manifest": module.artifact_binding(
                    candidate_resolution_path,
                    schema_version=candidate_tool.RESOLUTION_MANIFEST_SCHEMA,
                ),
                "candidate_pair_exclusions": module.artifact_binding(
                    pair_path,
                    rows=pair_rows,
                    schema_version=candidate_tool.PAIR_EXCLUSION_SCHEMA,
                ),
                "candidate_cohort_exclusions": module.artifact_binding(
                    cohort_exclusion_path,
                    rows=cohort_exclusion_rows,
                    schema_version=candidate_tool.COHORT_EXCLUSION_SCHEMA,
                ),
            }
            retained_ids = [row["base_fact_id"] for row in base_rows]
            postreview = {
                "schema_version": module.POSTREVIEW_SUCCESSOR_MANIFEST_SCHEMA,
                "status": module.POSTREVIEW_STATUS,
                "fact_review_contract": {
                    "mode": "fact_resolution_successor_cohort",
                    "selection_is_complete_retained_cohort": True,
                    "source_universe_base_fact_count": len(universe_ids),
                },
                "inputs": {
                    "fact_resolution": fact_resolution_bindings,
                    "candidate_repair": candidate_repair,
                },
                "outputs": {
                    "full_base_facts": module.artifact_binding(
                        paths["base"], rows=base_rows
                    ),
                    "distractor_candidates": module.artifact_binding(
                        paths["distractors"], rows=distractor_rows
                    ),
                    "neutral_reference_candidates": module.artifact_binding(
                        paths["neutrals"], rows=neutral_rows
                    ),
                    "split_manifest": module.artifact_binding(
                        paths["split"],
                        schema_version=split_manifest["schema_version"],
                    ),
                },
                "candidate_repair_contract": {
                    "resolution_id": candidate_context["resolution_id"],
                    "resolution_applied": True,
                    "pair_exclusion_count": 0,
                    "candidate_stage_cohort_exclusion_count": 1,
                    "fixed_point_reached": True,
                    "candidate_shortfall_count": 0,
                    "candidate_policy_relaxed": False,
                    "predecessor_candidate_reviews_blanket_valid": False,
                    "exact_v3_accepted_projection_evidence_carry_forward_eligible": True,
                    "candidate_id_or_row_sha_alone_sufficient_for_review_reuse": False,
                    "full_candidate_evidence_coverage_required": True,
                    "full_candidate_model_rereview_required": False,
                    "changed_or_new_candidate_fresh_model_review_required": True,
                    "predecessor_zh_reviews_valid": False,
                    "full_zh_translation_and_review_required": True,
                    "human_gold": False,
                },
                "lineage_digests": {
                    "ordered_input_base_fact_ids_sha256": module.sha256_value(
                        universe_ids
                    ),
                    "ordered_fact_stage_retained_base_fact_ids_sha256": module.sha256_value(
                        universe_ids
                    ),
                    "ordered_retained_base_fact_ids_sha256": module.sha256_value(
                        retained_ids
                    ),
                    "ordered_fact_stage_excluded_base_fact_ids_sha256": module.sha256_value(
                        []
                    ),
                    "ordered_candidate_stage_excluded_base_fact_ids_sha256": module.sha256_value(
                        candidate_excluded_ids
                    ),
                    "ordered_rejected_base_fact_ids_sha256": module.sha256_value(
                        candidate_excluded_ids
                    ),
                    "ordered_revised_base_fact_ids_sha256": module.sha256_value([]),
                },
                "review_invalidation": {
                    "pre_resolution_translation_reviews_valid": False,
                    "pre_postreview_split_translation_reviews_valid": False,
                    "predecessor_candidate_reviews_blanket_valid": False,
                    "exact_v3_accepted_projection_evidence_carry_forward_eligible": True,
                    "predecessor_translation_reviews_valid": False,
                    "candidate_id_or_row_sha_alone_sufficient_for_review_reuse": False,
                    "full_candidate_evidence_coverage_required": True,
                    "full_candidate_model_rereview_required": False,
                    "changed_or_new_candidate_fresh_model_review_required": True,
                    "new_distractor_review_required": True,
                    "new_neutral_review_required": True,
                    "new_translation_and_translation_review_required": True,
                },
            }
            write_json(postreview_path, postreview)

            kwargs = {
                "formal_manifest": module.read_json(paths["universe_manifest"]),
                "universe_ids": universe_ids,
                "base_rows": base_rows,
                "distractor_rows": distractor_rows,
                "neutral_rows": neutral_rows,
                "split_manifest": split_manifest,
                "paths": {
                    "full_base_facts": paths["base"],
                    "distractor_candidates": paths["distractors"],
                    "neutral_candidates": paths["neutrals"],
                    "split_manifest": paths["split"],
                },
                "fact_resolution_manifest_path": fact_resolution_path,
                "postreview_rebuild_manifest_path": postreview_path,
            }
            with mock.patch.object(
                module,
                "_load_semantic_materializer",
                return_value=semantic_materializer,
            ), mock.patch.object(
                module,
                "_load_candidate_resolution_tool",
                return_value=candidate_tool,
            ):
                successor = module._validate_successor_chain(**kwargs)
                self.assertEqual(successor["retained_record_count"], 2)
                self.assertEqual(
                    successor["candidate_resolution_excluded_record_count"], 1
                )
                self.assertEqual(
                    successor["ordered_retained_base_fact_ids_sha256"],
                    module.sha256_value(retained_ids),
                )

                tampered = json.loads(json.dumps(postreview))
                tampered["inputs"]["candidate_repair"][
                    "predecessor_postreview_manifest"
                ]["sha256"] = "0" * 64
                write_json(postreview_path, tampered)
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    module._validate_successor_chain(**kwargs)

                tampered = json.loads(json.dumps(postreview))
                tampered["review_invalidation"][
                    "predecessor_translation_reviews_valid"
                ] = True
                write_json(postreview_path, tampered)
                with self.assertRaisesRegex(ValueError, "invalidation contract"):
                    module._validate_successor_chain(**kwargs)

                tampered = json.loads(json.dumps(postreview))
                tampered["candidate_repair_contract"][
                    "full_candidate_model_rereview_required"
                ] = True
                write_json(postreview_path, tampered)
                with self.assertRaisesRegex(ValueError, "candidate-repair contract"):
                    module._validate_successor_chain(**kwargs)

                tampered = json.loads(json.dumps(postreview))
                tampered["schema_version"] = module.POSTREVIEW_MANIFEST_SCHEMA
                write_json(postreview_path, tampered)
                with self.assertRaisesRegex(ValueError, "successor manifest schema v3"):
                    module._validate_successor_chain(**kwargs)

    def test_limit_is_deterministic_and_does_not_claim_full_zh_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            manifest = module.execute(
                self.args(paths, "--limit", "1"), router_factory=FakeRouter
            )
            self.assertEqual(manifest["counts"]["selected_records"], 1)
            self.assertFalse(manifest["selection_is_full_universe"])
            self.assertFalse(
                manifest["completion_claims"]["full_pool_zh_translation_review_complete"]
            )
            rows = module.read_jsonl(
                paths["output"] / "zh_translation_review_records.jsonl"
            )
            self.assertEqual([row["base_fact_id"] for row in rows], ["pbf_1"])

    def test_accepted_projection_processes_only_projected_one_plus_one_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            projection_path, context, projection_tool = self.make_projection_context(
                paths
            )
            args = self.args(
                paths,
                "--accepted-candidate-projection-manifest",
                str(projection_path),
            )
            args.full_base_facts = None
            args.distractor_candidates = None
            args.neutral_candidates = None
            args.split_manifest = None
            with mock.patch.object(
                module,
                "_load_accepted_projection_tool",
                return_value=projection_tool,
            ), mock.patch.object(
                module,
                "_validate_accepted_projection",
                return_value={},
            ):
                manifest = module.execute(args, router_factory=FakeRouter)

            projection_tool.load_projection_context.assert_called_once_with(
                projection_manifest_path=projection_path.resolve()
            )
            self.assertEqual(
                manifest["status"],
                "completed_accepted_candidate_projection_zh_proxy_review",
            )
            self.assertEqual(manifest["counts"]["universe_records"], 3)
            self.assertEqual(manifest["counts"]["selected_records"], 2)
            self.assertFalse(manifest["selection_is_full_universe"])
            self.assertTrue(
                manifest["selection_is_complete_accepted_projection"]
            )
            self.assertFalse(
                manifest["completion_claims"]
                ["full_pool_zh_translation_review_complete"]
            )
            self.assertTrue(
                manifest["completion_claims"]
                ["accepted_candidate_projection_zh_translation_review_complete"]
            )
            self.assertEqual(module._manifest_exit_code(manifest), 0)
            rows = module.read_jsonl(
                paths["output"] / "zh_translation_review_records.jsonl"
            )
            self.assertEqual(
                [row["base_fact_id"] for row in rows],
                context["projected_target_ids"],
            )
            self.assertTrue(
                all(
                    len(row["source_distractors"]) == 1
                    and len(row["source_neutral_candidates"]) == 1
                    for row in rows
                )
            )

    def test_accepted_projection_rejects_limit_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            projection_path = Path(directory) / "projection.json"
            write_json(projection_path, {})
            with self.assertRaisesRegex(ValueError, "limit is not allowed"):
                module.execute(
                    self.args(
                        paths,
                        "--accepted-candidate-projection-manifest",
                        str(projection_path),
                        "--limit",
                        "1",
                    ),
                    router_factory=NoCallRouter,
                )

    def test_accepted_projection_requires_exact_projected_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            projection_path, _, projection_tool = self.make_projection_context(paths)
            with mock.patch.object(
                module,
                "_load_accepted_projection_tool",
                return_value=projection_tool,
            ), mock.patch.object(
                module,
                "_validate_accepted_projection",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    ValueError, "full_base_facts path differs from supplied input"
                ):
                    module.load_bound_inputs(
                        formal_universe_manifest_path=paths["universe_manifest"],
                        formal_universe_items_path=paths["universe_items"],
                        full_base_facts_path=paths["base"],
                        distractor_candidates_path=None,
                        neutral_candidates_path=None,
                        split_manifest_path=None,
                        accepted_candidate_projection_manifest_path=projection_path,
                    )

    def test_review_validation_requires_consistent_field_level_verdicts(self):
        item = {
            "source": {"answer_aliases_en": ["Alias"]},
            "distractors": [{"distractor_id": "d1"}],
            "neutral_candidates": [{"neutral_candidate_id": "n1"}],
        }
        review = FakeRouter.review("ignored", accepted=True)
        review["checks"]["distractors"][0]["distractor_id"] = "d1"
        review["checks"]["neutral_candidates"][0]["neutral_candidate_id"] = "n1"
        module.validate_review(review, item)
        review["checks"]["prompt"]["translation_equivalent"] = False
        with self.assertRaisesRegex(
            ValueError, "review_decision_inconsistent_with_field_checks"
        ):
            module.validate_review(review, item)

    def test_response_model_mismatch_fails_whole_batch_without_recursive_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            manifest = module.execute(
                self.args(paths, "--limit", "2"),
                router_factory=IdentityMismatchRouter,
            )
            generation_calls = [
                call
                for call in FakeRouter.calls
                if call[0] == "full_zh_translation_generation"
            ]
            self.assertEqual(len(generation_calls), 1)
            self.assertEqual(manifest["counts"]["quarantined_records"], 2)
            rows = module.read_jsonl(paths["output"] / "translation_initial.jsonl")
            self.assertTrue(all(row["terminal_status"] == "failed" for row in rows))
            self.assertTrue(
                all(
                    row["response_model_identity_status"] == "mismatch"
                    for row in rows
                )
            )

    def test_missing_response_model_fails_whole_batch_without_recursive_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            manifest = module.execute(
                self.args(paths, "--limit", "2"),
                router_factory=IdentityMissingRouter,
            )
            generation_calls = [
                call
                for call in FakeRouter.calls
                if call[0] == "full_zh_translation_generation"
            ]
            self.assertEqual(len(generation_calls), 1)
            self.assertEqual(manifest["counts"]["quarantined_records"], 2)
            rows = module.read_jsonl(paths["output"] / "translation_initial.jsonl")
            self.assertTrue(all(row["terminal_status"] == "failed" for row in rows))
            self.assertTrue(
                all(
                    row["response_model_identity_status"] == "not_observed"
                    and row["response_model_identity_error_type"]
                    == "ResponseModelMissing"
                    for row in rows
                )
            )

    def test_resume_revalidates_checkpoint_payload_and_upstream_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            module.execute(
                self.args(paths, "--limit", "1"), router_factory=FakeRouter
            )
            initial_path = paths["output"] / "translation_initial.jsonl"
            initial = module.read_jsonl(initial_path)
            initial[0]["parsed_response"]["prompt_zh"] = "仍为合法格式但内容已被改写："
            write_jsonl(initial_path, initial)
            with self.assertRaisesRegex(
                ValueError, "translation_record_sha256 mismatch"
            ):
                module.execute(
                    self.args(paths, "--limit", "1", "--resume"),
                    router_factory=NoCallRouter,
                )

            initial[0]["parsed_response"].pop("answer_zh")
            write_jsonl(initial_path, initial)
            with self.assertRaisesRegex(ValueError, "missing_answer_zh"):
                module.execute(
                    self.args(paths, "--limit", "1", "--resume"),
                    router_factory=NoCallRouter,
                )

    def test_model_specs_drop_secret_like_fields_and_force_json_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            config = module.load_config(paths["config"])
            config["model_roles"]["translation"]["primary"]["api_key"] = "secret"
            config["model_roles"]["translation"]["reviewer"]["base_url"] = (
                "https://secret.invalid"
            )
            specs = module._model_specs(config, self.args(paths))
            self.assertNotIn("api_key", specs["generator"])
            self.assertNotIn("base_url", specs["reviewer"])
            self.assertTrue(specs["generator"]["json_mode"])
            self.assertTrue(specs["reviewer"]["json_mode"])

    def test_expected_response_model_cannot_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            with self.assertRaisesRegex(
                ValueError, "generator expected response model must be non-empty"
            ):
                module.execute(
                    self.args(
                        paths,
                        "--generator-expected-response-model",
                        "",
                    ),
                    router_factory=NoCallRouter,
                )

    def test_expected_response_model_override_requires_config_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            args = self.args(
                paths,
                "--generator-expected-response-model",
                "unfrozen-alias",
            )
            config = module.load_config(paths["config"])
            with self.assertRaisesRegex(ValueError, "frozen config identity"):
                module._model_specs(config, args)

            config["model_roles"]["translation"]["primary"][
                "expected_response_model"
            ] = "explicit-new-version"
            args.generator_expected_response_model = "explicit-new-version"
            specs = module._model_specs(config, args)
            self.assertEqual(specs["generator"]["model"], module.DEFAULT_GENERATOR_ROUTE)

    def test_route_fingerprint_is_redacted_and_env_drift_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            paths["env"].write_text(
                "ALIYUN_BASE_URL=https://zh-generator-one.invalid/v1\n"
                "OPENAI_BASE_URL=https://zh-reviewer-one.invalid/v1\n",
                encoding="utf-8",
            )
            manifest = module.execute(
                self.args(paths, "--limit", "1"), router_factory=FakeRouter
            )
            serialized = json.dumps(manifest["route_identity"], sort_keys=True)
            self.assertNotIn("zh-generator-one", serialized)
            self.assertNotIn("zh-reviewer-one", serialized)
            self.assertFalse(
                manifest["route_identity"]["credentials_or_endpoints_included"]
            )
            self.assertEqual(len(manifest["source_fingerprints"]), 3)
            self.assertTrue(
                manifest["review_independence"]["model_identity_distinct"]
            )
            self.assertTrue(
                manifest["review_independence"]["provider_endpoint_distinct"]
            )
            self.assertFalse(
                manifest["review_independence"][
                    "provider_infrastructure_independence_claimed"
                ]
            )

            paths["env"].write_text(
                "ALIYUN_BASE_URL=https://zh-generator-two.invalid/v1\n"
                "OPENAI_BASE_URL=https://zh-reviewer-one.invalid/v1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "run fingerprint mismatch"):
                module.execute(
                    self.args(paths, "--limit", "1", "--resume"),
                    router_factory=NoCallRouter,
                )

    def test_shared_gateway_is_recorded_without_infrastructure_independence_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            config = module.read_json(paths["config"])
            shared = "https://shared-gateway-fixture.invalid/v1"
            config["provider_profiles"]["aliyun"]["base_url"] = shared
            config["provider_profiles"]["openai"]["base_url"] = shared
            write_json(paths["config"], config)
            manifest = module.execute(
                self.args(paths, "--limit", "1"), router_factory=FakeRouter
            )
            independence = manifest["review_independence"]
            self.assertTrue(independence["model_identity_distinct"])
            self.assertFalse(independence["provider_endpoint_distinct"])
            self.assertFalse(
                independence["provider_infrastructure_independence_claimed"]
            )
            self.assertEqual(independence["claim_scope"], "distinct_model_identity_only")

    def test_request_guard_detects_mutated_effective_route_before_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            env_path.write_text(
                "TEST_API_KEY=fixture-key\n"
                "TEST_BASE_URL=https://route-a.example/v1\n",
                encoding="utf-8",
            )
            config = {
                "execution": {
                    "timeout_seconds": 1,
                    "max_retries": 0,
                    "max_workers": 1,
                },
                "provider_profiles": {
                    "openai": {
                        "protocol": "openai_compatible",
                        "api_key_env": "TEST_API_KEY",
                        "base_url_env": "TEST_BASE_URL",
                    }
                },
            }
            spec = {"provider_profile": "openai", "model": "gpt-5.5"}
            router = module.runtime.ModelRouter(
                config, env_path, root / "events.jsonl"
            )
            identity = module.build_route_identity_set(
                config=config,
                env_path=env_path,
                routes=[(spec, "gpt-5.5-2026-04-24")],
                resolved_env=router.values,
            )
            module.attach_route_identity_guard(
                router,
                config=config,
                env_path=env_path,
                identity_set=identity,
            )
            router.values["TEST_BASE_URL"] = "https://route-b.example/v1"

            with mock.patch.object(router, "_one_call") as one_call:
                result = router.request_json(
                    "test_stage",
                    "test_item",
                    spec,
                    "{}",
                    lambda _value: None,
                )

            self.assertEqual(result.terminal_status, "failed")
            self.assertEqual(result.attempt_count, 1)
            self.assertEqual(
                result.attempts[0]["error_type"], "StaticProxyIdentityError"
            )
            one_call.assert_not_called()

    def test_request_guard_uses_router_config_not_caller_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            env_path.write_text("TEST_API_KEY=fixture-key\n", encoding="utf-8")
            frozen_config = {
                "execution": {
                    "timeout_seconds": 1,
                    "max_retries": 0,
                    "max_workers": 1,
                },
                "provider_profiles": {
                    "openai": {
                        "protocol": "openai_compatible",
                        "api_key_env": "TEST_API_KEY",
                        "base_url": "https://route-a.example/v1",
                    }
                },
            }
            live_config = json.loads(json.dumps(frozen_config))
            live_config["provider_profiles"]["openai"]["base_url"] = (
                "https://route-b.example/v1"
            )
            spec = {"provider_profile": "openai", "model": "gpt-5.5"}
            router = module.runtime.ModelRouter(
                live_config, env_path, root / "events.jsonl"
            )
            identity = module.build_route_identity_set(
                config=frozen_config,
                env_path=env_path,
                routes=[(spec, "gpt-5.5-2026-04-24")],
            )
            module.attach_route_identity_guard(
                router,
                config=frozen_config,
                env_path=env_path,
                identity_set=identity,
            )

            with mock.patch.object(router, "_one_call") as one_call:
                result = router.request_json(
                    "test_stage",
                    "test_item",
                    spec,
                    "{}",
                    lambda _value: None,
                )

            self.assertEqual(result.terminal_status, "failed")
            self.assertEqual(
                result.attempts[0]["error_type"], "StaticProxyIdentityError"
            )
            one_call.assert_not_called()

    def test_route_identity_rejects_duplicate_keys_and_noncanonical_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"provider_profile": "openai", "model": "gpt-5.5"}
            config = {
                "provider_profiles": {
                    "openai": {
                        "protocol": "openai_compatible",
                        "base_url": "https://route.example/v1",
                    }
                }
            }
            with self.assertRaisesRegex(ValueError, "duplicate model route"):
                module.build_route_identity_set(
                    config=config,
                    env_path=root / ".env",
                    routes=[(spec, "expected"), (spec, "expected")],
                    resolved_env={},
                )

            config["provider_profiles"]["openai"]["protocol"] = "anthropic "
            with self.assertRaisesRegex(ValueError, "unsupported model route protocol"):
                module.build_route_identity_set(
                    config=config,
                    env_path=root / ".env",
                    routes=[(spec, "expected")],
                    resolved_env={},
                )
            router_config = {
                **config,
                "execution": {
                    "timeout_seconds": 1,
                    "max_retries": 0,
                    "max_workers": 1,
                },
            }
            router = module.runtime.ModelRouter(
                router_config, root / ".env", root / "events.jsonl"
            )
            with self.assertRaisesRegex(RuntimeError, "unsupported_provider_protocol"):
                router._profile(spec)

    def test_strict_json_validation_rejects_extra_fields(self):
        item = {
            "base_fact_id": "pbf_1",
            "source": {"answer_aliases_en": ["Alias"]},
            "distractors": [{"distractor_id": "d_pbf_1"}],
            "neutral_candidates": [{"neutral_candidate_id": "n_pbf_1"}],
        }
        translation = FakeRouter.translation("pbf_1")
        translation["unexpected"] = "must fail"
        with self.assertRaisesRegex(ValueError, "invalid_translation_fields"):
            module.validate_translation(translation, item)
        review = FakeRouter.review("pbf_1", accepted=True)
        review["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "invalid_review_fields"):
            module.validate_review(review, item)

    def test_non_full_or_quarantined_manifest_has_nonzero_exit(self):
        complete = {
            "status": "completed_full_zh_proxy_review",
            "completion_claims": {"full_pool_zh_translation_review_complete": True},
            "counts": {"quarantined_records": 0},
        }
        self.assertEqual(module._manifest_exit_code(complete), 0)
        self.assertEqual(
            module._manifest_exit_code(
                {
                    **complete,
                    "status": "completed_with_quarantine_or_limited_scope",
                }
            ),
            2,
        )
        self.assertEqual(
            module._manifest_exit_code(
                {**complete, "counts": {"quarantined_records": 1}}
            ),
            2,
        )

    def test_cli_defaults_use_verified_generator_route(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.make_fixture(directory)
            args = self.args(paths)
            self.assertEqual(args.generator_model, module.DEFAULT_GENERATOR_ROUTE)
            self.assertEqual(args.generator_expected_response_model, "qwen3.8-max")
            self.assertEqual(args.reviewer_model, "gpt-5.5")
            self.assertEqual(
                args.reviewer_expected_response_model, "gpt-5.5-2026-04-24"
            )


if __name__ == "__main__":
    unittest.main()
