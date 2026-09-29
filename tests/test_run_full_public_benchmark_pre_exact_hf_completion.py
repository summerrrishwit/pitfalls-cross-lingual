import argparse
import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "run_full_public_benchmark_pre_exact_hf_completion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_full_public_benchmark_pre_exact_hf_completion", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
completion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(completion)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def binding(path: Path):
    return {
        "path": str(path.resolve()),
        "sha256": completion.sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def jsonl_binding(path: Path, schema_version: str):
    value = binding(path)
    value.update(
        {
            "record_count": sum(
                bool(line.strip())
                for line in path.read_text(encoding="utf-8").splitlines()
            ),
            "schema_version": schema_version,
        }
    )
    return value


class FakeCandidateModule:
    @staticmethod
    def source_fingerprints():
        return {"runner_sha256": "f" * 64}


class FakeValidators:
    candidate = FakeCandidateModule()

    def __init__(self, postreview=None, completed_reviews=None):
        self.postreview = postreview or {
            "distractors": [{}, {}],
            "neutrals": [{}, {}],
        }
        self.completed_reviews = {
            Path(path).resolve(): value
            for path, value in (completed_reviews or {}).items()
        }

    def load_postreview(self, path):
        return copy.deepcopy(self.postreview)

    def load_complete_candidate_review(self, *, predecessor_path, manifest_path):
        key = Path(manifest_path).resolve()
        if key not in self.completed_reviews:
            raise AssertionError(f"unexpected completed review load: {key}")
        return copy.deepcopy(self.completed_reviews[key])


class CompletionOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.universe = self.root / "full-8969-formal-universe-v1"
        self.postreview = (
            self.universe
            / "postreview-successor-r02"
            / "postreview_rebuild_manifest.json"
        )
        self.candidate = (
            self.universe
            / "candidate-review-r02"
            / "candidate_review_run_manifest.json"
        )
        self.config = self.root / "config.json"
        for path in (self.postreview, self.candidate, self.config):
            write_json(path, {})

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, **overrides):
        values = {
            "repo_root": PROJECT_ROOT,
            "universe_root": self.universe,
            "start_round": 2,
            "start_postreview_manifest": self.postreview,
            "start_candidate_review_manifest": self.candidate,
            "config": self.config,
            "state_dir": self.root / "state",
            "wait_for_external_writer": False,
            "acknowledge_interrupted_review": False,
            "external_poll_seconds": 1.0,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def orchestrator(self, *, validators=None, **overrides):
        return completion.CompletionOrchestrator(
            self.args(**overrides),
            validators=validators or FakeValidators(),
            now_fn=lambda: "2026-09-26T00:00:00+00:00",
            sleep_fn=lambda _: None,
        )

    def candidate_manifest(
        self,
        orchestrator,
        *,
        status="running",
        terminal_status_counts=None,
        overall_decision_counts=None,
    ):
        contract = {
            "tool_version": completion.CURRENT_CANDIDATE_TOOL_VERSION,
            "candidate_manifest": {
                "path": str(self.postreview.resolve()),
                "sha256": completion.sha256_file(self.postreview),
            },
            "selected_candidate_count": 4,
            "full_candidate_count": 4,
            "selection_is_full_candidate_set": True,
            "batch_size": 8,
            "checkpoint_compact_every": 128,
            "max_workers": 2,
            "expected_response_model": "gpt-5.5-2026-04-24",
            "config_sha256": completion.sha256_value(orchestrator.config),
            "source_fingerprints": FakeCandidateModule.source_fingerprints(),
            "reviewer_spec": {
                "provider_profile": "openai",
                "model": "gpt-5.5",
                "reasoning_effort": "low",
                "max_output_tokens": 16384,
                "max_retries": 2,
                "temperature": 0.0,
                "json_mode": True,
            },
        }
        manifest = {
            "schema_version": completion.CANDIDATE_MANIFEST_SCHEMA,
            "tool_version": completion.CANDIDATE_TOOL_VERSION,
            "status": status,
            "run_contract": contract,
            "run_contract_sha256": completion.sha256_value(contract),
        }
        if terminal_status_counts is not None:
            manifest["terminal_status_counts"] = terminal_status_counts
        if overall_decision_counts is not None:
            manifest["overall_decision_counts"] = overall_decision_counts
        write_json(self.candidate, manifest)
        return manifest

    def historical_v2_fixture(self):
        artifact_dir = self.postreview.parent
        full_path = artifact_dir / "full_base_facts.jsonl"
        distractor_path = artifact_dir / "distractor_candidates.jsonl"
        neutral_path = artifact_dir / "neutral_reference_candidates.jsonl"
        facts = [
            {
                "schema_version": "public-benchmark-full-provisional-base-fact-v1",
                "base_fact_id": base_fact_id,
                "canonical_fact_en": f"fact {base_fact_id}",
                "answer_en": base_fact_id,
                "answer_aliases_en": [base_fact_id],
                "human_gold": False,
                "hf_model_execution_status": "not_run",
                "hf_tokenizer_execution_status": "not_run",
            }
            for base_fact_id in ("fact-target", "fact-source-a", "fact-source-b")
        ]
        distractors = [
            {
                "schema_version": "public-benchmark-dual-distractor-candidate-v1",
                "distractor_id": candidate_id,
                "base_fact_id": "fact-target",
                "source_base_fact_id": source_id,
                "slot": slot,
            }
            for candidate_id, source_id, slot in (
                ("distractor-1", "fact-source-a", 1),
                ("distractor-2", "fact-source-b", 2),
            )
        ]
        neutrals = [
            {
                "schema_version": "public-benchmark-neutral-reference-candidate-v1",
                "neutral_candidate_id": candidate_id,
                "base_fact_id": "fact-target",
                "source_base_fact_id": source_id,
                "slot": slot,
            }
            for candidate_id, source_id, slot in (
                ("neutral-1", "fact-source-a", 1),
                ("neutral-2", "fact-source-b", 2),
            )
        ]
        write_jsonl(full_path, facts)
        write_jsonl(distractor_path, distractors)
        write_jsonl(neutral_path, neutrals)
        artifacts = {
            "full_base_facts": {
                "path": full_path,
                "value": facts,
                "binding": jsonl_binding(
                    full_path,
                    "public-benchmark-full-provisional-base-fact-v1",
                ),
            },
            "distractor_candidates": {
                "path": distractor_path,
                "value": distractors,
                "binding": jsonl_binding(
                    distractor_path,
                    "public-benchmark-dual-distractor-candidate-v1",
                ),
            },
            "neutral_reference_candidates": {
                "path": neutral_path,
                "value": neutrals,
                "binding": jsonl_binding(
                    neutral_path,
                    "public-benchmark-neutral-reference-candidate-v1",
                ),
            },
        }
        predecessor = {
            "distractors": distractors,
            "neutrals": neutrals,
            "artifacts": artifacts,
        }
        units = []
        fact_index = {row["base_fact_id"]: row for row in facts}
        for kind, rows, id_field in (
            ("distractor", distractors, "distractor_id"),
            ("neutral", neutrals, "neutral_candidate_id"),
        ):
            for candidate in rows:
                source_id = candidate["source_base_fact_id"]
                units.append(
                    {
                        "input_index": len(units),
                        "candidate_id": candidate[id_field],
                        "candidate_kind": kind,
                        "candidate_row_sha256": completion.sha256_value(candidate),
                        "target_base_fact_id": "fact-target",
                        "target_base_fact_row_sha256": completion.sha256_value(
                            fact_index["fact-target"]
                        ),
                        "source_base_fact_id": source_id,
                        "source_base_fact_row_sha256": completion.sha256_value(
                            fact_index[source_id]
                        ),
                    }
                )
        input_bindings = {
            label: {**value["binding"], "path": str(value["path"].resolve())}
            for label, value in artifacts.items()
        }
        selected_ids = [unit["candidate_id"] for unit in units]
        selected_by_kind = {
            kind: [
                unit["candidate_id"]
                for unit in units
                if unit["candidate_kind"] == kind
            ]
            for kind in ("distractor", "neutral")
        }
        route = {
            "schema_version": "redacted-model-route-identity-v1",
            "protocol": "openai",
            "provider_profile": "openai",
            "requested_model": "gpt-5.5",
            "expected_response_model": "gpt-5.5-2026-04-24",
            "normalized_base_url_sha256": "a" * 64,
            "normalized_request_route_sha256": "b" * 64,
            "credentials_or_endpoints_included": False,
        }
        route["route_identity_sha256"] = completion.sha256_value(route)
        route_identity = {
            "schema_version": "redacted-model-route-identity-set-v1",
            "record_count": 1,
            "records": [route],
            "credentials_or_endpoints_included": False,
        }
        route_identity["route_identity_set_sha256"] = completion.sha256_value(
            route_identity
        )
        contract = {
            "tool_version": completion.HISTORICAL_CANDIDATE_TOOL_VERSION,
            "candidate_manifest": {
                "path": str(self.postreview.resolve()),
                "sha256": completion.sha256_file(self.postreview),
                "schema_version": "public-benchmark-full-postreview-rebuild-manifest-v2",
                "manifest_kind": "postreview_rebuild",
            },
            "input_artifacts": input_bindings,
            "selected_candidate_ids_sha256": completion.sha256_value(selected_ids),
            "selected_candidate_count": len(units),
            "selected_candidate_ids_by_kind": {
                kind: {
                    "record_count": len(ids),
                    "ordered_ids_sha256": completion.sha256_value(ids),
                }
                for kind, ids in selected_by_kind.items()
            },
            "full_candidate_count": len(units),
            "selection_is_full_candidate_set": True,
            "config_sha256": completion.sha256_value({}),
            "source_fingerprints": copy.deepcopy(
                completion.HISTORICAL_SOURCE_FINGERPRINTS
            ),
            "route_identity": route_identity,
            "reviewer_spec": {
                "provider_profile": "openai",
                "model": "gpt-5.5",
                "reasoning_effort": "low",
                "max_output_tokens": 16384,
                "max_retries": 2,
                "temperature": 0.0,
                "json_mode": True,
            },
            "expected_response_model": "gpt-5.5-2026-04-24",
            "protocol_reviewer_identity": {
                "provider_profile": "openai",
                "requested_model": "gpt-5.5",
                "expected_response_model": "gpt-5.5-2026-04-24",
            },
            "prompt_contract_sha256": completion.HISTORICAL_PROMPT_CONTRACT_SHA256,
            "batch_size": 8,
            "checkpoint_compact_every": 128,
            "max_workers": 2,
            "behavior_blind": True,
            "human_gold": False,
        }
        manifest = {
            "schema_version": completion.CANDIDATE_MANIFEST_SCHEMA,
            "tool_version": completion.HISTORICAL_CANDIDATE_TOOL_VERSION,
            "status": "running",
            "run_contract": contract,
            "run_contract_sha256": completion.sha256_value(contract),
            "evidence_boundary": {
                "reviewer_type": "independent_model_proxy",
                "human_gold": False,
                "behavior_blind": True,
                "target_behavior_consumed": False,
                "hf_model_execution_count": 0,
                "hf_tokenizer_execution_count": 0,
                "behavior_execution_count": 0,
                "validation_behavior_exposure_count": 0,
                "sealed_behavior_exposure_count": 0,
                "review_freeze_performed": False,
                "split_freeze_performed": False,
                "candidate_promotion_performed": False,
            },
        }
        write_json(self.candidate, manifest)
        return predecessor, units, manifest

    def historical_v2_checkpoint_row(self, unit, run_contract_sha256, decision="accept"):
        if unit["candidate_kind"] == "distractor":
            judgments = {
                "answer_unique": True,
                "distractor_factually_false": True,
                "answer_alias_disjoint": True,
                "neutral_unrelated": None,
                "neutral_semantically_neutral": None,
            }
        else:
            judgments = {
                "answer_unique": None,
                "distractor_factually_false": None,
                "answer_alias_disjoint": None,
                "neutral_unrelated": True,
                "neutral_semantically_neutral": True,
            }
        verdict = {
            "candidate_id": unit["candidate_id"],
            "candidate_kind": unit["candidate_kind"],
            "overall_decision": decision,
            **judgments,
            "rationale": "Frozen v2 fixture rationale.",
            "confidence": "high",
        }
        adjudication = {
            "schema_version": completion.HISTORICAL_ADJUDICATION_SCHEMA,
            **unit,
            "overall_decision": decision,
            **judgments,
            "rationale": verdict["rationale"],
            "confidence": "high",
            "reviewer_type": "independent_model_proxy",
            "reviewer_id": "openai:gpt-5.5-2026-04-24",
            "requested_model": "gpt-5.5",
            "expected_response_model": "gpt-5.5-2026-04-24",
            "response_model": "gpt-5.5-2026-04-24",
            "response_model_identity_status": "matched",
            "review_method": completion.HISTORICAL_REVIEW_METHOD,
            "reviewed_at": "2026-09-25T00:00:00+00:00",
            "behavior_blind": True,
            "target_behavior_consumed": False,
            "human_gold": False,
        }
        adjudication.pop("input_index")
        return {
            "schema_version": completion.HISTORICAL_CHECKPOINT_SCHEMA,
            "tool_version": completion.HISTORICAL_CANDIDATE_TOOL_VERSION,
            **unit,
            "run_contract_sha256": run_contract_sha256,
            "terminal_status": "completed",
            "failure_stage": None,
            "model_verdict": verdict,
            "strict_adjudication": adjudication,
            "model_calls": [
                {
                    "terminal_status": "completed",
                    "prompt_version": completion.HISTORICAL_PROMPT_VERSION,
                    "provider_profile": "openai",
                    "requested_model": "gpt-5.5",
                    "expected_response_model": "gpt-5.5-2026-04-24",
                    "response_model": "gpt-5.5-2026-04-24",
                    "response_model_identity_status": "matched",
                    "credentials_or_endpoints_included": False,
                    "prompt_sha256": "c" * 64,
                    "batch_id": "fixture-batch",
                    "ordered_candidate_ids_sha256": "d" * 64,
                }
            ],
            "retry_history": [],
            "completed_at": "2026-09-25T00:00:00+00:00",
            "behavior_blind": True,
            "human_gold": False,
            "hf_model_execution_count": 0,
            "hf_tokenizer_execution_count": 0,
            "behavior_execution_count": 0,
            "validation_behavior_exposure_count": 0,
            "sealed_behavior_exposure_count": 0,
        }

    def test_candidate_decision_requires_full_terminal_coverage(self):
        self.assertEqual(
            completion.decide_candidate_action(
                status="running",
                full_count=4,
                completed_count=0,
                decision_counts={},
            ),
            "running",
        )
        self.assertEqual(
            completion.decide_candidate_action(
                status="completed",
                full_count=4,
                completed_count=4,
                decision_counts={"accept": 4},
            ),
            "terminal_all_accept",
        )
        self.assertEqual(
            completion.decide_candidate_action(
                status="completed",
                full_count=4,
                completed_count=4,
                decision_counts={"accept": 3, "defer": 1},
            ),
            "terminal_with_nonaccept",
        )
        self.assertEqual(
            completion.decide_candidate_action(
                status="completed_with_failures",
                full_count=4,
                completed_count=3,
                decision_counts={},
            ),
            "technical_recovery_required",
        )
        with self.assertRaises(completion.CompletionBlockedError):
            completion.decide_candidate_action(
                status="completed",
                full_count=4,
                completed_count=3,
                decision_counts={"accept": 3},
            )

    def test_interrupted_review_acknowledgement_is_explicit_cli_opt_in(self):
        common = [
            "--universe-root",
            str(self.universe),
            "--start-round",
            "2",
            "--start-postreview-manifest",
            str(self.postreview),
            "--start-candidate-review-manifest",
            str(self.candidate),
            "--config",
            str(self.config),
        ]
        self.assertFalse(
            completion.build_parser().parse_args(common).acknowledge_interrupted_review
        )
        self.assertTrue(
            completion.build_parser()
            .parse_args([*common, "--acknowledge-interrupted-review"])
            .acknowledge_interrupted_review
        )

    def test_candidate_contract_is_bound_to_predecessor_parameters_and_sha(self):
        orchestrator = self.orchestrator()
        manifest = self.candidate_manifest(orchestrator)
        predecessor = {"distractors": [{}, {}], "neutrals": [{}, {}]}
        result = orchestrator._candidate_manifest_contract(
            predecessor=predecessor
        )
        self.assertEqual(result["run_contract"], manifest["run_contract"])

        stale = copy.deepcopy(manifest)
        stale["run_contract"]["candidate_manifest"]["sha256"] = "0" * 64
        stale["run_contract_sha256"] = completion.sha256_value(
            stale["run_contract"]
        )
        write_json(self.candidate, stale)
        with self.assertRaisesRegex(
            completion.CompletionBlockedError, "predecessor SHA-256"
        ):
            orchestrator._candidate_manifest_contract(predecessor=predecessor)

        changed = copy.deepcopy(manifest)
        changed["run_contract"]["batch_size"] = 1
        changed["run_contract_sha256"] = completion.sha256_value(
            changed["run_contract"]
        )
        write_json(self.candidate, changed)
        with self.assertRaisesRegex(
            completion.CompletionBlockedError, "parameters or full-set coverage"
        ):
            orchestrator._candidate_manifest_contract(predecessor=predecessor)

    def test_exclusive_lock_has_one_owner_and_keeps_inode(self):
        lock_path = self.root / "lock"
        with completion.exclusive_lock(lock_path):
            with self.assertRaises(completion.OrchestratorLockedError):
                with completion.exclusive_lock(lock_path):
                    pass
        self.assertTrue(lock_path.is_file())
        with completion.exclusive_lock(lock_path):
            pass

    def test_atomic_state_and_append_only_journal(self):
        orchestrator = self.orchestrator()
        with completion.exclusive_lock(orchestrator.lock_path):
            orchestrator._load_or_initialize_state()
            first_journal = orchestrator.journal_path.read_bytes()
            orchestrator.state["status"] = "fixture_second_state"
            orchestrator._persist_state(event="fixture_second_event")

        self.assertTrue(orchestrator.state_path.is_file())
        self.assertTrue(orchestrator.journal_path.read_bytes().startswith(first_journal))
        events = [
            json.loads(line)
            for line in orchestrator.journal_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(
            [event["event"] for event in events],
            ["state_initialized", "fixture_second_event"],
        )
        self.assertEqual(
            events[-1]["state_sha256"],
            completion.sha256_file(orchestrator.state_path),
        )
        self.assertEqual(
            list(orchestrator.state_path.parent.glob(".state.json.*.tmp")), []
        )

    def test_atomic_write_preserves_old_file_if_replace_fails(self):
        path = self.root / "state.json"
        write_json(path, {"old": True})
        old_bytes = path.read_bytes()
        with mock.patch.object(completion.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                completion.atomic_write_json(path, {"new": True})
        self.assertEqual(path.read_bytes(), old_bytes)
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_external_running_candidate_only_reports_waiting(self):
        orchestrator = self.orchestrator()
        self.candidate_manifest(orchestrator, status="running")
        result = orchestrator.run()
        self.assertEqual(result["status"], "waiting_for_external_writer")
        self.assertFalse(result["automatic_command_started"])
        self.assertFalse(result["hf_or_behavior_work_performed"])
        state = json.loads(orchestrator.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "waiting_for_external_writer")
        self.assertFalse(state["safety_boundary"]["automatic_candidate_retry"])
        self.assertFalse(state["safety_boundary"]["automatic_zh"])

    def test_optional_wait_observes_running_to_terminal_without_resume(self):
        write_json(self.candidate, {"status": "running"})
        orchestrator = self.orchestrator(wait_for_external_writer=True)
        calls = {"run_locked": 0, "sleep": 0}

        def fake_run_locked():
            calls["run_locked"] += 1
            if calls["run_locked"] == 1:
                raise completion.ExternalWriterRunning(self.candidate)
            return {
                "status": "observed_terminal",
                "automatic_downstream_execution_performed": False,
            }

        def finish_external(_seconds):
            calls["sleep"] += 1
            write_json(self.candidate, {"status": "completed"})

        orchestrator.sleep_fn = finish_external
        with mock.patch.object(
            orchestrator, "_run_locked", side_effect=fake_run_locked
        ):
            result = orchestrator.run()
        self.assertEqual(result["status"], "observed_terminal")
        self.assertEqual(calls, {"run_locked": 2, "sleep": 1})

    def test_acknowledged_historical_v2_interruption_emits_partial_report_only(self):
        predecessor, units, manifest = self.historical_v2_fixture()
        run_sha = manifest["run_contract_sha256"]
        checkpoint_path = self.candidate.parent / completion.CANDIDATE_CHECKPOINT_NAME
        journal_path = self.candidate.parent / completion.CANDIDATE_JOURNAL_NAME
        first = self.historical_v2_checkpoint_row(units[0], run_sha)
        second = self.historical_v2_checkpoint_row(units[1], run_sha)
        write_jsonl(checkpoint_path, [first])
        journal_path.write_bytes(
            completion.canonical_json_bytes(second)
            + b"\n"
            + b'{"schema_version":"truncated'
        )

        previous_postreview = (
            self.universe
            / "postreview-successor-r01"
            / "postreview_rebuild_manifest.json"
        )
        previous_manifest = (
            self.universe
            / "candidate-review-r01"
            / "candidate_review_run_manifest.json"
        )
        previous_checkpoint = previous_manifest.parent / "checkpoint.jsonl"
        previous_adjudications = previous_manifest.parent / "adjudications.jsonl"
        write_json(previous_postreview, {"round": 1})
        write_json(previous_manifest, {"round": 1})
        previous_rows = [
            first["strict_adjudication"],
            second["strict_adjudication"],
        ]
        write_jsonl(previous_adjudications, previous_rows)
        write_jsonl(previous_checkpoint, [first, second])
        previous_review = {
            "manifest": {
                "run_contract": {"config_sha256": completion.sha256_value({})}
            },
            "adjudications": previous_rows,
            "adjudications_binding": binding(previous_adjudications),
            "checkpoint_path": previous_checkpoint,
            "checkpoint_binding": binding(previous_checkpoint),
            "checkpoint_rows": [first, second],
        }
        validators = FakeValidators(
            postreview=predecessor,
            completed_reviews={previous_manifest: previous_review},
        )
        orchestrator = self.orchestrator(
            validators=validators,
            acknowledge_interrupted_review=True,
        )
        result = orchestrator.run()

        self.assertEqual(result["status"], "protocol_review_required")
        self.assertFalse(result["coverage_complete"])
        self.assertTrue(result["review_interrupted"])
        self.assertEqual(result["completed_candidate_count"], 2)
        self.assertEqual(result["pending_candidate_count"], 2)
        report = json.loads(
            Path(result["stability_report_path"]).read_text(encoding="utf-8")
        )
        self.assertFalse(report["coverage_complete"])
        self.assertTrue(report["review_interrupted"])
        self.assertEqual(report["counts"]["completed_candidate_count"], 2)
        self.assertEqual(report["counts"]["pending_candidate_count"], 2)
        self.assertEqual(
            report["interrupted_review_evidence"]["checkpoint"]["record_count"],
            1,
        )
        self.assertEqual(
            report["interrupted_review_evidence"]["journal"]["record_count"],
            1,
        )
        self.assertTrue(
            report["integrity_checks"]["truncated_tail_detected"]
        )
        self.assertTrue(
            report["integrity_checks"]["merged_input_indices_contiguous_prefix"]
        )
        self.assertTrue(report["gate"]["protocol_review_required"])
        self.assertFalse(
            report["gate"]["partial_evidence_can_authorize_downstream"]
        )
        self.assertFalse(result["automatic_downstream_execution_performed"])

    def test_historical_v2_running_review_without_acknowledgement_still_waits(self):
        predecessor, _, _ = self.historical_v2_fixture()
        orchestrator = self.orchestrator(
            validators=FakeValidators(postreview=predecessor)
        )
        result = orchestrator.run()
        self.assertEqual(result["status"], "waiting_for_external_writer")
        self.assertFalse(result["automatic_command_started"])
        self.assertFalse(result["hf_or_behavior_work_performed"])
        self.assertFalse(
            (
                orchestrator.state_dir
                / "candidate_cross_round_partial_stability_r01_r02.json"
            ).exists()
        )

    def test_acknowledged_interruption_rejects_duplicate_compact_checkpoint(self):
        predecessor, units, manifest = self.historical_v2_fixture()
        row = self.historical_v2_checkpoint_row(
            units[0], manifest["run_contract_sha256"]
        )
        write_jsonl(
            self.candidate.parent / completion.CANDIDATE_CHECKPOINT_NAME,
            [row, row],
        )
        write_jsonl(
            self.candidate.parent / completion.CANDIDATE_JOURNAL_NAME,
            [],
        )
        orchestrator = self.orchestrator(
            validators=FakeValidators(postreview=predecessor),
            acknowledge_interrupted_review=True,
        )
        result = orchestrator.run()
        self.assertEqual(result["status"], "blocked_fail_closed")
        self.assertIn("duplicate interrupted compact checkpoint", result["reason"])
        self.assertFalse(result["automatic_downstream_execution_performed"])

    def test_historical_v2_completed_review_uses_frozen_validator(self):
        predecessor, units, manifest = self.historical_v2_fixture()
        run_sha = manifest["run_contract_sha256"]
        checkpoint_path = self.candidate.parent / completion.CANDIDATE_CHECKPOINT_NAME
        adjudications_path = self.candidate.parent / "candidate_semantic_adjudications.jsonl"
        checkpoint_rows = [
            self.historical_v2_checkpoint_row(unit, run_sha) for unit in units
        ]
        adjudications = [row["strict_adjudication"] for row in checkpoint_rows]
        write_jsonl(checkpoint_path, checkpoint_rows)
        write_jsonl(adjudications_path, adjudications)
        manifest.update(
            {
                "status": "completed",
                "selection_is_full_candidate_set": True,
                "candidate_review_complete_for_selected_scope": True,
                "candidate_review_complete_for_full_set": True,
                "terminal_status_counts": {"completed": len(units)},
                "overall_decision_counts": {"accept": len(units)},
                "artifacts": {
                    "checkpoint": jsonl_binding(
                        checkpoint_path, completion.HISTORICAL_CHECKPOINT_SCHEMA
                    ),
                    "adjudications": jsonl_binding(
                        adjudications_path,
                        completion.HISTORICAL_ADJUDICATION_SCHEMA,
                    ),
                },
            }
        )
        write_json(self.candidate, manifest)

        validators = completion.ExistingValidators.__new__(
            completion.ExistingValidators
        )
        validators.load_postreview = lambda _path: copy.deepcopy(predecessor)
        result = validators.load_complete_candidate_review(
            predecessor_path=self.postreview,
            manifest_path=self.candidate,
        )
        self.assertEqual(len(result["adjudications"]), 4)
        self.assertEqual(
            result["manifest"]["tool_version"],
            completion.HISTORICAL_CANDIDATE_TOOL_VERSION,
        )
        self.assertEqual(
            [row["input_index"] for row in result["checkpoint_rows"]],
            [0, 1, 2, 3],
        )

    def test_acknowledged_interruption_requires_contiguous_input_prefix(self):
        predecessor, units, manifest = self.historical_v2_fixture()
        row = self.historical_v2_checkpoint_row(
            units[1], manifest["run_contract_sha256"]
        )
        write_jsonl(
            self.candidate.parent / completion.CANDIDATE_CHECKPOINT_NAME,
            [row],
        )
        write_jsonl(
            self.candidate.parent / completion.CANDIDATE_JOURNAL_NAME,
            [],
        )
        orchestrator = self.orchestrator(
            validators=FakeValidators(postreview=predecessor),
            acknowledge_interrupted_review=True,
        )
        result = orchestrator.run()
        self.assertEqual(result["status"], "blocked_fail_closed")
        self.assertIn("not a contiguous prefix", result["reason"])

    def test_interrupted_acknowledgement_and_wait_are_mutually_exclusive(self):
        with self.assertRaisesRegex(
            completion.CompletionBlockedError, "mutually exclusive"
        ):
            self.orchestrator(
                wait_for_external_writer=True,
                acknowledge_interrupted_review=True,
            )

    def test_terminal_technical_failure_is_reported_without_retry(self):
        orchestrator = self.orchestrator()
        self.candidate_manifest(
            orchestrator,
            status="completed_with_failures",
            terminal_status_counts={"completed": 3, "failed": 1},
        )
        result = orchestrator.run()
        self.assertEqual(result["status"], "candidate_terminal_recovery_required")
        self.assertFalse(result["automatic_command_started"])
        self.assertFalse(result["hf_or_behavior_work_performed"])
        state = json.loads(orchestrator.state_path.read_text(encoding="utf-8"))
        self.assertEqual(
            state["status"], "candidate_terminal_recovery_required"
        )

    def test_stability_report_detects_accept_drift_and_same_prompt_batch(self):
        previous_postreview = (
            self.universe
            / "postreview-successor-r01"
            / "postreview_rebuild_manifest.json"
        )
        previous_manifest = (
            self.universe
            / "candidate-review-r01"
            / "candidate_review_run_manifest.json"
        )
        write_json(previous_postreview, {"round": 1})
        write_json(previous_manifest, {"round": 1})
        previous_checkpoint = previous_manifest.parent / "checkpoint.jsonl"
        current_checkpoint = self.candidate.parent / "checkpoint.jsonl"
        previous_adjudications = previous_manifest.parent / "adjudications.jsonl"
        current_adjudications = self.candidate.parent / "adjudications.jsonl"
        stable_call = {
            "terminal_status": "completed",
            "prompt_sha256": "1" * 64,
            "batch_id": "batch-stable",
            "ordered_candidate_ids_sha256": "2" * 64,
            "prompt_version": "v1",
            "requested_model": "gpt-5.5",
            "response_model": "gpt-5.5-2026-04-24",
        }
        changed_call = {**stable_call, "batch_id": "batch-changed"}
        bases = [
            {
                "candidate_id": f"candidate-{index}",
                "candidate_row_sha256": str(index) * 64,
                "candidate_kind": "distractor",
                "target_base_fact_id": f"target-{index}",
                "source_base_fact_id": f"source-{index}",
                "target_base_fact_row_sha256": "4" * 64,
                "source_base_fact_row_sha256": "5" * 64,
            }
            for index in (1, 2, 3)
        ]
        previous_rows = [
            {**bases[0], "overall_decision": "accept"},
            {**bases[1], "overall_decision": "accept"},
            {**bases[2], "overall_decision": "reject"},
        ]
        current_rows = [
            {**bases[0], "overall_decision": "reject"},
            {**bases[1], "overall_decision": "defer"},
            {**bases[2], "overall_decision": "reject"},
        ]
        write_jsonl(previous_adjudications, previous_rows)
        write_jsonl(current_adjudications, current_rows)
        write_jsonl(
            previous_checkpoint,
            [
                {"candidate_id": base["candidate_id"], "model_calls": [stable_call]}
                for base in bases
            ],
        )
        write_jsonl(
            current_checkpoint,
            [
                {
                    "candidate_id": base["candidate_id"],
                    "model_calls": [stable_call if index != 1 else changed_call],
                }
                for index, base in enumerate(bases)
            ],
        )
        previous_review = {
            "manifest": {
                "run_contract": {
                    "config_sha256": completion.sha256_value({})
                }
            },
            "adjudications": previous_rows,
            "adjudications_binding": binding(previous_adjudications),
            "checkpoint_path": previous_checkpoint,
            "checkpoint_binding": binding(previous_checkpoint),
        }
        current_review = {
            "adjudications": current_rows,
            "adjudications_binding": binding(current_adjudications),
            "checkpoint_path": current_checkpoint,
            "checkpoint_binding": binding(current_checkpoint),
        }
        validators = FakeValidators(
            completed_reviews={previous_manifest: previous_review}
        )
        orchestrator = self.orchestrator(validators=validators)
        result = orchestrator._cross_round_stability_report(
            current_review=current_review
        )
        report = result["report"]
        self.assertEqual(report["status"], "protocol_review_required")
        self.assertEqual(report["counts"]["exact_unchanged_candidate_overlap"], 3)
        self.assertEqual(report["counts"]["unchanged_candidate_decision_drift"], 2)
        self.assertEqual(
            report["counts"]["same_prompt_and_batch_decision_drift"], 1
        )
        self.assertEqual(
            report["decision_transition_counts"],
            {"accept->defer": 1, "accept->reject": 1, "reject->reject": 1},
        )
        self.assertTrue(report["gate"]["optional_stopping_risk_detected"])
        self.assertFalse(
            report["gate"][
                "automatic_resolution_successor_review_loop_authorized"
            ]
        )
        self.assertFalse(report["gate"]["automatic_zh_start_authorized"])

        orchestrator.state = {
            "candidate_stability_report": {
                "path": result["path"],
                "sha256": result["sha256"],
            }
        }
        validated = orchestrator._validate_protocol_review_state()
        self.assertEqual(validated["status"], "protocol_review_required")
        self.assertEqual(validated["unchanged_candidate_decision_drift"], 2)
        self.assertFalse(validated["automatic_downstream_execution_performed"])

    def test_every_terminal_decision_mix_stops_at_protocol_review(self):
        for action, decisions in (
            ("terminal_all_accept", {"accept": 4}),
            ("terminal_with_nonaccept", {"accept": 3, "reject": 1}),
        ):
            with self.subTest(action=action):
                state_dir = self.root / action
                orchestrator = self.orchestrator(state_dir=state_dir)
                report_path = state_dir / "stability.json"
                report = {
                    "status": "protocol_review_required",
                    "counts": {"unchanged_candidate_decision_drift": 0},
                    "gate": {
                        "automatic_resolution_successor_review_loop_authorized": False
                    },
                }
                write_json(report_path, report)
                stability = {
                    "path": str(report_path),
                    "sha256": completion.sha256_file(report_path),
                    "report": report,
                }
                with mock.patch.object(
                    orchestrator,
                    "inspect_candidate",
                    return_value={
                        "action": action,
                        "review": {},
                        "decision_counts": decisions,
                    },
                ), mock.patch.object(
                    orchestrator,
                    "_cross_round_stability_report",
                    return_value=stability,
                ), mock.patch.object(
                    orchestrator,
                    "_validate_protocol_review_state",
                    return_value={
                        "status": "protocol_review_required",
                        "unchanged_candidate_decision_drift": 0,
                        "automatic_downstream_execution_performed": False,
                    },
                ):
                    result = orchestrator.run()
                self.assertEqual(result["status"], "protocol_review_required")
                state = json.loads(
                    orchestrator.state_path.read_text(encoding="utf-8")
                )
                self.assertEqual(state["status"], "protocol_review_required")
                self.assertEqual(
                    state["protocol_review_reason"],
                    "automatic_multi_round_optional_stopping_not_authorized",
                )

    def test_module_has_no_downstream_execution_entry_point(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        defined_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("subprocess", imported_roots)
        self.assertNotIn("subprocess", source)
        self.assertTrue(
            {
                "candidate_command",
                "resolution_command",
                "zh_command",
                "_invoke",
                "_run_candidate",
                "_run_zh",
                "_ensure_resolution_and_successor",
                "_materialize_attestation_packet",
            }.isdisjoint(defined_names)
        )


if __name__ == "__main__":
    unittest.main()
