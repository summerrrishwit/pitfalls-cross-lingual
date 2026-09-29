import copy
import io
import json
import random
import re
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock


from scripts import materialize_full_public_benchmark_historical_exposure_contract_v2 as exposure


class FullHistoricalExposureContractV2Tests(unittest.TestCase):
    @staticmethod
    def _legacy_identifier_matches(payload, identifiers):
        encoded = sorted(
            (value.encode("utf-8") for value in identifiers),
            key=lambda value: (-len(value), value),
        )
        pattern = re.compile(b"|".join(re.escape(value) for value in encoded))
        return sorted(
            {match.group(0).decode("utf-8") for match in pattern.finditer(payload)}
        )

    def _git(self, root: Path, *args: str) -> None:
        subprocess.run(["git", "-C", str(root), *args], check=True)

    def _write_json(self, path: Path, value) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _make_universe(self, root: Path):
        universe_id = "formal_cohort_fixture_v1"
        items = [
            {
                "schema_version": exposure.SOURCE_ITEM_SCHEMA_VERSION,
                "universe_id": universe_id,
                "selection_index": index,
                "cohort_item_id": f"cohort_item_{index}",
                "source_base_fact_id": f"pbf_fixture_{index}",
                "source_id": f"source_fixture_{index}",
                "source_row_bindings": {
                    "behavior_row_sha256": str(index) * 64,
                    "review_queue_row_sha256": str(index + 1) * 64,
                    "base_fact_cluster_row_sha256": str(index + 2) * 64,
                },
                "member_count": 1,
                "member_bindings": [
                    {
                        "candidate_id": f"candidate_fixture_{index}",
                        "triple_record_sha256": str(index + 3) * 64,
                        "source_input_record_sha256": str(index + 4) * 64,
                    }
                ],
            }
            for index in range(2)
        ]
        items_path = root / "formal_cohort_universe_items.jsonl"
        items_path.write_bytes(exposure._jsonl_payload(items))
        cohort_ids = [row["cohort_item_id"] for row in items]
        base_ids = [row["source_base_fact_id"] for row in items]
        manifest = {
            "schema_version": exposure.SOURCE_UNIVERSE_SCHEMA_VERSION,
            "tool_version": "public-benchmark-finalizer-preflight-v2",
            "universe_id": universe_id,
            "universe_label": "fixture-full-universe-v1",
            "universe_status": "declared_immutable_not_reviewed",
            "selection_policy": {"mode": "full-pool"},
            "selection_source": {
                "mode": "full_pool",
                "record_count": len(items),
                "ordered_base_fact_ids_sha256": exposure.sha256_value(base_ids),
            },
            "historical_exposure_contract_status": exposure.SOURCE_CONTRACT_STATUS,
            "record_count": len(items),
            "ordered_cohort_item_ids_sha256": exposure.sha256_value(cohort_ids),
            "ordered_source_base_fact_ids_sha256": exposure.sha256_value(base_ids),
            "ordered_item_row_hashes_sha256": exposure.sha256_value(
                [exposure.sha256_value(row) for row in items]
            ),
            "items": exposure._binding(
                items_path,
                schema_version=exposure.SOURCE_ITEM_SCHEMA_VERSION,
                record_count=len(items),
            ),
            "review_complete": False,
            "canonical_freeze_authorized": False,
            "split_freeze_authorized": False,
        }
        manifest_path = root / "formal_cohort_universe_manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest_path, items

    def _make_repository(self, root: Path):
        self._git(root, "init", "-q")
        manifest_path, items = self._make_universe(root)
        historical = root / "deleted_behavior_results.jsonl"
        historical.write_text(
            json.dumps(
                {
                    "source_base_fact_id": "pbf_fixture_0",
                    "simulation_id": "old-sim",
                    "model": "fixture-target",
                    "selected_option": "B",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self._git(root, "add", ".")
        self._git(
            root,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture history",
        )
        historical.unlink()
        self._git(root, "add", "-u")
        self._git(
            root,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-q",
            "-m",
            "remove historical result",
        )
        (root / "current_behavior_results.jsonl").write_text(
            json.dumps(
                {
                    "source_base_fact_id": "pbf_fixture_1",
                    "model_id": "fixture-target",
                    "predicted_answer": "C",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "fact_review.jsonl").write_text(
            json.dumps(
                {
                    "source_base_fact_id": "pbf_fixture_1",
                    "model": "fixture-reviewer",
                    "raw_response": "review prose",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path, items

    def _materialize(self, root: Path):
        manifest_path, items = self._make_repository(root)
        original_manifest = manifest_path.read_bytes()
        output_dir = root / "pending-contract-v2"
        result = exposure.materialize_pending_contract(
            universe_manifest_path=manifest_path,
            repo_root=root,
            output_dir=output_dir,
            expected_record_count=2,
        )
        self.assertEqual(manifest_path.read_bytes(), original_manifest)
        return result, output_dir, items

    def _completed_attestation(self, output_dir: Path):
        template = exposure.read_json(output_dir / exposure.ATTESTATION_TEMPLATE_NAME)
        contract = exposure.read_json(output_dir / exposure.CONTRACT_NAME)
        created_at = datetime.fromisoformat(contract["created_at"])
        value = copy.deepcopy(template)
        value.update(
            {
                "status": exposure.AFFIRMED_STATUS,
                "attestation_source": "user_explicit_confirmation",
                "attester_role": "scope_owner",
                "attested_by": "fixture-scope-owner",
                "authority_basis": "owns the complete experiment storage and run history",
                "attested_at": (created_at + timedelta(seconds=2)).isoformat(),
                "attested_through": (created_at + timedelta(seconds=1)).isoformat(),
            }
        )
        value["statements"] = {
            field: True for field in exposure.REQUIRED_STATEMENTS
        }
        value["limitations_acknowledged"] = {
            field: True for field in exposure.REQUIRED_LIMITATION_ACKNOWLEDGEMENTS
        }
        value["scope_inventory"] = {
            kind: {"inventory_complete": True, "sources": []}
            for kind in exposure.SCOPE_KINDS
        }
        exposed = set()
        for row in value["repository_hit_adjudications"]:
            row["decision"] = "confirmed_exposure"
            row["rationale"] = "fixture behavior result is a real historical response"
            exposed.add(row["source_base_fact_id"])
        value["known_historical_exposure_source_base_fact_ids"] = sorted(exposed)
        value["completion_instructions"] = template["completion_instructions"]
        path = output_dir / "scope_owner_attestation.completed.v2.json"
        self._write_json(path, value)
        return path, value

    def test_materializer_scans_worktree_and_reachable_history_but_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output_dir, _ = self._materialize(Path(directory))
            contract = exposure.read_json(output_dir / exposure.CONTRACT_NAME)
            template = exposure.read_json(output_dir / exposure.ATTESTATION_TEMPLATE_NAME)

            self.assertEqual(result["status"], exposure.PENDING_STATUS)
            self.assertFalse(result["historical_exposure_complete"])
            self.assertFalse(result["split_freeze_authorized"])
            self.assertFalse(result["exact_hf_authorized"])
            self.assertEqual(contract["source_universe_v1"]["record_count"], 2)
            self.assertEqual(
                contract["source_universe_v1"]["historical_exposure_contract_status"],
                "not_predeclared",
            )
            self.assertTrue(contract["repository_scan"]["current_worktree_complete"])
            self.assertTrue(contract["repository_scan"]["reachable_git_blobs_complete"])
            self.assertEqual(
                contract["repository_scan"]["behavior_candidate_source_base_fact_ids"],
                ["pbf_fixture_0", "pbf_fixture_1"],
            )
            self.assertIsNone(template["attested_by"])
            self.assertIsNone(template["known_historical_exposure_source_base_fact_ids"])
            self.assertFalse(contract["authority_boundary"]["external_scope_claims_generated_by_tool"])
            self.assertFalse(contract["authority_boundary"]["template_is_attestation"])
            self.assertEqual(
                contract["execution_safety"],
                {
                    "network_used": False,
                    "model_inference_used": False,
                    "hf_checkpoint_loaded": False,
                    "tokenizer_loaded": False,
                    "behavior_run_performed": False,
                    "validation_or_sealed_exposure_assessed": False,
                },
            )

            pending = exposure.validate_contract(
                contract_path=output_dir / exposure.CONTRACT_NAME
            )
            self.assertEqual(
                pending["status"], "valid_pending_contract_waiting_for_scope_owner"
            )
            self.assertFalse(pending["ready_for_successor_universe_binding"])

    def test_identifier_matcher_preserves_legacy_leftmost_longest_semantics(self):
        cases = [
            (["a", "ab", "abc"], b"abc"),
            (["abc", "bc", "c"], b"abc bc c"),
            (["ab", "bcde"], b"abcde"),
            (["ababa", "bab", "aba"], b"ababaaba"),
            (["a", "bbbb"], b"abbbb"),
            (["a+b", "a.b", "[x]"], b"a+b a.b [x]"),
            (["中文", "中文标识", "标识"], "中文标识 标识".encode("utf-8")),
            (["a\x00b", "\x00b"], b"a\x00b\x00b"),
        ]
        for identifiers, payload in cases:
            with self.subTest(identifiers=identifiers, payload=payload):
                matcher = exposure._identifier_pattern(
                    {value: {"fixture"} for value in identifiers}
                )
                self.assertEqual(
                    exposure._matched_identifiers(payload, matcher),
                    self._legacy_identifier_matches(payload, identifiers),
                )

        generator = random.Random(20260928)
        alphabet = "abc"
        universe = [
            "".join(chars)
            for length in range(1, 6)
            for chars in __import__("itertools").product(alphabet, repeat=length)
        ]
        for _ in range(100):
            identifiers = generator.sample(universe, generator.randint(1, 40))
            payload = "".join(generator.choice(alphabet) for _ in range(128)).encode()
            matcher = exposure._identifier_pattern(
                {value: {"fixture"} for value in identifiers}
            )
            self.assertEqual(
                exposure._matched_identifiers(payload, matcher),
                self._legacy_identifier_matches(payload, identifiers),
            )

    def test_identifier_matcher_handles_large_payload_and_full_scale_identifier_count(self):
        identifiers = {
            f"pbf_{index:020d}": {f"base_{index}"} for index in range(8969)
        }
        first = "pbf_00000000000000000000"
        last = "pbf_00000000000000008968"
        payload = first.encode() + b"p" * (4 * 1024 * 1024) + last.encode()
        matcher = exposure._identifier_pattern(identifiers)
        self.assertEqual(
            exposure._matched_identifiers(payload, matcher), [first, last]
        )

    def test_git_blob_payloads_stream_each_oid_before_reading_next(self):
        payloads = {
            "a" * 40: b"first\x00blob",
            "b" * 40: b"x" * (128 * 1024),
        }
        sources = [
            {"git_blob_oid": "a" * 40},
            {"git_blob_oid": "b" * 40},
            {"git_blob_oid": "a" * 40},
        ]

        class FakeProcess:
            def __init__(self):
                self.events = []
                self.pending = None
                self.return_code = None
                self.stdin = self.FakeStdin(self)
                self.stdout = self.FakeStdout(self)
                self.stderr = io.BytesIO()

            class FakeStdin:
                def __init__(self, owner):
                    self.owner = owner
                    self.closed = False
                    self.flushed = False

                def write(self, value):
                    if self.owner.pending is not None or value.count(b"\n") != 1:
                        raise AssertionError("requests must be written one at a time")
                    object_id = value.rstrip(b"\n").decode("ascii")
                    self.owner.pending = {
                        "object_id": object_id,
                        "header": f"{object_id} blob {len(payloads[object_id])}\n".encode(),
                        "body": io.BytesIO(payloads[object_id] + b"\n"),
                    }
                    self.flushed = False
                    self.owner.events.append(("write", object_id))
                    return len(value)

                def flush(self):
                    self.flushed = True
                    self.owner.events.append(("flush", None))

                def close(self):
                    self.closed = True

            class FakeStdout:
                def __init__(self, owner):
                    self.owner = owner

                def readline(self):
                    if self.owner.pending is None or not self.owner.stdin.flushed:
                        raise AssertionError("response read must follow a flushed request")
                    self.owner.events.append(("readline", None))
                    return self.owner.pending["header"]

                def read(self, size):
                    if self.owner.pending is None:
                        raise AssertionError("no response is pending")
                    value = self.owner.pending["body"].read(size)
                    self.owner.events.append(("read", size))
                    if self.owner.pending["body"].tell() == len(
                        self.owner.pending["body"].getvalue()
                    ):
                        self.owner.pending = None
                    return value

                def close(self):
                    return None

            def poll(self):
                return self.return_code

            def wait(self):
                self.return_code = 0
                return 0

            def kill(self):
                self.return_code = -9

        process = FakeProcess()
        with mock.patch.object(exposure.subprocess, "Popen", return_value=process):
            results = list(exposure._git_blob_payloads(Path("."), sources))
        self.assertEqual([source for source, _ in results], sources)
        self.assertEqual(
            [payload for _, payload in results],
            [payloads[source["git_blob_oid"]] for source in sources],
        )
        self.assertEqual(
            [event[0] for event in process.events],
            ["write", "flush", "readline", "read", "read"] * len(sources),
        )

    def test_valid_explicit_scope_owner_attestation_is_read_only_and_not_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self._materialize(Path(directory))
            attestation_path, _ = self._completed_attestation(output_dir)
            before = (output_dir / exposure.CONTRACT_NAME).read_bytes()

            result = exposure.validate_contract(
                contract_path=output_dir / exposure.CONTRACT_NAME,
                scope_owner_attestation_path=attestation_path,
            )

            self.assertTrue(result["scope_owner_attestation_valid"])
            self.assertTrue(result["ready_for_successor_universe_binding"])
            self.assertFalse(result["historical_exposure_complete"])
            self.assertFalse(result["split_freeze_authorized"])
            self.assertFalse(result["exact_hf_authorized"])
            self.assertEqual(
                (output_dir / exposure.CONTRACT_NAME).read_bytes(), before
            )

    def test_template_itself_cannot_be_treated_as_an_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self._materialize(Path(directory))
            with self.assertRaisesRegex(
                exposure.ContractValidationError, "not explicitly affirmed"
            ):
                exposure.validate_contract(
                    contract_path=output_dir / exposure.CONTRACT_NAME,
                    scope_owner_attestation_path=output_dir
                    / exposure.ATTESTATION_TEMPLATE_NAME,
                )

    def test_external_scope_requires_explicit_complete_even_when_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self._materialize(Path(directory))
            attestation_path, value = self._completed_attestation(output_dir)
            value["scope_inventory"]["external_platforms"]["inventory_complete"] = False
            self._write_json(attestation_path, value)
            with self.assertRaisesRegex(
                exposure.ContractValidationError,
                "external_platforms.inventory_complete must be explicitly true",
            ):
                exposure.validate_contract(
                    contract_path=output_dir / exposure.CONTRACT_NAME,
                    scope_owner_attestation_path=attestation_path,
                )

    def test_repository_hit_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self._materialize(Path(directory))
            attestation_path, value = self._completed_attestation(output_dir)
            value["repository_hit_adjudications"] = value[
                "repository_hit_adjudications"
            ][1:]
            self._write_json(attestation_path, value)
            with self.assertRaisesRegex(
                exposure.ContractValidationError,
                "repository behavior candidates are not fully adjudicated",
            ):
                exposure.validate_contract(
                    contract_path=output_dir / exposure.CONTRACT_NAME,
                    scope_owner_attestation_path=attestation_path,
                )

    def test_disclosure_must_equal_repository_and_external_exposures(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output_dir, _ = self._materialize(Path(directory))
            attestation_path, value = self._completed_attestation(output_dir)
            value["known_historical_exposure_source_base_fact_ids"] = []
            self._write_json(attestation_path, value)
            with self.assertRaisesRegex(
                exposure.ContractValidationError,
                "do not equal repository plus external disclosures",
            ):
                exposure.validate_contract(
                    contract_path=output_dir / exposure.CONTRACT_NAME,
                    scope_owner_attestation_path=attestation_path,
                )

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "existing"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                exposure.materialize_pending_contract(
                    universe_manifest_path=Path("unused"),
                    repo_root=root,
                    output_dir=output_dir,
                    expected_record_count=2,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
