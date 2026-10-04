import copy
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


from scripts import materialize_full_public_benchmark_formal_universe_v2 as universe
from scripts import (
    materialize_full_public_benchmark_historical_exposure_contract_v2 as exposure,
)


class FullFormalUniverseV2SuccessorTests(unittest.TestCase):
    def _git(self, root: Path, *args: str) -> None:
        subprocess.run(["git", "-C", str(root), *args], check=True)

    def _write_json(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(universe._json_payload(value))

    def _make_source_universe(
        self,
        root: Path,
        *,
        directory_name: str = "formal-universe-v1",
        universe_id: str = "formal_cohort_fixture_v1",
        base_id_prefix: str = "pbf_fixture",
    ):
        output = root / directory_name
        output.mkdir(parents=True)
        items = []
        for index in range(2):
            items.append(
                {
                    "schema_version": exposure.SOURCE_ITEM_SCHEMA_VERSION,
                    "universe_id": universe_id,
                    "selection_index": index,
                    "cohort_item_id": f"cohort_item_fixture_{index}",
                    "source_base_fact_id": f"{base_id_prefix}_{index}",
                    "source_id": f"source_fixture_{index}",
                    "source_row_bindings": {
                        "behavior_row_sha256": f"{index + 1:064x}",
                        "review_queue_row_sha256": f"{index + 2:064x}",
                        "base_fact_cluster_row_sha256": f"{index + 3:064x}",
                    },
                    "member_count": 1,
                    "member_bindings": [
                        {
                            "candidate_id": f"candidate_fixture_{index}",
                            "triple_record_sha256": f"{index + 4:064x}",
                            "source_input_record_sha256": f"{index + 5:064x}",
                        }
                    ],
                }
            )
        items_path = output / universe.ITEMS_NAME
        items_path.write_bytes(exposure._jsonl_payload(items))
        cohort_ids = [row["cohort_item_id"] for row in items]
        base_ids = [row["source_base_fact_id"] for row in items]
        manifest = {
            "schema_version": exposure.SOURCE_UNIVERSE_SCHEMA_VERSION,
            "tool_version": "public-benchmark-finalizer-preflight-v2",
            "universe_id": universe_id,
            "universe_label": "fixture-full-universe-v1",
            "universe_status": "declared_immutable_not_reviewed",
            "selection_policy": {
                "policy_id": "fixture-full-pool-v1",
                "mode": "full-pool",
                "formal_cohort_purpose_explicit": True,
                "review_sample_inference_role_not_inherited": True,
            },
            "selection_source": {
                "mode": "full_pool",
                "source_role": "behavior_bundle",
                "record_count": len(items),
                "ordered_base_fact_ids_sha256": exposure.sha256_value(base_ids),
            },
            "source_artifacts": {"fixture": {"sha256": "a" * 64}},
            "comparison_canonical": {"fixture": {"sha256": "b" * 64}},
            "near_duplicate_audit_contract": {"policy": "fixture"},
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
        manifest_path = output / universe.MANIFEST_NAME
        self._write_json(manifest_path, manifest)
        return manifest_path, items

    def _fixture(self, root: Path):
        self._git(root, "init", "-q")
        source_manifest, source_items = self._make_source_universe(root)
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
            "fixture source universe",
        )
        (root / "historical_behavior_result.jsonl").write_text(
            json.dumps(
                {
                    "source_base_fact_id": "pbf_fixture_0",
                    "model_id": "fixture-target-model",
                    "predicted_answer": "fixture answer",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        packet_dir = root / "historical-exposure-contract-v2"
        exposure.materialize_pending_contract(
            universe_manifest_path=source_manifest,
            repo_root=root,
            output_dir=packet_dir,
            expected_record_count=2,
        )
        return (
            source_manifest,
            source_items,
            packet_dir / exposure.CONTRACT_NAME,
            packet_dir,
        )

    def _materialize(self, root: Path):
        source_manifest, source_items, contract_path, packet_dir = self._fixture(root)
        output_dir = root / "full-8969-formal-universe-v2"
        source_before = source_manifest.read_bytes()
        result = universe.materialize_successor_universe(
            source_universe_manifest_path=source_manifest,
            historical_exposure_contract_path=contract_path,
            output_dir=output_dir,
            universe_label="fixture-full-8969-formal-universe-v2",
            expected_record_count=2,
        )
        self.assertEqual(source_manifest.read_bytes(), source_before)
        return (
            result,
            output_dir,
            source_manifest,
            source_items,
            contract_path,
            packet_dir,
        )

    def _completed_attestation(self, packet_dir: Path) -> Path:
        template = exposure.read_json(packet_dir / exposure.ATTESTATION_TEMPLATE_NAME)
        contract = exposure.read_json(packet_dir / exposure.CONTRACT_NAME)
        created_at = datetime.fromisoformat(contract["created_at"])
        value = copy.deepcopy(template)
        value.update(
            {
                "status": exposure.AFFIRMED_STATUS,
                "attestation_source": "user_explicit_confirmation",
                "attester_role": "scope_owner",
                "attested_by": "fixture-scope-owner",
                "authority_basis": "owns the complete fixture history",
                "attested_at": (created_at + timedelta(seconds=2)).isoformat(),
                "attested_through": (created_at + timedelta(seconds=1)).isoformat(),
            }
        )
        value["statements"] = {
            field: True for field in exposure.REQUIRED_STATEMENTS
        }
        value["limitations_acknowledged"] = {
            field: True
            for field in exposure.REQUIRED_LIMITATION_ACKNOWLEDGEMENTS
        }
        value["scope_inventory"] = {
            kind: {"inventory_complete": True, "sources": []}
            for kind in exposure.SCOPE_KINDS
        }
        exposed = set()
        for row in value["repository_hit_adjudications"]:
            row["decision"] = "confirmed_exposure"
            row["rationale"] = "fixture historical behavior evidence"
            exposed.add(row["source_base_fact_id"])
        value["known_historical_exposure_source_base_fact_ids"] = sorted(exposed)
        path = packet_dir / "scope_owner_attestation.completed.v2.json"
        self._write_json(path, value)
        return path

    def test_materializes_exact_scope_successor_with_pending_contract_prebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                result,
                output_dir,
                source_manifest,
                source_items,
                contract_path,
                packet_dir,
            ) = self._materialize(root)
            manifest_path = output_dir / universe.MANIFEST_NAME
            manifest = universe.read_json(manifest_path)
            items = universe.read_jsonl(output_dir / universe.ITEMS_NAME)
            contract = exposure.read_json(contract_path)

            self.assertEqual(result["record_count"], 2)
            self.assertEqual(manifest["artifact_version"], universe.ARTIFACT_VERSION)
            self.assertEqual(
                manifest["lineage"]["predecessor_manifest"]["sha256"],
                universe.sha256_file(source_manifest),
            )
            self.assertEqual(
                manifest["lineage"]["predecessor_universe_id"],
                source_items[0]["universe_id"],
            )
            self.assertEqual(
                [row["source_base_fact_id"] for row in items],
                [row["source_base_fact_id"] for row in source_items],
            )
            self.assertEqual(
                [row["cohort_item_id"] for row in items],
                [row["cohort_item_id"] for row in source_items],
            )
            for source, successor in zip(source_items, items):
                self.assertEqual(successor["source_row_bindings"], source["source_row_bindings"])
                self.assertEqual(successor["member_bindings"], source["member_bindings"])
                self.assertEqual(
                    successor["predecessor_item_binding"]["row_sha256"],
                    universe.sha256_value(source),
                )

            authority = manifest["historical_exposure_authority"]
            self.assertEqual(authority["contract_id"], contract["contract_id"])
            self.assertEqual(
                authority["attestation_challenge_sha256"],
                contract["attestation_challenge_sha256"],
            )
            self.assertEqual(
                authority["contract"]["sha256"],
                universe.sha256_file(contract_path),
            )
            self.assertEqual(
                authority["scope_owner_attestation_template"]["sha256"],
                universe.sha256_file(packet_dir / exposure.ATTESTATION_TEMPLATE_NAME),
            )
            self.assertIsNone(
                authority["future_attestation_binding"][
                    "completed_scope_owner_attestation"
                ]
            )
            self.assertFalse(authority["future_attestation_binding"]["in_place_manifest_edit_allowed"])
            for field in (
                "historical_exposure_complete",
                "review_complete",
                "review_freeze_authorized",
                "canonical_freeze_authorized",
                "split_freeze_authorized",
                "exact_hf_authorized",
                "behavior_authorized",
            ):
                self.assertFalse(manifest[field])

            validated = universe.validate_successor_universe(
                universe_manifest_path=manifest_path,
                expected_record_count=2,
            )
            self.assertEqual(
                validated["status"],
                "valid_pending_formal_universe_v2_successor",
            )
            self.assertFalse(validated["scope_owner_attestation_bound"])
            self.assertIsNone(validated["future_attestation_binding_candidate"])

    def test_completed_attestation_only_yields_future_sha_binding_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output_dir, _, _, _, packet_dir = self._materialize(root)
            attestation_path = self._completed_attestation(packet_dir)
            manifest_path = output_dir / universe.MANIFEST_NAME
            before = manifest_path.read_bytes()

            result = universe.validate_successor_universe(
                universe_manifest_path=manifest_path,
                scope_owner_attestation_path=attestation_path,
                expected_record_count=2,
            )

            candidate = result["future_attestation_binding_candidate"]
            self.assertEqual(candidate["sha256"], universe.sha256_file(attestation_path))
            self.assertEqual(candidate["known_historical_exposure_count"], 1)
            self.assertFalse(result["scope_owner_attestation_bound"])
            self.assertFalse(result["historical_exposure_complete"])
            self.assertFalse(result["split_freeze_authorized"])
            self.assertFalse(result["exact_hf_authorized"])
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_validator_rejects_item_tampering_even_if_binding_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output_dir, _, _, _, _ = self._materialize(root)
            manifest_path = output_dir / universe.MANIFEST_NAME
            items_path = output_dir / universe.ITEMS_NAME
            rows = universe.read_jsonl(items_path)
            rows[0]["source_id"] = "tampered-source"
            items_path.write_bytes(universe._jsonl_payload(rows))
            manifest = universe.read_json(manifest_path)
            manifest["items"] = universe._binding(
                items_path,
                schema_version=universe.ITEM_SCHEMA_VERSION,
                record_count=2,
            )
            manifest["ordered_item_row_hashes_sha256"] = universe.sha256_value(
                [universe.sha256_value(row) for row in rows]
            )
            self._write_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                universe.UniverseValidationError,
                "do not exactly copy the predecessor ordered scope",
            ):
                universe.validate_successor_universe(
                    universe_manifest_path=manifest_path,
                    expected_record_count=2,
                )

    def test_contract_for_a_different_predecessor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest, _, contract_path, _ = self._fixture(root)
            other_manifest, _ = self._make_source_universe(
                root,
                directory_name="other-formal-universe-v1",
                universe_id="formal_cohort_other_v1",
                base_id_prefix="pbf_other",
            )
            with self.assertRaisesRegex(
                universe.UniverseValidationError,
                "contract targets a different predecessor universe",
            ):
                universe.materialize_successor_universe(
                    source_universe_manifest_path=other_manifest,
                    historical_exposure_contract_path=contract_path,
                    output_dir=root / "should-not-exist",
                    universe_label="invalid-cross-bound-successor",
                    expected_record_count=2,
                )
            self.assertTrue(source_manifest.is_file())
            self.assertFalse((root / "should-not-exist").exists())

    def test_pending_manifest_cannot_be_completed_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output_dir, _, _, _, _ = self._materialize(root)
            manifest_path = output_dir / universe.MANIFEST_NAME
            manifest = universe.read_json(manifest_path)
            manifest["historical_exposure_authority"]["future_attestation_binding"][
                "completed_scope_owner_attestation"
            ] = {"sha256": "f" * 64}
            self._write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                universe.UniverseValidationError,
                "stale or completed in place",
            ):
                universe.validate_successor_universe(
                    universe_manifest_path=manifest_path,
                    expected_record_count=2,
                )

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "existing"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                universe.materialize_successor_universe(
                    source_universe_manifest_path=Path("unused"),
                    historical_exposure_contract_path=Path("unused"),
                    output_dir=output_dir,
                    universe_label="unused",
                    expected_record_count=2,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
