import tempfile
import unittest
from pathlib import Path
from unittest import mock


from scripts import complete_static_g0a_historical_exposure as completion
from scripts import manage_static_g0a as g0a


class HistoricalExposureCompletionTests(unittest.TestCase):
    def make_pending_manifest(self, root: Path):
        manifest = {
            "schema_version": g0a.EXPOSURE_MANIFEST_SCHEMA,
            "created_at": "2026-09-16T09:00:00+00:00",
            "input_bundle": {"sha256": "1" * 64},
            "effective_staging_bundle": {"sha256": "2" * 64},
            "resolved_static_draft": {"sha256": "3" * 64},
            "resolved_static_draft_manifest": {"sha256": "4" * 64},
            "repository_scope": {
                "repo_root": str(root),
                "git_head": "a" * 40,
                "reachable_commit_count": 2,
                "tracked_file_count": 509,
                "tracked_paths_sha256": "5" * 64,
            },
            "simulation_results_scan": {
                "file_count": 13,
                "record_count": 5498,
                "scans": {"sha256": "6" * 64},
            },
            "repository_artifact_scan": {
                "policy": "exact-holdout-identifiers-plus-behavior-signature-v1",
                "inventory": {"sha256": "a" * 64},
                "results": {"sha256": "b" * 64},
            },
        }
        path = root / "historical_exposure_manifest.pending.json"
        g0a.write_json(path, manifest)
        return path, manifest

    def make_attestation(self, root: Path, pending_path: Path, manifest):
        scope = {field: True for field in completion.REQUIRED_SCOPE_FLAGS}
        scope["proxy_model_ids"] = list(completion.EXPECTED_PROXY_MODEL_IDS)
        value = {
            "schema_version": completion.ATTESTATION_SCHEMA,
            "status": "affirmed",
            "attestation_source": "user_explicit_confirmation",
            "attester_role": "scope_owner",
            "attested_by": "fixture-scope-owner",
            "attested_at": "2026-09-16T10:00:00+00:00",
            "statement": {
                "no_validation_or_sealed_historical_behavior_exposure": True,
                "known_historical_exposures": [],
                "scope_includes": scope,
            },
            "bindings": {
                "pending_manifest_sha256": g0a.sha256_file(pending_path),
                "input_bundle_sha256": manifest["input_bundle"]["sha256"],
                "effective_staging_bundle_sha256": manifest[
                    "effective_staging_bundle"
                ]["sha256"],
                "resolved_static_draft_sha256": manifest[
                    "resolved_static_draft"
                ]["sha256"],
                "resolved_static_draft_manifest_sha256": manifest[
                    "resolved_static_draft_manifest"
                ]["sha256"],
                "repository_git_head": manifest["repository_scope"]["git_head"],
                "repository_tracked_paths_sha256": manifest["repository_scope"][
                    "tracked_paths_sha256"
                ],
                "repository_scan_inventory_sha256": completion._repository_scan_inventory_sha256(
                    manifest
                ),
                "repository_scan_results_sha256": manifest["repository_artifact_scan"][
                    "results"
                ]["sha256"],
                "simulation_results_scan_sha256": manifest["simulation_results_scan"][
                    "scans"
                ]["sha256"],
            },
        }
        path = root / "scope_attestation.json"
        g0a.write_json(path, value)
        return path, value

    def make_mock_packet(self, root: Path, pending_path: Path, manifest):
        manifest.update(
            {
                "input_bundle": {"sha256": "1" * 64, "schema_version": "input-v1"},
                "effective_staging_bundle": {
                    "sha256": "2" * 64,
                    "schema_version": "staging-v1",
                },
                "resolved_static_draft": {
                    "sha256": "3" * 64,
                    "schema_version": "resolved-v1",
                },
                "resolved_static_draft_manifest": {
                    "sha256": "4" * 64,
                    "schema_version": g0a.RESOLVED_DRAFT_MANIFEST_SCHEMA,
                },
            }
        )
        g0a.write_json(pending_path, manifest)
        checks = []
        for index in range(64):
            base_fact_id = f"pbf_{index:020d}"
            checks.append(
                {
                    "schema_version": g0a.EXPOSURE_CHECK_SCHEMA,
                    "base_fact_id": base_fact_id,
                    "source_record_sha256": "7" * 64,
                    "effective_source_record_sha256": "8" * 64,
                    "input_record_sha256": "9" * 64,
                    "split_assignment": "validation" if index < 32 else "sealed",
                    "query_fingerprint_sha256": g0a.sha256_value(base_fact_id),
                    "checked_sources": ["fixture/simulation_results.jsonl"],
                    "local_simulation_identifier_hits": [],
                    "historically_exposed": None,
                    "terminal_status": completion.EXPECTED_PENDING_STATUS,
                    "reviewer_type": "codex_proxy",
                    "human_gold": False,
                }
            )
        return {
            "manifest": manifest,
            "manifest_path": pending_path,
            "loaded": {},
            "checks": checks,
            "registry": [],
            "scans": [],
            "scans_path": root / "simulation_results_scans.jsonl",
            "repo_root": root,
        }

    def make_repository_scan_manifest(self, root: Path, *, behavior_hit=False):
        inventory_id = "worktree_fixture"
        inventory = [
            {
                "schema_version": completion.pending.REPOSITORY_INVENTORY_SCHEMA,
                "inventory_id": inventory_id,
                "source_kind": "current_worktree_file",
                "relative_paths": ["fixture.jsonl"],
                "byte_count": 100,
                "content_sha256": "c" * 64,
                "identifier_match_count": 1,
                "matched_holdout_base_fact_count": 1,
                "behavior_hit_base_fact_count": 1 if behavior_hit else 0,
                "scan_status": "completed_exact_identifier_and_behavior_signature_scan",
            }
        ]
        results = [
            {
                "schema_version": completion.pending.REPOSITORY_SCAN_RESULT_SCHEMA,
                "inventory_id": inventory_id,
                "source_kind": "current_worktree_file",
                "relative_paths": ["fixture.jsonl"],
                "matched_identifiers": ["source_fixture"],
                "matched_holdout_base_fact_ids": ["pbf_fixture"],
                "behavior_signature_names": ["simulation_id"] if behavior_hit else [],
                "behavior_hit_base_fact_ids": ["pbf_fixture"] if behavior_hit else [],
                "historical_behavior_exposure_detected": behavior_hit,
                "scan_status": "completed_exact_identifier_and_behavior_signature_scan",
            }
        ]
        inventory_path = root / "repository_artifact_inventory.jsonl"
        results_path = root / "repository_exposure_scan_results.jsonl"
        g0a.write_jsonl(inventory_path, inventory)
        g0a.write_jsonl(results_path, results)
        return {
            "repository_artifact_scan": {
                "policy": "exact-holdout-identifiers-plus-behavior-signature-v1",
                "current_worktree_complete": True,
                "reachable_git_blobs_complete": True,
                "worktree_file_count": 1,
                "reachable_git_blob_count": 0,
                "identifier_matching_artifact_count": 1,
                "behavior_hit_base_fact_count": 1 if behavior_hit else 0,
                "behavior_hit_base_fact_ids": ["pbf_fixture"] if behavior_hit else [],
                "inventory": g0a._binding(
                    inventory_path,
                    schema_version=completion.pending.REPOSITORY_INVENTORY_SCHEMA,
                    record_count=1,
                    id_values=[inventory_id],
                    id_digest_field="inventory_ids_sha256",
                ),
                "results": g0a._binding(
                    results_path,
                    schema_version=completion.pending.REPOSITORY_SCAN_RESULT_SCHEMA,
                    record_count=1,
                    id_values=[inventory_id],
                    id_digest_field="inventory_ids_sha256",
                ),
            }
        }

    def test_scope_attestation_requires_every_explicit_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_path, manifest = self.make_pending_manifest(root)
            attestation_path, attestation = self.make_attestation(
                root, pending_path, manifest
            )
            accepted = completion._validate_scope_attestation(
                attestation_path=attestation_path,
                pending_manifest_path=pending_path,
                pending_manifest=manifest,
            )
            self.assertEqual(accepted["status"], "affirmed")

            attestation["statement"]["scope_includes"]["external_platforms"] = False
            g0a.write_json(attestation_path, attestation)
            with self.assertRaisesRegex(
                completion.HistoricalExposureBlockedError,
                "external_platforms must be explicitly true",
            ):
                completion._validate_scope_attestation(
                    attestation_path=attestation_path,
                    pending_manifest_path=pending_path,
                    pending_manifest=manifest,
                )

    def test_scope_attestation_rejects_disclosed_hit_and_stale_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_path, manifest = self.make_pending_manifest(root)
            attestation_path, attestation = self.make_attestation(
                root, pending_path, manifest
            )
            attestation["statement"]["known_historical_exposures"] = [
                {"base_fact_id": "pbf_hit"}
            ]
            g0a.write_json(attestation_path, attestation)
            with self.assertRaisesRegex(
                completion.HistoricalExposureBlockedError,
                "explicit empty known_historical_exposures",
            ):
                completion._validate_scope_attestation(
                    attestation_path=attestation_path,
                    pending_manifest_path=pending_path,
                    pending_manifest=manifest,
                )

            attestation["statement"]["known_historical_exposures"] = []
            attestation["bindings"]["repository_scan_results_sha256"] = "0" * 64
            g0a.write_json(attestation_path, attestation)
            with self.assertRaisesRegex(
                completion.HistoricalExposureBlockedError,
                "repository_scan_results_sha256",
            ):
                completion._validate_scope_attestation(
                    attestation_path=attestation_path,
                    pending_manifest_path=pending_path,
                    pending_manifest=manifest,
                )

    def test_any_repository_hit_remains_blocked(self):
        with self.assertRaisesRegex(
            completion.HistoricalExposureBlockedError,
            "contains a validation/sealed hit",
        ):
            completion._require_zero_repository_hits(
                matched_count=1,
                matches=[{"base_fact_id": "pbf_hit"}],
                label="fixture repository scan",
            )

    def test_repository_artifact_evidence_accepts_zero_hits_and_rejects_a_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_manifest = self.make_repository_scan_manifest(root)
            validated = completion._validate_repository_artifact_scan(
                manifest=clean_manifest,
                owner_path=root / "owner.json",
            )
            self.assertEqual(len(validated["inventory"]), 1)
            self.assertEqual(len(validated["results"]), 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hit_manifest = self.make_repository_scan_manifest(root, behavior_hit=True)
            with self.assertRaisesRegex(
                completion.HistoricalExposureBlockedError,
                "full-repository artifact scan contains a validation/sealed hit",
            ):
                completion._validate_repository_artifact_scan(
                    manifest=hit_manifest,
                    owner_path=root / "owner.json",
                )

    def test_completion_publishes_64_checks_without_mutating_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_path, manifest = self.make_pending_manifest(root)
            packet = self.make_mock_packet(root, pending_path, manifest)
            attestation_path, _ = self.make_attestation(root, pending_path, manifest)
            pending_before = pending_path.read_bytes()
            output = root / "completed"
            with mock.patch.object(
                completion, "_validate_pending_packet", return_value=packet
            ):
                result = completion.complete_historical_exposure(
                    pending_manifest_path=pending_path,
                    scope_attestation_path=attestation_path,
                    output_dir=output,
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["holdout_check_count"], 64)
            self.assertEqual(pending_path.read_bytes(), pending_before)
            completed_checks = g0a.read_jsonl(
                output / completion.COMPLETE_CHECKS_NAME
            )
            self.assertEqual(len(completed_checks), 64)
            self.assertTrue(
                all(
                    row["terminal_status"] == "completed"
                    and row["historically_exposed"] is False
                    and row["reviewer_type"] == "user_scope_owner_attestation"
                    for row in completed_checks
                )
            )
            complete_manifest = g0a.read_json(
                output / completion.COMPLETE_MANIFEST_NAME
            )
            self.assertEqual(complete_manifest["status"], "complete")
            self.assertTrue(complete_manifest["historical_exposure_complete"])
            self.assertFalse(complete_manifest["behavior_authorized"])
            self.assertEqual(complete_manifest["known_blockers"], [])

    def test_completion_does_not_publish_when_attestation_is_not_affirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending_path, manifest = self.make_pending_manifest(root)
            packet = self.make_mock_packet(root, pending_path, manifest)
            attestation_path, attestation = self.make_attestation(
                root, pending_path, manifest
            )
            attestation["status"] = "pending"
            g0a.write_json(attestation_path, attestation)
            output = root / "must-not-exist"
            with mock.patch.object(
                completion, "_validate_pending_packet", return_value=packet
            ), self.assertRaises(completion.HistoricalExposureBlockedError):
                completion.complete_historical_exposure(
                    pending_manifest_path=pending_path,
                    scope_attestation_path=attestation_path,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_existing_output_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                completion.complete_historical_exposure(
                    pending_manifest_path=Path("unused"),
                    scope_attestation_path=Path("unused"),
                    output_dir=output,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
