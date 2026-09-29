import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "freeze_pre_exact_hf_protocol.py"
SPEC = importlib.util.spec_from_file_location("freeze_pre_exact_hf_protocol", SCRIPT_PATH)
protocol = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


class PreExactHFProtocolFreezeTests(unittest.TestCase):
    CREATED_AT = "2026-09-17T00:00:00+00:00"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="freeze-pre-exact-hf-protocol-test-"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.generator_path = (
            self.root / "scripts" / "freeze_pre_exact_hf_protocol.py"
        )
        self.generator_path.parent.mkdir(parents=True)
        self.generator_path.write_text(
            "# fixture generator identity\n", encoding="utf-8"
        )
        self.protocol_document_path = self.root / "tasks" / "protocol.md"
        self.protocol_document_path.parent.mkdir(parents=True)
        self.protocol_document_path.write_text(
            "# Frozen fixture protocol\n", encoding="utf-8"
        )
        self.static_dir = self.root / "frozen-v1"
        self.static_dir.mkdir()
        self.bundle_path = self.static_dir / "frozen_static_bundle.jsonl"
        self.split_manifest_path = self.static_dir / "split_freeze_manifest.json"
        self.static_manifest_path = self.static_dir / "static_freeze_manifest.json"
        self.rows = self._fixture_rows()
        self._write_jsonl(self.bundle_path, self.rows)
        self._write_json(
            self.split_manifest_path,
            {
                "schema_version": "factual-split-freeze-manifest-v2",
                "split_policy_version": "fixture-split-policy-v1",
                "split_status": "frozen",
                "input_bundle": {
                    "schema_version": "factual-perturbation-input-bundle-v1",
                    "sha256": protocol.sha256_file(self.bundle_path),
                    "record_count": len(self.rows),
                    "base_fact_ids_sha256": protocol.sha256_value(
                        sorted(row["base_fact_id"] for row in self.rows)
                    ),
                },
            },
        )
        self._write_static_manifest()

    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _fixture_rows(self):
        rows = []
        development_rows = []
        for relation_index, relation in enumerate(protocol.EXPECTED_RELATIONS):
            for item_index in range(24):
                development_rows.append(
                    self._row(
                        base_fact_id=f"dev-{relation_index}-{item_index:02d}",
                        split="development",
                        relation=relation,
                        component_id=(
                            f"dev-component-{relation_index}-{item_index:02d}"
                        ),
                    )
                )

        # Exercise the real component-atomic assignment path with the same
        # 87x1, 1x2, 1x3, 1x4 component-size distribution as the frozen data.
        by_id = {row["base_fact_id"]: row for row in development_rows}
        grouped_ids = {
            "dev-component-size-4": [
                "dev-0-00",
                "dev-1-00",
                "dev-2-00",
                "dev-3-00",
            ],
            "dev-component-size-3": ["dev-0-01", "dev-1-01", "dev-2-01"],
            "dev-component-size-2": ["dev-0-02", "dev-3-01"],
        }
        for component_id, base_fact_ids in grouped_ids.items():
            for base_fact_id in base_fact_ids:
                row = by_id[base_fact_id]
                row["leakage_component_id"] = component_id
                row["postclosure_lineage"][
                    "postclosure_split_group_id"
                ] = component_id
        rows.extend(development_rows)

        for split in ("validation", "sealed"):
            for index in range(32):
                relation = protocol.EXPECTED_RELATIONS[index % 4]
                rows.append(
                    self._row(
                        base_fact_id=f"{split}-{index:02d}",
                        split=split,
                        relation=relation,
                        component_id=f"{split}-component-{index:02d}",
                    )
                )
        return rows

    @staticmethod
    def _row(base_fact_id, split, relation, component_id):
        return {
            "base_fact_id": base_fact_id,
            "source_id": f"source-{base_fact_id}",
            "probe_relation_id": relation,
            "leakage_component_id": component_id,
            "split_assignment": split,
            "postclosure_lineage": {
                "postclosure_split_group_id": component_id,
            },
            "static_mcq": {
                "variants": [
                    {"variant_id": f"{base_fact_id}-d1"},
                    {"variant_id": f"{base_fact_id}-d2"},
                ],
                "behavior_inputs": [
                    {"input_id": f"{base_fact_id}-input-{index:02d}"}
                    for index in range(10)
                ],
            },
        }

    def _write_static_manifest(self):
        self._write_json(
            self.static_manifest_path,
            {
                "schema_version": "static-g0a-freeze-manifest-v1",
                "status": "g0a_static_stimulus_frozen",
                "static_frozen": True,
                "artifacts": {
                    "frozen_static_bundle.jsonl": {
                        "path": self.bundle_path.name,
                        "sha256": protocol.sha256_file(self.bundle_path),
                        "byte_count": self.bundle_path.stat().st_size,
                    },
                    "split_freeze_manifest.json": {
                        "path": self.split_manifest_path.name,
                        "sha256": protocol.sha256_file(self.split_manifest_path),
                        "byte_count": self.split_manifest_path.stat().st_size,
                    },
                },
            },
        )

    def _generate(self, output_dir):
        with mock.patch.object(protocol, "__file__", str(self.generator_path)):
            return protocol.generate(
                project_root=self.root,
                static_manifest_path=self.static_manifest_path,
                protocol_document_path=self.protocol_document_path,
                output_dir=output_dir,
                created_at=self.CREATED_AT,
                completed_proxy_run_id="fixture-proxy-run",
            )

    @staticmethod
    def _read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_fold_manifest_is_exact_balanced_component_atomic_and_oof(self):
        output_dir = self.root / "pre-exact-hf-protocol-v1"
        self._generate(output_dir)
        fold_manifest = self._read_json(
            output_dir / "development_fold_manifest.json"
        )
        assignments = fold_manifest["assignments"]

        assigned_ids = [row["base_fact_id"] for row in assignments]
        development_ids = {
            row["base_fact_id"]
            for row in self.rows
            if row["split_assignment"] == "development"
        }
        validation_or_sealed_ids = {
            row["base_fact_id"]
            for row in self.rows
            if row["split_assignment"] != "development"
        }
        self.assertEqual(len(assignments), 96)
        self.assertEqual(len(assigned_ids), len(set(assigned_ids)))
        self.assertEqual(set(assigned_ids), development_ids)
        self.assertTrue(set(assigned_ids).isdisjoint(validation_or_sealed_ids))

        fold_counts = Counter(
            row["development_fold_id"] for row in assignments
        )
        self.assertEqual(
            fold_counts,
            Counter({fold_id: 24 for fold_id in protocol.FOLD_IDS}),
        )
        relation_by_fold = defaultdict(Counter)
        component_folds = defaultdict(set)
        for row in assignments:
            fold_id = row["development_fold_id"]
            relation_by_fold[fold_id][row["probe_relation_id"]] += 1
            component_folds[row["leakage_component_id"]].add(fold_id)
        expected_relations = Counter(
            {relation: 6 for relation in protocol.EXPECTED_RELATIONS}
        )
        for fold_id in protocol.FOLD_IDS:
            self.assertEqual(relation_by_fold[fold_id], expected_relations)
        self.assertTrue(all(len(folds) == 1 for folds in component_folds.values()))
        self.assertEqual(
            fold_manifest["component_summary"]["component_size_counts"],
            {"1": 87, "2": 1, "3": 1, "4": 1},
        )

        all_eval_ids = []
        source_by_id = {row["base_fact_id"]: row for row in self.rows}
        for fold in fold_manifest["folds"]:
            eval_ids = set(fold["eval_base_fact_ids"])
            train_ids = development_ids - eval_ids
            self.assertEqual(len(eval_ids), 24)
            self.assertEqual(len(train_ids), 72)
            self.assertFalse(eval_ids & train_ids)
            self.assertEqual(eval_ids | train_ids, development_ids)
            self.assertEqual(
                fold["train_base_fact_ids_sha256"],
                protocol.sha256_value(sorted(train_ids)),
            )
            eval_components = {
                source_by_id[base_fact_id]["leakage_component_id"]
                for base_fact_id in eval_ids
            }
            train_components = {
                source_by_id[base_fact_id]["leakage_component_id"]
                for base_fact_id in train_ids
            }
            self.assertFalse(eval_components & train_components)
            all_eval_ids.extend(eval_ids)
        self.assertEqual(len(all_eval_ids), 96)
        self.assertEqual(Counter(all_eval_ids), Counter(development_ids))

        with mock.patch.object(protocol, "__file__", str(self.generator_path)):
            verified = protocol.verify_existing(
                project_root=self.root,
                static_manifest_path=self.static_manifest_path,
                protocol_document_path=self.protocol_document_path,
                output_dir=output_dir,
                expected_completed_proxy_run_id="fixture-proxy-run",
            )
        self.assertTrue(verified["verified"])
        self.assertFalse(verified["pnt_authorized"])

    def test_fixed_timestamp_generation_is_byte_deterministic_without_proxy_outputs(self):
        output_dir = self.root / "deterministic-output"
        self.assertFalse((self.root / "runs").exists())
        self._generate(output_dir)
        first_fold = (output_dir / "development_fold_manifest.json").read_bytes()
        first_protocol = (
            output_dir / "perturbation_protocol_manifest.json"
        ).read_bytes()
        protocol_manifest = json.loads(first_protocol)
        self.assertFalse(
            protocol_manifest["proxy_output_independence"]
            ["proxy_behavior_outputs_read_by_generator"]
        )
        self.assertFalse(
            protocol_manifest["proxy_output_independence"]["proxy_labels_used"]
        )
        self.assertEqual(
            protocol_manifest["cohort_contract"]["development_replay_population"],
            "all_96_without_proxy_filtering",
        )

        shutil.rmtree(output_dir)
        self._generate(output_dir)
        self.assertEqual(
            (output_dir / "development_fold_manifest.json").read_bytes(),
            first_fold,
        )
        self.assertEqual(
            (output_dir / "perturbation_protocol_manifest.json").read_bytes(),
            first_protocol,
        )

    def test_upstream_sha_tamper_fails_closed(self):
        with self.bundle_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        output_dir = self.root / "tampered-output"
        with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
            self._generate(output_dir)
        self.assertFalse(output_dir.exists())

    def test_existing_output_is_not_overwritten(self):
        output_dir = self.root / "existing-output"
        self._generate(output_dir)
        fold_path = output_dir / "development_fold_manifest.json"
        protocol_path = output_dir / "perturbation_protocol_manifest.json"
        before = (fold_path.read_bytes(), protocol_path.read_bytes())

        with self.assertRaisesRegex(ValueError, "output directory already exists"):
            self._generate(output_dir)

        self.assertEqual(
            (fold_path.read_bytes(), protocol_path.read_bytes()),
            before,
        )


if __name__ == "__main__":
    unittest.main()
