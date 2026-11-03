import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from run_qwen3_prefill_followup import estimate_bank, source_order


class SourceEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"fact-{i:02}" for i in range(72)]
        self.engine = SimpleNamespace(torch=torch,
            folds={fid: "training" for fid in self.ids},
            relations={fid: f"relation-{i%4}" for i, fid in enumerate(self.ids)})
        facts = {fid: torch.ones(4096) for fid in self.ids}
        self.directions = {"evaluation": {"paired": facts, "centered_icl": facts}}
        self.norms = {"evaluation": {fid: 100. for fid in self.ids}}

    def test_nested_fact_sources_are_stratified_and_repeatable(self):
        order = source_order(self.engine, self.ids, 0)
        self.assertEqual(order, source_order(self.engine, list(reversed(self.ids)), 0))
        for size in (18, 36, 72):
            counts = [sum(self.engine.relations[fid] == f"relation-{i}" for fid in order[:size]) for i in range(4)]
            self.assertLessEqual(max(counts)-min(counts), 1)
        self.assertTrue(set(order[:18]) < set(order[:36]) < set(order))

    def test_norm_uses_training_reference_and_source_fold_cannot_leak(self):
        bank, manifest = estimate_bank(self.engine, 28, self.directions, self.norms, 18, 0)
        self.assertAlmostEqual(bank["evaluation"]["paired"][28].norm().item(), 100., places=4)
        self.assertEqual(bank["evaluation"]["paired"][:28].abs().sum().item(), 0.)
        self.assertEqual(manifest["evaluation"]["source_facts"], 18)
        chosen = manifest["evaluation"]["source_fact_ids"][0]
        self.engine.folds[chosen] = "evaluation"
        with self.assertRaises(AssertionError):
            estimate_bank(self.engine, 28, self.directions, self.norms, 18, 0)


if __name__ == "__main__":
    unittest.main()
