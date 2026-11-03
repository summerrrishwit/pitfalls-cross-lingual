import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from run_qwen3_prefill_repair import PrefillIntervention, AnswerPositionIntervention


class Block(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class FakeModel:
    def __init__(self):
        self.model = type("Inner", (), {"layers": [Block()]})()


class PrefillHookTests(unittest.TestCase):
    def test_prefill_cached_decode_and_teacher_forcing_share_coordinate(self):
        model = FakeModel()
        block = model.model.layers[0]
        with PrefillIntervention(model, {0: torch.ones(2)}, 3):
            prefill = block(torch.zeros(1, 3, 2))
            decode = block(torch.zeros(1, 1, 2))
            teacher = block(torch.zeros(1, 6, 2))
        self.assertTrue(torch.equal(prefill[0, 2], torch.ones(2)))
        self.assertEqual(prefill.sum().item(), 2)
        self.assertEqual(decode.sum().item(), 0)
        self.assertTrue(torch.equal(teacher[:, :3], prefill))
        self.assertEqual(teacher[:, 3:].sum().item(), 0)
        self.assertEqual(len(block._forward_pre_hooks), 0)

    def test_replacement_does_not_add_or_mutate_source(self):
        model = FakeModel()
        source = torch.full((1, 3, 2), 4.)
        with PrefillIntervention(model, {0: torch.ones(2)}, 3, replace=True):
            result = model.model.layers[0](source)
        self.assertTrue(torch.equal(result[0, 2], torch.ones(2)))
        self.assertTrue(torch.equal(source, torch.full_like(source, 4.)))

    def test_delta_injection_reaches_bfloat_source_and_timing_control(self):
        model = FakeModel()
        target = torch.tensor([1.5, 2.0], dtype=torch.bfloat16)
        source = torch.tensor([1.5078125, 1.984375], dtype=torch.bfloat16)
        delta = source.float()-target.float()
        inputs = target.repeat(1, 3, 1)
        with PrefillIntervention(model, {0: delta}, 3):
            result = model.model.layers[0](inputs)
        self.assertTrue(torch.equal(result[0, 2], source))
        with AnswerPositionIntervention(model, {0: delta}, 3):
            cached = model.model.layers[0](target.repeat(1, 1, 1))
            teacher = model.model.layers[0](target.repeat(1, 5, 1))
        self.assertTrue(torch.equal(cached[0, 0], source))
        self.assertTrue(torch.equal(teacher[0, 2:], source.repeat(3, 1)))
        self.assertTrue(torch.equal(teacher[0, :2], target.repeat(2, 1)))


if __name__ == "__main__":
    unittest.main()
