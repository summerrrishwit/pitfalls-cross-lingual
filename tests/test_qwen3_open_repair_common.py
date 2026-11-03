import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from qwen3_open_repair_common import Engine, ResidualInjection, injection_vectors


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.model=torch.nn.Module()
        self.model.model=torch.nn.Module()
        self.model.model.layers=torch.nn.ModuleList([torch.nn.Identity()])

    def test_teacher_forcing_patches_only_answer_predicting_positions(self):
        source=torch.zeros(1,5,4)
        with ResidualInjection(self.model,{0:torch.ones(4)},3):
            result=self.model.model.layers[0](source)
        self.assertTrue(torch.equal(result[0,:2],torch.zeros(2,4)))
        self.assertTrue(torch.equal(result[0,2:],torch.ones(3,4)))
        self.assertTrue(torch.equal(source,torch.zeros_like(source)))
        self.assertEqual(len(self.model.model.layers[0]._forward_pre_hooks),0)

    def test_cached_decoding_injects_once_at_current_token(self):
        with ResidualInjection(self.model,{0:torch.ones(4)},30):
            result=self.model.model.layers[0](torch.zeros(1,1,4))
        self.assertTrue(torch.equal(result,torch.ones(1,1,4)))

    def test_zero_hook_is_exact_noop(self):
        original=torch.randn(1,7,4)
        with ResidualInjection(self.model,{0:torch.zeros(4)},5):
            result=self.model.model.layers[0](original)
        self.assertTrue(torch.equal(result,original))

    def test_random_matches_combined_norm_and_repeats(self):
        bank={"fold":{"task":torch.tensor([[3.,0.,0.,0.]]),"difference":torch.tensor([[0.,4.,0.,0.]])}}
        spec={"id":"combined","parts":[{"kind":"task","layer":0,"scale":1.},{"kind":"difference","layer":0,"scale":1.}]}
        first=injection_vectors(torch,bank,"fold",spec,"v3",0)[0]
        again=injection_vectors(torch,bank,"fold",spec,"v3",0)[0]
        self.assertTrue(torch.equal(first,again))
        self.assertAlmostEqual(first.norm().item(),5.,places=5)

    def test_literal_wrapper_does_not_accept_negated_gold(self):
        engine=Engine.__new__(Engine);engine.cache={}
        row={"base_fact_id":"fact","language":"en","factual_completion_prompt":"The answer is", "answer_groups":[{"option_id":"gold","aliases_en":["Paris"],"aliases_zh":["巴黎"]}]}
        self.assertTrue(engine.label(row,"The answer is Paris.")["generation_correct"])
        self.assertIsNone(engine.label(row,"The answer is not Paris.")["generation_correct"])


if __name__=="__main__":unittest.main()
