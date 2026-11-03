import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_qwen3_open_repair import eligibility, fact_mean, metrics


class RepairGateTests(unittest.TestCase):
    def test_variants_do_not_inflate_fact_weight(self):
        self.assertEqual(fact_mean([("a",1.),("a",1.),("b",0.)]),.5)

    def test_unresolved_pairs_and_control_harm_cannot_pass(self):
        result={"cohort_inputs":480,"inputs":532,"unresolved_paired_inputs":1,
                "retention":{"original":{"paired_accuracy_gain":0.},"neutral":{"paired_accuracy_gain":0.},"targeted":{"paired_accuracy_gain":.1}},
                "resistant_unresolved":[],"resistant_facts":10,"resistant_harmed":[],
                "control_newly_harmed_facts":["control"],"control_unresolved_on_baseline_correct":[],
                "directed_generation_gain":.5,"directed_margin_gain":1.}
        constraints={"original_accuracy_drop_max":.05,"neutral_accuracy_drop_max":.05,"resistant_harm_rate_max":.05}
        reasons=eligibility(result,constraints)
        self.assertIn("unresolved_primary_pairs",reasons)
        self.assertIn("unrelated_control_harm",reasons)

    def test_known_resistant_harm_cannot_be_erased_by_another_missing_arm(self):
        baseline={key:{"base_fact_id":"fact","language":"zh","variant":arm,"generation_correct":True,
                       "designated_distractor_id":"foil"} for key,arm in [("a","original"),("b","neutral"),("c","targeted")]}
        records=[{"input_id":key,"base_fact_id":"fact","generation_correct":False if key=="a" else None} for key in baseline]
        result=metrics(records,baseline,{},set(),set(),{"fact":set(baseline)})
        self.assertEqual(result["resistant_harmed"],["fact"])


if __name__=="__main__":unittest.main()
