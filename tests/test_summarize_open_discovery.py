import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("discovery_summary", ROOT / "scripts/summarize_open_discovery.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiscoveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((ROOT / "configs/open_factual_discovery_v1.json").read_text())

    def candidate(self, fact="f1", attempt=1, positive=True):
        return {"candidate_id": f"{fact}-{attempt}", "base_fact_id": fact, "split_assignment": "development",
                "attempt_number": attempt, "generator_strategy_version": f"v{attempt}",
                "manipulation_family": "explicit_false_assertion_v1", "terminal_status": "completed",
                "static_review_passed": True, "original_correct": True, "neutral_correct": True,
                "targeted_correct": not positive, "targeted_distractor_hit": positive,
                "error_kind": "factual" if positive else "none", "replication_passed": positive}

    def request(self, candidates, request_id="r1", tokens=100):
        return {"request_id": request_id, "split_assignment": "development",
                "base_fact_ids": list({r["base_fact_id"] for r in candidates}),
                "candidate_ids": [r["candidate_id"] for r in candidates], "role": "target",
                "terminal_status": "completed", "total_tokens": tokens, "cost_usd": None}

    def test_batch_and_variants_do_not_inflate_independent_facts(self):
        first = self.candidate(positive=False)
        second = self.candidate(attempt=2)
        candidates = [first, second]
        result = module.summarize(self.policy, candidates, [self.request(candidates)])
        self.assertEqual(result["candidate_attempts"], 2)
        self.assertEqual(result["reproducible_defect_facts"], 1)
        self.assertEqual(result["request_attempts"], 1)

    def test_missing_pricing_and_tokens_are_unavailable(self):
        candidates = [self.candidate()]
        result = module.summarize(self.policy, candidates, [self.request(candidates, tokens=None)])
        self.assertIsNone(result["total_tokens"])
        self.assertIsNone(result["total_cost_usd"])
        self.assertIsNone(result["cost_usd_per_reproducible_fact"])
        self.assertEqual(result["token_budget_status"], "unverified_missing_usage")

    def test_retries_and_baseline_failures_count_in_budget(self):
        candidates = [self.candidate()]
        requests = [self.request(candidates)]
        retry = self.request(candidates, request_id="retry")
        retry["terminal_status"] = "failed"
        requests.append(retry)
        requests.append({"request_id": "baseline", "base_fact_ids": ["failed-baseline"], "candidate_ids": [],
                         "role": "baseline", "split_assignment": "development", "terminal_status": "completed", "total_tokens": 50})
        result = module.summarize(self.policy, candidates, requests)
        self.assertEqual(result["assigned_or_attempted_facts"], 2)
        self.assertEqual(result["request_attempts"], 3)
        self.assertEqual(result["reproducible_fact_yield"], 0.5)

    def test_unreviewed_or_unreproduced_errors_do_not_count_as_final_yield(self):
        one, two = self.candidate(), self.candidate("f2")
        one["replication_passed"] = None
        two["replication_passed"] = False
        two["static_review_passed"] = None
        result = module.summarize(self.policy, [one, two], [self.request([one, two])])
        self.assertEqual(result["open_defect_facts"], 1)
        self.assertEqual(result["reproducible_defect_facts"], 0)

    def test_non_target_factual_error_is_still_a_defect(self):
        candidate = self.candidate()
        candidate["targeted_distractor_hit"] = False
        result = module.summarize(self.policy, [candidate], [self.request([candidate])])
        self.assertEqual(result["reproducible_defect_facts"], 1)
        self.assertEqual(result["reproducible_directed_facts"], 0)

    def test_third_candidate_and_success_followup_are_rejected(self):
        cases = [[self.candidate(attempt=i, positive=False) for i in (1, 2, 3)],
                 [self.candidate(), self.candidate(attempt=2)]]
        for candidates in cases:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                module.summarize(self.policy, candidates, [self.request(candidates)])

    def test_feedback_cannot_use_missing_or_failed_baseline(self):
        for field, value in [("targeted_correct", None), ("original_correct", False)]:
            first = self.candidate(positive=False)
            first[field] = value
            candidates = [first, self.candidate(attempt=2)]
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.summarize(self.policy, candidates, [self.request(candidates)])

    def test_holdout_feedback_rejected(self):
        candidates = [self.candidate()]
        request = self.request(candidates)
        request["split_assignment"] = "validation"
        with self.assertRaisesRegex(ValueError, "Development"):
            module.summarize(self.policy, candidates, [request])

    def test_request_and_token_caps_rejected(self):
        candidates = [self.candidate()]
        policy = copy.deepcopy(self.policy)
        policy["max_model_request_attempts"] = 1
        with self.assertRaisesRegex(ValueError, "budget"):
            module.summarize(policy, candidates, [self.request(candidates), self.request(candidates, "r2")])
        with self.assertRaisesRegex(ValueError, "budget"):
            module.summarize(self.policy, candidates, [self.request(candidates, tokens=self.policy["max_total_tokens"] + 1)])

    def test_duplicate_requests_and_unaccounted_candidates_rejected(self):
        candidates = [self.candidate()]
        request = self.request(candidates)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.summarize(self.policy, candidates, [request, request])
        with self.assertRaisesRegex(ValueError, "accounting"):
            module.summarize(self.policy, candidates, [])


if __name__ == "__main__":
    unittest.main()
