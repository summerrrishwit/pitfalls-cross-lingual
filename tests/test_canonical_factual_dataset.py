import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_canonical_factual_dataset.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_canonical_factual_dataset", SCRIPT_PATH)
canonical = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(canonical)


def record(model, status="extracted", answer="Paris", **overrides):
    base = {
        "source_id": "raw_1",
        "source_dataset": "mmlu",
        "source_subset": "geography",
        "source_path": "data/source/mmlu.json",
        "source_original_index": 1,
        "source_question": "What is the capital of France?",
        "source_choices": ["Paris", "Berlin", "Rome", "Madrid"],
        "source_answer": "Paris",
        "candidate_id": "raw_1",
        "source_snapshot": {"record": {"choices": ["Paris", "Berlin", "Rome", "Madrid"]}},
        "terminal_status": "completed",
        "extraction_status": status,
        "exclusion_reason": None if status == "extracted" else "not_factual",
        "subject": "France" if status == "extracted" else None,
        "subject_type": "country" if status == "extracted" else None,
        "relation_raw": "capital" if status == "extracted" else None,
        "answer": answer if status == "extracted" else None,
        "answer_type": "city" if status == "extracted" else None,
        "canonical_fact": "The capital of France is Paris." if status == "extracted" else None,
        "extraction_confidence": 0.95,
        "validation_errors": [],
        "extraction": {"model": model},
    }
    base.update(overrides)
    return base


class CanonicalTripleTests(unittest.TestCase):
    def test_dual_answer_consensus_preserves_source_choices(self):
        decision, output = canonical.canonicalize_pair(
            record(canonical.PREFERRED_MODEL), record(canonical.SECONDARY_MODEL, answer="the Paris")
        )
        self.assertEqual(decision["canonical_decision"], "keep")
        self.assertEqual(output["source_choices"], ["Paris", "Berlin", "Rome", "Madrid"])
        self.assertEqual(output["answer"], "Paris")

    def test_single_model_extraction_requires_codex_review(self):
        decision, output = canonical.canonicalize_pair(
            record(canonical.PREFERRED_MODEL),
            record(canonical.SECONDARY_MODEL, status="not_extractable"),
        )
        self.assertIsNone(output)
        self.assertEqual(decision["canonical_decision"], "pending_review")
        self.assertEqual(decision["review_case"], "single_model_extraction")

    def test_answer_conflict_requires_codex_review(self):
        decision, output = canonical.canonicalize_pair(
            record(canonical.PREFERRED_MODEL), record(canonical.SECONDARY_MODEL, answer="Lyon")
        )
        self.assertIsNone(output)
        self.assertEqual(decision["canonical_decision"], "pending_review")
        self.assertEqual(decision["review_case"], "dual_model_answer_conflict")

    def test_codex_accepts_single_model_extraction(self):
        adjudication = {
            "source_id": "raw_1",
            "review_case": "single_model_extraction",
            "codex_decision": "accept",
            "selected_model": canonical.PREFERRED_MODEL,
            "reason_code": "atomic_fact_supported",
            "rationale": "The question directly asks one stable fact about France.",
            "review_prompt_version": canonical.CANONICAL_REVIEW_PROMPT_VERSION,
        }
        decision, output = canonical.canonicalize_pair(
            record(canonical.PREFERRED_MODEL),
            record(canonical.SECONDARY_MODEL, status="not_extractable"),
            adjudication,
        )
        self.assertEqual(decision["canonical_decision"], "keep")
        self.assertEqual(output["canonical_source_model"], canonical.PREFERRED_MODEL)
        self.assertEqual(output["source_choices"], ["Paris", "Berlin", "Rome", "Madrid"])

    def test_codex_rejects_answer_conflict(self):
        adjudication = {
            "source_id": "raw_1",
            "review_case": "dual_model_answer_conflict",
            "codex_decision": "reject",
            "selected_model": None,
            "reason_code": "answer_scope_unresolved",
            "rationale": "The two answers differ materially and neither scope can be preferred safely.",
            "review_prompt_version": canonical.CANONICAL_REVIEW_PROMPT_VERSION,
        }
        decision, output = canonical.canonicalize_pair(
            record(canonical.PREFERRED_MODEL),
            record(canonical.SECONDARY_MODEL, answer="Lyon"),
            adjudication,
        )
        self.assertIsNone(output)
        self.assertEqual(decision["canonical_reason"], "codex_rejected_dual_model_answer_conflict")

    def test_pending_review_is_not_counted_as_excluded(self):
        decisions, outputs, summary = canonical.determine_canonical_triples(
            [record(canonical.PREFERRED_MODEL)],
            [record(canonical.SECONDARY_MODEL, status="not_extractable")],
        )
        self.assertEqual(decisions[0]["canonical_decision"], "pending_review")
        self.assertEqual(outputs, [])
        self.assertEqual(summary["pending_review_count"], 1)
        self.assertEqual(summary["excluded_record_count"], 0)


class DownstreamConstructionTests(unittest.TestCase):
    def test_codex_taxonomy_maps_long_type_relation(self):
        entry = {
            "relation_raw_normalized": "type of energy released",
            "answer_type": "energy type",
        }
        status, relation_id, _, _ = canonical.codex_classify(entry)
        self.assertEqual((status, relation_id), ("mapped", "entity_energy_form"))

    def test_prompt_and_distractors_are_deterministic(self):
        taxonomy = {
            "relations": [
                {
                    "id": "entity_location",
                    "prompt_template_en": "The location associated with {subject} is",
                }
            ]
        }
        normalized = {
            **record(canonical.PREFERRED_MODEL),
            "normalization_status": "mapped",
            "relation_normalized": "entity_location",
        }
        prompts, summary = canonical.generate_factual_prompts([normalized], taxonomy)
        self.assertEqual(summary["status_counts"], {"generated": 1})
        self.assertEqual(prompts[0]["prompt_en"], "The capital of France is")
        self.assertEqual(prompts[0]["prompt_source"], "canonical_fact_answer_suffix")
        grouped, expanded, distractor_summary = canonical.build_distractor_candidates(prompts)
        self.assertEqual(grouped[0]["source_wrong_options"], ["Berlin", "Rome", "Madrid"])
        self.assertEqual(distractor_summary["candidate_count"], 3)
        self.assertFalse(any(item["distractor_verified"] for item in expanded))


if __name__ == "__main__":
    unittest.main()
