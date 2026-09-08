import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_candidate_relation_mapping.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("build_candidate_relation_mapping", SCRIPT_PATH)
candidate_mapping = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(candidate_mapping)


def entry(relation_raw, answer_type="concept", count=1):
    return {
        "signature_id": f"sig-{relation_raw}-{answer_type}",
        "relation_raw": relation_raw,
        "relation_raw_normalized": relation_raw.lower(),
        "subject_type": "concept",
        "answer_type": answer_type,
        "direction": f"concept -> {answer_type}",
        "count": count,
        "examples": [],
    }


class CandidateRelationMappingTests(unittest.TestCase):
    def test_definition_directions_are_distinct(self):
        description_status = candidate_mapping.classify(entry("term for definition"))
        definition_status = candidate_mapping.classify(entry("definition"))
        self.assertEqual(description_status[1], "description_named_term")
        self.assertEqual(definition_status[1], "term_definition")

    def test_overloaded_classification_requires_class_like_answer_type(self):
        self.assertEqual(candidate_mapping.classify(entry("classification", "attribute"))[0], "ambiguous")
        self.assertEqual(
            candidate_mapping.classify(entry("classification", "biological category"))[1],
            "entity_classification",
        )

    def test_broad_relation_is_ambiguous(self):
        self.assertEqual(candidate_mapping.classify(entry("is"))[0], "ambiguous")

    def test_every_inventory_signature_receives_an_explicit_decision(self):
        inventory = {
            "inventory_version": "relation-inventory-v1",
            "entries": [entry("definition"), entry("is"), entry("unseen multi word relation")],
        }
        taxonomy = {
            "taxonomy_version": candidate_mapping.TAXONOMY_VERSION,
            "relations": candidate_mapping.RELATIONS,
        }
        mapping = candidate_mapping.build_mapping(inventory, taxonomy)
        self.assertEqual(len(mapping["mappings"]), len(inventory["entries"]))
        self.assertEqual(
            {item["normalization_status"] for item in mapping["mappings"]},
            {"mapped", "ambiguous", "out_of_taxonomy"},
        )


if __name__ == "__main__":
    unittest.main()
