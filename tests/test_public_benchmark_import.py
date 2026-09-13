import gzip
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_public_benchmarks.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("import_public_benchmarks", SCRIPT_PATH)
importer = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT_SPEC.loader.exec_module(importer)


class PublicBenchmarkImportTests(unittest.TestCase):
    def test_global_mmlu_full_download_uses_page_cache(self):
        class Response:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "num_rows_total": 3,
                    "rows": [{"row_idx": index, "row": {"sample_id": str(index)}} for index in range(3)],
                }

        class Session:
            def __init__(self):
                self.call_count = 0

            def get(self, *args, **kwargs):
                self.call_count += 1
                return Response()

        spec = {
            "url": "https://example.invalid/rows",
            "revision": "a" * 40,
            "config": "en",
            "split": "test",
            "expected_record_count": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            session = Session()
            first, first_meta = importer.fetch_global_mmlu(
                session, spec, 1, None, 17, "global_mmlu", Path(directory)
            )
            second, second_meta = importer.fetch_global_mmlu(
                session, spec, 1, None, 17, "global_mmlu", Path(directory)
            )
        self.assertEqual(len(first), 3)
        self.assertEqual(first, second)
        self.assertEqual(first_meta["eligibility_scope"], "all_upstream_rows")
        self.assertEqual(second_meta["cache_hit_count"], 1)
        self.assertEqual(session.call_count, 1)

    def test_global_mmlu_maps_answer_label_to_choice_text(self):
        spec = {"revision": "a" * 40, "split": "test", "config": "en"}
        raw = [{"row_idx": 4, "row": {
            "sample_id": "history/test/4",
            "subject": "history",
            "subject_category": "humanities",
            "question": "Who wrote Hamlet?",
            "option_a": "Dante",
            "option_b": "William Shakespeare",
            "option_c": "Homer",
            "option_d": "Virgil",
            "answer": "B",
        }}]
        records, exclusions = importer.normalize_global_mmlu(raw, spec)
        self.assertEqual(exclusions, {})
        self.assertEqual(records[0]["answer"], "William Shakespeare")
        self.assertEqual(records[0]["choices"][1], "William Shakespeare")
        self.assertEqual(records[0]["upstream_id"], "history/test/4")

    def test_mkqa_keeps_short_open_answers_and_not_aliases_as_choices(self):
        spec = {
            "revision": "b" * 40,
            "accepted_answer_types": ["entity", "date", "number", "number_with_unit", "short_phrase"],
        }
        raw = [{
            "example_id": 7,
            "queries": {"en": "Who wrote Hamlet?"},
            "answers": {"en": [{
                "type": "entity",
                "text": "William Shakespeare",
                "aliases": ["Shakespeare"],
            }]},
        }]
        records, exclusions = importer.normalize_mkqa(raw, spec)
        self.assertEqual(exclusions, {})
        self.assertEqual(records[0]["source_format"], "open_qa")
        self.assertEqual(records[0]["choices"], [])
        self.assertEqual(records[0]["upstream_metadata"]["answer_aliases"], ["Shakespeare"])

    def test_mkqa_gzip_jsonl_parser(self):
        source = b'{"example_id": 1}\n{"example_id": 2}\n'
        self.assertEqual(
            [record["example_id"] for record in importer.parse_jsonl(gzip.decompress(gzip.compress(source)))],
            [1, 2],
        )

    def test_msqa_filters_to_english(self):
        spec = {"revision": "c" * 40, "language": "en-EN"}
        raw = [
            {"id": "EN-01", "language": "en-EN", "question": "Question?", "answer": "Answer", "category": "History"},
            {"id": "ZH-01", "language": "zh-ZH", "question": "问题？", "answer": "答案", "category": "History"},
        ]
        records, exclusions = importer.normalize_msqa(raw, spec)
        self.assertEqual([record["upstream_id"] for record in records], ["EN-01"])
        self.assertEqual(exclusions["non_target_language"], 1)

    def test_openbookqa_zip_member_is_jsonl(self):
        member = "dataset/test.jsonl"
        row = {
            "id": "1",
            "question": {
                "stem": "Which is a mammal?",
                "choices": [{"label": "A", "text": "whale"}, {"label": "B", "text": "trout"}],
            },
            "answerKey": "A",
        }
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(member, json.dumps(row) + "\n")
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            raw = importer.parse_jsonl(archive.read(member))
        records, exclusions = importer.normalize_openbookqa(
            raw, {"revision": "d" * 40, "split": "test"}
        )
        self.assertEqual(exclusions, {})
        self.assertEqual(records[0]["answer"], "whale")

    def test_deterministic_sample_is_source_specific(self):
        records = [{"upstream_id": str(index)} for index in range(20)]
        first = importer.deterministic_sample(records, 5, 17, "a")
        second = importer.deterministic_sample(list(reversed(records)), 5, 17, "a")
        other = importer.deterministic_sample(records, 5, 17, "b")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
