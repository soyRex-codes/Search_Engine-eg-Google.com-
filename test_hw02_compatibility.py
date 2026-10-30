"""Rajkumar Kushwaha | CSC734 | HW03 integration with earlier homework.

Check that promoting HW03 to main.py preserves the original index builder,
Boolean retrieval, and parallel index builder through hw02_index.py.
"""

from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import hw02_index
import main_parallel
import search_program_for_user_query as boolean_search


class EarlierHomeworkCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "file_1.txt": "The cat cat dog",
            "file_2.txt": "cat fish",
            "file_3.txt": "dog fish",
        }
        self.stop_words = {"the"}
        processed = hw02_index.preprocess_documents(self.documents, self.stop_words)
        self.index = hw02_index.build_inverted_index(processed)

    def test_original_index_builder(self):
        self.assertEqual(self.index, {
            "cat": ["file_1.txt", "file_2.txt"],
            "dog": ["file_1.txt", "file_3.txt"],
            "fish": ["file_2.txt", "file_3.txt"],
        })

    def test_boolean_search_uses_legacy_preprocessing(self):
        self.assertIs(boolean_search.preprocess_text, hw02_index.preprocess_text)
        self.assertEqual(boolean_search.evaluate_boolean_query(
            ["cat", "dog"], ("AND",), self.index), ["file_1.txt"])
        self.assertEqual(boolean_search.evaluate_boolean_query(
            ["cat", "dog"], ("OR",), self.index),
            ["file_1.txt", "file_2.txt", "file_3.txt"])

    def test_parallel_builder_matches_sequential_index(self):
        parallel_index, frequencies = main_parallel.build_index_parallel(
            self.documents, self.stop_words, workers=2, batch_size=1)
        self.assertEqual(parallel_index, self.index)
        self.assertEqual(frequencies, Counter(cat=3, dog=2, fish=2))


class ParallelBooleanIndexRegressionTests(unittest.TestCase):
    def test_total_order_includes_replica_suffixes(self):
        ids = ["file_10.txt", "file_2_c1.txt", "file_1_c2.txt", "file_2.txt",
               "file_1_c0.txt", "file_1.txt"]
        self.assertEqual(sorted(ids, key=hw02_index.document_sort_key), [
            "file_1.txt", "file_1_c0.txt", "file_1_c2.txt", "file_2.txt",
            "file_2_c1.txt", "file_10.txt",
        ])
        # Leading zeros must not give distinct filenames identical sort keys.
        self.assertNotEqual(hw02_index.document_sort_key("file_1.txt"),
                            hw02_index.document_sort_key("file_01.txt"))

    def test_unsupported_ids_have_clear_errors(self):
        for doc_id in ("notes.txt", "file_1_cx.txt", "folder/file_1.txt", 123):
            with self.subTest(doc_id=doc_id), self.assertRaisesRegex(
                    ValueError, "Unsupported document ID"):
                hw02_index.document_sort_key(doc_id)

    def test_shuffled_multibatch_indexes_and_boolean_results(self):
        fixtures = [
            {"file_10.txt": "cat fish", "file_2.txt": "dog fish", "file_1.txt": "cat dog"},
            {"file_10_c1.txt": "cat fish", "file_2_c1.txt": "dog fish",
             "file_1.txt": "cat dog", "file_2.txt": "cat fish", "file_1_c0.txt": "dog fish"},
        ]
        for documents in fixtures:
            with self.subTest(ids=list(documents)):
                processed = hw02_index.preprocess_documents(documents, set())
                sequential = hw02_index.build_inverted_index(processed)
                parallel, _ = main_parallel.build_index_parallel(
                    documents, set(), workers=2, batch_size=1)
                self.assertEqual(parallel, sequential)
                cat, dog, fish = (set(sequential[t]) for t in ("cat", "dog", "fish"))
                expected = {
                    ("AND", "AND"): cat & dog & fish,
                    ("AND", "OR"): (cat & dog) | fish,
                    ("OR", "AND"): cat | (dog & fish),
                    ("OR", "OR"): cat | dog | fish,
                }
                for operators, matching in expected.items():
                    self.assertEqual(boolean_search.evaluate_boolean_query(
                        ["cat", "dog", "fish"], operators, parallel),
                        sorted(matching, key=hw02_index.document_sort_key))

    def test_both_loaders_sort_deduplicate_and_preserve_file(self):
        raw = {"cat": ["file_10.txt", "file_2_c1.txt", "file_2.txt", "file_2_c1.txt"],
               "dog": ["file_2_c1.txt", "file_1.txt"]}
        expected = {"cat": ["file_2.txt", "file_2_c1.txt", "file_10.txt"],
                    "dog": ["file_1.txt", "file_2_c1.txt"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            for loader in (hw02_index.load_index, boolean_search.load_inverted_index):
                with self.subTest(loader=loader.__name__), redirect_stdout(io.StringIO()):
                    normalized = loader(path)
                self.assertEqual(normalized, expected)
                self.assertEqual(boolean_search.evaluate_boolean_query(
                    ["cat", "dog"], ("AND",), normalized), ["file_2_c1.txt"])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), raw)

    def test_malformed_indexes_are_rejected(self):
        for raw in ([], {123: []}, {"cat": "file_1.txt"}, {"cat": [123]},
                    {"cat": ["notes.txt"]}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                hw02_index.normalize_inverted_index(raw)


if __name__ == "__main__":
    unittest.main()
