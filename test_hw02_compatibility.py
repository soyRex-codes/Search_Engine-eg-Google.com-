"""Rajkumar Kushwaha | CSC734 | HW03 integration with earlier homework.

Check that promoting HW03 to main.py preserves the original index builder,
Boolean retrieval, and parallel index builder through hw02_index.py.
"""

from collections import Counter
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


if __name__ == "__main__":
    unittest.main()
