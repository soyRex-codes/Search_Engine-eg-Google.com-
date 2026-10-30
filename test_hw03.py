"""Rajkumar Kushwaha | CSC734 | Homework 03.

Run with: python -m unittest -v test_hw03
The reference scorer uses dense, unnormalized vectors and direct division,
independently of main.py's sparse normalized-vector implementation.
"""

from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import random
import string
import subprocess
import sys
import tempfile
import unittest

import nltk
from nltk.stem import PorterStemmer

import main as hw


def reference_scores(counts, i, j):
    """Compute three independent dense cosine scores from original counts."""
    vocabulary = sorted({term for row in counts for term in row})
    n = len(counts)
    idf = {term: math.log10(n / sum(term in row for row in counts))
           for term in vocabulary}
    result = []
    for method in range(3):
        dense = []
        for row in (counts[i], counts[j]):
            weights = []
            for term in vocabulary:
                tf = row.get(term, 0)
                if method == 0:
                    weight = tf
                elif method == 1:
                    weight = tf * idf[term]
                else:
                    weight = (1 + math.log10(tf)) * idf[term] if tf else 0
                weights.append(weight)
            dense.append(weights)
        a, b = dense
        denominator = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(
            math.fsum(x * x for x in b))
        numerator = math.fsum(x * y for x, y in zip(a, b))
        result.append(numerator / denominator if denominator else 0.0)
    return tuple(result)


class WeightingAndRankingTests(unittest.TestCase):
    def vectors(self, counts):
        """Build test inputs through the production weighting stage."""
        _, df = hw.collection_statistics(counts)
        return hw.build_normalized_vectors(counts, hw.compute_idf(df, len(counts)))

    def test_collection_frequency_is_not_document_frequency(self):
        cf, df = hw.collection_statistics([Counter(a=4, b=1), Counter(a=2)])
        self.assertEqual(cf, Counter(a=6, b=1))
        self.assertEqual(df, Counter(a=2, b=1))
        self.assertEqual(hw.compute_idf(df, 2)["a"], 0.0)
        self.assertAlmostEqual(hw.compute_idf(df, 2)["b"], math.log10(2))

    def test_hand_calculated_example(self):
        counts = [Counter(a=2, b=1), Counter(a=1, b=1), Counter(b=1, c=1)]
        vectors, zero = self.vectors(counts)
        self.assertEqual(zero, (0, 0, 0))
        expected_tf = {(0, 1): 3 / math.sqrt(10),
                       (0, 2): 1 / math.sqrt(10), (1, 2): 0.5}
        for pair, tf in expected_tf.items():
            scores = hw.cosine_scores(vectors[pair[0]], vectors[pair[1]])
            self.assertAlmostEqual(scores[0], tf, places=12)
            self.assertAlmostEqual(scores[1], 1 if pair == (0, 1) else 0, places=12)
            self.assertAlmostEqual(scores[2], 1 if pair == (0, 1) else 0, places=12)

    def test_sublinear_scaling_changes_relative_weights(self):
        counts = [Counter(a=10, b=1), Counter(a=1), Counter(b=1)]
        vectors, _ = self.vectors(counts)
        # a and b have equal IDF. log-scaled weights in doc 0 are (2, 1).
        scores = hw.cosine_scores(vectors[0], vectors[1])
        self.assertAlmostEqual(scores[0], 10 / math.sqrt(101), places=12)
        self.assertAlmostEqual(scores[1], 10 / math.sqrt(101), places=12)
        self.assertAlmostEqual(scores[2], 2 / math.sqrt(5), places=12)

    def test_identical_disjoint_and_proportional_vectors(self):
        counts = [Counter(a=2, b=1), Counter(a=2, b=1),
                  Counter(c=4), Counter(a=4, b=2)]
        vectors, _ = self.vectors(counts)
        for score in hw.cosine_scores(vectors[0], vectors[1]):
            self.assertAlmostEqual(score, 1.0, places=12)
        self.assertEqual(hw.cosine_scores(vectors[0], vectors[2]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(hw.cosine_scores(vectors[0], vectors[3])[0], 1.0)

    def test_zero_idf_and_empty_documents(self):
        vectors, zero = self.vectors([Counter(a=2), Counter(a=1)])
        self.assertEqual(zero, (0, 2, 2))
        self.assertEqual(hw.cosine_scores(vectors[0], vectors[1]), (1.0, 0.0, 0.0))
        vectors, zero = self.vectors([Counter(), Counter(), Counter()])
        self.assertEqual(zero, (3, 3, 3))
        expected = [(0.0, 0, 1), (0.0, 0, 2), (0.0, 1, 2)]
        self.assertEqual(hw.find_top_pairs(vectors, 3), [expected] * 3)

    def test_ties_and_zero_score_fill(self):
        vectors, _ = self.vectors([Counter(a=1), Counter(a=1),
                                  Counter(b=1), Counter(c=1)])
        for ranking in hw.find_top_pairs(vectors, 6):
            self.assertEqual(ranking[0], (1.0, 0, 1))
            self.assertEqual([(i, j) for _, i, j in ranking[1:]],
                             [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    def test_scores_against_independent_dense_reference(self):
        rng = random.Random(734)
        for trial in range(25):
            counts = [Counter({term: value for term in "abcdefgh"
                               if (value := rng.randrange(6)) > 0})
                      for _ in range(8)] + [Counter()]
            vectors, _ = self.vectors(counts)
            for i, j in combinations(range(len(counts)), 2):
                with self.subTest(trial=trial, pair=(i, j)):
                    actual = hw.cosine_scores(vectors[i], vectors[j])
                    expected = reference_scores(counts, i, j)
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b, places=12)

    def test_heap_selection_against_exhaustive_sort(self):
        rng = random.Random(3)
        counts = [Counter({term: value for term in "abcdefghijk"
                           if (value := rng.randrange(8)) > 0}) for _ in range(12)]
        vectors, _ = self.vectors(counts)
        all_scores = [(hw.cosine_scores(vectors[i], vectors[j]), i, j)
                      for i, j in combinations(range(len(vectors)), 2)]
        for k in (1, 7, 65, 66):
            actual = hw.find_top_pairs(vectors, k)
            for m in range(3):
                expected = sorted(((scores[m], i, j) for scores, i, j in all_scores),
                                  key=lambda item: (-item[0], item[1], item[2]))[:k]
                self.assertEqual(actual[m], expected)

    def test_invalid_k(self):
        """Reject invalid integer values as well as non-integer input types."""
        # The float and bool are intentional: neither is an acceptable K.
        for k in (0, -1, 4, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                hw.validate_k(k, 3)
        self.assertEqual(hw.validate_k(3, 3), 3)


class InputAndPreprocessingTests(unittest.TestCase):
    def test_preprocessing_matches_hw02_sequence(self):
        text = "The libraries, LIBRARY! U.S.A. searching-searches can't 2026 ..."
        stop_words = {"the", "ca"}
        stemmer = PorterStemmer()
        # Directly reproduce the supplied HW02 pipeline without the new cache.
        expected = []
        for token in nltk.word_tokenize(text):
            token = token.lower().translate(str.maketrans("", "", string.punctuation))
            if token and token not in stop_words:
                expected.append(stemmer.stem(token))
        self.assertEqual(hw.preprocess_text(text, stop_words, stemmer.stem), expected)

    def test_file_discovery_and_stopwords(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("file_10.txt", "file_2.txt", "file_1.txt", "ignore.md"):
                (root / name).write_text("", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "file_3.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual([p.name for p in hw.discover_documents(root)],
                             ["file_1.txt", "file_2.txt", "file_10.txt"])
            stop = root / "stopwords.dat"
            stop.write_text("\ufeffThe\n AND \n\n", encoding="utf-8")
            self.assertEqual(hw.load_stop_words(stop), {"the", "and"})

    def test_missing_folder_and_too_few_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                hw.discover_documents(Path(directory) / "missing")
            with self.assertRaises(ValueError):
                hw.discover_documents(Path(directory))

    def test_cli_success_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            (documents / "file_1.txt").write_text("cat cat dog", encoding="utf-8")
            (documents / "file_2.txt").write_text("cat dog", encoding="utf-8")
            stop = root / "stopwords.txt"
            stop.write_text("the\n", encoding="utf-8")
            command = [sys.executable, str(hw.SCRIPT_DIR / "main.py"),
                       "--documents", str(documents), "--stopwords", str(stop)]
            valid = subprocess.run(command + ["--k", "1"], capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(valid.stdout.count("with similarity of"), 3)
            self.assertIn("The number of unique words is: 2", valid.stdout)
            self.assertIn("Total execution time", valid.stdout)
            invalid = subprocess.run(command + ["--k", "2"], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 1)
            self.assertIn("K must be an integer", invalid.stderr)
            interactive = subprocess.run(command, input="abc\n0\n2\n1\n",
                                         capture_output=True, text=True)
            self.assertEqual(interactive.returncode, 0, interactive.stderr)
            self.assertEqual(interactive.stdout.count("Please enter an integer"), 3)

            # Relocate the program and use defaults from another working directory.
            relocated = root / "main.py"
            relocated.write_text((hw.SCRIPT_DIR / "main.py").read_text(encoding="utf-8"),
                                 encoding="utf-8")
            portable = subprocess.run([sys.executable, str(relocated)],
                                      input="\n\n1\n", cwd=root.parent,
                                      capture_output=True, text=True)
            self.assertEqual(portable.returncode, 0, portable.stderr)
            self.assertIn("Documents folder [Enter for documents]: ", portable.stdout)
            self.assertIn("Stop-word file [Enter for stopwords.txt]: ", portable.stdout)
            self.assertNotIn(str(root), portable.stdout)
            self.assertEqual(portable.stdout.count("with similarity of"), 3)


if __name__ == "__main__":
    unittest.main()
