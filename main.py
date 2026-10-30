"""Rajkumar Kushwaha | CSC734 | HW03.

Find the global top K distinct document pairs using three cosine measures.
All weighting, normalization, cosine calculations, and ranking are implemented
here. NLTK supplies only tokenization and Porter stemming, as in HW02.
"""

# ========================================================================
# PART 1: Imports, timing, and shared data structures
# ========================================================================
from time import perf_counter

# Includes dependency imports in the reported elapsed time when run as a script.
PROGRAM_STARTED = perf_counter()

import argparse
from collections import Counter
from functools import lru_cache
import heapq
import math
from pathlib import Path
import re
import string
import sys

import nltk
from nltk.stem import PorterStemmer


# All three-value tuples use this order: TF, TF-IDF, then sublinear TF-IDF.
METHODS = ("tf", "tf_idf", "wf_idf")
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
# Resolve defaults from this script's location, so moving the folder is safe.
SCRIPT_DIR = Path(__file__).resolve().parent
# Type aliases describe the data passed between functions:
# Weights: three weights for one term, one per method.
# Vector: only the terms present in one document (a sparse representation).
# Pair: (similarity, index of the first document, index of the second document).
# Document indices refer to the same positions in paths, counts, and vectors.
Weights = tuple[float, float, float]
Vector = dict[str, Weights]
Pair = tuple[float, int, int]


# ========================================================================
# PART 2: identification, document discovery, and stop words
# ========================================================================
def my_info_and_course() -> None:
    """Print the course, assignment, and student identification; return None."""
    print("================== CSC734-IR Homework 03 ==================")
    print("First Name: Rajkumar")
    print("Last Name: Kushwaha")
    print("=========================================================")

# Sort key for natural filename order, not lexicographic.
def document_sort_key(path: Path) -> tuple:
    """Returns a natural filename key: file_2.txt precedes file_10.txt."""
    # Split digit groups from text so filenames sort numerically, not as text.
    # For example, "file_10.txt" contains the integer 10 rather than string "10".
    pieces = re.split(r"(\d+)", path.name)
    return (tuple((0, int(p)) if p.isdigit() else (1, p.casefold())
                  for p in pieces), path.name)

# document discovery
def discover_documents(folder: Path) -> list[Path]:
    """Validate a folder; return naturally sorted direct-child .txt paths."""
    if not folder.is_dir():
        raise ValueError(f"Document folder does not exist: {folder}")
    paths = sorted((p for p in folder.glob("*.txt") if p.is_file()),
                   key=document_sort_key)
    if len(paths) < 2:
        raise ValueError("At least two .txt documents are required.")
    return paths

# Read stop words from a file, normalize, and return as a set.
def load_stop_words(path: Path) -> set[str]:
    """Read a UTF-8, one-word-per-line file; return normalized stop words."""
    # A set supports repeated membership checks during preprocessing.
    # utf-8-sig also accepts an optional byte-order mark at the file's start.
    return {line.strip().lower() for line in path.read_text(
        encoding="utf-8-sig").splitlines() if line.strip()}


# ========================================================================
# PART 3: preprocessing and per-document term counts
# ========================================================================
def preprocess_text(text: str, stop_words: set[str], stem) -> list[str]:
    """Return compatible tokens using the supplied cached stem function.

    NLTK word_tokenize, lowercase, remove ASCII
    punctuation, remove stop words, Porter stem. Does not deduplicate tokens:
    their multiplicities are needed for term frequencies.
    """
    processed = []
    # Tokenization identifies word-like units before normalization.
    for token in nltk.word_tokenize(text):
        # Deleted punctuation inside tokens too: "U.S.A." becomes "usa".
        token = token.lower().translate(PUNCTUATION_TABLE)
        # Discard empty/stop-word tokens, then stem the remaining words.
        # Keep repeated stems: their counts determine TF, not just presence.
        if token and token not in stop_words:
            processed.append(stem(token))
    return processed

# Build a list of term counts, one Counter per document, in the same order.
def build_term_counts(paths: list[Path], stop_words: set[str]) -> list[Counter]:
    """Reads each path once; return per-document counts in the same order.

    Cache stems within this invocation only. Every run still reads, tokenizes,
    and counts all documents from scratch; there is no saved-index shortcut.
    """
    stemmer = PorterStemmer()
    # Repeated words reuse their stem; the cache is rebuilt for each run.
    cached_stem = lru_cache(maxsize=100_000)(stemmer.stem)
    counts = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Example: ["cat", "cat", "dog"] -> {"cat": 2, "dog": 1}.
        # counts[i] belongs to paths[i]; later stages preserve this alignment.
        counts.append(Counter(preprocess_text(text, stop_words, cached_stem)))
    return counts


# ========================================================================
# PART 4: Collection statistics, IDF, and normalized document vectors
# ========================================================================
# Computes collection frequency and document frequency from per-document counts.
def collection_statistics(counts: list[Counter]) -> tuple[Counter, Counter]:
    """Return (collection frequency, document frequency) for counted documents."""
    collection_frequency = Counter()
    document_frequency = Counter()
    for document in counts:
        # Counter.update(mapping) adds occurrence counts, not just the keys.
        # A term appearing 5 times in one document and 2 in another has CF=7.
        collection_frequency.update(document)
        # Updating with keys counts a term once per document, irrespective of TF.
        # In the example above its DF=2, which is the statistic used for IDF.
        document_frequency.update(document.keys())
    return collection_frequency, document_frequency

# Computes unsmoothed IDF weights for all observed terms in the corpus.
def compute_idf(document_frequency: Counter, n_documents: int) -> dict[str, float]:
    """Return unsmoothed log10(N / df) weights for observed vocabulary terms."""
    # IDF(t) = log10(N / DF(t)). N includes empty preprocessed documents.
    # These terms were observed, so DF is at least 1; no division by zero.
    # A term in every document has IDF=log10(1)=0. Do not add smoothing.
    return {term: math.log10(n_documents / frequency)
            for term, frequency in document_frequency.items()}

# Builds normalized sparse vectors for all three weighting methods.
def build_normalized_vectors(counts: list[Counter], idf: dict[str, float]
                             ) -> tuple[list[Vector], tuple[int, int, int]]:
    """Return sparse vectors holding (TF, TF-IDF, WF-IDF) normalized weights.

    Each method has its own Euclidean length. Empty/zero-weight vectors remain
    zero, defining their reported similarity to every document as zero.
    Also return the number of zero vectors for each method.
    """
    vectors: list[Vector] = []
    zero_counts = [0, 0, 0]
    for document in counts:
        # Create all three raw weights together for each observed term:
        # TF = count; TF-IDF = count * IDF;
        # WF-IDF = (1 + log10(count)) * IDF.
        # Counter entries here are positive, so log10(0) is never evaluated.
        weights = {
            term: (float(tf), tf * idf[term],
                   (1.0 + math.log10(tf)) * idf[term])
            for term, tf in document.items()
        }
        # Each method needs its OWN length: sqrt(sum of squared weights).
        # Include every term in the document, not only terms shared by a pair.
        # fsum reduces rounding error when adding the squared weights.
        lengths = [math.sqrt(math.fsum(w[m] ** 2 for w in weights.values()))
                   for m in range(3)]
        # Record zero vectors so the output can explain their special handling.
        for m in range(3):
            zero_counts[m] += lengths[m] == 0.0
        # Divide each weight by its method's length to produce unit vectors.
        # If a length is zero, leave that method's coordinates at zero instead
        # of dividing by zero. Later its reported cosine will therefore be 0.
        vectors.append({
            term: (w[0] / lengths[0] if lengths[0] else 0.0,
                   w[1] / lengths[1] if lengths[1] else 0.0,
                   w[2] / lengths[2] if lengths[2] else 0.0)
            for term, w in weights.items()
        })
    return vectors, (zero_counts[0], zero_counts[1], zero_counts[2])


# ========================================================================
# PART 5: Three cosine similarities from one shared-term scan
# ========================================================================
def cosine_scores(left: Vector, right: Vector) -> Weights:
    """Return the three normalized dot products, scanning the smaller vector.

    Sharing the overlap scan avoids three separate dictionary-intersection
    operations per pair. Absent terms have zero weight in every method.
    """
    # Cosine(A, B) = dot(A, B) / (length(A) * length(B)).
    # The vectors are already normalized, so only the dot products remain.
    # Swapping the inputs does not change cosine; scan the smaller dictionary.
    if len(left) > len(right):
        left, right = right, left
    tf = tf_idf = wf_idf = 0.0
    lookup = right.get
    for term, a in left.items():
        b = lookup(term)
        # A missing term has weight zero, so only shared terms contribute.
        # Accumulate the three methods together to reuse this dictionary lookup.
        if b is not None:
            tf += a[0] * b[0]
            tf_idf += a[1] * b[1]
            wf_idf += a[2] * b[2]
    # Correct tiny floating-point overshoots; do not round before ranking.
    return (min(1.0, max(0.0, tf)), min(1.0, max(0.0, tf_idf)),
            min(1.0, max(0.0, wf_idf)))


# ========================================================================
# PART 6: Validate K and retain the globally closest document pairs
# ========================================================================
def validate_k(k: object, pair_count: int) -> int:
    """Return a valid integer K; reject incorrect types or values outside the range."""
    # Accept an arbitrary input value so wrong types can be checked and rejected.
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= pair_count:
        raise ValueError(f"K must be an integer from 1 to {pair_count:,}.")
    return k


def find_top_pairs(vectors: list[Vector], k: int) -> list[list[Pair]]:
    """Return three sorted top-K lists of (score, left index, right index).

    Visit i < j exactly once. A size-K min-heap per method retains only the
    best pairs. Negative indices make the heap's worst tied item the pair
    with larger indices. Final order is score descending, then i, then j.
    Include zero scores when K requires them; there are no dropped pairs.
    """
    n = len(vectors)
    validate_k(k, n * (n - 1) // 2)
    # One heap per method. A min-heap keeps its SMALLEST item at heap[0],
    # which will be the worst of the currently retained K candidates.
    heaps = [[], [], []]
    # i < j excludes self-pairs and reversed duplicates. With N documents,
    # this visits N*(N-1)/2 pairs regardless of the requested K.
    for i in range(n - 1):
        for j in range(i + 1, n):
            scores = cosine_scores(vectors[i], vectors[j])
            for heap, score in zip(heaps, scores):
                # Tuples compare left to right. A higher score is better;
                # on an exact computed tie, smaller document indices are better.
                # Negating the indices makes that preference work in a min-heap.
                item = (score, -i, -j)
                if len(heap) < k:
                    # Fill the heap until it contains K candidates.
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    # Replace the worst candidate only when this pair is better.
                    heapq.heapreplace(heap, item)
    # A heap is not a fully sorted list. Sort only the retained K candidates,
    # best first, and convert the negative indices back to ordinary indices.
    return [[(score, -i, -j) for score, i, j in sorted(heap, reverse=True)]
            for heap in heaps]


# ========================================================================
# PART 7: Displays corpus statistics and the three ranked result lists
# ========================================================================
def display_statistics(paths: list[Path], collection_frequency: Counter) -> None:
    """Print document count, processed vocabulary size, and top 10 terms."""
    print(f"The number of documents is: {len(paths)}")
    print(f"The number of unique words is: {len(collection_frequency)}")
    print("The top 10 most frequent words are:")
    # Sort by occurrence count descending, then alphabetically for equal counts.
    top = sorted(collection_frequency.items(), key=lambda item: (-item[1], item[0]))[:10]
    for rank, (term, frequency) in enumerate(top, 1):
        print(f"{rank}. {term} ({frequency} occurrences)")
    if not top:
        print("(No terms remain after preprocessing.)")


def display_results(paths: list[Path], results: list[list[Pair]], k: int) -> None:
    """Print K pairs per weighting method; scores are rounded only for display."""
    print("=========================================================")
    print(f"The top {k} closest document pairs are:")
    for number, (method, pairs) in enumerate(zip(METHODS, results), 1):
        print(f"\n{number}. Using {method}:")
        # Translate stored document indices back into the original filenames.
        for rank, (score, i, j) in enumerate(pairs, 1):
            print(f"{rank}. {paths[i].name}, {paths[j].name} "
                  f"with similarity of {score:.8f}")


# ========================================================================
# PART 8: Input collection, pipeline execution, timing, and error handling
# ========================================================================
def parse_arguments() -> argparse.Namespace:
    """Return optional CLI inputs; omitted inputs are requested interactively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, help="folder containing .txt documents")
    parser.add_argument("--stopwords", type=Path, help="one-stop-word-per-line UTF-8 file")
    parser.add_argument("--k", type=int, help="number of distinct pairs per method")
    return parser.parse_args()


def main() -> int:
    """Collect inputs, run the full pipeline, print results; return exit status."""
    # Overall flow: input paths/K -> term counts -> IDF and normalized vectors
    # -> top-K pairs -> formatted results and timing.
    args = parse_arguments()
    my_info_and_course()
    input_seconds = 0.0

    def prompt(message: str) -> str:
        """Return a stripped answer while tracking time spent awaiting input."""
        # Exclude time spent typing, while keeping actual processing time.
        nonlocal input_seconds
        before = perf_counter()
        try:
            return input(message).strip()
        finally:
            input_seconds += perf_counter() - before

    try:
        # A. Use command-line values, or prompt for anything not supplied.
        # Pressing Enter uses data beside main.py, with no personal path stored.
        if args.documents is None:
            answer = prompt("Documents folder [Enter for documents]: ")
            args.documents = Path(answer) if answer else SCRIPT_DIR / "documents"
        if args.stopwords is None:
            answer = prompt("Stop-word file [Enter for stopwords.txt]: ")
            args.stopwords = Path(answer) if answer else SCRIPT_DIR / "stopwords.txt"

        # Discovers the actual file count before deciding the largest valid K.
        paths = discover_documents(args.documents.expanduser())
        pair_count = len(paths) * (len(paths) - 1) // 2
        if args.k is None:
            while True:
                try:
                    args.k = validate_k(int(prompt(f"Enter K (1 to {pair_count:,}): ")),
                                        pair_count)
                    break
                except ValueError:
                    print(f"Please enter an integer from 1 to {pair_count:,}.")
        else:
            validate_k(args.k, pair_count)

        # B. Reads/preprocess every document and derives the corpus statistics.
        before = perf_counter()
        stop_words = load_stop_words(args.stopwords.expanduser())
        counts = build_term_counts(paths, stop_words)
        collection_frequency, document_frequency = collection_statistics(counts)
        preprocessing_seconds = perf_counter() - before
        display_statistics(paths, collection_frequency)

        # C. Precomputes weights and lengths once, then compare every pair.
        before = perf_counter()
        idf = compute_idf(document_frequency, len(paths))
        vectors, zero_counts = build_normalized_vectors(counts, idf)
        weighting_seconds = perf_counter() - before
        before = perf_counter()
        results = find_top_pairs(vectors, args.k)
        ranking_seconds = perf_counter() - before
        # D. Prints the selected pairs for all three weighting methods.
        display_results(paths, results, args.k)
        if any(zero_counts):
            print("\nZero-vector counts (tf, tf_idf, wf_idf):", zero_counts)
            print("Similarity involving a zero vector is reported as 0.")

        print(f"\nDistinct pairs evaluated per method: {pair_count:,}")
        print(f"Reading, preprocessing, and statistics: {preprocessing_seconds:.3f} seconds")
        print(f"Weighting and normalization: {weighting_seconds:.3f} seconds")
        print(f"Pair comparison and ranking: {ranking_seconds:.3f} seconds")
        # Flushes result output before sampling the total elapsed time.
        sys.stdout.flush()
        elapsed = perf_counter() - PROGRAM_STARTED - input_seconds
        print(f"Total execution time (excluding user input): {elapsed:.3f} seconds", flush=True)
        return 0
    # Reports expected setup/input problems without a long traceback.
    except LookupError:
        print("Missing NLTK tokenizer data. Run:\n"
              "  python -m nltk.downloader punkt punkt_tab", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.", file=sys.stderr)
        return 1


# Run the interactive/CLI program only when executed directly, so importing
# this module in test_hw03.py does not start prompts or document processing.
if __name__ == "__main__":
    raise SystemExit(main())
