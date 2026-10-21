# CSC734 Information Retrieval — HW03

Rajkumar Kushwaha's Python information-retrieval project. The latest version
compares documents using the vector-space model and returns the **K closest
document pairs** for three weighting methods: TF, TF-IDF, and sublinear TF-IDF.

The HW01/HW02 inverted-index builder, Boolean query search, and parallel indexing
experiment remain available through the commands below.

## Setup

Requires **Python 3.10 or later**. NLTK is the only third-party runtime dependency.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Use `python3` if that is your interpreter's command name.

## Run HW03

From the repository root:

```bash
python main.py
```

Press Enter to use the included `documents/` and `stopwords.txt`, then enter K.
Alternatively, supply all inputs as command-line arguments:

```bash
python main.py --documents documents --stopwords stopwords.txt --k 20
```

The included collection contains **1,459 documents**, giving **1,063,611 distinct
pairs**. K must be between 1 and that pair count. K selects pairs across the
collection; it does not request K neighbors for every document.

Output includes student/course identification, processed vocabulary size, the
10 most frequent processed terms, three ranked pair lists, and execution time.
Results are printed to the terminal. To save a run:

```bash
python main.py --documents documents --stopwords stopwords.txt --k 20 > output.txt
```

## Calculation

Preprocessing retains the HW02 sequence: NLTK tokenization, lowercase, ASCII
punctuation removal, supplied stop-word removal, and Porter stemming.

For term `t` in document `d`:

| Method | Weight |
| --- | --- |
| TF | `tf(t,d)` |
| TF-IDF | `tf(t,d) * log10(N / df(t))` |
| Sublinear TF-IDF | `(1 + log10(tf(t,d))) * log10(N / df(t))` when TF is positive; otherwise zero |

`tf` is an occurrence count. `df` counts documents containing the term, once per
document. IDF is unsmoothed. Each method uses its own Euclidean vector lengths;
cosine similarity is the dot product of the resulting unit vectors.

Each distinct pair is evaluated once (`i < j`). Self-pairs and reversed
duplicates are excluded. Different files with identical processed vectors are
valid results with similarity 1. A zero vector's similarity is reported as 0,
an explicit convention for a mathematically undefined case. Exactly equal
computed scores use natural document order to break ties. Tiny floating-point
differences can change the order of mathematically tied pairs.

Weights, normalization, cosine calculations, and ranking are implemented directly.
No ready-made vectorizer or built-in cosine function is used. Sparse dictionaries,
an in-run stem cache, a shared scoring pass, and bounded heaps reduce repeated
work. Every run rereads and processes the corpus; no saved similarity results are
used.

## Verification and performance

Run the HW03 tests and earlier-homework compatibility tests:

```bash
python -m unittest discover -v
```

HW03 tests cover hand-calculated examples, an independent dense reference scorer,
top-K selection against exhaustive sorting, zero vectors, ties, input validation,
and relocation of the program. Compatibility tests check the legacy index builder,
Boolean retrieval, and a two-worker parallel indexing run.

During development, an independent full sort selected the same top 10,000 pairs
for each method. Score differences were at floating-point precision; reordered
results were mathematical ties. A measured full K=20 run took about 2.5–2.8 seconds
on the development machine, and a K=10,000 run took about 3.1 seconds externally
measured. These are hardware-dependent measurements, not runtime guarantees.

See [BENCHMARK.txt](BENCHMARK.txt) for the original benchmark methodology,
[sample_output.txt](sample_output.txt) for a K=10 run, and
[How_to_run.txt](How_to_run.txt) for detailed execution instructions.

## Earlier homework commands

`main.py` now runs HW03. The previous indexing program is preserved as
`hw02_index.py`; its preprocessing interface is used by the legacy tools.

| Task | Command |
| --- | --- |
| Build/save the original inverted index | `python hw02_index.py` |
| Run HW02 Boolean queries | `python search_program_for_user_query.py` |
| Preview Boolean query matches | `python search_program_for_user_query.py --preview` |
| Build an index using parallel preprocessing | `python main_parallel.py` |

For Boolean search, provide `inverted_index.json`, `stopwords.txt`, and
`sample_queries.txt` at the prompts. Results are written to `output.txt`.
The [HW02 instructions](docs/HW02_README.md) retain the earlier workflow details.

## Repository files

| Path | Purpose |
| --- | --- |
| `main.py` | HW03 document similarity program |
| `test_hw03.py` | HW03 correctness and input tests |
| `test_hw02_compatibility.py` | Regression tests for earlier homework commands |
| `hw02_index.py` | Preserved sequential HW01/HW02 index builder |
| `search_program_for_user_query.py` | HW02 Boolean query retrieval |
| `main_parallel.py` | Earlier parallel index builder |
| `requirements.txt` | Tested NLTK version |
| `documents/`, `stopwords.txt` | Input corpus and stop-word list |
| `How_to_run.txt` | Detailed HW03 setup and usage |
| `BENCHMARK.txt`, `sample_output.txt` | Development measurements and example output |

The existing `documents_big/` synthetic collection, saved indexes, and Conda
environment file are retained from earlier homework. HW03 defaults to `documents/`.
Local virtual environments, caches, editor files, output, and ZIP archives are
excluded from new Git additions by `.gitignore`.

No license is currently specified for this repository.
