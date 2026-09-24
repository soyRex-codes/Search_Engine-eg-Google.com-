# Name: Rajkumar Kushwaha, Course: 734, Assignment: HW01
# Extra credit: process groups of documents in parallel using processes.

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter

# I reuse the functions from main.py so both versions process text the same way.
from main import (
    build_inverted_index,
    display_index_size,
    load_index,
    load_stop_words,
    my_info_and_course,
    preprocess_documents,
    read_documents,
    save_index,
)


# function1 - divide the documents into smaller groups
def document_batches(documents, batch_size=64):
    """Input: document dictionary and maximum number of documents per group.
    Output: a list of smaller document dictionaries.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batches = []
    current_batch = {}
    for doc_id, text in documents.items():
        # Keep the file name with its text, just like in main.py.
        current_batch[doc_id] = text
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = {}

    # Include the last group even if it has fewer documents than batch_size.
    if current_batch:
        batches.append(current_batch)
    return batches


# function2 - the work that each process does for one group
def process_batch(documents, stop_words):
    """Input: one group of documents and the provided stop-word set.
    Output: this group's inverted index and term-frequency Counter.
    """
    processed_documents = preprocess_documents(documents, stop_words)
    partial_index = build_inverted_index(processed_documents)
    term_frequency = Counter()
    for tokens in processed_documents.values():
        # For ["book", "book", "cat"], count book twice and cat once.
        # The index still lists each document only once for each term.
        term_frequency.update(tokens)
    return partial_index, term_frequency


# function3 - starts the processes and combines their results
def build_index_parallel(documents, stop_words, workers=4, batch_size=64):
    """Input: all documents, stop words, process limit, and group size.
    Output: the complete inverted index and term counts for all documents.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    batches = document_batches(documents, batch_size)
    my_inverted_index = {}
    term_frequency = Counter()
    if not batches:
        return my_inverted_index, term_frequency

    # Uses up to four processes by default, or fewer if there are fewer groups.
    number_of_processes = min(workers, len(batches))
    # "spawns" starts separate Python processes, including on my Mac.
    with ProcessPoolExecutor(
        max_workers=number_of_processes,
        mp_context=get_context("spawn"),
    ) as executor:
        tasks = []
        for batch in batches:
            # Submits every group first so processes can work at the same time.
            task = executor.submit(process_batch, batch, stop_words)
            tasks.append(task)

        # Reads results in the same order as the original groups. This keeps
        # documents order and the order of terms with equal counts consistent.
        for task in tasks:
            partial_index, partial_frequency = task.result()
            for term, doc_ids in partial_index.items():
                if term not in my_inverted_index:
                    my_inverted_index[term] = []
                # Each document belongs to one group, so its ID is not repeated.
                my_inverted_index[term].extend(doc_ids)

            # If two groups count "book" 3 and 5 times, the total becomes 8.
            term_frequency.update(partial_frequency)

    return my_inverted_index, term_frequency


# function4 - shows the most frequent terms after combining all groups
def display_frequent_terms(term_frequency, n):
    """Input: the combined term counts and number of terms to show.
    Output: prints and returns the top n terms with their counts.
    """
    top_terms = term_frequency.most_common(n)
    print(f"\nTop {n} frequent terms:")
    for term, frequency in top_terms:
        print(f"{term}: {frequency}")
    return top_terms


# function5 - runs the parallel version of HW01
def main():
    """Input: document and stop-word paths entered by the user.
    Output: printed results and inverted_index_parallel.json; no return value.
    """
    my_info_and_course()
    folder_path = input("Enter the folder path:")
    stop_words_file_path = input("Enter the stop words file path:")
    documents = read_documents(folder_path)
    if not documents:
        raise ValueError("No .txt documents found in the supplied folder")
    stop_words = load_stop_words(stop_words_file_path)

    start = perf_counter()
    my_inverted_index, term_frequency = build_index_parallel(
        documents, stop_words, workers=4
    )
    print(f"Parallel processing and merge time: {perf_counter() - start:.3f} seconds")

    # Prints and save after all groups have been combined into one index.
    display_index_size(my_inverted_index)
    display_frequent_terms(term_frequency, 10)
    save_index(my_inverted_index, "inverted_index_parallel.json")
    loaded_index = load_index("inverted_index_parallel.json")
    if loaded_index != my_inverted_index:
        raise RuntimeError("The saved index does not match the in-memory index")


# Child processes import this file. This check prevents them from asking
# for input or starting another set of processes when they import it.
if __name__ == "__main__":
    main()
