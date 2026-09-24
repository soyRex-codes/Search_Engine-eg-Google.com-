import nltk
import string
import unicodedata
from pathlib import Path
from nltk.stem import PorterStemmer
import json
from collections import Counter
from time import perf_counter

# Name: Rajkumar Kushwaha, Course: 734, Assignment: HW01

# function1 - to display my name and course
def my_info_and_course(): #function for name and course print
    print('================================== CSC734-IR Homework 01 ==================================')
    print('First Name: Rajkumar')
    print('Last Name: Kushwaha')
    print('===========================================================================================')

# function2
    """
    Reads all .txt files from the document folder.
    Input: folder_path - path to the folder containing the text documents.
    Output: documents - dictionary where the file name is the document ID 
            and the value is the original text from that file.
    """
    # I kept every document separate because later the inverted index
    # needs to know which document each term came from.
def read_documents(folder_path):
    # 1. Reading and storing the documents and its contents
    documents = {}  # dictionary/hasmap for storing all text but with their file boundaries for inverted index

    folder = Path(folder_path)
    # Goes through only the .txt files in the document folder.
    for doc_id in folder.glob(
            "*.txt"):  # file_path or doc_id = "documents/file_1.txt" # for simple file only, this was for test and learning
        with open(doc_id, "r", encoding="utf-8") as f:
            text = f.read()  # reads the file and stores then as str
            # store the read documents text -> str
            key = doc_id.name # # I am using the actual file name as the document ID.
            documents[key] = text  # gets the file name
    return documents

# function3
"""
    Loads the stop-word list provided for the hw
    Input: stop_words_file_path - path to the stopwords text file.
    Output: A set containing the stop words.
"""
def load_stop_words(stop_words_file_path):
    # 2. tokenization of the document contents
    with open(stop_words_file_path, "r", encoding="utf-8") as f:
        stop_words = f.read().splitlines()  # reads the text file "stop_words.txt" and creates a list where each item is one line from the file.
    # A set is useful here because I repeatedly check whether a token
    # exists in the stop-word collection.
    return set(stop_words)  # because we're going to repeatedly ask token in stop_words; and sets are designed for fast membership checking.

# fun4
""" Processes each document before it is added to the inverted index.  The preprocessing steps used here are:
    punctuation separation, tokenization, lowercase conversion, stop-word removal, and Porter stemming.
    Input: documents - dictionary containing the original document text. stop_words - set containing the provided stop words.
    Output: processed_documents - dictionary containing the processed token list for each document.
    """
def preprocess_documents(documents, stop_words):
    # Creates the Porter stemmer once and reuse it for all documents.
    stemmer = PorterStemmer()
    processed_documents = {}

    for doc_id, text in documents.items():
        # Replace ASCII and Unicode punctuation with spaces before tokenization.
        # This separates words instead of joining them: "nine-month" becomes
        # "nine month". Apostrophes, slashes, decimal points, and curly quotes
        # follow the same rule. Stop-word removal is applied to the new tokens.
        normalized_text = "".join(
            " " if character in string.punctuation
            or unicodedata.category(character).startswith("P") else character
            for character in text
        )
        tokens = nltk.word_tokenize(normalized_text)

        # Lowercase and stop-word removal
        filtered_tokens = []
        # Converts everything to lowercase so terms such as
        # "Information" and "information" are treated the same.
        for token in tokens:
            token = token.lower()
            # Keep the token only if it is not in the provided stop-word list.
            if not token:
                continue

            if token not in stop_words:
                filtered_tokens.append(token)

        # Stemming
        # Reduced related forms of words to a common stem.
        # Example: "libraries" may become "librari".
        stemmed_tokens = []

        for token in filtered_tokens:
            stemmed_token = stemmer.stem(token)
            stemmed_tokens.append(stemmed_token)
        # Preserves the document boundary after preprocessing.
        processed_documents[doc_id] = stemmed_tokens

    return processed_documents


# fun 5
"""Builds the inverted index from the processed document collection. 
Input: processed_documents - dictionary containing the processed tokens for every document.
Output: my_inverted_index - dictionary where each term points to the list of document IDs containing that term.
"""
def build_inverted_index(processed_documents):
    # we must not add the same doc_id multiple times when a term repeats inside one document
    my_inverted_index = {}
    for doc_id, tokens in processed_documents.items():
        # I only need one occurrence of a document ID in a term's postings list.
        # For example, ["cat", "run", "run"] becomes {"cat", "run"} for this step.
        non_duplicate_terms_within_document = set(tokens) # eg: this turns ["cat", "run", "run"] into ["cat", "run"]
        for term in non_duplicate_terms_within_document:
            # First time seeing a term, create its postings list.
            if term not in my_inverted_index:
                # Add the current document to the term's postings list.
                my_inverted_index[term] = []
            my_inverted_index[term].append(doc_id)
    return my_inverted_index


#fun6
""" Calculates and displays the size of the inverted index.
Input: my_inverted_index - completed inverted index.
Output:Returns the index size in bytes and megabytes. The values are also printed for the assignment output.
"""
def display_index_size(my_inverted_index):
    index_to_json_text = json.dumps(my_inverted_index) # returns a JSON string
    index_to_bytes = index_to_json_text.encode('utf-8')
    index_size_in_bytes = len(index_to_bytes) # this gives number of bytes
    index_size_in_mb = (index_size_in_bytes/(1024*1024))
    """
       1 KB = 1024 bytes 
       1 MB = 1024 × 1024 bytes
    """
    print(f"Index size in bytes: {index_size_in_bytes} bytes")
    print(f"Index size in MB: {index_size_in_mb:.4f} MB")
    return index_size_in_bytes, index_size_in_mb

#fun7
"""Finds and displays the n most frequent terms in the entire collection.
Input: processed_documents - processed tokens from all documents. n - number of frequent terms to display.
Output: top_terms - list containing the most frequent terms and their collection frequencies."""
def show_frequent_terms(processed_documents, n):
    term_frequency = Counter()
    # Counter is updated using every token occurrence from every document.
    # This gives collection frequency, not just the number of documents containing the term.
    for tokens in processed_documents.values():
        term_frequency.update(tokens)
    top_terms = term_frequency.most_common(n)
    print(f"\nTop {n} frequent terms:")
    for term, frequency in top_terms:
        print(f"{term}: {frequency}")
    return top_terms

#fun8
""" Saves the completed inverted index as a JSON file.
Input: my_inverted_index - inverted index to save. file_name - name/path of the output JSON file.
Output: No return value. The index is written to disk."""
def save_index(my_inverted_index, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(my_inverted_index, file)
    print(f"Index saved to {file_name}")

# fun 9
""" Loads an inverted index that was previously saved as JSON.
Input: file_name - path to the saved index file.
Output:loaded_index - reconstructed inverted-index dictionary."""
def load_index(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        loaded_index = json.load(file)
    print(f"Index loaded from {file_name}")
    return loaded_index


def main():
    # functions calls
    my_info_and_course() # function call to print name and course

    # user input for documents and stopwords paths
    folder_path = input("Enter the folder path:")
    stop_words_file_path = input("Enter the stop words file path:")

    # Start after entering the paths so my typing time is not counted.
    start_time = perf_counter()

    documents = read_documents(folder_path)
    stop_words = load_stop_words(stop_words_file_path)

    # function calls
    processed_documents = preprocess_documents(documents, stop_words)
    my_inverted_index = build_inverted_index(processed_documents)
    display_index_size(my_inverted_index)
    show_frequent_terms(processed_documents,10)
    save_index(my_inverted_index,"inverted_index.json")
    loaded_index = load_index("inverted_index.json")

    # Include reading, processing, indexing, displaying, saving, and loading.
    elapsed_time = perf_counter() - start_time
    print(f"\nTotal execution time: {elapsed_time:.3f} seconds")

if __name__ == '__main__':
    main()