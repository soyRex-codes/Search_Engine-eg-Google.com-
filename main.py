import nltk
import string
import unicodedata
from pathlib import Path
from nltk.stem import PorterStemmer
import json
from collections import Counter
from time import perf_counter

# Name: Rajkumar Kushwaha, Course: 734, Assignment: HW02

#fun1 - my intro
def my_info_and_course(): #function for name and course print
    """Print CSC734, student name, and homework number; return no value."""
    print('================== CSC734-IR Homework 02 ==================')
    print('First Name: Rajkumar')
    print('Last Name: Kushwaha')
    print('===========================================================================================')


# function1 - to display my name and course

# Translation table used to remove punctuation character form tokens
PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
r"""
The expression creates a character mapping table used to delete all standard punctuation marks
from a string via the Python String maketrans Reference.
How It Works:
• First Argument (""): An empty string indicating no characters are to be mapped or replaced.
• Second Argument (""): An empty string indicating no replacement strings are provided.
• Third Argument (string.punctuation): A string containing all ASCII punctuation marks
(! " # \$ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _  { | } ~`) designated for complete removal.
• Result: Passing this translation table to a string's translate() method
(e.g., text.translate(PUNCTUATION_TABLE)) efficiently strips out every punctuation mark in a single C-optimized operation
"""
def preprocess_text(text, stop_words): # reusable for both text and user_query normalization
    """
    Applies the same text processing steps that will be used for both
    documents and user queries.

    Input:
        text - raw document or query text.
        stop_words - set containing the provided stop words.

    Output:
        processed_tokens - normalized and stemmed tokens.
    """

    stemmer = PorterStemmer()

    # Step 1: Tokenize
    tokens = nltk.word_tokenize(text)

    processed_tokens = []

    for token in tokens:

        # Step 2: Lowercase
        token = token.lower()

        # Step 3: Remove punctuation anywhere inside the token.
        # Example: "U.S.A." becomes "usa".
        token = token.translate(PUNCTUATION_TABLE)

        # A punctuation-only token may become empty after translation.
        if not token:
            continue

        # Step 4: Remove stop words
        if token in stop_words:
            continue

        # Step 5: Stemming
        stemmed_token = stemmer.stem(token)

        processed_tokens.append(stemmed_token)

    return processed_tokens

# function2
def read_documents(folder_path):
    """Read .txt files in folder_path and return {file_name: document_text}."""
    # Keep each document separate so the index can record its document ID.
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

# function3 - load_stop_words
def load_stop_words(stop_words_file_path):
    """Read stop_words_file_path and return its words as a set."""
    # 2. tokenization of the document contents
    with open(stop_words_file_path, "r", encoding="utf-8") as f:
        stop_words = f.read().splitlines()  # reads the text file "stop_words.txt" and creates a list where each item is one line from the file.
    # A set is useful here because I repeatedly check whether a token
    # exists in the stop-word collection.
    return set(stop_words)  # because we're going to repeatedly ask token in stop_words; and sets are designed for fast membership checking.

# fun4 - normalization/preprocess_documents - Processes every document using the common preprocessing pipeline.
def preprocess_documents(documents, stop_words):
    """
    Processes every document using the common preprocessing pipeline.
    Input:
        documents - dictionary of {doc_id: raw text}.
        stop_words - set of provided stop words.
    Output:
        processed_documents - dictionary of {doc_id: processed tokens}.
    """
    processed_documents = {}
    for doc_id, text in documents.items():
        processed_tokens = preprocess_text(text, stop_words)
        processed_documents[doc_id] = processed_tokens
    return processed_documents


# fun 4.5
"""
for sorted inverted index:
sorted will look like this:
[
    "file_18.txt",
    "file_105.txt",
    "file_339.txt"
]
"""
def document_sort_key(doc_id):
    """Return the numeric suffix of a document ID such as file_339.txt."""
#file_339.txt -> 339
    return int(Path(doc_id).stem.split("_")[-1])

# fun 5
"""Builds the inverted index from the processed document collection.
Input: processed_documents - dictionary containing the processed tokens for every document.
Output: my_inverted_index - dictionary where each term points to the list of document IDs containing that term.
"""
def build_inverted_index(processed_documents):
    """Return term-to-sorted-document-ID postings from processed_documents."""
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
    for term in my_inverted_index:
        my_inverted_index[term].sort(key=document_sort_key)
    return my_inverted_index


#fun6
def display_index_size(my_inverted_index):
    """Print and return the size in bytes and MB of my_inverted_index."""
    index_to_json_text = json.dumps(my_inverted_index) # returns a JSON string
    index_to_bytes = index_to_json_text.encode('utf-8')
    index_size_in_bytes = len(index_to_bytes) # this gives number of bytes
    index_size_in_mb = (index_size_in_bytes/(1024*1024))
    # 1 MB = 1024 * 1024 bytes.
    print(f"Index size in bytes: {index_size_in_bytes} bytes")
    print(f"Index size in MB: {index_size_in_mb:.4f} MB")
    return index_size_in_bytes, index_size_in_mb

#fun7
def show_frequent_terms(processed_documents, n):
    """Print and return the n most frequent terms in processed_documents."""
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
def save_index(my_inverted_index, file_name):
    """Write my_inverted_index to file_name as JSON; return no value."""
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(my_inverted_index, file)
    print(f"Index saved to {file_name}")

# fun 9
def load_index(file_name):
    """Read file_name and return its saved inverted-index dictionary."""
    with open(file_name, "r", encoding="utf-8") as file:
        loaded_index = json.load(file)
    print(f"Index loaded from {file_name}")
    return loaded_index


def main():
    """Prompt for input paths, save the index, and print statistics; return none."""
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
