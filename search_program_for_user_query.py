import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from main import preprocess_text, load_stop_words, document_sort_key, my_info_and_course # reusable functions from main
from itertools import product # product() gives us every possible combination of choices.

# Rajkumar Kushwaha, CSC734 Information Retrieval, Homework 02

""" importing it will not run the HW1 program. Python will only make its functions available."""
"""
# file_flow

load inverted_index.json (done)
        ↓
load stopwords (done)
        ↓
load queries.txt (done)
        ↓
for each query (done)
        ↓
preprocess_text(query) (done)
        ↓
generate every AND/OR combination(done)
        ↓
Boolean retrieval using postings
        ↓
display matching documents
"""

# load inverted index
# fun1 - load inverted_index.json
""" Loads an inverted index that was previously saved as JSON.
Input: file_name - path to the saved index file.
Output: inverted_index - previously saved inverted index.
    The postings lists were sorted when the index was built in HW1."""
def load_inverted_index(file_name): ## (file_name= "inverted_index.json")
    """Read the saved JSON index and return its term-to-postings dictionary."""
    with open(file_name, "r", encoding="utf-8") as file:
        inverted_index = json.load(file)
    print(f"Index loaded from {file_name}")
    return inverted_index

#fun2 - load stop words - possibly unless we can reuse

# fun3 - load queries.txt from a file
"""
Function to load queries
Input: file_name - path to the queries file.
Output: queries - list containing each non-empty query as a string.
"""
def load_queries(query_file_path):
    """Return the non-empty query lines from query_file_path in file order."""
    queries = []
    with open(query_file_path, "r", encoding="utf-8") as file:
        for line in file:
            query = line.strip()

            if query:
                queries.append(query)
    return queries




# fun 4- generate every AND/OR combination
def generate_query_combinations(terms):
    """generate all possible combinations between query terms
    Input: terms  - list of query terms
    output: combinations - list of all possible combinations
    """
    combinations = [] # to store all combinations
    #check1: no terms means no query to generate
    if not terms:
        return combinations

    #check2: one term query has no boolean operators positions
    if len(terms) ==1:
        combinations.append((terms[0], ())) # add to the list of combinations found
        return combinations

    # for n terms: n-1 positions exists for AND/OR
    number_of_operator_positions = len(terms)-1
    operator_combinations = product(["AND", "OR"], repeat = number_of_operator_positions)

    for operators in operator_combinations:
        query_parts = [terms[0]] #term1 #parts of query terms starting at index 0

        for i, operator in enumerate(operators): # first iteration: i = 0, operator = "AND" , second iteration: i = 1 & operator = "AND"
            # We add: operator & terms[i+1]
            query_parts.append(operator) # term1 + operator
            query_parts.append(terms[i+1]) # term1 + operator + term2
            # first iteration: above query result : ["document", "AND", "index"]
            # second iteration: ["document", "AND", "index", "AND", "search"]
        boolean_query = " ".join(query_parts) # turns a list into a single string # eg: this ["document", "AND", "index", "AND", "search"] becomes document AND index AND search
        combinations.append((boolean_query, operators))
    return combinations

# AND: intersection of sorted postings
def intersect_postings(postings1, postings2):
    """
    looks documents that appear in both sorted postings lists.
    Input: postings1 - list of first sorted documents IDs and postings2 - second list of sorted documents IDs.
    output: intersected_postings - list of sorted postings IDs appearing in both sorted postings
    """
    intersection = []
    i = 0
    j = 0
    while i<len(postings1) and j<len(postings2):
        doc1 = postings1[i]
        doc2 = postings2[j]

        if doc1 == doc2:
            intersection.append(doc1)
            i += 1
            j += 1

        elif document_sort_key(doc1) < document_sort_key(doc2):
            i +=1
        else:
            j +=1
    return intersection


# next: for OR: union of sorted postings
# we want every document appearing in either postings list, without duplicates.
"""
input:
A → [2, 5, 8, 12]
B → [1, 5, 7, 8, 15]
Output: [1, 2, 5, 7, 8, 12, 15]
"""
def union_postings(postings1, postings2):
    """
    combines two sorted postings lists for a boolean OR operator
    Input: postings1 - list of first sorted documents IDs and postings2 - second list of sorted documents IDs.
    Output: union - sorted list containing documents IDs from either list without duplicates
    """
    union = [] # to store unions
    i = 0
    j = 0
    while i < len(postings1) and j < len(postings2):
        doc1 = postings1[i]
        doc2 = postings2[j]

        if doc1 == doc2:
            # same documents appear in both lists, so add it only once
            union.append(doc1)
            i += 1
            j += 1
        elif document_sort_key(doc1) < document_sort_key(doc2):
            # doc1 comes first numerically, so add it and move pointer i.
            """
            union.append(doc1) : Because doc1 is numerically smaller than doc2,
            it is the next chronological document to be added to the result set.
            Appending it now ensures our final union array maintains a strict sorted order.
            """
            union.append(doc1) # (Term_A OR Term_B), a document is considered a valid match if it appears in either list.
            i +=1
        else: # doc2 comes first numerically, so add it and move pointer j.
            union.append(doc2)
            j +=1
    # one postings list may still contain documents after the other ends
    while i < len(postings1):
        union.append(postings1[i])
        i +=1
    # If postings2 still has documents left, add all of them.
    while j < len(postings2):
        union.append(postings2[j])
        j += 1
    return union


# this function is for connecting each processed term to its real postings list
# To Retrieve postings for real query terms
def get_postings(term, inverted_index):
    """Return the sorted postings for term, or an empty list if it is absent."""
    return inverted_index.get(term, [])


def evaluate_boolean_query(terms, operators, inverted_index):
    """Evaluates one Boolean query combination.
    Boolean precedence: 1. AND 2. OR
    Input: terms - list of processed query terms. operators - tuple containing
            AND/OR operators. inverted_index - loaded inverted index.
    Output: result - sorted list of matching document IDs.
    """
    if not terms:
        return []
    # Start the first AND group with the first term.
    current_group = get_postings(terms[0], inverted_index)
    # This will hold completed AND groups separated by OR.
    groups = []
    for i, operator in enumerate(operators):
        next_term = terms[i + 1]
        next_postings = get_postings(next_term, inverted_index)
        if operator == "AND":
            # AND has higher precedence, so combine it
            # with the current group immediately.
            current_group = intersect_postings(
                current_group,
                next_postings
            )
        elif operator == "OR":
            # OR separates AND groups.
            groups.append(current_group)
            # Start a new group with the next term.
            current_group = next_postings
    # Add the final group.
    groups.append(current_group)
    # Start with the first group.
    result = groups[0]
    # OR all completed groups together.
    for group in groups[1:]:
        result = union_postings(result, group)
    return result

def display_query_results(boolean_query, results, show_all_results):
    """Display one expression and its sorted matching document IDs.
    Input: boolean_query - generated expression, results - matching document IDs,
           show_all_results - whether to print every ID or a short preview.
    Output: formatted result text; this function does not return a value.
    """
    display_query = boolean_query.replace(" AND ", " and ").replace(" OR ", " or ")
    print(f"{'=' * 16} Results for: {display_query} {'=' * 16}")

    if show_all_results:
        if results:
            for doc_id in results:
                print(doc_id)
        else:
            print("No matching documents")
    else:
        print("Number of results:", len(results))
        print("First 10 results:", results[:10])

    print("=" * 50)


def main(show_all_results=True):
    """Load the index and queries, then save every Boolean combination.
    Input: show_all_results - print all document IDs by default for HW02;
           False prints a short debugging preview.
    Output: output.txt beside this script, replaced on each run; no return value.
    """
    # Keep prompts visible in the console before redirecting the result text.
    index_file = input("Enter the inverted index file path: ")
    stop_words_file = input("Enter the stop words file path: ")
    query_file = input("Enter the query file path: ")

    output_path = Path(__file__).resolve().with_name("output.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(output_file):
            my_info_and_course()
            inverted_index = load_inverted_index(index_file)
            stop_words = load_stop_words(stop_words_file)
            queries = load_queries(query_file)

            # Process each query and write its Boolean results to output.txt.
            for query_number, query in enumerate(queries, start=1):
                if len(query.split()) > 4:
                    print("Skipping query with more than 4 words:", query)
                    continue

                processed_query = preprocess_text(query, stop_words)
                print(f"{'=' * 18} User Query {query_number}: {query} {'=' * 18}")
                if not show_all_results:
                    print("Processed query:", processed_query)

                if not processed_query:
                    print("No searchable terms remain after preprocessing.")
                    print()
                    continue

                combinations = generate_query_combinations(processed_query)

                for boolean_query, operators in combinations:
                    results = evaluate_boolean_query(
                        processed_query,
                        operators,
                        inverted_index
                    )

                    display_query_results(boolean_query, results, show_all_results)
                print()

    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main(show_all_results="--preview" not in sys.argv[1:])
