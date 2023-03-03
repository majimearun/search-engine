# importing libraries
import sys
import spacy
import pandas as pd
import pickle
import re
from LinkedList import LinkedList
from phrase_query_functions import phrase_query
from scoring_functions import get_term_frequency_scores
from edit_distance_functions import spell_check_query, autocomplete_result
from wildcard_query_functions import query_permuterm_index

# getting the summarizer pipeline which we earlier loaded in the setup.py file
with open(
    "/home/majime/programming/github/search-engine/models/summary_pipeline.pkl", "rb"
) as f:
    summary_pipeline = pickle.load(f)

# loading the spacy model for lemmitization of queries
nlp = spacy.load("en_core_web_sm")

# setting recursion limit for functions on linked list
sys.setrecursionlimit(10000)


def multi_query(
    queries: str,
    inverted_list: dict[str, LinkedList],
    perm_index: dict[str, LinkedList],
    rev_perm_index: dict[str, LinkedList],
    _and: bool = False,
):
    """Finds all the documents that match/contain words from the query string

    Args:
        queries (str): query string
        inverted_list (dict[str, LinkedList]): inverted list for each word in the corpus
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus
        _and (bool, optional): Whether to return the intersection of the documents matching the query words. Defaults to False.
    Returns:
        list[int]: sorted list of documents that match the query string
    """
    docs: list[int] = []
    for query in queries:
        if "*" in query:
            docs.append(
                query_permuterm_index(query, perm_index, rev_perm_index, inverted_list)
            )

        else:
            query: str = nlp(query)[0].lemma_.lower()
            intermediate_docs: list[int] = []
            try:
                for id in inverted_list[query]:
                    intermediate_docs.append(id.data)
            except:
                continue
            docs.append(intermediate_docs)
    if not _and:
        # return union of all sublists in docs
        result: list[int] = []
        for l in docs:
            for id in l:
                if id not in result:
                    result.append(id)
        return sorted(result)
    else:
        if len(docs) == 0:
            return []
        # return intersection of all sublists in docs
        result: set[int] = set(docs[0])
        for l in docs:
            result = result.intersection(set(l))
        return sorted(list(result))


def boolean_filter(
    queries: str,
    inverted_list: dict[str, LinkedList],
    perm_index: dict[str, LinkedList],
    rev_perm_index: dict[str, LinkedList],
    bi_word_index: dict[str, LinkedList],
    _phrase=False,
):
    """Filters out documents using a simple boolean retrieval

    Args:
        queries (str): query string
        inverted_list (dict[str, LinkedList]): inverse index for each word in the corpus
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus
        bi_word_index (dict[str, LinkedList]): biword index for each biword in the corpus
        _phrase (bool, optional): Whether the query is a phrase query or not. Defaults to False.

    Returns:
        list[int]: sorted list of documents that match the query string
    """
    # seperate queries into two lists, and and or lists. and words have double quotes around them
    if not _phrase:
        queries: list[str] = queries.split()
        and_queries: list[str] = []
        or_queries: list[str] = []
        for query in queries:
            # identifying and queries using double quotes
            if query[0] == '"' and query[-1] == '"':
                and_queries.append(query[1:-1])
            else:
                or_queries.append(query)
        if len(and_queries) == 0:
            return multi_query(or_queries, inverted_list, perm_index, rev_perm_index)
        # If there are any and queries we only take care of thyem as those in OR may or may not be present
        # Scoring takes care of the order (or queries are still part of the ranking later on)
        else:
            return multi_query(
                and_queries, inverted_list, perm_index, rev_perm_index, _and=True
            )
        # If you want to filter using both the AND and OR queries uncomment the following lines and change the else statement above to (elif len(or_queries) == 0):
        
        # and_results: list[int] = multi_query(
        #     and_queries, inverted_list, perm_index, rev_perm_index, _and=True
        # )
        # or_results: list[int] = multi_query(
        #     or_queries, inverted_list, perm_index, rev_perm_index
        # )
        # considering if even one word contains an and all words do, otherwise there is no point of the or words
        # return sorted(list(set(and_results) & set(or_results)))
    else:
        # all phrase queries are (logically) of and type, so removing all double quotes
        queries: list[str] = queries.replace('"', "")
        return phrase_query(queries, bi_word_index, perm_index, rev_perm_index)
    
    
def print_results(scores: list[tuple[int, float]], df: pd.DataFrame, show_summary: bool, ranked:bool):
    """Prints the results of the search

    Args:
        scores (list[tuple[int, float]]): sorted list of tuples containing document id and score
        df (pd.DataFrame): dataframe containing the corpus
        show_summary (bool): whether to show the summary of the document or not
        ranked (bool): whether the search was ranked or not
    """
    print(f"Documents Retrieved: {len(scores)}")
    if not ranked:
        print("Unranked Search Results: Boolean Retrieval")
    print(
        "------------------------------------------------------------------------------------------"
    )
    print(
        "------------------------------------------------------------------------------------------"
    )
    for i, score in enumerate(scores):
        row = df.loc[score[0]]
        print(f"Rank: {i + 1}")
        print(f"Document Name: {row.document_name}")
        print(f"Page Number: {row.page_number + 1}")
        print(f"Paragraph Number: {row.paragraph_number + 1}")
        print(f"Score: {score[1]}")
        print(
            "------------------------------------------------------------------------------------------"
        )
        if show_summary:
            summary = summary_pipeline(row.text, truncation=True)
            print(f"Summary: {summary[0]['summary_text']}")
        print(
            "------------------------------------------------------------------------------------------"
        )
        # replace any number f spaces with a single space
        print_text = re.sub(r"\s+", " ", row.text)
        print(f"Paragraph Text: \n{print_text}")
        print(
            "------------------------------------------------------------------------------------------"
        )
        print(
            "------------------------------------------------------------------------------------------"
        )
        print(
            "------------------------------------------------------------------------------------------"
        ) 
        
        
def search(
    query: str,
    inverted_list: dict[str, LinkedList],
    perm_index: dict[str, LinkedList],
    rev_perm_index: dict[str, LinkedList],
    bi_word_index: dict[str, LinkedList],
    main_df: pd.DataFrame,
    is_phrase: bool = False,
    ranked: bool = True,
    show_summary: bool = False,
    retrieve_n: int = None,
    spell_check: bool = False,
    autocomplete: bool = False,
    n_auto_results: int = 5,
):
    """Searches the corpus for documents that match the query string

    Args:
        query (str): query string
        inverted_list (dict[str, LinkedList]): inverted index for each word in the corpus
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus
        bi_word_index (dict[str, LinkedList]): biword index for each biword in the corpus
        main_df (pd.DataFrame): dataframe containing the corpus
        is_phrase (bool, optional): Whether the query is a phrase query or not. Defaults to False.
        ranked (bool, optional): SWhether the results should be ranked or not. Defaults to True.
        show_summary (bool, optional): Whether we need to show the summary of the retrieved documents. Defaults to False.
        retrieve_n (int, optional): Number of documents to be retrieved. Defaults to None.
        spell_check (bool, optional): Whether to perform spell check or not. Defaults to False.
        auto_complete (bool, optional): Whether to print auto complete options instead of search or not. Defaults to False.
        n_auto_results (int, optional): Number of auto complete results to be printed. Defaults to 5.
    """
    query = query.lower()
    if autocomplete:
        results = autocomplete_result(query, inverted_list, n_auto_results)
        print("Possible Options:")
        print("------------------------------------------------------------------------------------------")
        for i, result in enumerate(results):
            print(f"{i + 1}. {result}")
        print("------------------------------------------------------------------------------------------")
        return 
            
    # first we filter results using boolean retrieval
    filtered = boolean_filter(
        query,
        inverted_list,
        perm_index,
        rev_perm_index,
        bi_word_index,
        _phrase=is_phrase,
    )
    if len(filtered) == 0:
        if spell_check:
            print(
                f"No documents found with direct match with {query}. Performing spell check..."
            )
            corrected_queries: list[str] = []
            for q in query.split():
                if "*" not in q:
                    corrected_queries.append(spell_check_query(q, inverted_list))
                else:
                    corrected_queries.append(q)
            query = " ".join(corrected_queries)
            print(f"Corrected Query: {query}")
            filtered = boolean_filter(
                query,
                inverted_list,
                perm_index,
                rev_perm_index,
                bi_word_index,
                _phrase=is_phrase,
            )
            if len(filtered) == 0:
                print("No documents found even after spell check")
                return
        else:
            print("No documents found")
            return
    if ranked:
        scores = get_term_frequency_scores(
            main_df,
            query.split(),
            inverted_list,
            perm_index,
            rev_perm_index,
        )
        # get only scores for those documents that are in filtered
        scores = [x for x in scores if x[0] in filtered]

    else:
        scores = []
        for id in filtered:
            scores.append((id, None))
    if retrieve_n is not None:
        scores = scores[:retrieve_n]
    print_results(scores, main_df, show_summary, ranked)
    