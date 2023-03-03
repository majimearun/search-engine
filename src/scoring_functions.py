# importing libraries
import numpy as np
import sys
import spacy
import pandas as pd
from LinkedList import LinkedList
from wildcard_query_functions import (
    left_permuterm_indexing,
    query_permuterm_index,
    right_permuterm_indexing,
)


# loading the spacy model for lemmitization of queries
nlp = spacy.load("en_core_web_sm")

# setting recursion limit for functions on linked list
sys.setrecursionlimit(10000)


def tfidf(tf: int, _df: int, ndocs: int):
    """Calculates the tf*idf score for a given term frequency and document frequency

    Args:
        tf (int): term frequency
        _df (int): document frequency
        ndocs (int): total number of documents in the corpus

    Returns:
        float: tf*idf score
    """
    return (np.log(1 + tf)) * (np.log((1 + ndocs) / (_df + 1)) + 1)


def get_term_frequency_scores(
    df: pd.DataFrame,
    queries: list[str],
    inverted_list: dict[str, LinkedList],
    perm_index: dict[str, LinkedList],
    rev_perm_index: dict[str, LinkedList],
):
    """Calculates the tf*idf scores for each document in the corpus

    Args:
        df (pd.DataFrame): dataframe containing the corpus
        queries (list[str]): list of query words
        inverted_list (dict[str, LinkedList]): inverted index for each word in the corpus
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus

    Returns:
        list[tuple[int, float]]: sorted (descending based on score) list of tuples containing document id and tf*idf score
    """
    # removing quotes from queries
    queries = [q.replace('"', "") for q in queries]
    # lemmatizing queries
    queries = [nlp(q)[0].lemma_ for q in queries if "*" not in q]
    scores: dict[int, float] = {}
    for index, row in df.iterrows():
        score = 0
        text: str = row["tokenized"]
        for query in queries:
            if "*" not in query:
                tf = text.count(query)
                if query in inverted_list:
                    _df = len(inverted_list[query])
                    score += tfidf(tf, _df, len(df))
            else:
                if query[-1] == "*":
                    left_matches = left_permuterm_indexing(query, perm_index)
                    for word in left_matches:
                        _df = len(inverted_list[word])
                        tf = text.count(word)
                        score += tfidf(tf, _df, len(df))
                elif query[0] == "*":
                    right_matches = right_permuterm_indexing(query, rev_perm_index)
                    for word in right_matches:
                        _df = len(inverted_list[word])
                        tf = text.count(word)
                        score += tfidf(tf, _df, len(df))
                else:
                    matches = query_permuterm_index(
                        query, perm_index, rev_perm_index, inverted_list, True
                    )
                    for word in matches:
                        _df = len(inverted_list[word])
                        tf = text.count(word)
                        score += tfidf(tf, _df, len(df))
        scores[index] = score
    sorted_scores: list[tuple[int, float]] = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_scores
