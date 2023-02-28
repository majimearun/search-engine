# importing libraries
import numpy as np
import sys
import spacy
import pandas as pd
from setup import get_all_rotations
import pickle
from LinkedList import LinkedList

# getting the summarizer pipeline which we earlier loaded in the setup.py file
with open(
    "/home/majime/programming/github/ir-search-engine/models/summary_pipeline.pkl", "rb"
) as f:
    summary_pipeline = pickle.load(f)

# loading the spacy model for lemmitization of queries
nlp = spacy.load("en_core_web_sm")

# setting recursion limit for functions on linked list
sys.setrecursionlimit(10000)


def left_permuterm_indexing(query: str, perm_index: dict[str, LinkedList]):
    """Finds all the words in the corpus that match the wildcard on thee right end of the query word

    Args:
        query (str): query word (including wildcard)
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus

    Returns:
        list[str]: list of words in the corpus that match the wildcard on the right end of the query word
    """
    result: list[str] = []
    query = query + "$"
    rotations = get_all_rotations(query)
    for rotation in rotations:
        if rotation[0] == "*":
            # remove the wildcard and the $ from the rotation
            q = rotation[2:]
            if q in perm_index:
                # if the rotation is in the permuterm index, add all the words in the linked list to the result
                for word in perm_index[q]:
                    result.append(word.data)
    return result


def right_permuterm_indexing(query: str, rev_perm_index: dict[str, LinkedList]):
    """Finds all the words in the corpus that match the wildcard on the left end of the query word

    Args:
        query (str): query word (including wildcard)
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus

    Returns:
        list[str]: list of words in the corpus that match the wildcard on the left end of the query word
    """
    result: list[str] = []
    query = "$" + query
    # reversing as reverse permuterm index is built on the reversed word
    query = query[::-1]
    rotations = get_all_rotations(query)
    for rotation in rotations:
        if rotation[0] == "*":
            q = rotation[2:]
            if q in rev_perm_index:
                #  if the rotation is in the permuterm index, add all the words in the linked list to the result
                for word in rev_perm_index[q]:
                    result.append(word.data)
    return result


def query_permuterm_index(
    query: str,
    perm_index: dict[str, LinkedList],
    rev_perm_index: dict[str, LinkedList],
    inv_list: dict[str, LinkedList],
    ret_words: bool = False,
):
    """Finds all the documents that match the query word

    Args:
        query (str): query word
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus
        inv_list (dict[str, LinkedList]): inverted list for each word in the corpus
        ret_words (bool, optional): Whether we need to return the words matching the wildcard. If false, then instead returns the documents matching the wild card query. Defaults to False.

    Returns:
        list[str | int]: list of words in the corpus/ sorted list of documents that match the wildcard query
    """
    result: list[str] = []
    if "*" in query:
        if query[-1] == "*":
            result = left_permuterm_indexing(query, perm_index)
        elif query[0] == "*":
            result = right_permuterm_indexing(query, rev_perm_index)

        else:
            # split into two halves so we can find the words matching the wildcard on the left and right
            halves = query.split("*")
            left_result = left_permuterm_indexing(halves[0] + "*", perm_index)
            right_result = right_permuterm_indexing("*" + halves[-1], rev_perm_index)
            # finding the intersection of the two lists (common terms matching both and left and right halves)
            result = list(set(left_result) & set(right_result))
    if ret_words:
        return result
    docs: list[int] = []
    for word in result:
        #  since these words are from perm index, they definitely occur in the inverted list, but for safety we use try except
        for id in inv_list[word]:
            try:
                docs.append(id.data)
            except Exception as e:
                print(f"Exception at querying wildcard match: {id} \n\n {e}")
                continue
    return sorted(list(set(docs)))


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
        # return intersection of all sublists in docs
        result: set[int] = set(docs[0])
        for l in docs:
            result = result.intersection(set(l))
        return sorted(list(result))


def query_bi_word_index(query: str, bi_word_index: dict[str, LinkedList]):
    """Finds all the documents that match the biword query

    Args:
        query (str): biword query string
        bi_word_index (dict[str, LinkedList]): biword index for each biword in the corpus

    Returns:
        list[int]: sorted list of documents that match the biword query string
    """
    result: list[int] = []
    if query in bi_word_index:
        for id in bi_word_index[query]:
            result.append(id.data)
    return sorted(result)


def match_all_wildcards_in_biwords(
    biwords: list[str],
    perm_index: dict[str, LinkedList],
    rev_perm_index: dict[str, LinkedList],
):
    """Finds all the possible biwords from the biword query string that contain wildcard matches

    Args:
        biwords (list[str]): list of biwords in the query string
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (dict[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus

    Returns:
        list[str]: list of all possible biwords that match the wildcard query
    """
    word_possibilites: dict[str, list[str]] = {}
    for bw in biwords:
        words = bw.split()
        for i in range(len(words)):
            if "*" in words[i]:
                # No need for inverse list as we want only the words and we are setting ret_words = True
                word_possibilites[words[i]] = query_permuterm_index(
                    words[i], perm_index, rev_perm_index, None, ret_words=True
                )
            else:
                # if it doesnt contain a wildcard, then it is a normal word and only possible match is itself
                word_possibilites[words[i]] = [words[i]]

    new_biwords: list[str] = []
    for bw in biwords:
        words = bw.split()
        # creating a list with all possible biwrds matching the wildcard(s) it contains
        possibilities = [
            " ".join([word_possibilites[words[0]][i], word_possibilites[words[1]][j]])
            for i in range(len(word_possibilites[words[0]]))
            for j in range(len(word_possibilites[words[1]]))
        ]
        new_biwords.extend(possibilities)
    return new_biwords


def phrase_query(
    query: str,
    bi_word_index: dict[str, LinkedList],
    perm_index: dict[str, LinkedList],
    rev_perm_index: list[str, LinkedList],
):
    """Finds all the documents that match the phrase query string

    Args:
        query (str): phrase query string
        bi_word_index (dict[str, LinkedList]): biword index for each biword in the corpus
        perm_index (dict[str, LinkedList]): permuterm index for each possible rotation of words in the corpus
        rev_perm_index (list[str, LinkedList]): reverse permuterm index for each possible rotation of words in the corpus

    Returns:
        list[int]: sorted list of documents that match the phrase query string
    """
    words: list[str] = query.split()
    biwords: list[str] = []
    for i in range(len(words) - 1):
        biwords.append(words[i] + " " + words[i + 1])
    # new biwords is a list of all possible biwords that match the wildcard(s) in the query
    new_biwords = match_all_wildcards_in_biwords(biwords, perm_index, rev_perm_index)
    result = []
    for bw in new_biwords:
        result.append(query_bi_word_index(bw, bi_word_index))
    if len(result) == 0:
        # to prevent overflow error in next step
        return []
    final = set(result[0])
    if len(new_biwords) > len(biwords):
        # if length of the new biwords is greater than the original biwords, then there were wildcards in the query, so we need to take union so we dont lose any results
        for l in result:
            final = final.union(set(l))
    else:
        # if length of the new biwords is less than the original biwords, then there were no wildcards in the query, so we need can take intersection without losing any results
        for l in result:
            final = final.intersection(set(l))
    return sorted(list(final))


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
        _phrase (bool, optional): Wheteher the query is aphrase query or not. Defaults to False.

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
        if len(or_queries) == 0:
            return multi_query(
                and_queries, inverted_list, perm_index, rev_perm_index, _and=True
            )
        and_results: list[int] = multi_query(
            and_queries, inverted_list, perm_index, rev_perm_index, _and=True
        )
        or_results: list[int] = multi_query(
            or_queries, inverted_list, perm_index, rev_perm_index
        )
        # considering if even one word contains an and all words do, otherwise there is no point of the or words
        return sorted(list(set(and_results) & set(or_results)))
    else:
        # all phrase queries are (logically) of and type, so removing all double quotes
        queries: list[str] = queries.replace('"', "")
        return phrase_query(queries, bi_word_index, perm_index, rev_perm_index)


def tfidf(tf: int, _df: int, ndocs: int):
    """Calculates the tf*idf score for a given term frequency and document frequency

    Args:
        tf (int): term frequency
        _df (int): document frequency
        ndocs (int): total number of documents in the corpus

    Returns:
        float: tf*idf score
    """
    return (np.log(1 + tf)) * (np.log(ndocs / (_df + 1)))


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
    scores = []
    ndocs = len(df)
    for row in df.iterrows():
        text = str(row[1]["tokenized"])
        text = text.split()
        score = 0
        for q in queries:
            if "*" not in q:
                if q not in inverted_list:
                    continue
                doc_freq = len(inverted_list[q])
                score += tfidf(1 + text.count(q), doc_freq, ndocs)
            else:
                if q[-1] == "*":
                    left_result = left_permuterm_indexing(q, perm_index)
                    for word in left_result:
                        doc_freq = len(inverted_list[word])
                        score += tfidf(1 + text.count(word), doc_freq, ndocs)
                elif q[0] == "*":
                    right_result = right_permuterm_indexing(q, rev_perm_index)
                    for word in right_result:
                        doc_freq = len(inverted_list[word])
                        score += tfidf(1 + text.count(word), doc_freq, ndocs)
                else:
                    halves = q.split("*")
                    left_result = left_permuterm_indexing(halves[0] + "*", perm_index)
                    right_result = right_permuterm_indexing(
                        "*" + halves[-1], rev_perm_index
                    )
                    result = list(set(left_result) & set(right_result))
                    for word in result:
                        doc_freq = len(inverted_list[word])
                        score += tfidf(1 + text.count(word), doc_freq, ndocs)

        scores.append((row[0], score))
    return sorted(scores, key=lambda x: x[1], reverse=True)


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
        show_summary (bool, optional): Whether we need to show the summary of the retieved documents. Defaults to False.
        retrieve_n (int, optional): Number of dicuments to be retrieved. Defaults to None.
    """
    query = query.lower()
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
        # get only scores for thoise documents that are in filtered
        scores = [x for x in scores if x[0] in filtered]

    else:
        scores = []
        for id in filtered:
            scores.append((id, None))
    if retrieve_n is not None:
        scores = scores[:retrieve_n]
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
        row = main_df.loc[score[0]]
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
        print(f"Paragraph Text: \n{row.text}")
        print(
            "------------------------------------------------------------------------------------------"
        )
        print(
            "------------------------------------------------------------------------------------------"
        )
        print(
            "------------------------------------------------------------------------------------------"
        )
