# importing libraries
import numpy as np
import sys
import spacy
from setup import (
    create_inverted_list,
    get_all_rotations,
    permuterm_indexing,
    reverse_permuterm_indexing,
)

nlp = spacy.load("en_core_web_sm")
sys.setrecursionlimit(10000)


def left_permuterm_indexing(query, perm_index):
    result = []
    query = query + "$"
    rotations = get_all_rotations(query)
    for rotation in rotations:
        if rotation[0] == "*":
            q = rotation[2:]
            if q in perm_index:
                for word in perm_index[q]:
                    result.append(word.data)
    return result


def right_permuterm_indexing(query, rev_perm_index):
    result = []
    query = "$" + query
    query = query[::-1]
    rotations = get_all_rotations(query)
    for rotation in rotations:
        if rotation[0] == "*":
            q = rotation[2:]
            if q in rev_perm_index:
                for word in rev_perm_index[q]:
                    result.append(word.data)
    return result


def query_permuterm_index(query, perm_index, rev_perm_index, inv_list):
    result = []
    if "*" in query:
        if query[-1] == "*":
            result = left_permuterm_indexing(query, perm_index)
        elif query[0] == "*":
            result = right_permuterm_indexing(query, rev_perm_index)

        else:
            halves = query.split("*")
            left_result = left_permuterm_indexing(halves[0] + "*", perm_index)
            right_result = right_permuterm_indexing("*" + halves[-1], rev_perm_index)
            result = list(set(left_result) & set(right_result))

    docs = []
    for word in result:
        for id in inv_list[word]:
            docs.append(id.data)
    return sorted(list(set(docs)))


def multi_query(queries, inverted_list, perm_index, rev_perm_index, _and=False):
    docs = []
    for query in queries:
        if "*" in query:
            docs.append(
                query_permuterm_index(query, perm_index, rev_perm_index, inverted_list)
            )

        else:
            query = nlp(query)[0].lemma_.lower()
            intermediate_docs = []
            try:
                for id in inverted_list[query]:
                    intermediate_docs.append(id.data)
            except:
                continue
            docs.append(intermediate_docs)
    # return docs
    if not _and:
        # return union of all sublists in docs
        result = []
        for l in docs:
            for id in l:
                if id not in result:
                    result.append(id)
        return sorted(result)
    else:
        result = set(docs[0])
        for l in docs:
            result = result.intersection(set(l))
        return sorted(list(result))


def query_n_word_index(query, n_word_index):
    result = []
    if query in n_word_index:
        for id in n_word_index[query]:
            result.append(id.data)
    return result


def phrase_query(query, n_word_index):
    words = query.split()
    # TODO: List all possible biwords if wild card is used
    biwords = []
    for i in range(len(words) - 1):
        biwords.append(words[i] + " " + words[i + 1])
    result = []
    for bw in biwords:
        result.append(query_n_word_index(bw, n_word_index))
    if len(result) == 0:
        return []
    final = set(result[0])
    for l in result:
        final = final.intersection(set(l))
    return sorted(list(final))


def boolean_filter(
    queries, inverted_list, perm_index, rev_perm_index, n_word_index, _phrase=False
):
    # seperate queries into two lists, and and or lists. and words have double quotes around them
    if not _phrase:
        queries = queries.split()
        and_queries = []
        or_queries = []
        for query in queries:
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
        and_results = multi_query(
            and_queries, inverted_list, perm_index, rev_perm_index, _and=True
        )
        or_results = multi_query(or_queries, inverted_list, perm_index, rev_perm_index)
        return sorted(list(set(and_results) & set(or_results)))
    else:
        queries = queries.replace('"', "")
        return phrase_query(queries, n_word_index)


def tfidf(tf, _df, ndocs):
    return (np.log(1 + tf)) * (np.log(ndocs / _df))


def get_term_frequency_scores(df, queries, inverted_list, perm_index, rev_perm_index):
    scores = []
    doc_freq = {}
    ndocs = len(df)
    for q in queries:
        if q in inverted_list:
            doc_freq[q] = len(inverted_list[q]) + 1
        else:
            doc_freq[q] = 1
    for row in df.iterrows():
        text = str(row[1]["tokenized"])
        text = text.split()
        score = 0
        for q in queries:
            if q in text:
                if "*" not in q:
                    score += tfidf(1 + text.count(q), doc_freq[q], ndocs)
                else:
                    if q[-1] == "*":
                        left_result = left_permuterm_indexing(q, perm_index)
                        for word in left_result:
                            score += tfidf(1 + text.count(word), doc_freq[word], ndocs)
                    elif q[0] == "*":
                        right_result = right_permuterm_indexing(q, rev_perm_index)
                        for word in right_result:
                            score += tfidf(1 + text.count(word), doc_freq[word], ndocs)
                    else:
                        halves = q.split("*")
                        left_result = left_permuterm_indexing(
                            halves[0] + "*", perm_index
                        )
                        right_result = right_permuterm_indexing(
                            "*" + halves[-1], rev_perm_index
                        )
                        result = list(set(left_result) & set(right_result))
                        for word in result:
                            score += tfidf(1 + text.count(word), doc_freq[word], ndocs)

        scores.append((row[0], score))
    scores = [x for x in scores if x[1] > 0]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def search(
    query,
    inverted_list,
    perm_index,
    rev_perm_index,
    n_word_index,
    main_df,
    is_phrase=False,
    ranked=True,
):
    query = query.lower()
    filtered = boolean_filter(
        query,
        inverted_list,
        perm_index,
        rev_perm_index,
        n_word_index,
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
        scores = [x for x in scores if x[0] in filtered]

    else:
        scores = []
        for id in filtered:
            scores.append((id, None))
    print(f"Documents Retrieved: {len(scores)}")
    if not ranked:
        print("Unranked Search Results: Boolean Retrieval")
    print(
        "------------------------------------------------------------------------------------------"
    )
    print(
        "------------------------------------------------------------------------------------------"
    )
    for score in scores:
        row = main_df.loc[score[0]]
        print(f"Document Name: {row.document_name}")
        print(f"Page Number: {row.page_number + 1}")
        print(f"Score: {score[1]}")
        print(f"Paragraph Number: {row.paragraph_number + 1}")
        print(f"Paragraph Text: {row.text}")
        print(
            "------------------------------------------------------------------------------------------"
        )
        print(
            "------------------------------------------------------------------------------------------"
        )
        print(
            "------------------------------------------------------------------------------------------"
        )
