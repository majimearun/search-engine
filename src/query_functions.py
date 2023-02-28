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


def query_permuterm_index(query, perm_index, rev_perm_index, inv_list, ret_words=False):
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
    if ret_words:
        return result
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


def match_all_wildcards_in_biwords(biwords, perm_index, rev_perm_index):
    word_possibilites = {}
    for bw in biwords:
        words = bw.split()
        for i in range(len(words)):
            if "*" in words[i]:
                word_possibilites[words[i]] = query_permuterm_index(
                    words[i], perm_index, rev_perm_index, None, ret_words=True
                )
            else:
                word_possibilites[words[i]] = [words[i]]

    new_biwords = []
    for bw in biwords:
        words = bw.split()
        possibilities = [
            " ".join([word_possibilites[words[0]][i], word_possibilites[words[1]][j]])
            for i in range(len(word_possibilites[words[0]]))
            for j in range(len(word_possibilites[words[1]]))
        ]
        new_biwords.extend(possibilities)
    return new_biwords


def phrase_query(query, n_word_index, perm_index, rev_perm_index):
    words = query.split()
    biwords = []
    for i in range(len(words) - 1):
        biwords.append(words[i] + " " + words[i + 1])
    new_biwords = match_all_wildcards_in_biwords(biwords, perm_index, rev_perm_index)
    result = []
    for bw in new_biwords:
        result.append(query_n_word_index(bw, n_word_index))
    if len(result) == 0:
        return []
    final = set(result[0])
    if len(new_biwords) > len(biwords):
        for l in result:
            final = final.union(set(l))
    else:
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
        return phrase_query(queries, n_word_index, perm_index, rev_perm_index)


def tfidf(tf, _df, ndocs):
    return (np.log(1 + tf)) * (np.log(ndocs / (_df + 1)))


def get_term_frequency_scores(df, queries, inverted_list, perm_index, rev_perm_index):
    scores = []
    ndocs = len(df)
    for row in df.iterrows():
        text = str(row[1]["tokenized"])
        text = text.split()
        score = 0
        # print(queries)
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
    # scores = [x for x in scores if x[1] > 0]
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
