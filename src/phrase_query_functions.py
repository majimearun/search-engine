# Copyright © 2023 Arunachala Amuda Murugan (@majimearun)
#
# License: GNU General Public License v3.0

# importing libraries
import sys
import spacy
from LinkedList import LinkedList
from wildcard_query_functions import query_permuterm_index


# loading the spacy model for lemmitization of queries
nlp = spacy.load("en_core_web_sm")

# setting recursion limit for functions on linked list
sys.setrecursionlimit(10000)


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
                word_possibilites[words[i]] = [nlp(words[i])[0].lemma_.lower()]

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
