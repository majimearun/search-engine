# Copyright © 2023 Arunachala Amuda Murugan (@majimearun)
#
# License: GNU General Public License v3.0

# importing libraries
import sys
from setup import get_all_rotations
from LinkedList import LinkedList


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
