# importing libraries
import numpy as np
import sys
from LinkedList import LinkedList


# setting recursion limit for functions on linked list
sys.setrecursionlimit(10000)


def levenshtein_distance(s1: str, s2: str, swapping_importance: bool = True):
    """Calculates the levenshtein distance between two strings

    Args:
        s1 (str): first string
        s2 (str): second string
        swapping_importance (bool, optional): Whether swapping two characters is more important than deleting or inserting a character. Defaults to True.

    Returns:
        int: levenshtein distance between the two strings
    """
    levenshtein = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        levenshtein[i, 0] = i
    for j in range(len(s2) + 1):
        levenshtein[0, j] = j
    if not swapping_importance:
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    levenshtein[i, j] = levenshtein[i - 1, j - 1]
                else:
                    levenshtein[i, j] = (
                        min(
                            levenshtein[i - 1, j - 1],
                            levenshtein[i - 1, j],
                            levenshtein[i, j - 1],
                        )
                        + 1
                    )
        return levenshtein[len(s1), len(s2)]
    else:
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    levenshtein[i, j] = levenshtein[i - 1, j - 1]
                else:
                    levenshtein[i, j] = (
                        min(
                            levenshtein[i - 1, j - 1],
                            levenshtein[i - 1, j],
                            levenshtein[i, j - 1],
                        )
                        + 1
                    )
                    if (
                        i > 1
                        and j > 1
                        and s1[i - 1] == s2[j - 2]
                        and s1[i - 2] == s2[j - 1]
                    ):
                        levenshtein[i, j] = min(
                            levenshtein[i, j], levenshtein[i - 2, j - 2] + 1
                        )
        return levenshtein[len(s1), len(s2)]


def spell_check_query(query: str, inverted_list: dict[str, LinkedList]):
    """Spell checks the query and returns the corrected query

    Args:
        query (str): query string
        inverted_list (dict[str, LinkedList]): inverse index for each word in the corpus

    Returns:
        str: corrected query string
    """
    query = query.replace('"', "")
    query = query.split()
    for i in range(len(query)):
        if query[i] not in inverted_list:
            min_dist = 100000
            min_word = ""
            for word in inverted_list:
                dist = levenshtein_distance(query[i], word)
                if dist < min_dist:
                    min_dist = dist
                    min_word = word
            query[i] = min_word
    return " ".join(query)
