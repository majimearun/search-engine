import pandas as pd
import pickle
from LinkedList import LinkedList

def create_postings_list(x: str):
    """Creates a postings list for a given string
    
    Args:
        x (str): string to create postings list for
        
    Returns:
        list[str]: sorted postings list for the given string
    """
    # just in case any number is passed/occurs in the text
    x = str(x)
    posting_list = set()
    for word in x.split():
        posting_list.add(word.lower())
    posting_list = list(posting_list)
    # remove strings with only punctuations
    punctuations = """!()-[]{};:'"\,<>./?@#$%^&*_~=+"""
    for word in posting_list:
        if word in punctuations:
            posting_list.remove(word)
    return sorted((posting_list))


def create_inverted_list(df: pd.DataFrame, corpus: list[str]):
    """Creates an inverted list for a given corpus and set of documents. Inverted list a dictionary with keys as the words in the corpus and values as a sorted linked list of the documents in which the word occurs

    Args:
        df (pd.DataFrame): dataframe containing the postings list for each document
        corpus (list[str]): list of all the words in the corpus

    Returns:
        dict[str, LinkedList]: inverted list for the given corpus and set of documents
    """
    inverted_list = {}
    for word in corpus:
        inverted_list[word] = LinkedList()
    for row in df.iterrows():
        l = row[1]["posting_list"]
        for word in l:
            inverted_list[word].append(row[0])
    for word in inverted_list:
        inverted_list[word].sort()
    return inverted_list


def get_all_rotations(s: str):
    """Get all the rotations of a given string (clockwise downwards rotation)

    Args:
        s (str): string for which all the rotations are to be found

    Returns:
        list[str]: list of all the rotations of the given string
    """
    rotations = []
    for i in range(len(s)):
        rotations.append(s[i:] + s[:i])
    return rotations


def permuterm_indexing(inv_list: dict[str, LinkedList]):
    """Creates a permuterm index for a given inverted list

    Args:
        inv_list (dict[str, LinkedList]): inverted list using which the permuterm index is to be created

    Returns:
        dict[str, LinkedList]: permuterm index for the given inverted list
    """
    perm_index = {}
    for word in inv_list:
        word_perm = word + "$"
        rotations = get_all_rotations(word_perm)
        for rotation in rotations:
            q = rotation.split("$")[-1]
            if q not in perm_index:
                perm_index[q] = LinkedList()
            perm_index[q].append(word)
    return perm_index


def reverse_permuterm_indexing(inv_list: dict[str, LinkedList]):
    """Creates a reverse permuterm index for a given inverted list

    Args:
        inv_list (dict[str, LinkedList]): inverted list using which the reverse permuterm index is to be created

    Returns:
        dict[str, LinkedList]: reverse permuterm index for the given inverted list
    """
    rev_perm_index = {}
    for word in inv_list:
        word_perm = "$" + word
        word_perm = word_perm[::-1]
        rotations = get_all_rotations(word_perm)
        for rotation in rotations:
            q = rotation.split("$")[-1]
            if q not in rev_perm_index:
                rev_perm_index[q] = LinkedList()
            rev_perm_index[q].append(word)
    return rev_perm_index


def make_bi_word_index(df: pd.DataFrame):
    """Creates an bi-word index for a given dataframe

    Args:
        df (pd.DataFrame): dataframe whose `tokenized` column is to be used to create the bi-word index

    Returns:
        dict[str, LinkedList]: bi-word index for the given dataframe
    """
    bi_word_index = {}
    for row in df.iterrows():
        text = str(row[1]["tokenized"])
        text = text.split()
        for i in range(len(text) - 1):
            # tale two adjacent words as the key for the bi-word index
            bi_word = text[i] + " " + text[i + 1]
            if bi_word not in bi_word_index:
                bi_word_index[bi_word] = LinkedList()
            bi_word_index[bi_word].append(row[0])
    for key in bi_word_index:
        bi_word_index[key].sort()
    return bi_word_index

def startup_engine(*paths: tuple[str]):
    """Creates the inverted list, permuterm index, reverse permuterm index, bi-word index, corpus and the dataframe containing the index and text (normal and tokenized) for each document
    Args:
        paths (tuple[str]): paths to the csv files containing the text for which the indexes are to be created
        
    Returns:
        tuple[dict[str, LinkedList], dict[str, LinkedList], dict[str, LinkedList], dict[str, LinkedList], list[str], pd.DataFrame]: tuple containing the inverted list, permuterm index, reverse permuterm index, bi-word index, corpus and the dataframe containing the index and text (normal and tokenized) for each document
        
    """
    main_df = pd.read_csv(paths[0])
    main_df["posting_list"] = main_df["tokenized"].apply(create_postings_list)
    for path in paths[1:]:
        temp_df = pd.read_csv(path)
        temp_df["posting_list"] = temp_df["tokenized"].apply(create_postings_list)
        main_df = pd.concat([main_df, temp_df])
        
    main_df = main_df.reset_index(drop=True)
    corpus = set()
    for l in main_df.posting_list:
        for word in l:
            corpus.add(word)
    corpus = sorted(list(corpus))

    inverted_list = create_inverted_list(main_df, corpus)

    perm_index = permuterm_indexing(inverted_list)
    rev_perm_index = reverse_permuterm_indexing(inverted_list)

    bi_word_index = make_bi_word_index(main_df)
    
    return inverted_list, perm_index, rev_perm_index, bi_word_index, corpus, main_df

if __name__ == "__main__":
    # Run this file to create the summarizer model (pretrained transformer form huggingface). Needs to be run only once.
    from transformers import pipeline
    summary_pipeline = pipeline("summarization")
    
    with open("./models/summary_pipeline.pkl", "wb") as f:
        pickle.dump(summary_pipeline, f)
    