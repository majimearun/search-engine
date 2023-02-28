import pandas as pd
import pickle
from LinkedList import LinkedList


def create_postings_list(x):
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


def create_inverted_list(df, corpus):
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


def get_all_rotations(s):
    rotations = []
    for i in range(len(s)):
        rotations.append(s[i:] + s[:i])
    return rotations


def permuterm_indexing(inv_list):
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


def reverse_permuterm_indexing(inv_list):
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


def make_n_word_index(df):
    n_word_index = {}
    for row in df.iterrows():
        text = str(row[1]["tokenized"])
        text = text.split()
        for i in range(len(text) - 1):
            n_word = text[i] + " " + text[i + 1]
            if n_word not in n_word_index:
                n_word_index[n_word] = LinkedList()
            n_word_index[n_word].append(row[0])
    for key in n_word_index:
        n_word_index[key].sort()
    return n_word_index

def startup_engine():
    auto_df = pd.read_csv(
        "/home/majime/programming/github/ir-search-engine/data/tokenized/auto.csv"
    )
    property_df = pd.read_csv(
        "/home/majime/programming/github/ir-search-engine/data/tokenized/property.csv"
    )

    auto_df["posting_list"] = auto_df["tokenized"].apply(create_postings_list)
    property_df["posting_list"] = property_df["tokenized"].apply(create_postings_list)

    main_df = pd.concat([auto_df, property_df])
    main_df = main_df.reset_index(drop=True)

    corpus = set()
    for l in main_df.posting_list:
        for word in l:
            corpus.add(word)
    corpus = sorted(list(corpus))

    inverted_list = create_inverted_list(main_df, corpus)

    perm_index = permuterm_indexing(inverted_list)
    rev_perm_index = reverse_permuterm_indexing(inverted_list)

    n_word_index = make_n_word_index(main_df)
    
    return inverted_list, perm_index, rev_perm_index, n_word_index, corpus, main_df

if __name__ == "__main__":
    from transformers import pipeline
    summary_pipeline = pipeline("summarization")
    
    with open("/home/majime/programming/github/ir-search-engine/models/summary_pipeline.pkl", "wb") as f:
        pickle.dump(summary_pipeline, f)
    