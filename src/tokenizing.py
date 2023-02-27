# import libraries
import pandas as pd
import glob
import pickle
import spacy

nlp = spacy.load("en_core_web_sm")


def read_pickle_into_pages(files):
    """
    Takes in a path to the pickle files and returns a dataframe with the document name, page number and text.
    """
    main_df = pd.DataFrame(columns=["document_name", "page_number", "text"])
    for file in files:
        fh = open(file, "rb")
        doc_info = pickle.load(fh)
        text = [page[0] for page in doc_info["text"]]
        page_number = [page[1] for page in doc_info["text"]]
        df = pd.DataFrame(columns=["document_name", "page_number", "text"])
        df["document_name"] = [file.split("/")[-1][:-4]] * len(text)
        df["page_number"] = page_number
        df["text"] = text
        main_df = pd.concat([main_df, df])
    return main_df


def seperate_df_into_paragraphs(df, threshold=100):
    """
    Seperate the dataframe into paragraphs.
    """
    main_df = pd.DataFrame(
        columns=["document_name", "page_number", "paragraph_number", "text"]
    )
    for _, row in df.iterrows():
        text = row["text"].split("\n \n")
        text = (" ".join(text)).split("\n\n")
        text = (" ".join(text)).split(".\n")
        # remove empty strings
        text = [paragraph for paragraph in text if paragraph != ""]
        # if only spaces, remove
        text = [paragraph for paragraph in text if paragraph != " "]
        # if the length of the paragraph is less than threshold, remove
        text = [paragraph for paragraph in text if len(paragraph) > threshold]

        for paragraph_number, paragraph in enumerate(text):
            df = pd.DataFrame(
                columns=["document_name", "page_number", "paragraph_number", "text"]
            )
            df["document_name"] = [row["document_name"]]
            df["page_number"] = [row["page_number"]]
            df["paragraph_number"] = [paragraph_number]
            df["text"] = [paragraph]
            main_df = pd.concat([main_df, df])
    return main_df


def tokenize(
    df,
    nlp,
    allow_digits=False,
    allow_punct=False,
    allow_stopwords=False,
    allow_numbers=False,
):
    df["tokenized"] = df["text"].apply(
        lambda x: " ".join(
            [
                token.lemma_.lower()
                for token in nlp(x)
                if (token.is_alpha or allow_digits)
                and (not token.is_punct or allow_punct)
                and (not token.is_stop or allow_stopwords)
                and (not token.like_num or allow_numbers)
            ]
        )
    )


if __name__ == "__main__":
    auto_pkls = glob.glob(
        "/home/majime/programming/github/ir-search-engine/data/pkls/Auto/*.pkl"
    )
    property_pkls = glob.glob(
        "/home/majime/programming/github/ir-search-engine/data/pkls/Property/*.pkl"
    )
    auto_df = read_pickle_into_pages(auto_pkls)
    property_df = read_pickle_into_pages(property_pkls)

    auto_final = seperate_df_into_paragraphs(auto_df)
    auto_final.reset_index(inplace=True, drop=True)

    property_final = seperate_df_into_paragraphs(property_df)
    property_final.reset_index(inplace=True, drop=True)

    tokenize(auto_final, nlp)
    tokenize(property_final, nlp)

    auto_final.to_csv(
        "/home/majime/programming/github/ir-search-engine/data/tokenized/auto.csv",
        index=False,
    )
    property_final.to_csv(
        "/home/majime/programming/github/ir-search-engine/data/tokenized/property.csv",
        index=False,
    )
