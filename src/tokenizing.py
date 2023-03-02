import pandas as pd
import glob
import pickle
import spacy

# model used for lemmatizing
nlp = spacy.load("en_core_web_sm")


def read_pickle_into_pages(files: list[str]):
    """Takes in a path to the pickle files and returns a dataframe with the document name, page number and text.

    Args:
        files (list): list of paths to the pickle files

    Returns:
        pd.DataFrame: dataframe with the document name, page number and text
    """
    main_df: pd.DataFrame = pd.DataFrame(
        columns=["document_name", "page_number", "text"]
    )
    for file in files:
        # loading the pickle file from path
        fh = open(file, "rb")
        doc_info = pickle.load(fh)
        text = [page[0] for page in doc_info["text"]]
        # 0 indexed page numbers
        page_number = [page[1] for page in doc_info["text"]]
        # creating a dataframe with a row for each page
        df = pd.DataFrame(columns=["document_name", "page_number", "text"])
        # The document name is the last element of the path split by `/` and removing the `.pkl` extension
        df["document_name"] = [file.split("/")[-1][:-4]] * len(text)
        df["page_number"] = page_number
        df["text"] = text
        # concatenating the dataframe to the main dataframe
        main_df = pd.concat([main_df, df])
    return main_df


def seperate_df_into_paragraphs(
    df: pd.DataFrame, threshold: int = 100, break_paragraph_at: int = 3000
):
    """Seperate the dataframe from each row being a page to each row being a paragraph

    Args:
        df (pd.DataFrame): dataframe with the document name, page number and text
        threshold (int, optional): threshold for the length of the paragraph. Defaults to 100.
        break_paragraph_at (int, optional): maximum length of a paragraph. Defaults to 3000.

    Returns:
        pd.DataFrame: dataframe with the document name, page number, paragraph number and text
    """
    main_df = pd.DataFrame(
        columns=["document_name", "page_number", "paragraph_number", "text"]
    )
    for _, row in df.iterrows():
        # split by double new line (observed a spce in between in the txt files)
        text = row["text"].split("\n \n")
        text = (" \n\n".join(text)).split("\n\n")
        text = (" .\n".join(text)).split(".\n")
        # break the paragraphs if its length is over the break_paragraph_at parameter
        filtering = (" ".join(text)).split("\n")
        filtered_text = []
        string = ""
        for paragraph in filtering:
            if len(string) > break_paragraph_at:
                filtered_text.append(string)
                string = ""
            string += paragraph
        if string != "":
            filtered_text.append(string)
        text = filtered_text
        # remove empty paragraphs
        text = [paragraph for paragraph in text if paragraph != ""]
        # remove paragraphs with only spaces
        text = [paragraph for paragraph in text if paragraph != " "]
        # remove paragraphs with length less than the threshold
        text = [paragraph for paragraph in text if len(paragraph) > threshold]

        # creating a dataframe with a row for each paragraph (paragraph number and page number are 0 indexed)
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
    """removes stopwords, punctuations, digits and numbers and lemmatizes the text. Creates a new column called `tokenized` which contains the tokenized text. Modifies the dataframe in place.

    Args:
        df (pd.DataFrame): dataframe with the document name, page number, paragraph number and text
        nlp (spacy.lang.en.English): spacy model used for lemmatizing
        allow_digits (bool, optional): whether to allow digits. Defaults to False.
        allow_punct (bool, optional): whether to allow punctuations. Defaults to False.
        allow_stopwords (bool, optional): whether to allow stopwords. Defaults to False.
        allow_numbers (bool, optional): whether to allow numbers. Defaults to False.

    Returns:
        None
    """
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
    #  setting up the tokenized dataframes and saving them for future use
    auto_pkls = glob.glob(
        "/home/majime/programming/github/search-engine/data/pkls/Auto/*.pkl"
    )
    property_pkls = glob.glob(
        "/home/majime/programming/github/search-engine/data/pkls/Property/*.pkl"
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
        "./data/tokenized/auto.csv",
        index=False,
    )
    property_final.to_csv(
        "./data/tokenized/property.csv",
        index=False,
    )
