# Copyright (C) 2023 by Arunachala Amuda Murugan
# 
# Lisence: GNU General Public License v3.0

import fitz
import glob
import pickle


def clean(path: str):
    """Cleans the pdfs in a given folder into a pickle file (page-wise) per pdf, and saves it a folder with the same parent as the pdfs, but in a `pkl/` folder

    Args:
        path (str):takes in the path to the folder with pdfs to be cleaned

    Returns:
        None
    """
    # Adding the wildcard to the path to match all pdfs within the folder
    path += "*.pdf"
    # Getting paths of all the pdfs in the folder into a list
    pdfs: list[str] = glob.glob(path)
    for file in pdfs:
        doc = fitz.open(file)
        doc_info = {}
        doc_info["text"] = [(page.get_text(), i) for i, page in enumerate(doc)]
        fh = open(file.replace("pdf", "pkl"), "wb")
        pickle.dump(doc_info, fh)


if __name__ == "__main__":
    # works with the absolute path only for some reason
    clean("./data/pdfs/Auto/")
    clean("./data/pdfs/Property/")
