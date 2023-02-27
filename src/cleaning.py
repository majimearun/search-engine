import fitz
import glob
import pickle


def clean(path: str):
    """_summary_

    Args:
        path (str):takes in the path to the pdfs required to be cleaned into a pickle file (page-wise)
    """
    pdfs = glob.glob(path)
    for file in pdfs:
        doc = fitz.open(file)
        doc_info = {}
        doc_info["text"] = [(page.get_text(), i) for i, page in enumerate(doc)]
        doc_info["metadata"] = doc.metadata
        fh = open(file.replace("pdf", "pkl"), "wb")
        pickle.dump(doc_info, fh)


if __name__ == "__main__":
    # works with the absolute path only for some reason
    clean("/home/majime/programming/github/ir-search-engine/data/pdfs/Auto/*.pdf")
    clean("/home/majime/programming/github/ir-search-engine/data/pdfs/Property/*.pdf")
