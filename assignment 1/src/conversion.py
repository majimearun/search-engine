import fitz
import glob
import pickle

auto_pdfs = glob.glob(
    "/home/majime/programming/github/information-retrieval-assignments/assignment 1/pdfs/Auto/*.pdf"
)
property_pdfs = glob.glob(
    "/home/majime/programming/github/information-retrieval-assignments/assignment 1/pdfs/Property/*.pdf"
)

for file in auto_pdfs:
    doc = fitz.open(file)
    doc_info = {}
    doc_info["text"] = [(page.get_text(), i) for i, page in enumerate(doc)]
    doc_info["metadata"] = doc.metadata
    fh = open(file.replace("pdf", "pkl"), "wb")
    pickle.dump(doc_info, fh)
    
for file in property_pdfs:
    doc = fitz.open(file)
    doc_info = {}
    doc_info["text"] = [(page.get_text(), i) for i, page in enumerate(doc)]
    doc_info["metadata"] = doc.metadata
    fh = open(file.replace("pdf", "pkl"), "wb")
    pickle.dump(doc_info, fh)
    