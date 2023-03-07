# CS F469: Information Retrieval Project 1

**Problem Statement:** To build a search engine for a given corpus of documents. 

- By Arunachala Amuda Murugan (2021A7PS0205H) 
- Completed on 5th March 2023

## Feature List

1. A pipeline for cleaning and extracting text from the pdf filesm (for current and future ones as well).

2. Seperation of documents (paragraphs) and lemmatization.

3. Forming posting lists, inverted index and permuterm indexes (normal and reverse).

4. Boolean retrieval/filtering: Free (**AND** and **OR**) and phrase queries (biwords).

5. Wildcard matching in all kinds of queries (automated).

6. Ranking of documents based on TF-IDF scores (including all matches of any wildcard query word).

7. Spelling correction (edit distance and swapping adjacent characters) for all non wildcard query words.

8. Basic autocomplete suggestions instead of search results.
9. Summarization of retrieved documents.

## Usage/Replication Instructions

1. Clone or download the repository **(assignment branch)** and setup the environment from the `env.yml` file using

```
conda env create -f env.yml
```

1. Activate the environment using

```
conda activate search_engine
```

3. If you want to change the pdfs being read/converted, do the needful and modify the paths wherever necessary. If not just continue with the next step.

4. Run the `cleaning.py`, `tokenizing.py` and `setup.py` file in the same order.

5. See all the possible usage examples in `2021A7PS0205H_Info_Retr_Assignment1_Report.pdf` or in its corresponding `ipynb` file and fit it to use in your application.

6. Other statistics can also be found in the `2021A7PS0205H_Info_Retr_Assignment1_Report.pdf` file.

## Libraries Used

1. `NumPy` and `Pandas` for core functionality

2. `spaCy` for lemmatization

3. `lemminflect` for 'un'lemmatizing words

4. `fitz` for pdf extraction

5. `transformers` (huggingface) for summarization 
