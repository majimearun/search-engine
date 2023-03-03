# Search Engine

My first search engine built using Python, Numpy, Pandas and spaCy

## Functionalities

### Boolean retrieval and filtering: Free and phrase queries

By default passing a string is an **OR** query where we search for documents that contain any of the words in the query string. For example, if we pass the query string `python pandas` we will get all the documents that contain either `python` or `pandas` or both. 

Additionally, if we pass any of the words in the query string in double quotes, we will get all the documents that contain the phrase in the double quotes. For example, if we pass the query string `"python" "pandas"` we will get all the documents that contain both words `"python"` and `"pandas"` (an **AND** query).

**TODO:** Currently there are two things that are being implemented. 

1. Every word that needs to be present in the document as an and query needs to be in double quotes separately. As in if we want both the words `python` and `pandas`, we would need to pass the query string as `"python" "pandas"`. It would be better if we could pass the query string as `"python pandas"`.

2. Currently, even if one **AND** query is present in the string, we consider that the user wants any other words also included, otherwise they wouldn't have typed it in (i.e., we are considering a generic user base). So we are taking an intersection of all results which essentially equates to being an **AND** query of all the words. Instead if our ranking algorithm is implemented well enough, the filtering process can just focus on the **AND** queries and the rest of the words can be ignored (the ranking will take care of the the order the results are sent to the user).

If the option `is_phrase` is set to `True`, we nbow consider the order of the words to be important (phrase queries). It is implemented using biwords. So if we pass the string `I love python`, we will get all the documents that contain the phrase `I love` **AND** `love python` wherein each of the words in the biword is present in the document but the biwords themselves might be in a different order in the docuemt.

### Wildcard queries

Wildcard queries are queries that contain a wildcard character `*` which can be used to match any number of characters. For example, if we pass the query string `pyth*n` we will get all the documents that contain words that start with `pyth` and end with `n`. We can pass wildcard queries in three different ways:

1. At the beginning of the word: `*thon`

2. At the end of the word: `pyth*`

3. In the middle of the word: `py*on`

**Note**: Currently only one wildcard character `*` is supported per query word (though each query word in the query string can have its own wildcard character), so a query like `p*yth*n` is translated to `p*n` in the backend.

Additionally while using wildcard characters in a phrase query, it is retirved as on **OR** query on the biwords instead of an **AND** query.

**Wildcard queries are automtically identified** and the queried on, and don't need any extra effort on the user's part.

**TODO:** As of now the search functionality will consider any `*` as a wildcard query, so I am currently building an option to turn wilcard querying off so they can search for test containing an actual `*` in them.


### Ranking

Ranking is done based on the `tf-idf` scores of the documents with respect to the query. The `tf-idf` score is calculated document at a time and the results are sorted and sent to the main `search` function. Only those scores that pass the initial boolean filter will have their scores sent from the backend. If the user just wants a boolean filtered result he/she/they just need to turn the `ranked` parameter to `False`. 

If wildcard characters are present in the query, all the words that match the wildcard query contribute to the score. For example if we pass `dat*`, both `data` and `date`  (and any others that match as well) will contribute to the score.

While retieving the documents, the user can choose the number of documents they want retrieved using the `retrieve_n` parameter in the engine's `search` function. 

**Note:** Biwords do not have a seperate ranking and each word individually counts towards ranking. The initial filter takes care that the documents do indeed contain these biwords. Though performance comparisons are yet to be done, it is expected that the ranking will be better if the biwords are ranked as well.

### Spelling correction

Spelling correction is done using a slightly modified version of the `levenshtien edit distance` algorithm. If the `spell_check` option is set to `True` and the given query string does not match any documents, the engine will try to find the closest match to every word from the corpus and return the documents that contain those words.

Additionally, to the normal edit distance algorithm, I added a check if the word has two adjacent characters swapped (i.e.. now a new operation *swapping adjacent characters* is added to the normal *replacing, inserting, and deleting* options to find the minimum distance between two words).

Spell checks ignore the wildcard characters and only check for the words that are not wildcard queries. It currently converts phrase and **AND** queries to **OR** queries and then checks for the closest match as we are changing the query itself.

### Autocomplete suggestions

Using the same edit distance algorithm, we find words with the smallest distance to the last word in the query from the corpus. If two words have the same distance, we compare it based on its frequency in the corpus. The word that occurs more frequently is likelier to be the correct 'autocomplete' word.

To check for possible autocomplete results instead of a search, set the `autocomplete` parameter to `True` in the `search` function.

The lemmatized worda are converted back to possible 'un'lemmatized words using the `lemminflect` package.

## Usage instructions

1. Clone or download the repository and setup the environment from the `env.yml` file using

```
conda env create -f env.yml
```

2. Activate the environment using

```
conda activate search_engine
```

3. If you want to change the pdfs being read/converted, do the needful and modify the paths wherver necessary. If not just continue with the next step.

4. Run the `cleaning.py`,  `tokenizing.py` and `setup.py` file in the same order.

5. See all the possible usage examples in `exmaple_usage.ipynb` and fit it to use in your application.


