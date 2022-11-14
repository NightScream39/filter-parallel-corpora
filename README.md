# Filter-parallel-corpora
Filter en-ru parallel corpus with opusfilter and laserembeddings 

All filtered/deleted corpuses are in folder: 

* `/datasets`

Corpus filtered by opusfilter: 

   - **`WikiMatrix-filtered.en-ru.en`**
 
   - **`WikiMatrix-filtered.en-ru.ru`**

Were used:

- *`LengthFilter `*
- *`LengthRatioFilter `*
- *`LongWordFilter `*
- *`HtmlTagFilter `*
- *`TerminalPunctuationFilter `*
- *`CharacterScoreFilter `*
- *`NonAlphaFilter: (customfilter.py)`*
- *`NonAlphaFilter: (customfilter.py)`*
- *`UppercaseFilter: (UpperCaseFilter.py)`*
Corpus deleted by opusfilter:

- **`WikiMatrix-filteredDROP.en-ru.en `**

- **`WikiMatrix-filteredDROP.en-ru.ru`**

Corpus filtered by complex scores:

- **`WikiMatrix-filteredbyComplexFilter.en-ru.en `**
- **`WikiMatrix-filteredbyComplexFilter.en-ru.ru`**

Corpus deleted by complex scores:

- **`WikiMatrix-filteredDROPbyComplexFilter.en-ru.en`**
- **`WikiMatrix-filteredDROPbyComplexFilter.en-ru.ru`**

Corpus with summary deleted sentences:
- **`WikiMatrix-summaryDROP.en`**
- **`WikiMatrix-summaryDROP.en`**

# Install
Install requirements.txt:
- `pip install -r requirements.txt`

Set `PYTHONPATH` environment variable to varikn directory:

- `export PYTHONPATH=$PYTHONPATH:/*path to directory with builded files (varikn.py)*`

# Usage filter: 

> *If u want to create files with deleted sentences by opusfilter, uncomment `filterfalse: true` into the `opus.yaml`*

  With opusfilter and complex filter: 
  
  - `python fpc.py opus.yaml -o -c`
  
  With opusfilter: 
  
  - `python fpc.py opus.yaml -o`



