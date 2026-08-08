# Assignment 1 - Telugu Text Tokenization

**Course:** AI357 Natural Language Processing  
**Language:** Telugu  
**Main implementation:** `Telugu_NLP_Assignment.ipynb`

## Overview

This project implements a regex-based Telugu text preprocessing pipeline with:

- sentence tokenization
- word tokenization
- corpus statistics
- Parquet output for processed sentences

The current repository is notebook-driven. The code is implemented inside `Telugu_NLP_Assignment.ipynb`, and the processed outputs are already available under `output/indiccorp/`.

## What The Code Does

### 1. Word tokenization

The tokenizer walks through each sentence character by character and matches token patterns in priority order.

Matched token types include:

- URLs
- email addresses
- dates
- decimal numbers
- integers
- Telugu-script text
- Latin words
- punctuation

If nothing matches, the tokenizer falls back to consuming a single non-whitespace character.

### 2. Sentence tokenization

Sentence splitting uses a protection step before splitting so that dots inside:

- URLs
- email addresses
- decimal numbers

do not trigger sentence boundaries.

The sentence boundary logic splits on punctuation such as:

- `.`
- `!`
- `?`
- Telugu danda

### 3. Paragraph pipeline

`tokenize_paragraph()` combines the two steps:

1. split paragraph into sentences
2. tokenize each sentence into words

### 4. Corpus statistics

The `CorpusStats` class tracks:

- total sentences
- total words
- total characters in the original text
- total token characters
- unique token types

It then derives:

- average sentence length
- average word length
- type-token ratio

## Dependencies

The notebook imports:

- `re`
- `json`
- `os`
- `pathlib.Path`
- `pyarrow`
- `datasets`

## Current Output

The repository already contains the processed IndicCorpV2 Telugu outputs:

- `output/indiccorp/indiccorp_te_tokenized.parquet`
- `output/indiccorp/indiccorp_te_statistics.json`

### Statistics

```json
{
  "total_sentences": 20181498,
  "total_words": 249508148,
  "total_characters": 1563922281,
  "average_sentence_length": 12.3632,
  "average_word_length": 5.3489,
  "unique_token_types": 3737704,
  "type_token_ratio": 0.01498
}
```

## Notebook Sections

`Telugu_NLP_Assignment.ipynb` currently includes:

- imports
- regex pattern definitions
- `word_tokenize()`
- `_protect()` and `_restore()`
- `sentence_tokenize()`
- `tokenize_paragraph()`
- `CorpusStats`
- IndicCorpV2 task configuration
- OSCAR-2301 task configuration
- sample tokenization demo

## IndicCorpV2 Task

The notebook contains the current configuration for the Telugu subset of IndicCorpV2:

- dataset: `ai4bharat/IndicCorpV2`
- configuration: `indiccorp_v2`
- split: `tel_Telu`
- streaming: enabled
- text field: `text`

The notebook shows this task as already processed, and the output files above reflect that run.

## OSCAR-2301 Task

The notebook also includes a configuration stub for OSCAR-2301:

- dataset: `oscar-corpus/OSCAR-2301`
- configuration: `te`
- split: `train`
- streaming: enabled
- text field: `content`

This task is marked as pending access approval in the notebook.

## How To Use

### Open the notebook

```bash
python -m jupyter notebook Telugu_NLP_Assignment.ipynb
```

### Inspect the existing outputs

The processed corpus files are already in `output/indiccorp/`.

### Re-run the notebook

Run the cells from top to bottom to:

1. define the regex tokenizers
2. test sample tokenization
3. review the corpus statistics
4. review the IndicCorpV2 and OSCAR configurations

## Notes

- The earlier README mentioned standalone scripts such as `run_indiccorp.py` and `generate_stats_from_parquet.py`, but those are not present in the current repository layout.
- The current code lives in the notebook, so this README now reflects that structure.

## Repository Layout

```text
Assignment1/
|-- README.md
|-- Telugu_NLP_Assignment.ipynb
|-- NLP_Assignment1 (1).pdf
|-- output/
|   `-- indiccorp/
|       |-- indiccorp_te_tokenized.parquet
|       `-- indiccorp_te_statistics.json
`-- .gitignore
```
