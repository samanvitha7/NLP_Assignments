# NLP Lab Assignment 10: Word Embeddings for Sentiment and Analogies

`run_assignment10.py` implements the three questions in the assignment PDF. It
samples 10,000 training and 1,000 test reviews with stratification, preprocesses
review text, evaluates Google News Word2Vec, GloVe, and FastText embeddings,
and compares average and IDF-weighted sentence vectors in a one-hidden-layer
sentiment classifier. It also evaluates the standard `questions-words.txt`
analogy questions and records category and overall accuracy.

## Input files

Download and unpack the Amazon Reviews Polarity dataset from the Kaggle link in
the assignment. Pass its `train.csv` and `test.csv` paths to the script. The
expected files have three columns in this order: label, title, review text.
Headerless Kaggle CSVs are supported.

Download the following embeddings from the links in the assignment and pass
their local paths:

* Google News Word2Vec: binary `.bin` (or a gensim `.kv` file)
* GloVe: plain text word vectors
* FastText: `wiki-news-300d-1M-subword.vec`

Install dependencies in the Python environment used to run the script:

```bash
python -m pip install numpy pandas scikit-learn gensim
```

## Run

From the repository root:

```bash
python Assignment10/run_assignment10.py \
  --train path/to/train.csv \
  --test path/to/test.csv \
  --word2vec path/to/GoogleNews-vectors-negative300.bin \
  --glove path/to/glove.6B.300d.txt \
  --fasttext path/to/wiki-news-300d-1M-subword.vec
```

The analogy questions file is downloaded automatically on the first run. To
use a local copy, supply `--analogies path/to/questions-words.txt`. Use
`--seed` to reproduce the stratified samples and neural-network initialization.

## Outputs

Files are written to `Assignment10/results/`:

* `train_sample.csv` and `test_sample.csv`: sampled source rows.
* `sentiment_results.csv`: accuracy and macro F1 for each of the six
  embedding/representation combinations.
* `analogy_results.json`: overall and per-category analogy accuracies, plus
  up to 30 example predictions for each embedding.

Preprocessing lowercases text, removes punctuation and digits, tokenizes with
an alphabetic tokenizer, and removes English stopwords. TF-IDF weights are
fitted on the sampled training reviews only. Sentences with no in-vocabulary
words receive a zero vector. The classifier standardizes features using
training data only and uses early stopping.
