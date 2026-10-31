"""End-to-end implementation for NLP Lab Assignment 10.

Expected data files are the Kaggle Amazon Reviews ``train.csv`` and
``test.csv`` (columns: label, title, text). Embedding files are supplied by
the user because of their size and separate download licenses.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "results"
ANALOGY_URL = "https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt"
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def load_stopwords():
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)
    except ImportError:
        return set("a an the and or but if while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once is am are was were be been being have has had do does did having it its this that these those i me my we our you your he she they them what which who whom not no nor as very can will just".split())


STOPWORDS = load_stopwords()


def preprocess(text: object) -> list[str]:
    """Lowercase, remove digits/punctuation, tokenize, and remove stopwords."""
    if pd.isna(text):
        return []
    return [w for w in TOKEN_RE.findall(str(text).lower()) if w not in STOPWORDS]


def read_amazon_csv(path: Path) -> pd.DataFrame:
    # Kaggle's Amazon Reviews Polarity CSV may be headerless (label,title,text).
    frame = pd.read_csv(path, header=None, names=["label", "title", "text"],
                        dtype={"label": "int64", "title": "string", "text": "string"})
    if str(frame.iloc[0]["label"]).lower() == "label":
        frame = frame.iloc[1:].copy()
        frame["label"] = frame["label"].astype(int)
    frame["review"] = frame["title"].fillna("") + " " + frame["text"].fillna("")
    return frame


def stratified_samples(train_path: Path, test_path: Path, seed: int):
    train, test = read_amazon_csv(train_path), read_amazon_csv(test_path)
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        raise ValueError("Both source CSVs must contain at least two sentiment classes.")
    n_train, n_test = min(10000, len(train)), min(1000, len(test))
    tr, _ = train_test_split(train, train_size=n_train, random_state=seed,
                             stratify=train["label"])
    te, _ = train_test_split(test, train_size=n_test, random_state=seed,
                             stratify=test["label"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def load_vectors(kind: str, path: Path):
    """Load Google News W2V, GloVe text, or FastText .vec via gensim."""
    from gensim.models import KeyedVectors
    if kind == "word2vec":
        if path.suffix == ".model":
            from gensim.models import Word2Vec
            return Word2Vec.load(str(path)).wv
        return KeyedVectors.load(str(path), mmap="r") if path.suffix == ".kv" else KeyedVectors.load_word2vec_format(str(path), binary=path.suffix.lower() == ".bin")
    # GloVe and wiki-news FastText are distributed as text vectors. GloVe
    # dumps differ on whether they include the optional count/dimension row.
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first = stream.readline().split()
    has_header = len(first) == 2 and all(part.isdigit() for part in first)
    return KeyedVectors.load_word2vec_format(str(path), binary=False, no_header=not has_header)


def sentence_matrix(token_rows, vectors, tfidf=None, feature_index=None):
    dim = vectors.vector_size
    result = np.zeros((len(token_rows), dim), dtype=np.float32)
    for i, words in enumerate(token_rows):
        found = [(w, vectors[w]) for w in words if w in vectors]
        if not found:
            continue
        if tfidf is None:
            result[i] = np.mean([v for _, v in found], axis=0)
        else:
            weights = np.asarray([
                float(tfidf[i, feature_index[w]]) if w in feature_index else 0.0
                for w, _ in found
            ], dtype=np.float32)
            if weights.sum() <= 0:
                weights = np.ones(len(found), dtype=np.float32)
            result[i] = np.average(np.asarray([v for _, v in found]), axis=0, weights=weights)
    return result


def analogy_sections(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANALOGY_URL, path)
    sections, current = [], None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current = {"category": line[1:].strip(), "items": []}
                sections.append(current)
            elif current and len(line.split()) == 4:
                current["items"].append(line.lower().split())
    return sections


def evaluate_analogies(sections, vectors):
    # Google News keys are case sensitive while questions-words.txt is lower case.
    key_for_lower = {}
    for key in vectors.index_to_key:
        key_for_lower.setdefault(key.lower(), key)
        if key == key.lower():
            key_for_lower[key.lower()] = key
    rows, correct, total = [], 0, 0
    for section in sections:
        cat_total = cat_correct = 0
        for a, b, c, expected in section["items"]:
            if not all(w in key_for_lower for w in (a, b, c, expected)):
                continue
            try:
                prediction = vectors.most_similar(
                    positive=[key_for_lower[b], key_for_lower[c]],
                    negative=[key_for_lower[a]], topn=1)[0][0]
            except (KeyError, ValueError):
                continue
            hit = prediction.lower() == expected
            cat_total += 1
            cat_correct += int(hit)
            if total < 30:
                rows.append({"category": section["category"], "a": a, "b": b,
                             "c": c, "expected": expected, "predicted": prediction,
                             "correct": hit})
            total += 1
            correct += int(hit)
        section["evaluated"] = cat_total
        section["accuracy"] = cat_correct / cat_total if cat_total else None
    return {"accuracy": correct / total if total else None, "correct": correct,
            "evaluated": total, "categories": sections, "examples": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=Path("train.csv"))
    ap.add_argument("--test", type=Path, default=Path("test.csv"))
    ap.add_argument("--word2vec", type=Path, help="Google News 300d binary or gensim .kv")
    ap.add_argument("--glove", type=Path, help="GloVe text vectors")
    ap.add_argument("--fasttext", type=Path, help="wiki-news-300d-1M-subword.vec")
    ap.add_argument("--analogies", type=Path, default=Path("questions-words.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    missing = [name for name in ("word2vec", "glove", "fasttext") if getattr(args, name) is None]
    if missing:
        ap.error("provide embedding files with --word2vec, --glove, and --fasttext")

    OUT.mkdir(parents=True, exist_ok=True)
    train, test = stratified_samples(args.train, args.test, args.seed)
    train.to_csv(OUT / "train_sample.csv", index=False)
    test.to_csv(OUT / "test_sample.csv", index=False)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()
    train_tokens = [preprocess(x) for x in train["review"]]
    test_tokens = [preprocess(x) for x in test["review"]]
    # Fit IDF on training documents only, then use each word's document weight.
    tfidf = TfidfVectorizer(tokenizer=preprocess, preprocessor=None, token_pattern=None,
                            lowercase=False, vocabulary=None, dtype=np.float32)
    train_sparse = tfidf.fit_transform(train["review"])
    test_sparse = tfidf.transform(test["review"])
    feature_names = tfidf.get_feature_names_out()
    feature_index = {word: i for i, word in enumerate(feature_names)}
    analogy = analogy_sections(args.analogies)
    metrics, analogy_summary = [], {}

    for kind in ("word2vec", "glove", "fasttext"):
        vectors = load_vectors(kind, getattr(args, kind))
        analogy_summary[kind] = evaluate_analogies(analogy, vectors)
        for mode in ("average", "tfidf_weighted"):
            x_train = sentence_matrix(train_tokens, vectors,
                                      None if mode == "average" else train_sparse,
                                      feature_index)
            x_test = sentence_matrix(test_tokens, vectors,
                                     None if mode == "average" else test_sparse,
                                     feature_index)
            # A single hidden layer as required; scaling is fitted on train only.
            model = make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(128,), activation="relu", solver="adam",
                alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                max_iter=30, early_stopping=True, n_iter_no_change=4,
                random_state=args.seed, verbose=False))
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            metrics.append({"embedding": kind, "representation": mode,
                            "accuracy": accuracy_score(y_test, pred),
                            "macro_f1": f1_score(y_test, pred, average="macro"),
                            "train_samples": len(train), "test_samples": len(test),
                            "oov_test_sentences": int(np.sum(~np.any(x_test, axis=1)))})
            print(f"{kind:9s} {mode:14s} macro-F1={metrics[-1]['macro_f1']:.4f}")

    pd.DataFrame(metrics).to_csv(OUT / "sentiment_results.csv", index=False)
    (OUT / "analogy_results.json").write_text(json.dumps(analogy_summary, indent=2), encoding="utf-8")
    print(f"Saved samples, sentiment scores, and analogy evaluations under {OUT}")


if __name__ == "__main__":
    main()
