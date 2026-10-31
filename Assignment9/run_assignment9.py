"""
NLP Lab Assignment 9 - Hindi POS Tagging with MEMM and CRF
Comparative Study: Naive Bayes vs. HMM vs. MEMM vs. CRF

This script implements:
1. Maximum Entropy Markov Model (MEMM) based on arne-cl/nltk-maxent-pos-tagger
2. Conditional Random Field (CRF) tagger using sklearn-crfsuite
3. Naive Bayes & HMM from Assignment 7
4. Complete evaluation on dev and test sets
5. Per-tag F1 comparison and overall metrics
6. High-resolution bar charts and summary CSV files
"""

import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn_crfsuite

# ---------------------------------------------------------
# Constants and Directories
# ---------------------------------------------------------
START, END, UNK = "<S>", "</S>", "<UNK>"
DATA_DIR = os.path.join("Assignment9", "Hindi-POS-Data")
OUT_DIR = os.path.join("Assignment9", "results")
CHARTS_DIR = os.path.join(OUT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------
def read_pos_file(path):
    """
    Reads a POS-tagged corpus file where each line has: <word>\t<tag> or <word> <tag>.
    A blank line denotes sentence boundary.
    Returns: list of (words, tags) tuples.
    """
    sents, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if words:
                    sents.append((words, tags))
                    words, tags = [], []
                continue
            parts = line.split()
            words.append(parts[0])
            tags.append(parts[-1])
    if words:
        sents.append((words, tags))
    return sents


print("Loading datasets...")
train_sents = read_pos_file(os.path.join(DATA_DIR, "hindi-train-pos.txt"))
dev_sents = read_pos_file(os.path.join(DATA_DIR, "hindi-dev-pos.txt"))
test_sents = read_pos_file(os.path.join(DATA_DIR, "hindi-test-pos.txt"))

print(f"Loaded: Train={len(train_sents)} sents, Dev={len(dev_sents)} sents, Test={len(test_sents)} sents")

# Tag frequencies and canonical labels sorted by training frequency
tag_freq = Counter(t for _, ts in train_sents for t in ts)
labels = [t for t, _ in tag_freq.most_common()]
print(f"Total Unique Tags: {len(labels)}")
print("Top 10 tags by frequency:", labels[:10])


# ---------------------------------------------------------
# 2. Assignment Feature Extractor (Questions 2a - 2d)
# ---------------------------------------------------------
def extract_word_features(words, i):
    """
    Extracts features for words[i] according to Assignment 9 specifications:
    a. Prefixes Upto Length 4 (Prefix lengths 1, 2, 3, 4)
       - If prefix of particular length is not found, feature = 'NULL'
    b. Suffixes Upto Length 4 (Suffix lengths 1, 2, 3, 4)
       - If suffix of particular length is not found, feature = 'NULL'
    c. Context Features:
       - Previous Word
       - Current Word
       - Next Word
    d. Binary Length Feature:
       - If length <= 4: 'LESS', else 'MORE'
    """
    w = words[i]
    n = len(w)
    
    feats = {
        # Feature d: Binary Length Feature
        "length_feature": "LESS" if n <= 4 else "MORE",
        
        # Feature c: Context Features
        "word": w,
        "prev_word": words[i - 1] if i > 0 else START,
        "next_word": words[i + 1] if i < len(words) - 1 else END,
        
        # Feature a: Prefixes Upto Length 4
        "prefix_1": w[:1] if n >= 1 else "NULL",
        "prefix_2": w[:2] if n >= 2 else "NULL",
        "prefix_3": w[:3] if n >= 3 else "NULL",
        "prefix_4": w[:4] if n >= 4 else "NULL",
        
        # Feature b: Suffixes Upto Length 4
        "suffix_1": w[-1:] if n >= 1 else "NULL",
        "suffix_2": w[-2:] if n >= 2 else "NULL",
        "suffix_3": w[-3:] if n >= 3 else "NULL",
        "suffix_4": w[-4:] if n >= 4 else "NULL",
    }
    return feats


# ---------------------------------------------------------
# 3. Naive Bayes Tagger (from Assignment 7)
# ---------------------------------------------------------
class NaiveBayesTagger:
    """
    Generative Baseline:
    P(tag | feats) ~ P(tag) * prod_f P(feat_f | tag)
    Uses add-alpha smoothing. At test time, previous tag is the predicted tag.
    """
    FEATS = ["len", "word", "prev_word", "next_word", "prev_tag"]

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    @staticmethod
    def _feats(words, i, prev_tag):
        w = words[i]
        return {
            "len": "MORE" if len(w) > 4 else "LESS",
            "word": w,
            "prev_word": words[i - 1] if i > 0 else START,
            "next_word": words[i + 1] if i < len(words) - 1 else END,
            "prev_tag": prev_tag,
        }

    def fit(self, sents):
        self.tags = sorted({t for _, ts in sents for t in ts})
        self.tag_idx = {t: i for i, t in enumerate(self.tags)}
        T = len(self.tags)
        self.tag_count = np.zeros(T)
        cnt = {f: defaultdict(lambda: np.zeros(T)) for f in self.FEATS}
        for words, tags in sents:
            prev = START
            for i, t in enumerate(tags):
                ti = self.tag_idx[t]
                self.tag_count[ti] += 1
                for f, v in self._feats(words, i, prev).items():
                    cnt[f][v][ti] += 1
                prev = t
        self.log_prior = np.log(self.tag_count / self.tag_count.sum())
        self.vocab = {}
        self.logp = {}
        self.logunk = {}
        for f in self.FEATS:
            vals = list(cnt[f].keys())
            V = len(vals)
            self.vocab[f] = {v: j for j, v in enumerate(vals)}
            M = np.stack([cnt[f][v] for v in vals])
            denom = self.tag_count + self.alpha * (V + 1)
            self.logp[f] = np.log((M + self.alpha) / denom)
            self.logunk[f] = np.log(self.alpha / denom)
        return self

    def _score(self, feats):
        s = self.log_prior.copy()
        for f, v in feats.items():
            j = self.vocab[f].get(v)
            s += self.logp[f][j] if j is not None else self.logunk[f]
        return s

    def predict_sentence(self, words):
        out, prev = [], START
        for i in range(len(words)):
            t = self.tags[int(np.argmax(self._score(self._feats(words, i, prev))))]
            out.append(t)
            prev = t
        return out


# ---------------------------------------------------------
# 4. HMM Tagger (from Assignment 7)
# ---------------------------------------------------------
class HMMTagger:
    """
    First-order Bigram Hidden Markov Model.
    Transition probability: A[i, j] = P(tag_j | tag_i)
    Emission probability:   B[j, w] = P(word_w | tag_j)
    Decoding: Forward Viterbi algorithm.
    """
    def __init__(self, alpha_trans=0.01, alpha_emit=0.01):
        self.at, self.ae = alpha_trans, alpha_emit

    def fit(self, sents):
        self.tags = sorted({t for _, ts in sents for t in ts})
        self.tag_idx = {t: i for i, t in enumerate(self.tags)}
        T = len(self.tags)
        self.T = T
        trans = np.zeros((T + 1, T + 1))
        emit = defaultdict(lambda: np.zeros(T))
        tag_count = np.zeros(T)
        for words, tags in sents:
            ids = [self.tag_idx[t] for t in tags]
            seq = [T] + ids + [T]
            for a, b in zip(seq[:-1], seq[1:]):
                trans[a, b] += 1
            for w, ti in zip(words, ids):
                emit[w][ti] += 1
                tag_count[ti] += 1
        a_smooth = (trans + self.at) / (trans.sum(axis=1, keepdims=True) + self.at * (T + 1))
        self.logA = np.log(a_smooth[:T, :T])
        self.logA_start = np.log(a_smooth[T, :T])
        self.logA_end = np.log(a_smooth[:T, T])
        vocab = list(emit.keys())
        V = len(vocab)
        self.words = {w: i for i, w in enumerate(vocab)}
        M = np.stack([emit[w] for w in vocab], axis=1)
        denom = tag_count[:, None] + self.ae * (V + 1)
        self.logB = np.log((M + self.ae) / denom)
        self.logB_unk = np.log(self.ae / denom[:, 0])
        return self

    def _emit_matrix(self, words):
        cols = []
        for w in words:
            j = self.words.get(w)
            cols.append(self.logB[:, j] if j is not None else self.logB_unk)
        return np.stack(cols)

    def forward_viterbi(self, words):
        em = self._emit_matrix(words)
        n, T = em.shape
        delta = np.empty((n, T))
        back = np.zeros((n, T), dtype=int)
        delta[0] = self.logA_start + em[0]
        for i in range(1, n):
            cand = delta[i - 1][:, None] + self.logA
            back[i] = cand.argmax(axis=0)
            delta[i] = cand.max(axis=0) + em[i]
        last = delta[-1] + self.logA_end
        path = [int(last.argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i][path[-1]]))
        return [self.tags[k] for k in reversed(path)]


# ---------------------------------------------------------
# 5. Maximum Entropy Markov Model (MEMM)
# ---------------------------------------------------------
class MEMMTagger:
    """
    Maximum Entropy Markov Model (MEMM) POS Tagger.
    Modeled after Ratnaparkhi (1996) and arne-cl/nltk-maxent-pos-tagger:
      P(tag_t | words, history) = exp(w^T f(words, t, history)) / Z(words, history)
    
    Trained via multinomial log-linear Maximum Entropy loss (L-BFGS).
    Supports fast vectorised inference and greedy Markov sequence decoding.
    """
    def __init__(self, C=1.0, max_iter=100):
        self.C = C
        self.max_iter = max_iter
        self.vec = DictVectorizer(sparse=True)
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42
        )

    def fit(self, sents):
        train_feats = []
        train_labels = []
        for words, tags in sents:
            for i, t in enumerate(tags):
                pt = tags[i - 1] if i > 0 else START
                f = extract_word_features(words, i)
                f["prev_tag"] = pt
                train_feats.append(f)
                train_labels.append(t)

        print(f"  [MEMM] Vectorizing {len(train_feats)} instances...")
        X_train = self.vec.fit_transform(train_feats)
        print(f"  [MEMM] Training MaxEnt classifier (shape={X_train.shape})...")
        self.clf.fit(X_train, train_labels)

        self.classes = self.clf.classes_
        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Precompute transition weight matrix for fast inference:
        # trans_weights[prev_tag_idx, cur_tag_idx]
        self.all_prev_tags = list(self.classes) + [START]
        self.prev_tag_to_idx = {pt: i for i, pt in enumerate(self.all_prev_tags)}
        self.trans_weights = np.zeros((len(self.all_prev_tags), self.n_classes))

        for pt, p_idx in self.prev_tag_to_idx.items():
            feat_name = f"prev_tag={pt}"
            if feat_name in self.vec.vocabulary_:
                f_idx = self.vec.vocabulary_[feat_name]
                self.trans_weights[p_idx] = self.clf.coef_[:, f_idx]

        return self

    def predict_sents(self, sents):
        """
        Fast sentence prediction across multiple sentences.
        Computes observation scores in batch, then applies sequential Markov decoding.
        """
        obs_feats = []
        sent_lens = []
        for words, _ in sents:
            sent_lens.append(len(words))
            for i in range(len(words)):
                obs_feats.append(extract_word_features(words, i))

        X_obs = self.vec.transform(obs_feats)
        obs_scores = (X_obs @ self.clf.coef_.T) + self.clf.intercept_

        out_preds = []
        offset = 0
        start_idx = self.prev_tag_to_idx[START]

        for L in sent_lens:
            prev_idx = start_idx
            sent_pred = []
            for pos in range(offset, offset + L):
                scores = obs_scores[pos] + self.trans_weights[prev_idx]
                best_c = int(np.argmax(scores))
                best_tag = self.classes[best_c]
                sent_pred.append(best_tag)
                prev_idx = self.prev_tag_to_idx.get(best_tag, start_idx)
            offset += L
            out_preds.append(sent_pred)

        return out_preds


# ---------------------------------------------------------
# 6. CRF Tagger (sklearn-crfsuite)
# ---------------------------------------------------------
class CRFTagger:
    """
    Linear-chain Conditional Random Field (CRF) POS Tagger.
    Implemented using sklearn-crfsuite as per Question 2.
    Maximizes global sequence conditional likelihood:
      P(Y | X) = (1 / Z(X)) * exp(sum_i (sum_k lambda_k f_k(y_i, X, i) + sum_j mu_j t_j(y_{i-1}, y_i, X, i)))
    """
    def __init__(self, c1=0.1, c2=0.1, max_iterations=50):
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=False
        )

    @staticmethod
    def _sent2features(words):
        return [extract_word_features(words, i) for i in range(len(words))]

    def fit(self, sents):
        print("  [CRF] Extracting features for all sentences...")
        X_train = [self._sent2features(w) for w, _ in sents]
        y_train = [t for _, t in sents]
        print(f"  [CRF] Training CRF on {len(X_train)} sentences with L-BFGS...")
        self.crf.fit(X_train, y_train)
        return self

    def predict_sents(self, sents):
        X = [self._sent2features(w) for w, _ in sents]
        return self.crf.predict(X)


# ---------------------------------------------------------
# 7. Model Training & Evaluation Engine
# ---------------------------------------------------------
def evaluate_predictions(gold_seqs, pred_seqs, target_labels):
    gold = [t for seq in gold_seqs for t in seq]
    pred = [t for seq in pred_seqs for t in seq]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=target_labels, zero_division=0)
    per_tag = pd.DataFrame({
        "tag": target_labels,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s
    })
    summary = {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": float(np.mean(f)),
        "weighted_f1": float(np.average(f, weights=s)),
    }
    return per_tag, summary


# Dictionary of splits
splits = {
    "dev": dev_sents,
    "test": test_sents,
}

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Model 1: Naive Bayes
print("\n>>> 1. Training Naive Bayes...")
t0 = time.time()
nb_model = NaiveBayesTagger().fit(train_sents)
print(f"Naive Bayes trained in {time.time()-t0:.2f}s")

# Model 2: HMM
print("\n>>> 2. Training HMM (Bigram with Forward Viterbi)...")
t0 = time.time()
hmm_model = HMMTagger().fit(train_sents)
print(f"HMM trained in {time.time()-t0:.2f}s")

# Model 3: MEMM
print("\n>>> 3. Training MEMM (Maximum Entropy Markov Model)...")
t0 = time.time()
memm_model = MEMMTagger(C=1.0, max_iter=100).fit(train_sents)
print(f"MEMM trained in {time.time()-t0:.2f}s")

# Model 4: CRF
print("\n>>> 4. Training CRF (sklearn-crfsuite)...")
t0 = time.time()
crf_model = CRFTagger(c1=0.1, c2=0.1, max_iterations=50).fit(train_sents)
print(f"CRF trained in {time.time()-t0:.2f}s")


# ---------------------------------------------------------
# 8. Evaluation Across Splits
# ---------------------------------------------------------
print("\n" + "="*60)
print("EVALUATION & COMPARATIVE STUDY")
print("="*60)

model_runners = {
    "Naive Bayes": lambda sents: [[nb_model.predict_sentence(w) for w, _ in [s]][0] for s in sents],
    "HMM": lambda sents: [[hmm_model.forward_viterbi(w) for w, _ in [s]][0] for s in sents],
    "MEMM": lambda sents: memm_model.predict_sents(sents),
    "CRF": lambda sents: crf_model.predict_sents(sents),
}

per_tag_records = []
summary_records = []

for split_name, sents in splits.items():
    gold_seqs = [tags for _, tags in sents]
    print(f"\n--- Evaluating on {split_name.upper()} Set ({len(sents)} sents) ---")
    for model_name, runner in model_runners.items():
        t0 = time.time()
        pred_seqs = runner(sents)
        dur = time.time() - t0
        pt, summ = evaluate_predictions(gold_seqs, pred_seqs, labels)
        
        pt.insert(0, "split", split_name)
        pt.insert(0, "model", model_name)
        per_tag_records.append(pt)
        
        summary_records.append({
            "model": model_name,
            "split": split_name,
            "accuracy": summ["accuracy"],
            "macro_f1": summ["macro_f1"],
            "weighted_f1": summ["weighted_f1"],
            "eval_time_sec": round(dur, 2)
        })
        print(f"  {split_name:5s} | {model_name:15s} | Acc={summ['accuracy']:.4f} | Macro F1={summ['macro_f1']:.4f} | W-F1={summ['weighted_f1']:.4f} (Time: {dur:.2f}s)")

per_tag_df = pd.concat(per_tag_records, ignore_index=True)
summary_df = pd.DataFrame(summary_records)

# Save results
per_tag_df.to_csv(os.path.join(OUT_DIR, "per_tag_scores.csv"), index=False)
summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
print(f"\nSaved CSV summaries to {OUT_DIR}")


# ---------------------------------------------------------
# 9. Plotting Charts (Question 3)
# ---------------------------------------------------------
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

sns.set_theme(style="whitegrid")
models_list = ["Naive Bayes", "HMM", "MEMM", "CRF"]
palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# (a) Overall Performance Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, split_name in zip(axes, ["dev", "test"]):
    sub = summary_df[summary_df["split"] == split_name].set_index("model")
    sub[["accuracy", "macro_f1", "weighted_f1"]].plot(
        kind="bar",
        ax=ax,
        colormap="viridis",
        rot=0,
        width=0.75
    )
    ax.set_title(f"Overall Model Comparison ({split_name.capitalize()} Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords="offset points")

plt.tight_layout()
overall_chart_path = os.path.join(CHARTS_DIR, "overall_comparison.png")
fig.savefig(overall_chart_path, dpi=200)
plt.close(fig)
print(f"Saved: {overall_chart_path}")

# (b) Per-tag F1 Bar Charts for Dev and Test (ordered by frequency)
for split_name in ["dev", "test"]:
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(len(labels))
    w = 0.8 / len(models_list)
    
    for k, m in enumerate(models_list):
        vals = [
            per_tag_df[
                (per_tag_df["model"] == m) &
                (per_tag_df["tag"] == t) &
                (per_tag_df["split"] == split_name)
            ]["f1"].values[0]
            for t in labels
        ]
        ax.bar(
            x + (k - (len(models_list) - 1) / 2) * w,
            vals,
            w,
            label=m,
            color=palette[k]
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, fontweight="medium")
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Per-Tag F1-Score Comparison on {split_name.capitalize()} Set\n(Tags ordered from left to right by training frequency)",
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(ncol=len(models_list), fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, f"f1_per_tag_{split_name}.png")
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {chart_path}")


# ---------------------------------------------------------
# 10. CRF Learned Feature Interpretation
# ---------------------------------------------------------
print("\n" + "="*60)
print("INTERPRETING LEARNED CRF FEATURES")
print("="*60)

print("\nTop 10 Most Likely Label Transitions:")
for (from_tag, to_tag), weight in Counter(crf_model.crf.transition_features_).most_common(10):
    print(f"  {from_tag:8s} -> {to_tag:8s} : {weight:+.4f}")

print("\nTop 10 Most Informative State (Observation) Features:")
for (feat, tag), weight in Counter(crf_model.crf.state_features_).most_common(10):
    try:
        print(f"  {feat:30s} -> {tag:8s} : {weight:+.4f}")
    except UnicodeEncodeError:
        print(f"  {feat.encode('ascii', 'backslashreplace').decode('ascii'):30s} -> {tag:8s} : {weight:+.4f}")

print("\nAssignment 9 execution finished successfully!")

