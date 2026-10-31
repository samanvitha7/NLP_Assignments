# NLP Lab Assignment 9 — Hindi POS Tagging
## Maximum Entropy Markov Models (MEMM) & Conditional Random Fields (CRF)
### Comparative Study: Naive Bayes vs. HMM vs. MEMM vs. CRF

---

## 1. Overview & Problem Statement

This project implements and evaluates four sequence labeling paradigms on a morphologically rich Hindi Part-of-Speech (POS) dataset (Bureau of Indian Standards / BIS tagset):
1. **Maximum Entropy Markov Model (MEMM)**: Implemented based on [arne-cl/nltk-maxent-pos-tagger](https://github.com/arne-cl/nltk-maxent-pos-tagger) and Ratnaparkhi (1996), conditioning on observation features and previous predicted tags ($y_{t-1}$) with high-performance L-BFGS optimization.
2. **Conditional Random Field (CRF)**: Implemented using `sklearn-crfsuite` following the [official tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html), incorporating the four required feature families:
   - **Prefixes up to length 4** ($1, 2, 3, 4$): Padded with `'NULL'` when word length $< k$.
   - **Suffixes up to length 4** ($1, 2, 3, 4$): Padded with `'NULL'` when word length $< k$.
   - **Context Features**: Previous Word ($w_{i-1}$), Current Word ($w_i$), Next Word ($w_{i+1}$).
   - **Binary Length Feature**: `'LESS'` if $\text{len}(w) \le 4$, else `'MORE'`.
3. **Comprehensive Comparative Study**:
   - Compares **Naive Bayes** and **HMM** (developed in Assignment 7) with **MEMM** and **CRF**.
   - Evaluates performance across 24 POS tags on both Development and Test sets.
   - Generates summary CSVs and high-resolution grouped bar charts.

---

## 2. Directory Structure

```text
Assignment9/
├── Hindi-POS-Data/
│   ├── hindi-train-pos.txt             # 15,087 sentences, 325,710 tokens
│   ├── hindi-dev-pos.txt               # 1,866 sentences,  40,762 tokens
│   └── hindi-test-pos.txt              # 1,910 sentences,  41,415 tokens
├── results/
│   ├── summary.csv                     # Overall Accuracy, Macro F1, Weighted F1
│   ├── per_tag_scores.csv              # Per-tag Precision, Recall, F1, Support
│   └── charts/
│       ├── overall_comparison.png      # Bar chart of overall metrics on dev & test
│       ├── f1_per_tag_dev.png          # Grouped bar chart of per-tag F1 on dev set
│       └── f1_per_tag_test.png         # Grouped bar chart of per-tag F1 on test set
├── Lab9_POS_Tagger.ipynb               # Fully executed Jupyter Notebook with all outputs
├── run_assignment9.py                  # Standalone end-to-end Python script
├── generate_lab9_notebook.py           # Automated notebook generator and runner
├── NLP-Lab-Assignment-9-2026.pdf       # Assignment problem sheet
└── README.md                           # Comprehensive documentation & results report
```

---

## 3. Theoretical Comparison: The Four Paradigms

| Model | Paradigm | Objective Function | Normalization | Label Bias Problem? | Overlapping Features? | Test Accuracy | Test Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Naive Bayes** | Generative | Joint $P(X, Y) = P(Y) \prod_j P(x_j \mid Y)$ | Global over joint space | No | No (Independence assumed) | 92.57% | 84.44% |
| **HMM (Viterbi)**| Generative | Joint $P(X, Y) = \prod_t P(y_t \mid y_{t-1}) P(w_t \mid y_t)$ | Local transition & emission | No | No (Strict emission independence) | 92.92% | 83.28% |
| **MEMM** | Discriminative | Conditional $P(Y \mid X) = \prod_t P(y_t \mid x_t, y_{t-1})$ | **Local** per state $Z(x_t, y_{t-1})$ | **Yes** (Prone to label bias) | **Yes** (Arbitrary features) | 96.15% | 92.34% |
| **CRF** | Discriminative | Conditional $P(Y \mid X) = \frac{1}{Z(X)} \exp(\sum_t \dots)$ | **Global** over sequence $Z(X)$ | **No** (Immune) | **Yes** (Arbitrary features) | **96.51%** | **92.90%** |

### Key Theoretical Insights:
1. **Generative vs. Discriminative Leap:**
   - Generative models (NB & HMM) model the joint distribution $P(X, Y)$, forcing strong independence assumptions on the observation $X$.
   - Discriminative models (MEMM & CRF) directly model $P(Y \mid X)$, enabling the model to exploit rich overlapping sub-word features (prefixes, suffixes, length, context words). This produces an immediate **+3.59% accuracy gain** and **+9.62% Macro F1 gain** on the test set.
2. **The Label Bias Problem (Why CRF > MEMM):**
   - In MEMM, local normalization at state $t$ requires:
     $$\sum_{y'} P(y' \mid x_t, y_{t-1}) = 1$$
     States with few outgoing transitions pass their probability mass regardless of the observation $x_t$. Downstream observations cannot retroactively correct an upstream error.
   - In CRF, the global partition function $Z(X)$ normalizes over all candidate sequence paths simultaneously:
     $$P(Y \mid X) = \frac{1}{Z(X)} \prod_{t=1}^T \exp\left( \sum_k \lambda_k f_k(y_t, X, t) + \sum_j \mu_j g_j(y_{t-1}, y_t, X, t) \right)$$
     This eliminates label bias and enables downstream words to vote on earlier POS tags.

---

## 4. Quantitative Results

### Overall Performance Table (`results/summary.csv`)

| Model | Split | Accuracy | Macro F1 | Weighted F1 | Evaluation Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | Dev | 92.47% | 83.31% | 92.49% | 0.23s |
| **HMM (Viterbi)** | Dev | 92.99% | 82.75% | 93.09% | 0.34s |
| **MEMM** | Dev | 96.00% | 95.94% | 95.98% | 0.39s |
| **CRF** | Dev | **96.33%** | **96.57%** | **96.32%** | 0.37s |
| **Naive Bayes** | Test | 92.57% | 84.44% | 92.59% | 0.24s |
| **HMM (Viterbi)** | Test | 92.92% | 83.28% | 93.03% | 0.36s |
| **MEMM** | Test | 96.15% | 92.34% | 96.13% | 0.40s |
| **CRF** | Test | **96.51%** | **92.90%** | **96.50%** | 0.31s |

---

## 5. Detailed Per-Tag Performance Breakdown (Test Split)

Below is the comparison of F1-scores across tags ordered by training frequency:

| Tag | Category / Description | Support | Naive Bayes | HMM | MEMM | CRF | Key Finding / Linguistic Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `N_NN` | Common Noun | 10,790 | 0.9012 | 0.9112 | 0.9380 | **0.9433** | Sub-word suffixes and surrounding postposition context help CRF resolve ambiguous nouns. |
| `PSP` | Postposition (*ने, को, से, में*) | 8,247 | 0.9824 | 0.9862 | 0.9956 | **0.9961** | Function words with $\le 4$ length and unique 1-2 char suffixes are near-perfectly tagged. |
| `N_NNP` | Proper Noun (Names, Places) | 4,500 | 0.8407 | 0.8399 | 0.8923 | **0.9049** | **+6.5% gain over HMM.** Prefix/suffix features generalize to unseen Named Entities. |
| `V_VM` | Main Verb (*जा, कर, देख*) | 4,213 | 0.9384 | 0.9432 | 0.9634 | **0.9664** | Verbal inflections (*-ता, -ती, -ते, -कर, -या*) are captured by length 2-3 suffixes. |
| `V_VAUX` | Auxiliary Verb (*रहा, है, था*) | 3,118 | 0.9348 | 0.9392 | 0.9490 | **0.9525** | Strong transition weight ($V\_VM \to V\_VAUX$) confirms auxiliary placement in Hindi verb phrase. |
| `RD_PUNC`| Punctuation | 2,752 | 0.9995 | 0.9996 | 0.9996 | **0.9996** | Deterministic symbol matching across all models. |
| `JJ` | Adjective | 2,367 | 0.8865 | 0.9050 | 0.9222 | **0.9281** | Adjective-Noun agreement ($JJ \to N\_NN$) transition boosts CRF precision. |
| `PR_PRP` | Personal Pronoun (*वह, मैं, हम*) | 1,326 | 0.9071 | 0.9627 | 0.9788 | **0.9835** | High-frequency closed class; syntactic position strongly guides CRF. |
| `QT_QTC` | Cardinal Number (*एक, दो, 10*) | 788 | 0.9202 | 0.8708 | 0.9755 | **0.9724** | Length and digit patterns strongly separate numbers from nouns. |
| `CC_CCD` | Coordinating Conjunction (*और, तथा*)| 738 | 0.9616 | 0.9717 | **0.9878** | 0.9852 | Strong surrounding noun/verb context anchors coordination. |
| `PR_PRF` | Reflexive Pronoun (*अपना, खुद*) | 151 | 0.9034 | 0.6803 | 0.9934 | **0.9967** | **+31.6% jump over HMM.** HMM suffers from emission sparsity; CRF uses suffix *-ना* and *-द*. |
| `PR_PRL` | Relative Pronoun (*जो, जिसे*) | 134 | 0.7248 | 0.6877 | 0.9556 | **0.9556** | **+26.8% jump over HMM.** Context words (*BOS / EOS*) anchor relative clause boundaries. |
| `PR_PRQ` | Interrogative Pronoun (*क्या, कौन*)| 27 | 0.8302 | 0.7654 | **1.0000** | **1.0000** | **100% precision and recall** in CRF/MEMM via prefix *क-* and interrogative syntax. |
| `QT_QTO` | Ordinal Quantifier (*पहला, दूसरा*)| 25 | 0.6667 | 0.7674 | 0.9459 | **0.9600** | Ordinal suffixes (*-रा, -था, -वां*) uniquely identified by length 2-3 suffixes. |

---

## 6. Learned CRF Weights Analysis (Interpretability)

Linear-chain CRFs provide clear, linguistically grounded interpretability:

### Top 10 Positive Label Transitions ($y_{t-1} \to y_t$):
1. `V_VM     -> V_VAUX   (+4.6620)`: In Hindi, main verbs are followed immediately by auxiliary verbs (*कर रहा है*).
2. `N_NNP    -> N_NNP    (+4.2450)`: Multi-word named entities frequently follow each other (*नरेंद्र मोदी*).
3. `V_VAUX   -> V_VAUX   (+3.1364)`: Auxiliary chains (*रहा है*, *सकता था*).
4. `JJ       -> N_NN     (+2.9303)`: Attributive adjectives precede nouns (*सुंदर घर*).
5. `N_NN     -> V_VM     (+2.6614)`: SOV basic word order (Object Noun preceding Verb).
6. `RP_INTF  -> QT_QTF   (+2.4205)`: Intensifiers preceding quantifiers (*बहुत कम*).
7. `QT_QTC   -> QT_QTC   (+2.3845)`: Multi-word numbers (*दो लाख*).
8. `QT_QTC   -> N_NN     (+2.3809)`: Quantifiers modifying count nouns (*पाँच लोग*).
9. `N_NN     -> PSP      (+2.1892)`: Nouns followed by postpositions (*घर में*, *राम ने*).
10. `DM_DMD   -> N_NN     (+2.1814)`: Deictic demonstratives modifying nouns (*यह किताब*).

### Top Negative (Penalized) Transitions:
1. `PSP      -> PSP      (-3.9402)`: Consecutive postpositions almost never occur in Hindi grammar.
2. `V_VAUX   -> PSP      (-3.1145)`: Postpositions follow nominals, not auxiliary verbs.
3. `RD_PUNC  -> V_VAUX   (-2.8912)`: Auxiliary verbs do not follow punctuation without a main verb.

---

## 7. How to Run

### Option A: Run the Fully Executed Jupyter Notebook
Open `Assignment9/Lab9_POS_Tagger.ipynb` in VS Code or Jupyter Notebook:
- All 26 cells are already fully executed with all outputs, DataFrames, and inline charts rendered.
- You can also re-run all cells via `Kernel -> Restart and Run All`.

### Option B: Run via Command Line
Execute the end-to-end Python script:
```bash
python Assignment9/run_assignment9.py
```
This trains all models, evaluates them on Dev and Test sets, writes `results/summary.csv`, `results/per_tag_scores.csv`, and saves the charts to `results/charts/`.
