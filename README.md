```markdown
# 📩 SMS & Email Spam Classifier

An end-to-end Natural Language Processing (NLP) binary classification pipeline built with **Scikit-Learn**, **NLTK** . Designed specifically to prioritize **zero false positives** in spam detection.

---

## 📌 The Problem: Why 99% Accuracy Is a Trap

Standard evaluation metrics can be misleading when classes are imbalanced and the cost of classification errors is asymmetrical:

- **Class Imbalance:** In typical communication datasets, the distribution is skewed (~87% Ham vs. ~13% Spam). A dummy model predicting "Ham" for every input automatically yields an 87% accuracy rate despite offering zero predictive value.
- **Asymmetric Error Penalties:**
  - **False Negative (Spam in Inbox):** A minor nuisance.
  - **False Positive (Legitimate Mail in Spam):** A critical failure (e.g., missed job offers, urgent bills, or two-factor authentication tokens).
- **Core Optimization Target:** **Precision on the Spam class** ($1.00$ or $100\%$) takes absolute precedence over raw accuracy.

---

## ⚙️ Architecture & NLP Pipeline


```

Raw Message
│
▼
Text Normalization (Lowercasing, Tokenization, Alphanumeric Filtering)
│
▼
Noise Removal (Punctuation Stripping & NLTK Stopwords)
│
▼
Morphological Stemming (PorterStemmer: 'winning', 'wins' -> 'win')
│
▼
Vector Space Representation (TF-IDF with Top 3,000 Features)
│
▼
Classifier Inference (Multinomial Naive Bayes)
│
▼
Decision Output (Ham / Spam with Probability Score)

```

---

## 📊 Model Benchmarking

Over 11 algorithms were trained and evaluated on the same stratified test split:

| Algorithm / Family | Focus Area | Performance Note |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes (MNB)** | Frequency-based discrete features | **Top Performer: 100% Precision (0 False Positives)** |
| **Support Vector Classifier (SVC)** | High-dimensional margins (Sigmoid) | High accuracy, slightly slower inference |
| **Tree Ensembles (RF, ExtraTrees)** | Variance reduction & bagging | Stable, but computationally heavier |
| **Boosting (AdaBoost, XGBoost, GBDT)** | Sequential error correction | Strong recall, higher risk of false positives |
| **Ensembles (Voting, Stacking)** | Multi-model probabilistic consensus | Competitive, but MNB remained more optimal |

> **Key Takeaway:** Complex neural or ensemble architectures do not inherently beat classical probabilistic models on sparse, high-dimensional text matrices. Multinomial Naive Bayes produced the best precision-to-speed ratio.

---

```

---

## 🚀 Quickstart

### 1. Clone & Set Up Environment

```bash
git clone [https://github.com/NajeebAhmed69/SMS-Spam-Classifier-Model-Project.git](https://github.com/NajeebAhmed69/SMS-Spam-Classifier-Model-Project.git)
cd email-spam-classifier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


```


## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing:** NLTK (`punkt`, `stopwords`, `PorterStemmer`)
* **Machine Learning:** Scikit-Learn (TF-IDF, Naive Bayes, Ensembles), XGBoost
* **Serialization:** Pickle
* **Deployment:** Streamlit

```

```