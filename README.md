# Movie Review Sentiment Classification

This project focuses on binary sentiment classification of movie reviews using classical machine learning models.  
We compare multiple models and analyze their strengths, weaknesses, and limitations.

---

## Table of Contents

- [Dataset](#dataset)
- [Methods](#methods)
- [Models Used](#models-used)
- [Results](#results)
- [Key Insights](#key-insights)
- [Error Analysis](#error-analysis)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [How to Run](#how-to-run)

---

## Dataset

We use the IMDb movie review dataset, consisting of:

- 25,000 training reviews  
- 25,000 test reviews  

Balanced classes:
- Positive (1): 12,500  
- Negative (0): 12,500  

### Key observations:

- The dataset is perfectly balanced  
- Review lengths vary widely (mean ≈ 234 words)  
- Some reviews exceed 1000 words  

---

## Methods

### Text Representation

- TF-IDF Vectorization  
- Max features: 20,000  
- N-grams: (1, 2)

---

## Models Used

### Baseline
- Dummy Classifier (majority class)

### Classical Models
- Naive Bayes (MultinomialNB)
- Logistic Regression
- Linear SVM

### Tree Model
- Decision Tree  
  - Used to demonstrate overfitting

### Ensemble Methods
- Voting Classifier (NB + LR)
- Stacking Classifier (meta-learner)

---

## Results

| Model | Accuracy |
|------|---------|
| Stacked Ensemble | 0.8985 |
| Logistic Regression | 0.8956 |
| Linear SVM | 0.8901 |
| Voting Ensemble | 0.8891 |
| Naive Bayes | 0.8652 |
| Decision Tree | 0.7050 |
| Dummy Baseline | 0.5000 |

---

## Key Insights

### Linear models perform best
- Logistic Regression and SVM achieve ~90% accuracy  
- TF-IDF + linear models are highly effective for text classification  

### Decision Tree overfits
- Train accuracy: ~100%  
- Test accuracy: ~70%  
- Tree models struggle with sparse high-dimensional data  

### Ensembles slightly improve performance
- Stacking outperforms individual models  
- Learns optimal combination of predictions  

---

## Error Analysis

From misclassified examples:

### 1. Lack of context understanding
- Models rely on word frequency rather than meaning  

Example:
> “fail”, “miserably” → predicted negative  
Even if overall review is positive  

### 2. Long review bias
- Models sum word weights  
- Early positive words can dominate even in negative reviews  

---

## Limitations

- No understanding of context or semantics  
- Ignores word order beyond n-grams  
- Cannot capture sarcasm or nuanced sentiment  

---

## Future Work

- Word embeddings (Word2Vec, GloVe)  
- Deep learning models (LSTM, RNN)  
- Transformer-based models (BERT)  
- Context-aware representations  

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Make sure dataset is structured as:

```
aclImdb/
  train/
    pos/
    neg/
  test/
    pos/
    neg/
```
