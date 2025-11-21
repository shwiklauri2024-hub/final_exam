# 📧 Spam/Ham Email Classifier

A high-accuracy Machine Learning model built using Python and scikit-learn to automatically distinguish between unwanted **Spam** emails and legitimate **Ham** (non-spam) emails.

This project uses the classic **Multinomial Naive Bayes** algorithm, which is highly effective for text classification tasks.

## 🎯 Project Goal

The primary objective is to create a robust spam detection system that can automate the process of filtering an inbox, thereby enhancing user productivity and security.

## ✨ Key Features

- **Multinomial Naive Bayes:** A fast and effective probabilistic classifier optimized for text data.
- **Bag-of-Words Vectorization:** Converts human-readable text into numerical features the model can process.
- **Persistence:** The trained model and feature extractor are saved for immediate use without retraining.

## 🚀 Getting Started

### Prerequisites

This project requires Python 3.x and the following libraries:

```
pip install pandas scikit-learn joblib
```

### Installation and Execution

1. **Obtain the Code:**  
   Download the `spam_classifier.py` file to your project directory.

2. **Dataset (Optional but Recommended):**  
   Save a labeled dataset such as the SMS Spam Collection dataset as `spam.csv` in the root directory. If no dataset is found, a small demo dataset is used.

3. **Run the Training Script:**
```
python spam_classifier.py
```

## ⚙️ Model Architecture & Methodology

### Data Cleaning & Preparation

- Lowercase conversion  
- Punctuation removal  
- Label encoding  

### Feature Engineering (Bag-of-Words)

- Build vocabulary from corpus  
- Convert each email into a token-count vector  

### Classification: Multinomial Naive Bayes

A probabilistic classifier based on Bayes’ Theorem, well-suited for word-count data.

## 📊 Results and Evaluation

### Metrics

| Metric | Importance |
|--------|------------|
| Accuracy | Overall correctness |
| Precision | Avoids false positives |
| Recall | Avoids false negatives |
| F1-Score | Balances precision & recall |

### Confusion Matrix

|                | Predicted Ham (0) | Predicted Spam (1) |
|----------------|-------------------|--------------------|
| Actual Ham (0) | True Negatives    | False Positives     |
| Actual Spam (1)| False Negatives   | True Positives      |

## ⏭️ Next Steps

- Add TF-IDF  
- Hyperparameter tuning  
- Deploy with Flask/FastAPI  
