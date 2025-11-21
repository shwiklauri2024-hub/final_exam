📧 Spam/Ham Email Classifier

A high-accuracy Machine Learning model built using Python and scikit-learn to automatically distinguish between unwanted Spam emails and legitimate Ham (non-spam) emails.

This project uses the classic Multinomial Naive Bayes algorithm, which is highly effective for text classification tasks.

🎯 Project Goal

The primary objective is to create a robust spam detection system that can automate the process of filtering an inbox, thereby enhancing user productivity and security.

✨ Key Features

Multinomial Naive Bayes: A fast and effective probabilistic classifier optimized for text data.

Bag-of-Words Vectorization: Converts human-readable text into numerical features the model can process.

Persistence: The trained model and feature extractor are saved for immediate use without retraining.

🚀 Getting Started

Follow these steps to set up and run the spam classification model on your local machine.

Prerequisites

This project requires Python 3.x and the following libraries:

pip install pandas scikit-learn joblib


Installation and Execution

Obtain the Code:
Download the spam_classifier.py file to your project directory.

Dataset (Optional but Recommended):
For full training, obtain a labeled dataset (e.g., the SMS Spam Collection dataset) and save it as a CSV file named spam.csv in the root directory. Note: If spam.csv is not present, the script will use a small internal dataset for demonstration.

Run the Training Script:
Execute the Python file. It will clean the data, train the model, evaluate its performance, and save the model and vectorizer to disk.

python spam_classifier.py


⚙️ Model Architecture & Methodology

The classification pipeline involves three critical phases: Data Cleaning, Feature Engineering, and Training.

1. Data Cleaning & Preparation

Before training, the raw message text is cleaned to ensure consistency and reduce noise:

Lowercase Conversion: All text is converted to lowercase ('FREE GIFT' becomes 'free gift').

Punctuation Removal: Punctuation marks are stripped ('Click now!' becomes 'Click now').

Label Encoding: Text labels (ham, spam) are converted to numerical values (0, 1) for the machine learning model.

2. Feature Engineering (The Bag-of-Words Model)

Machine learning models require numerical input. The Count Vectorizer converts the collection of text messages into a matrix of token counts using the Bag-of-Words (BoW) model.

Process: The vectorizer scans all training emails to build a global list of unique words (the Vocabulary).

Output: Each email is then represented as a vector (a row in the matrix) where each column corresponds to a word in the vocabulary, and the cell value is the number of times that word appears in the email. Words like "free," "urgent," and "click" will have higher counts in spam messages, forming strong distinguishing features.

3. Classification Algorithm: Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It is particularly well-suited for discrete count data (like word counts from the BoW model).

The core assumption is that the presence of a particular word in an email is independent of the presence of other words, given the email's class (Spam or Ham). The model calculates the probability of an email belonging to the Spam class given the set of words it contains.

📊 Results and Evaluation

After training, the model's performance is tested on a portion of the data it has never seen (the test set). High performance in these metrics ensures a reliable spam filter.

Metric

Importance in Spam Filtering

Accuracy

Overall correctness (e.g., 98.5% of emails classified correctly).

Precision

Crucial: Measures how many of the emails predicted as Spam were actually Spam. High precision minimizes False Positives (marking legitimate emails as spam).

Recall

Measures how many of the actual Spam emails were correctly identified. High recall minimizes False Negatives (missing spam, which is undesirable).

F1-Score

The harmonic mean of Precision and Recall, providing a single score that balances both.

Confusion Matrix Visualization

The Confusion Matrix visually breaks down the model's predictions into the four possible outcomes:



Predicted Ham (0)

Predicted Spam (1)

Actual Ham (0)

True Negatives (Correctly identified Ham)

False Positives (Ham incorrectly flagged as Spam)

Actual Spam (1)

False Negatives (Spam missed by the filter)

True Positives (Correctly identified Spam)

⏭️ Next Steps and Enhancements

Advanced Vectorization: Integrate TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which weighs word counts by their importance across the entire corpus, often leading to better performance than simple count-based models.

Hyperparameter Tuning: Use GridSearchCV or RandomizedSearchCV to fine-tune the alpha parameter of the Multinomial Naive Bayes model for optimal results.

Deployment: Wrap the predict_email function in a lightweight web framework (like Flask or FastAPI) to create a web API for real-time classification.
