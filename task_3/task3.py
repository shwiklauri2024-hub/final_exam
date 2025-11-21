import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import re
import string

# --- Configuration ---
MODEL_PATH = 'spam_model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'

# --- 1. Data Preparation and Cleaning ---

def clean_text(text):
    """
    Performs basic text cleaning: removing punctuation and converting to lowercase.
    For a real-world application, this should include stop word removal and stemming/lemmatization.
    """
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text

def load_data():
    """
    Loads data. For a complete run, load 'spam.csv'.
    A small hardcoded DataFrame is used here for immediate testing.
    """
    try:
        # Attempt to load the full dataset if available
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    except FileNotFoundError:
        # Fallback to a small, hardcoded dataset for demonstration
        print("Warning: 'spam.csv' not found. Using a small internal dataset for demonstration.")
        data = {
            'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
            'message': [
                'Hey, how are you doing today?',
                'WINNER! Claim your FREE prize now by clicking this link.',
                'Just confirming the meeting at 2 PM tomorrow.',
                'URGENT! Your account will be suspended if you do not reply.',
                'The project report is attached for your review.',
                'Free trial offer expires tonight! Act fast to save!'
            ]
        }
        df = pd.DataFrame(data)

    df['message'] = df['message'].apply(clean_text)
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# --- 2. Training and Evaluation ---

def train_and_evaluate():
    """
    Main function to load data, train the model, and evaluate performance.
    """
    df = load_data()

    X = df['message']
    y = df['label_encoded']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Feature Engineering (Count Vectorizer / Bag of Words)
    # The vectorizer learns the vocabulary from the training data
    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    
    # 4. Model Training (Multinomial Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_vectors, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test_vectors)

    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    # Save the trained model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    return model, vectorizer

# --- 6. Prediction Function ---

def predict_email(message, model, vectorizer):
    """
    Uses the trained model to classify a new, single email message.
    """
    # Clean the input message
    cleaned_message = clean_text(message)
    # Vectorize the cleaned message (using the SAVED vectorizer)
    message_vector = vectorizer.transform([cleaned_message])
    
    # Predict
    prediction = model.predict(message_vector)[0]
    
    if prediction == 1:
        return "Spam (1)"
    else:
        return "Ham (0)"

if __name__ == '__main__':
    # Train and save the necessary components
    trained_model, trained_vectorizer = train_and_evaluate()

    # Demonstration of the prediction function
    print("\n--- Model Demonstration ---")
    
    test_spam = "Congratulations! You have won a million dollars, click link now!"
    test_ham = "Please confirm the delivery address for your package tomorrow."

    print(f"Test Email 1 (Spam): '{test_spam}' -> Classification: {predict_email(test_spam, trained_model, trained_vectorizer)}")
    print(f"Test Email 2 (Ham): '{test_ham}' -> Classification: {predict_email(test_ham, trained_model, trained_vectorizer)}")

    # Example of loading the saved model for later use
    # loaded_model = joblib.load(MODEL_PATH)
    # loaded_vectorizer = joblib.load(VECTORIZER_PATH)
    # print(f"\nLoaded model predicts for a new message: {predict_email('You won a free iPhone!', loaded_model, loaded_vectorizer)}")
