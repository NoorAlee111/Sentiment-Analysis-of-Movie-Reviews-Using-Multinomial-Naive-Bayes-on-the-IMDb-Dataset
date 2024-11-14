# - nltk: Provides tools for text processing, including tokenization and stopword removal.
# - sklearn: Used for machine learning algorithms and model evaluation.
# - pandas: Used to load, manipulate, and display the dataset.
# - numpy: Used for efficient numerical operations.

import os
import random
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Path to IMDb dataset
dataset_path = 'aclImdb'

# Function to load dataset with Pandas
def load_data(data_dir):
    # Create empty lists to hold reviews and their labels
    reviews = []
    labels = []

    # Load positive reviews
    pos_dir = os.path.join(data_dir, 'pos')
    for file_name in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, file_name), 'r', encoding='utf-8') as file:
            reviews.append(file.read())
            labels.append(1)  # 1 indicates a positive review

    # Load negative reviews
    neg_dir = os.path.join(data_dir, 'neg')
    for file_name in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, file_name), 'r', encoding='utf-8') as file:
            reviews.append(file.read())
            labels.append(0)  # 0 indicates a negative review

    # Create a Pandas DataFrame for easy handling
    data = pd.DataFrame({
        'review': reviews,
        'label': labels
    })

    return data

# Load dataset using Pandas
data = load_data(os.path.join(dataset_path, 'train'))

# Display first few rows of the dataset to verify
print(data.head())

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], test_size=0.2, random_state=42)

# Text Preprocessing
# Explanation:
# - Stopwords are common words (like "the", "is") that don't contribute to sentiment and are removed.
# - CountVectorizer: Converts text into a bag-of-words format for model training.

# Initialize CountVectorizer with stop words
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

# Fit and transform training data, transform test data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Training
# Explanation:
# - MultinomialNB: A Naive Bayes classifier commonly used for text classification.
# - It calculates the probability of each class and chooses the highest probability.

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Model Evaluation
# Explanation:
# - We predict sentiments of the test set and calculate accuracy.

# Predict and calculate accuracy
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict sentiment of a custom review
def predict_sentiment(review):
    review_vectorized = vectorizer.transform([review])  # Vectorize user input review
    prediction = model.predict(review_vectorized)  # Predict using trained model
    return "Positive" if prediction[0] == 1 else "Negative"

# Take user input for a new review and predict its sentiment
user_review = input("Enter a movie review: ")
print("Sentiment:", predict_sentiment(user_review))
