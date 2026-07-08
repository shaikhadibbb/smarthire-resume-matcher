import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import TFIDF_VECTORIZER_PATH

def fit_tfidf(texts, max_features=5000):
    """
    Fits a TF-IDF vectorizer on the given texts.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vectorizer.fit(texts)
    return vectorizer

def save_vectorizer(vectorizer, path=TFIDF_VECTORIZER_PATH):
    """
    Saves the TF-IDF vectorizer to disk using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved successfully to {path}")

def load_vectorizer(path=TFIDF_VECTORIZER_PATH):
    """
    Loads the TF-IDF vectorizer from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}")
    return joblib.load(path)

# Simple check block
if __name__ == "__main__":
    sample_texts = ["python developer with machine learning experience", "frontend developer with react skills"]
    vec = fit_tfidf(sample_texts)
    print("Vocabulary size:", len(vec.vocabulary_))
