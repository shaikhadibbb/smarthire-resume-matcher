"""Evaluation suite for the SmartHire machine learning pipeline.

Computes accuracy, precision, recall, and F1 metrics for classifiers, 
recommenders, and clustering models.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, silhouette_score
)
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import PROCESSED_DATA_DIR, CLASSIFIER_PATH, TFIDF_VECTORIZER_PATH, KMEANS_MODEL_PATH
from src.models.recommender import JobRecommender

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_classifier() -> None:
    """Evaluates the resume domain classifier model on the processed dataset."""
    logger.info("================ EVALUATING RESUME CLASSIFIER ================")
    resume_path = PROCESSED_DATA_DIR / "processed_resumes.csv"
    if not resume_path.exists() or not os.path.exists(CLASSIFIER_PATH):
        logger.warning("Classifier or resumes missing. Skipping classification evaluation.")
        return
        
    df = pd.read_csv(resume_path).dropna(subset=["clean_resume", "Category"])
    classifier = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    
    X = vectorizer.transform(df["clean_resume"])
    y = df["Category"]
    
    preds = classifier.predict(X)
    
    logger.info("Accuracy: %.4f", accuracy_score(y, preds))
    cm = confusion_matrix(y, preds)
    logger.info("Confusion Matrix Shape: %s", cm.shape)
    logger.info("Classification Report:\n%s", classification_report(y, preds, zero_division=0))

def evaluate_recommender(k: int = 10) -> None:
    """Evaluates the recommendation engine on 20 random resumes.

    Args:
        k: Precision@K threshold.
    """
    logger.info("================ EVALUATING RECOMMENDATION ENGINE ================")
    resume_path = PROCESSED_DATA_DIR / "processed_resumes.csv"
    if not resume_path.exists():
        logger.warning("Processed resumes missing. Skipping recommendation evaluation.")
        return
        
    df_resumes = pd.read_csv(resume_path).dropna(subset=["clean_resume"])
    
    # Load recommender
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    recommender = JobRecommender(vectorizer=vectorizer)
    recommender.load_jobs_and_vectorize()
    
    # Run evaluation on a random sample of 20 resumes
    sample_df = df_resumes.sample(n=20, random_state=42)
    precisions: List[float] = []
    
    for _, row in sample_df.iterrows():
        resume_text = str(row["clean_resume"])
        resume_category = str(row["Category"])
        
        # Get top-K recommendations
        recs = recommender.get_recommendations(resume_text, predicted_category=resume_category, top_n=k)
        
        if recs.empty:
            precisions.append(0.0)
            continue
            
        keyword = resume_category.split()[0].lower()
        
        matches = 0
        for _, rec_row in recs.iterrows():
            job_text = str(rec_row.get("title", "")).lower() + " " + str(rec_row.get("description", "")).lower()
            if keyword in job_text:
                matches += 1
                
        precisions.append(matches / k)
        
    mean_precision = np.mean(precisions)
    logger.info("Average Precision@%d (using category keyword matching): %.4f", k, mean_precision)

def evaluate_clustering() -> None:
    """Evaluates job clustering silhouette and inertia scores."""
    logger.info("================ EVALUATING CLUSTERING ================")
    if not os.path.exists(KMEANS_MODEL_PATH):
        logger.warning("Clustering model missing. Skipping clustering evaluation.")
        return
        
    data = joblib.load(KMEANS_MODEL_PATH)
    kmeans = data["kmeans"]
    vectorizer = data["vectorizer"]
    
    jobs_path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    df_jobs = pd.read_csv(jobs_path)
    combined_text = (
        df_jobs["clean_title"].fillna("") + " " +
        df_jobs["clean_skills"].fillna("") + " " +
        df_jobs["clean_description"].fillna("")
    )
    
    X = vectorizer.transform(combined_text)
    
    # Subsample for silhouette score to keep computation fast
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], min(2000, X.shape[0]), replace=False)
    X_sample = X[sample_indices]
    labels_sample = kmeans.labels_[sample_indices]
    
    score = silhouette_score(X_sample, labels_sample)
    logger.info("Silhouette Score (2000 samples): %.4f", score)
    logger.info("Inertia (Elbow Score): %.4f", kmeans.inertia_)

if __name__ == "__main__":
    evaluate_classifier()
    evaluate_recommender()
    evaluate_clustering()
