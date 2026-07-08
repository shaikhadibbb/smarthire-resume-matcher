"""Resume classification model training.

Fits a TF-IDF vectorizer and trains a Logistic Regression classifier to predict
resume categories. Annotates the job corpus with domain predictions, corrected
by title heuristics.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    PROCESSED_DATA_DIR, CLASSIFIER_PATH, TFIDF_VECTORIZER_PATH, RANDOM_STATE
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def heuristic_job_category(title: str, description: str, model_pred: str) -> str:
    """Corrects job classification domain using job title keywords to bridge domain gap.

    Args:
        title: The job title.
        description: The job description.
        model_pred: The initial classifier model prediction.

    Returns:
        The finalized category string.
    """
    import re
    title_lower = title.lower()
    
    # 1. Non-technical/non-corporate filter overrides (Map to "Other")
    non_tech_pattern = r"\b(substitute|program aide|aide|teacher|tutor|instructor|teaching|retail sales|cashier|clerk|merchandiser|store associate|stocker|nurse|nursing|caregiver|physician|medical assistant|dental|administrative assistant|receptionist|secretary|call center|learning and development|learning manager|training coordinator|curriculum)\b"
    if re.search(non_tech_pattern, title_lower):
        return "Other"
        
    # 2. Specific technical overrides
    if re.search(r"\b(data scientist|data science|machine learning|ml engineer|deep learning|nlp|computer vision|artificial intelligence|ai)\b", title_lower):
        return "Data Science"
    if re.search(r"\b(data analyst|business analyst|analytics|product analyst|reporting analyst)\b", title_lower):
        return "Business Analyst"
    if re.search(r"\b(frontend|front-end|web developer|web designer|react|angular|vue|ui/ux|wordpress|html|javascript|js)\b", title_lower):
        return "Web Designing"
    if re.search(r"\b(devops|cloud engineer|aws|sre|system administrator|sysadmin|linux administrator)\b", title_lower):
        return "DevOps Engineer"
    if re.search(r"\b(java developer|java engineer|spring boot|java)\b", title_lower):
        return "Java Developer"
    if re.search(r"\b(python developer|python engineer)\b", title_lower):
        return "Python Developer"
    if re.search(r"\b(dotnet|\.net)\b", title_lower) or re.search(r"\bc#(?:\b|[^\w]|$)", title_lower):
        return "DotNet Developer"
    if re.search(r"\b(database|dba|sql developer|oracle)\b", title_lower):
        return "Database"
    if re.search(r"\b(qa|testing|automation testing|test engineer|quality assurance)\b", title_lower):
        return "Automation Testing"
    if re.search(r"\b(hr manager|recruiter|human resources|talent acquisition|onboarding|hr)\b", title_lower):
        return "HR"
    if re.search(r"\b(sales|business development|marketing|account manager|salesforce)\b", title_lower):
        return "Sales"
    if re.search(r"\b(blockchain|solidity|ethereum)\b", title_lower):
        return "Blockchain"
    if re.search(r"\b(hadoop|spark|big data|etl developer|data engineer|data warehouse)\b", title_lower):
        return "ETL Developer"
    if re.search(r"\b(civil)\b", title_lower):
        return "Civil Engineer"
    if re.search(r"\b(mechanical)\b", title_lower):
        return "Mechanical Engineer"
    if re.search(r"\b(electrical)\b", title_lower):
        return "Electrical Engineering"
        
    return model_pred

def train_classifier() -> Tuple[Any, TfidfVectorizer]:
    """Trains the resume domain classifier and tags the job database.

    Fits a denser TF-IDF (1-3 ngrams, max 10000 features) and class-balanced model.
    Saves models and tags the job corpus with title-heuristics-corrected predictions.

    Returns:
        A tuple of (trained_classifier, fitted_vectorizer).
    """
    processed_resume_path: Path = PROCESSED_DATA_DIR / "processed_resumes.csv"
    
    if not processed_resume_path.exists():
        raise FileNotFoundError(f"Processed resumes not found at {processed_resume_path}. Please run preprocess.py first.")
        
    logger.info("Loading processed resumes...")
    df = pd.read_csv(processed_resume_path)
    df = df.dropna(subset=["clean_resume", "Category"])
    
    X = df["clean_resume"]
    y = df["Category"]
    
    logger.info("Dataset contains %d records across %d categories.", len(X), y.nunique())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Fit TF-IDF Vectorizer with ngram_range=(1,2), max_features=10000, min_df=2, max_df=0.8, sublinear_tf=True
    logger.info("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.8,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save vectorizer
    MODEL_SAVE_DIR = Path(CLASSIFIER_PATH).parent
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
    logger.info("Vectorizer saved successfully to %s", TFIDF_VECTORIZER_PATH)
    
    # Define candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=RANDOM_STATE),
        "LinearSVC": LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, dual=False),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=20, random_state=RANDOM_STATE),
        "SGDClassifier": SGDClassifier(loss='hinge', class_weight='balanced', alpha=0.0001, random_state=RANDOM_STATE),
        "MultinomialNB": MultinomialNB()
    }
    
    logger.info("Running Stratified 5-Fold Cross Validation to pick the best model...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    best_name = ""
    best_score = -1.0
    
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring='accuracy')
        mean_score = np.mean(scores)
        logger.info("CV Accuracy for %s: %.4f", name, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            
    logger.info("Selected Best Model: %s with CV Accuracy: %.4f", best_name, best_score)
    
    # Train the best model on full training set
    best_model = candidates[best_name]
    
    logger.info("Wrapping %s with CalibratedClassifierCV to enable calibrated probabilities...", best_name)
    best_model = CalibratedClassifierCV(estimator=best_model, cv=5)
        
    best_model.fit(X_train_vec, y_train)
    
    # Evaluate on test split
    predictions = best_model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)
    
    logger.info("Model Accuracy on test split: %.4f", acc)
    report = classification_report(y_test, predictions, zero_division=0)
    logger.info("Classification Report on test split:\n%s", report)
    
    # Save model
    joblib.dump(best_model, CLASSIFIER_PATH)
    logger.info("Classifier saved successfully to %s", CLASSIFIER_PATH)
    
    # Tag Job Listings Corpus with predicted categories
    jobs_path: Path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    if jobs_path.exists():
        logger.info("Annotating job database with corrected predicted categories...")
        df_jobs = pd.read_csv(jobs_path)
        job_texts = (
            df_jobs["clean_title"].fillna("") + " " +
            df_jobs["clean_skills"].fillna("") + " " +
            df_jobs["clean_description"].fillna("")
        )
        
        job_vecs = vectorizer.transform(job_texts)
        job_preds = best_model.predict(job_vecs)
        
        # Apply heuristic title correction
        final_preds = []
        for i, row in df_jobs.iterrows():
            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            final_cat = heuristic_job_category(title, desc, job_preds[i])
            final_preds.append(final_cat)
            
        df_jobs["predicted_category"] = final_preds
        df_jobs.to_csv(jobs_path, index=False)
        logger.info("Saved annotated job corpus to %s", jobs_path)
        
    return best_model, vectorizer

if __name__ == "__main__":
    train_classifier()
