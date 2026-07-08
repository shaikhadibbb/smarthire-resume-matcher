"""Fit/Shortlisting Predictor Model.

Generates a balanced synthetic dataset of 5,000 resume-job pairs including 
same-category negatives to force selectivity. Trains and evaluates Logistic 
Regression vs. XGBoost, serializing the best model.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import (
    PROCESSED_DATA_DIR, FIT_PREDICTOR_PATH, TFIDF_VECTORIZER_PATH, RANDOM_STATE
)
from src.features.text_features import load_vectorizer
from src.features.match_features import compute_match_features

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_fit_dataset(num_samples: int = 5000) -> pd.DataFrame:
    """Builds a balanced synthetic dataset of (resume, job) pairs.

    Includes category mismatches and same-category skill mismatches as negative classes.

    Args:
        num_samples: Total number of pairs to generate.

    Returns:
        A DataFrame containing match features and binary label.
    """
    logger.info("Building synthetic fit dataset of %d pairs...", num_samples)
    
    resume_path: Path = PROCESSED_DATA_DIR / "processed_resumes.csv"
    jobs_path: Path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    
    if not resume_path.exists() or not jobs_path.exists():
        raise FileNotFoundError("Processed resumes or jobs files are missing. Run preprocess.py and classifier.py first.")
        
    df_resumes = pd.read_csv(resume_path).dropna(subset=["clean_resume", "Category"])
    df_jobs = pd.read_csv(jobs_path)
    
    if "predicted_category" not in df_jobs.columns:
        raise ValueError("Job corpus must contain predicted_category columns. Run classifier.py first.")
        
    tfidf = load_vectorizer(TFIDF_VECTORIZER_PATH)
    
    # Pre-calculate job text vectors for speed
    job_desc_texts = df_jobs["clean_description"].fillna("")
    job_tfidf_vectors = tfidf.transform(job_desc_texts)
    
    data_rows: List[Dict[str, float]] = []
    
    target_pos = num_samples // 2
    target_neg_diff_cat = target_pos // 2
    target_neg_same_cat = target_pos // 2
    
    pos_count = 0
    neg_same_count = 0
    neg_diff_count = 0
    
    # Group resumes and jobs by category for fast access
    resumes_by_cat = {cat: grp for cat, grp in df_resumes.groupby("Category")}
    jobs_by_cat = {cat: grp for cat, grp in df_jobs.groupby("predicted_category")}
    
    all_categories = list(resumes_by_cat.keys())
    
    # 1. Generate Positive Matches & Same-Category Negatives
    logger.info("Generating same-category matches...")
    np.random.seed(RANDOM_STATE)
    
    # Loop over categories to ensure representation from all domains
    for cat in all_categories:
        if cat not in jobs_by_cat or jobs_by_cat[cat].empty:
            continue
            
        cat_resumes = resumes_by_cat[cat]
        cat_jobs = jobs_by_cat[cat]
        
        # Try a few pairs per category
        pairs_to_try = 150
        for _ in range(min(pairs_to_try, len(cat_resumes) * len(cat_jobs))):
            res_idx = np.random.randint(0, len(cat_resumes))
            job_sub_idx = np.random.randint(0, len(cat_jobs))
            
            resume_row = cat_resumes.iloc[res_idx]
            job_row = cat_jobs.iloc[job_sub_idx]
            job_idx = cat_jobs.index[job_sub_idx]
            
            resume_skills_str = resume_row.get("extracted_skills", "")
            resume_skills = [s.strip().lower() for s in resume_skills_str.split(",") if s.strip()] if isinstance(resume_skills_str, str) else None
            features = compute_match_features(
                resume_text=resume_row["clean_resume"],
                job_row=job_row,
                tfidf_vectorizer=tfidf,
                job_tfidf_vector=job_tfidf_vectors[job_idx],
                resume_skills=resume_skills
            )
            features["category_match"] = 1.0
            
            # Check if positive match
            if (features["skill_overlap_ratio"] > 0.20 or features["description_similarity"] > 0.15) and pos_count < target_pos:
                features["label"] = 1.0
                data_rows.append(features)
                pos_count += 1
            # Check if same-category negative match (low skill overlap)
            elif features["skill_overlap_ratio"] < 0.10 and features["description_similarity"] < 0.12 and neg_same_count < target_neg_same_cat:
                features["label"] = 0.0
                data_rows.append(features)
                neg_same_count += 1
                
            if pos_count >= target_pos and neg_same_count >= target_neg_same_cat:
                break
                
    # Relaxed thresholds to fill remaining positive slots if needed
    if pos_count < target_pos:
        logger.info("Filling remaining positive matches with relaxed thresholds...")
        for cat in all_categories:
            if pos_count >= target_pos:
                break
            if cat not in jobs_by_cat or jobs_by_cat[cat].empty:
                continue
            cat_resumes = resumes_by_cat[cat]
            cat_jobs = jobs_by_cat[cat]
            
            for _ in range(min(150, len(cat_resumes) * len(cat_jobs))):
                if pos_count >= target_pos:
                    break
                res_idx = np.random.randint(0, len(cat_resumes))
                job_sub_idx = np.random.randint(0, len(cat_jobs))
                resume_row = cat_resumes.iloc[res_idx]
                job_row = cat_jobs.iloc[job_sub_idx]
                job_idx = cat_jobs.index[job_sub_idx]
                
                resume_skills_str = resume_row.get("extracted_skills", "")
                resume_skills = [s.strip().lower() for s in resume_skills_str.split(",") if s.strip()] if isinstance(resume_skills_str, str) else None
                features = compute_match_features(
                    resume_text=resume_row["clean_resume"],
                    job_row=job_row,
                    tfidf_vectorizer=tfidf,
                    job_tfidf_vector=job_tfidf_vectors[job_idx],
                    resume_skills=resume_skills
                )
                features["category_match"] = 1.0
                features["label"] = 1.0
                data_rows.append(features)
                pos_count += 1
                
    # Same for same-category negatives
    if neg_same_count < target_neg_same_cat:
        logger.info("Filling remaining same-category negatives...")
        for cat in all_categories:
            if neg_same_count >= target_neg_same_cat:
                break
            if cat not in jobs_by_cat or jobs_by_cat[cat].empty:
                continue
            cat_resumes = resumes_by_cat[cat]
            cat_jobs = jobs_by_cat[cat]
            
            for _ in range(min(150, len(cat_resumes) * len(cat_jobs))):
                if neg_same_count >= target_neg_same_cat:
                    break
                res_idx = np.random.randint(0, len(cat_resumes))
                job_sub_idx = np.random.randint(0, len(cat_jobs))
                resume_row = cat_resumes.iloc[res_idx]
                job_row = cat_jobs.iloc[job_sub_idx]
                job_idx = cat_jobs.index[job_sub_idx]
                
                resume_skills_str = resume_row.get("extracted_skills", "")
                resume_skills = [s.strip().lower() for s in resume_skills_str.split(",") if s.strip()] if isinstance(resume_skills_str, str) else None
                features = compute_match_features(
                    resume_text=resume_row["clean_resume"],
                    job_row=job_row,
                    tfidf_vectorizer=tfidf,
                    job_tfidf_vector=job_tfidf_vectors[job_idx],
                    resume_skills=resume_skills
                )
                features["category_match"] = 1.0
                features["label"] = 0.0
                data_rows.append(features)
                neg_same_count += 1
                
    # 2. Generate Different-Category Negatives
    logger.info("Generating different-category negatives...")
    while neg_diff_count < target_neg_diff_cat:
        res_idx = np.random.randint(0, len(df_resumes))
        resume_row = df_resumes.iloc[res_idx]
        res_cat = resume_row["Category"]
        
        diff_cats = [c for c in all_categories if c != res_cat and c in jobs_by_cat]
        if not diff_cats:
            continue
        dest_cat = np.random.choice(diff_cats)
        dest_jobs = jobs_by_cat[dest_cat]
        
        job_sub_idx = np.random.randint(0, len(dest_jobs))
        job_row = dest_jobs.iloc[job_sub_idx]
        job_idx = dest_jobs.index[job_sub_idx]
        
        resume_skills_str = resume_row.get("extracted_skills", "")
        resume_skills = [s.strip().lower() for s in resume_skills_str.split(",") if s.strip()] if isinstance(resume_skills_str, str) else None
        features = compute_match_features(
            resume_text=resume_row["clean_resume"],
            job_row=job_row,
            tfidf_vectorizer=tfidf,
            job_tfidf_vector=job_tfidf_vectors[job_idx],
            resume_skills=resume_skills
        )
        features["category_match"] = 0.0
        features["label"] = 0.0
        data_rows.append(features)
        neg_diff_count += 1
        
    df_fit = pd.DataFrame(data_rows)
    logger.info("Generated fit dataset. Size: %d rows. Label distribution:\n%s", len(df_fit), df_fit["label"].value_counts(normalize=True))
    return df_fit

def train_fit_predictor() -> Any:
    """Trains Logistic Regression and XGBoost on fit features.

    Saves the model with the higher test F1-score to models/fit_predictor.pkl.

    Returns:
        The best trained model object.
    """
    df_fit = build_fit_dataset()
    
    # Feature columns for classification matching BUG 6 feature specs
    feature_cols = [
        "description_similarity", "skill_overlap_ratio", "skill_coverage_ratio",
        "experience_match", "category_match"
    ]
    
    X = df_fit[feature_cols]
    y = df_fit["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 1. Train Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    
    lr_preds = lr.predict(X_test)
    lr_f1 = f1_score(y_test, lr_preds)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds, zero_division=0)
    lr_rec = recall_score(y_test, lr_preds, zero_division=0)
    logger.info("Logistic Regression F1: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, AUC: %.4f",
                lr_f1, lr_acc, lr_prec, lr_rec, roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
    
    # 2. Train XGBoost
    logger.info("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_prec = precision_score(y_test, xgb_preds, zero_division=0)
    xgb_rec = recall_score(y_test, xgb_preds, zero_division=0)
    logger.info("XGBoost F1: %.4f, Accuracy: %.4f, Precision: %.4f, Recall: %.4f, AUC: %.4f",
                xgb_f1, xgb_acc, xgb_prec, xgb_rec, roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))
    
    # Save the best model based on F1
    best_model = xgb_model if xgb_f1 > lr_f1 else lr
    best_model_name = "XGBoost" if xgb_f1 > lr_f1 else "LogisticRegression"
    
    logger.info("Saving best model (%s) to %s...", best_model_name, FIT_PREDICTOR_PATH)
    
    Path(FIT_PREDICTOR_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, FIT_PREDICTOR_PATH)
    logger.info("Fit predictor saved.")
    
    return best_model

if __name__ == "__main__":
    train_fit_predictor()
