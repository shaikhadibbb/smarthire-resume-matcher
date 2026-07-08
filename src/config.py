"""Configuration module for the SmartHire project.

Defines directory paths, model save locations, and hyperparameters.
"""

from pathlib import Path
from typing import Dict, Any

# Project root directory (absolute path)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
INTERIM_DATA_DIR: Path = BASE_DIR / "data" / "interim"
PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"

# Model paths
MODEL_SAVE_DIR: Path = BASE_DIR / "models"
CLASSIFIER_PATH: Path = MODEL_SAVE_DIR / "classifier.pkl"
TFIDF_VECTORIZER_PATH: Path = MODEL_SAVE_DIR / "tfidf_vectorizer.pkl"
FIT_PREDICTOR_PATH: Path = MODEL_SAVE_DIR / "fit_predictor.pkl"
KMEANS_MODEL_PATH: Path = MODEL_SAVE_DIR / "kmeans_model.pkl"

# Skills database path
SKILLS_DB_PATH: Path = MODEL_SAVE_DIR / "skills_db.json"

# Figures directory
FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"

# Common settings
RANDOM_STATE: int = 42

# ML Model Hyperparameters
CLASSIFIER_TFIDF_MAX_FEATURES: int = 10000
CLASSIFIER_TEST_SIZE: float = 0.2

# Recommender settings
RECOMMENDER_TFIDF_MAX_FEATURES: int = 10000
RECOMMENDATION_COUNT: int = 10

# Clustering settings
CLUSTERING_K: int = 6
CLUSTERING_TOP_N_WORDS: int = 15

# Fit Predictor Heuristics
FIT_THRESHOLD: float = 0.35
