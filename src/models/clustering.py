"""Unsupervised clustering of job postings.

Groups jobs using KMeans, profiles cluster skill requirements, and performs 
skill-gap comparison for candidate career guidance.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import PROCESSED_DATA_DIR, KMEANS_MODEL_PATH, RANDOM_STATE
from src.features.match_features import extract_skills

# Technical blacklist to avoid drowning out domain skills with soft-skills
SOFT_SKILLS_BLACKLIST = {
    "communication", "leadership", "strategy", "presentation", "negotiation", "compliance", 
    "excel", "innovation", "project management", "business development", "customer support", 
    "crm", "sales", "safe", "cv", "risk management", "team management", "change management", 
    "vendor management", "mentoring", "coaching", "conflict resolution", "decision making", 
    "critical thinking", "problem solving", "analytical thinking", "creative thinking", 
    "design thinking", "strong stakeholder management", "budget management", "planning", 
    "reporting", "dashboard", "metrics", "kpi", "okr", "other", "nan", ""
}

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class JobClusterer:
    """K-Means clustering and skill-profiling manager."""

    def __init__(self, k: int = 6) -> None:
        """Initializes the clusterer.

        Args:
            k: Number of clusters.
        """
        self.k: int = k
        self.kmeans: KMeans = None
        self.vectorizer: TfidfVectorizer = None
        self.cluster_skills: Dict[int, List[str]] = {}
        self.cluster_skills_frequencies: Dict[int, Dict[str, int]] = {}
        self.category_skills: Dict[str, List[str]] = {}
        self.category_skills_frequencies: Dict[str, Dict[str, int]] = {}

    def fit(self, df_jobs: pd.DataFrame) -> pd.DataFrame:
        """Fits KMeans on the jobs corpus and builds cluster skill profiles.

        Args:
            df_jobs: Job listings DataFrame.

        Returns:
            The DataFrame with cluster labels.
        """
        logger.info("Fitting KMeans clustering with k=%d...", self.k)
        
        # Combine job title and skills ONLY to isolate technical centroids
        combined_text = (
            df_jobs["clean_title"].fillna("") + " " +
            df_jobs["clean_skills"].fillna("")
        )
        
        # Vectorize job text
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        job_vectors = self.vectorizer.fit_transform(combined_text)
        
        # Train KMeans
        self.kmeans = KMeans(n_clusters=self.k, random_state=RANDOM_STATE, n_init=10)
        self.kmeans.fit(job_vectors)
        
        # Save model elements
        MODEL_SAVE_DIR = Path(KMEANS_MODEL_PATH).parent
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kmeans": self.kmeans, "vectorizer": self.vectorizer}, KMEANS_MODEL_PATH)
        logger.info("KMeans saved to %s", KMEANS_MODEL_PATH)
        
        df_jobs_copy = df_jobs.copy()
        df_jobs_copy["cluster"] = self.kmeans.labels_
        
        # First, pre-calculate the total number of jobs in the entire corpus containing each skill
        # (This is needed to calculate the IDF score for each skill)
        from collections import Counter
        import math
        
        skill_corpus_counts = Counter()
        cluster_jobs_skills: Dict[int, List[List[str]]] = {}
        
        # Technical blacklist to avoid drowning out domain skills with soft-skills
        BLACKLIST = SOFT_SKILLS_BLACKLIST
        
        for cluster_id in range(self.k):
            cluster_subset = df_jobs_copy[df_jobs_copy["cluster"] == cluster_id]
            cluster_jobs_skills[cluster_id] = []
            
            for _, row in cluster_subset.iterrows():
                extracted_str = row.get("extracted_skills", "")
                if isinstance(extracted_str, str) and extracted_str.strip() and not pd.isna(extracted_str):
                    row_skills = [s.strip().lower() for s in extracted_str.split(",")]
                else:
                    row_text = (
                        str(row.get("clean_title", "")) + " " +
                        str(row.get("clean_skills", "")) + " " +
                        str(row.get("clean_description", ""))
                    )
                    row_skills = extract_skills(row_text, fuzzy=False)
                
                # Clean row skills: remove empty and 'nan'
                cleaned_row_skills = list(set([s for s in row_skills if s and s != "nan"]))
                cluster_jobs_skills[cluster_id].append(cleaned_row_skills)
                
                for s in cleaned_row_skills:
                    skill_corpus_counts[s] += 1
                    
        # Now, calculate the TF-IDF-based scores for each skill in each cluster
        N = len(df_jobs_copy)
        logger.info("Profiling cluster skills based on TF-IDF scoring...")
        for cluster_id in range(self.k):
            jobs_skills_list = cluster_jobs_skills[cluster_id]
            c_size = len(jobs_skills_list)
            if c_size == 0:
                self.cluster_skills[cluster_id] = []
                self.cluster_skills_frequencies[cluster_id] = {}
                continue
                
            # Count how many jobs in this cluster contain each skill
            freqs: Dict[str, int] = {}
            for job_skills in jobs_skills_list:
                for s in job_skills:
                    freqs[s] = freqs.get(s, 0) + 1
                    
            # Compute TF-IDF scores
            scores: Dict[str, float] = {}
            for s, count in freqs.items():
                if s in BLACKLIST:
                    continue
                # Filter out skills that appear in less than 3 jobs in this cluster to avoid noise
                if count < 3:
                    continue
                tf = count / c_size
                idf = math.log((N + 1) / (skill_corpus_counts[s] + 1))
                scores[s] = tf * idf
                
            # Sort skills by TF-IDF score
            sorted_skills = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
            # Take top 20 skills as the cluster profile
            top_skills = sorted_skills[:20]
            
            self.cluster_skills[cluster_id] = top_skills
            self.cluster_skills_frequencies[cluster_id] = freqs
            logger.info("Cluster %d Top TF-IDF Skills: %s", cluster_id, top_skills[:10])
            
        # Also build predicted category skill profiles
        logger.info("Profiling predicted category skills based on TF-IDF scoring...")
        self.category_skills = {}
        self.category_skills_frequencies = {}
        
        if "predicted_category" in df_jobs_copy.columns:
            for cat in df_jobs_copy["predicted_category"].unique():
                if pd.isna(cat):
                    continue
                cat_subset = df_jobs_copy[df_jobs_copy["predicted_category"] == cat]
                cat_size = len(cat_subset)
                
                freqs = {}
                for _, row in cat_subset.iterrows():
                    s_str = row.get("extracted_skills", "")
                    if isinstance(s_str, str) and s_str.strip() and s_str.lower() != "nan" and not pd.isna(s_str):
                        row_skills = list(set([s.strip().lower() for s in s_str.split(",") if s.strip()]))
                    else:
                        row_skills = []
                    for s in row_skills:
                        if s and s != "nan" and s not in BLACKLIST:
                            freqs[s] = freqs.get(s, 0) + 1
                            
                scores = {}
                for s, count in freqs.items():
                    if count < 2:
                        continue
                    tf = count / cat_size
                    idf = math.log((N + 1) / (skill_corpus_counts[s] + 1))
                    scores[s] = tf * idf
                    
                sorted_skills = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
                self.category_skills[cat] = sorted_skills[:20]
                self.category_skills_frequencies[cat] = freqs
                
        # Save cluster & category skills map to disk
        cluster_skills_path = MODEL_SAVE_DIR / "cluster_skills.pkl"
        joblib.dump({
            "skills": self.cluster_skills, 
            "freqs": self.cluster_skills_frequencies,
            "category_skills": self.category_skills,
            "category_freqs": self.category_skills_frequencies
        }, cluster_skills_path)
        logger.info("Cluster & category skills saved successfully to %s", cluster_skills_path)
        
        return df_jobs_copy

    def predict_cluster(self, clean_resume_text: str) -> int:
        """Predicts the cluster assignment for a clean resume.

        Args:
            clean_resume_text: Cleansed resume text.

        Returns:
            The predicted cluster ID.
        """
        if self.kmeans is None or self.vectorizer is None:
            data = joblib.load(KMEANS_MODEL_PATH)
            self.kmeans = data["kmeans"]
            self.vectorizer = data["vectorizer"]
            
        resume_vec = self.vectorizer.transform([clean_resume_text])
        cluster_id = self.kmeans.predict(resume_vec)[0]
        return int(cluster_id)

    def generate_skill_gap_report(self, clean_resume_text: str, target_cluster_id: int = None, predicted_category: str = None) -> Dict[str, Any]:
        """Calculates overlaps and sorted skill gaps against the cluster or category profile.

        Args:
            clean_resume_text: Cleansed resume text.
            target_cluster_id: Optional cluster ID to evaluate against.
            predicted_category: Optional predicted category domain to evaluate against (recommended).

        Returns:
            A dictionary containing matches, gaps, and alerts.
        """
        candidate_skills = extract_skills(clean_resume_text)
        candidate_skills = list(set([s for s in candidate_skills if s and s != "nan"]))
        
        cluster_skills_path = Path(KMEANS_MODEL_PATH).parent / "cluster_skills.pkl"
        if (not self.cluster_skills or not getattr(self, "category_skills", None)) and cluster_skills_path.exists():
            try:
                data = joblib.load(cluster_skills_path)
                self.cluster_skills = data.get("skills", {})
                self.cluster_skills_frequencies = data.get("freqs", {})
                self.category_skills = data.get("category_skills", {})
                self.category_skills_frequencies = data.get("category_freqs", {})
            except Exception as e:
                logger.error("Error loading cluster skills pkl: %s", e)
                
        if not self.cluster_skills:
            self.cluster_skills = {
                0: ["python", "sql", "pandas", "numpy", "scikit-learn", "machine learning", "tensorflow", "pytorch", "tableau", "powerbi"],
                1: ["java", "spring", "docker", "kubernetes", "git", "aws", "sql", "linux"],
                2: ["javascript", "react", "html", "css", "node.js", "typescript", "angular", "bootstrap"],
                3: ["sql", "excel", "powerbi", "tableau", "data analysis", "statistics", "python"],
                4: ["aws", "devops", "docker", "kubernetes", "linux", "jenkins", "terraform", "git"],
                5: ["excel", "project management", "agile", "scrum", "jira", "communication", "salesforce"]
            }
            self.cluster_skills_frequencies = {k: {s: 1 for s in v} for k, v in self.cluster_skills.items()}
            self.category_skills = {}
            self.category_skills_frequencies = {}

        # Use category profile if available, otherwise cluster profile
        if predicted_category and getattr(self, "category_skills", None) and predicted_category in self.category_skills:
            cluster_reqs = self.category_skills[predicted_category]
            freqs_map = self.category_skills_frequencies[predicted_category]
        else:
            if target_cluster_id is None:
                target_cluster_id = self.predict_cluster(clean_resume_text)
            cluster_reqs = self.cluster_skills.get(target_cluster_id, [])
            freqs_map = self.cluster_skills_frequencies.get(target_cluster_id, {})
            
        # Intersection
        has_skills = [s for s in cluster_reqs if s in candidate_skills]
        # Difference
        missing_skills = [s for s in cluster_reqs if s not in candidate_skills and s.lower() not in SOFT_SKILLS_BLACKLIST]
        
        # Sort missing skills by frequency in the cluster
        missing_skills = sorted(missing_skills, key=lambda s: freqs_map.get(s, 0), reverse=True)
        
        extra_skills = [s for s in candidate_skills if s not in cluster_reqs]
        
        warning_msg = ""
        if not candidate_skills:
            warning_msg = (
                "We couldn't detect specific skills. Make sure your resume lists technologies "
                "clearly (e.g., 'Python', 'SQL', 'Machine Learning')."
            )
            
        return {
            "cluster_id": target_cluster_id,
            "has_skills": has_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "warning": warning_msg
        }

if __name__ == "__main__":
    jobs_path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    if jobs_path.exists():
        df = pd.read_csv(jobs_path)
        c = JobClusterer()
        c.fit(df)
