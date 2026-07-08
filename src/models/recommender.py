"""Job Recommendation Engine.

Filters jobs by domain category, computes a weighted similarity score using
TF-IDF cosine similarity, Jaccard skill overlap, and experience match, 
and returns ranked results with justifications.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import PROCESSED_DATA_DIR, RECOMMENDATION_COUNT
from src.features.text_features import fit_tfidf
from src.features.match_features import extract_skills, calculate_skill_overlap, calculate_jaccard_similarity
from src.parsing.resume_parser import extract_experience_years

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Predefined nearest categories mapping in case of small clusters
NEAREST_CATEGORIES: Dict[str, List[str]] = {
    "Data Science": ["Business Analyst", "Python Developer"],
    "Python Developer": ["Data Science", "Web Designing"],
    "Web Designing": ["Java Developer", "DotNet Developer"],
    "Java Developer": ["DotNet Developer", "Web Designing"],
    "DotNet Developer": ["Java Developer", "Web Designing"],
    "DevOps Engineer": ["Network Security Engineer", "Database"],
    "Network Security Engineer": ["DevOps Engineer", "Database"],
    "Database": ["DevOps Engineer", "ETL Developer"],
    "ETL Developer": ["Database", "Hadoop"],
    "Hadoop": ["ETL Developer", "Database"],
    "Automation Testing": ["Testing", "Java Developer"],
    "Testing": ["Automation Testing", "Java Developer"],
    "Business Analyst": ["Data Science", "PMO"],
    "PMO": ["Business Analyst", "Operations Manager"],
    "Operations Manager": ["PMO", "Sales"],
    "Sales": ["Operations Manager", "HR"],
    "HR": ["Sales", "Operations Manager"],
    "Blockchain": ["Python Developer", "Java Developer"],
    "Civil Engineer": ["Mechanical Engineer", "Electrical Engineering"],
    "Mechanical Engineer": ["Civil Engineer", "Electrical Engineering"],
    "Electrical Engineering": ["Mechanical Engineer", "Civil Engineer"],
    "SAP Developer": ["Database", "Java Developer"],
    "Advocate": ["HR", "PMO"],
    "Arts": ["HR", "Operations Manager"],
    "Health and fitness": ["HR", "Operations Manager"],
}

class JobRecommender:
    """Cosine similarity and skill Jaccard recommendation engine."""

    def __init__(self, vectorizer: Any = None) -> None:
        """Initializes the recommender.

        Args:
            vectorizer: Optional pre-fitted TF-IDF vectorizer.
        """
        self.vectorizer = vectorizer
        self.df_jobs: pd.DataFrame = None
        self.job_vectors: Any = None
        
    def load_jobs_and_vectorize(self) -> None:
        """Loads preprocessed jobs, builds corpus text, and fits/transforms TF-IDF."""
        jobs_path: Path = PROCESSED_DATA_DIR / "merged_jobs.csv"
        if not jobs_path.exists():
            raise FileNotFoundError(f"Merged jobs not found at {jobs_path}. Please run preprocess.py first.")
            
        logger.info("Loading merged job corpus...")
        self.df_jobs = pd.read_csv(jobs_path)
        
        if "predicted_category" not in self.df_jobs.columns:
            logger.warning("predicted_category column not found. Running with fallback category.")
            self.df_jobs["predicted_category"] = "Data Science"
            
        logger.info("Fitting recommender TF-IDF on job description text...")
        if self.vectorizer is None:
            self.vectorizer = fit_tfidf(self.df_jobs["clean_description"].fillna(""), max_features=10000)
            
        self.job_vectors = self.vectorizer.transform(self.df_jobs["clean_description"].fillna(""))
        logger.info("Job corpus vectorized. Shape: %s", self.job_vectors.shape)

    def get_recommendations(
        self, 
        clean_resume_text: str, 
        predicted_category: str = "", 
        top_n: int = RECOMMENDATION_COUNT
    ) -> pd.DataFrame:
        """Finds top-N recommendations using domain-filtering and combined score ranking.

        Args:
            clean_resume_text: Pre-cleaned resume text.
            predicted_category: Category domain predicted for the resume.
            top_n: Number of recommendations to return.

        Returns:
            A DataFrame containing recommended jobs, match scores, and explanations.
        """
        if self.job_vectors is None or self.df_jobs is None:
            self.load_jobs_and_vectorize()
            
        # 1. Filter jobs by predicted category first
        filtered_df = self.df_jobs.copy()
        if predicted_category:
            cat_jobs = filtered_df[filtered_df["predicted_category"] == predicted_category]
            if len(cat_jobs) < 10:
                nearest = NEAREST_CATEGORIES.get(predicted_category, [])
                allowed_cats = [predicted_category] + nearest
                filtered_df = filtered_df[filtered_df["predicted_category"].isin(allowed_cats)]
            else:
                filtered_df = cat_jobs
                
        # Keep a copy of category-filtered df in case strict title filter returns empty
        cat_filtered_backup = filtered_df.copy()
                
        # If predicted category is Data Science, apply a strict title keyword filter
        if predicted_category == "Data Science":
            import re
            ds_pattern = r"\b(data|scientist|science|machine learning|ml|deep learning|nlp|computer vision|artificial intelligence|ai|analytics|analyst|research|quantitative|statistician|statistics|quant)\b"
            filtered_df = filtered_df[filtered_df["title"].str.lower().apply(lambda t: bool(re.search(ds_pattern, t)))]
                
        # Fallback only if strict filtering results in 0 matches
        if len(filtered_df) == 0:
            filtered_df = cat_filtered_backup
        if len(filtered_df) == 0:
            filtered_df = self.df_jobs.copy()
            
        # Get tfidf vectors for filtered subset
        filtered_indices = filtered_df.index.tolist()
        filtered_job_vectors = self.job_vectors[filtered_indices]
        
        # 2. Compute Cosine Similarity between resume and filtered job descriptions
        resume_vector = self.vectorizer.transform([clean_resume_text])
        cos_similarities = cosine_similarity(resume_vector, filtered_job_vectors).flatten()
        
        # 3. Extract skills and experience
        resume_skills = set(extract_skills(clean_resume_text))
        resume_exp = extract_experience_years(clean_resume_text)
        
        results: List[Dict[str, Any]] = []
        job_rows = filtered_df.to_dict(orient="records")
        for i, job_row in enumerate(job_rows):
            # Component A: Cosine Similarity on description
            tfidf_score = float(cos_similarities[i])
            
            # Component B: Jaccard Skill Similarity
            job_skills_str = job_row.get("extracted_skills", "")
            if isinstance(job_skills_str, str) and job_skills_str.strip():
                job_skills = set([s.strip().lower() for s in job_skills_str.split(",") if s.strip()])
            else:
                job_skills_str_fallback = job_row.get("skills", "")
                if isinstance(job_skills_str_fallback, str) and job_skills_str_fallback.strip():
                    job_skills = set([s.strip().lower() for s in job_skills_str_fallback.split(",") if s.strip()])
                else:
                    job_skills = set(extract_skills(str(job_row.get("clean_description", ""))))
                
            overlap_score = calculate_skill_overlap(list(resume_skills), list(job_skills))
            jacc_score = calculate_jaccard_similarity(resume_skills, job_skills)
            skill_score = (overlap_score + jacc_score) / 2
            
            # Component C: Experience Match Score (with absolute difference penalty)
            job_min_exp = float(job_row.get("min_experience", 0.0))
            exp_diff = abs(resume_exp - job_min_exp)
            exp_score = max(0.0, 1.0 - 0.15 * exp_diff)
            
            # Scale TF-IDF cosine similarity to boost realistic match percentage ranges
            scaled_tfidf = min(1.0, tfidf_score * 3.0)
            
            # Combined similarity score
            combined_score = 0.6 * skill_score + 0.2 * scaled_tfidf + 0.2 * exp_score
            
            # Generate justification
            matched_skills_list = list(resume_skills.intersection(job_skills))
            matched_skills_str = ", ".join(matched_skills_list[:4])
            
            if matched_skills_list:
                explanation = f"Matches your skills: {matched_skills_str}."
            else:
                explanation = "Good fit for your profile."
                
            if resume_exp >= job_min_exp:
                explanation += f" Meets experience requirement ({int(job_min_exp)} yrs)."
            else:
                explanation += f" Requires {int(job_min_exp)} yrs exp."
                
            job_row["tfidf_score"] = tfidf_score
            job_row["jaccard_score"] = skill_score  # Backwards compatibility
            job_row["skill_score"] = skill_score
            job_row["experience_score"] = exp_score
            job_row["similarity_score"] = combined_score
            job_row["explanation"] = explanation
            
            results.append(job_row)
            
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values(by="similarity_score", ascending=False).head(top_n)
            
        return df_results

if __name__ == "__main__":
    recommender = JobRecommender()
    recommender.load_jobs_and_vectorize()
    test_resume = "python developer data scientist numpy pandas machine learning sql"
    recs = recommender.get_recommendations(test_resume, "Data Science", top_n=3)
    for i, r in recs.iterrows():
        print(f"Title: {r['title']} | Score: {r['similarity_score']:.4f} | Explanation: {r['explanation']}")
