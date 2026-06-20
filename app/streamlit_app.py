"""Streamlit dashboard for the SmartHire resume-to-job matching application.

Integrates resume parsing, classification, unsupervised clustering, 
similarity recommendation, fit prediction, and career gap analysis in a clean web UI.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    CLASSIFIER_PATH, TFIDF_VECTORIZER_PATH, FIT_PREDICTOR_PATH, KMEANS_MODEL_PATH, PROCESSED_DATA_DIR
)
from src.parsing.resume_parser import extract_resume_text, extract_experience_years
from src.data.preprocess import clean_text
from src.features.match_features import compute_match_features, extract_skills
from src.models.recommender import JobRecommender
from src.models.clustering import JobClusterer

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

# Page configuration
st.set_page_config(
    page_title="SmartHire — Resume-to-Job Matching & Career Guidance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium visual layout)
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 0.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 2.2rem;
    }
    .card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .skill-have {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.35rem 0.7rem;
        border-radius: 4px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #A7F3D0;
    }
    .skill-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.7rem;
        border-radius: 4px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #FECACA;
    }
    .skill-extra {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.35rem 0.7rem;
        border-radius: 4px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="font-weight: 600; color: #1e3a8a;">SmartHire Portal</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #666; font-size: 0.95rem;">AI-Powered Resume-to-Job Matching & Career Guidance Engine (Classical ML Edition)</p>', unsafe_allow_html=True)
st.markdown("---")

def check_models_ready() -> bool:
    """Checks if all required model pickle checkpoints are available on disk.

    Returns:
        True if all files exist, False otherwise.
    """
    return (
        Path(CLASSIFIER_PATH).exists() and
        Path(TFIDF_VECTORIZER_PATH).exists() and
        Path(KMEANS_MODEL_PATH).exists()
    )

def is_empty(val):
    if pd.isna(val):
        return True
    if not isinstance(val, str):
        return True
    val_clean = val.strip().lower()
    return not val_clean or val_clean == "nan" or val_clean == "not specified"

@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """Loads and caches the trained scikit-learn and KMeans model parameters individually.

    Returns:
        A dictionary of models, or None if files are missing.
    """
    models = {}
    
    # Try loading classifier
    try:
        if Path(CLASSIFIER_PATH).exists():
            models["classifier"] = joblib.load(CLASSIFIER_PATH)
        else:
            models["classifier"] = None
    except Exception as e:
        logger.error("Error loading classifier: %s", e)
        models["classifier"] = None
        
    # Try loading tfidf vectorizer
    try:
        if Path(TFIDF_VECTORIZER_PATH).exists():
            models["tfidf"] = joblib.load(TFIDF_VECTORIZER_PATH)
        else:
            models["tfidf"] = None
    except Exception as e:
        logger.error("Error loading tfidf: %s", e)
        models["tfidf"] = None
        
    # Try loading KMeans
    try:
        if Path(KMEANS_MODEL_PATH).exists():
            kmeans_data = joblib.load(KMEANS_MODEL_PATH)
            models["kmeans"] = kmeans_data.get("kmeans")
            models["kmeans_tfidf"] = kmeans_data.get("vectorizer")
        else:
            models["kmeans"] = None
            models["kmeans_tfidf"] = None
    except Exception as e:
        logger.error("Error loading kmeans: %s", e)
        models["kmeans"] = None
        models["kmeans_tfidf"] = None
        
    return models

models_dict = load_all_models()

# If models_dict has no files at all, we can still load the UI and show listings, but we check if we have the database
# Sidebar layout for file upload and config
st.sidebar.header("Resume Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your resume file (PDF, DOCX, or TXT format)", 
    type=["pdf", "docx", "txt", "text"]
)

override_exp = st.sidebar.checkbox("Override parsed years of experience?")
user_exp = st.sidebar.slider("Select years of experience:", 0, 15, 2) if override_exp else None

if uploaded_file is not None:
    import uuid
    # Create temp folder to read file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    # Use a unique name to prevent accidental deletion of static files with the same name
    unique_name = f"temp_{uuid.uuid4().hex}_{uploaded_file.name}"
    temp_path = temp_dir / unique_name
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        import time
        
        # Step 1: Parsing
        with st.spinner("Parsing resume... extracting text and skills..."):
            resume_raw = extract_resume_text(temp_path)
            resume_cleaned = clean_text(resume_raw)
            cand_exp_parsed, is_fresher = extract_experience_years(resume_raw, return_fresher=True)
            cand_exp = user_exp if override_exp else cand_exp_parsed
            
            # Cleanup temp file
            if temp_path.exists():
                os.remove(temp_path)
            time.sleep(0.5)
            
        if len(resume_cleaned.strip()) == 0:
            st.error("No readable text could be extracted. Please make sure the document is not password-protected or empty.")
        else:
            # Step 2: Analyzing
            with st.spinner("Analyzing against job database..."):
                # Classification domain prediction (Model A)
                classifier_failed = False
                if models_dict.get("classifier") is not None and models_dict.get("tfidf") is not None:
                    try:
                        resume_tfidf = models_dict["tfidf"].transform([resume_cleaned])
                        predicted_cat = str(models_dict["classifier"].predict(resume_tfidf)[0])
                        confidences = models_dict["classifier"].predict_proba(resume_tfidf)
                        
                        # Apply temperature scaling to make confidence score intuitive
                        T = 0.55
                        scaled_conf = np.exp(np.log(np.maximum(confidences, 1e-12)) / T)
                        scaled_conf = scaled_conf / np.sum(scaled_conf)
                        confidence_pct = float(np.max(scaled_conf) * 100)
                    except Exception as e:
                        logger.error("Classifier prediction failed: %s", e)
                        classifier_failed = True
                else:
                    classifier_failed = True
                    
                if classifier_failed:
                    predicted_cat = "Data Science"
                    confidence_pct = 0.0
                    
                # Unsupervised clustering cluster ID
                clustering_failed = False
                try:
                    clusterer = JobClusterer()
                    cluster_id = clusterer.predict_cluster(resume_cleaned)
                    gap_report = clusterer.generate_skill_gap_report(resume_cleaned, target_cluster_id=cluster_id, predicted_category=predicted_cat)
                except Exception as e:
                    logger.error("Clustering analysis failed: %s", e)
                    clustering_failed = True
                    cluster_id = 0
                    gap_report = {
                        "missing_skills": [],
                        "extra_skills": [],
                        "warning": "Clustering model unavailable. Skill-gap analysis skipped."
                    }
                    
                resume_skills = extract_skills(resume_cleaned)
                from collections import Counter
                skill_counts = Counter(resume_skills)
                time.sleep(0.5)
                
            # Step 3: Generating recommendations
            with st.spinner("Generating recommendations..."):
                try:
                    rec_cat = "" if classifier_failed else predicted_cat
                    recommender = JobRecommender(vectorizer=models_dict.get("tfidf"))
                    recommender.load_jobs_and_vectorize()
                    recs_df = recommender.get_recommendations(
                        clean_resume_text=resume_cleaned,
                        predicted_category=rec_cat,
                        top_n=5
                    )
                except Exception as e:
                    logger.error("Recommender matching failed: %s", e)
                    recs_df = pd.DataFrame()
                time.sleep(0.5)
                
            # Layout splitting: Left = Resume Summary, Right = Matches/Career Advice
            left_col, right_col = st.columns([1, 2])
            
            # Left Column content: Profile Metrics
            with left_col:
                st.markdown("---")
                st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #333; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1.2rem;">Resume Profile Summary</p>', unsafe_allow_html=True)
                
                # Predicted Domain Card
                if classifier_failed:
                    st.markdown("""
                    <div class="card">
                        <strong style="color: #4B5563;">Predicted Domain Category</strong>
                        <div class="metric-value" style="margin-top: 0.5rem; margin-bottom: 0.2rem;">Unavailable</div>
                        <div style="font-size: 0.9rem; color: #6B7280;">Category model failed to load</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card">
                        <strong style="color: #4B5563;">Predicted Domain Category</strong>
                        <div class="metric-value" style="margin-top: 0.5rem; margin-bottom: 0.2rem;">{predicted_cat}</div>
                        <div style="font-size: 0.9rem; color: #6B7280;">Confidence: {confidence_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Warning if classifier confidence is low
                if not classifier_failed and confidence_pct < 30.0:
                    st.warning("Low confidence prediction. Results may be less accurate. Tailor your resume keywords.")
                    
                # Experience Card
                if is_fresher and not override_exp:
                    exp_display = "Fresher (0 years)"
                else:
                    exp_display = f"{cand_exp:.1f} Years"
                    
                st.markdown(f"""
                <div class="card">
                    <strong style="color: #4B5563;">Candidate Experience</strong>
                    <div class="metric-value" style="margin-top: 0.5rem; margin-bottom: 0.2rem;">{exp_display}</div>
                    <div style="font-size: 0.9rem; color: #6B7280;">Parsed from text: {cand_exp_parsed:.1f} yrs</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable section to review extracted text
                with st.expander("View parsed raw text"):
                    st.text_area("Extracted Resume Content", value=resume_raw, height=250, disabled=True)
                    
                # Build Downloadable Markdown Report
                report_skills_you_have = sorted(list(set([s for s in resume_skills if s.lower() not in SOFT_SKILLS_BLACKLIST])))
                
                import datetime
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                jobs_report_list = []
                if not recs_df.empty:
                    for idx, row in recs_df.iterrows():
                        jobs_report_list.append(f"* **{row['title']}** at **{row['company']}** ({row['location']}) — Match Score: {row['similarity_score']*100:.1f}%")
                else:
                    jobs_report_list.append("* No recommendations available")
                jobs_report_str = "\n".join(jobs_report_list)
                
                extracted_skills_str = ", ".join([s.upper() for s in sorted(list(set(resume_skills)))]) if resume_skills else "None"
                skills_to_learn_str = ", ".join([s.upper() for s in gap_report.get('missing_skills', [])]) if gap_report.get('missing_skills') else "None"
                
                domain_val = "Unavailable" if classifier_failed else f"{predicted_cat} (Confidence: {confidence_pct:.1f}%)"
                
                report_md = f"""# SmartHire Career Guidance Report
Generated on: {current_date}

## Profile Summary
* **Predicted Domain**: {domain_val}
* **Candidate Experience**: {exp_display}
* **Extracted Technical Skills**: {extracted_skills_str}

## Top 5 Recommended Jobs
{jobs_report_str}

## Skill Gap Analysis
* **Skills to Learn**: {skills_to_learn_str}

---
Note: Generated by SmartHire — Classical ML project (no LLMs used)
"""
                st.download_button(
                    label="Download Career Report",
                    data=report_md,
                    file_name="smarthire_career_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            # Right Column content: Matches and Guidance
            with right_col:
                has_detected_skills = len(resume_skills) > 0
                
                if has_detected_skills:
                    # Unsupervised discovery header
                    st.markdown("---")
                    st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #333; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1.2rem;">Unsupervised Career Guidance</p>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="card">
                        <strong style="color: #4B5563;">Job Family Discovery (Unsupervised)</strong>
                        <div class="metric-value" style="margin-top: 0.5rem; margin-bottom: 0.2rem;">Cluster {cluster_id}</div>
                        <div style="font-size: 0.9rem; color: #6B7280;">Analyzed relative to matching job cluster.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Warning if skills db is empty
                    if gap_report.get("warning"):
                        st.warning(gap_report["warning"])
                        
                    # Skill analysis grids
                    g_col1, g_col2 = st.columns([1, 1])
                    with g_col1:
                        st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #155724; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.8rem;">Skills You Have</p>', unsafe_allow_html=True)
                        has_skills_unique = sorted(list(set([s for s in resume_skills if s.lower() not in SOFT_SKILLS_BLACKLIST])))
                        if has_skills_unique:
                            tags = "".join([f'<span class="skill-have">{s.upper()}</span>' for s in has_skills_unique])
                            st.markdown(tags, unsafe_allow_html=True)
                        else:
                            st.warning("We couldn't detect specific skills. Make sure your resume lists technologies clearly (e.g., 'Python', 'SQL', 'Machine Learning').")
                            
                    with g_col2:
                        st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #721c24; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.8rem;">Recommended Skills to Learn</p>', unsafe_allow_html=True)
                        if gap_report.get("missing_skills"):
                            tags = "".join([f'<span class="skill-missing">{s.upper()}</span>' for s in gap_report["missing_skills"]])
                            st.markdown(tags, unsafe_allow_html=True)
                        else:
                            st.success("Fantastic! You possess all the top skills matched for this job family.")
                            
                    # Extra skills
                    extra_skills_unique = sorted(list(set([s for s in gap_report.get("extra_skills", []) if s.lower() not in SOFT_SKILLS_BLACKLIST])))
                    if extra_skills_unique:
                        st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #004085; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1.2rem; margin-bottom: 0.8rem;">Additional Skills Found</p>', unsafe_allow_html=True)
                        tags = "".join([
                            f'<span class="skill-extra">{s.upper()}<span style="background-color: #1E40AF; color: white; border-radius: 10px; padding: 0.1rem 0.4rem; font-size: 0.75rem; margin-left: 0.4rem; font-weight: bold;">{skill_counts[s]}</span></span>'
                            for s in extra_skills_unique
                        ])
                        st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.warning("No specific skills detected in resume. Ensure technical terms are spelled out (e.g., 'Python' not 'py').")
                    
                st.markdown("---")
                
                # Job Recommendations Similarity Engine
                st.markdown("---")
                st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #333; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;">Top 5 Recommended Jobs</p>', unsafe_allow_html=True)
                
                domain_caption = predicted_cat if not classifier_failed else "All Domains"
                st.caption(f"Jobs are filtered to your predicted domain ({domain_caption}) and ranked by combined similarity score. Scores above 70% indicate strong alignment; below 40% suggests exploring adjacent roles.")
                
                if classifier_failed:
                    st.warning("Category model unavailable. Showing all-domain jobs instead of filtered results.")
                    
                if not recs_df.empty:
                    # Display Sortable Recommendations Table
                    st.markdown('<p style="font-size: 0.8rem; font-weight: 600; color: #444; text-transform: uppercase; margin-top: 0.8rem; margin-bottom: 0.6rem;">Recommendations Overview (Sortable)</p>', unsafe_allow_html=True)
                    recs_summary_df = recs_df[["title", "company", "location", "similarity_score"]].copy()
                    
                    skills_matched_list = []
                    for idx, row in recs_df.iterrows():
                        skills_col = row.get("skills")
                        extracted_skills_col = row.get("extracted_skills")
                        if is_empty(skills_col) and is_empty(extracted_skills_col):
                            skills_matched_list.append("Job description not parsed for skills")
                        else:
                            if not is_empty(extracted_skills_col):
                                job_skills = set([s.strip().lower() for s in extracted_skills_col.split(",") if s.strip()])
                            else:
                                job_skills = set([s.strip().lower() for s in skills_col.split(",") if s.strip()])
                            overlap = sorted(list(set([s.lower() for s in resume_skills]).intersection(job_skills)))
                            if overlap:
                                skills_matched_list.append(", ".join([s.upper() for s in overlap]))
                            else:
                                skills_matched_list.append("None")
                                
                    recs_summary_df["Skills Matched"] = skills_matched_list
                    recs_summary_df["similarity_score"] = recs_summary_df["similarity_score"] * 100
                    recs_summary_df = recs_summary_df.rename(columns={
                        "title": "Job Title",
                        "company": "Company",
                        "location": "Location",
                        "similarity_score": "Match Score (%)"
                    })
                    st.dataframe(recs_summary_df, use_container_width=True)
                    
                    # Render expandable rows for each recommendation with custom HTML progress bars
                    st.markdown('<p style="font-size: 0.8rem; font-weight: 600; color: #444; text-transform: uppercase; margin-top: 1.2rem; margin-bottom: 0.6rem;">Detailed Job Breakdown</p>', unsafe_allow_html=True)
                    for idx, row in recs_df.reset_index(drop=True).iterrows():
                        match_score = row["similarity_score"]
                        title = row["title"]
                        company = row["company"]
                        loc = row["location"]
                        
                        exp_label = f"Match: {match_score*100:.1f}% | {title} at {company} ({loc})"
                        
                        with st.expander(exp_label):
                            pct = match_score * 100
                            if pct >= 70.0:
                                bar_color = "#28a745"
                                label_text = "Strong Match"
                            elif pct >= 40.0:
                                bar_color = "#ffc107"
                                label_text = "Medium Match"
                            else:
                                bar_color = "#dc3545"
                                label_text = "Low Match"
                                
                            progress_html = f"""
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; font-family: sans-serif;">
                                <div style="background-color: #e9ecef; border-radius: 4px; flex-grow: 1; height: 16px; overflow: hidden; border: 1px solid #dee2e6;">
                                    <div style="background-color: {bar_color}; width: {pct:.1f}%; height: 100%; transition: width 0.4s ease;"></div>
                                </div>
                                <span style="font-weight: bold; color: {bar_color}; font-size: 1.05rem; min-width: 55px; text-align: right;">{pct:.1f}%</span>
                                <span style="color: #6c757d; font-size: 0.9rem; font-weight: 500;">({label_text})</span>
                            </div>
                            """
                            st.markdown(progress_html, unsafe_allow_html=True)
                            
                            # Detailed breakdown metrics
                            st.write(f"**Why recommended?** {row['explanation']}")
                            st.write(f"**TF-IDF Cosine Similarity**: {row['tfidf_score']*100:.1f}% | **Skill Overlap**: {row['jaccard_score']*100:.1f}%")
                            st.write(f"**Required Experience**: {int(row['min_experience'])} to {int(row['max_experience'])} Years")
                            
                            # Matched skills sub-expander
                            skills_col = row.get("skills")
                            extracted_skills_col = row.get("extracted_skills")
                            if is_empty(skills_col) and is_empty(extracted_skills_col):
                                st.write("Job description not parsed for skills")
                            else:
                                if not is_empty(extracted_skills_col):
                                    job_skills = set([s.strip().lower() for s in extracted_skills_col.split(",") if s.strip()])
                                else:
                                    job_skills = set([s.strip().lower() for s in skills_col.split(",") if s.strip()])
                                overlap = sorted(list(set([s.lower() for s in resume_skills]).intersection(job_skills)))
                                overlap_display = ", ".join([s.upper() for s in overlap]) if overlap else "None"
                                with st.expander("Matched skills"):
                                    st.write(f"Matched skills: {overlap_display}")
                            
                            # Show raw skills and description, handling nan
                            skills_val = row['skills']
                            if pd.isna(skills_val) or not str(skills_val).strip() or str(skills_val).lower() == "nan":
                                skills_val = "Not Specified"
                            st.markdown(f"**Skills Tagged**: `{skills_val}`")
                            with st.expander("View full job description"):
                                st.write(row["description"])
                else:
                    st.info("No matching jobs found.")
                        
    except Exception as e:
        logger.error("Error processing resume upload: %s", e, exc_info=True)
        st.error(f"Error processing resume: {e}")
        
else:
    st.info("Upload a candidate resume (PDF/DOCX/TXT) in the left sidebar to begin career matching!")
    
    # Display baseline stats when idle
    st.markdown("---")
    st.markdown('<p style="font-size: 0.85rem; font-weight: 600; color: #333; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.8rem;">Available Job Listings Database</p>', unsafe_allow_html=True)
    jobs_path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    if jobs_path.exists():
        df_jobs = pd.read_csv(jobs_path)
        st.write(f"The job database contains **{len(df_jobs)}** records consolidated from **Naukri** and **LinkedIn**.")
        st.dataframe(
            df_jobs[["title", "company", "location", "skills", "source"]].sample(n=5, random_state=42),
            use_container_width=True
        )
    else:
        st.warning("Job listings database was not found. Please run the preprocessing scripts first.")

