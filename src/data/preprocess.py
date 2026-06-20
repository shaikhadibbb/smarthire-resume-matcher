"""Data preprocessing module for resumes and job postings.

Cleans text using NLTK WordNet lemmatization, handles negation stopwords, 
parses experience, and merges job corpora.
"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, List, Union
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure required NLTK resources are downloaded
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        logger.warning("NLTK download failed for %s: %s", resource, e)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Cleans and normalizes text using lemmatization and custom stopwords.

    Keeps numbers and version tags (e.g., Python 3.9, C++, C#) while removing 
    noise and special characters. Retains negations like "not", "no", "nor", "against".

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned, space-separated string of tokens.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    
    # Keep alphanumerics, dots (for versions like 3.9), pluses (for C++), hashes (for C#), spaces
    # Reject other special punctuation
    text = re.sub(r"[^\w\s\.\+#-]", " ", text)
    
    # Remove RT/CC patterns
    text = re.sub(r"\b(rt|cc)\b", " ", text)
    
    # Tokenize
    words = nltk.word_tokenize(text)
    
    # Customize stopwords: retain negative descriptors that change sentence meaning
    negations = {"not", "no", "nor", "against"}
    stop_words = set(stopwords.words("english")) - negations
    
    cleaned_tokens: List[str] = []
    for word in words:
        if word not in stop_words and len(word) > 1:
            # Lemmatize verbs, nouns, adjectives
            lemmatized = lemmatizer.lemmatize(word, pos="v")
            lemmatized = lemmatizer.lemmatize(lemmatized, pos="n")
            cleaned_tokens.append(lemmatized)
            
    # Remove extra spaces
    return " ".join(cleaned_tokens).strip()

def parse_naukri_experience(exp_str: Union[str, float]) -> Tuple[float, float]:
    """Parses experience strings from Naukri listings into min and max years.

    Args:
        exp_str: Raw experience string (e.g. "3 - 8 yrs", "5 yrs").

    Returns:
        A tuple of (min_experience, max_experience).
    """
    if not isinstance(exp_str, str) or pd.isna(exp_str):
        return 0.0, 2.0
    
    numbers = re.findall(r"\d+(?:\.\d+)?", exp_str)
    if len(numbers) >= 2:
        try:
            return float(numbers[0]), float(numbers[1])
        except ValueError:
            pass
    elif len(numbers) == 1:
        try:
            return float(numbers[0]), float(numbers[0])
        except ValueError:
            pass
            
    return 0.0, 2.0

def parse_linkedin_experience(exp_level: Union[str, float]) -> Tuple[float, float]:
    """Maps LinkedIn experience levels to numeric ranges.

    Args:
        exp_level: Experience string (e.g. "Entry level", "Associate").

    Returns:
        A tuple of (min_experience, max_experience).
    """
    if not isinstance(exp_level, str) or pd.isna(exp_level):
        return 0.0, 2.0
        
    level = exp_level.lower().strip()
    
    if "internship" in level:
        return 0.0, 1.0
    elif "entry" in level:
        return 0.0, 2.0
    elif "associate" in level:
        return 2.0, 5.0
    elif "mid" in level or "senior" in level:
        return 5.0, 10.0
    elif "director" in level:
        return 10.0, 15.0
    elif "executive" in level:
        return 12.0, 20.0
    else:
        return 0.0, 2.0

def parse_experience_from_text(text: str) -> Tuple[float, float]:
    """Parses experience requirements from text (e.g. "3+ years", "minimum 2 years", "5-7 years").
    
    Args:
        text: Raw text containing description.

    Returns:
        A tuple of (min_experience, max_experience).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0
        
    text_lower = text.lower()
    
    # Pattern A: "5-7 years" or "5 to 7 years" or "5 - 7 yrs"
    range_pattern = r"\b(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs)\b"
    match_range = re.search(range_pattern, text_lower)
    if match_range:
        try:
            return float(match_range.group(1)), float(match_range.group(2))
        except ValueError:
            pass
            
    # Pattern B: "3+ years", "3+ yrs", "3 years+"
    plus_pattern = r"\b(\d+)\+?\s*(?:years?|yrs)\b"
    match_plus = re.search(plus_pattern, text_lower)
    if match_plus:
        try:
            val = float(match_plus.group(1))
            return val, val + 3.0
        except ValueError:
            pass
            
    # Pattern C: "minimum 2 years", "at least 4 years"
    min_pattern = r"\b(?:minimum|min|at least)\s*(\d+)\s*(?:years?|yrs)\b"
    match_min = re.search(min_pattern, text_lower)
    if match_min:
        try:
            val = float(match_min.group(1))
            return val, val + 3.0
        except ValueError:
            pass
            
    return 0.0, 0.0

def preprocess_jobs() -> None:
    """Preprocesses and merges Naukri and LinkedIn job listings into a single corpus."""
    logger.info("Starting jobs preprocessing...")
    naukri_path = RAW_DATA_DIR / "naukri_dataset.csv"
    linkedin_path = RAW_DATA_DIR / "linkedin_dataset.csv"
    
    # Process Naukri
    logger.info("Loading Naukri dataset...")
    df_naukri = pd.read_csv(naukri_path)
    
    df_naukri["jobtitle"] = df_naukri["jobtitle"].fillna("Unknown Title")
    df_naukri["company"] = df_naukri["company"].fillna("Unknown Company")
    df_naukri["joblocation_address"] = df_naukri["joblocation_address"].fillna("India")
    df_naukri["skills"] = df_naukri["skills"].fillna("")
    df_naukri["jobdescription"] = df_naukri["jobdescription"].fillna("")
    
    exp_parsed = df_naukri["experience"].apply(parse_naukri_experience)
    df_naukri["min_experience"] = [e[0] for e in exp_parsed]
    df_naukri["max_experience"] = [e[1] for e in exp_parsed]
    
    # Parse experience from description too
    for i, row in df_naukri.iterrows():
        if df_naukri.at[i, "min_experience"] == 0.0:
            desc_min, desc_max = parse_experience_from_text(row["jobdescription"])
            if desc_min > 0.0:
                df_naukri.at[i, "min_experience"] = desc_min
                df_naukri.at[i, "max_experience"] = desc_max
                
    df_n_clean = pd.DataFrame({
        "title": df_naukri["jobtitle"],
        "company": df_naukri["company"],
        "location": df_naukri["joblocation_address"],
        "skills": df_naukri["skills"],
        "description": df_naukri["jobdescription"],
        "min_experience": df_naukri["min_experience"],
        "max_experience": df_naukri["max_experience"],
        "source": "Naukri"
    })
    
    # Process LinkedIn
    logger.info("Loading LinkedIn dataset...")
    df_linkedin = pd.read_csv(linkedin_path)
    
    df_linkedin["title"] = df_linkedin["title"].fillna("Unknown Title")
    df_linkedin["company_name"] = df_linkedin["company_name"].fillna("Unknown Company")
    df_linkedin["location"] = df_linkedin["location"].fillna("Remote")
    df_linkedin["skills_desc"] = df_linkedin["skills_desc"].fillna("")
    df_linkedin["description"] = df_linkedin["description"].fillna("")
    
    exp_parsed_li = df_linkedin["formatted_experience_level"].apply(parse_linkedin_experience)
    df_linkedin["min_experience"] = [e[0] for e in exp_parsed_li]
    df_linkedin["max_experience"] = [e[1] for e in exp_parsed_li]
    
    # Parse experience from description too
    for i, row in df_linkedin.iterrows():
        if df_linkedin.at[i, "min_experience"] == 0.0:
            desc_min, desc_max = parse_experience_from_text(row["description"])
            if desc_min > 0.0:
                df_linkedin.at[i, "min_experience"] = desc_min
                df_linkedin.at[i, "max_experience"] = desc_max
                
    df_l_clean = pd.DataFrame({
        "title": df_linkedin["title"],
        "company": df_linkedin["company_name"],
        "location": df_linkedin["location"],
        "skills": df_linkedin["skills_desc"],
        "description": df_linkedin["description"],
        "min_experience": df_linkedin["min_experience"],
        "max_experience": df_linkedin["max_experience"],
        "source": "LinkedIn"
    })
    
    # Combine datasets
    logger.info("Merging job datasets...")
    df_merged = pd.concat([df_n_clean, df_l_clean], ignore_index=True)
    
    # Clean text columns
    logger.info("Cleaning job text columns...")
    df_merged["clean_title"] = df_merged["title"].apply(clean_text)
    df_merged["clean_skills"] = df_merged["skills"].apply(clean_text)
    df_merged["clean_description"] = df_merged["description"].apply(clean_text)
    
    # Keep dataset size balanced for faster local training
    if len(df_merged) > 15000:
        logger.info("Subsampling combined job corpus to 10,000 rows.")
        df_merged = df_merged.sample(n=10000, random_state=42).reset_index(drop=True)
        
    logger.info("Pre-extracting database skills for all 10,000 jobs...")
    from src.features.match_features import extract_skills
    df_merged["extracted_skills"] = df_merged.apply(
        lambda r: ",".join(extract_skills(str(r["clean_title"]) + " " + str(r["clean_description"]), fuzzy=False)),
        axis=1
    )
        
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "merged_jobs.csv"
    df_merged.to_csv(processed_path, index=False)
    logger.info("Jobs preprocessing complete. Saved to %s", processed_path)

def preprocess_resumes() -> None:
    """Preprocesses the Resume classification dataset."""
    logger.info("Starting resumes preprocessing...")
    resume_path = RAW_DATA_DIR / "resume_dataset.csv"
    
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume dataset not found at {resume_path}")
        
    df_resume = pd.read_csv(resume_path)
    
    logger.info("Cleaning resume text columns...")
    df_resume["clean_resume"] = df_resume["Resume"].apply(clean_text)
    
    logger.info("Pre-extracting database skills for all resumes...")
    from src.features.match_features import extract_skills
    df_resume["extracted_skills"] = df_resume["clean_resume"].apply(
        lambda t: ",".join(extract_skills(str(t), fuzzy=False))
    )
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "processed_resumes.csv"
    df_resume.to_csv(processed_path, index=False)
    logger.info("Resumes preprocessing complete. Saved to %s", processed_path)

if __name__ == "__main__":
    preprocess_jobs()
    preprocess_resumes()
