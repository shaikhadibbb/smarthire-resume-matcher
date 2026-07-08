"""Feature engineering and matching logic between resumes and jobs.

Computes skill overlap, years of experience, and text similarity features.
"""

import os
import re
import json
import logging
import difflib
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import SKILLS_DB_PATH
from src.parsing.resume_parser import extract_experience_years

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Complete SKILLS_DB organized by category
DEFAULT_SKILL_DB: Dict[str, List[str]] = {
    "Data Science": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "tf", "pytorch",
        "keras", "matplotlib", "seaborn", "plotly", "tableau", "powerbi", "machine learning", "ml",
        "deep learning", "dl", "nlp", "natural language processing", "computer vision", "cv", "statistics",
        "statistical analysis", "regression", "classification", "clustering", "time series", "forecasting",
        "a/b testing", "hypothesis testing", "data mining", "etl", "data pipeline", "apache spark", "spark",
        "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery", "aws s3", "ec2", "lambda", "sagemaker",
        "azure ml", "gcp", "google cloud", "docker", "kubernetes", "mlflow", "wandb", "git", "github",
        "jupyter", "jupyter notebook", "colab", "excel", "vba", "rstudio", "spss", "sas", "matlab",
        "scipy", "statsmodels", "xgboost", "lightgbm", "catboost", "random forest", "svm",
        "logistic regression", "linear regression", "decision tree", "gradient boosting", "ensemble methods",
        "cross-validation", "grid search", "hyperparameter tuning", "feature engineering",
        "dimensionality reduction", "pca", "t-sne", "umap", "word2vec", "tf-idf", "bert", "sentiment analysis",
        "named entity recognition", "ner", "topic modeling", "lda", "recommendation systems",
        "collaborative filtering", "neural networks", "cnn", "rnn", "lstm", "gru", "transformer", "gan",
        "reinforcement learning", "opencv", "pillow", "nltk", "spacy", "gensim", "hugging face",
        "transformers", "langchain", "streamlit", "flask", "django", "fastapi", "rest api", "graphql",
        "mongodb", "postgresql", "mysql", "sqlite", "redis", "elasticsearch", "neo4j", "dask", "polars",
        "modin", "vaex", "ray", "feature store", "data lake", "data warehouse", "olap", "data modeling",
        "schema design", "data governance", "data quality", "data lineage", "metadata", "great expectations",
        "pandera", "pydantic"
    ],
    "Web Development": [
        "html", "html5", "css", "css3", "javascript", "js", "typescript", "ts", "react", "react.js",
        "vue", "vue.js", "angular", "node.js", "node", "express", "next.js", "nuxt.js", "gatsby", "redux",
        "mobx", "webpack", "vite", "babel", "npm", "yarn", "pnpm", "tailwind css", "bootstrap", "material ui",
        "mui", "sass", "less", "jquery", "ajax", "json", "xml", "soap", "rest", "api", "graphql",
        "websocket", "oauth", "jwt", "authentication", "authorization", "ci/cd", "jenkins", "github actions",
        "gitlab ci", "travis", "circleci", "terraform", "ansible", "nginx", "apache", "cdn", "cloudflare",
        "vercel", "netlify", "heroku", "aws", "azure", "gcp", "firebase", "supabase", "prisma", "orm",
        "sequelize", "mongoose", "sqlalchemy", "alembic", "migration", "testing", "jest", "mocha",
        "cypress", "selenium", "playwright", "unit testing", "integration testing", "e2e testing", "tdd",
        "bdd", "agile", "scrum", "kanban", "jira", "confluence", "figma", "sketch", "adobe xd",
        "responsive design", "mobile-first", "pwa", "spa", "ssr", "csr", "hydration", "seo", "accessibility",
        "a11y", "web performance", "lighthouse", "core web vitals", "service worker", "web assembly",
        "wasm", "three.js", "d3.js", "chart.js", "canvas", "svg", "webgl"
    ],
    "Mobile Development": [
        "swift", "objective-c", "kotlin", "java", "flutter", "dart", "react native", "ionic", "cordova",
        "xamarin", "android studio", "xcode", "cocoapods", "gradle", "fastlane", "app store", "play store",
        "firebase", "push notification", "geolocation", "maps", "arcore", "arkit", "core ml",
        "tensorflow lite", "on-device ml"
    ],
    "DevOps / Cloud": [
        "linux", "bash", "shell scripting", "powershell", "python scripting", "ansible", "puppet", "chef",
        "terraform", "cloudformation", "pulumi", "docker", "containerization", "kubernetes", "k8s", "helm",
        "istio", "service mesh", "jenkins", "gitlab ci", "github actions", "azure devops", "aws codepipeline",
        "codebuild", "codedeploy", "ec2", "ecs", "eks", "fargate", "lambda", "serverless", "api gateway",
        "cloudfront", "route53", "s3", "rds", "dynamodb", "aurora", "elasticache", "elasticsearch",
        "opensearch", "cloudwatch", "prometheus", "grafana", "datadog", "new relic", "splunk", "elk stack",
        "logstash", "kibana", "fluentd", "jaeger", "zipkin", "envoy", "nginx", "haproxy", "traefik",
        "letsencrypt", "ssl/tls", "vpn", "vpc", "networking", "subnet", "firewall", "security group",
        "iam", "rbac", "zero trust", "backup", "disaster recovery", "dr", "high availability", "ha",
        "load balancing", "auto scaling", "blue-green deployment", "canary deployment", "feature flags",
        "chaos engineering", "cost optimization", "finops"
    ],
    "Cybersecurity": [
        "penetration testing", "ethical hacking", "vulnerability assessment", "threat modeling",
        "risk assessment", "compliance", "gdpr", "hipaa", "soc2", "iso 27001", "nist", "cissp", "ceh",
        "oscp", "security+", "network security", "application security", "appsec", "devsecops", "sast",
        "dast", "iast", "sca", "dependency scanning", "container security", "secrets management",
        "vault", "hashicorp", "siem", "soar", "edr", "xdr", "mdr", "threat intelligence", "incident response",
        "forensics", "malware analysis", "reverse engineering", "cryptography", "encryption", "hashing",
        "digital signature", "pki", "blockchain", "smart contract", "web3"
    ],
    "Business / Management": [
        "project management", "product management", "program management", "portfolio management", "agile",
        "scrum", "safe", "lean", "six sigma", "pmp", "prince2", "itil", "cobit", "togaf", "business analysis",
        "requirements gathering", "strong stakeholder management", "change management", "risk management",
        "vendor management", "contract management", "budget management", "financial modeling", "forecasting",
        "planning", "strategy", "okr", "kpi", "metrics", "dashboard", "reporting", "presentation",
        "communication", "negotiation", "leadership", "team management", "mentoring", "coaching",
        "conflict resolution", "decision making", "critical thinking", "problem solving", "analytical thinking",
        "creative thinking", "innovation", "design thinking", "ux research", "user research", "market research",
        "competitive analysis", "swot", "pestle", "porter's five forces", "value chain", "business model canvas",
        "lean canvas", "customer journey mapping", "persona", "empathy map", "ideation", "prototyping",
        "usability testing", "a/b testing", "conversion rate optimization", "cro", "growth hacking",
        "digital marketing", "seo", "sem", "ppc", "google ads", "facebook ads", "social media marketing",
        "content marketing", "email marketing", "marketing automation", "hubspot", "salesforce", "crm",
        "sales", "business development", "account management", "customer success", "customer support",
        "helpdesk", "zendesk", "intercom", "freshdesk"
    ]
}

# Cache to optimize repeat lookups
_SKILLS_CACHE: Dict[Tuple[str, bool], List[str]] = {}

def load_skills_db() -> Dict[str, List[str]]:
    """Loads the categorized skills database from the JSON file.

    Returns:
        A dictionary mapping categories to lists of skills.
    """
    if os.path.exists(SKILLS_DB_PATH):
        try:
            with open(SKILLS_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading skills database JSON: %s. Using default fallback.", e)
    else:
        logger.warning("Skills database not found at %s. Using default fallback.", SKILLS_DB_PATH)
    return DEFAULT_SKILL_DB

def get_flat_skills_list() -> List[str]:
    """Flattens all categories in the skills database into a single list.

    Returns:
        A list of unique skill strings.
    """
    skills_db = load_skills_db()
    all_skills: Set[str] = set()
    for cat, skills in skills_db.items():
        for skill in skills:
            all_skills.add(skill.lower().strip())
    return list(all_skills)

def extract_skills(text: str, skill_db: List[str] = None, fuzzy: bool = True) -> List[str]:
    """Extracts skills present in text from the skills database.

    Uses case-insensitive regex matching with word boundaries to avoid partial matches
    (e.g., matching "java" from "javascript"). Also supports fuzzy matching for minor typos
    and handles custom context filters (e.g. for Excel).

    Args:
        text: The text to search (resume or job description).
        skill_db: An optional list of skill names to match against.
        fuzzy: Whether to run fuzzy matching checks.

    Returns:
        A list of normalized, matched skill strings with duplicates preserved for frequency.
    """
    if not text:
        return []
        
    if skill_db is None:
        cache_key = (text, fuzzy)
        if cache_key in _SKILLS_CACHE:
            return _SKILLS_CACHE[cache_key].copy()
            
        skill_db = get_flat_skills_list()
        
    text_lower = text.lower()
    extracted: List[str] = []
    
    # Pre-tokenize text for fuzzy matching helper
    tokens = re.findall(r'[a-z0-9+#.-]+', text_lower)
    
    for skill in skill_db:
        skill_clean = skill.strip().lower()
        if not skill_clean:
            continue
            
        # 1. Excel Context Filter Rule
        if skill_clean == "excel":
            excel_context_keywords = [
                "data", "analysis", "analytics", "analyst", "reporting", "sheet", 
                "pivot", "dashboard", "vlookup", "finance", "business", "statistics", 
                "statistical", "modeling", "vba"
            ]
            if not any(ctx in text_lower for ctx in excel_context_keywords):
                continue
                
        # 2. R programming language context check
        if skill_clean == "r":
            r_pattern = r"\br\b"
            exact_r = len(re.findall(r_pattern, text_lower))
            if exact_r > 0:
                r_context = [
                    "programming", "language", "studio", "stats", "statistics", "data", 
                    "python", "sql", "analysis", "modeling", "rstudio", "ggplot", "developer"
                ]
                if any(ctx in text_lower for ctx in r_context):
                    for _ in range(exact_r):
                        extracted.append("r")
            continue

        # 3. Exact Regex Matching with Word Boundaries
        special_chars = ["c++", "c#", "net", ".net", "ci/cd", "a/b testing", "co-lab", "next.js", "node.js", "vue.js", "react.js", "three.js", "d3.js", "chart.js"]
        if skill_clean in special_chars:
            pattern = r"(?:\s|^|[.,;.:\(\)])" + re.escape(skill_clean) + r"(?:\s|$|[.,;.:\(\)])"
        else:
            pattern = r"\b" + re.escape(skill_clean) + r"\b"
            
        exact_matches = list(re.finditer(pattern, text_lower))
        count = len(exact_matches)
        
        if count > 0:
            for _ in range(count):
                extracted.append(skill_clean)
        elif fuzzy and len(skill_clean) > 4:
            # 4. Optimized Fuzzy Matching for minor typos
            if not skill_clean[0] in text_lower:
                continue
                
            words_in_skill = skill_clean.split()
            n = len(words_in_skill)
            
            fuzzy_count = 0
            if n == 1:
                for token in tokens:
                    if not token or token[0] != skill_clean[0]:
                        continue
                    if len(token) >= len(skill_clean) - 1 and len(token) <= len(skill_clean) + 1:
                        ratio = difflib.SequenceMatcher(None, token, skill_clean).ratio()
                        if ratio >= 0.85:
                            fuzzy_count += 1
            else:
                for j in range(len(tokens) - n + 1):
                    phrase = " ".join(tokens[j:j+n])
                    if not phrase or phrase[0] != skill_clean[0]:
                        continue
                    ratio = difflib.SequenceMatcher(None, phrase, skill_clean).ratio()
                    if ratio >= 0.85:
                        fuzzy_count += 1
                        
            if fuzzy_count > 0:
                for _ in range(fuzzy_count):
                    extracted.append(skill_clean)
                    
    if skill_db is not None and 'cache_key' in locals():
        _SKILLS_CACHE[cache_key] = extracted.copy()
        
    return extracted

def calculate_skill_overlap(resume_skills: List[str], job_skills: List[str]) -> float:
    """Computes the ratio of job skills possessed by the candidate (Jaccard-like overlap).

    Args:
        resume_skills: List of candidate skills.
        job_skills: List of job skills.

    Returns:
        A float representing the match ratio [0, 1].
    """
    if not job_skills:
        return 0.5
        
    overlap = set(resume_skills).intersection(set(job_skills))
    return len(overlap) / len(job_skills)

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculates the Jaccard similarity index between two sets.

    Args:
        set1: The first set of elements.
        set2: The second set of elements.

    Returns:
        The Jaccard similarity index [0, 1].
    """
    if not set1 and not set2:
        return 0.0
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(set1.intersection(set2)) / len(union)

def compute_match_features(
    resume_text: str, 
    job_row: Dict[str, Any], 
    tfidf_vectorizer: Any, 
    job_tfidf_vector: Any = None,
    resume_skills: List[str] = None
) -> Dict[str, float]:
    """Computes matching features between a single resume and a job row.

    Args:
        resume_text: Cleaned resume text.
        job_row: A dictionary representing job attributes.
        tfidf_vectorizer: A fitted TF-IDF vectorizer.
        job_tfidf_vector: Precomputed job TF-IDF vector to save time.
        resume_skills: Precomputed list of candidate skills.

    Returns:
        A dictionary containing similarity and match features.
    """
    clean_desc = job_row.get("clean_description", "")
    clean_desc = clean_desc if isinstance(clean_desc, str) else ""
    clean_skills = job_row.get("clean_skills", "")
    clean_skills = clean_skills if isinstance(clean_skills, str) else ""
    
    # 1. Cosine similarity on full description
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    
    if job_tfidf_vector is not None:
        sim = float(cosine_similarity(resume_tfidf, job_tfidf_vector)[0][0])
    else:
        job_tfidf = tfidf_vectorizer.transform([clean_desc])
        sim = float(cosine_similarity(resume_tfidf, job_tfidf)[0][0])
        
    # 2. Skill overlap
    if resume_skills is None:
        resume_skills = extract_skills(resume_text, fuzzy=True)
    
    # Check if job has a pre-extracted skills list (from extracted_skills or skills)
    job_skills_str = job_row.get("extracted_skills", "")
    if isinstance(job_skills_str, str) and job_skills_str.strip():
        job_skills = [s.strip().lower() for s in job_skills_str.split(",")]
    else:
        job_skills_str_fallback = job_row.get("skills", "")
        if isinstance(job_skills_str_fallback, str) and job_skills_str_fallback.strip():
            job_skills = [s.strip().lower() for s in job_skills_str_fallback.split(",")]
        else:
            job_skills = extract_skills(clean_desc, fuzzy=False)
        
    skill_overlap = calculate_skill_overlap(resume_skills, job_skills)
    skill_coverage = len(set(resume_skills).intersection(set(job_skills))) / len(set(resume_skills)) if resume_skills else 0.0
    
    # 3. Experience Match
    cand_exp = extract_experience_years(resume_text)
    min_exp = float(job_row.get("min_experience", 0.0))
    
    exp_diff = cand_exp - min_exp
    exp_match = 1.0 if cand_exp >= min_exp else (cand_exp / min_exp if min_exp > 0.0 else 1.0)
    
    # Determine category match
    job_category = job_row.get("predicted_category", "")
    resume_category = job_row.get("resume_category", "")
    cat_match = 1.0 if job_category and resume_category and job_category == resume_category else 0.0
    
    return {
        "description_similarity": float(sim),
        "skill_overlap_ratio": float(skill_overlap),
        "skill_coverage_ratio": float(skill_coverage),
        "candidate_experience": float(cand_exp),
        "job_min_experience": float(min_exp),
        "experience_diff": float(exp_diff),
        "experience_match": float(exp_match),
        "category_match": float(cat_match)
    }
