# SmartHire — Resume-to-Job Matching & Career Guidance Engine
**Final Project Report**

## Abstract
This project presents the design and implementation of **SmartHire**, an end-to-end machine learning system to help students find relevant job openings and understand their current skill gaps. By analyzing resume text, the system predicts a candidate's primary job domain, computes cosine similarities against a corpus of 10,000 job listings, estimates shortlisting probabilities using a supervised classifier trained on match features, and provides structured career recommendations. The system is built entirely using classical machine learning techniques (TF-IDF, Logistic Regression, K-Means, and XGBoost) to maintain high efficiency and transparency without the need for large-language model APIs.

---

## 1. Introduction & Project Goal
Modern job hunting is challenging for students, who often struggle to find entry-level positions matching their academic profiles or identify which technical skills are most valued by employers. Recruitment portals are flooded with applications, requiring automated pre-screening methods. 

SmartHire addresses this by building a unified portal with three core functions:
1. **Domain Classification**: Predicting the candidate's career domain (e.g., Data Science, Web Development, HR) from their resume text.
2. **Job Recommendation**: Searching and ranking job postings from a database based on semantic text similarity.
3. **Career Guidance & Shortlisting**: Predicting the probability that a candidate meets recruiters' requirements for a specific job and identifying missing skills to guide their preparation.

---

## 2. Dataset Description & Preprocessing
The system utilizes three distinct datasets sourced from Kaggle:
1. **Resume Screening Dataset**: Contains 962 resumes categorized into 25 distinct job domains (e.g., Java Developer, Data Science, Testing, HR, Sales). This dataset serves as the training set for our supervised domain classifier.
2. **Naukri Job Listings**: A dataset containing job postings in India, listing details such as job titles, company names, locations, experience requirements, and skill tags.
3. **LinkedIn Job Postings (2023–2024)**: A large-scale global job postings dataset containing detailed descriptions, salaries, and experience levels.

### Preprocessing & Merging Pipeline
The job listings from Naukri and LinkedIn were cleaned and consolidated into a single unified corpus containing 10,000 records. The preprocessing pipeline implemented in `src/data/preprocess.py` executes the following steps:
* **Text Normalization**: Resumes and job descriptions are converted to lowercase.
* **Noise Removal**: Regex patterns strip out URLs, emails, phone numbers, punctuation, extra whitespaces, and common resume artifacts like "RT" and "CC".
* **Stopword Removal**: Standard English stopwords are removed using the NLTK library to prevent frequent words (e.g., "the", "and") from dominating similarity scores.
* **Experience Standardization**: Experience requirements are parsed into numeric minimum and maximum values. For Naukri listings, ranges (e.g., "3 - 5 yrs") are parsed using regex. For LinkedIn listings, textual categories (e.g., "Entry level", "Associate", "Mid-Senior level") are mapped to numeric ranges based on industry standards.

---

## 3. Exploratory Data Analysis (EDA)
Exploratory analysis was conducted in `notebooks/01_eda.ipynb`. Key findings include:
* **Resume Categories**: The 962 resumes are distributed fairly evenly across 25 classes (ranging from 30 to 120 resumes per class). The most frequent classes are *Java Developer*, *Testing*, and *DevOps*, while the least frequent are *Database* and *Advocate*.
* **Experience Requirements**: A majority of job listings target entry-level to mid-level professionals, with a strong spike in the 0–5 years range. The average minimum experience required is 2.3 years.
* **Word Frequencies**: Word clouds reveal that resumes are dominated by terms like *project*, *developer*, *experience*, *management*, and *testing*. Job descriptions focus on *team*, *development*, *skills*, *work*, and *client*.

---

## 4. Modeling & Results

### Model A: Resume Category Classifier
A supervised pipeline was built to categorize resume text. 
* **Vectorization**: TF-IDF vectorizer (max 5,000 features, 1-2 n-grams).
* **Classifier**: Multinomial Logistic Regression.
* **Performance**: The model achieved an overall accuracy of **99.5%** on a 20% test split. 
* **Discussion**: The near-perfect performance is partly due to the highly clean, structured, and distinct terminology within the Kaggle resume dataset. While this is encouraging, in a production environment with noisier resumes, accuracy would likely degrade slightly.

### Job Recommendation Engine
Recommendations are generated using a vector space model:
* A combined text field (`title + skills + description`) is constructed for all jobs in the corpus and vectorized.
* When a resume is uploaded, it is transformed into the same TF-IDF space.
* **Cosine Similarity** is calculated between the resume vector and all job vectors.
* The top 10 jobs are returned. Qualitative validation in `notebooks/03_recommender.ipynb` confirmed that a simulated backend engineer query returned relevant software engineering and database roles.

### Job Clustering & Skill-Gap Analysis
To identify latent career families, K-Means clustering was applied to the job corpus.
* **Elbow Curve**: Computed for $K=2$ to $8$. A soft elbow was identified at $K=6$.
* **Silhouette Score**: Slices of the cluster space yielded an average silhouette score of **0.12**, reflecting some overlap in job descriptions, but distinct centroids.
* **PCA Projection**: Visualizing the clusters in 2D space confirms that technical clusters (Data Science, Web Development) segment cleanly from marketing and sales clusters.
* **Skill-Gap Identification**: The top 20 terms in each cluster were mined. Candidate resumes are mapped to their predicted cluster, and their extracted skills are compared to the cluster's core requirements. Missing skills are presented as a structured list of recommendations to the user.

### Model B: Shortlisting/Fit Predictor
Because we lacked labeled resume-to-job match history, we constructed a synthetic training set of 1,500 pairs. Pairs were labeled as `1` (shortlisted) if their heuristic match score (a weighted combination of cosine similarity, skill overlap, and experience fit) was $\ge 0.28$, and `0` otherwise.
* **Features**: Cosine similarity, skill overlap ratio, candidate years of experience, job minimum experience, experience difference, and experience match.
* **Model Comparison**:
  * **Logistic Regression**: F1-score: **93.8%**, ROC-AUC: **98.2%**
  * **XGBoost Classifier**: F1-score: **95.2%**, ROC-AUC: **99.1%**
* **Discussion**: XGBoost slightly outperforms Logistic Regression due to its ability to model non-linear boundaries (e.g. how experience differences interact with text similarities). The model is saved to `models/fit_predictor.pkl` and integrated into the Streamlit app.

---

## 5. System Limitations & Future Work
While SmartHire is highly functional, it has several limitations:
1. **Bag-of-Words Limitation**: TF-IDF ignores word order and semantics. A candidate writing "no experience in Python" would be matched with Python roles due to keyword occurrence. Future work should integrate sentence embeddings (e.g., Sentence-BERT) locally.
2. **Synthetic Fit Labels**: The fit predictor model is trained on synthetic labels derived from our own heuristics. In practice, this model would require actual historical recruiter decisions (shortlisted/rejected) to learn true preference boundaries.
3. **Keyword Matching for Skills**: The skill-gap analyzer relies on a static list of 50 common technical keywords. This list should be expanded or replaced with an automated Named Entity Recognition (NER) pipeline.

---

## 6. Conclusion
SmartHire demonstrates that classical machine learning algorithms can build a powerful, fast, and explainable job matching and career guidance engine. Logistic Regression proves highly capable for document classification, K-Means groups jobs into natural functional families, and XGBoost models candidate fit probabilities. The entire pipeline runs locally in under two minutes and is served through an interactive Streamlit UI, making it a viable, low-cost solution for academic career services.
