import os
import urllib.request
import sys

# Add src to python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import RAW_DATA_DIR

DATASET_URLS = {
    "resume_dataset.csv": "https://raw.githubusercontent.com/raghavendranhp/Resume_screening/master/UpdatedResumeDataSet.csv",
    "naukri_dataset.csv": "https://raw.githubusercontent.com/vikashjha2050/Job-recommendation-system/master/naukri_com-job_sample.csv",
    "linkedin_dataset.csv": "https://raw.githubusercontent.com/sherinrose2019k-ops/LinkedIn-Job-Postings-Analysis/main/linkedin_cleaned.csv"
}

def download_datasets():
    """
    Downloads raw datasets from public GitHub mirrors and saves them in data/raw.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Target directory for raw data: {RAW_DATA_DIR}")
    
    for filename, url in DATASET_URLS.items():
        dest_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.exists(dest_path):
            print(f"{filename} already exists at {dest_path}. Skipping download.")
            continue
            
        print(f"Downloading {filename} from {url}...")
        try:
            # Add User-Agent header to avoid HTTP 403 forbidden
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                with open(dest_path, "wb") as f:
                    f.write(response.read())
            print(f"Successfully downloaded {filename}.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise e

if __name__ == "__main__":
    download_datasets()
