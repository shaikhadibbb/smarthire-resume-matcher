"""Resume parsing utilities.

Extracts text from PDF, DOCX, and TXT files, and parses candidate years of experience.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Union, Tuple
import pdfplumber
import docx

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_pdf(file_path: Path) -> str:
    """Extracts text from a PDF file using pdfplumber.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text.
    """
    text: str = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error("Error parsing PDF %s: %s", file_path, e)
    return text

def parse_docx(file_path: Path) -> str:
    """Extracts text from a DOCX file using python-docx.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted text.
    """
    text: str = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
    except Exception as e:
        logger.error("Error parsing DOCX %s: %s", file_path, e)
    return text

def parse_txt(file_path: Path) -> str:
    """Extracts text from a TXT file.

    Args:
        file_path: Path to the TXT file.

    Returns:
        Extracted text.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error("Error parsing TXT %s: %s", file_path, e)
        return ""

def extract_resume_text(file_path: Union[str, Path]) -> str:
    """Detects file type and extracts text from PDF, DOCX, or TXT.

    Args:
        file_path: Path to the resume file.

    Returns:
        Extracted resume text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file type is unsupported and fallback failed.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("Resume file not found at: %s", path)
        raise FileNotFoundError(f"Resume file not found at: {path}")
        
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        return parse_pdf(path)
    elif ext == ".docx":
        return parse_docx(path)
    elif ext in [".txt", ".text"]:
        return parse_txt(path)
    else:
        logger.warning("Unsupported file extension %s. Falling back to plain text parsing.", ext)
        try:
            return parse_txt(path)
        except Exception as e:
            raise ValueError(f"Unsupported file format {ext} and fallback failed: {e}")

def extract_experience_years(text: str, return_fresher: bool = False) -> Union[float, Tuple[float, bool]]:
    """Extracts candidate years of experience from text using regex.

    Supports explicit patterns ("5 years of experience") and year ranges (e.g. "2018 - present").
    Sums distinct ranges up to a max cap of 40 years.

    Args:
        text: The resume text.
        return_fresher: Whether to also return a boolean indicating fresher status.

    Returns:
        Total years of experience (float) or a tuple of (experience, is_fresher).
    """
    if not text:
        if return_fresher:
            return 0.0, False
        return 0.0

    # Normalize unicode hyphens/dashes to standard hyphen
    text_clean = re.sub(r"[\u2013\u2014]", "-", text.lower())
    current_year = 2026  # Anchored at 2026 based on local metadata year
    
    # Months pattern (both words and optional slash numbers like 05/ or 5/)
    months_words = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    month_prefix = rf"(?:(?:{months_words}|\d{{1,2}})[\s,/-]+)?"
    
    # 1. Look for date ranges (e.g. 2018 - present or 2015 - 2021)
    # Present-ended ranges (e.g. "2018 - present", "Jan 2018 to present", "05/2018 - present")
    present_pattern = rf"\b{month_prefix}(\d{{4}})\s*(?:-|to)\s*(?:present|current|now|active|202[4-6])\b"
    present_matches = re.findall(present_pattern, text_clean)
    
    # Range between two years (e.g. "2015 - 2019", "Jan 2015 to Dec 2019", "05/2015 - 08/2019")
    between_pattern = rf"\b{month_prefix}(\d{{4}})\s*(?:-|to)\s*{month_prefix}(\d{{4}})\b"
    between_matches = re.findall(between_pattern, text_clean)
    
    ranges_years = []
    
    for start_str in present_matches:
        try:
            start = int(start_str)
            if 1980 < start <= current_year:
                diff = current_year - start
                if 0 <= diff < 40:
                    ranges_years.append(diff)
        except ValueError:
            pass
            
    for start_str, end_str in between_matches:
        try:
            start = int(start_str)
            end = int(end_str)
            if 1980 < start <= current_year and 1980 < end <= current_year:
                diff = end - start
                if 0 <= diff < 40:
                    ranges_years.append(diff)
        except ValueError:
            pass
            
    total_years = 0.0
    if ranges_years:
        total_years = min(sum(ranges_years), 40.0)
    else:
        # 2. Look for explicit mentions of years of experience
        explicit_patterns = [
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?experience",
            r"(?:experience|exp)[\s:]*(\d+(?:\.\d+)?)\s*\+?\s*years?",
            r"\b(\d+(?:\.\d+)?)\s*(?:years?|yrs)\b"
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text_clean)
            if matches:
                try:
                    values = [float(m) for m in matches if float(m) < 40.0]
                    if values:
                        total_years = max(values)
                        break
                except ValueError:
                    continue
                    
    is_fresher = False
    if total_years == 0.0:
        fresher_keywords = ["fresher", "graduate", "student", "intern", "seeking opportunity", "looking for", "entry level", "0 years"]
        if any(kw in text_clean for kw in fresher_keywords):
            is_fresher = True
            
    if return_fresher:
        return total_years, is_fresher
    return total_years
