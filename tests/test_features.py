"""Unit tests for the feature engineering and parsing modules.

Validates text cleaning, case-insensitive skill extraction, experience extraction, 
and edge case safety (e.g. empty strings).
"""

import os
import sys
import unittest
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess import clean_text
from src.features.match_features import extract_skills, calculate_skill_overlap
from src.parsing.resume_parser import extract_experience_years

class TestSmartHireFeatures(unittest.TestCase):
    """Test suite validating text processing and extraction functions."""

    def test_clean_text_noise_removal(self) -> None:
        """Tests that text cleaning removes URLs, emails, and formatting noise."""
        raw_text = "Check out http://github.com/profile or mail me at hello@test.com! Python 3.9 & C++."
        cleaned = clean_text(raw_text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("hello", cleaned)
        self.assertIn("python", cleaned)
        self.assertIn("c++", cleaned)
        
    def test_skill_extraction_case_insensitivity(self) -> None:
        """Tests that skill extraction is case-insensitive."""
        text_caps = "Experienced in PYTHON, SQL, and TABLEAU."
        skills = extract_skills(text_caps)
        self.assertIn("python", skills)
        self.assertIn("sql", skills)
        self.assertIn("tableau", skills)
        
    def test_skill_extraction_boundaries(self) -> None:
        """Tests that word boundaries prevent partial matching (e.g. Java matching JavaScript)."""
        text_java = "I write javascript code."
        skills = extract_skills(text_java)
        self.assertIn("javascript", skills)
        self.assertNotIn("java", skills)  # Word boundaries must prevent "java" from matching "javascript"

    def test_skill_extraction_samples(self) -> None:
        """Tests skill extraction on 5 different sample resume texts."""
        samples = [
            "Data scientist with machine learning, python, and pytorch experience.",
            "Web developer skilled in react, javascript, node.js, and tailwind css.",
            "Devops engineer with aws, docker, kubernetes, and jenkins skills.",
            "Cybersecurity analyst skilled in penetration testing, threat modeling, and cryptography.",
            "Project manager with agile, scrum, and business analysis certifications."
        ]
        
        expected_skills = [
            ["python", "machine learning", "pytorch"],
            ["react", "javascript", "node.js", "tailwind css"],
            ["aws", "docker", "kubernetes", "jenkins"],
            ["penetration testing", "threat modeling", "cryptography"],
            ["agile", "scrum", "business analysis"]
        ]
        
        for i, text in enumerate(samples):
            skills = extract_skills(text)
            for exp_s in expected_skills[i]:
                self.assertIn(exp_s, skills)

    def test_experience_extraction_formats(self) -> None:
        """Tests experience parser on explicit, range, and shorthand formats."""
        test_cases = {
            "I have 5 years of experience in coding.": 5.0,
            "Exp: 3+ years in front end development.": 3.0,
            "Software engineer from 2018 - 2022.": 4.0,
            "Software engineer from 2018 – 2022.": 4.0,  # en-dash
            "Software engineer from 2018 — 2022.": 4.0,  # em-dash
            "Software engineer from 2018 to 2022.": 4.0,  # to
            "Software engineer from 05/2018 - 08/2022.": 4.0,  # slash months
            "Research assistant from jan 2015 - present.": 11.0,  # 2026 - 2015 = 11 yrs
            "No prior industrial experience, fresh graduate.": 0.0
        }
        
        for text, exp_val in test_cases.items():
            parsed_val = extract_experience_years(text)
            self.assertEqual(parsed_val, exp_val)

    def test_empty_resume_handling(self) -> None:
        """Tests that empty resume text does not cause crashes and returns empty lists/values."""
        skills = extract_skills("")
        self.assertEqual(skills, [])
        
        exp = extract_experience_years("")
        self.assertEqual(exp, 0.0)
        
        cleaned = clean_text("")
        self.assertEqual(cleaned, "")

if __name__ == "__main__":
    unittest.main()
