"""
===============================================================
Task 3 — UNSTRUCTURED TEXT → EXCEL (AI Extraction Pipeline)
===============================================================

This script:
    1) Reads unstructured text files from: data/unstructured/
    2) Uses an OpenAI LLM to extract structured fields in JSON format
    3) Writes a clean Excel file into: output/extracted_output.xlsx

--------------------------------------------------------------
TODO ITEMS
--------------------------------------------------------------
1. Improve LLM Prompt
   - Make JSON formatting strict
   - Force null for missing fields
   - Prevent comments or extra text

2. Add Better JSON Validation
   - Handle malformed responses
   - Strip extra characters safely
   - Implement second-pass repair logic

3. Add Field Validation
   - Validate email format
   - Validate phone number patterns
   - Validate date formats (optional)

4. Add Error Handling & Logging
   - Log failed files into errors.xlsx
   - Do not break the entire pipeline

5. Improve Excel Formatting
   - Auto-adjust column width
   - Add timestamped filenames
"""

import os
import glob
import json
import re
from dotenv import load_dotenv
import openai
import pandas as pd

# ------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# ------------------------------------------------------
# LLM Extraction Logic
# ------------------------------------------------------
def extract_fields(text: str) -> dict:
    """
    Extracts candidate information from unstructured text using an LLM.

    TODO for candidate:
    -------------------
    - Rewrite system message to enforce strict JSON
    - Add 'respond only with JSON, no explanation'
    - Add fallback regex extraction for name/email/phone
    """

    system_prompt = (
        "Extract the following fields and respond ONLY with a valid JSON object:\n"
        '{"candidate_name":"","role":"","date":"","summary":"","email":"","phone":""}\n\n'
        "Rules:\n"
        "- If a field is missing, return null\n"
        "- Reply ONLY with JSON, nothing else\n"
    )

    user_prompt = f"TEXT TO ANALYZE:\n{text}\n\nReturn JSON only."

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict JSON extraction machine."},
                {"role": "user", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = resp["choices"][0]["message"]["content"]

    except Exception:
        return {
            "candidate_name": None,
            "role": None,
            "date": None,
            "summary": None,
            "email": None,
            "phone": None,
            "error": "LLM_REQUEST_FAILED"
        }

    # ------------------------------------------------------
    # JSON Parsing — with Repair Logic
    # ------------------------------------------------------
    try:
        data = json.loads(content)

    except Exception:
        # Candidate TODO: More robust repair logic
        start = content.find("{")
        end = content.rfind("}")
        try:
            data = json.loads(content[start:end + 1])
        except Exception:
            data = {
                "candidate_name": None,
                "role": None,
                "date": None,
                "summary": None,
                "email": None,
                "phone": None,
                "error": "JSON_PARSE_FAIL"
            }

    # ------------------------------------------------------
    # Email Validation — Candidate can improve this
    # ------------------------------------------------------
    if "email" in data and data["email"]:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", data["email"]):
            data["email"] = None

    return data


# ------------------------------------------------------
# Processing Pipeline
# ------------------------------------------------------
def process_all():
    """
    Reads all .txt files, performs extraction, and writes to Excel.

    TODO for candidate:
    -------------------
    - Add support for multiple file formats (.pdf, .docx) via OCR or libraries
    - Add logging for failures
    - Add timestamp to filename
    """

    records = []
    files = glob.glob("data/unstructured/*")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()

        extracted = extract_fields(text)
        extracted["document_name"] = os.path.basename(fp)
        records.append(extracted)

    df = pd.DataFrame(records)

    os.makedirs("output", exist_ok=True)
    df.to_excel("output/extracted_output.xlsx", index=False)

    print("✔ Extraction Completed")
    print("→ Saved to output/extracted_output.xlsx")


# ------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------
if __name__ == "__main__":
    process_all()
