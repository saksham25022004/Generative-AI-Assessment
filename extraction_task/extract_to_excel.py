"""
Resume Extraction Pipeline - Extract structured data from resume images

TODO:
1. Extract text/data from images in data/images/
2. Use OpenAI to structure the data into JSON format
3. Write results to output/extracted_resumes.xlsx

Fields to extract:
- candidate_name
- email
- phone
- education
- experience_summary
- skills
"""

import os
import glob
from dotenv import load_dotenv
import openai
import pandas as pd
import base64

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4o-mini"  # Vision-capable model

def extract_from_image(image_path: str) -> dict:
    """
    TODO: Extract structured data from resume image using OpenAI Vision API.
    
    Steps:
    1. Read image and encode to base64
    2. Send to OpenAI with gpt-4o-mini (vision-capable)
    3. Request structured JSON output
    4. Parse and return the data
    
    Returns:
        dict with candidate info or error
    """
    # YOUR CODE HERE
    pass


def process_all_resumes():
    """
    TODO: Process all resume images and create Excel output.
    
    Steps:
    1. Get all images from data/images/
    2. Extract data from each image
    3. Compile results into DataFrame
    4. Save to output/extracted_resumes.xlsx
    """
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    process_all_resumes()