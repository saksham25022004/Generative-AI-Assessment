# Extract to Excel - Implementation Guide

Extract structured candidate data from resume images using OpenAI Vision API.

## Task Overview

**Input:** `data/images/*` (resume images)  
**Output:** `output/extracted_output.xlsx`

---

## Requirements

### 1. Image to Text Extraction
- Read images from `data/images/`
- Use OpenAI Vision API (gpt-4o-mini) to analyze resume images
- Encode images to base64 before sending to API

### 2. Structured Data Extraction
Extract these fields as JSON:
- `candidate_name`
- `role`
- `date`
- `summary`
- `email`
- `phone`

Prompt the model to return ONLY valid JSON with null for missing fields.

### 3. Data Validation
- Validate email format using regex
- Validate phone number patterns
- Handle JSON parsing errors gracefully

### 4. Error Handling
- Continue processing if individual images fail
- Log errors with filename
- Add `document_name` field to track source

### 5. Excel Output
- Create DataFrame with all records
- Save to `output/extracted_output.xlsx`
- Include all extracted fields plus document name

---

## Testing

Run the script and verify Excel output contains structured data from all resume images.

---

## Expected Output Structure

| candidate_name | role | date | summary | email | phone | document_name |
|----------------|------|------|---------|-------|-------|---------------|