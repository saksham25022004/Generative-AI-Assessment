# Task 3 — Extract → Excel

## Story

The **Talent Acquisition team** receives hundreds of candidate summaries, reports, and interview notes every week — but most of them are **unstructured text files**.

They've asked your team to build a small **AI-powered document extraction tool** that can read these files, identify key details like candidate name, role, and contact info, and automatically write the results into an Excel sheet.

This will later be part of a larger internal HR analytics system that tracks hiring pipelines, sourcing metrics, and candidate trends.

Your task is to design and refine a simple **text-to-structure pipeline** using **OpenAI models**, and export the data cleanly using **Pandas + OpenPyXL**.

---

## What's Provided

Inside the folder `extraction_task/`, you'll find:

| File | Description |
|------|-------------|
| `extract_to_excel.py` | Starter script with TODOs to improve extraction and formatting |
| `data/unstructured/` | Contains raw sample candidate files (e.g., resumes or summaries) |
| `output/` | The folder where your final Excel file will be saved |
| `README_RESULTS.md` | Write your notes and improvement summary here |

---

## Your Mission (Step-by-Step)

### Understand the Current Script

Open `extract_to_excel.py` — it currently:

- Reads all `.txt` files from `data/unstructured/`
- Uses OpenAI to extract structured information
- Saves the results to `output/extracted_output.xlsx`

**Your job is to enhance this script** so that the extraction is more reliable, structured, and validated.

---

### Improve the Prompt Design

Right now, the system prompt inside `extract_fields()` looks like this:

```python
system = 'Extract the following fields and respond ONLY with a valid JSON object: {"candidate_name":"","role":"","date":"","summary":"","email":"","phone":""}'
```

This works, but it's too basic.

**Your task:**

- Make the instruction clearer and enforce JSON format strictly
- Ask the LLM to fill missing fields as `null` if not available
- Prevent free-form responses (no explanations or text outside JSON)

**Example improved instruction:**

```
"Extract candidate_name, role, date, summary, email, and phone.
Respond only in JSON. If data is missing, return null. No text outside JSON."
```

This ensures that even if the text has variations like:

```yaml
Candidate: Alice Singh | Role: Data Scientist
Summary: Python & ML specialist | Contact: alice@example.com
```

The model still extracts consistent keys.

---

### Add Validation Logic

Once you get the model's response, it's parsed with `json.loads()`. But sometimes the model may produce malformed JSON or partial text.

**Improve this by:**

- Trimming unwanted characters before parsing
- Handling `JSONDecodeError` gracefully
- Logging any failed file to the console instead of stopping the entire script
- Optionally, store failed extractions in a separate sheet
- **Bonus:** Add regex checks for email and phone to clean invalid values

---

### Enhance the Output Format

The output Excel file should look clean and easy to read:

| candidate_name | role | date | summary | email | phone | document_name |
|----------------|------|------|---------|-------|-------|---------------|

** Your task:**

- Ensure all fields are consistently lowercase in headers
- Make sure each document becomes one row
- Save the file to `extraction_task/output/extracted_output.xlsx`

**Optional improvements:**

- Auto-adjust Excel column widths using OpenPyXL
- Add a timestamp to the filename (e.g., `extracted_output_2025-11-13.xlsx`)

---

### Test the Script

Run:

```bash
cd extraction_task
python extract_to_excel.py
```

You should see logs like:

```bash
Extracting data/unstructured/
Wrote output/extracted_output.xlsx
```

Open the Excel file in `output/` and verify that each text file corresponds to one row.

---

###  Add More Sample Data (Optional)

If you want to test robustness, you can add new files inside `data/unstructured/`

Include variations — missing fields, reordered info, or messy formatting — and ensure your extraction still works well.

---

## Deliverables

At the end of this task, you should have:

- A working script that runs without errors
- Clean, structured data saved in Excel
- Improved prompt, validation, and error handling
- Notes about your changes in `README_RESULTS.md`

---

## Hints

- Use `try/except` when parsing LLM output to avoid breaking the flow
- Keep the prompt short and deterministic — "Respond only with JSON" is key
- You can test locally with `gpt-4o-mini` for fast results
- Even small improvements in cleaning can make the output much more consistent

---