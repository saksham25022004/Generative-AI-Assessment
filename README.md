# 🧠 Generative AI Engineer Assessment

Welcome to the **Generative AI Engineer Assessment**.  
This hands-on challenge evaluates your ability to build, optimize, and reason about **real-world Generative AI systems** using **LLMs**, **Vector Databases**, and **APIs**.

---

## 📦 Overview

You’ll work across three independent yet connected tasks that simulate real Generative AI engineering challenges:

| Task | Name | Focus Area | Description |
|------|------|-------------|--------------|
| **Task 1** | **RAG Workflow** | LLMs + Vector Search | Build a Retrieval-Augmented Generation pipeline using **Postgres** and **OpenAI embeddings** to answer questions from company documents. |
| **Task 2** | **Weather Agent** | LLM Reasoning + API Calls | Create an AI-powered FastAPI agent that understands natural language, extracts city names, and fetches real-time weather using an external API. |
| **Task 3** | **Extract → Excel** | Information Extraction | Use an LLM to read unstructured text and output structured candidate data into a formatted Excel file. |


---

## 🧠 What You’ll Learn & Demonstrate

By completing this assessment, you’ll showcase your ability to:
- Apply **Retrieval-Augmented Generation (RAG)** principles effectively.  
- Integrate **LLMs** with APIs and external data pipelines.  
- Implement robust **data ingestion, parsing, and validation** workflows.  
- Write clean, modular, and production-friendly Python/FastAPI code.  
- Think like an **AI Engineer** — building solutions, not demos.

---

## ⚙️ Tech Stack

- **Language:** Python 3.9+  
- **Frameworks:** FastAPI, Pandas  
- **LLM API:** OpenAI GPT-4o-mini, text-embedding-3-small  
- **Vector DB:** Postgres  
- **Utilities:** dotenv, tqdm, openpyxl, requests  

---

## 🚀 How to Start

1. Begin with [`INSTRUCTIONS.md`](./INSTRUCTIONS.md) for setup and environment details.  
2. Complete the tasks **in order (1 → 2 → 3)** to simulate a real GenAI workflow:
   - Start with RAG (data retrieval + response generation)
   - Move to the Agent (reasoning + API integration)
   - Finish with Extraction (structured output generation)
3. Document your improvements in each task’s `README_RESULTS.md`.

---

## 🏁 Final Submission

When you finish:
- Ensure all three tasks run locally without errors.
- Verify all required files (`.py`, `.md`, `output/`, etc.) are present.
- Add your notes and reasoning in each task’s result file.

---

> 💡 **Tip:** Treat this as a real engineering sprint. Each task builds practical AI engineering depth — from working with embeddings and vector search to integrating APIs and automating data processing.