# Task 1 — RAG Workflow

## Story

Your AI team is building an internal Generative AI Knowledge Assistant that helps teams quickly search company documents — product notes, technical specs, onboarding guides, marketing strategies, etc.

To power this assistant, you need a reliable Retrieval-Augmented Generation (RAG) pipeline that uses:

* PostgreSQL with the `pgvector` extension
* OpenAI embeddings for semantic understanding
* FastAPI for serving natural-language answers
* LLM generation for contextual responses

This RAG system will allow teams to ask free-form questions and get grounded answers based only on internal documents — with zero hallucination.

---

## Context

Traditional LLMs guess answers when they don't know something. Your job is to prevent that by using a retrieval-first workflow:

1. Store document chunks + embeddings inside PostgreSQL
2. Embed the user question
3. Retrieve top relevant chunks using vector similarity
4. Generate a grounded response using GPT

You will implement this end-to-end during this task.

---

## What You Have

Inside your `rag_pg/` folder:

| File | Description |
|------|-------------|
| `ingest_pg.py` | Reads text files, chunks them, generates embeddings, stores records in PostgreSQL (`pgvector`) |
| `rag_api.py` | FastAPI endpoint that retrieves chunks + calls an LLM to answer questions |
| `data/docs/` | Sample documents for ingestion |
| `README_RESULTS.md` | Add your notes, improvements, architecture explanation |

---

## Your Mission

### 1. Install PostgreSQL

* **Windows**: [PostgreSQL Installer](https://www.postgresql.org/download/windows/)

Then enable extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create table:

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(1536)
);
```

---

### 2. Ingest and Embed Documents

Run ingestion:

```bash
cd rag_pg
python ingest_pg.py
```

**What the script does:**

* Reads each `.txt` file
* Splits into chunks
* Generates embeddings via `text-embedding-3-small`
* Inserts into Postgres with their metadata

**Your responsibilities:**

* Improve chunking (overlap, token-aware splitting)
* Add batching to embedding calls
* Add retry logic for rate limits
* Add file metadata (like `source_filename`)

After ingestion, verify:

```sql
SELECT COUNT(*) FROM documents;
```

---

### 3. Build the Retrieval & Generation API

Launch the API:

```bash
uvicorn rag_api:app --reload --port 8001
```

Visit:

```
http://127.0.0.1:8001/docs
```

**Workflow in `rag_api.py`:**

1. Embed the query
2. Run vector similarity search:

```sql
SELECT content, metadata 
FROM documents
ORDER BY embedding <-> query_embedding
LIMIT 4;
```

3. Construct context
4. Send everything into LLM
5. Return both answer + retrieved sources

---

### 4. Strengthen the Prompt

Your LLM must be prevented from hallucinating.

Use this template:

```
You are an assistant answering questions using ONLY the information in the provided context.
If the answer is not present in the context, reply: "I don't know."

Context:
{retrieved_chunks}

Question: {user_query}
```

**Your tasks:**

* Improve clarity
* Force grounded responses
* Format answer cleanly
* Optionally include citations:

```
[source: sample2.txt]
```

---

### 5. Test and Evaluate Your RAG

Try questions like:

* "What are the main product goals?"
* "How does the onboarding workflow work?"
* "Does the document mention pricing strategy?"

**Check:**

* Did the correct chunks return?
* Is the answer grounded?
* What happens when the context has no answer?

---

## Deliverables

By the end of this task, you should provide:

* Working RAG service using PostgreSQL + pgvector
* Improved `ingest_pg.py` (chunking, batching, metadata)
* Improved `rag_api.py` (prompt, retrieval quality)
* Detailed notes in `README_RESULTS.md`

---

## Architecture Diagram

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         FastAPI Server (rag_api.py)     │
│  ┌──────────────────────────────────┐   │
│  │ 1. Embed Query                   │   │
│  │ 2. Vector Similarity Search      │   │
│  │ 3. Construct Context             │   │
│  │ 4. LLM Generation                │   │
│  └──────────────────────────────────┘   │
└────────┬───────────────────┬────────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌──────────────────┐
│  PostgreSQL +   │  │   OpenAI API     │
│    pgvector     │  │ (Embeddings +    │
│                 │  │  GPT-4)          │
└─────────────────┘  └──────────────────┘
         ▲
         │
┌────────┴────────────────────────────────┐
│  Document Ingestion (ingest_pg.py)     │
│  ┌──────────────────────────────────┐   │
│  │ 1. Read Documents                │   │
│  │ 2. Chunk Text                    │   │
│  │ 3. Generate Embeddings (Batch)   │   │
│  │ 4. Store in PostgreSQL           │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
         ▲
         │
    ┌────┴────┐
    │ data/   │
    │ docs/   │
    └─────────┘
```

---

## Notes

* Use `text-embedding-3-small` for cost-effectiveness (1536 dimensions)
* Implement proper error handling and logging
* Consider adding caching for frequently asked questions
* Monitor API costs and implement rate limiting if needed