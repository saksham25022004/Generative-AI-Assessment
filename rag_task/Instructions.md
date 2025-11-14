# RAG System - Implementation Guide

Build a Retrieval-Augmented Generation system using PostgreSQL + pgvector + OpenAI.

---

## Setup

### 1. Database Configuration
```sql
-- Create database
CREATE DATABASE rag_knowledge_base;

-- Enable pgvector
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    doc_name TEXT,
    chunk TEXT,
    embedding VECTOR(1536),
    metadata JSONB
);
```

### 2. Environment Variables (.env)
```
OPENAI_API_KEY=your_key
DATABASE_URL=postgresql://user:pass@localhost/rag_knowledge_base
```

---

## Part 1: Document Ingestion (rag_ingest.py)

### Requirements

**PDF Text Extraction**
- Use PyPDF2 or pdfplumber
- Extract text from all PDFs in `data/docs/`

**Text Chunking**
- Chunk size: ~500 tokens
- Overlap: 50 tokens between chunks
- Maintain sentence boundaries

**Embedding Generation**
- Model: `text-embedding-3-small`
- Dimension: 1536
- Batch API calls for efficiency
- Add retry logic for rate limits

**Database Storage**
- Insert: doc_name, chunk, embedding, metadata
- Metadata: `{"chunk_index": N, "total_chunks": M}`
- Handle errors without stopping pipeline

### Run
```bash
python rag_ingest.py
```

---

## Part 2: RAG API (rag_api.py)

### `/query` Endpoint Implementation

**1. Query Embedding**
```python
query_embedding = generate_embedding(user_query)
```

**2. Vector Similarity Search**
```sql
SELECT doc_name, chunk, embedding <-> %s::vector AS distance
FROM documents
ORDER BY distance
LIMIT k;
```

**3. Context Construction**
```python
context = "\n\n".join([f"[Document {i}]\n{chunk}" for i, chunk in enumerate(chunks)])
```

**4. LLM Prompt**
```
You are an assistant answering questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}
```

**5. Response Format**
```json
{
  "answer": "...",
  "sources": [
    {"doc_name": "file.pdf", "chunk": "...", "relevance_score": 0.85}
  ],
  "query": "..."
}
```

### Run
```bash
uvicorn rag_api:app --port 8001
```

---

## Testing

**Test Ingestion**
```sql
SELECT COUNT(*) FROM documents;
SELECT doc_name, COUNT(*) FROM documents GROUP BY doc_name;
```

**Test API**
- Visit: `http://localhost:8001/docs`
- Try queries related to PDF content
- Verify sources are relevant
- Check answer grounding

**Test Cases**
- Query with answer in docs
- Query with no answer (should say "don't know")
- Multi-document queries
- Edge cases (empty query, etc.)

---

## Key Points

- Ensure embedding dimensions match (1536)
- Use parameterized queries for SQL injection safety
- Log errors without breaking pipeline
- Validate PDF extraction quality before chunking
- Test vector search returns relevant chunks