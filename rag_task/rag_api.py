"""
RAG API - FastAPI service for question answering with RAG

TODO:
1. Implement vector similarity search
2. Build context from retrieved chunks
3. Generate grounded responses with LLM
4. Return answer with sources
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
import psycopg2
from typing import List, Dict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="RAG Knowledge Assistant")


class QueryRequest(BaseModel):
    query: str
    k: int = 3


class SourceDocument(BaseModel):
    doc_name: str
    chunk: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str


def generate_embedding(text: str) -> List[float]:
    """
    TODO: Generate embedding for query using OpenAI.
    
    Model: text-embedding-3-small
    """
    pass


def retrieve_chunks(query_embedding: List[float], k: int) -> List[Dict]:
    """
    TODO: Vector similarity search using pgvector.
    
    SQL Query:
    SELECT doc_name, chunk, embedding <-> %s::vector AS distance
    FROM documents
    ORDER BY distance
    LIMIT %s;
    
    Returns:
        List of dicts with doc_name, chunk, distance
    """
    pass


def build_prompt(chunks: List[str], query: str) -> str:
    """
    TODO: Construct prompt with context and instructions.
    
    Include:
    - Clear instructions to use only provided context
    - Numbered context chunks
    - User query
    - Instruction to say "I don't know" if answer not in context
    """
    pass


def generate_answer(prompt: str) -> str:
    """
    TODO: Generate answer using gpt-4o-mini.
    
    Settings:
    - temperature: 0.1 (factual responses)
    - max_tokens: 500
    """
    pass


@app.get("/")
def root():
    return {"message": "RAG API", "docs": "/docs"}


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    TODO: Main RAG endpoint.
    
    Steps:
    1. Generate query embedding
    2. Retrieve top-k similar chunks
    3. Build prompt with context
    4. Generate answer with LLM
    5. Format response with sources
    
    Handle errors:
    - No documents found
    - OpenAI API failures
    - Database errors
    """
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8001, reload=True)