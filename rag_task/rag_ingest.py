"""
RAG Ingestion - Load PDFs into PostgreSQL with pgvector

TODO:
1. Extract text from PDFs in data/docs/
2. Chunk text with overlap
3. Generate embeddings using OpenAI
4. Store in PostgreSQL with pgvector
"""

import os
import glob
from dotenv import load_dotenv
import openai
import psycopg2

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    TODO: Extract text from PDF file.
    
    Hint: Use PyPDF2 or pdfplumber library
    """
    pass


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    TODO: Split text into overlapping chunks.
    
    Args:
        text: Full document text
        chunk_size: Target tokens per chunk
        overlap: Overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    pass


def generate_embedding(text: str) -> list:
    """
    TODO: Generate embedding using OpenAI.
    
    Model: text-embedding-3-small
    """
    pass


def store_chunks(doc_name: str, chunks: list, embeddings: list):
    """
    TODO: Store chunks and embeddings in PostgreSQL.
    
    Table schema:
    - id: SERIAL PRIMARY KEY
    - doc_name: TEXT
    - chunk: TEXT
    - embedding: VECTOR(1536)
    - metadata: JSONB
    """
    pass


def ingest_all_pdfs():
    """
    TODO: Process all PDFs in data/docs/ folder.
    
    Steps:
    1. Get all PDF files
    2. Extract text from each
    3. Chunk the text
    4. Generate embeddings
    5. Store in database
    """
    pass


if __name__ == "__main__":
    ingest_all_pdfs()