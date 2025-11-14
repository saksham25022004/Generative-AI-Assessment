"""
RAG API with PostgreSQL & pgvector
===================================
TODO LIST:
====================
This file provides the RAG API structure. You still need to:

1. COMPLETE DATABASE SETUP:
   - Install PostgreSQL locally (no Docker)
   - Create database: rag_knowledge_base
   - Enable pgvector extension
   - Create documents table with proper schema
   - Verify connection

2. CREATE & IMPROVE ingest_pg.py:
   - Implement token-aware chunking with overlap
   - Add batch embedding generation
   - Add retry logic for API rate limits
   - Include metadata (source_filename, chunk_index, etc.)
   - Add error handling and logging

3. PREPARE SAMPLE DATA:
   - Create data/docs/ folder
   - Add 5-10 sample .txt documents
   - Include various document types (product notes, technical specs, etc.)

4. IMPROVE THIS FILE (rag_api.py):
   - Review all TODO comments below
   - Strengthen the prompt template
   - Consider re-ranking strategies
   - Add filtering capabilities
   - Improve error messages

5. TEST & EVALUATE:
   - Run ingestion and verify data
   - Test with various queries
   - Check answer quality and grounding
   - Verify citation accuracy
   - Test edge cases (no answer, ambiguous queries)

6. DOCUMENT YOUR WORK:
   - Create README_RESULTS.md
   - Explain architecture decisions
   - Document improvements made
   - Include test results and evaluation
   - Note challenges and solutions

See inline TODO comments throughout this file for specific improvement areas.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import os
import openai
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# ============================================================================
# Configuration & Setup
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Environment configuration with fallback defaults
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
VECTOR_DIM = int(os.getenv('VECTOR_DIM', '1536'))  # text-embedding-3-small dimension
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')

# Validate critical environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in environment variables")

# Configure OpenAI client
openai.api_key = OPENAI_API_KEY

# Configure logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Initialization
# ============================================================================

app = FastAPI(
    title='RAG Knowledge Assistant API',
    description='Semantic search and question answering over internal documents',
    version='2.0.0',
    docs_url='/docs',
    redoc_url='/redoc'
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class QueryRequest(BaseModel):
    """
    Request schema for the /query endpoint.
    
    Attributes:
        query: The user's natural language question
        k: Number of top similar chunks to retrieve (default: 3)
        include_metadata: Whether to return chunk metadata in response
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User's question in natural language"
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include source metadata in response"
    )
    
    @validator('query')
    def query_must_not_be_empty(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()


class SourceDocument(BaseModel):
    """
    Schema for a retrieved source document.
    
    Attributes:
        doc_name: Original filename of the source document
        chunk: The text content of the retrieved chunk
        distance: Cosine distance (lower = more similar)
        relevance_score: Normalized similarity score (0-1, higher = more relevant)
        metadata: Additional document metadata (optional)
    """
    doc_name: str
    chunk: str
    distance: float
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """
    Response schema for the /query endpoint.
    
    Attributes:
        answer: The generated answer from the LLM
        sources: List of source documents used to generate the answer
        query: Echo of the original query
        model_used: The LLM model used for generation
    """
    answer: str
    sources: List[SourceDocument]
    query: str
    model_used: str


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    database_connected: bool
    openai_configured: bool
    vector_dimension: int


# ============================================================================
# Database Utilities
# ============================================================================

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Ensures proper connection cleanup even if errors occur.
    
    Usage:
        with get_db_connection() as conn:
            # Use connection
            ...
    
    Yields:
        psycopg2.connection: Active database connection
    
    Raises:
        HTTPException: If database connection fails
    """
    conn = None
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            cursor_factory=RealDictCursor  # Return results as dictionaries
        )
        logger.info("Database connection established")
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")


# ============================================================================
# OpenAI Integration with Retry Logic
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_embedding(text: str) -> List[float]:
    """
    Generate embeddings for text using OpenAI's embedding model.
    Includes automatic retry logic for transient failures.
    
    Args:
        text: Input text to embed
    
    Returns:
        List of floats representing the embedding vector
    
    Raises:
        HTTPException: If embedding generation fails after retries
    """
    try:
        logger.info(f"Generating embedding for query: {text[:50]}...")
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = response['data'][0]['embedding']
        logger.info(f"Embedding generated successfully (dim: {len(embedding)})")
        return embedding
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate embedding: {str(e)}"
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_llm_response(prompt: str, model: str = LLM_MODEL) -> str:
    """
    Generate a response using OpenAI's chat completion API.
    Includes retry logic for rate limits and transient errors.
    
    Args:
        prompt: The full prompt including context and question
        model: OpenAI model to use (default: gpt-4o-mini)
    
    Returns:
        Generated answer text
    
    Raises:
        HTTPException: If LLM generation fails after retries
    """
    try:
        logger.info(f"Generating LLM response using model: {model}")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful AI assistant specializing in answering questions '
                        'based on company documentation. You provide accurate, concise answers '
                        'and always ground your responses in the provided context.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=0.1,  # Low temperature for more factual responses
            max_tokens=500,
            top_p=0.9
        )
        answer = response['choices'][0]['message']['content']
        logger.info("LLM response generated successfully")
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI LLM error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate response: {str(e)}"
        )


# ============================================================================
# Core RAG Pipeline Functions
# ============================================================================

def retrieve_relevant_chunks(
    query_embedding: List[float],
    k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve the k most similar document chunks using vector similarity.
    Uses cosine distance operator (<->) for efficient nearest neighbor search.
    
    Args:
        query_embedding: Vector representation of the user's query
        k: Number of chunks to retrieve
    
    Returns:
        List of dictionaries containing doc_name, chunk, distance, and metadata
    
    Note:
        The pgvector extension uses the <-> operator for L2 distance.
        Lower distance = higher similarity.
    
    TODO - CANDIDATE TASKS:
        1. Consider implementing re-ranking with a cross-encoder model for better precision
        2. Add filtering by metadata (e.g., document type, date range)
        3. Implement hybrid search (combine vector + keyword search)
        4. Add query expansion or reformulation for better recall
        5. Consider implementing MMR (Maximal Marginal Relevance) for diversity
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Vector similarity search using pgvector
            # The <-> operator computes L2 (Euclidean) distance
            # For cosine similarity, use <=> operator if configured
            query = """
                SELECT 
                    doc_name,
                    chunk,
                    metadata,
                    embedding <-> %s::vector AS distance
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """
            
            logger.info(f"Executing vector similarity search (k={k})")
            cur.execute(query, (query_embedding, query_embedding, k))
            results = cur.fetchall()
            logger.info(f"Retrieved {len(results)} relevant chunks")
            
            # TODO: Keep distances for debugging and evaluation purposes
            # TODO: Experiment with different distance thresholds to filter low-quality matches
            
            return results


def calculate_relevance_score(distance: float, max_distance: float = 2.0) -> float:
    """
    Convert distance to a normalized relevance score (0-1).
    
    Args:
        distance: L2 distance from vector similarity search
        max_distance: Maximum expected distance for normalization
    
    Returns:
        Relevance score between 0 (not relevant) and 1 (highly relevant)
    """
    # Invert and normalize: closer distance = higher score
    score = max(0.0, 1.0 - (distance / max_distance))
    return round(score, 4)


def construct_prompt(context_chunks: List[str], user_query: str) -> str:
    """
    Construct an optimized prompt for the LLM that enforces grounded responses.
    
    Args:
        context_chunks: List of retrieved document chunks
        user_query: The original user question
    
    Returns:
        Formatted prompt string with instructions and context
    
    Design Principles:
        - Clear instructions to prevent hallucination
        - Numbered context for easier citation
        - Explicit fallback behavior for unknown answers
    
    TODO - CANDIDATE TASKS:
        1. Improve prompt clarity and specificity
        2. Add few-shot examples for better response quality
        3. Implement citation formatting (e.g., [source: filename.txt])
        4. Add constraints for answer length and format
        5. Test different prompt templates and compare results
        6. Consider role-specific prompts (e.g., technical vs. business queries)
    """
    # Number each context chunk for potential citation
    numbered_context = "\n\n".join([
        f"[Document {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    # TODO: CANDIDATE - Strengthen this prompt template
    # Consider adding:
    # - Specific formatting instructions
    # - Examples of good vs bad answers
    # - Citation requirements
    # - Tone and style guidelines
    prompt = f"""You are an AI assistant answering questions based on internal company documentation.

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context below
2. If the context contains the answer, provide a clear and concise response
3. If the context does NOT contain enough information to answer, respond with:
   "I don't have enough information in the available documents to answer this question."
4. You may cite sources by referencing [Document N] when helpful
5. Do not make assumptions or add information not present in the context

CONTEXT:
{numbered_context}

QUESTION: {user_query}

ANSWER:"""
    
    return prompt


# ============================================================================
# API Endpoints
# ============================================================================

@app.get('/', response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        'message': 'RAG Knowledge Assistant API',
        'version': '2.0.0',
        'docs': '/docs'
    }


@app.get('/health', response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system status.
    Checks database connectivity and OpenAI configuration.
    """
    db_connected = False
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                db_connected = True
    except Exception as e:
        logger.warning(f"Health check database error: {e}")
    
    return HealthResponse(
        status='healthy' if db_connected else 'degraded',
        database_connected=db_connected,
        openai_configured=bool(OPENAI_API_KEY),
        vector_dimension=VECTOR_DIM
    )


@app.post('/query', response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Main RAG endpoint: Retrieve relevant documents and generate an answer.
    
    Workflow:
        1. Generate embedding for user query
        2. Retrieve k most similar document chunks from PostgreSQL
        3. Construct a grounded prompt with retrieved context
        4. Generate answer using LLM
        5. Return answer with source attributions
    
    Args:
        request: QueryRequest containing the user's question and parameters
    
    Returns:
        QueryResponse with generated answer and source documents
    
    Raises:
        HTTPException: For various error conditions (400, 502, 503)
    
    TODO - CANDIDATE TASKS:
        1. Add query preprocessing (spell check, query expansion)
        2. Implement response caching for common questions
        3. Add conversation history support for follow-up questions
        4. Implement confidence scoring for answers
        5. Add A/B testing framework for different prompt strategies
        6. Consider streaming responses for better UX
        7. Add analytics/logging for query patterns
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Step 1: Generate query embedding
        query_embedding = generate_embedding(request.query)
        
        # Step 2: Retrieve relevant chunks using vector similarity
        # TODO: Consider adding pre-filtering by metadata or document type
        retrieved_docs = retrieve_relevant_chunks(query_embedding, k=request.k)
        
        if not retrieved_docs:
            logger.warning("No documents found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents available in the knowledge base"
            )
        
        # Step 3: Extract chunks and construct context
        # TODO: Consider implementing chunk re-ranking here
        context_chunks = [doc['chunk'] for doc in retrieved_docs]
        prompt = construct_prompt(context_chunks, request.query)
        
        # Step 4: Generate LLM response
        # TODO: Add response validation to check if answer is grounded
        answer = generate_llm_response(prompt, model=LLM_MODEL)
        
        # Step 5: Format source documents with relevance scores
        sources = [
            SourceDocument(
                doc_name=doc['doc_name'],
                chunk=doc['chunk'][:500] + '...' if len(doc['chunk']) > 500 else doc['chunk'],
                distance=float(doc['distance']),
                relevance_score=calculate_relevance_score(float(doc['distance'])),
                metadata=doc.get('metadata') if request.include_metadata else None
            )
            for doc in retrieved_docs
        ]
        
        logger.info("Query processed successfully")
        
        # TODO: Log query analytics (query, answer quality, latency, sources used)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            model_used=LLM_MODEL
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Application Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks.
    Log configuration and verify critical services.
    """
    logger.info("=" * 60)
    logger.info("RAG Knowledge Assistant API Starting")
    logger.info("=" * 60)
    logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Vector Dimension: {VECTOR_DIM}")
    logger.info(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configured'}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown tasks.
    """
    logger.info("RAG Knowledge Assistant API Shutting Down")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        'rag_api:app',
        host='0.0.0.0',
        port=8001,
        reload=True,  # Disable in production
        log_level='info'
    )