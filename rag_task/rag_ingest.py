"""
PostgreSQL + pgvector Document Ingestion Pipeline
==================================================
TODO LIST:
====================
This script provides an improved ingestion pipeline. Tasks to complete:

1. SETUP:
   - Ensure PostgreSQL is running
   - Verify pgvector extension is enabled
   - Create .env file with OPENAI_API_KEY and DATABASE_URL
   - Create data/docs/ folder with sample .txt files

2. IMPROVEMENTS TO MAKE:
   - Test different chunk sizes and overlap values
   - Experiment with different splitting strategies
   - Add support for other file formats (PDF, DOCX, etc.)
   - Implement deduplication logic
   - Add incremental ingestion (skip already processed files)

3. METADATA ENHANCEMENTS:
   - Add document category/tags
   - Include author information
   - Track version numbers
   - Add last_updated timestamps

4. VALIDATION:
   - Run ingestion: python ingest_pg.py
   - Check database: SELECT COUNT(*), COUNT(DISTINCT doc_name) FROM documents;
   - Verify embeddings: SELECT doc_name, embedding FROM documents LIMIT 1;
   - Test retrieval quality with sample queries

5. DOCUMENT RESULTS:
   - Note ingestion statistics (time, chunks, errors)
   - Document any issues encountered
   - Add findings to README_RESULTS.md
"""

import os
import glob
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging

from dotenv import load_dotenv
import openai
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

# ============================================================================
# Configuration & Setup
# ============================================================================

# Load environment variables
load_dotenv()

# Environment configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))  # Embeddings per API call
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Overlapping tokens

# Validate critical environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env file")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL must be set in .env file")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize tokenizer for token-aware chunking
try:
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
except KeyError:
    # Fallback to cl100k_base if model-specific encoding not found
    tokenizer = tiktoken.get_encoding("cl100k_base")

# ============================================================================
# Database Utilities
# ============================================================================

def get_db_connection() -> psycopg2.extensions.connection:
    """
    Establish connection to PostgreSQL database.
    
    Returns:
        Active database connection
    
    Raises:
        psycopg2.Error: If connection fails
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        logger.info("✓ Database connection established")
        return conn
    except psycopg2.Error as e:
        logger.error(f"✗ Database connection failed: {e}")
        raise


def ensure_table_exists(conn: psycopg2.extensions.connection) -> None:
    """
    Create documents table if it doesn't exist.
    Includes proper schema with metadata support.
    
    Args:
        conn: Active database connection
    
    TODO - CANDIDATE TASKS:
        1. Add indexes for frequently queried metadata fields
        2. Consider partitioning for large datasets
        3. Add full-text search index for keyword search
        4. Implement table versioning for schema migrations
    """
    with conn.cursor() as cur:
        # Create table with comprehensive schema
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                doc_name TEXT NOT NULL,
                chunk TEXT NOT NULL,
                chunk_index INTEGER,
                total_chunks INTEGER,
                metadata JSONB,
                embedding VECTOR({VECTOR_DIM}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for faster vector similarity search
        # Using IVFFlat for efficient approximate nearest neighbor search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        # Create index for document lookups
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_doc_name_idx 
            ON documents(doc_name);
        """)
        
        # Create GIN index for metadata queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_metadata_idx 
            ON documents 
            USING gin(metadata);
        """)
        
        conn.commit()
        logger.info("✓ Table 'documents' verified/created with indexes")


def check_existing_document(conn: psycopg2.extensions.connection, doc_name: str) -> bool:
    """
    Check if a document has already been ingested.
    
    Args:
        conn: Active database connection
        doc_name: Name of the document to check
    
    Returns:
        True if document exists, False otherwise
    
    TODO - CANDIDATE: Implement incremental ingestion
        - Compare file modification times
        - Re-ingest only if source file changed
        - Add version tracking
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents WHERE doc_name = %s", (doc_name,))
        count = cur.fetchone()[0]
        return count > 0


# ============================================================================
# Text Chunking with Token Awareness
# ============================================================================

def chunk_text_token_aware(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks using token-aware splitting.
    This ensures chunks respect token limits for embedding models.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
    
    Returns:
        List of text chunks
    
    Strategy:
        1. Encode text into tokens
        2. Create sliding windows with overlap
        3. Decode tokens back to text
        4. This preserves word boundaries better than character splitting
    
    TODO - CANDIDATE IMPROVEMENTS:
        1. Implement semantic chunking (split at sentence/paragraph boundaries)
        2. Add recursive splitting for very long documents
        3. Preserve special formatting (headers, lists, code blocks)
        4. Add chunk metadata (position, hierarchy level)
    """
    # Encode text to tokens
    tokens = tokenizer.encode(text)
    
    # Handle edge case: text shorter than chunk_size
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Calculate end position
        end = start + chunk_size
        
        # Extract chunk tokens
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Add to chunks list
        chunks.append(chunk_text.strip())
        
        # Move start position (with overlap)
        start += (chunk_size - overlap)
    
    logger.info(f"  → Split into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
    return chunks


def chunk_text_semantic(text: str, max_tokens: int = CHUNK_SIZE) -> List[str]:
    """
    Split text at natural boundaries (paragraphs, sentences) while respecting token limits.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
    
    Returns:
        List of semantically coherent chunks
    
    TODO - CANDIDATE: This is a more advanced chunking strategy
        Implement this for better chunk quality:
        1. Split by paragraphs first
        2. If paragraph too long, split by sentences
        3. If sentence too long, use token-aware splitting
        4. Combine small chunks to reach optimal size
    """
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = len(tokenizer.encode(para))
        
        # If paragraph alone exceeds max, split it
        if para_tokens > max_tokens:
            # Save current chunk if exists
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Use token-aware splitting for oversized paragraph
            para_chunks = chunk_text_token_aware(para, max_tokens, CHUNK_OVERLAP)
            chunks.extend(para_chunks)
        
        # If adding paragraph would exceed limit, save current chunk
        elif current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        
        # Otherwise, add to current chunk
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]


# ============================================================================
# Embedding Generation with Batching & Retry Logic
# ============================================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(openai.error.RateLimitError),
    reraise=True
)
def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    Includes automatic retry logic for rate limits.
    
    Args:
        texts: List of text chunks to embed
    
    Returns:
        List of embedding vectors
    
    Features:
        - Automatic batching for efficiency
        - Exponential backoff for rate limits
        - Progress logging
    
    TODO - CANDIDATE IMPROVEMENTS:
        1. Add embedding caching to avoid re-embedding same text
        2. Implement parallel processing for multiple batches
        3. Add cost tracking and estimation
        4. Consider using smaller/faster models for non-critical documents
    """
    if not texts:
        return []
    
    try:
        logger.info(f"  → Generating embeddings for {len(texts)} chunks...")
        
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        
        # Extract embeddings from response
        embeddings = [item['embedding'] for item in response['data']]
        
        logger.info(f"  ✓ Generated {len(embeddings)} embeddings")
        return embeddings
        
    except openai.error.RateLimitError as e:
        logger.warning(f"  ⚠ Rate limit hit, retrying... ({e})")
        raise  # Will be caught by retry decorator
    except openai.error.OpenAIError as e:
        logger.error(f"  ✗ OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"  ✗ Unexpected error generating embeddings: {e}")
        raise


def process_embeddings_in_batches(
    texts: List[str],
    batch_size: int = BATCH_SIZE
) -> List[List[float]]:
    """
    Process large number of texts in batches to respect API limits.
    
    Args:
        texts: List of all text chunks
        batch_size: Number of texts per API call
    
    Returns:
        List of all embeddings
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch = texts[i:i + batch_size]
        
        logger.info(f"  Processing batch {batch_num}/{total_batches}")
        embeddings = generate_embeddings_batch(batch)
        all_embeddings.extend(embeddings)
        
        # Rate limiting: brief pause between batches
        if i + batch_size < len(texts):
            time.sleep(0.5)
    
    return all_embeddings


# ============================================================================
# Document Ingestion Pipeline
# ============================================================================

def extract_metadata_from_file(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from file path and system attributes.
    
    Args:
        file_path: Path to the source file
    
    Returns:
        Dictionary of metadata
    
    TODO - CANDIDATE: Enhance metadata extraction
        1. Parse document headers for title, author, date
        2. Detect document type/category
        3. Extract keywords or tags
        4. Add custom metadata from filename patterns
    """
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "source_file": path.name,
        "file_path": str(path.absolute()),
        "file_size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "ingested_at": datetime.now().isoformat(),
        # TODO: Add document type detection
        # TODO: Add category/tag extraction
        # TODO: Add language detection
    }


def ingest_document(
    file_path: str,
    conn: psycopg2.extensions.connection,
    force_reingest: bool = False
) -> Tuple[int, int]:
    """
    Ingest a single document into the database.
    
    Args:
        file_path: Path to the text file
        conn: Active database connection
        force_reingest: If True, re-ingest even if document exists
    
    Returns:
        Tuple of (chunks_processed, chunks_inserted)
    
    Workflow:
        1. Read file content
        2. Check if already ingested (skip if exists)
        3. Chunk text intelligently
        4. Generate embeddings in batches
        5. Insert into database with metadata
    
    TODO - CANDIDATE ENHANCEMENTS:
        1. Add support for multiple file formats (PDF, DOCX, MD)
        2. Implement text cleaning/preprocessing
        3. Add deduplication logic
        4. Track ingestion statistics
    """
    doc_name = os.path.basename(file_path)
    
    # Check if document already exists
    if not force_reingest and check_existing_document(conn, doc_name):
        logger.info(f"⊘ Skipping {doc_name} (already ingested)")
        return (0, 0)
    
    try:
        # Read file content
        logger.info(f"Processing: {doc_name}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"Skipping {doc_name} (empty file)")
            return (0, 0)
        
        # Extract metadata
        base_metadata = extract_metadata_from_file(file_path)
        
        # Chunk text (use semantic chunking for better quality)
        chunks = chunk_text_semantic(content, CHUNK_SIZE)
        total_chunks = len(chunks)
        
        if not chunks:
            logger.warning(f"No chunks created for {doc_name}")
            return (0, 0)
        
        # Generate embeddings in batches
        embeddings = process_embeddings_in_batches(chunks, BATCH_SIZE)
        
        # Prepare data for insertion
        rows = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create chunk-specific metadata
            chunk_metadata = {
                **base_metadata,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "token_count": len(tokenizer.encode(chunk))
            }
            
            rows.append((
                doc_name,
                chunk,
                idx,
                total_chunks,
                json.dumps(chunk_metadata),
                embedding
            ))
        
        # Batch insert into database
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO documents 
                (doc_name, chunk, chunk_index, total_chunks, metadata, embedding) 
                VALUES %s
                """,
                rows,
                page_size=100
            )
            conn.commit()
        
        logger.info(f"  ✓ Ingested {doc_name}: {len(chunks)} chunks")
        return (len(chunks), len(chunks))
        
    except Exception as e:
        logger.error(f"  ✗ Error ingesting {doc_name}: {e}", exc_info=True)
        conn.rollback()
        return (len(chunks) if 'chunks' in locals() else 0, 0)


# ============================================================================
# Main Ingestion Orchestration
# ============================================================================

def ingest_all_documents(
    docs_dir: str = 'data/docs',
    pattern: str = '*.txt',
    force_reingest: bool = False
) -> Dict[str, Any]:
    """
    Ingest all documents from the specified directory.
    
    Args:
        docs_dir: Directory containing documents
        pattern: File pattern to match (e.g., '*.txt', '*.md')
        force_reingest: If True, re-ingest all documents
    
    Returns:
        Dictionary with ingestion statistics
    
    TODO - CANDIDATE: Add more features
        1. Support multiple file patterns
        2. Recursive directory traversal
        3. Parallel processing for multiple files
        4. Resume capability for interrupted ingestions
        5. Dry-run mode to preview changes
    """
    logger.info("=" * 70)
    logger.info("Starting Document Ingestion Pipeline")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Find all matching files
    search_pattern = os.path.join(docs_dir, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        logger.warning(f"No files found matching: {search_pattern}")
        return {
            "status": "no_files",
            "files_found": 0,
            "files_processed": 0,
            "total_chunks": 0,
            "duration_seconds": 0
        }
    
    logger.info(f"Found {len(files)} files to process")
    logger.info("")
    
    # Connect to database
    conn = get_db_connection()
    ensure_table_exists(conn)
    
    # Process each file
    stats = {
        "files_found": len(files),
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "total_chunks": 0,
        "chunks_inserted": 0,
        "errors": []
    }
    
    for file_path in files:
        try:
            chunks_processed, chunks_inserted = ingest_document(
                file_path,
                conn,
                force_reingest
            )
            
            if chunks_inserted > 0:
                stats["files_processed"] += 1
                stats["total_chunks"] += chunks_processed
                stats["chunks_inserted"] += chunks_inserted
            else:
                stats["files_skipped"] += 1
                
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{os.path.basename(file_path)}: {str(e)}")
            logger.error(f"Failed to process {file_path}: {e}")
    
    # Close database connection
    conn.close()
    
    # Calculate duration
    duration = time.time() - start_time
    stats["duration_seconds"] = round(duration, 2)
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Ingestion Summary")
    logger.info("=" * 70)
    logger.info(f"Files Found:      {stats['files_found']}")
    logger.info(f"Files Processed:  {stats['files_processed']}")
    logger.info(f"Files Skipped:    {stats['files_skipped']}")
    logger.info(f"Files Failed:     {stats['files_failed']}")
    logger.info(f"Total Chunks:     {stats['total_chunks']}")
    logger.info(f"Chunks Inserted:  {stats['chunks_inserted']}")
    logger.info(f"Duration:         {stats['duration_seconds']}s")
    
    if stats['errors']:
        logger.warning(f"\n Errors encountered:")
        for error in stats['errors']:
            logger.warning(f"  - {error}")
    
    logger.info("=" * 70)
    logger.info("Ingestion Complete!")
    logger.info("=" * 70)
    
    return stats


def verify_ingestion(conn: Optional[psycopg2.extensions.connection] = None):
    """
    Verify ingestion results and display statistics.
    
    TODO - CANDIDATE: Add more verification checks
        1. Verify embedding dimensions
        2. Check for duplicate chunks
        3. Validate metadata completeness
        4. Test sample queries
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total chunks
            cur.execute("SELECT COUNT(*) as count FROM documents")
            total_chunks = cur.fetchone()['count']
            
            # Unique documents
            cur.execute("SELECT COUNT(DISTINCT doc_name) as count FROM documents")
            unique_docs = cur.fetchone()['count']
            
            # Sample document
            cur.execute("""
                SELECT doc_name, COUNT(*) as chunk_count 
                FROM documents 
                GROUP BY doc_name 
                ORDER BY chunk_count DESC 
                LIMIT 5
            """)
            top_docs = cur.fetchall()
            
            logger.info("\n Database Verification:")
            logger.info(f"  Total Chunks:      {total_chunks}")
            logger.info(f"  Unique Documents:  {unique_docs}")
            logger.info(f"\n  Top Documents by Chunk Count:")
            for doc in top_docs:
                logger.info(f"    - {doc['doc_name']}: {doc['chunk_count']} chunks")
    
    finally:
        if should_close:
            conn.close()


# ============================================================================
# Command-Line Interface
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ingest documents into PostgreSQL with pgvector'
    )
    parser.add_argument(
        '--dir',
        default='data/docs',
        help='Directory containing documents (default: data/docs)'
    )
    parser.add_argument(
        '--pattern',
        default='*.txt',
        help='File pattern to match (default: *.txt)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-ingestion of existing documents'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data, skip ingestion'
    )
    
    args = parser.parse_args()
    
    try:
        if args.verify_only:
            verify_ingestion()
        else:
            stats = ingest_all_documents(
                docs_dir=args.dir,
                pattern=args.pattern,
                force_reingest=args.force
            )
            
            # Verify after ingestion
            if stats['chunks_inserted'] > 0:
                verify_ingestion()
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Fatal error: {e}", exc_info=True)
        exit(1)