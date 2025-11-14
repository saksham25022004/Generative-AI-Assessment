# Quick Start

1. Install Python 3.9+ and create virtual environment:
   ```bash
   python3 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install PostgreSQL locally:
   - Windows: use the installer from https://www.postgresql.org/download/windows/

3. Create database and enable pgvector:
   ```sql
   -- run in psql or pgAdmin
   CREATE DATABASE assessment;
   \c assessment
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. Update `.env` from `.env.example` with your values:
   - `DATABASE_URL=postgresql://postgres:password@localhost:5432/assessment`
   - `OPENAI_API_KEY=sk-...`
   - `WEATHER_API_KEY=...`
   - `VECTOR_DIM=1536`

5. (Optional) Run quick verify:
   ```bash
   python verify_setup.py
   ```

6. RAG: ingest documents and run FastAPI
```bash
cd rag_task
python rag_ingest.py  
uvicorn app:app --reload --port 8001
```
Open: http://127.0.0.1:8001/docs

7. Agent:
```bash
cd agent_task
uvicorn agent_weather:app --reload --port 8002
```

8. Extraction:
```bash
cd extraction_task
python extract_to_excel.py
# output will be in extraction_task/output/extracted_output.xlsx
```