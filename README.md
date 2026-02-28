# Market Analyst Agent

## Setup

### 1) Create and activate local uv environment
```
uv venv
```

Activate:
```
.venv\Scripts\activate
```

### 2) Install dependencies
```
uv pip install -r requirements.txt
```

### 3) Configure environment variables
Create and fill `.env`:
```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
PGVECTOR_CONNECTION_STRING=postgresql+psycopg://user:password@host:5432/dbname
TAVILY_API_KEY=
```

### 4) Run API (optional)
```
uvicorn app.main:app --reload --app-dir backend
```

### 5) Run notebook
Open `notebooks/agent_playground.ipynb` and execute the cells.

### 6) Ingest fundamentals (PDF -> pgvector)
```
python scripts/ingest_fundamentals.py --pdf "data/raw/ACME_2023.pdf" --company "ACME" --year 2023
```
Ensure the target Postgres database has the `pgvector` extension installed:
```
CREATE EXTENSION IF NOT EXISTS vector;
```
