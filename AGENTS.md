# AGENTS.md

## Project Overview
This repository implements a **Market Analyst Agent** backend (FastAPI + LangChain) with four analysis paths:
- News/web-search agent
- Technical chart agent
- Fundamental (RAG over filings) agent
- Supervisor agent that orchestrates all three
- Next.js frontend dashboard with tabbed UI, ingestion controls, and live streaming event viewer

Primary backend entrypoint:
- `backend/app/main.py`

## Runtime and Python Executable (uv venv)
Use the UV-managed virtual environment Python directly:
- Relative path: `.venv\Scripts\python.exe`
- Absolute path (this workspace): `c:\Users\rushi\OneDrive - ImmersiLearn Education Services LLP\Projects\LLM Projects\market-analyst-agent\.venv\Scripts\python.exe`

Common commands:
```powershell
# Create venv
uv venv

# Install deps
uv pip install -r requirements.txt

# Verify interpreter
.\.venv\Scripts\python.exe --version

# Run API
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

## Environment Variables
Configured via `.env` (loaded from repo root/cwd in `backend/app/main.py`).

Core vars used across the codebase:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION` (default `2024-02-01`)
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDINGS_ENDPOINT` (optional override)
- `AZURE_OPENAI_EMBEDDINGS_API_KEY` (optional override)
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_KEY`
- `PGVECTOR_CONNECTION_STRING`
- `TAVILY_API_KEY`

## Code Structure

### API Layer
File: `backend/app/main.py`

Responsibilities:
- Creates FastAPI app
- Defines strict request/response schemas (Pydantic)
- Exposes agent endpoints
- Normalizes/validates inputs before calling agents
- Enables CORS for local frontend origins (`localhost:3000`, `127.0.0.1:3000`)
- Serves generated technical chart files from `/static` (mapped to local `data/`)
- Exposes NDJSON streaming endpoints for live agent execution updates

Endpoints:
- `GET /health`
- `POST /agents/web-search`
- `POST /agents/web-search/stream`
- `POST /agents/technical`
- `POST /agents/technical/stream`
- `POST /agents/fundamental`
- `POST /agents/fundamental/stream`
- `POST /agents/supervisor`
- `POST /agents/supervisor/stream`
- `POST /agents/ingest` (multipart form with PDF upload)
- `GET /documents/ingested`
- `DELETE /documents/ingested`

### News Agent
File: `backend/app/agents/news/web_search_agent.py`

Responsibilities:
- Builds LangChain agent with Tavily search tool
- Uses Azure OpenAI chat model
- Returns a callable agent for invocation

### Technical Agent
File: `backend/app/agents/technical/technical_chart_agent.py`

Responsibilities:
- Downloads OHLCV data via `yfinance`
- Computes indicators (`MACD`, `RSI`) via `pandas_ta`
- Plots candlestick + indicators via `mplfinance`
- Uses vision-capable LLM analysis of generated chart image
- Forces headless matplotlib backend (`Agg`) to avoid Tkinter/main-thread runtime issues in server mode

Output (`TechnicalAnalysisResult`):
- `symbol`
- `image_path`
- `summary`
- `latest_values` (`close`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`)

### Fundamental Agent
File: `backend/app/agents/fundamental/fundamental_agent.py`

Responsibilities:
- Builds PGVector-backed retriever tool (`company_retriever`)
- Creates LangChain tool-calling agent (`create_agent`)
- Supports modes:
  - `general` (structured fundamental summary)
  - `qa` (question-specific answer)
  - `auto` (selects between general/qa)
- Tracks and returns source metadata for retrieved chunks
- Returns source metadata including `ticker`
- Supports streaming execution via `stream_fundamentals(...)`

Output (`FundamentalAnalysisResult`):
- `mode`
- `company`
- `answer`
- `sources`

### Supervisor Agent
File: `backend/app/agents/supervisor/supervisor_agent.py`

Responsibilities:
- Builds supervisor agent with 3 tools:
  - `technical_subagent`
  - `fundamental_subagent`
  - `news_subagent`
- Enforces structured JSON synthesis prompt
- Parses/normalizes synthesis JSON
- Has deterministic fallback if any tool call is skipped
- Supports streaming execution via `stream_market_supervised(...)`

Output (`SupervisorAnalysisResult`):
- `symbol`
- `company`
- `technical`
- `fundamental`
- `news`
- `synthesis`

### Document Ingestion Service
File: `backend/app/services/document_ingestion.py`

Responsibilities:
- Extracts Markdown from PDF via Azure Document Intelligence
- Splits Markdown into hierarchical chunks
- Adds metadata (`company`, `ticker`, `doc_type`, `year`, `source_path`, `chunk_index`)
- Stores chunks in PGVector using Azure embeddings

Output (`IngestionResult`):
- `company`
- `ticker`
- `source_path`
- `chunks_stored`
- `collection_name`
- `markdown_path`

### Vector Document Registry Service
File: `backend/app/services/vector_document_registry.py`

Responsibilities:
- Lists ingested document groups from PGVector tables (`langchain_pg_collection`, `langchain_pg_embedding`)
- Returns aggregated document records with metadata and chunk counts
- Deletes document chunks by (`collection_name`, `source_path`)

## Agent Invocation Contract
For LangChain `create_agent` calls, use message-based invocation:
```python
{"messages": [HumanMessage(content="...")]}
```

At API level:
- `/agents/web-search` request payload is:
```json
{"messages": ["your query"]}
```

Streaming API shape:
- Media type: `application/x-ndjson`
- Each line is a JSON object:
```json
{"event":"started|stream|progress|final","data":{...}}
```
- The `final` event contains final agent result payload.

## Data Flow Summary
1. Client calls FastAPI endpoint.
2. Pydantic validates and normalizes request.
3. Endpoint calls relevant agent/service.
4. Agent may call tools (retriever/web/technical/fundamental).
5. Result is shaped into typed response model.
6. FastAPI returns validated JSON response.

Streaming flow:
1. Client calls `.../stream` endpoint.
2. Server yields NDJSON events in real time (`started` -> intermediate `stream/progress` -> `final`).
3. Frontend appends events into live timeline panel and renders final result when `final` arrives.

## Notebook and Testing
- Notebook playground: `notebooks/agent_playground.ipynb`
- It includes direct agent tests and API-level tests for all endpoints.

## Frontend
- App path: `frontend/app/page.tsx`
- API base default: `NEXT_PUBLIC_API_BASE_URL` falls back to `/api` (for reverse-proxy deployment)
- Left-tab workflow includes:
  - PDF ingestion
  - Web search
  - Technical analysis
  - Fundamental analysis
  - Supervisor orchestration
- Ingested documents dropdown is shared across tabs.
- Fundamental and Supervisor forms use indexed-doc based dropdowns for company selection.
- Supervisor symbol remains manually editable by design.
- Ingestion tab shows all indexed docs and supports delete action.
- Web/Fundamental/Technical/Supervisor tabs show live stream event traces and final outputs.

## Docker + Nginx Deployment (Unified Container)
This repo supports a single-container full-stack deployment where frontend + backend run together:

- Docker build file: `Dockerfile`
- Nginx config: `deploy/nginx/default.conf`
- Container entrypoint: `deploy/entrypoint.sh`

Runtime routing:
- Frontend served at `/` (static Next.js export)
- Backend exposed at `/api/*` via Nginx reverse proxy to FastAPI on `127.0.0.1:8000`
- Example health endpoint in deployed mode: `GET /api/health`

Local Docker smoke test:
```powershell
docker build -t market-analyst-agent:fullstack-local .
docker run --rm -p 8080:80 market-analyst-agent:fullstack-local
```
Then verify:
- `http://localhost:8080/`
- `http://localhost:8080/api/health`

## Azure Container Instances (ACI) Deployment
Deployment script:
- `scripts/deploy_aci.ps1`

What the script does:
1. Validates Azure CLI login context.
2. Ensures resource group exists.
3. Ensures ACR exists and has admin enabled.
4. Builds/pushes Docker image via `az acr build`.
5. Recreates ACI container group with public IP.
6. Injects env vars from repo-root `.env` as secure environment variables.

Default exposed port in ACI:
- `80` (Nginx entrypoint)

Run deployment:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy_aci.ps1
```

Optional fixed ACR name:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy_aci.ps1 -AcrName <youracrname>
```

## Quick Run Checklist
1. `uv venv`
2. `uv pip install -r requirements.txt`
3. Fill `.env`
4. Start API with `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend`
5. Start frontend:
   - `cd frontend`
   - `npm.cmd install`
   - `npm.cmd run dev`
6. Open `http://localhost:3000` (frontend) and run workflows
