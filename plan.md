# Market Analyst Agent Plan

## Goal
Build a multi-agent market analyst system with a supervisor agent orchestrating three sub-agents (technical, fundamental, news), exposed via a FastAPI backend and a Next.js frontend, with a Jupyter notebook for iterative testing of agents and API logic.

## High-Level Architecture
- Supervisor agent coordinates:
  - Technical Analysis Agent: generates chart with MACD + RSI, uses vision model to interpret chart, returns technical signals.
  - Fundamental Analysis Agent: retrieves company-specific PDF chunks from vector DB, analyzes income statement, balance sheet, cash flow across years.
  - News Analysis Agent: uses Tavily API to search web, summarizes recent/news sentiment and key events.
- Supervisor compiles:
  - Sectioned report: Technical / Fundamental / News
  - Final 6-month growth rating (1–10) with rationale.

## Implementation Phases
1. **Backend (FastAPI)**
   - Core agent orchestration
   - Retrieval + analysis logic
   - API endpoints for single company analysis and chat mode
   - Notebook to run and validate agents before API integration
2. **Frontend (Next.js)**
   - Company input mode
   - Query/chat mode
   - Results dashboard with per-agent sections and rating

## Workstreams
### A. Backend & Agent Core
1. Define data contracts (request/response schemas) — **Completed**
2. Implement Technical Analysis Agent — **Completed**
   - Fetch historical prices
   - Compute MACD/RSI
   - Render chart image
   - Run vision model for signals
3. Implement Fundamental Analysis Agent — **Pending**
   - Metadata filter by company
   - Retrieve vector chunks
   - Parse financial statements (IS/BS/CF)
   - Multi-year trend analysis
4. Implement News Analysis Agent — **Completed**
   - Tavily API integration
   - Rerank/cluster results
   - Summarize + sentiment
5. Implement Supervisor Agent — **Pending**
   - Tool calling to sub-agents
   - Synthesize final report + 6-month rating
6. Expose FastAPI endpoints — **In Progress**
   - `POST /analyze` (company)
   - `POST /chat` (company + query)
7. Observability and logging

### B. Document Ingestion (PDF → Vector DB)
1. Upload PDFs (annual reports / financial documents) — **Pending**
2. Azure Document Intelligence prebuilt model → markdown output — **Pending**
3. Chunk by headers using LangChain Markdown Header Splitter — **Pending**
4. Store chunks in Postgres with pgvector + metadata (company, doc type, year) — **Pending**


### B. Notebook (Prototype + Test)
1. Setup notebook environment — **Completed**
2. Run each sub-agent independently — **In Progress**
3. Run supervisor end-to-end — **Pending**
4. Validate JSON outputs for API parity — **Pending**

### C. Frontend (Next.js)
1. Input page with two modes
2. Results rendering
3. Chat interface

## File Structure & Status
- `.env` — Local environment variables for API keys and endpoints. **Implemented**
- `.gitignore` — Git ignore rules for venv, data outputs, secrets. **Implemented**
- `README.md` — Setup and run instructions. **Implemented**
- `requirements.txt` — Python dependencies list. **Implemented**
- `plan.md` — Project plan and status tracking. **Implemented**
- `notebooks/agent_playground.ipynb` — Run and test agents in a notebook. **Implemented**
- `backend/app/main.py` — FastAPI app with agent endpoints. **Implemented**
- `backend/app/agents/news/web_search_agent.py` — Tavily + Azure OpenAI web search agent. **Implemented**
- `backend/app/agents/technical/technical_chart_agent.py` — Chart build + indicators + vision analysis. **Implemented**
- `backend/app/agents/fundamental/` — Fundamental analysis agent module. **Pending**
- `backend/app/agents/supervisor/` — Supervisor agent module. **Pending**
- `backend/app/api/` — Future route modules. **Pending**
- `backend/app/core/` — Shared core utilities/config. **Pending**
- `backend/app/db/` — DB connectors and vector storage. **Pending**
- `backend/app/schemas/` — Request/response schemas. **In Progress**
- `backend/app/services/` — External services integrations. **Pending**
- `backend/app/utils/` — Helpers and shared utilities. **Pending**
- `backend/app/prompts/` — Prompt templates. **Pending**
- `backend/tests/` — Backend tests. **Pending**
- `frontend/` — Next.js app. **Pending**
- `scripts/` — Utility scripts. **Pending**
- `data/raw/` — Raw PDFs and source documents. **Pending**
- `data/processed/` — Generated artifacts (charts). **Implemented**
- `data/chunks/` — Chunked document outputs. **Pending**
- `configs/` — Config files. **Pending**
- `docs/` — Project documentation. **Pending**

## Open Decisions
- Charting & TA library (mplfinance, plotly, or tradingview)
- Vision model provider (OpenAI Vision, Gemini, etc.)
- Vector DB choice (FAISS/Chroma/Pinecone)
- PDF parsing pipeline (pdfplumber, unstructured)

## Milestones
1. Backend skeleton + notebook scaffold — **Completed**
2. Technical Analysis Agent working end-to-end — **Completed**
3. Fundamental Analysis Agent working end-to-end — **Pending**
4. News Analysis Agent working end-to-end — **Completed**
5. Supervisor synthesis + API endpoints — **Pending**
6. Frontend integration — **Pending**

