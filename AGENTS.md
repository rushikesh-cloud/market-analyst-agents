# AGENTS.md

## Project overview
Market Analyst Agents is a FastAPI backend with multiple agentic analysis paths for:
- web/news analysis
- technical chart analysis
- fundamental RAG over ingested annual report data
- supervisor synthesis across all three

Primary app entrypoint:
- `backend/app/main.py`

## Runtime architecture
- API layer (FastAPI): validates input/output schemas and routes to agent/service functions.
- Agent layer:
  - `news/web_search_agent.py`: Tavily + Azure OpenAI via LangChain `create_agent`.
  - `technical/technical_chart_agent.py`: yfinance + indicators + chart image + vision summary.
  - `fundamental/fundamental_agent.py`: pgvector retrieval + agentic answer generation.
  - `supervisor/supervisor_agent.py`: orchestrates technical + fundamental + news tools and produces structured synthesis.
- Service layer:
  - `services/document_ingestion.py`: PDF -> Azure Document Intelligence markdown -> chunking -> pgvector.

## Current API surface
Defined in `backend/app/main.py`.

### `GET /health`
- Returns `{ "status": "ok" }`.

### `POST /agents/web-search`
- Request: `WebSearchRequest`
  - `query: str` (trimmed, non-empty)
- Response: `WebSearchResponse`
  - `query: str`
  - `answer: str`
- Invocation format uses LangChain v1 style:
  - `agent.invoke({"messages": [{"role": "user", "content": query}]})`

### `POST /agents/technical`
- Request: `TechnicalRequest`
  - `symbol: str` (trimmed, uppercased, non-empty)
  - `period: str` default `3mo`
  - `interval: str` default `1d`
- Response: `TechnicalResponse`
  - `symbol: str`
  - `image_path: str`
  - `summary: str`
  - `latest_values: Dict[str, Optional[float]]`

### `POST /agents/fundamental`
- Request: `FundamentalRequest`
  - `company: str` (non-empty)
  - `question: Optional[str]`
  - `mode: Literal["auto", "general", "qa"]`
  - `collection: str` default `fundamental_docs`
  - `top_k: int` in `[1, 50]`
- Response: `FundamentalResponse`
  - `mode: str`
  - `company: str`
  - `answer: str`
  - `sources: list[SourceDocument]`

### `POST /agents/supervisor`
- Request: `SupervisorRequest`
  - `symbol: str`
  - `company: str`
  - `fundamental_question: Optional[str]`
  - `news_query: Optional[str]`
  - `technical_period: str`
  - `technical_interval: str`
  - `collection: str`
  - `top_k: int` in `[1, 50]`
- Response: `SupervisorResponse`
  - `symbol`, `company`
  - `technical: TechnicalPayload`
  - `fundamental: FundamentalPayload`
  - `news: NewsPayload`
  - `synthesis: SynthesisPayload`

### `POST /agents/ingest`
- Multipart form upload for PDF ingestion.
- Fields: `company`, `doc_type`, `year`, `collection`, `azure_model`, `embeddings_deployment`, `file`.
- Returns `IngestionResponse` with chunk count and source metadata.

## Agent implementation details

### News agent (`backend/app/agents/news/web_search_agent.py`)
- Builds `create_agent(...)` with:
  - Azure chat model (`AzureChatOpenAI`)
  - Tavily search tool (`TavilySearchResults(max_results=5)`)
- System prompt enforces web-backed market summary behavior.

### Technical agent (`backend/app/agents/technical/technical_chart_agent.py`)
- Data fetch: `yfinance.download(...)`
- Indicators: MACD and RSI via `pandas_ta`
- Plot: candlestick + MACD + RSI via `mplfinance`
- Vision summary: sends chart image as base64 data URL to Azure chat model
- Safety normalization:
  - model response content is normalized to string
  - non-finite numeric values (`NaN`, `inf`) are converted to `None`

### Fundamental agent (`backend/app/agents/fundamental/fundamental_agent.py`)
- Creates `create_agent(...)` with custom retriever tool (`company_retriever`).
- Vector store: `PGVector` with collection filter by `company`.
- Supports modes:
  - `general`: structured fundamentals summary
  - `qa`: answer targeted question
  - `auto`: chooses based on question presence
- Uses messages-based invocation payload:
  - `{"messages": [{"role": "user", "content": prompt}]}`

### Supervisor agent (`backend/app/agents/supervisor/supervisor_agent.py`)
- Creates orchestrator agent with 3 tools:
  - `technical_subagent`
  - `fundamental_subagent`
  - `news_subagent`
- Enforces JSON-only final synthesis target.
- Includes fallback logic when JSON parse fails or tool calls are missed.
- Uses messages-based invocation payload for all `create_agent` calls.

## Ingestion pipeline details (`backend/app/services/document_ingestion.py`)
1. Read PDF bytes.
2. Azure Document Intelligence (`prebuilt-layout` by default) returns markdown.
3. Split markdown by heading hierarchy.
4. Attach metadata per chunk:
   - `company`, `doc_type`, `year`, `source_path`, `chunk_index`
5. Store chunks in pgvector collection.

## Environment variables
Required/used across modules:
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

## Local run and tooling

### `uv` environment
- Virtual environment directory: `.venv`
- Python executable (repo-relative):
  - `.venv\Scripts\python.exe`
- Python executable (absolute on this machine):
  - `C:\Users\rushi\OneDrive - ImmersiLearn Education Services LLP\Projects\LLM Projects\market-analyst-agents\.venv\Scripts\python.exe`

### Typical commands
- Install deps:
  - `uv pip install -r requirements.txt`
- Run API:
  - `uvicorn app.main:app --reload --app-dir backend`
- Run ingestion script with uv Python explicitly:
  - `.venv\Scripts\python.exe scripts\ingest_fundamentals.py --pdf "documents\apple-10k-report.pdf" --company "APPLE" --year 2024`

## Repository map (key paths)
- `backend/app/main.py` - FastAPI app + request/response models.
- `backend/app/agents/news/web_search_agent.py` - web search agent builder.
- `backend/app/agents/technical/technical_chart_agent.py` - technical workflow.
- `backend/app/agents/fundamental/fundamental_agent.py` - fundamental RAG workflow.
- `backend/app/agents/supervisor/supervisor_agent.py` - orchestrator/synthesis.
- `backend/app/services/document_ingestion.py` - PDF ingestion to pgvector.
- `scripts/ingest_fundamentals.py` - ingestion CLI utility.
- `notebooks/agent_playground.ipynb` - experimentation notebook.
- `docs/architecture.md` - architecture notes and diagrams.

## Notes for future changes
- Keep LangChain agent invocations in messages format (`{"messages": [...]}`) for `create_agent`-based agents.
- Maintain strict API schema validation (`extra="forbid"`) in `main.py`.
- Preserve JSON-safe numeric outputs (`Optional[float]` where non-finite values can occur).
