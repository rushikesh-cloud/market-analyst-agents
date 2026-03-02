# AGENTS.md

## Important Points (Current)
- Backend is FastAPI (`backend/app/main.py`) with strict Pydantic validation (`extra="forbid"`).
- All LangChain `create_agent` invocations use messages payload format: `{"messages": [{"role": "user", "content": "..."}]}`.
- Guardrails are LLM-based (`backend/app/guardrails.py`) and enforce market/stock-only scope.
- Agent-level logging is implemented across news, technical, fundamental, supervisor, and supervisor-chat modules.
- Ingestion metadata includes `company`, `ticker`, `year`, `doc_type`, `source_path`, `chunk_index`.
- Ingested document list/delete are vector-DB backed (pgvector tables), not file-registry backed.
- Web search response is normalized to human-readable `answer` + extracted `sources` list.
- Frontend is Next.js (`frontend/`) with left-tab workflow matching dashboard style.
- Supervisor Chat supports persistent session memory with LangGraph Postgres checkpointer.

## Runtime Architecture
- API layer: `backend/app/main.py`
- Agents:
  - `backend/app/agents/news/web_search_agent.py`
  - `backend/app/agents/technical/technical_chart_agent.py`
  - `backend/app/agents/fundamental/fundamental_agent.py`
  - `backend/app/agents/supervisor/supervisor_agent.py`
  - `backend/app/agents/supervisor/supervisor_chat_agent.py`
- Services:
  - `backend/app/services/document_ingestion.py`
  - `backend/app/services/supervisor_chat_memory.py`

## Key API Endpoints
- `GET /health`
- `POST /agents/web-search`
  - Returns `query`, `answer`, `sources`
- `POST /agents/technical`
- `POST /agents/fundamental`
- `POST /agents/supervisor`
- `POST /agents/ingest`
  - Accepts multipart: `company`, `ticker`, `year`, `collection`, `file`, etc.
- `GET /agents/ingested-docs`
  - Lists ingested docs from pgvector metadata groups
- `DELETE /agents/ingested-docs/{doc_id}`
  - Deletes matching chunks from vector DB
- `GET /agents/supervisor-chat/sessions`
- `POST /agents/supervisor-chat/sessions`
- `GET /agents/supervisor-chat/sessions/{session_id}/messages`
- `POST /agents/supervisor-chat/message`

## Supervisor Chat Memory
- Uses `langgraph-checkpoint-postgres` (`PostgresSaver`) with `thread_id = session_id`.
- Chat sessions/messages persist in Postgres tables:
  - `supervisor_chat_sessions`
  - `supervisor_chat_messages`
- Session context (`symbol`, `company`) is stored and updated per session.
- Chat request forwards frontend-provided `symbol` and `company` into supervisor analysis.

## Frontend Notes
- App path: `frontend/app/page.tsx`
- Tabs:
  - PDF Ingestion
  - Web Search Agent
  - Technical Agent
  - Fundamental Agent
  - Supervisor Agent
  - Supervisor Chat
- Chat tab has explicit `ticker` and `company` fields.
- Selecting an ingested document auto-populates chat ticker/company and other agent forms.
- Results and chat history are rendered in markdown.

## Environment Variables
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_KEY`
- `PGVECTOR_CONNECTION_STRING`
- `TAVILY_API_KEY`
- Optional: `AZURE_OPENAI_GUARDRAIL_DEPLOYMENT`

## Dependencies Added Recently
- `langgraph`
- `langgraph-checkpoint-postgres`

## Local Run
- Backend:
  - `uvicorn app.main:app --reload --app-dir backend`
- Frontend:
  - `cd frontend`
  - `npm install`
  - `npm run dev`
  - Optional: `NEXT_PUBLIC_API_BASE=http://localhost:8000`

## Guardrails / Conventions
- Keep responses market-scope only.
- Keep structured responses JSON-safe.
- Preserve symbol/company passthrough from frontend into supervisor/supervisor-chat analysis.
- Avoid raw tool payload rendering in UI; prefer parsed answer + source links.
