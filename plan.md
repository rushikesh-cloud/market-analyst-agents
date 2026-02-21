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
1. Define data contracts (request/response schemas)
2. Implement Technical Analysis Agent
   - Fetch historical prices
   - Compute MACD/RSI
   - Render chart image
   - Run vision model for signals
3. Implement Fundamental Analysis Agent
   - Metadata filter by company
   - Retrieve vector chunks
   - Parse financial statements (IS/BS/CF)
   - Multi-year trend analysis
4. Implement News Analysis Agent
   - Tavily API integration
   - Rerank/cluster results
   - Summarize + sentiment
5. Implement Supervisor Agent
   - Tool calling to sub-agents
   - Synthesize final report + 6-month rating
6. Expose FastAPI endpoints
   - `POST /analyze` (company)
   - `POST /chat` (company + query)
7. Observability and logging

### B. Document Ingestion (PDF → Vector DB)
1. Upload PDFs (annual reports / financial documents)
2. Azure Document Intelligence prebuilt model → markdown output
3. Chunk by headers using LangChain Markdown Header Splitter
4. Store chunks in Postgres with pgvector + metadata (company, doc type, year)


### B. Notebook (Prototype + Test)
1. Setup notebook environment
2. Run each sub-agent independently
3. Run supervisor end-to-end
4. Validate JSON outputs for API parity

### C. Frontend (Next.js)
1. Input page with two modes
2. Results rendering
3. Chat interface

## Open Decisions
- Charting & TA library (mplfinance, plotly, or tradingview)
- Vision model provider (OpenAI Vision, Gemini, etc.)
- Vector DB choice (FAISS/Chroma/Pinecone)
- PDF parsing pipeline (pdfplumber, unstructured)

## Milestones
1. Backend skeleton + notebook scaffold
2. Technical Analysis Agent working end-to-end
3. Fundamental Analysis Agent working end-to-end
4. News Analysis Agent working end-to-end
5. Supervisor synthesis + API endpoints
6. Frontend integration

