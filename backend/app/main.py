from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.fundamental.fundamental_agent import analyze_fundamentals
from app.agents.news.web_search_agent import (
    build_web_search_agent,
    invoke_market_web_search,
    parse_web_search_result,
)
from app.agents.supervisor.supervisor_agent import analyze_market_supervised
from app.agents.supervisor.supervisor_chat_agent import run_supervisor_chat_turn
from app.agents.technical.technical_chart_agent import analyze_stock_technical
from app.guardrails import GuardrailViolation, ensure_market_query
from app.services.document_ingestion import (
    delete_ingested_document_from_pgvector,
    ingest_pdf_to_pgvector,
    list_ingested_documents_from_pgvector,
)
from app.services.supervisor_chat_memory import (
    add_supervisor_chat_message,
    create_supervisor_chat_session,
    get_supervisor_chat_session,
    init_supervisor_chat_tables,
    list_supervisor_chat_messages,
    list_supervisor_chat_sessions,
    update_supervisor_chat_session_context,
)


app = FastAPI(title="Market Analyst Agent API")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load env vars from .env at app startup.
# Support running uvicorn from repo root or from backend/ directory.
_repo_root = Path(__file__).resolve().parents[2]
_env_candidates = [
    _repo_root / ".env",
    Path.cwd() / ".env",
]

for _env_path in _env_candidates:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
        break
else:
    load_dotenv(override=False)

init_supervisor_chat_tables()

_web_search_agent: Optional[Any] = None


def get_web_search_agent():
    global _web_search_agent
    if _web_search_agent is None:
        _web_search_agent = build_web_search_agent()
    return _web_search_agent


def _extract_agent_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(str(text))
                    elif item is not None:
                        parts.append(str(item))
                if parts:
                    return "\n".join(parts)
            if content:
                return str(content)
        output = payload.get("output")
        if output is not None:
            return str(output)
    return str(payload)


@app.exception_handler(GuardrailViolation)
async def guardrail_exception_handler(_: Request, exc: GuardrailViolation) -> JSONResponse:
    logger.warning("Guardrail violation: %s", str(exc))
    return JSONResponse(
        status_code=422,
        content={"error": "guardrail_violation", "message": str(exc)},
    )


class WebSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class WebSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    sources: list[dict]


class TechnicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1)
    period: str = "3mo"
    interval: str = "1d"

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("symbol must not be empty")
        return cleaned


class TechnicalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    image_path: str
    summary: str
    latest_values: Dict[str, Optional[float]]


class IngestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    company: str
    ticker: Optional[str]
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]
    doc_type: str
    year: Optional[str]


class IngestedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    company: str
    ticker: Optional[str]
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]
    doc_type: str = "annual_report"
    year: Optional[str]


class FundamentalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str = Field(..., min_length=1)
    question: Optional[str] = None
    mode: Literal["auto", "general", "qa"] = "auto"
    collection: str = "fundamental_docs"
    top_k: int = Field(8, ge=1, le=50)

    @field_validator("company", "collection")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: Optional[str] = None
    ticker: Optional[str] = None
    year: Optional[str] = None
    doc_type: Optional[str] = None
    source_path: Optional[str] = None
    chunk_index: Optional[int] = None


class FundamentalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    company: str
    answer: str
    sources: list[SourceDocument]


class SupervisorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1)
    company: str = Field(..., min_length=1)
    fundamental_question: Optional[str] = None
    news_query: Optional[str] = None
    technical_period: str = "3mo"
    technical_interval: str = "1d"
    collection: str = "fundamental_docs"
    top_k: int = Field(8, ge=1, le=50)

    @field_validator("symbol", "company", "collection")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("fundamental_question", "news_query")
    @classmethod
    def validate_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class TechnicalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    image_path: str
    summary: str
    latest_values: Dict[str, Optional[float]]


class FundamentalPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    company: str
    answer: str
    sources: list[SourceDocument]


class NewsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: str
    sources: list[dict] = Field(default_factory=list)


class SynthesisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investment_rating_6m: Optional[int] = Field(default=None, ge=1, le=10)
    stance: Literal["Bullish", "Neutral", "Bearish"]
    technical_section: str
    fundamental_section: str
    news_section: str
    risks: list[str]
    final_thesis: str


class SupervisorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    company: str
    technical: TechnicalPayload
    fundamental: FundamentalPayload
    news: NewsPayload
    synthesis: SynthesisPayload


class SupervisorChatCreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., min_length=1)
    symbol: Optional[str] = None
    company: Optional[str] = None


class SupervisorChatSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    symbol: Optional[str]
    company: Optional[str]
    created_at: str
    updated_at: str


class SupervisorChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class SupervisorChatTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    symbol: Optional[str] = None
    company: Optional[str] = None


class SupervisorChatTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session: SupervisorChatSession
    assistant_message: SupervisorChatMessage


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agents/web-search", response_model=WebSearchResponse)
def run_web_search(payload: WebSearchRequest) -> WebSearchResponse:
    """
    Minimal endpoint to exercise the web search agent.
    Expects: {"query": "..."}
    """
    query = ensure_market_query(payload.query, field_name="query")
    logger.info("API /agents/web-search")
    agent = get_web_search_agent()
    result = invoke_market_web_search(agent, query)
    parsed = parse_web_search_result(result, query)
    return WebSearchResponse(query=parsed["query"], answer=parsed["answer"], sources=parsed["sources"])


@app.post("/agents/technical", response_model=TechnicalResponse)
def run_technical(payload: TechnicalRequest) -> TechnicalResponse:
    """
    Minimal endpoint to exercise the technical analysis agent.
    Expects: {"symbol": "AAPL", "period": "3mo", "interval": "1d"}
    """
    logger.info("API /agents/technical")
    result = analyze_stock_technical(payload.symbol, period=payload.period, interval=payload.interval)
    return TechnicalResponse(
        symbol=result.symbol,
        image_path=result.image_path,
        summary=result.summary,
        latest_values=result.latest_values,
    )


@app.post("/agents/fundamental", response_model=FundamentalResponse)
def run_fundamental(payload: FundamentalRequest) -> FundamentalResponse:
    """
    Agentic RAG over company-specific annual report chunks in pgvector.
    mode=auto -> general if no question, qa otherwise.
    """
    logger.info("API /agents/fundamental")
    result = analyze_fundamentals(
        company=payload.company,
        question=payload.question,
        mode=payload.mode,
        collection_name=payload.collection,
        top_k=payload.top_k,
    )
    return FundamentalResponse(
        mode=result.mode,
        company=result.company,
        answer=result.answer,
        sources=result.sources,
    )


@app.post("/agents/supervisor", response_model=SupervisorResponse)
def run_supervisor(payload: SupervisorRequest) -> SupervisorResponse:
    """
    Supervisor orchestration over technical + fundamental + news agents.
    """
    logger.info("API /agents/supervisor")
    result = analyze_market_supervised(
        symbol=payload.symbol,
        company=payload.company,
        fundamental_question=payload.fundamental_question,
        news_query=payload.news_query,
        technical_period=payload.technical_period,
        technical_interval=payload.technical_interval,
        collection_name=payload.collection,
        top_k=payload.top_k,
    )
    return SupervisorResponse(
        symbol=result.symbol,
        company=result.company,
        technical=result.technical,
        fundamental=result.fundamental,
        news=result.news,
        synthesis=result.synthesis,
    )


@app.post("/agents/ingest", response_model=IngestionResponse)
async def ingest_document(
    company: str = Form(...),
    ticker: Optional[str] = Form(None),
    doc_type: str = Form("annual_report"),
    year: Optional[str] = Form(None),
    collection: str = Form("fundamental_docs"),
    azure_model: str = Form("prebuilt-layout"),
    embeddings_deployment: Optional[str] = Form(None),
    file: UploadFile = File(...),
) -> IngestionResponse:
    """
    Upload a PDF and ingest it into pgvector.
    """
    logger.info("API /agents/ingest")
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = file.filename or "uploaded.pdf"
    target_path = upload_dir / safe_name
    content = await file.read()
    target_path.write_bytes(content)

    result = ingest_pdf_to_pgvector(
        pdf_path=target_path,
        company=company.strip(),
        ticker=ticker.strip().upper() if ticker else None,
        doc_type=doc_type.strip(),
        year=year.strip() if year else None,
        collection_name=collection.strip(),
        azure_model_id=azure_model.strip(),
        embeddings_deployment=embeddings_deployment.strip() if embeddings_deployment else None,
    )

    record_list = list_ingested_documents_from_pgvector(collection_name=result.collection_name)
    matched = next(
        (
            item
            for item in record_list
            if item.get("collection_name") == result.collection_name
            and item.get("company") == result.company
            and item.get("source_path") == result.source_path
            and (item.get("ticker") or None) == (result.ticker or None)
            and (item.get("year") or None) == (year.strip() if year else None)
        ),
        None,
    )
    if matched is None:
        matched = {
            "id": "",
            "doc_type": doc_type.strip(),
            "year": year.strip() if year else None,
        }

    return IngestionResponse(
        id=matched["id"],
        company=result.company,
        ticker=result.ticker,
        source_path=result.source_path,
        chunks_stored=result.chunks_stored,
        collection_name=result.collection_name,
        markdown_path=result.markdown_path,
        doc_type=str(matched.get("doc_type") or doc_type.strip()),
        year=matched.get("year"),
    )


@app.get("/agents/supervisor-chat/sessions", response_model=list[SupervisorChatSession])
def get_supervisor_chat_sessions() -> list[SupervisorChatSession]:
    logger.info("API /agents/supervisor-chat/sessions")
    sessions = list_supervisor_chat_sessions()
    return [SupervisorChatSession(**item) for item in sessions]


@app.post("/agents/supervisor-chat/sessions", response_model=SupervisorChatSession)
def create_supervisor_chat_session_endpoint(payload: SupervisorChatCreateSessionRequest) -> SupervisorChatSession:
    logger.info("API create /agents/supervisor-chat/sessions")
    created = create_supervisor_chat_session(
        title=payload.title.strip(),
        symbol=payload.symbol.strip().upper() if payload.symbol else None,
        company=payload.company.strip().upper() if payload.company else None,
    )
    return SupervisorChatSession(**created)


@app.get("/agents/supervisor-chat/sessions/{session_id}/messages", response_model=list[SupervisorChatMessage])
def get_supervisor_chat_history(session_id: str) -> list[SupervisorChatMessage]:
    logger.info("API /agents/supervisor-chat/sessions/%s/messages", session_id)
    session = get_supervisor_chat_session(session_id)
    if session is None:
        return []
    messages = list_supervisor_chat_messages(session_id)
    return [SupervisorChatMessage(**item) for item in messages]


@app.post("/agents/supervisor-chat/message", response_model=SupervisorChatTurnResponse)
def run_supervisor_chat_turn_endpoint(payload: SupervisorChatTurnRequest) -> SupervisorChatTurnResponse:
    logger.info("API /agents/supervisor-chat/message")
    session = get_supervisor_chat_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session_id not found")

    message_text = ensure_market_query(payload.message, field_name="message")
    symbol = payload.symbol.strip().upper() if payload.symbol else session.get("symbol")
    company = payload.company.strip().upper() if payload.company else session.get("company")

    update_supervisor_chat_session_context(session_id=payload.session_id, symbol=symbol, company=company)
    add_supervisor_chat_message(session_id=payload.session_id, role="user", content=message_text)
    turn = run_supervisor_chat_turn(
        session_id=payload.session_id,
        user_message=message_text,
        symbol=symbol,
        company=company,
    )
    assistant = add_supervisor_chat_message(
        session_id=payload.session_id,
        role="assistant",
        content=str(turn["answer"]),
    )
    updated_session = get_supervisor_chat_session(payload.session_id) or session
    return SupervisorChatTurnResponse(
        session=SupervisorChatSession(**updated_session),
        assistant_message=SupervisorChatMessage(**assistant),
    )


@app.get("/agents/ingested-docs", response_model=list[IngestedDocument])
def get_ingested_docs() -> list[IngestedDocument]:
    logger.info("API /agents/ingested-docs")
    items = list_ingested_documents_from_pgvector()
    return [IngestedDocument(**item) for item in items]


@app.delete("/agents/ingested-docs/{doc_id}")
def delete_ingested_doc(doc_id: str) -> dict:
    logger.info("API delete ingested-doc id=%s", doc_id)
    deleted_count = delete_ingested_document_from_pgvector(doc_id=doc_id)
    if deleted_count <= 0:
        return {"deleted": False, "message": "No matching vectors found for document id."}
    return {"deleted": True, "deleted_chunks": deleted_count}
