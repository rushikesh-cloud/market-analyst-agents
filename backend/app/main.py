from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from langchain.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.fundamental.fundamental_agent import analyze_fundamentals, stream_fundamentals
from app.agents.news.web_search_agent import build_web_search_agent
from app.agents.supervisor.supervisor_agent import analyze_market_supervised, stream_market_supervised
from app.agents.technical.technical_chart_agent import analyze_stock_technical
from app.services.agent_run_logger import log_agent_run
from app.services.document_ingestion import ingest_pdf_to_pgvector
from app.services.query_guardrail import validate_market_query
from app.services.vector_document_registry import delete_ingested_document, list_ingested_documents

app = FastAPI(title="Market Analyst Agent API")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="data"), name="static")


@app.options("/{rest_of_path:path}")
def preflight_handler(rest_of_path: str) -> Response:
    return Response(status_code=204)

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

_web_search_agent: Optional[Any] = None


def get_web_search_agent():
    global _web_search_agent
    if _web_search_agent is None:
        _web_search_agent = build_web_search_agent()
    return _web_search_agent


class WebSearchRequest(BaseModel):
    messages: list[str] = Field(default_factory=list, min_length=1)

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: list[str]) -> list[str]:
        cleaned = [msg.strip() for msg in value if isinstance(msg, str) and msg.strip()]
        if not cleaned:
            raise ValueError("messages must contain at least one non-empty query string")
        return cleaned


class WebSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: Dict[str, Any]


class TechnicalRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    period: str = Field(default="3mo", min_length=2, max_length=10)
    interval: str = Field(default="1d", min_length=2, max_length=10)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("symbol must be a non-empty ticker")
        return symbol


class TechnicalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    image_path: str
    summary: str
    latest_values: Dict[str, float]


class IngestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: str
    ticker: Optional[str] = None
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: Optional[str] = None
    ticker: Optional[str] = None
    year: Optional[str] = None
    doc_type: Optional[str] = None
    source_path: Optional[str] = None
    chunk_index: Optional[int] = None


class IngestedDocumentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_name: str
    source_path: str
    company: Optional[str] = None
    ticker: Optional[str] = None
    doc_type: Optional[str] = None
    year: Optional[str] = None
    chunks_stored: int


class IngestedDocumentsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[IngestedDocumentItem]


class DeleteIngestedDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: str = Field(default="fundamental_docs", min_length=1)
    source_path: str = Field(..., min_length=1)


class DeleteIngestedDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_name: str
    source_path: str
    deleted_chunks: int


class FundamentalRequest(BaseModel):
    company: str = Field(..., min_length=1, max_length=100)
    question: Optional[str] = None
    mode: Literal["auto", "general", "qa"] = "auto"
    collection: str = Field(default="fundamental_docs", min_length=1)
    top_k: int = Field(default=8, ge=1, le=25)

    @field_validator("company")
    @classmethod
    def _normalize_company(cls, value: str) -> str:
        company = value.strip()
        if not company:
            raise ValueError("company must be non-empty")
        return company

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _validate_mode_question(self) -> "FundamentalRequest":
        if self.mode == "qa" and not self.question:
            raise ValueError("question is required when mode='qa'")
        return self


class FundamentalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    company: str
    answer: str
    sources: list[SourceDocument]


class SupervisorRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    company: str = Field(..., min_length=1, max_length=100)
    fundamental_question: Optional[str] = None
    news_query: Optional[str] = None
    technical_period: str = Field(default="3mo", min_length=2, max_length=10)
    technical_interval: str = Field(default="1d", min_length=2, max_length=10)
    collection: str = Field(default="fundamental_docs", min_length=1)
    top_k: int = Field(default=8, ge=1, le=25)

    @field_validator("symbol")
    @classmethod
    def _normalize_supervisor_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("symbol must be non-empty")
        return symbol

    @field_validator("company")
    @classmethod
    def _normalize_supervisor_company(cls, value: str) -> str:
        company = value.strip()
        if not company:
            raise ValueError("company must be non-empty")
        return company

    @field_validator("fundamental_question", "news_query")
    @classmethod
    def _strip_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SupervisorTechnical(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    image_path: str
    summary: str
    latest_values: Dict[str, float]


class SupervisorFundamental(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    company: str
    answer: str
    sources: list[SourceDocument]


class SupervisorNews(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    answer: str


class SupervisorSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    investment_rating_6m: Optional[int] = Field(default=None, ge=1, le=10)
    stance: str
    technical_section: str
    fundamental_section: str
    news_section: str
    risks: list[str]
    final_thesis: str


class SupervisorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str
    company: str
    technical: SupervisorTechnical
    fundamental: SupervisorFundamental
    news: SupervisorNews
    synthesis: SupervisorSynthesis


def _messages_payload(query: str) -> dict[str, list[HumanMessage]]:
    return {"messages": [HumanMessage(content=query)]}


def _extract_final_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content:
                return str(content)
        output = payload.get("output")
        if isinstance(output, str):
            return output
    if hasattr(payload, "content"):
        content = getattr(payload, "content", None)
        if content:
            return str(content)
    return str(payload)


def _safe_json(value: Any) -> Any:
    try:
        return jsonable_encoder(value)
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _safe_json(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "dict"):
        try:
            return _safe_json(value.dict())
        except Exception:
            return str(value)
    if hasattr(value, "content"):
        data = {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": getattr(value, "content", None),
            "name": getattr(value, "name", None),
            "id": getattr(value, "id", None),
            "tool_calls": getattr(value, "tool_calls", None),
            "additional_kwargs": getattr(value, "additional_kwargs", None),
            "response_metadata": getattr(value, "response_metadata", None),
        }
        return {k: _safe_json(v) for k, v in data.items()}
    return str(value)


def _ndjson_line(event: str, data: Any) -> str:
    return json.dumps({"event": event, "data": _safe_json(data)}, ensure_ascii=False) + "\n"


def _extract_message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is not None:
        return str(content)
    return str(message)


def _extract_final_messages(payload: Any) -> list[str]:
    if payload is None:
        return []

    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            collected = [_extract_message_content(msg).strip() for msg in messages]
            return [msg for msg in collected if msg]

        output = payload.get("output")
        if isinstance(output, str) and output.strip():
            return [output.strip()]

    text = _extract_final_text(payload).strip()
    return [text] if text else []


def _count_total_tokens(payload: Any) -> Optional[int]:
    total = 0
    found_any = False
    visited: set[int] = set()

    def walk(value: Any) -> None:
        nonlocal total, found_any
        marker = id(value)
        if marker in visited:
            return
        visited.add(marker)

        if isinstance(value, dict):
            token_usage = value.get("token_usage")
            if isinstance(token_usage, dict):
                token_total = token_usage.get("total_tokens")
                if isinstance(token_total, (int, float)):
                    total += int(token_total)
                    found_any = True
            direct_total = value.get("total_tokens")
            if isinstance(direct_total, (int, float)):
                total += int(direct_total)
                found_any = True
            for sub in value.values():
                walk(sub)
            return

        if isinstance(value, (list, tuple, set)):
            for sub in value:
                walk(sub)
            return

        response_metadata = getattr(value, "response_metadata", None)
        if isinstance(response_metadata, dict):
            walk(response_metadata)
        usage_metadata = getattr(value, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            walk(usage_metadata)
        additional_kwargs = getattr(value, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            walk(additional_kwargs)
        content = getattr(value, "content", None)
        if content is not None and not isinstance(content, str):
            walk(content)
        tool_calls = getattr(value, "tool_calls", None)
        if isinstance(tool_calls, list):
            walk(tool_calls)

    walk(payload)
    if not found_any:
        return None
    return total


def _log_agent_run_safe(
    *,
    agent_name: str,
    company: Optional[str],
    symbol: Optional[str],
    input_query: Optional[str],
    input_messages_count: int,
    result_payload: Any,
) -> None:
    try:
        log_agent_run(
            agent_name=agent_name,
            company=company,
            symbol=symbol,
            input_query=input_query,
            final_messages=_extract_final_messages(result_payload),
            total_tokens=_count_total_tokens(result_payload),
            message_count=input_messages_count,
            raw_result=_safe_json(result_payload),
        )
    except Exception as exc:
        logger.exception("Failed to log agent run for %s: %s", agent_name, exc)


def _enforce_market_guardrail(
    *,
    query: str,
    agent_name: str,
    company: Optional[str] = None,
    symbol: Optional[str] = None,
) -> None:
    try:
        decision = validate_market_query(
            query=query,
            company=company,
            symbol=symbol,
            agent_name=agent_name,
        )
    except Exception as exc:
        logger.exception("Guardrail check failed for %s: %s", agent_name, exc)
        raise HTTPException(
            status_code=503,
            detail="Guardrail validation is temporarily unavailable. Please retry.",
        ) from exc

    if not decision.allowed:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "out_of_scope_query",
                "message": "Request must be related to company market/investment analysis.",
                "reason": decision.reason,
            },
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agents/web-search", response_model=WebSearchResponse)
def run_web_search(payload: WebSearchRequest) -> WebSearchResponse:
    """
    Minimal endpoint to exercise the web search agent.
    Expects: {"messages": ["..."]}
    """
    query = payload.messages[-1].strip()
    if not query:
        raise HTTPException(status_code=422, detail="messages must include a non-empty final query")
    _enforce_market_guardrail(query=query, agent_name="web_search")

    agent = get_web_search_agent()
    result = agent.invoke(_messages_payload(query))
    _log_agent_run_safe(
        agent_name="web_search",
        company=None,
        symbol=None,
        input_query=query,
        input_messages_count=len(payload.messages),
        result_payload=result,
    )
    return WebSearchResponse(result=result)


@app.post("/agents/web-search/stream")
def run_web_search_stream(payload: WebSearchRequest) -> StreamingResponse:
    query = payload.messages[-1].strip()
    if not query:
        raise HTTPException(status_code=422, detail="messages must include a non-empty final query")
    _enforce_market_guardrail(query=query, agent_name="web_search_stream")

    def event_stream():
        agent = get_web_search_agent()
        yield _ndjson_line("started", {"query": query})
        last_values_payload: Any = None
        for chunk in agent.stream(_messages_payload(query), stream_mode=["updates", "messages", "values"]):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                stream_mode, data = chunk
                if stream_mode == "values":
                    last_values_payload = data
                yield _ndjson_line("stream", {"stream_mode": stream_mode, "payload": data})
            else:
                yield _ndjson_line("stream", {"payload": chunk})

        yield _ndjson_line(
            "final",
            {
                "result": last_values_payload,
                "answer": _extract_final_text(last_values_payload),
            },
        )
        _log_agent_run_safe(
            agent_name="web_search_stream",
            company=None,
            symbol=None,
            input_query=query,
            input_messages_count=len(payload.messages),
            result_payload=last_values_payload,
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/agents/technical", response_model=TechnicalResponse)
def run_technical(payload: TechnicalRequest) -> TechnicalResponse:
    """
    Minimal endpoint to exercise the technical analysis agent.
    Expects: {"symbol": "AAPL", "period": "3mo", "interval": "1d"}
    """
    technical_intent = (
        f"Technical stock market analysis for ticker {payload.symbol} "
        f"with period {payload.period} and interval {payload.interval}."
    )
    _enforce_market_guardrail(
        query=technical_intent,
        agent_name="technical",
        symbol=payload.symbol,
    )
    result = analyze_stock_technical(payload.symbol, period=payload.period, interval=payload.interval)
    result_payload = {
        "symbol": result.symbol,
        "image_path": result.image_path,
        "summary": result.summary,
        "latest_values": result.latest_values,
    }
    _log_agent_run_safe(
        agent_name="technical",
        company=None,
        symbol=payload.symbol,
        input_query=f"symbol={payload.symbol}, period={payload.period}, interval={payload.interval}",
        input_messages_count=1,
        result_payload=result_payload,
    )
    return TechnicalResponse(
        symbol=result.symbol,
        image_path=result.image_path,
        summary=result.summary,
        latest_values=result.latest_values,
    )


@app.post("/agents/technical/stream")
def run_technical_stream(payload: TechnicalRequest) -> StreamingResponse:
    technical_intent = (
        f"Technical stock market analysis for ticker {payload.symbol} "
        f"with period {payload.period} and interval {payload.interval}."
    )
    _enforce_market_guardrail(
        query=technical_intent,
        agent_name="technical_stream",
        symbol=payload.symbol,
    )

    def event_stream():
        yield _ndjson_line("started", {"symbol": payload.symbol, "period": payload.period, "interval": payload.interval})
        yield _ndjson_line("progress", {"phase": "fetch_and_indicator_calc"})
        result = analyze_stock_technical(payload.symbol, period=payload.period, interval=payload.interval)
        result_payload = {
            "symbol": result.symbol,
            "image_path": result.image_path,
            "summary": result.summary,
            "latest_values": result.latest_values,
        }
        yield _ndjson_line(
            "final",
            {
                "result": result_payload
            },
        )
        _log_agent_run_safe(
            agent_name="technical_stream",
            company=None,
            symbol=payload.symbol,
            input_query=f"symbol={payload.symbol}, period={payload.period}, interval={payload.interval}",
            input_messages_count=1,
            result_payload=result_payload,
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/agents/fundamental", response_model=FundamentalResponse)
def run_fundamental(payload: FundamentalRequest) -> FundamentalResponse:
    """
    Agentic RAG over company-specific annual report chunks in pgvector.
    mode=auto -> general if no question, qa otherwise.
    """
    guardrail_query = payload.question or f"Fundamental market analysis for {payload.company}"
    _enforce_market_guardrail(
        query=guardrail_query,
        agent_name="fundamental",
        company=payload.company,
    )

    result = analyze_fundamentals(
        company=payload.company,
        question=payload.question,
        mode=payload.mode,
        collection_name=payload.collection,
        top_k=payload.top_k,
    )
    result_payload = {
        "mode": result.mode,
        "company": result.company,
        "answer": result.answer,
        "sources": result.sources,
    }
    _log_agent_run_safe(
        agent_name="fundamental",
        company=payload.company,
        symbol=None,
        input_query=payload.question or f"general fundamental analysis for {payload.company}",
        input_messages_count=1,
        result_payload=result_payload,
    )
    return FundamentalResponse(
        mode=result.mode,
        company=result.company,
        answer=result.answer,
        sources=result.sources,
    )


@app.post("/agents/fundamental/stream")
def run_fundamental_stream(payload: FundamentalRequest) -> StreamingResponse:
    guardrail_query = payload.question or f"Fundamental market analysis for {payload.company}"
    _enforce_market_guardrail(
        query=guardrail_query,
        agent_name="fundamental_stream",
        company=payload.company,
    )

    def event_stream():
        yield _ndjson_line(
            "started",
            {
                "company": payload.company,
                "mode": payload.mode,
                "collection": payload.collection,
                "top_k": payload.top_k,
            },
        )
        final_payload: Any = None
        observed_messages_count = 1
        for item in stream_fundamentals(
            company=payload.company,
            question=payload.question,
            mode=payload.mode,
            collection_name=payload.collection,
            top_k=payload.top_k,
        ):
            if isinstance(item, dict):
                if item.get("event") == "final":
                    final_payload = item.get("result")
                elif item.get("stream_mode") == "values":
                    final_payload = item.get("payload")
                if "messages" in item and isinstance(item["messages"], list):
                    observed_messages_count = max(observed_messages_count, len(item["messages"]))
            yield _ndjson_line(item.get("event", "stream"), item)
        _log_agent_run_safe(
            agent_name="fundamental_stream",
            company=payload.company,
            symbol=None,
            input_query=payload.question or f"general fundamental analysis for {payload.company}",
            input_messages_count=observed_messages_count,
            result_payload=final_payload,
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/agents/supervisor", response_model=SupervisorResponse)
def run_supervisor(payload: SupervisorRequest) -> SupervisorResponse:
    """
    Supervisor orchestration over technical + fundamental + news agents.
    """
    guardrail_query = (
        f"Company={payload.company}; Symbol={payload.symbol}; "
        f"FundamentalFocus={payload.fundamental_question or ''}; "
        f"NewsFocus={payload.news_query or ''}"
    )
    _enforce_market_guardrail(
        query=guardrail_query,
        agent_name="supervisor",
        company=payload.company,
        symbol=payload.symbol,
    )

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
    result_payload = {
        "symbol": result.symbol,
        "company": result.company,
        "technical": result.technical,
        "fundamental": result.fundamental,
        "news": result.news,
        "synthesis": result.synthesis,
    }
    _log_agent_run_safe(
        agent_name="supervisor",
        company=payload.company,
        symbol=payload.symbol,
        input_query=payload.news_query or payload.fundamental_question or f"{payload.company} {payload.symbol}",
        input_messages_count=1,
        result_payload=result_payload,
    )
    return SupervisorResponse(
        symbol=result.symbol,
        company=result.company,
        technical=result.technical,
        fundamental=result.fundamental,
        news=result.news,
        synthesis=result.synthesis,
    )


@app.post("/agents/supervisor/stream")
def run_supervisor_stream(payload: SupervisorRequest) -> StreamingResponse:
    guardrail_query = (
        f"Company={payload.company}; Symbol={payload.symbol}; "
        f"FundamentalFocus={payload.fundamental_question or ''}; "
        f"NewsFocus={payload.news_query or ''}"
    )
    _enforce_market_guardrail(
        query=guardrail_query,
        agent_name="supervisor_stream",
        company=payload.company,
        symbol=payload.symbol,
    )

    def event_stream():
        yield _ndjson_line(
            "started",
            {
                "symbol": payload.symbol,
                "company": payload.company,
                "collection": payload.collection,
            },
        )
        final_payload: Any = None
        for item in stream_market_supervised(
            symbol=payload.symbol,
            company=payload.company,
            fundamental_question=payload.fundamental_question,
            news_query=payload.news_query,
            technical_period=payload.technical_period,
            technical_interval=payload.technical_interval,
            collection_name=payload.collection,
            top_k=payload.top_k,
        ):
            if isinstance(item, dict):
                if item.get("event") == "final":
                    final_payload = item.get("result")
                elif item.get("stream_mode") == "values":
                    final_payload = item.get("payload")
            yield _ndjson_line(item.get("event", "stream"), item)
        _log_agent_run_safe(
            agent_name="supervisor_stream",
            company=payload.company,
            symbol=payload.symbol,
            input_query=payload.news_query or payload.fundamental_question or f"{payload.company} {payload.symbol}",
            input_messages_count=1,
            result_payload=final_payload,
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/documents/ingested", response_model=IngestedDocumentsResponse)
def get_ingested_documents(
    collection: Optional[str] = Query(default="fundamental_docs"),
) -> IngestedDocumentsResponse:
    records = list_ingested_documents(collection_name=collection or None)
    return IngestedDocumentsResponse(
        items=[
            IngestedDocumentItem(
                collection_name=record.collection_name,
                source_path=record.source_path,
                company=record.company,
                ticker=record.ticker,
                doc_type=record.doc_type,
                year=record.year,
                chunks_stored=record.chunks_stored,
            )
            for record in records
        ]
    )


@app.delete("/documents/ingested", response_model=DeleteIngestedDocumentResponse)
def remove_ingested_document(payload: DeleteIngestedDocumentRequest) -> DeleteIngestedDocumentResponse:
    deleted_chunks = delete_ingested_document(
        source_path=payload.source_path.strip(),
        collection_name=payload.collection.strip(),
    )
    return DeleteIngestedDocumentResponse(
        collection_name=payload.collection.strip(),
        source_path=payload.source_path.strip(),
        deleted_chunks=deleted_chunks,
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

    return IngestionResponse(
        company=result.company,
        ticker=result.ticker,
        source_path=result.source_path,
        chunks_stored=result.chunks_stored,
        collection_name=result.collection_name,
        markdown_path=result.markdown_path,
    )
