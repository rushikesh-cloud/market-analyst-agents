from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.fundamental.fundamental_agent import analyze_fundamentals
from app.agents.news.web_search_agent import build_web_search_agent
from app.agents.supervisor.supervisor_agent import analyze_market_supervised
from app.agents.technical.technical_chart_agent import analyze_stock_technical
from app.services.document_ingestion import ingest_pdf_to_pgvector


app = FastAPI(title="Market Analyst Agent API")

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

    company: str
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agents/web-search", response_model=WebSearchResponse)
def run_web_search(payload: WebSearchRequest) -> WebSearchResponse:
    """
    Minimal endpoint to exercise the web search agent.
    Expects: {"query": "..."}
    """
    agent = get_web_search_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": payload.query}]})
    answer = _extract_agent_text(result)
    return WebSearchResponse(query=payload.query, answer=answer)


@app.post("/agents/technical", response_model=TechnicalResponse)
def run_technical(payload: TechnicalRequest) -> TechnicalResponse:
    """
    Minimal endpoint to exercise the technical analysis agent.
    Expects: {"symbol": "AAPL", "period": "3mo", "interval": "1d"}
    """
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
        doc_type=doc_type.strip(),
        year=year.strip() if year else None,
        collection_name=collection.strip(),
        azure_model_id=azure_model.strip(),
        embeddings_deployment=embeddings_deployment.strip() if embeddings_deployment else None,
    )

    return IngestionResponse(
        company=result.company,
        source_path=result.source_path,
        chunks_stored=result.chunks_stored,
        collection_name=result.collection_name,
        markdown_path=result.markdown_path,
    )
