from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.agents.news.web_search_agent import build_web_search_agent
from app.agents.technical.technical_chart_agent import analyze_stock_technical
from app.agents.fundamental.fundamental_agent import analyze_fundamentals
from app.services.document_ingestion import ingest_pdf_to_pgvector

app = FastAPI(title="Market Analyst Agent API")

# Load env vars from .env at app startup
load_dotenv()

_web_search_agent: Optional[Any] = None


def get_web_search_agent():
    global _web_search_agent
    if _web_search_agent is None:
        _web_search_agent = build_web_search_agent()
    return _web_search_agent


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class WebSearchResponse(BaseModel):
    result: Dict[str, Any]


class TechnicalRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    period: str = "3mo"
    interval: str = "1d"


class TechnicalResponse(BaseModel):
    symbol: str
    image_path: str
    summary: str
    latest_values: Dict[str, float]


class IngestionResponse(BaseModel):
    company: str
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]


class FundamentalRequest(BaseModel):
    company: str = Field(..., min_length=1)
    question: Optional[str] = None
    mode: str = "auto"  # auto | general | qa
    collection: str = "fundamental_docs"
    top_k: int = 8


class FundamentalResponse(BaseModel):
    mode: str
    company: str
    answer: str
    sources: list[dict]


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
    result = agent.invoke({"input": payload.query.strip()})
    return WebSearchResponse(result=result)


@app.post("/agents/technical", response_model=TechnicalResponse)
def run_technical(payload: TechnicalRequest) -> TechnicalResponse:
    """
    Minimal endpoint to exercise the technical analysis agent.
    Expects: {"symbol": "AAPL", "period": "3mo", "interval": "1d"}
    """
    symbol = payload.symbol.strip()
    result = analyze_stock_technical(symbol, period=payload.period, interval=payload.interval)
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
        company=payload.company.strip(),
        question=payload.question.strip() if payload.question else None,
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
