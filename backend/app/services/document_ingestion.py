from __future__ import annotations

import os
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
import psycopg
try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import AzureOpenAIEmbeddings


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


@dataclass
class IngestionResult:
    company: str
    ticker: Optional[str]
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]


def _normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return dsn


def _encode_doc_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _decode_doc_id(doc_id: str) -> Dict[str, Any]:
    padding = "=" * (-len(doc_id) % 4)
    raw = base64.urlsafe_b64decode(doc_id + padding).decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Invalid document id payload")
    return parsed


def extract_markdown_from_pdf(
    pdf_path: Path,
    *,
    endpoint: Optional[str] = None,
    key: Optional[str] = None,
    model_id: str = "prebuilt-layout",
) -> str:
    endpoint = endpoint or _env("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = key or _env("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    pdf_bytes = pdf_path.read_bytes()

    poller = client.begin_analyze_document(
        model_id,
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        output_content_format="markdown",
    )
    result = poller.result()
    if not result.content:
        raise RuntimeError("Document Intelligence returned empty content.")
    return result.content


def split_markdown_into_chunks(markdown: str) -> List[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )
    return splitter.split_text(markdown)


def _build_embeddings(deployment: Optional[str] = None) -> AzureOpenAIEmbeddings:
    deployment = deployment or _env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT") or _env("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY") or _env("AZURE_OPENAI_KEY")
    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _attach_metadata(
    chunks: Iterable[Document],
    *,
    company: str,
    ticker: Optional[str],
    doc_type: str,
    year: Optional[str],
    source_path: str,
) -> List[Document]:
    enriched: List[Document] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        metadata.update(
            {
                "company": company,
                "ticker": ticker,
                "doc_type": doc_type,
                "year": year,
                "source_path": source_path,
                "chunk_index": index,
            }
        )
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched


def store_chunks_pgvector(
    chunks: List[Document],
    *,
    collection_name: str,
    connection_string: str,
    embeddings: Optional[AzureOpenAIEmbeddings] = None,
) -> None:
    embeddings = embeddings or _build_embeddings()
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        pre_delete_collection=False,
    )


def ingest_pdf_to_pgvector(
    *,
    pdf_path: Path,
    company: str,
    ticker: Optional[str] = None,
    doc_type: str = "annual_report",
    year: Optional[str] = None,
    collection_name: str = "fundamental_docs",
    connection_string: Optional[str] = None,
    markdown_output_path: Optional[Path] = None,
    azure_endpoint: Optional[str] = None,
    azure_key: Optional[str] = None,
    azure_model_id: str = "prebuilt-layout",
    embeddings_deployment: Optional[str] = None,
) -> IngestionResult:
    markdown = extract_markdown_from_pdf(
        pdf_path,
        endpoint=azure_endpoint,
        key=azure_key,
        model_id=azure_model_id,
    )

    if markdown_output_path:
        markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_output_path.write_text(markdown, encoding="utf-8")

    chunks = split_markdown_into_chunks(markdown)
    enriched_chunks = _attach_metadata(
        chunks,
        company=company,
        ticker=ticker,
        doc_type=doc_type,
        year=year,
        source_path=str(pdf_path),
    )

    connection_string = connection_string or _env("PGVECTOR_CONNECTION_STRING")
    store_chunks_pgvector(
        enriched_chunks,
        collection_name=collection_name,
        connection_string=connection_string,
        embeddings=_build_embeddings(embeddings_deployment),
    )

    return IngestionResult(
        company=company,
        ticker=ticker,
        source_path=str(pdf_path),
        chunks_stored=len(enriched_chunks),
        collection_name=collection_name,
        markdown_path=str(markdown_output_path) if markdown_output_path else None,
    )


def list_ingested_documents_from_pgvector(
    *,
    collection_name: Optional[str] = None,
    connection_string: Optional[str] = None,
) -> List[Dict[str, Any]]:
    connection_string = _normalize_pg_dsn(connection_string or _env("PGVECTOR_CONNECTION_STRING"))
    sql = """
        SELECT
            c.name AS collection_name,
            COALESCE(e.cmetadata->>'company', '') AS company,
            NULLIF(COALESCE(e.cmetadata->>'ticker', ''), '') AS ticker,
            NULLIF(COALESCE(e.cmetadata->>'year', ''), '') AS year,
            COALESCE(e.cmetadata->>'doc_type', '') AS doc_type,
            COALESCE(e.cmetadata->>'source_path', '') AS source_path,
            COUNT(*)::int AS chunks_stored
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE (%s::text IS NULL OR c.name = %s::text)
        GROUP BY
            c.name,
            COALESCE(e.cmetadata->>'company', ''),
            NULLIF(COALESCE(e.cmetadata->>'ticker', ''), ''),
            NULLIF(COALESCE(e.cmetadata->>'year', ''), ''),
            COALESCE(e.cmetadata->>'doc_type', ''),
            COALESCE(e.cmetadata->>'source_path', '')
        ORDER BY c.name, company, year NULLS LAST, source_path;
    """
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (collection_name, collection_name))
            rows = cur.fetchall()

    docs: List[Dict[str, Any]] = []
    for row in rows:
        key_payload = {
            "collection_name": row[0],
            "company": row[1],
            "ticker": row[2],
            "year": row[3],
            "doc_type": row[4],
            "source_path": row[5],
        }
        docs.append(
            {
                "id": _encode_doc_id(key_payload),
                "collection_name": row[0],
                "company": row[1],
                "ticker": row[2],
                "year": row[3],
                "doc_type": row[4],
                "source_path": row[5],
                "chunks_stored": row[6],
                "markdown_path": None,
            }
        )
    return docs


def delete_ingested_document_from_pgvector(
    *,
    doc_id: str,
    connection_string: Optional[str] = None,
) -> int:
    payload = _decode_doc_id(doc_id)
    collection_name = str(payload.get("collection_name") or "")
    company = str(payload.get("company") or "")
    source_path = str(payload.get("source_path") or "")
    ticker = payload.get("ticker")
    year = payload.get("year")
    doc_type = payload.get("doc_type")
    if not collection_name or not company or not source_path:
        raise ValueError("Invalid document id. Missing required metadata fields.")

    connection_string = _normalize_pg_dsn(connection_string or _env("PGVECTOR_CONNECTION_STRING"))
    sql = """
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE e.collection_id = c.uuid
          AND c.name = %s::text
          AND COALESCE(e.cmetadata->>'company', '') = %s::text
          AND COALESCE(e.cmetadata->>'source_path', '') = %s::text
          AND (%s::text IS NULL OR COALESCE(e.cmetadata->>'ticker', '') = %s::text)
          AND (%s::text IS NULL OR COALESCE(e.cmetadata->>'year', '') = %s::text)
          AND (%s::text IS NULL OR COALESCE(e.cmetadata->>'doc_type', '') = %s::text);
    """
    ticker_text = None if ticker is None else str(ticker)
    year_text = None if year is None else str(year)
    doc_type_text = None if doc_type is None else str(doc_type)
    with psycopg.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    collection_name,
                    company,
                    source_path,
                    ticker_text,
                    ticker_text,
                    year_text,
                    year_text,
                    doc_type_text,
                    doc_type_text,
                ),
            )
            deleted = cur.rowcount or 0
        conn.commit()
    return int(deleted)
