from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
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
    source_path: str
    chunks_stored: int
    collection_name: str
    markdown_path: Optional[str]


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
    api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY") or _env("AZURE_OPENAI_API_KEY")
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
        source_path=str(pdf_path),
        chunks_stored=len(enriched_chunks),
        collection_name=collection_name,
        markdown_path=str(markdown_output_path) if markdown_output_path else None,
    )
