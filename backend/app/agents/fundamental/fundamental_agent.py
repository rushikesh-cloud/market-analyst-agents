from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_community.vectorstores.pgvector import PGVector
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from app.guardrails import ensure_market_query

logger = logging.getLogger(__name__)


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_KEY"),
        azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.2,
    )


def _build_embeddings() -> AzureOpenAIEmbeddings:
    deployment = _env("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT") or _env("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY") or _env("AZURE_OPENAI_KEY")
    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _get_vector_store(collection_name: str) -> PGVector:
    return PGVector(
        connection_string=_env("PGVECTOR_CONNECTION_STRING"),
        collection_name=collection_name,
        embedding_function=_build_embeddings(),
    )


def _format_docs(docs: List[Any]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        header = f"[{i}] company={meta.get('company')} year={meta.get('year')} doc_type={meta.get('doc_type')}"
        content = doc.page_content.strip()
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


def _sources_from_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        sources.append(
            {
                "company": meta.get("company"),
                "ticker": meta.get("ticker"),
                "year": meta.get("year"),
                "doc_type": meta.get("doc_type"),
                "source_path": meta.get("source_path"),
                "chunk_index": meta.get("chunk_index"),
            }
        )
    return sources


def _extract_final_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, AIMessage):
        return str(payload.content)
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
            if isinstance(content, list):
                parts: List[str] = []
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
        if isinstance(output, str):
            return output
        if output is not None:
            return str(output)
    return str(payload)


def _to_agent_messages_input(user_text: str) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": user_text}]}


@dataclass
class FundamentalAnalysisResult:
    mode: str
    company: str
    answer: str
    sources: List[Dict[str, Any]]


def _build_fundamental_agent(
    *,
    company: str,
    collection_name: str,
    top_k: int,
) -> tuple[Any, List[Any]]:
    vector_store = _get_vector_store(collection_name)
    llm = _build_llm()
    logger.debug("Building fundamental agent for company=%s collection=%s top_k=%s", company, collection_name, top_k)

    retrieved_docs: List[Any] = []

    @tool("company_retriever")
    def company_retriever(query: str) -> str:
        """
        Retrieve relevant chunks for the given query from the specified company annual report.
        """
        logger.info("Fundamental retriever invoked for company=%s", company)
        logger.debug("Retriever query: %s", query)
        docs = vector_store.similarity_search(
            query,
            k=top_k,
            filter={"company": company},
        )
        logger.debug("Retrieved %s documents for company=%s", len(docs), company)
        retrieved_docs.extend(docs)
        return _format_docs(docs)

    system_prompt = (
        "You are a fundamental analysis agent. You MUST use the company_retriever tool "
        "to fetch context before answering. Ask targeted retrieval queries as needed. "
        "Only answer using retrieved context. If context is missing, say so explicitly."
    )

    agent = create_agent(llm, [company_retriever], system_prompt=system_prompt)
    return agent, retrieved_docs


def analyze_fundamentals(
    *,
    company: str,
    question: Optional[str] = None,
    mode: str = "auto",
    collection_name: str = "fundamental_docs",
    top_k: int = 8,
) -> FundamentalAnalysisResult:
    """
    Agentic RAG for fundamentals:
    - mode="general": summarize Balance Sheet, Income Statement, Cash Flow.
    - mode="qa": answer a specific question.
    - mode="auto": if question provided -> qa, else general.
    """

    mode = (mode or "auto").lower().strip()
    logger.info("Starting fundamental analysis: company=%s mode=%s collection=%s top_k=%s", company, mode, collection_name, top_k)
    try:
        if mode == "auto":
            mode = "qa" if question and question.strip() else "general"
        if question:
            ensure_market_query(question, field_name="question")

        agent, retrieved_docs = _build_fundamental_agent(
            company=company,
            collection_name=collection_name,
            top_k=top_k,
        )

        if mode == "general":
            prompt = (
                f"Company: {company}\n\n"
                "You need to provide a concise fundamental analysis. Use the tool multiple times "
                "with targeted queries for:\n"
                "- balance sheet\n"
                "- income statement\n"
                "- cash flow statement\n\n"
                "Then respond with the following sections:\n"
                "- Balance Sheet Summary\n"
                "- Income Statement Summary\n"
                "- Cash Flow Summary\n"
                "- Overall Assessment\n"
                "- Key Risks / Data Gaps\n"
            )
            response = agent.invoke(_to_agent_messages_input(prompt))
            logger.debug("Fundamental general response payload type=%s", type(response).__name__)
            answer = _extract_final_text(response)
            logger.info("Completed fundamental general analysis for company=%s", company)

            return FundamentalAnalysisResult(
                mode="general",
                company=company,
                answer=answer,
                sources=_sources_from_docs(retrieved_docs),
            )

        if not question or not question.strip():
            raise ValueError("Question is required when mode is 'qa'.")

        prompt = (
            f"Company: {company}\n"
            f"Question: {question}\n\n"
            "Use the retriever tool to get the most relevant chunks before answering."
        )
        response = agent.invoke(_to_agent_messages_input(prompt))
        logger.debug("Fundamental qa response payload type=%s", type(response).__name__)
        answer = _extract_final_text(response)
        logger.info("Completed fundamental qa analysis for company=%s", company)

        return FundamentalAnalysisResult(
            mode="qa",
            company=company,
            answer=answer,
            sources=_sources_from_docs(retrieved_docs),
        )
    except Exception:
        logger.exception("Fundamental analysis failed for company=%s", company)
        raise
