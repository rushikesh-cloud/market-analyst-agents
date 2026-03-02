from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import AzureChatOpenAI

from app.guardrails import ensure_market_query

logger = logging.getLogger(__name__)


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_answer_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return _message_content_to_text(last.get("content"))
            return _message_content_to_text(getattr(last, "content", ""))
        output = payload.get("output")
        if output is not None:
            return str(output)
    return str(payload)


def _extract_web_sources(payload: Any) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return sources
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return sources

    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if str(role) != "tool":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = _message_content_to_text(content).strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                sources.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": item.get("content"),
                        "score": item.get("score"),
                    }
                )
        elif isinstance(parsed, dict):
            sources.append(
                {
                    "title": parsed.get("title"),
                    "url": parsed.get("url"),
                    "content": parsed.get("content"),
                    "score": parsed.get("score"),
                }
            )

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for src in sources:
        key = str(src.get("url") or src.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(src)
    return deduped


def parse_web_search_result(payload: Any, query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "answer": _extract_answer_text(payload),
        "sources": _extract_web_sources(payload),
    }


def build_web_search_agent() -> Any:
    logger.debug("Building web search agent")
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_KEY"),
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.2,
        )
        tools = [TavilySearchResults(max_results=5)]
        system_prompt = (
            "You are a market research assistant. Use web search to gather up-to-date facts "
            "and summarize them clearly with sources."
        )
        agent = create_agent(llm, tools, system_prompt=system_prompt)
        logger.info("Web search agent ready")
        return agent
    except Exception:
        logger.exception("Failed to build web search agent")
        raise


def invoke_market_web_search(agent: Any, query: str) -> Any:
    cleaned_query = ensure_market_query(query, field_name="query")
    logger.info("Invoking web search agent")
    logger.debug("Web search query: %s", cleaned_query)
    try:
        return agent.invoke({"messages": [{"role": "user", "content": cleaned_query}]})
    except Exception:
        logger.exception("Web search agent invocation failed")
        raise

