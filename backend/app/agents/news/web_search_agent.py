from __future__ import annotations

import logging
import os
from typing import Any

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


def build_web_search_agent() -> Any:
    """
    Web search agent using Tavily + Azure OpenAI.

    Required env vars:
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_KEY
    - AZURE_OPENAI_DEPLOYMENT
    - AZURE_OPENAI_API_VERSION (optional; defaults to 2024-02-01)
    - TAVILY_API_KEY
    """
    logger.debug("Building web search agent")
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_KEY"),
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.2,
        )

        tools = [
            TavilySearchResults(max_results=5),
        ]

        system_prompt = (
            "You are a market research assistant. Use web search to gather up-to-date facts "
            "and summarize them clearly with sources."
        )
        # LangChain v1 create_agent API
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
