from __future__ import annotations

import os
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
try:
    # LangChain < 1.0
    from langchain.prompts import ChatPromptTemplate
except ModuleNotFoundError:
    # LangChain 1.0+ moved prompts to langchain_core
    from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_web_search_agent() :
    """
    Web search agent using Tavily + Azure OpenAI.

    Required env vars:
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_DEPLOYMENT
    - AZURE_OPENAI_API_VERSION (optional; defaults to 2024-02-01)
    - TAVILY_API_KEY
    """

    llm = AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_API_KEY"),
        azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.2,
    )

    tools = [
        TavilySearchResults(max_results=5),
    ]

    system_prompt = """You are a market research assistant. Use web search to gather up-to-date facts and summarize them clearly with sources."""
    # LangChain v1 create_agent API
    agent = create_agent(llm, tools, system_prompt=system_prompt)

    return agent
