from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver

from app.agents.supervisor.supervisor_agent import analyze_market_supervised

_CHECKPOINTER_SETUP_DONE = False


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalize_pg_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return dsn


def _db_uri() -> str:
    return _normalize_pg_dsn(_env("PGVECTOR_CONNECTION_STRING"))


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_KEY"),
        azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.2,
    )


def _extract_text(payload: Any) -> str:
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


def _ensure_checkpointer_setup() -> None:
    global _CHECKPOINTER_SETUP_DONE
    if _CHECKPOINTER_SETUP_DONE:
        return
    with PostgresSaver.from_conn_string(_db_uri()) as checkpointer:
        checkpointer.setup()
    _CHECKPOINTER_SETUP_DONE = True


def run_supervisor_chat_turn(
    *,
    session_id: str,
    user_message: str,
    symbol: Optional[str] = None,
    company: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_checkpointer_setup()
    llm = _build_llm()
    default_symbol = (symbol or "").strip().upper() or "AAPL"
    default_company = (company or "").strip().upper() or default_symbol

    @tool("run_supervisor_analysis")
    def run_supervisor_analysis(
        symbol: str = default_symbol,
        company: str = default_company,
        fundamental_question: str = "",
        news_query: str = "",
    ) -> str:
        """
        Call this tool to run the market analysis supervisor agent with the given parameters.
        """
        result = analyze_market_supervised(
            symbol=symbol,
            company=company,
            fundamental_question=fundamental_question or None,
            news_query=news_query or None,
            collection_name="fundamental_docs",
            top_k=8,
        )
        payload = {
            "symbol": result.symbol,
            "company": result.company,
            "technical": result.technical,
            "fundamental": result.fundamental,
            "news": result.news,
            "synthesis": result.synthesis,
        }
        return json.dumps(payload)

    system_prompt = (
        "You are a market supervisor chat assistant. "
        "You can keep session memory and answer follow-ups. "
        "When a user asks for stock/company analysis, call run_supervisor_analysis. "
        "Always provide concise markdown in final answer with sections and bullets."
    )
    with PostgresSaver.from_conn_string(_db_uri()) as checkpointer:
        agent = create_agent(llm, [run_supervisor_analysis], system_prompt=system_prompt, checkpointer=checkpointer)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            {"configurable": {"thread_id": session_id}},
        )
    return {
        "answer": _extract_text(result),
        "raw": result,
    }
