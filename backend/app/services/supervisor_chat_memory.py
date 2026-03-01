from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Any, Iterator, Literal, Optional

import psycopg
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver

from app.agents.supervisor.supervisor_agent import analyze_market_supervised


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _normalize_connection_string(connection_string: str) -> str:
    if connection_string.startswith("postgresql+psycopg://"):
        return connection_string.replace("postgresql+psycopg://", "postgresql://", 1)
    return connection_string


def _db_connection_string() -> str:
    return _normalize_connection_string(
        os.getenv("LANGGRAPH_CHECKPOINT_CONNECTION_STRING")
        or _env("PGVECTOR_CONNECTION_STRING")
    )


@lru_cache(maxsize=1)
def _chat_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_KEY"),
        azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=0.1,
    )


@dataclass
class SupervisorChatSession:
    session_id: str
    title: str
    symbol: str
    company: str
    collection_name: str
    technical_period: str
    technical_interval: str
    top_k: int
    created_at: datetime
    updated_at: datetime


@dataclass
class SupervisorChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class SupervisorChatTurn:
    session: SupervisorChatSession
    assistant_message: str
    messages: list[SupervisorChatMessage]


_session_table_lock = Lock()
_session_table_ready = False

_checkpointer_lock = Lock()
_checkpointer_cm: Optional[Iterator[PostgresSaver]] = None
_checkpointer: Optional[PostgresSaver] = None


def _ensure_session_table() -> None:
    global _session_table_ready
    if _session_table_ready:
        return
    with _session_table_lock:
        if _session_table_ready:
            return
        with psycopg.connect(_db_connection_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS supervisor_chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        company TEXT NOT NULL,
                        collection_name TEXT NOT NULL DEFAULT 'fundamental_docs',
                        technical_period TEXT NOT NULL DEFAULT '3mo',
                        technical_interval TEXT NOT NULL DEFAULT '1d',
                        top_k INTEGER NOT NULL DEFAULT 8,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_supervisor_chat_sessions_updated_at
                    ON supervisor_chat_sessions (updated_at DESC)
                    """
                )
            conn.commit()
        _session_table_ready = True


def _get_checkpointer() -> PostgresSaver:
    global _checkpointer_cm, _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    with _checkpointer_lock:
        if _checkpointer is not None:
            return _checkpointer
        cm = PostgresSaver.from_conn_string(_db_connection_string())
        saver = cm.__enter__()
        saver.setup()
        _checkpointer_cm = cm
        _checkpointer = saver
        return saver


def close_supervisor_chat_memory() -> None:
    global _checkpointer_cm, _checkpointer
    with _checkpointer_lock:
        cm = _checkpointer_cm
        _checkpointer_cm = None
        _checkpointer = None
    if cm is not None:
        cm.__exit__(None, None, None)


def _row_to_session(row: tuple[Any, ...]) -> SupervisorChatSession:
    return SupervisorChatSession(
        session_id=str(row[0]),
        title=str(row[1]),
        symbol=str(row[2]),
        company=str(row[3]),
        collection_name=str(row[4]),
        technical_period=str(row[5]),
        technical_interval=str(row[6]),
        top_k=int(row[7]),
        created_at=row[8],
        updated_at=row[9],
    )


def create_supervisor_chat_session(
    *,
    title: Optional[str],
    symbol: str,
    company: str,
    collection_name: str = "fundamental_docs",
    technical_period: str = "3mo",
    technical_interval: str = "1d",
    top_k: int = 8,
) -> SupervisorChatSession:
    _ensure_session_table()
    session_id = str(uuid.uuid4())
    clean_symbol = symbol.strip().upper()
    clean_company = company.strip().upper()
    clean_title = (title or "").strip() or f"{clean_company} ({clean_symbol})"
    with psycopg.connect(_db_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO supervisor_chat_sessions (
                    session_id, title, symbol, company, collection_name,
                    technical_period, technical_interval, top_k
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING
                    session_id, title, symbol, company, collection_name,
                    technical_period, technical_interval, top_k, created_at, updated_at
                """,
                (
                    session_id,
                    clean_title,
                    clean_symbol,
                    clean_company,
                    collection_name.strip(),
                    technical_period.strip(),
                    technical_interval.strip(),
                    int(top_k),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    if not row:
        raise RuntimeError("Unable to create chat session.")
    return _row_to_session(row)


def list_supervisor_chat_sessions(*, limit: int = 100) -> list[SupervisorChatSession]:
    _ensure_session_table()
    with psycopg.connect(_db_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    session_id, title, symbol, company, collection_name,
                    technical_period, technical_interval, top_k, created_at, updated_at
                FROM supervisor_chat_sessions
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (max(1, min(limit, 500)),),
            )
            rows = cur.fetchall()
    return [_row_to_session(row) for row in rows]


def get_supervisor_chat_session(session_id: str) -> Optional[SupervisorChatSession]:
    _ensure_session_table()
    with psycopg.connect(_db_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    session_id, title, symbol, company, collection_name,
                    technical_period, technical_interval, top_k, created_at, updated_at
                FROM supervisor_chat_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
    return _row_to_session(row) if row else None


def _touch_session_updated_at(session_id: str) -> None:
    with psycopg.connect(_db_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE supervisor_chat_sessions SET updated_at = NOW() WHERE session_id = %s",
                (session_id,),
            )
        conn.commit()


def _extract_message_text(message: Any) -> str:
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


def _extract_final_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            return _extract_message_text(messages[-1])
        output = payload.get("output")
        if isinstance(output, str):
            return output
    if hasattr(payload, "content"):
        return _extract_message_text(payload)
    return str(payload)


def _build_supervisor_markdown(result: Any) -> str:
    synthesis = result.synthesis
    lines = [
        f"## {result.company} ({result.symbol})",
        f"- **Stance:** {synthesis.get('stance', 'N/A')}",
        f"- **6M Rating:** {synthesis.get('investment_rating_6m', 'N/A')} / 10",
        "",
        "### Technical",
        str(synthesis.get("technical_section", "") or "_No technical section provided._"),
        "",
        "### Fundamental",
        str(synthesis.get("fundamental_section", "") or "_No fundamental section provided._"),
        "",
        "### News",
        str(synthesis.get("news_section", "") or "_No news section provided._"),
        "",
        "### Risks",
    ]
    risks = synthesis.get("risks") or []
    if isinstance(risks, list) and risks:
        lines.extend([f"- {str(item)}" for item in risks])
    else:
        lines.append("- No explicit risks provided.")
    lines.extend(
        [
            "",
            "### Final Thesis",
            str(synthesis.get("final_thesis", "") or "_No final thesis provided._"),
        ]
    )
    return "\n".join(lines)


def _build_chat_agent(session: SupervisorChatSession):
    default_symbol = session.symbol
    default_company = session.company
    default_collection = session.collection_name
    default_period = session.technical_period
    default_interval = session.technical_interval
    default_top_k = session.top_k

    @tool("run_supervisor_analysis")
    def run_supervisor_analysis(
        question: str,
        symbol: str = "",
        company: str = "",
    ) -> str:
        """Run a full supervisor analysis and return markdown results."""
        active_symbol = (symbol or default_symbol).strip().upper()
        active_company = (company or default_company).strip().upper()
        result = analyze_market_supervised(
            symbol=active_symbol,
            company=active_company,
            fundamental_question=(question or "").strip() or None,
            news_query=f"{active_company} {active_symbol} latest company news catalysts risks",
            technical_period=default_period,
            technical_interval=default_interval,
            collection_name=default_collection,
            top_k=default_top_k,
        )
        return _build_supervisor_markdown(result)

    system_prompt = (
        "You are a market supervisor chat analyst. "
        "For market/company analysis requests, call run_supervisor_analysis. "
        f"Default symbol is {default_symbol}. Default company is {default_company}. "
        "If the user asks about another company/symbol, pass those values to the tool. "
        "Return concise markdown and keep sections readable."
    )
    return create_agent(
        model=_chat_llm(),
        tools=[run_supervisor_analysis],
        system_prompt=system_prompt,
        checkpointer=_get_checkpointer(),
    )


def get_supervisor_chat_history(session_id: str) -> tuple[SupervisorChatSession, list[SupervisorChatMessage]]:
    session = get_supervisor_chat_session(session_id)
    if not session:
        raise KeyError(f"Session not found: {session_id}")

    config = {"configurable": {"thread_id": session.session_id}}
    state = _build_chat_agent(session).get_state(config)
    values = getattr(state, "values", {}) or {}
    messages = values.get("messages", []) if isinstance(values, dict) else []

    history: list[SupervisorChatMessage] = []
    for message in messages:
        msg_type = str(getattr(message, "type", "") or "").lower()
        if msg_type in {"human", "user"}:
            role: Literal["user", "assistant"] = "user"
        elif msg_type in {"ai", "assistant"}:
            role = "assistant"
        else:
            continue
        text = _extract_message_text(message).strip()
        if text:
            history.append(SupervisorChatMessage(role=role, content=text))
    return session, history


def send_supervisor_chat_message(*, session_id: str, message: str) -> SupervisorChatTurn:
    session = get_supervisor_chat_session(session_id)
    if not session:
        raise KeyError(f"Session not found: {session_id}")

    clean_message = message.strip()
    if not clean_message:
        raise ValueError("message must be non-empty")

    config = {"configurable": {"thread_id": session.session_id}}
    response = _build_chat_agent(session).invoke(
        {"messages": [HumanMessage(content=clean_message)]},
        config=config,
    )
    assistant_message = _extract_final_text(response).strip()
    _touch_session_updated_at(session.session_id)
    refreshed_session, history = get_supervisor_chat_history(session.session_id)
    return SupervisorChatTurn(
        session=refreshed_session,
        assistant_message=assistant_message,
        messages=history,
    )
