from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg

_table_ready = False


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def _normalize_connection_string(connection_string: str) -> str:
    if connection_string.startswith("postgresql+psycopg://"):
        return connection_string.replace("postgresql+psycopg://", "postgresql://", 1)
    return connection_string


def _connection_string(connection_string: Optional[str]) -> str:
    return _normalize_connection_string(connection_string or _env("PGVECTOR_CONNECTION_STRING"))


def ensure_agent_log_table(*, connection_string: Optional[str] = None) -> None:
    global _table_ready
    if _table_ready:
        return
    sql = """
        CREATE TABLE IF NOT EXISTS agent_run_logs (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            agent_name TEXT NOT NULL,
            company TEXT,
            symbol TEXT,
            input_query TEXT,
            final_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            total_tokens INTEGER,
            message_count INTEGER NOT NULL DEFAULT 0,
            raw_result JSONB
        );
    """
    with psycopg.connect(_connection_string(connection_string)) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    _table_ready = True


def log_agent_run(
    *,
    agent_name: str,
    company: Optional[str],
    symbol: Optional[str],
    input_query: Optional[str],
    final_messages: list[str],
    total_tokens: Optional[int],
    message_count: int,
    raw_result: Any = None,
    connection_string: Optional[str] = None,
) -> None:
    ensure_agent_log_table(connection_string=connection_string)
    sql = """
        INSERT INTO agent_run_logs (
            created_at,
            agent_name,
            company,
            symbol,
            input_query,
            final_messages,
            total_tokens,
            message_count,
            raw_result
        ) VALUES (
            %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb
        );
    """
    payload_result = raw_result if raw_result is not None else {}
    with psycopg.connect(_connection_string(connection_string)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    datetime.now(timezone.utc),
                    agent_name,
                    company,
                    symbol,
                    input_query,
                    json.dumps(final_messages, ensure_ascii=False),
                    total_tokens,
                    int(message_count),
                    json.dumps(payload_result, ensure_ascii=False, default=str),
                ),
            )
        conn.commit()
